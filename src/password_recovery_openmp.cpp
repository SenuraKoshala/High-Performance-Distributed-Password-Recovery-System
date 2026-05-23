#include <functional>
/*
 * ============================================================
 *  High-Performance Password Recovery System
 *  OpenMP Parallel Implementation  —  Group 30
 *  EE7218 / EC7207  High Performance Computing
 * ============================================================
 *
 *  Build (WSL / Linux):
 *    sudo apt install libssl-dev     (once only)
 *    g++ -O2 -std=c++17 -fopenmp password_recovery_openmp.cpp -lssl -lcrypto -o recover_omp
 *
 *  Run:
 *    ./recover_omp
 *
 *  Optionally control thread count at runtime:
 *    OMP_NUM_THREADS=8 ./recover_omp
 * ============================================================
 *
 *  Parallelization Strategy (per method):
 *
 *  [Brute Force]
 *    The outermost character of every candidate string is used
 *    as the split point.  Each thread owns a disjoint slice of
 *    the charset for that first character and independently
 *    recurses through all shorter suffixes.  A shared
 *    atomic flag (found_flag) lets every thread bail out the
 *    moment any thread finds the answer, avoiding wasted work.
 *
 *  [Dictionary Attack]
 *    The wordlist is read into a std::vector up front (single-
 *    threaded I/O).  The vector is then distributed with
 *    #pragma omp parallel for schedule(dynamic, 64), so each
 *    thread grabs 64-word chunks.  Dynamic scheduling balances
 *    load when words vary in length.  A shared atomic flag
 *    stops all threads on first match.
 *
 *  [Rule-Based Attack]
 *    Same pre-load approach as Dictionary.  Each thread takes a
 *    word, expands its ~70 mutations entirely in thread-local
 *    memory (no sharing), and tests them independently.
 *    Dynamic scheduling with chunk=32 amortises the heavier
 *    per-word mutation cost while keeping load balanced.
 * ============================================================
 */

#include <openssl/evp.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/* ─────────────────────────────────────────────────────────────
   ANSI colours
───────────────────────────────────────────────────────────── */
#define RESET    "\033[0m"
#define BOLD     "\033[1m"
#define GREEN    "\033[32m"
#define CYAN     "\033[36m"
#define YELLOW   "\033[33m"
#define RED      "\033[31m"
#define GRAY     "\033[90m"
#define MAGENTA  "\033[35m"
#define WHITE    "\033[97m"

/* ─────────────────────────────────────────────────────────────
   Thread-safe MD5  (each call allocates its own EVP_MD_CTX)
───────────────────────────────────────────────────────────── */
std::string md5(const std::string& s) {
    unsigned char buf[EVP_MAX_MD_SIZE];
    unsigned int  len = 0;
    EVP_MD_CTX*   ctx = EVP_MD_CTX_new();          // per-call alloc → thread-safe
    EVP_DigestInit_ex(ctx, EVP_md5(), nullptr);
    EVP_DigestUpdate(ctx, s.data(), s.size());
    EVP_DigestFinal_ex(ctx, buf, &len);
    EVP_MD_CTX_free(ctx);
    std::ostringstream ss;
    for (unsigned int i = 0; i < len; ++i)
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)buf[i];
    return ss.str();
}

/* ─────────────────────────────────────────────────────────────
   Timer
───────────────────────────────────────────────────────────── */
using Clock = std::chrono::steady_clock;
using TP    = std::chrono::time_point<Clock>;
double elapsed(TP t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/* ─────────────────────────────────────────────────────────────
   Progress line  — called only from master thread
───────────────────────────────────────────────────────────── */
void progress(long long tested, double secs, const std::string& last) {
    double kps = tested / (secs > 0 ? secs : 1e-9) / 1000.0;
    std::cout << GRAY
              << "\r  Tested: " << std::setw(10) << tested
              << "  Speed: "    << std::setw(9)  << std::fixed
              << std::setprecision(1) << kps << " k/s"
              << "  Last: "     << last.substr(0, 20)
              << std::string(4, ' ')
              << RESET << std::flush;
}

/* ─────────────────────────────────────────────────────────────
   Result struct
───────────────────────────────────────────────────────────── */
struct Result {
    std::string method;
    bool        found   = false;
    std::string cracked = "";
    long long   tested  = 0;
    double      secs    = 0.0;
    int         threads = 1;
};

/* ─────────────────────────────────────────────────────────────
   Print result for one method
───────────────────────────────────────────────────────────── */
void printResult(const Result& r) {
    std::string countStr;
    {
        std::string raw = std::to_string(r.tested);
        int insert = (int)raw.size() % 3;
        for (int i = 0; i < (int)raw.size(); ++i) {
            if (i > 0 && (i - insert) % 3 == 0) countStr += ',';
            countStr += raw[i];
        }
    }

    std::cout << "\n";
    if (r.found) {
        std::cout << "  " << BOLD << GREEN << "✔  Found!" << RESET
                  << "  Password: " << BOLD << "\"" << r.cracked << "\"" << RESET
                  << "  |  Time: " << BOLD << std::fixed << std::setprecision(2)
                  << r.secs << "s" << RESET
                  << "  |  Tested: " << BOLD << countStr << " candidates" << RESET
                  << "  |  Threads: " << BOLD << r.threads << RESET
                  << "\n";
    } else {
        std::cout << "  " << BOLD << RED << "✘  Not found" << RESET
                  << "  |  Time: " << BOLD << std::fixed << std::setprecision(2)
                  << r.secs << "s" << RESET
                  << "  |  Tested: " << BOLD << countStr << " candidates" << RESET
                  << "  |  Threads: " << BOLD << r.threads << RESET
                  << "\n";
    }
}

/* ─────────────────────────────────────────────────────────────
   Comparison summary table
───────────────────────────────────────────────────────────── */
void printSummary(const std::vector<Result>& results, const std::string& pw) {
    auto fmt = [](long long n) -> std::string {
        std::string raw = std::to_string(n), out;
        int ins = (int)raw.size() % 3;
        for (int i = 0; i < (int)raw.size(); ++i) {
            if (i > 0 && (i - ins) % 3 == 0) out += ',';
            out += raw[i];
        }
        return out;
    };

    std::cout << "\n" << BOLD << CYAN
              << "  ╔════════════════════════════════════════════════════════════════════╗\n"
              << "  ║                     COMPARISON SUMMARY (OpenMP)                   ║\n"
              << "  ╠════════════════════════════════════════════════════════════════════╣\n"
              << RESET << BOLD
              << "  ║  Password cracked   :  " << std::left << std::setw(46)
              << pw << BOLD << CYAN << "║\n"
              << "  ╠══════════════════╦════════════╦════════════╦════════════════╦═══════╣\n"
              << "  ║ Method           ║  Result    ║  Time      ║  Candidates    ║ Thd   ║\n"
              << "  ╠══════════════════╬════════════╬════════════╬════════════════╬═══════╣\n"
              << RESET;

    for (const auto& r : results) {
        std::string resultCol;
        if (r.found)
            resultCol = std::string(GREEN) + BOLD + "  Found!   " + RESET;
        else
            resultCol = std::string(RED)   +        " Not found " + RESET;

        std::ostringstream timeStr;
        timeStr << std::fixed << std::setprecision(2) << r.secs << "s";

        std::cout
            << "  ║ " << std::left  << std::setw(16) << r.method << " ║"
            << resultCol
            << " ║ " << std::right << std::setw(8)  << timeStr.str() << "  "
            << " ║ " << std::setw(14) << fmt(r.tested) << " ║ "
            << std::setw(5) << r.threads << " ║\n";
    }

    std::cout << BOLD << CYAN
              << "  ╚══════════════════╩════════════╩════════════╩════════════════╩═══════╝\n"
              << RESET << "\n";
}

/* ═════════════════════════════════════════════════════════════
   METHOD 1 — BRUTE FORCE  (OpenMP parallel)
   ─────────────────────────────────────────────────────────────
   Strategy:
     • For length L, the first character is divided among threads.
       Thread i handles charset[i], charset[i+nthreads], …
     • Each thread recursively fills remaining L-1 positions
       using its own private string buffer (no shared mutable state).
     • std::atomic<bool> found_flag: any thread can set it;
       all threads check it at every recursion step and exit early.
     • std::atomic<long long> global_tested: updated via
       fetch_add (lock-free on x86) for accurate totals.
═════════════════════════════════════════════════════════════ */
Result runBruteForce(const std::string& target, int maxLen) {

    std::cout << "\n" << BOLD << CYAN
              << "  ┌────────────────────────────────────────┐\n"
              << "  │   METHOD : Brute Force  [OpenMP]       │\n"
              << "  └────────────────────────────────────────┘\n" << RESET;

    const std::string charset =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";
    const int CS = (int)charset.size();

    Result r;
    r.method = "Brute Force";

    std::atomic<bool>      found_flag{false};
    std::atomic<long long> global_tested{0};
    std::string            cracked_pw;

    TP t0 = Clock::now();

    for (int len = 1; len <= maxLen && !found_flag; ++len) {
        std::cout << GRAY << "  → length " << len << " ..." << RESET << "\n";

        // Parallel loop over the FIRST character of every candidate.
        // Each iteration is completely independent.
        #pragma omp parallel shared(found_flag, global_tested, cracked_pw)
        {
            // Thread-private candidate buffer
            std::string buf(len, charset[0]);

            // Recursive lambda (thread-private)
            // depth=0  →  test the fully built candidate
            // depth>0  →  enumerate this position
            std::function<void(int)> gen = [&](int depth) {
                if (found_flag) return;          // early exit

                if (depth == 0) {
                    long long cnt = global_tested.fetch_add(1, std::memory_order_relaxed);
                    if (cnt % 500000 == 0 && omp_get_thread_num() == 0)
                        progress(cnt, elapsed(t0), buf);

                    if (md5(buf) == target) {
                        found_flag = true;
                        #pragma omp critical
                        cracked_pw = buf;       // save atomically
                    }
                    return;
                }

                for (int ci = 0; ci < CS && !found_flag; ++ci) {
                    buf[len - depth] = charset[ci];
                    gen(depth - 1);
                }
            };

            // Distribute first-character slices across threads
            #pragma omp for schedule(dynamic, 1) nowait
            for (int ci = 0; ci < CS; ++ci) {
                if (found_flag) continue;   // omp for must finish all iters
                buf[0] = charset[ci];
                if (len == 1) {
                    // single-char password: test directly
                    long long cnt = global_tested.fetch_add(1, std::memory_order_relaxed);
                    if (cnt % 500000 == 0 && omp_get_thread_num() == 0)
                        progress(cnt, elapsed(t0), buf);
                    if (md5(buf) == target) {
                        found_flag = true;
                        #pragma omp critical
                        cracked_pw = buf;
                    }
                } else {
                    gen(len - 1);   // fill positions 1 .. len-1
                }
            }
        } // end parallel
    }

    r.found   = found_flag.load();
    r.cracked = cracked_pw;
    r.tested  = global_tested.load();
    r.secs    = elapsed(t0);
    r.threads = omp_get_max_threads();

    progress(r.tested, r.secs, "");
    std::cout << "\n";
    return r;
}

/* ═════════════════════════════════════════════════════════════
   METHOD 2 — DICTIONARY ATTACK  (OpenMP parallel)
   ─────────────────────────────────────────────────────────────
   Strategy:
     • Read the entire wordlist into a vector (serial I/O).
     • Parallel for with dynamic scheduling divides the vector
       across threads; chunk size 64 keeps overhead low.
     • Each thread calls md5() independently (thread-safe).
     • Atomic found_flag stops all threads immediately on a hit.
═════════════════════════════════════════════════════════════ */
Result runDictionary(const std::string& target, const std::string& wordlistPath) {

    std::cout << "\n" << BOLD << YELLOW
              << "  ┌────────────────────────────────────────┐\n"
              << "  │   METHOD : Dictionary Attack [OpenMP]  │\n"
              << "  └────────────────────────────────────────┘\n" << RESET;
    std::cout << GRAY << "  Loading wordlist: " << wordlistPath << " ...\n" << RESET;

    Result r;
    r.method = "Dictionary";

    // ── Load wordlist (serial) ──────────────────────────────
    std::vector<std::string> words;
    {
        std::ifstream file(wordlistPath);
        if (!file) {
            std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
            r.secs = 0;
            return r;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) words.push_back(std::move(line));
        }
    }
    std::cout << GRAY << "  Loaded " << words.size() << " words.\n" << RESET;

    std::atomic<bool>      found_flag{false};
    std::atomic<long long> global_tested{0};
    std::string            cracked_pw;

    TP t0 = Clock::now();
    long long N = (long long)words.size();

    #pragma omp parallel shared(found_flag, global_tested, cracked_pw)
    {
        #pragma omp for schedule(dynamic, 64) nowait
        for (long long i = 0; i < N; ++i) {
            if (found_flag) continue;

            const std::string& word = words[i];
            long long cnt = global_tested.fetch_add(1, std::memory_order_relaxed);
            if (cnt % 500000 == 0 && omp_get_thread_num() == 0)
                progress(cnt, elapsed(t0), word);

            if (md5(word) == target) {
                found_flag = true;
                #pragma omp critical
                cracked_pw = word;
            }
        }
    }

    r.found   = found_flag.load();
    r.cracked = cracked_pw;
    r.tested  = global_tested.load();
    r.secs    = elapsed(t0);
    r.threads = omp_get_max_threads();

    progress(r.tested, r.secs, "");
    std::cout << "\n";
    return r;
}

/* ─────────────────────────────────────────────────────────────
   Mutation rules  (identical to serial version — thread-safe
   because it only reads its input and returns a new vector)
───────────────────────────────────────────────────────────── */
std::vector<std::string> mutate(const std::string& w) {
    std::vector<std::string> v;
    v.reserve(80);

    v.push_back(w);

    std::string cap = w;
    if (!cap.empty()) cap[0] = (char)toupper((unsigned char)cap[0]);
    v.push_back(cap);

    std::string up = w;
    for (char& c : up) c = (char)toupper((unsigned char)c);
    v.push_back(up);

    for (const std::string& s : {
            "1","2","12","123","1234","12345","123456",
            "0","!","@","#",
            "2022","2023","2024","2025",
            "99","01","007","69"}) {
        v.push_back(w   + s);
        v.push_back(cap + s);
    }

    for (const std::string& p : {"1","my","the","123"})
        v.push_back(p + w);

    std::string leet = w;
    for (char& c : leet)
        switch (tolower((unsigned char)c)) {
            case 'a': c='@'; break;  case 'e': c='3'; break;
            case 'i': c='1'; break;  case 'o': c='0'; break;
            case 's': c='$'; break;
        }
    v.push_back(leet);
    for (const std::string& s : {"1","123","!"}) v.push_back(leet + s);

    std::string cleet = leet;
    if (!cleet.empty()) cleet[0] = (char)toupper((unsigned char)w[0]);
    v.push_back(cleet);
    for (const std::string& s : {"1","123","!","2024"}) v.push_back(cleet + s);

    std::string rev = w;
    std::reverse(rev.begin(), rev.end());
    v.push_back(rev);

    v.push_back(w + w);

    return v;
}

/* ═════════════════════════════════════════════════════════════
   METHOD 3 — RULE-BASED ATTACK  (OpenMP parallel)
   ─────────────────────────────────────────────────────────────
   Strategy:
     • Pre-load wordlist into vector (serial I/O).
     • Each parallel thread takes a chunk of words (dynamic,32).
     • mutate() is called in thread-local context; it allocates
       only on the thread's own heap — no sharing.
     • All ~70 mutations per word are tested before moving on,
       keeping the mutation batch as one logical work unit.
     • Atomic found_flag + critical section to capture the hit.
═════════════════════════════════════════════════════════════ */
Result runRuleBased(const std::string& target, const std::string& wordlistPath) {

    std::cout << "\n" << BOLD << MAGENTA
              << "  ┌────────────────────────────────────────┐\n"
              << "  │   METHOD : Rule-Based Attack [OpenMP]  │\n"
              << "  └────────────────────────────────────────┘\n" << RESET;
    std::cout << GRAY
              << "  Loading wordlist and applying 9 mutation rules ...\n"
              << "  (capitalise, leet-speak, suffixes, prefixes, reverse ...)\n"
              << RESET;

    Result r;
    r.method = "Rule-Based";

    // ── Load wordlist (serial) ──────────────────────────────
    std::vector<std::string> words;
    {
        std::ifstream file(wordlistPath);
        if (!file) {
            std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
            r.secs = 0;
            return r;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) words.push_back(std::move(line));
        }
    }
    std::cout << GRAY << "  Loaded " << words.size() << " words.\n" << RESET;

    std::atomic<bool>      found_flag{false};
    std::atomic<long long> global_tested{0};
    std::string            cracked_pw;

    TP t0 = Clock::now();
    long long N = (long long)words.size();

    #pragma omp parallel shared(found_flag, global_tested, cracked_pw)
    {
        #pragma omp for schedule(dynamic, 32) nowait
        for (long long i = 0; i < N; ++i) {
            if (found_flag) continue;

            // mutate() is pure (no shared writes) — fully thread-safe
            std::vector<std::string> candidates = mutate(words[i]);

            for (const std::string& candidate : candidates) {
                if (found_flag) break;

                long long cnt = global_tested.fetch_add(1, std::memory_order_relaxed);
                if (cnt % 500000 == 0 && omp_get_thread_num() == 0)
                    progress(cnt, elapsed(t0), candidate);

                if (md5(candidate) == target) {
                    found_flag = true;
                    #pragma omp critical
                    cracked_pw = candidate;
                    break;
                }
            }
        }
    }

    r.found   = found_flag.load();
    r.cracked = cracked_pw;
    r.tested  = global_tested.load();
    r.secs    = elapsed(t0);
    r.threads = omp_get_max_threads();

    progress(r.tested, r.secs, "");
    std::cout << "\n";
    return r;
}

/* ─────────────────────────────────────────────────────────────
   Built-in demo wordlist (used when no wordlist.txt exists)
───────────────────────────────────────────────────────────── */
void writeDemoWordlist(const std::string& path) {
    static const char* words[] = {
        "password","123456","qwerty","admin","letmein","welcome",
        "monkey","dragon","master","sunshine","princess","football",
        "shadow","superman","batman","trustno1","iloveyou","hello",
        "login","pass","secret","abc123","test","guest","root",
        "computer","internet","michael","jessica","hunter","ranger",
        "soccer","baseball","network","server","linux","windows",
        "cookie","summer","flower","chicken","pepper","cheese",
        nullptr
    };
    std::ofstream f(path);
    for (int i = 0; words[i]; ++i) f << words[i] << "\n";
}

/* ─────────────────────────────────────────────────────────────
   Interactive menu
───────────────────────────────────────────────────────────── */
int printMenu() {
    std::cout << "\n"
              << BOLD << "  Select attack mode:\n" << RESET
              << "\n"
              << "    " << CYAN    << "[1]" << RESET << "  Brute Force\n"
              << "         Tries every possible combination of characters.\n"
              << "         " << GRAY << "Best for: short passwords (≤4 chars)" << RESET << "\n"
              << "\n"
              << "    " << YELLOW  << "[2]" << RESET << "  Dictionary Attack\n"
              << "         Tests each word from the wordlist exactly as-is.\n"
              << "         " << GRAY << "Best for: common plain words" << RESET << "\n"
              << "\n"
              << "    " << MAGENTA << "[3]" << RESET << "  Rule-Based Attack\n"
              << "         Applies 9 mutation rules to every wordlist word.\n"
              << "         " << GRAY << "Best for: tweaked passwords  (e.g. P@$$w0rd1)" << RESET << "\n"
              << "\n"
              << "    " << WHITE   << "[4]" << RESET << "  Run All Three  "
              << GRAY << "(shows comparison table)" << RESET << "\n"
              << "\n"
              << "  Your choice [1-4]: " << BOLD;

    int choice = 0;
    std::string line;
    std::getline(std::cin, line);
    std::cout << RESET;
    if (!line.empty()) choice = line[0] - '0';
    return choice;
}

/* ─────────────────────────────────────────────────────────────
   MAIN
───────────────────────────────────────────────────────────── */
int main() {

    // ── Banner ─────────────────────────────────────────────
    std::cout << BOLD << CYAN
        << "\n"
        << "  ╔════════════════════════════════════════════════════╗\n"
        << "  ║   Password Recovery System  –  OpenMP Edition     ║\n"
        << "  ║   EE7218 / EC7207  ·  High Performance Computing  ║\n"
        << "  ║   Group 30                                         ║\n"
        << "  ╚════════════════════════════════════════════════════╝\n"
        << RESET << "\n";

    // ── Thread count info ───────────────────────────────────
    std::cout << GRAY
              << "  OpenMP threads available : "
              << BOLD << omp_get_max_threads() << RESET << "\n"
              << GRAY
              << "  (override with: OMP_NUM_THREADS=N ./recover_omp)\n"
              << RESET;

    // ── Step 1: get the password to crack ──────────────────
    std::cout << "\n  Enter the password to crack: " << BOLD;
    std::string password;
    std::getline(std::cin, password);
    std::cout << RESET;

    if (password.empty()) {
        std::cout << RED << "\n  [!] No password entered. Exiting.\n" << RESET;
        return 1;
    }

    // ── Step 2: hash it ─────────────────────────────────────
    std::string target = md5(password);

    std::cout << "\n"
              << "  Password  :  " << BOLD << password << RESET << "\n"
              << "  MD5 hash  :  " << BOLD << target   << RESET << "\n"
              << "\n"
              << GRAY
              << "  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
              << "  The original password is now hidden from the\n"
              << "  cracker. Each method sees only the hash.\n"
              << "  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
              << RESET;

    // ── Step 3: resolve wordlist ────────────────────────────
    std::string wordlist = "wordlist.txt";
    {
        std::ifstream check(wordlist);
        if (!check) {
            wordlist = "/tmp/demo_wordlist.txt";
            writeDemoWordlist(wordlist);
            std::cout << GRAY
                      << "\n  No wordlist.txt found — using built-in demo wordlist.\n"
                      << "  Tip: place rockyou.txt as 'wordlist.txt' for real testing.\n"
                      << RESET;
        } else {
            std::cout << GRAY << "\n  Wordlist: " << wordlist << RESET << "\n";
        }
    }

    // ── Step 4: brute-force length cap ─────────────────────
    int maxLen = std::min((int)password.size(), 5);
    if ((int)password.size() > 5) {
        std::cout << YELLOW
                  << "\n  Note: password is " << password.size() << " characters long.\n"
                  << "  Brute force is capped at length 5 to keep it fast.\n"
                  << RESET;
    }

    // ── Step 5: mode selection ──────────────────────────────
    int choice = printMenu();
    while (choice < 1 || choice > 4) {
        std::cout << RED << "  Please enter 1, 2, 3 or 4: " << BOLD << RESET;
        std::string line;
        std::getline(std::cin, line);
        if (!line.empty()) choice = line[0] - '0';
    }

    // ── Step 6: run selected method(s) ──────────────────────
    std::vector<Result> results;

    auto divider = [&]() {
        std::cout << "\n" << GRAY
                  << "  ─────────────────────────────────────────────────\n"
                  << RESET;
    };

    if (choice == 1 || choice == 4) {
        divider();
        Result r = runBruteForce(target, maxLen);
        printResult(r);
        results.push_back(r);
    }

    if (choice == 2 || choice == 4) {
        divider();
        Result r = runDictionary(target, wordlist);
        printResult(r);
        results.push_back(r);
    }

    if (choice == 3 || choice == 4) {
        divider();
        Result r = runRuleBased(target, wordlist);
        printResult(r);
        results.push_back(r);
    }

    // ── Step 7: comparison table ────────────────────────────
    if (choice == 4) {
        printSummary(results, password);
    } else {
        std::cout << "\n";
    }

    return 0;
}