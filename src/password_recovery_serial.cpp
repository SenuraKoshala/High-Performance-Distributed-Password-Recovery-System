/*
 * ============================================================
 *  High-Performance Password Recovery System
 *  Serial Implementation  —  Group 30
 *  EE7218 / EC7207  High Performance Computing
 * ============================================================
 *
 *  Build (WSL / Linux):
 *    sudo apt install libssl-dev     (once only)
 *    g++ -O2 -std=c++17 password_recovery_serial.cpp -lssl -lcrypto -o recover
 *
 *  Run:
 *    ./recover
 * ============================================================
 */

#include <openssl/evp.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/* ─────────────────────────────────────────────────────────────
   ANSI colours  (work in WSL terminal)
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
#define BG_GREEN "\033[42m"
#define BG_RED   "\033[41m"

/* ─────────────────────────────────────────────────────────────
   Compute MD5 hex digest for a string
───────────────────────────────────────────────────────────── */
std::string md5(const std::string& s) {
    unsigned char buf[EVP_MAX_MD_SIZE];
    unsigned int  len = 0;
    EVP_MD_CTX*   ctx = EVP_MD_CTX_new();
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
   Live progress line  (overwrites itself in the terminal)
───────────────────────────────────────────────────────────── */
void progress(long long tested, double secs, const std::string& last) {
    double kps = tested / (secs > 0 ? secs : 1e-9) / 1000.0;
    std::cout << GRAY
              << "\r  Tested: " << std::setw(10) << tested
              << "  Speed: "   << std::setw(9) << std::fixed
              << std::setprecision(1) << kps << " k/s"
              << "  Last: "    << last.substr(0, 20)
              << std::string(4, ' ')
              << RESET << std::flush;
}

/* ─────────────────────────────────────────────────────────────
   Result struct  — filled by each method
───────────────────────────────────────────────────────────── */
struct Result {
    std::string method;
    bool        found   = false;
    std::string cracked = "";
    long long   tested  = 0;
    double      secs    = 0.0;
};

/* ─────────────────────────────────────────────────────────────
   Print the result line for one method
   Format:  Found! | Time: 0.31s | Tested: 14,000 candidates
            Not found | Time: 4.20s | Tested: 1,200,000 candidates
───────────────────────────────────────────────────────────── */
void printResult(const Result& r) {

    // format the candidate count with commas  e.g. 1200000 → "1,200,000"
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
                  << "\n";
    } else {
        std::cout << "  " << BOLD << RED << "✘  Not found" << RESET
                  << "  |  Time: " << BOLD << std::fixed << std::setprecision(2)
                  << r.secs << "s" << RESET
                  << "  |  Tested: " << BOLD << countStr << " candidates" << RESET
                  << "\n";
    }
}

/* ─────────────────────────────────────────────────────────────
   Print the comparison summary table
───────────────────────────────────────────────────────────── */
void printSummary(const std::vector<Result>& results, const std::string& pw) {

    // helper: comma-formatted number
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
              << "  ╔══════════════════════════════════════════════════════════════╗\n"
              << "  ║                    COMPARISON SUMMARY                       ║\n"
              << "  ╠══════════════════════════════════════════════════════════════╣\n"
              << RESET
              << BOLD
              << "  ║  Password cracked   :  " << std::left << std::setw(39)
              << pw << BOLD << CYAN << "║\n"
              << "  ╠══════════════════╦════════════╦════════════╦════════════════╣\n"
              << "  ║ Method           ║  Result    ║  Time      ║  Candidates    ║\n"
              << "  ╠══════════════════╬════════════╬════════════╬════════════════╣\n"
              << RESET;

    for (const auto& r : results) {

        std::string resultCol, resetCol = RESET;
        if (r.found) {
            resultCol = std::string(GREEN) + BOLD + "  Found!   " + RESET;
        } else {
            resultCol = std::string(RED)   +        " Not found " + RESET;
        }

        std::ostringstream timeStr;
        timeStr << std::fixed << std::setprecision(2) << r.secs << "s";

        std::cout
            << "  ║ " << std::left  << std::setw(16) << r.method  << " ║"
            << resultCol
            << " ║ " << std::right << std::setw(8)  << timeStr.str()  << "  "
            << " ║ " << std::setw(14) << fmt(r.tested) << " ║\n";
    }

    std::cout << BOLD << CYAN
              << "  ╚══════════════════╩════════════╩════════════╩════════════════╝\n"
              << RESET << "\n";
}

/* ═════════════════════════════════════════════════════════════
   METHOD 1 — BRUTE FORCE
   Tries every combination of characters up to maxLen.
   Character set: a-z + 0-9  (36 chars — keeps demo time low).
   For a password of length 4 that is 36^4 = ~1.7 million tries.
═════════════════════════════════════════════════════════════ */
Result runBruteForce(const std::string& target, int maxLen) {

    std::cout << "\n" << BOLD << CYAN
              << "  ┌────────────────────────────────────────┐\n"
              << "  │   METHOD : Brute Force                 │\n"
              << "  └────────────────────────────────────────┘\n" << RESET;
    std::cout << GRAY
              << "  Trying every combination of a-z and 0-9\n"
              << "  up to length " << maxLen << " ...\n" << RESET;

    const std::string charset =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";

    Result r;
    r.method = "Brute Force";

    std::string buf;
    buf.reserve(maxLen);
    long long tested = 0;
    TP t0 = Clock::now();

    std::function<bool(int)> gen = [&](int depth) -> bool {
        if (depth == 0) {
            ++tested;
            if (tested % 500000 == 0)
                progress(tested, elapsed(t0), buf);
            if (md5(buf) == target) {
                r.found   = true;
                r.cracked = buf;
                return true;
            }
            return false;
        }
        for (char c : charset) {
            buf.push_back(c);
            if (gen(depth - 1)) return true;
            buf.pop_back();
        }
        return false;
    };

    for (int len = 1; len <= maxLen && !r.found; ++len) {
        std::cout << GRAY << "  → length " << len << " ..." << RESET << "\n";
        gen(len);
    }

    r.tested = tested;
    r.secs   = elapsed(t0);
    progress(tested, r.secs, buf);
    std::cout << "\n";
    return r;
}

/* ═════════════════════════════════════════════════════════════
   METHOD 2 — DICTIONARY ATTACK
   Reads every word from the wordlist and tests it verbatim.
═════════════════════════════════════════════════════════════ */
Result runDictionary(const std::string& target, const std::string& wordlistPath) {

    std::cout << "\n" << BOLD << YELLOW
              << "  ┌────────────────────────────────────────┐\n"
              << "  │   METHOD : Dictionary Attack           │\n"
              << "  └────────────────────────────────────────┘\n" << RESET;
    std::cout << GRAY << "  Testing each word from: "
              << wordlistPath << " ...\n" << RESET;

    Result r;
    r.method = "Dictionary";

    std::ifstream file(wordlistPath);
    if (!file) {
        std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
        r.secs = 0;
        return r;
    }

    long long tested = 0;
    TP t0 = Clock::now();
    std::string word;

    while (std::getline(file, word)) {
        if (!word.empty() && word.back() == '\r') word.pop_back();
        if (word.empty()) continue;

        ++tested;
        if (tested % 500000 == 0)
            progress(tested, elapsed(t0), word);

        if (md5(word) == target) {
            r.found   = true;
            r.cracked = word;
            break;
        }
    }

    r.tested = tested;
    r.secs   = elapsed(t0);
    progress(tested, r.secs, word);
    std::cout << "\n";
    return r;
}

/* ─────────────────────────────────────────────────────────────
   Mutation rules applied to every wordlist word.
   Returns ~60–80 variants per word.
───────────────────────────────────────────────────────────── */
std::vector<std::string> mutate(const std::string& w) {
    std::vector<std::string> v;
    v.reserve(80);

    // Rule 1: original
    v.push_back(w);

    // Rule 2: Capitalize first letter
    std::string cap = w;
    if (!cap.empty()) cap[0] = (char)toupper((unsigned char)cap[0]);
    v.push_back(cap);

    // Rule 3: ALL CAPS
    std::string up = w;
    for (char& c : up) c = (char)toupper((unsigned char)c);
    v.push_back(up);

    // Rule 4: append digit / year suffixes
    for (const std::string& s : {
            "1","2","12","123","1234","12345","123456",
            "0","!","@","#",
            "2022","2023","2024","2025",
            "99","01","007","69"}) {
        v.push_back(w   + s);  // password1
        v.push_back(cap + s);  // Password1
    }

    // Rule 5: prepend common prefixes
    for (const std::string& p : {"1","my","the","123"})
        v.push_back(p + w);

    // Rule 6: leet-speak  a→@ e→3 i→1 o→0 s→$
    std::string leet = w;
    for (char& c : leet)
        switch (tolower((unsigned char)c)) {
            case 'a': c='@'; break;  case 'e': c='3'; break;
            case 'i': c='1'; break;  case 'o': c='0'; break;
            case 's': c='$'; break;
        }
    v.push_back(leet);
    for (const std::string& s : {"1","123","!"}) v.push_back(leet + s);

    // Rule 7: Capitalize + leet  →  "P@$$w0rd"
    std::string cleet = leet;
    if (!cleet.empty()) cleet[0] = (char)toupper((unsigned char)w[0]);
    v.push_back(cleet);
    for (const std::string& s : {"1","123","!","2024"}) v.push_back(cleet + s);

    // Rule 8: reverse
    std::string rev = w;
    std::reverse(rev.begin(), rev.end());
    v.push_back(rev);

    // Rule 9: double word
    v.push_back(w + w);

    return v;
}

/* ═════════════════════════════════════════════════════════════
   METHOD 3 — RULE-BASED ATTACK
   For every word in the wordlist, generates ~70 mutated
   variants and tests each one.
═════════════════════════════════════════════════════════════ */
Result runRuleBased(const std::string& target, const std::string& wordlistPath) {

    std::cout << "\n" << BOLD << MAGENTA
              << "  ┌────────────────────────────────────────┐\n"
              << "  │   METHOD : Rule-Based Attack           │\n"
              << "  └────────────────────────────────────────┘\n" << RESET;
    std::cout << GRAY
              << "  Loading wordlist and applying 9 mutation rules ...\n"
              << "  (capitalise, leet-speak, suffixes, prefixes, reverse ...)\n"
              << RESET;

    Result r;
    r.method = "Rule-Based";

    std::ifstream file(wordlistPath);
    if (!file) {
        std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
        r.secs = 0;
        return r;
    }

    long long tested = 0;
    TP t0 = Clock::now();
    std::string word;

    while (std::getline(file, word)) {
        if (!word.empty() && word.back() == '\r') word.pop_back();
        if (word.empty()) continue;

        for (const std::string& candidate : mutate(word)) {
            ++tested;
            if (tested % 500000 == 0)
                progress(tested, elapsed(t0), candidate);

            if (md5(candidate) == target) {
                r.found   = true;
                r.cracked = candidate;
                goto done;
            }
        }
    }
    done:

    r.tested = tested;
    r.secs   = elapsed(t0);
    progress(tested, r.secs, word);
    std::cout << "\n";
    return r;
}

/* ─────────────────────────────────────────────────────────────
   Write a built-in demo wordlist to disk
   (used when no wordlist.txt is present)
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
   Print the interactive menu and return the user's choice
───────────────────────────────────────────────────────────── */
int printMenu() {
    std::cout << "\n"
              << BOLD << "  Select attack mode:\n" << RESET
              << "\n"
              << "    " << CYAN  << "[1]" << RESET << "  Brute Force\n"
              << "         Tries every possible combination of characters.\n"
              << "         " << GRAY << "Best for: short passwords (≤4 chars)" << RESET << "\n"
              << "\n"
              << "    " << YELLOW << "[2]" << RESET << "  Dictionary Attack\n"
              << "         Tests each word from the wordlist exactly as-is.\n"
              << "         " << GRAY << "Best for: common plain words" << RESET << "\n"
              << "\n"
              << "    " << MAGENTA << "[3]" << RESET << "  Rule-Based Attack\n"
              << "         Applies 9 mutation rules to every wordlist word.\n"
              << "         " << GRAY << "Best for: tweaked passwords  (e.g. P@$$w0rd1)" << RESET << "\n"
              << "\n"
              << "    " << WHITE << "[4]" << RESET << "  Run All Three  " << GRAY << "(shows comparison table)" << RESET << "\n"
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
        << "  ╔══════════════════════════════════════════════════╗\n"
        << "  ║   Password Recovery System  –  Serial Edition   ║\n"
        << "  ║   EE7218 / EC7207  ·  High Performance Computing ║\n"
        << "  ║   Group 30                                       ║\n"
        << "  ╚══════════════════════════════════════════════════╝\n"
        << RESET << "\n";

    // ── Step 1: get the password to crack ──────────────────
    std::cout << "  Enter the password to crack: " << BOLD;
    std::string password;
    std::getline(std::cin, password);
    std::cout << RESET;

    if (password.empty()) {
        std::cout << RED << "\n  [!] No password entered. Exiting.\n" << RESET;
        return 1;
    }

    // ── Step 2: hash it — this becomes the "target hash" ───
    std::string target = md5(password);

    std::cout << "\n"
              << "  Password  :  " << BOLD << password << RESET << "\n"
              << "  MD5 hash  :  " << BOLD << target   << RESET << "\n"
              << "\n"
              << GRAY
              << "  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
              << "  The original password is now hidden from the\n"
              << "  cracker. Each method below sees only the hash\n"
              << "  and must recover the password from scratch.\n"
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
                      << "\n  No wordlist.txt found in current directory.\n"
                      << "  Using built-in demo wordlist.\n"
                      << "  Tip: place rockyou.txt here as 'wordlist.txt'\n"
                      << "       for real-world testing.\n"
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
                  << "  Brute force will NOT find passwords longer than 5 chars —\n"
                  << "  that is exactly why dictionary and rule-based methods exist.\n"
                  << RESET;
    }

    // ── Step 5: mode selection menu ─────────────────────────
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

    // ── Step 7: comparison table (only when running all 3) ──
    if (choice == 4) {
        printSummary(results, password);
    } else {
        std::cout << "\n";
    }

    return 0;
}