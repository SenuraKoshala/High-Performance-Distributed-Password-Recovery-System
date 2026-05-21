/*
 * ============================================================
 *  High-Performance Password Recovery System
 *  MPI Distributed Memory Implementation  —  Group 30
 *  EE7218 / EC7207  High Performance Computing
 * ============================================================
 *
 *  Build (WSL / Linux):
 *    sudo apt install libssl-dev mpich   (once only)
 *    mpicxx -O2 -std=c++17 password_recovery_mpi.cpp -lssl -lcrypto -o recover_mpi
 *
 *  Run (e.g. 4 processes):
 *    mpirun -np 4 ./recover_mpi
 *
 *  On a single machine with many cores:
 *    mpirun -np 8 --oversubscribe ./recover_mpi
 * ============================================================
 *
 *  MPI Architecture — Master / Worker Model
 *  ─────────────────────────────────────────
 *  Rank 0  =  MASTER
 *    • Handles all terminal I/O (banner, menu, prompts, results).
 *    • Broadcasts the target hash and configuration to workers.
 *    • For Brute Force: distributes "first-character" slices.
 *    • For Dict / Rule-Based: reads wordlist, scatters word
 *      chunks to workers using a work-queue (tag-based send/recv).
 *    • Listens for FOUND / DONE messages from workers and
 *      signals all workers to stop once a result arrives.
 *
 *  Ranks 1..N-1  =  WORKERS
 *    • Receive their slice of the search space from master.
 *    • Hash candidates and report immediately on a match.
 *    • Send a DONE message when their chunk is exhausted.
 *    • Loop for more work (dynamic work-queue pattern) until
 *      master sends a STOP signal.
 *
 *  Tags used for MPI messages:
 *    TAG_WORK  (1)  — master → worker: here is your next chunk
 *    TAG_FOUND (2)  — worker → master: I found the password
 *    TAG_DONE  (3)  — worker → master: my chunk is finished
 *    TAG_STOP  (4)  — master → worker: stop, answer found or exhausted
 *    TAG_COUNT (5)  — worker → master: partial tested count
 *
 *  Work distribution per method:
 *    Brute Force  : first-character index sent as a single int;
 *                   worker recurses through all suffix positions.
 *    Dictionary   : master streams batches of BATCH_SIZE words
 *                   encoded as a flat char buffer; each worker
 *                   requests more work when its batch is done.
 *    Rule-Based   : same batched approach; worker calls mutate()
 *                   on each received word independently.
 * ============================================================
 */

#include <mpi.h>
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
#include <cstring>

/* ─────────────────────────────────────────────────────────────
   ANSI colours  (shown only by rank 0 / master)
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
   MPI message tags
───────────────────────────────────────────────────────────── */
constexpr int TAG_WORK  = 1;   // master → worker: chunk descriptor
constexpr int TAG_FOUND = 2;   // worker → master: cracked password
constexpr int TAG_DONE  = 3;   // worker → master: finished chunk
constexpr int TAG_STOP  = 4;   // master → worker: halt
constexpr int TAG_COUNT = 5;   // worker → master: tested count update

/* ─────────────────────────────────────────────────────────────
   Work-queue batch size for dict/rule-based
───────────────────────────────────────────────────────────── */
constexpr int BATCH_SIZE = 200;   // words per chunk sent to a worker
constexpr int MAX_WORD   = 256;   // max bytes per word incl. null

/* ─────────────────────────────────────────────────────────────
   MD5  (each call is fully self-contained → process-safe)
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
   Progress line  (master only)
───────────────────────────────────────────────────────────── */
void progress(long long tested, double secs, const std::string& last) {
    double kps = tested / (secs > 0 ? secs : 1e-9) / 1000.0;
    std::cout << GRAY
              << "\r  Tested: " << std::setw(10) << tested
              << "  Speed: "    << std::setw(9) << std::fixed
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
    int         procs   = 1;
};

/* ─────────────────────────────────────────────────────────────
   Print result (master only)
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
                  << "  |  Processes: " << BOLD << r.procs << RESET
                  << "\n";
    } else {
        std::cout << "  " << BOLD << RED << "✘  Not found" << RESET
                  << "  |  Time: " << BOLD << std::fixed << std::setprecision(2)
                  << r.secs << "s" << RESET
                  << "  |  Tested: " << BOLD << countStr << " candidates" << RESET
                  << "  |  Processes: " << BOLD << r.procs << RESET
                  << "\n";
    }
}

/* ─────────────────────────────────────────────────────────────
   Comparison summary table (master only)
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
              << "  ╔═════════════════════════════════════════════════════════════════════╗\n"
              << "  ║                  COMPARISON SUMMARY  (MPI)                         ║\n"
              << "  ╠═════════════════════════════════════════════════════════════════════╣\n"
              << RESET << BOLD
              << "  ║  Password cracked   :  " << std::left << std::setw(47)
              << pw << BOLD << CYAN << "║\n"
              << "  ╠══════════════════╦════════════╦════════════╦════════════════╦══════╣\n"
              << "  ║ Method           ║  Result    ║  Time      ║  Candidates    ║ Proc ║\n"
              << "  ╠══════════════════╬════════════╬════════════╬════════════════╬══════╣\n"
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
            << " ║ " << std::right << std::setw(8) << timeStr.str() << "  "
            << " ║ " << std::setw(14) << fmt(r.tested) << " ║ "
            << std::setw(4) << r.procs << " ║\n";
    }

    std::cout << BOLD << CYAN
              << "  ╚══════════════════╩════════════╩════════════╩════════════════╩══════╝\n"
              << RESET << "\n";
}

/* ─────────────────────────────────────────────────────────────
   Mutation rules  (identical to serial — pure function,
   safe to call in any process)
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

/* ─────────────────────────────────────────────────────────────
   Encode / decode a batch of words into a flat char buffer.

   Wire format:
     [ word0 \0 word1 \0 ... wordN-1 \0 ]
   Total buffer size = BATCH_SIZE * MAX_WORD bytes (fixed-size
   MPI send to avoid variable-length messages).
   Unused slots are zero-filled; receiver stops at empty string.
───────────────────────────────────────────────────────────── */
static const int BATCH_BUF = BATCH_SIZE * MAX_WORD;

void encodeBatch(const std::vector<std::string>& words,
                 int start, int count,
                 char* buf)
{
    std::memset(buf, 0, BATCH_BUF);
    for (int i = 0; i < count; ++i) {
        const std::string& w = words[start + i];
        std::strncpy(buf + i * MAX_WORD, w.c_str(), MAX_WORD - 1);
    }
}

std::vector<std::string> decodeBatch(const char* buf) {
    std::vector<std::string> out;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        const char* p = buf + i * MAX_WORD;
        if (*p == '\0') break;
        out.emplace_back(p);
    }
    return out;
}

/* ─────────────────────────────────────────────────────────────
   Built-in demo wordlist (used when no wordlist.txt found)
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

/* ═════════════════════════════════════════════════════════════
   METHOD 1 — BRUTE FORCE  (MPI)
   ─────────────────────────────────────────────────────────────
   Master distributes first-character indices (0..CS-1) to
   workers one at a time using a dynamic work-queue.

   Protocol:
     MASTER                          WORKER
     ──────                          ──────
     Send TAG_WORK  int ci →         Receive ci
                                     Recurse all suffixes for charset[ci]
                                     Send TAG_DONE  long long tested →
     Receive TAG_DONE or TAG_FOUND
     If TAG_FOUND: broadcast STOP
     Else: send next ci (or STOP)
═════════════════════════════════════════════════════════════ */

/* ── Worker side of Brute Force ── */
void bfWorker(int rank, const std::string& target,
              const std::string& charset, int maxLen)
{
    (void)rank;
    const int CS = (int)charset.size();

    while (true) {
        // Receive work: a first-char index, or -1 = STOP
        int ci;
        MPI_Recv(&ci, 1, MPI_INT, 0, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (ci == -1) break;   // STOP signal

        long long local_tested = 0;
        std::string found_pw;
        bool found = false;

        // For each length, fix buf[0] = charset[ci] and recurse
        for (int len = 1; len <= maxLen && !found; ++len) {
            std::string buf(len, charset[0]);
            buf[0] = charset[ci];

            if (len == 1) {
                ++local_tested;
                if (md5(buf) == target) {
                    found    = true;
                    found_pw = buf;
                    break;
                }
                continue;
            }

            // Recursive suffix generation
            std::function<bool(int)> gen = [&](int depth) -> bool {
                if (depth == 0) {
                    ++local_tested;
                    if (md5(buf) == target) {
                        found    = true;
                        found_pw = buf;
                        return true;
                    }
                    return false;
                }
                for (int c2 = 0; c2 < CS && !found; ++c2) {
                    buf[len - depth] = charset[c2];
                    if (gen(depth - 1)) return true;
                }
                return false;
            };
            gen(len - 1);
        }

        if (found) {
            // Send found password back to master
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            // Wait for STOP
            int stop;
            MPI_Recv(&stop, 1, MPI_INT, 0, TAG_STOP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            // Report count and ask for next chunk
            MPI_Send(&local_tested, 1, MPI_LONG_LONG_INT, 0,
                     TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

/* ── Master side of Brute Force ── */
Result bfMaster(const std::string& target,
                const std::string& charset, int maxLen,
                int nprocs)
{
    Result r;
    r.method = "Brute Force";
    r.procs  = nprocs;

    const int CS      = (int)charset.size();
    int       nextCi  = 0;          // next first-char index to dispatch
    int       active  = 0;          // workers currently busy
    long long total   = 0;
    bool      found   = false;
    std::string cracked;

    TP t0 = Clock::now();

    // Seed every worker with its first job
    for (int w = 1; w < nprocs && nextCi < CS; ++w) {
        MPI_Send(&nextCi, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
        ++nextCi;
        ++active;
    }

    // Dynamic work-queue loop
    while (active > 0) {
        MPI_Status status;
        // Probe for any incoming message (DONE or FOUND)
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int src = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (tag == TAG_FOUND) {
            char pwbuf[MAX_WORD] = {};
            MPI_Recv(pwbuf, MAX_WORD, MPI_CHAR, src, TAG_FOUND,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cracked = pwbuf;
            found   = true;

            // Send STOP to the reporting worker (it's waiting for it)
            int stop = -1;
            MPI_Send(&stop, 1, MPI_INT, src, TAG_STOP, MPI_COMM_WORLD);

            // Send STOP to all other active workers
            for (int w = 1; w < nprocs; ++w) {
                if (w == src) continue;
                int ci = -1;
                MPI_Send(&ci, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
            }
            break;

        } else if (tag == TAG_DONE) {
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_DONE,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += cnt;
            progress(total, elapsed(t0), "");

            if (nextCi < CS) {
                // Give the worker its next first-char index
                MPI_Send(&nextCi, 1, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);
                ++nextCi;
            } else {
                // No more work — send STOP
                int ci = -1;
                MPI_Send(&ci, 1, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);
                --active;
            }
        }
    }

    r.found   = found;
    r.cracked = cracked;
    r.tested  = total;
    r.secs    = elapsed(t0);
    return r;
}

/* ── Public entry point ── */
Result runBruteForce(const std::string& target, int maxLen,
                     int rank, int nprocs)
{
    const std::string charset =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";

    if (rank == 0) {
        std::cout << "\n" << BOLD << CYAN
                  << "  ┌────────────────────────────────────────┐\n"
                  << "  │   METHOD : Brute Force  [MPI]          │\n"
                  << "  └────────────────────────────────────────┘\n" << RESET;
        std::cout << GRAY
                  << "  Charset: a-z A-Z 0-9  |  Max length: " << maxLen
                  << "  |  Processes: " << nprocs << "\n" << RESET;

        Result r = bfMaster(target, charset, maxLen, nprocs);
        return r;
    } else {
        bfWorker(rank, target, charset, maxLen);
        return Result{};   // unused on workers
    }
}

/* ═════════════════════════════════════════════════════════════
   METHOD 2 — DICTIONARY ATTACK  (MPI)
   ─────────────────────────────────────────────────────────────
   Master reads the wordlist once and streams BATCH_SIZE-word
   chunks to each worker on demand (dynamic work-queue).

   Protocol  (same tag scheme as Brute Force):
     TAG_WORK  →  flat char buffer  (BATCH_BUF bytes)
                  or a single-byte buffer with count==0 → STOP
     TAG_DONE  →  long long tested count
     TAG_FOUND →  char[MAX_WORD] cracked password
     TAG_STOP  →  int -1 (ack to finder)
═════════════════════════════════════════════════════════════ */

/* ── Worker side of Dictionary ── */
void dictWorker(int rank, const std::string& target) {
    (void)rank;
    char buf[BATCH_BUF];

    while (true) {
        MPI_Status status;
        MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_STOP) break;

        std::vector<std::string> batch = decodeBatch(buf);
        if (batch.empty()) break;

        long long local_tested = 0;
        bool found = false;
        std::string found_pw;

        for (const std::string& word : batch) {
            ++local_tested;
            if (md5(word) == target) {
                found    = true;
                found_pw = word;
                break;
            }
        }

        if (found) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            // Wait for STOP ack
            MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, TAG_STOP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            MPI_Send(&local_tested, 1, MPI_LONG_LONG_INT, 0,
                     TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

/* ── Master side of Dictionary ── */
Result dictMaster(const std::string& target,
                  const std::vector<std::string>& words,
                  int nprocs)
{
    Result r;
    r.method = "Dictionary";
    r.procs  = nprocs;

    long long total    = 0;
    int       nextIdx  = 0;
    int       active   = 0;
    bool      found    = false;
    std::string cracked;
    long long N        = (long long)words.size();

    TP t0 = Clock::now();
    char buf[BATCH_BUF];

    // Seed every worker with its first batch
    for (int w = 1; w < nprocs; ++w) {
        int count = (int)std::min((long long)BATCH_SIZE, N - nextIdx);
        if (count <= 0) break;
        encodeBatch(words, nextIdx, count, buf);
        MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_WORK, MPI_COMM_WORLD);
        nextIdx += count;
        ++active;
    }

    while (active > 0) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int src = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (tag == TAG_FOUND) {
            char pwbuf[MAX_WORD] = {};
            MPI_Recv(pwbuf, MAX_WORD, MPI_CHAR, src, TAG_FOUND,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cracked = pwbuf;
            found   = true;

            // ACK the finder
            MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_STOP, MPI_COMM_WORLD);

            // STOP all other active workers
            for (int w = 1; w < nprocs; ++w) {
                if (w == src) continue;
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
            }
            break;

        } else if (tag == TAG_DONE) {
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_DONE,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += cnt;
            progress(total, elapsed(t0), words[nextIdx > 0 ? nextIdx - 1 : 0]);

            int count = (int)std::min((long long)BATCH_SIZE, N - nextIdx);
            if (count > 0) {
                encodeBatch(words, nextIdx, count, buf);
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_WORK, MPI_COMM_WORLD);
                nextIdx += count;
            } else {
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_STOP, MPI_COMM_WORLD);
                --active;
            }
        }
    }

    r.found   = found;
    r.cracked = cracked;
    r.tested  = total;
    r.secs    = elapsed(t0);
    return r;
}

/* ── Public entry point ── */
Result runDictionary(const std::string& target,
                     const std::string& wordlistPath,
                     int rank, int nprocs)
{
    if (rank == 0) {
        std::cout << "\n" << BOLD << YELLOW
                  << "  ┌────────────────────────────────────────┐\n"
                  << "  │   METHOD : Dictionary Attack [MPI]     │\n"
                  << "  └────────────────────────────────────────┘\n" << RESET;
        std::cout << GRAY << "  Loading wordlist: " << wordlistPath
                  << " ...\n" << RESET;

        std::vector<std::string> words;
        {
            std::ifstream file(wordlistPath);
            if (!file) {
                std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
                // Signal all workers to stop immediately
                char buf[BATCH_BUF] = {};
                for (int w = 1; w < nprocs; ++w)
                    MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
                Result r;
                r.method = "Dictionary";
                r.procs  = nprocs;
                return r;
            }
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) words.push_back(std::move(line));
            }
        }
        std::cout << GRAY << "  Loaded " << words.size()
                  << " words  |  Processes: " << nprocs << "\n" << RESET;

        return dictMaster(target, words, nprocs);

    } else {
        dictWorker(rank, target);
        return Result{};
    }
}

/* ═════════════════════════════════════════════════════════════
   METHOD 3 — RULE-BASED ATTACK  (MPI)
   ─────────────────────────────────────────────────────────────
   Same batch distribution as Dictionary.  Worker receives a
   batch of base words, calls mutate() on each, then tests all
   resulting candidates.  The mutation step is entirely local
   to each worker — zero inter-process communication during
   the hash-comparison phase.
═════════════════════════════════════════════════════════════ */

/* ── Worker side of Rule-Based ── */
void ruleWorker(int rank, const std::string& target) {
    (void)rank;
    char buf[BATCH_BUF];

    while (true) {
        MPI_Status status;
        MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_STOP) break;

        std::vector<std::string> batch = decodeBatch(buf);
        if (batch.empty()) break;

        long long local_tested = 0;
        bool found = false;
        std::string found_pw;

        for (const std::string& word : batch) {
            if (found) break;
            for (const std::string& candidate : mutate(word)) {
                ++local_tested;
                if (md5(candidate) == target) {
                    found    = true;
                    found_pw = candidate;
                    break;
                }
            }
        }

        if (found) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, TAG_STOP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            MPI_Send(&local_tested, 1, MPI_LONG_LONG_INT, 0,
                     TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

/* ── Master side of Rule-Based  (reuses dictMaster logic) ── */
Result ruleMaster(const std::string& target,
                  const std::vector<std::string>& words,
                  int nprocs)
{
    // Identical dispatch to dictMaster; workers handle mutations internally.
    // We duplicate it here with updated method name for clarity.
    Result r;
    r.method = "Rule-Based";
    r.procs  = nprocs;

    long long total   = 0;
    int       nextIdx = 0;
    int       active  = 0;
    bool      found   = false;
    std::string cracked;
    long long N       = (long long)words.size();

    TP t0 = Clock::now();
    char buf[BATCH_BUF];

    for (int w = 1; w < nprocs; ++w) {
        int count = (int)std::min((long long)BATCH_SIZE, N - nextIdx);
        if (count <= 0) break;
        encodeBatch(words, nextIdx, count, buf);
        MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_WORK, MPI_COMM_WORLD);
        nextIdx += count;
        ++active;
    }

    while (active > 0) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int src = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (tag == TAG_FOUND) {
            char pwbuf[MAX_WORD] = {};
            MPI_Recv(pwbuf, MAX_WORD, MPI_CHAR, src, TAG_FOUND,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cracked = pwbuf;
            found   = true;

            MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_STOP, MPI_COMM_WORLD);
            for (int w = 1; w < nprocs; ++w) {
                if (w == src) continue;
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
            }
            break;

        } else if (tag == TAG_DONE) {
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_DONE,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += cnt;
            if (nextIdx > 0 && nextIdx <= (int)words.size())
                progress(total, elapsed(t0), words[nextIdx - 1]);

            int count = (int)std::min((long long)BATCH_SIZE, N - nextIdx);
            if (count > 0) {
                encodeBatch(words, nextIdx, count, buf);
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_WORK, MPI_COMM_WORLD);
                nextIdx += count;
            } else {
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_STOP, MPI_COMM_WORLD);
                --active;
            }
        }
    }

    r.found   = found;
    r.cracked = cracked;
    r.tested  = total;
    r.secs    = elapsed(t0);
    return r;
}

/* ── Public entry point ── */
Result runRuleBased(const std::string& target,
                    const std::string& wordlistPath,
                    int rank, int nprocs)
{
    if (rank == 0) {
        std::cout << "\n" << BOLD << MAGENTA
                  << "  ┌────────────────────────────────────────┐\n"
                  << "  │   METHOD : Rule-Based Attack [MPI]     │\n"
                  << "  └────────────────────────────────────────┘\n" << RESET;
        std::cout << GRAY
                  << "  Applying 9 mutation rules per word ...\n"
                  << "  Processes: " << nprocs << "\n" << RESET;

        std::vector<std::string> words;
        {
            std::ifstream file(wordlistPath);
            if (!file) {
                std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
                char buf[BATCH_BUF] = {};
                for (int w = 1; w < nprocs; ++w)
                    MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
                Result r;
                r.method = "Rule-Based";
                r.procs  = nprocs;
                return r;
            }
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) words.push_back(std::move(line));
            }
        }
        std::cout << GRAY << "  Loaded " << words.size() << " words.\n" << RESET;

        return ruleMaster(target, words, nprocs);

    } else {
        ruleWorker(rank, target);
        return Result{};
    }
}

/* ─────────────────────────────────────────────────────────────
   Interactive menu  (rank 0 only)
───────────────────────────────────────────────────────────── */
int printMenu() {
    std::cout << "\n"
              << BOLD << "  Select attack mode:\n" << RESET
              << "\n"
              << "    " << CYAN    << "[1]" << RESET << "  Brute Force\n"
              << "         " << GRAY << "Best for: short passwords (≤4 chars)" << RESET << "\n"
              << "\n"
              << "    " << YELLOW  << "[2]" << RESET << "  Dictionary Attack\n"
              << "         " << GRAY << "Best for: common plain words" << RESET << "\n"
              << "\n"
              << "    " << MAGENTA << "[3]" << RESET << "  Rule-Based Attack\n"
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
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ── All configuration lives on rank 0 and is broadcast ─ */

    // We use fixed-size broadcast buffers for simplicity
    char   target_buf[33]  = {};   // MD5 hex = 32 chars + null
    char   wordlist_buf[512] = {};
    int    maxLen  = 0;
    int    choice  = 0;
    char   pw_buf[MAX_WORD] = {};   // original password (for display only)

    /* ── Rank 0: interact with the user ─────────────────── */
    if (rank == 0) {

        std::cout << BOLD << CYAN
            << "\n"
            << "  ╔═══════════════════════════════════════════════════╗\n"
            << "  ║   Password Recovery System  –  MPI Edition       ║\n"
            << "  ║   EE7218 / EC7207  ·  High Performance Computing  ║\n"
            << "  ║   Group 30                                         ║\n"
            << "  ╚═══════════════════════════════════════════════════╝\n"
            << RESET << "\n";

        std::cout << GRAY
                  << "  MPI processes : " << BOLD << nprocs << RESET << "\n"
                  << GRAY
                  << "  Architecture  : Master (rank 0) + "
                  << nprocs - 1 << " Worker(s)\n"
                  << RESET;

        std::cout << "\n  Enter the password to crack: " << BOLD;
        std::string password;
        std::getline(std::cin, password);
        std::cout << RESET;

        if (password.empty()) {
            std::cout << RED << "\n  [!] No password entered. Exiting.\n" << RESET;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::string target = md5(password);
        std::strncpy(target_buf, target.c_str(), 32);
        std::strncpy(pw_buf,     password.c_str(), MAX_WORD - 1);

        std::cout << "\n"
                  << "  Password  :  " << BOLD << password << RESET << "\n"
                  << "  MD5 hash  :  " << BOLD << target   << RESET << "\n"
                  << GRAY
                  << "\n  ┄ The hash is broadcast to all workers. ┄\n"
                  << "  ┄ Each worker tests its own slice.       ┄\n"
                  << RESET;

        // Resolve wordlist
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
        std::strncpy(wordlist_buf, wordlist.c_str(), 511);

        maxLen = std::min((int)password.size(), 5);
        if ((int)password.size() > 5) {
            std::cout << YELLOW
                      << "\n  Note: brute force capped at length 5.\n"
                      << RESET;
        }

        choice = printMenu();
        while (choice < 1 || choice > 4) {
            std::cout << RED << "  Please enter 1, 2, 3 or 4: " << BOLD << RESET;
            std::string line;
            std::getline(std::cin, line);
            if (!line.empty()) choice = line[0] - '0';
        }
    }

    /* ── Broadcast config to all workers ─────────────────── */
    MPI_Bcast(target_buf,   33,  MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(wordlist_buf, 512, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxLen,      1,   MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&choice,      1,   MPI_INT,  0, MPI_COMM_WORLD);

    std::string target(target_buf);
    std::string wordlist(wordlist_buf);

    /* ── Run selected method(s) ───────────────────────────── */
    std::vector<Result> results;

    auto divider = [&]() {
        if (rank == 0)
            std::cout << "\n" << GRAY
                      << "  ─────────────────────────────────────────────────\n"
                      << RESET;
    };

    if (choice == 1 || choice == 4) {
        divider();
        Result r = runBruteForce(target, maxLen, rank, nprocs);
        if (rank == 0) { printResult(r); results.push_back(r); }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (choice == 2 || choice == 4) {
        divider();
        Result r = runDictionary(target, wordlist, rank, nprocs);
        if (rank == 0) { printResult(r); results.push_back(r); }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (choice == 3 || choice == 4) {
        divider();
        Result r = runRuleBased(target, wordlist, rank, nprocs);
        if (rank == 0) { printResult(r); results.push_back(r); }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0 && choice == 4) {
        printSummary(results, std::string(pw_buf));
    } else if (rank == 0) {
        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}