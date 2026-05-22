/*
 * ============================================================
 *  High-Performance Password Recovery System
 *  Hybrid Implementation  —  MPI + OpenMP  —  Group 30
 *  EE7218 / EC7207  High Performance Computing
 * ============================================================
 *
 *  Build (WSL / Linux):
 *    sudo apt install libssl-dev mpich   (once only)
 *    mpicxx -O2 -std=c++17 -fopenmp password_recovery_hybrid.cpp \
 *           -lssl -lcrypto -o recover_hybrid
 *
 *  Run (e.g. 4 MPI processes, each using 4 OpenMP threads = 16 workers):
 *    mpirun -np 4 ./recover_hybrid
 *
 *  Control OpenMP threads per MPI rank:
 *    OMP_NUM_THREADS=4 mpirun -np 4 ./recover_hybrid
 *
 *  Oversubscribe on a single machine:
 *    mpirun -np 4 --oversubscribe ./recover_hybrid
 * ============================================================
 *
 *  Two-Level Parallelism Architecture
 *  ────────────────────────────────────────────────────────────
 *  LEVEL 1 — MPI (Distributed Memory)
 *    Rank 0 = Master  : handles all I/O, broadcasts config,
 *                       runs a dynamic work-queue, collects results.
 *    Rank 1..N-1 = Workers : each receives a chunk of the search
 *                       space from master, processes it using
 *                       OpenMP threads internally, reports back.
 *
 *  LEVEL 2 — OpenMP (Shared Memory, inside each MPI rank)
 *    Each worker rank spawns T OpenMP threads.
 *    Threads share the received chunk and process it in parallel
 *    using atomic flags for early exit and lock-free counters.
 *
 *  Combined parallelism = N_mpi_workers × T_omp_threads
 *  Example: 4 processes × 4 threads = 16 parallel hash streams.
 *
 *  Per-method strategy:
 *
 *  [Brute Force]
 *    Master sends one "first-character index" (ci) per message.
 *    Worker's OpenMP threads each own a sub-slice of the second
 *    character, recursing through all remaining suffix positions.
 *    → MPI splits by first char, OpenMP splits by second char.
 *
 *  [Dictionary Attack]
 *    Master streams BATCH_SIZE-word chunks to workers.
 *    Within each worker, OpenMP parallel-for distributes the
 *    batch words across threads (dynamic, chunk=32).
 *
 *  [Rule-Based Attack]
 *    Same batch distribution as Dictionary.
 *    Each OpenMP thread receives a word, calls mutate() locally
 *    (thread-private allocation), hashes all ~70 variants.
 *
 *  MPI Tags:
 *    TAG_WORK  (1) master→worker  chunk descriptor
 *    TAG_FOUND (2) worker→master  cracked password
 *    TAG_DONE  (3) worker→master  chunk done, send me more
 *    TAG_STOP  (4) master→worker  halt
 * ============================================================
 */

#include <mpi.h>
#include <omp.h>
#include <openssl/evp.h>

#include <algorithm>
#include <atomic>
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
   ANSI colours  (printed only by rank 0)
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
   MPI tags
───────────────────────────────────────────────────────────── */
constexpr int TAG_WORK  = 1;
constexpr int TAG_FOUND = 2;
constexpr int TAG_DONE  = 3;
constexpr int TAG_STOP  = 4;

/* ─────────────────────────────────────────────────────────────
   Batch parameters
───────────────────────────────────────────────────────────── */
constexpr int BATCH_SIZE = 300;          // words per MPI chunk
constexpr int MAX_WORD   = 256;          // max bytes per word + null
constexpr int BATCH_BUF  = BATCH_SIZE * MAX_WORD;

/* ─────────────────────────────────────────────────────────────
   MD5  — fully re-entrant; safe to call from any thread
   (EVP_MD_CTX is allocated per-call on the thread's stack/heap)
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
   Progress  (master / rank-0 only)
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
    int         threads = 1;
};

/* ─────────────────────────────────────────────────────────────
   Print result (rank 0 only)
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
                  << "  |  Time: "  << BOLD << std::fixed << std::setprecision(2)
                  << r.secs << "s"  << RESET
                  << "  |  Tested: " << BOLD << countStr << " candidates" << RESET
                  << "  |  " << BOLD << r.procs << " proc × "
                  << r.threads << " threads" << RESET
                  << "\n";
    } else {
        std::cout << "  " << BOLD << RED << "✘  Not found" << RESET
                  << "  |  Time: "   << BOLD << std::fixed << std::setprecision(2)
                  << r.secs << "s"   << RESET
                  << "  |  Tested: " << BOLD << countStr << " candidates" << RESET
                  << "  |  " << BOLD << r.procs << " proc × "
                  << r.threads << " threads" << RESET
                  << "\n";
    }
}

/* ─────────────────────────────────────────────────────────────
   Comparison summary table (rank 0 only)
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
              << "  ╔══════════════════════════════════════════════════════════════════════════╗\n"
              << "  ║            COMPARISON SUMMARY  (MPI + OpenMP Hybrid)                   ║\n"
              << "  ╠══════════════════════════════════════════════════════════════════════════╣\n"
              << RESET << BOLD
              << "  ║  Password cracked   :  "
              << std::left << std::setw(51) << pw << BOLD << CYAN << "║\n"
              << "  ╠══════════════════╦════════════╦═══════════╦════════════════╦════════════╣\n"
              << "  ║ Method           ║  Result    ║  Time     ║  Candidates    ║ Proc×Thd   ║\n"
              << "  ╠══════════════════╬════════════╬═══════════╬════════════════╬════════════╣\n"
              << RESET;

    for (const auto& r : results) {
        std::string resultCol;
        if (r.found)
            resultCol = std::string(GREEN) + BOLD + "  Found!   " + RESET;
        else
            resultCol = std::string(RED)   +        " Not found " + RESET;

        std::ostringstream timeStr, ptStr;
        timeStr << std::fixed << std::setprecision(2) << r.secs << "s";
        ptStr   << r.procs << "×" << r.threads;

        std::cout
            << "  ║ " << std::left  << std::setw(16) << r.method  << " ║"
            << resultCol
            << " ║ " << std::right << std::setw(7)  << timeStr.str() << "  "
            << " ║ " << std::setw(14) << fmt(r.tested) << " ║ "
            << std::setw(10) << ptStr.str() << " ║\n";
    }

    std::cout << BOLD << CYAN
              << "  ╚══════════════════╩════════════╩═══════════╩════════════════╩════════════╝\n"
              << RESET << "\n";
}

/* ─────────────────────────────────────────────────────────────
   Mutation rules — pure function, thread-safe and process-safe
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
   Batch encode / decode  (same wire format as MPI version)
───────────────────────────────────────────────────────────── */
void encodeBatch(const std::vector<std::string>& words,
                 int start, int count, char* buf) {
    std::memset(buf, 0, BATCH_BUF);
    for (int i = 0; i < count; ++i)
        std::strncpy(buf + i * MAX_WORD, words[start + i].c_str(), MAX_WORD - 1);
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
   Demo wordlist fallback
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
   METHOD 1 — BRUTE FORCE  (MPI + OpenMP)
   ─────────────────────────────────────────────────────────────
   Two-level decomposition:
     MPI level  : master assigns first-character index ci (0..CS-1)
                  to workers via a dynamic work-queue.
     OpenMP level: worker spawns T threads; each thread owns a
                  disjoint slice of the SECOND character and
                  recurses through all remaining suffix positions.

   Example with charset size 62, T=4 threads:
     MPI gives worker ci=5  →  all candidates starting with 'f'
     Thread 0: f[a..p][...]   Thread 1: f[q..5][...]  etc.

   Early-exit:
     std::atomic<bool> found_flag shared across OMP threads.
     On match: thread sets flag, saves password in critical section,
     loop breaks. Worker then sends TAG_FOUND to master, which
     broadcasts TAG_STOP to all other workers.
═════════════════════════════════════════════════════════════ */

/* ── Worker ── */
void bfWorker(int rank, const std::string& target,
              const std::string& charset, int maxLen)
{
    (void)rank;
    const int CS = (int)charset.size();

    while (true) {
        int ci;
        MPI_Recv(&ci, 1, MPI_INT, 0, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (ci == -1) break;   // STOP

        std::atomic<bool>      found_flag{false};
        std::atomic<long long> local_tested{0};
        std::string            found_pw;

        // OpenMP parallel region — each thread handles a
        // slice of the second character for all lengths.
        #pragma omp parallel shared(found_flag, local_tested, found_pw)
        {
            for (int len = 1; len <= maxLen && !found_flag; ++len) {

                if (len == 1) {
                    // Single-char: only one candidate for this ci
                    #pragma omp single nowait
                    {
                        std::string buf(1, charset[ci]);
                        local_tested.fetch_add(1, std::memory_order_relaxed);
                        if (md5(buf) == target) {
                            found_flag = true;
                            #pragma omp critical
                            found_pw = buf;
                        }
                    }
                } else {
                    // len >= 2: fix buf[0]=charset[ci],
                    // distribute charset[1] across threads.
                    #pragma omp for schedule(dynamic, 1) nowait
                    for (int c2 = 0; c2 < CS; ++c2) {
                        if (found_flag) continue;

                        // Thread-private buffer
                        std::string buf(len, charset[0]);
                        buf[0] = charset[ci];
                        buf[1] = charset[c2];

                        // Recurse positions 2..len-1
                        std::function<void(int)> gen = [&](int depth) {
                            if (found_flag) return;
                            if (depth == 0) {
                                local_tested.fetch_add(
                                    1, std::memory_order_relaxed);
                                if (md5(buf) == target) {
                                    found_flag = true;
                                    #pragma omp critical
                                    found_pw = buf;
                                }
                                return;
                            }
                            for (int cx = 0; cx < CS && !found_flag; ++cx) {
                                buf[len - depth] = charset[cx];
                                gen(depth - 1);
                            }
                        };
                        gen(len - 2);   // 2 chars already fixed
                    }
                }
                #pragma omp barrier   // all threads sync before next length
            }
        } // end omp parallel

        if (found_flag) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            int stop;
            MPI_Recv(&stop, 1, MPI_INT, 0, TAG_STOP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            long long cnt = local_tested.load();
            MPI_Send(&cnt, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

/* ── Master ── */
Result bfMaster(const std::string& target,
                const std::string& charset,
                int maxLen, int nprocs, int nthreads)
{
    Result r;
    r.method  = "Brute Force";
    r.procs   = nprocs;
    r.threads = nthreads;

    const int CS     = (int)charset.size();
    int       nextCi = 0;
    int       active = 0;
    long long total  = 0;
    bool      found  = false;
    std::string cracked;

    TP t0 = Clock::now();

    // Seed workers
    for (int w = 1; w < nprocs && nextCi < CS; ++w) {
        MPI_Send(&nextCi, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
        ++nextCi; ++active;
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

            int stop = -1;
            MPI_Send(&stop, 1, MPI_INT, src, TAG_STOP, MPI_COMM_WORLD);
            for (int w = 1; w < nprocs; ++w) {
                if (w == src) continue;
                int ci = -1;
                MPI_Send(&ci, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
            }
            break;

        } else { // TAG_DONE
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_DONE,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += cnt;
            progress(total, elapsed(t0), "");

            if (nextCi < CS) {
                MPI_Send(&nextCi, 1, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);
                ++nextCi;
            } else {
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

/* ── Entry point ── */
Result runBruteForce(const std::string& target, int maxLen,
                     int rank, int nprocs, int nthreads)
{
    const std::string charset =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";

    if (rank == 0) {
        std::cout << "\n" << BOLD << CYAN
                  << "  ┌─────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Brute Force  [MPI + OpenMP]      │\n"
                  << "  └─────────────────────────────────────────────┘\n"
                  << RESET;
        std::cout << GRAY
                  << "  Charset: a-z A-Z 0-9  |  Max length: " << maxLen << "\n"
                  << "  MPI processes: " << nprocs
                  << "  |  OpenMP threads/process: " << nthreads
                  << "  |  Total workers: " << (nprocs - 1) * nthreads
                  << "\n" << RESET;
        return bfMaster(target, charset, maxLen, nprocs, nthreads);
    } else {
        bfWorker(rank, target, charset, maxLen);
        return Result{};
    }
}

/* ═════════════════════════════════════════════════════════════
   METHOD 2 — DICTIONARY ATTACK  (MPI + OpenMP)
   ─────────────────────────────────────────────────────────────
   MPI level  : master streams BATCH_SIZE-word chunks to workers
                via a dynamic work-queue (TAG_WORK / TAG_DONE).
   OpenMP level: worker distributes the received batch across T
                threads using parallel-for with dynamic scheduling.
                Each thread independently hashes its words.

   Thread safety:
     • md5() allocates EVP_MD_CTX per call — no shared OpenSSL state.
     • found_flag is std::atomic<bool> (lock-free on x86).
     • found_pw is written inside #pragma omp critical (once only).
     • local_tested uses fetch_add (lock-free increment).
═════════════════════════════════════════════════════════════ */

/* ── Worker ── */
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

        std::atomic<bool>      found_flag{false};
        std::atomic<long long> local_tested{0};
        std::string            found_pw;
        long long N = (long long)batch.size();

        #pragma omp parallel shared(found_flag, local_tested, found_pw)
        {
            #pragma omp for schedule(dynamic, 32) nowait
            for (long long i = 0; i < N; ++i) {
                if (found_flag) continue;

                local_tested.fetch_add(1, std::memory_order_relaxed);

                if (md5(batch[i]) == target) {
                    found_flag = true;
                    #pragma omp critical
                    found_pw = batch[i];
                }
            }
        }

        if (found_flag) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, TAG_STOP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            long long cnt = local_tested.load();
            MPI_Send(&cnt, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

/* ── Master ── */
Result dictMaster(const std::string& target,
                  const std::vector<std::string>& words,
                  int nprocs, int nthreads)
{
    Result r;
    r.method  = "Dictionary";
    r.procs   = nprocs;
    r.threads = nthreads;

    long long total   = 0;
    int       nextIdx = 0;
    int       active  = 0;
    bool      found   = false;
    std::string cracked;
    long long N = (long long)words.size();

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

        } else { // TAG_DONE
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

/* ── Entry point ── */
Result runDictionary(const std::string& target,
                     const std::string& wordlistPath,
                     int rank, int nprocs, int nthreads)
{
    if (rank == 0) {
        std::cout << "\n" << BOLD << YELLOW
                  << "  ┌─────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Dictionary Attack [MPI + OpenMP] │\n"
                  << "  └─────────────────────────────────────────────┘\n"
                  << RESET;
        std::cout << GRAY << "  Loading wordlist: " << wordlistPath << " ...\n" << RESET;

        std::vector<std::string> words;
        {
            std::ifstream file(wordlistPath);
            if (!file) {
                std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
                char tmp[BATCH_BUF] = {};
                for (int w = 1; w < nprocs; ++w)
                    MPI_Send(tmp, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
                Result r; r.method = "Dictionary"; r.procs = nprocs;
                return r;
            }
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) words.push_back(std::move(line));
            }
        }
        std::cout << GRAY
                  << "  Loaded " << words.size() << " words"
                  << "  |  MPI processes: " << nprocs
                  << "  |  OpenMP threads/process: " << nthreads
                  << "  |  Total workers: " << (nprocs - 1) * nthreads
                  << "\n" << RESET;

        return dictMaster(target, words, nprocs, nthreads);
    } else {
        dictWorker(rank, target);
        return Result{};
    }
}

/* ═════════════════════════════════════════════════════════════
   METHOD 3 — RULE-BASED ATTACK  (MPI + OpenMP)
   ─────────────────────────────────────────────────────────────
   MPI level  : same batch-dispatch as Dictionary.
   OpenMP level: each thread receives one base word, calls
                mutate() entirely in thread-local memory (no
                sharing of the mutation vector), then hashes
                all ~70 variants independently.

   Why mutate() is safe under OpenMP:
     It is a pure function — reads only its argument, allocates
     a new std::vector on the calling thread's heap, returns it
     by value. Zero shared mutable state.

   Chunk size: dynamic, 16 words/thread — smaller than Dictionary
   because each word costs ~70× more work; finer chunks keep
   all threads evenly loaded.
═════════════════════════════════════════════════════════════ */

/* ── Worker ── */
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

        std::atomic<bool>      found_flag{false};
        std::atomic<long long> local_tested{0};
        std::string            found_pw;
        long long N = (long long)batch.size();

        #pragma omp parallel shared(found_flag, local_tested, found_pw)
        {
            #pragma omp for schedule(dynamic, 16) nowait
            for (long long i = 0; i < N; ++i) {
                if (found_flag) continue;

                // mutate() is called in thread-local context —
                // the returned vector lives on this thread's heap
                std::vector<std::string> candidates = mutate(batch[i]);

                for (const std::string& candidate : candidates) {
                    if (found_flag) break;

                    local_tested.fetch_add(1, std::memory_order_relaxed);

                    if (md5(candidate) == target) {
                        found_flag = true;
                        #pragma omp critical
                        found_pw = candidate;
                        break;
                    }
                }
            }
        }

        if (found_flag) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, TAG_STOP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            long long cnt = local_tested.load();
            MPI_Send(&cnt, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

/* ── Master ── */
Result ruleMaster(const std::string& target,
                  const std::vector<std::string>& words,
                  int nprocs, int nthreads)
{
    Result r;
    r.method  = "Rule-Based";
    r.procs   = nprocs;
    r.threads = nthreads;

    long long total   = 0;
    int       nextIdx = 0;
    int       active  = 0;
    bool      found   = false;
    std::string cracked;
    long long N = (long long)words.size();

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

        } else { // TAG_DONE
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

/* ── Entry point ── */
Result runRuleBased(const std::string& target,
                    const std::string& wordlistPath,
                    int rank, int nprocs, int nthreads)
{
    if (rank == 0) {
        std::cout << "\n" << BOLD << MAGENTA
                  << "  ┌─────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Rule-Based Attack [MPI + OpenMP] │\n"
                  << "  └─────────────────────────────────────────────┘\n"
                  << RESET;
        std::cout << GRAY
                  << "  9 mutation rules per word  |  MPI processes: " << nprocs
                  << "  |  OpenMP threads/process: " << nthreads
                  << "  |  Total workers: " << (nprocs - 1) * nthreads
                  << "\n" << RESET;

        std::vector<std::string> words;
        {
            std::ifstream file(wordlistPath);
            if (!file) {
                std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
                char tmp[BATCH_BUF] = {};
                for (int w = 1; w < nprocs; ++w)
                    MPI_Send(tmp, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
                Result r; r.method = "Rule-Based"; r.procs = nprocs;
                return r;
            }
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) words.push_back(std::move(line));
            }
        }
        std::cout << GRAY << "  Loaded " << words.size() << " words.\n" << RESET;

        return ruleMaster(target, words, nprocs, nthreads);
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
              << "         " << GRAY << "Best for: tweaked passwords (e.g. P@$$w0rd1)" << RESET << "\n"
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

    // MPI_THREAD_FUNNELED: only the main thread makes MPI calls.
    // OpenMP threads stay inside the worker's parallel region;
    // all MPI sends/recvs happen on the main thread before/after.
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "[ERROR] MPI does not support MPI_THREAD_FUNNELED.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Capture OpenMP thread count (same on all ranks)
    int nthreads = omp_get_max_threads();

    /* ── Broadcast buffers ───────────────────────────────── */
    char target_buf[33]   = {};
    char wordlist_buf[512] = {};
    char pw_buf[MAX_WORD] = {};
    int  maxLen = 0;
    int  choice = 0;

    /* ── Rank 0: user interaction ────────────────────────── */
    if (rank == 0) {

        std::cout << BOLD << CYAN
            << "\n"
            << "  ╔══════════════════════════════════════════════════════╗\n"
            << "  ║   Password Recovery System  –  Hybrid Edition       ║\n"
            << "  ║   MPI + OpenMP  |  EE7218 / EC7207  |  Group 30    ║\n"
            << "  ╚══════════════════════════════════════════════════════╝\n"
            << RESET << "\n";

        std::cout << GRAY
                  << "  MPI processes          : " << BOLD << nprocs  << RESET << "\n"
                  << GRAY
                  << "  OpenMP threads/process : " << BOLD << nthreads << RESET << "\n"
                  << GRAY
                  << "  Total parallel workers : "
                  << BOLD << (nprocs - 1) * nthreads << RESET
                  << GRAY << "  (master excluded)\n" << RESET;

        std::cout << "\n  Enter the password to crack: " << BOLD;
        std::string password;
        std::getline(std::cin, password);
        std::cout << RESET;

        if (password.empty()) {
            std::cout << RED << "\n  [!] No password entered. Exiting.\n" << RESET;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::string target = md5(password);
        std::strncpy(target_buf,   target.c_str(),   32);
        std::strncpy(pw_buf,       password.c_str(),  MAX_WORD - 1);

        std::cout << "\n"
                  << "  Password  :  " << BOLD << password << RESET << "\n"
                  << "  MD5 hash  :  " << BOLD << target   << RESET << "\n"
                  << GRAY
                  << "\n  ┄ Hash broadcast to all MPI ranks.            ┄\n"
                  << "  ┄ Each rank uses " << nthreads
                  << " OpenMP threads internally. ┄\n"
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
        if ((int)password.size() > 5)
            std::cout << YELLOW
                      << "\n  Note: brute force capped at length 5.\n"
                      << RESET;

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
        Result r = runBruteForce(target, maxLen, rank, nprocs, nthreads);
        if (rank == 0) { printResult(r); results.push_back(r); }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (choice == 2 || choice == 4) {
        divider();
        Result r = runDictionary(target, wordlist, rank, nprocs, nthreads);
        if (rank == 0) { printResult(r); results.push_back(r); }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (choice == 3 || choice == 4) {
        divider();
        Result r = runRuleBased(target, wordlist, rank, nprocs, nthreads);
        if (rank == 0) { printResult(r); results.push_back(r); }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0 && choice == 4)
        printSummary(results, std::string(pw_buf));
    else if (rank == 0)
        std::cout << "\n";

    MPI_Finalize();
    return 0;
}