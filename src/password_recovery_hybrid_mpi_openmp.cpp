/*
 * ============================================================
 *  High-Performance Password Recovery System
 *  Hybrid MPI + OpenMP Implementation  —  Group 30
 *  EE7218 / EC7207  High Performance Computing
 * ============================================================
 *
 *  Build:
 *    mpicxx -O2 -std=c++17 -fopenmp password_recovery_hybrid_mpi_openmp.cpp -lssl -lcrypto -o recover_hybrid
 *
 *  Run:
 *    OMP_NUM_THREADS=4 mpirun -np 4 ./recover_hybrid
 * ============================================================
 *
 *  Two-Layer Parallelism:
 *    Layer 1 — MPI: Master/Worker across processes (inter-node)
 *    Layer 2 — OpenMP: threads within each worker (intra-node)
 *    Uses MPI_THREAD_FUNNELED: only master thread calls MPI.
 * ============================================================
 */

#include <mpi.h>
#include <omp.h>
#include <openssl/evp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
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
   MPI message tags
───────────────────────────────────────────────────────────── */
constexpr int TAG_WORK  = 1;
constexpr int TAG_FOUND = 2;
constexpr int TAG_DONE  = 3;
constexpr int TAG_STOP  = 4;
constexpr int TAG_COUNT = 5;

/* ─────────────────────────────────────────────────────────────
   Batch constants for dict / rule-based work distribution
───────────────────────────────────────────────────────────── */
constexpr int BATCH_SIZE = 200;
constexpr int MAX_WORD   = 256;
static const int BATCH_BUF = BATCH_SIZE * MAX_WORD;

/* ─────────────────────────────────────────────────────────────
   MD5  (thread-safe: allocates own EVP_MD_CTX per call)
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
   Result struct  — includes both procs and threads
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
                  << "  |  Procs: " << BOLD << r.procs << RESET
                  << "  |  Threads/proc: " << BOLD << r.threads << RESET
                  << "\n";
    } else {
        std::cout << "  " << BOLD << RED << "✘  Not found" << RESET
                  << "  |  Time: " << BOLD << std::fixed << std::setprecision(2)
                  << r.secs << "s" << RESET
                  << "  |  Tested: " << BOLD << countStr << " candidates" << RESET
                  << "  |  Procs: " << BOLD << r.procs << RESET
                  << "  |  Threads/proc: " << BOLD << r.threads << RESET
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
              << "  ╔═══════════════════════════════════════════════════════════════════════════╗\n"
              << "  ║                  COMPARISON SUMMARY  (MPI + OpenMP Hybrid)               ║\n"
              << "  ╠═══════════════════════════════════════════════════════════════════════════╣\n"
              << RESET << BOLD
              << "  ║  Password cracked   :  " << std::left << std::setw(53)
              << pw << BOLD << CYAN << "║\n"
              << "  ╠══════════════════╦════════════╦════════════╦════════════════╦══════╦══════╣\n"
              << "  ║ Method           ║  Result    ║  Time      ║  Candidates    ║ Proc ║ Thd  ║\n"
              << "  ╠══════════════════╬════════════╬════════════╬════════════════╬══════╬══════╣\n"
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
            << std::setw(4) << r.procs << " ║ "
            << std::setw(4) << r.threads << " ║\n";
    }

    std::cout << BOLD << CYAN
              << "  ╚══════════════════╩════════════╩════════════╩════════════════╩══════╩══════╝\n"
              << RESET << "\n";
}

/* ─────────────────────────────────────────────────────────────
   Mutation rules  (pure function → safe in any thread/process)
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
   Encode / decode word batches for MPI transfer
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
   Built-in demo wordlist
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
   METHOD 1 — BRUTE FORCE  (MPI + OpenMP Hybrid)
   Master distributes first-char indices; each worker uses
   OpenMP threads to parallelize suffix generation.
═════════════════════════════════════════════════════════════ */

void bfWorker(int rank, const std::string& target,
              const std::string& charset, int maxLen)
{
    (void)rank;
    const int CS = (int)charset.size();

    while (true) {
        int ci;
        MPI_Recv(&ci, 1, MPI_INT, 0, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (ci == -1) break;

        std::atomic<bool> found_flag{false};
        std::string found_pw;
        long long local_tested = 0;

        for (int len = 1; len <= maxLen && !found_flag; ++len) {
            if (len == 1) {
                std::string buf(1, charset[ci]);
                ++local_tested;
                if (md5(buf) == target) { found_flag = true; found_pw = buf; }
                continue;
            }

            // OpenMP parallelizes the second character across threads
            // MPI calls are OUTSIDE this region (funneled model)
            #pragma omp parallel shared(found_flag, found_pw)
            {
                std::string buf(len, charset[0]);
                buf[0] = charset[ci];

                #pragma omp for schedule(dynamic, 1) nowait reduction(+:local_tested)
                for (int c2 = 0; c2 < CS; ++c2) {
                    if (found_flag) continue;
                    buf[1] = charset[c2];

                    // Recurse remaining positions (thread-private)
                    std::function<void(int)> gen = [&](int depth) {
                        if (found_flag) return;
                        if (depth == 0) {
                            ++local_tested;
                            if (md5(buf) == target) {
                                found_flag = true;
                                #pragma omp critical
                                found_pw = buf;
                            }
                            return;
                        }
                        for (int c3 = 0; c3 < CS && !found_flag; ++c3) {
                            buf[len - depth] = charset[c3];
                            gen(depth - 1);
                        }
                    };
                    gen(len - 2);
                }
            } // end omp parallel
        }

        if (found_flag) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            int stop;
            MPI_Recv(&stop, 1, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            MPI_Send(&local_tested, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

Result bfMaster(const std::string& target, const std::string& charset,
                int maxLen, int nprocs) {
    Result r;
    r.method = "Brute Force";
    r.procs  = nprocs;
    r.threads = omp_get_max_threads();
    const int CS = (int)charset.size();
    int nextCi = 0, active = 0;
    long long total = 0;
    bool found = false;
    std::string cracked;
    TP t0 = Clock::now();

    for (int w = 1; w < nprocs && nextCi < CS; ++w) {
        MPI_Send(&nextCi, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
        ++nextCi; ++active;
    }

    while (active > 0) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int src = status.MPI_SOURCE, tag = status.MPI_TAG;

        if (tag == TAG_FOUND) {
            char pwbuf[MAX_WORD] = {};
            MPI_Recv(pwbuf, MAX_WORD, MPI_CHAR, src, TAG_FOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cracked = pwbuf; found = true;
            int stop = -1;
            MPI_Send(&stop, 1, MPI_INT, src, TAG_STOP, MPI_COMM_WORLD);
            for (int w = 1; w < nprocs; ++w) {
                if (w == src) continue;
                int ci = -1;
                MPI_Send(&ci, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
            }
            break;
        } else if (tag == TAG_DONE) {
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    r.found = found; r.cracked = cracked; r.tested = total; r.secs = elapsed(t0);
    return r;
}

Result runBruteForce(const std::string& target, int maxLen, int rank, int nprocs) {
    const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    if (rank == 0) {
        std::cout << "\n" << BOLD << CYAN
                  << "  ┌────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Brute Force  [MPI+OpenMP]       │\n"
                  << "  └────────────────────────────────────────────┘\n" << RESET;
        std::cout << GRAY << "  Processes: " << nprocs
                  << "  |  Threads/proc: " << omp_get_max_threads()
                  << "  |  Max length: " << maxLen << "\n" << RESET;
        return bfMaster(target, charset, maxLen, nprocs);
    } else {
        bfWorker(rank, target, charset, maxLen);
        return Result{};
    }
}

/* ═════════════════════════════════════════════════════════════
   METHOD 2 — DICTIONARY ATTACK  (MPI + OpenMP Hybrid)
   Master streams word batches; workers use OpenMP to test
   batch words in parallel across threads.
═════════════════════════════════════════════════════════════ */

void dictWorker(int rank, const std::string& target) {
    (void)rank;
    char buf[BATCH_BUF];

    while (true) {
        MPI_Status status;
        MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_STOP) break;

        std::vector<std::string> batch = decodeBatch(buf);
        if (batch.empty()) break;

        std::atomic<bool> found_flag{false};
        std::string found_pw;
        long long local_tested = 0;
        long long N = (long long)batch.size();

        // OpenMP parallel over the batch — each thread tests different words
        #pragma omp parallel for schedule(dynamic, 16) shared(found_flag, found_pw) reduction(+:local_tested)
        for (long long i = 0; i < N; ++i) {
            if (found_flag) continue;
            ++local_tested;
            if (md5(batch[i]) == target) {
                found_flag = true;
                #pragma omp critical
                found_pw = batch[i];
            }
        }

        // MPI communication outside parallel region (funneled)
        if (found_flag) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            MPI_Send(&local_tested, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

Result dictMaster(const std::string& target,
                  const std::vector<std::string>& words, int nprocs) {
    Result r;
    r.method = "Dictionary";
    r.procs  = nprocs;
    r.threads = omp_get_max_threads();
    long long total = 0;
    int nextIdx = 0, active = 0;
    bool found = false;
    std::string cracked;
    long long N = (long long)words.size();
    TP t0 = Clock::now();
    char buf[BATCH_BUF];

    for (int w = 1; w < nprocs; ++w) {
        int count = (int)std::min((long long)BATCH_SIZE, N - nextIdx);
        if (count <= 0) break;
        encodeBatch(words, nextIdx, count, buf);
        MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_WORK, MPI_COMM_WORLD);
        nextIdx += count; ++active;
    }

    while (active > 0) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int src = status.MPI_SOURCE, tag = status.MPI_TAG;

        if (tag == TAG_FOUND) {
            char pwbuf[MAX_WORD] = {};
            MPI_Recv(pwbuf, MAX_WORD, MPI_CHAR, src, TAG_FOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cracked = pwbuf; found = true;
            MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_STOP, MPI_COMM_WORLD);
            for (int w = 1; w < nprocs; ++w) {
                if (w == src) continue;
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
            }
            break;
        } else if (tag == TAG_DONE) {
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    r.found = found; r.cracked = cracked; r.tested = total; r.secs = elapsed(t0);
    return r;
}

Result runDictionary(const std::string& target, const std::string& wordlistPath,
                     int rank, int nprocs) {
    if (rank == 0) {
        std::cout << "\n" << BOLD << YELLOW
                  << "  ┌────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Dictionary Attack [MPI+OpenMP]  │\n"
                  << "  └────────────────────────────────────────────┘\n" << RESET;
        std::vector<std::string> words;
        {
            std::ifstream file(wordlistPath);
            if (!file) {
                std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
                char buf[BATCH_BUF] = {};
                for (int w = 1; w < nprocs; ++w)
                    MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
                Result r; r.method = "Dictionary"; r.procs = nprocs; return r;
            }
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) words.push_back(std::move(line));
            }
        }
        std::cout << GRAY << "  Loaded " << words.size() << " words  |  Procs: "
                  << nprocs << "  |  Threads/proc: " << omp_get_max_threads() << "\n" << RESET;
        return dictMaster(target, words, nprocs);
    } else {
        dictWorker(rank, target);
        return Result{};
    }
}

/* ═════════════════════════════════════════════════════════════
   METHOD 3 — RULE-BASED ATTACK  (MPI + OpenMP Hybrid)
   Same batch distribution as dict. Workers use OpenMP to
   parallelize mutate+hash across words in the batch.
═════════════════════════════════════════════════════════════ */

void ruleWorker(int rank, const std::string& target) {
    (void)rank;
    char buf[BATCH_BUF];

    while (true) {
        MPI_Status status;
        MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_STOP) break;

        std::vector<std::string> batch = decodeBatch(buf);
        if (batch.empty()) break;

        std::atomic<bool> found_flag{false};
        std::string found_pw;
        long long local_tested = 0;
        long long N = (long long)batch.size();

        // OpenMP: each thread takes a word, mutates it, tests all variants
        #pragma omp parallel for schedule(dynamic, 4) shared(found_flag, found_pw) reduction(+:local_tested)
        for (long long i = 0; i < N; ++i) {
            if (found_flag) continue;
            std::vector<std::string> candidates = mutate(batch[i]);
            for (const std::string& c : candidates) {
                if (found_flag) break;
                ++local_tested;
                if (md5(c) == target) {
                    found_flag = true;
                    #pragma omp critical
                    found_pw = c;
                    break;
                }
            }
        }

        if (found_flag) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            MPI_Send(&local_tested, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

Result ruleMaster(const std::string& target,
                  const std::vector<std::string>& words, int nprocs) {
    Result r;
    r.method = "Rule-Based";
    r.procs  = nprocs;
    r.threads = omp_get_max_threads();
    long long total = 0;
    int nextIdx = 0, active = 0;
    bool found = false;
    std::string cracked;
    long long N = (long long)words.size();
    TP t0 = Clock::now();
    char buf[BATCH_BUF];

    for (int w = 1; w < nprocs; ++w) {
        int count = (int)std::min((long long)BATCH_SIZE, N - nextIdx);
        if (count <= 0) break;
        encodeBatch(words, nextIdx, count, buf);
        MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_WORK, MPI_COMM_WORLD);
        nextIdx += count; ++active;
    }

    while (active > 0) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int src = status.MPI_SOURCE, tag = status.MPI_TAG;

        if (tag == TAG_FOUND) {
            char pwbuf[MAX_WORD] = {};
            MPI_Recv(pwbuf, MAX_WORD, MPI_CHAR, src, TAG_FOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cracked = pwbuf; found = true;
            MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_STOP, MPI_COMM_WORLD);
            for (int w = 1; w < nprocs; ++w) {
                if (w == src) continue;
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
            }
            break;
        } else if (tag == TAG_DONE) {
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    r.found = found; r.cracked = cracked; r.tested = total; r.secs = elapsed(t0);
    return r;
}

Result runRuleBased(const std::string& target, const std::string& wordlistPath,
                    int rank, int nprocs) {
    if (rank == 0) {
        std::cout << "\n" << BOLD << MAGENTA
                  << "  ┌────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Rule-Based Attack [MPI+OpenMP]  │\n"
                  << "  └────────────────────────────────────────────┘\n" << RESET;
        std::vector<std::string> words;
        {
            std::ifstream file(wordlistPath);
            if (!file) {
                std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
                char buf[BATCH_BUF] = {};
                for (int w = 1; w < nprocs; ++w)
                    MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
                Result r; r.method = "Rule-Based"; r.procs = nprocs; return r;
            }
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) words.push_back(std::move(line));
            }
        }
        std::cout << GRAY << "  Loaded " << words.size() << " words  |  Procs: "
                  << nprocs << "  |  Threads/proc: " << omp_get_max_threads() << "\n" << RESET;
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
              << BOLD << "  Select attack mode:\n" << RESET << "\n"
              << "    " << CYAN    << "[1]" << RESET << "  Brute Force\n"
              << "         " << GRAY << "Best for: short passwords (≤4 chars)" << RESET << "\n\n"
              << "    " << YELLOW  << "[2]" << RESET << "  Dictionary Attack\n"
              << "         " << GRAY << "Best for: common plain words" << RESET << "\n\n"
              << "    " << MAGENTA << "[3]" << RESET << "  Rule-Based Attack\n"
              << "         " << GRAY << "Best for: tweaked passwords  (e.g. P@$$w0rd1)" << RESET << "\n\n"
              << "    " << WHITE   << "[4]" << RESET << "  Run All Three  "
              << GRAY << "(shows comparison table)" << RESET << "\n\n"
              << "  Your choice [1-4]: " << BOLD;
    int choice = 0;
    std::string line;
    std::getline(std::cin, line);
    std::cout << RESET;
    if (!line.empty()) choice = line[0] - '0';
    return choice;
}

/* ─────────────────────────────────────────────────────────────
   MAIN  —  MPI_Init_thread with MPI_THREAD_FUNNELED
───────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "ERROR: MPI does not support MPI_THREAD_FUNNELED.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    char   target_buf[33]    = {};
    char   wordlist_buf[512] = {};
    int    maxLen = 0, choice = 0;
    char   pw_buf[MAX_WORD]  = {};

    if (rank == 0) {
        std::cout << BOLD << CYAN
            << "\n"
            << "  ╔═══════════════════════════════════════════════════════╗\n"
            << "  ║   Password Recovery System  –  MPI+OpenMP Hybrid     ║\n"
            << "  ║   EE7218 / EC7207  ·  High Performance Computing     ║\n"
            << "  ║   Group 30                                            ║\n"
            << "  ╚═══════════════════════════════════════════════════════╝\n"
            << RESET << "\n";

        std::cout << GRAY
                  << "  MPI processes         : " << BOLD << nprocs << RESET << "\n"
                  << GRAY
                  << "  OpenMP threads/proc   : " << BOLD << omp_get_max_threads() << RESET << "\n"
                  << GRAY
                  << "  Total parallelism     : " << BOLD << nprocs * omp_get_max_threads() << RESET << "\n"
                  << GRAY
                  << "  MPI thread level      : FUNNELED\n"
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
        std::strncpy(pw_buf, password.c_str(), MAX_WORD - 1);

        std::cout << "\n"
                  << "  Password  :  " << BOLD << password << RESET << "\n"
                  << "  MD5 hash  :  " << BOLD << target   << RESET << "\n"
                  << GRAY
                  << "\n  ┄ Hash broadcast to all workers (MPI + OpenMP threads). ┄\n"
                  << RESET;

        std::string wordlist = "wordlist.txt";
        { std::ifstream check(wordlist);
          if (!check) {
              wordlist = "/tmp/demo_wordlist.txt";
              writeDemoWordlist(wordlist);
              std::cout << GRAY << "\n  No wordlist.txt — using built-in demo wordlist.\n" << RESET;
          } else {
              std::cout << GRAY << "\n  Wordlist: " << wordlist << RESET << "\n";
          }
        }
        std::strncpy(wordlist_buf, wordlist.c_str(), 511);

        maxLen = std::min((int)password.size(), 5);
        if ((int)password.size() > 5)
            std::cout << YELLOW << "\n  Note: brute force capped at length 5.\n" << RESET;

        choice = printMenu();
        while (choice < 1 || choice > 4) {
            std::cout << RED << "  Please enter 1, 2, 3 or 4: " << BOLD << RESET;
            std::string line;
            std::getline(std::cin, line);
            if (!line.empty()) choice = line[0] - '0';
        }
    }

    // Broadcast config to all workers
    MPI_Bcast(target_buf,   33,  MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(wordlist_buf, 512, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxLen,      1,   MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&choice,      1,   MPI_INT,  0, MPI_COMM_WORLD);

    std::string target(target_buf);
    std::string wordlist(wordlist_buf);

    // Run selected method(s)
    std::vector<Result> results;
    auto divider = [&]() {
        if (rank == 0)
            std::cout << "\n" << GRAY
                      << "  ─────────────────────────────────────────────────\n" << RESET;
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

    if (rank == 0 && choice == 4)
        printSummary(results, std::string(pw_buf));
    else if (rank == 0)
        std::cout << "\n";

    MPI_Finalize();
    return 0;
}
