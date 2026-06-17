#include <mpi.h>
#include <omp.h>
#include "common.h"

/* ═════════════════════════════════════════════════════════════
   METHOD 1 — BRUTE FORCE  (MPI + OpenMP)
   MPI level  : master assigns first-character index (ci) to workers.
   OpenMP level: worker threads each own a sub-slice of the second
                 character, recursing through remaining positions.
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

        std::atomic<bool>      found_flag{false};
        std::atomic<long long> local_tested{0};
        std::string            found_pw;

        for (int len = 1; len <= maxLen && !found_flag; ++len) {
            if (len == 1) {
                std::string buf(1, charset[ci]);
                local_tested.fetch_add(1, std::memory_order_relaxed);
                if (md5(buf) == target) {
                    found_flag = true;
                    found_pw = buf;
                }
            } else {
                #pragma omp parallel for schedule(dynamic, 1) shared(found_flag, local_tested, found_pw)
                for (int c2 = 0; c2 < CS; ++c2) {
                    if (found_flag) continue;
                    std::string buf(len, charset[0]);
                    buf[0] = charset[ci];
                    buf[1] = charset[c2];

                    std::function<void(int)> gen = [&](int depth) {
                        if (found_flag) return;
                        if (depth == 0) {
                            local_tested.fetch_add(1, std::memory_order_relaxed);
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
                    gen(len - 2);
                }
            }
        }

        if (found_flag) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            long long cnt = local_tested.load();
            MPI_Send(&cnt, 1, MPI_LONG_LONG_INT, 0, TAG_FOUND, MPI_COMM_WORLD);
            int stop;
            MPI_Recv(&stop, 1, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            long long cnt = local_tested.load();
            MPI_Send(&cnt, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

Result bfMaster(const std::string& target,
                const std::string& charset,
                int maxLen, int nprocs, int nthreads)
{
    Result r;
    r.method  = "Brute Force";
    r.procs   = nprocs;
    r.threads = nthreads;

    const int CS     = (int)charset.size();
    int       nextCi = 0, active = 0;
    long long total  = 0;
    bool      found  = false;
    std::string cracked;
    TP t0 = Clock::now();

    // Track which workers are still alive (in their receive loop)
    std::vector<bool> alive(nprocs, false);

    for (int w = 1; w < nprocs && nextCi < CS; ++w) {
        MPI_Send(&nextCi, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
        ++nextCi; ++active;
        alive[w] = true;
    }

    while (active > 0) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int src = status.MPI_SOURCE, tag = status.MPI_TAG;

        if (tag == TAG_FOUND) {
            char pwbuf[MAX_WORD] = {};
            MPI_Recv(pwbuf, MAX_WORD, MPI_CHAR, src, TAG_FOUND,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_FOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += cnt;
            cracked = pwbuf; found = true;
            int stop = -1;
            MPI_Send(&stop, 1, MPI_INT, src, TAG_STOP, MPI_COMM_WORLD);
            alive[src] = false;
            // Only send stop to workers that are still alive
            for (int w = 1; w < nprocs; ++w) {
                if (w == src || !alive[w]) continue;
                int ci = -1;
                MPI_Send(&ci, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
            }
            break;
        } else {
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
                alive[src] = false;
                --active;
            }
        }
    }

    r.found = found; r.cracked = cracked; r.tested = total; r.secs = elapsed(t0);
    return r;
}

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
                  << "  └─────────────────────────────────────────────┘\n" << RESET
                  << GRAY
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
   MPI level  : master streams BATCH_SIZE-word chunks to workers.
   OpenMP level: parallel-for with dynamic scheduling across threads.
═════════════════════════════════════════════════════════════ */

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
            long long cnt = local_tested.load();
            MPI_Send(&cnt, 1, MPI_LONG_LONG_INT, 0, TAG_FOUND, MPI_COMM_WORLD);
            MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, TAG_STOP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            long long cnt = local_tested.load();
            MPI_Send(&cnt, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

Result dictMaster(const std::string& target,
                  const std::vector<std::string>& words,
                  int nprocs, int nthreads)
{
    Result r;
    r.method  = "Dictionary";
    r.procs   = nprocs;
    r.threads = nthreads;

    long long total = 0, N = (long long)words.size();
    int nextIdx = 0, active = 0;
    bool found = false;
    std::string cracked;
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
            MPI_Recv(pwbuf, MAX_WORD, MPI_CHAR, src, TAG_FOUND,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_FOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += cnt;
            cracked = pwbuf; found = true;
            MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_STOP, MPI_COMM_WORLD);
            for (int w = 1; w < nprocs; ++w) {
                if (w == src) continue;
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
            }
            break;
        } else {
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

    r.found = found; r.cracked = cracked; r.tested = total; r.secs = elapsed(t0);
    return r;
}

Result runDictionary(const std::string& target,
                     const std::string& wordlistPath,
                     int rank, int nprocs, int nthreads)
{
    if (rank == 0) {
        std::cout << "\n" << BOLD << YELLOW
                  << "  ┌─────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Dictionary Attack [MPI + OpenMP] │\n"
                  << "  └─────────────────────────────────────────────┘\n" << RESET
                  << GRAY << "  Loading wordlist: " << wordlistPath << " ...\n" << RESET;

        std::vector<std::string> words;
        std::ifstream file(wordlistPath);
        if (!file) {
            std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
            char tmp[BATCH_BUF] = {};
            for (int w = 1; w < nprocs; ++w)
                MPI_Send(tmp, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
            Result r; r.method = "Dictionary"; r.procs = nprocs; return r;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) words.push_back(std::move(line));
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
   MPI level  : same batch-dispatch as Dictionary.
   OpenMP level: each thread calls mutate() in thread-local memory,
                 then hashes all ~70 variants independently.
═════════════════════════════════════════════════════════════ */

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
                // mutate() is a pure function — fully thread-local
                for (const std::string& candidate : mutate(batch[i])) {
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
            long long cnt = local_tested.load();
            MPI_Send(&cnt, 1, MPI_LONG_LONG_INT, 0, TAG_FOUND, MPI_COMM_WORLD);
            MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, TAG_STOP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else {
            long long cnt = local_tested.load();
            MPI_Send(&cnt, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }
}

Result ruleMaster(const std::string& target,
                  const std::vector<std::string>& words,
                  int nprocs, int nthreads)
{
    Result r;
    r.method  = "Rule-Based";
    r.procs   = nprocs;
    r.threads = nthreads;

    long long total = 0, N = (long long)words.size();
    int nextIdx = 0, active = 0;
    bool found = false;
    std::string cracked;
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
            MPI_Recv(pwbuf, MAX_WORD, MPI_CHAR, src, TAG_FOUND,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_FOUND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += cnt;
            cracked = pwbuf; found = true;
            MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_STOP, MPI_COMM_WORLD);
            for (int w = 1; w < nprocs; ++w) {
                if (w == src) continue;
                MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
            }
            break;
        } else {
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

    r.found = found; r.cracked = cracked; r.tested = total; r.secs = elapsed(t0);
    return r;
}

Result runRuleBased(const std::string& target,
                    const std::string& wordlistPath,
                    int rank, int nprocs, int nthreads)
{
    if (rank == 0) {
        std::cout << "\n" << BOLD << MAGENTA
                  << "  ┌─────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Rule-Based Attack [MPI + OpenMP] │\n"
                  << "  └─────────────────────────────────────────────┘\n" << RESET
                  << GRAY
                  << "  9 mutation rules per word  |  MPI processes: " << nprocs
                  << "  |  OpenMP threads/process: " << nthreads
                  << "  |  Total workers: " << (nprocs - 1) * nthreads
                  << "\n" << RESET;

        std::vector<std::string> words;
        std::ifstream file(wordlistPath);
        if (!file) {
            std::cout << RED << "  [!] Wordlist file not found.\n" << RESET;
            char tmp[BATCH_BUF] = {};
            for (int w = 1; w < nprocs; ++w)
                MPI_Send(tmp, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD);
            Result r; r.method = "Rule-Based"; r.procs = nprocs; return r;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) words.push_back(std::move(line));
        }
        std::cout << GRAY << "  Loaded " << words.size() << " words.\n" << RESET;
        return ruleMaster(target, words, nprocs, nthreads);
    } else {
        ruleWorker(rank, target);
        return Result{};
    }
}

/* ─────────────────────────────────────────────────────────────
   MAIN
───────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {

    // MPI_THREAD_FUNNELED: only the main thread makes MPI calls.
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "[ERROR] MPI does not support MPI_THREAD_FUNNELED.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int nthreads = omp_get_max_threads();

    char target_buf[33]    = {};
    char wordlist_buf[512] = {};
    char pw_buf[MAX_WORD]  = {};
    int  maxLen = 0, choice = 0;

    if (rank == 0) {
        std::cout << BOLD << CYAN
            << "\n"
            << "  ╔══════════════════════════════════════════════════════╗\n"
            << "  ║   Password Recovery System  –  Hybrid Edition       ║\n"
            << "  ║   MPI + OpenMP  |  EE7218 / EC7207  |  Group 30    ║\n"
            << "  ╚══════════════════════════════════════════════════════╝\n" << RESET
            << "\n"
            << GRAY << "  MPI processes          : " << BOLD << nprocs   << RESET << "\n"
            << GRAY << "  OpenMP threads/process : " << BOLD << nthreads  << RESET << "\n"
            << GRAY << "  Total parallel workers : " << BOLD << (nprocs - 1) * nthreads
            << RESET << GRAY << "  (master excluded)\n" << RESET;

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
                  << " OpenMP threads internally. ┄\n" << RESET;

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
            std::cout << YELLOW << "\n  Note: brute force capped at length 5.\n" << RESET;

        choice = printMenu();
        while (choice < 1 || choice > 4) {
            std::cout << RED << "  Please enter 1, 2, 3 or 4: " << BOLD << RESET;
            std::string line;
            std::getline(std::cin, line);
            if (!line.empty()) choice = line[0] - '0';
        }
    }

    MPI_Bcast(target_buf,   33,  MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(wordlist_buf, 512, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxLen,      1,   MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&choice,      1,   MPI_INT,  0, MPI_COMM_WORLD);

    std::string target(target_buf);
    std::string wordlist(wordlist_buf);

    std::vector<Result> results;
    auto divider = [&]() {
        if (rank == 0)
            std::cout << "\n" << GRAY
                      << "  ─────────────────────────────────────────────────\n" << RESET;
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