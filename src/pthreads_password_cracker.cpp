#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <pthread.h>
#include <openssl/md5.h>
#include <atomic>
#include <vector>

using namespace std;
using namespace chrono;

// ─── Constants (mirrors serial.cpp) ──────────────────────────────────────────
const string CHARSET      = "abcdefghijklmnopqrstuvwxyz0123456789";
const int    MAX_LENGTH   = 5;
const int    CHARSET_SIZE = CHARSET.size();
const int    NUM_THREADS  = 8;  // Tune to your CPU core count

// ─── Shared state between threads ────────────────────────────────────────────
struct SharedState {
    const string* target_hash;    // Read-only: the hash we're cracking
    string        result;         // Write: recovered password (protected by mutex)
    bool          found;          // Write: discovery flag (protected by mutex)
    long long     candidates_tested; // Write: cumulative counter (protected by mutex)

    pthread_mutex_t mutex;        // Protects result, found, candidates_tested
};

// ─── Per-thread arguments ─────────────────────────────────────────────────────
// Static partitioning: each thread owns a contiguous [start, end) slice
// of the candidate index space for a fixed password length.
// This avoids dynamic task allocation overhead and prevents false sharing
// because each thread has its own local counter (flushed to shared at end).
struct ThreadArgs {
    int           length;         // Password length for this search round
    long long     start_index;    // Inclusive start of this thread's slice
    long long     end_index;      // Exclusive end of this thread's slice
    SharedState*  state;          // Pointer to shared state
};

// COMMIT 2: Core Thread Function
// - computeMD5() and indexToCandidate() (same as serial.cpp)
// - threadWorker(): local counting to avoid false sharing,
//   early-exit on found, mutex-protected writes

// ─── Hash & index utilities (identical to serial.cpp)     ///    we use to hash අංකයක් password එකක් බවට හරවනවා────────────────────────
string computeMD5(const string& input) {            // password → hash //
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(input.c_str()),
        input.size(), digest);

    char hex_str[33];
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        sprintf(&hex_str[i * 2], "%02x", digest[i]);
    hex_str[32] = '\0';
    return string(hex_str);
}

string indexToCandidate(long long index, int length) {      // number → password //
    string candidate(length, CHARSET[0]);
    for (int i = length - 1; i >= 0; i--) {
        candidate[i] = CHARSET[index % CHARSET_SIZE];
        index /= CHARSET_SIZE;
    }
    return candidate;
}

long long computeTotalCandidates(int length) {
    long long total = 1;
    for (int i = 0; i < length; i++) total *= CHARSET_SIZE;
    return total;
}

// ─── Thread worker ────────────────────────────────────────────────────────────
// Key design decisions:
//  1. LOCAL counter (local_count) — accumulated per-thread, flushed once at end.
//     Avoids hammering the shared mutex on every hash, eliminating false sharing.
//  2. Periodic found-check (every 1000 candidates) — avoids locking on every
//     iteration while still reacting quickly to another thread's discovery.
//  3. Single mutex lock on success — only the winning thread writes result.
void* threadWorker(void* arg) {
    ThreadArgs*  args  = static_cast<ThreadArgs*>(arg);
    SharedState* state = args->state;

    long long local_count = 0;  // ← false-sharing avoidance: thread-local counter    // තමන්ගේ counter  //

    for (long long i = args->start_index; i < args->end_index; i++) {

        // Check every 1000 iterations whether another thread already found it
        if (i % 1000 == 0) {              // හැම 1000 tries වලදී — වෙන කෙනෙක් හොයාගත්තද? //
            pthread_mutex_lock(&state->mutex);
            bool already_found = state->found;
            pthread_mutex_unlock(&state->mutex);
            if (already_found) break;  // Early exit             // ඔව් නම් STOP
        }

        string candidate = indexToCandidate(i, args->length);
        string hash      = computeMD5(candidate);
        local_count++;

        if (hash == *state->target_hash) {               // ඔව් නම් STOP  // ✅ FOUND
            // ── Critical section: write result ──
            pthread_mutex_lock(&state->mutex);
            if (!state->found) {           // Double-check: avoid overwriting
                state->found  = true;
                state->result = candidate;
            }
            pthread_mutex_unlock(&state->mutex);
            break;
        }
    }

    // Flush local count to shared counter (single lock per thread lifetime)
    pthread_mutex_lock(&state->mutex);
    state->candidates_tested += local_count;
    pthread_mutex_unlock(&state->mutex);

    return nullptr;
}

// COMMIT 3: Search Orchestration, Reporting & main()
// - searchLength(): spawns NUM_THREADS with static slice partitioning
// - recoverPassword(): outer loop over lengths 1..MAX_LENGTH
// - printReport(): mirrors serial.cpp format + shows thread count & speedup hint
// - main(): same 5-password test suite as serial.cpp

// ─── Search one password length using NUM_THREADS pthreads ───────────────────
bool searchLength(int length,
                  const string& target_hash,
                  string& result,
                  long long& candidates_tested)
{
    long long total = computeTotalCandidates(length);

    // ── Shared state init ──
    SharedState state;
    state.target_hash       = &target_hash;
    state.found             = false;
    state.candidates_tested = 0;
    pthread_mutex_init(&state.mutex, nullptr);

    // ── Static partitioning: divide [0, total) evenly across threads ──
    // Each thread gets a contiguous chunk → no dynamic scheduling overhead,
    // no work-stealing overhead. Works well here because all candidates take
    // roughly equal time to hash (uniform work per iteration).
    long long chunk = (total + NUM_THREADS - 1) / NUM_THREADS;

    vector<pthread_t>   threads(NUM_THREADS);  // thrad 8 create 
    vector<ThreadArgs>  thread_args(NUM_THREADS);   // agrgument create 

    for (int t = 0; t < NUM_THREADS; t++) {    // value assine value for argument
        thread_args[t].length      = length;
        thread_args[t].start_index = t * chunk;
        thread_args[t].end_index   = min((t + 1) * chunk, total);
        thread_args[t].state       = &state;

        pthread_create(&threads[t], nullptr, threadWorker, &thread_args[t]);   // create and process thereds
    }

    // ── Join all threads ──
    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], nullptr);   // waiting for process 

    pthread_mutex_destroy(&state.mutex);

    result            = state.result;
    candidates_tested += state.candidates_tested;
    return state.found;
}

// ─── Outer recovery loop (identical structure to serial.cpp) ─────────────────
bool recoverPassword(const string& target_hash,
                     string& recovered,
                     long long& total_tested,
                     double& elapsed_seconds)
{
    auto start   = high_resolution_clock::now();
    bool found   = false;
    total_tested = 0;

    for (int len = 1; len <= MAX_LENGTH && !found; len++) {
        cout << "[*] Searching length " << len
             << "  (candidates: " << computeTotalCandidates(len)
             << ")  [threads: " << NUM_THREADS << "]\n";

        found = searchLength(len, target_hash, recovered, total_tested);
    }

    auto end        = high_resolution_clock::now();
    elapsed_seconds = duration<double>(end - start).count();
    return found;
}

// ─── Performance report ───────────────────────────────────────────────────────
void printReport(bool found,
                 const string& recovered,
                 const string& target_hash,
                 long long total_tested,
                 double elapsed)
{
    cout << "\n========================================\n";
    cout << "     PERFORMANCE REPORT (Pthreads)      \n";
    cout << "========================================\n";
    cout << fixed << setprecision(6);
    cout << "  Status           : " << (found ? "FOUND" : "NOT FOUND") << "\n";
    if (found)
        cout << "  Recovered pass   : " << recovered << "\n";
    cout << "  Target hash      : " << target_hash           << "\n";
    cout << "  Candidates tested: " << total_tested          << "\n";
    cout << "  Elapsed time     : " << elapsed << " seconds\n";
    cout << "  Throughput       : "
         << (long long)(total_tested / elapsed) << " hashes/sec\n";
    cout << "  Threads used     : " << NUM_THREADS            << "\n";
    cout << "  Partitioning     : Static (contiguous chunks) \n";
    cout << "  Charset size     : " << CHARSET_SIZE           << "\n";
    cout << "  Max length tried : " << MAX_LENGTH             << "\n";
    cout << "========================================\n\n";

    if (found) {
        string verify = computeMD5(recovered);
        cout << "  Verification     : "
             << (verify == target_hash ? "PASSED ✓" : "FAILED ✗") << "\n";
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    string test_passwords[] = {"a", "ab", "abc", "z9", "abc1"};

    for (const string& pw : test_passwords) {
        string target_hash = computeMD5(pw);

        cout << "\n[TEST] Password: \"" << pw << "\"\n";
        cout << "[TEST] MD5 hash: "    << target_hash << "\n\n"; // hash හදනවා  // crack කරනවා   // results print කරනවා

        string    recovered;
        long long total_tested = 0;
        double    elapsed      = 0.0;

        bool found = recoverPassword(target_hash, recovered, total_tested, elapsed);
        printReport(found, recovered, target_hash, total_tested, elapsed);
    }

    return 0;
}



//                       Results print කරනවා — time, speed, verification