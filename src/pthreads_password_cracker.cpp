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

// ─── Hash & index utilities (identical to serial.cpp) ────────────────────────
string computeMD5(const string& input) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(input.c_str()),
        input.size(), digest);

    char hex_str[33];
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        sprintf(&hex_str[i * 2], "%02x", digest[i]);
    hex_str[32] = '\0';
    return string(hex_str);
}

string indexToCandidate(long long index, int length) {
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

    long long local_count = 0;  // ← false-sharing avoidance: thread-local counter

    for (long long i = args->start_index; i < args->end_index; i++) {

        // Check every 1000 iterations whether another thread already found it
        if (i % 1000 == 0) {
            pthread_mutex_lock(&state->mutex);
            bool already_found = state->found;
            pthread_mutex_unlock(&state->mutex);
            if (already_found) break;  // Early exit
        }

        string candidate = indexToCandidate(i, args->length);
        string hash      = computeMD5(candidate);
        local_count++;

        if (hash == *state->target_hash) {
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