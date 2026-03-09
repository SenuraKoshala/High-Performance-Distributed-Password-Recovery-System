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
