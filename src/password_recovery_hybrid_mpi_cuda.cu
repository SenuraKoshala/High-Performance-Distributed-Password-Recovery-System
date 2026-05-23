/*
 * ============================================================
 *  High-Performance Password Recovery System
 *  Hybrid MPI + CUDA Implementation  —  Group 30
 *  EE7218 / EC7207  High Performance Computing
 * ============================================================
 *
 *  Build:
 *    nvcc -O2 -std=c++17 password_recovery_hybrid_mpi_cuda.cu \
 *         -lssl -lcrypto -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include \
 *         -L/usr/lib/x86_64-linux-gnu/openmpi/lib -o recover_mpi_cuda
 *
 *    Or with mpicxx wrapper:
 *    nvcc -O2 -std=c++17 password_recovery_hybrid_mpi_cuda.cu \
 *         -lssl -lcrypto $(mpicxx --showme:compile) $(mpicxx --showme:link) \
 *         -o recover_mpi_cuda
 *
 *  Run (e.g. 4 MPI processes, each using 1 GPU):
 *    mpirun -np 4 ./recover_mpi_cuda
 *
 * ============================================================
 *
 *  Two-Layer Parallelism Architecture:
 *    Layer 1 — MPI: Master (rank 0) distributes index ranges
 *              to workers across nodes.
 *    Layer 2 — CUDA: Each worker launches GPU kernels to
 *              brute-force hash candidates in parallel on GPU.
 *
 *  GPU Strategy (Brute Force only — best GPU fit):
 *    - Host divides the candidate space into batches
 *    - Each batch = CUDA kernel launch with thousands of threads
 *    - Device-side MD5 (RFC 1321): each CUDA thread computes
 *      MD5 for one candidate independently
 *    - atomicCAS on device flag when match found
 *    - Result copied back to host, reported to master via MPI
 *
 *  For Dict/Rule-Based: falls back to CPU (host-side) since
 *  string manipulation and variable-length words are not ideal
 *  for GPU. MPI still distributes work across processes.
 * ============================================================
 */

#include <mpi.h>
#include <openssl/evp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

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
   MPI tags & batch constants (same as other implementations)
───────────────────────────────────────────────────────────── */
constexpr int TAG_WORK  = 1;
constexpr int TAG_FOUND = 2;
constexpr int TAG_DONE  = 3;
constexpr int TAG_STOP  = 4;
constexpr int BATCH_SIZE = 200;
constexpr int MAX_WORD   = 256;
static const int BATCH_BUF = BATCH_SIZE * MAX_WORD;

/* ─────────────────────────────────────────────────────────────
   CUDA configuration
───────────────────────────────────────────────────────────── */
constexpr int CUDA_THREADS_PER_BLOCK = 256;
constexpr int CUDA_BATCH_SIZE = 1 << 20;  // ~1M candidates per kernel launch

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

/* ─────────────────────────────────────────────────────────────
   Host-side MD5 (OpenSSL EVP — for dict/rule-based on CPU)
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
   Timer / Progress / Result  (same as other files)
───────────────────────────────────────────────────────────── */
using Clock = std::chrono::steady_clock;
using TP    = std::chrono::time_point<Clock>;
double elapsed(TP t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

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

struct Result {
    std::string method;
    bool        found   = false;
    std::string cracked = "";
    long long   tested  = 0;
    double      secs    = 0.0;
    int         procs   = 1;
    std::string accel   = "CPU";
};

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
                  << "  |  Tested: " << BOLD << countStr << RESET
                  << "  |  Procs: " << BOLD << r.procs << RESET
                  << "  |  Accel: " << BOLD << r.accel << RESET << "\n";
    } else {
        std::cout << "  " << BOLD << RED << "✘  Not found" << RESET
                  << "  |  Time: " << BOLD << std::fixed << std::setprecision(2)
                  << r.secs << "s" << RESET
                  << "  |  Tested: " << BOLD << countStr << RESET
                  << "  |  Procs: " << BOLD << r.procs << RESET
                  << "  |  Accel: " << BOLD << r.accel << RESET << "\n";
    }
}

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
              << "  ╔══════════════════════════════════════════════════════════════════════╗\n"
              << "  ║                COMPARISON SUMMARY  (MPI + CUDA)                     ║\n"
              << "  ╠══════════════════════════════════════════════════════════════════════╣\n"
              << RESET << BOLD
              << "  ║  Password cracked   :  " << std::left << std::setw(48)
              << pw << BOLD << CYAN << "║\n"
              << "  ╠══════════════════╦════════════╦════════════╦════════════════╦════════╣\n"
              << "  ║ Method           ║  Result    ║  Time      ║  Candidates    ║ Accel  ║\n"
              << "  ╠══════════════════╬════════════╬════════════╬════════════════╬════════╣\n"
              << RESET;
    for (const auto& r : results) {
        std::string resultCol = r.found
            ? std::string(GREEN) + BOLD + "  Found!   " + RESET
            : std::string(RED) + " Not found " + RESET;
        std::ostringstream ts;
        ts << std::fixed << std::setprecision(2) << r.secs << "s";
        std::cout << "  ║ " << std::left << std::setw(16) << r.method << " ║"
                  << resultCol << " ║ " << std::right << std::setw(8) << ts.str()
                  << "   ║ " << std::setw(14) << fmt(r.tested) << " ║ "
                  << std::left << std::setw(6) << r.accel << " ║\n";
    }
    std::cout << BOLD << CYAN
              << "  ╚══════════════════╩════════════╩════════════╩════════════════╩════════╝\n"
              << RESET << "\n";
}

/* ═════════════════════════════════════════════════════════════
   DEVICE-SIDE MD5  (RFC 1321)
   Each CUDA thread calls device_md5() independently.
   No shared memory needed — fully parallel.

   MD5 processes data in 64-byte blocks. For short passwords
   (≤55 bytes), everything fits in a single block with padding.
═════════════════════════════════════════════════════════════ */

// MD5 round constants (F, G, H, I)
__device__ __forceinline__ uint32_t F(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
__device__ __forceinline__ uint32_t G(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
__device__ __forceinline__ uint32_t H(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
__device__ __forceinline__ uint32_t I(uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | ~z); }
__device__ __forceinline__ uint32_t ROTL(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

// Per-round shift amounts
__device__ __constant__ uint32_t d_S[64] = {
    7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
    5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
    4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
    6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
};

// Pre-computed T[i] = floor(2^32 * |sin(i+1)|)
__device__ __constant__ uint32_t d_T[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,
    0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,
    0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,
    0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,
    0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,
    0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,
    0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,
    0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,
    0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

/*
 * device_md5: compute MD5 of input (up to 55 bytes) into 4 uint32s
 *   input:  pointer to message bytes
 *   len:    message length
 *   digest: output array of 4 uint32_t (16 bytes = 128 bits)
 */
__device__ void device_md5(const uint8_t* input, int len, uint32_t* digest) {
    // Build padded 64-byte block (single block for len ≤ 55)
    uint32_t M[16];
    memset(M, 0, 64);
    uint8_t* block = (uint8_t*)M;
    for (int i = 0; i < len; ++i) block[i] = input[i];
    block[len] = 0x80;                         // append bit '1'
    M[14] = (uint32_t)(len * 8);               // length in bits (low 32)
    M[15] = 0;                                 // length in bits (high 32)

    // Initial hash values
    uint32_t a0 = 0x67452301;
    uint32_t b0 = 0xefcdab89;
    uint32_t c0 = 0x98badcfe;
    uint32_t d0 = 0x10325476;

    uint32_t a = a0, b = b0, c = c0, d = d0;

    // 64 rounds
    for (int i = 0; i < 64; ++i) {
        uint32_t f, g;
        if (i < 16)      { f = F(b,c,d); g = i; }
        else if (i < 32) { f = G(b,c,d); g = (5*i + 1) % 16; }
        else if (i < 48) { f = H(b,c,d); g = (3*i + 5) % 16; }
        else             { f = I(b,c,d); g = (7*i) % 16; }

        uint32_t temp = d;
        d = c;
        c = b;
        b = b + ROTL(a + f + d_T[i] + M[g], d_S[i]);
        a = temp;
    }

    digest[0] = a0 + a;
    digest[1] = b0 + b;
    digest[2] = c0 + c;
    digest[3] = d0 + d;
}

/*
 * Convert 4 uint32_t MD5 digest to 16 raw bytes for comparison
 * (little-endian, as per MD5 spec)
 */
__device__ void md5_to_bytes(const uint32_t* digest, uint8_t* out) {
    for (int i = 0; i < 4; ++i) {
        out[i*4 + 0] = (uint8_t)(digest[i]);
        out[i*4 + 1] = (uint8_t)(digest[i] >> 8);
        out[i*4 + 2] = (uint8_t)(digest[i] >> 16);
        out[i*4 + 3] = (uint8_t)(digest[i] >> 24);
    }
}

/* ─────────────────────────────────────────────────────────────
   Device charset (same a-z A-Z 0-9 = 62 chars)
───────────────────────────────────────────────────────────── */
__device__ __constant__ char d_charset[63] =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
__device__ __constant__ int d_charset_size = 62;

/*
 * Convert a linear index to a candidate password on device.
 * Same logic as host indexToCandidate but runs on GPU.
 */
__device__ void index_to_candidate(long long idx, int len, char* out) {
    for (int i = len - 1; i >= 0; --i) {
        out[i] = d_charset[idx % 62];
        idx /= 62;
    }
    out[len] = '\0';
}

/* ─────────────────────────────────────────────────────────────
   Host helper: parse hex MD5 string into 16 raw bytes
   for comparison on device
───────────────────────────────────────────────────────────── */
void hex_to_bytes(const std::string& hex, uint8_t* out) {
    for (int i = 0; i < 16; ++i) {
        unsigned int byte;
        sscanf(hex.c_str() + i * 2, "%02x", &byte);
        out[i] = (uint8_t)byte;
    }
}

/* ═════════════════════════════════════════════════════════════
   CUDA KERNEL: brute_force_kernel
   Each thread computes one candidate password from its global
   index, hashes it with device_md5, and compares against target.
   If match found, atomicCAS sets the found flag and copies
   the winning index.
═════════════════════════════════════════════════════════════ */
__global__ void brute_force_kernel(
    long long start_idx,    // first candidate index in this batch
    int       pw_len,       // password length to try
    const uint8_t* d_target,// 16-byte target MD5 hash on device
    int*      d_found,      // device flag: 0=not found, 1=found
    long long* d_found_idx  // index of found candidate
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long idx = start_idx + tid;

    // Early exit if another thread already found it
    if (*d_found) return;

    // Generate candidate from index
    char candidate[8];  // max 7 chars + null
    index_to_candidate(idx, pw_len, candidate);

    // Compute MD5
    uint32_t digest[4];
    device_md5((const uint8_t*)candidate, pw_len, digest);

    // Convert to bytes for comparison
    uint8_t hash_bytes[16];
    md5_to_bytes(digest, hash_bytes);

    // Compare with target
    bool match = true;
    for (int i = 0; i < 16; ++i) {
        if (hash_bytes[i] != d_target[i]) { match = false; break; }
    }

    if (match) {
        // atomicCAS: only first thread to find sets the flag
        if (atomicCAS(d_found, 0, 1) == 0) {
            *d_found_idx = idx;
        }
    }
}

/* ─────────────────────────────────────────────────────────────
   Host-side GPU launcher
   Searches all candidates of a given length on GPU.
   Returns true if found, fills result_pw.
───────────────────────────────────────────────────────────── */
bool gpu_brute_force_length(
    int pw_len,
    const uint8_t* h_target,  // 16-byte target hash (host)
    long long& total_tested,
    std::string& result_pw
) {
    // Total candidates for this length: 62^pw_len
    long long total = 1;
    for (int i = 0; i < pw_len; ++i) total *= 62;

    // Allocate device memory
    uint8_t*   d_target;
    int*       d_found;
    long long* d_found_idx;

    CUDA_CHECK(cudaMalloc(&d_target, 16));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_idx, sizeof(long long)));

    CUDA_CHECK(cudaMemcpy(d_target, h_target, 16, cudaMemcpyHostToDevice));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice));

    bool found = false;
    long long found_idx = -1;

    // Launch kernel in batches of CUDA_BATCH_SIZE
    for (long long offset = 0; offset < total; offset += CUDA_BATCH_SIZE) {
        long long batch = std::min((long long)CUDA_BATCH_SIZE, total - offset);
        int blocks = (int)((batch + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

        brute_force_kernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
            offset, pw_len, d_target, d_found, d_found_idx);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check if found
        int h_found = 0;
        CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
        total_tested += batch;

        if (h_found) {
            CUDA_CHECK(cudaMemcpy(&found_idx, d_found_idx, sizeof(long long), cudaMemcpyDeviceToHost));
            found = true;
            break;
        }
    }

    // Reconstruct password from index on host
    if (found && found_idx >= 0) {
        const char* charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        char pw[8] = {};
        long long idx = found_idx;
        for (int i = pw_len - 1; i >= 0; --i) {
            pw[i] = charset[idx % 62];
            idx /= 62;
        }
        result_pw = std::string(pw, pw_len);
    }

    cudaFree(d_target);
    cudaFree(d_found);
    cudaFree(d_found_idx);
    return found;
}

/* ═════════════════════════════════════════════════════════════
   METHOD 1 — BRUTE FORCE  (MPI + CUDA)
   Master distributes password-length ranges to workers.
   Each worker uses GPU to search its assigned range.
═════════════════════════════════════════════════════════════ */

void bfWorker(int rank, const std::string& target_hex, int maxLen) {
    // Each worker selects GPU based on rank (multi-GPU support)
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) cudaSetDevice((rank - 1) % deviceCount);

    uint8_t target_bytes[16];
    hex_to_bytes(target_hex, target_bytes);

    while (true) {
        // Receive work: password length to search, or -1 = STOP
        int pw_len;
        MPI_Recv(&pw_len, 1, MPI_INT, 0, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (pw_len == -1) break;

        long long local_tested = 0;
        std::string found_pw;
        bool found = gpu_brute_force_length(pw_len, target_bytes, local_tested, found_pw);

        if (found) {
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

Result bfMaster(const std::string& target, int maxLen, int nprocs) {
    Result r;
    r.method = "Brute Force";
    r.procs  = nprocs;
    r.accel  = "CUDA";
    int nextLen = 1, active = 0;
    long long total = 0;
    bool found = false;
    std::string cracked;
    TP t0 = Clock::now();

    // Seed workers with password lengths
    for (int w = 1; w < nprocs && nextLen <= maxLen; ++w) {
        MPI_Send(&nextLen, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
        ++nextLen; ++active;
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
                int len = -1;
                MPI_Send(&len, 1, MPI_INT, w, TAG_WORK, MPI_COMM_WORLD);
            }
            break;
        } else if (tag == TAG_DONE) {
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += cnt;
            progress(total, elapsed(t0), "");
            if (nextLen <= maxLen) {
                MPI_Send(&nextLen, 1, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);
                ++nextLen;
            } else {
                int len = -1;
                MPI_Send(&len, 1, MPI_INT, src, TAG_WORK, MPI_COMM_WORLD);
                --active;
            }
        }
    }
    r.found = found; r.cracked = cracked; r.tested = total; r.secs = elapsed(t0);
    return r;
}

Result runBruteForce(const std::string& target, int maxLen, int rank, int nprocs) {
    if (rank == 0) {
        std::cout << "\n" << BOLD << CYAN
                  << "  ┌──────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Brute Force  [MPI+CUDA]           │\n"
                  << "  └──────────────────────────────────────────────┘\n" << RESET;
        std::cout << GRAY << "  Processes: " << nprocs
                  << "  |  Accelerator: CUDA GPU"
                  << "  |  Max length: " << maxLen << "\n" << RESET;
        return bfMaster(target, maxLen, nprocs);
    } else {
        bfWorker(rank, target, maxLen);
        return Result{};
    }
}

/* ═════════════════════════════════════════════════════════════
   HELPER FUNCTIONS for Dict/Rule-Based (CPU fallback)
═════════════════════════════════════════════════════════════ */
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
    for (const std::string& s : {"1","2","12","123","1234","12345","123456",
            "0","!","@","#","2022","2023","2024","2025","99","01","007","69"}) {
        v.push_back(w + s); v.push_back(cap + s);
    }
    for (const std::string& p : {"1","my","the","123"}) v.push_back(p + w);
    std::string leet = w;
    for (char& c : leet) switch(tolower((unsigned char)c)) {
        case 'a': c='@'; break; case 'e': c='3'; break;
        case 'i': c='1'; break; case 'o': c='0'; break; case 's': c='$'; break;
    }
    v.push_back(leet);
    for (const std::string& s : {"1","123","!"}) v.push_back(leet + s);
    std::string cleet = leet;
    if (!cleet.empty()) cleet[0] = (char)toupper((unsigned char)w[0]);
    v.push_back(cleet);
    for (const std::string& s : {"1","123","!","2024"}) v.push_back(cleet + s);
    std::string rev = w; std::reverse(rev.begin(), rev.end()); v.push_back(rev);
    v.push_back(w + w);
    return v;
}

void encodeBatch(const std::vector<std::string>& words, int start, int count, char* buf) {
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
void writeDemoWordlist(const std::string& path) {
    static const char* words[] = {
        "password","123456","qwerty","admin","letmein","welcome",
        "monkey","dragon","master","sunshine","princess","football",
        "shadow","superman","batman","trustno1","iloveyou","hello",
        "login","pass","secret","abc123","test","guest","root",
        "computer","internet","michael","jessica","hunter","ranger",
        "soccer","baseball","network","server","linux","windows",
        "cookie","summer","flower","chicken","pepper","cheese", nullptr
    };
    std::ofstream f(path);
    for (int i = 0; words[i]; ++i) f << words[i] << "\n";
}

/* ═════════════════════════════════════════════════════════════
   METHOD 2 — DICTIONARY ATTACK  (MPI, CPU fallback)
═════════════════════════════════════════════════════════════ */
void dictWorker(int rank, const std::string& target) {
    (void)rank; char buf[BATCH_BUF];
    while (true) {
        MPI_Status status;
        MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_STOP) break;
        auto batch = decodeBatch(buf);
        if (batch.empty()) break;
        long long local_tested = 0; bool found = false; std::string found_pw;
        for (const auto& word : batch) {
            ++local_tested;
            if (md5(word) == target) { found = true; found_pw = word; break; }
        }
        if (found) {
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

Result dictMaster(const std::string& target, const std::vector<std::string>& words, int nprocs) {
    Result r; r.method = "Dictionary"; r.procs = nprocs; r.accel = "CPU";
    long long total = 0; int nextIdx = 0, active = 0; bool found = false;
    std::string cracked; long long N = (long long)words.size();
    TP t0 = Clock::now(); char buf[BATCH_BUF];
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
            for (int w = 1; w < nprocs; ++w) { if (w != src) MPI_Send(buf, BATCH_BUF, MPI_CHAR, w, TAG_STOP, MPI_COMM_WORLD); }
            break;
        } else if (tag == TAG_DONE) {
            long long cnt;
            MPI_Recv(&cnt, 1, MPI_LONG_LONG_INT, src, TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += cnt;
            int count = (int)std::min((long long)BATCH_SIZE, N - nextIdx);
            if (count > 0) { encodeBatch(words, nextIdx, count, buf); MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_WORK, MPI_COMM_WORLD); nextIdx += count; }
            else { MPI_Send(buf, BATCH_BUF, MPI_CHAR, src, TAG_STOP, MPI_COMM_WORLD); --active; }
        }
    }
    r.found = found; r.cracked = cracked; r.tested = total; r.secs = elapsed(t0);
    return r;
}

Result runDictionary(const std::string& target, const std::string& wordlistPath, int rank, int nprocs) {
    if (rank == 0) {
        std::cout << "\n" << BOLD << YELLOW
                  << "  ┌──────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Dictionary Attack [MPI, CPU]      │\n"
                  << "  └──────────────────────────────────────────────┘\n" << RESET;
        std::vector<std::string> words;
        { std::ifstream file(wordlistPath);
          if (!file) { std::cout << RED << "  [!] Wordlist not found.\n" << RESET;
              char buf[BATCH_BUF] = {}; for (int w=1;w<nprocs;++w) MPI_Send(buf,BATCH_BUF,MPI_CHAR,w,TAG_STOP,MPI_COMM_WORLD);
              Result r; r.method="Dictionary"; r.procs=nprocs; return r; }
          std::string line;
          while (std::getline(file,line)) { if (!line.empty()&&line.back()=='\r') line.pop_back(); if (!line.empty()) words.push_back(std::move(line)); }
        }
        std::cout << GRAY << "  Loaded " << words.size() << " words  |  Procs: " << nprocs << "  |  CPU fallback\n" << RESET;
        return dictMaster(target, words, nprocs);
    } else { dictWorker(rank, target); return Result{}; }
}

/* ═════════════════════════════════════════════════════════════
   METHOD 3 — RULE-BASED ATTACK  (MPI, CPU fallback)
═════════════════════════════════════════════════════════════ */
void ruleWorker(int rank, const std::string& target) {
    (void)rank; char buf[BATCH_BUF];
    while (true) {
        MPI_Status status;
        MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TAG_STOP) break;
        auto batch = decodeBatch(buf); if (batch.empty()) break;
        long long local_tested = 0; bool found = false; std::string found_pw;
        for (const auto& word : batch) {
            if (found) break;
            for (const auto& c : mutate(word)) {
                ++local_tested;
                if (md5(c) == target) { found = true; found_pw = c; break; }
            }
        }
        if (found) {
            char pwbuf[MAX_WORD] = {};
            std::strncpy(pwbuf, found_pw.c_str(), MAX_WORD - 1);
            MPI_Send(pwbuf, MAX_WORD, MPI_CHAR, 0, TAG_FOUND, MPI_COMM_WORLD);
            MPI_Recv(buf, BATCH_BUF, MPI_CHAR, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        } else { MPI_Send(&local_tested, 1, MPI_LONG_LONG_INT, 0, TAG_DONE, MPI_COMM_WORLD); }
    }
}

Result runRuleBased(const std::string& target, const std::string& wordlistPath, int rank, int nprocs) {
    if (rank == 0) {
        std::cout << "\n" << BOLD << MAGENTA
                  << "  ┌──────────────────────────────────────────────┐\n"
                  << "  │   METHOD : Rule-Based Attack [MPI, CPU]      │\n"
                  << "  └──────────────────────────────────────────────┘\n" << RESET;
        std::vector<std::string> words;
        { std::ifstream file(wordlistPath);
          if (!file) { std::cout << RED << "  [!] Wordlist not found.\n" << RESET;
              char buf[BATCH_BUF]={}; for(int w=1;w<nprocs;++w) MPI_Send(buf,BATCH_BUF,MPI_CHAR,w,TAG_STOP,MPI_COMM_WORLD);
              Result r; r.method="Rule-Based"; r.procs=nprocs; return r; }
          std::string line;
          while(std::getline(file,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();if(!line.empty())words.push_back(std::move(line));}
        }
        std::cout << GRAY << "  Loaded " << words.size() << " words  |  Procs: " << nprocs << "  |  CPU fallback\n" << RESET;
        // Reuse dictMaster logic — worker handles mutations
        Result r; r.method="Rule-Based"; r.procs=nprocs; r.accel="CPU";
        long long total=0; int nextIdx=0,active=0; bool found=false; std::string cracked;
        long long N=(long long)words.size(); TP t0=Clock::now(); char buf[BATCH_BUF];
        for(int w=1;w<nprocs;++w){int count=(int)std::min((long long)BATCH_SIZE,N-nextIdx);if(count<=0)break;
            encodeBatch(words,nextIdx,count,buf);MPI_Send(buf,BATCH_BUF,MPI_CHAR,w,TAG_WORK,MPI_COMM_WORLD);nextIdx+=count;++active;}
        while(active>0){MPI_Status st;MPI_Probe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&st);
            int src=st.MPI_SOURCE,tag=st.MPI_TAG;
            if(tag==TAG_FOUND){char pwbuf[MAX_WORD]={};MPI_Recv(pwbuf,MAX_WORD,MPI_CHAR,src,TAG_FOUND,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                cracked=pwbuf;found=true;MPI_Send(buf,BATCH_BUF,MPI_CHAR,src,TAG_STOP,MPI_COMM_WORLD);
                for(int w=1;w<nprocs;++w){if(w!=src)MPI_Send(buf,BATCH_BUF,MPI_CHAR,w,TAG_STOP,MPI_COMM_WORLD);}break;
            }else if(tag==TAG_DONE){long long cnt;MPI_Recv(&cnt,1,MPI_LONG_LONG_INT,src,TAG_DONE,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                total+=cnt;int count=(int)std::min((long long)BATCH_SIZE,N-nextIdx);
                if(count>0){encodeBatch(words,nextIdx,count,buf);MPI_Send(buf,BATCH_BUF,MPI_CHAR,src,TAG_WORK,MPI_COMM_WORLD);nextIdx+=count;}
                else{MPI_Send(buf,BATCH_BUF,MPI_CHAR,src,TAG_STOP,MPI_COMM_WORLD);--active;}}}
        r.found=found;r.cracked=cracked;r.tested=total;r.secs=elapsed(t0);return r;
    } else { ruleWorker(rank, target); return Result{}; }
}

/* ─────────────────────────────────────────────────────────────
   Interactive menu (rank 0 only)
───────────────────────────────────────────────────────────── */
int printMenu() {
    std::cout << "\n" << BOLD << "  Select attack mode:\n" << RESET << "\n"
              << "    " << CYAN    << "[1]" << RESET << "  Brute Force  " << GRAY << "(GPU accelerated)" << RESET << "\n\n"
              << "    " << YELLOW  << "[2]" << RESET << "  Dictionary Attack  " << GRAY << "(CPU)" << RESET << "\n\n"
              << "    " << MAGENTA << "[3]" << RESET << "  Rule-Based Attack  " << GRAY << "(CPU)" << RESET << "\n\n"
              << "    " << WHITE   << "[4]" << RESET << "  Run All Three  "
              << GRAY << "(comparison table)" << RESET << "\n\n"
              << "  Your choice [1-4]: " << BOLD;
    int choice = 0; std::string line;
    std::getline(std::cin, line); std::cout << RESET;
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

    // Each worker selects its GPU
    if (rank > 0) {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 0) cudaSetDevice((rank - 1) % deviceCount);
    }

    char target_buf[33] = {}, wordlist_buf[512] = {}, pw_buf[MAX_WORD] = {};
    int maxLen = 0, choice = 0;

    if (rank == 0) {
        std::cout << BOLD << CYAN
            << "\n"
            << "  ╔═══════════════════════════════════════════════════════╗\n"
            << "  ║   Password Recovery System  –  MPI+CUDA Hybrid       ║\n"
            << "  ║   EE7218 / EC7207  ·  High Performance Computing     ║\n"
            << "  ║   Group 30                                            ║\n"
            << "  ╚═══════════════════════════════════════════════════════╝\n"
            << RESET << "\n";

        // Show GPU info
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        std::cout << GRAY << "  MPI processes  : " << BOLD << nprocs << RESET << "\n"
                  << GRAY << "  CUDA GPUs      : " << BOLD << deviceCount << RESET << "\n";
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << GRAY << "    GPU " << i << ": " << BOLD << prop.name
                      << RESET << GRAY << "  (" << prop.multiProcessorCount << " SMs, "
                      << prop.totalGlobalMem / (1024*1024) << " MB)\n" << RESET;
        }

        std::cout << "\n  Enter the password to crack: " << BOLD;
        std::string password; std::getline(std::cin, password);
        std::cout << RESET;
        if (password.empty()) {
            std::cout << RED << "\n  [!] No password entered.\n" << RESET;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::string target = md5(password);
        std::strncpy(target_buf, target.c_str(), 32);
        std::strncpy(pw_buf, password.c_str(), MAX_WORD - 1);

        std::cout << "\n  Password : " << BOLD << password << RESET
                  << "\n  MD5 hash : " << BOLD << target << RESET << "\n"
                  << GRAY << "\n  ┄ Brute force runs on GPU; Dict/Rule use CPU. ┄\n" << RESET;

        std::string wordlist = "wordlist.txt";
        { std::ifstream check(wordlist);
          if (!check) { wordlist = "/tmp/demo_wordlist.txt"; writeDemoWordlist(wordlist);
              std::cout << GRAY << "\n  No wordlist.txt — using demo wordlist.\n" << RESET;
          } else { std::cout << GRAY << "\n  Wordlist: " << wordlist << RESET << "\n"; }
        }
        std::strncpy(wordlist_buf, wordlist.c_str(), 511);
        maxLen = std::min((int)password.size(), 5);
        if ((int)password.size() > 5)
            std::cout << YELLOW << "\n  Note: brute force capped at length 5.\n" << RESET;
        choice = printMenu();
        while (choice < 1 || choice > 4) {
            std::cout << RED << "  Enter 1-4: " << RESET;
            std::string line; std::getline(std::cin, line);
            if (!line.empty()) choice = line[0] - '0';
        }
    }

    MPI_Bcast(target_buf, 33, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(wordlist_buf, 512, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::string target(target_buf), wordlist(wordlist_buf);
    std::vector<Result> results;

    if (choice == 1 || choice == 4) {
        Result r = runBruteForce(target, maxLen, rank, nprocs);
        if (rank == 0) { printResult(r); results.push_back(r); }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (choice == 2 || choice == 4) {
        Result r = runDictionary(target, wordlist, rank, nprocs);
        if (rank == 0) { printResult(r); results.push_back(r); }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (choice == 3 || choice == 4) {
        Result r = runRuleBased(target, wordlist, rank, nprocs);
        if (rank == 0) { printResult(r); results.push_back(r); }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0 && choice == 4) printSummary(results, std::string(pw_buf));
    else if (rank == 0) std::cout << "\n";

    MPI_Finalize();
    return 0;
}

