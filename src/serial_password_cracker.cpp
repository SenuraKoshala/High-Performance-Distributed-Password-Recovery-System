#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <openssl/md5.h>

using namespace std;
using namespace chrono;

const string CHARSET   = "abcdefghijklmnopqrstuvwxyz0123456789";
const int    MAX_LENGTH = 5;   // Maximum password length to try
const int    CHARSET_SIZE = CHARSET.size();


string computeMD5(const string& input) {
    unsigned char digest[MD5_DIGEST_LENGTH];   // 16 bytes

    // OpenSSL MD5: hashes `input` and writes result to `digest`
    MD5(reinterpret_cast<const unsigned char*>(input.c_str()),
        input.size(),
        digest);

    // Convert raw bytes to a readable hex string (e.g. "900150983cd24fb0...")
    char hex_str[33];
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++)
        sprintf(&hex_str[i * 2], "%02x", digest[i]);
    hex_str[32] = '\0';

    return string(hex_str);
}

string indexToCandidate(long long index, int length) {
    string candidate(length, CHARSET[0]);   // fill with first char

    // Work backwards from the last character position
    for (int i = length - 1; i >= 0; i--) {
        candidate[i] = CHARSET[index % CHARSET_SIZE];
        index /= CHARSET_SIZE;
    }
    return candidate;
}

long long computeTotalCandidates(int length) {
    long long total = 1;
    for (int i = 0; i < length; i++)
        total *= CHARSET_SIZE;
    return total;
}

bool searchLength(int length,
                  const string& target_hash,
                  string& result,
                  long long& candidates_tested)
{
    long long total = computeTotalCandidates(length);

    bool found = false;

    for (long long i = 0; i < total; i++) {


        // Generate candidate string from numeric index
        string candidate = indexToCandidate(i, length);

        // Hash the candidate
        string hash = computeMD5(candidate);

        // Increment the counter
        candidates_tested++;

        // Compare with target
        if (hash == target_hash) {
            
            found  = true;
            result = candidate;
            break;  
        }
    }

    return found;
}

bool recoverPassword(const string& target_hash,
                     string& recovered,
                     long long& total_tested,
                     double& elapsed_seconds)
{
    auto start = high_resolution_clock::now();
    bool found = false;
    total_tested = 0;

    for (int len = 1; len <= MAX_LENGTH && !found; len++) {
        cout << "[*] Searching length " << len
             << "  (candidates: " << computeTotalCandidates(len) << ")\n";

        // [OPENMP] The searchLength function contains the
        // parallel for loop. No changes needed here in the
        // outer loop for the shared-memory version.
        found = searchLength(len, target_hash, recovered, total_tested);
    }

    auto end   = high_resolution_clock::now();
    elapsed_seconds = duration<double>(end - start).count();
    return found;
}

void printReport(bool found,
                 const string& recovered,
                 const string& target_hash,
                 long long total_tested,
                 double elapsed)
{
    cout << "\n========================================\n";
    cout << "       PERFORMANCE REPORT (Serial)      \n";
    cout << "========================================\n";
    cout << fixed << setprecision(6);
    cout << "  Status          : " << (found ? "FOUND" : "NOT FOUND") << "\n";
    if (found)
        cout << "  Recovered pass  : " << recovered << "\n";
    cout << "  Target hash     : " << target_hash      << "\n";
    cout << "  Candidates tested: " << total_tested    << "\n";
    cout << "  Elapsed time    : " << elapsed << " seconds\n";
    cout << "  Throughput      : "
         << (long long)(total_tested / elapsed) << " hashes/sec\n";
    cout << "  Charset size    : " << CHARSET_SIZE     << "\n";
    cout << "  Max length tried: " << MAX_LENGTH       << "\n";
    cout << "========================================\n\n";

    // Verification check (required in your report: 100% match)
    if (found) {
        string verify = computeMD5(recovered);
        cout << "  Verification    : "
             << (verify == target_hash ? "PASSED ✓" : "FAILED ✗") << "\n";
    }
}


int main() {

    string test_passwords[] = {"a", "ab", "abc", "z9", "abc1"};

    for (const string& pw : test_passwords) {
        string target_hash = computeMD5(pw);

        cout << "\n[TEST] Password: \"" << pw << "\"\n";
        cout << "[TEST] MD5 hash: "   << target_hash << "\n\n";

        string   recovered;
        long long total_tested  = 0;
        double    elapsed       = 0.0;

        bool found = recoverPassword(target_hash,
                                     recovered,
                                     total_tested,
                                     elapsed);

        printReport(found, recovered, target_hash, total_tested, elapsed);
    }

    return 0;
}