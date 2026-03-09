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

int main() {


    return 0;
}