/*
 * ============================================================
 *  common.h  —  Shared utilities for Password Recovery System
 *  EE7218 / EC7207  High Performance Computing  —  Group 30
 * ============================================================
 *
 *  Contains all helper/utility code shared across implementations:
 *    - ANSI colour macros
 *    - MPI tag & batch constants
 *    - MD5 hashing (OpenSSL EVP)
 *    - Timer, progress display, Result struct, result printing
 *    - Password mutation rules (for rule-based attack)
 *    - Batch encode/decode for MPI message packing
 *    - Demo wordlist generator
 *    - Interactive menu
 * ============================================================
 */

#ifndef COMMON_H
#define COMMON_H

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

#include <openssl/evp.h>

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
inline std::string md5(const std::string& s) {
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
inline double elapsed(TP t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/* ─────────────────────────────────────────────────────────────
   Progress  (master / rank-0 only)
───────────────────────────────────────────────────────────── */
inline void progress(long long tested, double secs, const std::string& last) {
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
inline void printResult(const Result& r) {
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
inline void printSummary(const std::vector<Result>& results, const std::string& pw) {
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
inline std::vector<std::string> mutate(const std::string& w) {
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
inline void encodeBatch(const std::vector<std::string>& words,
                 int start, int count, char* buf) {
    std::memset(buf, 0, BATCH_BUF);
    for (int i = 0; i < count; ++i)
        std::strncpy(buf + i * MAX_WORD, words[start + i].c_str(), MAX_WORD - 1);
}

inline std::vector<std::string> decodeBatch(const char* buf) {
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
inline void writeDemoWordlist(const std::string& path) {
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

/* ─────────────────────────────────────────────────────────────
   Interactive menu  (rank 0 only)
───────────────────────────────────────────────────────────── */
inline int printMenu() {
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

#endif // COMMON_H
