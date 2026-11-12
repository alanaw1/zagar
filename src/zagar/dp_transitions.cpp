#include "dp_transitions.hpp"

#include <algorithm>
#include <stdexcept>

namespace zagar {

namespace {

// Small Pascal table, big enough for any level count at the supported N.
std::vector<std::vector<uint64_t>> binomial_table(int n) {
    std::vector<std::vector<uint64_t>> B(
        static_cast<size_t>(n) + 1, std::vector<uint64_t>(static_cast<size_t>(n) + 1, 0)
    );
    for (int i = 0; i <= n; ++i) {
        B[static_cast<size_t>(i)][0] = 1;
        for (int k = 1; k <= i; ++k) {
            B[static_cast<size_t>(i)][static_cast<size_t>(k)] =
                B[static_cast<size_t>(i) - 1][static_cast<size_t>(k) - 1]
                + B[static_cast<size_t>(i) - 1][static_cast<size_t>(k)];
        }
    }
    return B;
}

uint64_t mul_checked(uint64_t a, uint64_t b) {
    uint64_t out;
    if (__builtin_mul_overflow(a, b, &out)) {
        throw std::overflow_error(
            "zagar: locus transition weight exceeded uint64; N is too large "
            "for the C++ backend"
        );
    }
    return out;
}

uint64_t add_checked(uint64_t a, uint64_t b) {
    uint64_t out;
    if (__builtin_add_overflow(a, b, &out)) {
        throw std::overflow_error(
            "zagar: locus transition weight exceeded uint64; N is too large "
            "for the C++ backend"
        );
    }
    return out;
}

struct Recursion {
    const std::vector<int>& hist;
    const std::vector<int>& levels;
    const std::vector<std::vector<uint64_t>>& B;
    std::vector<int>& work;
    std::unordered_map<std::string, uint64_t>& out;
    std::string key;                  // reused across leaves

    void run(size_t i, int s, uint64_t w) {
        if (i == levels.size()) {
            // Drop trailing zeros, keeping at least one byte.
            size_t end = work.size();
            while (end > 1 && work[end - 1] == 0) {
                --end;
            }
            key.assign(end + 1, '\0');
            for (size_t t = 0; t < end; ++t) {
                key[t] = static_cast<char>(static_cast<unsigned char>(work[t]));
            }
            key[end] = static_cast<char>(static_cast<unsigned char>(s));
            auto it = out.find(key);
            if (it == out.end()) {
                out.emplace(key, w);
            } else {
                it->second = add_checked(it->second, w);
            }
            return;
        }

        const int r = levels[i];
        const int h = hist[static_cast<size_t>(r)];
        const int max_k2 = (r >= 2) ? h : 0;
        for (int k2 = 0; k2 <= max_k2; ++k2) {
            const int max_k1 = (r >= 1) ? (h - k2) : 0;
            for (int k1 = 0; k1 <= max_k1; ++k1) {
                // C(h, k2) * C(h - k2, k1) * 2^k1
                uint64_t mult = mul_checked(
                    B[static_cast<size_t>(h)][static_cast<size_t>(k2)],
                    B[static_cast<size_t>(h - k2)][static_cast<size_t>(k1)]
                );
                mult = mul_checked(mult, uint64_t(1) << k1);

                work[static_cast<size_t>(r)] -= k1 + k2;
                if (k1) {
                    work[static_cast<size_t>(r - 1)] += k1;
                }
                if (k2) {
                    work[static_cast<size_t>(r - 2)] += k2;
                }

                run(i + 1, s + k1 + 2 * k2, mul_checked(w, mult));

                work[static_cast<size_t>(r)] += k1 + k2;
                if (k1) {
                    work[static_cast<size_t>(r - 1)] -= k1;
                }
                if (k2) {
                    work[static_cast<size_t>(r - 2)] -= k2;
                }
            }
        }
    }
};

}  // namespace

std::unordered_map<std::string, uint64_t> locus_transitions(
    const std::vector<int>& hist
) {
    if (hist.empty()) {
        throw std::invalid_argument("hist must not be empty");
    }
    int total = 0;
    int max_h = 0;
    for (int c : hist) {
        if (c < 0) {
            throw std::invalid_argument("hist counts must be non-negative");
        }
        total += c;
        max_h = std::max(max_h, c);
    }

    std::vector<int> levels;
    levels.reserve(hist.size());
    for (size_t r = 0; r < hist.size(); ++r) {
        if (hist[r]) {
            levels.push_back(static_cast<int>(r));
        }
    }

    const auto B = binomial_table(max_h);
    std::vector<int> work(hist);
    std::unordered_map<std::string, uint64_t> out;
    // One state per distinct (histogram, s); the count is modest, but
    // reserving avoids a handful of rehashes in the hot path.
    out.reserve(64);

    Recursion rec{hist, levels, B, work, out, std::string()};
    rec.run(0, 0, 1);
    (void)total;
    return out;
}

}  // namespace zagar
