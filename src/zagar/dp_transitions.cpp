#include "dp_transitions.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

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

namespace {

std::vector<std::vector<cpp_int>> binomial_table_big(int n) {
    std::vector<std::vector<cpp_int>> B(
        static_cast<size_t>(n) + 1, std::vector<cpp_int>(static_cast<size_t>(n) + 1, 0)
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

// Trailing zeros dropped, at least one byte kept: `_trim` in zagar.exact_dp.
std::string trim_key(const std::vector<int>& h) {
    size_t end = h.size();
    while (end > 1 && h[end - 1] == 0) {
        --end;
    }
    std::string k(end, '\0');
    for (size_t t = 0; t < end; ++t) {
        k[t] = static_cast<char>(static_cast<unsigned char>(h[t]));
    }
    return k;
}

// As Recursion above, but the locus weight and the layer weight are applied at
// the leaf and the result accumulated straight into the next layer.  No
// intermediate transition table exists, and the next layer merges duplicate
// states as a side effect of being a map.
struct LayerStep {
    const std::vector<int>& hist;
    const std::vector<int>& levels;
    const std::vector<std::vector<uint64_t>>& B;
    std::vector<int>& work;
    const std::vector<cpp_int>& wt;     // wt[s] = C(s, phi1_p) C(2N - s, phi0_p)
    int lo;
    int hi;
    int reach;
    int total;
    const cpp_int& weight;
    std::unordered_map<std::string, cpp_int>& nxt;
    std::string key;

    void run(size_t i, int s, uint64_t w) {
        if (i == levels.size()) {
            const int left = total - s;
            if (left < lo || left > hi) {
                return;                     // the remaining loci cannot absorb it
            }
            if (s >= static_cast<int>(wt.size())
                    || wt[static_cast<size_t>(s)] == 0) {
                return;                     // this locus admits no such column sum
            }
            size_t end = work.size();
            while (end > 1 && work[end - 1] == 0) {
                --end;
            }
            if (static_cast<int>(end) - 1 > reach) {
                return;                     // a row would be left needing too much
            }
            key.assign(end, '\0');
            for (size_t t = 0; t < end; ++t) {
                key[t] = static_cast<char>(static_cast<unsigned char>(work[t]));
            }
            nxt[key] += weight * wt[static_cast<size_t>(s)] * w;
            return;
        }

        const int r = levels[i];
        const int h = hist[static_cast<size_t>(r)];
        const int max_k2 = (r >= 2) ? h : 0;
        for (int k2 = 0; k2 <= max_k2; ++k2) {
            const int max_k1 = (r >= 1) ? (h - k2) : 0;
            for (int k1 = 0; k1 <= max_k1; ++k1) {
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

std::string count_A12_dp(
    int N,
    const std::vector<int>& a_vec,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1,
    bool reorder_loci
) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer");
    }
    if (N > DP_TRANSITIONS_MAX_N) {
        throw std::invalid_argument(
            "N exceeds the C++ backend's 64-bit multiplicity bound; use the "
            "pure-Python backend"
        );
    }
    if (static_cast<int>(a_vec.size()) != N) {
        throw std::invalid_argument("a_vec must have length N");
    }
    if (phi0.size() != phi1.size()) {
        throw std::invalid_argument("phi0 and phi1 must have the same length");
    }

    const int P = static_cast<int>(phi0.size());
    long long sum_a = 0;
    for (int a : a_vec) {
        sum_a += a;
    }

    const bool phis_zero =
        std::all_of(phi0.begin(), phi0.end(), [](int x) { return x == 0; })
        && std::all_of(phi1.begin(), phi1.end(), [](int x) { return x == 0; });
    if (sum_a == 0 && phis_zero) {
        return "1";                         // A and X both forced to zero
    }
    if (P == 0) {
        return sum_a == 0 ? "1" : "0";
    }

    // Narrow admissible windows prune soonest.
    std::vector<std::pair<int, int>> loci;      // (phi0_p, phi1_p)
    loci.reserve(static_cast<size_t>(P));
    for (int p = 0; p < P; ++p) {
        loci.emplace_back(phi0[static_cast<size_t>(p)], phi1[static_cast<size_t>(p)]);
    }
    if (reorder_loci) {
        std::stable_sort(
            loci.begin(), loci.end(),
            [&](const std::pair<int, int>& x, const std::pair<int, int>& y) {
                return (2 * N - x.first) - x.second < (2 * N - y.first) - y.second;
            }
        );
    }

    const auto Bb = binomial_table_big(2 * N);
    std::vector<std::vector<cpp_int>> colw(
        static_cast<size_t>(P),
        std::vector<cpp_int>(static_cast<size_t>(2 * N) + 1, 0)
    );
    for (int p = 0; p < P; ++p) {
        const int ph0 = loci[static_cast<size_t>(p)].first;
        const int ph1 = loci[static_cast<size_t>(p)].second;
        for (int s = 0; s <= 2 * N; ++s) {
            colw[static_cast<size_t>(p)][static_cast<size_t>(s)] =
                Bb[static_cast<size_t>(s)][static_cast<size_t>(ph1)]
                * Bb[static_cast<size_t>(2 * N - s)][static_cast<size_t>(ph0)];
        }
    }

    std::vector<int> suf_lo(static_cast<size_t>(P) + 1, 0);
    std::vector<int> suf_hi(static_cast<size_t>(P) + 1, 0);
    for (int p = P - 1; p >= 0; --p) {
        suf_lo[static_cast<size_t>(p)] = suf_lo[static_cast<size_t>(p) + 1]
            + loci[static_cast<size_t>(p)].second;
        suf_hi[static_cast<size_t>(p)] = suf_hi[static_cast<size_t>(p) + 1]
            + (2 * N - loci[static_cast<size_t>(p)].first);
    }

    const int width = *std::max_element(a_vec.begin(), a_vec.end()) + 1;
    std::vector<int> start(static_cast<size_t>(width), 0);
    for (int a : a_vec) {
        start[static_cast<size_t>(a)] += 1;
    }

    std::unordered_map<std::string, cpp_int> layer;
    layer[trim_key(start)] = 1;
    const auto Bs = binomial_table(N);

    for (int p = 0; p < P; ++p) {
        std::unordered_map<std::string, cpp_int> nxt;
        nxt.reserve(layer.size() * 2 + 16);
        const auto& wt = colw[static_cast<size_t>(p)];
        const int lo = suf_lo[static_cast<size_t>(p) + 1];
        const int hi = suf_hi[static_cast<size_t>(p) + 1];
        const int reach = 2 * (P - p - 1);

        for (const auto& kv : layer) {
            std::vector<int> hist(kv.first.size());
            int total = 0;
            for (size_t i = 0; i < kv.first.size(); ++i) {
                hist[i] = static_cast<int>(static_cast<unsigned char>(kv.first[i]));
                total += static_cast<int>(i) * hist[i];
            }
            std::vector<int> levels;
            levels.reserve(hist.size());
            for (size_t r = 0; r < hist.size(); ++r) {
                if (hist[r]) {
                    levels.push_back(static_cast<int>(r));
                }
            }
            std::vector<int> work(hist);
            LayerStep step{hist, levels, Bs, work, wt, lo, hi, reach, total,
                           kv.second, nxt, std::string()};
            step.run(0, 0, 1);
        }

        layer = std::move(nxt);
        if (layer.empty()) {
            return "0";
        }
    }

    // All row sums exhausted: the histogram with every row at remaining sum 0.
    const std::string final_key(1, static_cast<char>(static_cast<unsigned char>(N)));
    auto it = layer.find(final_key);
    if (it == layer.end()) {
        return "0";
    }
    return it->second.convert_to<std::string>();
}

}  // namespace zagar
