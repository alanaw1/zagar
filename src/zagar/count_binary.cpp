// Refactored from Rcpp version to pure C++ with pybind11 bindings
// Original: count_binary.cpp (Rcpp/RcppParallel)

#include "count_binary.hpp"

#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cmath>

#ifdef ZAGAR_HAS_OPENMP
#include <omp.h>
#endif

namespace zagar {

using boost::multiprecision::cpp_int;

// Constants for overflow detection
static constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();
static constexpr uint64_t SAFE_MULT_THRESHOLD = 0xFFFFFFFFULL; // 2^32 - 1

namespace detail {

// --- Conjugate of q (with leading 0); L = max(q)+2
std::vector<int> conjugate(const std::vector<int>& q, int L) {
    std::vector<int> counts(L, 0);
    for (int v : q) {
        if (v < 0 || v >= L) {
            throw std::invalid_argument("q has values outside [0, L-1]");
        }
        counts[static_cast<size_t>(v)] += 1;
    }
    std::vector<int> c(L, 0);
    int acc = 0;
    for (int t = L - 1; t >= 1; --t) {
        acc += counts[static_cast<size_t>(t)];
        c[static_cast<size_t>(t)] = acc;
    }
    return c;
}

// --- Pad p as [0, p..., 0...0] with L trailing zeros
std::vector<int> pad_p(const std::vector<int>& p, int L) {
    std::vector<int> out;
    out.reserve(1 + p.size() + L);
    out.push_back(0);
    out.insert(out.end(), p.begin(), p.end());
    out.insert(out.end(), static_cast<size_t>(L), 0);
    return out;
}

// --- Precompute Pascal binomial table up to n (inclusive) using cpp_int
std::vector<std::vector<cpp_int>> precompute_binomial(int n) {
    std::vector<std::vector<cpp_int>> B(n + 1, std::vector<cpp_int>(n + 1));
    for (int i = 0; i <= n; ++i) {
        for (int k = 0; k <= n; ++k) {
            if (k == 0 || k == i) {
                B[i][k] = 1;
            } else if (k > i) {
                B[i][k] = 0;
            } else {
                B[i][k] = B[i - 1][k] + B[i - 1][k - 1];
            }
        }
    }
    return B;
}

// --- Precompute Pascal binomial table using uint64_t (for small n)
std::vector<std::vector<uint64_t>> precompute_binomial_u64(int n) {
    std::vector<std::vector<uint64_t>> B(n + 1, std::vector<uint64_t>(n + 1));
    for (int i = 0; i <= n; ++i) {
        for (int k = 0; k <= n; ++k) {
            if (k == 0 || k == i) {
                B[i][k] = 1;
            } else if (k > i) {
                B[i][k] = 0;
            } else {
                B[i][k] = B[i - 1][k] + B[i - 1][k - 1];
            }
        }
    }
    return B;
}

// --- Full recursion with uint64_t optimization
cpp_int count_recursion_binary(
    const std::vector<int>& p,
    std::vector<int>& c,
    int j, int k,
    int s_sum, int p_sum, int c_sum,
    MemoTable& table,
    int L,
    const std::vector<std::vector<cpp_int>>& B,
    const std::vector<std::vector<uint64_t>>& B_u64
) {
    if (s_sum == p[static_cast<size_t>(j)]) {
        auto it = table.find(c);
        if (it != table.end()) {
            return it->second;
        } else {
            cpp_int result = count_recursion_binary(
                p, c,
                j + 1, 1,
                0,
                p[static_cast<size_t>(j + 2)],
                c[1],
                table, L, B, B_u64
            );
            if (c[1] != p[static_cast<size_t>(j + 1)]) {
                table.emplace(c, result);
            }
            return result;
        }
    } else {
        int r_k = c[static_cast<size_t>(k)] - c[static_cast<size_t>(k + 1)];
        int GR_condition = c_sum - s_sum - p_sum;

        int smallest = std::max(0, p[static_cast<size_t>(j)] - s_sum - c[static_cast<size_t>(k + 1)]);
        int largest = std::min(std::min(r_k, p[static_cast<size_t>(j)] - s_sum), GR_condition);

        if (smallest > largest) {
            return cpp_int(0);
        }

        uint64_t result_u64 = 0;
        bool overflow = false;

        for (int s_k = smallest; s_k <= largest; ++s_k) {
            c[static_cast<size_t>(k)] -= s_k;

            cpp_int value = count_recursion_binary(
                p, c,
                j, k + 1,
                s_sum + s_k,
                p_sum + p[static_cast<size_t>(j + k + 1)],
                c_sum + c[static_cast<size_t>(k + 1)],
                table, L, B, B_u64
            );

            c[static_cast<size_t>(k)] += s_k;

            if (!overflow) {
                if (value <= U64_MAX) {
                    uint64_t val_u64 = static_cast<uint64_t>(value);
                    uint64_t binom_u64 = B_u64[static_cast<size_t>(r_k)][static_cast<size_t>(s_k)];

                    if (binom_u64 <= SAFE_MULT_THRESHOLD && val_u64 <= SAFE_MULT_THRESHOLD) {
                        uint64_t product = binom_u64 * val_u64;
                        if (result_u64 <= U64_MAX - product) {
                            result_u64 += product;
                            continue;
                        }
                    }
                }
                overflow = true;
            }

            cpp_int result = cpp_int(result_u64);
            result += B[static_cast<size_t>(r_k)][static_cast<size_t>(s_k)] * value;

            for (int s_k2 = s_k + 1; s_k2 <= largest; ++s_k2) {
                c[static_cast<size_t>(k)] -= s_k2;

                cpp_int value2 = count_recursion_binary(
                    p, c,
                    j, k + 1,
                    s_sum + s_k2,
                    p_sum + p[static_cast<size_t>(j + k + 1)],
                    c_sum + c[static_cast<size_t>(k + 1)],
                    table, L, B, B_u64
                );

                c[static_cast<size_t>(k)] += s_k2;
                result += B[static_cast<size_t>(r_k)][static_cast<size_t>(s_k2)] * value2;
            }
            return result;
        }

        return cpp_int(result_u64);
    }
}

} // namespace detail

// --- Main API: count_binary_matrices ---
CountResult count_binary_matrices(
    const std::vector<int>& p_in,
    const std::vector<int>& q_in,
    bool sort_p_desc
) {
    std::vector<int> p = p_in;
    std::vector<int> q = q_in;

    if (p.empty() || q.empty()) {
        throw std::invalid_argument("p and q must be non-empty.");
    }
    if (*std::min_element(p.begin(), p.end()) < 0 ||
        *std::min_element(q.begin(), q.end()) < 0) {
        throw std::invalid_argument("p and q must be nonnegative integers.");
    }

    long long sum_p = std::accumulate(p.begin(), p.end(), 0LL);
    long long sum_q = std::accumulate(q.begin(), q.end(), 0LL);
    if (sum_p != sum_q) {
        return CountResult{"0", 0};
    }

    if (sort_p_desc) {
        std::sort(p.begin(), p.end(), std::greater<int>());
    }

    int n = static_cast<int>(q.size());
    int L = *std::max_element(q.begin(), q.end()) + 2;

    std::vector<int> c = detail::conjugate(q, L);
    std::vector<int> p_pad = detail::pad_p(p, L);

    auto B = detail::precompute_binomial(n);
    auto B_u64 = detail::precompute_binomial_u64(n);

    MemoTable table;
    table.reserve(1 << 12);

    std::vector<int> root(L, 0);
    table.emplace(root, cpp_int(1));

    cpp_int number = detail::count_recursion_binary(
        p_pad, c, 1, 1, 0,
        p_pad[2],
        c[1],
        table, L, B, B_u64
    );

    table[c] = number;

    std::string number_str = number.convert_to<std::string>();
    return CountResult{number_str, static_cast<int>(table.size())};
}

// --- Helper for A12: process a single configuration ---
static cpp_int process_a12_config(
    const std::vector<int>& p,
    const std::vector<int>& cv1,
    const std::vector<int>& cv2,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1,
    int P,
    int N,
    int ncols,
    int L,
    MemoTable& table,
    const std::vector<std::vector<cpp_int>>& B_enum,
    const std::vector<std::vector<uint64_t>>& B_enum_u64,
    const std::vector<std::vector<cpp_int>>& B_weight,
    const std::vector<std::vector<uint64_t>>& B_weight_u64
) {
    std::vector<int> q(ncols);
    for (int i = 0; i < P; ++i) {
        q[i] = cv1[i];
        q[P + i] = cv2[i];
    }

    std::vector<int> p_pad = detail::pad_p(p, L);
    std::vector<int> c = detail::conjugate(q, L);

    cpp_int A_size = detail::count_recursion_binary(
        p_pad, c,
        1, 1,
        0,
        p_pad[2],
        c[1],
        table,
        L,
        B_enum,
        B_enum_u64
    );

    table[c] = A_size;

    // Compute weight
    uint64_t weight_u64 = 1;
    bool weight_overflow = false;

    for (int i = 0; i < P; ++i) {
        int s = cv1[i] + cv2[i];
        // n1 = count of A=1 positions, k1 = phi1 (count of A=1 AND X=1)
        // n2 = count of A=0 positions, k2 = phi0 (count of A=0 AND X=1)
        int n1 = s;
        int k1 = phi1[i];
        int n2 = 2 * N - s;
        int k2 = phi0[i];

        if (k1 < 0 || k1 > n1 || k2 < 0 || k2 > n2) {
            return cpp_int(0);
        }

        uint64_t b1 = B_weight_u64[n1][k1];
        uint64_t b2 = B_weight_u64[n2][k2];

        if (!weight_overflow) {
            if (weight_u64 <= SAFE_MULT_THRESHOLD && b1 <= SAFE_MULT_THRESHOLD) {
                weight_u64 *= b1;
                if (weight_u64 <= SAFE_MULT_THRESHOLD && b2 <= SAFE_MULT_THRESHOLD) {
                    weight_u64 *= b2;
                } else {
                    weight_overflow = true;
                }
            } else {
                weight_overflow = true;
            }
        }
    }

    if (!weight_overflow) {
        return cpp_int(weight_u64) * A_size;
    } else {
        cpp_int weight = 1;
        for (int i = 0; i < P; ++i) {
            int s = cv1[i] + cv2[i];
            // n1 = count of A=1 positions, k1 = phi1 (count of A=1 AND X=1)
            // n2 = count of A=0 positions, k2 = phi0 (count of A=0 AND X=1)
            int n1 = s;
            int k1 = phi1[i];
            int n2 = 2 * N - s;
            int k2 = phi0[i];
            weight *= B_weight[n1][k1];
            weight *= B_weight[n2][k2];
        }
        return weight * A_size;
    }
}

// --- Main API: getA12 ---
A12Result getA12(
    int N,
    const std::vector<int>& a_vec,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1,
    bool sort_p_desc
) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }

    int P = static_cast<int>(phi0.size());
    if (static_cast<int>(phi1.size()) != P) {
        throw std::invalid_argument("phi0 and phi1 must have the same length.");
    }

    std::vector<int> p = a_vec;
    if (sort_p_desc) {
        std::sort(p.begin(), p.end(), std::greater<int>());
    }

    long long sum_p = std::accumulate(p.begin(), p.end(), 0LL);

    // Coordinate-level feasibility check
    for (int i = 0; i < P; ++i) {
        if (phi1[i] > 2 * N - phi0[i]) {
            return A12Result{"0"};
        }
    }

    // Feasible set zero check
    for (int i = 0; i < P; ++i) {
        int Lp = phi1[i];
        int Up = 2 * N - phi0[i];

        long long S_p = 0;
        for (int s = Lp; s <= Up; ++s) {
            int a_min = std::max(0, s - N);
            int a_max = std::min(N, s);
            if (a_min <= a_max) {
                S_p += (a_max - a_min + 1);
            }
        }

        if (S_p == 0) {
            return A12Result{"0"};
        }
    }

    int ncols = 2 * P;

    auto B_enum = detail::precompute_binomial(ncols);
    auto B_enum_u64 = detail::precompute_binomial_u64(ncols);
    auto B_weight = detail::precompute_binomial(2 * N);
    auto B_weight_u64 = detail::precompute_binomial_u64(2 * N);

    // Enumerate all feasible configurations
    std::vector<std::pair<std::vector<int>, std::vector<int>>> all_configs;
    std::vector<int> v1(P, 0), v2(P, 0);

    std::function<void(int)> enumerate_configs = [&](int p_idx) {
        if (p_idx == P) {
            long long sum_q = 0;
            for (int i = 0; i < P; ++i) {
                sum_q += v1[i] + v2[i];
            }
            if (sum_q != sum_p) {
                return;
            }
            all_configs.emplace_back(v1, v2);
            return;
        }

        int Lp = phi1[p_idx];
        int Up = 2 * N - phi0[p_idx];

        for (int s = Lp; s <= Up; ++s) {
            int a_min = std::max(0, s - N);
            int a_max = std::min(N, s);
            if (a_min > a_max) continue;

            for (int a = a_min; a <= a_max; ++a) {
                v1[p_idx] = a;
                v2[p_idx] = s - a;
                enumerate_configs(p_idx + 1);
            }
        }
    };

    enumerate_configs(0);

    if (all_configs.empty()) {
        return A12Result{"0"};
    }

    // Group configurations by L value
    std::unordered_map<int, std::vector<size_t>> configs_by_L;
    for (size_t idx = 0; idx < all_configs.size(); ++idx) {
        const auto& cfg = all_configs[idx];
        int max_q = 0;
        for (int i = 0; i < P; ++i) {
            max_q = std::max(max_q, cfg.first[i]);
            max_q = std::max(max_q, cfg.second[i]);
        }
        int L = max_q + 2;
        configs_by_L[L].push_back(idx);
    }

    cpp_int total_sum = 0;

    for (auto& kv : configs_by_L) {
        int L = kv.first;
        const std::vector<size_t>& indices = kv.second;

        MemoTable table;
        table.reserve(1 << 12);

        std::vector<int> root(L, 0);
        table.emplace(root, cpp_int(1));

        for (size_t idx : indices) {
            const auto& cfg = all_configs[idx];
            cpp_int contrib = process_a12_config(
                p, cfg.first, cfg.second,
                phi0, phi1,
                P, N, ncols, L,
                table,
                B_enum, B_enum_u64,
                B_weight, B_weight_u64
            );
            total_sum += contrib;
        }
    }

    return A12Result{total_sum.convert_to<std::string>()};
}

// --- Main API: getA12_parallel ---
A12Result getA12_parallel(
    int N,
    const std::vector<int>& a_vec,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1,
    int n_threads,
    bool sort_p_desc
) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }

    int P = static_cast<int>(phi0.size());
    if (static_cast<int>(phi1.size()) != P) {
        throw std::invalid_argument("phi0 and phi1 must have the same length.");
    }

    std::vector<int> p = a_vec;
    if (sort_p_desc) {
        std::sort(p.begin(), p.end(), std::greater<int>());
    }

    long long sum_p = std::accumulate(p.begin(), p.end(), 0LL);

    // Coordinate-level feasibility check
    for (int i = 0; i < P; ++i) {
        if (phi1[i] > 2 * N - phi0[i]) {
            return A12Result{"0"};
        }
    }

    // Feasible set zero check
    for (int i = 0; i < P; ++i) {
        int Lp = phi1[i];
        int Up = 2 * N - phi0[i];

        long long S_p = 0;
        for (int s = Lp; s <= Up; ++s) {
            int a_min = std::max(0, s - N);
            int a_max = std::min(N, s);
            if (a_min <= a_max) {
                S_p += (a_max - a_min + 1);
            }
        }

        if (S_p == 0) {
            return A12Result{"0"};
        }
    }

    int ncols = 2 * P;

    auto B_enum = detail::precompute_binomial(ncols);
    auto B_enum_u64 = detail::precompute_binomial_u64(ncols);
    auto B_weight = detail::precompute_binomial(2 * N);
    auto B_weight_u64 = detail::precompute_binomial_u64(2 * N);

    // Build first-coordinate feasible pairs
    std::vector<std::pair<int, int>> first_pairs;
    {
        int L0 = phi1[0];
        int U0 = 2 * N - phi0[0];

        for (int s = L0; s <= U0; ++s) {
            int a_min = std::max(0, s - N);
            int a_max = std::min(N, s);
            if (a_min > a_max) continue;

            for (int a0 = a_min; a0 <= a_max; ++a0) {
                int b0 = s - a0;
                first_pairs.emplace_back(a0, b0);
            }
        }
    }

    if (first_pairs.empty()) {
        return A12Result{"0"};
    }

    std::vector<cpp_int> partial_sums(first_pairs.size());

#ifdef ZAGAR_HAS_OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t idx0 = 0; idx0 < first_pairs.size(); ++idx0) {
#else
    (void)n_threads; // Suppress unused warning
    for (size_t idx0 = 0; idx0 < first_pairs.size(); ++idx0) {
#endif
        cpp_int local_sum = 0;

        std::vector<int> v1(P, 0), v2(P, 0);
        v1[0] = first_pairs[idx0].first;
        v2[0] = first_pairs[idx0].second;

        // Enumerate configurations for this first pair
        std::vector<std::pair<std::vector<int>, std::vector<int>>> local_configs;

        std::function<void(int)> enumerate_suffix = [&](int p_idx) {
            if (p_idx == P) {
                long long sum_q = 0;
                for (int i = 0; i < P; ++i) {
                    sum_q += v1[i] + v2[i];
                }
                if (sum_q != sum_p) {
                    return;
                }
                local_configs.emplace_back(v1, v2);
                return;
            }

            int Lp = phi1[p_idx];
            int Up = 2 * N - phi0[p_idx];

            for (int s = Lp; s <= Up; ++s) {
                int a_min = std::max(0, s - N);
                int a_max = std::min(N, s);
                if (a_min > a_max) continue;

                for (int a_val = a_min; a_val <= a_max; ++a_val) {
                    v1[p_idx] = a_val;
                    v2[p_idx] = s - a_val;
                    enumerate_suffix(p_idx + 1);
                }
            }
        };

        enumerate_suffix(1);

        if (local_configs.empty()) {
            partial_sums[idx0] = 0;
            continue;
        }

        // Group by L value
        std::unordered_map<int, std::vector<size_t>> configs_by_L;
        for (size_t i = 0; i < local_configs.size(); ++i) {
            const auto& cfg = local_configs[i];
            int max_q = 0;
            for (int j = 0; j < P; ++j) {
                max_q = std::max(max_q, cfg.first[j]);
                max_q = std::max(max_q, cfg.second[j]);
            }
            int L = max_q + 2;
            configs_by_L[L].push_back(i);
        }

        // Process each L-group with shared memo table
        for (auto& kv : configs_by_L) {
            int L = kv.first;
            const std::vector<size_t>& indices = kv.second;

            MemoTable table;
            table.reserve(1 << 10);

            std::vector<int> root(L, 0);
            table.emplace(root, cpp_int(1));

            for (size_t cfg_idx : indices) {
                const auto& cfg = local_configs[cfg_idx];
                cpp_int contrib = process_a12_config(
                    p, cfg.first, cfg.second,
                    phi0, phi1,
                    P, N, ncols, L,
                    table,
                    B_enum, B_enum_u64,
                    B_weight, B_weight_u64
                );
                local_sum += contrib;
            }
        }

        partial_sums[idx0] = local_sum;
    }

    cpp_int total_sum = 0;
    for (size_t i = 0; i < partial_sums.size(); ++i) {
        total_sum += partial_sums[i];
    }

    return A12Result{total_sum.convert_to<std::string>()};
}

// --- Main API: getA12_pruning (with sum-based pruning and symmetry optimization) ---
// Combines two optimizations:
// 1. Sum-based pruning: tracks running sum during enumeration, prunes when target unreachable
// 2. Symmetry: (v1, v2) and (v2, v1) have same count/weight, only enumerate v1 <= v2
A12Result getA12_pruning(
    int N,
    const std::vector<int>& a_vec,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1,
    int n_threads,
    bool sort_p_desc
) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }

    int P = static_cast<int>(phi0.size());
    if (static_cast<int>(phi1.size()) != P) {
        throw std::invalid_argument("phi0 and phi1 must have the same length.");
    }

    std::vector<int> p = a_vec;
    if (sort_p_desc) {
        std::sort(p.begin(), p.end(), std::greater<int>());
    }

    long long sum_p = std::accumulate(p.begin(), p.end(), 0LL);

    // Coordinate-level feasibility check
    for (int i = 0; i < P; ++i) {
        if (phi1[i] > 2 * N - phi0[i]) {
            return A12Result{"0"};
        }
    }

    // Feasible set zero check
    for (int i = 0; i < P; ++i) {
        int Lp = phi1[i];
        int Up = 2 * N - phi0[i];

        long long S_p = 0;
        for (int s = Lp; s <= Up; ++s) {
            int a_min = std::max(0, s - N);
            int a_max = std::min(N, s);
            if (a_min <= a_max) {
                S_p += (a_max - a_min + 1);
            }
        }

        if (S_p == 0) {
            return A12Result{"0"};
        }
    }

    int ncols = 2 * P;

    auto B_enum = detail::precompute_binomial(ncols);
    auto B_enum_u64 = detail::precompute_binomial_u64(ncols);
    auto B_weight = detail::precompute_binomial(2 * N);
    auto B_weight_u64 = detail::precompute_binomial_u64(2 * N);

    // Precompute suffix sum bounds for pruning
    std::vector<long long> suffix_min_sum(P + 1, 0);
    std::vector<long long> suffix_max_sum(P + 1, 0);
    for (int j = P - 1; j >= 0; --j) {
        int min_s = phi1[j];
        int max_s = 2 * N - phi0[j];
        suffix_min_sum[j] = suffix_min_sum[j + 1] + min_s;
        suffix_max_sum[j] = suffix_max_sum[j + 1] + max_s;
    }

    // Build first-coordinate feasible pairs (with symmetry: only v1[0] <= v2[0])
    std::vector<std::pair<int, int>> first_pairs;
    {
        int L0 = phi1[0];
        int U0 = 2 * N - phi0[0];

        for (int s = L0; s <= U0; ++s) {
            int a_min = std::max(0, s - N);
            int a_max = std::min(N, s);
            if (a_min > a_max) continue;

            // Only enumerate pairs where a0 <= b0 (symmetry)
            for (int a0 = a_min; a0 <= a_max; ++a0) {
                int b0 = s - a0;
                if (a0 <= b0) {  // Symmetry constraint
                    first_pairs.emplace_back(a0, b0);
                }
            }
        }
    }

    if (first_pairs.empty()) {
        return A12Result{"0"};
    }

    std::vector<cpp_int> partial_sums(first_pairs.size());

#ifdef ZAGAR_HAS_OPENMP
    if (n_threads > 0) {
        omp_set_num_threads(n_threads);
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t idx0 = 0; idx0 < first_pairs.size(); ++idx0) {
#else
    (void)n_threads;
    for (size_t idx0 = 0; idx0 < first_pairs.size(); ++idx0) {
#endif
        cpp_int local_sum = 0;

        std::vector<int> v1(P, 0), v2(P, 0);
        v1[0] = first_pairs[idx0].first;
        v2[0] = first_pairs[idx0].second;

        // Store configs with their symmetry multiplier (1 if v1==v2, 2 if v1<v2)
        struct ConfigWithMult {
            std::vector<int> cv1, cv2;
            int multiplier;
        };
        std::vector<ConfigWithMult> local_configs;

        // Track comparison state: 0 = equal so far, 1 = v1 < v2, -1 = v1 > v2 (skip)
        std::function<void(int, long long, int)> enumerate_suffix = [&](int p_idx, long long current_sum, int cmp_state) {
            if (p_idx == P) {
                // The total column sum must equal the total row sum.  For
                // P >= 2 the incremental bounds below already force this at the
                // last coordinate, but they are only applied for p_idx >= 1:
                // the first coordinate comes from first_pairs, which is built
                // without reference to sum_p.  So when P == 1 nothing else
                // checks it, infeasible configurations reach
                // process_a12_config, and its recursion indexes p_pad past the
                // end -- returning a wrong count or faulting, depending on what
                // happens to be in memory.  Mirrors the explicit check in
                // getA12's enumerate_configs.
                if (current_sum != sum_p) {
                    return;
                }
                if (cmp_state >= 0) {  // v1 <= v2
                    int mult = (cmp_state == 0) ? 1 : 2;  // equal -> 1, less -> 2
                    local_configs.push_back({v1, v2, mult});
                }
                return;
            }

            int Lp = phi1[p_idx];
            int Up = 2 * N - phi0[p_idx];

            for (int s = Lp; s <= Up; ++s) {
                long long new_sum = current_sum + s;

                // Sum-based pruning
                if (new_sum + suffix_min_sum[p_idx + 1] > sum_p) break;
                if (new_sum + suffix_max_sum[p_idx + 1] < sum_p) continue;

                int a_min = std::max(0, s - N);
                int a_max = std::min(N, s);
                if (a_min > a_max) continue;

                for (int a_val = a_min; a_val <= a_max; ++a_val) {
                    int b_val = s - a_val;
                    v1[p_idx] = a_val;
                    v2[p_idx] = b_val;

                    int new_cmp_state;
                    if (cmp_state != 0) {
                        // Already decided
                        new_cmp_state = cmp_state;
                    } else {
                        // Still equal, compare this position
                        if (a_val < b_val) {
                            new_cmp_state = 1;  // v1 < v2 established
                        } else if (a_val > b_val) {
                            new_cmp_state = -1;  // v1 > v2, will be skipped
                        } else {
                            new_cmp_state = 0;  // still equal
                        }
                    }

                    // Skip if v1 > v2 already established
                    if (new_cmp_state >= 0) {
                        enumerate_suffix(p_idx + 1, new_sum, new_cmp_state);
                    }
                }
            }
        };

        // Initial comparison state based on first pair
        int init_cmp = (v1[0] < v2[0]) ? 1 : ((v1[0] > v2[0]) ? -1 : 0);
        long long first_sum = v1[0] + v2[0];
        enumerate_suffix(1, first_sum, init_cmp);

        if (local_configs.empty()) {
            partial_sums[idx0] = 0;
            continue;
        }

        // Group by L value
        std::unordered_map<int, std::vector<size_t>> configs_by_L;
        for (size_t i = 0; i < local_configs.size(); ++i) {
            const auto& cfg = local_configs[i];
            int max_q = 0;
            for (int j = 0; j < P; ++j) {
                max_q = std::max(max_q, cfg.cv1[j]);
                max_q = std::max(max_q, cfg.cv2[j]);
            }
            int L = max_q + 2;
            configs_by_L[L].push_back(i);
        }

        // Process each L-group with shared memo table
        for (auto& kv : configs_by_L) {
            int L = kv.first;
            const std::vector<size_t>& indices = kv.second;

            MemoTable table;
            table.reserve(1 << 10);

            std::vector<int> root(L, 0);
            table.emplace(root, cpp_int(1));

            for (size_t cfg_idx : indices) {
                const auto& cfg = local_configs[cfg_idx];
                cpp_int contrib = process_a12_config(
                    p, cfg.cv1, cfg.cv2,
                    phi0, phi1,
                    P, N, ncols, L,
                    table,
                    B_enum, B_enum_u64,
                    B_weight, B_weight_u64
                );
                // Apply symmetry multiplier
                local_sum += cpp_int(cfg.multiplier) * contrib;
            }
        }

        partial_sums[idx0] = local_sum;
    }

    cpp_int total_sum = 0;
    for (size_t i = 0; i < partial_sums.size(); ++i) {
        total_sum += partial_sums[i];
    }

    return A12Result{total_sum.convert_to<std::string>()};
}

// --- Helper: Compute binomial coefficient C(n, k) for large n/k ---
static cpp_int binomial_large(int n, int k) {
    if (k < 0 || k > n) return cpp_int(0);
    if (k == 0 || k == n) return cpp_int(1);
    if (k > n - k) k = n - k;  // Take advantage of symmetry

    cpp_int result = 1;
    for (int i = 0; i < k; ++i) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

// --- Helper: Compute log2 of a cpp_int ---
static double log2_cpp_int(const cpp_int& x) {
    if (x <= 0) return -std::numeric_limits<double>::infinity();
    if (x == 1) return 0.0;

    // Convert to string and use length for rough estimate
    std::string s = x.convert_to<std::string>();
    size_t num_digits = s.size();

    // log2(x) ≈ log2(10) * (number of digits - 1) + log2(leading digits)
    // For better precision, extract leading digits
    double leading = 0.0;
    size_t lead_count = std::min<size_t>(15, num_digits);
    for (size_t i = 0; i < lead_count; ++i) {
        leading = leading * 10.0 + (s[i] - '0');
    }

    // log2(x) = log2(leading) + log2(10) * (num_digits - lead_count)
    return std::log2(leading) + std::log2(10.0) * static_cast<double>(num_digits - lead_count);
}

// --- Main API: getA1 ---
A1Result getA1(
    int N,
    int P,
    const std::vector<int>& a_vec
) {
    // Validate inputs
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }
    if (P <= 0) {
        throw std::invalid_argument("P must be a positive integer.");
    }
    if (static_cast<int>(a_vec.size()) != N) {
        throw std::invalid_argument("Length of a_vec must equal N.");
    }

    int two_P = 2 * P;
    for (int a : a_vec) {
        if (a < 0 || a > two_P) {
            throw std::invalid_argument("All entries of a_vec must be between 0 and 2*P.");
        }
    }

    // Compute product of binomial coefficients: prod(C(2*P, a_i))
    cpp_int binom_product = 1;
    for (int a : a_vec) {
        binom_product *= binomial_large(two_P, a);
    }

    // Multiply by 2^(2*N*P)
    long long exponent = 2LL * N * P;
    cpp_int power_of_2 = cpp_int(1) << static_cast<unsigned>(exponent);
    cpp_int result = binom_product * power_of_2;

    // Compute log2 value
    double log2_val = log2_cpp_int(result);

    return A1Result{result.convert_to<std::string>(), log2_val};
}

// --- Main API: getA2 ---
A2Result getA2(
    int N,
    int P,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1
) {
    // Validate inputs
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }
    if (P <= 0) {
        throw std::invalid_argument("P must be a positive integer.");
    }
    if (static_cast<int>(phi0.size()) != P) {
        throw std::invalid_argument("Length of phi0_vec must equal P.");
    }
    if (static_cast<int>(phi1.size()) != P) {
        throw std::invalid_argument("Length of phi1_vec must equal P.");
    }

    int two_N = 2 * N;
    for (int j = 0; j < P; ++j) {
        if (phi0[j] < 0 || phi0[j] > two_N) {
            throw std::invalid_argument("All entries of phi0_vec must be between 0 and 2*N.");
        }
        if (phi1[j] < 0 || phi1[j] > two_N) {
            throw std::invalid_argument("All entries of phi1_vec must be between 0 and 2*N.");
        }
        if (phi0[j] + phi1[j] > two_N) {
            throw std::invalid_argument("phi0_j + phi1_j must be <= 2*N for all j.");
        }
    }

    // Compute product: prod(C(2*N, phi0+phi1) * C(phi0+phi1, phi1) * 2^(2*N-(phi0+phi1)))
    cpp_int result = 1;
    for (int j = 0; j < P; ++j) {
        int sum = phi0[j] + phi1[j];

        // C(2*N, sum)
        cpp_int term = binomial_large(two_N, sum);

        // C(sum, phi1)
        term *= binomial_large(sum, phi1[j]);

        // 2^(2*N - sum)
        int power_exp = two_N - sum;
        cpp_int power_term = cpp_int(1) << static_cast<unsigned>(power_exp);
        term *= power_term;

        result *= term;
    }

    // Compute log2 value
    double log2_val = log2_cpp_int(result);

    return A2Result{result.convert_to<std::string>(), log2_val};
}

} // namespace zagar
