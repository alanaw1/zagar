#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <boost/multiprecision/cpp_int.hpp>

namespace zagar {

using boost::multiprecision::cpp_int;

// --- Hash-based key for vector<int> (much faster than string serialization)
struct VectorHash {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = v.size();
        for (auto x : v) {
            seed ^= static_cast<std::size_t>(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Type alias for the memoization table
using MemoTable = std::unordered_map<std::vector<int>, cpp_int, VectorHash>;

// Result structures
struct CountResult {
    std::string number;      // Big integer as decimal string
    int memo_entries;        // Memoization table size
};

struct A12Result {
    std::string number;      // Big integer as decimal string
};

struct A1Result {
    std::string number;      // Big integer as decimal string
    double log2_value;       // log2 of the result
};

struct A2Result {
    std::string number;      // Big integer as decimal string
    double log2_value;       // log2 of the result
};

// Main API functions

/**
 * Count 0/1 matrices with row sums p and column sums q.
 *
 * @param p Row sums (nonnegative integers)
 * @param q Column sums (nonnegative integers)
 * @param sort_p_desc If true, sort p in decreasing order for speed
 * @return CountResult with number (as string) and memo_entries
 */
CountResult count_binary_matrices(
    const std::vector<int>& p,
    const std::vector<int>& q,
    bool sort_p_desc = false
);

/**
 * Compute weighted combinatorial sum A12.
 *
 * @param N Positive integer parameter
 * @param a_vec Row sums
 * @param phi0 First constraint vector
 * @param phi1 Second constraint vector (must have same length as phi0)
 * @param sort_p_desc If true, sort p in decreasing order
 * @return A12Result with number (as string)
 */
A12Result getA12(
    int N,
    const std::vector<int>& a_vec,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1,
    bool sort_p_desc = true
);

/**
 * Compute weighted combinatorial sum A12 using parallel processing.
 *
 * @param N Positive integer parameter
 * @param a_vec Row sums
 * @param phi0 First constraint vector
 * @param phi1 Second constraint vector
 * @param n_threads Number of threads (-1 = use all available)
 * @param sort_p_desc If true, sort p in decreasing order
 * @return A12Result with number (as string)
 */
A12Result getA12_parallel(
    int N,
    const std::vector<int>& a_vec,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1,
    int n_threads = -1,
    bool sort_p_desc = true
);

/**
 * Compute A1: Number of Arrays Subject to Row Margin Constraints.
 *
 * Formula: prod(C(2*P, a_i)) * 2^(2*N*P) for all a_i in a_vec
 *
 * @param N Positive integer (number of rows)
 * @param P Positive integer (number of columns)
 * @param a_vec Row margin constraints (length N, values in [0, 2*P])
 * @return A1Result with number (as string) and log2_value
 */
A1Result getA1(
    int N,
    int P,
    const std::vector<int>& a_vec
);

/**
 * Compute A2: Number of Arrays Subject to Column Margin Constraints.
 *
 * Formula: prod(C(2*N, phi0_j+phi1_j) * C(phi0_j+phi1_j, phi1_j) * 2^(2*N-(phi0_j+phi1_j)))
 *
 * @param N Positive integer (number of rows)
 * @param P Positive integer (number of columns)
 * @param phi0 First constraint vector (length P, values in [0, 2*N])
 * @param phi1 Second constraint vector (length P, phi0_j+phi1_j <= 2*N)
 * @return A2Result with number (as string) and log2_value
 */
A2Result getA2(
    int N,
    int P,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1
);

// Internal helper functions (exposed for potential reuse)
namespace detail {

std::vector<int> conjugate(const std::vector<int>& q, int L);
std::vector<int> pad_p(const std::vector<int>& p, int L);
std::vector<std::vector<cpp_int>> precompute_binomial(int n);
std::vector<std::vector<uint64_t>> precompute_binomial_u64(int n);

cpp_int count_recursion_binary(
    const std::vector<int>& p,
    std::vector<int>& c,
    int j, int k,
    int s_sum, int p_sum, int c_sum,
    MemoTable& table,
    int L,
    const std::vector<std::vector<cpp_int>>& B,
    const std::vector<std::vector<uint64_t>>& B_u64
);

} // namespace detail

} // namespace zagar
