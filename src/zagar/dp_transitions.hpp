#ifndef ZAGAR_DP_TRANSITIONS_HPP
#define ZAGAR_DP_TRANSITIONS_HPP

#include <boost/multiprecision/cpp_int.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace zagar {

using boost::multiprecision::cpp_int;

/**
 * Largest N for which the transition weights are guaranteed to fit in
 * uint64_t.  Summed over every transition out of one histogram, the weight is
 * at most 4^N: each of the N rows independently stays put, drops one level (in
 * C(2,1) = 2 ways) or drops two.  4^31 < 2^64 <= 4^32, so 31 is the bound.
 * The implementation checks for overflow regardless rather than relying on
 * callers to respect this.
 */
constexpr int DP_TRANSITIONS_MAX_N = 31;

/**
 * One locus's transitions out of a histogram of remaining row sums.
 *
 * Used by the pure-Python driver in zagar.exact_dp; `count_A12_dp` below does
 * not call it, since it accumulates transitions directly into the next layer.
 *
 * `hist[r]` is the number of rows whose remaining row sum is r.  Each row may
 * contribute 0, 1 or 2 ancestry-1 haplotypes at this locus, moving it down 0,
 * 1 or 2 levels; a row contributing 1 does so in C(2,1) = 2 ways, and that
 * factor is folded into the returned weight.
 *
 * Returns a map from the resulting state to the number of ways of reaching it.
 * The key packs the new histogram with its trailing zeros dropped (at least
 * one byte is kept, matching `_trim` in zagar.exact_dp), followed by one final
 * byte holding `s`, the number of ancestry-1 haplotypes placed at this locus.
 * That concatenation is injective because `s` is always the last byte.
 *
 * Distinct choices can land on the same state -- the map from per-level counts
 * to the new histogram is linear and not injective -- so weights are summed.
 *
 * @throws std::invalid_argument if a count is negative or the histogram is
 *         empty.
 * @throws std::overflow_error if a weight would exceed uint64_t.
 */
std::unordered_map<std::string, uint64_t> locus_transitions(
    const std::vector<int>& hist
);

/**
 * The whole histogram dynamic program, with no Python boundary inside it.
 *
 * Equivalent to `zagar.exact_dp.count_A12_dp`, which remains the reference
 * implementation and the fallback for builds without this extension.  Running
 * the layer loop here rather than in Python removes two costs: nothing is
 * marshalled per transition, and -- because transitions are accumulated
 * straight into the next layer, which merges duplicate states anyway -- the
 * intermediate per-histogram transition table disappears entirely.
 *
 * Layer weights are `cpp_int`, so the 331-bit counts at N = P = 10 are no
 * obstacle.  The per-transition multiplicities are still `uint64_t` and
 * bounded by 4^N, hence the DP_TRANSITIONS_MAX_N check.
 *
 * `a_vec` must already be padded to length N; validation of the constraints
 * themselves is left to the Python wrapper so that error messages stay in one
 * place.
 *
 * @param reorder_loci  process the most constraining loci first.  Affects only
 *                      speed: permuting loci is a bijection on arrays that
 *                      preserves the row sums.
 * @return the count in base 10, for the caller to turn into a Python int.
 * @throws std::invalid_argument on inconsistent sizes, non-positive N, or
 *         N > DP_TRANSITIONS_MAX_N.
 * @throws std::overflow_error if a multiplicity would exceed uint64_t.
 */
std::string count_A12_dp(
    int N,
    const std::vector<int>& a_vec,
    const std::vector<int>& phi0,
    const std::vector<int>& phi1,
    bool reorder_loci
);

}  // namespace zagar

#endif  // ZAGAR_DP_TRANSITIONS_HPP
