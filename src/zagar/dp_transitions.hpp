#ifndef ZAGAR_DP_TRANSITIONS_HPP
#define ZAGAR_DP_TRANSITIONS_HPP

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace zagar {

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

}  // namespace zagar

#endif  // ZAGAR_DP_TRANSITIONS_HPP
