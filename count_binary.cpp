// [[Rcpp::depends(RcppParallel, BH)]]
// [[Rcpp::plugins(cpp14)]]

#include <Rcpp.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <boost/multiprecision/cpp_int.hpp>
#include <RcppParallel.h>

using namespace Rcpp;
using boost::multiprecision::cpp_int;
namespace rp = RcppParallel;

// --- Hash-based key for vector<int> (much faster than string serialization)
struct VectorHash {
  std::size_t operator()(const std::vector<int>& v) const {
    std::size_t seed = v.size();
    for (auto x : v) {
      // Combine hash using FNV-1a inspired mixing
      seed ^= static_cast<std::size_t>(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

// Type alias for the optimized memoization table
using MemoTable = std::unordered_map<std::vector<int>, cpp_int, VectorHash>;

// --- Conjugate of q (with leading 0); L = max(q)+2
static std::vector<int> conjugate_cpp(const std::vector<int>& q, int L) {
  std::vector<int> counts(L, 0);  // counts[t] = # columns with sum == t
  for (int v : q) {
    if (v < 0 || v >= L) stop("q has values outside [0, L-1]");
    counts[(size_t)v] += 1;
  }
  std::vector<int> c(L, 0);
  // c[0] = 0; for t=1..L-1, c[t] = # columns with sum >= t  (tail cumulative)
  int acc = 0;
  for (int t = L - 1; t >= 1; --t) {
    acc += counts[(size_t)t];
    c[(size_t)t] = acc;
  }
  return c;
}

// --- Pad p as [0, p..., 0...0] with L trailing zeros
static std::vector<int> pad_p_cpp(const std::vector<int>& p, int L) {
  std::vector<int> out;
  out.reserve(1 + (int)p.size() + L);
  out.push_back(0);
  out.insert(out.end(), p.begin(), p.end());
  out.insert(out.end(), (size_t)L, 0);
  return out;
}

// --- Precompute Pascal binomial table up to n (inclusive) using cpp_int
static std::vector<std::vector<cpp_int>> precompute_binomial_cpp(int n) {
  std::vector<std::vector<cpp_int>> B(n + 1, std::vector<cpp_int>(n + 1));
  for (int i = 0; i <= n; ++i) {
    for (int k = 0; k <= n; ++k) {
      if (k == 0 || k == i) {
        B[i][k] = 1;
      } else if (k > i) {
        B[i][k] = 0;
      } else {
        // Pascal recurrence
        B[i][k] = B[i - 1][k] + B[i - 1][k - 1];
      }
    }
  }
  return B;
}

// --- Precompute Pascal binomial table using uint64_t (for small n)
// Returns table and max safe n where all values fit in uint64_t
static std::vector<std::vector<uint64_t>> precompute_binomial_u64(int n) {
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

// Constants for overflow detection
static constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();
// Safe threshold for multiplication: if both operands < SAFE_MULT_THRESHOLD, product won't overflow
static constexpr uint64_t SAFE_MULT_THRESHOLD = 0xFFFFFFFFULL; // 2^32 - 1

// --- Full recursion with uint64_t optimization
// Uses uint64_t arithmetic when values are small, falls back to cpp_int when needed.
// p: padded with leading 0 and L trailing 0s
// c: conjugate vector length L (c[0]==0, c[t] for t>=1)
// j,k: the "Python-style" indices you used (start at 1)
// s_sum, p_sum, c_sum: running totals as in original code
// table: memo keyed by vector<int> directly (using VectorHash)
// L: as defined (max(q)+2)
// B: binomial table (cpp_int) with B[n][k] = C(n,k)
// B_u64: binomial table (uint64_t) for fast arithmetic
static cpp_int count_recursion_binary_cpp(
    const std::vector<int>& p,
    std::vector<int>& c,
    int j, int k,
    int s_sum, int p_sum, int c_sum,
    MemoTable& table,
    int L,
    const std::vector<std::vector<cpp_int>>& B,
    const std::vector<std::vector<uint64_t>>& B_u64
) {
  if (s_sum == p[(size_t)j]) {
    // lookup value under key c (vector directly)
    auto it = table.find(c);
    if (it != table.end()) {
      return it->second;
    } else {
      cpp_int result = count_recursion_binary_cpp(
        p, c,
        j + 1, 1,
        0,
        p[(size_t)(j + 2)],  // p[j+2] in Python
        c[(size_t)1],        // c[1]   in Python
        table, L, B, B_u64
      );
      // if c[1] != p[j+1], cache it (as in Python)
      if (c[(size_t)1] != p[(size_t)(j + 1)]) {
        table.emplace(c, result);
      }
      return result;
    }
  } else {
    // r_k = c[k] - c[k+1]
    int r_k = c[(size_t)k] - c[(size_t)(k + 1)];
    // Gale–Ryser upper bound
    int GR_condition = c_sum - s_sum - p_sum;

    // smallest = max(0, p[j]-s_sum - c[k+1])
    int smallest = std::max(0, p[(size_t)j] - s_sum - c[(size_t)(k + 1)]);
    // largest = min( min(r_k, p[j]-s_sum), GR_condition )
    int largest  = std::min(std::min(r_k, p[(size_t)j] - s_sum), GR_condition);

    if (smallest > largest) {
      return cpp_int(0);
    }

    // Try to use uint64_t accumulation
    uint64_t result_u64 = 0;
    bool overflow = false;

    for (int s_k = smallest; s_k <= largest; ++s_k) {
      // c[k] -= s_k
      c[(size_t)k] -= s_k;

      cpp_int value = count_recursion_binary_cpp(
        p, c,
        j, k + 1,
        s_sum + s_k,
        p_sum + p[(size_t)(j + k + 1)],
        c_sum + c[(size_t)(k + 1)],
        table, L, B, B_u64
      );

      // revert c[k]
      c[(size_t)k] += s_k;

      if (!overflow) {
        // Check if value fits in uint64_t
        if (value <= U64_MAX) {
          uint64_t val_u64 = static_cast<uint64_t>(value);
          uint64_t binom_u64 = B_u64[(size_t)r_k][(size_t)s_k];

          // Check for multiplication overflow
          if (binom_u64 <= SAFE_MULT_THRESHOLD && val_u64 <= SAFE_MULT_THRESHOLD) {
            uint64_t product = binom_u64 * val_u64;
            // Check for addition overflow
            if (result_u64 <= U64_MAX - product) {
              result_u64 += product;
              continue;
            }
          }
        }
        // Overflow detected, fall back to cpp_int for remaining iterations
        overflow = true;
      }

      // Fall back path: accumulate directly to cpp_int result
      // We need to restart with cpp_int from scratch
      cpp_int result = cpp_int(result_u64);
      // Add current term
      result += B[(size_t)r_k][(size_t)s_k] * value;

      // Continue with remaining terms using cpp_int
      for (int s_k2 = s_k + 1; s_k2 <= largest; ++s_k2) {
        c[(size_t)k] -= s_k2;

        cpp_int value2 = count_recursion_binary_cpp(
          p, c,
          j, k + 1,
          s_sum + s_k2,
          p_sum + p[(size_t)(j + k + 1)],
          c_sum + c[(size_t)(k + 1)],
          table, L, B, B_u64
        );

        c[(size_t)k] += s_k2;
        result += B[(size_t)r_k][(size_t)s_k2] * value2;
      }
      return result;
    }

    return cpp_int(result_u64);
  }
}

//' Count 0/1 matrices with row sums p and column sums q (full C++ version).
//'
//' Faithful port of your Python (conjugate, padding, binomials, recursion + memoization).
//' Returns the count as a **decimal string**. Optionally sorts `p` in nonincreasing order
//' for speed (does not change the count).
//'
//' @param p Integer vector of row sums (nonnegative).
//' @param q Integer vector of column sums (nonnegative).
//' @param sort_p_desc Logical; if TRUE, sort `p` decreasing (recommended). Default FALSE to
//' match your original Python exactly.
//' @return A list with elements:
//' \itemize{
//'   \item number: character (big integer in base 10)
//'   \item memo_entries: integer (how many memo states stored)
//' }
//' @examples
//' \dontrun{
//'   Rcpp::sourceCpp("count_binary_full.cpp")
//'   p <- c(2L, 1L)
//'   q <- c(1L, 1L, 1L)
//'   out <- claude_count_binary_matrices_cpp(p, q, sort_p_desc = TRUE)
//'   out$number
//'   # If desired:
//'   # gmp::as.bigz(out$number)
//' }
// [[Rcpp::export]]
Rcpp::List claude_count_binary_matrices_cpp(Rcpp::IntegerVector pR,
                                     Rcpp::IntegerVector qR,
                                     bool sort_p_desc = false) {
  std::vector<int> p(pR.begin(), pR.end());
  std::vector<int> q(qR.begin(), qR.end());

  if (p.empty() || q.empty()) stop("p and q must be non-empty.");
  if (*std::min_element(p.begin(), p.end()) < 0 ||
      *std::min_element(q.begin(), q.end()) < 0) {
    stop("p and q must be nonnegative integers.");
  }

  long long sum_p = std::accumulate(p.begin(), p.end(), 0LL);
  long long sum_q = std::accumulate(q.begin(), q.end(), 0LL);
  if (sum_p != sum_q) {
    return Rcpp::List::create(
      _["number"] = std::string("0"),
      _["memo_entries"] = 0
    );
  }

  if (sort_p_desc) {
    std::sort(p.begin(), p.end(), std::greater<int>()); // optional speedup
  }

  int n = (int)q.size();
  int L = *std::max_element(q.begin(), q.end()) + 2;

  // Build conjugate and padded p
  std::vector<int> c = conjugate_cpp(q, L);
  std::vector<int> p_pad = pad_p_cpp(p, L);

  // Precompute binomials up to n (both cpp_int and uint64_t versions)
  auto B = precompute_binomial_cpp(n);
  auto B_u64 = precompute_binomial_u64(n);

  // Memo table: key = vector<int> directly, value = cpp_int count
  MemoTable table;
  table.reserve(1 << 12); // heuristic; adjust as needed

  // root state: tuple(0,...,0) -> 1
  std::vector<int> root(L, 0);
  table.emplace(root, cpp_int(1));

  // Initial recursion call (same as Python):
  // j=1, k=1, s_sum=0, p_sum=p[2], c_sum=c[1]
  cpp_int number = count_recursion_binary_cpp(
    p_pad, c, 1, 1, 0,
    p_pad[(size_t)2],   // p[2]
    c[(size_t)1],       // c[1]
    table, L, B, B_u64
  );

  // store final c state -> number (as in Python)
  table[c] = number;

  // Return as decimal string + memo size
  std::string number_str = number.convert_to<std::string>();
  return Rcpp::List::create(
    _["number"] = number_str,
    _["memo_entries"] = (int)table.size()
  );
}


// [[Rcpp::export]]
Rcpp::List claude_getA12_cpp(
    int N,
    Rcpp::IntegerVector a_vecR,   // row sums p
    Rcpp::IntegerVector phi0R,
    Rcpp::IntegerVector phi1R,
    bool sort_p_desc = true       // mirrors claude_count_binary_matrices_cpp
) {
  using boost::multiprecision::cpp_int;

  // --- Basic checks
  if (N <= 0) {
    Rcpp::stop("N must be a positive integer.");
  }

  int P = phi0R.size();
  if (phi1R.size() != P) {
    Rcpp::stop("phi0_vec and phi1_vec must have the same length.");
  }

  std::vector<int> phi0(phi0R.begin(), phi0R.end());
  std::vector<int> phi1(phi1R.begin(), phi1R.end());
  std::vector<int> p(a_vecR.begin(), a_vecR.end());  // row sums

  // Optional: sort p in decreasing order
  if (sort_p_desc) {
    std::sort(p.begin(), p.end(), std::greater<int>());
  }

  // Precompute sum(p)
  long long sum_p = std::accumulate(p.begin(), p.end(), 0LL);

  // --- Coordinate-level feasibility check
  for (int i = 0; i < P; ++i) {
    if (phi1[i] > 2 * N - phi0[i]) {
      return Rcpp::List::create(
        Rcpp::_["number"] = std::string("0")
      );
    }
  }

  // ----------------------------------------------------------------------
  // *** FEASIBLE SET ZERO CHECK (no size limit) ***
  //
  // For each coordinate p, feasible (a, b) satisfy:
  //   a + b = s in [phi1[p], 2N - phi0[p]]
  //   0 <= a <= N, 0 <= b <= N
  //
  // Count per coordinate:
  //   S_p = sum_s max(0, min(N, s) - max(0, s - N) + 1)
  //
  // If S_p == 0 for any coordinate, return 0 immediately.
  // ----------------------------------------------------------------------

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
      return Rcpp::List::create(
        Rcpp::_["number"] = std::string("0")
      );
    }
  }

  // ----------------------------------------------------------------------
  // Enumerate all feasible configurations (no size limit)
  // ----------------------------------------------------------------------

  int ncols = 2 * P;

  // Precompute binomial tables (both cpp_int and uint64_t versions)
  std::vector<std::vector<cpp_int>> B_enum = precompute_binomial_cpp(ncols);
  std::vector<std::vector<uint64_t>> B_enum_u64 = precompute_binomial_u64(ncols);
  std::vector<std::vector<cpp_int>> B_weight = precompute_binomial_cpp(2 * N);
  std::vector<std::vector<uint64_t>> B_weight_u64 = precompute_binomial_u64(2 * N);

  // --- Phase 1: Enumerate all feasible (v1, v2) configurations ---
  std::vector<std::pair<std::vector<int>, std::vector<int>>> all_configs;

  std::vector<int> v1(P, 0), v2(P, 0);

  std::function<void(int)> enumerate_configs = [&](int p_idx) {
    if (p_idx == P) {
      // Check sum constraint
      long long sum_q = 0;
      for (int i = 0; i < P; ++i) {
        sum_q += v1[i] + v2[i];
      }
      if (sum_q != sum_p) {
        return;
      }
      // Store this configuration
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
    return Rcpp::List::create(
      Rcpp::_["number"] = std::string("0")
    );
  }

  // --- Phase 2: Group configurations by L value (max_q + 2) ---
  // This allows us to reuse memoization tables across configs with the same L
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

  // --- Phase 3: Process each L-group with a shared memoization table ---
  cpp_int total_sum = 0;

  for (auto& kv : configs_by_L) {
    int L = kv.first;
    const std::vector<size_t>& indices = kv.second;

    // Precompute p_pad for this L (same for all configs in this group)
    std::vector<int> p_pad = pad_p_cpp(p, L);

    // Shared memoization table for this L-group
    MemoTable table;
    table.reserve(1 << 12);

    // Initialize root state
    std::vector<int> root(L, 0);
    table.emplace(root, cpp_int(1));

    for (size_t idx : indices) {
      const auto& cfg = all_configs[idx];
      const std::vector<int>& cv1 = cfg.first;
      const std::vector<int>& cv2 = cfg.second;

      // Construct q = (v1, v2)
      std::vector<int> q(ncols);
      for (int i = 0; i < P; ++i) {
        q[i] = cv1[i];
        q[P + i] = cv2[i];
      }

      std::vector<int> c = conjugate_cpp(q, L);

      cpp_int A_size = count_recursion_binary_cpp(
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

      // Compute weight using uint64_t when possible
      uint64_t weight_u64 = 1;
      bool weight_overflow = false;

      for (int i = 0; i < P; ++i) {
        int s = cv1[i] + cv2[i];
        int n1 = s;
        int k1 = phi0[i];
        int n2 = 2 * N - s;
        int k2 = phi1[i];

        if (k1 < 0 || k1 > n1 || k2 < 0 || k2 > n2) {
          weight_u64 = 0;
          break;
        }

        uint64_t b1 = B_weight_u64[n1][k1];
        uint64_t b2 = B_weight_u64[n2][k2];

        // Check for overflow in multiplication chain
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

      if (weight_u64 != 0 || weight_overflow) {
        if (!weight_overflow) {
          total_sum += cpp_int(weight_u64) * A_size;
        } else {
          // Fall back to cpp_int weight calculation
          cpp_int weight = 1;
          for (int i = 0; i < P; ++i) {
            int s = cv1[i] + cv2[i];
            int n1 = s;
            int k1 = phi0[i];
            int n2 = 2 * N - s;
            int k2 = phi1[i];
            weight *= B_weight[n1][k1];
            weight *= B_weight[n2][k2];
          }
          total_sum += weight * A_size;
        }
      }
    }
  }

  std::string total_str = total_sum.convert_to<std::string>();
  return Rcpp::List::create(
    Rcpp::_["number"] = total_str
  );
}

// Worker for parallel A12 computation
// Parallelizes by first-coordinate pairs, with memo sharing within L-groups
struct A12Worker : public rp::Worker {
  const std::vector<int> &p;
  const std::vector<int> &phi0;
  const std::vector<int> &phi1;
  const std::vector< std::pair<int,int> > &first_pairs;
  const std::vector<std::vector<cpp_int>> &B_enum;
  const std::vector<std::vector<uint64_t>> &B_enum_u64;
  const std::vector<std::vector<cpp_int>> &B_weight;
  const std::vector<std::vector<uint64_t>> &B_weight_u64;
  const long long sum_p;
  const int P;
  const int N;
  const int ncols;

  std::vector<cpp_int> &partial_sums;

  A12Worker(const std::vector<int> &p_,
            const std::vector<int> &phi0_,
            const std::vector<int> &phi1_,
            const std::vector< std::pair<int,int> > &first_pairs_,
            const std::vector<std::vector<cpp_int>> &B_enum_,
            const std::vector<std::vector<uint64_t>> &B_enum_u64_,
            const std::vector<std::vector<cpp_int>> &B_weight_,
            const std::vector<std::vector<uint64_t>> &B_weight_u64_,
            long long sum_p_,
            int P_,
            int N_,
            int ncols_,
            std::vector<cpp_int> &partial_sums_)
    : p(p_), phi0(phi0_), phi1(phi1_),
      first_pairs(first_pairs_),
      B_enum(B_enum_), B_enum_u64(B_enum_u64_),
      B_weight(B_weight_), B_weight_u64(B_weight_u64_),
      sum_p(sum_p_),
      P(P_), N(N_), ncols(ncols_),
      partial_sums(partial_sums_) {}

  void operator()(std::size_t begin, std::size_t end) override {
    for (std::size_t idx0 = begin; idx0 < end; ++idx0) {
      cpp_int local_sum = 0;

      std::vector<int> v1(P, 0), v2(P, 0);
      v1[0] = first_pairs[idx0].first;
      v2[0] = first_pairs[idx0].second;

      // Enumerate all configurations for this first_pair
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

        std::vector<int> p_pad = pad_p_cpp(p, L);

        MemoTable table;
        table.reserve(1 << 10);

        std::vector<int> root(L, 0);
        table.emplace(root, cpp_int(1));

        for (size_t cfg_idx : indices) {
          const auto& cfg = local_configs[cfg_idx];
          const std::vector<int>& cv1 = cfg.first;
          const std::vector<int>& cv2 = cfg.second;

          std::vector<int> q(ncols);
          for (int i = 0; i < P; ++i) {
            q[i] = cv1[i];
            q[P + i] = cv2[i];
          }

          std::vector<int> c = conjugate_cpp(q, L);

          cpp_int A_size = count_recursion_binary_cpp(
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

          uint64_t weight_u64 = 1;
          bool weight_overflow = false;

          for (int i = 0; i < P; ++i) {
            int s = cv1[i] + cv2[i];
            int n1 = s;
            int k1 = phi0[i];
            int n2 = 2 * N - s;
            int k2 = phi1[i];

            if (k1 < 0 || k1 > n1 || k2 < 0 || k2 > n2) {
              weight_u64 = 0;
              break;
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

          if (weight_u64 != 0 || weight_overflow) {
            if (!weight_overflow) {
              local_sum += cpp_int(weight_u64) * A_size;
            } else {
              cpp_int weight = 1;
              for (int i = 0; i < P; ++i) {
                int s = cv1[i] + cv2[i];
                int n1 = s;
                int k1 = phi0[i];
                int n2 = 2 * N - s;
                int k2 = phi1[i];
                weight *= B_weight[n1][k1];
                weight *= B_weight[n2][k2];
              }
              local_sum += weight * A_size;
            }
          }
        }
      }

      partial_sums[idx0] = local_sum;
    }
  }
};

// [[Rcpp::export]]
Rcpp::List claude_getA12_cpp_parallel(
    int N,
    Rcpp::IntegerVector a_vecR,
    Rcpp::IntegerVector phi0R,
    Rcpp::IntegerVector phi1R,
    int n_threads = -1,
    bool sort_p_desc = true
) {
  if (N <= 0) {
    Rcpp::stop("N must be a positive integer.");
  }

  int P = phi0R.size();
  if (phi1R.size() != P) {
    Rcpp::stop("phi0_vec and phi1_vec must have the same length.");
  }

  std::vector<int> phi0(phi0R.begin(), phi0R.end());
  std::vector<int> phi1(phi1R.begin(), phi1R.end());
  std::vector<int> p(a_vecR.begin(), a_vecR.end());

  if (sort_p_desc) {
    std::sort(p.begin(), p.end(), std::greater<int>());
  }

  long long sum_p = std::accumulate(p.begin(), p.end(), 0LL);

  for (int i = 0; i < P; ++i) {
    if (phi1[i] > 2 * N - phi0[i]) {
      return Rcpp::List::create(
        Rcpp::_["number"] = std::string("0")
      );
    }
  }

  // ----------------------------------------------------------------------
  // *** FEASIBLE SET ZERO CHECK (no size limit) ***
  // ----------------------------------------------------------------------

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
      return Rcpp::List::create(
        Rcpp::_["number"] = std::string("0")
      );
    }
  }

  int ncols = 2 * P;

  std::vector<std::vector<cpp_int>> B_enum   = precompute_binomial_cpp(ncols);
  std::vector<std::vector<uint64_t>> B_enum_u64 = precompute_binomial_u64(ncols);
  std::vector<std::vector<cpp_int>> B_weight = precompute_binomial_cpp(2 * N);
  std::vector<std::vector<uint64_t>> B_weight_u64 = precompute_binomial_u64(2 * N);

  // Build first-coordinate feasible pairs
  std::vector< std::pair<int,int> > first_pairs;
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
    return Rcpp::List::create(
      Rcpp::_["number"] = std::string("0")
    );
  }

  std::vector<cpp_int> partial_sums(first_pairs.size());

  A12Worker worker(
    p, phi0, phi1,
    first_pairs,
    B_enum, B_enum_u64,
    B_weight, B_weight_u64,
    sum_p,
    P, N, ncols,
    partial_sums
  );

  rp::parallelFor(
    0,
    (std::size_t)first_pairs.size(),
    worker
  );

  cpp_int total_sum = 0;
  for (std::size_t i = 0; i < partial_sums.size(); ++i) {
    total_sum += partial_sums[i];
  }

  std::string total_str = total_sum.convert_to<std::string>();
  return Rcpp::List::create(
    Rcpp::_["number"] = total_str
  );
}
