"""Tests for zagar.asymptotics (Theorem 3.1, all four regimes).

The main correctness check compares the closed-form correction terms used in
``zagar.asymptotics`` against the expanded polynomials of the reference R
implementation (``four_cases_freeze_101626.R``), which were derived
independently.  Agreement to floating-point precision across the parameter
space is strong evidence that neither has a transcription error.
"""

import math
import random

import pytest

from zagar.asymptotics import (
    classify_case,
    densities_from_constraints,
    entropy_bits,
    log2_A12_asymptotic,
)

L2E = math.log2(math.e)


# ---------------------------------------------------------------------------
# Reference implementation: literal port of the R script's expanded polynomials
#
# Nothing below is meant to be readable.  It is a transcription, kept in the
# shape the R source had it, so that a disagreement with zagar.asymptotics
# points at one implementation or the other rather than at a paraphrase made on
# the way in.  ``rho`` is the R script's name for ``abar``.
# ---------------------------------------------------------------------------


def _H(*p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def ref_case1(N, P, rho, f0, f1):
    term_1 = 2 * N * P * _H(1 - rho - f0, rho - f1, f1, f0)
    term_2 = (
        -(N - 1) * 0.5 * math.log2(rho * (1 - rho))
        - P * 0.5 * math.log2(f0 * f1)
        - (P - 1) * 0.5 * math.log2(1 - f0 - f1)
        - 0.5 * math.log2((1 - rho - f0) * (rho - f1))
        - ((N + 2 * P) / 2) * math.log2(math.pi)
        - N * 0.5 * math.log2(P)
        - P * math.log2(N)
        - (N + 2 * P)
    )
    coeff_N2 = (
        f0**3*f1**2*rho**2 - f0**3*f1**2*rho + f0**3*f1**2
        - f0**3*f1*rho**3 + f0**3*f1*rho**2 - f0**3*f1*rho
        + f0**2*f1**3*rho**2 - f0**2*f1**3*rho + f0**2*f1**3
        - 2*f0**2*f1**2*rho**2 + 2*f0**2*f1**2*rho - 2*f0**2*f1**2
        - f0**2*f1*rho**4 + 3*f0**2*f1*rho**3 - 3*f0**2*f1*rho**2 + 2*f0**2*f1*rho
        + f0*f1**3*rho**3 - 2*f0*f1**3*rho**2 + 2*f0*f1**3*rho - f0*f1**3
        - f0*f1**2*rho**4 + f0*f1**2*rho**3 - f0*f1**2*rho + f0*f1**2
        + f0*f1*rho**4 - 2*f0*f1*rho**3 + 2*f0*f1*rho**2 - f0*f1*rho
    )
    coeff_NP = (
        6*f0**3*f1**3 - 6*f0**3*f1**2*rho**2 - 6*f0**3*f1**2*rho
        + 6*f0**3*f1*rho**3
        - 6*f0**2*f1**3*rho**2 + 18*f0**2*f1**3*rho - 12*f0**2*f1**3
        - 12*f0**2*f1**2*rho**2 + 12*f0**2*f1**2*rho
        + 6*f0**2*f1*rho**4 - 6*f0**2*f1*rho**3
        - 6*f0*f1**3*rho**3 + 18*f0*f1**3*rho**2 - 18*f0*f1**3*rho + 6*f0*f1**3
        + 6*f0*f1**2*rho**4 - 18*f0*f1**2*rho**3 + 18*f0*f1**2*rho**2 - 6*f0*f1**2*rho
    )
    coeff_P2 = (
        f0**3*f1**2*rho**2 - f0**3*f1**2*rho
        - f0**3*f1*rho**3 + f0**3*f1*rho
        + f0**3*rho**3 - f0**3*rho**2
        + f0**2*f1**3*rho**2 - f0**2*f1**3*rho
        - 3*f0**2*f1**2*rho**2 + 3*f0**2*f1**2*rho
        - f0**2*f1*rho**4 + 3*f0**2*f1*rho**3 - 2*f0**2*f1*rho
        + f0**2*rho**4 - 3*f0**2*rho**3 + 2*f0**2*rho**2
        + f0*f1**3*rho**3 - 3*f0*f1**3*rho**2 + 2*f0*f1**3*rho
        - f0*f1**2*rho**4 + f0*f1**2*rho**3 + 3*f0*f1**2*rho**2 - 3*f0*f1**2*rho
        + 2*f0*f1*rho**4 - 4*f0*f1*rho**3 + f0*f1*rho**2 + f0*f1*rho
        - f0*rho**4 + 2*f0*rho**3 - f0*rho**2
        - f1**3*rho**3 + 2*f1**3*rho**2 - f1**3*rho
        + f1**2*rho**4 - f1**2*rho**3 - f1**2*rho**2 + f1**2*rho
        - f1*rho**4 + 2*f1*rho**3 - f1*rho**2
    )
    coeff_const = (
        6*f0**3*f1**3 - 7*f0**3*f1**2*rho**2 - 5*f0**3*f1**2*rho - f0**3*f1**2
        + 7*f0**3*f1*rho**3
        - 7*f0**2*f1**3*rho**2 + 19*f0**2*f1**3*rho - 13*f0**2*f1**3
        - 9*f0**2*f1**2*rho**2 + 9*f0**2*f1**2*rho + 2*f0**2*f1**2
        + 7*f0**2*f1*rho**4 - 8*f0**2*f1*rho**3
        - 7*f0*f1**3*rho**3 + 21*f0*f1**3*rho**2 - 21*f0*f1**3*rho + 7*f0*f1**3
        + 7*f0*f1**2*rho**4 - 20*f0*f1**2*rho**3 + 18*f0*f1**2*rho**2
        - 4*f0*f1**2*rho - f0*f1**2
    )
    # In the R source coeff_N and coeff_P are both -coeff_NP.
    num = (
        N**2 * coeff_N2 + N * P * coeff_NP - N * coeff_NP
        + P**2 * coeff_P2 - P * coeff_NP + coeff_const
    )
    den = 24*N*P*f0*f1*rho*(f1 - rho)*(rho - 1)*(f0 + f1 - 1)*(f0 + rho - 1)
    return term_1 + term_2 + (num / den) * L2E


def ref_case2(N, P, rho, f0):
    term1 = 2 * N * P * _H(rho, f0, 1 - rho - f0)
    term2 = (
        -(N - 1) * 0.5 * math.log2(P)
        - (2 * P - 1) * 0.5 * math.log2(N)
        - (N - 1) * 0.5 * math.log2(rho * (1 - rho))
        - P * 0.5 * math.log2(f0 * rho * (1 - rho - f0))
        - (N + 2 * P - 1) * 0.5 * math.log2(math.pi)
        - (N + 2 * P - 1)
    )
    num = (
        N**2*f0**2*rho**2 - N**2*f0**2*rho + N**2*f0**2 + N**2*f0*rho**3
        - 2*N**2*f0*rho**2 + 2*N**2*f0*rho - N**2*f0 - 6*N*P*f0**2*rho**2
        + 6*N*P*f0**2*rho - 6*N*P*f0*rho**3 + 12*N*P*f0*rho**2 - 6*N*P*f0*rho
        + 6*N*f0**2*rho**2 - 6*N*f0**2*rho + 6*N*f0*rho**3 - 12*N*f0*rho**2
        + 6*N*f0*rho + P**2*f0**2*rho**2 - 2*P**2*f0**2*rho + P**2*f0**2
        + P**2*f0*rho**3 - 3*P**2*f0*rho**2 + 3*P**2*f0*rho - P**2*f0
        - P**2*rho**3 + 2*P**2*rho**2 - P**2*rho + 6*P*f0**2*rho**2
        - 6*P*f0**2*rho + 6*P*f0*rho**3 - 12*P*f0*rho**2 + 6*P*f0*rho
        - 7*f0**2*rho**2 + 7*f0**2*rho - f0**2 - 7*f0*rho**3 + 14*f0*rho**2
        - 8*f0*rho + f0
    )
    den = 24 * N * P * f0 * rho * (1 - rho) * (1 - rho - f0)
    return term1 + term2 + (num / den) * L2E


def ref_case4(N, P, rho):
    term1 = 2 * N * P * _H(rho, 1 - rho)
    term2 = (
        -(N - 1) * 0.5 * math.log2(P)
        - (P - 1) * 0.5 * math.log2(N)
        - (N + P - 1) * 0.5 * math.log2(rho * (1 - rho))
        - (N + P - 1) * 0.5 * math.log2(math.pi)
        - (N + P - 1)
    )
    e4 = (6*rho**2 - 6*rho + 1) * (N + P - 1)**2 / (16*N*P*rho*(1 - rho)) * L2E
    h3 = (
        -((2*rho - 1)**2)
        * (5*N**2 + 6*N*P - 6*N + 5*P**2 - 6*P + 1)
        / (48*N*P*rho*(1 - rho))
        * L2E
    )
    return term1 + term2 + e4 + h3


# ---------------------------------------------------------------------------
# TODO (Alan): the test functions go below.  See the checklist in the plan.
#
# Agreement with the reference polynomials -- the main event.  Draw random
# (N, P) in [2, 400] and random densities in each regime, and compare at
# rel=1e-12.  Note there is no ref_case3: Case 3 is checked by asserting that
# log2_A12_asymptotic at (abar, 1 - abar, f1) equals ref_case2(N, P, 1 - abar,
# f1), i.e. straight through the A -> 1 - A bijection.
#   - test_case1_matches_reference          (Case 1, interior)
#   - test_case2_matches_reference          (f1 = abar)
#   - test_case3_matches_reference_through_the_bijection
#   - test_case4_matches_reference          (abar = f1 = 1 - f0)
#   - test_case4_reproduces_legacy_symmetric_approximation:
#       at abar = f0 = f1 = 1/2 the new Case 4 must reproduce the old
#       _compute_approximations formula exactly, which is what keeps getA12's
#       published output unchanged.  The legacy value is
#         upper  = 2NP - 0.5*(P*log2(pi*N) + N*log2(pi*P) - log2(pi*N*P))
#         legacy = upper - ((P + N - 1)**2)/(8*N*P) * L2E
#
# Agreement with exact counts.  Parametrize over (N, P, abar, f0, f1,
# exact_log2, case) and assert 0 < approx - exact < 0.04 bits -- the sign
# matters, the expansion always overestimates at finite size.
#
# Every value below was re-derived from count_A12_dp and agrees to 4 decimal
# places, with one exception, noted:
#   (4, 4, 0.5,   0.25, 0.25, 43.7463, 1)   (4, 4, 0.5,  0.25, 0.5, 29.9929, 2)
#   (4, 4, 0.375, 0.25, 0.25, 41.0235, 1)   (4, 8, 0.5,  0.25, 0.5, 63.3956, 2)
#   (4, 6, 1/3,   0.25, 0.25, 60.2523, 1)   (6, 6, 0.5,  0.5,  1/6, 72.8085, 3)
#   (4, 6, 0.5,   0.25, 0.25, 68.0568, 1)   (4, 4, 0.5,  0.5,  0.5, 19.6530, 4)
#   (10,10, 0.5,  0.25, 0.25, 330.0200, 1)  (4, 4, 0.25, 0.75, 0.25, 14.8899, 4)
#    ^ NOT independently verified: the exact     (6, 6, 0.5,  0.5,  0.5, 49.3671, 4)
#      count at N = P = 10 in Case 1 exceeds     (8, 8, 0.25, 0.75, 0.25, 72.7581, 4)
#      16 GB in count_A12_dp.  It is carried
#      over from the reference suite on trust.
#      The implied error is +0.0094 bits, which
#      fits the trend, but it is the one row
#      here resting on the reference alone.
#
# Note the band is tight at (4, 4, 0.25, 0.75, 0.25): the error there is
# +0.0358, so 0.04 is close to binding and not an arbitrary round number.
#
# Classification, helpers, error handling
#   - interior and all three boundaries classify correctly
#   - the tolerance: classify_case(0.4, 0.3, 0.4 - 1e-12) == 2, but
#     classify_case(0.4, 0.3, 0.4 - 1e-3) == 1
#   - infeasible or degenerate triples raise ValueError: (0.3, 0.3, 0.5),
#     (0.9, 0.5, 0.2), (0.0, 0.3, 0.1), (0.5, 0.0, 0.1), (1.0, 0.1, 0.1)
#   - N <= 0 or P <= 0 raises
#   - entropy_bits on (0.5,0.5), four quarters, (1.0,0.0), and (0.5,0.4) raising
#   - densities_from_constraints(4, [4]*4, [2]*4, [2]*4) -> (0.5, 0.25, 0.25)
#   - densities_from_constraints rejects the non-uniform paper example, with a
#     message matching "semi-regular"
#   - the details payload: exactly the keys {log2_A12, case, entropy,
#     dimension, K, log2_delta}; dimension == N + 2P in Case 1; log2_A12 equal
#     to the plain call.  Add the identity the reference never checked:
#       2*N*P*H - 0.5*log2_delta - 0.5*D*log2(pi) - D + K*L2E == log2_A12
#     This is what pins log2_delta to the real log2(Delta) rather than half of
#     it.  Build it from the details dict alone.
# ---------------------------------------------------------------------------
