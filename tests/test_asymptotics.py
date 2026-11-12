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
# Agreement with the reference polynomials
#
# This is the real correctness proof for the closed forms.  Each test draws
# random (N, P) and random densities strictly inside one regime, and compares
# against the independently derived expanded polynomial at rel=1e-12 -- loose
# enough to absorb the different operation grouping, tight enough that a wrong
# coefficient cannot hide.
#
# The small slacks (0.02, 0.06) keep draws away from the regime boundaries,
# where classify_case would correctly pick a different case and the comparison
# against the wrong ref_ function would fail for reasons of the test's own
# making.  Because those slacks mean draws can be rejected, each filtered test
# counts its iterations and asserts the loop actually ran.
# ---------------------------------------------------------------------------


def test_case1_matches_reference():
    rng = random.Random(0)
    checked = 0
    for _ in range(600):
        N, P = rng.randint(2, 400), rng.randint(2, 400)
        f0, f1 = rng.uniform(0.02, 0.45), rng.uniform(0.02, 0.45)
        if (1 - f0) - f1 < 0.06:
            continue
        abar = rng.uniform(f1 + 0.02, 1 - f0 - 0.02)
        assert classify_case(abar, f0, f1) == 1
        assert log2_A12_asymptotic(N, P, abar, f0, f1) == pytest.approx(
            ref_case1(N, P, abar, f0, f1), rel=1e-12
        )
        checked += 1
    assert checked > 0, "every draw was filtered; the test proved nothing"


def test_case2_matches_reference():
    rng = random.Random(1)
    checked = 0
    for _ in range(300):
        N, P = rng.randint(2, 400), rng.randint(2, 400)
        abar = rng.uniform(0.05, 0.6)
        f0 = rng.uniform(0.02, 0.9 - abar)
        if 1 - abar - f0 <= 0.02:
            continue
        assert classify_case(abar, f0, abar) == 2
        assert log2_A12_asymptotic(N, P, abar, f0, abar) == pytest.approx(
            ref_case2(N, P, abar, f0), rel=1e-12
        )
        checked += 1
    assert checked > 0, "every draw was filtered; the test proved nothing"


def test_case3_matches_reference_through_the_bijection():
    """There is no ref_case3; Case 3 is checked through A -> 1 - A.

    The bijection sends (abar, f0, f1) to (1 - abar, f1, f0), so a Case 3
    point must agree with ref_case2 evaluated at (1 - abar, f1).  Asserting
    the classification first matters: without it, a bug that routed every
    point to Case 2 would leave this test passing.
    """
    rng = random.Random(2)
    for _ in range(300):
        N, P = rng.randint(2, 400), rng.randint(2, 400)
        abar = rng.uniform(0.4, 0.95)
        f0 = 1 - abar
        f1 = rng.uniform(0.02, abar - 0.02)
        assert classify_case(abar, f0, f1) == 3
        assert log2_A12_asymptotic(N, P, abar, f0, f1) == pytest.approx(
            ref_case2(N, P, 1 - abar, f1), rel=1e-12
        )


def test_case4_matches_reference():
    rng = random.Random(3)
    for _ in range(300):
        N, P = rng.randint(2, 400), rng.randint(2, 400)
        abar = rng.uniform(0.05, 0.95)
        assert classify_case(abar, 1 - abar, abar) == 4
        assert log2_A12_asymptotic(N, P, abar, 1 - abar, abar) == pytest.approx(
            ref_case4(N, P, abar), rel=1e-12
        )


def test_case4_reproduces_legacy_symmetric_approximation():
    """The old regular-case formula is Case 4 at abar = 1/2.

    This is the backward-compatibility guarantee for getA12's published
    output: whatever else changed in 0.2.0, the number a 0.1.0 caller saw at
    the symmetric point must be the number they see now.
    """
    for N, P in [(5, 5), (10, 10), (7, 13), (50, 50), (100, 80)]:
        upper = 2*N*P - 0.5*(
            P*math.log2(math.pi*N) + N*math.log2(math.pi*P) - math.log2(math.pi*N*P)
        )
        legacy = upper - ((P + N - 1) ** 2) / (8 * N * P) * L2E
        assert log2_A12_asymptotic(N, P, 0.5, 0.5, 0.5) == pytest.approx(
            legacy, rel=1e-12
        )


# ---------------------------------------------------------------------------
# TODO (step 3): agreement with exact counts; classification, helpers and
# error handling; the details payload.
# ---------------------------------------------------------------------------
