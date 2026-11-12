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
# Agreement with exact counts
#
# Steps above prove exact_dp is right and that the closed forms match the R
# derivation.  Both could hold while the expansion described a different
# quantity; this is the test that connects the two.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,P,abar,f0,f1,exact_log2,case",
    [
        (4, 4, 0.5, 0.25, 0.25, 43.7463, 1),
        (4, 4, 0.375, 0.25, 0.25, 41.0235, 1),
        (4, 6, 1 / 3, 0.25, 0.25, 60.2523, 1),
        (4, 6, 0.5, 0.25, 0.25, 68.0568, 1),
        # The one row not independently re-derived: the exact count at
        # N = P = 10 in Case 1 exceeds 16 GB in count_A12_dp, so this oracle
        # is carried over from the reference suite on trust.  The implied
        # error, +0.0094 bits, fits the trend of its neighbours.
        (10, 10, 0.5, 0.25, 0.25, 330.0200, 1),
        (4, 4, 0.5, 0.25, 0.5, 29.9929, 2),
        (4, 8, 0.5, 0.25, 0.5, 63.3956, 2),
        (6, 6, 0.5, 0.5, 1 / 6, 72.8085, 3),
        (4, 4, 0.5, 0.5, 0.5, 19.6530, 4),
        (4, 4, 0.25, 0.75, 0.25, 14.8899, 4),
        (6, 6, 0.5, 0.5, 0.5, 49.3671, 4),
        (8, 8, 0.25, 0.75, 0.25, 72.7581, 4),
    ],
)
def test_close_to_exact_counts(N, P, abar, f0, f1, exact_log2, case):
    """Within 0.04 bits of exact enumeration even at these very small sizes.

    The lower bound is the interesting half: the expansion must *over*estimate
    at finite size, so a sign error would pass a two-sided bound.  0.04 is
    close to binding -- at (4, 4, 0.25, 0.75, 0.25) the error is +0.0358.
    """
    got = log2_A12_asymptotic(N, P, abar, f0, f1, details=True)
    assert got["case"] == case
    err = got["log2_A12"] - exact_log2
    assert 0 < err < 0.04, f"error {err:+.4f} bits outside the expected band"


# ---------------------------------------------------------------------------
# Case classification, helpers and error handling
# ---------------------------------------------------------------------------


def test_classify_interior_and_boundaries():
    assert classify_case(0.5, 0.25, 0.25) == 1
    assert classify_case(0.4, 0.3, 0.4) == 2
    assert classify_case(0.7, 0.3, 0.2) == 3
    assert classify_case(0.3, 0.7, 0.3) == 4


def test_classify_uses_tolerance_near_the_boundary():
    """A point a hair inside Case 1 is treated as the boundary case.

    The Case 1 correction diverges like 1 / (abar - f1) as f1 -> abar, so the
    boundary formula is the right one there.  This is the only test that pins
    the boundary-absorption design, so it is also the one that would notice if
    the tolerance were dropped from classify_case's feasibility checks.
    """
    assert classify_case(0.4, 0.3, 0.4 - 1e-12) == 2
    assert classify_case(0.4, 0.3, 0.4 - 1e-3) == 1


def test_classify_absorbs_a_hair_past_the_boundary():
    """A point just *outside* the feasible region is absorbed, not rejected.

    The two feasibility checks in classify_case are themselves tolerant, so a
    triple intended as Case 2 or Case 3 that floating-point arithmetic has
    nudged a hair into infeasibility still classifies rather than raising.
    This is the half of the tolerance design that the test above cannot reach:
    there f1 < abar, so the ``f1 > abar`` branch never fires.
    """
    assert classify_case(0.4, 0.3, 0.4 + 1e-12) == 2
    assert classify_case(0.4, 0.6 + 1e-12, 0.2) == 3
    # ... while a point genuinely outside the region must still raise.
    with pytest.raises(ValueError):
        classify_case(0.4, 0.3, 0.4 + 1e-3)
    with pytest.raises(ValueError):
        classify_case(0.4, 0.6 + 1e-3, 0.2)


@pytest.mark.parametrize(
    "abar,f0,f1",
    [
        (0.3, 0.3, 0.5),   # f1 > abar: no ancestry 1 to carry the alt alleles
        (0.9, 0.5, 0.2),   # abar > 1 - f0: no ancestry 0 to carry them
        (0.0, 0.3, 0.1),   # abar on the closed boundary
        (0.5, 0.0, 0.1),   # f0 on the closed boundary
        (1.0, 0.1, 0.1),   # abar on the closed boundary
    ],
)
def test_infeasible_or_degenerate_raises(abar, f0, f1):
    with pytest.raises(ValueError):
        classify_case(abar, f0, f1)


def test_nonpositive_sizes_raise():
    with pytest.raises(ValueError):
        log2_A12_asymptotic(0, 5, 0.5, 0.25, 0.25)
    with pytest.raises(ValueError):
        log2_A12_asymptotic(5, -1, 0.5, 0.25, 0.25)


def test_entropy_bits():
    assert entropy_bits(0.5, 0.5) == pytest.approx(1.0)
    assert entropy_bits(0.25, 0.25, 0.25, 0.25) == pytest.approx(2.0)
    assert entropy_bits(1.0, 0.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        entropy_bits(0.5, 0.4)


def test_entropy_bits_rejects_negative_components():
    """Components summing to 1 but with a negative entry must still raise.

    This is the case the sum check alone cannot catch, and a density computed
    as 1 - abar - f0 is exactly where a negative would come from.
    """
    with pytest.raises(ValueError):
        entropy_bits(1.2, -0.2)


def test_densities_from_constraints():
    d = densities_from_constraints(4, [4] * 4, [2] * 4, [2] * 4)
    assert (d["N"], d["P"]) == (4, 4)
    assert (d["abar"], d["f0"], d["f1"]) == (0.5, 0.25, 0.25)


def test_densities_denominators_differ_between_row_and_column():
    """abar is out of 2P, f0 and f1 out of 2N; a square case cannot tell them
    apart, so use N != P."""
    d = densities_from_constraints(2, [3] * 2, [1] * 4, [2] * 4)
    assert d["abar"] == 3 / 8      # 2P = 8
    assert d["f0"] == 1 / 4        # 2N = 4
    assert d["f1"] == 2 / 4


def test_densities_rejects_non_uniform():
    with pytest.raises(ValueError, match="semi-regular"):
        densities_from_constraints(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1])


@pytest.mark.parametrize(
    "N,a_vec,phi0_vec,phi1_vec",
    [
        (3, [4, 4], [2], [2]),          # a_vec is not length N
        (2, [2, 2], [2, 2], [2]),       # phi vectors differ in length
    ],
)
def test_densities_rejects_bad_lengths(N, a_vec, phi0_vec, phi1_vec):
    with pytest.raises(ValueError):
        densities_from_constraints(N, a_vec, phi0_vec, phi1_vec)


# ---------------------------------------------------------------------------
# The details payload
# ---------------------------------------------------------------------------


def test_details_payload_is_consistent():
    d = log2_A12_asymptotic(20, 20, 0.5, 0.25, 0.25, details=True)
    assert set(d) == {"log2_A12", "case", "entropy", "dimension", "K", "log2_delta"}
    assert d["dimension"] == 20 + 2 * 20
    assert d["log2_A12"] == pytest.approx(
        log2_A12_asymptotic(20, 20, 0.5, 0.25, 0.25)
    )


@pytest.mark.parametrize(
    "N,P,abar,f0,f1",
    [
        (20, 20, 0.5, 0.25, 0.25),      # case 1
        (20, 30, 0.4, 0.3, 0.4),        # case 2
        (30, 20, 0.7, 0.3, 0.2),        # case 3
        (20, 20, 0.3, 0.7, 0.3),        # case 4
    ],
)
def test_details_satisfies_the_assembly_identity(N, P, abar, f0, f1):
    """log2_A12 must equal the assembled form of the other four fields.

    Deliberately rebuilt from the formula longhand rather than by calling
    _assemble: calling it would compare the module to itself and pass even if
    _assemble were wrong.  Written this way, the test pins log2_delta to the
    real log2(Delta) -- the reference implementation back-solved it and
    reported half that value, which its own key-set assertion never caught.
    """
    d = log2_A12_asymptotic(N, P, abar, f0, f1, details=True)
    rebuilt = (
        2 * N * P * d["entropy"]
        - 0.5 * d["log2_delta"]
        - 0.5 * d["dimension"] * math.log2(math.pi)
        - d["dimension"]
        + d["K"] * L2E
    )
    assert rebuilt == pytest.approx(d["log2_A12"], rel=1e-15)
