"""Tests for zagar.asymptotics (Theorem 3.1, all four regimes).

Correctness rests on three oracles, none of which needs anything outside this
package:

* **Exact counts.**  ``zagar.exact_dp`` enumerates ``|A12|`` directly, so
  :func:`test_close_to_exact_counts` is what ties the expansion to the
  quantity it claims to approximate.  It is the only oracle here that could
  notice the formula describing a different object altogether.
* **The complement bijection.**  ``A -> 1 - A`` leaves ``|A12|`` unchanged and
  sends ``(abar, f0, f1)`` to ``(1 - abar, f1, f0)``, so the expansion must be
  invariant under it exactly rather than asymptotically.  The bijection itself
  is checked against enumeration by
  ``test_complement_bijection_preserves_the_count`` in ``test_exact_dp.py``,
  so the invariance tests below do not rest on an unchecked claim.
* **The 0.1.0 symmetric formula.**  Case 4 at ``abar = 1/2`` must reproduce
  what ``getA12(return_approx=True)`` published before this release.

During development the closed forms were also compared against the expanded
polynomials of the reference R derivation (``four_cases_freeze_101626.R``) over
several thousand random points in every regime, agreeing to 1e-12 relative.
That was a one-time check on the transcription and is deliberately not part of
this suite: the R script is not distributed with the package, so nobody
installing ``zagar`` could reproduce it, and a failure years from now would be
undebuggable.  It is preserved in the history at commit 88f3508.
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
# Invariance under the complement bijection
#
# Replacing A by 1 - A is a bijection on admixed arrays: row sums go from a_n
# to 2P - a_n, the cell types (1,1) and (0,1) are exchanged along with (1,0)
# and (0,0), and X is untouched.  So it maps (abar, f0, f1) to
# (1 - abar, f1, f0) and preserves the count exactly.
#
# Every term of _K_case1 is built from alpha, {p0, pt} and {f0, f1} in
# combinations this map permutes, so it is a real constraint on the correction:
# a slip that breaks that structure shows up here.  It cannot see a slip that
# preserves it -- a mis-scaled denominator, say -- which is what the exact
# counts and the golden table are for.
# ---------------------------------------------------------------------------


def test_complement_invariance_in_the_interior():
    """Case 1 is where this bites, since all four cell types are present.

    H, log2 Delta, D and K must each be invariant, and the tolerance is set
    about twenty times the observed worst case (5e-16 over these draws) so
    that it measures the formula rather than the floating-point noise.
    """
    rng = random.Random(17)
    checked = 0
    for _ in range(400):
        N, P = rng.randint(2, 400), rng.randint(2, 400)
        f0, f1 = rng.uniform(0.02, 0.45), rng.uniform(0.02, 0.45)
        if (1 - f0) - f1 < 0.06:
            continue
        abar = rng.uniform(f1 + 0.02, 1 - f0 - 0.02)
        assert classify_case(abar, f0, f1) == 1
        assert classify_case(1 - abar, f1, f0) == 1
        assert log2_A12_asymptotic(N, P, abar, f0, f1) == pytest.approx(
            log2_A12_asymptotic(N, P, 1 - abar, f1, f0), rel=1e-14
        )
        checked += 1
    assert checked > 0, "every draw was filtered; the test proved nothing"


def test_complement_invariance_on_the_case4_edge():
    """On the doubly degenerate edge the image is again a Case 4 point.

    Independent of the test above: Case 4 has its own correction term, which
    shares no code with Case 1's.
    """
    rng = random.Random(19)
    for _ in range(200):
        N, P = rng.randint(2, 400), rng.randint(2, 400)
        abar = rng.uniform(0.05, 0.95)
        assert classify_case(abar, 1 - abar, abar) == 4
        assert classify_case(1 - abar, abar, 1 - abar) == 4
        assert log2_A12_asymptotic(N, P, abar, 1 - abar, abar) == pytest.approx(
            log2_A12_asymptotic(N, P, 1 - abar, abar, 1 - abar), rel=1e-14
        )


def test_complement_exchanges_case2_and_case3():
    """A structural guard, not a check on the arithmetic -- deliberately so.

    ``_case3`` is implemented as ``_case2`` evaluated at the image point, so
    the value assertion below is true by construction and cannot fail today.
    What it does pin is the classification: the complement must exchange the
    two faces.  If Case 3 is ever given an implementation of its own, this
    stops being tautological and becomes the test of it -- which is why it is
    written now, with the circularity stated rather than hidden.
    """
    rng = random.Random(23)
    checked = 0
    for _ in range(200):
        N, P = rng.randint(2, 400), rng.randint(2, 400)
        abar = rng.uniform(0.05, 0.6)
        f0 = rng.uniform(0.02, 0.9 - abar)
        if 1 - abar - f0 <= 0.02:
            continue
        assert classify_case(abar, f0, abar) == 2
        assert classify_case(1 - abar, abar, f0) == 3
        assert log2_A12_asymptotic(N, P, abar, f0, abar) == pytest.approx(
            log2_A12_asymptotic(N, P, 1 - abar, abar, f0), rel=1e-14
        )
        checked += 1
    assert checked > 0, "every draw was filtered; the test proved nothing"


# ---------------------------------------------------------------------------
# Backward compatibility with the 0.1.0 published formula
# ---------------------------------------------------------------------------


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
# The oracle that connects the expansion to the thing it approximates.  The
# invariance tests above and the golden table below could both pass while the
# formula described a different quantity; this is the one that would not.
#
# Every value was produced by count_A12_dp from integer constraints built with
# exact rational arithmetic, and checked to round-trip back through
# densities_from_constraints to the densities in the row.  That round trip is
# not optional: round(2 * 9 * 0.25) is 4, not 5, so the obvious way of writing
# a row silently enumerates a different parameter point.
#
# Case 1 is the expensive regime -- N = P = 9 takes 47 s and 2.5 GiB, and
# N = P = 10 exceeds 16 GiB -- while the degenerate regimes are cheap, which is
# why cases 2, 3 and 4 reach further.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,P,abar,f0,f1,exact_log2,case",
    [
        (4, 4, 0.5, 0.25, 0.25, 43.7463, 1),
        (4, 4, 0.375, 0.25, 0.25, 41.0235, 1),
        (4, 6, 1 / 3, 0.25, 0.25, 60.2523, 1),
        (4, 6, 0.5, 0.25, 0.25, 68.0568, 1),
        (6, 6, 0.5, 0.25, 0.25, 108.5082, 1),
        (8, 8, 0.5, 0.25, 0.25, 203.8089, 1),
        # f0 = f1 = 1/3 rather than 1/4: at N = 9, 2N * 0.25 is not an integer,
        # so a quarter-density row cannot be built here at all.
        (9, 9, 0.5, 1 / 3, 1 / 3, 248.9962, 1),
        (4, 4, 0.5, 0.25, 0.5, 29.9929, 2),
        (4, 8, 0.5, 0.25, 0.5, 63.3956, 2),
        (6, 6, 0.5, 0.25, 0.5, 75.2987, 2),
        (8, 8, 0.5, 0.25, 0.5, 142.9976, 2),
        (5, 5, 0.4, 0.3, 0.4, 53.0782, 2),
        (6, 6, 0.5, 0.5, 1 / 6, 72.8085, 3),
        (6, 8, 0.5, 0.5, 0.25, 102.6406, 3),
        # Same count as the (8, 8) Case 2 row above, because the two points are
        # complement images of each other.  The bijection is visible in the
        # oracle data, not only in the formula.
        (8, 8, 0.5, 0.5, 0.25, 142.9976, 3),
        (4, 4, 0.5, 0.5, 0.5, 19.6530, 4),
        (4, 4, 0.25, 0.75, 0.25, 14.8899, 4),
        (6, 6, 0.5, 0.5, 0.5, 49.3671, 4),
        (8, 8, 0.25, 0.75, 0.25, 72.7581, 4),
        (10, 10, 0.5, 0.5, 0.5, 153.7482, 4),
        (12, 12, 0.25, 0.75, 0.25, 179.1586, 4),
    ],
)
def test_close_to_exact_counts(N, P, abar, f0, f1, exact_log2, case):
    """Within 0.04 bits of exact enumeration even at these very small sizes.

    The lower bound is the interesting half: the expansion must *over*estimate
    at finite size, so a sign error would pass a two-sided bound.  0.04 is
    close to binding -- at (4, 4, 0.25, 0.75, 0.25) the error is +0.0358,
    the largest over all 21 rows.
    """
    got = log2_A12_asymptotic(N, P, abar, f0, f1, details=True)
    assert got["case"] == case
    err = got["log2_A12"] - exact_log2
    assert 0 < err < 0.04, f"error {err:+.4f} bits outside the expected band"


# ---------------------------------------------------------------------------
# Golden values
#
# Regression, not correctness.  These are this implementation's own output,
# recorded once it had passed everything else in this file and the one-time
# comparison against the reference R derivation described in the module
# docstring.  They exist to make any later change to the closed forms visible,
# including changes far too small for the 0.04-bit band above to notice.
#
# A failure here means the output moved.  It does not by itself mean the output
# became wrong, and the right response is to find out which -- starting with
# the exact-count and invariance tests, which do carry mathematical content.
#
# rel=1e-12 rather than something tighter so the table survives a different
# libm or a reordered sum.  It is still four orders of magnitude finer than the
# smallest coefficient slip worth worrying about: the least consequential
# mutation probed while designing this table moved a value by 0.002 bits, or
# 2e-8 relative at these magnitudes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,P,abar,f0,f1,expected",
    [
        (5, 5, 0.5, 0.25, 0.25, 72.35534171838236),
        (20, 30, 0.5, 0.25, 0.25, 2169.6157555414584),
        (7, 13, 0.4, 0.3, 0.2, 288.31588441477317),
        (200, 150, 0.6, 0.2, 0.1, 103542.56063238299),
        (5, 5, 0.4, 0.3, 0.4, 53.09747014490303),
        (20, 30, 0.5, 0.25, 0.5, 1574.3909047061406),
        (150, 200, 0.3, 0.5, 0.3, 86786.9204012328),
        (30, 20, 0.7, 0.3, 0.2, 1580.4335862541614),
        (6, 6, 0.5, 0.5, 1 / 6, 72.82408957892738),
        (100, 80, 0.8, 0.2, 0.5, 22785.644738424107),
        (4, 4, 0.5, 0.5, 0.5, 19.66748185150658),
        (20, 20, 0.3, 0.7, 0.3, 594.8397008412535),
        (100, 80, 0.25, 0.75, 0.25, 12293.532615087332),
    ],
)
def test_golden_values(N, P, abar, f0, f1, expected):
    assert log2_A12_asymptotic(N, P, abar, f0, f1) == pytest.approx(
        expected, rel=1e-12
    )


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
