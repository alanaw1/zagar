"""Tests for zagar.exact_dp and the count_A12 backend dispatcher.

The dynamic program is checked three ways: against a brute-force enumeration of
every admixed array at tiny sizes, against the independent column-sum
enumeration in ``_core``, and against the one value published in the manuscript.
"""

import math
import random
from itertools import combinations, product

import pytest

import zagar
from zagar.exact_dp import count_A12_dp

PAPER_EXAMPLE = (3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1])
PAPER_EXAMPLE_COUNT = 133520


# ---------------------------------------------------------------------------
# Brute force, for the smallest cases only
# ---------------------------------------------------------------------------


def brute_A12(N, a_vec, phi0_vec, phi1_vec):
    """Enumerate (A_row, X_row) per individual and match the locus totals.

    This shares no code and no reasoning with ``count_A12_dp``: it builds every
    admixed array cell by cell, rather than collapsing each locus into a count
    in ``{0, 1, 2}``.  That independence is the point -- it is the only check
    here that does not assume the generating identity of ``exact_dp`` is right.

    Cost is ``prod_n (C(2P, a_n) * 2^(2P))``, so it is usable only for
    ``N <= 3``, ``P <= 2`` or thereabouts.
    """
    P = len(phi0_vec)
    W = 2 * P
    per_row = []
    for n in range(N):
        opts = []
        for ones in combinations(range(W), a_vec[n]):
            A = [0] * W
            for i in ones:
                A[i] = 1
            for xbits in product((0, 1), repeat=W):
                c0 = [0] * P
                c1 = [0] * P
                for p in range(P):
                    for hap in (0, 1):
                        j = hap * P + p
                        if xbits[j]:
                            (c1 if A[j] else c0)[p] += 1
                opts.append((tuple(c0), tuple(c1)))
        per_row.append(opts)
    tgt0, tgt1 = tuple(phi0_vec), tuple(phi1_vec)
    total = 0
    for combo in product(*per_row):
        s0 = [0] * P
        s1 = [0] * P
        for c0, c1 in combo:
            for p in range(P):
                s0[p] += c0[p]
                s1[p] += c1[p]
        if tuple(s0) == tgt0 and tuple(s1) == tgt1:
            total += 1
    return total


def random_instance(rng, max_N=3, max_P=2):
    """A random instance that is not necessarily feasible.

    ``phi1`` is drawn against the room ``phi0`` leaves, so that
    ``phi0_p + phi1_p <= 2N`` always holds and the instance is at least
    *valid*; whether any array satisfies it is left open on purpose, since a
    count of zero is as much a thing to get right as a count of a million.
    """
    N = rng.randint(1, max_N)
    P = rng.randint(1, max_P)
    a = [rng.randint(0, 2 * P) for _ in range(N)]
    phi0 = [rng.randint(0, 2 * N) for _ in range(P)]
    phi1 = [rng.randint(0, 2 * N - x) for x in phi0]
    return N, a, phi0, phi1


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_paper_example():
    """The one value published with the manuscript."""
    assert count_A12_dp(*PAPER_EXAMPLE) == PAPER_EXAMPLE_COUNT


def test_matches_brute_force():
    """The strongest check available: brute_A12 shares no reasoning with the DP.

    It builds every array cell by cell, so it does not assume the ``C(2, e)``
    collapse, the histogram state or any of the pruning.  A wrong generating
    identity would show up here and nowhere else.
    """
    rng = random.Random(1)
    for _ in range(120):
        N, a, phi0, phi1 = random_instance(rng)
        assert count_A12_dp(N, a, phi0, phi1) == brute_A12(N, a, phi0, phi1)


def test_matches_column_sum_enumeration():
    """The two backends are independent algorithms; they must agree exactly."""
    rng = random.Random(7)
    for _ in range(120):
        N, a, phi0, phi1 = random_instance(rng, max_N=4, max_P=4)
        assert count_A12_dp(N, a, phi0, phi1) == zagar.getA12(N, a, phi0, phi1)


def test_non_uniform_margins():
    """Margins need not be semi-regular; the paper example already is not."""
    assert count_A12_dp(2, [3, 1], [1, 2], [2, 1]) == brute_A12(
        2, [3, 1], [1, 2], [2, 1]
    )
    assert count_A12_dp(3, [4, 2, 0], [2, 1], [1, 3]) == brute_A12(
        3, [4, 2, 0], [2, 1], [1, 3]
    )


def test_reordering_loci_does_not_change_the_count():
    """Permuting loci is a bijection on arrays preserving the row sums.

    ``reorder_loci`` is a speed heuristic, so this needs no oracle: it turns
    the docstring's claim into something the suite enforces.
    """
    rng = random.Random(11)
    for _ in range(40):
        N, a, phi0, phi1 = random_instance(rng, max_N=3, max_P=3)
        assert count_A12_dp(N, a, phi0, phi1, reorder_loci=True) == count_A12_dp(
            N, a, phi0, phi1, reorder_loci=False
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_all_zero_constraints():
    # A is all zeros and X is all zeros: exactly one array.
    assert count_A12_dp(3, [0, 0, 0], [0, 0], [0, 0]) == 1


def test_infeasible_returns_zero():
    # Every row sum is 0, so A is identically 0 and no haplotype carries
    # ancestry 1 -- but locus 0 demands three alternative alleles on
    # ancestry-1 haplotypes.
    assert count_A12_dp(2, [0, 0], [0], [3]) == 0
    # a_n = 4 = 2P, so A is identically 1 and every haplotype is ancestry 1;
    # no ancestry-0 allele can be placed, yet phi0 asks for one.
    assert count_A12_dp(2, [4, 4], [1, 0], [0, 0]) == 0


def test_single_individual():
    """With phi0 = phi1 = 0, X is forced to zero and the count is C(2P, a)."""
    for P in (1, 2, 3):
        for a in range(2 * P + 1):
            phi0 = [0] * P
            phi1 = [0] * P
            assert count_A12_dp(1, [a], phi0, phi1) == brute_A12(1, [a], phi0, phi1)


def test_short_a_vec_is_zero_padded():
    """A short a_vec means the other individuals carry no ancestry-1 entries.

    This matches the behaviour of the column-sum backend, which pads to N, so
    the two remain interchangeable.  It is a compatibility contract rather than
    a mathematical necessity, which is why it is pinned here.
    """
    assert count_A12_dp(2, [1], [0], [0]) == count_A12_dp(2, [1, 0], [0], [0])
    assert zagar.getA12(5, [2, 2], [1, 1], [1, 1]) == count_A12_dp(
        5, [2, 2], [1, 1], [1, 1]
    )
    assert count_A12_dp(5, [2, 2], [1, 1], [1, 1]) == count_A12_dp(
        5, [2, 2, 0, 0, 0], [1, 1], [1, 1]
    )


@pytest.mark.parametrize(
    "N,a,phi0,phi1",
    [
        (2, [1, 1, 1], [0], [0]),      # a_vec longer than N
        (2, [1, 1], [0, 0], [0]),      # phi vectors differ in length
        (2, [9, 1], [0], [0]),         # a_n > 2P
        (2, [1, 1], [5], [0]),         # phi0 > 2N
        (2, [1, 1], [3], [3]),         # phi0 + phi1 > 2N
        (0, [], [], []),               # N must be positive
    ],
)
def test_invalid_inputs_raise(N, a, phi0, phi1):
    with pytest.raises(ValueError):
        count_A12_dp(N, a, phi0, phi1)


# ---------------------------------------------------------------------------
# The count_A12 dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_backends_agree():
    """All backends compute the same integer; they differ only in cost."""
    rng = random.Random(3)
    for _ in range(25):
        N, a, phi0, phi1 = random_instance(rng, max_N=3, max_P=3)
        values = {
            m: zagar.count_A12(N, a, phi0, phi1, method=m)
            for m in ("auto", "dp", "column-sums", "column-sums-parallel",
                      "column-sums-pruning")
        }
        assert len(set(values.values())) == 1, values


def test_dispatcher_defaults_to_dp():
    assert zagar.count_A12(*PAPER_EXAMPLE) == PAPER_EXAMPLE_COUNT
    assert (
        zagar.count_A12(*PAPER_EXAMPLE, method="dp") == PAPER_EXAMPLE_COUNT
    )
    assert (
        zagar.count_A12(*PAPER_EXAMPLE, method="column-sums") == PAPER_EXAMPLE_COUNT
    )


def test_dispatcher_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        zagar.count_A12(*PAPER_EXAMPLE, method="nope")


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("dp", {"reorder_loci": False}),
        ("column-sums", {"sort_p_desc": False}),
        ("column-sums-parallel", {"n_threads": 1}),
    ],
)
def test_dispatcher_passes_kwargs_through(method, kwargs):
    """A backend-specific keyword must reach the backend, not raise."""
    assert (
        zagar.count_A12(*PAPER_EXAMPLE, method=method, **kwargs)
        == PAPER_EXAMPLE_COUNT
    )


# ---------------------------------------------------------------------------
# Asymptotics against exact counts, end to end
#
# Distinct from test_close_to_exact_counts in test_asymptotics.py: there the
# exact values are hard-coded, here they are enumerated, so this also exercises
# the round trip from densities to integer constraints and back.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,P,abar,f0,f1",
    [
        (4, 4, 0.5, 0.25, 0.25),
        (4, 4, 0.5, 0.25, 0.5),
        (4, 4, 0.5, 0.5, 0.5),
        (5, 5, 0.4, 0.3, 0.4),
    ],
)
def test_asymptotic_tracks_exact_count(N, P, abar, f0, f1):
    a = [round(2 * P * abar)] * N
    phi0 = [round(2 * N * f0)] * P
    phi1 = [round(2 * N * f1)] * P
    exact = count_A12_dp(N, a, phi0, phi1)
    assert exact > 0
    # The constraints we built must be the ones the densities describe.
    d = zagar.densities_from_constraints(N, a, phi0, phi1)
    assert (d["abar"], d["f0"], d["f1"]) == pytest.approx((abar, f0, f1))
    approx = zagar.log2_A12_asymptotic(N, P, abar, f0, f1)
    assert approx - math.log2(exact) == pytest.approx(0.0, abs=0.05)
