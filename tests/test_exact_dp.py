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
from zagar.exact_dp import _locus_transitions, count_A12_dp

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


def test_complement_bijection_preserves_the_count():
    """``A -> 1 - A`` is a bijection on admixed arrays, so it cannot change
    ``|A12|``.

    Complementing ``A`` sends row sum ``a_n`` to ``2P - a_n`` and exchanges the
    cell types ``(1,1) <-> (0,1)`` and ``(1,0) <-> (0,0)``, so ``phi0`` and
    ``phi1`` swap while ``X`` is untouched.  Nothing here requires semi-regular
    margins, which is why the paper example is included alongside the random
    draws.

    ``zagar.asymptotics`` leans on this identity -- ``_case3`` is implemented
    as ``_case2`` at the image point -- so verifying it against enumeration is
    what keeps the invariance tests in ``test_asymptotics.py`` from resting on
    an unchecked claim.
    """
    N, a, phi0, phi1 = PAPER_EXAMPLE
    P = len(phi0)
    assert (
        count_A12_dp(N, [2 * P - v for v in a], phi1, phi0) == PAPER_EXAMPLE_COUNT
    )

    rng = random.Random(5)
    nonzero = 0
    for _ in range(200):
        N, a, phi0, phi1 = random_instance(rng, max_N=3, max_P=3)
        P = len(phi0)
        count = count_A12_dp(N, a, phi0, phi1)
        assert count == count_A12_dp(N, [2 * P - v for v in a], phi1, phi0)
        nonzero += count > 0
    # Both sides are zero on an infeasible instance, so a draw that is mostly
    # infeasible would satisfy the identity without exercising it.
    assert nonzero > 20, f"only {nonzero} feasible draws; the test is weak"


# ---------------------------------------------------------------------------
# The C++ transition backend
#
# _core.locus_transitions is a port of _locus_transitions, which is ~86% of the
# dynamic program's runtime.  The Python version stays as the reference the
# port is checked against, so these tests compare the two directly rather than
# only comparing final counts: a port that was wrong on some histogram the DP
# never visits would otherwise pass.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    zagar.exact_dp._locus_transitions_cpp is None,
    reason="built without the C++ extension",
)
def test_cpp_transitions_match_python_exhaustively():
    """Every histogram of at most 6 rows over at most 5 levels."""
    cpp = zagar.exact_dp._locus_transitions_cpp
    checked = 0
    for width in range(1, 6):
        for total in range(1, 7):
            for hist in product(range(total + 1), repeat=width):
                if sum(hist) != total:
                    continue
                if width > 1 and hist[-1] == 0:
                    continue        # not trimmed; the DP never produces these
                got = {(h, s): w for h, s, w in cpp(bytes(hist))}
                want = {(h, s): w for h, s, w in _locus_transitions(hist)}
                assert got == want, hist
                checked += 1
    assert checked > 200, f"only {checked} histograms covered"


@pytest.mark.skipif(
    zagar.exact_dp._locus_transitions_cpp is None,
    reason="built without the C++ extension",
)
def test_cpp_transitions_match_python_on_random_histograms():
    """Wider and taller than the exhaustive sweep reaches."""
    cpp = zagar.exact_dp._locus_transitions_cpp
    rng = random.Random(31)
    for _ in range(300):
        total = rng.randint(1, 12)
        width = rng.randint(1, 8)
        hist = [0] * width
        for _ in range(total):
            hist[rng.randrange(width)] += 1
        while len(hist) > 1 and hist[-1] == 0:
            hist.pop()
        got = {(h, s): w for h, s, w in cpp(bytes(hist))}
        want = {(h, s): w for h, s, w in _locus_transitions(tuple(hist))}
        assert got == want, hist


@pytest.mark.skipif(
    zagar.exact_dp._locus_transitions_cpp is None,
    reason="built without the C++ extension",
)
def test_cpp_transitions_raise_rather_than_wrap_on_overflow():
    """Weights are uint64; past the documented bound it must refuse, not wrap.

    Total weight out of one histogram is at most 4**sum(hist), so 31 rows is
    the last safe size.  A silent wrap here would produce a plausible-looking
    wrong count, which is the failure mode this package has already been bitten
    by once.
    """
    assert zagar.exact_dp._CPP_MAX_N == 31
    zagar.exact_dp._locus_transitions_cpp(bytes([0, 0, 31]))      # fine
    with pytest.raises(OverflowError):
        zagar.exact_dp._locus_transitions_cpp(bytes([0, 0, 40]))


@pytest.mark.parametrize("cache", [False, True])
def test_backends_and_cache_agree_on_the_count(cache):
    """The count must not depend on the transition backend or the cache.

    ``cache_transitions`` and the C++ port are both pure optimisations, so this
    is the end-to-end statement of that claim.
    """
    rng = random.Random(13)
    for _ in range(40):
        N, a, phi0, phi1 = random_instance(rng, max_N=4, max_P=3)
        want = count_A12_dp(N, a, phi0, phi1)
        assert count_A12_dp(N, a, phi0, phi1, cache_transitions=cache) == want
    assert count_A12_dp(*PAPER_EXAMPLE, cache_transitions=cache) == PAPER_EXAMPLE_COUNT


@pytest.mark.skipif(
    zagar.exact_dp._count_A12_dp_cpp is None,
    reason="built without the C++ extension",
)
def test_cpp_and_python_backends_agree():
    """The whole-DP C++ backend against the Python driver, the reference.

    ``backend="cpp"`` re-implements the driver -- locus ordering, the weight
    table, the suffix bounds, all three prunes, the final key -- so this is a
    differential test of two implementations of one algorithm, not a test of
    one implementation twice.  It is the only thing standing between a
    transcription slip in the C++ and a wrong count.
    """
    rng = random.Random(23)
    for _ in range(250):
        N, a, phi0, phi1 = random_instance(rng, max_N=4, max_P=3)
        want = count_A12_dp(N, a, phi0, phi1, backend="python")
        assert count_A12_dp(N, a, phi0, phi1, backend="cpp") == want
    for args in [
        PAPER_EXAMPLE,
        (3, [0, 0, 0], [0, 0], [0, 0]),     # all-zero: empty products are 1
        (2, [0, 0], [0], [3]),              # infeasible, no ancestry 1
        (2, [4, 4], [1, 0], [0, 0]),        # infeasible, no ancestry 0
        (1, [2], [0], [0]),                 # single individual
        (2, [1], [0], [0]),                 # short a_vec, zero-padded
        (4, [4] * 4, [2] * 4, [2] * 4),     # Case 1 interior
        (5, [5] * 5, [2] * 5, [3] * 5),     # non-symmetric, N = P = 5
        (6, [6] * 6, [3] * 6, [3] * 6),     # Case 1 interior, N = P = 6
        (3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1]),   # non-uniform margins
    ]:
        assert (count_A12_dp(*args, backend="cpp")
                == count_A12_dp(*args, backend="python"))


@pytest.mark.skipif(
    zagar.exact_dp._count_A12_dp_cpp is None,
    reason="built without the C++ extension",
)
@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_reorder_loci_is_a_no_op_for_both_backends(backend):
    """Each backend sorts loci itself, so each needs the invariant checked."""
    rng = random.Random(29)
    for _ in range(60):
        N, a, phi0, phi1 = random_instance(rng, max_N=3, max_P=3)
        assert (count_A12_dp(N, a, phi0, phi1, backend=backend,
                             reorder_loci=True)
                == count_A12_dp(N, a, phi0, phi1, backend=backend,
                                reorder_loci=False))


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        count_A12_dp(*PAPER_EXAMPLE, backend="nope")


@pytest.mark.parametrize("backend", ["auto", "cpp", "python"])
def test_invalid_inputs_raise_on_every_backend(backend):
    """Validation happens before backend selection, so it must not vary.

    The C++ path would otherwise be free to accept something the reference
    rejects, or to report it differently.
    """
    for N, a, phi0, phi1 in [
        (2, [1, 1, 1], [0], [0]),
        (2, [1, 1], [0, 0], [0]),
        (2, [9, 1], [0], [0]),
        (2, [1, 1], [5], [0]),
        (2, [1, 1], [3], [3]),
        (0, [], [], []),
    ]:
        with pytest.raises(ValueError):
            count_A12_dp(N, a, phi0, phi1, backend=backend)


def test_python_fallback_gives_the_same_count(monkeypatch):
    """Forcing the pure-Python kernel must not change any answer.

    Guards the fallback path that a build without the extension would take.
    """
    monkeypatch.setattr(zagar.exact_dp, "_locus_transitions_cpp", None)
    assert count_A12_dp(*PAPER_EXAMPLE) == PAPER_EXAMPLE_COUNT
    rng = random.Random(17)
    for _ in range(30):
        N, a, phi0, phi1 = random_instance(rng, max_N=3, max_P=3)
        assert count_A12_dp(N, a, phi0, phi1) == zagar.getA12(N, a, phi0, phi1)


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
            for m in ("auto", "dp", "histogram-dp",
                      "column-sums", "miller-harrison",
                      "column-sums-parallel", "miller-harrison-parallel",
                      "column-sums-pruning", "miller-harrison-pruning")
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
        ("dp", {"backend": "python"}),
        ("dp", {"backend": "cpp"}),
        ("miller-harrison", {"sort_p_desc": False}),
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
