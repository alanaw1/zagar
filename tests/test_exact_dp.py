"""Tests for zagar.exact_dp and the count_A12 backend dispatcher.

The dynamic program is checked three ways: against a brute-force enumeration of
every admixed array at tiny sizes, against the independent column-sum
enumeration in ``_core``, and against the one value published in the manuscript.
"""

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
                    for l in (0, 1):
                        j = l * P + p
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
# TODO (Alan): the test functions go below.  See the checklist in the plan.
#
# Correctness
#   - the manuscript value: count_A12_dp(*PAPER_EXAMPLE) == PAPER_EXAMPLE_COUNT
#   - agreement with brute_A12 over ~120 random tiny instances
#   - agreement with zagar.getA12 over ~120 instances up to max_N=max_P=4
#   - non-uniform margins (the paper example already is non-uniform)
#   - reorder_loci=True and reorder_loci=False give the same count
#
# Edge cases
#   - all-zero constraints -> 1
#   - infeasible -> 0 (e.g. (2, [0,0], [0], [3]) and (2, [4,4], [1,0], [0,0]);
#     work out for yourself why each is infeasible before asserting it)
#   - single individual, every a in [0, 2P], against brute_A12
#   - a short a_vec is zero-padded, and agrees with getA12 on the same input
#   - parametrized invalid inputs raise ValueError: a_vec longer than N;
#     phi vectors of different length; a_n > 2P; phi0 > 2N; phi0 + phi1 > 2N;
#     N <= 0
#
# The count_A12 dispatcher
#   - all four methods agree on random instances
#   - "auto" defaults to dp; "column-sums" gives the same answer
#   - an unknown method raises ValueError matching "unknown method"
#   - kwargs reach the backend (e.g. method="dp", reorder_loci=False)
#
# End to end
#   - the asymptotic value tracks log2 of the exact count to within 0.05 bits
#     at (4,4,.5,.25,.25), (4,4,.5,.25,.5), (4,4,.5,.5,.5), (5,5,.4,.3,.4)
# ---------------------------------------------------------------------------
