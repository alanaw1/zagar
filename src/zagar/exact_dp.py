"""Exact enumeration of doubly constrained admixed arrays by dynamic programming.

This is an alternative backend to the Miller--Harrison style counter in
``_core``.  Both compute the same integer

    |A12| = #{ [A, X] : row sums of A are a_vec,
                        Phi_{p0} = phi0_vec[p], Phi_{p1} = phi1_vec[p] }.

Method
------
Let ``e_{np}`` in ``{0, 1, 2}`` be the number of individual ``n``'s two
haplotypes at locus ``p`` that carry ancestry 1.  Summing over the placements of
``X`` within each locus first gives

    |A12| = sum over N x P matrices (e_{np}) with row sums a_n of
            prod_{n,p} C(2, e_{np}) * prod_p C(s_p, phi1_p) C(2N - s_p, phi0_p),

where ``s_p = sum_n e_{np}``.  The identity collapses each locus's two columns of
``A``, together with all of ``X``, into one column with entries in ``{0, 1, 2}``:
an entry equal to 1 can be realised in ``C(2, 1) = 2`` ways, and a column with
sum ``s`` admits ``C(s, phi1) C(2N - s, phi0)`` placements of the alternative
alleles.

The program then processes one locus at a time.  Rows with equal *remaining* row
sums are interchangeable, so the state is the histogram of those remaining sums
rather than the per-row vector, and states are memoised.  The locus weight is
applied at each step, so the sum over feasible column-sum vectors is never
enumerated --- which is the step that dominates the Miller--Harrison approach.

Complexity
----------
The state is a composition of ``N`` into at most ``max(a_vec) + 1`` parts, and
the number of reachable states is what makes this exponential: the widest layer
holds 11,802 states at ``N = P = 8`` and 175,070 at ``N = P = 10``, growing by
a factor of seven or eight per unit of ``N``.

Only one layer is held at a time, so that growth is modest in absolute terms --
the whole ``N = P = 10`` run fits in 54 MB.

Measured at the Case 1 interior point ``abar = 1/2``, ``f0 = f1 = 1/4``, with
``cache_transitions=False`` (Apple M2, 16 GB; peak RSS includes a ~10 MB
interpreter baseline).  Only even ``N`` appears because ``2N/4`` has to be an
integer for that density to be realisable at all --- at ``N = 11`` it is 5.5,
and ``round(5.5)`` is 6, not 5:

=======  ==================  =====================  ==================
``N=P``  ``backend="cpp"``   ``backend="python"``   no extension built
=======  ==================  =====================  ==================
4        0.000 s, 24 MB      0.000 s, 23 MB         0.001 s, 24 MB
6        0.004 s, 24 MB      0.041 s, 24 MB         0.19 s, 25 MB
8        0.44 s, 26 MB       2.2 s, 28 MB           9.1 s, 65 MB
10       45 s, 54 MB         166 s, 65 MB           720 s, 542 MB
=======  ==================  =====================  ==================

For scale, the column-sum enumeration in ``_core`` takes 0.014 s at
``N = P = 4``, 2.0 s and 492 MB at ``N = P = 5``, and exhausts 16 GB at
``N = P = 6``.  All of these algorithms are exponential; this one simply has a
much smaller base.

``backend="python"`` is the driver in this module, which still calls the C++
``_core.locus_transitions`` for each histogram; the last column is a build with
no extension, where that enumeration runs in Python too.  The gap between the
first two columns is the cost of the Python boundary.  ``"python"`` marshals a
``(new_hist, s, ways)`` triple per transition --- 7.27 million of them at
``N = P = 8``, over half of which its own prunes then discard --- and holds a
transition table per histogram; ``"cpp"`` accumulates straight into the next
layer and builds no table at all.  The advantage narrows as ``N`` grows (10x at
6, 4.9x at 8, 3.7x at 10) because the enumeration itself comes to dominate.

Past the table, ``N = P = 11`` at ``f0 = f1 = 5/22`` takes 7.8 min and 122 MB
under ``"cpp"`` against 24 min under ``"python"``, and the two agree on all 404
bits of the answer.  Time grows roughly tenfold per unit of ``N`` while memory
grows only two- or threefold, so time is the binding constraint and ``N = 13``
is about the practical ceiling.

This point, and not ``abar = f0 = f1 = 1/2``, is the expensive one.  Setting
``abar = 1/2`` does maximise the number of reachable histograms, but taking
``f0 = f1 = 1/2`` saturates every locus: ``f0 + f1 = 1`` forces ``X == 1``
everywhere (Case 4 of :mod:`zagar.asymptotics`), leaving each locus a single
admissible column sum.  That is far cheaper --- at ``N = P = 7``, 0.0010 s
against 0.034 s under ``"cpp"``.  Skewing ``abar`` away from 1/2 likewise
shrinks the state space.

Non-uniform constraints are supported.  Loci are processed in an order chosen to
prune early, which is valid because permuting loci is a bijection on arrays that
preserves the row sums.
"""

from __future__ import annotations

from math import comb
from typing import Dict, List, Sequence, Tuple

try:
    from ._core import DP_TRANSITIONS_MAX_N as _CPP_MAX_N
    from ._core import count_A12_dp as _count_A12_dp_cpp
    from ._core import locus_transitions as _locus_transitions_cpp
except ImportError:                     # pragma: no cover
    _locus_transitions_cpp = None       # pure-Python fallback
    _count_A12_dp_cpp = None
    _CPP_MAX_N = 0

_BACKENDS = ("auto", "cpp", "python")

__all__ = ["count_A12_dp"]


def _pad_a_vec(N, a_vec):
    """Zero-pad ``a_vec`` to length ``N``, as the ``_core`` backend does.

    A short ``a_vec`` means the remaining individuals carry no ancestry-1
    entries at all, so padding with zeros is the reading that keeps this
    function interchangeable with :func:`zagar.getA12`.
    """
    a_vec = [int(x) for x in a_vec]
    if len(a_vec) > N:
        raise ValueError(f"a_vec has length {len(a_vec)}, which exceeds N = {N}")
    if len(a_vec) < N:
        a_vec = a_vec + [0] * (N - len(a_vec))
    return a_vec


def _validate(N, a_vec, phi0_vec, phi1_vec):
    P = len(phi0_vec)
    if N <= 0:
        raise ValueError("N must be a positive integer")
    if len(phi1_vec) != P:
        raise ValueError("phi0_vec and phi1_vec must have the same length")
    if any(not (0 <= a <= 2 * P) for a in a_vec):
        raise ValueError("every entry of a_vec must lie in [0, 2P]")
    if any(not (0 <= x <= 2 * N) for x in phi0_vec):
        raise ValueError("every entry of phi0_vec must lie in [0, 2N]")
    if any(not (0 <= x <= 2 * N) for x in phi1_vec):
        raise ValueError("every entry of phi1_vec must lie in [0, 2N]")
    if any(x + y > 2 * N for x, y in zip(phi0_vec, phi1_vec)):
        raise ValueError("phi0_vec[p] + phi1_vec[p] must not exceed 2N")
    return P


def _trim(h: List[int]):
    """Drop trailing zeros so that equivalent histograms share one key.

    Returns ``bytes`` when every count fits in a byte, which it does whenever
    ``N <= 255``; this is three to four times smaller than the equivalent tuple
    of ints.  Falls back to a tuple otherwise, though the state space is far
    out of reach at such ``N``, and the C++ transition backend only accepts
    ``bytes``.
    """
    end = len(h)
    while end > 1 and h[end - 1] == 0:
        end -= 1
    view = h[:end]
    try:
        return bytes(view)
    except ValueError:          # a count exceeds 255
        return tuple(view)


def _locus_transitions(hist) -> List[Tuple[object, int, int]]:
    """Ways one locus can reduce the remaining row sums.

    The pure-Python equivalent of ``_core.locus_transitions``, kept as a
    fallback for builds without the extension and as the reference the C++
    version is tested against.  The two return the same triples, so either can
    be dropped into the driver below; the C++ one is 20-40x faster and is used
    whenever it is available and ``N`` is within its 64-bit weight bound.

    Parameters
    ----------
    hist : bytes or tuple of int
        ``hist[r]`` is the number of rows whose remaining row sum is ``r``.

    Returns
    -------
    list of (bytes or tuple, int, int)
        One ``(new_hist, s, ways)`` triple per reachable state, where ``s`` is
        the number of ancestry-1 haplotypes placed at this locus.  The factor
        ``C(2, 1) = 2`` for each row taking the value 1 is included in
        ``ways``.  Distinct choices can land on the same state, so weights are
        summed rather than listed separately.
    """
    levels = [r for r in range(len(hist)) if hist[r]]
    out: Dict[Tuple[Tuple[int, ...], int], int] = {}
    work = list(hist)

    def rec(i: int, s: int, w: int) -> None:
        if i == len(levels):
            key = (_trim(work), s)
            out[key] = out.get(key, 0) + w
            return
        r = levels[i]
        h = hist[r]
        max_k2 = h if r >= 2 else 0
        for k2 in range(max_k2 + 1):
            max_k1 = (h - k2) if r >= 1 else 0
            for k1 in range(max_k1 + 1):
                mult = comb(h, k2) * comb(h - k2, k1) * (1 << k1)
                work[r] -= k1 + k2
                if k1:
                    work[r - 1] += k1
                if k2:
                    work[r - 2] += k2
                rec(i + 1, s + k1 + 2 * k2, w * mult)
                work[r] += k1 + k2
                if k1:
                    work[r - 1] -= k1
                if k2:
                    work[r - 2] -= k2

    rec(0, 0, 1)
    return [(h, s, w) for (h, s), w in out.items()]


def count_A12_dp(
    N: int,
    a_vec: Sequence[int],
    phi0_vec: Sequence[int],
    phi1_vec: Sequence[int],
    reorder_loci: bool = True,
    cache_transitions: bool = False,
    backend: str = "auto",
) -> int:
    """Exact ``|A12|`` by dynamic programming over loci.

    Parameters
    ----------
    N : int
        Number of individuals.
    a_vec : sequence of int
        ``a_vec[n]`` is the number of ancestry-1 entries in row ``n`` of ``A``;
        each must lie in ``[0, 2P]``.  A vector shorter than ``N`` is zero-
        padded, matching :func:`zagar.getA12`; a longer one is an error.
    phi0_vec, phi1_vec : sequence of int
        Length ``P``.  At locus ``p`` there are ``phi0_vec[p]`` alternative
        alleles on ancestry-0 haplotypes and ``phi1_vec[p]`` on ancestry-1
        haplotypes.  Each must lie in ``[0, 2N]``, and their sum must not
        exceed ``2N``.
    reorder_loci : bool, optional
        Process the most constraining loci first (default True).  This only
        affects speed; permuting loci does not change the count.
    cache_transitions : bool, optional
        Memoise each histogram's transition table across layers (default
        False).  A layer is keyed by histogram, so a table is only ever reused
        if the same histogram recurs in a *later* layer, while each cached
        table holds a few hundred triples.  Measured at ``abar = 1/2``,
        ``f0 = f1 = 1/4``:

        =======  ===============  ================  ==================
        ``N=P``  locus enum.      ``False``         ``True``
        =======  ===============  ================  ==================
        8        C++ kernel       2.1 s, 28 MB      2.2 s, 598 MB
        9        C++ kernel       16 s, 35 MB       18 s, 2883 MB
        8        Python kernel    9.1 s, 65 MB      7.1 s, 614 MB
        9        Python kernel    72 s, 174 MB      55 s, 2971 MB
        =======  ===============  ================  ==================

        With the C++ kernel the cache is slower *and* uses two orders of
        magnitude more memory -- recomputing costs less than the dictionary
        lookup it saves -- so there is no reason to enable it.  It helps only
        in a build without the extension, and even there it buys about 23% of
        the time for ten to seventeen times the memory.  Kept as an option
        rather than deleted because that trade is occasionally the right one,
        but it is not the default and should not be.

        This parameter concerns the *transition* enumeration, which is a
        different axis from ``backend`` below: it applies only to
        ``backend="python"``, since ``backend="cpp"`` builds no transition
        table to cache.
    backend : {"auto", "cpp", "python"}, optional
        Which implementation of the dynamic program to run.  All three return
        the same integer.

        ``"auto"`` (default)
            ``"cpp"`` when the compiled extension is present and ``N`` is
            within its 64-bit multiplicity bound, otherwise ``"python"``.
        ``"cpp"``
            The whole layer loop in C++ (``_core.count_A12_dp``).  About 4x
            faster than ``"python"`` here and roughly 20x faster than a build
            with no extension at all, because no transition crosses the
            Python boundary and no intermediate transition table is built.
            Raises :exc:`ValueError` if the extension is missing.
        ``"python"``
            The driver in this module, which still uses the C++ transition
            kernel for each histogram when it is available.  This is the
            reference implementation: it is what ``"cpp"`` is tested against,
            and it is what runs when ``N`` exceeds the C++ bound.

    Returns
    -------
    int
        The exact count, as an arbitrary-precision Python integer.  Returns 0
        if the constraints are infeasible.

    Raises
    ------
    ValueError
        If the inputs have inconsistent lengths or out-of-range entries.

    Examples
    --------
    The worked example of the manuscript, with ``N = 3`` individuals and
    ``P = 4`` loci:

    >>> count_A12_dp(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1])
    133520
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {_BACKENDS}"
        )
    a_vec = _pad_a_vec(N, a_vec)
    P = _validate(N, a_vec, phi0_vec, phi1_vec)

    # Validation happens above so that both backends report the same errors.
    use_cpp = backend == "cpp" or (
        backend == "auto" and _count_A12_dp_cpp is not None and N <= _CPP_MAX_N
    )
    if use_cpp:
        if _count_A12_dp_cpp is None:
            raise ValueError(
                "backend='cpp' requires the compiled extension, which is not "
                "available in this build"
            )
        return _count_A12_dp_cpp(N, list(a_vec), list(phi0_vec), list(phi1_vec),
                                 reorder_loci)

    if (sum(a_vec) == 0 and all(x == 0 for x in phi0_vec)
            and all(x == 0 for x in phi1_vec)):
        return 1
    if P == 0:
        return 1 if all(a == 0 for a in a_vec) else 0

    loci: List[Tuple[int, int]] = list(zip(phi0_vec, phi1_vec))
    if reorder_loci:
        # A locus admits s in [phi1, 2N - phi0]; narrow windows prune soonest.
        loci.sort(key=lambda t: (2 * N - t[0]) - t[1])

    # Per-locus weight table: colw[p][s] = C(s, phi1_p) * C(2N - s, phi0_p).
    colw: List[List[int]] = [
        [comb(s, ph1) * comb(2 * N - s, ph0) for s in range(2 * N + 1)]
        for ph0, ph1 in loci
    ]
    # Suffix bounds on the total ancestry-1 count still to be placed.
    suf_lo = [0] * (P + 1)
    suf_hi = [0] * (P + 1)
    for p in range(P - 1, -1, -1):
        ph0, ph1 = loci[p]
        suf_lo[p] = suf_lo[p + 1] + ph1
        suf_hi[p] = suf_hi[p + 1] + (2 * N - ph0)

    width = max(a_vec) + 1
    start = [0] * width
    for a in a_vec:
        start[a] += 1

    # The C++ enumeration needs ``bytes`` keys, which ``_trim`` gives whenever
    # N <= 255, and 64-bit weights, which hold whenever N <= _CPP_MAX_N.
    transitions = _locus_transitions
    if _locus_transitions_cpp is not None and N <= _CPP_MAX_N:
        transitions = _locus_transitions_cpp

    # Forward layered dynamic program.  ``layer`` maps a histogram of remaining
    # row sums to the total weight of all ways of reaching it with the loci
    # processed so far.  Only one layer is held at a time, so the whole state
    # space is never materialised.
    layer = {_trim(start): 1}
    trans_cache: Dict[object, List[Tuple[object, int, int]]] = {}

    for p in range(P):
        w_table = colw[p]
        lo_next, hi_next = suf_lo[p + 1], suf_hi[p + 1]
        reach = 2 * (P - p - 1)          # ancestry-1 entries one row can still shed
        nxt: Dict[object, int] = {}
        for hist, weight in layer.items():
            # The ancestry-1 entries still to place are determined by the
            # histogram, so this is computed once per state, not per transition.
            total = sum(r * c for r, c in enumerate(hist))
            if cache_transitions:
                trans = trans_cache.get(hist)
                if trans is None:
                    trans = transitions(hist)
                    trans_cache[hist] = trans
            else:
                trans = transitions(hist)
            for new_hist, s, ways in trans:
                cw = w_table[s]
                if not cw:
                    continue
                # Prune: no row may be left needing more than the loci can give.
                if len(new_hist) - 1 > reach:
                    continue
                # Prune: what remains must fit the windows [phi1_q, 2N - phi0_q]
                # of the loci not yet processed.
                total_left = total - s
                if not (lo_next <= total_left <= hi_next):
                    continue
                add = weight * ways * cw
                prev = nxt.get(new_hist)
                nxt[new_hist] = add if prev is None else prev + add
        layer = nxt
        if not layer:
            return 0
        # Histograms shrink as rows are exhausted, so stale transition entries
        # are never reused; drop them once the cache outgrows a layer.  The
        # bound counts cached tables, each of which holds a few hundred
        # triples, which is why the cache dominates peak memory when enabled.
        if cache_transitions and len(trans_cache) > 4 * len(layer) + 1024:
            trans_cache.clear()

    return layer.get(_trim([N]), 0)
