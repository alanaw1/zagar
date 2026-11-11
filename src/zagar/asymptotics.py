"""Asymptotic enumeration of doubly constrained admixed arrays.

This module implements the four-regime asymptotic expansion for

    |A12| = #{ two-way diploid admixed arrays [A, X] with
               A_{n.}^{(1)} = 2 P abar   for every individual n,
               Phi_{p0} = 2 N f0  and  Phi_{p1} = 2 N f1  for every locus p }

in the *semi-regular* setting, where every individual carries the same global
ancestry proportion ``abar`` and every locus the same ancestry-specific allele
frequencies ``f0`` and ``f1``.  The expansion is valid as ``N, P -> oo`` with
``N / P`` bounded away from 0 and infinity.

Note that ``f0`` and ``f1`` are *joint* proportions, measured against all
``2N`` haplotypes at a locus, and not frequencies conditional on ancestry.
The feasible region is therefore ``f1 <= abar <= 1 - f0``.

Writing ``alpha = abar (1 - abar)``, the result has the form

    |A12| = pi^(D/2) / ( (2 pi)^D sqrt(Delta) ) * Q^(-2 N P) * exp(K) * (1 + o(1))

so that, in bits,

    log2 |A12| = 2 N P * H  -  log2(Delta)/2  -  (D/2) log2(pi)  -  D
                 +  K * log2(e),

where ``H = -log2(Q)`` is the Shannon entropy of the cell types the constraints
leave available, ``Delta`` is the determinant of the saddle-point Hessian, and
``D`` is the dimension of the saddle-point integral.

Which cell types survive depends on where ``(abar, f0, f1)`` sits in the
feasible region.  With all four present, their densities are

======  ================================  =================
Cell    Meaning                           Density
======  ================================  =================
(0, 0)  ancestry 0, reference allele      ``1 - abar - f0``
(1, 0)  ancestry 1, reference allele      ``abar - f1``
(1, 1)  ancestry 1, alternative allele    ``f1``
(0, 1)  ancestry 0, alternative allele    ``f0``
======  ================================  =================

which sum to 1 and recover the margins: ancestry 1 totals
``(abar - f1) + f1 = abar``, and alternative alleles on ancestry-0 haplotypes
total ``f0``.  Driving one density to zero collapses the expansion onto a
boundary, giving four regimes:

======  ==============================  ===========================================
Case    Condition                       Cell types available
======  ==============================  ===========================================
1       ``f1 < abar < 1 - f0``          all four: (0,0), (0,1), (1,0), (1,1)
2       ``f1 = abar < 1 - f0``          three: (1,0) forced empty
3       ``f1 < abar = 1 - f0``          three: (0,0) forced empty
4       ``abar = f1 = 1 - f0``          two: X == 1 everywhere
======  ==============================  ===========================================

Case 3 needs no formula of its own.  The bijection ``A -> 1 - A`` sends
``(abar, f0, f1)`` to ``(1 - abar, f1, f0)`` and carries Case 3 onto Case 2, so
:func:`_case3` delegates to :func:`_case2` at the image point.

The correction terms ``K_r`` are written here in the closed forms derived in
the manuscript rather than as expanded polynomials in ``(N, P, abar, f0, f1)``.
The two agree exactly (see ``tests/test_asymptotics.py``, which checks the
closed forms against the expanded polynomials of the reference R
implementation), but the closed forms are shorter, show the structure of each
term, and are better conditioned: measured against exact rational arithmetic
they carry relative error at the 1e-16 level, where the expanded degree-10
numerator reaches 1e-11 through cancellation between its terms.

Boundary tolerance
------------------
:func:`log2_A12_asymptotic` classifies a parameter triple with
``math.isclose``-style tolerance: a point with ``f1`` within ``tol`` of
``abar`` is treated as Case 2, and similarly for Cases 3 and 4.  This is
deliberate.  The Case 1 correction behaves like ``-1 / (24 N P (abar - f1))``
as ``f1 -> abar`` and like ``-1 / (24 N P (1 - abar - f0))`` as
``abar -> 1 - f0``, so it diverges near the boundary; the boundary formula is
the appropriate one there.
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple, Sequence

__all__ = [
    "entropy_bits",
    "classify_case",
    "log2_A12_asymptotic",
    "densities_from_constraints",
]

_LOG2E = math.log2(math.e)


class _Parts(NamedTuple):
    """Everything one regime computes, so nothing is evaluated twice.

    Each ``_caseN`` returns this rather than a bare float, which keeps
    :func:`log2_A12_asymptotic` free of a second round of ``entropy_bits`` and
    ``_K_caseN`` calls when ``details=True``, and lets ``log2_delta`` be
    reported as itself instead of back-solved from ``log2_A12``.

    The five fields satisfy, exactly,

        log2_A12 == _assemble(N, P, entropy, log2_delta, dimension, K)
    """

    log2_A12: float
    entropy: float
    log2_delta: float
    dimension: int
    K: float


def _near(x: float, y: float, tol: float = 1e-8) -> bool:
    """Combined absolute/relative closeness test.

    For parameters in [0, 1] the default tolerance treats numbers within
    ``1e-8`` as equal.

    Returns
    -------
    bool
        True when ``|x - y| <= tol * max(1, |x|, |y|)``.
    """
    raise NotImplementedError("see spec section 5")


def entropy_bits(*p: float) -> float:
    """Shannon entropy, in bits, of a probability vector.

    Parameters
    ----------
    *p : float
        Components of the distribution.  They must sum to 1 to within ``1e-8``.
        Zero components are allowed and contribute nothing, following the
        convention ``0 log 0 = 0``.

    Returns
    -------
    float
        ``-sum(p_i log2 p_i)``.

    Raises
    ------
    ValueError
        If the components do not sum to 1, or if any is negative.

    Notes
    -----
    Accumulate with :func:`math.fsum` rather than the builtin ``sum``: this is
    called with up to four densities that can differ by orders of magnitude
    near a boundary, and the result is then multiplied by ``2 N P``, so an
    error in the last bits of ``H`` is amplified by the largest factor in the
    whole expansion.

    Examples
    --------
    >>> entropy_bits(0.5, 0.5)
    1.0
    >>> entropy_bits(0.25, 0.25, 0.25, 0.25)
    2.0
    """
    raise NotImplementedError("see spec section 5")


def classify_case(abar: float, f0: float, f1: float, tol: float = 1e-8) -> int:
    """Return the regime (1-4) of Theorem 3.1 for a density triple.

    Parameters
    ----------
    abar, f0, f1 : float
        Density parameters, each strictly in (0, 1), with ``f1 <= abar <= 1-f0``.
    tol : float, optional
        Tolerance used for the boundary equalities (default ``1e-8``).

    Returns
    -------
    int
        1 (interior), 2 (``f1 = abar``), 3 (``abar = 1 - f0``) or
        4 (both, so ``X`` is identically 1).

    Raises
    ------
    ValueError
        If the parameters are outside (0, 1), or if ``f1 > abar`` or
        ``abar > 1 - f0`` so that no admissible array exists.

    Notes
    -----
    The infeasibility checks must themselves be tolerant, or a point intended
    as Case 2 that lands a hair on the wrong side of ``f1 = abar`` through
    floating-point arithmetic will raise instead of classifying.  See the
    module docstring on why the boundary is absorbing rather than exact.

    Examples
    --------
    >>> classify_case(0.5, 0.25, 0.25)
    1
    >>> classify_case(0.3, 0.7, 0.3)
    4
    """
    raise NotImplementedError("see spec section 5")


# ---------------------------------------------------------------------------
# Correction terms K_r (natural log; multiply by log2(e) for bits)
# ---------------------------------------------------------------------------


def _K_case1(N: int, P: int, abar: float, f0: float, f1: float) -> float:
    """Cubic-quartic correction in the interior regime.

    Four terms, of orders ``N/P``, ``P/N``, ``1`` and ``1/(NP)``.  Writing
    ``p0 = 1 - abar - f0`` and ``pt = abar - f1`` for the two densities that
    vanish on the boundaries, and ``s = p0 + pt = 1 - f0 - f1``, every
    denominator is a product of ``alpha``, ``f0``, ``f1``, ``p0``, ``pt`` and
    ``s`` --- which is why this diverges as either boundary is approached.

    See spec section 4, Case 1.
    """
    raise NotImplementedError("see spec section 4, Case 1")


def _K_case2(N: int, P: int, abar: float, f0: float) -> float:
    """Correction on the singly degenerate face ``f1 = abar``.

    ``f1`` is not a parameter here: it equals ``abar`` by hypothesis.  Write
    ``tau = 1 - abar - f0`` for the one density still free to vanish (which is
    what distinguishes this face from Case 4).

    See spec section 4, Case 2.
    """
    raise NotImplementedError("see spec section 4, Case 2")


def _K_case4(N: int, P: int, abar: float) -> float:
    """Correction on the doubly degenerate edge ``abar = f1 = 1 - f0``.

    Only ``abar`` survives as a parameter: ``X == 1`` everywhere, so the array
    is determined by ``A`` alone and the expansion reduces to the one-matrix
    problem.  Both terms carry ``1 / alpha`` and vanish at neither endpoint, so
    unlike Cases 1-3 there is no further boundary to fall off.

    See spec section 4, Case 4.
    """
    raise NotImplementedError("see spec section 4, Case 4")


# ---------------------------------------------------------------------------
# Per-case assembly
# ---------------------------------------------------------------------------


def _assemble(N: int, P: int, H: float, log2_delta: float, D: int, K: float) -> float:
    """Combine entropy, Gaussian determinant and correction into log2 |A12|.

    Implements, in one place for all four regimes,

        2 N P H  -  log2(Delta)/2  -  (D/2) log2(pi)  -  D  +  K log2(e)

    See spec section 2.
    """
    raise NotImplementedError("see spec section 2")


def _case1(N: int, P: int, abar: float, f0: float, f1: float) -> _Parts:
    """The interior regime: all four cell types present, ``D = N + 2P``.

    See spec section 4, Case 1, for ``H`` and ``log2 Delta``.
    """
    raise NotImplementedError("see spec section 4, Case 1")


def _case2(N: int, P: int, abar: float, f0: float) -> _Parts:
    """The face ``f1 = abar``: ``(1,0)`` empty, ``D = N + 2P - 1``.

    See spec section 4, Case 2, for ``H`` and ``log2 Delta``.
    """
    raise NotImplementedError("see spec section 4, Case 2")


def _case3(N: int, P: int, abar: float, f1: float) -> _Parts:
    """The face ``abar = 1 - f0``: ``(0,0)`` empty, ``D = N + 2P - 1``.

    The exact bijection ``A -> 1 - A`` maps ``(abar, f0, f1)`` to
    ``(1 - abar, f1, f0)``, carrying Case 3 onto Case 2.  Since the bijection
    is exact rather than asymptotic, *every* field of the returned
    :class:`_Parts` is already correct at the image point --- the count, the
    entropy (``H(1 - abar, f1, abar - f1)``, which is what Case 2 evaluates
    there), the determinant, the dimension and the correction.  So this needs
    no arithmetic of its own, only the right call.

    Note that ``f0`` is not an argument: on this face ``f0 = 1 - abar``, so
    passing it would be redundant and would invite the two to disagree.
    """
    raise NotImplementedError("one line; see spec section 4, Case 3")


def _case4(N: int, P: int, abar: float) -> _Parts:
    """The edge ``abar = f1 = 1 - f0``: ``X == 1`` throughout, ``D = N + P - 1``.

    See spec section 4, Case 4, for ``H`` and ``log2 Delta``.
    """
    raise NotImplementedError("see spec section 4, Case 4")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def log2_A12_asymptotic(
    N: int,
    P: int,
    abar: float,
    f0: float,
    f1: float,
    tol: float = 1e-8,
    details: bool = False,
):
    """Asymptotic value of ``log2 |A12|`` for semi-regular constraints.

    Parameters
    ----------
    N : int
        Number of individuals (so ``2 N`` haplotypes per locus).
    P : int
        Number of loci (so ``2 P`` haplotype positions per individual).
    abar : float
        Global ancestry proportion, identical across individuals.  Each row of
        ``A`` carries ``2 P abar`` ones.
    f0, f1 : float
        Ancestry-specific allele frequencies at every locus, measured relative
        to all ``2 N`` haplotypes there: ``Phi_{p0} = 2 N f0`` alternative
        alleles on ancestry-0 haplotypes and ``Phi_{p1} = 2 N f1`` on
        ancestry-1 haplotypes.  Note these are *joint* proportions, not
        frequencies conditional on ancestry.
    tol : float, optional
        Tolerance for the boundary equalities (default ``1e-8``); see the module
        docstring.
    details : bool, optional
        If True, return a dict with the value broken into its parts.

    Returns
    -------
    float or dict
        ``log2 |A12|``, or a dict with keys ``log2_A12``, ``case``, ``entropy``,
        ``log2_delta``, ``dimension`` and ``K``.

    Raises
    ------
    ValueError
        If the parameters are infeasible or outside (0, 1), or if ``N`` or
        ``P`` is not positive.

    Notes
    -----
    The expansion carries a ``(1 + o(1))`` relative error as ``N, P -> oo`` with
    ``N / P`` bounded.  At finite sizes it slightly overestimates the count: at
    ``N = P = 4`` the error is under 0.04 bits in every regime.

    Dispatch is a classification followed by one ``_caseN`` call.  Build the
    ``details`` payload from the :class:`_Parts` that call already returned
    rather than recomputing ``entropy`` and ``K`` or back-solving
    ``log2_delta``; the latter is how the reference implementation came to
    report half the true ``log2(Delta)``.

    Examples
    --------
    >>> round(log2_A12_asymptotic(10, 10, 0.5, 0.25, 0.25), 4)
    330.0294
    >>> log2_A12_asymptotic(8, 8, 0.25, 0.75, 0.25, details=True)["case"]
    4
    """
    raise NotImplementedError("see spec section 6")


def densities_from_constraints(
    N: int,
    a_vec: Sequence[int],
    phi0_vec: Sequence[int],
    phi1_vec: Sequence[int],
    tol: float = 1e-8,
) -> Dict[str, float]:
    """Recover ``(abar, f0, f1)`` from integer constraint vectors.

    Parameters
    ----------
    N : int
        Number of individuals.
    a_vec : sequence of int
        Length ``N``; row sums of ``A``.  Must be constant.
    phi0_vec, phi1_vec : sequence of int
        Length ``P``; ancestry-specific sums.  Each must be constant.
    tol : float, optional
        Accepted for signature compatibility with the rest of the module; the
        conversion itself is exact.

    Returns
    -------
    dict
        ``{"N", "P", "abar", "f0", "f1"}``.

    Raises
    ------
    ValueError
        If the lengths are inconsistent, or if the constraints are not
        semi-regular, in which case the asymptotic formulas of this module do
        not apply.  The message says ``semi-regular`` in that second case.

    Notes
    -----
    The denominators differ between the row and column constraints --- ``2 P``
    for ``abar``, ``2 N`` for ``f0`` and ``f1`` --- which follows directly from
    the definitions in the module docstring: a row sum ranges over an
    individual's ``2 P`` haplotype positions, a locus sum over the ``2 N``
    haplotypes at that locus.

    Examples
    --------
    >>> d = densities_from_constraints(4, [4] * 4, [2] * 4, [2] * 4)
    >>> (d["abar"], d["f0"], d["f1"])
    (0.5, 0.25, 0.25)
    """
    raise NotImplementedError("see spec section 5")
