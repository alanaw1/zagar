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

from ast import Return
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
    return abs(x - y) <= tol * max(1.0, abs(x), abs(y))


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
    >>> entropy_bits(1.0, 0.0)          # the 0 log 0 = 0 convention
    0.0
    """
    total = math.fsum(p)
    if not _near(total, 1.0):
        raise ValueError(f"Components must sum to 1, got {total}")
    if any(p_i < -1e-12 for p_i in p):
        raise ValueError("Components must be non-negative")
    # ``+ 0.0`` turns -0.0 into 0.0: a degenerate distribution has every
    # surviving term zero, and negating that sum yields negative zero.
    # Entropy is nonnegative by definition, so reporting -0.0 is a wart.
    return -math.fsum(p_i * math.log2(p_i) for p_i in p if p_i > 0.0) + 0.0


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
    # Check domain
    if min(abar, f0, f1) <= 0.0 or max(abar, f0, f1) >= 1.0:
        raise ValueError("Parameters must each be in (0, 1)")
    # Check that f1 <= abar <= 1 - f0
    if f1 > abar and not _near(f1, abar, tol):
        raise ValueError(f"f1 = {f1} > abar = {abar}; |A12| = 0")
    if abar > 1 - f0 and not _near(abar, 1 - f0, tol):
        raise ValueError(f"abar = {abar} > 1 - f0 = {1 - f0}; |A12| = 0")
    # Identify boundaries
    at_f1 = _near(f1, abar, tol)
    at_f0 = _near(abar, 1 - f0, tol)

    if at_f1 and at_f0:
        return 4
    if at_f1:
        return 2
    if at_f0:
        return 3
    return 1


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
    alpha = abar * (1 - abar)
    p0 = 1 - abar - f0            # density of cell type (0, 0)
    pt = abar - f1                # density of cell type (1, 0)
    s = p0 + pt                   # = 1 - f0 - f1
    t1 = -N/(24*P) * (abar**2 - abar + 1) / alpha
    t2 = -P/(24*N) * (1-f0)*(1-f1)*(f0+f1) / (f0*f1*s)
    t3 = -(1/4) * (1 - 1/N - 1/P) * (1 - p0*pt/(alpha*s))
    t4 = (
        (7*abar**2 - 7*abar + 1)/alpha
        + 6*p0*pt/(alpha*s)
        - (p0**2 + pt**2 + p0*pt)/(p0*pt*s)
    ) / (24*N*P)
    return t1 + t2 + t3 + t4


def _K_case2(N: int, P: int, abar: float, f0: float) -> float:
    """Correction on the singly degenerate face ``f1 = abar``.

    ``f1`` is not a parameter here: it equals ``abar`` by hypothesis.  Write
    ``tau = 1 - abar - f0`` for the one density still free to vanish (which is
    what distinguishes this face from Case 4).

    See spec section 4, Case 2.
    """
    alpha = abar * (1 - abar)
    tau = 1 - abar - f0
    return (
        -N/(24*P) * (abar**2 - abar + 1) / alpha
        - 1/4
        + 1/(4*N)
        + 1/(4*P)
        - P/(24*N) * (1-f0)*(f0+abar)*(1-abar) / (f0*abar*tau)
        + (7*abar**2 - 7*abar + 1) / (24*N*P*alpha)
    )


def _K_case4(N: int, P: int, abar: float) -> float:
    """Correction on the doubly degenerate edge ``abar = f1 = 1 - f0``.

    Only ``abar`` survives as a parameter: ``X == 1`` everywhere, so the array
    is determined by ``A`` alone and the expansion reduces to the one-matrix
    problem.  Both terms carry ``1 / alpha`` and vanish at neither endpoint, so
    unlike Cases 1-3 there is no further boundary to fall off.

    See spec section 4, Case 4.
    """
    alpha = abar * (1 - abar)
    term1 = (6*abar**2 - 6*abar + 1) * (N + P - 1)**2 / (16*N*P*alpha)
    term2 = (
        (2*abar - 1)**2
        * (5*N**2 + 6*N*P - 6*N + 5*P**2 - 6*P + 1)
        / (48*N*P*alpha)
    )
    return term1 - term2


# ---------------------------------------------------------------------------
# Per-case assembly
# ---------------------------------------------------------------------------


def _assemble(N: int, P: int, H: float, log2_delta: float, D: int, K: float) -> float:
    """Combine entropy, Gaussian determinant and correction into log2 |A12|.

    Implements, in one place for all four regimes,

        2 N P H  -  log2(Delta)/2  -  (D/2) log2(pi)  -  D  +  K log2(e)

    See spec section 2.
    """
    return 2*N*P*H - log2_delta/2 - (D/2)*math.log2(math.pi) - D + K*_LOG2E


def _case1(N: int, P: int, abar: float, f0: float, f1: float) -> _Parts:
    """The interior regime: all four cell types present, ``D = N + 2P``.

    See spec section 4, Case 1, for ``H`` and ``log2 Delta``.
    """
    alpha = abar * (1 - abar)
    H = entropy_bits(1 - abar - f0, abar - f1, f1, f0)
    log2_delta = (
        N*math.log2(P)
        + 2*P*math.log2(N)
        + (N - 1)*math.log2(alpha)
        + P*math.log2(f0*f1)
        + (P - 1)*math.log2(1 - f0 - f1)
        + math.log2((abar - f1)*(1 - abar - f0))
    )
    D = N + 2*P
    K = _K_case1(N, P, abar, f0, f1)
    value = _assemble(N, P, H, log2_delta, D, K)
    return _Parts(value, H, log2_delta, D, K)


def _case2(N: int, P: int, abar: float, f0: float) -> _Parts:
    """The face ``f1 = abar``: ``(1,0)`` empty, ``D = N + 2P - 1``.

    See spec section 4, Case 2, for ``H`` and ``log2 Delta``.
    """
    alpha = abar * (1 - abar)
    H = entropy_bits(abar, f0, 1 - abar - f0)
    log2_delta = (
        (N - 1)*math.log2(P)
        + (2*P - 1)*math.log2(N)
        + (N - 1)*math.log2(alpha)
        + P*math.log2(abar*f0*(1 - abar - f0))
    )
    D = N + 2*P - 1
    K = _K_case2(N, P, abar, f0)
    value = _assemble(N, P, H, log2_delta, D, K)
    return _Parts(value, H, log2_delta, D, K)


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
    return _case2(N, P, 1 - abar, f1)


def _case4(N: int, P: int, abar: float) -> _Parts:
    """The edge ``abar = f1 = 1 - f0``: ``X == 1`` throughout, ``D = N + P - 1``.

    See spec section 4, Case 4, for ``H`` and ``log2 Delta``.
    """
    alpha = abar * (1 - abar)
    H = entropy_bits(abar, 1 - abar)
    log2_delta = (
        (N - 1)*math.log2(P)
        + (P - 1)*math.log2(N)
        + (N + P - 1)*math.log2(alpha)
    )
    D = N + P - 1
    K = _K_case4(N, P, abar)
    value = _assemble(N, P, H, log2_delta, D, K)
    return _Parts(value, H, log2_delta, D, K)


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
    the three densities held fixed and ``N / P`` bounded.  At finite sizes the
    error is a few hundredths of a bit, and its *direction* depends on where in
    the feasible region the densities sit: holding ``abar = 1/2``, it runs
    about ``+0.010`` bits along ``f0 = f1 = 1/4`` but ``-0.016`` bits at
    ``N = P = 4`` along ``f0 = f1 = 1/8``.  Nothing here establishes a sign;
    over every margin small enough to check exactly, 252 of the 680 at
    ``N = P = 8`` are underestimates.

    Accuracy degrades if the densities are not held fixed, which the signature
    cannot prevent a caller from doing.  At one alternative allele per locus
    per ancestry -- ``phi0 = phi1 = 1``, so ``f0 = f1 = 1/(2N)`` -- two of the
    four cell types have density ``O(1/N)``, the dropped terms stop being
    small, and the error grows with ``N`` instead of shrinking: ``-0.016``
    bits at ``N = P = 4`` to ``-0.049`` at ``N = P = 8``.

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
    if N <= 0 or P <= 0:
        raise ValueError(f"N = {N} and P = {P} must be positive integers")
    case = classify_case(abar, f0, f1, tol)
    if case == 1:
        parts = _case1(N, P, abar, f0, f1)
    elif case == 2:
        parts = _case2(N, P, abar, f0)
    elif case == 3:
        parts = _case3(N, P, abar, f1)
    else:  # case == 4
        parts = _case4(N, P, abar)

    if not details:
        return parts.log2_A12

    return dict(log2_A12=parts.log2_A12,
                case=case,
                entropy=parts.entropy,
                log2_delta=parts.log2_delta,
                dimension=parts.dimension,
                K=parts.K)


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
    P = len(phi0_vec)

    if len(a_vec) != N:
        raise ValueError(f"Length of a_vec = {len(a_vec)} != N = {N}")
    if len(phi1_vec) != P:
        raise ValueError(
            f"phi1_vec has length {len(phi1_vec)}, but phi0_vec has length {P}; "
            "they must agree"
        )
    for name, vec in (("a_vec", a_vec), ("phi0_vec", phi0_vec),
                      ("phi1_vec", phi1_vec)):
        if any(x != vec[0] for x in vec):
            raise ValueError(
                f"{name} is not constant; the asymptotic formulas require "
                "semi-regular constraints"
            )

    return dict(N=N,
                P=P,
                abar=a_vec[0] / (2 * P),
                f0=phi0_vec[0] / (2 * N),
                f1=phi1_vec[0] / (2 * N))
