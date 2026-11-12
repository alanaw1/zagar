"""
Exact and Approximate Enumeration of Admixed Arrays

This Python package provides efficient C++ implementations for counting
admixed arrays and binary matrices with specified row and column sums.

Functions
---------
count_binary_matrices(p, q, sort_p_desc=False)
    Count Binary Matrices with Row Sums :math:`\\boldsymbol{p}` and Column Sums :math:`\\boldsymbol{q}`

getA12(N, a_vec, phi0_vec, phi1_vec, sort_p_desc=True)
    Cardinality of Doubly Constrained Admixed Arrays

getA12_parallel(N, a_vec, phi0_vec, phi1_vec, n_threads=-1, sort_p_desc=True)
    Cardinality of Doubly Constrained Admixed Arrays (Parallelized)

getA12_pruning(N, a_vec, phi0_vec, phi1_vec, n_threads=-1, sort_p_desc=True)
    Cardinality of Doubly Constrained Admixed Arrays (with Pruning and Symmetry)

count_A12(N, a_vec, phi0_vec, phi1_vec, method="auto")
    Exact :math:`|\\mathscr{A}_{12}|` with a Selectable Backend

count_A12_dp(N, a_vec, phi0_vec, phi1_vec, reorder_loci=True)
    Exact :math:`|\\mathscr{A}_{12}|` by Dynamic Programming over Loci

log2_A12_asymptotic(N, P, abar, f0, f1, tol=1e-8, details=False)
    Asymptotic :math:`\\log_2 |\\mathscr{A}_{12}|` for Semi-Regular Constraints

classify_case(abar, f0, f1, tol=1e-8)
    Regime (1-4) of Theorem 3.1 for a Density Triple

entropy_bits(*p)
    Shannon Entropy, in Bits, of a Probability Vector

densities_from_constraints(N, a_vec, phi0_vec, phi1_vec, tol=1e-8)
    Recover :math:`(\\bar a, f_0, f_1)` from Integer Constraint Vectors

getA1(N, P, a_vec, return_log=False)
    Cardinality of :math:`\\mathscr{A}_1`: Number of Arrays Subject to Row Margin Constraints

getA2(N, P, phi0_vec, phi1_vec, return_log=False)
    Cardinality of :math:`\\mathscr{A}_2`: Number of Arrays Subject to Column Margin Constraints

compareA1A2(a_vec, phi0_vec, phi1_vec, use_large_NP=False)
    Compare Cardinalities of :math:`\\mathscr{A}_1` and :math:`\\mathscr{A}_2`

get_constraints(A, X)
    Compute Marginal Constraint Vectors from Admixed Array

count_matrices(p, q, sort_p_desc=False)
    Wrapper for ``count_binary_matrices``


Documentation
-------------
"""

import math
import warnings
from typing import List, Tuple

import numpy as np

from ._core import (
    count_binary_matrices,
    getA12 as _getA12_core,
    getA12_parallel as _getA12_parallel_core,
    getA12_pruning as _getA12_pruning_core,
    getA1,
    getA2,
    has_openmp,
)
from .asymptotics import (
    classify_case,
    densities_from_constraints,
    entropy_bits,
    log2_A12_asymptotic,
)
from .exact_dp import count_A12_dp

__version__ = "0.1.0"
__all__ = [
    "count_binary_matrices",
    "getA12",
    "getA12_parallel",
    "getA12_pruning",
    "getA1",
    "getA2",
    "compareA1A2",
    "get_constraints",
    "count_matrices",
    "has_openmp",
    # exact enumeration
    "count_A12",
    "count_A12_dp",
    # asymptotic enumeration (Theorem 3.1, all four regimes)
    "log2_A12_asymptotic",
    "classify_case",
    "entropy_bits",
    "densities_from_constraints",
]

#: Algorithms accepted by :func:`count_A12`.
_EXACT_METHODS = ("dp", "column-sums", "column-sums-parallel", "column-sums-pruning")

#: Accepted spellings that map onto :data:`_EXACT_METHODS`.  The
#: ``miller-harrison`` names are there because that is what the column-sum
#: family is called in the literature and in this package's own docs.
_METHOD_ALIASES = {
    "auto": "dp",
    "histogram-dp": "dp",
    "miller-harrison": "column-sums",
    "miller-harrison-parallel": "column-sums-parallel",
    "miller-harrison-pruning": "column-sums-pruning",
}


def _is_regular_case(N, a_vec, phi0_vec, phi1_vec):
    """Check if inputs form the regular case (all elements equal to expected values)."""
    P = len(phi0_vec)
    # Check: all a_vec equal to P, all phi0_vec equal to N, all phi1_vec equal to N
    if len(set(a_vec)) != 1 or a_vec[0] != P:
        return False
    if len(set(phi0_vec)) != 1 or phi0_vec[0] != N:
        return False
    if len(set(phi1_vec)) != 1 or phi1_vec[0] != N:
        return False
    return True


def _is_semiregular(a_vec, phi0_vec, phi1_vec):
    """True when every constraint vector is constant, as Theorem 3.1 requires."""
    return (
        len(set(a_vec)) == 1
        and len(set(phi0_vec)) == 1
        and len(set(phi1_vec)) == 1
    )


def _compute_approximations(N, P):
    """
    Compute Mathematical Approximations for Regular Case.

    This is the symmetric point :math:`\\bar a = f_0 = f_1 = 1/2`, which is
    Case 4 of :func:`log2_A12_asymptotic` at :math:`\\bar a = 1/2`; the two
    agree to floating-point precision.  It is retained so that the output of
    :func:`getA12` is unchanged for callers who relied on it.

    Returns
    -------
    dict with:
    - ``log2_upper_bound``: Upper bound on :math:`\\log_2(count)`
    - ``log2_approx``: Approximation valid when :math:`N = \\Theta(P)`
    """
    # Upper bound: 2*N*P - 0.5*(P*log2(pi*N) + N*log2(pi*P) - log2(pi*N*P))
    pi = math.pi
    log2_upper_bound = (
        2 * N * P
        - 0.5 * (P * math.log2(pi * N) + N * math.log2(pi * P) - math.log2(pi * N * P))
    )

    # Approximation (valid for N = Theta(P)):
    # Upper bound - (P + N - 1)^2 / (8*N*P) * log2(e)
    log2_e = math.log2(math.e)
    log2_approx = log2_upper_bound - ((P + N - 1) ** 2) / (8 * N * P) * log2_e

    return {
        "log2_upper_bound": log2_upper_bound,
        "log2_approx": log2_approx,
    }


def _approx_result(N, a_vec, phi0_vec, phi1_vec, count):
    """Assemble the ``return_approx=True`` payload shared by the getA12 family.

    The symmetric point keeps its historical keys; other semi-regular margins
    now get the four-regime value of :func:`log2_A12_asymptotic`, together with
    the regime that applies and the densities it was evaluated at.  Non-uniform
    margins get no asymptotic value, because Theorem 3.1 does not cover them.
    """
    P = len(phi0_vec)
    is_regular = _is_regular_case(N, a_vec, phi0_vec, phi1_vec)
    result = {"count": count, "is_regular": is_regular}
    # The core backend zero-pads a short a_vec; do the same before asking
    # whether the constraints are semi-regular.
    if len(a_vec) < N:
        a_vec = list(a_vec) + [0] * (N - len(a_vec))

    if is_regular:
        approx = _compute_approximations(N, P)
        result["log2_upper_bound"] = approx["log2_upper_bound"]
        result["log2_approx"] = approx["log2_approx"]
        result["case"] = 4
    elif _is_semiregular(a_vec, phi0_vec, phi1_vec):
        d = densities_from_constraints(N, a_vec, phi0_vec, phi1_vec)
        try:
            detail = log2_A12_asymptotic(
                N, P, d["abar"], d["f0"], d["f1"], details=True
            )
        except ValueError as exc:
            warnings.warn(
                f"no asymptotic value for these densities: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            result["log2_approx"] = detail["log2_A12"]
            result["case"] = detail["case"]
            result["densities"] = (d["abar"], d["f0"], d["f1"])

    if "log2_approx" in result:
        warnings.warn(
            "log2_approx is asymptotic: accurate when N and P are large with "
            "bounded ratio, and a slight overestimate at small sizes.",
            RuntimeWarning,
            stacklevel=3,
        )
    return result


def count_A12(N, a_vec, phi0_vec, phi1_vec, method="auto", **kwargs):
    """Exact :math:`|\\mathscr{A}_{12}|` with a selectable backend.

    Parameters
    ----------
    N : int
        Number of individuals.
    a_vec : sequence of int
        Length ``N``; row sums of ``A``.
    phi0_vec, phi1_vec : sequence of int
        Length ``P``; ancestry-specific sums at each locus.
    method : str, optional
        Which algorithm to use.  Two families are available, and they share no
        code: see the Notes.

        ``"auto"`` (default)
            Currently ``"dp"``, which dominated the alternatives on every case
            tested.
        ``"dp"`` (alias ``"histogram-dp"``)
            Dynamic program over loci, keyed on the histogram of remaining row
            sums (:mod:`zagar.exact_dp`).  Runs in C++ when the compiled
            extension is present; pass ``backend="python"`` to force the
            reference implementation.
        ``"column-sums"`` (alias ``"miller-harrison"``)
            Enumeration over feasible column-sum vectors, counting binary
            matrices by the Miller--Harrison recursion (:func:`getA12`).
        ``"column-sums-parallel"``, ``"column-sums-pruning"``
            Threaded and pruned variants of the same (:func:`getA12_parallel`,
            :func:`getA12_pruning`), aliased ``"miller-harrison-parallel"``
            and ``"miller-harrison-pruning"``.
    **kwargs
        Passed through to the chosen algorithm, e.g. ``n_threads`` for the
        parallel variant, or ``reorder_loci`` / ``backend`` for ``"dp"``.

    Returns
    -------
    int
        The exact count.

    Raises
    ------
    ValueError
        If ``method`` is not recognised.

    Notes
    -----
    All methods return the same integer; they differ only in time and memory.
    The two families are independent algorithms -- one folds the loci into a
    histogram of remaining row sums and never enumerates a column-sum vector,
    the other enumerates those vectors and counts matrices for each -- so
    agreement between them is a genuine cross-check, and the test suite pins
    it over hundreds of random instances.

    The dynamic program is the faster of the two by orders of magnitude:
    ``"column-sums"`` exhausts 16 GB at ``N = P = 6``, where ``"dp"`` needs
    0.01 s and 24 MB.  The column-sum family is kept because
    :func:`count_binary_matrices` is useful in its own right, because it is
    that independent cross-check, and because :func:`getA1`, :func:`getA2`
    and :func:`compareA1A2` build on the same machinery.

    Examples
    --------
    >>> import zagar
    >>> zagar.count_A12(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1])
    133520
    >>> zagar.count_A12(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1],
    ...                 method="column-sums")
    133520
    >>> zagar.count_A12(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1],
    ...                 method="miller-harrison")
    133520
    >>> zagar.count_A12(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1],
    ...                 method="dp", backend="python")
    133520
    """
    method = _METHOD_ALIASES.get(method, method)
    if method == "dp":
        return count_A12_dp(N, a_vec, phi0_vec, phi1_vec, **kwargs)
    if method == "column-sums":
        return getA12(N, a_vec, phi0_vec, phi1_vec, **kwargs)
    if method == "column-sums-parallel":
        return getA12_parallel(N, a_vec, phi0_vec, phi1_vec, **kwargs)
    if method == "column-sums-pruning":
        return getA12_pruning(N, a_vec, phi0_vec, phi1_vec, **kwargs)
    raise ValueError(
        f"unknown method {method!r}; expected one of {_EXACT_METHODS} "
        f"or an alias from {tuple(_METHOD_ALIASES)}"
    )


def get_constraints(A: np.ndarray, X: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
    """
    Compute Marginal Constraint Vectors from Admixed Array

    Given binary matrices :math:`\\mathbf{A}` (ancestry) and :math:`\\mathbf{X}` (allele dosage) 
    that make up the admixed array :math:`[\\mathbf{A},\\mathbf{X}]`, computes three marginal
    constraint vectors used in constrained admixed array enumeration.

    Parameters
    ----------
    A : np.ndarray
        Binary matrix of shape (N, 2P) representing ancestry assignments.
        A[n, p] = 1 means individual n has ancestry 1 at position p.
        May contain np.nan values for missing data.
    X : np.ndarray
        Binary matrix of shape (N, 2P) representing allele values.
        X[n, p] = 1 means individual n has allele 1 at position p.
        May contain np.nan values for missing data.

    Returns
    -------
    a_vec : list of int
        Length N. Entry n is the row sum of A (total ancestry-1 count for
        individual n). This is the row margin constraint for :math:`\\mathscr{A}_1`.
        NA values in A are ignored.
    phi0_vec : list of int
        Length P. Entry p is the sum of X values at positions where A=0,
        across columns p and P+p for all individuals.
        This is the first column margin constraint for :math:`\\mathscr{A}_2`.
    phi1_vec : list of int
        Length P. Entry p is the sum of X values at positions where A=1,
        across columns p and P+p for all individuals.
        This is the second column margin constraint for :math:`\\mathscr{A}_2`.

    Notes
    -----
    If either A or X contains NA/NaN values, a warning is issued but computation
    continues. NA values are ignored in all sums (they contribute 0 to the totals).

    Examples
    --------
    >>> import numpy as np
    >>> import zagar
    >>> np.random.seed(42)
    >>> N, P = 6, 5
    >>> A = np.random.randint(0, 2, size=(N, 2*P)).astype(float)
    >>> X = np.random.randint(0, 2, size=(N, 2*P)).astype(float)
    >>> a_vec, phi0_vec, phi1_vec = zagar.get_constraints(A, X)
    >>> len(a_vec) == N
    True
    >>> len(phi0_vec) == P and len(phi1_vec) == P
    True

    >>> # With missing data
    >>> A[0, 0] = np.nan
    >>> X[1, 2] = np.nan
    >>> a_vec, phi0_vec, phi1_vec = zagar.get_constraints(A, X)  # warns but continues
    """
    A = np.asarray(A)
    X = np.asarray(X)

    N, cols = A.shape
    P = cols // 2

    if X.shape != A.shape:
        raise ValueError(f"A and X must have the same shape. Got A: {A.shape}, X: {X.shape}")
    if cols != 2 * P:
        raise ValueError(f"Number of columns must be even (2*P). Got {cols}")

    # Check for NA values and warn
    na_in_A = np.isnan(A).any() if np.issubdtype(A.dtype, np.floating) else False
    na_in_X = np.isnan(X).any() if np.issubdtype(X.dtype, np.floating) else False

    if na_in_A or na_in_X:
        na_count_A = np.isnan(A).sum() if np.issubdtype(A.dtype, np.floating) else 0
        na_count_X = np.isnan(X).sum() if np.issubdtype(X.dtype, np.floating) else 0
        warnings.warn(
            f"Missing values detected: {na_count_A} NA(s) in A, {na_count_X} NA(s) in X. "
            "These will be ignored in calculations."
        )

    # a_vec: row sums of A (length N), ignoring NAs
    a_vec = np.nansum(A, axis=1).astype(int).tolist()

    # Split into first P and last P columns
    A1, A2 = A[:, :P], A[:, P:]
    X1, X2 = X[:, :P], X[:, P:]

    # phi0_vec[p] = sum over n of [(1-A[n,p])*X[n,p] + (1-A[n,P+p])*X[n,P+p]]
    # Using nansum to ignore NA values
    phi0_vec = np.nansum((1 - A1) * X1, axis=0) + np.nansum((1 - A2) * X2, axis=0)
    phi0_vec = phi0_vec.astype(int).tolist()

    # phi1_vec[p] = sum over n of [A[n,p]*X[n,p] + A[n,P+p]*X[n,P+p]]
    phi1_vec = np.nansum(A1 * X1, axis=0) + np.nansum(A2 * X2, axis=0)
    phi1_vec = phi1_vec.astype(int).tolist()

    return a_vec, phi0_vec, phi1_vec


def getA12(N, a_vec, phi0_vec, phi1_vec, sort_p_desc=True, return_approx=False):
    """
    Cardinality of Doubly Constrained Admixed Arrays

    Uses a modified version of Miller and Harrison's EXACT algorithm
    to compute the size of a set of admixed arrays with specified
    global ancestry vector :math:`\\boldsymbol{a}` (``a_vec``, a row
    sum constraint) and ancestry-specific allele frequencies
    :math:`(\\boldsymbol{\\phi}_0,\\boldsymbol{\\phi}_1)` (``phi0_vec`` and
    ``phi1_vec``, column sum constraint).

    Optionally, it can return an approximate count and upper bound
    (in base two logarithm scale), based on a saddle-point approximation.
    Currently only works for the following special regular case:

    - ``phi0_vec`` and ``phi1_vec`` entries all equal to ``N``
    - ``a_vec`` values all equal to the size of the list ``phi0_vec``

    Parameters
    ----------
    N : int
        Positive integer parameter (number of individuals)
    a_vec : list of int
        Row sums (length N, all equal to P for regular case)
    phi0_vec : list of int
        First constraint vector (length P)
    phi1_vec : list of int
        Second constraint vector (must have same length as phi0_vec)
    sort_p_desc : bool, optional
        If True, sort p in decreasing order for speed (default: True)
    return_approx : bool, optional
        - If True and inputs form a "regular case" (all a_vec equal P, all phi0_vec equal N, all phi1_vec equal N), 
            return a dict with: 'count' (exact count), 'is_regular' (True), 
            'log2_upper_bound' (upper bound on log2(count)), and 'log2_approx' 
            (approximation valid when N and P are large with bounded ratio).
        - If False or not regular case, 
            return just the count (default: False)

    Returns
    -------
    int or dict
        If return_approx=False: The computed :math:`|\\mathscr{A}_{12}|` value (arbitrary precision int)

        If return_approx=True: dict with count and approximation info

    Examples
    --------
    >>> import zagar
    >>> # Basic usage
    >>> zagar.getA12(N=5, a_vec=[2, 2], phi0_vec=[1, 1], phi1_vec=[1, 1])
    7632
    >>> # With approximation for regular case
    >>> result = zagar.getA12(N=4, a_vec=[4]*4, phi0_vec=[4]*4, phi1_vec=[4]*4, return_approx=True)
    >>> result['count']
    824410
    >>> round(result['log2_approx'], 4)  # Approximation of log2(count)
    19.6675
    """
    a_vec = list(a_vec)
    phi0_vec = list(phi0_vec)
    phi1_vec = list(phi1_vec)

    count = _getA12_core(N, a_vec, phi0_vec, phi1_vec, sort_p_desc)

    if not return_approx:
        return count

    return _approx_result(N, a_vec, phi0_vec, phi1_vec, count)


def getA12_parallel(N, a_vec, phi0_vec, phi1_vec, n_threads=-1, sort_p_desc=True, return_approx=False):
    """
    Cardinality of Doubly Constrained Admixed Arrays (Parallelized)

    This is a parallelized version of ``getA12``, which relies on
    OpenMP. If the user already has OpenMP, this function should be used
    in place of ``getA12`` to speed up calculations.

    Parallelization is achieved by assigning workers to each value of the
    first coordinate of the constrained set :math:`\\mathscr{S}`.

    Parameters
    ----------
    N : int
        Positive integer parameter (number of individuals)
    a_vec : list of int
        Row sums (length N, typically all equal to P for regular case)
    phi0_vec : list of int
        First constraint vector (length P)
    phi1_vec : list of int
        Second constraint vector (must have same length as phi0_vec)
    n_threads : int, optional
        Number of threads (-1 = use all available, default: -1)
    sort_p_desc : bool, optional
        If True, sort p in decreasing order for speed (default: True)
    return_approx : bool, optional
        If True and inputs form a "regular case" (all a_vec equal P,
        all phi0_vec equal N, all phi1_vec equal N), return a dict with:
            - 'count': exact count
            - 'is_regular': True
            - 'log2_upper_bound': upper bound on log2(count)
            - 'log2_approx': approximation valid when N and P are large with bounded ratio
        If False or not regular case, returns just the count (default: False)

    Returns
    -------
    int or dict
        If return_approx=False: The computed :math:`|\\mathscr{A}_{12}|` value (arbitrary precision int)

        If return_approx=True: dict with count and approximation info

    Notes
    -----
    This function uses OpenMP for parallelization when available.
    If OpenMP is not available at compile time, it falls back to serial execution.

    Examples
    --------
    >>> import zagar
    >>> # Basic usage
    >>> zagar.getA12_parallel(N=5, a_vec=[2, 2], phi0_vec=[1, 1], phi1_vec=[1, 1])
    7632
    >>> # With approximation for regular case
    >>> result = zagar.getA12_parallel(N=7, a_vec=[7]*7, phi0_vec=[7]*7, phi1_vec=[7]*7, return_approx=True)
    >>> result['count']
    1014691016274501451776
    >>> result['is_regular']
    True
    """
    a_vec = list(a_vec)
    phi0_vec = list(phi0_vec)
    phi1_vec = list(phi1_vec)

    count = _getA12_parallel_core(N, a_vec, phi0_vec, phi1_vec, n_threads, sort_p_desc)

    if not return_approx:
        return count

    return _approx_result(N, a_vec, phi0_vec, phi1_vec, count)


def getA12_pruning(N, a_vec, phi0_vec, phi1_vec, n_threads=-1, sort_p_desc=True, return_approx=False):
    """
    Cardinality of Doubly Constrained Admixed Arrays (with Pruning and Symmetry)

    This is an optimized version of ``getA12_parallel`` that combines two optimizations:

    1. **Sum-based pruning**: Tracks the running sum of column values during
       enumeration and prunes branches when it becomes impossible to reach
       the target sum.

    2. **Symmetry exploitation**: Configurations (v1, v2) and (v2, v1) have
       identical counts and weights. Only enumerates configurations where
       v1 <= v2 lexicographically and multiplies by 2 when v1 < v2, reducing
       work by up to 50%.

    Together, these optimizations provide significant speedup (up to 50x or more)
    compared to ``getA12_parallel``, especially for non-uniform constraint vectors.

    Parameters
    ----------
    N : int
        Positive integer parameter (number of individuals)
    a_vec : list of int
        Row sums (length N, typically all equal to P for regular case)
    phi0_vec : list of int
        First constraint vector (length P)
    phi1_vec : list of int
        Second constraint vector (must have same length as phi0_vec)
    n_threads : int, optional
        Number of threads (-1 = use all available, default: -1)
    sort_p_desc : bool, optional
        If True, sort p in decreasing order for speed (default: True)
    return_approx : bool, optional
        If True and inputs form a "regular case" (all a_vec equal P,
        all phi0_vec equal N, all phi1_vec equal N), return a dict with:
            - 'count': exact count
            - 'is_regular': True
            - 'log2_upper_bound': upper bound on log2(count)
            - 'log2_approx': approximation valid when N and P are large with bounded ratio
        If False or not regular case, returns just the count (default: False)

    Returns
    -------
    int or dict
        If return_approx=False: The computed :math:`|\\mathscr{A}_{12}|` value (arbitrary precision int)

        If return_approx=True: dict with count and approximation info

    Notes
    -----
    This function uses OpenMP for parallelization when available.

    Examples
    --------
    >>> import zagar
    >>> # Basic usage (same API as getA12_parallel)
    >>> zagar.getA12_pruning(N=5, a_vec=[2, 2], phi0_vec=[1, 1], phi1_vec=[1, 1])
    7632
    >>> # The pruning and symmetry optimisations change the cost, not the count
    >>> zagar.getA12_pruning(N=4, a_vec=[4]*4, phi0_vec=[4]*4, phi1_vec=[4]*4)
    824410
    >>> zagar.getA12_pruning(N=4, a_vec=[4]*4, phi0_vec=[4]*4,
    ...                      phi1_vec=[4]*4) == zagar.getA12(
    ...     N=4, a_vec=[4]*4, phi0_vec=[4]*4, phi1_vec=[4]*4)
    True
    """
    a_vec = list(a_vec)
    phi0_vec = list(phi0_vec)
    phi1_vec = list(phi1_vec)

    count = _getA12_pruning_core(N, a_vec, phi0_vec, phi1_vec, n_threads, sort_p_desc)

    if not return_approx:
        return count

    return _approx_result(N, a_vec, phi0_vec, phi1_vec, count)


def count_matrices(p, q, sort_p_desc=False):
    """
    Wrapper for ``count_binary_matrices``

    This is a wrapper for ``count_binary_matrices`` that returns just the count.

    Parameters
    ----------
    p : list or array-like
        Row sums (nonnegative integers)
    q : list or array-like
        Column sums (nonnegative integers)
    sort_p_desc : bool, optional
        If True, sort p in decreasing order for speed optimization

    Returns
    -------
    int
        Number of binary matrices with the given marginals

    Examples
    --------
    >>> import zagar
    >>> zagar.count_matrices([2, 1], [1, 1, 1])
    3
    """
    result = count_binary_matrices(list(p), list(q), sort_p_desc)
    return result["number"]


def compareA1A2(a_vec, phi0_vec, phi1_vec, use_large_NP=False):
    """
    Compare Cardinalities of :math:`\\mathscr{A}_1` and :math:`\\mathscr{A}_2`

    This function computes :math:`|\\mathscr{A}_1|` and :math:`|\\mathscr{A}_2|` and compares their sizes.
    Optionally, it can also compute an asymptotic approximation for large :math:`N`, :math:`P`
    based on entropy calculations.

    Parameters
    ----------
    a_vec : list of int
        Row margin constraints (length N)
    phi0_vec : list of int
        First constraint vector (length P)
    phi1_vec : list of int
        Second constraint vector (length P, same length as phi0_vec)
    use_large_NP : bool, optional
        If True, compute asymptotic comparison using entropy criterion (default: False)

    Returns
    -------
    dict
        Dictionary with keys:
            - 'log2_A1': float - log2 of :math:`|\\mathscr{A}_1|`
            - 'log2_A2': float - log2 of :math:`|\\mathscr{A}_2|`
            - 'outcome': str - "A1 is larger" or "A2 is at least as large"
        If use_large_NP=True, also includes:
            - 'large_NP_stat': float - the asymptotic criterion value (LHS - RHS)
            - 'large_NP_outcome': str - "A1 is roughly larger" or "A2 is roughly at least as large"

    Examples
    --------
    >>> import zagar
    >>> result = zagar.compareA1A2([25]*50, [20]*50, [22]*50)
    >>> print(result['outcome'])
    A2 is at least as large
    >>> result = zagar.compareA1A2([25]*50, [20]*50, [22]*50, use_large_NP=True)
    >>> round(result['large_NP_stat'], 4)
    -0.1695
    >>> print(result['large_NP_outcome'])
    A2 is roughly at least as large

    Notes
    -----
    The asymptotic condition for A1 > A2 (when use_large_NP=True) is:
        H(``a_bar``) > H(``f0``, ``f1``, 1-``f0``-``f1``) - (``f0`` + ``f1``)
    where:
        - ``a_bar`` = ``a_vec`` / (2*P), the normalized row margins
        - ``f0`` = ``phi0_vec`` / (2*N), ``f1`` = ``phi1_vec`` / (2*N), the normalized column margins
        - H() denotes entropy
    """
    # Get dimensions from input vectors
    N = len(a_vec)
    P = len(phi0_vec)

    if len(phi1_vec) != P:
        raise ValueError("phi0_vec and phi1_vec must have the same length.")

    # Compute log2(A1) and log2(A2) using C++ functions
    log_A1 = getA1(N, P, list(a_vec), return_log=True)
    log_A2 = getA2(N, P, list(phi0_vec), list(phi1_vec), return_log=True)

    # Determine exact comparison outcome
    if log_A1 > log_A2:
        outcome = "A1 is larger"
    else:
        outcome = "A2 is at least as large"

    result = {
        'log2_A1': log_A1,
        'log2_A2': log_A2,
        'outcome': outcome,
    }

    # Compute asymptotic approximation if requested
    if use_large_NP:
        # Compute normalized vectors
        a_bar_vec = [a / (2 * P) for a in a_vec]
        f0_vec = [phi0 / (2 * N) for phi0 in phi0_vec]
        f1_vec = [phi1 / (2 * N) for phi1 in phi1_vec]

        # Helper function for binary entropy term: x * log2(1/x) if x > 0, else 0
        def entropy_term(x):
            if x <= 0 or x >= 1:
                return 0.0
            return x * math.log2(1.0 / x)

        # Compute RHS: mean of [H(f0, f1, 1-f0-f1) - (f0 + f1)] over all j
        rhs_values = []
        for f0, f1 in zip(f0_vec, f1_vec):
            # H(f0, f1, 1-f0-f1) = entropy of three-way distribution
            f_rest = 1.0 - f0 - f1
            f_entropy = entropy_term(f0) + entropy_term(f1) + entropy_term(f_rest)
            rhs_values.append(f_entropy - f0 - f1)
        ineq_rhs = sum(rhs_values) / len(rhs_values)

        # Compute LHS: mean of H(a_bar, 1-a_bar) over all i
        lhs_values = []
        for a_bar in a_bar_vec:
            # H(a_bar, 1-a_bar) = binary entropy
            a_bar_entropy = entropy_term(a_bar) + entropy_term(1.0 - a_bar)
            lhs_values.append(a_bar_entropy)
        ineq_lhs = sum(lhs_values) / len(lhs_values)

        # Criterion: LHS - RHS. If > 0, A1 > A2 asymptotically
        crit_val = ineq_lhs - ineq_rhs

        if crit_val > 0:
            large_NP_outcome = "A1 is roughly larger"
        else:
            large_NP_outcome = "A2 is roughly at least as large"

        result['large_NP_stat'] = crit_val
        result['large_NP_outcome'] = large_NP_outcome

    return result
