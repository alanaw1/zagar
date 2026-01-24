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
    getA1,
    getA2,
    has_openmp,
)

__version__ = "0.1.0"
__all__ = [
    "count_binary_matrices",
    "getA12",
    "getA12_parallel",
    "getA1",
    "getA2",
    "compareA1A2",
    "get_constraints",
    "count_matrices",
    "has_openmp",
]


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


def _compute_approximations(N, P):
    """
    Compute Mathematical Approximations for Regular Case.

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
    >>> result['log2_approx']  # Approximation of log2(count)
    19.6675
    """
    a_vec = list(a_vec)
    phi0_vec = list(phi0_vec)
    phi1_vec = list(phi1_vec)

    count = _getA12_core(N, a_vec, phi0_vec, phi1_vec, sort_p_desc)

    if not return_approx:
        return count

    P = len(phi0_vec)
    is_regular = _is_regular_case(N, a_vec, phi0_vec, phi1_vec)

    result = {
        "count": count,
        "is_regular": is_regular,
    }

    if is_regular:
        print("Computing estimated log-counts based on saddle-point approximation.")
        print("Note: log2_approx is good only when P and N are large with bounded ratio")
        approx = _compute_approximations(N, P)
        result["log2_upper_bound"] = approx["log2_upper_bound"]
        result["log2_approx"] = approx["log2_approx"]

    return result


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

    P = len(phi0_vec)
    is_regular = _is_regular_case(N, a_vec, phi0_vec, phi1_vec)

    result = {
        "count": count,
        "is_regular": is_regular,
    }

    if is_regular:
        print("Computing estimated log-counts based on saddle-point approximation.")
        print("Note: log2_approx is good only when P and N are large with bounded ratio")
        approx = _compute_approximations(N, P)
        result["log2_upper_bound"] = approx["log2_upper_bound"]
        result["log2_approx"] = approx["log2_approx"]

    return result


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
    >>> result = zagar.compareA1A2([25]*50, [20]*50, [22]*50, use_large_NP=True)
    >>> print(result['large_NP_stat'])

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
