API Reference (Python)
======================

.. module:: zagar

count_binary_matrices
---------------------

.. function:: count_binary_matrices(p, q, sort_p_desc=False)

   Count Binary Matrices with Row Sums :math:`\boldsymbol{p}` and Column Sums :math:`\boldsymbol{q}`.

   The underlying algorithm is a minor modification of the original EXACT algorithm implemented
   in Miller and Harrison (2013) *The Annals of Statistics*. Compared to their original C version,
   this version uses modern C++ optimizations (e.g., ``std::unordered_map`` in C++
   versus ``khash`` from ``klib`` for hashing) and performs operations on flexible
   integer types, which may be more efficient when the problem size is not too large
   (e.g., ``uint64_t`` rather than arbitrary precision types for smaller input values).

   :param p: Row sums (nonnegative integers)
   :type p: list of int
   :param q: Column sums (nonnegative integers)
   :type q: list of int
   :param sort_p_desc: If True, sort p in decreasing order for speed (default: False)
   :type sort_p_desc: bool, optional
   :returns: ``{'number': int, 'memo_entries': int}``

      - ``number``: The count of binary matrices (arbitrary precision)
      - ``memo_entries``: Number of entries in the memoization table

   :rtype: dict

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> result = zagar.count_binary_matrices([2, 1], [1, 1, 1])
      >>> result['number']
      3


count_matrices
--------------

.. function:: count_matrices(p, q, sort_p_desc=False)

   Wrapper for ``count_binary_matrices``

   This is a wrapper for ``count_binary_matrices`` that returns just the count.

   :param p: Row sums (nonnegative integers)
   :type p: list or array-like
   :param q: Column sums (nonnegative integers)
   :type q: list or array-like
   :param sort_p_desc: If True, sort p in decreasing order for speed optimization
   :type sort_p_desc: bool, optional
   :returns: Number of binary matrices with the given marginals
   :rtype: int

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> zagar.count_matrices([2, 1], [1, 1, 1])
      3


getA12
------

.. function:: getA12(N, a_vec, phi0_vec, phi1_vec, sort_p_desc=True, return_approx=False)

   Cardinality of Doubly Constrained Admixed Arrays

   Uses a modified version of Miller and Harrison's EXACT algorithm
   to compute the size of a set of admixed arrays with specified
   global ancestry vector :math:`\boldsymbol{a}` (``a_vec``, a row
   sum constraint) and ancestry-specific allele frequencies
   :math:`(\boldsymbol{\phi}_0,\boldsymbol{\phi}_1)` (``phi0_vec`` and
   ``phi1_vec``, column sum constraint).

   Optionally, it can return an approximate count and upper bound
   (in base two logarithm scale), based on a saddle-point approximation.
   Currently only works for the following special regular case:

   - ``phi0_vec`` and ``phi1_vec`` entries all equal to ``N``
   - ``a_vec`` values all equal to the size of the list ``phi0_vec``

   :param N: Positive integer parameter (number of individuals)
   :type N: int
   :param a_vec: Row sums (length N, all equal to P for regular case)
   :type a_vec: list of int
   :param phi0_vec: First constraint vector (length P)
   :type phi0_vec: list of int
   :param phi1_vec: Second constraint vector (must have same length as phi0_vec)
   :type phi1_vec: list of int
   :param sort_p_desc: If True, sort p in decreasing order for speed (default: True)
   :type sort_p_desc: bool, optional
   :param return_approx:
      - If True and inputs form a "regular case" (all a_vec equal P, all phi0_vec equal N, all phi1_vec equal N),
        return a dict with: 'count' (exact count), 'is_regular' (True),
        'log2_upper_bound' (upper bound on log2(count)), and 'log2_approx'
        (approximation valid when N and P are large with bounded ratio).
      - If False or not regular case, return just the count (default: False)
   :type return_approx: bool, optional
   :returns:
      If return_approx=False: The computed :math:`|\mathscr{A}_{12}|` value (arbitrary precision int)

      If return_approx=True: dict with count and approximation info
   :rtype: int or dict

   **Examples**

   .. code-block:: python

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


getA12_parallel
---------------

.. function:: getA12_parallel(N, a_vec, phi0_vec, phi1_vec, n_threads=-1, sort_p_desc=True, return_approx=False)

   Cardinality of Doubly Constrained Admixed Arrays (Parallelized)

   This is a parallelized version of ``getA12``, which relies on
   OpenMP. If the user already has OpenMP, this function should be used
   in place of ``getA12`` to speed up calculations.

   Parallelization is achieved by assigning workers to each value of the
   first coordinate of the constrained set :math:`\mathscr{S}`.

   :param N: Positive integer parameter (number of individuals)
   :type N: int
   :param a_vec: Row sums (length N, typically all equal to P for regular case)
   :type a_vec: list of int
   :param phi0_vec: First constraint vector (length P)
   :type phi0_vec: list of int
   :param phi1_vec: Second constraint vector (must have same length as phi0_vec)
   :type phi1_vec: list of int
   :param n_threads: Number of threads (-1 = use all available, default: -1)
   :type n_threads: int, optional
   :param sort_p_desc: If True, sort p in decreasing order for speed (default: True)
   :type sort_p_desc: bool, optional
   :param return_approx: If True and inputs form a "regular case" (all a_vec equal P,
      all phi0_vec equal N, all phi1_vec equal N), return a dict with:

      - 'count': exact count
      - 'is_regular': True
      - 'log2_upper_bound': upper bound on log2(count)
      - 'log2_approx': approximation valid when N and P are large with bounded ratio

      If False or not regular case, returns just the count (default: False)
   :type return_approx: bool, optional
   :returns:
      If return_approx=False: The computed :math:`|\mathscr{A}_{12}|` value (arbitrary precision int)

      If return_approx=True: dict with count and approximation info
   :rtype: int or dict

   .. note::

      This function uses OpenMP for parallelization when available.
      If OpenMP is not available at compile time, it falls back to serial execution.

   **Examples**

   .. code-block:: python

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


getA1
-----

.. function:: getA1(N, P, a_vec, return_log=False)

   Cardinality of :math:`\mathscr{A}_1`: Number of Arrays Subject to Row Margin Constraints

   Formula: prod(C(2*P, a_n)) * 2^(2*N*P) for all a_n in a_vec

   Mathematically,
   :math:`2^{2NP} \prod_{n=1}^N \binom{2P}{a_{n\cdot}}` for all :math:`a_{n\cdot}` in :math:`\boldsymbol{a}`

   :param N: Positive integer (number of rows)
   :type N: int
   :param P: Positive integer (number of columns)
   :type P: int
   :param a_vec: Row margin constraints (length N, values in [0, 2*P])
   :type a_vec: list of int
   :param return_log: If True, return log2 of result as float (default: False)
   :type return_log: bool, optional
   :returns:
      If return_log=False: The exact :math:`|\mathscr{A}_1|` value (arbitrary precision)

      If return_log=True: :math:`\log_2(|\mathscr{A}_1|)` as a float
   :rtype: int or float

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> zagar.getA1(N=50, P=50, a_vec=[25]*50)
      >>> zagar.getA1(N=50, P=50, a_vec=[25]*50, return_log=True)


getA2
-----

.. function:: getA2(N, P, phi0_vec, phi1_vec, return_log=False)

   Cardinality of :math:`\mathscr{A}_2`: Number of Arrays Subject to Column Margin Constraints

   Formula: prod(C(2*N, phi0_p+phi1_p) * C(phi0_j+phi1_p, phi1_p) * 2^(2*N-(phi0_p+phi1_p)))

   Mathematically,
   :math:`\prod_{p=1}^P \binom{2N}{\phi_{p,0}+\phi_{p,1}} \binom{\phi_{p,0}+\phi_{p,1}}{\phi_{p,0}} 2^{2N - (\phi_{p,0}+\phi_{p,1})}`

   :param N: Positive integer (number of rows)
   :type N: int
   :param P: Positive integer (number of columns)
   :type P: int
   :param phi0_vec: First constraint vector (length P, values in [0, 2*N])
   :type phi0_vec: list of int
   :param phi1_vec: Second constraint vector (length P, phi0_j+phi1_j <= 2*N)
   :type phi1_vec: list of int
   :param return_log: If True, return log2 of result as float (default: False)
   :type return_log: bool, optional
   :returns:
      If return_log=False: The exact :math:`|\mathscr{A}_2|` value (arbitrary precision)

      If return_log=True: :math:`\log_2(|\mathscr{A}_2|)` as a float
   :rtype: int or float

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> zagar.getA2(N=50, P=50, phi0_vec=[20]*50, phi1_vec=[22]*50)
      >>> zagar.getA2(N=50, P=50, phi0_vec=[20]*50, phi1_vec=[22]*50, return_log=True)


compareA1A2
-----------

.. function:: compareA1A2(a_vec, phi0_vec, phi1_vec, use_large_NP=False)

   Compare Cardinalities of :math:`\mathscr{A}_1` and :math:`\mathscr{A}_2`

   This function computes :math:`|\mathscr{A}_1|` and :math:`|\mathscr{A}_2|` and compares their sizes.
   Optionally, it can also compute an asymptotic approximation for large :math:`N`, :math:`P`
   based on entropy calculations.

   :param a_vec: Row margin constraints (length N)
   :type a_vec: list of int
   :param phi0_vec: First constraint vector (length P)
   :type phi0_vec: list of int
   :param phi1_vec: Second constraint vector (length P, same length as phi0_vec)
   :type phi1_vec: list of int
   :param use_large_NP: If True, compute asymptotic comparison using entropy criterion (default: False)
   :type use_large_NP: bool, optional
   :returns: Dictionary with keys:

      - ``'log2_A1'``: float - log2 of :math:`|\mathscr{A}_1|`
      - ``'log2_A2'``: float - log2 of :math:`|\mathscr{A}_2|`
      - ``'outcome'``: str - "A1 is larger" or "A2 is at least as large"

      If use_large_NP=True, also includes:

      - ``'large_NP_stat'``: float - the asymptotic criterion value (LHS - RHS)
      - ``'large_NP_outcome'``: str - "A1 is roughly larger" or "A2 is roughly at least as large"

   :rtype: dict

   .. note::

      The asymptotic condition for A1 > A2 (when use_large_NP=True) is:
          H(``a_bar``) > H(``f0``, ``f1``, 1-``f0``-``f1``) - (``f0`` + ``f1``)
      where:
          - ``a_bar`` = ``a_vec`` / (2*P), the normalized row margins
          - ``f0`` = ``phi0_vec`` / (2*N), ``f1`` = ``phi1_vec`` / (2*N), the normalized column margins
          - H() denotes entropy

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> result = zagar.compareA1A2([25]*50, [20]*50, [22]*50)
      >>> print(result['outcome'])
      >>> result = zagar.compareA1A2([25]*50, [20]*50, [22]*50, use_large_NP=True)
      >>> print(result['large_NP_stat'])


get_constraints
---------------

.. function:: get_constraints(A, X)

   Compute Marginal Constraint Vectors from Admixed Array

   Given binary matrices :math:`\mathbf{A}` (ancestry) and :math:`\mathbf{X}` (allele dosage)
   that make up the admixed array :math:`[\mathbf{A},\mathbf{X}]`, computes three marginal
   constraint vectors used in constrained admixed array enumeration.

   :param A: Binary matrix of shape (N, 2P) representing ancestry assignments.
      A[n, p] = 1 means individual n has ancestry 1 at position p.
      May contain np.nan values for missing data.
   :type A: np.ndarray
   :param X: Binary matrix of shape (N, 2P) representing allele values.
      X[n, p] = 1 means individual n has allele 1 at position p.
      May contain np.nan values for missing data.
   :type X: np.ndarray
   :returns: Tuple of three lists:

      - ``a_vec``: list of int - Length N. Entry n is the row sum of A (total ancestry-1 count for
        individual n). This is the row margin constraint for :math:`\mathscr{A}_1`.
        NA values in A are ignored.
      - ``phi0_vec``: list of int - Length P. Entry p is the sum of X values at positions where A=0,
        across columns p and P+p for all individuals.
        This is the first column margin constraint for :math:`\mathscr{A}_2`.
      - ``phi1_vec``: list of int - Length P. Entry p is the sum of X values at positions where A=1,
        across columns p and P+p for all individuals.
        This is the second column margin constraint for :math:`\mathscr{A}_2`.

   :rtype: tuple of (list, list, list)

   .. note::

      If either A or X contains NA/NaN values, a warning is issued but computation
      continues. NA values are ignored in all sums (they contribute 0 to the totals).

   **Examples**

   .. code-block:: python

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


has_openmp
----------

.. function:: has_openmp()

   Check if OpenMP support is available.

   Returns True if the package was compiled with OpenMP support,
   False otherwise. When OpenMP is not available, getA12_parallel
   falls back to serial execution.

   :returns: True if OpenMP is available, False otherwise.
   :rtype: bool

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> zagar.has_openmp()
      False  # or True if compiled with OpenMP


Module Attributes
-----------------

.. data:: __version__

   Current version of the zagar package.

   .. code-block:: python

      >>> import zagar
      >>> zagar.__version__
      '0.1.0'
