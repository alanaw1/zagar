API Reference (Python)
======================

.. module:: zagar

count_binary_matrices
---------------------

.. function:: count_binary_matrices(p, q, sort_p_desc=False)

   Count Binary Matrices with Row Sums :math:`\boldsymbol{p}` and Column Sums :math:`\boldsymbol{q}`.

   The underlying algorithm is a minor modification of the original EXACT algorithm implemented
   in `Miller and Harrison (2013) <https://doi.org/10.1214/13-AOS1131>`_. Compared to their original C version,
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

   Optionally, it can return an approximate count (in base two logarithm
   scale) alongside the exact one, based on a saddle-point approximation.

   .. versionchanged:: 0.2.0

      The approximation now covers every *semi-regular* margin, meaning any
      constraints in which all entries of :math:`\boldsymbol{a}` are equal, all
      entries of :math:`\boldsymbol{\phi}_0` are equal, and all entries of
      :math:`\boldsymbol{\phi}_1` are equal. Previously it was available only
      at the fully symmetric point. Three things changed:

      - the returned dictionary gains ``case``, the regime of the asymptotic approximation
        that applies (see :func:`classify_case`); and, away from the symmetric
        point, ``densities``, the triple
        :math:`(\bar{a}, f_0, f_1)` the formula was evaluated at;
      - a :exc:`RuntimeWarning` is emitted whenever an approximation is
        returned, where earlier versions printed a message to standard output;
      - ``log2_upper_bound`` is still returned, but only at the symmetric
        point, because the bound it implements is a construction specific to
        that point and does not generalize.

      Margins that are not semi-regular are outside the range covered by the approximation
      formula, so they receive the exact count alone, with no approximation and no warning.

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
   :param return_approx: If True, return a dictionary carrying the exact count
      together with whatever approximation applies, rather than the count
      alone (default: False). The keys always present are:

      - ``count``: the exact count, as an arbitrary precision integer
      - ``is_regular``: True only at the fully symmetric point

      Present whenever the margins are semi-regular:

      - ``log2_approx``: :math:`\log_2 |\mathscr{A}_{12}|` from approximation formula
      - ``case``: which of the four regimes applies, an integer in
        :math:`\{1,2,3,4\}`

      Present only away from the symmetric point:

      - ``densities``: the triple :math:`(\bar{a}, f_0, f_1)`

      Present only at the symmetric point:

      - ``log2_upper_bound``: an upper bound on :math:`\log_2 |\mathscr{A}_{12}|`

   :type return_approx: bool, optional
   :returns:
      If return_approx=False: The computed :math:`|\mathscr{A}_{12}|` value (arbitrary precision int)

      If return_approx=True: dict with the count and approximation info
   :rtype: int or dict

   .. warning::

      ``log2_approx`` is asymptotic. It is accurate when :math:`N` and
      :math:`P` are both large with bounded ratio and the three densities are
      held fixed. At small sizes it is off by a few hundredths of a bit, and
      the direction depends on the densities rather than on the size: along
      :math:`f_0 = f_1 = 1/4` with :math:`\bar{a} = 1/2` the error runs about
      :math:`+0.010` bits, while along :math:`f_0 = f_1 = 1/8` it is
      :math:`-0.016` bits at :math:`N = P = 4`. A :exc:`RuntimeWarning` says
      so at each call. See :doc:`asymptotics_ex` for where the sign turns.

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> # Basic usage
      >>> zagar.getA12(N=5, a_vec=[2, 2], phi0_vec=[1, 1], phi1_vec=[1, 1])
      7632
      >>> # The symmetric point: Case 4, and the only one with an upper bound
      >>> result = zagar.getA12(N=4, a_vec=[4]*4, phi0_vec=[4]*4, phi1_vec=[4]*4, return_approx=True)
      >>> result['count']
      824410
      >>> round(result['log2_approx'], 4)  # Approximation of log2(count)
      19.6675
      >>> result['is_regular'], result['case']
      (True, 4)
      >>> # Semi-regular but not symmetric: new in 0.2.0, Case 1 here
      >>> result = zagar.getA12(N=4, a_vec=[4]*4, phi0_vec=[2]*4, phi1_vec=[2]*4, return_approx=True)
      >>> result['case'], result['densities']
      (1, (0.5, 0.25, 0.25))
      >>> 'log2_upper_bound' in result  # does not generalize off the symmetric point
      False


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
   :param return_approx: If True, return a dictionary carrying the exact count
      together with whatever approximation applies (default: False). The
      payload is identical to :func:`getA12`'s, and is produced by the same
      code; see that entry for the full list of keys and when each appears.

      .. versionchanged:: 0.2.0

         Covers every semi-regular margin rather than the symmetric point
         alone, adds ``case`` and ``densities``, and emits a
         :exc:`RuntimeWarning` where earlier versions printed to standard
         output.

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


getA12_pruning
--------------

.. function:: getA12_pruning(N, a_vec, phi0_vec, phi1_vec, n_threads=-1, sort_p_desc=True, return_approx=False)

   Cardinality of Doubly Constrained Admixed Arrays (Pruned and Symmetry-Reduced)

   A third variant of ``getA12``, combining two optimizations on top of the
   parallel version. The first is *sum-based pruning*: the enumeration tracks
   the running total of the column sums chosen so far, and abandons a branch as
   soon as the loci still to come cannot bring that total to
   :math:`\sum_n a_n`. The second is a *symmetry reduction*: at each locus the
   pair :math:`(v_1, v_2)` and its swap :math:`(v_2, v_1)` contribute equally,
   so only one of the two is enumerated and its contribution is doubled.

   Both are speed optimizations. The count is identical to ``getA12`` and
   ``getA12_parallel``.

   :param N: Positive integer parameter (number of individuals)
   :type N: int
   :param a_vec: Row sums (length N)
   :type a_vec: list of int
   :param phi0_vec: First constraint vector (length P)
   :type phi0_vec: list of int
   :param phi1_vec: Second constraint vector (must have same length as phi0_vec)
   :type phi1_vec: list of int
   :param n_threads: Number of threads (-1 = use all available, default: -1)
   :type n_threads: int, optional
   :param sort_p_desc: If True, sort p in decreasing order for speed (default: True)
   :type sort_p_desc: bool, optional
   :param return_approx: As in :func:`getA12`; the payload is produced by the
      same code and has the same keys.
   :type return_approx: bool, optional
   :returns:
      If return_approx=False: The computed :math:`|\mathscr{A}_{12}|` value (arbitrary precision int)

      If return_approx=True: dict with the count and approximation info
   :rtype: int or dict

   .. versionadded:: 0.2.0

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> zagar.getA12_pruning(N=5, a_vec=[2, 2], phi0_vec=[1, 1], phi1_vec=[1, 1])
      7632
      >>> # The optimizations change the cost, not the count
      >>> zagar.getA12_pruning(N=4, a_vec=[4]*4, phi0_vec=[4]*4, phi1_vec=[4]*4)
      824410


count_A12
---------

.. function:: count_A12(N, a_vec, phi0_vec, phi1_vec, method="auto", **kwargs)

   Exact Cardinality, with a Choice of Algorithm

   The recommended entry point for exact counting. It computes
   :math:`|\mathscr{A}_{12}|` exactly, and lets you choose which of two
   independent algorithms does the work.

   The two methods share no code and arrive at the answer differently. The
   **column-sum** method (``getA12`` and its variants) enumerates the feasible
   column-sum vectors :math:`(\boldsymbol{v}_1, \boldsymbol{v}_2)` and, for
   each, counts the binary matrices with those column sums by the
   Miller--Harrison recursion. The **dynamic program** never forms such a
   vector: it folds the loci one at a time, keeping only a histogram of the row
   sums each individual has left to account for, which is what makes it faster
   by orders of magnitude.

   :param N: Positive integer parameter (number of individuals)
   :type N: int
   :param a_vec: Row sums (length N)
   :type a_vec: list of int
   :param phi0_vec: First constraint vector (length P)
   :type phi0_vec: list of int
   :param phi1_vec: Second constraint vector (must have same length as phi0_vec)
   :type phi1_vec: list of int
   :param method: Which algorithm to use (default: ``"auto"``):

      - ``"auto"``: currently ``"dp"``, which was faster on every case tested
      - ``"dp"``, alias ``"histogram-dp"``: the dynamic program
        (:func:`count_A12_dp`)
      - ``"column-sums"``, alias ``"miller-harrison"``: the column-sum
        enumeration (:func:`getA12`)
      - ``"column-sums-parallel"``, alias ``"miller-harrison-parallel"``:
        threaded (:func:`getA12_parallel`)
      - ``"column-sums-pruning"``, alias ``"miller-harrison-pruning"``: pruned
        and symmetry-reduced (:func:`getA12_pruning`)

   :type method: str, optional
   :param kwargs: Passed through to the chosen algorithm, for example
      ``n_threads`` for the parallel variant, or ``reorder_loci`` and
      ``backend`` for ``"dp"``.
   :returns: The exact count
   :rtype: int
   :raises ValueError: If ``method`` is not one of the names above

   .. note::

      Because the algorithmic methods are independent, agreement between them provides a
      genuine check. Our unit tests verify their agreement over hundreds of random instances.
      However, the methods differ considerably in time and memory. At :math:`N = P = 6` the column-sum
      enumeration exhausts 16 GB, where the dynamic program needs 0.01 s and 24 MB. Regardless, the
      column-sum methods are kept for three reasons: (1) :func:`count_binary_matrices` is useful in its
      own right, (2) they are the independent cross-check just described, and (3) :func:`getA1`, :func:`getA2`
      and :func:`compareA1A2` build on the same column-sum machinery.

   .. versionadded:: 0.2.0

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> # The worked example of the manuscript: N = 3 individuals, P = 4 loci
      >>> zagar.count_A12(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1])
      133520
      >>> # The same number by the other algorithm
      >>> zagar.count_A12(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1],
      ...                 method="miller-harrison")
      133520


count_A12_dp
------------

.. function:: count_A12_dp(N, a_vec, phi0_vec, phi1_vec, reorder_loci=True, cache_transitions=False, backend="auto")

   Exact Cardinality by Dynamic Programming over Loci

   Counts :math:`|\mathscr{A}_{12}|` exactly by processing one locus at a
   time. This is the algorithm :func:`count_A12` selects by default; call it
   directly when you want its specific options.

   The method rests on a re-parametrization. Let :math:`e_{np} \in \{0,1,2\}`
   be the number of individual :math:`n`'s two haplotypes that carry ancestry
   1 at locus :math:`p`. Summing over the placements of :math:`\mathbf{X}`
   within each locus first gives

   .. math::

      |\mathscr{A}_{12}| = \sum_{(e_{np}) \,:\, \sum_p e_{np} = a_n}
        \ \prod_{n,p} \binom{2}{e_{np}}
        \ \prod_p \binom{s_p}{\phi_{1,p}} \binom{2N - s_p}{\phi_{0,p}},
        \qquad s_p = \sum_n e_{np},

   which collapses each locus's two columns of :math:`\mathbf{A}`, together
   with all of :math:`\mathbf{X}`, into a single column with entries in
   :math:`\{0,1,2\}`. The program then folds the loci in turn. Individuals
   whose *remaining* row sums are equal are interchangeable, so the state it
   carries is the histogram of those remaining sums rather than the vector of
   them, and the sum over feasible column-sum vectors is never enumerated.

   Non-uniform constraints are supported; nothing here assumes the margins are
   semi-regular.

   :param N: Positive integer parameter (number of individuals)
   :type N: int
   :param a_vec: Row sums (length N). A shorter vector is zero-padded, matching
      :func:`getA12`; a longer one is an error.
   :type a_vec: list of int
   :param phi0_vec: First constraint vector (length P)
   :type phi0_vec: list of int
   :param phi1_vec: Second constraint vector (must have same length as phi0_vec)
   :type phi1_vec: list of int
   :param reorder_loci: If True, process the most constraining loci first
      (default: True). This affects speed only — permuting loci is a bijection
      on admixed arrays that preserves the row sums.
   :type reorder_loci: bool, optional
   :param cache_transitions: If True, memoize each histogram's transition table
      across layers (default: False). This is a poor trade and is off for that
      reason: it costs an order of magnitude in memory to save at most about a
      quarter of the time, and with the compiled backend it is slower as well.
      It has no effect when ``backend="cpp"``.
   :type cache_transitions: bool, optional
   :param backend: Which implementation runs (default: ``"auto"``). All three
      return the same integer.

      - ``"auto"``: ``"cpp"`` when the compiled extension is present and
        :math:`N` is within its 64-bit bound, otherwise ``"python"``
      - ``"cpp"``: the whole computation in C++
      - ``"python"``: the pure-Python driver, which is the reference
        implementation the compiled one is tested against

   :type backend: str, optional
   :returns: The exact count, as an arbitrary precision integer. Returns 0 if
      the constraints are infeasible.
   :rtype: int
   :raises ValueError: If the inputs have inconsistent lengths or out-of-range
      entries, or if ``backend`` is not one of the names above

   .. note::

      Code profiling has shown that this approach is more time- and memory-efficient
      than the earlier column-sum approach (``getA12`` methods). At a deliberately
      expensive parameter point, :math:`\bar{a} = 1/2` and :math:`f_0 = f_1 = 1/4`,
      :math:`N = P = 8` takes 0.44 s and :math:`N = P = 10` takes 45 s, against
      0.014 s at :math:`N = P = 4` and 16 GB exhausted at :math:`N = P = 6` for the
      column-sum enumeration. Across those same runs, time grew by roughly a factor of
      ten per unit of :math:`N` while memory grew two- or threefold.

      Thus time is the binding constraint, and extrapolating that rate puts the practical
      ceiling near :math:`N = 13`. That being said, the rate is a measurement of the
      present implementation at this one parameter point.

   .. versionadded:: 0.2.0

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> zagar.count_A12_dp(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1])
      133520
      >>> # The reference implementation agrees, as it must
      >>> zagar.count_A12_dp(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1],
      ...                    backend="python")
      133520
      >>> # Infeasible constraints give zero, not an error
      >>> zagar.count_A12_dp(2, [0, 0], [0], [3])
      0


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
      >>> # The exact count is astronomically large, so the log is the useful form
      >>> round(zagar.getA1(N=50, P=50, a_vec=[25]*50, return_log=True), 4)
      8884.1223
      >>> len(str(zagar.getA1(N=50, P=50, a_vec=[25]*50)))   # digits in the exact count
      2675


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
      >>> round(zagar.getA2(N=50, P=50, phi0_vec=[20]*50, phi1_vec=[22]*50,
      ...                   return_log=True), 4)
      9570.754
      >>> len(str(zagar.getA2(N=50, P=50, phi0_vec=[20]*50, phi1_vec=[22]*50)))
      2882


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
      A2 is at least as large
      >>> result = zagar.compareA1A2([25]*50, [20]*50, [22]*50, use_large_NP=True)
      >>> round(result['large_NP_stat'], 4)
      -0.1695
      >>> print(result['large_NP_outcome'])
      A2 is roughly at least as large


log2_A12_asymptotic
-------------------

.. function:: log2_A12_asymptotic(N, P, abar, f0, f1, details=False)

   Asymptotic Value of :math:`\log_2 |\mathscr{A}_{12}|` for Semi-Regular Margins

   Evaluates the asymptotic formula, which gives :math:`\log_2 |\mathscr{A}_{12}|` as an
   expansion in :math:`N` and :math:`P` when the constraints are
   *semi-regular* — every individual has the same global ancestry, and every
   locus the same pair of ancestry-specific allele frequencies. The margins are
   then described by three densities rather than by vectors:

   - :math:`\bar{a}`, the fraction of each individual's :math:`2P` haplotype
     slots carrying ancestry 1;
   - :math:`f_0`, the fraction of the :math:`2N` haplotypes at a locus that
     carry ancestry 0 *and* the alternative allele;
   - :math:`f_1`, likewise for ancestry 1.

   Use :func:`densities_from_constraints` to obtain these from integer
   constraint vectors.

   The expansion takes the form

   .. math::

      \log_2 |\mathscr{A}_{12}| \;=\; 2NP\,H \;-\; \tfrac{1}{2}\log_2 \Delta
        \;-\; \tfrac{D}{2}\log_2 \pi \;-\; D \;+\; K \log_2 e,

   in which :math:`H` is the Shannon entropy, in bits, of the four cell types
   the constraints leave available; :math:`\Delta` is the determinant arising
   from the saddle point; :math:`D` is the dimension of the constraint space;
   and :math:`K` is the leading correction.

   Which cell types survive is what divides the parameter space into four
   regimes, and the function selects among them automatically. See
   :func:`classify_case`.

   :param N: Number of individuals (positive)
   :type N: int
   :param P: Number of loci (positive)
   :type P: int
   :param abar: Global ancestry density :math:`\bar{a}`, strictly between 0 and 1
   :type abar: float
   :param f0: Ancestry-0 alternative allele density :math:`f_0`
   :type f0: float
   :param f1: Ancestry-1 alternative allele density :math:`f_1`
   :type f1: float
   :param details: If True, return a dictionary of the expansion's parts rather
      than the value alone (default: False)
   :type details: bool, optional
   :returns:
      If details=False: the value of :math:`\log_2 |\mathscr{A}_{12}|`

      If details=True: a dict with

      - ``log2_A12``: the same value
      - ``case``: the regime, an integer in :math:`\{1,2,3,4\}`
      - ``entropy``: :math:`H`, in bits
      - ``log2_delta``: :math:`\log_2 \Delta`
      - ``dimension``: :math:`D`
      - ``K``: the correction term

   :rtype: float or dict
   :raises ValueError: If :math:`N` or :math:`P` is not positive, or if the
      densities lie outside the feasible region
      :math:`f_1 \leq \bar{a} \leq 1 - f_0`

   .. warning::

      This is an asymptotic value, accurate when :math:`N` and :math:`P` are
      both large with bounded ratio **and the three densities are held
      fixed**. At finite size it is off by a few hundredths of a bit in either
      direction. Against exact enumeration it is high by :math:`+0.011` bits
      at :math:`N = P = 4` along :math:`\bar{a} = 1/2`,
      :math:`f_0 = f_1 = 1/4`, but low by :math:`-0.016` bits at the same size
      along :math:`f_0 = f_1 = 1/8`. The sign is not guaranteed: over every
      margin small enough to check exactly, 252 of the 680 at :math:`N = P = 8`
      are underestimates. :doc:`asymptotics_ex` tabulates this.

      Letting the densities shrink with :math:`N` violates the assumptions of
      the asymptotic approximation, and the error may grow rather than shrink — see
      the same page.

   .. versionadded:: 0.2.0

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> round(zagar.log2_A12_asymptotic(10, 10, 0.5, 0.25, 0.25), 4)
      330.0294
      >>> d = zagar.log2_A12_asymptotic(10, 10, 0.5, 0.25, 0.25, details=True)
      >>> d['case'], d['dimension']   # interior regime, D = N + 2P
      (1, 30)


classify_case
-------------

.. function:: classify_case(abar, f0, f1)

   Which Regime of the Asymptotic Approximation Applies

   Each of the :math:`2NP` haplotype-locus cells of an admixed array has a
   type :math:`(A, X) \in \{0,1\}^2`, and the three densities fix how common
   each type is:

   .. math::

      \Pr[(1,1)] = f_1, \quad \Pr[(0,1)] = f_0, \quad
      \Pr[(1,0)] = \bar{a} - f_1, \quad \Pr[(0,0)] = 1 - \bar{a} - f_0.

   Requiring all four to be nonnegative gives the feasible region
   :math:`f_1 \leq \bar{a} \leq 1 - f_0`, and the regimes are the places where
   a cell type empties out:

   .. list-table::
      :header-rows: 1
      :widths: 10 26 26 38

      * - Case
        - Condition
        - Empty cell type
        - Consequence
      * - 1
        - interior
        - none
        - all four types present, :math:`D = N + 2P`
      * - 2
        - :math:`f_1 = \bar{a}`
        - :math:`(1,0)`
        - :math:`X = 1` wherever :math:`A = 1`, so
          :math:`\sum_p \phi_{1,p} = \sum_n a_n` and :math:`D = N + 2P - 1`
      * - 3
        - :math:`\bar{a} = 1 - f_0`
        - :math:`(0,0)`
        - :math:`X = 1` wherever :math:`A = 0`, so
          :math:`\sum_p \phi_{0,p} = 2NP - \sum_n a_n` and
          :math:`D = N + 2P - 1`
      * - 4
        - both
        - :math:`(1,0)` and :math:`(0,0)`
        - :math:`\mathbf{X} \equiv 1`, so the array is determined by
          :math:`\mathbf{A}` alone and :math:`D = N + P - 1`

   In each degenerate case one constraint becomes implied by the others, which
   is why the dimension :math:`D` drops.

   The boundaries are tested with a small tolerance, and this is deliberate
   rather than defensive: the Case 1 correction term diverges like
   :math:`1/(\bar{a} - f_1)` as :math:`f_1 \to \bar{a}`, so near a boundary the
   boundary formula is the correct one, not merely the safer one. A triple that
   floating-point arithmetic has nudged a hair outside the feasible region is
   absorbed for the same reason; one genuinely outside it raises.

   :param abar: Global ancestry density :math:`\bar{a}`
   :type abar: float
   :param f0: Ancestry-0 alternative allele density :math:`f_0`
   :type f0: float
   :param f1: Ancestry-1 alternative allele density :math:`f_1`
   :type f1: float
   :returns: The case, an integer in :math:`\{1,2,3,4\}`
   :rtype: int
   :raises ValueError: If the densities are infeasible or degenerate, meaning
      :math:`f_1 > \bar{a}`, or :math:`\bar{a} > 1 - f_0`, or any density on
      the closed boundary of :math:`[0,1]`

   .. versionadded:: 0.2.0

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> zagar.classify_case(0.5, 0.25, 0.25)   # interior
      1
      >>> zagar.classify_case(0.4, 0.3, 0.4)     # f1 == abar
      2
      >>> zagar.classify_case(0.7, 0.3, 0.2)     # abar == 1 - f0
      3
      >>> zagar.classify_case(0.3, 0.7, 0.3)     # both
      4


entropy_bits
------------

.. function:: entropy_bits(*p)

   Shannon Entropy in Bits

   Computes :math:`H(\boldsymbol{p}) = -\sum_i p_i \log_2 p_i`, with the usual
   convention :math:`0 \log_2 0 = 0`. This is the leading term of the asymptotic
   approximation, where the arguments are the densities of the cell types that the
   constraints leave available --- four of them in Case 1, three in Cases 2 and 3, two in
   Case 4.

   :param p: The component probabilities, which must be nonnegative and sum
      to 1
   :type p: float
   :returns: The entropy in bits
   :rtype: float
   :raises ValueError: If any component is negative, or if the components do
      not sum to 1

   .. versionadded:: 0.2.0

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> zagar.entropy_bits(0.5, 0.5)
      1.0
      >>> zagar.entropy_bits(0.25, 0.25, 0.25, 0.25)
      2.0
      >>> zagar.entropy_bits(1.0, 0.0)     # 0 log 0 = 0
      0.0


densities_from_constraints
--------------------------

.. function:: densities_from_constraints(N, a_vec, phi0_vec, phi1_vec)

   Densities :math:`(\bar{a}, f_0, f_1)` from Integer Constraint Vectors

   Converts semi-regular constraint vectors into the three densities that
   the asymptotic approximation formula is stated in terms of, so that
   :func:`log2_A12_asymptotic` and :func:`classify_case` can be applied
   to a concrete problem.

   The two denominators differ, and it is worth being explicit about why.
   Each individual has :math:`2P` haplotype slots across the loci, so

   .. math::

      \bar{a} = \frac{a_n}{2P},

   while each locus is observed on :math:`2N` haplotypes across the cohort, so

   .. math::

      f_0 = \frac{\phi_{0,p}}{2N}, \qquad f_1 = \frac{\phi_{1,p}}{2N}.

   These coincide only when :math:`N = P`.

   Note also that :math:`f_0` and :math:`f_1` are *joint* proportions over all
   :math:`2N` haplotypes at a locus, not frequencies conditional on ancestry:
   :math:`f_0 + f_1` is the overall alternative allele frequency there.

   :param N: Positive integer parameter (number of individuals)
   :type N: int
   :param a_vec: Row sums, all equal (length N)
   :type a_vec: list of int
   :param phi0_vec: First constraint vector, all equal (length P)
   :type phi0_vec: list of int
   :param phi1_vec: Second constraint vector, all equal (same length as phi0_vec)
   :type phi1_vec: list of int
   :returns: ``{'N': int, 'P': int, 'abar': float, 'f0': float, 'f1': float}``
   :rtype: dict
   :raises ValueError: If the vectors have inconsistent lengths, or if the
      margins are not semi-regular — that is, if the entries of any one of the
      three vectors are not all equal

   .. note::

      A density is only realizable when the corresponding product is an
      integer. At :math:`N = 11`, for instance, :math:`2N f_0 = 5.5` for
      :math:`f_0 = 1/4`, so a quarter density cannot be expressed at that size
      at all. Building constraint vectors by rounding is a reliable way to
      compute at a parameter point other than the intended one.

   .. versionadded:: 0.2.0

   **Examples**

   .. code-block:: python

      >>> import zagar
      >>> d = zagar.densities_from_constraints(4, [4]*4, [2]*4, [2]*4)
      >>> d['abar'], d['f0'], d['f1']
      (0.5, 0.25, 0.25)
      >>> # N != P, where the two denominators come apart: abar over 2P = 8,
      >>> # the frequencies over 2N = 4
      >>> d = zagar.densities_from_constraints(2, [3]*2, [1]*4, [2]*4)
      >>> d['abar'], d['f0'], d['f1']
      (0.375, 0.25, 0.5)


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
      >>> isinstance(zagar.has_openmp(), bool)
      True


Module Attributes
-----------------

.. data:: __version__

   Current version of the zagar package.

   .. code-block:: python

      >>> import zagar
      >>> zagar.__version__
      '0.2.0'
