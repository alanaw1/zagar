Python Example
==============

Basics
------

To make sure Python works, try to generate a binary matrix (which we refer to as :math:`\mathbf{X}`).

.. code-block:: python

   >>> import numpy as np
   >>> import random

   # Seed the generator so this page is reproducible; drop this line to get
   # a different matrix each time
   >>> random.seed(0)

   # Define matrix dimensions
   >>> rows = 6
   >>> cols = 8

   # Generate random binary matrix
   >>> X = np.array([[random.choice([0, 1]) for _ in range(cols)] for _ in range(rows)])
   >>> print(X)
   [[1 1 0 1 1 1 1 1]
    [1 0 0 1 0 0 1 0]
    [1 0 0 1 1 0 1 1]
    [1 0 1 1 1 0 0 0]
    [1 0 1 1 0 1 0 0]
    [0 0 0 1 0 0 1 1]]
   >>> print(X.shape)
   (6, 8)

If the above works, we are ready to proceed with performing combinatorial calculations.
First, import **zagar**. (If you have not installed it, go to the `Installation page <./installation.html>`__ on
this website and follow instructions there.)

.. code-block:: python

   >>> import zagar

Binary matrices form a class of rich and practical discrete structures.

- *Rich*, because they lead to interesting mathematical theory (see e.g., `Barvinok 2010 <https://doi.org/10.1016/j.aim.2009.12.001>`__, `Cooper et al. 2019 <https://doi.org/10.37236/8092>`__, `Liebenau and Wormald 2022 <https://doi.org/10.1002/rsa.21105>`__ and many others not listed);
- *Practical*, because they represent many data objects encountered in reality (e.g., contingency tables in statistics, genetic polymorphism data, codes for information transmission).

With **zagar**, we can investigate the following combinatorial question:

   Given marginal constraints, like the row and column sums, of a binary matrix, how many such matrices
   are there that satisfy them?

The answer to this question is challenging in general. But for reasonably sized matrix dimensions, the function ``count_matrices`` calculates the answer. Let us use the binary matrix :math:`\mathbf{X}` generated earlier to demonstrate its usage.

.. code-block:: python

   # Compute row and column margins
   >>> row_margins = X.sum(axis=1).tolist()
   >>> col_margins = X.sum(axis=0).tolist()

   # Check
   >>> print(row_margins)
   [7, 3, 5, 4, 4, 3]
   >>> print(col_margins)
   [5, 1, 2, 6, 3, 2, 4, 3]

   # Count number of binary matrices with observed marginal constraints
   >>> zagar.count_matrices(p=row_margins, q=col_margins, sort_p_desc=True)
   12880

There are :math:`12,880` such binary matrices --- a huge count, for a matrix with only 48 entries.

Admixed Arrays
--------------

Closely related to a binary matrix is an admixed array. An *admixed array* is a pair :math:`[\mathbf{A},\mathbf{X}]`, where :math:`\mathbf{A}=(A_{np})` and :math:`\mathbf{X}=(X_{np})` lie in :math:`\{0,1\}^{N\times 2P}`. Here, :math:`N` and :math:`P` are positive integers providing the dimensionality of the underlying binary matrices; they correspond to the number of individuals and the number of genetic loci in an admixed population dataset.

Let us generate a random admixed array.

.. code-block:: python

   # Define array parameters
   >>> N = 6
   >>> P = 5

   # Generate random admixed array (seeded again, so this block stands alone)
   >>> random.seed(1)
   >>> X = np.array([[random.choice([0, 1]) for _ in range(2*P)] for _ in range(N)])
   >>> A = np.array([[random.choice([0, 1]) for _ in range(2*P)] for _ in range(N)])

   >>> print(A)
   [[0 0 1 1 1 0 1 0 1 1]
    [0 0 0 0 0 0 1 1 1 1]
    [1 0 1 0 0 1 0 1 1 0]
    [1 1 1 1 1 0 1 1 0 0]
    [0 0 0 1 0 0 0 0 1 0]
    [1 0 1 0 0 1 1 0 0 0]]
   >>> print(X)
   [[0 0 1 0 1 1 1 1 0 0]
    [1 0 1 1 0 1 1 0 0 1]
    [0 0 0 0 1 0 1 0 0 1]
    [1 0 1 0 0 1 1 0 1 0]
    [0 1 0 1 1 0 1 1 1 1]
    [0 1 0 1 1 0 1 1 0 1]]

The following two summary statistics (modulo normalization factors that are irrelevant here) are studied in genetic studies of admixed populations.

- *Individual global ancestry*: For each individual :math:`n\in[N]`, their global ancestry dosage is :math:`A_{n\cdot}=\sum_{p=1}^{2P}A_{np}`. Equivalently, by gathering the values across all individuals we obtain a vector of individual global ancestry dosages called the *ancestry dosage vector*, :math:`\boldsymbol{A}=(A_{1\cdot},\ldots,A_{N\cdot})`.
- *Ancestry-specific allele frequencies*: For each locus :math:`p\in[P]`, the pair :math:`(\Phi_{p,0},\Phi_{p,1})` records the allele dosage specific to each ancestry. Mathematically, :math:`\Phi_{p,0}=\sum_{n=1}^N \left[(1-A_{np})X_{np}+(1-A_{n(P+p)})X_{n(P+p)}\right]` and :math:`\Phi_{p,1}=\sum_{n=1}^N \left(A_{np}X_{np}+A_{n(P+p)}X_{n(P+p)}\right)`. Again, by gathering the values across all loci we obtain two *ancestry-specific allele dosage vectors*: :math:`\boldsymbol{\Phi}_0=(\Phi_{1,0},\ldots,\Phi_{P,0})` and :math:`\boldsymbol{\Phi}_1=(\Phi_{1,1},\ldots,\Phi_{P,1})`.

We compute these summaries for the admixed array generated above.

.. code-block:: python

   # Individual global ancestry (row sums of A)
   >>> row_sum_A = A.sum(axis=1).tolist()
   >>> print(row_sum_A)
   [6, 4, 5, 7, 2, 4]

   # Ancestry-specific allele frequencies
   >>> phi0 = []
   >>> phi1 = []
   >>> for p in range(P):
   ...     s0 = 0
   ...     s1 = 0
   ...     for n in range(N):
   ...         a1 = A[n, p]
   ...         a2 = A[n, P + p]
   ...         x1 = X[n, p]
   ...         x2 = X[n, P + p]
   ...
   ...         s0 += (1 - a1) * x1 + (1 - a2) * x2
   ...         s1 += a1 * x1 + a2 * x2
   ...
   ...     phi0.append(int(s0))
   ...     phi1.append(int(s1))

   >>> print(phi0)
   [4, 4, 4, 3, 6]
   >>> print(phi1)
   [1, 4, 2, 2, 2]

   # Optional: sanity checks
   # 1) row sum constraints must sum to no. 1's in A
   # 2) column sum constraints must sum to no. 1's in X
   >>> bool(sum(row_sum_A) == A.sum())
   True
   >>> bool((sum(phi0) + sum(phi1)) == X.sum())
   True

The code above was written to help reinforce understanding of how these summaries are calculated from data. These summaries can also be obtained with a single call to the function ``get_constraints``:

.. code-block:: python

   # Compute phi0, phi1 and row_sum_A
   >>> constraint_list = zagar.get_constraints(A, X)
   >>> constraint_list[0] # row_sum_A
   [6, 4, 5, 7, 2, 4]
   >>> constraint_list[1] # phi0
   [4, 4, 4, 3, 6]
   >>> constraint_list[2] # phi1
   [1, 4, 2, 2, 2]


Now assume that these summaries are viewed as constraints. We can ask an analogous combinatorial question of our admixed array.

   Given the marginal constraints (on individual global ancestry and ancestry-specific allele frequencies) of an admixed array,
   how many admixed arrays are there that satisfy them?

We concretely define three sets of constrained admixed arrays.

- *Individual global ancestry constraint*: :math:`\mathscr{A}_{1}=\{[\mathbf{A},\mathbf{X}]:~(\forall n \in [N])(A_{n\cdot}=a_{n\cdot})\}`. Equivalently, :math:`\mathscr{A}_{1}` denotes the set of admixed arrays for which :math:`\boldsymbol{A}=\boldsymbol{a}`, for some fixed non-negative integer :math:`N`-vector :math:`\boldsymbol{a}=(a_{1\cdot},\ldots,a_{N\cdot})`.
- *Ancestry-specific allele frequency constraint*: :math:`\mathscr{A}_{2}=\{[\mathbf{A},\mathbf{X}]:~(\forall p\in[P])(\Phi_{p,0}=\phi_{p,0} \wedge \Phi_{p,1}=\phi_{p,1})\}`. Equivalently, :math:`\mathscr{A}_{2}` denotes the set of admixed arrays for which :math:`\boldsymbol{\Phi}_0=\boldsymbol{\phi}_0` and :math:`\boldsymbol{\Phi}_1=\boldsymbol{\phi}_1`, for some fixed non-negative integer :math:`P`-vectors :math:`\boldsymbol{\phi}_0=(\phi_{1,0},\ldots,\phi_{P,0})` and :math:`\boldsymbol{\phi}_1=(\phi_{1,1},\ldots,\phi_{P,1})`.
- *Both constraints*: :math:`\mathscr{A}_{12}` denotes the set of admixed arrays for which :math:`\boldsymbol{A}=\boldsymbol{a}`, :math:`\boldsymbol{\Phi}_0=\boldsymbol{\phi}_0` and :math:`\boldsymbol{\Phi}_1=\boldsymbol{\phi}_1`.

With **zagar**, we can compute the sizes of these constrained sets of arrays using ``getA1``, ``getA2`` and ``count_A12``.

The first two have closed forms and are immediate. The third is the hard one,
and is what most of this package is about.

.. code-block:: python

   # Size of A1
   >>> zagar.getA1(N=N, P=P, a_vec=row_sum_A)
   14529541998506384596716748800000

   # Size of A2
   >>> zagar.getA2(N=N, P=P, phi0_vec=phi0, phi1_vec=phi1)
   56038944358306978725888000000

   # Size of A12
   >>> zagar.count_A12(N, row_sum_A, phi0, phi1)
   551871989144398349945520

Note that :math:`|\mathscr{A}_{12}|` is far smaller than either
:math:`|\mathscr{A}_1|` or :math:`|\mathscr{A}_2|`: imposing both families of
constraints at once is much more restrictive than imposing either alone.

``count_A12`` lets you choose the algorithm, and the choice matters. The
default is a dynamic program over loci; ``getA12`` is the original enumeration
over column-sum vectors, using the Miller--Harrison recursion. Both return the
same integer, as they must:

.. code-block:: python

   >>> zagar.count_A12(N, row_sum_A, phi0, phi1, method="miller-harrison")
   551871989144398349945520

On this small example the first call takes about 0.01 sec and the second about
3 sec. In general, you should use ``count_A12`` with the default setting unless
you have a particular reason to want the other algorithm (e.g., to check one against the other).

If you have OpenMP installed, ``getA12_parallel`` (or
``method="miller-harrison-parallel"``) threads the column-sum enumeration.

Saddle-Point Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us consider a regular admixed array, defined by :math:`\boldsymbol{a}=(P,...,P)` and :math:`\boldsymbol{\phi}_0=\boldsymbol{\phi}_1=(N,...,N)`. In this constrained case, a `saddle-point approximation <https://doi.org/10.2307/2686100>`__ to a `Cauchy integral <https://mathworld.wolfram.com/CauchyIntegralFormula.html>`__ can be used to estimate the size of :math:`\mathscr{A}_{12}`:

:math:`\log_2(|\mathscr{A}_{12}|)\approx 2NP - \frac{1}{2}[N\log_2(\pi P) + P\log_2(\pi N)-\log_2(\pi NP)] - \log_2(e)\cdot(N+P-1)^2/(8NP)`

We demonstrate this functionality in **zagar**.

.. code-block:: python

   >>> import math

   # Array parameters
   >>> N = 9
   >>> P = 3

   # Define regular constraints
   >>> a_reg = [P] * N       # length N, all entries = P
   >>> phi0_reg = [N] * P    # length P, all entries = N
   >>> phi1_reg = [N] * P    # length P, all entries = N

   # Compute size of A12 and report the approximation and upper bound.
   # A RuntimeWarning is raised to say the value is asymptotic; in earlier
   # versions this was printed to standard output instead.
   >>> res = zagar.getA12(N=N, a_vec=a_reg, phi0_vec=phi0_reg, phi1_vec=phi1_reg, return_approx=True)
   >>> print(res)
   {'count': 25697841152, 'is_regular': True,
    'log2_upper_bound': 35.40699628357531,
    'log2_approx': 34.59881989492918, 'case': 4}

   # Saddle-point approximation
   >>> print(res['log2_approx'])
   34.59881989492918

   # Upper bound
   >>> print(res['log2_upper_bound'])
   35.40699628357531

   # Compute log2(A12)
   >>> round(math.log2(25697841152), 4)
   34.5809


For :math:`(N,P)=(9,3)`, there are :math:`|\mathscr{A}_{12}|=25,697,841,152` (more than 25 billion!) regular admixed arrays. This is equivalent to about :math:`\log_2(|\mathscr{A}_{12}|)=34.6` bits, which is well-approximated by the saddle-point formula.

Beyond the Regular Case (v0.2.0 update)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The formula above is specific to the fully symmetric point. As of v0.2.0
the approximation covers every *semi-regular* margin, that is, any constraints in
which all entries of :math:`\boldsymbol{a}` are equal, all entries of
:math:`\boldsymbol{\phi}_0` are equal, and all entries of
:math:`\boldsymbol{\phi}_1` are equal. Such margins are described by three
densities:

- :math:`\bar{a}`, the fraction of each individual's :math:`2P` haplotype slots carrying ancestry 1;
- :math:`f_0` and :math:`f_1`, the fractions of the :math:`2N` haplotypes at a locus that carry the alternative allele on an ancestry-0 and an ancestry-1 background respectively,

and where those densities sit in the feasible region
:math:`f_1 \leq \bar{a} \leq 1 - f_0` determines which of four regimes
applies. The returned dictionary reports it.

.. code-block:: python

   # Semi-regular, but not symmetric: abar = 1/2, f0 = f1 = 1/4
   >>> res = zagar.getA12(N=4, a_vec=[4]*4, phi0_vec=[2]*4, phi1_vec=[2]*4, return_approx=True)
   >>> res['count']
   14755332110760
   >>> res['is_regular'], res['case'], res['densities']
   (False, 1, (0.5, 0.25, 0.25))

   # How good is it here?
   >>> import math
   >>> round(res['log2_approx'] - math.log2(res['count']), 4)
   0.0107

The densities can be read off a set of constraints directly, and the regime
identified without computing anything:

.. code-block:: python

   >>> zagar.densities_from_constraints(4, [4]*4, [2]*4, [2]*4)
   {'N': 4, 'P': 4, 'abar': 0.5, 'f0': 0.25, 'f1': 0.25}
   >>> zagar.classify_case(0.5, 0.25, 0.25)
   1

That error is :math:`+0.011` bits here, and along this line of densities
(:math:`\bar{a} = 1/2`, :math:`f_0 = f_1 = 1/4`) it stays positive and shrinks
slowly as the arrays grow, reaching :math:`+0.008` bits by :math:`N=P=11`. The
sign is not a general property, though: elsewhere in the feasible region the
approximation undershoots instead. :doc:`asymptotics_ex` shows where. It is an
asymptotic result in any case, so a :exc:`RuntimeWarning` accompanies every
approximate value to say so.

Margins that are not semi-regular fall outside the approximation formula. For those,
``return_approx=True`` gives the exact count alone, with no approximation and
no warning.

Going Further
-------------

The approximation above was quoted for one set of constraints and one regime.
Which regime a margin falls into, how the formula is assembled from the cell
type densities, and how closely it tracks the exact counts across the whole
feasible region are reported on the next page.

.. toctree::
   :maxdepth: 2

   asymptotics_ex
