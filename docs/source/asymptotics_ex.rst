Asymptotics Example
===================

Exact enumeration of :math:`\mathscr{A}_{12}` is expensive, and unavoidably so:
the cost grows roughly tenfold for each unit increase in :math:`N`, which puts
:math:`N = P = 13` or thereabouts beyond practical reach. Past that point the
answer has to come from an asymptotic formula instead.

This page covers that formula --- Theorem 3.1 --- as **zagar** implements it:
what it needs from you, which of its four regimes applies, how to read the
pieces it is assembled from, and how accurate it actually is.

Semi-Regular Margins
--------------------

The formula does not apply to arbitrary constraints. It requires the margins to
be *semi-regular*: every individual has the same global ancestry, and every
locus the same pair of ancestry-specific allele dosages. In symbols,
:math:`\boldsymbol{a} = (a, \ldots, a)`,
:math:`\boldsymbol{\phi}_0 = (\phi_0, \ldots, \phi_0)` and
:math:`\boldsymbol{\phi}_1 = (\phi_1, \ldots, \phi_1)`.

Three numbers then describe the problem completely, in place of three vectors:

.. math::

   \bar{a} = \frac{a}{2P}, \qquad f_0 = \frac{\phi_0}{2N},
   \qquad f_1 = \frac{\phi_1}{2N}.

The denominators differ, and the difference is easy to lose sight of. Each
individual has :math:`2P` haplotype slots across the loci, so :math:`\bar{a}`
is taken over :math:`2P`; each locus is observed on :math:`2N` haplotypes
across the cohort, so the allele densities are taken over :math:`2N`. They
coincide only when :math:`N = P`.

``densities_from_constraints`` performs the conversion, and refuses margins
that are not semi-regular rather than silently averaging them:

.. code-block:: python

   >>> import zagar
   >>> zagar.densities_from_constraints(4, [4]*4, [2]*4, [2]*4)
   {'N': 4, 'P': 4, 'abar': 0.5, 'f0': 0.25, 'f1': 0.25}

   >>> # N != P, so the two denominators come apart: abar over 2P = 8,
   >>> # the allele densities over 2N = 4
   >>> d = zagar.densities_from_constraints(2, [3]*2, [1]*4, [2]*4)
   >>> d['abar'], d['f0'], d['f1']
   (0.375, 0.25, 0.5)

Note that :math:`f_0` and :math:`f_1` are *joint* proportions over all
:math:`2N` haplotypes at a locus, not frequencies conditional on ancestry.
Their sum :math:`f_0 + f_1` is the overall alternative allele frequency there.

Four Regimes
------------

Each of the :math:`2NP` haplotype-locus cells of an admixed array carries a
pair :math:`(A, X) \in \{0,1\}^2`, and the three densities fix how common each
of the four possible pairs is:

.. math::

   \Pr[(1,1)] = f_1, \qquad \Pr[(0,1)] = f_0, \qquad
   \Pr[(1,0)] = \bar{a} - f_1, \qquad \Pr[(0,0)] = 1 - \bar{a} - f_0.

Two consequences follow immediately. First, requiring all four to be
nonnegative gives the feasible region

.. math::

   f_1 \;\leq\; \bar{a} \;\leq\; 1 - f_0,

so the region is not an extra modelling assumption --- it is just the statement
that no cell type can occur a negative number of times. Second, the regimes of
Theorem 3.1 are exactly the places where one of the four types runs out:

.. list-table::
   :header-rows: 1
   :widths: 8 22 22 48

   * - Case
     - Where
     - Empty type
     - What that forces
   * - 1
     - interior
     - none
     - all four types occur; :math:`D = N + 2P`
   * - 2
     - :math:`f_1 = \bar{a}`
     - :math:`(1,0)`
     - :math:`X = 1` wherever :math:`A = 1`, hence
       :math:`\sum_p \phi_{1,p} = \sum_n a_n`; :math:`D = N + 2P - 1`
   * - 3
     - :math:`\bar{a} = 1 - f_0`
     - :math:`(0,0)`
     - :math:`X = 1` wherever :math:`A = 0`, hence
       :math:`\sum_p \phi_{0,p} = 2NP - \sum_n a_n`; :math:`D = N + 2P - 1`
   * - 4
     - both
     - :math:`(1,0)`, :math:`(0,0)`
     - :math:`\mathbf{X} \equiv 1`, so the array is determined by
       :math:`\mathbf{A}` alone; :math:`D = N + P - 1`

Here :math:`D` is the dimension of the constraint space. In Case 1 the
:math:`N` row constraints and :math:`2P` column constraints are independent. In
Cases 2 and 3 the identity in the last column makes one of them redundant, so
:math:`D` drops by one. In Case 4, :math:`\mathbf{X} \equiv 1` means
:math:`\phi_{0,p} + \phi_{1,p} = 2N` at every locus, so the two column
constraints per locus collapse into one, leaving :math:`P` of them rather than
:math:`2P`, and the row-sum identity costs one more.

``classify_case`` reports the regime without computing anything:

.. code-block:: python

   >>> zagar.classify_case(0.5, 0.25, 0.25)   # interior
   1
   >>> zagar.classify_case(0.5, 0.25, 0.5)    # f1 == abar
   2
   >>> zagar.classify_case(0.5, 0.5, 0.25)    # abar == 1 - f0
   3
   >>> zagar.classify_case(0.5, 0.5, 0.5)     # both
   4

Densities outside the feasible region are rejected:

.. code-block:: python

   >>> zagar.classify_case(0.3, 0.3, 0.5)     # f1 > abar
   Traceback (most recent call last):
       ...
   ValueError: f1 = 0.5 > abar = 0.3; |A12| = 0

The boundaries are tested with a small tolerance, and that is a mathematical
decision rather than a defensive one. The Case 1 correction term behaves like
:math:`-1/\bigl(24NP(\bar{a} - f_1)\bigr)` as :math:`f_1 \to \bar{a}`, so it
does not merely lose accuracy near the boundary --- it diverges. The boundary
formula is the correct one there, so a point a hair inside Case 1 is treated as
Case 2.

The Shape of the Formula
------------------------

In every regime the answer has the same five-part form:

.. math::

   \log_2 |\mathscr{A}_{12}| \;=\; 2NP\,H
     \;-\; \tfrac{1}{2}\log_2 \Delta
     \;-\; \tfrac{D}{2}\log_2 \pi
     \;-\; D
     \;+\; K \log_2 e .

The leading term is the entropy count: :math:`H` is the Shannon entropy in bits
of the cell-type distribution above, and :math:`2NP\,H` is how many bits it
takes to describe an array whose cells follow it. Everything after that is the
correction for the constraints actually holding exactly rather than on average.
:math:`\Delta` is the determinant arising at the saddle point, :math:`D` the
dimension counted in the table, and :math:`K` the leading finite-size
correction.

Passing ``details=True`` returns all five pieces, so you can see which part of
the expansion a change in the densities actually moves:

.. code-block:: python

   >>> d = zagar.log2_A12_asymptotic(10, 10, 0.5, 0.25, 0.25, details=True)
   >>> d['case'], d['dimension']
   (1, 30)
   >>> d['entropy']                 # H(1/4, 1/4, 1/4, 1/4) = 2 bits
   2.0
   >>> round(d['log2_A12'], 4)
   330.0294

A Worked Example in Each Regime
-------------------------------

Holding :math:`N = P = 10` and :math:`\bar{a} = 1/2` fixed and varying only the
allele densities walks through all four regimes in turn:

.. list-table::
   :header-rows: 1
   :widths: 10 10 8 10 14 14 18

   * - :math:`f_0`
     - :math:`f_1`
     - Case
     - :math:`D`
     - :math:`H`
     - :math:`K`
     - :math:`\log_2 |\mathscr{A}_{12}|`
   * - 0.25
     - 0.25
     - 1
     - 30
     - 2.000
     - -0.602
     - 330.029
   * - 0.25
     - 0.50
     - 2
     - 29
     - 1.500
     - -0.701
     - 233.535
   * - 0.50
     - 0.25
     - 3
     - 29
     - 1.500
     - -0.701
     - 233.535
   * - 0.50
     - 0.50
     - 4
     - 19
     - 1.000
     - -0.451
     - 153.762

Reading across: as cell types are forced out, the entropy falls from 2 bits to
1, and the dimension falls from 30 to 19. The leading term :math:`2NP\,H`
shrinks accordingly, which is why the counts differ by so much.

The Cases 2 and 3 rows are identical, and not by coincidence. Replacing
:math:`\mathbf{A}` by :math:`1 - \mathbf{A}` is a bijection on admixed arrays:
it sends row sums :math:`a_n` to :math:`2P - a_n` and exchanges the cell types
:math:`(1,1) \leftrightarrow (0,1)` and
:math:`(1,0) \leftrightarrow (0,0)`, so it maps
:math:`(\bar{a}, f_0, f_1)` to :math:`(1 - \bar{a}, f_1, f_0)` while leaving
:math:`|\mathscr{A}_{12}|` unchanged. The two rows above are images of one
another under it:

.. code-block:: python

   >>> x = zagar.log2_A12_asymptotic(10, 10, 0.5, 0.25, 0.5)
   >>> y = zagar.log2_A12_asymptotic(10, 10, 0.5, 0.5, 0.25)
   >>> x == y
   True

How Accurate Is It?
-------------------

Asymptotic results are only useful if you know when to trust them, so the
comparison against exact enumeration is worth doing explicitly. Each row below
enumerates :math:`|\mathscr{A}_{12}|` exactly and compares.

.. list-table::
   :header-rows: 1
   :widths: 8 8 10 12 12 18 18 14

   * - :math:`N`
     - :math:`P`
     - :math:`\bar{a}`
     - :math:`f_0`
     - :math:`f_1`
     - Exact (bits)
     - Asymptotic
     - Error
   * - 4
     - 4
     - 0.5
     - 0.25
     - 0.25
     - 43.7463
     - 43.7570
     - +0.0107
   * - 6
     - 6
     - 0.5
     - 0.25
     - 0.25
     - 108.5082
     - 108.5203
     - +0.0121
   * - 8
     - 8
     - 0.5
     - 0.25
     - 0.25
     - 203.8089
     - 203.8198
     - +0.0109
   * - 10
     - 10
     - 0.5
     - 0.25
     - 0.25
     - 330.0197
     - 330.0294
     - +0.0097
   * - 11
     - 11
     - 0.5
     - 5/22
     - 5/22
     - 404.0798
     - 404.0882
     - +0.0084
   * - 4
     - 4
     - 0.5
     - 0.25
     - 0.5
     - 29.9929
     - 30.0038
     - +0.0109
   * - 8
     - 8
     - 0.5
     - 0.25
     - 0.5
     - 142.9976
     - 143.0131
     - +0.0155
   * - 8
     - 8
     - 0.5
     - 0.5
     - 0.25
     - 142.9976
     - 143.0131
     - +0.0155
   * - 4
     - 4
     - 0.25
     - 0.75
     - 0.25
     - 14.8899
     - 14.9257
     - +0.0358
   * - 12
     - 12
     - 0.25
     - 0.75
     - 0.25
     - 179.1586
     - 179.1720
     - +0.0134

Three things are worth drawing out.

The errors are **tiny in relative terms**. At :math:`N = P = 10` the
approximation is wrong by 0.0097 bits in a count of 330 bits --- a relative
error of :math:`3 \times 10^{-5}`, on a number with 100 decimal digits.

They are **consistently positive**. The expansion always overestimates at
finite size, across every regime and every size tested. That sign is a property
of the derivation, not an accident of these particular points, and the test
suite asserts it rather than merely bounding the magnitude.

They **shrink as the arrays grow**, which is what makes it an asymptotic
result: :math:`+0.0107` bits at :math:`N = P = 4` falling to :math:`+0.0084`
at :math:`N = P = 11`. The worst row in the table,
:math:`+0.0358` at :math:`(4, 4, 0.25, 0.75, 0.25)`, is also the smallest
array in it.

Note again that the two :math:`(8, 8)` rows in Cases 2 and 3 agree exactly, on
the exact counts as well as the approximations. The complement bijection is
visible in the enumeration, not just in the formula.

Choosing Between Exact and Asymptotic
-------------------------------------

The practical question is which to reach for. The exact routines are the right
answer whenever they finish, and the table below says when that is, measured at
the hardest parameter point --- :math:`\bar{a} = 1/2`, :math:`f_0 = f_1 = 1/4`,
which maximises the number of reachable states:

.. list-table::
   :header-rows: 1
   :widths: 14 30 28 28

   * - :math:`N = P`
     - ``count_A12`` (default)
     - ``method="miller-harrison"``
     - ``log2_A12_asymptotic``
   * - 4
     - 0.000 s
     - 0.014 s
     - instant
   * - 6
     - 0.004 s
     - exhausts 16 GB
     - instant
   * - 8
     - 0.44 s
     - --
     - instant
   * - 10
     - 45 s
     - --
     - instant
   * - 11
     - 7.8 min
     - --
     - instant
   * - 13
     - about 12 h
     - --
     - instant

So in round terms: up to :math:`N = P \approx 10` the exact count is a
reasonable thing to ask for interactively; beyond :math:`N = P \approx 13` it
is not available at any price, and the asymptotic value is the only answer
there is. In the overlap, computing both is a cheap way to confirm the
approximation is behaving as expected on your parameters.

The asymptotic value takes microseconds regardless of :math:`N` and
:math:`P` --- it is a closed-form expression, so the sizes enter only as
numbers:

.. code-block:: python

   >>> round(zagar.log2_A12_asymptotic(10**6, 10**6, 0.5, 0.25, 0.25), 2)
   3999968125402.55

A Caution on Representability
-----------------------------

A density is only realisable if it corresponds to whole numbers of haplotypes.
At :math:`N = 11`, for instance, :math:`2N f_0 = 5.5` when :math:`f_0 = 1/4`,
so a quarter density simply cannot be expressed at that size --- which is why
the :math:`N = P = 11` row of the accuracy table uses :math:`5/22` instead.

Building constraint vectors by rounding is therefore a reliable way to compute
at a parameter point other than the one you intended, and Python's
round-half-to-even makes it worse rather than better: ``round(4.5)`` is 4 and
``round(5.5)`` is 6. Round-tripping through ``densities_from_constraints`` is a
cheap way to catch it:

.. code-block:: python

   >>> # Intending f0 = f1 = 1/4 at N = P = 11
   >>> phi = round(2 * 11 * 0.25)
   >>> phi
   6
   >>> d = zagar.densities_from_constraints(11, [11]*11, [phi]*11, [phi]*11)
   >>> d['f0'] == 0.25                      # not the point we asked for
   False
   >>> round(d['f0'], 4)
   0.2727
