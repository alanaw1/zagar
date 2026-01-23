R Example
=========

Basics
------

To make sure R works, try to generate a binary matrix (which we refer to as :math:`\mathbf{X}`).

.. code-block:: R

   # Set seed for reproducibility
   > set.seed(42)

   # Define matrix dimensions
   > rows <- 6
   > cols <- 8

   # Generate random binary matrix
   > X <- do.call("cbind",
                lapply(rep(0.5,cols), function(x) {rbinom(rows,1,x)}))

   > print(X)
            [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8]
    [1,]    1    1    1    0    0    1    0    0
    [2,]    1    0    0    1    1    1    0    1
    [3,]    0    1    0    1    0    0    1    0
    [4,]    1    1    1    0    1    1    1    1
    [5,]    1    0    1    1    0    0    0    1
    [6,]    1    1    0    1    1    1    0    1

   > dim(X)
   [1] 6 8

If the above works, we are ready to proceed with performing combinatorial calculations.
First, load **zagar**. (If you have not installed it, go to the `Installation page <./installation.html>`__ on
this website and follow instructions there.)

.. code-block:: R

   > library(zagar)

Binary matrices form a class of rich and practical discrete structures.

- *Rich*, because they lead to interesting mathematical theory (see e.g., `Barvinok 2010 <https://doi.org/10.1016/j.aim.2009.12.001>`__, `Cooper et al. 2019 <https://doi.org/10.37236/8092>`__, `Liebenau and Wormald 2022 <https://doi.org/10.1002/rsa.21105>`__ and many others not listed);
- *Practical*, because they represent many data objects encountered in reality (e.g., contingency tables in statistics, genetic polymorphism data, codes for information transmission).

With **zagar**, we can investigate the following combinatorial question:

   Given marginal constraints, like the row and column sums, of a binary matrix, how many such matrices
   are there that satisfy them?

The answer to this question is challenging in general. But for reasonably sized matrix dimensions, the function ``countBinaryMatrices`` calculates the answer. Let us use the binary matrix :math:`\mathbf{X}` generated earlier to demonstrate its usage.

.. code-block:: R

   # Compute row and column margins
   > row_margins <- rowSums(X)
   > col_margins <- colSums(X)

   # Check
   >>> print(row_margins)
   [1] 4 5 3 7 4 6
   >>> print(col_margins)
   [1] 5 4 3 4 3 4 2 4

   # Count number of binary matrices with observed marginal constraints
   > countBinaryMatrices(p=row_margins, q=col_margins, sort_p_desc=TRUE)
    Enumerating using EXACT algorithm...
    No. memo entries = 43
    [1] "549278"    

There are :math:`549,278` such binary matrices --- a huge count!

Admixed Arrays
--------------

Closely related to a binary matrix is an admixed array. An *admixed array* is a pair :math:`[\mathbf{A},\mathbf{X}]`, where :math:`\mathbf{A}=(A_{np})` and :math:`\mathbf{X}=(X_{np})` lie in :math:`\{0,1\}^{N\times 2P}`. Here, :math:`N` and :math:`P` are positive integers providing the dimensionality of the underlying binary matrices; they correspond to the number of individuals and the number of genetic loci in an admixed population dataset.

Let us generate a random admixed array.

.. code-block:: R

   # Define array parameters
   > N <- 6
   > P <- 5

   # Generate random admixed array
   > X <- matrix(sample(0:1, size=N*2*P, replace=TRUE), nrow=N, ncol=2*P)
   > A <- matrix(sample(0:1, size=N*2*P, replace=TRUE), nrow=N, ncol=2*P)

   > print(A)
         [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
    [1,]    1    1    1    1    0    0    1    1    0     0
    [2,]    1    0    1    0    1    0    1    1    0     0
    [3,]    1    1    0    1    0    0    1    1    0     1
    [4,]    1    0    0    1    0    1    1    0    1     1
    [5,]    1    0    1    1    1    1    1    1    0     1
    [6,]    1    1    1    1    0    0    0    0    0     1
   
   > print(X)
         [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
    [1,]    1    1    1    0    1    1    0    1    1     1
    [2,]    1    1    0    0    0    1    0    1    1     0
    [3,]    1    0    0    1    1    0    0    1    0     1
    [4,]    1    1    0    1    1    1    0    1    1     0
    [5,]    1    1    0    1    0    1    1    1    1     0
    [6,]    0    0    0    1    0    0    1    0    0     1

The following two summary statistics (modulo normalization factors that are irrelevant here) are studied in genetic studies of admixed populations.

- *Individual global ancestry*: For each individual :math:`n\in[N]`, their global ancestry dosage is :math:`A_{n\cdot}=\sum_{p=1}^{2P}A_{np}`. Equivalently, by gathering the values across all individuals we obtain a vector of individual global ancestry dosages called the *ancestry dosage vector*, :math:`\boldsymbol{A}=(A_{1\cdot},\ldots,A_{N\cdot})`.
- *Ancestry-specific allele frequencies*: For each locus :math:`p\in[P]`, the pair :math:`(\Phi_{p,0},\Phi_{p,1})` records the allele dosage specific to each ancestry. Mathematically, :math:`\Phi_{p,0}=\sum_{n=1}^N \left[(1-A_{np})X_{np}+(1-A_{n(P+p)})X_{n(P+p)}\right]` and :math:`\Phi_{p,1}=\sum_{n=1}^N \left(A_{np}X_{np}+A_{n(P+p)}X_{n(P+p)}\right)`. Again, by gathering the values across all loci we obtain two *ancestry-specific allele dosage vectors*: :math:`\boldsymbol{\Phi}_0=(\Phi_{1,0},\ldots,\Phi_{P,0})` and :math:`\boldsymbol{\Phi}_1=(\Phi_{1,1},\ldots,\Phi_{P,1})`.

We compute these summaries for the admixed array generated above.

.. code-block:: R

   # Individual global ancestry (row sums of A)
   > row_sum_A = rowSums(A)
   > print(row_sum_A)
   [1] 6 5 6 6 8 5

   # Ancestry-specific allele frequencies
   > phi0 <- c()
   > phi1 <- c()
   > for (p in 1:P) {
   +   s0 <- 0
   +   s1 <- 0
   +   for (n in 1:N) {
   +     a1 <- A[n, p]
   +     a2 <- A[n, P + p]
   +     x1 <- X[n, p]
   +     x2 <- X[n, P + p]
   +
   +     s0 <- s0 + (1 - a1) * x1 + (1 - a2) * x2
   +     s1 <- s1 + a1 * x1 + a2 * x2
   +   }
   +   phi0[p] <- s0
   +   phi1[p] <- s1
   + }

   > phi0
   [1] 2 4 1 3 4
   > phi1
   [1] 7 2 5 5 2

   # Optional: sanity checks
   # 1) row sum constraints must sum to no. 1's in A
   # 2) column sum constraints must sum to no. 1's in X
   > sum(row_sum_A) == sum(A)
   [1] TRUE
   > (sum(phi0) + sum(phi1)) == sum(X)
   [1] TRUE

The code above was written to help reinforce understanding of how these summaries are calculated from data. These summaries can also be obtained with a single call to the function ``getConstraints``:

.. code-block:: R

   # Compute phi0, phi1 and row_sum_A
   > constraint_list <- getConstraints(A, X)
   > constraint_list$a_vec # row_sum_A
   [1] 6 5 6 6 8 5 
   > constraint_list$phi0_vec # phi0
   [1] 2 4 1 3 4
   > constraint_list$phi1_vec # phi1
   [1] 7 2 5 5 2


Now assume that these summaries are viewed as constraints. We can ask an analogous combinatorial question of our admixed array.

   Given the marginal constraints (on individual global ancestry and ancestry-specific allele frequencies) of an admixed array,
   how many admixed arrays are there that satisfy them?

We concretely define three sets of constrained admixed arrays.

- *Individual global ancestry constraint*: :math:`\mathscr{A}_{1}=\{[\mathbf{A},\mathbf{X}]:~(\forall n \in [N])(A_{n\cdot}=a_{n\cdot})\}`. Equivalently, :math:`\mathscr{A}_{1}` denotes the set of admixed arrays for which :math:`\boldsymbol{A}=\boldsymbol{a}`, for some fixed non-negative integer :math:`N`-vector :math:`\boldsymbol{a}=(a_{1\cdot},\ldots,a_{N\cdot})`.
- *Ancestry-specific allele frequency constraint*: :math:`\mathscr{A}_{2}=\{[\mathbf{A},\mathbf{X}]:~(\forall p\in[P])(\Phi_{p,0}=\phi_{p,0} \wedge \Phi_{p,1}=\phi_{p,1})\}`. Equivalently, :math:`\mathscr{A}_{2}` denotes the set of admixed arrays for which :math:`\boldsymbol{\Phi}_0=\boldsymbol{\phi}_0` and :math:`\boldsymbol{\Phi}_1=\boldsymbol{\phi}_1`, for some fixed non-negative integer :math:`P`-vectors :math:`\boldsymbol{\phi}_0=(\phi_{1,0},\ldots,\phi_{P,0})` and :math:`\boldsymbol{\phi}_1=(\phi_{1,1},\ldots,\phi_{P,1})`.
- *Both constraints*: :math:`\mathscr{A}_{12}` denotes the set of admixed arrays for which :math:`\boldsymbol{A}=\boldsymbol{a}`, :math:`\boldsymbol{\Phi}_0=\boldsymbol{\phi}_0` and :math:`\boldsymbol{\Phi}_1=\boldsymbol{\phi}_1`.

With **zagar**, we can compute the sizes of these constrained sets of arrays using ``getA1``, ``getA2`` and ``getA12``. 

.. code-block:: R

   # Size of A1
   > getA1(N=N, P=P, a_vec=row_sum_A)
   Big Integer ('bigz') :
   [1] 30512038196863407653105172480000

   # Size of A2
   > getA2(N=N, P=P, phi0_vec=phi0, phi1_vec=phi1)
   Big Integer ('bigz') :
   [1] 7845452210162977021624320000
   
   # Size of A12
   > getA12(N=N, P=P, a_vec=row_sum_A, phi0_vec=phi0, phi1_vec=phi1)
   Big Integer ('bigz') :
   [1] 574155698908329217448832

Saddle-Point Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us consider a regular admixed array, defined by :math:`\boldsymbol{a}=(P,...,P)` and :math:`\boldsymbol{\phi}_0=\boldsymbol{\phi}_1=(N,...,N)`. In this constrained case, a `saddle-point approximation <https://doi.org/10.2307/2686100>`__ to a `Cauchy integral <https://mathworld.wolfram.com/CauchyIntegralFormula.html>`__ can be used to estimate the size of :math:`\mathscr{A}_{12}`:

:math:`\log_2(|\mathscr{A}_{12}|)\approx 2NP - \frac{1}{2}[N\log_2(\pi P) + P\log_2(\pi N)-\log_2(\pi NP)] - \log_2(e)\cdot(N+P-1)^2/(8NP)`

We demonstrate this functionality in **zagar**.

.. code-block:: R

   # Array parameters
   > N <- 9
   > P <- 3

   # Define regular constraints
   > a_reg <- rep(P, N)       # length N, all entries = P
   > phi0_reg <- rep(N, P)    # length P, all entries = N
   > phi1_reg <- rep(N, P)    # length P, all entries = N

   # Compute size of A12 and report the approximation and upper bound
   > res <- getA12(N=N, P=P, 
                   a_vec=a_reg, phi0_vec=phi0_reg, 
                   phi1_vec=phi1_reg, return_approx=TRUE)
   Computing estimated log-counts based on saddle-point approximation.
   Note: log2_approx is good only when P and N are large with bounded ratio
   > print(res)
   $count
   Big Integer ('bigz') :
   [1] 25697841152

   $is_regular
   [1] TRUE

   $log2_upper_bound
   [1] 35.407

   $log2_approx
   [1] 34.59882

   # Saddle-point approximation
   > print(res$log2_approx)
   [1] 34.59882

   # Upper bound
   > print(res$log2_upper_bound)
   [1] 35.407

   # Compute log2(A12)
   > log2(25697841152)
   [1] 34.58093  # well-approximated by saddle-point approximation

For :math:`(N,P)=(9,3)`, there are :math:`|\mathscr{A}_{12}|=25,697,841,152` (more than 25 billion!) regular admixed arrays. This is equivalent to about :math:`\log_2(|\mathscr{A}_{12}|)=34.6` bits, which is well-approximated by the saddle-point formula.

