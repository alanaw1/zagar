"""Tests for the zagar package."""

import warnings

import pytest
import numpy as np
import zagar


class TestCountBinaryMatrices:
    """Tests for count_binary_matrices function."""

    def test_simple_case(self):
        """Test a simple 2x3 matrix case."""
        # 2x3 matrix with row sums [2,1] and col sums [1,1,1]
        # The 3 valid matrices are:
        # [[1,1,0], [0,0,1]], [[1,0,1], [0,1,0]], [[0,1,1], [1,0,0]]
        result = zagar.count_binary_matrices([2, 1], [1, 1, 1])
        assert result["number"] == 3
        assert result["memo_entries"] > 0

    def test_convenience_wrapper(self):
        """Test the count_matrices convenience function."""
        count = zagar.count_matrices([2, 1], [1, 1, 1])
        assert count == 3

    def test_single_row(self):
        """Test with a single row."""
        result = zagar.count_binary_matrices([3], [1, 1, 1])
        assert result["number"] == 1

    def test_single_column(self):
        """Test with a single column."""
        result = zagar.count_binary_matrices([1, 1, 1], [3])
        assert result["number"] == 1

    def test_identity_like(self):
        """Test identity-like matrix counting."""
        # 3x3 with row sums [1,1,1] and col sums [1,1,1]
        # This counts permutation matrices = 3! = 6
        result = zagar.count_binary_matrices([1, 1, 1], [1, 1, 1])
        assert result["number"] == 6

    def test_mismatched_sums(self):
        """Test that mismatched sums return 0."""
        result = zagar.count_binary_matrices([1, 2], [1, 1])
        assert result["number"] == 0

    def test_empty_input_raises(self):
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError):
            zagar.count_binary_matrices([], [1, 2])
        with pytest.raises(ValueError):
            zagar.count_binary_matrices([1, 2], [])

    def test_negative_values_raises(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError):
            zagar.count_binary_matrices([-1, 2], [1, 0])
        with pytest.raises(ValueError):
            zagar.count_binary_matrices([1, 2], [1, -1, 1])

    def test_sort_p_desc_option(self):
        """Test that sort_p_desc gives same result."""
        p = [1, 3, 2]
        q = [2, 2, 2]
        result_unsorted = zagar.count_binary_matrices(p, q, sort_p_desc=False)
        result_sorted = zagar.count_binary_matrices(p, q, sort_p_desc=True)
        assert result_unsorted["number"] == result_sorted["number"]

    def test_larger_matrix(self):
        """Test a larger matrix to verify arbitrary precision."""
        # 5x5 with all row/col sums = 3
        p = [3] * 5
        q = [3] * 5
        result = zagar.count_binary_matrices(p, q, sort_p_desc=True)
        assert result["number"] > 0
        assert isinstance(result["number"], int)


class TestGetA12:
    """Tests for getA12 function."""

    def test_basic(self):
        """Test basic A12 computation."""
        result = zagar.getA12(N=5, a_vec=[2, 2], phi0_vec=[1, 1], phi1_vec=[1, 1])
        assert isinstance(result, int)
        assert result >= 0

    def test_invalid_N_raises(self):
        """Test that invalid N raises ValueError."""
        with pytest.raises(ValueError):
            zagar.getA12(N=0, a_vec=[2], phi0_vec=[1], phi1_vec=[1])
        with pytest.raises(ValueError):
            zagar.getA12(N=-1, a_vec=[2], phi0_vec=[1], phi1_vec=[1])

    def test_mismatched_phi_lengths_raises(self):
        """Test that mismatched phi0/phi1 lengths raise ValueError."""
        with pytest.raises(ValueError):
            zagar.getA12(N=5, a_vec=[2, 2], phi0_vec=[1, 1], phi1_vec=[1])

    def test_infeasible_returns_zero(self):
        """Test that infeasible constraints return 0."""
        # phi1 > 2*N - phi0 is infeasible
        result = zagar.getA12(N=2, a_vec=[2], phi0_vec=[0], phi1_vec=[10])
        assert result == 0


class TestGetA12Parallel:
    """Tests for getA12_parallel function."""

    def test_matches_serial(self):
        """Test that parallel result matches serial."""
        N, a_vec, phi0_vec, phi1_vec = 5, [2, 2], [1, 1], [1, 1]
        serial = zagar.getA12(N, a_vec, phi0_vec, phi1_vec)
        parallel = zagar.getA12_parallel(N, a_vec, phi0_vec, phi1_vec)
        assert serial == parallel

    def test_explicit_threads(self):
        """Test with explicit thread count."""
        result = zagar.getA12_parallel(
            N=5, a_vec=[2, 2], phi0_vec=[1, 1], phi1_vec=[1, 1], n_threads=2
        )
        assert isinstance(result, int)

    def test_single_thread(self):
        """Test with single thread (should work like serial)."""
        N, a_vec, phi0_vec, phi1_vec = 5, [2, 2], [1, 1], [1, 1]
        serial = zagar.getA12(N, a_vec, phi0_vec, phi1_vec)
        single = zagar.getA12_parallel(N, a_vec, phi0_vec, phi1_vec, n_threads=1)
        assert serial == single


class TestGetA12Pruning:
    """Tests for getA12_pruning, which had no dedicated tests before 0.2.0.

    It was exported and documented but only ever exercised indirectly, so a
    regression in the pruning or symmetry logic would not have been caught.
    """

    def test_matches_serial(self):
        N, a_vec, phi0_vec, phi1_vec = 5, [2, 2], [1, 1], [1, 1]
        serial = zagar.getA12(N, a_vec, phi0_vec, phi1_vec)
        pruned = zagar.getA12_pruning(N, a_vec, phi0_vec, phi1_vec)
        assert serial == pruned

    def test_explicit_threads(self):
        result = zagar.getA12_pruning(
            N=5, a_vec=[2, 2], phi0_vec=[1, 1], phi1_vec=[1, 1], n_threads=2
        )
        assert isinstance(result, int)

    def test_single_thread(self):
        N, a_vec, phi0_vec, phi1_vec = 5, [2, 2], [1, 1], [1, 1]
        serial = zagar.getA12(N, a_vec, phi0_vec, phi1_vec)
        single = zagar.getA12_pruning(N, a_vec, phi0_vec, phi1_vec, n_threads=1)
        assert serial == single

    def test_n4_p4(self):
        N = P = 4
        assert zagar.getA12_pruning(
            N=N, a_vec=[P] * N, phi0_vec=[N] * P, phi1_vec=[N] * P
        ) == 824410

    def test_infeasible_returns_zero(self):
        assert zagar.getA12_pruning(2, [0, 0], [0], [3]) == 0


class TestModuleAttributes:
    """Tests for module-level attributes."""

    def test_version(self):
        """Test that version is defined."""
        assert hasattr(zagar, "__version__")
        assert isinstance(zagar.__version__, str)

    def test_all_exports(self):
        """Test that __all__ contains expected functions."""
        expected = [
            "count_binary_matrices", "getA12", "getA12_parallel",
            "getA1", "getA2", "compareA1A2", "get_constraints", "count_matrices", "has_openmp"
        ]
        for name in expected:
            assert name in zagar.__all__
            assert hasattr(zagar, name)

    def test_has_openmp(self):
        """Test that has_openmp returns a boolean."""
        result = zagar.has_openmp()
        assert isinstance(result, bool)


class TestAccuracyCountBinaryMatrices:
    """Accuracy tests from R unit tests for count_binary_matrices."""

    def test_darwins_finches(self):
        """Test Darwin's finches dataset."""
        p = [14, 13, 14, 10, 12, 2, 10, 1, 10, 11, 6, 2, 17]
        q = [4, 4, 11, 10, 10, 8, 9, 10, 8, 9, 3, 10, 4, 7, 9, 3, 3]
        result = zagar.count_binary_matrices(p, q, sort_p_desc=True)
        assert result["number"] == 67149106137567626

    def test_manlys_lizards(self):
        """Test Manly's lizards dataset."""
        p = [7, 23, 6, 6, 2, 18, 8, 8, 1, 22, 2, 2, 9, 2, 1, 18, 9, 3, 3, 1]
        q = [13, 4, 9, 4, 3, 2, 3, 4, 4, 5, 2, 5, 2, 10, 10, 10, 7, 6, 6, 3, 3, 11, 8, 11, 6]
        result = zagar.count_binary_matrices(p, q, sort_p_desc=True)
        assert result["number"] == 55838420515731001979319625577023858901579264

    def test_galaskiewicz_ceos_and_clubs(self):
        """Test Galaskiewicz's CEOs and clubs dataset."""
        p = [3, 11, 22, 12, 3, 4, 4, 4, 6, 3, 4, 5, 5, 3, 9]
        q = [3, 3, 2, 3, 3, 3, 4, 3, 4, 2, 3, 2, 4, 7, 5, 5, 6, 5, 5, 5, 3, 3, 4, 5, 3, 3]
        result = zagar.count_binary_matrices(p, q, sort_p_desc=True)
        assert result["number"] == 25533540876226059861329182058955286218365274646344655


class TestAccuracyGetA12:
    """Accuracy tests from R unit tests for getA12."""

    def test_n4_p4(self):
        """Test N=P=4 parameterization."""
        N = 4
        P = 4
        result = zagar.getA12(
            N=N,
            a_vec=[P] * N,
            phi0_vec=[N] * P,
            phi1_vec=[N] * P
        )
        assert result == 824410

    def test_n6_p6(self):
        """Test N=P=6 parameterization."""
        N = 6
        P = 6
        result = zagar.getA12(
            N=N,
            a_vec=[P] * N,
            phi0_vec=[N] * P,
            phi1_vec=[N] * P
        )
        assert result == 726090271230576


class TestAccuracyGetA12Parallel:
    """Accuracy tests from R unit tests for getA12_parallel."""

    def test_n4_p4_parallel(self):
        """Test N=P=4 parameterization with parallel."""
        N = 4
        P = 4
        result = zagar.getA12_parallel(
            N=N,
            a_vec=[P] * N,
            phi0_vec=[N] * P,
            phi1_vec=[N] * P
        )
        assert result == 824410

    def test_n6_p6_parallel(self):
        """Test N=P=6 parameterization with parallel."""
        N = 6
        P = 6
        result = zagar.getA12_parallel(
            N=N,
            a_vec=[P] * N,
            phi0_vec=[N] * P,
            phi1_vec=[N] * P
        )
        assert result == 726090271230576

    def test_n7_p7_parallel(self):
        """Test N=P=7 parameterization with parallel."""
        N = 7
        P = 7
        result = zagar.getA12_parallel(
            N=N,
            a_vec=[P] * N,
            phi0_vec=[N] * P,
            phi1_vec=[N] * P
        )
        assert result == 1014691016274501451776


class TestGetA1:
    """Tests for getA1 function."""

    def test_basic_small(self):
        """Test a small case with known value."""
        # N=2, P=2, a_vec=[2,2]
        # Formula: C(4,2)^2 * 2^8 = 6^2 * 256 = 36 * 256 = 9216
        result = zagar.getA1(N=2, P=2, a_vec=[2, 2])
        assert result == 9216

    def test_return_log(self):
        """Test log2 return mode."""
        import math
        result = zagar.getA1(N=2, P=2, a_vec=[2, 2], return_log=True)
        expected = math.log2(9216)
        assert abs(result - expected) < 0.01

    def test_arbitrary_precision(self):
        """Test with large values requiring arbitrary precision."""
        result = zagar.getA1(N=50, P=50, a_vec=[25]*50)
        assert isinstance(result, int)
        assert result > 0

    def test_invalid_N_raises(self):
        """Test that invalid N raises ValueError."""
        with pytest.raises(ValueError):
            zagar.getA1(N=0, P=2, a_vec=[1, 1])

    def test_mismatched_a_vec_length_raises(self):
        """Test that a_vec length must equal N."""
        with pytest.raises(ValueError):
            zagar.getA1(N=3, P=2, a_vec=[1, 1])


class TestGetA2:
    """Tests for getA2 function."""

    def test_basic_small(self):
        """Test a small case."""
        # N=2, P=2, phi0_vec=[1,1], phi1_vec=[1,1]
        # For each j: C(4,2)*C(2,1)*2^(4-2) = 6*2*4 = 48
        # Product for P=2: 48^2 = 2304
        result = zagar.getA2(N=2, P=2, phi0_vec=[1, 1], phi1_vec=[1, 1])
        assert result == 2304

    def test_return_log(self):
        """Test log2 return mode."""
        import math
        result = zagar.getA2(N=2, P=2, phi0_vec=[1, 1], phi1_vec=[1, 1], return_log=True)
        expected = math.log2(2304)
        assert abs(result - expected) < 0.01

    def test_arbitrary_precision(self):
        """Test with large values requiring arbitrary precision."""
        result = zagar.getA2(N=50, P=50, phi0_vec=[20]*50, phi1_vec=[22]*50)
        assert isinstance(result, int)
        assert result > 0

    def test_invalid_N_raises(self):
        """Test that invalid N raises ValueError."""
        with pytest.raises(ValueError):
            zagar.getA2(N=0, P=2, phi0_vec=[1, 1], phi1_vec=[1, 1])

    def test_phi_out_of_range_raises(self):
        """Test that phi values must satisfy constraints."""
        with pytest.raises(ValueError):
            zagar.getA2(N=2, P=2, phi0_vec=[3, 1], phi1_vec=[3, 1])


class TestCompareA1A2:
    """Tests for compareA1A2 function."""

    def test_basic_comparison(self):
        """Test basic comparison returns expected keys."""
        result = zagar.compareA1A2([2, 2], [1, 1], [1, 1])
        assert 'log2_A1' in result
        assert 'log2_A2' in result
        assert 'outcome' in result

    def test_with_large_NP(self):
        """Test comparison with asymptotic analysis."""
        result = zagar.compareA1A2([25]*50, [20]*50, [22]*50, use_large_NP=True)
        assert 'large_NP_stat' in result
        assert 'large_NP_outcome' in result

    def test_mismatched_phi_lengths_raises(self):
        """Test that phi0 and phi1 must have same length."""
        with pytest.raises(ValueError):
            zagar.compareA1A2([1, 1], [1, 1], [1])


class TestGetA12Approximation:
    """Tests for getA12 return_approx feature."""

    def test_regular_case_detected(self):
        """Test that regular case is correctly detected."""
        N, P = 4, 4
        with pytest.warns(RuntimeWarning, match="asymptotic"):
            result = zagar.getA12(
                N=N,
                a_vec=[P] * N,
                phi0_vec=[N] * P,
                phi1_vec=[N] * P,
                return_approx=True
            )
        assert result["is_regular"] is True
        assert "count" in result
        assert "log2_upper_bound" in result
        assert "log2_approx" in result
        # New in 0.2.0: the symmetric point is Case 4 of Theorem 3.1.
        assert result["case"] == 4

    def test_non_regular_case_detected(self):
        """Test that non-regular case is correctly detected."""
        result = zagar.getA12(
            N=5,
            a_vec=[2, 2],
            phi0_vec=[1, 1],
            phi1_vec=[1, 1],
            return_approx=True
        )
        assert result["is_regular"] is False
        assert "count" in result
        assert "log2_upper_bound" not in result
        assert "log2_approx" not in result

    def test_return_approx_false_returns_int(self):
        """Test that return_approx=False returns int (default behavior)."""
        result = zagar.getA12(N=4, a_vec=[4]*4, phi0_vec=[4]*4, phi1_vec=[4]*4)
        assert isinstance(result, int)
        assert result == 824410

    def test_approximation_values_reasonable(self):
        """Test that approximation values are reasonable."""
        import math
        N, P = 4, 4
        with pytest.warns(RuntimeWarning, match="asymptotic"):
            result = zagar.getA12(
                N=N,
                a_vec=[P] * N,
                phi0_vec=[N] * P,
                phi1_vec=[N] * P,
                return_approx=True
            )
        exact_log2 = math.log2(result["count"])
        # Upper bound should be >= exact
        assert result["log2_upper_bound"] >= exact_log2 
        # Approximation should be close to exact for small N,P
        assert abs(result["log2_approx"] - exact_log2) < 2  # within 2 bits

    def test_count_matches_non_approx(self):
        """Test that count in approx mode matches non-approx mode."""
        N, P = 6, 6
        non_approx = zagar.getA12(N=N, a_vec=[P]*N, phi0_vec=[N]*P, phi1_vec=[N]*P)
        with pytest.warns(RuntimeWarning, match="asymptotic"):
            approx = zagar.getA12(
                N=N, a_vec=[P]*N, phi0_vec=[N]*P, phi1_vec=[N]*P,
                return_approx=True
            )
        assert approx["count"] == non_approx

    def test_semiregular_but_not_symmetric_gets_an_approximation(self):
        """New in 0.2.0: Theorem 3.1 covers every semi-regular margin.

        In 0.1.0 this returned only the count, because the approximation was
        available at the symmetric point alone.
        """
        import math
        with pytest.warns(RuntimeWarning, match="asymptotic"):
            result = zagar.getA12(4, [4]*4, [2]*4, [2]*4, return_approx=True)
        assert result["is_regular"] is False
        assert result["case"] == 1
        assert result["densities"] == (0.5, 0.25, 0.25)
        # The symmetric-point-only upper bound does not apply off that point.
        assert "log2_upper_bound" not in result
        err = result["log2_approx"] - math.log2(result["count"])
        assert 0 < err < 0.04

    def test_non_uniform_margins_get_no_approximation_and_no_warning(self):
        """Theorem 3.1 does not cover non-uniform margins, so stay silent."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")       # any warning becomes a failure
            result = zagar.getA12(3, [6, 1, 7], [2, 0, 2, 1], [1, 3, 4, 1],
                                  return_approx=True)
        assert result["count"] == 133520
        assert "log2_approx" not in result
        assert "case" not in result

    def test_all_three_variants_return_identical_payloads(self):
        """getA12, getA12_parallel and getA12_pruning share _approx_result."""
        payloads = []
        for fn in (zagar.getA12, zagar.getA12_parallel, zagar.getA12_pruning):
            with pytest.warns(RuntimeWarning, match="asymptotic"):
                payloads.append(fn(4, [4]*4, [2]*4, [2]*4, return_approx=True))
        assert payloads[0] == payloads[1] == payloads[2]


class TestGetA12ParallelApproximation:
    """Tests for getA12_parallel return_approx feature."""

    def test_regular_case_detected(self):
        """Test that regular case is correctly detected in parallel."""
        N, P = 4, 4
        with pytest.warns(RuntimeWarning, match="asymptotic"):
            result = zagar.getA12_parallel(
                N=N,
                a_vec=[P] * N,
                phi0_vec=[N] * P,
                phi1_vec=[N] * P,
                return_approx=True
            )
        assert result["is_regular"] is True
        assert "count" in result
        assert "log2_upper_bound" in result
        assert "log2_approx" in result

    def test_non_regular_case_detected(self):
        """Test that non-regular case is correctly detected in parallel."""
        result = zagar.getA12_parallel(
            N=5,
            a_vec=[2, 2],
            phi0_vec=[1, 1],
            phi1_vec=[1, 1],
            return_approx=True
        )
        assert result["is_regular"] is False
        assert "count" in result
        assert "log2_upper_bound" not in result

    def test_return_approx_false_returns_int(self):
        """Test that return_approx=False returns int (default behavior)."""
        result = zagar.getA12_parallel(N=4, a_vec=[4]*4, phi0_vec=[4]*4, phi1_vec=[4]*4)
        assert isinstance(result, int)
        assert result == 824410

    def test_parallel_matches_serial_with_approx(self):
        """Test that parallel with approx matches serial with approx."""
        N, P = 6, 6
        with pytest.warns(RuntimeWarning, match="asymptotic"):
            serial = zagar.getA12(
                N=N, a_vec=[P]*N, phi0_vec=[N]*P, phi1_vec=[N]*P,
                return_approx=True
            )
        with pytest.warns(RuntimeWarning, match="asymptotic"):
            parallel = zagar.getA12_parallel(
                N=N, a_vec=[P]*N, phi0_vec=[N]*P, phi1_vec=[N]*P,
                return_approx=True
            )
        assert serial["count"] == parallel["count"]
        assert serial["is_regular"] == parallel["is_regular"]
        assert abs(serial["log2_upper_bound"] - parallel["log2_upper_bound"]) < 1e-10
        assert abs(serial["log2_approx"] - parallel["log2_approx"]) < 1e-10


class TestGetConstraints:
    """Tests for get_constraints function."""

    def test_basic_shape(self):
        """Test that output shapes are correct."""
        np.random.seed(42)
        N, P = 6, 5
        A = np.random.randint(0, 2, size=(N, 2*P)).astype(float)
        X = np.random.randint(0, 2, size=(N, 2*P)).astype(float)
        a_vec, phi0_vec, phi1_vec = zagar.get_constraints(A, X)
        assert len(a_vec) == N
        assert len(phi0_vec) == P
        assert len(phi1_vec) == P

    def test_output_types(self):
        """Test that outputs are lists of integers."""
        np.random.seed(42)
        N, P = 4, 3
        A = np.random.randint(0, 2, size=(N, 2*P)).astype(float)
        X = np.random.randint(0, 2, size=(N, 2*P)).astype(float)
        a_vec, phi0_vec, phi1_vec = zagar.get_constraints(A, X)
        assert isinstance(a_vec, list)
        assert isinstance(phi0_vec, list)
        assert isinstance(phi1_vec, list)
        assert all(isinstance(v, int) for v in a_vec)
        assert all(isinstance(v, int) for v in phi0_vec)
        assert all(isinstance(v, int) for v in phi1_vec)

    def test_a_vec_is_row_sums(self):
        """Test that a_vec equals row sums of A."""
        A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 1, 0, 0]], dtype=float)
        X = np.random.randint(0, 2, size=A.shape).astype(float)
        a_vec, _, _ = zagar.get_constraints(A, X)
        expected = [2, 2, 2]  # row sums of A
        assert a_vec == expected

    def test_phi_computation(self):
        """Test phi0_vec and phi1_vec computation for known case."""
        # N=2, P=2 (so 4 columns)
        # A[n,p]=1 means ancestry 1, A[n,p]=0 means ancestry 0
        A = np.array([[1, 0, 0, 1],
                      [0, 1, 1, 0]], dtype=float)
        X = np.array([[1, 1, 1, 1],
                      [1, 1, 1, 1]], dtype=float)
        a_vec, phi0_vec, phi1_vec = zagar.get_constraints(A, X)
        # phi0_vec[p] = sum of X where A=0 at columns p and P+p
        # p=0: A[:,0]=[1,0], A[:,2]=[0,1]
        #      X where A=0: X[1,0]=1, X[0,2]=1 -> phi0_vec[0] = 2
        # p=1: A[:,1]=[0,1], A[:,3]=[1,0]
        #      X where A=0: X[0,1]=1, X[1,3]=1 -> phi0_vec[1] = 2
        assert phi0_vec == [2, 2]
        # phi1_vec[p] = sum of X where A=1
        # p=0: A=1 at (0,0) and (1,2) -> X[0,0]=1, X[1,2]=1 -> phi1_vec[0] = 2
        # p=1: A=1 at (1,1) and (0,3) -> X[1,1]=1, X[0,3]=1 -> phi1_vec[1] = 2
        assert phi1_vec == [2, 2]

    def test_mismatched_shapes_raises(self):
        """Test that mismatched A and X shapes raise ValueError."""
        A = np.zeros((3, 4))
        X = np.zeros((3, 6))
        with pytest.raises(ValueError, match="same shape"):
            zagar.get_constraints(A, X)

    def test_odd_columns_raises(self):
        """Test that odd number of columns raises ValueError."""
        A = np.zeros((3, 5))
        X = np.zeros((3, 5))
        with pytest.raises(ValueError, match="even"):
            zagar.get_constraints(A, X)

    def test_na_values_warn(self):
        """Test that NA values trigger a warning."""
        np.random.seed(42)
        N, P = 4, 3
        A = np.random.randint(0, 2, size=(N, 2*P)).astype(float)
        X = np.random.randint(0, 2, size=(N, 2*P)).astype(float)
        A[0, 0] = np.nan
        X[1, 2] = np.nan
        with pytest.warns(UserWarning, match="Missing values detected"):
            zagar.get_constraints(A, X)

    def test_na_values_ignored_in_a_vec(self):
        """Test that NA values in A are ignored for a_vec calculation."""
        A = np.array([[1, 1, 1, 1],
                      [1, 0, 1, 0]], dtype=float)
        X = np.ones_like(A)
        A[0, 0] = np.nan  # Make one entry NA
        with pytest.warns(UserWarning):
            a_vec, _, _ = zagar.get_constraints(A, X)
        # Row 0: was [1,1,1,1] -> [nan,1,1,1], sum ignoring NA = 3
        # Row 1: [1,0,1,0], sum = 2
        assert a_vec == [3, 2]

    def test_na_values_ignored_in_phi_vecs(self):
        """Test that NA values are ignored in phi computation."""
        A = np.array([[0, 0, 0, 0],
                      [1, 1, 1, 1]], dtype=float)
        X = np.array([[1, 1, 1, 1],
                      [1, 1, 1, 1]], dtype=float)
        X[0, 0] = np.nan  # NA in X
        with pytest.warns(UserWarning):
            _, phi0_vec, phi1_vec = zagar.get_constraints(A, X)
        # p=0: phi0 would normally include X[0,0]=1 (since A[0,0]=0), but it's NA
        #      phi0_vec[0] = X[0,2]=1 (A[0,2]=0) -> 1
        #      phi1_vec[0] = X[1,0]=1 + X[1,2]=1 = 2
        assert phi0_vec[0] == 1
        assert phi1_vec[0] == 2

    def test_integer_input_accepted(self):
        """Test that integer arrays work (no NA handling needed)."""
        A = np.array([[1, 0, 0, 1],
                      [0, 1, 1, 0]], dtype=int)
        X = np.array([[1, 1, 1, 1],
                      [1, 1, 1, 1]], dtype=int)
        # Should not warn since integer arrays can't have NaN
        a_vec, phi0_vec, phi1_vec = zagar.get_constraints(A, X)
        assert a_vec == [2, 2]
        assert phi0_vec == [2, 2]
        assert phi1_vec == [2, 2]

    def test_all_zeros(self):
        """Test with all-zero matrices."""
        N, P = 3, 2
        A = np.zeros((N, 2*P))
        X = np.zeros((N, 2*P))
        a_vec, phi0_vec, phi1_vec = zagar.get_constraints(A, X)
        assert a_vec == [0] * N
        assert phi0_vec == [0] * P
        assert phi1_vec == [0] * P

    def test_all_ones(self):
        """Test with all-one matrices."""
        N, P = 3, 2
        A = np.ones((N, 2*P))
        X = np.ones((N, 2*P))
        a_vec, phi0_vec, phi1_vec = zagar.get_constraints(A, X)
        assert a_vec == [2*P] * N  # Each row has 2*P ones
        assert phi0_vec == [0] * P  # No positions where A=0
        assert phi1_vec == [2*N] * P  # All positions have A=1, X=1
