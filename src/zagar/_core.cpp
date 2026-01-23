#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "count_binary.hpp"

namespace py = pybind11;

// Helper: Convert string representation of big integer to Python int
static py::object string_to_pyint(const std::string& num_str) {
    return py::int_(py::str(num_str));
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "zagar: Combinatorial counting algorithms for binary matrices";

    // count_binary_matrices function
    m.def("count_binary_matrices", [](
        const std::vector<int>& p,
        const std::vector<int>& q,
        bool sort_p_desc
    ) -> py::dict {
        auto result = zagar::count_binary_matrices(p, q, sort_p_desc);
        py::dict d;
        d["number"] = string_to_pyint(result.number);
        d["memo_entries"] = result.memo_entries;
        return d;
    },
    py::arg("p"),
    py::arg("q"),
    py::arg("sort_p_desc") = false,
    R"doc(
Count Binary Matrices with Row Sums :math:`\boldsymbol{p}` and Column Sums :math:`\boldsymbol{q}`.

The underlying algorithm is a minor modification of the original EXACT algorithm implemented 
by Miller and Harrison (2013) *Annals of Statistics*. Compared to their original C version, 
this version uses modern C++ optimizations (e.g., ``std::unordered_map`` in C++ 
versus ``khash`` from ``klib`` for hashing) and performs operations on flexible 
integer types, which may be more efficient when the problem size is not too large 
(e.g., ``uint64_t`` rather than arbitrary precision types for smaller input values).

Parameters
----------
p : list of int
    Row sums (nonnegative integers)
q : list of int
    Column sums (nonnegative integers)
sort_p_desc : bool, optional
    If True, sort p in decreasing order for speed (default: False)

Returns
-------
dict
    {'number': int, 'memo_entries': int}
        - number: The count of binary matrices (arbitrary precision)
        - memo_entries: Number of entries in the memoization table

Examples
--------
>>> import zagar
>>> result = zagar.count_binary_matrices([2, 1], [1, 1, 1])
>>> result['number']
3
)doc");

    // getA12 function
    m.def("getA12", [](
        int N,
        const std::vector<int>& a_vec,
        const std::vector<int>& phi0_vec,
        const std::vector<int>& phi1_vec,
        bool sort_p_desc
    ) -> py::object {
        auto result = zagar::getA12(N, a_vec, phi0_vec, phi1_vec, sort_p_desc);
        return string_to_pyint(result.number);
    },
    py::arg("N"),
    py::arg("a_vec"),
    py::arg("phi0_vec"),
    py::arg("phi1_vec"),
    py::arg("sort_p_desc") = true,
    R"doc(
Compute weighted combinatorial sum A12.

Parameters
----------
N : int
    Positive integer parameter
a_vec : list of int
    Row sums
phi0_vec : list of int
    First constraint vector
phi1_vec : list of int
    Second constraint vector (must have same length as phi0_vec)
sort_p_desc : bool, optional
    If True, sort p in decreasing order (default: True)

Returns
-------
int
    The computed A12 value (arbitrary precision)
)doc");

    // getA12_parallel function
    m.def("getA12_parallel", [](
        int N,
        const std::vector<int>& a_vec,
        const std::vector<int>& phi0_vec,
        const std::vector<int>& phi1_vec,
        int n_threads,
        bool sort_p_desc
    ) -> py::object {
        auto result = zagar::getA12_parallel(N, a_vec, phi0_vec, phi1_vec, n_threads, sort_p_desc);
        return string_to_pyint(result.number);
    },
    py::arg("N"),
    py::arg("a_vec"),
    py::arg("phi0_vec"),
    py::arg("phi1_vec"),
    py::arg("n_threads") = -1,
    py::arg("sort_p_desc") = true,
    R"doc(
Compute weighted combinatorial sum A12 using parallel processing.

Parameters
----------
N : int
    Positive integer parameter
a_vec : list of int
    Row sums
phi0_vec : list of int
    First constraint vector
phi1_vec : list of int
    Second constraint vector (must have same length as phi0_vec)
n_threads : int, optional
    Number of threads (-1 = use all available, default: -1)
sort_p_desc : bool, optional
    If True, sort p in decreasing order (default: True)

Returns
-------
int
    The computed A12 value (arbitrary precision)

Notes
-----
This function uses OpenMP for parallelization when available.
If OpenMP is not available at compile time, it falls back to
serial execution.
)doc");

    // getA1 function
    m.def("getA1", [](
        int N,
        int P,
        const std::vector<int>& a_vec,
        bool return_log
    ) -> py::object {
        auto result = zagar::getA1(N, P, a_vec);
        if (return_log) {
            return py::float_(result.log2_value);
        } else {
            return string_to_pyint(result.number);
        }
    },
    py::arg("N"),
    py::arg("P"),
    py::arg("a_vec"),
    py::arg("return_log") = false,
    R"doc(
Cardinality of :math:`\mathscr{A}_1`: Number of Arrays Subject to Row Margin Constraints

Formula: prod(C(2*P, a_n)) * 2^(2*N*P) for all a_n in a_vec

Mathematically,
:math:`2^{2NP} \prod_{n=1}^N \binom{2P}{a_{n\cdot}}` for all :math:`a_{n\cdot}` in :math:`\boldsymbol{a}`

Parameters
----------
N : int
    Positive integer (number of rows)
P : int
    Positive integer (number of columns)
a_vec : list of int
    Row margin constraints (length N, values in [0, 2*P])
return_log : bool, optional
    If True, return log2 of result as float (default: False)

Returns
-------
int or float
    If return_log=False: The exact :math:`|\mathscr{A}_1|` value (arbitrary precision)

    If return_log=True: :math:`\log_2(|\mathscr{A}_1|)` as a float

Examples
--------
>>> import zagar
>>> zagar.getA1(N=50, P=50, a_vec=[25]*50)
>>> zagar.getA1(N=50, P=50, a_vec=[25]*50, return_log=True)
)doc");

    // getA2 function
    m.def("getA2", [](
        int N,
        int P,
        const std::vector<int>& phi0,
        const std::vector<int>& phi1,
        bool return_log
    ) -> py::object {
        auto result = zagar::getA2(N, P, phi0, phi1);
        if (return_log) {
            return py::float_(result.log2_value);
        } else {
            return string_to_pyint(result.number);
        }
    },
    py::arg("N"),
    py::arg("P"),
    py::arg("phi0_vec"),
    py::arg("phi1_vec"),
    py::arg("return_log") = false,
    R"doc(
Cardinality of :math:`\mathscr{A}_2`: Number of Arrays Subject to Column Margin Constraints

Formula: prod(C(2*N, phi0_p+phi1_p) * C(phi0_j+phi1_p, phi1_p) * 2^(2*N-(phi0_p+phi1_p)))

Mathematically,
:math:`\prod_{p=1}^P \binom{2N}{\phi_{p,0}+\phi_{p,1}} \binom{\phi_{p,0}+\phi_{p,1}}{\phi_{p,0}} 2^{2N - (\phi_{p,0}+\phi_{p,1})}` 

Parameters
----------
N : int
    Positive integer (number of rows)
P : int
    Positive integer (number of columns)
phi0_vec : list of int
    First constraint vector (length P, values in [0, 2*N])
phi1_vec : list of int
    Second constraint vector (length P, phi0_j+phi1_j <= 2*N)
return_log : bool, optional
    If True, return log2 of result as float (default: False)

Returns
-------
int or float
    If return_log=False: The exact :math:`|\mathscr{A}_2|` value (arbitrary precision)

    If return_log=True: :math:`\log_2(|\mathscr{A}_2|)` as a float

Examples
--------
>>> import zagar
>>> zagar.getA2(N=50, P=50, phi0_vec=[20]*50, phi1_vec=[22]*50)
>>> zagar.getA2(N=50, P=50, phi0_vec=[20]*50, phi1_vec=[22]*50, return_log=True)
)doc");

    // has_openmp function
    m.def("has_openmp", []() -> bool {
#ifdef ZAGAR_HAS_OPENMP
        return true;
#else
        return false;
#endif
    },
    R"doc(
Check if OpenMP support is available.

Returns True if the package was compiled with OpenMP support,
False otherwise. When OpenMP is not available, getA12_parallel
falls back to serial execution.

Returns
-------
bool
    True if OpenMP is available, False otherwise.

Examples
--------
>>> import zagar
>>> zagar.has_openmp()
False  # or True if compiled with OpenMP
)doc");

    // Version info
    m.attr("__version__") = "0.1.0";
}
