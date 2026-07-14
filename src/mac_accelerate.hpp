/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

The main author may be contacted at c.diddens@utwente.nl

================================================================================*/

#pragma once

// Only meaningful on macOS, where Apple's Accelerate framework provides the
// "Sparse Solvers" API used below. On any other platform this header (and
// its matching mac_accelerate.cpp) compiles to an empty translation unit, so
// it can unconditionally be part of the source list on all platforms.
#ifdef __APPLE__

#include <string>
#include <vector>
#include <cstdint>

#include <Accelerate/Accelerate.h>

namespace pyoomph
{

    // Subset of Accelerate's SparseFactorization_t that is meaningful for the matrices pyoomph
    // hands over (real, either general or symmetric, sparse matrices). Kept as pyoomph's own type
    // rather than exposing SparseFactorization_t directly so that callers (in particular the
    // pybind11 bindings) do not need to include Accelerate/Accelerate.h themselves.
    enum class MacAccelerateMethod
    {
        QR,             // General (unsymmetric, possibly rectangular) sparse QR. Always applicable.
        Cholesky,       // Symmetric positive definite; requires a square matrix.
        LDLT,           // Symmetric indefinite (Bunch-Kaufman pivoting); requires a square matrix.
        LDLTUnpivoted,  // Symmetric indefinite, no pivoting; requires a square matrix.
        LDLTSBK,        // Symmetric indefinite (supernodal Bunch-Kaufman); requires a square matrix.
        LDLTTPP,        // Symmetric indefinite (threshold partial pivoting); requires a square matrix.
        CholeskyAtA     // Cholesky factorization of AtA; used for least-squares on rectangular A.
    };

    // Parses one of "qr", "cholesky", "ldlt", "ldlt_unpivoted", "ldlt_sbk", "ldlt_tpp",
    // "cholesky_at_a" (case-insensitive); throws std::invalid_argument otherwise.
    MacAccelerateMethod mac_accelerate_method_from_string(const std::string &name);
    std::string mac_accelerate_method_to_string(MacAccelerateMethod method);
    // All method names accepted by mac_accelerate_method_from_string(), in a fixed order -
    // exposed so Python-side code can offer/validate choices without duplicating the list.
    std::vector<std::string> mac_accelerate_available_methods();

    // Holds a factorized sparse matrix (via Apple's Accelerate Sparse Solvers) and lets it be
    // solved against one or more right-hand sides, and re-factorized with a different method or
    // matrix without constructing a new solver.
    class MacAccelerateSparseSolver
    {
    public:
        MacAccelerateSparseSolver() = default;
        ~MacAccelerateSparseSolver();

        // Disallow copies: Accelerate's opaque factorization keeps raw pointers into this
        // instance's own buffers (see releaseFactorization()/m_values etc below).
        MacAccelerateSparseSolver(const MacAccelerateSparseSolver &) = delete;
        MacAccelerateSparseSolver &operator=(const MacAccelerateSparseSolver &) = delete;

        // factorize(n_rows, n_cols, indptr, indices, data, method)
        //
        // CSR arrays as produced by scipy.sparse.csr_matrix:
        //   indptr  : array of length n_rows+1
        //   indices : array of length nnz (column indices)
        //   data    : array of length nnz
        //
        // For a symmetric method (Cholesky/LDLT*), n_rows must equal n_cols, and only the upper
        // triangle (row <= col) of the given CSR matrix is used - entries below the diagonal are
        // silently ignored, mirroring the "give me one triangle" contract of the underlying
        // Accelerate API.
        void factorize(int n_rows, int n_cols,
                       const std::vector<long long> &indptr,
                       const std::vector<long long> &indices,
                       const std::vector<double> &data,
                       MacAccelerateMethod method = MacAccelerateMethod::QR);

        // Re-run the numerical (and, if necessary, symbolic) factorization of the last matrix
        // passed to factorize() with a different method, without the caller having to resupply
        // the CSR arrays again.
        void refactorize(MacAccelerateMethod method);

        // solve(b) -> x. b must have length rows(); returns a vector of length cols(), reusing
        // the cached factorization (equivalent to calling this repeatedly for new right-hand
        // sides without re-factorizing).
        std::vector<double> solve(const std::vector<double> &b) const;

        bool isFactorized() const { return m_hasFactorization; }
        int rows() const { return m_n_rows; }
        int cols() const { return m_n_cols; }
        MacAccelerateMethod method() const { return m_method; }

    private:
        void releaseFactorization();
        void buildMatrixAndFactorize(MacAccelerateMethod method);

        int m_n_rows = 0;
        int m_n_cols = 0;
        MacAccelerateMethod m_method = MacAccelerateMethod::QR;

        // Cached CSR input (kept so refactorize() can rebuild with a different method).
        std::vector<long long> m_csr_indptr;
        std::vector<long long> m_csr_indices;
        std::vector<double> m_csr_data;

        // Backing storage for the CSC representation; must outlive m_matrix and
        // m_factorization since Accelerate keeps raw pointers into it.
        std::vector<long long> m_colStarts;
        std::vector<long> m_colStarts32;
        std::vector<int32_t> m_rowIndices;
        std::vector<double> m_values;

        SparseMatrix_Double m_matrix{};
        SparseOpaqueFactorization_Double m_factorization{};
        bool m_hasFactorization = false;
    };

} // namespace pyoomph

#endif // __APPLE__
