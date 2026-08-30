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

#include "mac_accelerate.hpp"

#ifdef __APPLE__

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <cstdlib>

namespace pyoomph
{

    namespace
    {

        std::string statusToString(SparseStatus_t status)
        {
            switch (status)
            {
            case SparseStatusOK:
                return "SparseStatusOK";
            case SparseFactorizationFailed:
                return "SparseFactorizationFailed";
            case SparseMatrixIsSingular:
                return "SparseMatrixIsSingular";
            case SparseInternalError:
                return "SparseInternalError";
            case SparseParameterError:
                return "SparseParameterError";
            case SparseStatusReleased:
                return "SparseStatusReleased";
            default:
            {
                std::ostringstream oss;
                oss << "Unknown SparseStatus_t(" << static_cast<int>(status) << ")";
                return oss.str();
            }
            }
        }

        // Passed to SparseFactor as SparseSymbolicFactorOptions::reportError. Its mere presence is the
        // point: when this pointer is NULL (as with the 2-argument SparseFactor overload we used to
        // call), Accelerate does not RETURN SparseMatrixIsSingular for a rank-deficient matrix - the QR
        // path halts the whole process via __builtin_trap() (a SIGILL through libSparse's _SparseTrap),
        // which no C++ catch and no Newton retry can survive. A singular Jacobian is routine mid-Newton
        // (an injected abort, a diverging step, a fold), so a Mac defaulting to Accelerate would die on
        // it. With a non-NULL reportError, Accelerate instead returns with .status set, letting
        // checkStatus() raise the catchable MacAccelerateNumericalFailure the retry logic expects.
        // The message is already conveyed by the status, so this callback only needs to exist.
        void reportSparseError(const char * /*message*/) {}

        void checkStatus(SparseStatus_t status, const char *where)
        {
            if (status == SparseStatusOK) return;
            const std::string msg = std::string(where) + " failed: " + statusToString(status);
            // Only these two say something about the MATRIX, and only those are worth retrying with a
            // smaller step; the rest say something about the call or about Accelerate itself, where a
            // retry would only hide the message. See MacAccelerateNumericalFailure in the header.
            if (status == SparseMatrixIsSingular || status == SparseFactorizationFailed)
                throw MacAccelerateNumericalFailure(msg);
            throw std::runtime_error(msg);
        }

        // Every symmetric method requires a square matrix and only ever looks at one stored
        // triangle (we always supply the upper one, see buildMatrixAndFactorize()).
        bool methodIsSymmetric(MacAccelerateMethod method)
        {
            switch (method)
            {
            case MacAccelerateMethod::Cholesky:
            case MacAccelerateMethod::LDLT:
            case MacAccelerateMethod::LDLTUnpivoted:
            case MacAccelerateMethod::LDLTSBK:
            case MacAccelerateMethod::LDLTTPP:
                return true;
            case MacAccelerateMethod::QR:
            case MacAccelerateMethod::CholeskyAtA:
                return false;
            }
            return false;
        }

        SparseFactorization_t toAccelerateType(MacAccelerateMethod method)
        {
            switch (method)
            {
            case MacAccelerateMethod::Cholesky:
                return SparseFactorizationCholesky;
            case MacAccelerateMethod::LDLT:
                return SparseFactorizationLDLT;
            case MacAccelerateMethod::LDLTUnpivoted:
                return SparseFactorizationLDLTUnpivoted;
            case MacAccelerateMethod::LDLTSBK:
                return SparseFactorizationLDLTSBK;
            case MacAccelerateMethod::LDLTTPP:
                return SparseFactorizationLDLTTPP;
            case MacAccelerateMethod::QR:
                return SparseFactorizationQR;
            case MacAccelerateMethod::CholeskyAtA:
                return SparseFactorizationCholeskyAtA;
            }
            throw std::invalid_argument("Unknown MacAccelerateMethod");
        }

        std::string toLower(const std::string &s)
        {
            std::string res = s;
            std::transform(res.begin(), res.end(), res.begin(), [](unsigned char c)
                           { return std::tolower(c); });
            return res;
        }

    } // namespace

    std::vector<std::string> mac_accelerate_available_methods()
    {
        return {"qr", "cholesky", "ldlt", "ldlt_unpivoted", "ldlt_sbk", "ldlt_tpp", "cholesky_at_a"};
    }

    MacAccelerateMethod mac_accelerate_method_from_string(const std::string &name)
    {
        const std::string n = toLower(name);
        if (n == "qr")
            return MacAccelerateMethod::QR;
        if (n == "cholesky")
            return MacAccelerateMethod::Cholesky;
        if (n == "ldlt")
            return MacAccelerateMethod::LDLT;
        if (n == "ldlt_unpivoted")
            return MacAccelerateMethod::LDLTUnpivoted;
        if (n == "ldlt_sbk")
            return MacAccelerateMethod::LDLTSBK;
        if (n == "ldlt_tpp")
            return MacAccelerateMethod::LDLTTPP;
        if (n == "cholesky_at_a")
            return MacAccelerateMethod::CholeskyAtA;
        throw std::invalid_argument("Unknown Accelerate sparse factorization method '" + name +
                                    "'. Available: qr, cholesky, ldlt, ldlt_unpivoted, ldlt_sbk, ldlt_tpp, cholesky_at_a");
    }

    std::string mac_accelerate_method_to_string(MacAccelerateMethod method)
    {
        switch (method)
        {
        case MacAccelerateMethod::QR:
            return "qr";
        case MacAccelerateMethod::Cholesky:
            return "cholesky";
        case MacAccelerateMethod::LDLT:
            return "ldlt";
        case MacAccelerateMethod::LDLTUnpivoted:
            return "ldlt_unpivoted";
        case MacAccelerateMethod::LDLTSBK:
            return "ldlt_sbk";
        case MacAccelerateMethod::LDLTTPP:
            return "ldlt_tpp";
        case MacAccelerateMethod::CholeskyAtA:
            return "cholesky_at_a";
        }
        return "unknown";
    }

    MacAccelerateSparseSolver::~MacAccelerateSparseSolver()
    {
        releaseFactorization();
    }

    void MacAccelerateSparseSolver::releaseFactorization()
    {
        if (m_hasFactorization)
        {
            SparseCleanup(m_factorization);
            m_hasFactorization = false;
        }
    }

    void MacAccelerateSparseSolver::factorize(int n_rows, int n_cols,
                                              const std::vector<long long> &indptr,
                                              const std::vector<long long> &indices,
                                              const std::vector<double> &data,
                                              MacAccelerateMethod method)
    {
        if (static_cast<long long>(indptr.size()) != n_rows + 1)
        {
            throw std::invalid_argument("indptr must have length n_rows + 1");
        }
        const long long nnz = indptr[n_rows];
        if (static_cast<long long>(indices.size()) != nnz ||
            static_cast<long long>(data.size()) != nnz)
        {
            throw std::invalid_argument("indices/data length must equal indptr[-1] (nnz)");
        }

        m_n_rows = n_rows;
        m_n_cols = n_cols;
        m_csr_indptr = indptr;
        m_csr_indices = indices;
        m_csr_data = data;

        buildMatrixAndFactorize(method);
    }

    void MacAccelerateSparseSolver::refactorize(MacAccelerateMethod method)
    {
        if (m_csr_indptr.empty())
        {
            throw std::runtime_error("refactorize() called before factorize()");
        }
        buildMatrixAndFactorize(method);
    }

    void MacAccelerateSparseSolver::buildMatrixAndFactorize(MacAccelerateMethod method)
    {
        releaseFactorization();

        const bool symmetric = methodIsSymmetric(method);
        if (symmetric && m_n_rows != m_n_cols)
        {
            throw std::invalid_argument("Method '" + mac_accelerate_method_to_string(method) +
                                        "' requires a square matrix");
        }

        const int n_rows = m_n_rows;
        const int n_cols = m_n_cols;
        const auto &ip = m_csr_indptr;
        const auto &ind = m_csr_indices;
        const auto &dat = m_csr_data;

        // ---- Convert CSR (row-major) -> CSC (column-major) ----
        // Standard counting-sort transpose-conversion, O(nnz + n). For symmetric methods, only
        // the upper triangle (row <= col) is kept, matching Accelerate's "one stored triangle"
        // contract for SparseSymmetric matrices.
        m_colStarts.assign(n_cols + 1, 0);
        for (int row = 0; row < n_rows; ++row)
        {
            for (long long k = ip[row]; k < ip[row + 1]; ++k)
            {
                const long long col = ind[k];
                if (col < 0 || col >= n_cols)
                {
                    throw std::invalid_argument("column index out of range in CSR data");
                }
                if (symmetric && row > col)
                    continue;
                m_colStarts[col + 1]++;
            }
        }
        for (int c = 0; c < n_cols; ++c)
        {
            m_colStarts[c + 1] += m_colStarts[c];
        }

        const long long kept_nnz = m_colStarts[n_cols];
        m_rowIndices.assign(kept_nnz, 0);
        m_values.assign(kept_nnz, 0.0);
        // Remember where each stored value came from, so a later numeric-only refactorization can
        // refill m_values by a gather instead of redoing this conversion.
        m_csc_from_csr.assign(kept_nnz, 0);
        std::vector<long long> writeCursor(m_colStarts.begin(), m_colStarts.end());

        for (int row = 0; row < n_rows; ++row)
        {
            for (long long k = ip[row]; k < ip[row + 1]; ++k)
            {
                const long long col = ind[k];
                if (symmetric && row > col)
                    continue;
                const double val = dat[k];
                const long long dest = writeCursor[col]++;
                m_rowIndices[dest] = static_cast<int32_t>(row);
                m_values[dest] = val;
                m_csc_from_csr[dest] = k;
            }
        }

        // Accelerate wants int32_t column starts (length n_cols+1)
        m_colStarts32.assign(m_colStarts.begin(), m_colStarts.end());

        // ---- Build SparseMatrixStructure / SparseMatrix_Double ----
        SparseAttributes_t attributes{};
        attributes.transpose = false;
        attributes.triangle = SparseUpperTriangle; // ignored for general (non-symmetric) matrices
        attributes.kind = symmetric ? SparseSymmetric : SparseOrdinary;
        attributes._reserved = 0;

        SparseMatrixStructure structure{};
        structure.rowCount = n_rows;
        structure.columnCount = n_cols;
        structure.columnStarts = m_colStarts32.data();
        structure.rowIndices = m_rowIndices.data();
        structure.attributes = attributes;
        structure.blockSize = 1;

        m_matrix = SparseMatrix_Double{};
        m_matrix.structure = structure;
        m_matrix.data = m_values.data();

        // ---- Factorize with the requested method ----
        // The 4-argument overload is used solely to install reportError (see reportSparseError above);
        // every other field is Accelerate's own default. malloc/free are passed explicitly because the
        // struct annotates them _Nonnull - the system allocator is what the short overload uses anyway,
        // and macOS malloc satisfies the documented 16-byte alignment requirement.
        SparseSymbolicFactorOptions sfoptions{};
        sfoptions.control = SparseDefaultControl;
        sfoptions.orderMethod = SparseOrderDefault;
        sfoptions.order = nullptr;
        sfoptions.ignoreRowsAndColumns = nullptr;
        sfoptions.malloc = &std::malloc;
        sfoptions.free = &std::free;
        sfoptions.reportError = &reportSparseError;

        SparseNumericFactorOptions nfoptions{};
        nfoptions.control = SparseDefaultControl;
        nfoptions.scalingMethod = SparseScalingDefault;
        nfoptions.scaling = nullptr;
        nfoptions.pivotTolerance = 0.0;
        nfoptions.zeroTolerance = 0.0;

        m_factorization = SparseFactor(toAccelerateType(method), m_matrix, sfoptions, nfoptions);
        const SparseStatus_t status = m_factorization.status;
        if (status != SparseStatusOK)
        {
            // reportError kept SparseFactor from trapping, so the returned object is valid but may hold
            // partial allocations; release them before reporting, or every singular Newton iterate would
            // leak one factorization. m_hasFactorization stays false, so the destructor will not free it
            // again. The status is read before the cleanup, since SparseCleanup invalidates the object.
            SparseCleanup(m_factorization);
            m_factorization = SparseOpaqueFactorization_Double{};
            checkStatus(status, ("SparseFactor (" + mac_accelerate_method_to_string(method) + ")").c_str());
        }
        m_hasFactorization = true;
        m_method = method;
        m_n_symbolic++;
    }

    // See the declaration in mac_accelerate.hpp. SparseRefactor reuses the symbolic factorization held
    // inside m_factorization and recomputes only the numbers, which is valid exactly while the pattern
    // is unchanged -- so the pattern is checked here rather than trusted.
    bool MacAccelerateSparseSolver::refactorize_values_only(const std::vector<long long> &indptr,
                                                            const std::vector<long long> &indices,
                                                            const std::vector<double> &data)
    {
        if (!m_hasFactorization) return false;
        if (m_csr_indptr.size() != indptr.size()) return false;
        if (m_csr_indices.size() != indices.size()) return false;
        if (m_csr_data.size() != data.size()) return false;
        if (m_csc_from_csr.size() != m_values.size()) return false; // Factorized before the map existed
        if (!std::equal(m_csr_indptr.begin(), m_csr_indptr.end(), indptr.begin())) return false;
        if (!std::equal(m_csr_indices.begin(), m_csr_indices.end(), indices.begin())) return false;

        m_csr_data = data;
        for (size_t dest = 0; dest < m_values.size(); ++dest)
        {
            m_values[dest] = data[static_cast<size_t>(m_csc_from_csr[dest])];
        }
        // m_matrix.data already points at m_values.data(); the assignment above wrote through it, and
        // m_values has not been reallocated, so the pointer Accelerate holds is still valid.
        SparseRefactor(m_matrix, &m_factorization);
        if (m_factorization.status != SparseStatusOK)
        {
            // Deliberately not an error. Reusing the symbolic factorization fixes the pivoting, and the
            // new values may not tolerate it where a fresh factorization would have chosen differently.
            // Drop what we have and report failure so the caller does a full factorize; turning a
            // recoverable situation into an exception would make the optimisation a liability.
            releaseFactorization();
            return false;
        }
        m_n_numeric_only++;
        return true;
    }

    std::vector<double> MacAccelerateSparseSolver::solve(const std::vector<double> &b) const
    {
        if (!m_hasFactorization)
        {
            throw std::runtime_error("solve() called before factorize()");
        }
        if (static_cast<int>(b.size()) != m_n_rows)
        {
            throw std::invalid_argument("rhs length must equal number of rows of factorized matrix");
        }

        // Accelerate's SparseSolve expects the rhs/solution buffer sized to max(n_rows, n_cols)
        // and overwrites it in place with the solution in the first n_cols entries.
        const int workLen = std::max(m_n_rows, m_n_cols);
        std::vector<double> work(workLen, 0.0);
        for (int i = 0; i < m_n_rows; ++i)
            work[i] = b[i];

        DenseVector_Double xb{};
        xb.count = workLen;
        xb.data = work.data();

        SparseSolve(m_factorization, xb);

        std::vector<double> result(m_n_cols);
        for (int i = 0; i < m_n_cols; ++i)
            result[i] = work[i];
        return result;
    }

} // namespace pyoomph

#endif // __APPLE__
