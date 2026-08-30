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

// Bindings for Spectra (https://spectralib.org), the header-only Arnoldi eigensolver that
// cmake/ThirdPartySpectra.cmake downloads together with Eigen. Only compiled into _pyoomph_core
// when PYOOMPH_HAS_SPECTRA is on, and the only translation unit that may include Eigen's or
// Spectra's headers - see the include scoping in CMakeLists.txt for why that is not merely tidy.
//
// What is bound is deliberately only the Arnoldi iteration itself. The operator it iterates is a
// Python callable, and everything problem-specific - assembling J and M, forming and factorising
// J - sigma*M, back-transforming the eigenvalues - stays in pyoomph/solvers/spectra.py. The reason
// is that pyoomph has no C++ linear solver to call: MKL Pardiso is a ctypes wrapper in
// pyoomph/solvers/pardiso.py, and the C++ core already calls *out* to Python for every LU it needs
// (see GeneralSolverCallback in ../solver.cpp). One sparse triangular solve per matrix-vector
// product dwarfs the cost of the callback, so there is nothing to win by marshalling matrices down
// here and a lot of duplicated solver-selection logic to lose.
//
// Spectra has no generalized non-symmetric solver at all - SymGEigsSolver/SymGEigsShiftSolver
// require A symmetric and B positive definite, which pyoomph mass matrices violate: they are
// positive semi-definite and singular (pressure and pinned rows carry no time derivative). The
// Python side therefore applies the shift-and-invert transform itself, exactly as SLEPc's ST does,
// and hands us the standard-problem operator C = (J - sigma*M)^{-1} M. The infinite eigenvalues
// that a singular M produces land at nu = 0, which LargestMagn never converges to.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Eigen is "primarily MPL2" - a few of its headers carry third-party BSD or LGPL code (see its
// COPYING.README). EIGEN_MPL2_ONLY turns including any of those into a compile error, so the licence
// statement in README.md is enforced by the build rather than merely asserted. Nothing needed here is
// affected: the LGPL parts are the iterative linear solvers (IncompleteLUT), and this file only uses
// Eigen's dense core through Spectra.
#define EIGEN_MPL2_ONLY

#include <Eigen/Core>
#include <Spectra/GenEigsSolver.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace pyoomph
{
	namespace spectra_bindings
	{
		typedef std::complex<double> Complex;

		// Maps the sort-rule names used by pyoomph/solvers/spectra.py onto Spectra's enum. The
		// names are Spectra's own, spelled out rather than encoded as integers so that a mismatch
		// between the two sides is a clear error instead of a silently different spectrum.
		Spectra::SortRule parse_sort_rule(const std::string &name)
		{
			if (name == "LargestMagn") return Spectra::SortRule::LargestMagn;
			if (name == "SmallestMagn") return Spectra::SortRule::SmallestMagn;
			if (name == "LargestReal") return Spectra::SortRule::LargestReal;
			if (name == "SmallestReal") return Spectra::SortRule::SmallestReal;
			if (name == "LargestImag") return Spectra::SortRule::LargestImag;
			if (name == "SmallestImag") return Spectra::SortRule::SmallestImag;
			throw std::invalid_argument("unknown Spectra sort rule '" + name + "'");
		}

		std::string comp_info_name(Spectra::CompInfo info)
		{
			switch (info)
			{
			case Spectra::CompInfo::Successful: return "successful";
			case Spectra::CompInfo::NotConverging: return "not_converging";
			case Spectra::CompInfo::NumericalIssue: return "numerical_issue";
			default: return "unknown";
			}
		}

		// The operator Spectra iterates: y = op(x), with op a Python callable. Spectra only ever
		// asks for rows()/cols()/perform_op(), and never copies or reorders the operator, so
		// holding the callable and one scratch buffer by reference is safe for the lifetime of a
		// single solve.
		// The template parameter is S rather than Scalar because Spectra requires the member type
		// to be called exactly Scalar, and a template parameter of that name would shadow it.
		template <typename S>
		class PythonMatProd
		{
		public:
			using Scalar = S;
			using Index = Eigen::Index;

			PythonMatProd(const nb::callable &op, Index n) : m_op(op), m_n(n), m_buffer(n), m_calls(0) {}

			Index rows() const { return m_n; }
			Index cols() const { return m_n; }

			void perform_op(const Scalar *x_in, Scalar *y_out) const
			{
				// The GIL is held throughout compute() (see the note in the binding functions), so
				// nothing has to be acquired here.
				std::copy(x_in, x_in + m_n, m_buffer.begin());

				// The array handed to the callback is a VIEW onto m_buffer and is only valid for the
				// duration of the call - it is overwritten by the next matrix-vector product. The
				// Python side never retains it (both the Pardiso and the SuperLU wrappers copy the
				// right-hand side before solving), which is what makes reusing one buffer legitimate.
				size_t shape[1] = {(size_t) m_n};
				nb::ndarray<nb::numpy, Scalar, nb::ndim<1>, nb::c_contig> x_view(m_buffer.data(), 1, shape, nb::handle());

				nb::object result = m_op(x_view);
				++m_calls;

				// The callback returns a freshly allocated vector (Pardiso and SuperLU both do), so
				// there is nothing to be gained from an out-parameter. A wrong dtype or length would
				// otherwise surface as a cast_error mentioning only nanobind internals.
				nb::ndarray<nb::numpy, Scalar, nb::ndim<1>, nb::c_contig> y_arr;
				try
				{
					y_arr = nb::cast<nb::ndarray<nb::numpy, Scalar, nb::ndim<1>, nb::c_contig>>(result);
				}
				catch (const nb::cast_error &)
				{
					throw std::runtime_error(
						std::string("the Spectra operator callback must return a contiguous 1d numpy array of ") +
						(Eigen::NumTraits<Scalar>::IsComplex ? "complex128" : "float64") + " with " +
						std::to_string((long long) m_n) + " entries");
				}
				if ((Index) y_arr.shape(0) != m_n)
				{
					throw std::runtime_error(
						"the Spectra operator callback returned " + std::to_string((long long) y_arr.shape(0)) +
						" entries, expected " + std::to_string((long long) m_n));
				}
				std::copy(y_arr.data(), y_arr.data() + m_n, y_out);
			}

			long num_calls() const { return m_calls; }

		private:
			nb::callable m_op;
			Index m_n;
			mutable std::vector<Scalar> m_buffer;
			mutable long m_calls;
		};

		// Copies Spectra's results into freshly owned numpy arrays. Eigenvalues and eigenvectors are
		// returned as complex128 in BOTH instantiations: Spectra's real GenEigsSolver already produces
		// complex Ritz values (a real non-symmetric matrix has complex-conjugate pairs), and
		// GenericEigenSolver.solve() is contractually complex anyway, so converting here saves the
		// Python side a cast it would otherwise always perform.
		//
		// Spectra hands the eigenvectors back as an n x nconv matrix of columns, while
		// GenericEigenSolver.solve() promises row i to be eigenvector i. Transposing during this copy
		// keeps that detail from leaking into Python, and avoids handing out a Fortran-ordered array
		// that the caller would silently re-copy later.
		template <typename Derived>
		nb::object copy_eigenvalues(const Eigen::MatrixBase<Derived> &vals, Eigen::Index nconv)
		{
			Complex *data = new Complex[(size_t) std::max<Eigen::Index>(nconv, 1)];
			for (Eigen::Index i = 0; i < nconv; i++) data[i] = Complex(vals(i));
			nb::capsule owner(data, [](void *p) noexcept { delete[] (Complex *) p; });
			size_t shape[1] = {(size_t) nconv};
			return nb::cast(nb::ndarray<nb::numpy, Complex, nb::ndim<1>>(data, 1, shape, owner));
		}

		template <typename Derived>
		nb::object copy_eigenvectors(const Eigen::MatrixBase<Derived> &vecs, Eigen::Index nconv, Eigen::Index n)
		{
			Complex *data = new Complex[(size_t) std::max<Eigen::Index>(nconv * n, 1)];
			for (Eigen::Index k = 0; k < nconv; k++)
				for (Eigen::Index i = 0; i < n; i++)
					data[k * n + i] = Complex(vecs(i, k));
			nb::capsule owner(data, [](void *p) noexcept { delete[] (Complex *) p; });
			size_t shape[2] = {(size_t) nconv, (size_t) n};
			return nb::cast(nb::ndarray<nb::numpy, Complex, nb::ndim<2>>(data, 2, shape, owner));
		}

		// The shared implementation of both entry points. Spectra itself rejects nev/ncv outside
		// 1 <= nev <= n-2 and nev+2 <= ncv <= n by throwing std::invalid_argument, which nanobind
		// turns into a Python ValueError, so those bounds are not re-checked here - the Python side
		// clamps them before ever getting this far, and a mistake there should say so loudly.
		template <typename Scalar>
		nb::object eigensolve(const nb::callable &op_callable, int64_t n, int64_t nev, int64_t ncv,
							  int64_t maxit, double tol, nb::object v0, const std::string &sortrule)
		{
			if (n <= 0) throw std::invalid_argument("the matrix dimension n must be positive");

			PythonMatProd<Scalar> op(op_callable, (Eigen::Index) n);
			Spectra::GenEigsSolver<PythonMatProd<Scalar>> eigs(op, (Eigen::Index) nev, (Eigen::Index) ncv);

			if (v0.is_none())
			{
				eigs.init();
			}
			else
			{
				auto v0_arr = nb::cast<nb::ndarray<nb::numpy, Scalar, nb::ndim<1>, nb::c_contig>>(v0);
				if ((int64_t) v0_arr.shape(0) != n)
					throw std::invalid_argument("the start vector v0 must have n entries");
				eigs.init(v0_arr.data());
			}

			// NOT wrapped in nb::gil_scoped_release. perform_op() calls straight back into Python on
			// every matrix-vector product, so the GIL would have to be re-acquired each time for no
			// gain: there is no other thread to let run, and the expensive part - MKL Pardiso through
			// ctypes, or SuperLU inside scipy - drops the GIL itself for the duration of the solve.
			const Eigen::Index nconv = eigs.compute(parse_sort_rule(sortrule), (Eigen::Index) maxit, tol);

			// CompInfo is reported, not thrown. Spectra still returns the converged subset when it
			// gives up (NotConverging), which is the same situation as SLEPc's nconv < nev and is
			// something pyoomph's callers already cope with; deciding whether to widen ncv, retry or
			// accept a partial result belongs next to the other knobs in Python.
			nb::tuple out = nb::make_tuple(
				copy_eigenvalues(eigs.eigenvalues(), nconv),
				copy_eigenvectors(eigs.eigenvectors(), nconv, (Eigen::Index) n),
				(int64_t) nconv,
				(int64_t) eigs.num_iterations(),
				comp_info_name(eigs.info()),
				(int64_t) op.num_calls());
			return out;
		}

		// Two entry points rather than one taking a dtype flag: Scalar is a compile-time template
		// parameter of GenEigsSolver, so a flag would only re-dispatch to these same two
		// instantiations while making the Python-visible signature - and the generated stub - silent
		// about what the callback must accept and return.
		//
		// Note that the complex instantiation needs Spectra from master, not the v1.0.0 release; see
		// the comment on PYOOMPH_SPECTRA_REF in cmake/ThirdPartySpectra.cmake.
		nb::object spectra_eigensolve_real(nb::callable op, int64_t n, int64_t nev, int64_t ncv,
										   int64_t maxit, double tol, nb::object v0, const std::string &sortrule)
		{
			return eigensolve<double>(op, n, nev, ncv, maxit, tol, v0, sortrule);
		}

		nb::object spectra_eigensolve_complex(nb::callable op, int64_t n, int64_t nev, int64_t ncv,
											  int64_t maxit, double tol, nb::object v0, const std::string &sortrule)
		{
			return eigensolve<Complex>(op, n, nev, ncv, maxit, tol, v0, sortrule);
		}
	}
}

void PyReg_Spectra(nb::module_ &m)
{
	using namespace pyoomph::spectra_bindings;

	// Always Spectra's GenEigsSolver, never SymEigsSolver, even when J and M are symmetric:
	// C = (J - sigma*M)^{-1} M is self-adjoint in the M inner product, not the Euclidean one, so the
	// symmetric Lanczos driver would be solving a different problem. Symmetry is exploited in the
	// factorisation instead (Pardiso's mtype -2), on the Python side.
	const char *doc =
		"Run Spectra's implicitly restarted Arnoldi iteration on a matrix-free operator.\n\n"
		"op is called as op(x) with a contiguous 1d numpy array of %s and must return a\n"
		"contiguous 1d numpy array of the same dtype and length n. The array passed in is a view\n"
		"onto an internal buffer that is overwritten by the next call, so it must not be retained.\n\n"
		"Returns (eigenvalues, eigenvectors, nconv, niter, info, nmatvec), where eigenvalues is\n"
		"complex128 of length nconv, eigenvectors is complex128 of shape (nconv, n) with row i\n"
		"belonging to eigenvalue i, and info is one of 'successful', 'not_converging' or\n"
		"'numerical_issue'. A partial result (nconv < nev) is returned rather than raised.\n\n"
		"This solves the STANDARD problem op(x) = nu*x. Generalized problems are handled by the\n"
		"caller applying a shift-and-invert transform, see pyoomph.solvers.spectra.";
	std::string doc_real = doc, doc_complex = doc;
	doc_real.replace(doc_real.find("%s"), 2, "float64");
	doc_complex.replace(doc_complex.find("%s"), 2, "complex128");

	m.def("spectra_eigensolve_real", &spectra_eigensolve_real,
		  "op"_a, "n"_a, "nev"_a, "ncv"_a, "maxit"_a, "tol"_a, "v0"_a.none(), "sortrule"_a, doc_real.c_str());
	m.def("spectra_eigensolve_complex", &spectra_eigensolve_complex,
		  "op"_a, "n"_a, "nev"_a, "ncv"_a, "maxit"_a, "tol"_a, "v0"_a.none(), "sortrule"_a, doc_complex.c_str());
}
