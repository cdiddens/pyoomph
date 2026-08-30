#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
#
#  @section LICENSE
#
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================

# Tests for the Spectra eigensolver backend (pyoomph/solvers/spectra.py, idname "spectra").
#
# The backend exists for the platforms that have no PETSc/SLEPc - Windows above all - so what is
# protected here is the capability the ARPACK-based backends lack rather than the eigenvalues alone:
# it must TARGET an eigenvalue, real or complex, on a pencil whose mass matrix is singular and
# possibly complex. Every eigenvalue assertion is backed by the generalized residual
# ||J v - lambda M v|| computed from the matrices the solver itself returns, so a test cannot pass by
# agreeing with a second implementation that is wrong in the same way.
#
# Note that the shift-and-invert transform lives on the Python side (Spectra has no generalized
# solver at all), which is why the target/shift policy - _choose_sigma - gets a test of its own for
# both of the caller shapes that disagree about it.

import numpy
import scipy.linalg
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.equations.navier_stokes import NavierStokesEquations
import pyoomph._pyoomph_core as _pyoomph_core

pytestmark = pytest.mark.skipif(not getattr(_pyoomph_core, "has_spectra", False),
                                reason="this pyoomph build was compiled without Spectra")


# ---------------------------------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------------------------------

class _TransientDiffusionProblem(Problem):
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(1, v))

        eqs = _Eqs()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


class _NavierStokesProblem(Problem):
    # Navier-Stokes rather than Stokes so that there ARE time derivatives: the velocity rows carry
    # partial_t while the pressure rows (and the Dirichlet rows) do not, which makes M positive
    # semi-definite and SINGULAR. That is the case Spectra's own generalized solvers cannot take -
    # they require a positive definite B - and the reason the shift-invert transform is applied on the
    # python side instead. Plain Stokes has no mass matrix at all and is refused earlier.
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))
        eqs = NavierStokesEquations(dynamic_viscosity=1, mass_density=1)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        self.add_equations(eqs @ "domain")


class _PendulumProblem(Problem):
    # 2 dofs only: below Spectra's 1 <= nev <= n-2, so it must take the dense path.
    def define_problem(self):
        class _Eqs(ODEEquations):
            def define_fields(self):
                self.define_ode_variable("phi", "psi")

            def define_residuals(self):
                phi, phi_test = var_and_test("phi")
                psi, psi_test = var_and_test("psi")
                self.add_residual((partial_t(psi) + sin(phi)) * phi_test)
                self.add_residual((partial_t(phi) - psi) * psi_test)

        self.add_equations(_Eqs() @ "pendulum")


def _residuals(evals, evects, J, M):
    """Generalized residual ||J v - lambda M v|| / ||v|| for each returned pair.

    Must be called while the Problem context is still open: the J and M handed back by solve() are
    scipy views straight onto oomph-lib's own matrix buffers, which are freed with the problem.
    """
    out = []
    for lam, v in zip(evals, evects):
        out.append(numpy.linalg.norm(J @ v - lam * (M @ v)) / numpy.linalg.norm(v))
    return numpy.array(out)


# ---------------------------------------------------------------------------------------------------
# Real problems
# ---------------------------------------------------------------------------------------------------

def test_real_pencil_matches_scipy_and_satisfies_the_pencil():
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("scipy")
        p.initialise()
        p.solve()
        ev_scipy, _ = p.solve_eigenproblem(5, shift=0)

        p.set_eigensolver("spectra")
        ev_spectra, _ = p.solve_eigenproblem(5, shift=0)
        es = p.get_eigen_solver()
        evals, evects, J, M = es.solve(5, shift=0)

        a = numpy.sort_complex(ev_scipy)
        b = numpy.sort_complex(ev_spectra)
        scale = max(1.0, float(numpy.max(numpy.abs(a))))
        assert numpy.max(numpy.abs(a - b)) < 1e-8 * scale
        assert numpy.max(_residuals(evals, evects, J, M)) < 1e-8 * scale


def test_singular_mass_matrix_yields_only_finite_eigenvalues():
    with _NavierStokesProblem() as p:
        p.quiet()
        p.set_eigensolver("spectra")
        p.initialise()
        p.solve()
        es = p.get_eigen_solver()
        evals, evects, J, M = es.solve(5, shift=0)

        # A singular M has infinite eigenvalues. They must not reach the caller, and the finite ones
        # must genuinely satisfy the pencil rather than merely be finite.
        assert len(evals) > 0
        assert numpy.all(numpy.isfinite(evals))
        scale = max(1.0, float(numpy.max(numpy.abs(evals))))
        assert numpy.max(_residuals(evals, evects, J, M)) < 1e-7 * scale


def test_tiny_problem_takes_the_dense_path():
    # n=2 violates Spectra's 1 <= nev <= n-2; without the dense branch this raises from Spectra.
    with _PendulumProblem() as p:
        p.quiet()
        p.set_eigensolver("spectra")
        ode = p.get_ode("pendulum")
        ode.set_value(phi=0.01, psi=0)
        p.solve()
        evs, _ = p.solve_eigenproblem(2)
    assert numpy.max(numpy.abs(numpy.sort_complex(evs) - numpy.array([-1j, 1j]))) < 1e-8


# ---------------------------------------------------------------------------------------------------
# Targets - the capability the ARPACK backends do not have
# ---------------------------------------------------------------------------------------------------

def test_targets_are_advertised():
    # The bifurcation code branches on exactly these two, so they are part of the contract.
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("spectra")
        es = p.get_eigen_solver()
        assert es.supports_target() is True
        assert es.supports_complex_target() is True


def test_real_target_selects_the_nearest_eigenvalue():
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("spectra")
        p.initialise()
        p.solve()
        es = p.get_eigen_solver()
        all_evals, _, _, _ = es.solve(8, shift=0)
        # Aim between two eigenvalues of the interior of the computed set, so "nearest" is a real
        # choice and not just "the one at the end".
        wanted = sorted(all_evals.real)[len(all_evals) // 2]
        target = float(wanted) + 0.01 * abs(float(wanted))
        evals, evects, J, M = es.solve(3, shift=0.0, target=target)

        assert abs(evals[0] - wanted) < 1e-6 * max(1.0, abs(wanted))
        assert numpy.max(_residuals(evals, evects, J, M)) < 1e-7 * max(1.0, abs(wanted))


def _hopf_pencil(n=30, omega=1.7):
    """A real pencil with one known conjugate pair at +-1j*omega and everything else real-negative."""
    import scipy.sparse as sp
    J = numpy.zeros((n, n))
    J[0, 1] = -omega
    J[1, 0] = omega
    for i in range(2, n):
        J[i, i] = -(0.5 + 0.3 * i)
    M = numpy.eye(n)
    M[n - 1, n - 1] = 0.0          # singular, as a pyoomph mass matrix is
    J[n - 1, n - 1] = -1.0
    return sp.csr_matrix(J), sp.csr_matrix(M), 1j * omega


def test_complex_target_hits_the_hopf_pair():
    # The shape get_hopf_eigenvector uses: the target IS an eigenvalue (so J-target*M is singular) and
    # the shift is deliberately nudged just off it. There the SHIFT must be respected.
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("spectra")
        p.initialise()
        es = p.get_eigen_solver()
        J, M, lam = _hopf_pencil()
        evals, evects, Jr, Mr = es.solve(1, shift=(1j + 1e-7) * abs(lam), target=lam,
                                         custom_J_and_M=(J, M), sort=False)
        assert len(evals) >= 1
        assert numpy.min(numpy.abs(evals - lam)) < 1e-7 * abs(lam)
        assert numpy.max(_residuals(evals, evects, Jr, Mr)) < 1e-8


def test_uninformative_shift_does_not_beat_the_target():
    # The shape NormalFormCalculator.get_left_eigenvector uses: shift=1e-7 with a target far from it.
    # Shift-inverting at 1e-7 converges to the modes nearest zero, and the caller then rejects the
    # answer as belonging to a different mode - so the TARGET must win here.
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("spectra")
        p.initialise()
        es = p.get_eigen_solver()
        J, M, lam = _hopf_pencil()
        evals, evects, Jr, Mr = es.solve(2, shift=1e-7, target=lam, custom_J_and_M=(J, M))
    closest = numpy.argmin(numpy.abs(evals - lam))
    assert abs(evals[closest] - lam) < 1e-6 * abs(lam)


# ---------------------------------------------------------------------------------------------------
# Factorisation backends
# ---------------------------------------------------------------------------------------------------

def test_complex_pencil_with_a_complex_target():
    # A genuinely complex, non-Hermitian pencil with a singular mass matrix - the shape the azimuthal
    # and normal-mode stability problems produce - solved against a complex target. This is the one
    # capability no other pyoomph backend has without a separately built complex PETSc.
    import scipy.sparse as sp
    n = 30
    rng = numpy.random.default_rng(11)
    Jd = numpy.diag(-(0.5 + 0.3 * numpy.arange(n)) + 1j * (0.2 + 0.15 * numpy.arange(n)))
    Jd += 0.1 * rng.standard_normal((n, n)) + 0.1j * rng.standard_normal((n, n))
    Md = numpy.eye(n) + 0.3j * numpy.eye(n)
    Md[n - 1, n - 1] = 0.0        # singular
    J, M = sp.csr_matrix(Jd), sp.csr_matrix(Md)

    # Reference: a dense eigendecomposition of the same pencil, an implementation Spectra shares
    # nothing with.
    ref = scipy.linalg.eig(Jd, b=Md, right=False)
    ref = ref[numpy.isfinite(ref)]
    wanted = ref[numpy.argsort(numpy.abs(ref - (-2.0 + 1.0j)))[0]]

    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("spectra")
        p.initialise()
        es = p.get_eigen_solver()
        evals, evects, Jr, Mr = es.solve(3, shift=0.0, target=complex(wanted),
                                         custom_J_and_M=(J, M))
        assert numpy.iscomplexobj(numpy.asarray(Jr.todense()))
        assert abs(evals[0] - wanted) < 1e-7 * max(1.0, abs(wanted))
        assert numpy.max(_residuals(evals, evects, Jr, Mr)) < 1e-8 * max(1.0, abs(wanted))


def test_superlu_path_matches_the_pardiso_one():
    # On a Windows wheel MKL is an optional dependency, so SuperLU is the live path there rather than
    # a fallback. Both must give the same spectrum.
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("spectra")
        p.initialise()
        p.solve()
        es = p.get_eigen_solver()
        es.use_pardiso = True
        ev_pardiso, _, _, _ = es.solve(5, shift=0)
        es.use_pardiso = False
        ev_splu, _, _, _ = es.solve(5, shift=0)

        a, b = numpy.sort_complex(ev_pardiso), numpy.sort_complex(ev_splu)
        scale = max(1.0, float(numpy.max(numpy.abs(a))))
        assert numpy.max(numpy.abs(a - b)) < 1e-8 * scale


def test_left_eigenvectors_are_still_refused():
    # Unimplemented in every backend; pinned so it is not mistaken for a Spectra-specific gap.
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.set_eigensolver("spectra")
        p.initialise()
        es = p.get_eigen_solver()
        with pytest.raises(RuntimeError):
            es.solve(2, shift=0, with_left_eigenvectors=True)
