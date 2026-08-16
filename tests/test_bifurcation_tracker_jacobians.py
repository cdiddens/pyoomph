#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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

# Every augmented assembly handler in src/bifurcation.cpp must agree with ITSELF: its Jacobian and
# its parameter derivative have to be the exact derivatives of its own get_residuals(). They kept
# not being, in one recurring way.
#
# pyoomph's eigen rows are (J + lambda*M)V = 0, for lambda*M*V + J*V = 0 and a perturbation
# ~exp(lambda*t). Upstream oomph-lib's HopfHandler writes the Omega*M block the other way round.
# pyoomph flipped the residuals but several derivative blocks inherited from oomph-lib, or written by
# hand alongside them, kept the old signs. Each such block multiplies something that is zero in the
# common case, so nothing complained:
#
#   - Omega*dM/dp is zero unless the MASS MATRIX DEPENDS ON THE PARAMETER
#     (MyHopfHandler::get_dresiduals_dparameter had oomph-lib's signs)
#   - M_imag is zero unless the m!=0 formulation is RESIDUAL-STABILIZED (fixed in a516c1b)
#   - the lambda*dM/dU blocks only exist during EIGENBRANCH tracking
#   - the azimuthal m-mode parameter derivative was overwritten by the base-mode one, so it is only
#     wrong where the m!=0 Jacobian DIFFERS from the axisymmetric one
#
# So the problems below are built to make exactly those terms nonzero: a mass matrix that depends
# both on a parameter and on the solution, and a stabilized axisymmetric formulation at m=1.
#
# The instrument is the same one that found the a516c1b bug: finite-difference EVERY column of the
# augmented Jacobian, not one direction. The eigenvector blocks are invisible to a single directional
# probe. This works from Python because build_augmented_dofs pushes pointers into the handler's own
# eigenvector storage, so the whole augmented dof vector is reachable through set_current_dofs().
#
# All cheap: 2x2 meshes, no eigensolver is ever called - the eigenvector guesses are made up, which
# is legitimate because residual/Jacobian consistency does not care where the state came from.

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


# ----------------------------------------------------------------------------------------------
# the finite-difference instrument
# ----------------------------------------------------------------------------------------------

def _fd_all_columns(p, nbase, eps=1e-6):
    """Compare every column of the augmented Jacobian with a central difference of the residual.

    Returns (worst relative deviation, label of the worst column). Restores the dofs it perturbs.
    """
    naug = p.ndof()
    x0 = numpy.array(p.get_current_dofs()[0])
    J = numpy.asarray(p.assemble_jacobian(with_residual=False).todense())

    def label(i):
        if i < nbase:
            return "base column %d" % i
        k, r = divmod(i - nbase, nbase)
        if (k + 1) * nbase <= naug - nbase:      # a full eigenvector block
            return "eigenvector block %d, column %d" % (k, r)
        return "scalar column %d (of %d)" % (i, naug)

    worst, worst_at = 0.0, "none"
    for j in range(naug):
        x = x0.copy(); x[j] += eps
        p.set_current_dofs(x); rp = numpy.array(p.get_residuals())
        x = x0.copy(); x[j] -= eps
        p.set_current_dofs(x); rm = numpy.array(p.get_residuals())
        col_fd = (rp - rm) / (2 * eps)
        scale = max(numpy.max(numpy.abs(col_fd)), numpy.max(numpy.abs(J[:, j])), 1e-30)
        rel = numpy.max(numpy.abs(J[:, j] - col_fd)) / scale
        if rel > worst:
            worst, worst_at = rel, label(j)
    p.set_current_dofs(x0)
    return worst, worst_at


def _fd_parameter_derivative(p, param, eps=1e-6):
    """Compare the handler's dResidual/dparameter with a central difference over the parameter.

    This is the ONLY thing that reaches get_dresiduals_dparameter: the augmented Jacobian's own
    parameter column is built by a different code path in the analytic-Hessian branch, so a
    column-by-column FD of the Jacobian does not cover it.
    """
    ana = numpy.array(p.get_parameter_derivative(param))
    par = p.get_global_parameter(param)
    p0 = par.value
    par.value = p0 + eps
    rp = numpy.array(p.get_residuals())
    par.value = p0 - eps
    rm = numpy.array(p.get_residuals())
    par.value = p0
    fd = (rp - rm) / (2 * eps)
    scale = max(numpy.max(numpy.abs(fd)), numpy.max(numpy.abs(ana)), 1e-30)
    return numpy.max(numpy.abs(ana - fd)) / scale


# ----------------------------------------------------------------------------------------------
# a Cartesian oscillator whose mass matrix depends on BOTH a parameter and the solution
# ----------------------------------------------------------------------------------------------

class _OscillatorEqs(Equations):
    """u' = w, w' = -A*u - damping, with the time-derivative terms multiplied by (B + u^2/2).

    The (B + u^2/2) factor is the point of the whole class: it makes dM/dB nonzero (which is what
    exposes a wrong Omega*dM/dp sign) and dM/dU nonzero (which is what exposes a wrong lambda*dM/dU
    block during eigenbranch tracking). A constant mass matrix hides both.
    """

    def define_fields(self):
        self.define_scalar_field("u", "C2")
        self.define_scalar_field("w", "C2")

    def define_residuals(self):
        u, vu = var_and_test("u")
        w, vw = var_and_test("w")
        pr = self.get_current_code_generator().get_problem()
        A, B = pr.A, pr.B
        mass = B + 0.5 * u ** 2
        self.add_residual(weak(mass * partial_t(u) - w, vu) + 0.1 * weak(grad(u), grad(vu)))
        self.add_residual(weak(mass * partial_t(w) + A * u + 0.3 * u ** 2 * w, vw)
                          + 0.1 * weak(grad(w), grad(vw)))


class _OscillatorProblem(Problem):
    def define_problem(self):
        self.A = self.define_global_parameter(A=1.0)
        self.B = self.define_global_parameter(B=1.5)
        self.add_mesh(RectangularQuadMesh(N=2, size=[1, 1]))
        eqs = _OscillatorEqs()
        eqs += DirichletBC(u=0, w=0) @ "bottom"
        self.add_equations(eqs @ "domain")


def _perturbed_oscillator(analytic_hessian=True, seed=7):
    """A problem at a random-ish state with a random complex eigenvector guess."""
    p = _OscillatorProblem()
    p.setup_for_stability_analysis(analytic_hessian=analytic_hessian)
    p.initialise()
    n = p.ndof()
    rng = numpy.random.default_rng(seed)
    p.set_current_dofs(numpy.array(p.get_current_dofs()[0]) + 0.05 * rng.standard_normal(n))
    V = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    V /= numpy.linalg.norm(V)
    return p, n, V


# ----------------------------------------------------------------------------------------------
# MyHopfHandler
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("analytic_hessian", [True, False])
def test_hopf_augmented_jacobian_is_exact(analytic_hessian):
    """Both branches of MyHopfHandler::get_jacobian, every augmented column.

    Covers the Omega columns and the (Phi,Psi) blocks, which are where the oomph-lib signs would
    survive. The analytic branch additionally exercises the Hessian-vector products.
    """
    p, n, V = _perturbed_oscillator(analytic_hessian=analytic_hessian)
    with p:
        p.activate_bifurcation_tracking("A", bifurcation_type="hopf", eigenvector=V, omega=0.4)
        worst, where = _fd_all_columns(p, n)
        assert worst < 1e-5, "Hopf augmented Jacobian (analytic_hessian=%s): %.3e at %s" % (
            analytic_hessian, worst, where)


@pytest.mark.parametrize("param", ["A", "B"])
def test_hopf_parameter_derivative_matches_its_own_residual(param):
    """MyHopfHandler::get_dresiduals_dparameter, which had oomph-lib's Omega*dM/dp signs.

    "B" is the case that matters: it enters the mass matrix only, so the eigen rows of dR/dB are
    made up entirely of the Omega*dM/dB terms and a flipped sign is a 200% error. "A" enters the
    Jacobian only and passed all along - it is here so that a future edit cannot fix one and break
    the other. This function feeds arclength continuation of a tracked Hopf point in a second
    parameter, so a wrong sign there moves the continuation path.
    """
    p, n, V = _perturbed_oscillator()
    with p:
        p.activate_bifurcation_tracking("A", bifurcation_type="hopf", eigenvector=V, omega=0.4)
        rel = _fd_parameter_derivative(p, param)
        assert rel < 1e-5, "Hopf dResidual/d%s disagrees with a finite difference: %.3e" % (param, rel)


def test_hopf_eigenbranch_jacobian_is_exact():
    """Complex eigenbranch continuation: the augmented system grows a lambda*M block and a lambda column.

    The lambda*dM/dU blocks were written as dMdU_Eig(i,j)*Eig_local[...], but dMdU_Eig is ALREADY
    contracted with the eigenvector - rows [0,n) hold dM/dU.Phi and rows [n,2n) hold dM/dU.Psi. So
    they carried a spurious extra factor, and the Psi equation read the Phi rows. Both are invisible
    unless the mass matrix depends on the solution, which is why _OscillatorEqs has u^2 in it.
    """
    p, n, V = _perturbed_oscillator()
    with p:
        p.activate_bifurcation_tracking(None, bifurcation_type="complex", eigenvector=V,
                                        eigenvalue_for_branch_tracking=-0.2 + 0.4j)
        worst, where = _fd_all_columns(p, n)
        assert worst < 1e-5, "Hopf eigenbranch augmented Jacobian: %.3e at %s" % (worst, where)


# ----------------------------------------------------------------------------------------------
# AzimuthalSymmetryBreakingHandler
# ----------------------------------------------------------------------------------------------

class _StabilizedAxisymProblem(Problem):
    """Stabilized Navier-Stokes on an axisymmetric moving mesh: the m!=0 mass matrix is complex.

    SUPG/PSPG put the residual into the test function, so the i*m/r factors of the azimuthal
    expansion reach the partial_t terms and M_imag is nonzero. The viscosity is the parameter, so the
    m!=0 Jacobian's parameter derivative also differs from the axisymmetric one.
    """

    def define_problem(self):
        from pyoomph.equations.stabilized_ns import StabilizedNavierStokes
        self.A = self.define_global_parameter(A=1.0)
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(RectangularQuadMesh(N=2, size=[1, 1], lower_left=[1, 0]))
        eqs = StabilizedNavierStokes(space="C1C1", stabilization="SUPGPSPGLSIC",
                                     viscous_form="stress", dynamic_viscosity=self.A,
                                     mass_density=1)
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "bottom"
        eqs += LaplaceSmoothedMesh()
        eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "bottom"
        self.add_equations(eqs @ "domain")


def _perturbed_azimuthal(analytic_hessian, seed=11):
    p = _StabilizedAxisymProblem()
    p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=analytic_hessian)
    p.initialise()
    n = p.ndof()
    rng = numpy.random.default_rng(seed)
    p.set_current_dofs(numpy.array(p.get_current_dofs()[0]) + 0.05 * rng.standard_normal(n))
    V = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    V /= numpy.linalg.norm(V)
    return p, n, V


def test_azimuthal_augmented_jacobian_is_exact_without_analytic_hessian():
    """The FD-Hessian branch of AzimuthalSymmetryBreakingHandler::get_jacobian.

    a516c1b checked the analytic branch. This branch had the Omega column of the REAL eigen row as a
    "+=" where the residual's -Omega*(M_real*Vi + M_imag*Vr) demands a "-=". Unlike the M_imag sign
    fixed alongside it, the M_real term there never vanishes, so this one is wrong for every
    azimuthal run with analytic Hessians switched off - which is what every tutorial that only scans
    eigenvalues uses.
    """
    p, n, V = _perturbed_azimuthal(analytic_hessian=False)
    with p:
        p.activate_bifurcation_tracking("A", bifurcation_type="azimuthal", azimuthal_mode=1,
                                        eigenvector=V, omega=0.3)
        worst, where = _fd_all_columns(p, n)
        assert worst < 1e-5, "azimuthal FD-Hessian augmented Jacobian: %.3e at %s" % (worst, where)


@pytest.mark.parametrize("analytic_hessian", [True, False])
def test_azimuthal_parameter_derivative_matches_its_own_residual(analytic_hessian):
    """AzimuthalSymmetryBreakingHandler::get_dresiduals_dparameter at m=1.

    It assembles the m!=0 real part, then the imaginary part, then the base state - and the base
    call used to write its matrices into the buffers already holding the m!=0 real ones, so the
    eigen rows were built from the AXISYMMETRIC parameter derivative. Only visible where the m!=0
    Jacobian differs from the base one, i.e. at m!=0 with a parameter inside the operator.
    """
    p, n, V = _perturbed_azimuthal(analytic_hessian=analytic_hessian)
    with p:
        p.activate_bifurcation_tracking("A", bifurcation_type="azimuthal", azimuthal_mode=1,
                                        eigenvector=V, omega=0.3)
        rel = _fd_parameter_derivative(p, "A")
        assert rel < 1e-5, "azimuthal dResidual/dA (analytic_hessian=%s): %.3e" % (
            analytic_hessian, rel)


def test_azimuthal_eigenbranch_refuses_without_an_analytic_hessian():
    """Refusing beats assembling a Jacobian that does not match the residual.

    The FD-Hessian branch never grew the lambda*M blocks or the lambda column that eigenbranch
    tracking needs, while get_residuals() does add them. Its refusal had been commented out, so the
    Newton step was simply taken with an inconsistent Jacobian. MyHopfHandler and MyFoldHandler both
    throw in the same situation.
    """
    p, n, V = _perturbed_azimuthal(analytic_hessian=False)
    with p:
        # the refusal may come from the activation itself or from the first assembly, depending on
        # whether the handler builds a Jacobian while setting up - either is fine, silence is not
        with pytest.raises(Exception, match="(?i)analytic(al)? hessian"):
            p.activate_bifurcation_tracking(None, bifurcation_type="normal_mode", azimuthal_mode=1,
                                            eigenvector=V,
                                            eigenvalue_for_branch_tracking=-0.2 + 0.3j)
            p.assemble_jacobian(with_residual=False)
