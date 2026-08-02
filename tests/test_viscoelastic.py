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

# Constitutive-equation validation for pyoomph.equations.viscoelastic.
#
# Every test here drives the constitutive equation with an IMPOSED, spatially uniform velocity
# gradient and no momentum coupling, so the conformation tensor is homogeneous and its evolution is
# an ODE that can be compared against an exact solution (Oldroyd-B) or against a reference
# integration of the conformation equation in numpy (all other models). That isolates what is
# actually under test -- the log-conformation transformation, the eigenvalue decompositions in
# pyoomph.expressions.tensor_funcs and their analytic Jacobians -- from mesh resolution, momentum
# coupling and stabilisation, none of which are involved.
#
# The sharpest of these is the start-up of simple shear: the first normal stress difference builds
# up as 1-exp(-s)(1+s) rather than the 1-exp(-s) of the shear stress, and getting that transient
# right requires the sign of the rotational part Omega of the Fattal-Kupferman decomposition to be
# consistent with pyoomph's grad(u)[i,j]=d(u_i)/d(x_j) convention. A flipped sign still reproduces
# the steady state, so the steady tests alone would not catch it.

import math

import numpy
import pytest

from pyoomph import Problem, Equations, DirichletBC
from pyoomph.expressions import var, vector, matrix, matproduct
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.equations.viscoelastic import (ViscoelasticEquations, OldroydB, Giesekus, PTT, FENE_CR, FENE_P,
                                            symmetric_2x2_matrix_log, oldroyd_b_shear_conformation)
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _ImposedVelocity(Equations):
    """Substitutes a prescribed expression for the velocity field, so no momentum equation is solved."""

    def __init__(self, expression):
        super().__init__()
        self.expression = expression

    def define_fields(self):
        self.define_field_by_substitution("velocity", self.expression, also_on_interface=True)


class _HomogeneousProblem(Problem):
    def __init__(self, velocity, model, formulation, relaxation_time, axisymmetric, stabilization=None):
        super().__init__()
        self.stabilization = stabilization
        self.velocity = velocity
        self.model = model
        self.formulation = formulation
        self.relaxation_time = relaxation_time
        self.axisymmetric = axisymmetric

    def define_problem(self):
        if self.axisymmetric:
            self.set_coordinate_system("axisymmetric")
            # Away from the axis: the axisymmetric decomposition divides by the radius, and r=0 is a
            # boundary of the domain, not a point where the constitutive equation has to hold.
            self.add_mesh(RectangularQuadMesh(N=1, size=[1, 1], lower_left=[1, 0]))
        else:
            self.add_mesh(RectangularQuadMesh(N=1))
        eqs = _ImposedVelocity(self.velocity)
        eqs += ViscoelasticEquations(model=self.model, relaxation_time=self.relaxation_time,
                                     polymer_viscosity=1, formulation=self.formulation,
                                     add_polymer_stress_to_momentum=False, space="C1",
                                     stabilization=self.stabilization)
        self.add_equations(eqs @ "domain")


def _run(tmp_path, velocity, model=None, formulation="log-conf", relaxation_time=1.0,
         tend=2.0, dt=0.002, axisymmetric=False, stabilization=None):
    """Integrates the constitutive equation under an imposed velocity and returns C at one node."""
    model = model if model is not None else OldroydB()
    with _HomogeneousProblem(velocity, model, formulation, relaxation_time, axisymmetric,
                             stabilization) as problem:
        problem.set_output_directory(str(tmp_path))
        problem.initialise()
        problem.run(tend, startstep=dt, maxstep=dt, temporal_error=None, outstep=False)
        mesh = problem.get_mesh("domain")
        indices = mesh.get_nodal_field_indices()
        node = next(iter(mesh.nodes()))
        prefix = "log_conformation" if formulation == "log-conf" else "conformation"

        def value(component, default):
            name = prefix + "_" + component
            return node.value(indices[name]) if name in indices else default

        # The out-of-plane component is only an unknown where the model needs it; where it is not,
        # it sits at its rest value (Psi_zz=0, i.e. C_zz=1).
        out_of_plane = "aa" if axisymmetric else "zz"
        rest = 0.0 if formulation == "log-conf" else 1.0
        field = numpy.diag([0.0, 0.0, value(out_of_plane, rest)])
        field[0, 0], field[1, 1] = value("xx", rest), value("yy", rest)
        field[0, 1] = field[1, 0] = value("xy", 0.0)
        if formulation == "log-conf":
            eigenvalues, eigenvectors = numpy.linalg.eigh(field)
            return eigenvectors @ numpy.diag(numpy.exp(eigenvalues)) @ eigenvectors.T
        return field


# ----------------------------------------------------------------------------------------------
# Reference solutions
# ----------------------------------------------------------------------------------------------

def _relaxation(model, C):
    """
    g(C) of the model, assembled in numpy from C*(C^-1 g(C)).

    Every model's log_relaxation_matrix is a linear combination of the identity, C and C^-1 with
    scalar coefficients, so it evaluates on numpy arrays unchanged. That the two forms a model
    declares really are the same function is checked separately, in
    test_model_relaxation_forms_agree.
    """
    log_form = model.log_relaxation_matrix(C, numpy.linalg.inv(C), float(numpy.trace(C)), numpy.identity(3))
    return C @ log_form


def _reference_conformation(model, L, relaxation_time, tend, steps=5000):
    """
    Integrates dC/dt = L*C + C*L^t - g(C)/lambda from C=identity with classical RK4.

    This deliberately does NOT use the log-conformation representation: it is the plain
    conformation equation in numpy, so agreement with the pyoomph result confirms that the log
    transform, the eigenvalue decomposition, the finite element assembly and the time stepping are
    all consistent with the model they claim to represent.
    """
    def rhs(C):
        return L @ C + C @ L.T - _relaxation(model, C) / relaxation_time

    C = numpy.identity(3)
    h = tend / steps
    for _ in range(steps):
        k1 = rhs(C)
        k2 = rhs(C + 0.5 * h * k1)
        k3 = rhs(C + 0.5 * h * k2)
        k4 = rhs(C + h * k3)
        C = C + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return C


def _oldroyd_b_startup_shear(shear_rate, relaxation_time, t):
    """Exact Oldroyd-B solution for simple shear switched on at t=0 from equilibrium."""
    W = relaxation_time * shear_rate
    s = t / relaxation_time
    C = numpy.identity(3)
    C[0, 0] = 1 + 2 * W ** 2 * (1 - math.exp(-s) * (1 + s))
    C[0, 1] = C[1, 0] = W * (1 - math.exp(-s))
    return C


# ----------------------------------------------------------------------------------------------
# Oldroyd-B against its closed-form solutions
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("formulation", ["log-conf", "conformation"])
def test_startup_of_simple_shear(tmp_path, formulation):
    """The transient that pins down the sign of the rotational part of the decomposition."""
    shear_rate, relaxation_time, tend = 1.0, 1.0, 2.0
    C = _run(tmp_path, vector(shear_rate * var("coordinate_y"), 0), formulation=formulation,
             relaxation_time=relaxation_time, tend=tend)
    reference = _oldroyd_b_startup_shear(shear_rate, relaxation_time, tend)
    assert numpy.max(numpy.abs(C - reference)) < 5e-6


def test_startup_of_simple_shear_converges_in_time(tmp_path):
    """What is left over above is BDF2 truncation error, so it must fall like dt^2."""
    shear_rate, relaxation_time, tend = 1.0, 1.0, 2.0
    reference = _oldroyd_b_startup_shear(shear_rate, relaxation_time, tend)
    errors = []
    for dt in (0.004, 0.002, 0.001):
        C = _run(tmp_path, vector(shear_rate * var("coordinate_y"), 0),
                 relaxation_time=relaxation_time, tend=tend, dt=dt)
        errors.append(numpy.max(numpy.abs(C - reference)))
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert coarse / fine > 3.0, "expected second order convergence, got " + str(errors)


@pytest.mark.parametrize("weissenberg", [0.5, 2.0, 5.0])
def test_steady_simple_shear(tmp_path, weissenberg):
    """C=[[1+2Wi^2, Wi],[Wi, 1]], the standard Oldroyd-B viscometric solution."""
    relaxation_time = 1.0
    shear_rate = weissenberg / relaxation_time
    # Long enough that the exp(-t/lambda) transient is below the tolerance.
    C = _run(tmp_path, vector(shear_rate * var("coordinate_y"), 0),
             relaxation_time=relaxation_time, tend=25.0, dt=0.01)
    reference = numpy.identity(3)
    reference[0, 0] = 1 + 2 * weissenberg ** 2
    reference[0, 1] = reference[1, 0] = weissenberg
    assert numpy.max(numpy.abs(C - reference)) < 1e-5 * max(1.0, reference[0, 0])


def test_steady_planar_extension(tmp_path):
    """
    C_xx=1/(1-2*lambda*rate), C_yy=1/(1+2*lambda*rate).

    Unlike shear this keeps the eigenframe fixed and the off-diagonal at zero, which exercises the
    already-diagonal branch of DiagonalizeSymmetricTensor rather than the general one.
    """
    rate, relaxation_time = 0.2, 1.0
    velocity = vector(rate * var("coordinate_x"), -rate * var("coordinate_y"))
    C = _run(tmp_path, velocity, relaxation_time=relaxation_time, tend=25.0, dt=0.01)
    reference = numpy.diag([1 / (1 - 2 * relaxation_time * rate), 1 / (1 + 2 * relaxation_time * rate), 1.0])
    assert numpy.max(numpy.abs(C - reference)) < 1e-5


def test_supg_survives_a_stagnation_point(tmp_path):
    """
    SUPG with a velocity field that vanishes somewhere in the domain.

    Planar extension u=(rate*x, -rate*y) on the unit square puts an exact stagnation point at the
    corner (0,0). The intrinsic time tau has to stay differentiable there: writing it via
    square_root(dot(u,u)) and squaring leaves d|u|/du = u/|u| in the Jacobian, which is 0/0 at a
    stagnation point and sent the first Newton step to 1e105 on the cylinder benchmark. The channel
    flow of test_supg_does_not_move_a_converged_solution cannot catch that, because there u.grad(w)
    vanishes wherever u does and the singular factor is multiplied by zero.
    """
    rate, relaxation_time = 0.2, 1.0
    velocity = vector(rate * var("coordinate_x"), -rate * var("coordinate_y"))
    C = _run(tmp_path, velocity, relaxation_time=relaxation_time, tend=25.0, dt=0.01,
             stabilization="SUPG")
    reference = numpy.diag([1 / (1 - 2 * relaxation_time * rate), 1 / (1 + 2 * relaxation_time * rate), 1.0])
    assert numpy.max(numpy.abs(C - reference)) < 1e-5


def test_rest_state_is_exactly_preserved(tmp_path):
    """
    Without flow the solution must stay at C=identity.

    This is the fully degenerate case -- all eigenvalues of C coincide -- so it runs entirely
    through the epsilon branch of the log-conformation decomposition, where the returned Jacobian is
    truncated. The residual there is still exact, so the answer must be exact too.
    """
    C = _run(tmp_path, vector(0, 0), tend=5.0, dt=0.05)
    assert numpy.max(numpy.abs(C - numpy.identity(3))) < 1e-12


# ----------------------------------------------------------------------------------------------
# Axisymmetric coordinates
# ----------------------------------------------------------------------------------------------

def test_axisymmetric_uniaxial_extension(tmp_path):
    """
    Uniaxial extension u = (-rate*r/2, rate*z), where the azimuthal component of the velocity
    gradient is nonzero. That is the entry the planar case never sees, and the only reason the
    axisymmetric decomposition in tensor_funcs is a separate class.
    """
    rate, relaxation_time = 0.2, 1.0
    velocity = vector(-rate * var("coordinate_x") / 2, rate * var("coordinate_y"))
    C = _run(tmp_path, velocity, relaxation_time=relaxation_time, tend=25.0, dt=0.01, axisymmetric=True)
    # Steady Oldroyd-B in a diagonal velocity gradient: C_ii = 1/(1-2*lambda*L_ii).
    reference = numpy.diag([1 / (1 + relaxation_time * rate),
                            1 / (1 - 2 * relaxation_time * rate),
                            1 / (1 + relaxation_time * rate)])
    assert numpy.max(numpy.abs(C - reference)) < 1e-5


def test_axisymmetric_shear_matches_the_planar_result(tmp_path):
    """
    An axial velocity varying with radius, u_z = rate*r, is the one shear that leaves the azimuthal
    direction alone: u_r stays zero, so the azimuthal velocity gradient u_r/r vanishes and the
    planar start-up solution must come back, with r and z playing the roles of y and x. (The
    superficially similar u_r = rate*z does NOT qualify -- it moves material radially outwards and
    therefore stretches it azimuthally.)
    """
    shear_rate, relaxation_time, tend = 1.0, 1.0, 2.0
    C = _run(tmp_path, vector(0, shear_rate * var("coordinate_x")),
             relaxation_time=relaxation_time, tend=tend, axisymmetric=True)
    planar = _oldroyd_b_startup_shear(shear_rate, relaxation_time, tend)
    # Swap the two in-plane directions: here it is C_zz that grows, not C_rr.
    reference = numpy.identity(3)
    reference[1, 1] = planar[0, 0]
    reference[0, 1] = reference[1, 0] = planar[0, 1]
    assert numpy.max(numpy.abs(C - reference)) < 5e-6


# ----------------------------------------------------------------------------------------------
# The nonlinear models, against a reference integration of the conformation equation
# ----------------------------------------------------------------------------------------------

_MODELS = [
    ("giesekus", Giesekus(alpha=0.3)),
    ("ptt-linear", PTT(epsilon=0.25, kind="linear")),
    ("ptt-exponential", PTT(epsilon=0.25, kind="exponential")),
    ("fene-cr", FENE_CR(L=4)),
    ("fene-p", FENE_P(L=4)),
]


@pytest.mark.parametrize("name,model", _MODELS, ids=[n for n, _ in _MODELS])
def test_nonlinear_models_in_shear(tmp_path, name, model):
    shear_rate, relaxation_time, tend = 2.0, 1.0, 3.0
    L = numpy.zeros((3, 3))
    L[0, 1] = shear_rate
    C = _run(tmp_path, vector(shear_rate * var("coordinate_y"), 0), model=model,
             relaxation_time=relaxation_time, tend=tend, dt=0.001)
    reference = _reference_conformation(model, L, relaxation_time, tend)
    assert numpy.max(numpy.abs(C - reference)) < 1e-4 * max(1.0, numpy.max(numpy.abs(reference)))


@pytest.mark.parametrize("name,model", _MODELS, ids=[n for n, _ in _MODELS])
def test_nonlinear_models_in_planar_extension(tmp_path, name, model):
    """
    Extension is where the out-of-plane component matters: for FENE-P the trace grows, so f differs
    from its equilibrium value a and C_zz is dragged away from 1. The models whose relaxation
    function vanishes at an eigenvalue of 1 must instead keep C_zz at exactly 1.
    """
    rate, relaxation_time, tend = 0.4, 1.0, 3.0
    L = numpy.diag([rate, -rate, 0.0])
    velocity = vector(rate * var("coordinate_x"), -rate * var("coordinate_y"))
    C = _run(tmp_path, velocity, model=model, relaxation_time=relaxation_time, tend=tend, dt=0.001)
    reference = _reference_conformation(model, L, relaxation_time, tend)
    assert numpy.max(numpy.abs(C - reference)) < 1e-4 * max(1.0, numpy.max(numpy.abs(reference)))
    if isinstance(model, FENE_P):
        assert abs(C[2, 2] - 1.0) > 1e-3, "FENE-P should not leave C_zz at its rest value here"
    else:
        assert abs(C[2, 2] - 1.0) < 1e-10


def test_fene_trace_stays_below_the_extensibility_limit(tmp_path):
    """The point of the FENE models: tr(C) < L^2 even in a strong extensional flow."""
    L_extensibility = 4.0
    rate = 5.0  # far beyond the Oldroyd-B singularity at rate=0.5
    velocity = vector(rate * var("coordinate_x"), -rate * var("coordinate_y"))
    C = _run(tmp_path, velocity, model=FENE_CR(L=L_extensibility), tend=10.0, dt=0.002)
    assert numpy.trace(C) < L_extensibility ** 2


# ----------------------------------------------------------------------------------------------
# Coupled to the momentum equation: planar Poiseuille flow
# ----------------------------------------------------------------------------------------------

# Everything above drives the constitutive equation with an imposed velocity, so none of it can see
# the polymer stress that is handed back to the momentum equation. Poiseuille flow does: for
# Oldroyd-B the polymer contributes tau_xy = eta_p*shear_rate exactly, so the total shear stress is
# (eta_s+eta_p)*shear_rate and the velocity profile is the Newtonian parabola formed with the TOTAL
# viscosity. Getting the coupling sign or magnitude wrong changes the effective viscosity and thus
# the centreline velocity, which is what the test measures.

_ETA_S, _ETA_P, _LAMBDA, _FORCE = 0.3, 0.7, 1.0, 1.0
_ETA_0 = _ETA_S + _ETA_P


class _PoiseuilleProblem(Problem):
    def __init__(self, elements_across, stabilization=None):
        super().__init__()
        self.elements_across = elements_across
        self.stabilization = stabilization

    def define_problem(self):
        # Periodic in the flow direction, so the fully developed profile is the exact solution of
        # the discrete problem too and no artificial inflow or outflow condition is needed.
        self.add_mesh(RectangularQuadMesh(N=[2, self.elements_across], size=[0.25, 1.0],
                                          periodic=[True, False]))
        navier_stokes = NavierStokesEquations(dynamic_viscosity=_ETA_S, mass_density=1,
                                              bulkforce=vector(_FORCE, 0))
        eqs = navier_stokes
        eqs += ViscoelasticEquations(model=OldroydB(), relaxation_time=_LAMBDA,
                                     polymer_viscosity=_ETA_P, space="C2",
                                     stabilization=self.stabilization)
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "top"
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "bottom"
        eqs += navier_stokes.create_pressure_fixation(value=0) @ "bottom"
        self.add_equations(eqs @ "domain")


def _poiseuille_errors(tmp_path, elements_across, stabilization=None):
    with _PoiseuilleProblem(elements_across, stabilization) as problem:
        problem.set_output_directory(str(tmp_path / ("poiseuille" + str(elements_across)
                                                     + str(stabilization))))
        problem.initialise()
        # A stationary solve straight from the rest state C=identity, i.e. from the fully
        # degenerate point of the log-conformation decomposition. That it converges at all is the
        # reason the conformation tensor is built from SymmetricMatrixExponential rather than from
        # the eigendecomposition; see ViscoelasticEquations._in_plane_exponential.
        problem.solve()
        mesh = problem.get_mesh("domain")
        indices = mesh.get_nodal_field_indices()
        velocity_error = conformation_error = 0.0
        centreline = 0.0
        for node in mesh.nodes():
            y = node.x(1)
            reference = _FORCE / (2 * _ETA_0) * y * (1 - y)
            shear_rate = _FORCE / (2 * _ETA_0) * (1 - 2 * y)
            centreline = max(centreline, node.value(indices["velocity_x"]))
            velocity_error = max(velocity_error,
                                 abs(node.value(indices["velocity_x"]) - reference),
                                 abs(node.value(indices["velocity_y"])))
            psi = numpy.zeros((3, 3))
            psi[0, 0] = node.value(indices["log_conformation_xx"])
            psi[1, 1] = node.value(indices["log_conformation_yy"])
            psi[0, 1] = psi[1, 0] = node.value(indices["log_conformation_xy"])
            eigenvalues, eigenvectors = numpy.linalg.eigh(psi)
            C = eigenvectors @ numpy.diag(numpy.exp(eigenvalues)) @ eigenvectors.T
            conformation_error = max(conformation_error,
                                     abs(C[0, 0] - (1 + 2 * (_LAMBDA * shear_rate) ** 2)),
                                     abs(C[0, 1] - _LAMBDA * shear_rate),
                                     abs(C[1, 1] - 1))
        return velocity_error, conformation_error, centreline


def test_poiseuille_flow_reproduces_the_total_viscosity(tmp_path):
    velocity_error, conformation_error, centreline = _poiseuille_errors(tmp_path, 16)
    assert velocity_error < 1e-5
    assert conformation_error < 1e-3
    # The discriminating number: with the total viscosity the centreline velocity is F/(8*eta_0).
    # A sign error in the polymer stress would give F/(8*(eta_s-eta_p)), which is negative here.
    assert abs(centreline - _FORCE / (8 * _ETA_0)) < 1e-5


def test_supg_does_not_move_a_converged_solution(tmp_path):
    """
    SUPG is residual-based, so its perturbation multiplies the strong residual and vanishes at the
    exact solution. Switching it on must therefore change the discrete solution only where that
    residual is nonzero -- not the answer a converged computation gives. Poiseuille is the sharpest
    case for this: the exact velocity and conformation lie in the FE space, so a stabilisation that
    were inconsistent would show up immediately as a shifted profile.
    """
    plain = _poiseuille_errors(tmp_path, 16)
    supg = _poiseuille_errors(tmp_path, 16, stabilization="SUPG")
    assert supg[0] < 1e-5 and supg[1] < 1e-3
    assert abs(supg[2] - plain[2]) < 1e-9, "centreline velocity moved: " + str((plain[2], supg[2]))


def test_poiseuille_flow_converges_under_refinement(tmp_path):
    """
    The exact velocity and conformation fields are quadratic and so lie in the C2 space, but
    Psi=log(C) does not, so there is a genuine interpolation error that has to vanish with the mesh.
    """
    coarse = _poiseuille_errors(tmp_path, 8)
    fine = _poiseuille_errors(tmp_path, 16)
    assert coarse[0] / fine[0] > 4.0, "velocity errors: " + str((coarse[0], fine[0]))
    assert coarse[1] / fine[1] > 4.0, "conformation errors: " + str((coarse[1], fine[1]))


# ----------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------

def test_symmetric_2x2_matrix_log_inverts_the_exponential():
    """
    The symbolic matrix logarithm used to prescribe inflow values of the log-conformation tensor.

    Wi=0 is in the list on purpose: it makes the conformation tensor isotropic, so the eigenvalues
    coincide and the b coefficient of a*I + b*M is 0/0. That is not a corner case to be tolerated but
    the ordinary situation on a channel's symmetry line, which is exactly where this helper gets used.
    """
    for weissenberg in (0.0, 1e-8, 0.3, 1.0, 4.0):
        C = numpy.array([[1 + 2 * weissenberg ** 2, weissenberg], [weissenberg, 1.0]])
        expression = symmetric_2x2_matrix_log(oldroyd_b_shear_conformation(weissenberg))
        got = numpy.array([[float(expression[i, j]) for j in range(2)] for i in range(2)])
        eigenvalues, eigenvectors = numpy.linalg.eigh(C)
        reference = eigenvectors @ numpy.diag(numpy.log(eigenvalues)) @ eigenvectors.T
        assert numpy.max(numpy.abs(got - reference)) < 1e-12


def test_ptt_rejects_a_nonzero_slip_parameter():
    """The Gordon-Schowalter derivative is not what the decomposition in tensor_funcs implements."""
    with pytest.raises(NotImplementedError):
        PTT(epsilon=0.1, xi=0.2)


@pytest.mark.parametrize("name,model", _MODELS + [("oldroyd-b", OldroydB())],
                         ids=[n for n, _ in _MODELS] + ["oldroyd-b"])
def test_model_relaxation_forms_agree(name, model):
    """
    Each model declares its relaxation twice: g(C) for the conformation formulation and C^-1 g(C)
    for the log-conformation one. They have to be the same function, or the two formulations would
    silently solve different physics -- and the reference integration used by the tests above,
    which goes through the log form, would be validating the log form against itself.
    """
    # An arbitrary symmetric positive definite conformation tensor, well away from equilibrium.
    C = numpy.array([[2.4, 0.7, 0.0], [0.7, 1.3, 0.0], [0.0, 0.0, 0.8]])
    identity = matrix([[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)])
    C_expression = matrix([[C[i, j] for j in range(3)] for i in range(3)])
    Cinv_expression = matrix([[numpy.linalg.inv(C)[i, j] for j in range(3)] for i in range(3)])
    trace = float(numpy.trace(C))

    direct = model.relaxation_matrix(C_expression, trace, identity)
    via_log = matproduct(C_expression, model.log_relaxation_matrix(C_expression, Cinv_expression, trace, identity))
    for i in range(3):
        for j in range(3):
            assert abs(float(direct[i, j]) - float(via_log[i, j])) < 1e-12
