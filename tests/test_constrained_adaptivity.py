# Tests for combining local C1 dof-constraints (ConstrainFieldsToC1Space /
# ConstrainPositionsToC1Space) with genuine oomph-lib hanging nodes on
# adaptively refined (non-conforming) meshes.
#
# This combination used to be refused outright. It is now supported by flattening
# each constrained/hanging dof into real free leaf dofs, composing the C1
# constraint with the refinement hang map (see dev_docs/hanging_nodes_redesign.md
# section 5.5).
#
# Correctness oracle: the problems here are LINEAR (Poisson, Laplace-smoothed
# mesh), so a single Newton step drives the residual to machine zero *iff* the
# assembled (analytic) Jacobian equals the true Jacobian. A wrong hang/constraint
# composition (wrong master weights or wrong residual redistribution) would leave
# a non-zero residual.

import numpy as np
import pytest
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import CircularMesh, RectangularQuadMesh
from pyoomph.equations.poisson import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh, ConstrainPositionsToC1Space


def _max_abs_residual(problem):
    return float(np.max(np.abs(np.asarray(problem.get_residuals()))))


class ConstrainedPoissonProblem(Problem):
    def __init__(self, mode="field", moving=False):
        super().__init__()
        self.mode = mode      # "none" | "field" | "field_where"
        self.moving = moving  # constrain the (moving) mesh position instead

    def define_problem(self):
        self += CircularMesh(radius=1, segments=["NE"])
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "circumference"
        # Constraining to C1 needs a C1 space present in the bulk element.
        eqs += ScalarField("_dummyC1", space="C1") + DirichletBC(_dummyC1=0)
        if self.moving:
            eqs += LaplaceSmoothedMesh()
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "circumference"
            eqs += ConstrainPositionsToC1Space()
        elif self.mode == "field":
            eqs += ConstrainFieldsToC1Space("u")
        elif self.mode == "field_where":
            eqs += ConstrainFieldsToC1Space("u", where=lambda x: x[0] > 0.0)
        self += eqs @ "domain"

    def _adapt_refine(self):
        self += RefineToLevel(2) @ "domain"
        self += RefineToLevel(4) @ "domain/circumference"


@pytest.mark.parametrize("mode", ["field", "field_where"])
def test_constrained_field_adaptivity(mode):
    with ConstrainedPoissonProblem(mode=mode) as problem:
        problem._adapt_refine()
        problem.solve()
        # Linear problem: residual ~0 after the Newton step certifies the analytic
        # Jacobian (hence the composed hang/constraint handling).
        assert _max_abs_residual(problem) < 1e-9


def test_constrained_position_adaptivity():
    with ConstrainedPoissonProblem(moving=True) as problem:
        problem._adapt_refine()
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9


class Constrained3DBrickProblem(Problem):
    """3D brick version: the flattening composition is dimension-agnostic."""

    def __init__(self, constrain=True):
        super().__init__()
        self.constrain = constrain

    def define_problem(self):
        from pyoomph.meshes.simplemeshes import CuboidBrickMesh
        self += CuboidBrickMesh(size=1, N=2)
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "left" + NeumannBC(u=1) @ "right"
        eqs += ScalarField("_dummyC1", space="C1") + DirichletBC(_dummyC1=0)
        if self.constrain:
            eqs += ConstrainFieldsToC1Space("u")
        self += eqs @ "domain"


def test_constrained_adaptivity_3d_brick():
    with Constrained3DBrickProblem(constrain=True) as problem:
        problem += RefineToLevel(1) @ "domain"
        problem += RefineToLevel(3) @ "domain/right"
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9


class _SurfaceField(InterfaceEquations):
    """A linear surface field s coupled to the bulk trace u, for interface-dof tests."""

    def define_fields(self):
        self.define_scalar_field("s", "C2")

    def define_residuals(self):
        s, st = var_and_test("s")
        u = var("u")
        self.add_weak(grad(s), grad(st))
        self.add_weak(s - u, st)


class InterfaceConstrainedProblem(Problem):
    def __init__(self, constrain=True):
        super().__init__()
        self.constrain = constrain

    def define_problem(self):
        self += RectangularQuadMesh(size=[1, 1], N=4)
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "left"
        eqs += ScalarField("_dummyC1", space="C1") + DirichletBC(_dummyC1=0)
        self += eqs @ "domain"
        seqs = _SurfaceField()
        if self.constrain:
            # Constrain the interface (surface) field s to C1 on the refined interface.
            seqs += ConstrainFieldsToC1Space("s")
        self += seqs @ "domain/right"


def test_interface_dof_constraint_reduces_dofs():
    # The constraint must actually pin the interface mid-edge dofs (this only works once the
    # interface mesh's setup_additional_dof_constraints is invoked and the (mode, index) argument
    # order is correct).
    with InterfaceConstrainedProblem(constrain=False) as problem:
        problem.initialise()
        problem.reapply_boundary_conditions()
        ndof_free = problem.ndof()
    with InterfaceConstrainedProblem(constrain=True) as problem:
        problem.initialise()
        problem.reapply_boundary_conditions()
        ndof_constrained = problem.ndof()
    assert ndof_constrained < ndof_free


def test_interface_dof_constraint_adaptivity():
    with InterfaceConstrainedProblem(constrain=True) as problem:
        problem += RefineToLevel(2) @ "domain"
        problem += RefineToLevel(4) @ "domain/right"
        problem.solve()
        assert _max_abs_residual(problem) < 1e-9


# --- Two coupled domains sharing a mutual interface with inter-domain interaction ---
# Verifies that C1 constraining still works (both close to and at the interface) when an interface
# element reads a field from the *opposite* domain, i.e. that the opposite-side hangbuffers the JIT
# code uses are also filled correctly. Refinement is kept matched on both sides of the interface (the
# current requirement); the level jump lives just inside each domain, so hanging occurs close to the
# interface while the constrained mid-edge nodes sit on it.

class _TwoDomainMesh2d(MeshTemplate):
    def __init__(self, N=4):
        super().__init__()
        self.N = N

    def define_geometry(self):
        N = self.N
        A = self.new_domain("domainA")
        B = self.new_domain("domainB")

        def nid(ix, iy):
            return self.add_node_unique(ix / N, iy / N)  # x in [0,2] (interface at x=1), y in [0,1]

        for ix in range(2 * N):
            for iy in range(N):
                dom = A if ix < N else B
                dom.add_quad_2d_C1(nid(ix, iy), nid(ix + 1, iy), nid(ix, iy + 1), nid(ix + 1, iy + 1))
        for iy in range(N):
            self.add_facet_to_boundary("left", [nid(0, iy), nid(0, iy + 1)])
            self.add_facet_to_boundary("right", [nid(2 * N, iy), nid(2 * N, iy + 1)])
            self.add_facet_to_boundary("interface", [nid(N, iy), nid(N, iy + 1)])


class _ConnectTAtInterface(InterfaceEquations):
    def define_fields(self):
        self.define_scalar_field("lam", "C2")  # Lagrange multiplier enforcing T continuity

    def define_residuals(self):
        my, myt = var_and_test("T")
        opp = var("T", domain=self.get_opposite_side_of_interface())
        oppt = testfunction("T", domain=self.get_opposite_side_of_interface())
        lam, lamt = var_and_test("lam")
        self.add_residual(weak(my - opp, lamt))
        self.add_residual(weak(lam, myt))
        self.add_residual(weak(-lam, oppt))


class TwoDomainInterfaceProblem(Problem):
    def __init__(self, constrain=True):
        super().__init__()
        self.constrain = constrain

    def define_problem(self):
        self.add_mesh(_TwoDomainMesh2d(N=4))
        for dom, (bnd, val) in [("domainA", ("left", 0)), ("domainB", ("right", 1))]:
            eqs = PoissonEquation(name="T", space="C2", source=1) + DirichletBC(T=val) @ bnd
            eqs += ScalarField("_dummyC1", space="C1") + DirichletBC(_dummyC1=0)
            if self.constrain:
                eqs += ConstrainFieldsToC1Space("T")
            self += eqs @ dom
        self += _ConnectTAtInterface() @ "domainA/interface"


def test_two_domain_interface_constrained_adaptivity():
    with TwoDomainInterfaceProblem(constrain=True) as problem:
        # Matched refinement on both sides of the interface; the jump lives just inside each domain.
        problem += RefineToLevel(1) @ "domainA"
        problem += RefineToLevel(1) @ "domainB"
        problem += RefineToLevel(3) @ "domainA/interface"
        problem += RefineToLevel(3) @ "domainB/interface"
        problem.solve()
        # Saddle-point (Lagrange multiplier + C1 constraint) conditioning limits the residual to ~1e-12
        # rather than machine zero; a separate analytic-vs-FD Jacobian check confirms the hangbuffers
        # (incl. the opposite-domain path) are exact.
        assert _max_abs_residual(problem) < 1e-8
