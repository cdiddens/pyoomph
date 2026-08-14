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

# Restarting from a state file must put the problem in the state the writer was in, and the run must
# then continue as if it had never been interrupted. Two different claims, and the second one is the
# one that was broken:
#
# The state itself was always restored exactly. But the next solve(timestep=...) took the branch for
# the very first unsteady step -- a freshly built problem has _taken_already_an_unsteady_step False --
# which re-initialises dt, re-applies the initial condition and resets the step counter, so the step
# ran with the degraded FIRST-ORDER start instead of continuing the scheme. The restarted run then
# drifted from the uninterrupted one by O(dt^2).
#
# That is invisible on a problem that has settled: with du/dt ~ 0, BDF1 and BDF2 agree, and a
# diffusion problem run to near-steady state reproduces to 1e-16 either way. The moving-mesh case
# below, driven by a boundary that keeps moving, showed 4.9e-4. Hence a genuinely time-dependent case
# is part of this file on purpose.
#
# Compared here: the residual vector, every history level, the pinned values and the Jacobian, all of
# which have to be bit-identical right after loading; then the same quantities after continuing, where
# each side has done its own Newton solves and round-off is allowed.

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import PseudoElasticMesh

DT = 0.05
STEPS_BEFORE = 3
STEPS_AFTER = 1


class DiffusionEqs(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        x, y = var("coordinate_x"), var("coordinate_y")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v))
                          - weak(1 + 10 * exp(-30 * ((x - 0.3) ** 2 + (y - 0.7) ** 2)), v))


class Line1dEqs(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        x = var("coordinate_x")
        self.add_residual(weak(grad(u), grad(v)) - weak(1 + 50 * exp(-200 * (x - 0.35) ** 2), v))


class Line1dProblem(Problem):
    def define_problem(self):
        self += LineMesh(N=20, size=1)
        self += (Line1dEqs() + DirichletBC(u=0) @ "left" + SpatialErrorEstimator(u=1)) @ "domain"
        self.max_refinement_level = 3
        self.write_states = False


def test_state_of_an_adaptively_refined_1d_mesh(tmp_path):
    # A refined 1d mesh addressed its elements through their tree root, and every son of a binary tree
    # reported root_pt()==NULL: the 2d and 3d son constructors inherit Root_pt from the father, the 1d
    # one did not (src/mesh1d.hpp). Writing a state segfaulted, which is how it was found - through the
    # gcl_glycerol_water_capillary tutorial, a 1d mesh with initial adaptivity.
    fname = str(tmp_path / "line.dump")
    with Line1dProblem() as writer:
        writer.set_output_directory(str(tmp_path / "w1d"))
        writer.solve(spatial_adapt=3)
        refined_elements = writer.get_mesh("domain").nelement()
        assert refined_elements > 20, "the mesh was not refined, so this proves nothing"
        reference = numpy.asarray(writer.get_history_dofs(0))
        writer.save_state(fname)

    with Line1dProblem() as reader:
        reader.set_output_directory(str(tmp_path / "r1d"))
        reader.load_state(fname)
        assert reader.get_mesh("domain").nelement() == refined_elements
        assert numpy.array_equal(numpy.asarray(reader.get_history_dofs(0)), reference)


class FacetValueEqs(InterfaceEquations):
    """A D0 field on an interface, pinned to a value that depends on where the interface is.

    D0 lives in the interface element's own internal Data, so it is stored by the state file's
    interface block and by nothing else - which is what makes it the probe below."""

    def __init__(self, target):
        super().__init__()
        self.target = target

    def define_fields(self):
        self.define_scalar_field("lam", "D0")

    def define_residuals(self):
        lam, lamtest = var_and_test("lam")
        self.add_residual(weak(lam - self.target, lamtest))


class NestedInterfaceProblem(Problem):
    """Carries D0 fields on an interface AND on an interface of that interface (a point here)."""

    def define_problem(self):
        self += RectangularQuadMesh(N=4, size=[1, 1])
        eqs = DiffusionEqs() + DirichletBC(u=0) @ "bottom"
        eqs += FacetValueEqs(2 + var("coordinate_x")) @ "top"
        eqs += FacetValueEqs(7.25) @ "top/left"
        eqs += FacetValueEqs(-3.5) @ "top/right"
        self += eqs @ "domain"
        self.write_states = False


def _interface_facet_values(problem):
    """{interface name: {structural key: internal-Data values}} over every interface mesh."""
    out = {}
    for mesh in problem._interfacemeshes:
        nelem = mesh.nelement()
        keys = numpy.asarray(mesh.get_interface_element_structural_keys(),
                             dtype=numpy.int64).reshape(nelem, 3)
        assert not numpy.any(keys[:, 0] < 0), (mesh.get_full_name(), keys)
        per_element = {}
        for ie, e in enumerate(mesh.elements()):
            key = tuple(int(c) for c in keys[ie])
            assert key not in per_element, "two elements of %s share the key %s" % (mesh.get_full_name(), key)
            per_element[key] = [e.internal_data_pt(i).value(j)
                                for i in range(e.ninternal_data())
                                for j in range(e.internal_data_pt(i).nvalue())]
        out[mesh.get_full_name()] = per_element
    return out


def test_state_file_of_an_interface_on_an_interface(tmp_path):
    """An interface OF an interface must be addressable in a state file, and its values must come back.

    Such an element hangs off a face element, which has no refinement tree and no base element index of
    its own, so asking it for a structural key produced (-1,-1,-1) and save_state refused to write
    anything at all - "Interface mesh elements without a global base index". That is every free surface
    meeting a wall, so it took out the state files of a whole class of problems. The key is the chain of
    face indices down to the bulk element now (src/mesh.cpp, pack_face_chain).

    Refined once before writing, because the same point interface used to take the process down with
    "pure virtual method called" on any adaptation that carried its discontinuous fields across
    (src/mesh.cpp, sample_position and the point branch of restore_discontinuous_data).

    The reader never solves: a solve would recompute lam from its own residual and hide a load that
    restored nothing."""
    fname = str(tmp_path / "nested.dump")
    with NestedInterfaceProblem() as writer:
        writer.set_output_directory(str(tmp_path / "w_nested"))
        writer.solve()
        writer.refine_uniformly()
        assert writer.get_mesh("domain").nelement() > 16, "the mesh was not refined, so this proves less"
        before = _interface_facet_values(writer)
        # the adaptation carried the point's value over rather than resetting it to zero
        assert [v for vals in before["domain/top/left"].values() for v in vals] == [7.25]
        writer.save_state(fname)

    assert set(before) >= {"domain/top", "domain/top/left", "domain/top/right"}, sorted(before)
    # the nested interfaces are what this is about, and they hold what they were pinned to
    assert [v for vals in before["domain/top/left"].values() for v in vals] == [7.25]
    assert [v for vals in before["domain/top/right"].values() for v in vals] == [-3.5]
    # ... and the two of them are told apart, rather than sharing the key of their common parent facet
    assert set(before["domain/top/left"]) != set(before["domain/top/right"])

    with NestedInterfaceProblem() as reader:
        reader.set_output_directory(str(tmp_path / "r_nested"))
        reader.load_state(fname)          # ... and not a single solve after it
        after = _interface_facet_values(reader)
    assert after == before


class MovingMeshEqs(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(1, v))


class TopMotion(InterfaceEquations):
    def define_residuals(self):
        # keeps moving, so the time discretisation order actually matters
        self.set_Dirichlet_condition("mesh_y", 1 + 0.2 * sin(3 * var("time")))


class RestartProblem(Problem):
    def __init__(self, kind, solver=None):
        super().__init__()
        self.kind = kind
        self.solver = solver
        self.write_states = False
        self.eigen_data_in_states = False
        self.continuation_data_in_states = False

    def define_problem(self):
        if self.solver is not None:
            self.set_linear_solver(self.solver)
        self += RectangularQuadMesh(N=4, size=[1, 1])
        if self.kind == "movmesh":
            eqs = MovingMeshEqs() + PseudoElasticMesh()
            eqs += DirichletBC(u=0, mesh_x=True, mesh_y=True) @ "bottom"
            eqs += DirichletBC(mesh_x=True) @ "left"
            eqs += DirichletBC(mesh_x=True) @ "right"
            eqs += TopMotion() @ "top"
        else:
            eqs = DiffusionEqs() + DirichletBC(u=0) @ "left"
        self += eqs @ "domain"


def _advance(problem, nsteps, adaptive):
    if adaptive:
        now = problem.get_current_time(dimensional=False, as_float=True)
        problem.run(now + nsteps * DT, startstep=DT, temporal_error=1e-3, outstep=False, maxstep=3 * DT)
    else:
        for _ in range(nsteps):
            problem.solve(timestep=DT)


def _snapshot(problem):
    residual, jacobian = problem.assemble_jacobian(with_residual=True)
    out = {"res": numpy.asarray(residual),
           "pinned": numpy.asarray(problem.get_current_pinned_values(True)),
           "jac": numpy.asarray(jacobian.tocoo().data),
           "time": problem.get_current_time(dimensional=False, as_float=True),
           "steps_done": problem.timestepper.get_num_unsteady_steps_done(),
           "dts": [problem.timestepper.time_pt().dt(i) for i in range(problem.timestepper.time_pt().ndt())]}
    for level in range(4):
        out["hist%d" % level] = numpy.asarray(problem.get_history_dofs(level))
    return out


def _assert_same(a, b, what, tol):
    assert a["time"] == b["time"], "%s: time %r vs %r" % (what, a["time"], b["time"])
    assert a["dts"] == b["dts"], "%s: dt %r vs %r" % (what, a["dts"], b["dts"])
    assert a["steps_done"] == b["steps_done"], (
        "%s: %d unsteady steps done vs %d -- a restarted run that thinks it is starting over takes its "
        "next step with the degraded first-order scheme" % (what, a["steps_done"], b["steps_done"]))
    for key in ("res", "jac", "pinned", "hist0", "hist1", "hist2", "hist3"):
        x, y = a[key], b[key]
        assert len(x) == len(y), "%s: %s has %d entries vs %d" % (what, key, len(x), len(y))
        if len(x) == 0:
            continue
        deviation = float(numpy.amax(numpy.abs(x - y)))
        assert deviation <= tol, "%s: %s differs by %.3e (tolerance %.0e)" % (what, key, deviation, tol)


# The continuation is compared bitwise with SuperLU and only to round-off with whatever solver the
# problem would use anyway. SuperLU (scipy) factorises from scratch every time, so its solve is a pure
# function of the matrix and a correctly restarted run reproduces the uninterrupted one exactly.
#
# Pardiso lands 2.2e-16 away instead, and it is worth knowing why, because it is not sloppiness: it
# reuses the SYMBOLIC factorisation whenever the sparsity pattern is unchanged (phase 22 instead of
# phase 12, PardisoSolver.reuse_symbolic_factorisation, on by default). A restarted run reaches that
# analysis with a different matrix than an uninterrupted one, so the reused analysis is not the same,
# and the numeric factorisation differs in the last bits. Verified by elimination: with
# reuse_symbolic_factorisation=False, Pardiso is bitwise as well. It is NOT thread nondeterminism
# either - unchanged with MKL_NUM_THREADS=1. (try_to_reuse_solver, which would reuse the NUMERIC
# factors, is off by default and plays no part here.)
@pytest.mark.parametrize("kind", ["transient", "tempadapt", "movmesh"])
@pytest.mark.parametrize("solver,continuation_tol", [("superlu", 0.0), (None, 1e-10)])
def test_restart_reproduces_the_state_and_the_continuation(tmp_path, kind, solver, continuation_tol):
    adaptive = kind == "tempadapt"
    fname = str(tmp_path / (kind + ".dump"))

    with RestartProblem(kind, solver) as writer:
        writer.set_output_directory(str(tmp_path / ("w_" + kind + str(solver))))
        writer.solve()
        _advance(writer, STEPS_BEFORE, adaptive)
        at_write_time = _snapshot(writer)
        writer.save_state(fname)
        _advance(writer, STEPS_AFTER, adaptive)
        uninterrupted = _snapshot(writer)

    with RestartProblem(kind, solver) as reader:
        reader.set_output_directory(str(tmp_path / ("r_" + kind + str(solver))))
        reader.load_state(fname)
        # The state as such: nothing here may differ at all, not even in the last bit
        _assert_same(at_write_time, _snapshot(reader), "state right after loading " + kind, tol=0.0)
        _advance(reader, STEPS_AFTER, adaptive)
        # The continuation: each side ran its own Newton solves, so round-off is allowed - but nothing
        # more. An O(dt^2) deviation here means the restarted run is integrating differently.
        _assert_same(uninterrupted, _snapshot(reader), "continuation after loading " + kind, tol=continuation_tol)
