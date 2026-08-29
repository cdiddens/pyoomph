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

# Correctness gates for the per-block JACOBIAN_BLOCK_* property flags computed during code generation
# (jacobian_block_flags / mass_matrix_block_flags in the JIT func table, see src/jitbridge.h).
#
# What is being protected here:
#
#   Each flag claims a PROOF about a whole (row class, column class) block of the elemental matrix,
#   read off the symbolic block expression rather than from numbers. A consumer -- a Schur complement
#   that wants to reuse a factorisation, a fieldsplit preconditioner picking a symmetric KSP -- acts on
#   that claim without ever checking it, so a wrong bit is silently wrong linear algebra, not a crash.
#
#   The gate below is therefore one-sided in the same way the contract is: EVERY set bit is verified
#   numerically, at several dof states, on the assembled global matrices (an elemental property carries
#   over to the global class block, since the latter is a sum of the former). Nothing asserts that a
#   clear bit is false -- "unset" means "not proven", and a conservative codegen is allowed to give up.
#
#   The second half of the file is the opposite check: that the analysis is not trivially proving
#   nothing. Poisson must come out symmetric AND constant, transient diffusion must lose CONSTANT but
#   keep CONSTANT_FIXED_DT, a wind read from a global parameter must lose CONSTANT that the same wind
#   written as a literal keeps, Stokes must be symmetric where Navier-Stokes is not, and a moving mesh
#   must lose constancy everywhere.

import os
import subprocess
import sys
import textwrap

import numpy
import pytest
from scipy.sparse import csr_matrix

import pyoomph._pyoomph_core as _core
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions.cb import CustomMathExpression
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.navier_stokes import NavierStokesEquations, StokesEquations
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

SYMMETRIC = _core.JACOBIAN_BLOCK_SYMMETRIC
ANTISYMMETRIC = _core.JACOBIAN_BLOCK_ANTISYMMETRIC
CONSTANT = _core.JACOBIAN_BLOCK_CONSTANT
CONSTANT_FIXED_DT = _core.JACOBIAN_BLOCK_CONSTANT_FIXED_DT


# ---------------------------------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------------------------------

class _PoissonProblem(Problem):
    """The reference case: a linear, steady, fixed-mesh operator, so everything must be provable."""

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(grad(u), grad(v)) - weak(1, v))

        eqs = _Eqs()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


class _TransientDiffusionProblem(Problem):
    """Same operator plus d/dt: the Jacobian now carries the BDF weight, the mass matrix does not."""

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


class _ConvectionDiffusionProblem(Problem):
    """Where the wind comes from is the whole point: the same weak form is constant with a literal
    wind, parameter-dependent with a wind read from a global parameter, and dof-dependent with a wind
    that is itself an unknown."""

    def __init__(self, wind="literal"):
        super().__init__()
        self.wind = wind

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))
        if self.wind == "literal":
            wind = vector(1, 0)
        elif self.wind == "parameter":
            # Nonzero, so that the block is numerically nonsymmetric as well as unprovable: at the
            # default value of 0 the convective term disappears from the numbers and the test below
            # would be comparing a pure diffusion block against itself.
            self.get_global_parameter("w").value = 1.0
            wind = vector(self.get_global_parameter("w"), 0)
        else:
            wind = var("velocity")

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("c", "C2")

            def define_residuals(self):
                c, q = var_and_test("c")
                self.add_residual(weak(grad(c), grad(q)) + weak(dot(wind, grad(c)), q))

        eqs = _Eqs()
        if self.wind == "unknown":
            eqs += NavierStokesEquations(dynamic_viscosity=1, mass_density=1)
            for b in ["left", "right", "bottom"]:
                eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
            eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
            eqs += DirichletBC(pressure=0) @ "bottom/left"
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(c=0) @ b
        self.add_equations(eqs @ "domain")


class _CavityProblem(Problem):
    """Lid-driven cavity, Stokes or Navier-Stokes. The difference between the two is exactly the
    convective term, i.e. exactly what a symmetry and a constancy proof must react to."""

    def __init__(self, navier=False):
        super().__init__()
        self.navier = navier

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))
        if self.navier:
            eqs = NavierStokesEquations(dynamic_viscosity=1, mass_density=1)
        else:
            eqs = StokesEquations(dynamic_viscosity=1)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        self.add_equations(eqs @ "domain")


class _MovingMeshProblem(Problem):
    """A Laplace-smoothed moving mesh. dx, the shape derivatives and the normal all become functions of
    unknowns, so no block may claim constancy any more."""

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(grad(u), grad(v)) - weak(1, v))

        eqs = _Eqs() + LaplaceSmoothedMesh()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0, mesh_x=True, mesh_y=True) @ b
        self.add_equations(eqs @ "domain")


class _CallbackProblem(Problem):
    """A python callback in the diffusivity. Callbacks are required to be deterministic functions of
    their arguments, so the block's constancy is decided by the arguments alone: a coordinate of a
    fixed mesh is constant, the unknown and the time are not."""

    def __init__(self, argument="coordinate"):
        super().__init__()
        self.argument = argument

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=4))
        argument = self.argument

        class _Bump(CustomMathExpression):
            def eval(self, arg):
                return 1.0 + 0.5 * numpy.sin(arg[0])

        class _Eqs(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "C2")

            def define_residuals(self):
                u, v = var_and_test("u")
                arg = {"coordinate": var("coordinate_x"), "unknown": u, "time": var("time")}[argument]
                self.add_residual(weak(_Bump()(arg) * grad(u), grad(v)) - weak(1, v))

        eqs = _Eqs()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")


# ---------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------

def _class_dofs(problem, meshname="domain"):
    """(class names, list of global equation numbers per class). The elemental flags are statements
    about (row class, column class) blocks, and the global block of a class pair is just the sum of the
    elemental ones, so the properties carry over and can be checked on the assembled matrix."""
    mesh = problem.get_mesh(meshname)
    names = mesh.element_pt(0)._get_contribution_tables()[0]
    groups = [set() for _ in names]
    for e in range(mesh.nelement()):
        el = mesh.element_pt(e)
        cidx = el._get_dof_contribution_indices()
        for i in range(el.ndof()):
            eq = el.eqn_number(i)
            if eq >= 0 and cidx[i] >= 0:
                groups[cidx[i]].add(eq)
    return names, [numpy.array(sorted(g), dtype=int) for g in groups]


def _tables(problem, meshname="domain"):
    """(contributes_to_jacobian, contributes_to_mass_matrix, jacobian_flags, mass_flags) of the code
    behind `meshname`, for the currently active residual."""
    el = problem.get_mesh(meshname).element_pt(0)
    _names, jac, mass = el._get_contribution_tables()
    jflags, mflags = el._get_block_flags()
    return jac, mass, jflags, mflags


def _matrices(p):
    """(mass matrix, Jacobian). The Jacobian is taken from the plain assembly rather than from the
    eigenproblem one, because only that one carries the time-stepper weights -- which is the very thing
    CONSTANT and CONSTANT_FIXED_DT differ about."""
    J = p.assemble_jacobian(with_residual=False).tocsr().copy()
    n, _, _, Mv, Mc, Mr, _, _, _Jv, _Jc, _Jr = p.assemble_eigenproblem_matrices(0.0)
    M = csr_matrix((Mv, Mc, Mr), shape=(n, n)).copy()
    return M, J


def _block(A, rows, cols):
    if len(rows) == 0 or len(cols) == 0:
        return numpy.zeros((len(rows), len(cols)))
    return numpy.asarray(A[rows][:, cols].todense())


def _perturb_dofs(p, base, seed, amplitude=1.0):
    """Always perturbs the SAME base state, so repeated calls do not compound into an ever wilder mesh."""
    rng = numpy.random.default_rng(seed)
    p.set_current_dofs(base + amplitude * rng.standard_normal(len(base)))


def _set_dt(p, dt):
    p.timestepper.time_pt().set_dt(0, dt)
    p.timestepper.set_weights()


# ---------------------------------------------------------------------------------------------------
# Gate 1 -- every set bit is true
# ---------------------------------------------------------------------------------------------------

def _check_soundness(p, meshname="domain", tol=1e-9, amplitude=1.0):
    """Verify numerically every bit the code generator claims. Returns the number of bits checked, so a
    caller can assert the run was not vacuous."""
    names, groups = _class_dofs(p, meshname)
    jac, mass, jflags, mflags = _tables(p, meshname)
    checked = 0

    def blocks_at(states):
        """The (M, J) class blocks at each of the given dof/time/dt states."""
        out = []
        for prepare in states:
            prepare()
            M, J = _matrices(p)
            out.append(([[_block(J, r, c) for c in groups] for r in groups],
                        [[_block(M, r, c) for c in groups] for r in groups]))
        return out

    dt0 = p.timestepper.time_pt().dt(0)
    t0 = p.get_current_time(dimensional=False, as_float=True)
    dofs0 = numpy.asarray(p.get_current_dofs()[0]).copy()
    restore = lambda: p.set_current_dofs(dofs0)
    states = [restore,
              lambda: _perturb_dofs(p, dofs0, 1234, amplitude),
              lambda: _perturb_dofs(p, dofs0, 5678, amplitude)]
    sampled = blocks_at(states)

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            for which, present, flags in ((0, jac, jflags), (1, mass, mflags)):
                f = flags[i][j]
                if f == 0:
                    continue
                what = "%s block %s vs %s" % ("jacobian" if which == 0 else "mass", ni, nj)
                assert present[i][j], "%s: flagged but not even marked as contributing" % what
                if f & SYMMETRIC or f & ANTISYMMETRIC:
                    sign = 1.0 if f & SYMMETRIC else -1.0
                    assert not (f & SYMMETRIC and f & ANTISYMMETRIC), \
                        "%s: claimed symmetric and antisymmetric at once" % what
                    for s in sampled:
                        A, B = s[which][i][j], s[which][j][i]
                        scale = max(1.0, numpy.abs(A).max() if A.size else 1.0)
                        assert numpy.abs(A - sign * B.T).max() <= tol * scale, \
                            "%s: claimed %s but is not" % (what, "symmetric" if sign > 0 else "antisymmetric")
                    checked += 1
                if f & CONSTANT or f & CONSTANT_FIXED_DT:
                    ref = sampled[0][which][i][j]
                    scale = max(1.0, numpy.abs(ref).max() if ref.size else 1.0)
                    for s in sampled[1:]:
                        assert numpy.abs(s[which][i][j] - ref).max() <= tol * scale, \
                            "%s: claimed constant but moved with the dofs" % what
                    checked += 1

    # Time, the global parameters and dt are state too. Restored afterwards, so the caller keeps a
    # usable problem.
    def constant_blocks_survive(change, undo, needs_full_constant, label):
        change()
        M, J = _matrices(p)
        for i in range(len(names)):
            for j in range(len(names)):
                for which, flags, A in ((0, jflags, J), (1, mflags, M)):
                    f = flags[i][j]
                    if not (f & CONSTANT) and (needs_full_constant or not (f & CONSTANT_FIXED_DT)):
                        continue
                    ref = sampled[0][which][i][j]
                    now = _block(A, groups[i], groups[j])
                    scale = max(1.0, numpy.abs(ref).max() if ref.size else 1.0)
                    assert numpy.abs(now - ref).max() <= tol * scale, \
                        "%s block %s vs %s: claimed constant but moved when %s" % (
                            "jacobian" if which == 0 else "mass", names[i], names[j], label)
        undo()

    restore()
    constant_blocks_survive(lambda: p.set_current_time(t0 + 3.0, dimensional=False),
                            lambda: p.set_current_time(t0, dimensional=False), False, "time advanced")
    for name in sorted(n for n in p.get_global_parameter_names() if not n.startswith("_")):
        param = p.get_global_parameter(name)
        old = param.value
        constant_blocks_survive(lambda: setattr(param, "value", old + 1.375),
                                lambda: setattr(param, "value", old), False, "a global parameter changed")
    # CONSTANT (but not CONSTANT_FIXED_DT) must additionally survive a change of dt.
    constant_blocks_survive(lambda: _set_dt(p, dt0 * 0.5), lambda: _set_dt(p, dt0), True, "dt changed")
    return checked


@pytest.mark.parametrize("factory,expect_bits", [
    (lambda: _PoissonProblem(), True),
    (lambda: _TransientDiffusionProblem(), True),
    (lambda: _ConvectionDiffusionProblem("literal"), True),
    # A wind read from a global parameter leaves the single block of that problem with no provable
    # property at all, which is the correct answer and the one case where the gate has nothing to do.
    (lambda: _ConvectionDiffusionProblem("parameter"), False),
    (lambda: _ConvectionDiffusionProblem("unknown"), True),
    (lambda: _CavityProblem(navier=False), True),
    (lambda: _CavityProblem(navier=True), True),
    (lambda: _MovingMeshProblem(), True),
    (lambda: _CallbackProblem("coordinate"), True),
    (lambda: _CallbackProblem("unknown"), False),
    (lambda: _CallbackProblem("time"), True),
], ids=["poisson", "transient_diffusion", "convdiff_literal", "convdiff_parameter", "convdiff_unknown",
        "stokes", "navier_stokes", "moving_mesh", "callback_of_coordinate", "callback_of_unknown",
        "callback_of_time"])
def test_every_set_bit_is_true(factory, expect_bits):
    """The one-sided contract: a set bit is a promise about the assembled matrix, so check them all."""
    with factory() as p:
        p.quiet()
        p.initialise()
        _set_dt(p, 0.25)
        p.solve()
        # A moving mesh tolerates only a small perturbation: randomising the nodal positions by O(1)
        # turns elements inside out and the assembly stops meaning anything.
        amplitude = 0.02 if isinstance(p, _MovingMeshProblem) else 1.0
        checked = _check_soundness(p, amplitude=amplitude)
        assert (checked > 0) == expect_bits, "the gate verified %d bits, expected %s" % (
            checked, "at least one" if expect_bits else "none")


# ---------------------------------------------------------------------------------------------------
# Gate 2 -- the analysis is not trivially proving nothing
# ---------------------------------------------------------------------------------------------------

def _index(names, name):
    return names.index(name)


def test_poisson_block_is_fully_provable():
    """The best case there is: linear, steady, fixed mesh. If this one is not fully flagged, the
    analysis has given up somewhere it should not have."""
    with _PoissonProblem() as p:
        p.quiet()
        p.initialise()
        p.solve()
        names, _groups = _class_dofs(p)
        _jac, _mass, jflags, _mflags = _tables(p)
        i = _index(names, "domain/u")
        assert jflags[i][i] == SYMMETRIC | CONSTANT | CONSTANT_FIXED_DT


def test_transient_diffusion_loses_constant_but_not_constant_at_fixed_dt():
    """The distinction the two constancy bits exist for: the Jacobian of d/dt carries the BDF weight, so
    it is constant only as long as dt is. The mass matrix half carries no weight and stays constant
    outright."""
    with _TransientDiffusionProblem() as p:
        p.quiet()
        p.initialise()
        _set_dt(p, 0.25)
        p.solve(timestep=0.25)
        names, groups = _class_dofs(p)
        _jac, mass, jflags, mflags = _tables(p)
        i = _index(names, "domain/u")
        assert jflags[i][i] == SYMMETRIC | CONSTANT_FIXED_DT
        assert mass[i][i], "the mass matrix block should exist"
        assert mflags[i][i] == SYMMETRIC | CONSTANT | CONSTANT_FIXED_DT

        # ...and the bit difference is real: the Jacobian block really does move with dt.
        before = _block(_matrices(p)[1], groups[i], groups[i])
        _set_dt(p, 0.5)
        after = _block(_matrices(p)[1], groups[i], groups[i])
        assert numpy.abs(after - before).max() > 1e-6, \
            "dt did not move the block, so CONSTANT_FIXED_DT vs CONSTANT proves nothing here"


@pytest.mark.parametrize("argument,constant", [("coordinate", True), ("unknown", False), ("time", False)],
                         ids=["callback_of_coordinate", "callback_of_unknown", "callback_of_time"])
def test_a_callback_is_as_constant_as_its_arguments(argument, constant):
    """A callback is a deterministic function of its arguments, so it must not block constancy by
    itself - only its arguments may. Treating every callback as variable (as the analysis first did)
    would make the coordinate case unprovable."""
    with _CallbackProblem(argument) as p:
        p.quiet()
        p.initialise()
        p.solve()
        names, _groups = _class_dofs(p)
        _jac, _mass, jflags, _mflags = _tables(p)
        i = _index(names, "domain/u")
        assert bool(jflags[i][i] & CONSTANT) is constant
        assert bool(jflags[i][i] & CONSTANT_FIXED_DT) is constant


@pytest.mark.parametrize("wind,constant", [("literal", True), ("parameter", False), ("unknown", False)],
                         ids=["literal_wind", "parameter_wind", "unknown_wind"])
def test_convection_makes_the_block_nonsymmetric_but_not_necessarily_variable(wind, constant):
    """A first-derivative term destroys symmetry regardless of where its coefficient comes from, while
    constancy depends entirely on that: a literal wind is a number, a wind read from a global parameter
    or from the velocity unknown is not."""
    with _ConvectionDiffusionProblem(wind) as p:
        p.quiet()
        p.initialise()
        p.solve()
        names, groups = _class_dofs(p)
        _jac, _mass, jflags, _mflags = _tables(p)
        i = _index(names, "domain/c")
        assert not (jflags[i][i] & (SYMMETRIC | ANTISYMMETRIC))
        assert bool(jflags[i][i] & CONSTANT) is constant
        assert bool(jflags[i][i] & CONSTANT_FIXED_DT) is constant
        # ...and SYMMETRIC is not merely unproven here, the block genuinely is not symmetric.
        B = _block(_matrices(p)[1], groups[i], groups[i])
        assert numpy.abs(B - B.T).max() > 1e-6


def test_stokes_is_symmetric_where_navier_stokes_is_not():
    """The velocity-velocity block is the convective term's only home, so it is the block that must
    react to the difference between the two."""
    with _CavityProblem(navier=False) as p:
        p.quiet()
        p.initialise()
        p.solve()
        names, _groups = _class_dofs(p)
        _jac, _mass, jflags, _mflags = _tables(p)
        i = _index(names, "domain/velocity_x")
        assert jflags[i][i] & SYMMETRIC
        assert jflags[i][i] & CONSTANT

    with _CavityProblem(navier=True) as p:
        p.quiet()
        p.initialise()
        p.solve()
        names, _groups = _class_dofs(p)
        _jac, _mass, jflags, _mflags = _tables(p)
        i = _index(names, "domain/velocity_x")
        assert jflags[i][i] == 0, "the convective term makes the velocity block neither symmetric nor constant"


def test_the_velocity_pressure_pair_is_antisymmetric():
    """With pyoomph's weak form the pressure enters the momentum equation as -p*div(v) and the
    continuity equation as +div(u)*q, so the off-diagonal pair is ANTIsymmetric, not symmetric. Asserted
    against the numbers as well, since it is the sign convention that decides which of the two bits is
    the correct one."""
    with _CavityProblem(navier=False) as p:
        p.quiet()
        p.initialise()
        p.solve()
        names, groups = _class_dofs(p)
        _jac, _mass, jflags, _mflags = _tables(p)
        u, q = _index(names, "domain/velocity_x"), _index(names, "domain/pressure")
        assert jflags[u][q] & ANTISYMMETRIC
        assert jflags[q][u] & ANTISYMMETRIC, "the bit must be set on both mirror entries"
        assert not (jflags[u][q] & SYMMETRIC)
        J = _matrices(p)[1]
        A, B = _block(J, groups[u], groups[q]), _block(J, groups[q], groups[u])
        assert numpy.abs(A).max() > 1e-6
        assert numpy.abs(A + B.T).max() <= 1e-9 * numpy.abs(A).max()
        assert numpy.abs(A - B.T).max() > 1e-6, "if it were symmetric too, the pair would say nothing"


def test_a_moving_mesh_clears_the_constancy_bits():
    """Every integrated block carries dx, and on a moving mesh dx is a function of unknowns. So is every
    shape derivative. Nothing may claim constancy here."""
    with _MovingMeshProblem() as p:
        p.quiet()
        p.initialise()
        p.solve()
        _jac, _mass, jflags, mflags = _tables(p)
        for flags in (jflags, mflags):
            for row in flags:
                for f in row:
                    assert not (f & (CONSTANT | CONSTANT_FIXED_DT))


def test_the_azimuthal_residual_set_is_skipped():
    """Azimuthal (and Cartesian normal-mode) stability blocks are complex, so proving J_ij = J_ji^T
    would need conjugate-transpose semantics that this analysis does not have. Codegen skips those
    residual sets entirely, which the contract reads as "nothing proven" -- the conservative answer, and
    the one that must not silently become "everything proven"."""
    with _PoissonProblem() as p:
        p.quiet()
        p.setup_for_stability_analysis(azimuthal_stability=True)
        p.initialise()
        p.solve()
        _jac, _mass, jflags, _mflags = _tables(p)
        assert any(f for row in jflags for f in row), "expected the default residual to be flagged"
        assert p._set_solved_residual("real_contrib_azimuthal_stability", True, True)
        try:
            _jac, _mass, jflags, mflags = _tables(p)
            assert not any(f for row in jflags for f in row)
            assert not any(f for row in mflags for f in row)
        finally:
            p._set_solved_residual("", False, True)


# ---------------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------------

_DETERMINISM_SCRIPT = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, %r)
    import test_jacobian_block_flags as T

    with T._CavityProblem(navier=True) as p:
        p.quiet()
        p.initialise()
        p.solve()
        jac, mass, jflags, mflags = T._tables(p)
        print("PYOOMPH_FLAGS " + json.dumps([jflags, mflags]))
""") % os.path.dirname(os.path.abspath(__file__))


def test_the_tables_are_deterministic(tmp_path):
    """The same problem must produce the same tables every time: the analysis walks maps keyed by
    pointers to fields, spaces and basis functions, and heap addresses are not reproducible across
    processes. Run in a subprocess per build rather than twice in one process -- a second Problem in the
    same interpreter can segfault in the JIT loader (see tests/test_multiple_problems.py)."""
    env = dict(os.environ)
    results = []
    for run in range(2):
        wd = tmp_path / ("run%d" % run)
        wd.mkdir()
        out = subprocess.run([sys.executable, "-c", _DETERMINISM_SCRIPT], cwd=str(wd), env=env,
                             capture_output=True, text=True, timeout=900)
        assert out.returncode == 0, out.stdout + out.stderr
        lines = [l for l in out.stdout.splitlines() if l.startswith("PYOOMPH_FLAGS ")]
        assert len(lines) == 1, out.stdout + out.stderr
        results.append(lines[0])
    assert results[0] == results[1]
