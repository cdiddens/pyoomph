#  ========================================================================
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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
#  ========================================================================

"""The threaded element loop must change the time and nothing else.

Everything here compares with ``array_equal`` rather than ``allclose``: the whole design of
``dev_docs/openmp_assembly.md`` - chunks in element order, a gather sorted stably by target slot -
exists to make the threaded assembly reproduce the serial one BIT FOR BIT. An ``allclose`` here
would pass just as well against a version that got the summation order wrong, which is exactly the
regression this file is meant to catch.

Two things every test does deliberately:

* ``_set_num_assembly_threads`` rather than ``set_num_threads``. The latter also threads the linear
  solver, and a threaded direct solver is NOT bit-reproducible - comparing after a solve would then
  compare two different converged states and blame the assembly for it.
* ``_get_parallel_assemblies_done()``. A fast path that silently fell back looks exactly like a
  working one, because both give the right answer; without this check every comparison below could
  be comparing the serial loop with itself.
"""

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.generic import *
from pyoomph.equations.poisson import *
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.additional import EquationCompilationFlags
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, LineMesh
from pyoomph import _pyoomph_core

NTHREADS = 4

# A build configured without OpenMP (PYOOMPH_USE_OPENMP=OFF, or AUTO on a toolchain that has none)
# still accepts _set_num_assembly_threads and still gives the right answer - it just never runs the
# threaded loop, so every bit-identity test below would fail on an assertion about a path that does
# not exist. The tests that check the loop DECLINES are left running: they pass either way.
requires_openmp = pytest.mark.skipif(not _pyoomph_core.has_openmp,
                                     reason="this build has no OpenMP; there is no threaded element loop to compare against")


def _snapshot(problem):
    """Residual, Jacobian and the separately assembled residual vector, as raw arrays."""
    res, jac = problem.assemble_jacobian(with_residual=True)
    return [numpy.array(res), jac.data.copy(), jac.indices.copy(), jac.indptr.copy(),
            numpy.array(problem.get_residuals())]


def _assert_threaded_matches_serial(problem, expect_threaded=True):
    serial = _snapshot(problem)
    before = problem._get_parallel_assemblies_done()
    problem._set_num_assembly_threads(NTHREADS)
    threaded = _snapshot(problem)
    ran = problem._get_parallel_assemblies_done() - before
    problem._set_num_assembly_threads(1)
    if expect_threaded:
        assert ran > 0, "the threaded element loop declined; the comparison below would prove nothing"
    names = ["residual (with Jacobian)", "Jacobian values", "Jacobian columns", "Jacobian row starts",
             "residual (on its own)"]
    for name, a, b in zip(names, serial, threaded):
        assert numpy.array_equal(a, b), name + " is not bit-identical between 1 and %d threads" % NTHREADS
    return ran


class _Poisson(Problem):
    def __init__(self, n=12, fd_jacobian=False):
        super().__init__()
        self.n = n
        self.fd_jacobian = fd_jacobian

    def define_problem(self):
        self += RectangularQuadMesh(N=self.n)
        eqs = PoissonEquation(name="u", source=1)
        eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
        if self.fd_jacobian:
            eqs += EquationCompilationFlags(analytical_jacobian=False)
        self += eqs @ "domain"


class _Cavity(Problem):
    """Lid-driven cavity: a two-space element (C2 velocity, C1 pressure), so its nodes carry the
    dummy values that interpolate_hang_values() writes - i.e. it exercises the hanging pre-pass."""

    def __init__(self, n=16):
        super().__init__()
        self.n = n

    def define_problem(self):
        self += RectangularQuadMesh(N=self.n)
        eqs = NavierStokesEquations(dynamic_viscosity=1, mass_density=1)
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "bottom"
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "left"
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "right"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        self += eqs @ "domain"


class _Callback(CustomMathExpression):
    def eval(self, args):
        return 1.0 + 0.5 * numpy.sin(args[0])


class _AdaptivePoisson(Problem):
    """A peaked source so the error estimator really refines, leaving hanging nodes behind - and a
    Python callback in the source, so the workers have to take the GIL to evaluate it."""

    def define_problem(self):
        self += RectangularQuadMesh(N=8)
        x = var("coordinate")
        peak = 100 * exp(-200 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
        eqs = PoissonEquation(name="u", source=_Callback()(var("coordinate_x")) * peak)
        eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


class _Bratu(Problem):
    """u_t = u_xx + lam*exp(u): a fold in lam, for the augmented and eigenvalue paths."""

    def __init__(self):
        super().__init__()
        self.lam = self.get_global_parameter("lam")

    def define_problem(self):
        self += LineMesh(N=120)
        u, v = var_and_test("u")
        eqs = ScalarField("u", space="C2")
        eqs += WeakContribution(partial_t(u), v) + WeakContribution(grad(u), grad(v))
        eqs += WeakContribution(-self.lam * exp(u), v)
        eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
        self += eqs @ "domain"


class _Pitchfork(Problem):
    """u_t = u_xx + lam*u - u^3: a pitchfork at lam = pi^2. The pitchfork handler is one of the two
    that switch a code's residual form per element while assembling (see thread_state.hpp)."""

    def __init__(self):
        super().__init__()
        self.lam = self.get_global_parameter("lam")

    def define_problem(self):
        self += LineMesh(N=120)
        u, v = var_and_test("u")
        eqs = ScalarField("u", space="C2")
        eqs += WeakContribution(partial_t(u), v) + WeakContribution(grad(u), grad(v))
        eqs += WeakContribution(-self.lam * u + u ** 3, v)
        eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
        self += eqs @ "domain"


@requires_openmp
def test_plain_problem_is_bit_identical(tmp_path):
    with _Poisson() as p:
        p.set_output_directory(str(tmp_path))
        p.initialise()
        _assert_threaded_matches_serial(p)


@requires_openmp
def test_two_space_element_is_bit_identical(tmp_path):
    with _Cavity() as p:
        p.set_output_directory(str(tmp_path))
        p.initialise()
        p.solve()
        _assert_threaded_matches_serial(p)


@requires_openmp
def test_hanging_nodes_and_python_callback_are_bit_identical(tmp_path):
    with _AdaptivePoisson() as p:
        p.set_output_directory(str(tmp_path))
        p.initialise()
        p.max_refinement_level = 4
        for _ in range(3):
            p.solve()
            p.adapt()
        p.solve()
        hanging = sum(1 for n in p.get_mesh("domain").nodes() if n.is_hanging())
        assert hanging > 0, "the mesh did not refine, so the hanging pre-pass is not being tested"
        _assert_threaded_matches_serial(p)


@requires_openmp
def test_eigenproblem_matrices_are_bit_identical(tmp_path):
    with _Bratu() as p:
        p.set_output_directory(str(tmp_path))
        p.initialise()
        p.lam.value = 0.5
        p.solve()

        def grab():
            return [numpy.array(x).copy() for x in p.assemble_eigenproblem_matrices(0) if hasattr(x, "__len__")]

        serial = grab()
        before = p._get_parallel_assemblies_done()
        p._set_num_assembly_threads(NTHREADS)
        threaded = grab()
        assert p._get_parallel_assemblies_done() - before > 0
        for i, (a, b) in enumerate(zip(serial, threaded)):
            assert numpy.array_equal(a, b), "eigenproblem array %d differs" % i


@requires_openmp
def test_arclength_continuation_path_is_bit_identical(tmp_path):
    """Arclength goes through the augmented (base-problem) assembly and its multi-assembly, which is
    a different element loop from the plain Jacobian one."""
    def run(threads):
        with _Bratu() as p:
            p.set_output_directory(str(tmp_path / ("arc%d" % threads)))
            p.initialise()
            p.lam.value = 0.5
            p.solve()
            p._set_num_assembly_threads(threads)
            ds = 0.05
            path = []
            for _ in range(8):
                ds = p.arclength_continuation(p.lam, ds)
                path.append((p.lam.value, float(numpy.max(p.get_current_dofs()[0]))))
            return numpy.array(path), p._get_parallel_assemblies_done()

    serial, n_serial = run(1)
    threaded, n_threaded = run(NTHREADS)
    assert n_serial == 0 and n_threaded > 0
    assert numpy.array_equal(serial, threaded)


@requires_openmp
def test_fold_tracking_is_bit_identical(tmp_path):
    def run(threads):
        with _Bratu() as p:
            p.set_output_directory(str(tmp_path / ("fold%d" % threads)))
            p.initialise()
            p.lam.value = 0.5
            p.solve()
            p._set_num_assembly_threads(threads)
            p.activate_bifurcation_tracking(p.lam, "fold")
            p.solve()
            return p.lam.value, numpy.array(p.get_current_dofs()[0]), p._get_parallel_assemblies_done()

    lam_s, dofs_s, n_s = run(1)
    lam_t, dofs_t, n_t = run(NTHREADS)
    assert n_s == 0 and n_t > 0
    assert lam_s == lam_t
    assert numpy.array_equal(dofs_s, dofs_t)


@requires_openmp
def test_pitchfork_tracking_is_bit_identical(tmp_path):
    """The pitchfork handler writes functable->current_res_jac per element; without the per-thread
    channel in thread_state.hpp two workers would overwrite each other's residual form."""
    with _Pitchfork() as p:
        p.set_output_directory(str(tmp_path))
        p.initialise()
        p.lam.value = 8.0
        p.solve()
        _, evec = p.solve_eigenproblem(3)
        p.activate_bifurcation_tracking(p.lam, "pitchfork", eigenvector=numpy.real(evec[0]))
        # After activation: the dof vector is augmented, so this is the state both arms restart from.
        dofs0 = list(p.get_current_dofs()[0])

        p.solve()
        serial = (p.lam.value, numpy.array(p.get_current_dofs()[0]))
        before = p._get_parallel_assemblies_done()

        p.set_current_dofs(dofs0)
        p.lam.value = 8.0
        p._set_num_assembly_threads(NTHREADS)
        p.solve()
        threaded = (p.lam.value, numpy.array(p.get_current_dofs()[0]))
        assert p._get_parallel_assemblies_done() - before > 0
        assert serial[0] == threaded[0]
        assert numpy.array_equal(serial[1], threaded[1])


def test_finite_difference_jacobian_declines(tmp_path):
    """An FD Jacobian perturbs nodal data shared with the neighbouring elements, so no scheme can
    thread it. It must fall back, not race."""
    with _Poisson(n=6, fd_jacobian=True) as p:
        p.set_output_directory(str(tmp_path))
        p.initialise()
        p._set_num_assembly_threads(NTHREADS)
        p.assemble_jacobian(with_residual=True)
        assert p._get_parallel_assemblies_done() == 0


def test_without_frozen_sparsity_declines(tmp_path):
    with _Poisson(n=6) as p:
        p.set_output_directory(str(tmp_path))
        p.use_frozen_sparsity = False
        p.initialise()
        p._set_num_assembly_threads(NTHREADS)
        p.assemble_jacobian(with_residual=True)
        assert p._get_parallel_assemblies_done() == 0


def test_set_num_threads_reaches_the_element_loop(tmp_path):
    with _Poisson(n=6) as p:
        p.set_output_directory(str(tmp_path))
        p.initialise()
        assert p._get_num_assembly_threads() == 1
        p.set_num_threads(NTHREADS)
        assert p._get_num_assembly_threads() == NTHREADS
        p.set_num_threads(None)   # "the backend's own default" for the solver, serial for the loop
        assert p._get_num_assembly_threads() == 1
