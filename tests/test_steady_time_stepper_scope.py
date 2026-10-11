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

"""The time steppers must be left exactly as they were found.

Everything that wants a stationary residual, Jacobian or mass matrix out of a possibly transient
problem has to make the time steppers steady first, and used to do it by hand: save ``is_steady()``,
call ``make_steady()``, do the work, call ``undo_make_steady()`` on the ones that were not steady
before. Three of those hand-rolled blocks had a ``raise`` between their two halves, so the restore
was skipped and the problem stayed steady for good.

That is a silent failure, which is what makes it worth a test. A leaked steady flag is sticky -
``set_weights()`` does not clear it, a state dump does not record it, and neither a stationary solve
nor an eigensolve heals it, because they all use the same "restore only what was not steady" idiom
and so leave an already-steady problem steady. What a later transient run then does is print
advancing times and write output at the right instants while the solution never moves, because every
``dt`` weight is zero and the generated *steady* residual body is the one being called.

The blocks are now ``with problem.steady_time_steppers():``. The tests below cover the scope itself
and then, per converted site, that the restore survives an exception raised inside it.

Note the oracle for the scope: restore what was *found*, not unconditionally undo. Two shipped call
paths depend on that - ``switch_to_hopf_orbit`` makes the problem steady on purpose and then calls
``get_hopf_lyapunov_coefficient``, whose block must leave it steady, and
``tests/hopf_lyapunov_worker.py`` does the same by hand.
"""

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.solvers.generic import GenericEigenSolver
from pyoomph.utils.halleys_method import HalleySolver


class _DecayEqs(ODEEquations):
    """A single nonlinear ODE, ``dt(u) = -u**3`` - transient, and nothing else to go wrong."""

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u) + u ** 3, v))


class _Decay(Problem):
    def define_problem(self):
        self += (_DecayEqs() + InitialCondition(u=1)) @ "ode"


def _steadiness(problem):
    return [problem.time_stepper_pt(i).is_steady() for i in range(problem.ntime_stepper())]


# --------------------------------------------------------------------------------------------
# 1. The scope itself
# --------------------------------------------------------------------------------------------

def test_the_block_is_steady_inside_and_restores_on_the_way_out(tmp_path):
    with _Decay() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()
        before = _steadiness(problem)
        assert not any(before)                      # non-vacuity: it starts unsteady
        with problem.steady_time_steppers():
            assert all(_steadiness(problem))
        assert _steadiness(problem) == before


def test_the_block_restores_when_it_raises(tmp_path):
    """The whole point: the hand-rolled version skipped its restore here."""
    with _Decay() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()
        before = _steadiness(problem)
        with pytest.raises(ValueError):
            with problem.steady_time_steppers():
                raise ValueError("boom")
        assert _steadiness(problem) == before


def test_the_blocks_nest(tmp_path):
    with _Decay() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()
        before = _steadiness(problem)
        with problem.steady_time_steppers():
            with problem.steady_time_steppers():
                assert all(_steadiness(problem))
            # The inner block found it steady, so it must NOT have undone it.
            assert all(_steadiness(problem))
        assert _steadiness(problem) == before


def test_an_already_steady_problem_is_left_steady(tmp_path):
    """What switch_to_hopf_orbit and hopf_lyapunov_worker.py rely on."""
    with _Decay() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()
        for i in range(problem.ntime_stepper()):
            problem.time_stepper_pt(i).make_steady()
        with problem.steady_time_steppers():
            pass
        assert all(_steadiness(problem))


# --------------------------------------------------------------------------------------------
# 2. Per converted site: the restore survives the exception that used to skip it
# --------------------------------------------------------------------------------------------

def test_halley_restores_after_its_non_convergence_error(tmp_path):
    """HalleySolver.solve raises on max_iterations from inside the block."""
    with _Decay() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.setup_for_stability_analysis(analytic_hessian=True)  # Halley needs the Hessian
        problem.initialise()
        before = _steadiness(problem)
        with pytest.raises(RuntimeError, match="did not converge"):
            HalleySolver(problem).solve(max_iterations=1, accuracy=1e-30)
        assert _steadiness(problem) == before


class _RaisingEigenSolver(GenericEigenSolver):
    """An eigensolver that fails the way a backend fails: from inside solve()."""

    idname = "raising_stub"

    def supports_complex_target(self) -> bool:
        return True

    def distributed_possible(self) -> bool:
        return True

    def solve(self, neval, shift=None, sort=True, which="LM", OPpart=None, v0=None, target=None,
              custom_J_and_M=None, with_left_eigenvectors=False, quiet=True):
        raise RuntimeError("eigensolver stub failure")


def test_eigensolve_restores_after_a_backend_failure(tmp_path):
    """_solve_eigenproblem_helper used to skip its restore whenever the eigensolver threw."""
    with _Decay() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()
        problem.set_eigensolver(_RaisingEigenSolver(problem))
        before = _steadiness(problem)
        with pytest.raises(RuntimeError, match="eigensolver stub failure"):
            problem.solve_eigenproblem(1)
        assert _steadiness(problem) == before


class _RealOnlyEigenSolver(_RaisingEigenSolver):
    """A real build: cannot target a complex value, which is what get_hopf_lyapunov_coefficient
    refuses on."""

    idname = "real_only_stub"

    def supports_complex_target(self) -> bool:
        return False


def test_hopf_lyapunov_restores_after_its_complex_target_refusal(tmp_path):
    """The 397-line block with five raises in it."""
    from pyoomph.generic.bifurcation_tools import get_hopf_lyapunov_coefficient

    with _Decay() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()
        problem.set_eigensolver(_RealOnlyEigenSolver(problem))
        before = _steadiness(problem)
        with pytest.raises(Exception):
            get_hopf_lyapunov_coefficient(problem, "decay_rate", omega=1.0)
        assert _steadiness(problem) == before


# --------------------------------------------------------------------------------------------
# 3. The same block also restores the two global parameters it borrows
# --------------------------------------------------------------------------------------------

class _DrivenEqs(ODEEquations):
    """A damped oscillator driven by whatever PeriodicDrivingResponse hands it."""

    def __init__(self, driving):
        super().__init__()
        self.driving = driving

    def define_fields(self):
        self.define_ode_variable("y")
        self.define_ode_variable("ydot")

    def define_residuals(self):
        y, y_test = var_and_test("y")
        ydot, ydot_test = var_and_test("ydot")
        self.add_weak(partial_t(y) - ydot, ydot_test)
        self.add_weak(partial_t(ydot) + 0.1 * ydot + y - self.driving, y_test)


class _Driven(Problem):
    def __init__(self):
        super().__init__()
        #: Set to PeriodicDrivingResponse.get_driving_mode() before initialise().
        self.driving: Expression = Expression(0)

    def define_problem(self):
        self += _DrivenEqs(self.driving) @ "oscillator"


def test_the_response_pencil_restores_both_parameters_it_borrows(tmp_path):
    """``_assemble_response_pencil`` sets omega to 1 and the Hopf parameter to 0 for the assembly.
    It always put omega back; the Hopf parameter it did not, which clobbers it for a caller that is
    Hopf-tracking on that very parameter."""
    from pyoomph.utils.periodic_driving_response import PeriodicDrivingResponse

    with _Driven() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        pdr = PeriodicDrivingResponse(problem)
        problem.driving = pdr.get_driving_mode()
        problem.solve()
        pdr.hopf_param.value = 0.25            # non-vacuity: something to lose
        pdr.omega_param.value = 0.75
        before_steady = _steadiness(problem)
        pdr._assemble_response_pencil()
        assert pdr.hopf_param.value == 0.25
        assert pdr.omega_param.value == 0.75
        assert _steadiness(problem) == before_steady


# --------------------------------------------------------------------------------------------
# 4. The tensor cache borrows four things, and must put all four back
# --------------------------------------------------------------------------------------------

class _ParamEqs(ODEEquations):
    """``dt(u) = -k*u**3`` - nonlinear, and with a global parameter the tensor cache has to sweep."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def define_fields(self):
        self.define_ode_variable("u")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u) + self.k * u ** 3, v))


class _WithParam(Problem):
    def define_problem(self):
        self.k = self.get_global_parameter("k")
        self.k.value = 1.0
        self += (_ParamEqs(self.k.get_symbol()) + InitialCondition(u=1)) @ "ode"


def test_the_tensor_cache_restores_everything_it_borrowed_when_it_throws(tmp_path):
    """``update_tensor_cache`` zeroes the dofs, sweeps every global parameter, makes the steppers
    steady and turns the custom residual off, then does three heavy assemblies and a Hessian build.
    Without the analytic Hessian the build raises - and before the try/finally that left the problem
    with zeroed dofs, a swept parameter and the custom residual disabled."""
    from pyoomph.generic.assembly import FixedMeshMaxQuadraticNonlinearAssembly

    with _WithParam() as problem:
        problem.set_output_directory(str(tmp_path))
        problem.quiet()
        problem.initialise()          # deliberately NO setup_for_stability_analysis
        # No solve: the initial condition already puts the dof at 1, and the stationary problem here
        # is u**3 == 0, whose Jacobian is singular at its own solution.
        problem.k.value = 2.5
        problem.use_custom_residual_jacobian = True
        dofs_before = numpy.array(problem.get_current_dofs()[0])
        steady_before = _steadiness(problem)
        assert numpy.max(numpy.abs(dofs_before)) > 1e-6    # non-vacuity: there is something to lose

        assembler = FixedMeshMaxQuadraticNonlinearAssembly()
        assembler._set_problem(problem)
        with pytest.raises(Exception):
            assembler.update_tensor_cache()

        assert numpy.allclose(numpy.array(problem.get_current_dofs()[0]), dofs_before)
        assert problem.k.value == 2.5
        assert problem.use_custom_residual_jacobian is True
        assert _steadiness(problem) == steady_before
        assert not assembler._tensor_cache_valid
