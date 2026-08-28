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

# Reporting an error after another Problem has been destroyed.
#
# oomph-lib's TerminateHelper owns a namespace-level std::stringstream that every OomphLibException
# writes its banner into. Problem's CONSTRUCTOR allocates it (TerminateHelper::setup()) and Problem's
# DESTRUCTOR frees it and sets it to null (TerminateHelper::clean_up_memory()) -- so with more than
# one Problem alive, the first one destroyed pulled the stream out from under all the others, and the
# next error any of them raised segfaulted while writing its message.
#
# The trigger is ordinary:
#
#     p = Problem(); ... solve ...
#     p = Problem(); ... solve that fails ...     # rebinding destroyed the first one
#
# i.e. any loop over cases that reuses the variable. Fixed by reference-counting the stream in the
# vendored copy (//FOR PYOOMPH), plus null guards on the two places that dereference it -- an error
# can also be raised when no Problem is alive at all.
#
# Run in subprocesses: the failure mode is a SIGSEGV, which would take the pytest process with it.

import os
import subprocess
import sys
import textwrap

import pytest

_PREAMBLE = textwrap.dedent("""
    from pyoomph import Problem, DirichletBC
    from pyoomph.equations.navier_stokes import NavierStokesEquations
    from pyoomph.meshes.simplemeshes import RectangularQuadMesh

    class Cavity(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=4))
            eqs = NavierStokesEquations(dynamic_viscosity=0.05, mass_density=1)
            for b in ["left", "right", "bottom"]:
                eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
            eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
            eqs += DirichletBC(pressure=0) @ "bottom/left"
            self.add_equations(eqs @ "domain")

    def solve(name, fail):
        p = Cavity()
        with p:
            p.set_output_directory(name)
            p.quiet()
            p.initialise()
            if fail:
                p.max_residuals = 1e-12   # any residual trips it -> NewtonSolverError
            try:
                p.solve()
                return "solved"
            except Exception as e:
                return type(e).__name__
""")


def _run(tmp_path, body, timeout=300):
    script = tmp_path / "case.py"
    script.write_text(_PREAMBLE + textwrap.dedent(body))
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, (
        "exited %d (a negative value is the killing signal -- -11 is SIGSEGV)\n"
        "--- stdout ---\n%s\n--- stderr tail ---\n%s"
        % (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:]))
    return proc.stdout


@pytest.mark.slow
def test_error_after_rebinding_the_problem_variable(tmp_path):
    """The reported case: the second Problem replaces the first, then fails."""
    out = _run(tmp_path, """
        print("A", solve("a", False), flush=True)
        print("B", solve("b", True), flush=True)
        print("DONE", flush=True)
    """)
    assert "A solved" in out
    assert "B RuntimeError" in out, "the failing solve did not raise cleanly: %r" % out
    assert "DONE" in out


@pytest.mark.slow
def test_error_in_each_of_several_problems_in_turn(tmp_path):
    """Several failures in a row, each after the previous Problem has been destroyed."""
    out = _run(tmp_path, """
        for i in range(3):
            print("R%d" % i, solve("r%d" % i, True), flush=True)
        print("DONE", flush=True)
    """)
    for i in range(3):
        assert "R%d RuntimeError" % i in out, "iteration %d did not raise cleanly: %r" % (i, out)
    assert "DONE" in out


@pytest.mark.slow
def test_error_while_another_problem_is_still_alive(tmp_path):
    """The other order: keep the first Problem alive and fail in the second."""
    out = _run(tmp_path, """
        keep = Cavity()
        with keep:
            keep.set_output_directory("keep"); keep.quiet(); keep.initialise(); keep.solve()
        print("KEEP solved", flush=True)
        print("B", solve("b", True), flush=True)
        print("DONE", flush=True)
    """)
    assert "KEEP solved" in out
    assert "B RuntimeError" in out
    assert "DONE" in out
