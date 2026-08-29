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

"""activate_bifurcation_tracking(blocksolve=True) must raise, not segfault.

It installs one of oomph-lib's own block linear solvers, and both of those open with

    FoldHandler* handler_pt = static_cast<FoldHandler*>(problem_pt->assembly_handler_pt());

then call a FoldHandler method on the result. pyoomph installs MyFoldHandler / MyHopfHandler, which
derive from oomph::AssemblyHandler and NOT from those classes, so the cast reinterprets an unrelated
object and the first member access is undefined behaviour. Measured before this guard: a plain SERIAL
Bratu fold track died with "Caught signal number 11 SEGV", with the default linear solver and with
petsc_mumps alike.

Serial is the point. The refusal that used to exist covered only --distribute, as though this were an
MPI restriction; it is not, and the serial and replicated crashes went straight through it. So the
test that matters most is the plain one below - an MPI-only test would have passed against the old
code and proved nothing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the Bratu problem the MPI bifurcation tests already drive, rather than a throwaway one: it is
# known to have a fold in `lam` and is exactly the case the SEGV was measured on.
from mpi_bifurcation_worker import BratuProblem


@pytest.mark.parametrize("bifurcation_type", ["fold", "hopf"])
def test_blocksolve_is_refused_with_a_message(tmp_path, bifurcation_type):
    """The two types that actually install a block solver. The message has to name the flag."""
    with BratuProblem(N=4) as p:
        p.set_output_directory(str(tmp_path / bifurcation_type))
        p.quiet()
        p.initialise()
        with pytest.raises(RuntimeError, match="blocksolve"):
            p.activate_bifurcation_tracking("lam", bifurcation_type, blocksolve=True)


def test_blocksolve_is_refused_for_a_type_that_ignored_it(tmp_path):
    """The types that never installed a block solver silently ACCEPTED the flag and did nothing.

    That is why this refuses rather than warns: a caller passing blocksolve=True to a pitchfork track
    was getting the ordinary full augmented solve with no indication the argument had been dropped.
    """
    with BratuProblem(N=4) as p:
        p.set_output_directory(str(tmp_path / "pf"))
        p.quiet()
        p.initialise()
        with pytest.raises(RuntimeError, match="blocksolve"):
            p.activate_bifurcation_tracking("lam", "pitchfork", blocksolve=True)


def test_the_refusal_happens_before_anything_is_installed(tmp_path):
    """A guard that fires after the handler is swapped would leave the problem in a tracking state."""
    with BratuProblem(N=4) as p:
        p.set_output_directory(str(tmp_path / "clean"))
        p.quiet()
        p.initialise()
        with pytest.raises(RuntimeError, match="blocksolve"):
            p.activate_bifurcation_tracking("lam", "fold", blocksolve=True)
        assert p.get_bifurcation_tracking_mode() == "", \
            "the refusal left a bifurcation handler installed"


def test_the_default_still_activates(tmp_path):
    """The control: blocksolve=False must be untouched, or the refusals above prove nothing."""
    with BratuProblem(N=6) as p:
        p.set_output_directory(str(tmp_path / "ok"))
        p.quiet()
        p.initialise()
        p.lam.value = 4.0
        p.solve()
        p.solve_eigenproblem(2, quiet=True)
        p.activate_bifurcation_tracking("lam", "fold")
        assert p.get_bifurcation_tracking_mode() == "fold"
        p.deactivate_bifurcation_tracking()
