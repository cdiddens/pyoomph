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

# Gmsh's Mesh.HighOrderOptimize, which pyoomph asks for on every order-2 mesh, used to be able to end
# the process outright.
#
# Gmsh's General.AbortOnError defaults to 2, "throw an exception unless in interactive mode", and its
# C API turns that throw into the error the Python binding raises - where the throw can unwind. The
# high-order optimizer raises its Msg::Error from inside an OpenMP region, where it cannot: the
# process dies of std::terminate. SIGABRT, no traceback, nothing to catch, and a script that has been
# solving for an hour is simply gone. A strongly anisotropic order-2 mesh walks into it - the worker's
# torus of minor radius 0.03 is enough:
#
#   Error   : Failed to reach critical value in pass 0 for measure(s): ScaledJac
#   terminate called after throwing an instance of 'std::runtime_error'
#
# So meshing now runs with AbortOnError=1, where gmsh reports rather than throws, and the reported
# errors are classified afterwards, outside any OpenMP region: an error that says the optimizer could
# not reach its quality target is a warning (gmsh returns the mesh it started from, which is the mesh
# Mesh.HighOrderOptimize=0 would have produced), and anything else is raised as a RuntimeError. That
# last half matters as much as the first: with gmsh no longer throwing, nothing else reports those.
#
# The meshing half of this runs in a subprocess. A regression is SIGABRT, not a failed assertion, and
# in-process it would take the whole pytest run with it.

import os
import subprocess
import sys

import pytest

from pyoomph.meshes.gmsh import GmshTemplate, _report_gmsh_errors, _GMSH_NONFATAL_ERRORS

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "gmsh_high_order_worker.py")


def _mesh_the_torus(tmp_path, high_order_optimize=None):
    cmd = [sys.executable, "-u", _WORKER, "--outdir", str(tmp_path / "torus")]
    if high_order_optimize is not None:
        cmd += ["--high-order-optimize", str(high_order_optimize)]
    proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=900)
    return proc, (proc.stdout or "") + (proc.stderr or "")


@pytest.mark.parametrize("high_order_optimize", [None, 0],
                         ids=["optimizer_as_configured", "optimizer_switched_off"])
def test_an_anisotropic_order_two_mesh_does_not_kill_the_process(tmp_path, high_order_optimize):
    proc, out = _mesh_the_torus(tmp_path, high_order_optimize)
    assert proc.returncode != -6, \
        "the high-order optimizer aborted the process again (SIGABRT):\n" + out[-3000:]
    assert proc.returncode == 0, "the worker failed:\n" + out[-3000:]
    nnode = [line for line in out.splitlines() if line.startswith("NNODE ")]
    assert len(nnode) == 1 and int(nnode[0].split()[1]) > 0, "no mesh came out:\n" + out[-3000:]


def test_the_optimizer_giving_up_is_reported_and_says_what_to_do(tmp_path):
    """Surviving is not enough: the run has to say that the elements were left uncurved."""
    _proc, out = _mesh_the_torus(tmp_path)
    assert "Failed to reach critical value" in out, \
        "the torus no longer provokes the optimizer, so this file guards nothing:\n" + out[-3000:]
    assert "high_order_optimize=0" in out, "the warning does not name the way out:\n" + out[-3000:]


def test_the_mesh_is_the_same_whether_the_optimizer_gave_up_or_never_ran(tmp_path):
    """What the warning claims: an optimizer that fails leaves the mesh it was handed."""
    _p1, out1 = _mesh_the_torus(tmp_path / "a")
    _p2, out2 = _mesh_the_torus(tmp_path / "b", high_order_optimize=0)
    n1 = [l for l in out1.splitlines() if l.startswith("NNODE ")][0]
    n2 = [l for l in out2.splitlines() if l.startswith("NNODE ")][0]
    assert n1 == n2, "the failed optimization changed the mesh: " + n1 + " vs " + n2


# ----------------------------------------------------------------------------------------------
# The classification, without meshing anything
# ----------------------------------------------------------------------------------------------

def test_an_unknown_error_is_raised():
    """The half that keeps AbortOnError=1 honest: gmsh no longer throws, so an error it reports has
    to be raised here or it is lost."""
    with pytest.raises(RuntimeError, match="The 1D mesh seems not to be forming a closed loop"):
        _report_gmsh_errors(["Info: Meshing 1D...",
                             "Error: The 1D mesh seems not to be forming a closed loop"], None)


def test_the_optimizers_own_complaints_are_not_raised(capsys):
    _report_gmsh_errors(["Info: Optimizing mesh (HighOrder)...",
                         "Error: Failed to reach critical value in pass 0 for measure(s): ScaledJac",
                         "Error: Optimization failed (some measures below critical value)"], None)
    printed = capsys.readouterr().out
    assert "ScaledJac" in printed and "high_order_optimize=0" in printed


def test_one_unknown_error_among_the_known_ones_still_raises():
    """A quality complaint must not cover for a real failure that came with it."""
    with pytest.raises(RuntimeError, match="Invalid boundary mesh"):
        _report_gmsh_errors(["Error: Failed to reach critical value in pass 0 for measure(s): ScaledJac",
                             "Error: Invalid boundary mesh for parametrization"], None)


def test_a_clean_log_reports_nothing(capsys):
    _report_gmsh_errors(["Info: Meshing 2D...", "Warning: something cosmetic"], None)
    assert capsys.readouterr().out == ""


def test_the_default_still_asks_for_the_optimization():
    """Switching it off by default would change every curved mesh pyoomph builds; the point here is
    that it can be switched off, not that it is."""
    assert GmshTemplate().high_order_optimize == 1
    assert _GMSH_NONFATAL_ERRORS, "the non-fatal list is what keeps a torus meshable at all"
