#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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

# The two PreCICE_Coupling tutorials must run on a machine with no preCICE.
#
# They are documentation of the coupling, but both are written to run monolithically as well - the
# source term and every boundary value come from an analytic solution, and the script integrates the
# squared error against it. That branch is ordinary pyoomph and needs no preCICE, yet it was
# unreachable without libprecice for as long as pyoomph/solvers/precice_adapter.py imported precice at
# module scope: the tutorial pass reported both scripts as skipped on every machine that has no
# preCICE, the GitHub runners every single night. The import now sits in initialize_problem().
#
# What is checked here, and why each case is needed:
#
#   the adapter imports without preCICE      the regression itself.
#   "precice" not in sys.modules afterwards  this is what makes the first case meaningful on duarte,
#                                            which HAS preCICE - without it the test would pass there
#                                            with a module-level import restored.
#   a participant still demands the package  a lazy import must not silently turn the coupling into a
#                                            no-op, and the error has to stay the ModuleNotFoundError
#                                            that citools/test_all_tutorial_scripts.py reads to
#                                            report a skip rather than a failure.
#
# The absence of preCICE is reproduced with a package of that name on PYTHONPATH that raises
# ModuleNotFoundError when imported, rather than by uninstalling anything. Hiding the real one by
# dropping the user site directory is not an option here: on a development checkout that is where the
# editable pyoomph install lives too.

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "precice_import_worker.py")


def _blocker_on_path(tmp_path):
    """A 'precice' package that cannot be imported, prepended to the child's PYTHONPATH."""
    pkg = tmp_path / "no_precice" / "precice"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "# Shadows a real preCICE install, to reproduce a machine that has none.\n"
        "raise ModuleNotFoundError(\"No module named 'precice'\", name='precice')\n")
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(tmp_path / "no_precice") + (os.pathsep + inherited if inherited else "")
    return env


def _run(phase, env=None, timeout=300):
    return subprocess.run([sys.executable, "-u", _WORKER, "--phase", phase],
                          cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)


def test_adapter_imports_without_precice(tmp_path):
    proc = _run("import_only", env=_blocker_on_path(tmp_path))
    assert proc.returncode == 0, (
        "the preCICE adapter must import on a machine without preCICE -- this is what keeps the two "
        "PreCICE_Coupling tutorials from being skipped there.\n--- stdout ---\n%s\n--- stderr ---\n%s"
        % (proc.stdout[-3000:], proc.stderr[-3000:]))
    assert "ADAPTER_IMPORTED" in proc.stdout
    # Without this line the test would also pass with the module-level import back, on any machine
    # that happens to have preCICE installed.
    assert "PRECICE_PRESENT no" in proc.stdout, (
        "the blocker did not take effect, so this run proves nothing.\n--- stdout ---\n%s"
        % proc.stdout[-2000:])
    assert "PYOOMPH_WORKER_DONE" in proc.stdout


def test_adapter_does_not_pull_in_precice_where_it_exists():
    # Same import, this machine's own environment: the assertion that matters is the one inside the
    # worker ("precice" not in sys.modules), which is what a re-added top-level import trips over.
    proc = _run("import_only")
    assert proc.returncode == 0, (
        "importing the adapter must not import preCICE itself.\n--- stdout ---\n%s\n--- stderr ---\n%s"
        % (proc.stdout[-3000:], proc.stderr[-3000:]))
    assert "ADAPTER_IMPORTED" in proc.stdout
    assert "PYOOMPH_WORKER_DONE" in proc.stdout


def test_a_participant_still_requires_precice(tmp_path):
    proc = _run("coupled", env=_blocker_on_path(tmp_path))
    assert proc.returncode != 0, (
        "asking for a preCICE participant without preCICE must fail, not quietly do nothing.\n"
        "--- stdout ---\n%s" % proc.stdout[-3000:])
    assert "COUPLED_RAN" not in proc.stdout
    last = (proc.stderr.strip().splitlines() or [""])[-1]
    # citools/test_all_tutorial_scripts.py reads the LAST line to decide skip-vs-failure.
    assert last.strip() == "ModuleNotFoundError: No module named 'precice'", (
        "the last line is what the tutorial harness classifies on, and it reads:\n%s" % last)
