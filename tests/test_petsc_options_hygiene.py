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

# What PetscFinalize says about the options database when the run is over.
#
# PETSc audits its options database at PetscFinalize and warns about every option nothing ever read
# ("There are N unused database options ... could be spelling mistake"). That is a typo detector for
# what the USER typed on the command line, and it is worth having. It stops being worth anything the
# moment it also reports things the user did not type and cannot act on, which is what it did:
#
#   * pyoomph's own defaults. slepc_mumps is the autodetected eigensolver wherever PETSc has MUMPS, and
#     its use_mumps() configures SLEPc's spectral transform when the eigensolver is CONSTRUCTED, long
#     before anyone asks for an eigenvalue -- so every run that solved with pardiso or superlu and
#     never touched an eigenproblem ended with five st_-prefixed options listed as suspect.
#   * pyoomph's own command line. petsc4py was handed the raw sys.argv, and PETSc records every
#     dash-prefixed token it is given, so --outdir, --distribute, -P ... came back as unused options.
#
# Both are now accounted for (pyoomph/solvers/petsc.py), and the third test here is the reason to be
# careful about how: a genuine typo must still be reported, or the check has simply been switched off.
#
# Has to run out-of-process: none of this is observable until the interpreter shuts down.

import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "petsc_options_worker.py")

_LEFTOVER_MARKER = "unused database option"


def _run_worker(tmp_path, *extra_args):
    """The worker's combined output, or a skip if this PETSc cannot exercise the case."""
    pytest.importorskip("petsc4py", reason="PETSc options hygiene is only testable with petsc4py")
    cmd = [sys.executable, _WORKER, "--outdir", os.path.join(str(tmp_path), "out")] + list(extra_args)
    proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    if "PYOOMPH_NO_MUMPS" in out:
        pytest.skip("this PETSc has no MUMPS, so no st_ options are set to be left over")
    assert proc.returncode == 0, "worker failed:\n" + out[-3000:]
    assert "PYOOMPH_WORKER_DONE" in out, "worker did not reach the end:\n" + out[-3000:]
    return out


def test_pyoomphs_own_defaults_are_not_reported(tmp_path):
    """The reported case: SLEPc/MUMPS configured, superlu doing the solving, no eigenproblem at all."""
    out = _run_worker(tmp_path)
    assert _LEFTOVER_MARKER not in out, "PetscFinalize reported pyoomph's own options:\n" + out[-3000:]


def test_pyoomphs_own_command_line_flags_are_not_reported(tmp_path):
    """--outdir is already in every invocation above; -P is the one short flag pyoomph owns."""
    out = _run_worker(tmp_path, "--distribute", "-P", "max_newton_iterations=5")
    assert _LEFTOVER_MARKER not in out, "PetscFinalize reported pyoomph's own flags:\n" + out[-3000:]


def test_a_real_typo_is_still_reported(tmp_path):
    """The check must still do its job -- otherwise the two tests above are passing for the wrong
    reason, namely that the warning was switched off wholesale."""
    out = _run_worker(tmp_path, "-not_a_petsc_option", "3")
    assert _LEFTOVER_MARKER in out, "a mistyped PETSc option went unreported:\n" + out[-3000:]
    assert "-not_a_petsc_option" in out
