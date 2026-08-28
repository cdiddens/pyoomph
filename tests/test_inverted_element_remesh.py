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

"""RemeshWhen(on_inverted_element=True): remeshing as the response to a folded mesh.

The case is the notch of `dev_docs/examples/inverted_element_notch.py`. A Laplace-smoothed mesh on a
unit square, with a Gaussian notch of linearly growing depth pushed into the top edge. The domain
stays meshable throughout; what folds at t = 0.1565 is the harmonic extension into a non-convex
shape. That matters, because it makes the fold a discretisation artefact a remesh genuinely repairs
rather than a domain collapsing, which nothing could repair.

**No step size gets past it.** The deformation is a function of t alone, so the fold sits at a fixed
TIME: halving dt only approaches it more slowly, and the mesh is folded at t_fold + eps for every
eps > 0. Without the trigger the run therefore dies there, and `test_without_the_trigger_the_run_dies_at_the_fold`
pins exactly that - it is what stops the positive test below from passing for some unrelated reason.

Everything runs in a subprocess, because `set_detect_inverted_elements` is process-wide.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "inverted_remesh_worker.py")
_FOLD_TIME = 0.1565      # measured, six digits: 0.156540
_ENDTIME = 0.30          # comfortably past it, and short enough to stay a fast test


def _mpi_reason():
    """None if an MPI run is possible here, else the reason to skip."""
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    return None


def _run(tmp_path, on_inverted, detect_only=False, nproc=1, distribute=False, timeout=1800):
    os.makedirs(str(tmp_path), exist_ok=True)
    cmd = [sys.executable, "-u", _WORKER, "--outdir", str(tmp_path / "out"), "--endtime", str(_ENDTIME)]
    if on_inverted:
        cmd.append("--on-inverted")
    if detect_only:
        cmd.append("--detect-only")
    if distribute:
        cmd.append("--distribute")
    env = dict(os.environ)
    if nproc > 1:
        cmd = ["mpirun", "-n", str(nproc)] + cmd
        # Importing pyoomph calls MPI_Init, so THIS pytest process is already a singleton MPI job and
        # owns an Open MPI session directory under TMPDIR. A nested mpirun collides with it and dies
        # with exit code 1 and NO diagnostics at all - which is exactly what this test hit first, and
        # what the same comment in test_mpi_eigenvalues.py already records. Give the child its own.
        ompi_tmp = os.path.join(str(tmp_path), "_ompi_session")
        os.makedirs(ompi_tmp, exist_ok=True)
        env["TMPDIR"] = ompi_tmp
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=_HERE, env=env)
    out = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_RESULT "):
            out.append(json.loads(line[len("PYOOMPH_RESULT "):]))
    if not out:
        tail = "CMD=" + repr(cmd) + "\nSTDOUT=" + repr(proc.stdout[-1500:]) + "\nSTDERR=" + repr(proc.stderr[-1500:])
        raise AssertionError("worker produced no result (exit %d):\n%s" % (proc.returncode, tail))
    for r in out:
        assert "error" not in r, r["error"]
    return out


def test_the_option_is_off_by_default():
    from pyoomph.equations.generic import RemeshingOptions
    assert RemeshingOptions().on_inverted_element is False
    # And the detection it would arm is off too, in a process where nothing asked for it. This is
    # what keeps the ~2% assembly cost of the check off everybody else's runs.
    from pyoomph._pyoomph_core import get_detect_inverted_elements
    assert get_detect_inverted_elements() is False


def test_arming_it_switches_the_detection_on(tmp_path):
    # In a subprocess: the switch is process-wide, so doing this inline would arm it for every test
    # that runs afterwards in the same worker.
    code = (
        "import sys; sys.path.insert(0,%r)\n"
        "from inverted_remesh_worker import NotchProblem\n"
        "from pyoomph._pyoomph_core import get_detect_inverted_elements\n"
        "print('BEFORE',get_detect_inverted_elements())\n"
        "with NotchProblem(on_inverted=True) as p:\n"
        "    p.set_output_directory(%r); p.quiet(); p.initialise()\n"
        "    print('AFTER',get_detect_inverted_elements())\n"
        "    print('DOMAINS',len(p._domains_remesh_on_inversion))\n"
    ) % (_HERE, str(tmp_path / "out"))
    proc = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True, text=True,
                          timeout=600, cwd=_HERE)
    assert "BEFORE False" in proc.stdout, proc.stdout + proc.stderr
    assert "AFTER True" in proc.stdout, proc.stdout + proc.stderr
    assert "DOMAINS 1" in proc.stdout, proc.stdout + proc.stderr


def test_with_detection_but_no_trigger_the_run_dies_at_the_fold(tmp_path):
    """The control, and it has to be DETECTION ON, TRIGGER OFF rather than "nothing switched on".

    With detection off the run does not fail at all - it finishes, quietly, on a mesh that has been
    inside out since t = 0.1565, because the J pyoomph integrates with is sqrt(det(g_ab)) and is
    non-negative by construction. That is the pre-detector behaviour and the whole reason the
    detector exists, so it is the wrong control: it would make the positive test below look like an
    improvement over a run that "worked".
    """
    r = _run(tmp_path, on_inverted=False, detect_only=True)[0]
    assert r.get("failed"), "the run was expected to fail at the fold, but finished"
    assert r["remeshes"] == 0, "nothing should have remeshed with the trigger off"
    assert r["inversion_reports"] > 0, "the detector never fired, so this is not the control it claims"
    # It stalls just past the fold rather than exactly on it: the step that discovers the fold is
    # taken from before it, so t lands within one (capped) step of _FOLD_TIME and then stops. What
    # matters is that it stops far short of the end time, not the third digit.
    assert _FOLD_TIME - 1e-3 < r["t"] < _FOLD_TIME + 0.06, \
        "expected the run to stall near the fold time %g, got t=%g" % (_FOLD_TIME, r["t"])
    assert r["t"] < _ENDTIME - 0.05


def test_with_the_trigger_the_run_gets_past_the_fold(tmp_path):
    r = _run(tmp_path, on_inverted=True)[0]
    assert not r.get("failed"), "the run failed: " + str(r.get("failed"))
    assert r["t"] >= _ENDTIME - 1e-6, "t=%g did not reach %g" % (r["t"], _ENDTIME)
    assert r["remeshes"] > 0, "it got past the fold without ever remeshing, which cannot be right"
    assert r["inversion_reports"] > 0, "nothing ever reported an inversion, so the trigger is untested"
    csqr = float(r["csqr"])
    assert csqr == csqr and csqr > 0, "the solution is NaN or empty after the remeshes"


@pytest.mark.slow
@pytest.mark.skipif(_mpi_reason() is not None, reason=str(_mpi_reason()))
@pytest.mark.parametrize("distribute", [False, True])
def test_mpi_matches_serial(tmp_path, distribute):
    """Replicated and distributed must reach the same place as serial.

    Both MPI modes split the ELEMENT loop by rank, so an inversion is seen only by whichever rank
    holds the folded element while remeshing is collective. Before the report was reduced across the
    ranks, `mpirun -n 2` on this case did not merely disagree - it HUNG, with the throwing rank out
    of the element loop and the others still inside the assembly's collectives.
    """
    serial = _run(tmp_path / "serial", on_inverted=True)[0]
    par = _run(tmp_path / "par", on_inverted=True, nproc=2, distribute=distribute)
    assert len(par) == 2

    # How closely the integrated solution may be expected to match is not the same question in the
    # two modes, and using one number for both would either hide a defect or fail for no reason.
    # REPLICATED: every rank holds the whole mesh, so the remesh is the serial remesh and what comes
    # out is the serial answer to round-off. DISTRIBUTED: the remesher rebuilds the geometry from
    # boundary data gathered across the ranks and does not reproduce the serial mesh element for
    # element, so after three remeshes and their interpolations the two runs are solving on genuinely
    # different meshes. Measured spread there is ~8e-5 relative; 1e-3 leaves room for that without
    # being loose enough to hide anything real, since a wrong partition or a lost remesh moves this
    # by percent. The two things that MUST still agree exactly are below.
    rtol = 1e-3 if distribute else 1e-8

    for r in par:
        assert not r.get("failed"), "rank %d failed: %s" % (r["rank"], r.get("failed"))
        assert r["distributed"] is distribute
        # The step sequence and the number of remeshes are decisions, not floating-point results:
        # every rank reduces the inversion report before acting on it, so a disagreement here means
        # the ranks took different branches - the failure mode this whole mechanism exists to prevent.
        assert abs(r["t"] - serial["t"]) < 1e-9, "t %g vs serial %g" % (r["t"], serial["t"])
        assert r["remeshes"] == serial["remeshes"], \
            "remeshed %d times against %d serially" % (r["remeshes"], serial["remeshes"])
        assert abs(r["csqr"] - serial["csqr"]) < rtol * max(1.0, abs(serial["csqr"]))

    # Between the two ranks of ONE run there is nothing left to differ: the same collective produced
    # both, so this stays tight in either mode.
    assert abs(par[0]["csqr"] - par[1]["csqr"]) < 1e-12 * max(1.0, abs(par[0]["csqr"]))
