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


# Eigendynamics animations on a mesh distributed with --distribute.
#
# An animation frame is the base state plus Re(factor*eigenvector), and a mirrored half of the same
# frame uses a different factor. The plot code runs on rank 0 alone inside run_with_global_mesh_data,
# so the perturbation is part of the merge REQUEST and every rank applies it
# (merge_perturbed_global_mesh_data, pyoomph/meshes/meshdatamerge.py). Before that it was refused
# outright, and docs/source/tutorial/plotting/eigendynamics.py was the last tutorial that could not run
# distributed.
#
# Everything compared here is numbering-independent -- counts, a digest over the sorted coordinates,
# permutation-invariant field statistics -- except the L2 distance between the perturbed and the base
# state, which is taken WITHIN one run, where both merged entries share the merged node order.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_eigendynamics_worker.py")

#: The perturbed state goes through the eigensolver, so it agrees with a serial run only to round-off.
_RTOL = 1e-8


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:  # noqa: BLE001
        return "MPI unavailable: " + str(e)
    try:
        from petsc4py import PETSc  # type:ignore  # noqa: F401
        import slepc4py  # type:ignore  # noqa: F401
    except Exception:
        return "petsc4py/slepc4py not available (the eigensolve needs SLEPc)"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(tmpdir, case, nproc=None, distribute=False, timeout=900):
    cmd = ["mpirun", "-n", str(nproc)] if nproc is not None else []
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir), "--case", case]
    if distribute:
        cmd += ["--distribute"]
    env = dict(os.environ)
    if nproc is not None:
        # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
        # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
        ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
        os.makedirs(ompi_tmp, exist_ok=True)
        env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose: the failure mode of a request that is not replayed by every rank is a
        # DEADLOCK, and that has to surface as a failure rather than as a suite that never finishes.
        raise AssertionError("%s did not finish within %d s -- suspect a collective that not every rank "
                             "reached.\n--- stdout tail ---\n%s" % (" ".join(cmd), timeout, (e.stdout or "")[-3000:]))
    per_rank = [json.loads(line[len("PYOOMPH_MPI_RESULT "):])
                for line in proc.stdout.splitlines() if line.startswith("PYOOMPH_MPI_RESULT ")]
    assert per_rank, "no result from %s (exit %d)\n--- stdout ---\n%s\n--- stderr ---\n%s" % (
        " ".join(cmd), proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:])
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (r["rank"], r["error"], r.get("traceback", ""))
    return per_rank


def _assert_same_summary(what, got, ref):
    assert got["nnode"] == ref["nnode"] and got["nelem"] == ref["nelem"], \
        "%s: merged %d nodes / %d elements, serial %d / %d" % (what, got["nnode"], got["nelem"],
                                                               ref["nnode"], ref["nelem"])
    assert got["coord_digest"] == ref["coord_digest"], "%s: the merged node positions differ from serial" % what
    for field in ("u_sum", "u_sqsum", "u_absum", "u_max", "uplus_sum", "uplus_sqsum", "uplus_max"):
        assert got[field] == pytest.approx(ref[field], rel=_RTOL), \
            "%s: %s is %r, serial %r" % (what, field, got[field], ref[field])


@pytest.mark.parametrize("case", ["symmetric", "scope", "plotter"])
@pytest.mark.parametrize("nproc", [2, 3])
def test_perturbed_merge_matches_serial(tmp_path, case, nproc):
    """The animated state, merged from a distributed mesh, must be the state a serial run draws.

    ``symmetric`` calls the collective from every rank, ``scope`` goes through the request scope that
    the plotter really uses (rank 0 asks, the others serve), and ``plotter`` drives the whole thing
    through Problem.create_eigendynamics_animation.
    """
    serial = _run(tmp_path / ("serial_" + case), case)[0]
    per_rank = _run(tmp_path / ("mpi_%s_%d" % (case, nproc)), case, nproc=nproc, distribute=True)
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    root = [r for r in per_rank if r["rank"] == 0][0]
    assert root["distributed"] is True
    assert root["ndof"] == serial["ndof"], "the distributed run solves a different problem"

    if case == "plotter":
        assert set(root["frames"]) == set(serial["frames"])
        for name in sorted(serial["frames"]):
            _assert_same_summary("frame " + name, root["frames"][name], serial["frames"][name])
        return

    _assert_same_summary("base", root["base"], serial["base"])
    for side in ("right", "left"):
        _assert_same_summary(side, root[side], serial[side])
        assert root[side + "_l2"] == pytest.approx(serial[side + "_l2"], rel=_RTOL), \
            ("%s: |perturbed-base|^2 is %r, serially %r. A perturbation that does not reach every rank "
             "lands here." % (side, root[side + "_l2"], serial[side + "_l2"]))
    # The two mirror halves must really be different states, or none of the above proves anything
    assert root["right"]["u_sum"] != pytest.approx(root["left"]["u_sum"], rel=1e-6)
    assert root["right"]["u_sum"] != pytest.approx(root["base"]["u_sum"], rel=1e-6)
    if case == "scope":
        # The perturbed extraction must leave the ordinary cache as it found it: the plain request
        # after the two perturbed ones has to give exactly the plain request before them.
        _assert_same_summary("the plain request after the perturbed ones", root["base_again"], root["base"])


@pytest.mark.parametrize("nproc", [2, 3])
def test_no_rank_is_left_perturbed(tmp_path, nproc):
    """Every rank must have its dofs back afterwards -- the request restores in a finally, on all of them."""
    for r in _run(tmp_path / ("restore_%d" % nproc), "scope", nproc=nproc, distribute=True):
        assert r["dof_drift"] == 0.0, "rank %d kept a perturbed state (drift %r)" % (r["rank"], r["dof_drift"])


@pytest.mark.parametrize("nproc", [2, 3])
def test_a_perturbation_that_misses_a_rank_is_detectable(tmp_path, nproc):
    """The comparison above has to be able to see a partial perturbation, or it proves nothing.

    The literal old bug -- rank 0 perturbing on its own inside the plot block -- cannot be reproduced:
    set_current_dofs is collective, so one rank calling it alone deadlocks. This is the same defect
    without the deadlock: everybody perturbs, but only the dofs of rank 0's block.
    """
    serial = _run(tmp_path / "serial_partial", "partial")[0]
    root = [r for r in _run(tmp_path / ("partial_%d" % nproc), "partial", nproc=nproc, distribute=True)
            if r["rank"] == 0][0]
    assert root["right_l2"] != pytest.approx(serial["right_l2"], rel=1e-3), \
        "a perturbation restricted to one rank's dofs gave the same answer as the full one, so the " \
        "assertions in the other tests cannot see the difference either"


@pytest.mark.parametrize("case", ["symmetric", "plotter"])
def test_replicated_mpirun_is_the_unchanged_local_path(tmp_path, case):
    """Without --distribute nothing is merged and the old code path runs; it must still match serial."""
    serial = _run(tmp_path / ("serial_rep_" + case), case)[0]
    per_rank = _run(tmp_path / ("rep_" + case), case, nproc=2, distribute=False)
    root = [r for r in per_rank if r["rank"] == 0][0]
    assert root["distributed"] is False
    if case == "plotter":
        for name in sorted(serial["frames"]):
            _assert_same_summary("frame " + name, root["frames"][name], serial["frames"][name])
    else:
        _assert_same_summary("base", root["base"], serial["base"])
        _assert_same_summary("right", root["right"], serial["right"])
