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

"""A 1-D submesh under --distribute, on a rank that gets none of it.

`Mesh::distribute()` calls `setup_tree_forest()` on **every** rank, on purpose: the call is
collective, so a rank whose partition holds none of this mesh's elements still has to enter it. For
a 1-D mesh that reaches `TemplatedMeshBase1d::setup_binary_tree_forest()`, which builds one tree
root per element and hands the vector to `oomph::BinaryTreeForest` - whose constructor called
`find_neighbours()` unconditionally, and that throws on an empty forest.

`QuadTreeForest` has had an early return for exactly this since `quadtree.cc:878`, and
`OcTreeForest` likewise; only the binary tree lacked it, so the gap was 1-D only and nothing in the
suite exercised a `LineMesh` under `--distribute`. Measured on a printhead problem whose restrictor
is a `LineMesh` of about twenty elements: rank 1 raised "Trying to setup the neighbour scheme for an
empty forest" and then segfaulted.

The 1-D mesh here shares no node with the 2-D one, which is what makes METIS put all of it on one
rank - the same reason the printhead's restrictor, coupled only through the chamber ODE's external
data, ends up undivided. Without that the test would partition the line too and never reach the
empty forest.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_line_mesh_worker.py")


def _mpi_reason():
    """None if a distributed run is possible here, else the reason to skip."""
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi, have_pymetis
        if not has_mpi():
            return "pyoomph was built without MPI"
        if not have_pymetis():
            return "PyMetis is not installed, so the mesh cannot be partitioned"
    except Exception as e:  # noqa: BLE001
        return "MPI unavailable: " + str(e)
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(tmpdir, nproc, distribute, timeout=900):
    cmd = []
    if nproc is not None:
        cmd += ["mpirun", "-n", str(nproc)]
    cmd += [sys.executable, "-u", _WORKER,
            "--outdir", os.path.join(str(tmpdir), "out_%s_%s" % (nproc, distribute))]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a deadlock (nproc=%s, distribute=%s).\n"
            "--- stdout tail ---\n%s" % (timeout, nproc, distribute, (e.stdout or "")[-3000:]))
    per_rank = [json.loads(l[len("PYOOMPH_MPI_RESULT "):]) for l in proc.stdout.splitlines()
                if l.startswith("PYOOMPH_MPI_RESULT ")]
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d -- a negative value is the killing signal)\n"
            "--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == (nproc or 1), "reported from %d of %s ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, r.get("traceback", r.get("error"))
    return per_rank


def test_a_disconnected_line_mesh_survives_the_partition(tmp_path):
    per_rank = _run(tmp_path, nproc=2, distribute=True)
    assert all(r["distributed"] for r in per_rank)
    # The point of the case: at least one rank holds none of the 1-D mesh and still has to build a
    # forest for it. Without the empty-forest guard that rank died here.
    assert min(r["line_elements"] for r in per_rank) == 0
    assert sum(r["line_elements"] for r in per_rank) >= 5


def test_the_distributed_answer_matches_the_serial_one(tmp_path):
    serial = _run(tmp_path, nproc=None, distribute=False)[0]
    per_rank = _run(tmp_path, nproc=2, distribute=True)
    # Same global problem, just partitioned: the dof count is what every rank agrees on.
    assert all(r["ndof"] == serial["ndof"] for r in per_rank)
    assert sum(r["domain_elements"] for r in per_rank) >= serial["domain_elements"]
