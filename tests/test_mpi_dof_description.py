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

# Problem.get_dof_description() under --distribute.
#
# It walks the mesh tree and asks each mesh to classify the dofs it owns, numbering the type names as
# it goes. Distributed, a mesh can have no elements at all on a rank - an interface that lies
# entirely on somebody else - and such a mesh used to answer with no names either, which shifted
# every later mesh's type indices on that rank and made the walk raise outright. What is checked here
# is therefore agreement: the answer must be the same on every rank, and the same one a serial run
# gives, since the residual vector the callers index it with is gathered.

import collections
import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_dof_description_worker.py")


def _mpi_reason():
    """None if a distributed run is possible here, else the reason to skip."""
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:
        return "MPI unavailable: " + str(e)
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(nproc, outdir, distribute, timeout=1800):
    cmd = ["mpirun", "-n", str(nproc)]
    if nproc > 1 and os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--outdir", str(outdir)]
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(outdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    results = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            with open(line[len("PYOOMPH_MPI_RESULT "):].strip()) as f:
                results.append(json.load(f))
    if len(results) != nproc:
        raise AssertionError("got %d of %d results (exit %d)\n--- stdout tail ---\n%s\n"
                             "--- stderr tail ---\n%s" % (len(results), nproc, proc.returncode,
                                                          proc.stdout[-3000:], proc.stderr[-3000:]))
    for r in results:
        assert "error" not in r, "rank %d failed: %s\n%s" % (r["rank"], r["error"],
                                                             r.get("traceback", ""))
    return sorted(results, key=lambda r: r["rank"])


def _histogram(result):
    """How many dofs of each type name, which is what survives the renumbering a partition causes."""
    return collections.Counter(result["names"][t] for t in result["types"])


@pytest.mark.parametrize("nproc", [2, 3])
def test_distributed_dof_description_agrees_with_serial(nproc, tmp_path):
    serial = _run(1, tmp_path / "serial", distribute=False)[0]
    assert not serial["distributed"]
    assert min(serial["types"]) >= 0, "the serial run left dofs undescribed"

    per_rank = _run(nproc, tmp_path / ("np%d" % nproc), distribute=True)
    assert per_rank[0]["distributed"], "the run was not distributed at all"
    # The case the walk used to raise on has to still be in the partition, or this passes for the
    # wrong reason.
    assert any(n == 0 for r in per_rank for n in r["nelement"].values()), \
        "no rank ended up with an empty mesh, so the empty-mesh case was not exercised"
    for r in per_rank:
        # Every dof of the WHOLE problem, not just this rank's share: the callers index the answer
        # with the gathered residual vector.
        assert len(r["types"]) == serial["ndof"]
        assert min(r["types"]) >= 0, "rank %d left %d dofs undescribed" % (
            r["rank"], sum(1 for t in r["types"] if t < 0))
        assert r["names"] == per_rank[0]["names"], "rank %d disagrees about the type names" % r["rank"]
        assert r["types"] == per_rank[0]["types"], "rank %d disagrees about the dof types" % r["rank"]
    assert per_rank[0]["names"] == serial["names"]
    # The dof ORDER differs (distributing renumbers them by partition), the composition does not.
    assert _histogram(per_rank[0]) == _histogram(serial)
