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

# Curved boundaries under --distribute. The same acceptance criterion as the serial module
# (tests/test_curved_boundaries.py): every node on a curved boundary satisfies the boundary's implicit
# equation to machine precision -- now required of EVERY rank, since each holds its own elements,
# macro elements and halo layer.
#
# This combination had never been run. Distributed adaptivity was covered, curved boundaries were
# covered, and the two together were not -- which is where both defects below were hiding.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_curved_worker.py")
_EXACT = 1e-14


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


def _run(kinds, nproc, tmpdir, level=1, timeout=1200):
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--kinds", ",".join(kinds), "--level", str(level),
            "--outdir", str(tmpdir), "--distribute"]
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics. Give the
    # child its own. (Same reasoning as test_mpi_adaptivity._run_distributed.)
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise AssertionError("mpirun did not finish within %d s -- suspect a distributed deadlock.\n"
                             "--- stdout tail ---\n%s" % (timeout, (e.stdout or b"")[-3000:]))
    results = {}
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            payload = json.loads(line[len("PYOOMPH_MPI_RESULT "):])
            results.setdefault(payload["kind"], []).append(payload)
    if not results:
        raise AssertionError("no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n"
                             "--- stderr tail ---\n%s" % (proc.returncode, proc.stdout[-3000:],
                                                          proc.stderr[-3000:]))
    for kind, per_rank in results.items():
        assert len(per_rank) == nproc, "%s reported from %d of %d ranks" % (kind, len(per_rank), nproc)
        for r in per_rank:
            assert "error" not in r, "%s failed on rank %d: %s\n%s" % (
                kind, r["rank"], r["error"], r.get("traceback", ""))
    return results


_KINDS_2D = ["quad", "tri"]
_KINDS_3D = ["sphere", "tetball", "gmshball"]


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_curved_boundaries_stay_exact(nproc, tmp_path):
    # Before this worked, two things went wrong in sequence, both of them only reachable here.
    #
    # First a segfault: the macro element held raw Node pointers, and Mesh::distribute() deletes the
    # elements and nodes a rank does not own. A macro element belongs to the ROOT element and is shared
    # by every son, so a son surviving on this rank could be left pointing at freed root vertices, and
    # classify_halo_and_haloed_nodes() then called get_x() on it. (The pre-branch implementation held
    # node pointers the same way and would have crashed identically; nothing had ever combined
    # --distribute with a curved boundary.) The macro element now stores vertex positions by value.
    #
    # Then a geometry corruption -- quadtree neighbour finding off by 0.447 -- because
    # map_nodes_on_macro_element only ever set history level t=0, leaving t>=1 on the straight
    # interpolant. oomph's synchronise_nonhanging_nodes compares get_x(t,...) against x(t,...) at every
    # t while distributing, so it saw conforming nodes as needing repair and moved them. Serial runs
    # never compare the two, which is why the "TODO: Time loop" that had been sitting there was
    # invisible until now.
    results = _run(_KINDS_2D + _KINDS_3D, nproc, tmp_path)
    for kind in _KINDS_2D + _KINDS_3D:
        assert kind in results, "no result for " + kind
        # Every rank that holds part of the curved boundary must have it exact. A rank holding none is
        # legitimate -- these meshes are deliberately coarse, and on four ranks the spherical octant's
        # boundary does not reach all of them -- so the "somebody has it" check is global.
        assert sum(r["bnodes"] for r in results[kind]) > 0, kind + ": no rank holds any boundary node"
        for r in results[kind]:
            assert r["worst"] < _EXACT, "%s: rank %d worst |r-R| = %.3e" % (kind, r["rank"], r["worst"])
