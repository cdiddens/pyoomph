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

# Nodal boundary membership under --distribute. See dev_docs/boundary_node_membership.md.
#
# The post-adapt repair removes memberships that no tagged face backs. Under MPI a rank can hold a
# node without holding the element whose tagged face carries it, so it deliberately decides only for
# nodes that lie in at least one of its own (non-halo) elements -- for those, the halo layer
# guarantees every incident element is present -- and the owners push their decisions onto the halo
# copies afterwards.
#
# That push is the only part of the machinery serial runs cannot exercise, and getting it wrong is
# silent: nothing else in pyoomph or oomph-lib ever compares boundary membership between ranks (the
# halo consistency check compares geometry, refinement level, flags and error, and nothing else). So
# what is checked here is agreement, not just correctness.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_boundary_membership_worker.py")
_KINDS = ["hex", "tet"]
_BOUNDARIES = ["wall", "side"]


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


def _run(kinds, nproc, tmpdir, level=2, timeout=1800):
    cmd = ["mpirun", "-n", str(nproc)]
    if os.environ.get("PYOOMPH_MPI_OVERSUBSCRIBE", "1") == "1":
        cmd += ["--oversubscribe"]
    cmd += [sys.executable, _WORKER, "--kinds", ",".join(kinds), "--level", str(level),
            "--outdir", str(tmpdir), "--distribute"]
    # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
    # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # The most likely way this machinery fails is a deadlock: the cross-rank push is a collective
        # placed just outside adapt_mesh's "if (nelement()>0)" block, and moving it inside would hang
        # exactly one rank. Say so, rather than reporting a bare timeout.
        raise AssertionError("mpirun did not finish within %d s -- suspect a distributed deadlock in "
                             "the boundary-membership push.\n--- stdout tail ---\n%s"
                             % (timeout, (e.stdout or "")[-3000:]))
    results = {}
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            # The line carries a path, not the payload: these results are tens of kB of node positions
            # and mpirun truncates a long line at 4096 characters.
            with open(line[len("PYOOMPH_MPI_RESULT "):].strip()) as f:
                payload = json.load(f)
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


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_boundary_membership_matches_the_facets(nproc, tmp_path):
    # The slab of tests/slab_mesh.py has every element with two faces on "wall", so its vertical edges
    # have both ends on "wall" without lying on it. In serial the same mesh produces 21 (hex) and 31
    # (tet) spuriously marked nodes at one refinement, 135 and 303 at two, and none once repaired.
    results = _run(_KINDS, nproc, tmp_path)
    for kind in _KINDS:
        per_rank = results[kind]
        assert sum(r["nelem"] for r in per_rank) > 0, kind

        for r in per_rank:
            # Only judges nodes this rank can decide, so it is a valid per-rank assertion even for a
            # rank whose halo layer reaches nodes it knows nothing about.
            assert tuple(r["selfcheck"]) == (0, 0), "%s rank %d: %s" % (kind, r["rank"], r["selfcheck"])

        # The global picture. Per rank neither list is conclusive -- a halo-only node can sit on a
        # facet owned elsewhere -- but their unions across all ranks are, and they must be equal.
        for bname in _BOUNDARIES:
            marked = set().union(*(set(r["marked"][bname]) for r in per_rank))
            on_facets = set().union(*(set(r["on_facets"][bname]) for r in per_rank))
            assert marked, "%s/%s: no rank holds any node of this boundary" % (kind, bname)
            assert marked - on_facets == set(), "%s/%s: %d nodes marked on no facet anywhere" % (
                kind, bname, len(marked - on_facets))
            assert on_facets - marked == set(), "%s/%s: the repair dropped %d genuine memberships" % (
                kind, bname, len(on_facets - marked))


@pytest.mark.parametrize("nproc", [2, 4])
def test_ranks_agree_about_every_shared_node(nproc, tmp_path):
    # The property the cross-rank push exists for. A rank that cannot decide a node leaves it marked
    # and receives the owner's verdict; if that message never arrived, or arrived against the wrong
    # element list, the two copies of the same node would end up on different boundaries -- and nothing
    # else in the code would ever notice.
    results = _run(_KINDS, nproc, tmp_path)
    for kind in _KINDS:
        per_rank = results[kind]
        seen = {}   # position -> (rank, frozenset of boundary names it is marked on there)
        for r in per_rank:
            here = {}
            for bname in _BOUNDARIES:
                for pos in r["marked"][bname]:
                    here.setdefault(pos, set()).add(bname)
            # Every node the rank holds, including those on no boundary at all -- otherwise a node
            # wrongly stripped on one rank would just look absent rather than inconsistent.
            for pos in set().union(*(set(r["on_facets"][b]) for b in _BOUNDARIES)):
                here.setdefault(pos, set())
            for pos, bnames in here.items():
                if pos in seen:
                    other_rank, other = seen[pos]
                    assert other == frozenset(bnames), (
                        "%s: node at %s is on %s on rank %d but on %s on rank %d" % (
                            kind, pos, sorted(other), other_rank, sorted(bnames), r["rank"]))
                else:
                    seen[pos] = (r["rank"], frozenset(bnames))
        assert seen, kind + ": nothing to compare"
