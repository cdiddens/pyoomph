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


# Distributed correctness of EnforcedInterfacialLaplaceSmoothing, i.e. of the reference arclength it
# measures along an interface at setup. Why that is a distributed problem at all, and what it looked
# like when it was wrong, is in the header of tests/mpi_arclength_smoothing_worker.py.
#
# What is compared is the arclength AT each node, addressed by position, against a serial reference --
# not the arclength profile of a rank's own piece. A rank holds a stretch of the curve, so its values
# are a contiguous subset of the serial ones; the defect this pins down is precisely that the stretch
# used to start at zero instead of where it really begins.
#
# Halo nodes are included on purpose. _s_fixed_ is a PINNED value, so oomph never synchronises it
# between a halo node and its owner, and a halo copy carrying a different reference arclength makes the
# halo element assemble a different equation from the element it stands in for.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_arclength_smoothing_worker.py")

#: Node positions of a distributed run differ from the serial ones only by the linear solve, so they
#: agree far better than this; it is the tolerance of the position match that finds a node in the
#: reference, not of the comparison itself.
_POSITION_ATOL = 1e-9
_ARCLENGTH_ATOL = 1e-9


def _mpi_reason():
    if shutil.which("mpirun") is None:
        return "mpirun not found"
    try:
        from pyoomph.generic.mpi import has_mpi
        if not has_mpi():
            return "pyoomph was built without MPI"
    except Exception as e:  # noqa: BLE001
        return "MPI unavailable: " + str(e)
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]


def _run(tmpdir, nproc=None, distribute=False, timeout=600):
    cmd = ["mpirun", "-n", str(nproc)] if nproc is not None else []
    cmd += [sys.executable, _WORKER, "--outdir", str(tmpdir)]
    if distribute:
        cmd += ["--distribute"]
    env = dict(os.environ)
    if nproc is not None:
        # Importing pyoomph calls MPI_Init, so this pytest process already owns an Open MPI session
        # directory under TMPDIR; a nested mpirun collides with it and dies with no diagnostics.
        ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
        os.makedirs(ompi_tmp, exist_ok=True)
        env["TMPDIR"] = ompi_tmp
    proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    per_rank = [json.loads(line[len("PYOOMPH_MPI_RESULT "):])
                for line in proc.stdout.splitlines() if line.startswith("PYOOMPH_MPI_RESULT ")]
    assert per_rank, "no result from %s (exit %d)\n--- stdout ---\n%s\n--- stderr ---\n%s" % (
        " ".join(cmd), proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:])
    return per_rank


def _lookup(reference, x, y):
    """The serial reference node at (x, y), or None if there is none within the tolerance."""
    best, bestd = None, None
    for rx, ry, ral in reference:
        d = abs(rx - x) + abs(ry - y)
        if bestd is None or d < bestd:
            best, bestd = (rx, ry, ral), d
    return best if bestd is not None and bestd <= _POSITION_ATOL else None


@pytest.mark.parametrize("nproc,distribute", [(2, False), (2, True), (3, True)])
def test_reference_arclength_is_the_one_of_the_whole_interface(tmp_path, nproc, distribute):
    reference = _run(tmp_path / "serial")[0]
    serial = reference["nodes"]
    total = max(n[2] for n in serial)
    assert total > 1.0, "the reference interface should be longer than the unit square's top edge"

    per_rank = _run(tmp_path / "mpi", nproc=nproc, distribute=distribute)
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    assert per_rank[0]["distributed"] is distribute
    assert per_rank[0]["ndof"] == reference["ndof"], "the distributed run solves a different problem"

    seen = 0
    for r in per_rank:
        for x, y, al in r["nodes"]:
            ref = _lookup(serial, x, y)
            assert ref is not None, \
                "rank %d has an interface node at (%r, %r) that the serial run does not" % (r["rank"], x, y)
            assert abs(al - ref[2]) <= _ARCLENGTH_ATOL, \
                ("rank %d measures the reference arclength at (%r, %r) as %r, serially it is %r. A rank "
                 "that starts counting at zero on its own piece of the interface reports exactly this."
                 % (r["rank"], x, y, al, ref[2]))
            seen += 1
    assert seen >= len(serial), \
        "the ranks together hold %d interface nodes, the serial run has %d" % (seen, len(serial))
