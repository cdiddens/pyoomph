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

# Periodic orbit tracking and Floquet multipliers on a DISTRIBUTED problem.
#
# PeriodicOrbitHandler used to be entirely replicated -- Tadd, x0, n0, du0ds and Count were
# global-Ndof std::vectors indexed by global equation number -- and --distribute was refused on the
# Python side. It now carries an AugmentedDofDistributionHelper like the four bifurcation-tracking
# handlers: the time-point unknowns and the orbit's reference data are DoubleVectorWithHaloEntries on
# the base distribution, eqn_number() runs its naive number through the helper's translation table,
# and a synchronise() override refreshes the halos and broadcasts the period after each Newton step.
#
# The test problem is Stuart-Landau kinetics applied pointwise on a 1D mesh plus diffusion. Its
# spatially UNIFORM state is exactly u=cos(t), v=sin(t) with T=2*pi -- diffusion annihilates a
# uniform field -- so the guess handed to the handler is the answer, on a mesh with enough elements
# to distribute. That also means no Hopf tracker is needed to reach the orbit, which matters:
# switch_to_hopf_orbit() needs the first Lyapunov coefficient from the Python custom assembler in
# bifurcation_tools.py, and THAT is still serial (dev_docs/mpi_augmented_systems.md Part II).
#
# WHAT EACH ASSERTION IS FOR:
#
#   - the period against 2*pi, and against the serial run. The exact value is what says the
#     distributed assembly is right at all; the serial comparison is what says nothing about the
#     answer depends on how the mesh was cut up. It comes out bit-identical.
#   - the Floquet multipliers against serial. These go through the extra step the orbit solve does
#     not: under --distribute the augmented rows are interleaved per rank, so a gathered orbit
#     Jacobian is NOT in the time-major order the condensation slices along, and it has to be
#     permuted back through PeriodicOrbitHandler::get_naive_equation_order(). Without that the
#     condensation cuts the wrong blocks -- which the structure check in pyoomph/generic/floquet.py
#     catches rather than answering wrongly, so a regression here fails loudly.
#   - the sampled orbit states (sample_absum). These exercise set_dofs_to_interpolated_values(), the
#     one part of the handler that writes base dofs wholesale rather than through an element: it
#     cannot use global_value() (a row this rank neither owns nor halos has no entry to reach), so it
#     writes owned rows and then pushes to the halos. A missing push shows up here as a difference
#     against serial.
#   - that the dof counts really are the distributed ones, i.e. the run under test was distributed at
#     all rather than silently falling back.

import json
import os
import shutil
import subprocess
import sys

import numpy
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "floquet_worker.py")

_N = 40
_NT = 24
_NPROC = 4


def _mpi_reason():
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


def _run(tmpdir, nproc, distribute, timeout=1800):
    os.makedirs(str(tmpdir), exist_ok=True)
    cmd = [sys.executable, _WORKER, "--outdir", str(tmpdir), "--case", "pde",
           "--N", str(_N), "--NT", str(_NT)]
    if nproc > 1:
        cmd = ["mpirun", "-n", str(nproc)] + cmd
    if distribute:
        cmd.append("--distribute")
    env = dict(os.environ)
    # Own TMPDIR for the nested mpirun; see tests/test_mpi_bifurcation_tracking.py for why.
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    out = subprocess.run(cmd, cwd=str(tmpdir), capture_output=True, text=True,
                         timeout=timeout, env=env)
    lines = [l for l in out.stdout.splitlines() if l.startswith("PYOOMPH_FLOQUET_RESULT ")]
    assert len(lines) == 1, out.stdout[-4000:] + out.stderr[-4000:]
    res = json.loads(lines[0][len("PYOOMPH_FLOQUET_RESULT "):])
    assert "error" not in res, res.get("traceback", res.get("error"))
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    return res


def _multipliers(res):
    """The multipliers in a conjugation-invariant canonical order.

    Asking for the n dominant multipliers can cut a complex conjugate pair in half, and which half
    survives depends on the order numpy.linalg.eig returns them in -- which the partitioning changes.
    Both halves say the same thing, so each is folded onto the positive-imaginary representative
    before sorting.
    """
    F = numpy.array(res["mult_re"]) + 1j * numpy.array(res["mult_im"])
    F = numpy.where(F.imag >= 0, F, numpy.conjugate(F))
    return numpy.array(sorted(F, key=lambda z: (abs(z), z.real, z.imag)))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("mpi_floquet")
    serial = _run(base / "serial", 1, False)
    dist = _run(base / "dist", _NPROC, True)
    return serial, dist


def test_the_distributed_run_really_was_distributed(runs):
    serial, dist = runs
    assert serial["distributed"] is False and dist["distributed"] is True, (serial, dist)
    assert dist["nproc"] == _NPROC, dist
    # nbase/nT/ndof are global counts, so they must agree with the serial run exactly.
    for key in ("nbase", "nT", "ndof"):
        assert serial[key] == dist[key], (key, serial[key], dist[key])


def test_period_matches_serial_and_the_exact_value(runs):
    serial, dist = runs
    # The uniform limit cycle has period exactly 2*pi; what is left is the time-discretization error
    # of order-3 collocation at NT=24, which is ~7e-6 here and identical in both runs.
    assert abs(serial["T"] - 2 * numpy.pi) < 1e-4, serial["T"]
    assert dist["T"] == serial["T"], (dist["T"], serial["T"])


def test_floquet_multipliers_match_serial(runs):
    serial, dist = runs
    a, b = _multipliers(serial), _multipliers(dist)
    assert len(a) == len(b) == 6, (a, b)
    # The remaining difference is the assembly's summation order, which the partitioning changes.
    assert numpy.max(numpy.abs(a - b)) < 1e-10, (a, b)
    assert numpy.min(numpy.abs(a - 1.0)) < 1e-6, a  # the trivial multiplier is still there


def test_orbit_sampling_matches_serial(runs):
    serial, dist = runs
    a = numpy.array(serial["sample_absum"])
    b = numpy.array(dist["sample_absum"])
    assert len(a) == len(b) == 8, (a, b)
    assert numpy.allclose(a, b, rtol=1e-12, atol=1e-9), (a, b)
