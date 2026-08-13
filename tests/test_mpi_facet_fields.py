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

# The MPI gate for the interior-facet skeleton.
#
# tests/test_internal_facet_fields.py covers the skeleton serially. Distributed, one thing is new and it
# is not covered anywhere else: the two elements sharing an interior facet may live on different ranks,
# and the choice of which of them ENUMERATES the facet (the "near" side, whose outward normal the facet
# terms use) is made independently on each rank. Nothing communicates it. It happens to agree today only
# because oomph::Mesh::distribute re-adds the retained elements in their original order, so a rule of the
# form "the first of the two elements seen wins" picks the same one everywhere.
#
# That is an invariant of vendored code, relied upon by pyoomph, and until this file existed nothing
# noticed if it broke. A broken one does not crash: the flux across the affected facets is doubled or
# dropped, Newton converges as usual, and the answer is the solution of a different equation.
#
# The assertions are therefore two-layered:
#
#   * `obs_meas` and `n_facet_elements` certify the ENUMERATION alone, without reference to the
#     solution -- a duplicated facet inflates both, a dropped one deflates them, in units of whole
#     facets rather than round-off;
#   * with a LINEAR manufactured solution the interior-penalty DG answer is exact, because the exact
#     solution lies in the "D1"/"D2" space and the scheme is consistent. It stays exact only if every
#     facet contributes exactly once: the flux term -avg(grad u).jump(v)n does not vanish at the exact
#     solution, so a mis-enumerated facet moves the answer off it.
#
# pytest runs serially, so each test launches tests/mpi_facet_fields_worker.py under mpirun; the serial
# reference is computed in-process from that same module, so a globally consistent but wrong field
# cannot pass by agreeing with itself.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_facet_fields_worker.py")


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
    try:
        from petsc4py import PETSc  # type:ignore
        if not PETSc.Sys.hasExternalPackage("mumps"):
            return "PETSc has no MUMPS support (no distributed-capable direct solver)"
    except Exception:
        return "petsc4py not available (no distributed-capable direct solver)"
    return None


_SKIP_REASON = _mpi_reason()
pytestmark = [pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON)),
              pytest.mark.slow]

_OBS_RTOL = 1e-9


def _run(nproc, tmpdir, args, timeout=600, distribute=True):
    """Launch the worker under mpirun and return the list of per-rank result dicts."""
    cmd = ["mpirun", "-n", str(nproc), sys.executable, _WORKER, "--outdir", str(tmpdir)] + list(args)
    if distribute:
        cmd += ["--distribute"]
    # Importing pyoomph calls MPI_Init, so THIS pytest process is already a (singleton) MPI job and owns
    # an Open MPI session directory under TMPDIR. A nested mpirun collides with it and dies immediately
    # with exit code 1 and no diagnostics at all. Give the child its own TMPDIR.
    env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmpdir), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    env["TMPDIR"] = ompi_tmp
    try:
        proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        # Bounded on purpose: a refusal or a facet decision taken on one rank alone leaves the others in
        # a collective for ever, and that has to surface as a FAILURE rather than as a hung suite.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a one-sided refusal or facet decision "
            "(args=%s, nproc=%d).\n--- stdout tail ---\n%s" % (timeout, args, nproc, (e.stdout or "")[-3000:]))
    per_rank = []
    for line in proc.stdout.splitlines():
        if line.startswith("PYOOMPH_MPI_RESULT "):
            per_rank.append(json.loads(line[len("PYOOMPH_MPI_RESULT "):]))
    if not per_rank:
        raise AssertionError(
            "no results from mpirun (exit %d)\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s"
            % (proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:]))
    assert len(per_rank) == nproc, "reported from %d of %d ranks" % (len(per_rank), nproc)
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))
        # Everything below compares against a serial reference, so a run in which --distribute quietly
        # did nothing would agree perfectly and pass without ever entering the code under test.
        assert r["distributed"] == distribute, \
            "rank %d reports distributed=%s for a run launched with distribute=%s" % (
                r["rank"], r["distributed"], distribute)
        if distribute:
            assert r["n_bulk_nonhalo"] < r["n_bulk_local"], \
                "rank %d holds no halo elements at all (%d of %d) -- it has the whole mesh, so no " \
                "facet is shared across ranks and the halo scheme is untested" % (
                    r["rank"], r["n_bulk_nonhalo"], r["n_bulk_local"])
    return per_rank


def _serial(**kwargs):
    """The serial reference, computed in-process from the same worker module."""
    sys.path.insert(0, _HERE)
    try:
        import mpi_facet_fields_worker
        return mpi_facet_fields_worker.solve_case(**kwargs)
    finally:
        sys.path.remove(_HERE)


# `n_interior_facets_local` is this rank's slice of the mesh (halo layer included) and has no serial
# counterpart; everything else reported is a global quantity, so a partition-dependent value is a defect.
_PER_RANK_KEYS = ("rank", "nproc", "n_interior_facets_local", "n_bulk_local", "n_bulk_nonhalo")


def _assert_ranks_agree(per_rank):
    ref = per_rank[0]
    for r in per_rank[1:]:
        for k in ref:
            if k in _PER_RANK_KEYS:
                continue
            same = (r[k] == pytest.approx(ref[k], rel=_OBS_RTOL, abs=1e-12)
                    if isinstance(ref[k], float) else r[k] == ref[k])
            assert same, "rank %d disagrees with rank 0 about %s: %r vs %r" % (r["rank"], k, r[k], ref[k])


def _assert_same_state(what, ref, got):
    assert got["ndof"] == ref["ndof"], "%s: ndof %d vs %d" % (what, ref["ndof"], got["ndof"])
    # The enumeration, before anything about the solution is looked at. A near-side disagreement between
    # two ranks changes these by whole facets.
    assert got["n_facet_elements"] == ref["n_facet_elements"], \
        "%s: %d facet elements assembled vs %d serially -- a facet is enumerated twice or not at all" % (
            what, got["n_facet_elements"], ref["n_facet_elements"])
    assert got["obs_meas"] == pytest.approx(ref["obs_meas"], rel=1e-12), \
        "%s: skeleton measure %.17g vs %.17g" % (what, got["obs_meas"], ref["obs_meas"])
    for k in sorted(k for k in ref if k.startswith("obs_")):
        assert got[k] == pytest.approx(ref[k], rel=_OBS_RTOL, abs=1e-12), \
            "%s: %s = %.17g vs %.17g" % (what, k, ref[k], got[k])


# ---------------------------------------------------------------------------------------------------
# 1 -- distributed DG solves the same problem as serial DG
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("tris", [0, 1])
def test_distributed_dg_linear_solution_stays_exact(tmp_path, nproc, tris):
    """The sharpest available statement: the discrete answer is the exact linear one, distributed.

    It is sharp because consistency is not enough on its own -- the flux term survives at the exact
    solution, so it holds only if each facet is assembled exactly once, from the side whose outward
    normal the terms were written for."""
    args = ["--size", "8", "--tris", str(tris), "--exact", "linear"]
    per_rank = _run(nproc, tmp_path, args)
    for r in per_rank:
        assert abs(r["obs_uerr2"]) < 1e-14, \
            "rank %d: |u - u_exact|^2 = %.3e, so the distributed DG scheme is not the serial one" % (
                r["rank"], r["obs_uerr2"])
        assert abs(r["obs_jump2"]) < 1e-12, "rank %d: the solution is not continuous" % r["rank"]
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed DG (n=%d, tris=%d) vs serial" % (nproc, tris),
                       _serial(N=8, tris=bool(tris), exact="linear", outdir=str(tmp_path / "serial")),
                       per_rank[0])


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_dg_matches_serial_for_a_non_representable_solution(tmp_path, nproc):
    """sin(pi x) sin(pi y): the discrete answer is not the exact one, so this compares the two runs'
    DISCRETISATIONS rather than their distance from a solution both happen to reproduce."""
    per_rank = _run(nproc, tmp_path, ["--size", "8", "--exact", "sin"])
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed DG sin (n=%d) vs serial" % nproc,
                       _serial(N=8, exact="sin", outdir=str(tmp_path / "serial")), per_rank[0])


def test_distributed_dg_matches_serial_for_a_higher_order_space(tmp_path):
    """"D2": more than one dof per facet side, so a mis-paired facet shows up as a wrong coupling rather
    than only as a wrong count."""
    per_rank = _run(4, tmp_path, ["--size", "6", "--space", "D2", "--exact", "sin"])
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed DG D2 vs serial",
                       _serial(N=6, space="D2", exact="sin", outdir=str(tmp_path / "serial")),
                       per_rank[0])


# ---------------------------------------------------------------------------------------------------
# 2 -- UNKNOWNS on the skeleton under --distribute
# ---------------------------------------------------------------------------------------------------
#
# The facet's dofs live in the facet element's own internal Data, and a facet shared by two processes
# exists on both, so without an ownership rule oomph-lib numbers them once per holder: two independent
# copies of what is meant to be one single-valued trace. `ndof` is the direct evidence -- it comes out
# inflated by exactly the number of shared facets times the dofs per facet, and the answer can still
# look plausible, because a projection that never feeds back into the bulk is happy with two copies.
#
# Problem.setup_interior_facet_halo_scheme() gives the skeleton its halo/haloed element lists and marks
# the halo side's Data as halo; oomph-lib's own machinery then copies the equation numbers and the
# values across. What these tests pin down is that the pairing is right: a mis-paired list does not
# crash, it silently puts one facet's equation numbers on another facet.

@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("facet_space", ["DL", "D0"])
def test_distributed_facet_unknown_is_numbered_once(tmp_path, nproc, facet_space):
    args = ["--size", "8", "--facet-space", facet_space, "--exact", "linear"]
    per_rank = _run(nproc, tmp_path, args)
    ref = _serial(N=8, exact="linear", facet_space=facet_space, outdir=str(tmp_path / "serial"))
    for r in per_rank:
        # The sharp one. Numbered per holder instead of per facet, this is larger than serial by whole
        # facets' worth of dofs, whatever the solution then does.
        assert r["ndof"] == ref["ndof"], \
            "rank %d: ndof %d vs %d serially -- the facet unknowns are numbered more than once" % (
                r["rank"], r["ndof"], ref["ndof"])
        # A linear bulk trace is exactly representable in DL, so its projection error is round-off.
        # (D0 cannot follow a linear trace, but its Galerkin projection still matches the facet MEAN,
        # so the SIGNED error vanishes there too.)
        assert abs(r["obs_perr1"]) < 1e-11, "rank %d: signed projection error %.3e" % (r["rank"], r["obs_perr1"])
        if facet_space == "DL":
            assert abs(r["obs_perr2"]) < 1e-14, "rank %d: projection error %.3e" % (r["rank"], r["obs_perr2"])
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed facet unknown (%s, n=%d) vs serial" % (facet_space, nproc),
                       ref, per_rank[0])


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_facet_unknown_survives_adaptation(tmp_path, nproc):
    """The skeleton is not adapted incrementally: it is deleted and rebuilt from the refined bulk mesh,
    so the halo scheme has to be rebuilt with it and the facet values carried across the rebuild.

    Refining also puts the 2:1 branch of the facet enumeration to work, where a coarse facet faces
    several fine ones -- which is where a partition-dependent near-side choice would show up first."""
    args = ["--size", "6", "--facet-space", "DL", "--exact", "sin", "--adapt", "2"]
    per_rank = _run(nproc, tmp_path, args, timeout=900)
    ref = _serial(N=6, exact="sin", facet_space="DL", adapt=2, outdir=str(tmp_path / "serial"))
    assert ref["n_facet_elements"] > 100, \
        "the reference run did not actually refine (%d facets) -- this test would prove nothing" % ref["n_facet_elements"]
    for r in per_rank:
        assert abs(r["obs_perr2"]) < 1e-14, \
            "rank %d: the trace was not carried across the adaptation (error %.3e)" % (r["rank"], r["obs_perr2"])
    _assert_ranks_agree(per_rank)
    _assert_same_state("adapted distributed facet unknown (n=%d) vs serial" % nproc, ref, per_rank[0])


@pytest.mark.parametrize("nproc", [2, 4])
def test_state_file_saved_serially_loads_distributed(tmp_path, nproc):
    """State files address elements by a partition-independent key, so a file written serially has to
    come back on any number of processes -- refined mesh, skeleton and facet values included. The halo
    scheme is rebuilt for whatever mesh the file describes, which is not the one the run started with."""
    ref = _serial(N=6, exact="sin", facet_space="DL", adapt=2, outdir=str(tmp_path), state="save")
    per_rank = _run(nproc, tmp_path, ["--size", "6", "--facet-space", "DL", "--exact", "sin",
                                      "--adapt", "2", "--state", "load"], timeout=900)
    _assert_ranks_agree(per_rank)
    _assert_same_state("state file loaded on %d ranks vs saved serially" % nproc, ref, per_rank[0])


def test_nodal_dg_facet_space_is_refused_before_distributing(tmp_path):
    """"D1"/"D2" on the skeleton cannot be carried through a mesh rebuild, and distributing performs
    one. That has to be said up front, by every rank, rather than discovered deep inside the
    distribution -- and it must name the way out."""
    # The bulk space has to carry a D2 trace, or a different (earlier, and unrelated) check fires.
    proc_args = ["--size", "6", "--space", "D2", "--facet-space", "D2", "--exact", "linear"]
    cmd_env = dict(os.environ)
    ompi_tmp = os.path.join(str(tmp_path), "_ompi_session")
    os.makedirs(ompi_tmp, exist_ok=True)
    cmd_env["TMPDIR"] = ompi_tmp
    cmd = ["mpirun", "-n", "2", sys.executable, _WORKER, "--outdir", str(tmp_path)] + proc_args + ["--distribute"]
    proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True, timeout=420, env=cmd_env)
    per_rank = [json.loads(l[len("PYOOMPH_MPI_RESULT "):]) for l in proc.stdout.splitlines()
                if l.startswith("PYOOMPH_MPI_RESULT ")]
    assert len(per_rank) == 2, "reported from %d of 2 ranks" % len(per_rank)
    for r in per_rank:
        assert "error" in r, "rank %d accepted a nodal discontinuous facet space" % r["rank"]
        assert "nodal discontinuous space" in r["error"], r["error"]
        assert "'DL' or 'D0'" in r["error"], "the refusal does not say what to use instead: %s" % r["error"]


# ---------------------------------------------------------------------------------------------------
# 3 -- the replicated mode, where the mesh is whole on every rank and only the element loop is split
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nproc", [2, 4])
def test_replicated_dg_matches_serial(tmp_path, nproc):
    """`mpirun` without `--distribute`. Every rank holds the whole mesh, so the skeleton is enumerated
    identically everywhere and only the assembly is split -- which means a facet element assigned to no
    rank, or to two, is a defect of the element RANGE rather than of the facet logic."""
    per_rank = _run(nproc, tmp_path, ["--size", "8", "--exact", "sin"], distribute=False)
    _assert_ranks_agree(per_rank)
    _assert_same_state("replicated DG (n=%d) vs serial" % nproc,
                       _serial(N=8, exact="sin", outdir=str(tmp_path / "serial")), per_rank[0])


# ---------------------------------------------------------------------------------------------------
# 4 -- the other two dimensions
# ---------------------------------------------------------------------------------------------------
#
# Everything above is 2d. There are three independent facet enumerators, one per dimension
# (`fill_internal_facet_buffers` in src/mesh1d.cpp, src/mesh2d.cpp, src/mesh3d.cpp), and each makes its
# own near-side choice; only the 2d one had ever been run under --distribute.
#
# 1d is the sharp one, for two reasons. Its rule is the INVERTED one -- it keeps the LATER of the two
# elements as the near side, where 2d and 3d keep the structurally smaller -- so a "fix" that unified
# the three would flip every 1d facet normal and with it the sign of every jump(). And a 1d facet is a
# point, so `obs_meas` is the plain count of interior facets and `obs_nx` is a plain signed count of
# which way they face: both are integers, and a single mis-enumerated facet moves them by a whole unit
# rather than by something one has to choose a tolerance for.
#
# 3d only admits conforming meshes (`TemplatedMeshBase3d::fill_internal_facet_buffers` refuses the rest),
# so these cases stay uniform: no adaptation, and the check is that a partitioned brick mesh enumerates
# the same 144 facets that a serial one does.

_N1D = 32          # 31 interior facets over up to 4 ranks
_N3D = 4           # 64 bricks, 144 interior facets -- large enough that no rank holds the whole mesh


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_1d_dg_linear_solution_stays_exact(tmp_path, nproc):
    """1d DG distributed, with the exact solution inside the discrete space.

    `obs_nx == -(N-1)` is the direct statement of the 1d near-side convention: every facet normal points
    in -x, unanimously, on every rank. One facet taken from the other side would read -(N-3)."""
    per_rank = _run(nproc, tmp_path, ["--dim", "1", "--size", str(_N1D), "--exact", "linear"])
    ref = _serial(dim=1, N=_N1D, exact="linear", outdir=str(tmp_path / "serial"))
    for r in per_rank:
        assert abs(r["obs_uerr2"]) < 1e-14, \
            "rank %d: |u - u_exact|^2 = %.3e in 1d" % (r["rank"], r["obs_uerr2"])
        assert r["obs_meas"] == pytest.approx(_N1D - 1), \
            "rank %d counts %.17g interior facets, not %d" % (r["rank"], r["obs_meas"], _N1D - 1)
        assert r["obs_nx"] == pytest.approx(-(_N1D - 1)), \
            "rank %d: sum of facet normals %.17g, not %d -- the 1d near-side rule is not unanimous" % (
                r["rank"], r["obs_nx"], -(_N1D - 1))
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed 1d DG (n=%d) vs serial" % nproc, ref, per_rank[0])


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_1d_dg_matches_serial_for_a_non_representable_solution(tmp_path, nproc):
    """sin(pi x) in 1d: compares the two runs' discretisations rather than their distance from a
    solution both reproduce exactly."""
    per_rank = _run(nproc, tmp_path, ["--dim", "1", "--size", str(_N1D), "--exact", "sin"])
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed 1d DG sin (n=%d) vs serial" % nproc,
                       _serial(dim=1, N=_N1D, exact="sin", outdir=str(tmp_path / "serial")),
                       per_rank[0])


@pytest.mark.parametrize("nproc", [2, 4])
@pytest.mark.parametrize("facet_space", ["DL", "D0"])
def test_distributed_1d_facet_unknown_is_numbered_once(tmp_path, nproc, facet_space):
    """A facet unknown on a point. `ndof` is again the sharp assertion: numbered per holder rather than
    per facet it comes out larger by exactly the number of facets straddling a partition boundary."""
    per_rank = _run(nproc, tmp_path, ["--dim", "1", "--size", str(_N1D), "--exact", "linear",
                                      "--facet-space", facet_space])
    ref = _serial(dim=1, N=_N1D, exact="linear", facet_space=facet_space,
                  outdir=str(tmp_path / "serial"))
    assert ref["ndof"] == 2 * _N1D + (_N1D - 1), \
        "the serial reference does not have one facet dof per interior facet (%d)" % ref["ndof"]
    for r in per_rank:
        assert r["ndof"] == ref["ndof"], \
            "rank %d: ndof %d vs %d serially -- the 1d facet unknowns are numbered more than once" % (
                r["rank"], r["ndof"], ref["ndof"])
        # On a point, "DL" and "D0" are the same one-value space, so both reproduce the trace exactly.
        assert abs(r["obs_perr2"]) < 1e-14, "rank %d: trace error %.3e" % (r["rank"], r["obs_perr2"])
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed 1d facet unknown (%s, n=%d) vs serial" % (facet_space, nproc),
                       ref, per_rank[0])


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_3d_dg_linear_solution_stays_exact(tmp_path, nproc):
    """3d bricks distributed. The orientation sums are integers here too: with all N-1 interior planes
    per direction facing the same way, each of nx, ny, nz is exactly N-1."""
    per_rank = _run(nproc, tmp_path, ["--dim", "3", "--size", str(_N3D), "--exact", "linear"],
                    timeout=900)
    ref = _serial(dim=3, N=_N3D, exact="linear", outdir=str(tmp_path / "serial"))
    assert ref["n_facet_elements"] == 3 * (_N3D - 1) * _N3D ** 2, \
        "the serial reference does not see every interior brick facet (%d)" % ref["n_facet_elements"]
    for r in per_rank:
        assert abs(r["obs_uerr2"]) < 1e-13, \
            "rank %d: |u - u_exact|^2 = %.3e in 3d" % (r["rank"], r["obs_uerr2"])
        for comp in "xyz":
            assert r["obs_n" + comp] == pytest.approx(_N3D - 1, rel=1e-12), \
                "rank %d: sum of facet normals along %s is %.17g, not %d" % (
                    r["rank"], comp, r["obs_n" + comp], _N3D - 1)
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed 3d DG (n=%d) vs serial" % nproc, ref, per_rank[0])


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_3d_facet_unknown_is_numbered_once(tmp_path, nproc):
    """A "DL" unknown on a quadrilateral facet of a partitioned brick mesh: three dofs per facet, so a
    mis-paired halo list puts a whole triple on the wrong facet rather than a single value."""
    per_rank = _run(nproc, tmp_path, ["--dim", "3", "--size", str(_N3D), "--exact", "linear",
                                      "--facet-space", "DL"], timeout=900)
    ref = _serial(dim=3, N=_N3D, exact="linear", facet_space="DL", outdir=str(tmp_path / "serial"))
    for r in per_rank:
        assert r["ndof"] == ref["ndof"], \
            "rank %d: ndof %d vs %d serially -- the 3d facet unknowns are numbered more than once" % (
                r["rank"], r["ndof"], ref["ndof"])
        # A linear bulk trace is bilinear on a brick facet, which "DL" (1, x, y) cannot follow exactly,
        # so only the SIGNED error vanishes; obs_perr2 is compared against serial instead.
        assert abs(r["obs_perr1"]) < 1e-11, \
            "rank %d: signed trace error %.3e" % (r["rank"], r["obs_perr1"])
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed 3d facet unknown (n=%d) vs serial" % nproc, ref, per_rank[0])


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_3d_dg_matches_serial_for_a_non_representable_solution(tmp_path, nproc):
    per_rank = _run(nproc, tmp_path, ["--dim", "3", "--size", str(_N3D), "--exact", "sin",
                                      "--facet-space", "DL"], timeout=900)
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed 3d DG sin (n=%d) vs serial" % nproc,
                       _serial(dim=3, N=_N3D, exact="sin", facet_space="DL",
                               outdir=str(tmp_path / "serial")), per_rank[0])
