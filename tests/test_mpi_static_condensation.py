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

# The MPI gate for static condensation (dev_docs/static_condensation.md section 9).
#
# pytest itself runs serially, so each test launches tests/mpi_condensation_worker.py under
# `mpirun -n N ... --distribute`. What is being protected here is not the same thing as in
# tests/test_static_condensation.py; distributed, three failure modes are new and none of them
# announces itself:
#
#   1. a component's E rows live on OTHER ranks, so if the Schur update (or the operator exchange that
#      carries X and y to them) is wrong, the neighbouring rows get a wrong matrix. Newton still
#      converges -- to the right answer, from a wrong Jacobian, in more steps -- so the guard is the
#      NEWTON STEP COUNT and the converged state together, against a serial reference;
#   2. only the owner reconstructs the eliminated dofs, so a missing halo synchronisation afterwards
#      leaves stale copies that a neighbour's elements then read. That moves the answer, not the
#      residual, so the eliminated ELEMENT-INTERNAL values are checksummed explicitly -- integral
#      observables of the retained field barely notice;
#   3. a refusal decided from a rank's own block is one-sided by construction, and half the ranks
#      throwing while the others enter the next collective is a HANG, not a failure. Every refusal
#      test therefore runs under a bounded timeout and asserts that all ranks report the same message.
#
# The serial reference is computed in-process from the same worker module, so a globally consistent but
# wrong field cannot pass by agreeing with itself.

import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "mpi_condensation_worker.py")


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

# The two arms solve the identical discrete system: condensation is an exact algebraic elimination, so
# the only thing separating them is the arithmetic of two different-but-equivalent matrices.
_OBS_RTOL = 1e-9
_RES_TOL = 1e-7


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
        # Bounded on purpose, and the assertion message says what a timeout means here: a refusal or a
        # plan decision taken on one rank alone leaves the others in a collective for ever, and that has
        # to surface as a FAILURE rather than as a suite that never returns.
        raise AssertionError(
            "mpirun did not finish within %d s -- suspect a one-sided refusal or plan decision "
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
    return per_rank


def _ok(per_rank):
    for r in per_rank:
        assert "error" not in r, "failed on rank %d: %s\n%s" % (
            r["rank"], r["error"], r.get("traceback", ""))
    return per_rank


def _serial(**kwargs):
    """The uncondensed serial reference, computed in-process from the same worker module."""
    sys.path.insert(0, _HERE)
    try:
        import mpi_condensation_worker
        return mpi_condensation_worker.solve_case(**kwargs)
    finally:
        sys.path.remove(_HERE)


def _serial_hdg(**kwargs):
    sys.path.insert(0, _HERE)
    try:
        import mpi_condensation_worker
        return mpi_condensation_worker.hdg_case(**kwargs)
    finally:
        sys.path.remove(_HERE)


def _assert_ranks_agree(per_rank):
    """Everything reported is a global quantity, so a partition-dependent answer is a defect."""
    ref = per_rank[0]
    _per_rank_keys = ("rank", "n_selected", "n_components_owned", "n_components_remote",
                      "n_foreign_E", "n_operator_sends", "n_operator_recvs")
    for r in per_rank[1:]:
        for k in ref:
            if k in _per_rank_keys:
                continue  # per-rank by construction: this rank's share of the mesh and of the work
            assert r[k] == pytest.approx(ref[k], rel=_OBS_RTOL) if isinstance(ref[k], float) else r[k] == ref[k], \
                "rank %d disagrees with rank 0 about %s: %r vs %r" % (r["rank"], k, r[k], ref[k])


def _assert_same_state(what, a, b):
    """Same discrete answer, eliminated dofs included."""
    assert a["ndof"] == b["ndof"], "%s: ndof %d vs %d" % (what, a["ndof"], b["ndof"])
    assert a["newton_steps"] == b["newton_steps"], \
        "%s: Newton steps %s vs %s -- the two arms are not solving the same linearised system" % (
            what, a["newton_steps"], b["newton_steps"])
    for k in sorted(k for k in a if k.startswith("obs_")):
        assert b[k] == pytest.approx(a[k], rel=_OBS_RTOL, abs=1e-12), \
            "%s: %s = %.17g vs %.17g" % (what, k, a[k], b[k])
    # The direct evidence that the ELIMINATED dofs were reconstructed, and reconstructed on every rank
    # that needs them: these are sums over the element-internal (DL pressure) values, of which the
    # gradient modes are exactly what condensation removes from the linear system.
    for k in ("internal_sum", "internal_sqsum"):
        scale = max(abs(a[k]), 1.0)
        assert abs(b[k] - a[k]) <= 1e-9 * scale, \
            "%s: %s = %.17g vs %.17g (the eliminated element-internal values differ)" % (what, k, a[k], b[k])


# ---------------------------------------------------------------------------------------------------
# 1 -- the equivalence that matters: a distributed condensed solve is the serial uncondensed one
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_condensed_solve_matches_serial(tmp_path, nproc):
    per_rank = _ok(_run(nproc, tmp_path, ["--size", "8", "--condense", "1"]))
    for r in per_rank:
        # Without this the rest is vacuous: every other assertion is also satisfied by a run in which
        # condensation quietly declined and the full system was solved (dev_docs/structural_assembly.md
        # section 4.1 -- the fast path needs a positive signal that it engaged).
        assert r["condensed"], "rank %d never condensed a Jacobian" % r["rank"]
        assert r["maxres"] < _RES_TOL, "rank %d: max|residual| = %.3e" % (r["rank"], r["maxres"])
        # Two solves, one plan: the second must reuse it. A plan costs three communication rounds, so
        # rebuilding it per assembly would make the feature slower than not having it.
        assert r["plan_rebuilds"] == 1, \
            "rank %d built %d plans for two solves of one pattern" % (r["rank"], r["plan_rebuilds"])
        assert r["n_components_owned"] > 0, "rank %d owns no component at all" % r["rank"]
    # ...and the cross-rank paths have to be exercised, or this is a serial test in fancy dress: some
    # rank must hold an E row of somebody else's component (so X and y travel and the Schur update is
    # applied by a rank that never saw A_LL), and some component must reach an E dof whose row it does
    # not own (so the increment of that dof is recovered from the halo-synchronised value, not from dx).
    assert sum(r["n_components_remote"] for r in per_rank) > 0, \
        "no component is shared between ranks -- the operator exchange never ran"
    assert sum(r["n_operator_sends"] for r in per_rank) > 0
    assert sum(r["n_operator_sends"] for r in per_rank) == sum(r["n_operator_recvs"] for r in per_rank), \
        "the operator send and receive plans do not match -- one rank would wait for a message nobody sends"
    assert sum(r["n_foreign_E"] for r in per_rank) > 0, \
        "every E dof is locally owned -- the value-based dx_E recovery was never used"
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed condensed (n=%d) vs serial uncondensed" % nproc,
                       _serial(N=8, condense=False, outdir=str(tmp_path / "serial")), per_rank[0])


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_condensed_transient_matches_serial(tmp_path, nproc):
    """Three BDF2 steps. A transient run re-linearises at a new state every step, so it exercises the
    per-step operator exchange and the per-step reconstruction rather than a single one of each."""
    per_rank = _ok(_run(nproc, tmp_path, ["--size", "8", "--condense", "1", "--transient", "3"]))
    for r in per_rank:
        assert r["condensed"]
        assert r["plan_rebuilds"] == 1, "the plan must survive a time step"
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed condensed transient (n=%d) vs serial uncondensed" % nproc,
                       _serial(N=8, condense=False, transient=3, outdir=str(tmp_path / "serial")),
                       per_rank[0])


def test_distributed_condensed_matches_distributed_uncondensed(tmp_path):
    """The same comparison with the partition held fixed, so the only difference is the switch.

    Complementary to the serial reference rather than redundant with it: this one cannot be satisfied by
    a distributed run that happens to agree with serial for an unrelated reason, and it fails loudly if
    condensation changes what the SAME partition computes."""
    on = _ok(_run(4, tmp_path / "on", ["--size", "8", "--condense", "1"]))
    off = _ok(_run(4, tmp_path / "off", ["--size", "8", "--condense", "0"]))
    assert on[0]["condensed"] and not off[0]["condensed"]
    _assert_same_state("distributed condensed vs distributed uncondensed", off[0], on[0])


@pytest.mark.parametrize("nproc", [2, 4])
def test_distributed_hdg_condensed_solve_matches_serial(tmp_path, nproc):
    """The end-to-end case this feature and the interior-facet halo scheme exist for together.

    HDG puts its trace unknowns on the skeleton, so it needs facet fields to survive --distribute; and
    its whole point is that the bulk unknowns can then be eliminated, so it needs condensation as well.
    Neither half is worth much without the other, and each has a way of being quietly wrong that the
    other would mask -- a trace numbered twice still gives a plausible answer, and an elimination that
    silently declined gives the right one. Hence: the answer against serial, AND the positive signal
    that the elimination ran.

    "D1"/"DL" rather than "D2"/"D2" only because an affine trace is the natural one for a "D1" bulk;
    a nodal facet space distributes as well (tests/test_mpi_facet_fields.py)."""
    per_rank = _ok(_run(nproc, tmp_path, ["--size", "8", "--mode", "hdg", "--condense", "1",
                                          "--space", "D1", "--facet-space", "DL"], timeout=600))
    for r in per_rank:
        assert r["condensed"], "rank %d never condensed a Jacobian" % r["rank"]
        assert r["maxres"] < _RES_TOL, "rank %d: max|residual| = %.3e" % (r["rank"], r["maxres"])
        assert r["plan_rebuilds"] == 1, \
            "rank %d built %d plans for two solves of one pattern" % (r["rank"], r["plan_rebuilds"])
        # Distributed there is no need to move the row cuts: oomph renumbers each rank's dofs into one
        # contiguous range, so an element's block is inside one rank's rows by construction.
        assert r["row_cuts"] == [], "rank %d asked for a snapped row split on a distributed run" % r["rank"]
    _assert_ranks_agree(per_rank)
    _assert_same_state("distributed condensed HDG (n=%d) vs serial uncondensed" % nproc,
                       _serial_hdg(N=8, condense=False, space="D1", facet_space="DL",
                                   outdir=str(tmp_path / "serial")),
                       per_rank[0])


def test_interface_adopting_condensed_dofs_matches_serial(tmp_path):
    """Requirement 2 of dev_docs/static_condensation.md, distributed.

    An interface element adopts the adjacent bulk elements' DL pressure Data and writes into the
    condensed dofs' OWN rows. Distributed that is the case the whole post-assembly design exists for:
    the entries reaching a component's rows may be assembled on a rank that does not own them, and the
    owner only sees them after the merge."""
    per_rank = _ok(_run(2, tmp_path, ["--size", "8", "--condense", "1", "--interface", "1"]))
    for r in per_rank:
        assert r["condensed"]
    _assert_ranks_agree(per_rank)
    _assert_same_state("interface case, distributed condensed vs serial uncondensed",
                       _serial(N=8, condense=False, interface=True, outdir=str(tmp_path / "serial")),
                       per_rank[0])


# ---------------------------------------------------------------------------------------------------
# 2 -- the collective votes. What is being tested is as much "it returned at all" as "it refused".
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nproc", [2, 4])
def test_structurally_singular_selection_is_refused_on_every_rank(tmp_path, nproc):
    """Condensing the whole DL pressure, constant mode included, is structurally singular.

    The guard is evaluated from each rank's own owned block, so a rank-local throw would leave any rank
    that saw nothing wrong sitting in the next collective. All ranks must therefore throw, and throw the
    SAME message -- the one the voting picked."""
    per_rank = _run(nproc, tmp_path, ["--size", "8", "--mode", "refuse"], timeout=420)
    _ok(per_rank)
    msgs = set()
    for r in per_rank:
        assert r["refused"], "rank %d accepted a structurally singular selection" % r["rank"]
        assert "structurally singular" in r["message"]
        msgs.add(r["message"])
    assert len(msgs) == 1, "the ranks refused with different messages: %s" % msgs
    assert "[rank " in next(iter(msgs)), "the refusal does not say which rank decided it"


@pytest.mark.parametrize("nproc", [2, 4])
def test_component_split_across_ranks_is_refused_collectively(tmp_path, nproc):
    """The distributed-only refusal: an interior-penalty DG field couples across every facet, so its
    selected dofs form one component spanning the mesh and therefore the ranks. No rank holds the block
    to be inverted, and no rank may throw on its own."""
    per_rank = _run(nproc, tmp_path, ["--size", "6", "--mode", "straddle"], timeout=420)
    _ok(per_rank)
    msgs = set()
    for r in per_rank:
        assert r["refused"], "rank %d accepted a component split across ranks" % r["rank"]
        assert "split across MPI ranks" in r["message"]
        msgs.add(r["message"])
    assert len(msgs) == 1, "the ranks refused with different messages: %s" % msgs


# ---------------------------------------------------------------------------------------------------
# 3 -- the REPLICATED mode: mpirun without --distribute
# ---------------------------------------------------------------------------------------------------
#
# The mesh is whole on every rank and only the linear system's ROWS are split. The ownership argument
# still holds, and more easily than distributed -- "the rank owning a row holds that dof's Data
# non-halo" is automatic when every rank holds every Data -- but one thing is new: a block can only be
# eliminated on the rank that owns ALL of its rows, and the uniform row split has no idea where the
# blocks are. pyoomph moves the cut points off them where the numbering allows it, which is what these
# two tests separate:
#
#   * element-INTERNAL selections (an HDG trace formulation) are numbered element by element, so the
#     cuts can step over each block and the elimination is served;
#   * a selection mixing NODAL and internal dofs (Crouzeix-Raviart: bubble velocity plus DL pressure)
#     cannot be, because oomph-lib numbers every nodal value before any internal one. That is refused,
#     collectively, with a message that says so rather than one about non-element-local selections.
#
# And there is no halo synchronisation replicated: each rank is a separate copy of the whole problem,
# kept in step only because every rank applies the same increment. The eliminated dofs are not in that
# increment, so their reconstructed values are allgathered -- if they were not, the ranks would drift
# apart silently from the first Newton step.

@pytest.mark.parametrize("nproc", [2, 4])
def test_replicated_hdg_condensed_solve_matches_serial(tmp_path, nproc):
    per_rank = _ok(_run(nproc, tmp_path, ["--size", "8", "--mode", "hdg", "--condense", "1"],
                        timeout=420, distribute=False))
    for r in per_rank:
        assert r["condensed"], "rank %d never condensed a Jacobian" % r["rank"]
        assert r["maxres"] < _RES_TOL, "rank %d: max|residual| = %.3e" % (r["rank"], r["maxres"])
        assert r["plan_rebuilds"] == 1, \
            "rank %d built %d plans for two solves of one pattern" % (r["rank"], r["plan_rebuilds"])
        # Without this the test would also pass on a run where the cuts were left uniform and every
        # block happened to sit inside one -- which is not what is being claimed.
        assert len(r["row_cuts"]) == nproc + 1, \
            "rank %d: no snapped row distribution was asked for (%r)" % (r["rank"], r["row_cuts"])
        assert r["row_cuts"] == sorted(r["row_cuts"]) and r["row_cuts"][-1] == r["ndof"]
    # Every rank must have computed the SAME cuts: the row distribution is decided with no
    # communication at all, so a rank disagreeing would build a different matrix layout than its peers.
    assert len({tuple(r["row_cuts"]) for r in per_rank}) == 1, \
        "the ranks disagree about the row cut points: %s" % [r["row_cuts"] for r in per_rank]
    _assert_ranks_agree(per_rank)
    _assert_same_state("replicated condensed HDG (n=%d) vs serial uncondensed" % nproc,
                       _serial_hdg(N=8, condense=False, outdir=str(tmp_path / "serial")), per_rank[0])


def test_replicated_mixed_nodal_and_internal_selection_is_refused(tmp_path):
    """Crouzeix-Raviart replicated: the bubble velocity is nodal and the DL pressure modes are internal,
    so the two halves of every element's block are hundreds of equations apart and no contiguous row
    split keeps one of them on a single rank. Refused on every rank, with the replicated explanation."""
    per_rank = _run(2, tmp_path, ["--size", "6", "--condense", "1"], timeout=420, distribute=False)
    msgs = set()
    for r in per_rank:
        assert "error" in r, "rank %d accepted an unservable replicated selection" % r["rank"]
        assert "REPLICATED MPI run" in r["error"], r["error"]
        assert "NODAL and element-internal" in r["error"], r["error"]
        msgs.add(r["error"].split("Traceback")[0])
    assert len(msgs) == 1, "the ranks refused with different messages: %s" % msgs
