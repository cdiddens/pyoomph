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

# Worker for tests/test_mpi_adaptive_recovery.py -- launched under mpirun, with and without
# --distribute. See dev_docs/adaptive_resolve_recovery.md.

import argparse
import json
import traceback

import numpy

from pyoomph import Problem, DirichletBC
from pyoomph.expressions import var, exp
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.generic import SpatialErrorEstimator
from pyoomph.generic.adaptive_recovery import AdaptiveResolveRecovery
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class SabotagedPoisson(Problem):
    """The re-solve after the first in-solve adaptation is made to fail.

    The request is issued from **rank 0 only**, on purpose. That is the partition-dependent case:
    Problem::consume_newton_abort_request() Allreduces it before throwing, so a decision one rank
    reached becomes a failure every rank sees - which is exactly the precondition the recovery
    handler relies on, since restoring a state is collective.
    """

    def __init__(self, N=8, rank_zero_only=True):
        super().__init__()
        self.N = N
        self.rank_zero_only = rank_zero_only
        # All adaptation happens inside solve(), so the counter below sees only the loop under test.
        self.initial_adaption_steps = 0
        self.adapt_count = 0
        self.armed = False
        self.ndof_before_sabotaged_adapt = None

    def define_problem(self):
        x = var("coordinate")
        eqs = PoissonEquation(source=exp(-((x[0]-0.5)**2+(x[1]-0.5)**2)/0.002))
        eqs += DirichletBC(u=0) @ ["bottom", "top", "left", "right"]
        eqs += SpatialErrorEstimator(u=1)
        self += RectangularQuadMesh(N=self.N)
        self += eqs @ "domain"

    def _adapt(self):
        self.adapt_count += 1
        if self.adapt_count == 1:
            self.ndof_before_sabotaged_adapt = int(self.ndof())
            self.armed = True
        return super()._adapt()

    def actions_before_newton_solve(self):
        if self.armed:
            self.armed = False
            if (not self.rank_zero_only) or get_mpi_rank() == 0:
                self._request_newton_abort("sabotage: pretend the re-solve after the adaptation diverged")
        return super().actions_before_newton_solve()


def recover(outdir, distributed, strategy):
    p = SabotagedPoisson()
    with p:
        p.set_output_directory(outdir)
        p.quiet()
        if distributed:
            p.set_linear_solver("petsc_mumps")
        p.adaptive_resolve_recovery = AdaptiveResolveRecovery(strategies=[strategy], quiet=True)
        p.solve(spatial_adapt=3)
        dofs = numpy.array(p.get_current_dofs()[0])
        return {
            "ndof": int(p.ndof()),
            "ndof_before_sabotaged_adapt": p.ndof_before_sabotaged_adapt,
            "total_failures": int(p.adaptive_resolve_recovery.total_failures),
            "finite": bool(numpy.all(numpy.isfinite(dofs))),
            # Order-independent: load_state may renumber the dofs, and under --distribute the
            # gathered vector is in the distributed numbering anyway. The sum is a cheap way to ask
            # whether every rank ended up with the same solution.
            "dofsum": float(numpy.sum(dofs)),
            # The recovered problem must still be usable, which under MPI also means every later
            # collective still matches up.
            "resolve_ok": _still_usable(p),
        }


def _still_usable(p):
    p.solve(spatial_adapt=1)
    dofs = numpy.array(p.get_current_dofs()[0])
    return bool(numpy.all(numpy.isfinite(dofs)))


def snapshot_roundtrip(outdir, distributed):
    """_snapshot_state/_restore_state on their own, which is where the MPI-specific code lives:
    the per-rank buffer when replicated, the broadcast of rank 0's merged stream when distributed."""
    p = SabotagedPoisson()
    p.rank_zero_only = False
    p.adapt_count = -10**6  # never arm
    with p:
        p.set_output_directory(outdir)
        p.quiet()
        if distributed:
            p.set_linear_solver("petsc_mumps")
        p.solve(spatial_adapt=2)
        before_ndof = int(p.ndof())
        before = numpy.sort(numpy.array(p.get_current_dofs()[0]))
        snap = p._snapshot_state()
        p.solve(spatial_adapt=2)   # move away: different mesh, different solution
        moved_ndof = int(p.ndof())
        p._restore_state(snap)
        after = numpy.sort(numpy.array(p.get_current_dofs()[0]))
        return {
            "snapshot_bytes": int(len(snap)),
            "before_ndof": before_ndof,
            "moved_ndof": moved_ndof,
            "restored_ndof": int(p.ndof()),
            "restored_exact": bool(numpy.array_equal(before, after)),
            "resolve_ok": _still_usable(p),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["recover", "snapshot"])
    ap.add_argument("--strategy", default="accept_unadapted")
    ap.add_argument("--outdir", required=True)
    # --distribute is consumed by pyoomph's own argument handling; we only need to know about it.
    args, extra = ap.parse_known_args()
    distributed = "--distribute" in extra

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "distributed": distributed}
    try:
        if args.mode == "recover":
            payload.update(recover(args.outdir, distributed, args.strategy))
        else:
            payload.update(snapshot_roundtrip(args.outdir, distributed))
    except Exception as e:  # noqa: BLE001
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
