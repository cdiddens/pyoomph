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

# Worker for tests/test_mpi_newton_abort.py -- launched under `mpirun ... --distribute`.

import argparse
import json
import traceback

import numpy

from pyoomph import Problem, Equations, DirichletBC, IntegralObservables
from pyoomph.expressions import var, dot
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class RejectOnceOnRankZero(Equations):
    """Rejects a single Newton step, and only from rank 0.

    The partition-dependent case on purpose: whatever decides to reject -- an interface about to
    self-intersect, in the real user -- normally only sees it on the ranks holding that part of the
    mesh, while abandoning the solve has to be unanimous.
    """

    def __init__(self, reject_at):
        super().__init__()
        self.reject_at = reject_at
        self.done = False

    def before_newton_convergence_check(self, eqtree):
        if self.reject_at < 0 or self.done or get_mpi_rank() != 0:
            return True
        problem = self.get_current_code_generator().get_problem()
        if float(problem.get_current_time(dimensional=False)) >= self.reject_at:
            self.done = True
            return False
        return True


class Cavity(Problem):
    def __init__(self, N=6, reject_at=-1.0):
        super().__init__()
        self.N = N
        self.reject_at = reject_at

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        eqs = NavierStokesEquations(dynamic_viscosity=0.05, mass_density=1)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        eqs += DirichletBC(pressure=0) @ "bottom/left"
        eqs += RejectOnceOnRankZero(self.reject_at)
        eqs += IntegralObservables(ke=dot(var("velocity"), var("velocity")))
        self.add_equations(eqs @ "domain")


def dof_roundtrip(outdir):
    """get_current_dofs()/set_current_dofs() on a distributed problem.

    Both index a row-partitioned vector by GLOBAL equation number, which used to run off the end of
    the local buffer and corrupt the heap -- silently, since the abort usually landed much later in
    an unrelated allocation. Correctness, not just survival: the vector must be the same on every
    rank, writing it back must change nothing, and a modification must take effect exactly.
    """
    p = Cavity()
    with p:
        p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        p.solve()

        d0 = numpy.array(p.get_current_dofs()[0])
        ke0 = float(p.get_mesh("domain").evaluate_all_observables()["ke"])
        p.set_current_dofs(list(d0))
        d1 = numpy.array(p.get_current_dofs()[0])
        ke1 = float(p.get_mesh("domain").evaluate_all_observables()["ke"])
        halved = d0 * 0.5
        p.set_current_dofs(list(halved))
        d2 = numpy.array(p.get_current_dofs()[0])
        return {
            "ndof": int(p.ndof()),
            "n": int(len(d0)),
            # Index-weighted, so a gather that gets the ORDER wrong is caught, not just the multiset.
            "checksum": float(numpy.sum(d0 * numpy.arange(1, len(d0) + 1))),
            "finite": bool(numpy.all(numpy.isfinite(d0))),
            "roundtrip_exact": bool(numpy.array_equal(d0, d1)),
            "ke_shift": abs(ke1 - ke0),
            "set_exact": bool(numpy.array_equal(d2, halved)),
        }


def transient_rejection(outdir, reject_at):
    """The use case the abort exists for: a rejected step must reduce dt and the run must continue."""
    p = Cavity(reject_at=reject_at)
    with p:
        p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        p.run(0.5, startstep=0.1, temporal_error=1e-3, outstep=False)
        return {
            "final_time": float(p.get_current_time(dimensional=False)),
            "ke": float(p.get_mesh("domain").evaluate_all_observables()["ke"]),
            "ndof": int(p.ndof()),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dofs", "transient"])
    ap.add_argument("--reject-at", type=float, default=-1.0)
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
    try:
        if args.mode == "dofs":
            payload.update(dof_roundtrip(args.outdir))
        else:
            payload.update(transient_rejection(args.outdir, args.reject_at))
    except Exception as e:  # noqa: BLE001
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
