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

# Worker for test_mpi_error_estimator.py. Solves one of three cases and prints a per-rank result:
#
#     PYOOMPH_MPI_RESULT <json>
#
# Everything lives inside main() and is guarded by __main__, so that pytest's collection (which
# imports every file in tests/) cannot trip over the argument parsing -- see the workers fixed in
# c80e7f0.
#
# The three cases each cover a code path that the existing MPI suites do not reach:
#
#   interface   The Z2 patch-local recovery frame on a mesh with a co-dimension. Run WITHOUT
#               --distribute on purpose: LagrZ2ErrorEstimator only takes its coefficient-broadcast
#               branch when the mesh is NOT distributed while MPI is initialised, and that is the
#               branch where a receiving rank rebuilds the sender's frame from the element list.
#               With --distribute each rank assembles its own patches and the branch never runs.
#   groups      Two compound-flux groups, i.e. n_compound_flux > 1. That number is the element count
#               of the MPI_Allreduce on flux_norm, so all ranks must agree on it.
#   ndof        The desired_ndof controller: the threshold bisection is a fixed number of MPI_SUM
#               reductions and must give every rank the same answer, and the element count behind it
#               has to skip halo copies or shared elements are counted once per holder.

import argparse
import json
import os
import sys
import traceback


def build_problem(case):
    from pyoomph import Problem, DirichletBC, SpatialErrorEstimator, InterfaceEquations
    from pyoomph.expressions import var, var_and_test, grad, weak, testfunction, exp, tanh
    from pyoomph.equations.poisson import PoissonEquation
    from pyoomph.meshes.simplemeshes import RectangularQuadMesh

    class BoundaryFront(InterfaceEquations):
        def define_fields(self):
            self.define_scalar_field("v", "C2")

        def define_residuals(self):
            x = var("coordinate")
            self.add_residual(weak(var("v")-tanh((x[1]-0.5)/0.08), testfunction("v")))

        def define_error_estimators(self):
            self.add_spatial_error_estimator(grad(var("v"), nondim=True))

    class P(Problem):
        def define_problem(self):
            x = var("coordinate")
            eqs = PoissonEquation(source=exp(-((x[0]-0.5)**2+(x[1]-0.5)**2)/0.01))
            eqs += DirichletBC(u=0) @ "bottom"
            if case == "interface":
                # The estimator lives on a vertical boundary: x is constant along it, which is the
                # configuration the global-coordinate recovery cannot parametrise at all.
                eqs += BoundaryFront() @ "left"
                eqs += SpatialErrorEstimator(u=1)
            elif case == "groups":
                eqs += SpatialErrorEstimator(u=1, group="a")
                eqs += SpatialErrorEstimator("mesh", group="b", normalize_relative=0, weight=2)
            else:
                eqs += SpatialErrorEstimator(u=1, normalize_relative=0)
            self += RectangularQuadMesh(N=8)
            self += eqs @ "domain"

    return P()


def solve_case(case, outdir):
    problem = build_problem(case)
    with problem:
        problem.set_output_directory(outdir)
        problem.quiet(True)
        problem.max_refinement_level = 5
        problem.initial_adaption_steps = 0
        if case == "ndof":
            problem.desired_ndof = 6000
        problem.solve()
        from pyoomph.generic.mpi import get_mpi_sum
        for _ in range(8):
            nref, nunref = problem._adapt()
            problem.solve()
            # The break has to be agreed globally. nref/nunref are rank-local, so a rank that
            # finished while another had work left would leave the loop early and the next _adapt --
            # which is collective -- would hang waiting for it. Problem's own adaptation loops
            # MPI-sum these for the same reason.
            if int(get_mpi_sum(int(nref))) == 0 and int(get_mpi_sum(int(nunref))) == 0:
                break
        mesh = problem.get_mesh("domain")
        res = {"ndof": int(problem.ndof()),
               "nelem_nonhalo": int(sum(0 if e.is_halo() else 1 for e in mesh.elements()))}
        if case == "interface":
            imesh = problem.get_mesh("domain/left")
            imesh._enable_adaptation()
            errs = list(imesh.get_elemental_errors())
            imesh._disable_adaptation()
            # Sorted, because the element enumeration is partition-dependent while the multiset of
            # errors is not. A sum alone would hide two errors swapping places with a third.
            res["interface_errors"] = sorted(round(float(e), 10) for e in errs)
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--outdir", required=True)
    args, _ = parser.parse_known_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "case": args.case}
    try:
        payload.update(solve_case(args.case, os.path.join(args.outdir, args.case)))
    except Exception as e:
        payload["error"] = type(e).__name__+": "+str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT "+json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
