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

# Worker for tests/test_mpi_structural_assembly.py -- launched under `mpirun ... --distribute`.
# Solves a lid-driven cavity with problem.keep_structural_zeros off or on and prints one
# PYOOMPH_MPI_RESULT line per rank. Kept separate from tests/mpi_worker.py because that one is driven
# by the box_cases refinement matrix, which is not what is being certified here.

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.navier_stokes import NavierStokesEquations
from pyoomph.meshes.simplemeshes import CuboidBrickMesh, RectangularQuadMesh
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class CavityProblem(Problem):
    def __init__(self, dim=2, N=16):
        super().__init__()
        self.dim = dim
        self.N = N

    def define_problem(self):
        if self.dim == 3:
            self.add_mesh(CuboidBrickMesh(N=self.N))
            eqs = NavierStokesEquations(dynamic_viscosity=0.05, mass_density=1)
            for b in ["left", "right", "front", "back", "bottom"]:
                eqs += DirichletBC(velocity_x=0, velocity_y=0, velocity_z=0) @ b
            eqs += DirichletBC(velocity_x=1, velocity_y=0, velocity_z=0) @ "top"
            eqs += DirichletBC(pressure=0) @ "bottom/left/front"
        else:
            self.add_mesh(RectangularQuadMesh(N=self.N))
            eqs = NavierStokesEquations(dynamic_viscosity=0.05, mass_density=1)
            for b in ["left", "right", "bottom"]:
                eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
            eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
            eqs += DirichletBC(pressure=0) @ "bottom/left"
        # Partition-independent observables: evaluate_integral_function skips halo elements and
        # MPI_Allreduce-sums, so these certify the FIELD, not just each rank's slice of it.
        eqs += IntegralObservables(ke=dot(var("velocity"), var("velocity")), vx=var("velocity_x"))
        self.add_equations(eqs @ "domain")


def solve_case(dim, N, structural, outdir=None):
    prob = CavityProblem(dim=dim, N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        p.keep_structural_zeros = structural
        p.solve()
        # Solve a second time from the converged state: the first solve can never reuse anything, so
        # only a repeat exercises the "pattern unchanged -> reuse the symbolic factorisation" branch.
        p.solve()
        obs = p.get_mesh("domain").evaluate_all_observables()
        res = {
            # get_residuals() is gathered to full length, so this is identical on every rank.
            "maxres": float(numpy.max(numpy.abs(numpy.asarray(p.get_residuals())))),
            "ndof": int(p.ndof()),
            "structure_id": int(p.jacobian_structure_id),
        }
        for name, val in obs.items():
            res["obs_" + name] = float(val)
        return res


def _row_maps(values, cols, row_start):
    """The local CSR as one {column: value} dict per row.

    Compared as maps, not as arrays, on purpose: the frozen route emits each row's columns in
    ascending order while oomph-lib emits them in first-seen order, so an element-wise comparison
    would report a difference in LAYOUT as a difference in the MATRIX.
    """
    return [dict(zip(cols[row_start[i]:row_start[i + 1]].tolist(),
                     values[row_start[i]:row_start[i + 1]].tolist()))
            for i in range(len(row_start) - 1)]


def compare_frozen_distributed(dim, N, outdir=None):
    """Assemble the same converged state through both distributed routes and diff them.

    This is the direct certificate for Phase 2b: pyoomph's frozen distributed assembly reproduces
    oomph-lib's parallel_sparse_assemble() exactly. Comparing the assembled matrix rather than the
    solution is deliberate -- a solver can converge to the right answer from a slightly wrong
    Jacobian, so a solution-level check would not see a defective merge permutation.
    """
    prob = CavityProblem(dim=dim, N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()
        p.solve()  # converge first, so both assemblies see identical dof values

        range_before = tuple(p._get_assembly_element_range())
        p.use_frozen_distributed_sparsity = False
        r0, ndof, nnz0, nloc0, v0, c0, s0 = p._assemble_residual_jacobian("")
        # oomph-lib's own routine re-tunes the element range from the elemental timings it just
        # measured (recompute_load_balanced_assembly). Replicated that moves the range, so the frozen
        # assembly below runs on a DIFFERENT slice than the plan the solve above built - which is
        # exactly the staleness the plan's recorded range exists to catch.
        range_after = tuple(p._get_assembly_element_range())
        p.use_frozen_distributed_sparsity = True
        r1, ndof, nnz1, nloc1, v1, c1, s1 = p._assemble_residual_jacobian("")

        out = {"ndof": int(ndof), "nrow_local": int(nloc1), "nnz_oomph": int(nnz0),
               "nnz_frozen": int(nnz1),
               "range_before": list(range_before), "range_after": list(range_after),
               "range_retuned": bool(range_before != range_after),
               # Zero means the frozen route never engaged and this test proved nothing.
               "plans_built": int(p._get_distributed_frozen_rebuild_count()),
               "maxres_diff": float(numpy.max(numpy.abs(r1 - r0))) if len(r0) else 0.0}
        if nloc0 != nloc1:
            out["error"] = "nrow_local differs between the two routes: %d vs %d" % (nloc0, nloc1)
            return out
        A, B = _row_maps(v0, c0, s0), _row_maps(v1, c1, s1)
        worst, missing, extra_nonzero = 0.0, 0, 0
        for ra, rb in zip(A, B):
            for col, val in ra.items():
                if col in rb:
                    worst = max(worst, abs(val - rb[col]))
                elif val != 0.0:
                    missing += 1  # oomph-lib stored a real entry the frozen pattern has no room for
            for col, val in rb.items():
                if col not in ra and val != 0.0:
                    extra_nonzero += 1  # the frozen route put a real entry where oomph-lib has none
        out["max_value_diff"] = worst
        out["missing_nonzero"] = missing
        out["extra_nonzero"] = extra_nonzero
        return out


def compare_frozen_distributed_residuals(dim, N, outdir=None):
    """The frozen residual-only distributed assembly must reproduce oomph-lib's.

    Under MPI, get_residuals() installs a ParallelResidualsHandler and goes through the whole of
    parallel_sparse_assemble() with zero matrices -- recomputing my_eqns, exchanging equation numbers
    and merging by bisection per row, all to sum a vector, once per Newton step. pyoomph substitutes
    a much smaller frozen plan for it.

    Checked at TWO states. A converged residual is ~1e-10, so comparing two near-zero vectors would
    pass whatever the routine did; the unsolved initial state, where the lid boundary condition makes
    the residual O(1), is what actually constrains it.
    """
    prob = CavityProblem(dim=dim, N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.initialise()

        def both():
            p.use_frozen_distributed_sparsity = False
            a = numpy.array(p.get_residuals(), copy=True)
            p.use_frozen_distributed_sparsity = True
            b = numpy.array(p.get_residuals(), copy=True)
            return a, b

        a0, b0 = both()                      # unsolved: residual is O(1)
        p.solve()
        a1, b1 = both()                      # converged: the state a Newton step asks about
        return {
            "ndof": int(p.ndof()),
            # Zero means the frozen route never engaged and this test proved nothing.
            "res_plans_built": int(p._get_distributed_residual_rebuild_count()),
            "init_norm": float(numpy.max(numpy.abs(a0))),
            "init_maxdiff": float(numpy.max(numpy.abs(b0 - a0))),
            "conv_norm": float(numpy.max(numpy.abs(a1))),
            "conv_maxdiff": float(numpy.max(numpy.abs(b1 - a1))),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--structural", type=int, default=0)
    ap.add_argument("--mode", default="solve", choices=["solve", "compare-distributed", "compare-residuals"])
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(),
               "case": "dim%d_N%d_struct%d" % (args.dim, args.size, args.structural)}
    try:
        if args.mode == "compare-distributed":
            payload.update(compare_frozen_distributed(args.dim, args.size, outdir=args.outdir))
        elif args.mode == "compare-residuals":
            payload.update(compare_frozen_distributed_residuals(args.dim, args.size, outdir=args.outdir))
        else:
            payload.update(solve_case(args.dim, args.size, bool(args.structural), outdir=args.outdir))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    # Straight to the real stdout, not through print(): pyoomph's default MPI console mode
    # ("condensed") wraps sys.stdout and MUTES every rank but 0 on a replicated run, and a test that
    # only ever sees rank 0 cannot notice a rank disagreeing.
    sys.__stdout__.write("PYOOMPH_MPI_RESULT " + json.dumps(payload) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
