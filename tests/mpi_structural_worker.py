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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--structural", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(),
               "case": "dim%d_N%d_struct%d" % (args.dim, args.size, args.structural)}
    try:
        payload.update(solve_case(args.dim, args.size, bool(args.structural), outdir=args.outdir))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
