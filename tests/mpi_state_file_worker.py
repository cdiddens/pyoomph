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

# Worker for test_mpi_state_files.py. Solves a small Poisson problem with a sharp source (so that
# adaptive refinement actually happens, and non-uniformly) and either writes a state file or loads one
# and reports what arrived.
#
# The fingerprint is deliberately numbering-independent: it sums each nodal value weighted by the
# node's position, over the non-halo elements only. A state that landed on the right nodes reproduces
# it; one whose values were shifted between nodes does not, however plausible the plot would look.

import argparse
import json
import sys
import traceback

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc, get_mpi_sum


class PoissonEqs(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")
        self.define_scalar_field("d0field", "D0")

    def define_residuals(self):
        u, v = var_and_test("u")
        d0, d0t = var_and_test("d0field")
        x, y = var("coordinate_x"), var("coordinate_y")
        source = 1 + x * y + 50 * exp(-200 * ((x - 0.35) ** 2 + (y - 0.6) ** 2))
        self.add_residual(weak(grad(u), grad(v)) - weak(source, v) + weak(d0 - u, d0t))


class StateProblem(Problem):
    def __init__(self, N=6, adapt=False):
        super().__init__()
        self.N, self.adapt = N, adapt
        self.write_states = False
        self.eigen_data_in_states = False
        self.continuation_data_in_states = False

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1, 1])
        eqs = PoissonEqs() + DirichletBC(u=0) @ "left"
        if self.adapt:
            eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"
        self.max_refinement_level = 2


def fingerprint(problem):
    mesh = problem.get_mesh("domain")
    weighted, square, count = 0.0, 0.0, 0
    for i in range(mesh.nelement()):
        if mesh.element_pt(i).non_halo_proc_ID() >= 0:
            continue  # a halo copy: its owner counts it
        element = mesh.element_pt(i)
        for j in range(element.nnode()):
            node = element.node_pt(j)
            value = node.value(0)
            weighted += value * (1 + node.x(0) + 2 * node.x(1))
            square += value * value
            count += 1
    return [get_mpi_sum(weighted), get_mpi_sum(square), get_mpi_sum(count)]


def run_case(mode, fname, resave_to=None, N=6, adapt=False, outdir="_state_test"):
    result = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "mode": mode}
    with StateProblem(N=N, adapt=adapt) as problem:
        problem.set_output_directory(outdir)
        problem.solve()
        if adapt:
            problem.solve(spatial_adapt=2)
        result["distributed"] = bool(problem.is_distributed())
        if mode == "save":
            result["fingerprint"] = fingerprint(problem)
            problem.save_state(fname)
        else:
            mesh = problem.get_mesh("domain")
            result["nnode_before"] = mesh.nnode()
            # Wreck the state first: a load that quietly does nothing must not pass
            for i in range(mesh.nnode()):
                mesh.node_pt(i).set_value(0, -12345.0)
            problem.load_state(fname)
            result["fingerprint"] = fingerprint(problem)
            if resave_to is not None:
                problem.save_state(resave_to)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["save", "load"])
    parser.add_argument("file")
    parser.add_argument("--resave-to", default=None)
    parser.add_argument("--outdir", default="_state_test")
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--adapt", action="store_true")
    parser.add_argument("--distribute", action="store_true")  # consumed by pyoomph itself
    args, _ = parser.parse_known_args()
    try:
        out = run_case(args.mode, args.file, resave_to=args.resave_to, N=args.size,
                       adapt=args.adapt, outdir=args.outdir)
    except BaseException as e:
        out = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "error": str(e),
               "traceback": traceback.format_exc()}
    print("PYOOMPH_MPI_RESULT " + json.dumps(out), flush=True)
