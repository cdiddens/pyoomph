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


# Launched by tests/test_mpi_arclength_smoothing.py as
#     mpirun -n N python mpi_arclength_smoothing_worker.py --outdir D [--distribute]
# (and directly, without mpirun, for the serial reference).
#
# EnforcedInterfacialLaplaceSmoothing keeps the nodes of an interface at their original relative
# spacing by solving for the arclength along it and shifting the nodes tangentially. The reference
# arclength it measures at setup is a property of the WHOLE curve, and that is what distribution
# breaks: each rank sees a piece of the interface and used to start counting from zero on its own
# piece, so the reference configuration was discontinuous across the partition boundary and the two
# ranks disagreed about the nodes they share. On the spreading droplet of
# docs/source/tutorial/ale/spread that turned a converging solve (Newton residual 4e-6 after one step)
# into a diverging one (8e-2, then inf).
#
# The interface here is deliberately deformed AND stretched non-uniformly by the prescribed motion of
# the top boundary, so the tangential shift has real work to do: with the smoothing switched off the
# top nodes keep their x positions, with it they redistribute. Comparing the positions against a
# serial run is therefore a test of the arclength, not just of the mesh solve.

import argparse
import json
import os
import sys

import numpy

from pyoomph import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh, EnforcedInterfacialLaplaceSmoothing
from pyoomph.equations.generic import ElementSpace
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_nproc, get_mpi_rank


class SmoothedTopProblem(Problem):
    """A unit square whose top boundary is pushed up by a bump, with the top nodes free to slide."""

    def __init__(self, N=8, amplitude=0.35):
        super().__init__()
        self.N, self.amplitude = N, amplitude

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N)
        eqs = LaplaceSmoothedMesh()
        # No field lives on this domain except the mesh position, so the coordinate space cannot be
        # deduced and has to be stated.
        eqs += ElementSpace("C2")
        # The bump is not symmetric about x=0.5, so equidistributing the arclength moves the nodes by
        # visibly different amounts along the interface rather than by a global shift.
        x = var("coordinate_x")
        eqs += DirichletBC(mesh_y=1 + self.amplitude * x ** 2 * sin(pi * x)) @ "top"
        for b in ["left", "right"]:
            eqs += DirichletBC(mesh_x=True) @ b
        eqs += DirichletBC(mesh_y=True) @ "bottom"
        eqs += EnforcedInterfacialLaplaceSmoothing().with_corners("left", "right") @ "top"
        self += eqs @ "domain"


def _top_nodes(problem):
    """This rank's nodes of the top interface: (x, y, reference arclength), sorted by x."""
    mesh = problem.get_mesh("domain/top")
    index = mesh.has_interface_dof_id("_s_fixed_top")
    out = []
    for n in mesh.nodes():
        out.append((float(n.x(0)), float(n.x(1)), float(n.value(n.additional_value_index(index)))))
    return sorted(out)


def run_case(N=8, outdir="_arclength_smoothing"):
    with SmoothedTopProblem(N=N) as problem:
        problem.set_output_directory(outdir)
        problem.quiet()
        problem.solve()
        nodes = _top_nodes(problem)
        return {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(),
                "distributed": bool(problem.is_distributed()),
                "ndof": int(problem.ndof()),
                "nodes": nodes,
                # The end of the reference arclength is the length of the whole curve: a rank that
                # measured only its own piece reports a fraction of it. None where this rank holds no
                # part of the interface at all, which happens from three ranks on.
                "max_arclength": max((n[2] for n in nodes), default=None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=8)
    args, _ = ap.parse_known_args()
    print("PYOOMPH_MPI_RESULT " + json.dumps(run_case(N=args.size, outdir=args.outdir)), flush=True)


if __name__ == "__main__":
    main()
