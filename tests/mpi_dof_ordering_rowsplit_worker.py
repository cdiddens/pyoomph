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

# Worker for tests/test_mpi_dof_ordering_rowsplit.py.
#
# Solves a Crouzeix-Raviart cavity, optionally under a dof layout and optionally with static
# condensation, and reports the row cuts, the layout's blocks and a summary of the converged state.

import argparse
import json
import os
import traceback

from pyoomph import DirichletBC, ElementBlockOrdering, InitialCondition, NodalBlockOrdering, Problem
from pyoomph.equations.generic import StaticCondensation
from pyoomph.equations.navier_stokes import StokesEquations
from pyoomph.generic.mpi import get_mpi_rank
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class _CRCavity(Problem):
    """Lid-driven Stokes on Crouzeix-Raviart triangles: C2TB velocities, whose cell-interior bubble
    node belongs to exactly one element, and a DL pressure. The condensation selection pairs the
    bubble velocity (NODAL) with the pressure gradient modes (element INTERNAL), and oomph numbers
    every nodal value before any internal one -- which is what puts the two halves of a block hundreds
    of equations apart and makes this the case replicated MPI refuses."""

    def __init__(self, condense):
        super().__init__()
        self.condense = condense

    def define_problem(self):
        self += RectangularQuadMesh(N=4, split_in_tris="left")
        ns = StokesEquations(dynamic_viscosity=1, mode="CR")
        eqs = ns + ns.create_pressure_fixation(value=0)
        if self.condense:
            eqs += StaticCondensation(velocity="bubble", pressure=[1, 2])
        eqs += InitialCondition(velocity_x=0, velocity_y=0)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        self += eqs @ "domain"


def _run(args):
    payload = {"rank": get_mpi_rank(), "layout": args.layout, "condense": bool(args.condense)}
    try:
        p = _CRCavity(bool(args.condense))
        p.set_output_directory(os.path.join(args.outdir, "out"))
        p.quiet()
        p.set_linear_solver("superlu")
        if args.layout == "elem":
            p.dof_ordering = ElementBlockOrdering("domain/velocity_*", "domain/pressure")
        elif args.layout == "nodal":
            p.dof_ordering = NodalBlockOrdering("domain/velocity_x", "domain/velocity_y", "domain/pressure")
        p.initialise()

        blocks = [list(b) for b in p._dof_ordering_blocks()]
        cuts = list(p._dof_ordering_row_cuts())
        payload["nblocks"] = len(blocks)
        payload["maxblock"] = max((b[1] - b[0] + 1 for b in blocks), default=0)
        payload["cuts"] = cuts
        # The property the cuts exist for. Interior cut points only: 0 and ndof are not cuts.
        payload["nstraddled"] = sum(1 for b in blocks for c in cuts[1:-1] if b[0] < c <= b[1])
        payload["cond_cuts"] = list(p._condensation_row_cuts())

        p.solve()
        m = p.get_mesh("domain")
        payload.update(
            ndof=int(p.ndof()),
            distributed=bool(p.is_distributed()),
            condensed=bool(p._last_jacobian_was_condensed()) if args.condense else False,
            newton=len(p.get_last_residual_convergence()),
            checksum=sum(n.x(0) + n.x(1) + sum(n.value(i) for i in range(n.nvalue())) for n in m.nodes()),
        )
    except Exception as e:
        payload["error"] = str(e)[:600]
        payload["traceback"] = traceback.format_exc()[-1500:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--layout", default="none")
    ap.add_argument("--condense", type=int, default=0)
    args, _ = ap.parse_known_args()  # --distribute is consumed by Problem.initialise()
    os.makedirs(args.outdir, exist_ok=True)
    _run(args)


main()
