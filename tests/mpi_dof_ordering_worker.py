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

# Worker for tests/test_mpi_dof_ordering.py: solve the moving-mesh/interface problem under a given dof
# ordering and report a rank-local summary of the converged state.
#
# Run as:  mpirun -n N python3 mpi_dof_ordering_worker.py --outdir DIR [--mode reverse] [--distribute]

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyoomph.generic.mpi import get_mpi_rank  # noqa: E402

from test_dof_description_walk import _MovingWithInterfaces  # noqa: E402


def _run(args):
    payload = {"rank": get_mpi_rank(), "mode": args.mode}
    try:
        p = _MovingWithInterfaces()
        p.set_output_directory(os.path.join(args.outdir, "out"))
        p.quiet()
        p._dof_ordering_mode = args.mode
        p.initialise()
        p.solve()
        m = p.get_mesh("domain")
        # Order-independent summaries of this rank's nodal state. A dof vector cannot be compared
        # across the two arms distributed (the ranks own different dofs and the reordering is
        # rank-local), and even replicated it is permuted by construction -- so what is compared is
        # the state where it lives, reduced to sums that do not depend on the node ordering either.
        checksum = 0.0
        sqsum = 0.0
        nstate = 0
        for n in m.nodes():
            vals = [n.x(d) for d in range(m.get_dimension())] + [n.value(i) for i in range(n.nvalue())]
            checksum += sum(vals)
            sqsum += sum(v * v for v in vals)
            nstate += len(vals)
        payload.update(ndof=int(p.ndof()), distributed=bool(p.is_distributed()),
                       nstate=nstate, checksum=checksum, sqsum=sqsum,
                       newton=len(p.get_last_residual_convergence()))
    except Exception as e:
        payload["error"] = str(e)
        payload["traceback"] = traceback.format_exc()
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", default="")
    args, _ = ap.parse_known_args()  # --distribute is consumed by Problem.initialise()
    os.makedirs(args.outdir, exist_ok=True)
    _run(args)


main()
