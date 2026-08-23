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

# Worker launched by tests/test_mpi_periodic.py as
#     mpirun -n N python mpi_periodic_worker.py --spec '<json>' --outdir <dir> [--N 16] --distribute
# It is NOT a test module itself (the name deliberately does not start with "test_", so pytest ignores it).
#
# Each rank solves the requested cases from tests/periodic_cases.py and prints one machine-readable line
# per case:
#     PYOOMPH_MPI_RESULT <json>
# with the rank, the case name and the measurements. The harness parses those lines and compares them
# both ACROSS RANKS and against its own serial reference. A case that raises reports "error" instead, so
# a crash in one case still yields a diagnosable result rather than a silent non-zero exit.
#
# --distribute is consumed by pyoomph's own command-line parser inside Problem.initialise(); it is passed
# through here rather than acted on.

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="JSON list of case names")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--N", type=int, default=16)
    args, _ = parser.parse_known_args()

    import periodic_cases

    rank, nproc = get_mpi_rank(), get_mpi_nproc()
    for case in json.loads(args.spec):
        payload = {"rank": rank, "nproc": nproc, "case": case}
        try:
            payload.update(periodic_cases.run_case(
                case, N=args.N, outdir=os.path.join(args.outdir, case)))
        except Exception as e:
            payload["error"] = type(e).__name__ + ": " + str(e)
            payload["traceback"] = traceback.format_exc()[-2000:]
        print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
