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

# Worker for tests/test_bifurcation_scan.py -- Problem.find_bifurcation_via_eigenvalues.
#
# Two defects and one new option are covered, on the Hopf of the ramped Brusselator (the problem
# lives in adapt_while_tracking_worker.py, which this imports rather than duplicating):
#
#   * the search did not converge. It kept no bracket -- only the last sign and a ds it rescaled --
#     so after the first sign change it ran a damped recursion with a fixed point of its own rather
#     than a bracketed root-find, and settled at a parameter value that was NOT the root. Measured
#     before the fix: B = 2.39063 with Re = +1.38e-3, against a root at 2.3875, and then it span
#     until the collapsing arclength step killed the continuation with an OomphException. Any
#     epsilon tighter than about 1e-2 hit it; the loose values that "worked" returned whatever point
#     the march happened to stop on.
#
#   * it refused to start from an unstable solution, although all the search needs is a sign CHANGE,
#     which is as detectable from above the axis as from below.
#
#   * track_eigenvector follows the MODE by eigenvector overlap instead of a fixed index into
#     whatever the eigensolver returned. Off by default.

import argparse
import json
import os
import sys

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_while_tracking_worker import BrusselatorProblem  # noqa: E402

from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc  # noqa: E402


def run(start_B, initstep, epsilon, neigen, track, outdir, max_steps=80):
    res = {"start_B": start_B, "initstep": initstep, "epsilon": epsilon,
           "neigen": neigen, "track": track}
    with BrusselatorProblem() as p:
        p.set_output_directory(outdir)
        p.set_linear_solver("petsc_mumps")
        p.quiet()
        p.get_global_parameter("A0").value = 1.0
        p.get_global_parameter("B").value = start_B
        p.solve()
        ev0, _ = p.solve_eigenproblem(neigen)
        res["start_real_part"] = float(numpy.real(ev0[0]))

        steps = 0
        last = None
        try:
            for param, ev in p.find_bifurcation_via_eigenvalues(
                    "B", initstep, neigen=neigen, max_ds=0.08, epsilon=epsilon,
                    track_eigenvector=track):
                steps += 1
                last = (float(param), complex(ev))
                if steps > max_steps:
                    res["outcome"] = "exhausted"
                    break
            else:
                res["outcome"] = "converged"
        except Exception as e:
            res["outcome"] = "raised"
            res["error"] = type(e).__name__ + ": " + str(e)[:300]
        res["steps"] = steps
        if last is not None:
            res["B"] = last[0]
            res["real_part"] = float(numpy.real(last[1]))
            res["omega"] = abs(float(numpy.imag(last[1])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--start-B", type=float, default=1.5)
    ap.add_argument("--initstep", type=float, default=0.08)
    ap.add_argument("--epsilon", type=float, default=1e-7)
    ap.add_argument("--neigen", type=int, default=6)
    ap.add_argument("--track", action="store_true")
    args, _ = ap.parse_known_args()
    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
    try:
        payload.update(run(args.start_B, args.initstep, args.epsilon, args.neigen,
                           args.track, args.outdir))
    except Exception as e:
        import traceback
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-3000:]
    # sys.__stdout__: the MPI console mutes stdout on every rank but 0.
    print("PYOOMPH_SCAN_RESULT " + json.dumps(payload), file=sys.__stdout__, flush=True)


if __name__ == "__main__":
    main()
