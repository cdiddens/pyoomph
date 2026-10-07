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
#
#   * stay_stable_file: never leave the stable side. A step that lands unstable is DISCARDED -- the
#     saved stable state is reloaded -- and retried with a secant prediction, so the search creeps up
#     to the bifurcation from below and the caller never has to solve on the unstable branch. It has
#     no in-tree callers but is used from external scripts, and had no coverage at all.

import argparse
import json
import os
import sys
from typing import Any

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_while_tracking_worker import BrusselatorProblem  # noqa: E402

from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc  # noqa: E402


def _instrument_stay_stable(problem_cls, record):
    """Count the reloads, and record dparameter/ds both after the reload and AT THE HANDOVER.

    The stay_stable_file retry computes a secant step as a PARAMETER delta and hands it to
    arclength_continuation, whose ds is an ARCLENGTH. The two are the same number only while
    dparameter/ds is 1 at the moment of the handover, which is what reset_arc_length_parameters()
    guarantees.

    Which of the two readings matters depends on a setting, and that is the point of recording both:

      * with continuation_data_in_states False (the default) load_state restores no tangent, so
        dparameter/ds is already 1 after the reload and the reset is redundant;
      * with it True, load_state restores the real tangent -- measured 0.7071 on this problem -- and
        the reset is the only thing that puts it back to 1.

    So a test that only runs the default cannot see the reset doing anything: removing it leaves the
    default case passing. The handover reading is the invariant that holds in both.
    """
    import pyoomph.generic.problem as _P
    orig_load = _P.Problem.load_state
    orig_cont = _P.Problem.arclength_continuation

    def patched_load(self, *a, **k):
        r = orig_load(self, *a, **k)
        record["reloads"] += 1
        record["dparam_ds_after_reload"].append(
            float(self.get_arc_length_parameter_derivative()))
        record["expect_secant"] = True
        return r

    def patched_cont(self, param, ds, **k):
        if record.pop("expect_secant", False):
            # The step that consumes the secant delta. dparameter/ds HERE is what decides whether
            # the delta is interpreted as the parameter increment it is.
            record["dparam_ds_at_handover"].append(
                float(self.get_arc_length_parameter_derivative()))
            before = float(self.get_global_parameter("B").value)
            r = orig_cont(self, param, ds, **k)
            after = float(self.get_global_parameter("B").value)
            record["dB_over_ds"].append((after-before)/ds if ds else float("nan"))
            return r
        return orig_cont(self, param, ds, **k)

    _P.Problem.load_state = patched_load               #type:ignore
    _P.Problem.arclength_continuation = patched_cont   #type:ignore
    return (orig_load, orig_cont)


def run(start_B, initstep, epsilon, neigen, track, outdir, max_steps=80, stay_stable=False,
        continuation_data_in_states=False):
    res = {"start_B": start_B, "initstep": initstep, "epsilon": epsilon,
           "neigen": neigen, "track": track, "stay_stable": stay_stable,
           "continuation_data_in_states": continuation_data_in_states}
    with BrusselatorProblem() as p:
        p.set_output_directory(outdir)
        p.set_linear_solver("petsc_mumps")
        p.quiet()
        p.continuation_data_in_states = continuation_data_in_states
        p.get_global_parameter("A0").value = 1.0
        p.get_global_parameter("B").value = start_B
        p.solve()
        ev0, _ = p.solve_eigenproblem(neigen)
        res["start_real_part"] = float(numpy.real(ev0[0]))

        record = {"reloads": 0, "dparam_ds_after_reload": [],
                  "dparam_ds_at_handover": [], "dB_over_ds": []}
        originals = None
        if stay_stable:
            originals = _instrument_stay_stable(type(p), record)

        steps = 0
        last = None
        yielded = []
        try:
            for param, ev in p.find_bifurcation_via_eigenvalues(
                    "B", initstep, neigen=neigen, max_ds=0.08, epsilon=epsilon,
                    track_eigenvector=track,
                    stay_stable_file="stay_stable.dump" if stay_stable else None):
                steps += 1
                last = (float(param), complex(ev))
                yielded.append([float(param), float(numpy.real(ev))])
                if steps > max_steps:
                    res["outcome"] = "exhausted"
                    break
            else:
                res["outcome"] = "converged"
        except Exception as e:
            res["outcome"] = "raised"
            res["error"] = type(e).__name__ + ": " + str(e)[:300]
        if originals is not None:
            import pyoomph.generic.problem as _P
            _P.Problem.load_state, _P.Problem.arclength_continuation = originals  #type:ignore
        res["steps"] = steps
        res["yielded"] = yielded
        res["reloads"] = record["reloads"]
        res["dparam_ds_after_reload"] = record["dparam_ds_after_reload"]
        res["dparam_ds_at_handover"] = record["dparam_ds_at_handover"]
        res["dB_over_ds"] = record["dB_over_ds"]
        res["stay_stable_dump_exists"] = os.path.isfile(
            os.path.join(outdir, "stay_stable.dump")) if stay_stable else None
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
    ap.add_argument("--stay-stable", action="store_true")
    ap.add_argument("--continuation-data-in-states", action="store_true")
    args, _ = ap.parse_known_args()
    payload: "dict[str, Any]" = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
    try:
        payload.update(run(args.start_B, args.initstep, args.epsilon, args.neigen,
                           args.track, args.outdir, stay_stable=args.stay_stable,
                           continuation_data_in_states=args.continuation_data_in_states))
    except Exception as e:
        import traceback
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-3000:]
    # sys.__stdout__: the MPI console mutes stdout on every rank but 0.
    print("PYOOMPH_SCAN_RESULT " + json.dumps(payload), file=sys.__stdout__, flush=True)


if __name__ == "__main__":
    main()
