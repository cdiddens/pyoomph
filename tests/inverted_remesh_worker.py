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

"""Worker for tests/test_inverted_element_remesh.py. Runs in its own process because
``set_detect_inverted_elements`` is a PROCESS-WIDE switch: a test that armed it would otherwise
leave every later test in the same process paying for the check and, worse, subject to it.

Launched plain, under ``mpirun``, and under ``mpirun ... --distribute``. Prints one
PYOOMPH_RESULT line per rank.

The problem is the notch of dev_docs/examples/inverted_element_notch.py, cut down so a case costs a
few seconds: a unit square, a Laplace-smoothed mesh, and a Gaussian notch of linearly growing depth
pushed into the top edge. The *domain* stays perfectly meshable throughout - it is the harmonic
extension into that non-convex shape that folds, at t = 0.1565, which is exactly the situation a
remesh repairs and a smaller time step cannot.
"""

import argparse
import json
import sys
import traceback

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.generic import RemeshWhen, RemeshingOptions
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class DiffusingScalar(Equations):
    """The only non-geometric unknown: without one there is no temporal error norm to drive the
    adaptive time stepping with, and nothing for a remesh to interpolate."""

    def define_fields(self):
        self.define_scalar_field("c", "C2")

    def define_residuals(self):
        c, ctest = var_and_test("c")
        self.add_residual(weak(partial_t(c, ALE="auto"), ctest) + weak(0.02 * grad(c), grad(ctest)))


class NotchProblem(Problem):
    def __init__(self, on_inverted=False, N=8):
        super().__init__()
        self.on_inverted = on_inverted
        self.N = N

    def define_problem(self):
        mesh = RectangularQuadMesh(N=self.N)
        mesh.remesher = Remesher2d(mesh)
        self.add_mesh(mesh)
        eqs = LaplaceSmoothedMesh()
        eqs += DiffusingScalar()
        xi = var("lagrangian")
        t = var("time")
        eqs += DirichletBC(mesh_x=0, mesh_y=True) @ "left"
        eqs += DirichletBC(mesh_x=1, mesh_y=True) @ "right"
        eqs += DirichletBC(mesh_y=0, mesh_x=True) @ "bottom"
        eqs += DirichletBC(mesh_y=1 - 0.85 * t * exp(-((xi[0] - 0.5) / 0.12) ** 2), mesh_x=True) @ "top"
        eqs += InitialCondition(c=exp(-((xi[0] - 0.5) ** 2 + (xi[1] - 0.5) ** 2) / 0.05))
        eqs += IntegralObservables(csqr=var("c") ** 2)
        # The quality thresholds are deliberately OFF, so the inverted-element trigger is the only
        # thing that can ask for a remesh. With them on, the quality criterion remeshes preventively
        # and the run never folds at all - which is the better way to use the framework, and exactly
        # why it would make this test prove nothing.
        eqs += RemeshWhen(RemeshingOptions(max_expansion=None, min_expansion=None,
                                           min_quality_decrease=None,
                                           on_inverted_element=self.on_inverted))
        self.add_equations(eqs @ "domain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--endtime", type=float, default=0.30)
    ap.add_argument("--on-inverted", action="store_true")
    # Detection without the remesh trigger: the control arm. NOT the same as passing nothing at all -
    # with detection off the run sails straight past the fold and finishes, because pyoomph's J is
    # sqrt(det(g_ab)), non-negative by construction, so an inside-out element integrates happily and
    # nothing notices. That is the pre-detector behaviour and the reason the detector exists.
    ap.add_argument("--detect-only", action="store_true")
    args, _rest = ap.parse_known_args()

    if args.on_inverted or args.detect_only:
        from pyoomph._pyoomph_core import set_detect_inverted_elements
        set_detect_inverted_elements(True)

    res = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
    try:
        with NotchProblem(on_inverted=args.on_inverted) as p:
            p.set_output_directory(args.outdir)
            p.quiet()
            remeshes = [0]
            orig = p.force_remesh

            def counting_remesh(*a, **k):
                remeshes[0] += 1
                return orig(*a, **k)
            p.force_remesh = counting_remesh

            # _get_inversion_reports() is a LIVE counter, cleared by every clean step and by every
            # remesh, so reading it at the end reports zero on a run that inverted hundreds of times.
            # Accumulate what it held just before each reset instead.
            total_reports = [0]
            orig_reset = p._reset_inversion_counter

            def accumulating_reset(*a, **k):
                total_reports[0] += int(p._get_inversion_reports())
                return orig_reset(*a, **k)
            p._reset_inversion_counter = accumulating_reset
            p.initialise()
            res["distributed"] = bool(p.is_distributed())
            t, dt = 0.0, 0.02
            try:
                while t < args.endtime - 1e-9:
                    dt = float(p.solve(timestep=dt, temporal_error=1e-3))
                    dt = min(dt, 0.05)
                    t = float(p.get_current_time(as_float=True, dimensional=False))
            except Exception as e:
                res["failed"] = type(e).__name__
            res["t"] = t
            res["remeshes"] = remeshes[0]
            res["inversion_reports"] = total_reports[0] + int(p._get_inversion_reports())
            # Partition-independent (evaluate_all_observables skips halos and Allreduce-sums), so
            # serial, replicated and distributed can be compared on it directly.
            res["csqr"] = float(p.get_mesh("domain").evaluate_all_observables()["csqr"])
    except Exception:
        res["error"] = traceback.format_exc()[-1500:]
    print("PYOOMPH_RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
