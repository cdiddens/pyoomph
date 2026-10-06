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

# Worker for test_mpi_python_fold.py: the PYTHON FoldTracker (pyoomph/generic/bifurcation_tools.py)
# tracking a fold under MPI. The first of the custom-assembler family to run there at all.
#
# Same problem as tests/mpi_bifurcation_worker.py uses for the C++ MyFoldHandler -- 2D Bratu, whose
# branch turns at a genuine limit point -- so the two routes can be compared against each other as
# well as against serial. That is the strongest check available: two independent implementations of
# the same augmented system, one of which has been correct under --distribute for a while.
#
# The critical parameter alone is a weak certificate: it is one number read off a converged system
# that every rank agrees about by construction. "eigfunc_usqr", the mesh integral of the squared
# tracked eigenvector, is what pins down WHERE on the mesh the eigenvector's entries ended up, which
# is what a wrong equation-table translation or a border in the wrong column would move while
# leaving the parameter alone. See the same argument in tests/test_mpi_bifurcation_tracking.py.

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc
from pyoomph.generic.bifurcation_tools import FoldTracker


class BratuEquations(Equations):
    """laplace(u) + lam*exp(u) = 0. The time derivative only gives the eigenproblem a mass matrix;
    the fold is a property of the steady residual."""

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(self.lam * exp(u), v))


class BratuProblem(Problem):
    def __init__(self, N=8):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1, 1], name="domain")
        self.lam = self.define_global_parameter(lam=4.0)
        eqs = BratuEquations(self.lam)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        # Partition-independent: evaluate_integral_function skips halo elements and MPI_Allreduce-sums.
        eqs += IntegralObservables(usqr=var("u") ** 2)
        self += eqs @ "domain"
        self.setup_for_stability_analysis(analytic_hessian=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--distribute", action="store_true")
    ap.add_argument("--nonlinear-constraint", action="store_true")
    ap.add_argument("--cxx", action="store_true", help="use the C++ MyFoldHandler instead, for comparison")
    args, _ = ap.parse_known_args()

    payload: dict = {"rank": get_mpi_rank(), "nproc": max(get_mpi_nproc(), 1),
                     "route": "cxx" if args.cxx else "python"}
    try:
        with BratuProblem(args.N) as p:
            p.set_output_directory(args.outdir)
            p.initialise()
            p.solve()
            payload["ndof_base"] = int(p.ndof())

            # A guess for the null vector: the least stable eigenvector of the base state.
            p.solve_eigenproblem(1)
            payload["eigenvalue_guess"] = complex(p.get_last_eigenvalues()[0]).real

            if args.cxx:
                p.activate_bifurcation_tracking("lam", "fold")
                p.solve()
                payload["critical"] = float(p.lam.value)
                payload["ndof_aug"] = int(p.ndof())
                p.deactivate_bifurcation_tracking()
            else:
                tracker = FoldTracker(p, "lam", eigenvector=0,
                                      nonlinear_length_constraint=args.nonlinear_constraint)
                payload["supports_mpi"] = bool(tracker.supports_mpi())
                p.set_custom_assembler(tracker)
                payload["ndof_aug"] = int(p.ndof())
                payload["groups"] = list(tracker.group_is_scalar)
                L = tracker.augmented_layout
                payload["aug_layout"] = [L.n, L.first_row, L.nrow_local, L.distributed]
                p.solve()
                payload["critical"] = float(p.lam.value)
                # The residual history, so a test can judge the RATE and not only the answer: a stale
                # rank-0-only scalar still converges, just linearly.
                # Read AFTER the solve: get_last_residual_convergence() is the history oomph kept for
                # it, complete including the final step, which a per-step hook misses because oomph
                # leaves the loop as soon as it has converged.
                payload["newton_residuals"] = [float(x) for x in p.get_last_residual_convergence()]
                p.set_custom_assembler(None)

            # Where the tracked eigenvector landed on the mesh. Numbering-independent, unlike the
            # eigenvector itself: it goes back through set_current_dofs(), which scatters by GLOBAL
            # equation number, and is integrated over non-halo elements with an allreduce.
            evec = numpy.real(numpy.asarray(p.get_last_eigenvectors()[0]))
            payload["evect_len"] = int(len(evec))
            nrm = float(numpy.linalg.norm(evec))
            if nrm > 0:
                evec = evec / nrm
            old = p.get_current_dofs()[0]
            p.set_current_dofs(evec)
            payload["eigfunc_usqr"] = float(p.get_mesh("domain").evaluate_all_observables()["usqr"])
            p.set_current_dofs(old)
            payload["ndof_after"] = int(p.ndof())
        payload["ok"] = True
    except Exception as e:
        payload["ok"] = False
        payload["error"] = repr(e)
        payload["traceback"] = traceback.format_exc()[-2500:]
    out = sys.__stdout__ or sys.stdout
    out.write("PYOOMPH_MPI_RESULT " + json.dumps(payload) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
