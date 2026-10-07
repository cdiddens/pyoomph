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

# Worker for tests/test_adapt_while_tracking.py -- does ADAPTING the mesh while a C++ bifurcation
# tracker is installed keep the bifurcation tracked? Runs serially and under `mpirun`, with and
# without --distribute, and prints one PYOOMPH_MPI_RESULT line per rank.
#
# Bratu with a second parameter, u'' + lam*exp(u) - b*u = 0, on a 1D mesh: the fold in lam exists for
# every b, so tracking lam and continuing in b traces a locus, and refining the mesh changes the base
# ndof without changing the physics.
#
# The four cases, and what each one is about:
#
#   adapt         bare Problem.adapt() with the tracker installed. Was broken SERIALLY: the
#                 adaptation deactivates the tracker and records a pending reactivation, nothing
#                 called it back (actions_after_newton_solve is where it used to live, and a bare
#                 adapt() does no Newton solve), and the next solve() then CLEARED the pending
#                 request -- so the tracker was dropped without a word and the next solve was an
#                 ordinary Newton solve sitting on a singular Jacobian.
#
#   solve_adapt   solve(spatial_adapt=N) with the tracker installed, i.e. the designed path and the
#                 one the arclength refusal's message recommends as the workaround. Was broken the
#                 same way and for a sharper reason: the Newton solve that oomph runs right after the
#                 adaptation, from inside steady_newton_solve(max_adapt), is the one that would have
#                 triggered the reactivation -- and it is a plain untracked solve AT the fold. It
#                 converges linearly at the 1/4 residual ratio Newton gives at a quadratic fold and
#                 dies on the iteration cap, so the reactivation behind it never ran.
#
#   arclength     tracked arclength continuation in the SECOND parameter with spatial_adapt=0, i.e.
#                 an ordinary fold locus. This always worked; it is here as the control that says a
#                 failure in the other cases is about the adaptation and not about tracked
#                 continuation in general.
#
#   arclength_adapt   the same with spatial_adapt>0, which is REFUSED. The refusal is the assertion:
#                 adapting inside an arclength step changes ndof halfway through it, so the arclength
#                 constraint the step solves against stops meaning anything, and oomph rejects the
#                 step and halves Ds for ever instead of failing. Measured: 40+ rejections down to
#                 Ds = 1e-13.
#
# What is reported, and why it is comparable across partitions: the tracked critical parameter and
# the base ndof are global, so every rank must agree and serial, replicated and --distribute must
# all land on the same fold.

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import Problem, Equations, InitialCondition, DirichletBC
from pyoomph.expressions import var_and_test, grad, exp, partial_t
from pyoomph.equations.generic import SpatialErrorEstimator
from pyoomph.meshes.simplemeshes import LineMesh
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class Bratu(Equations):
    def __init__(self, lam, b):
        super().__init__()
        self.lam, self.b = lam, b

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        # The time derivative is only there so a mass matrix exists for the eigensolve.
        self.add_weak(partial_t(u), v)
        self.add_weak(grad(u), grad(v))
        self.add_weak(-self.lam * exp(u) + self.b * u, v)


class BratuProblem(Problem):
    def __init__(self, N=20):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.add_mesh(LineMesh(N=self.N))
        eqs = Bratu(self.get_global_parameter("lam"), self.get_global_parameter("b"))
        eqs += InitialCondition(u=0)
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        eqs += SpatialErrorEstimator(u=1)
        self += eqs @ "domain"


def _find_fold(problem):
    """Walk up the branch in lam until the leading eigenvalue changes sign, then track the fold."""
    lam = problem.get_global_parameter("lam")
    lam.value = 1.0
    problem.solve()
    ds = 0.1
    last = None
    for _ in range(40):
        ds = problem.arclength_continuation("lam", ds)
        problem.solve_eigenproblem(2)
        ev = problem.get_last_eigenvalues()
        if last is not None and numpy.real(ev[0]) * numpy.real(last) < 0:
            break
        last = ev[0]
    else:
        raise RuntimeError("no sign change in the leading eigenvalue: the fold was not bracketed")
    problem.activate_bifurcation_tracking("lam", "fold")
    problem.solve()
    return float(lam.value)


def run_case(case, outdir, N=20, max_refinement_level=3, adapt_levels=1):
    res = {}
    with BratuProblem(N=N) as problem:
        problem.set_output_directory(outdir)
        # The augmented fold system has a structurally zero diagonal entry, which a plain PETSc LU
        # refuses ("Matrix is missing diagonal entry"). MUMPS pivots and is also the distributed
        # direct solver, so one solver covers every regime this worker runs in.
        problem.set_linear_solver("petsc_mumps")
        problem.max_refinement_level = max_refinement_level
        # The defaults leave this smooth solution alone, and an adapt that changes nothing proves
        # nothing at all.
        problem.max_permitted_error = 1e-7
        problem.min_permitted_error = 1e-9
        problem.set_arc_length_parameter(scale_arc_length=False)
        problem.quiet()
        problem.get_global_parameter("b").value = 0.0

        res["lam_c_coarse"] = _find_fold(problem)
        # ndof() is the AUGMENTED count while tracking (base + eigenvector + the parameter), so the
        # base count is what has to be reported for the mesh to be comparable.
        res["ndof_tracked_coarse"] = problem.ndof()
        res["mode_after_tracking"] = problem.get_bifurcation_tracking_mode()

        if case == "adapt":
            nref, nunref = problem.adapt()
            # The state RIGHT HERE is what the defect was about: the tracker has to be back on
            # before anything else happens, because the next thing anything does is a Newton solve
            # and that solve is sitting on the fold.
            res["mode_right_after_adapt"] = problem.get_bifurcation_tracking_mode()
            res["pending_right_after_adapt"] = \
                problem._bifurcation_reactivation_after_adaptation is not None
            res["nrefined"] = int(nref)
            res["nunrefined"] = int(nunref)
            problem.solve()
            res["ndof_tracked_fine"] = problem.ndof()
            res["lam_c_fine"] = float(problem.get_global_parameter("lam").value)
            res["mode_at_end"] = problem.get_bifurcation_tracking_mode()
        elif case == "solve_adapt":
            # adapt_levels > 1 exercises oomph's multi-level loop, where the reactivation now happens
            # once PER LEVEL and the augmented ndof therefore changes inside that loop.
            problem.solve(spatial_adapt=adapt_levels)
            res["ndof_tracked_fine"] = problem.ndof()
            res["lam_c_fine"] = float(problem.get_global_parameter("lam").value)
            res["mode_at_end"] = problem.get_bifurcation_tracking_mode()
        elif case == "arclength":
            ds = 0.05
            locus = []
            for _ in range(3):
                ds = problem.arclength_continuation("b", ds, spatial_adapt=0)
                locus.append([float(problem.get_global_parameter("b").value),
                              float(problem.get_global_parameter("lam").value)])
            res["locus"] = locus
            res["ndof_tracked_fine"] = problem.ndof()
            res["mode_at_end"] = problem.get_bifurcation_tracking_mode()
        elif case == "arclength_adapt":
            # Expected to be refused, by message. Caught here rather than in the driver so that the
            # refusal is observed on EVERY rank: a guard that fires on one rank only would leave the
            # others in the next collective.
            try:
                problem.arclength_continuation("b", 0.05, spatial_adapt=1)
            except RuntimeError as e:
                res["refusal"] = str(e)
            else:
                res["refusal"] = None
            res["mode_at_end"] = problem.get_bifurcation_tracking_mode()
        else:
            raise ValueError("unknown case " + str(case))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    choices=["adapt", "solve_adapt", "arclength", "arclength_adapt"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--adapt-levels", type=int, default=1)
    args, _ = ap.parse_known_args()
    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "case": args.case,
               "adapt_levels": args.adapt_levels}
    try:
        payload.update(run_case(args.case, args.outdir, N=args.size,
                                adapt_levels=args.adapt_levels))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-3000:]
    # sys.__stdout__: pyoomph's MPI console mutes stdout on every rank but 0 (mode "condensed"), so
    # a per-rank comparison would otherwise compare rank 0 with itself.
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), file=sys.__stdout__, flush=True)


if __name__ == "__main__":
    main()
