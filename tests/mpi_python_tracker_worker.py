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

# Worker for test_mpi_python_trackers.py: the PYTHON FoldTracker (pyoomph/generic/bifurcation_tools.py)
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

# The continuation parameter's name per case. The Brusselator's is B; the others use lam.
_PARAM = {"fold": "lam", "pitchfork": "lam", "hopf": "B",
          "eigenbranch_real": "lam", "eigenbranch_complex": "B"}
from pyoomph.generic.bifurcation_tools import (ComplexEigenbranchTracker, FoldTracker,
                                                HopfTracker, PitchForkTracker,
                                                RealEigenbranchTracker)


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


class ReactionDiffusionEquations(Equations):
    """laplace(u) + lam*u - u^3 = 0: the symmetry u -> -u of u=0 breaks in a pitchfork."""

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(self.lam * u - u ** 3, v))


class PitchforkProblem(Problem):
    """A 1 x 1.05 rectangle, NOT the unit square, for the reason mpi_bifurcation_worker.py documents:
    on the square the (1,2) and (2,1) Dirichlet modes are degenerate and an eigenvalue request can cut
    inside the degenerate pair, where which copy comes back is the Krylov solver's own business. The
    aspect ratio splits them and the bifurcation is unchanged -- still the symmetry breaking of u=0 in
    the first mode, at lam = pi^2*(1 + 1/1.05^2)."""

    ASPECT = 1.05

    def __init__(self, N=8):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1.0, self.ASPECT], name="domain")
        self.lam = self.define_global_parameter(lam=1.0)
        eqs = ReactionDiffusionEquations(self.lam)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        eqs += IntegralObservables(usqr=var("u") ** 2)
        self += eqs @ "domain"
        self.setup_for_stability_analysis(analytic_hessian=True)


class BrusselatorEquations(Equations):
    """The Brusselator, whose uniform state (u,v) = (A, B/A) loses stability at a Hopf: a pair of
    complex conjugate eigenvalues crosses the axis at B = 1 + A^2 plus a diffusive correction."""

    def __init__(self, A, B):
        super().__init__()
        self.A, self.B = A, B

    def define_fields(self):
        self.define_scalar_field("u", "C2")
        self.define_scalar_field("v", "C2")

    def define_residuals(self):
        u, ut = var_and_test("u")
        v, vt = var_and_test("v")
        self.add_residual(weak(partial_t(u), ut) + 0.02 * weak(grad(u), grad(ut))
                          - weak(self.A - (self.B + 1) * u + u ** 2 * v, ut))
        self.add_residual(weak(partial_t(v), vt) + 0.1 * weak(grad(v), grad(vt))
                          - weak(self.B * u - u ** 2 * v, vt))


class HopfProblem(Problem):
    """Brusselator on a line with no-flux ends, so the uniform state is an exact solution."""

    def __init__(self, N=20):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += LineMesh(N=self.N, size=1, name="domain")
        A = 1.0
        self.lam = self.define_global_parameter(B=2.5)   # named lam here so one code path fits all
        eqs = BrusselatorEquations(A, self.lam)
        eqs += InitialCondition(u=A, v=self.lam / A)
        eqs += IntegralObservables(usqr=var("u") ** 2 + var("v") ** 2)
        self += eqs @ "domain"
        self.setup_for_stability_analysis(analytic_hessian=True)


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
    ap.add_argument("--cxx", action="store_true", help="use the C++ handler instead, for comparison")
    ap.add_argument("--case", default="fold",
                    choices=["fold", "pitchfork", "hopf", "eigenbranch_real", "eigenbranch_complex"])
    args, _ = ap.parse_known_args()

    payload: dict = {"rank": get_mpi_rank(), "nproc": max(get_mpi_nproc(), 1),
                     "route": "cxx" if args.cxx else "python", "case": args.case}
    try:
        if args.case == "pitchfork":
            problem = PitchforkProblem(args.N)
        elif args.case in ("hopf", "eigenbranch_complex"):
            problem = HopfProblem(args.N if args.N != 8 else 20)
        else:
            problem = BratuProblem(args.N)
        with problem as p:
            p.set_output_directory(args.outdir)
            p.initialise()
            p.solve()
            payload["ndof_base"] = int(p.ndof())

            # A guess for the null vector: the least stable eigenvector of the base state. A Hopf
            # needs the complex pair, so more than one.
            # SIX for the Hopf, not four. At four the request truncated where the returned ORDER
            # stopped being the same in every regime: np=3 put a different mode in slot 0 and the
            # tracker converged to a second, equally real Hopf at B = 2.243 instead of 2.000, so the
            # test was comparing two different bifurcations. At six the spectrum comes back identical
            # at np=1, 2 and 3 -- +0.25 +- 0.968i first, then the real modes -- and slot 0 is
            # unambiguous. Same lesson as the pitchfork's aspect ratio: do not put the cut where the
            # answer is not well defined.
            p.solve_eigenproblem(6 if args.case in ("hopf", "eigenbranch_complex") else 1)
            evals = [complex(x) for x in p.get_last_eigenvalues()]
            guess = 0
            payload["eigenvalue_guess"] = evals[guess].real
            payload["omega_guess"] = evals[guess].imag

            if args.cxx:
                p.activate_bifurcation_tracking(_PARAM[args.case], args.case)
                p.solve()
                if args.case.startswith("eigenbranch"):
                    # No parameter is driven here: what the solve pins is the eigenvalue, so that is
                    # the number to compare across regimes.
                    lam0 = complex(p.get_last_eigenvalues()[0])
                    payload["critical"] = float(lam0.real)
                    payload["tracked_omega"] = float(lam0.imag)
                else:
                    payload["critical"] = float(p.lam.value)
                payload["ndof_aug"] = int(p.ndof())
                p.deactivate_bifurcation_tracking()
            else:
                if args.case == "pitchfork":
                    tracker = PitchForkTracker(p, _PARAM[args.case], eigenvector=guess,
                                               nonlinear_length_constraint=args.nonlinear_constraint)
                elif args.case == "hopf":
                    tracker = HopfTracker(p, _PARAM[args.case], eigenvector=guess,
                                          nonlinear_length_constraint=args.nonlinear_constraint)
                elif args.case == "eigenbranch_real":
                    # Follows an eigenvalue along the branch rather than pinning a bifurcation, so
                    # there is no parameter unknown: the eigenvalue itself is the extra scalar.
                    tracker = RealEigenbranchTracker(p, guess, nonlinear_length_constraint=args.nonlinear_constraint)
                elif args.case == "eigenbranch_complex":
                    tracker = ComplexEigenbranchTracker(p, guess, nonlinear_length_constraint=args.nonlinear_constraint)
                else:
                    tracker = FoldTracker(p, _PARAM[args.case], eigenvector=guess,
                                          nonlinear_length_constraint=args.nonlinear_constraint)
                payload["supports_mpi"] = bool(tracker.supports_mpi())
                p.set_custom_assembler(tracker)
                payload["ndof_aug"] = int(p.ndof())
                payload["groups"] = list(tracker.group_is_scalar)
                L = tracker.augmented_layout
                payload["aug_layout"] = [L.n, L.first_row, L.nrow_local, L.distributed]
                p.solve()
                if args.case.startswith("eigenbranch"):
                    # No parameter is driven here: what the solve pins is the eigenvalue, so that is
                    # the number to compare across regimes.
                    lam0 = complex(p.get_last_eigenvalues()[0])
                    payload["critical"] = float(lam0.real)
                    payload["tracked_omega"] = float(lam0.imag)
                else:
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
