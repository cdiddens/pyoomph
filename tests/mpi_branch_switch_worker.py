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

# Branch switching on a PDE, serially and under mpirun (with and without --distribute).
#
# The 1-dof ODEs of tests/branch_switch_worker.py cannot be partitioned and are the only closed-form
# check on the SIGNS and FACTORS of the normal-form coefficients, so they stay there, serial. What
# they cannot show is anything that scales with the number of degrees of freedom, and three of the
# defects this file was written for do exactly that:
#
#   * b3 came from a finite difference whose step was fd_eps*zeta with zeta unit-normalised in the
#     EUCLIDEAN dof norm, i.e. fd_eps/sqrt(N) per dof -- at or below the roundoff floor of the dofs
#     on any real mesh, and worse the finer the mesh.
#   * the pitchfork/transcritical discriminant compared |b2| against |b3| directly, and that ratio
#     grows like sqrt(N), so the verdict moved with the mesh.
#   * psi01 and wst came from a plain spsolve of a matrix that is EXACTLY singular at a branch point.
#
# Two problems, both derived from tests/mpi_bifurcation_worker.py's PitchforkProblem and keeping its
# 1 x 1.05 aspect ratio for the reason documented there (the square makes the (1,2)/(2,1) Dirichlet
# modes degenerate, and a truncation inside a degenerate pair has no well-defined answer):
#
#   pitchfork      u_t = laplace(u) + lam*u - u^3     odd nonlinearity  -> b2 == 0 exactly
#   transcritical  u_t = laplace(u) + lam*u - u^2     even nonlinearity -> b2 != 0
#
# Both keep u = 0 as an exact solution for EVERY lam, which is what makes this a real test: a switch
# that does not work does not fail loudly, it quietly stays on u = 0. Both cross at the first
# Dirichlet mode, lam_c = pi^2*(1 + 1/1.05^2).
#
# As in tests/mpi_bifurcation_worker.py, the dof numbering is NOT comparable between a serial and a
# distributed run -- distribute() renumbers -- so everything compared across runs is
# numbering-independent: the parameter, the coefficients (which are reductions), and mesh integrals.

import argparse
import json
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


LAM_C = numpy.pi ** 2 * (1.0 + 1.0 / 1.05 ** 2)


class BratuEquations(Equations):
    """laplace(u) + lam*exp(u) = 0 -- here only for its FOLD, and only for the coefficients.

    The two branch-point problems above sit on the trivial branch u = 0 with a POLYNOMIAL
    nonlinearity, and that makes them blind to one thing: the Hessian contraction is then exactly
    affine along the perturbation direction, so the central difference that supplies b3 is exact at
    every step length and b3 does not move when the step does. (Measured: changing the step by three
    decades left b3 identical to 15 digits on both.) Bratu has a NONZERO base state and an exp(),
    so its b3 has a genuine truncation-versus-cancellation optimum and a step sweep shows the V.

    A fold refuses to switch branches, which is fine -- get_normal_form1d computes every coefficient
    before it decides what kind of point it is at, so --phase coefficients gets what it needs.
    """

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
        self.lam = self.define_global_parameter(lam=4.0)

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1.0, 1.0])
        eqs = BratuEquations(self.lam)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        eqs += IntegralObservables(usqr=var("u") ** 2, uphi=var("u"))
        self += eqs @ "domain"


class BranchSwitchEquations(Equations):
    """u_t = laplace(u) + lam*u - u^p, p = 3 (pitchfork) or p = 2 (transcritical)."""

    def __init__(self, lam, power):
        super().__init__()
        self.lam = lam
        self.power = power

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v))
                          - weak(self.lam * u - u ** self.power, v))


class BranchSwitchProblem(Problem):
    ASPECT = 1.05

    def __init__(self, N=8, power=3):
        super().__init__()
        self.N = N
        self.power = power
        self.lam = self.define_global_parameter(lam=1)

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1.0, self.ASPECT])
        eqs = BranchSwitchEquations(self.lam, self.power)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        # usqr is the numbering-independent observable every cross-run comparison uses; uphi is the
        # SIGNED projection onto the first Dirichlet mode, which is what tells a pitchfork's two arms
        # apart (usqr cannot -- it is the same on both).
        phi = sin(pi * var("coordinate_x")) * sin(pi * var("coordinate_y") / BranchSwitchProblem.ASPECT)
        eqs += IntegralObservables(usqr=var("u") ** 2, uphi=var("u") * phi)
        self += eqs @ "domain"


def _observe(problem):
    obs = problem.get_mesh("domain").evaluate_all_observables()
    return {"lam": float(problem.lam.value),
            "usqr": float(obs["usqr"]), "uphi": float(obs["uphi"])}


def _to_bifurcation(problem, kind):
    """Solve the base branch, find the critical point and sit exactly on it."""
    problem.lam.value = 4.0 if kind == "bratu" else LAM_C - 1.0
    problem.solve()
    problem.solve_eigenproblem(1 if kind != "bratu" else 2)
    problem.activate_bifurcation_tracking("lam", "fold" if kind == "bratu" else None)
    problem.solve()
    return float(problem.lam.value)


def _analytic_amplitude(kind, dlam):
    """|<u,phi>|/<phi,phi> on the bifurcating branch, to leading order in dlam = lam - lam_c.

    With u = A*phi + O(A^2) and phi = sin(pi x) sin(pi y/L) normalised by nothing in particular, the
    Galerkin projection of laplace(u) + lam*u - u^p onto phi gives

        pitchfork (p=3):  A^2 = dlam * <phi,phi> / <phi^4>
        transcritical (p=2):  A   = dlam * <phi,phi> / <phi^3>

    The integrals are over the 1 x L rectangle and are done in closed form below. This is a LEADING
    ORDER statement, so the assertions that use it allow a few percent; its job is to say the landing
    is on the right branch, not to measure it.
    """
    L = BranchSwitchProblem.ASPECT
    # <phi,phi> = L/4 ; <phi^3> = (4/(3 pi))^2 * L ... computed numerically for clarity
    import numpy as _np
    nx = 400
    x = (_np.arange(nx) + 0.5) / nx
    y = (_np.arange(nx) + 0.5) / nx * L
    P = _np.sin(_np.pi * x)[:, None] * _np.sin(_np.pi * y / L)[None, :]
    dA = (1.0 / nx) * (L / nx)
    pp = float(_np.sum(P * P) * dA)
    if kind == "pitchfork":
        p4 = float(_np.sum(P ** 4) * dA)
        return numpy.sqrt(max(dlam, 0.0) * pp / p4)
    p3 = float(_np.sum(P ** 3) * dA)
    return dlam * pp / p3


def run(args):
    res = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "kind": args.kind, "N": args.N}
    if args.kind == "bratu":
        prob = BratuProblem(N=args.N)
    else:
        prob = BranchSwitchProblem(N=args.N, power=3 if args.kind == "pitchfork" else 2)
    with prob as problem:
        problem.set_output_directory(args.outdir)
        problem.quiet()
        # The same pair tests/mpi_bifurcation_worker.py uses, serially as well as under mpirun: a
        # distributed run needs a distributed-capable direct solver, and forcing the same choice in
        # the serial reference keeps the comparison about the partitioning and nothing else.
        problem.set_linear_solver("petsc_mumps")
        problem.set_eigensolver("slepc")
        problem.setup_for_stability_analysis(analytic_hessian=True)
        # --distribute is consumed by pyoomph itself from sys.argv; nothing to do here.
        problem.initialise()
        res["distributed"] = bool(problem.is_distributed())
        res["ndof"] = int(problem.ndof())

        lam_at = _to_bifurcation(problem, args.kind)
        res["lam_bif"] = lam_at

        nf = problem.classify_bifurcation("lam", fd_eps=args.fd_eps)
        res["type"] = nf.get("type")
        for k in ("a", "a_rel", "b1", "b2", "b3", "b2_rel", "norm_b2v",
                  "psi01_residual", "psi01_orth", "L_zeta", "LT_zeta_star"):
            if k in nf:
                res[k] = float(numpy.real(nf[k]))
        res["evect_len"] = int(len(problem.get_last_eigenvectors()[0]))

        if args.phase == "coefficients":
            return res

        # The dofs and the parameter, not a state file: returning to the bifurcation is all that is
        # needed here -- switch_branch wants the solution at the point plus the normal form, and
        # deactivates tracking itself -- and get_current_dofs/set_current_dofs gather and scatter by
        # global equation number, so they are right in both MPI regimes and cost nothing.
        #
        # This used to use save_state/load_state into a BytesIO, which HUNG under --distribute: the
        # merged stream reached rank 0 alone and loading the empty one back split the ranks. That is
        # fixed (save_state broadcasts it now, see dev_docs/distributed_state_files.md), but the dof
        # capture is the smaller thing to depend on for a test about branch switching.
        base_dofs = numpy.array(problem.get_current_dofs()[0])
        base_lam = float(problem.lam.value)

        landed = []
        for direction in ((1, -1) if args.kind == "pitchfork" else (1,)):
            problem.lam.value = base_lam
            problem.set_current_dofs(base_dofs)
            ds = problem.switch_branch("lam", normal_form=nf, direction=direction, quiet=True)
            if ds is None:
                landed.append(None)
                continue
            steps = []
            for _ in range(4):
                steps.append(_observe(problem))
                ds = problem.arclength_continuation("lam", ds)
            steps.append(_observe(problem))
            landed.append({"ds": float(ds), "steps": steps})
        res["landed"] = landed
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["pitchfork", "transcritical", "bratu"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--phase", default="full", choices=["full", "coefficients"])
    ap.add_argument("--fd-eps", dest="fd_eps", type=float, default=None,
                    help="relative FD step for b3; sweeping it is how the default was chosen")
    args, _rest = ap.parse_known_args()
    try:
        res = run(args)
        res["ok"] = True
    except BaseException as e:  # noqa: BLE001 - the traceback has to reach the launcher
        res = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "ok": False,
               "error": type(e).__name__ + ": " + str(e), "traceback": traceback.format_exc()}
    print("PYOOMPH_BRANCH_SWITCH_RESULT " + json.dumps(res), flush=True)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
