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

# Worker for tests/test_mpi_bifurcation_tracking.py -- launched under `mpirun ...`, with and
# without --distribute. Solves a bifurcation-tracking problem through the augmented assembly
# handlers of src/bifurcation.cpp and prints one PYOOMPH_MPI_RESULT line per rank.
#
# The critical parameter alone is a decent but incomplete certificate: it is one number that every
# rank reads off the same converged augmented system. What pins down that the EIGENVECTOR block was
# assembled and updated correctly is "eigfunc_usqr", the integral of the squared (normalised)
# eigenfunction over the mesh. Getting there runs the tracked eigenvector back through
# set_current_dofs(), which scatters BY GLOBAL EQUATION NUMBER, and integrates over non-halo
# elements with an MPI_Allreduce -- so an eigenvector whose entries landed in the wrong global slots
# (a broken eqn_number translation, a missing halo synchronise) moves it, while leaving the critical
# parameter alone.
#
# Note the dof numbering is NOT comparable between a serial and a distributed run -- distribute()
# renumbers so that each rank owns a contiguous block -- so every quantity compared across the two
# has to be numbering-independent. The critical parameter, omega and mesh integrals are; a dof
# vector is not.

import argparse
import json
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class BratuEquations(Equations):
    """laplace(u) + lam*exp(u) = 0, the textbook fold: the branch turns at lam = lam_crit.

    Both solution branches exist below lam_crit and none above it, so the fold is a genuine limit
    point with a single real null eigenvector -- what MyFoldHandler is written for.
    """

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        # The time derivative is what gives the eigenproblem a non-empty mass matrix; the fold
        # itself is a property of the steady residual and does not depend on it.
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(self.lam * exp(u), v))


class BratuProblem(Problem):
    """2D Bratu on a unit square with u=0 all around."""

    def __init__(self, N=8):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N)
        self.lam = self.define_global_parameter(lam=4.0)
        eqs = BratuEquations(self.lam)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        # Partition-independent: evaluate_integral_function skips halo elements and MPI_Allreduce-sums.
        eqs += IntegralObservables(usqr=var("u") ** 2)
        self += eqs @ "domain"


class BrusselatorEquations(Equations):
    """The Brusselator reaction-diffusion system, whose uniform state loses stability at a Hopf.

    The homogeneous steady state is (u,v) = (A, B/A) and it goes unstable to an oscillation at
    B = 1 + A^2 (plus a diffusive correction on a finite domain), i.e. a pair of complex conjugate
    eigenvalues crosses the axis -- what MyHopfHandler is written for.
    """

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


class BrusselatorProblem(Problem):
    """Brusselator on a line with no-flux ends, so the uniform state is an exact solution."""

    def __init__(self, N=20):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += LineMesh(N=self.N, size=1)
        A = 1.0
        self.B = self.define_global_parameter(B=2.5)
        eqs = BrusselatorEquations(A, self.B)
        eqs += InitialCondition(u=A, v=self.B / A)
        eqs += IntegralObservables(usqr=var("u") ** 2 + var("v") ** 2)
        self += eqs @ "domain"


def _eigenvector_norms(p, index=0):
    """Scale certificates of the TRACKED eigenvector, invariant under the dof permutation.

    After a tracking solve pyoomph replaces the last eigenvectors with the handler's own (see
    Problem.solve), so this is the augmented system's eigenvector block, gathered to full global
    length on every rank. The 2-norm and max-abs say nothing about WHERE on the mesh the entries
    landed -- eigfunc_usqr below does that -- but they do pin the SCALE that the normalisation
    constraint c.y = 1 is supposed to fix, which is the quantity a wrong Count or a wrong element
    count would move.
    """
    ev = numpy.array(p.get_last_eigenvectors()[index])
    return {"evect_len": int(len(ev)),
            "evect_norm": float(numpy.linalg.norm(ev)),
            "evect_absmax": float(numpy.amax(numpy.absolute(ev)))}


def _eigenfunction_observables(p, index=0):
    """Integral observables of the TRACKED eigenfunction, with the dofs restored afterwards.

    mode="abs", i.e. the entrywise |v|, because the tracked eigenvector is only determined up to a
    sign (fold) or a complex phase (Hopf): the normalisation constraint c.y = 1 fixes the scale but
    serial and distributed runs may still land on different phases. |v| is invariant under both, so
    what is left is the shape on the mesh, which is what we want to certify.
    """
    backup_dofs, backup_pinned = p.set_eigenfunction_as_dofs(index, mode="abs")
    try:
        obs = p.get_mesh("domain").evaluate_all_observables()
        return {("eigfunc_" + k): float(v) for k, v in obs.items()}
    finally:
        p.set_all_values_at_current_time(backup_dofs, backup_pinned, False)


def fold_case(N=8, lam0=4.0, outdir=None, eigenvector_scaling="unit", with_guess=True):
    """Locate the Bratu fold in lam by eigen-solve + fold tracking."""
    prob = BratuProblem(N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.set_eigensolver("slepc")
        p.initialise()
        p.lam.value = lam0
        p.solve()
        if with_guess:
            # The guess for the fold's null eigenvector: the eigenvalue closest to zero from above.
            p.solve_eigenproblem(2, quiet=True)
        # else: no eigenvector at all, so MyFoldHandler's no-guess constructor derives one itself by
        # solving against d(residual)/d(lam). That is a different code path, and the one the tutorial
        # kuramoto_sivanshinsky_bifurcation.py takes.
        p.activate_bifurcation_tracking("lam", "fold", eigenvector_scaling=eigenvector_scaling)
        p.solve()

        res = {
            "ndof": int(p.ndof()),
            "distributed": bool(p.is_distributed()),
            "param": float(p.lam.value),
            # The residual of the CONVERGED augmented system: a tracking solve that "converged" onto
            # a system assembled inconsistently across ranks shows up here and nowhere else.
            "max_residual": float(numpy.amax(numpy.absolute(p.get_residuals()))),
        }
        res.update(_eigenvector_norms(p))
        p.deactivate_bifurcation_tracking()
        # Deactivation must put the problem back exactly as it was -- under --distribute that means
        # the LOCAL dof count, since the base distribution is a partitioned one.
        n_global, nrow_local, first_row, distributed = p._get_dof_distribution_info()
        res["ndof_after_deactivate"] = int(p.ndof())
        res["nrow_local_after_deactivate"] = int(nrow_local)
        # AFTER deactivating, so that the eigenvector is written into the plain (non-augmented) dof
        # vector: while tracking is active the dofs are the augmented ones, whose length and layout
        # differ between serial and distributed, and padding a base-length eigenvector into them
        # measures the padding rather than the eigenfunction.
        res.update(_eigenfunction_observables(p, 0))
        return res


def hopf_case(N=20, B0=2.5, outdir=None):
    """Locate the Brusselator Hopf in B by eigen-solve + Hopf tracking."""
    prob = BrusselatorProblem(N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.set_eigensolver("slepc")
        p.initialise()
        p.B.value = B0
        p.solve()
        p.solve_eigenproblem(4, quiet=True)
        p.activate_bifurcation_tracking("B", "hopf")
        p.solve()

        res = {
            "ndof": int(p.ndof()),
            "distributed": bool(p.is_distributed()),
            "param": float(p.B.value),
            # The Hopf frequency lives on rank 0's dof vector alone when distributed, so this also
            # certifies that synchronise() broadcast it back to the other ranks.
            "omega": float(p._get_bifurcation_omega()),
            "max_residual": float(numpy.amax(numpy.absolute(p.get_residuals()))),
        }
        res.update(_eigenvector_norms(p))
        p.deactivate_bifurcation_tracking()
        n_global, nrow_local, first_row, distributed = p._get_dof_distribution_info()
        res["ndof_after_deactivate"] = int(p.ndof())
        res["nrow_local_after_deactivate"] = int(nrow_local)
        # After deactivating, for the reason given in fold_case().
        res.update(_eigenfunction_observables(p, 0))
        return res


class ReactionDiffusionEquations(Equations):
    """u_t = laplace(u) + lam*u - u^3.

    The trivial state u=0 is a solution for every lam and loses stability where lam reaches an
    eigenvalue of -laplace; because the nonlinearity is odd, that crossing is a PITCHFORK. Both the
    pitchfork and the azimuthal cases below are built on this, since it is the cheapest system with
    a genuine symmetry-breaking bifurcation whose eigenproblem SLEPc can actually factorise here.
    """

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v))
                          - weak(self.lam * u - u ** 3, v))


class PitchforkProblem(Problem):
    """Reaction-diffusion on a unit square with u=0 all around: pitchfork at lam = 2*pi^2."""

    def __init__(self, N=8):
        super().__init__()
        self.N = N
        self.lam = self.define_global_parameter(lam=1)

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N)
        eqs = ReactionDiffusionEquations(self.lam)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        eqs += IntegralObservables(usqr=var("u") ** 2)
        self += eqs @ "domain"


class AzimuthalReactionProblem(Problem):
    """The same reaction-diffusion, axisymmetric, tracked in the m=1 azimuthal mode.

    Exercises AzimuthalSymmetryBreakingHandler: the m != 0 residual forms, the axis dofs that the
    handler forces to zero (base_dofs_forced_zero / eigen_dofs_forced_zero), and -- since a scalar
    field carries no imaginary azimuthal contribution -- its has_imaginary_part == false layout,
    [u | Re(v) | param], where the parameter is the only scalar unknown.
    """

    def __init__(self, N=8):
        super().__init__()
        self.N = N
        self.lam = self.define_global_parameter(lam=1)

    def define_problem(self):
        self.set_coordinate_system(axisymmetric)
        self += RectangularQuadMesh(N=self.N)
        eqs = ReactionDiffusionEquations(self.lam)
        eqs += DirichletBC(u=0) @ ["right", "top", "bottom"]
        eqs += IntegralObservables(usqr=var("u") ** 2)
        self += eqs @ "domain"


def _reaction_case(prob, bifurcation_type, lam_start, lam_step, azimuthal_m, outdir):
    """Shared body of the pitchfork and azimuthal reaction-diffusion cases."""
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.set_eigensolver("slepc")
        if azimuthal_m is not None:
            p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=True)
        p.initialise()
        p.lam.value = lam_start
        p.solve()
        # Walk up to the onset for a guess. do_solve=False: the trivial state u=0 solves the system
        # for every lam, so there is nothing to re-solve between steps.
        for _l, _ev in p.find_bifurcation_via_eigenvalues("lam", initstep=lam_step, do_solve=False,
                                                          neigen=3, azimuthal_m=azimuthal_m,
                                                          epsilon=1e-2):
            pass
        p.activate_bifurcation_tracking("lam", bifurcation_type=bifurcation_type)
        p.solve()

        res = {
            "ndof": int(p.ndof()),
            "distributed": bool(p.is_distributed()),
            "param": float(p.lam.value),
            "omega": float(p._get_bifurcation_omega()),
            "max_residual": float(numpy.amax(numpy.absolute(p.get_residuals()))),
        }
        res.update(_eigenvector_norms(p))
        p.deactivate_bifurcation_tracking()
        n_global, nrow_local, first_row, distributed = p._get_dof_distribution_info()
        res["ndof_after_deactivate"] = int(p.ndof())
        res["nrow_local_after_deactivate"] = int(nrow_local)
        # After deactivating, for the reason given in fold_case().
        res.update(_eigenfunction_observables(p, 0))
        return res


def pitchfork_case(N=8, outdir=None):
    """Symmetry-breaking onset of u=0 in the reaction-diffusion square: MyPitchForkHandler."""
    return _reaction_case(PitchforkProblem(N=N), "pitchfork", 10.0, 4.0, None, outdir)


def azimuthal_case(N=8, outdir=None):
    """m=1 onset of u=0 in the axisymmetric reaction-diffusion: AzimuthalSymmetryBreakingHandler."""
    return _reaction_case(AzimuthalReactionProblem(N=N), "azimuthal", 10.0, 4.0, 1, outdir)


def fold_noguess_case(N=8, lam0=4.0, outdir=None):
    """The same fold, but found without ever solving an eigenproblem: see fold_case(with_guess)."""
    return fold_case(N=N, lam0=lam0, outdir=outdir, with_guess=False)


_CASES = {"fold": fold_case, "hopf": hopf_case, "azimuthal": azimuthal_case,
          "pitchfork": pitchfork_case, "fold_noguess": fold_noguess_case}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--case", default="fold", choices=sorted(_CASES))
    ap.add_argument("--eigenvector-scaling", default="unit", choices=["unit", "auto"])
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(),
               "case": "%s_N%d" % (args.case, args.size)}
    try:
        kwargs = {}
        if args.eigenvector_scaling != "unit":
            # Only the fold case takes it; the others would reject an unexpected keyword.
            kwargs["eigenvector_scaling"] = args.eigenvector_scaling
        payload.update(_CASES[args.case](N=args.size, outdir=args.outdir, **kwargs))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
