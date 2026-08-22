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


def _converged_max_residual(p):
    """max|R| of the augmented system, measured the way the tracking solve converged it.

    Problem.solve() is a STATIONARY solve: it makes every timestepper steady, converges, and restores
    them to unsteady before returning. So a get_residuals() call afterwards assembles the UNSTEADY
    Jacobian - including the BDF d/dt term - while the tracker converged the steady one, and the
    eigenvector rows J*Y come out at the size of that d/dt term rather than at the Newton tolerance.
    On the pitchfork case that is 8.3e-4 against Newton's own 3.5e-11: same dofs to the last bit, same
    parameter, only 225 of 452 rows differ and they are exactly the J*Y ones.

    Making the timesteppers steady for the measurement is what makes this number mean what the
    caller's comment says it means.
    """
    n = p.ntime_stepper()
    was_steady = [p.time_stepper_pt(i).is_steady() for i in range(n)]
    for i in range(n):
        p.time_stepper_pt(i).make_steady()
    try:
        return float(numpy.amax(numpy.absolute(p.get_residuals())))
    finally:
        for i in range(n):
            if not was_steady[i]:
                p.time_stepper_pt(i).undo_make_steady()


def _sorted_spectrum(evals):
    """Eigenvalues in a comparable order. Numbering-independent, so it survives distribute()."""
    ev = numpy.array(sorted(evals, key=lambda z: (round(numpy.real(z), 9), round(numpy.imag(z), 9))))
    return [float(numpy.real(z)) for z in ev], [float(abs(numpy.imag(z))) for z in ev]


def _eigen_during_tracking(p, neigen, shift, azimuthal_m=None):
    """The BASE state's eigenproblem, solved once with the tracker installed and once without it.

    This is the whole point of the feature: while a locus is being followed, the base state's
    remaining spectrum is what tells you a codim-2 point is coming. The A/B against the same state
    with the tracker removed is the assertion that matters -- the elemental assembly is the same
    either way (oomph's get_eigenproblem_matrices installs its own EigenProblemHandler), so anything
    that differs can only have come from the row layout, which is what
    Problem::BaseDofDistributionScope puts back.

    A NON-ZERO shift is mandatory here and pyoomph refuses a zero one: the tracker has converged the
    base state onto the bifurcation, so lambda = 0 (fold/pitchfork) or +-i*omega (Hopf/azimuthal) is
    an exact eigenvalue and that is exactly where shift-invert would factorise.
    """
    kw = {} if azimuthal_m is None else {"azimuthal_m": azimuthal_m}
    # solve_eigenproblem overwrites the last eigenvalues/vectors, and everything below this in the
    # worker still wants the TRACKED ones (_eigenvector_norms, _eigenfunction_observables). The
    # handler's own vector is reachable through _get_bifurcation_eigenvector(), but restoring what
    # was there keeps the rest of the worker reading exactly as it did.
    saved_vals, saved_vects = p.get_last_eigenvalues().copy(), p.get_last_eigenvectors().copy()
    res = {}
    try:
        tracked, _ = p.solve_eigenproblem(neigen, shift=shift, quiet=True, **kw)
        res["track_eig_re"], res["track_eig_im"] = _sorted_spectrum(tracked)
        # The row block the eigenproblem was actually assembled on. Under --distribute these must
        # tile [0, base_ndof) across the ranks; if the augmented layout had leaked through, nrow
        # would be the augmented count instead.
        nrow, nrow_local, first_row, distributed = p._get_base_dof_distribution_info()
        res["eig_nrow"] = int(nrow)
        res["eig_nrow_local"] = int(nrow_local)
        res["eig_first_row"] = int(first_row)
        res["eig_row_distributed"] = bool(distributed)
    finally:
        p._last_eigenvalues, p._last_eigenvectors = saved_vals, saved_vects
    return res


def _eigen_after_deactivation(p, neigen, shift, azimuthal_m=None):
    """The same eigenproblem once the tracker is gone -- the B half of the A/B above."""
    kw = {} if azimuthal_m is None else {"azimuthal_m": azimuthal_m}
    plain, _ = p.solve_eigenproblem(neigen, shift=shift, quiet=True, **kw)
    re, im = _sorted_spectrum(plain)
    return {"plain_eig_re": re, "plain_eig_im": im}


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


def fold_case(N=8, lam0=4.0, outdir=None, eigenvector_scaling="unit", with_guess=True, with_eigen=False):
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
            "max_residual": _converged_max_residual(p),
        }
        res.update(_eigenvector_norms(p))
        if with_eigen:
            res.update(_eigen_during_tracking(p, 4, shift=0.5))
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
        if with_eigen:
            res.update(_eigen_after_deactivation(p, 4, shift=0.5))
        return res


def hopf_case(N=20, B0=2.5, outdir=None, with_eigen=False):
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
            "max_residual": _converged_max_residual(p),
        }
        res.update(_eigenvector_norms(p))
        if with_eigen:
            # A Hopf sits at +-i*omega, not at 0, so the shift is placed near the real axis and away
            # from both: what has to come back is a complex pair at |Im| = omega.
            res.update(_eigen_during_tracking(p, 6, shift=0.5))
        p.deactivate_bifurcation_tracking()
        n_global, nrow_local, first_row, distributed = p._get_dof_distribution_info()
        res["ndof_after_deactivate"] = int(p.ndof())
        res["nrow_local_after_deactivate"] = int(nrow_local)
        # After deactivating, for the reason given in fold_case().
        res.update(_eigenfunction_observables(p, 0))
        if with_eigen:
            res.update(_eigen_after_deactivation(p, 6, shift=0.5))
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
    """Reaction-diffusion on a rectangle with u=0 all around: pitchfork at the first Dirichlet mode.

    A 1 x 1.05 rectangle rather than the unit square, and the aspect ratio is load-bearing. On the
    SQUARE the (1,2) and (2,1) Dirichlet modes are degenerate by symmetry -- here at lam-59.26 and
    lam-29.63 twice -- and _eigen_during_tracking asks for exactly 4 eigenvalues, so the truncation
    falls INSIDE that degenerate pair. Which of the two a Krylov solver returns in the fourth slot is
    then its own business: under --distribute the tracked half came back with both copies of -29.63
    and the untracked half with one copy plus the next mode at -79.18, and the A/B assertion compared
    two spectra that were both correct. Verified at n=6, where both halves agree entry for entry, and
    by 1 x 1.05, which splits the pair to -29.63/-26.87 and makes n=4 unambiguous again.

    Not a solver defect and not worth a tolerance: an nev cut inside a degenerate cluster has no
    well-defined answer, so the fix is to not put one there. The bifurcation is unaffected -- it is
    still the symmetry breaking of u=0 in the first mode, only at lam = pi^2*(1 + 1/1.05^2).
    """

    ASPECT = 1.05

    def __init__(self, N=8):
        super().__init__()
        self.N = N
        self.lam = self.define_global_parameter(lam=1)

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1.0, self.ASPECT])
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


def _reaction_case(prob, bifurcation_type, lam_start, lam_step, azimuthal_m, outdir, with_eigen=False):
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
            "max_residual": _converged_max_residual(p),
        }
        res.update(_eigenvector_norms(p))
        if with_eigen:
            # The azimuthal case asks for the AXISYMMETRIC (m=0) spectrum while an m=1 bifurcation is
            # being tracked -- the check "does the base state itself fold underneath this locus?",
            # and the case where the mode of the eigensolve differs from the tracked one. It works
            # here precisely because azimuthal tracking has already released the strong axis
            # conditions, so no renumbering is needed; the m=0 axis conditions come back as an
            # EigenMatrixSetDofsToZero manipulator instead.
            res.update(_eigen_during_tracking(p, 4, shift=0.5,
                                              azimuthal_m=0 if azimuthal_m is not None else None))
            # The tracked mode must survive an eigensolve at another one: the tracker reads the same
            # global "azimuthal_m" parameter when it assembles its eigen rows.
            if azimuthal_m is not None:
                res["m_after_eigen"] = float(p._azimuthal_mode_param_m.value)
            # ...and so must the tracked state itself. Re-converging has to land back on the same
            # critical parameter; a corrupted augmented system moves it.
            p.solve()
            res["param_after_eigen"] = float(p.lam.value)
        p.deactivate_bifurcation_tracking()
        n_global, nrow_local, first_row, distributed = p._get_dof_distribution_info()
        res["ndof_after_deactivate"] = int(p.ndof())
        res["nrow_local_after_deactivate"] = int(nrow_local)
        # After deactivating, for the reason given in fold_case().
        res.update(_eigenfunction_observables(p, 0))
        if with_eigen:
            res.update(_eigen_after_deactivation(p, 4, shift=0.5,
                                                 azimuthal_m=0 if azimuthal_m is not None else None))
        return res


def pitchfork_case(N=8, outdir=None, with_eigen=False):
    """Symmetry-breaking onset of u=0 in the reaction-diffusion square: MyPitchForkHandler."""
    return _reaction_case(PitchforkProblem(N=N), "pitchfork", 10.0, 4.0, None, outdir, with_eigen)


def azimuthal_case(N=8, outdir=None, with_eigen=False):
    """m=1 onset of u=0 in the axisymmetric reaction-diffusion: AzimuthalSymmetryBreakingHandler."""
    return _reaction_case(AzimuthalReactionProblem(N=N), "azimuthal", 10.0, 4.0, 1, outdir, with_eigen)


def fold_noguess_case(N=8, lam0=4.0, outdir=None, with_eigen=False):
    """The same fold, but found without ever solving an eigenproblem: see fold_case(with_guess)."""
    return fold_case(N=N, lam0=lam0, outdir=outdir, with_guess=False, with_eigen=with_eigen)


class AxisVectorEquations(Equations):
    """A vector field on an axisymmetric domain, so that AxisymmetryBC has something to pin.

    A SCALAR field is regular at r=0 for m=0 and needs no axis condition at all; it takes a vector,
    whose radial and azimuthal components must vanish there at m=0, for the mode-dependent Dirichlet
    machinery to exist. That machinery is what makes an m!=0 eigenproblem need a renumbering.
    """

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def define_fields(self):
        self.define_vector_field("v", "C2")

    def define_residuals(self):
        v, w = var_and_test("v")
        self.add_residual(weak(partial_t(v), w) + weak(grad(v), grad(w))
                          - weak(self.lam * v - dot(v, v) * v, w))


class AxisVectorProblem(Problem):
    def __init__(self, N=6):
        super().__init__()
        self.N = N
        self.lam = self.define_global_parameter(lam=1)

    def define_problem(self):
        from pyoomph.equations.generic import AxisymmetryBC
        self.set_coordinate_system(axisymmetric)
        self += RectangularQuadMesh(N=self.N)
        eqs = AxisVectorEquations(self.lam)
        eqs += DirichletBC(v_x=0, v_y=0, v_phi=0) @ ["right", "top", "bottom"]
        eqs += AxisymmetryBC() @ "left"
        self += eqs @ "domain"


def _dirichlet_flags(p):
    """The activation state of every Dirichlet condition on every mesh, as a comparable snapshot."""
    return [list(m._get_dirichlet_active_flags()) for m in p._iterate_all_meshes()]


def eigen_refusals_case(N=6, outdir=None, with_eigen=True):
    """The eigensolve-while-tracking refusals, and that a refusal leaves nothing behind.

    Serial only -- it asserts on error messages, not on numbers, so there is nothing for a
    distributed run to compare against. It lives here because it needs its own process anyway (a
    second Problem in one interpreter segfaults in the JIT loader).
    """
    res = {}
    with AxisVectorProblem(N=N) as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.set_eigensolver("slepc")
        p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=True)
        p.initialise()
        p.lam.value = 5.0
        p.solve()
        p.solve_eigenproblem(3, quiet=True)
        p.activate_bifurcation_tracking("lam", "fold")

        # A zero shift is the DEFAULT of solve_eigenproblem and is exactly the one value that cannot
        # work while tracking: lambda=0 is an exact eigenvalue of the state the tracker converged to.
        try:
            p.solve_eigenproblem(3, quiet=True)
            res["zero_shift_refused"] = False
        except RuntimeError as e:
            res["zero_shift_refused"] = True
            res["zero_shift_message"] = str(e)
        # shift=None is refused for the same reason: SLEPc then targets 0 and factorises there.
        try:
            p.solve_eigenproblem(3, shift=None, quiet=True)
            res["none_shift_refused"] = False
        except RuntimeError:
            res["none_shift_refused"] = True

        # m != 0 while tracking a FOLD: the strong axis conditions are still on and releasing them
        # renumbers, which would pull the augmented dof vector out from under the handler.
        before = _dirichlet_flags(p)
        try:
            p.solve_eigenproblem(3, shift=0.5, azimuthal_m=1, quiet=True)
            res["m1_refused"] = False
        except RuntimeError as e:
            res["m1_refused"] = True
            res["m1_message"] = str(e)
        # And the refusal must leave nothing behind: _before_eigen_solve has ALREADY deactivated the
        # axis conditions by the time it reports that a renumbering is needed, so they have to be put
        # back. A problem describing boundary conditions its numbering does not have is worse than
        # the missing feature.
        res["dirichlet_flags_restored"] = (_dirichlet_flags(p) == before)

        # m = 0 still works, and the tracker is unharmed by the two refusals.
        evals, _ = p.solve_eigenproblem(3, shift=0.5, quiet=True)
        res["m0_eig_re"], res["m0_eig_im"] = _sorted_spectrum(evals)
        nrow, _nrow_local, _first, _dist = p._get_base_dof_distribution_info()
        res["eig_nrow"] = int(nrow)
        res["ndof"] = int(p.ndof())
        p.solve()
        res["param_after_refusals"] = float(p.lam.value)
        p.deactivate_bifurcation_tracking()
        res["ndof_after_deactivate"] = int(p.ndof())
    return res


_CASES = {"fold": fold_case, "hopf": hopf_case, "azimuthal": azimuthal_case,
          "pitchfork": pitchfork_case, "fold_noguess": fold_noguess_case,
          "eigen_refusals": eigen_refusals_case}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--case", default="fold", choices=sorted(_CASES))
    ap.add_argument("--eigenvector-scaling", default="unit", choices=["unit", "auto"])
    # Off by default so the existing tracking assertions keep measuring exactly what they did: the
    # extra eigensolves are harmless but they do run the tracker through more code.
    ap.add_argument("--eigen-during-tracking", action="store_true",
                    help="also solve the base state's eigenproblem with the tracker installed, and "
                         "again once it is removed, and report both spectra")
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(),
               "case": "%s_N%d" % (args.case, args.size)}
    try:
        kwargs = {"with_eigen": args.eigen_during_tracking}
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
