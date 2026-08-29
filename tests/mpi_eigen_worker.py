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

# Worker for tests/test_mpi_eigenvalues.py -- launched under `mpirun ...`, with and without
# --distribute. Solves a generalized eigenproblem through SLEPc and prints one PYOOMPH_MPI_RESULT
# line per rank.
#
# The eigenvalues alone are a weak certificate: SLEPc computes them collectively, so every rank
# necessarily reports the same numbers whether or not pyoomph handed it the right matrix rows or
# gathered the eigenvectors into the right global slots. What pins those down is "eigfunc_l2", the
# integral of the squared eigenfunction over the mesh. Getting there runs the eigenvector back
# through set_eigenfunction_as_dofs() -> set_current_dofs(), which scatters BY GLOBAL EQUATION
# NUMBER, and then integrates over non-halo elements with an MPI_Allreduce. A gathered eigenvector
# whose entries landed in the wrong order still has the right 2-norm and still solves nothing
# visible in the eigenvalues, but it puts the eigenfunction in the wrong place on the mesh and this
# integral moves.
#
# Note the dof numbering is NOT comparable between a serial and a distributed run -- distribute()
# renumbers so that each rank owns a contiguous block -- so every quantity compared across the two
# has to be numbering-independent. Eigenvalues and mesh integrals are; a dof vector is not.

import argparse
import json
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.navier_stokes import NavierStokesEquations, NoSlipBC
from pyoomph.equations.advection_diffusion import AdvectionDiffusionEquations
from pyoomph.equations.generic import AxisymmetryBC
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class DiffusionEquations(Equations):
    """-laplace(u) with a first-order time derivative, i.e. the plainest possible mass matrix."""

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)))


class DiffusionProblem(Problem):
    """Dirichlet in x, Neumann in y, so the spectrum is -(n^2+m^2)*pi^2 with n>=1, m>=0.

    The leading eigenvalue -pi^2 is non-degenerate, which is what makes an eigenVECTOR comparison
    meaningful: at a repeated eigenvalue any basis of the eigenspace is a valid answer, and serial
    and distributed runs would be free to return different ones.
    """

    def __init__(self, N=8):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N)
        eqs = DiffusionEquations()
        eqs += DirichletBC(u=0) @ "left"
        eqs += DirichletBC(u=0) @ "right"
        # Partition-independent: evaluate_integral_function skips halo elements and MPI_Allreduce-sums.
        eqs += IntegralObservables(usqr=var("u") ** 2)
        self += eqs @ "domain"


class AzimuthalProblem(Problem):
    """Axisymmetric diffusion set up for azimuthal stability, i.e. the COMPLEX eigenproblem path.

    m != 0 is what brings the imaginary residual contribution into play, and with it the branch in
    get_J_M_n_and_type() that decides whether the eigenproblem is complex -- a decision made from a
    per-rank nonzero count, so it is exactly the branch that can make two ranks disagree.
    """

    def __init__(self, N=8):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.set_coordinate_system(axisymmetric)
        self += RectangularQuadMesh(N=self.N)
        eqs = DiffusionEquations()
        eqs += DirichletBC(u=0) @ "right"
        eqs += DirichletBC(u=0) @ "top"
        eqs += DirichletBC(u=0) @ "bottom"
        eqs += IntegralObservables(usqr=var("u") ** 2)
        self += eqs @ "domain"


class AxisymmetricFlowProblem(Problem):
    """Axisymmetric Boussinesq-free flow with an AxisymmetryBC, i.e. the MATRIX-MANIPULATOR path.

    At m != 0 the axis conditions are imposed by rewriting rows of J and M rather than by pinning:
    setup_forced_zero_dof_list_for_eigenproblems() installs an EigenMatrixSetDofsToZero for
    domain/left/{velocity_y,pressure,T} and the corner interfaces below it. Distributed, that
    surgery cannot be done on the assembled global matrix any more -- no rank has one -- so it moves
    onto the PETSc matrices, and this is the problem that exercises it.

    The corner interfaces (domain/bottom/left) are the reason resolve_equations_by_name() has to
    tolerate a locally absent submesh: a corner is one point and belongs to a single partition.
    """

    def __init__(self, N=6):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.set_coordinate_system(axisymmetric)
        self += RectangularQuadMesh(N=self.N)
        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=1)
        eqs += AdvectionDiffusionEquations(fieldnames="T", diffusivity=1, space="C1")
        eqs += DirichletBC(T=0) @ "bottom"
        eqs += DirichletBC(T=-1) @ "top"
        eqs += NoSlipBC() @ ["top", "right", "bottom"]
        eqs += AxisymmetryBC() @ "left"
        eqs += DirichletBC(pressure=0) @ "bottom/right"
        eqs += IntegralObservables(usqr=var("T") ** 2 + dot(var("velocity"), var("velocity")))
        self += eqs @ "domain"


def _eigenfunction_observables(p, index=0):
    """Integral observables of eigenfunction ``index``, with the dofs restored afterwards.

    mode="abs", i.e. the entrywise |v|, because an eigenvector is only determined up to a complex
    phase: SLEPc is free to return e^{i*phi}*v for any phi, and serial and distributed runs do pick
    different ones. mode="real" would then keep only cos(phi) of the eigenfunction and this integral
    would differ between the two runs by that factor alone -- a real effect with nothing wrong
    underneath, which is exactly what it looked like the first time this test was run. |v| is
    invariant under the phase, so what is left is the shape, which is what we want to certify.
    """
    backup_dofs, backup_pinned = p.set_eigenfunction_as_dofs(index, mode="abs")
    try:
        obs = p.get_mesh("domain").evaluate_all_observables()
        return {("eigfunc_" + k): float(v) for k, v in obs.items()}
    finally:
        p.set_all_values_at_current_time(backup_dofs, backup_pinned, False)


_PROBLEMS = {"diffusion": DiffusionProblem, "azimuthal": AzimuthalProblem,
             "axiflow": AxisymmetricFlowProblem}


def solve_case(N=8, neigen=3, eigensolver="slepc", azimuthal_m=None, problem="diffusion",
               outdir=None):
    prob = _PROBLEMS[problem](N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.set_eigensolver(eigensolver)
        if azimuthal_m is not None:
            p.setup_for_stability_analysis(azimuthal_stability=True, analytic_hessian=False)
        p.initialise()
        p.solve()
        if azimuthal_m is not None:
            evals, evects = p.solve_eigenproblem(neigen, azimuthal_m=azimuthal_m, quiet=True)
        else:
            evals, evects = p.solve_eigenproblem(neigen, quiet=True)

        # Zero here means the matrix-manipulator path never ran, so a test relying on it proved
        # nothing. Per-rank on purpose: only the ranks owning part of the axis have rows to zero.
        zeroed = 0
        for manip in p.get_eigen_solver().matrix_manipulators:
            zeroed += len(getattr(manip, "zeromap", ()))

        # The row split the EIGENSOLVER actually used, which is not the same question as whether the
        # mesh was distributed: without --distribute the matrices are replicated and the solver imposes
        # its own split. Reported because the eigenvalues come out the same whether the solve was split
        # or redundant, so nothing else in this payload would notice a solve that quietly went serial.
        layout = getattr(p.get_eigen_solver(), "last_parallel_layout", (0, 0, False))

        res = {
            "eigen_nrow_local": int(layout[0]),
            "eigen_first_row": int(layout[1]),
            "eigen_parallel": bool(layout[2]),
            "zeromap_size": int(zeroed),
            # Which branch get_J_M_n_and_type() took. False in an azimuthal case would mean the
            # imaginary contribution never materialised and the complex path went untested.
            "complex_assembly": bool(p.get_eigen_solver().last_assembly_was_complex),
            "ndof": int(p.ndof()),
            "distributed": bool(p.is_distributed()),
            "nconv": int(len(evals)),
            "evals_re": [float(numpy.real(e)) for e in evals],
            "evals_im": [float(numpy.imag(e)) for e in evals],
            # Full global length on every rank, or the gather did not happen.
            "evect_len": int(evects.shape[1]) if len(evals) else 0,
            # A sum over all entries, so it is invariant under the dof PERMUTATION -- which is exactly
            # why it says nothing about where on the mesh the eigenvector landed, and why the test
            # only uses it to check that the ranks returned the same vector rather than each its own
            # row block. eigfunc_* below is what constrains the placement.
            "evect0_absum": float(numpy.sum(numpy.abs(evects[0]))) if len(evals) else 0.0,
            "evect0_norm": float(numpy.linalg.norm(evects[0])) if len(evals) else 0.0,
        }
        if len(evals):
            res.update(_eigenfunction_observables(p, 0))
        return res


def rotate_case(N=8, normalize_max=True, outdir=None):
    """rotate_eigenvectors() must give the same answer from a local row block as from a full vector.

    The PETSc eigensolver replicates eigenvectors (see _vector_to_global_array), so nothing in the
    normal flow reaches rotate_eigenvectors' distributed branch and it would go untested. Slicing a
    replicated eigenvector down to the rank's own rows and feeding that back in is exactly the input
    that branch is written for, and the full-vector answer restricted to the same rows is the
    reference it has to reproduce.

    The point of the comparison is the reduction operators: the phase comes from a complex MEAN and
    the magnitude from either a MAX or a mean, and a partition where the ranks own unequal numbers of
    the selected dofs -- the normal case -- gives a different answer if any of them is combined as
    though it were one of the others.
    """
    prob = DiffusionProblem(N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        p.set_linear_solver("petsc_mumps")
        p.set_eigensolver("slepc")
        p.initialise()
        p.solve()
        p.solve_eigenproblem(2, quiet=True)
        evs = numpy.array(p.get_last_eigenvectors())
        n_global, nrow_local, first_row, distributed = p._get_dof_distribution_info()

        full = p.rotate_eigenvectors(evs, "domain/u", normalize_dofs=True,
                                     normalize_max=normalize_max)
        sliced = p.rotate_eigenvectors(evs[:, first_row:first_row + nrow_local], "domain/u",
                                       normalize_dofs=True, normalize_max=normalize_max)
        ref = full[:, first_row:first_row + nrow_local]
        sel = numpy.array(sorted(p.dof_strings_to_global_equations("domain/u")), dtype=numpy.int64)
        per_ev, cancel = [], []
        for k in range(ref.shape[0]):
            scale = float(numpy.amax(numpy.absolute(ref[k])))
            err = float(numpy.amax(numpy.absolute(sliced[k] - ref[k])))
            per_ev.append(err / scale if scale > 0 else err)
            # |mean| / mean|.| over the selected dofs. The phase is the angle of the MEAN, so when the
            # eigenmode is antisymmetric over those dofs the mean is a near-total cancellation and its
            # angle amplifies any difference in summation order enormously -- which is a property of
            # what rotate_eigenvectors is asked to compute, not of how it is reduced. Reported so a
            # loose tolerance below is visibly tied to the cancellation rather than just tolerated.
            vals = evs[k][sel]
            denom = float(numpy.sum(numpy.absolute(vals)))
            cancel.append(float(numpy.absolute(numpy.sum(vals))) / denom if denom > 0 else 0.0)
        return {"distributed": bool(distributed), "nrow_local": int(nrow_local),
                "first_row": int(first_row), "n_global": int(n_global),
                "n_selected_local": int(numpy.sum((sel >= first_row) * (sel < first_row + nrow_local))),
                "rel_err_per_ev": per_ev, "cancellation_per_ev": cancel,
                "rel_err": max(per_ev) if per_ev else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--neigen", type=int, default=3)
    ap.add_argument("--eigensolver", default="slepc")
    ap.add_argument("--azimuthal-m", type=int, default=-1)
    ap.add_argument("--problem", default="diffusion", choices=sorted(_PROBLEMS))
    ap.add_argument("--mode", default="eigen", choices=["eigen", "rotate"])
    ap.add_argument("--normalize-max", type=int, default=1)
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    azi = None if args.azimuthal_m < 0 else args.azimuthal_m
    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(),
               "case": "%s_N%d_%s_m%s" % (args.problem, args.size, args.eigensolver, str(azi))}
    try:
        if args.mode == "rotate":
            payload.update(rotate_case(N=args.size, normalize_max=bool(args.normalize_max),
                                       outdir=args.outdir))
        else:
            payload.update(solve_case(N=args.size, neigen=args.neigen,
                                      eigensolver=args.eigensolver, azimuthal_m=azi,
                                      problem=args.problem, outdir=args.outdir))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
