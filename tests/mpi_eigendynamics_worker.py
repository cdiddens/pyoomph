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


# Worker for test_mpi_eigendynamics.py.
#
# An eigendynamics animation (Problem.create_eigendynamics_animation) draws the base state plus
# Re(factor*eigenvector), one frame per phase, and a mirrored half of a frame uses a different factor.
# The plot code runs on rank 0 alone inside run_with_global_mesh_data, so the perturbation cannot be
# applied there: it is part of the merge REQUEST instead, and every rank applies it
# (merge_perturbed_global_mesh_data in pyoomph/meshes/meshdatamerge.py).
#
# What is compared against a serial run is numbering-independent: counts, a digest over the sorted node
# coordinates, permutation-invariant field statistics, and -- the sharp one -- the scalar
# sum((u_perturbed - u_base)**2) taken WITHIN one run, where the two merged entries share the merged
# node order. An implementation that perturbs on rank 0 only gets roughly the rank-0 fraction of that
# number, which is the "naive" case below.

import argparse
import hashlib
import json
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.meshdatacache import MeshDataCacheKey
from pyoomph.meshes.meshdatamerge import (merge_global_mesh_data, merge_perturbed_global_mesh_data,
                                          needs_merging, run_with_global_mesh_data)
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc, get_mpi_bcast


class DiffusionEqs(Equations):
    """u_t = laplace(u) on a rectangle: the leading eigenvalue is -(1/Lx^2+1/Ly^2)*pi^2, non-degenerate.

    Degeneracy would make this untestable -- at a repeated eigenvalue serial and distributed runs may
    return different vectors from the same eigenspace and nothing is comparable.
    """

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(partial_t(u), v) + weak(grad(u), grad(v)) - weak(1, v))
        # A local expression, so the merge's eager evaluation of those is covered as well: it happens
        # inside the perturbed window and must therefore see the perturbed state.
        self.add_local_function("uplus", u + 10 * var("coordinate_x"))


class EigenAnimProblem(Problem):
    def __init__(self, N=6):
        super().__init__()
        self.N = N
        self.write_states = False  # save_state does not support distributed meshes

    def define_problem(self):
        # Not a square, so that the second mode is not degenerate with anything either.
        self += RectangularQuadMesh(N=[self.N, self.N + 2], size=[1, 1.7])
        eqs = DiffusionEqs()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self += eqs @ "domain"


def _digest(rows):
    """Order-independent digest of an array of coordinate rows."""
    if len(rows) == 0:
        return "empty"
    flat = numpy.asarray(rows, dtype=numpy.float64).round(10) + 0.0  # +0.0 turns -0.0 into 0.0
    order = numpy.lexsort(tuple(flat[:, i] for i in range(flat.shape[1] - 1, -1, -1)))
    return hashlib.sha1(flat[order].tobytes()).hexdigest()


def _summarize(entry):
    """Everything about a merged entry that a partition cannot change."""
    co = entry.get_coordinates()
    res = {"nnode": int(co.shape[1]), "nelem": int(entry.elem_indices.shape[0]),
           "coord_digest": _digest(co.transpose())}
    for field in ("u", "uplus"):
        values = numpy.asarray(entry.get_data(field), dtype=float)
        res[field + "_sum"] = float(numpy.sum(values))
        res[field + "_sqsum"] = float(numpy.sum(values ** 2))
        res[field + "_absum"] = float(numpy.sum(numpy.abs(values)))
        res[field + "_max"] = float(numpy.amax(values))
    return res


def _values(entry):
    return numpy.asarray(entry.get_data("u"), dtype=float)


def _key():
    return MeshDataCacheKey.create(nondimensional=False, tesselate_tri=True, global_mesh=True)


def _merged(problem, mesh, factor=None, index=0):
    """The merged entry of the base state (factor None) or of a perturbed one. Collective either way."""
    if not needs_merging(mesh):
        # Serial or replicated: the same states, without any of the merge machinery.
        if factor is None:
            return problem.get_cached_mesh_data(mesh, nondimensional=False, tesselate_tri=True)
        problem.invalidate_cached_mesh_data()
        olddofs, _ = problem.get_current_dofs()
        try:
            problem.set_current_dofs(numpy.asarray(olddofs)
                                     + numpy.real(factor * problem.get_last_eigenvectors()[index]))
            entry = problem.get_cached_mesh_data(mesh, nondimensional=False, tesselate_tri=True)
            # As MatplotlibPlotter._get_mesh_data does: local expressions are lazy, so they have to be
            # materialised before the state is restored, or they describe the base state.
            for name in entry.local_expr_indices.keys():
                if name not in entry.nodal_field_inds.keys():
                    entry.nodal_local_exprs[name] = numpy.asarray(entry.get_data(name))
            return entry
        finally:
            problem.set_current_dofs(olddofs)
            problem.invalidate_cached_mesh_data()
    if factor is None:
        return merge_global_mesh_data(mesh, _key())
    return merge_perturbed_global_mesh_data(mesh, _key(), index, complex(factor))


def _fix_eigenvector_phase(problem):
    """Pin the arbitrary sign/phase of the eigenvector, so serial and distributed describe one field.

    An eigenvector is defined up to a complex factor, and the solvers do not agree on which one -- least
    of all across partitions. Anchoring on the largest entry is enough here: the modes are real, so this
    just fixes the sign, and it is a statement about the field, not about the dof numbering.
    """
    evs = numpy.array(problem.get_last_eigenvectors())
    for i in range(len(evs)):
        k = int(numpy.argmax(numpy.abs(evs[i])))
        anchor = evs[i][k]
        if abs(anchor) > 0:
            evs[i] = evs[i] * (numpy.conj(anchor) / abs(anchor))
    problem._last_eigenvectors = evs


def run_case(case="symmetric", N=6, outdir="_eigendynamics"):
    result = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "case": case}
    with EigenAnimProblem(N=N) as problem:
        problem.set_output_directory(outdir)
        problem.quiet()
        problem.set_linear_solver("petsc_mumps")
        problem.set_eigensolver("slepc")
        problem.initialise()
        problem.solve()
        problem.solve_eigenproblem(1)
        _fix_eigenvector_phase(problem)
        mesh = problem.get_mesh("domain")
        result["distributed"] = bool(mesh.is_mesh_distributed())
        result["ndof"] = int(problem.ndof())
        dofs_before = numpy.asarray(problem.get_current_dofs()[0])
        # exp(i*m*pi) for m=1, i.e. the mirrored half of the same frame
        right, left = 0.37 + 0.11j, -(0.37 + 0.11j)

        if case == "symmetric":
            # Every rank calls the collective directly, without the request scope.
            base = _merged(problem, mesh)
            pert_r = _merged(problem, mesh, right)
            pert_l = _merged(problem, mesh, left)
        elif case == "scope":
            # The real animation shape: rank 0 asks, the others serve. The trailing plain request is
            # the cache-pollution assertion -- it must give exactly the first one back.
            got = {}

            def ask():
                got["base"] = _merged(problem, mesh)
                got["right"] = _merged(problem, mesh, right)
                got["left"] = _merged(problem, mesh, left)
                got["base_again"] = _merged(problem, mesh)

            run_with_global_mesh_data({"": problem}, ask, context="the eigendynamics test")
            base, pert_r, pert_l = got.get("base"), got.get("right"), got.get("left")
            if get_mpi_rank() == 0:
                result["base_again"] = _summarize(got["base_again"])
        elif case == "partial":
            # What a perturbation that does not reach every rank looks like, so that the comparison
            # above is shown to be discriminating rather than insensitive. The literal old bug -- rank 0
            # perturbing inside the plot block -- cannot even be reproduced here: set_current_dofs is
            # collective, so calling it on one rank alone deadlocks rather than returning wrong data.
            # This is the same thing without the deadlock: the perturbation is applied by everyone, but
            # only on the dofs of rank 0's block.
            base = _merged(problem, mesh)
            _n, nrow_local, first_row, _dist = problem._get_dof_distribution_info()
            width = int(get_mpi_bcast(int(nrow_local)))  # rank 0's block, known to every rank
            start = int(get_mpi_bcast(int(first_row)))
            masked = numpy.zeros(len(problem.get_last_eigenvectors()[0]), dtype=complex)
            masked[start:start + width] = problem.get_last_eigenvectors()[0][start:start + width]
            olddofs, _ = problem.get_current_dofs()
            problem.invalidate_cached_mesh_data()
            try:
                problem.set_current_dofs(numpy.asarray(olddofs) + numpy.real(right * masked))
                pert_r = _merged(problem, mesh) if needs_merging(mesh) else \
                    problem.get_cached_mesh_data(mesh, nondimensional=False, tesselate_tri=True)
            finally:
                problem.set_current_dofs(olddofs)
                problem.invalidate_cached_mesh_data()
            pert_l = None
        elif case == "plotter":
            # The public API, end to end: create_eigendynamics_animation drives a plotter whose
            # define_plot records the merged data instead of drawing it. This is the case that covers
            # the eigenvector INDEX being plumbed through to the plotter at all.
            from pyoomph.output.plotting import MatplotlibPlotter

            recorded = {}

            class _RecordingPlotter(MatplotlibPlotter):
                def define_plot(self):
                    step = self._output_step
                    for mirror in (False, True):
                        entry = self._get_mesh_data("domain", mirror_x=mirror)
                        recorded[(step, mirror)] = _summarize(entry)

            plotter = _RecordingPlotter(eigenvector=None)
            problem.create_eigendynamics_animation("eigenanim", plotter, max_amplitude=0.5,
                                                   numperiods=1, numouts=2)
            base = _merged(problem, mesh)
            pert_r = pert_l = None
            if get_mpi_rank() == 0:
                # Frame 0 is t=0, where left and right differ only by the mirror factor exp(i*m*pi);
                # with m=0 (no azimuthal mode here) they coincide, which is the correct behaviour and
                # what is asserted. What matters is that both are the PERTURBED state.
                result["frames"] = {"%d_%s" % (step, mirror): summary
                                    for (step, mirror), summary in sorted(recorded.items())}
        else:
            raise RuntimeError("unknown case " + case)

        # Nobody may be left holding a perturbed state, on any rank.
        result["dof_drift"] = float(numpy.amax(numpy.abs(
            numpy.asarray(problem.get_current_dofs()[0]) - dofs_before)))

        if get_mpi_rank() == 0:
            assert base is not None
            result["base"] = _summarize(base)
            ub = _values(base)
            if pert_r is not None:
                result["right"] = _summarize(pert_r)
                result["right_l2"] = float(numpy.sum((_values(pert_r) - ub) ** 2))
            if pert_l is not None:
                result["left"] = _summarize(pert_l)
                result["left_l2"] = float(numpy.sum((_values(pert_l) - ub) ** 2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="_eigendynamics")
    ap.add_argument("--case", default="symmetric")
    ap.add_argument("--size", type=int, default=6)
    args, _ = ap.parse_known_args()
    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
    try:
        payload.update(run_case(case=args.case, N=args.size, outdir=args.outdir))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
