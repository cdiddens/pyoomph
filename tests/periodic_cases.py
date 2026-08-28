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

# Shared problem definitions for the periodic-boundary MPI campaign. Imported by the serial reference
# inside tests/test_mpi_periodic.py and by the worker it launches under `mpirun -n N ... --distribute`,
# so both sides solve a bit-identical problem and differ only in how it is partitioned.
#
# Periodicity in pyoomph is pointer aliasing, not a constraint equation: BoundaryNodeBase::
# make_node_periodic points the copy node's Value/Eqn_number arrays at the master's. Two things can
# therefore go wrong under --distribute, and the measured quantities are picked to separate them:
#
#   * ndof -- if a periodic link is silently lost (Data::~Data deep-copies every surviving copy and
#     clears Copied_node_pt without a word), the copies start contributing their own equations and
#     ndof GROWS. This is the sharp structural oracle and it is exact, not approximate.
#   * integral observables -- Mesh::evaluate_integral_function skips halo elements and MPI_Allreduce-
#     sums, so these are true global integrals; they certify the FIELD, i.e. that the shared
#     Eqn_number array was not double-bumped by synchronise_eqn_numbers and that halo/haloed
#     classification agreed about who owns each side of the seam.
#   * seam_jump -- max |u(x_master) - u(x_copy)| evaluated from the two sides of the seam. Zero by
#     construction if the aliasing survived, O(1) if it did not. Redundant with ndof in principle,
#     but it fails loudly in the case where a broken link happens not to change the dof count.
#
# Deliberately NOT compared: nelement() (per-rank, includes halos) and nodal values (a rank only holds
# its own partition).

import numpy as np

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh  # not in "from pyoomph import *"
from pyoomph.equations.additional import RefineMaxElementSize
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh

# All periodicity goes through PeriodicBC, which matches the two boundaries by KD-tree and defers
# corner nodes to MeshFromTemplateBase._link_periodic_corner_nodes. "line1d" exercises the 1D case (a
# single node per boundary, binary-tree periodic connection), "quad2d_x" one seam in 2D, and
# "quad2d_xy" two seams that meet in a corner, which is the only case that reaches the deferred
# corner pass.
CASES = ["line1d", "quad2d_x", "quad2d_xy", "line1d_nonper", "quad2d_nonper"]

# Not in CASES: both are expected to FAIL under --distribute and exist only for the tests that assert
# those refusals. Both solve fine serially.
#   quad2d_x_adaptive -- refining a distributed periodic mesh, see
#       Problem._require_no_distributed_periodic_refinement
#   quad2d_x_ale      -- periodic boundaries on a moving mesh, see
#       Problem._require_no_distributed_periodic_position_dofs
REFUSED_CASES = ["quad2d_x_adaptive", "quad2d_x_ale"]

# Wavenumbers of the source term. Deliberately whole multiples of 2*pi over the unit cell, so the
# exact solution is genuinely periodic and a broken seam shows up as a kink rather than as a small
# discretisation-level difference.
KX = 2
KY = 3


class ScreenedPoisson(Equations):
    """-lap(u) + u = f. Screened rather than plain Poisson because the doubly periodic case has no
    boundary left to pin, and plain Poisson would be singular there."""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) + weak(u - self.source, v))


class PeriodicProblem(Problem):
    def __init__(self, case="quad2d_x", N=16):
        super().__init__()
        self.case = case
        self.N = N

    def define_problem(self):
        case = self.case
        if case in ("line1d", "line1d_nonper"):
            periodic = case == "line1d"
            self.add_mesh(LineMesh(N=self.N, size=1.0))
            src = cos(2 * pi * KX * var("coordinate_x"))
            eqs = ScreenedPoisson(src)
            eqs += IntegralObservables(intu=var("u"), intu2=var("u") ** 2, area=1)
            if periodic:
                eqs += PeriodicBC("right", offset=[1.0]) @ "left"
            else:
                # Same discretisation, no periodicity: the control that proves the oracles above can
                # tell the two apart at all.
                eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
            self.add_equations(eqs @ "domain")
            return

        doubly = case == "quad2d_xy"
        adaptive = case == "quad2d_x_adaptive"
        ale = case == "quad2d_x_ale"
        periodic_x = case in ("quad2d_x", "quad2d_x_adaptive", "quad2d_x_ale")
        self.add_mesh(RectangularQuadMesh(N=self.N, size=[1.0, 1.0]))
        src = cos(2 * pi * KX * var("coordinate_x")) * cos(2 * pi * KY * var("coordinate_y"))
        eqs = ScreenedPoisson(src)
        eqs += IntegralObservables(intu=var("u"), intu2=var("u") ** 2, area=1,
                                   intgradu2=dot(grad(var("u")), grad(var("u"))))
        if adaptive:
            # A size criterion rather than a Z2 estimator: the initial adaption loop runs before the
            # first solve, so the field is still identically zero and an error estimator would flag
            # nothing. The refusal only fires on an adaption that actually changed the mesh, so the
            # criterion has to bite.
            self.max_refinement_level = 2
            eqs += RefineMaxElementSize(0.5 / (self.N * self.N))
        if ale:
            # A moving mesh, so that the nodal POSITIONS become unknowns. make_periodic() never
            # aliases positions, so the periodic copy then carries dofs of its own -- which is the
            # combination the distributed run has to refuse. mesh_y is left free on "top" so there is
            # something to solve for.
            eqs += LaplaceSmoothedMesh()
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "bottom"
            eqs += DirichletBC(mesh_x=True) @ "top"
        if doubly:
            eqs += PeriodicBC("right", offset=[1.0, 0]) @ "left"
            eqs += PeriodicBC("top", offset=[0, 1.0]) @ "bottom"
        else:
            eqs += DirichletBC(u=0) @ "top" + DirichletBC(u=0) @ "bottom"
            if periodic_x:
                eqs += PeriodicBC("right", offset=[1.0, 0]) @ "left"
            else:
                eqs += DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "right"
        self.add_equations(eqs @ "domain")


def _seam_jump(problem, case):
    """max |u| difference between the two sides of each periodic seam, measured on the mesh itself.

    Rank-local by construction (a rank only holds its own partition), so the harness takes the max
    over ranks rather than expecting every rank to see every seam node. A rank that holds neither
    side of any seam contributes 0.0, which is the right neutral element for a max.
    """
    if case in ("line1d_nonper", "quad2d_nonper"):
        return 0.0
    mesh = problem.get_mesh("domain")
    worst = 0.0
    for i in range(mesh.nnode()):
        n = mesh.node_pt(i)
        if not n.is_a_copy():
            continue
        m = n.copied_node_pt()
        for j in range(n.nvalue()):
            worst = max(worst, abs(n.value(j) - m.value(j)))
    return float(worst)


def run_case(case, N=16, outdir=None):
    """Solve one case and return the partition-independent measurements as a plain dict.

    Nothing here knows about distribution: the worker is launched with --distribute on the command
    line and pyoomph's own parser picks it up inside Problem.initialise(), so the serial reference and
    the distributed run go through exactly the same code."""
    prob = PeriodicProblem(case=case, N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.solve()
        m = p.get_mesh("domain")
        conv = p.get_last_residual_convergence()
        res = {
            # get_residuals() is gathered to full length on a distributed problem, so this is the same
            # number on every rank. All cases are linear, so it is machine zero iff the Jacobian is
            # exact -- which under --distribute additionally requires the halo exchange not to have
            # written through a periodic node's aliased Value array.
            "maxres": float(np.max(np.abs(np.asarray(p.get_residuals())))),
            "ndof": int(p.ndof()),
            "newton_conv": [float(c) for c in conv],
            "seam_jump": _seam_jump(p, case),
            "has_periodic_nodes": bool(m.has_periodic_nodes()),
        }
        for name, val in m.evaluate_all_observables().items():
            res["obs_" + name] = float(val)
        return res
