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

# Shared problem definitions for the 2D adaptive-mesh campaign on branch mixed_adapt. Imported by three
# places, which is the point: the serial tests (tests/test_adaptive_2d_campaign.py), the MPI harness
# (tests/test_mpi_adaptivity.py, which also runs them serially in-process to produce its reference), and
# the worker the harness launches under `mpirun -n N ... --distribute` (tests/mpi_worker.py). Keeping the
# definitions in one place is what makes the distributed-vs-serial comparison meaningful: both sides solve
# a bit-identical problem and differ only in how it is partitioned. It also means the serial and the MPI
# campaign cannot drift apart.
#
# All cases live on the box [-0.5,0.5]^2 in four discretisations -- pure quads, two triangle splits, and a
# genuinely MIXED quad+tri mesh with a cross-shape interface at x=0 -- at three refinement states:
#   (0,0): non-adaptive base mesh
#   (1,1): uniform level 1 (conforming, no hanging nodes)
#   (1,3): uniform level 1 plus a level-3 band on "top" -> non-uniform, TWO levels of 2:1 hanging jump
#
# The measured quantities (see measure()) are chosen to be partition-independent, so a distributed run must
# reproduce the serial numbers exactly:
#   * max|residual| -- all cases are linear, so this is ~0 iff the (hanging-node) Jacobian is exact. Since
#     the get_residuals() fix it is gathered to full length, hence identical on every rank.
#   * ndof         -- global on a distributed problem; certifies the same discretisation and the same
#                     hanging-node structure was built.
#   * integral observables -- Mesh::evaluate_integral_function skips halo elements and MPI_Allreduce-sums,
#                     so these are true global integrals and certify the FIELD, not just the residual.
# Deliberately NOT compared: nelement() (per-rank, includes halos) and nodal values (a rank only holds its
# own partition).

from pyoomph import *
from pyoomph.equations.additional import RefineMaxElementSize, RefineAccordingToElement  # not in "from pyoomph import *"
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.navier_stokes import StokesEquations
from pyoomph.equations.ALE import (LaplaceSmoothedMesh, ConstrainPositionsToC1Space,
                                   UnconstrainPositionsFromC1Space)
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

_BND = ["left", "right", "top", "bottom"]

MESH_KINDS = ["quad", "tri_left", "tri_crossed", "mixed"]
EQUATIONS = ["poisson1", "poisson2", "mixed12", "constrain12", "unconstrain12", "neumann",
             "stokes_th", "stokes_cr", "ale", "ale_posc1", "ale_posc1_unc"]
LEVELS = [(0, 0), (1, 1), (1, 3)]

# The prescribed outflow ("evaporation") through the free top surface in the ALE case.
J_EVAP = 0.1


class MixedBoxMesh(MeshTemplate):
    """[-0.5,0.5]^2 with quads for x<0 and triangles for x>0: one cross-shape quad<->tri interface at x=0."""

    def __init__(self, N=4, name="domain"):
        super().__init__()
        self.N = N
        self.dname = name

    def define_geometry(self):
        N = self.N
        dom = self.new_domain(self.dname)
        idx = {}

        def node(i, j):
            if (i, j) not in idx:
                idx[(i, j)] = self.add_node_unique(-0.5 + 1.0 * i / N, -0.5 + 1.0 * j / N)
            return idx[(i, j)]

        for i in range(N):
            for j in range(N):
                a, b, c, d = node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)
                if i < N // 2:
                    dom.add_quad_2d_C1(a, b, c, d)
                else:
                    dom.add_tri_2d_C1(a, b, d)
                    dom.add_tri_2d_C1(a, d, c)
        for j in range(N):
            self.add_facet_to_boundary("left", [node(0, j), node(0, j + 1)])
            self.add_facet_to_boundary("right", [node(N, j), node(N, j + 1)])
        for i in range(N):
            self.add_facet_to_boundary("bottom", [node(i, 0), node(i + 1, 0)])
            self.add_facet_to_boundary("top", [node(i, N), node(i + 1, N)])


def make_mesh(kind, N=4):
    if kind == "quad":
        return RectangularQuadMesh(name="domain", size=1, N=N, lower_left=[-0.5, -0.5])
    if kind in ("tri_left", "tri_crossed"):
        return RectangularQuadMesh(name="domain", size=1, N=N, lower_left=[-0.5, -0.5],
                                   split_in_tris=kind.split("_", 1)[1])
    if kind == "mixed":
        return MixedBoxMesh(N=N)
    raise ValueError("unknown mesh kind: " + str(kind))


class BoxProblem(Problem):
    def __init__(self, kind="quad", eq="poisson1", levels=(1, 3), N=4):
        super().__init__()
        self.kind, self.eq, self.levels, self.N = kind, eq, levels, N

    def define_problem(self):
        self += make_mesh(self.kind, self.N)
        x, y = var(["coordinate_x", "coordinate_y"])
        if self.eq in ("poisson1", "poisson2"):
            space = "C1" if self.eq == "poisson1" else "C2"
            eqs = PoissonEquation(source=1, space=space)
            eqs += DirichletBC(u=0) @ _BND
            eqs += IntegralObservables(intu=var("u"), intu2=var("u") ** 2)
        elif self.eq in ("mixed12", "constrain12", "unconstrain12"):
            # u on C2 driving v on C1: mixed continuous spaces on one mesh, so C1 owns a separate hang slot.
            # Three variants of the same problem, differing only in the C1 constraint on u:
            #   mixed12       -- none (baseline)
            #   constrain12   -- ConstrainFieldsToC1Space("u") everywhere: u is degraded to the C1 space
            #   unconstrain12 -- the same, plus UnconstrainFieldsFromC1Space("u") @ "top", restoring u's C2
            #                    dofs along the top edge only
            # so ndof(constrain12) < ndof(unconstrain12) < ndof(mixed12) certifies that the constraint bites
            # AND that the boundary unconstrain undoes it. On a hanging-node mesh the constrained non-vertex
            # nodes must receive a REGISTERED linear hang onto their C1 corners; merely pinning them would
            # leave the analytic Jacobian inconsistent with the residual (Newton would not converge in one
            # step), which is what newton_steps checks.
            #
            # NB the BCs: u and v are both Dirichlet on left/right/bottom and NATURAL (zero-flux) on "top".
            # Two reasons. (a) With u Dirichlet on top as well, its top-edge C2 mid-node values would be
            # pinned anyway and the boundary unconstrain would be a silent no-op -- the test would pass
            # while testing nothing. (b) With u and v carrying the SAME boundary conditions, the Green
            # identity int(v) == int(u^2) holds EXACTLY (all boundary terms drop), and it holds discretely
            # too whenever u and v live in the same discrete space -- i.e. exactly for constrain12. That is
            # asserted as an independent oracle: a mis-hung or sloppily restricted C1 constraint breaks it.
            eqs = PoissonEquation(source=1, space="C2")
            eqs += PoissonEquation(source=var("u"), space="C1", name="v")
            eqs += DirichletBC(u=0, v=0) @ ["left", "right", "bottom"]
            if self.eq != "mixed12":
                eqs += ConstrainFieldsToC1Space("u")
            if self.eq == "unconstrain12":
                eqs += UnconstrainFieldsFromC1Space("u") @ "top"
            eqs += IntegralObservables(intu=var("u"), intv=var("v"), intu2=var("u") ** 2)
        elif self.eq == "neumann":
            # Neumann fluxes on adaptive/mixed meshes: the flux is integrated over FACE elements whose
            # parent may be a hanging-node element of either shape. "right" is a constant flux; "top" is
            # spatially varying AND is the boundary that carries the refinement band, so its face elements
            # sit on refined parents while "right" mixes refined and unrefined ones.
            x = var("coordinate_x")
            eqs = PoissonEquation(source=1, space="C2")
            eqs += DirichletBC(u=0) @ ["left", "bottom"]
            eqs += NeumannBC(u=1) @ "right"
            eqs += NeumannBC(u=x) @ "top"
            eqs += IntegralObservables(intu=var("u"), intu2=var("u") ** 2)
        elif self.eq in ("ale", "ale_posc1", "ale_posc1_unc"):
            # Moving mesh (ALE): Stokes on a Laplace-smoothed mesh whose top surface is free in y, with a
            # prescribed outflow standing in for evaporation. The nodal POSITIONS are unknowns coupled to
            # the flow, so this exercises the hanging-node machinery on the position dofs as well as on the
            # fields. "area" detects the mesh motion itself -- without it a frozen mesh would pass.
            st = StokesEquations(mode="TH", dynamic_viscosity=1)
            eqs = st
            eqs += LaplaceSmoothedMesh()
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ ["left", "right", "bottom"]
            eqs += DirichletBC(mesh_x=True) @ "top"
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ ["left", "right", "bottom"]
            eqs += DirichletBC(velocity_x=0, velocity_y=J_EVAP) @ "top"
            eqs += DirichletBC(pressure=0) @ "bottom"
            # ConstrainPositionsToC1Space degrades the mesh POSITION to the C1 space (the moving-mesh
            # analogue of ConstrainFieldsToC1Space); the "_unc" variant restores the C2 position dofs on the
            # free surface, which is the combination a curved free boundary needs.
            if self.eq != "ale":
                eqs += ConstrainPositionsToC1Space()
            if self.eq == "ale_posc1_unc":
                eqs += UnconstrainPositionsFromC1Space() @ "top"
            eqs += IntegralObservables(area=1, intuy=var("velocity_y"),
                                       intu2=dot(var("velocity"), var("velocity")))
        elif self.eq in ("stokes_th", "stokes_cr"):
            # Stokes in the box driven by the bulk force f = (-y, x). Taylor-Hood is the mixed C2/C1 case;
            # Crouzeix-Raviart has bubble-enriched velocity and an ELEMENT-INTERNAL (discontinuous) pressure,
            # which exercises the other pressure-fixation path.
            mode = "TH" if self.eq == "stokes_th" else "CR"
            st = StokesEquations(mode=mode, dynamic_viscosity=1, bulkforce=vector(-y, x))
            eqs = st
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ _BND
            eqs += st.create_pressure_fixation(value=0)
            # NB: int(velocity_x) and int(velocity_y) are identically ZERO by symmetry for this forcing on
            # this box, so they would compare pure round-off (~1e-13) against pure round-off. The angular
            # momentum int(x*u_y - y*u_x) is what the forcing actually drives, and is the right observable.
            eqs += IntegralObservables(intcurl=x * var("velocity_y") - y * var("velocity_x"),
                                       intu2=dot(var("velocity"), var("velocity")))
        else:
            raise ValueError("unknown equation: " + str(self.eq))
        self += eqs @ "domain"
        self._add_refinement_criterion()

    def _add_refinement_criterion(self):
        # `levels` is (lo, hi) or (lo, hi, criterion). The criterion selects WHICH refinement-driving
        # equation states the requirement; the resulting mesh is meant to be the same either way, which is
        # what lets the distributed-vs-serial comparison stay meaningful across all of them.
        #
        # Why more than one: pyoomph's refinement criteria set a per-element error OVERRIDE, and those are
        # computed rank-locally -- that is what defect C was (dev_docs/adaptive_refinement.md section
        # 9.8). The dangerous shape is a criterion stated on an INTERFACE mesh ("domain/top"): a rank holds
        # halo copies of bulk elements without holding the interface elements that would override their
        # error, so the ranks disagree about what to refine. Every criterion here is exercised in exactly
        # that boundary-restricted position for that reason.
        lo, hi = self.levels[0], self.levels[1]
        crit = self.levels[2] if len(self.levels) > 2 else "level"
        h = 1.0 / self.N  # base element edge length on the unit box

        def centroid_x(e):
            nodes = e.nodes()
            return sum(n.x(0) for n in nodes) / len(nodes)

        if crit == "level":
            if lo:
                self += RefineToLevel(lo) @ "domain"
            if hi and hi != lo:
                self += RefineToLevel(hi) @ "domain/top"
        elif crit == "size":
            # Stated as a maximum element SIZE rather than a level. RefineMaxElementSize compares against
            # get_current_cartesian_nondim_size(), which is an AREA on the bulk mesh and a LENGTH on the
            # 1D interface mesh -- hence the two different thresholds for the same target level. Each is
            # set a hair below the size at the target level so rounding cannot land on the boundary.
            if lo:
                self += RefineMaxElementSize(0.99 * (h / 2 ** (lo - 1)) ** 2) @ "domain"
            if hi and hi != lo:
                self += RefineMaxElementSize(0.99 * h / 2 ** (hi - 1)) @ "domain/top"
        elif crit == "callback":
            # A per-element callback, in two positions. On the bulk it gives this campaign its only 2:1
            # interface running through the mesh INTERIOR (at x=0) rather than along a boundary, which a
            # partition cut is far more likely to lie along.
            #
            # That bulk form alone cannot detect a rank-local criterion, though: it reads nothing but the
            # element's own geometry, so a halo copy necessarily agrees with its owner. (Verified -- with
            # the error synchronisation disabled this criterion still passes, while "size" fails.) The
            # interface-restricted form below is what carries the teeth, for the reason in the note above:
            # a rank holds halo bulk elements without the interface elements that would override them.
            if lo:
                self += RefineAccordingToElement(lambda e: lo + (1 if centroid_x(e) < 0 else 0)) @ "domain"
            if hi and hi != lo:
                self += RefineAccordingToElement(lambda e: hi) @ "domain/top"
        else:
            raise ValueError("unknown refinement criterion: " + str(crit))


def solve_case(kind, eq, levels, N=4, outdir=None):
    """Solve one case and return the partition-independent measurements. The caller owns the Problem
    lifetime via the returned dict only -- the problem itself is torn down here."""
    import numpy as np
    prob = BoxProblem(kind=kind, eq=eq, levels=tuple(levels), N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        # levels may carry a trailing criterion tag (see BoxProblem._add_refinement_criterion), so take
        # the cap from the numeric entries only.
        p.max_refinement_level = max(max(levels[0], levels[1]), 1) + 1
        if eq == "stokes_cr":
            # Crouzeix-Raviart on refined triangles is ill-conditioned enough that MKL Pardiso's
            # STATIC pivoting leaves backward errors of order 1e0 and Newton diverges on the
            # resulting increments. The Jacobian is not at fault - umfpack solves the same system to
            # machine precision - so let the solver escalate its pivoting and refactorise instead.
            p.initialise()
            solver = p.get_la_solver()
            if hasattr(solver, "repair_bad_solves"):
                solver.repair_bad_solves = True
        p.solve()
        m = p.get_mesh("domain")
        conv = p.get_last_residual_convergence()
        res = {
            "maxres": float(np.max(np.abs(np.asarray(p.get_residuals())))),
            "ndof": int(p.ndof()),
            # Newton convergence history (max residual before the solve and after each iteration). The
            # meaningful Jacobian oracle is conv[1]/conv[0] -- how much ONE Newton step removes. For a
            # linear problem with an exact analytic Jacobian that ratio is ~1e-14; a constrained or hanging
            # dof that was pinned instead of being given a registered hang makes the Jacobian inconsistent
            # with the residual and the ratio collapses towards 1. Counting iterations instead would be
            # tolerance-dependent: an ill-conditioned discretisation (Crouzeix-Raviart on triangles) can
            # land at 1.9e-8 after a perfect first step and take a cosmetic second one purely because the
            # Newton tolerance is 1e-8.
            "newton_steps": max(len(conv) - 1, 0),
            "newton_conv": [float(c) for c in conv],
        }
        for name, val in m.evaluate_all_observables().items():
            res["obs_" + name] = float(val)
        return res


def case_id(kind, eq, levels):
    base = "%s-%s-%d%d" % (eq, kind, levels[0], levels[1])
    # The criterion is part of the identity: the same (kind, eq, levels) solved through a different
    # refinement criterion is a different case and must not collide with it in the results dict.
    return base if len(levels) < 3 or levels[2] == "level" else base + "-" + str(levels[2])
