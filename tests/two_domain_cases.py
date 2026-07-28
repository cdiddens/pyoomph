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

# Shared problem definitions for the COUPLED-INTERFACE adaptivity campaign. Same three-way sharing as
# box_cases.py: the serial tests (test_adaptive_interface_coupling.py), the MPI harness
# (test_mpi_interface_coupling.py) and the worker it launches under mpirun all import this one module, so
# the serial and distributed runs solve a bit-identical problem and differ only in how it is partitioned.
# It exposes the same solve_case()/case_id() interface as box_cases, so mpi_worker.py needs no changes.
#
# What is under test here is NOT the refinement engine (box_cases covers that) but the fact that two
# domains sharing an interface are adapted INDIVIDUALLY by oomph-lib. Each case therefore drives
# refinement ASYMMETRICALLY -- the criterion is stated for the "lower" domain only -- and the "upper"
# domain has no reason of its own to follow. Without Problem.enforce_interface_conformity() the
# opposite-element matcher (InterfaceMesh::connect_interface_elements_by_kdtree, which pairs interface
# elements by exact vertex-position sets) then has nothing to pair up and the run dies with
# "Cannot locate opposite element". See dev_docs/interface_refinement_coupling.md.
#
# The unit square, split at y=0.5 into "lower" and "upper" with the shared boundary named "interface":
#
#   kinds  quad / tri_left / tri_crossed  -- both domains the same family
#          mixed                          -- quads below, triangles above: the two sides of the interface
#                                            belong to DIFFERENT element families, so a facet subdivided
#                                            by a quad split has to be matched by a triangle split
#
# The measured quantities are the same partition-independent set as box_cases (gathered residual, global
# ndof, MPI-reduced integral observables) plus "nonconforming", which must be 0: the direct statement of
# the invariant, checked rather than inferred from the absence of a crash.

import math

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.navier_stokes import StokesEquations
from pyoomph.equations.ALE import LaplaceSmoothedMesh, ConnectMeshAtInterface
from pyoomph.equations.generic import (ConnectFieldsAtInterface, RefineToLevel, RefineMaxElementSize,
                                       RefineAccordingToElement, SpatialErrorEstimator)
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

MESH_KINDS = ["quad", "tri_left", "tri_crossed", "mixed"]
EQUATIONS = ["connect1", "connect2", "connect12", "ale"]
# (level applied uniformly to BOTH domains, level the asymmetric criterion drives in "lower", criterion)
LEVELS = [(0, 0, "level"), (1, 2, "level"), (1, 2, "size"), (1, 2, "callback"),
          (1, 2, "interface"), (1, 2, "estimator"), (0, 3, "level")]

_LOWER_BND = ["left", "right", "bottom"]
_UPPER_BND = ["left", "right", "top"]


class MixedTwoDomainMesh(MeshTemplate):
    """The unit square as quads for y<0.5 and triangles for y>0.5, sharing the boundary "interface".

    The point of this kind is that the two sides of the coupled interface are DIFFERENT element families.
    They still meet in matching line facets, so conformity is well-defined -- but a quad refining its
    bottom edge and a triangle refining its top edge have to end up with the same two sub-segments, which
    is exactly the property the opposite-element matcher relies on and nothing else in the suite checks.
    """

    def __init__(self, N=4):
        super().__init__()
        self.N = N

    def define_geometry(self):
        N = self.N
        half = N // 2
        lower = self.new_domain("lower")
        upper = self.new_domain("upper")
        idx = {}

        def node(i, j):
            if (i, j) not in idx:
                idx[(i, j)] = self.add_node_unique(1.0 * i / N, 1.0 * j / N)
            return idx[(i, j)]

        for i in range(N):
            for j in range(N):
                a, b, c, d = node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)
                if j < half:
                    lower.add_quad_2d_C1(a, b, c, d)
                else:
                    upper.add_tri_2d_C1(a, b, d)
                    upper.add_tri_2d_C1(a, d, c)
        for j in range(half):
            self.add_facet_to_boundary("left", [node(0, j), node(0, j + 1)])
            self.add_facet_to_boundary("right", [node(N, j), node(N, j + 1)])
        for j in range(half, N):
            self.add_facet_to_boundary("left", [node(0, j), node(0, j + 1)])
            self.add_facet_to_boundary("right", [node(N, j), node(N, j + 1)])
        for i in range(N):
            self.add_facet_to_boundary("bottom", [node(i, 0), node(i + 1, 0)])
            self.add_facet_to_boundary("top", [node(i, N), node(i + 1, N)])
            self.add_facet_to_boundary("interface", [node(i, half), node(i + 1, half)])


def make_mesh(kind, N=4):
    if kind == "mixed":
        return MixedTwoDomainMesh(N=N)
    split = False if kind == "quad" else kind.split("_", 1)[1]
    return RectangularQuadMesh(name=lambda x, y: "lower" if y < 0.5 else "upper",
                               size=1, N=N, lower_left=[0, 0], split_in_tris=split,
                               boundary_names={"lower_upper": "interface"})


def centroid_x(e):
    return e.get_Eulerian_midpoint()[0]


class TwoDomainProblem(Problem):
    def __init__(self, kind="quad", eq="connect1", levels=(1, 2, "level"), N=4):
        super().__init__()
        self.kind, self.eq, self.levels, self.N = kind, eq, tuple(levels), N

    def define_problem(self):
        self += make_mesh(self.kind, self.N)
        x = var("coordinate_x")

        if self.eq in ("connect1", "connect2", "connect12"):
            # Poisson in both domains, u tied across the interface by Lagrange multipliers. u == y is the
            # exact solution of every variant, and it is representable in every discretisation here, so
            # the field is pinned down independently of the mesh -- see solve_case's "maxuerr".
            #
            # connect12 gives the two domains DIFFERENT spaces (C2 below, C1 above). That is the case
            # where the coupling space itself has to be negotiated
            # (get_interface_field_connection_space), and where a hanging node on one side of the
            # interface meets a differently-interpolated node on the other.
            spaces = {"connect1": ("C1", "C1"), "connect2": ("C2", "C2"), "connect12": ("C2", "C1")}[self.eq]
            self += PoissonEquation(name="u", source=0, space=spaces[0]) @ "lower"
            self += PoissonEquation(name="u", source=0, space=spaces[1]) @ "upper"
            self += DirichletBC(u=0) @ "lower/bottom"
            self += DirichletBC(u=1) @ "upper/top"
            self += ConnectFieldsAtInterface("u") @ "lower/interface"
            self += IntegralObservables(intu=var("u"), intu2=var("u") ** 2) @ "lower"
            self += IntegralObservables(intu=var("u"), intu2=var("u") ** 2) @ "upper"
        elif self.eq == "ale":
            # Stokes on a Laplace-smoothed mesh in both domains, with the mesh POSITIONS coupled across the
            # interface (ConnectMeshAtInterface) as well as the velocity. Two things this adds over the
            # Poisson cases: the interface geometry is itself an unknown, so the facet positions the
            # conformity machinery keys on are solution-dependent; and the position dofs carry their own
            # hanging-node structure.
            for dom, bnds in (("lower", _LOWER_BND), ("upper", _UPPER_BND)):
                eqs = StokesEquations(mode="TH", dynamic_viscosity=1, bulkforce=vector(0, -1))
                eqs += LaplaceSmoothedMesh()
                eqs += DirichletBC(mesh_x=True, mesh_y=True) @ bnds
                eqs += DirichletBC(velocity_x=0, velocity_y=0) @ bnds
                eqs += IntegralObservables(area=1, intu2=dot(var("velocity"), var("velocity")))
                self += eqs @ dom
            self += DirichletBC(pressure=0) @ "lower/bottom"
            self += ConnectFieldsAtInterface(["velocity_x", "velocity_y"]) @ "lower/interface"
            self += ConnectMeshAtInterface() @ "lower/interface"
        else:
            raise ValueError("unknown equation: " + str(self.eq))

        self._add_refinement_criterion()

    def _add_refinement_criterion(self):
        """The asymmetric part: state a refinement requirement for "lower" that "upper" cannot see.

        Which criterion is used matters, and not for the reason one might expect. What decides whether a
        criterion reaches the opposite domain is not the criterion itself but WHERE it is stated:

          * on the bulk ("level", "size", "callback", "estimator") -- reads only the lower domain's own
            elements. The upper domain is told nothing, so the two sides diverge. These are the cases
            that fail without enforce_interface_conformity(); verified by running the suite under
            PYOOMPH_DISABLE_INTERFACE_CONFORMITY=1.
          * on the interface ("interface") -- InterfaceMesh._override_bulk_errors_where_necessary already
            pushes the error onto BOTH adjacent bulk elements, so this one is symmetric by construction
            and passes with or without the fix. It is kept because that symmetry is a property that must
            not silently regress, not because it exercises the new code.
        """
        lo, hi = self.levels[0], self.levels[1]
        crit = self.levels[2] if len(self.levels) > 2 else "level"
        if lo:
            # Applied to BOTH domains: the conforming starting point the asymmetry is measured against.
            self += RefineToLevel(lo) @ "lower"
            self += RefineToLevel(lo) @ "upper"
        if not hi or hi == lo:
            return
        h = 1.0 / self.N
        if crit == "level":
            self += RefineToLevel(hi) @ "lower"
        elif crit == "size":
            # Cartesian element SIZE, i.e. an AREA in 2d -- and the base area depends on how the family
            # fills a cell of the h x h grid. Deriving it from h alone (as if everything were a quad)
            # made this criterion a silent no-op on triangles: their elements were already below the
            # threshold at level lo, so nothing was refined and the case tested nothing at all.
            base = {"quad": h * h, "tri_left": 0.5 * h * h, "tri_crossed": 0.25 * h * h,
                    "mixed": h * h}[self.kind]  # "mixed" is quads BELOW, and the criterion is on "lower"
            # Just under the area at level hi-1: those elements refine, level-hi ones do not.
            self += RefineMaxElementSize(0.99 * base / 4 ** (hi - 1)) @ "lower"
        elif crit == "callback":
            # Position-dependent, so the refinement level jumps ALONG the interface as well as across it:
            # the upper domain then has to follow a pattern, not just a uniform level.
            self += RefineAccordingToElement(lambda e: hi if centroid_x(e) < 0.5 else lo) @ "lower"
        elif crit == "interface":
            self += RefineToLevel(hi) @ "lower/interface"
        elif crit == "estimator":
            # The realistic case: no explicit level anywhere, just a Z2 error estimator on a field that is
            # only sharp in the lower domain. Nothing states a level, so the two domains genuinely
            # disagree about how fine the interface should be.
            self += PoissonEquation(name="w", source=1.0 / (0.01 + var("coordinate_x") ** 2),
                                    space="C2") @ "lower"
            self += DirichletBC(w=0) @ ["lower/bottom", "lower/left", "lower/right"]
            self += SpatialErrorEstimator(w=1) @ "lower"
        else:
            raise ValueError("unknown refinement criterion: " + str(crit))


def solve_case(kind, eq, levels, N=4, outdir=None):
    """Solve one case and return the partition-independent measurements.

    `kind` also accepts the four-domain layouts (see FOUR_DOMAIN_KINDS at the bottom of this file), for
    which `eq` and `levels` are ignored -- that keeps the MPI harness, which drives everything through
    solve_case/case_id, working for both topologies without needing to know about either.
    """
    import numpy as np
    if kind in FOUR_DOMAIN_KINDS:
        return solve_four_domain_case(kind, outdir=outdir)
    prob = TwoDomainProblem(kind=kind, eq=eq, levels=tuple(levels), N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.max_refinement_level = max(levels[0], levels[1]) + 1
        p.initialise()
        # Repairs done while setting the mesh up (the uneven part of the initial uniform refinement is
        # deliberately applied after distribute, so it genuinely has to be repaired). Only what happens
        # from here on is a statement about the ADAPT path.
        repairs_at_init = p._interface_conformity_repairs
        # The refinement is driven by explicit criteria, not by a converging error estimate, so a couple
        # of adapt steps is enough for all of them to have taken effect.
        for _ in range(3):
            p.solve(spatial_adapt=1)
        repairs_during_adapt = p._interface_conformity_repairs - repairs_at_init

        # The Newton history of that last solve is useless as a Jacobian oracle: the field was already
        # converged from the previous pass, so it starts at machine zero and one step "reduces" nothing.
        # Wipe the coupled field on the FINAL mesh and solve it once more, from a residual of O(1).
        # Done nodally, so it stays correct per rank on a distributed mesh. Not done for "ale", whose
        # dofs include the nodal POSITIONS -- zeroing those collapses the mesh rather than resetting it.
        if eq in ("connect1", "connect2", "connect12"):
            for dom in ("lower", "upper"):
                m = p.get_mesh(dom)
                iu = m.get_nodal_field_indices()["u"]
                for n in m.nodes():
                    if not n.is_pinned(iu):
                        n.set_value(iu, 0.0)
            p.solve()
        conv = p.get_last_residual_convergence()
        res = {
            "maxres": float(np.max(np.abs(np.asarray(p.get_residuals())))),
            "ndof": int(p.ndof()),
            "newton_steps": max(len(conv) - 1, 0),
            "newton_conv": [float(c) for c in conv],
            # The invariant itself, stated rather than inferred. Zero means both sides of the interface
            # carry identical boundary facets, which is precisely what the opposite-element matcher needs.
            # Collective under MPI, so every rank calls it and every rank gets the same number.
            "nonconforming": int(p.check_interface_conformity(throw_on_mismatch=False, when="end of case")),
            # How many elements the post-adapt repair had to refine AFTER the fact. Zero is the good
            # case: it means the two sides agreed before either acted, rather than one of them being
            # merged away and then refined back -- a round trip that is correct but re-interpolates the
            # sons from the merged father and loses the fine-scale solution.
            "repairs_during_adapt": int(repairs_during_adapt),
        }
        for dom in ("lower", "upper"):
            for name, val in p.get_mesh(dom).evaluate_all_observables().items():
                res["obs_" + dom + "_" + name] = float(val)
        if eq in ("connect1", "connect2", "connect12"):
            # u == y exactly. A torn interface shows up here long before it shows up in an integral: a
            # mis-paired opposite element leaves the two sides connected to the WRONG neighbour, which
            # this catches even when the residual is happily converged.
            #
            # NB the tolerance. The Lagrange-multiplier formulation of ConnectFieldsAtInterface does not
            # reproduce a linear field to machine zero even on a single UNREFINED, unadapted mesh (~3e-10
            # on 81 dofs), so this bound is about the coupling, not about the refinement. A torn interface
            # misses it by orders of magnitude.
            worst = 0.0
            for dom in ("lower", "upper"):
                m = p.get_mesh(dom)
                # Look the index up by NAME. The "estimator" criterion adds a second field to the lower
                # domain, so index 0 is not necessarily u -- reading it positionally would silently
                # measure the wrong field (and did, until this was keyed on the name).
                iu = m.get_nodal_field_indices()["u"]
                for n in m.nodes():
                    worst = max(worst, abs(n.value(iu) - n.x(1)))
            res["maxuerr"] = float(worst)
        return res


def case_id(kind, eq, levels):
    if kind in FOUR_DOMAIN_KINDS:
        return kind
    crit = levels[2] if len(levels) > 2 else "level"
    return "%s-%s-%d%d-%s" % (eq, kind, levels[0], levels[1], crit)


# --- Four domains meeting at a cross point ----------------------------------------------------------
#
#         A | B
#         --+--
#         C | D
#
# A different topology from everything above, and the two things it adds are worth stating.
#
# The coupling graph is a CYCLE (A-B-D-C-A), not a chain, so the reconciliation has to close a loop
# rather than propagate along one. D shares no interface with A at all -- they touch only at the cross
# point -- so a refinement demand raised in A can only reach D by travelling around the cycle.
#
# And the cross point itself is four DISTINCT nodes, one per domain, tied pairwise by four Lagrange
# multipliers (A=B, A=C, B=D, C=D). Only three of those four constraints are independent; the fourth
# follows. That is a genuine over-constraint at a single point, and it is
# ConnectFieldsAtInterface.pin_redundant_lagrange_multipliers that has to notice.
#
# Exact solution u = y everywhere, so a mis-paired interface shows up directly as a nodal error.

FOUR_DOMAIN_KINDS = ["four_corner", "four_diagonal", "four_away"]
_FOUR_DOMS = ["A", "B", "C", "D"]
# RectangularQuadMesh names an auto-generated internal interface after the two domains it separates.
_FOUR_IFACES = [("A", "A_B"), ("A", "A_C"), ("B", "B_D"), ("C", "C_D")]


def _four_domain_of(x, y):
    if y >= 0.5:
        return "A" if x < 0.5 else "B"
    return "C" if x < 0.5 else "D"


class FourDomainProblem(Problem):
    def __init__(self, kind="four_corner", N=4):
        super().__init__()
        self.kind, self.N = kind, N

    def define_problem(self):
        self += RectangularQuadMesh(N=[self.N, self.N], size=[1, 1], name=_four_domain_of)
        for d in _FOUR_DOMS:
            self += PoissonEquation(name="u", source=0, space="C1") @ d
            self += IntegralObservables(intu=var("u"), intu2=var("u") ** 2) @ d
        self += DirichletBC(u=0) @ "C/bottom"
        self += DirichletBC(u=0) @ "D/bottom"
        self += DirichletBC(u=1) @ "A/top"
        self += DirichletBC(u=1) @ "B/top"
        for dom, nm in _FOUR_IFACES:
            self += ConnectFieldsAtInterface("u") @ (dom + "/" + nm)

        if self.kind == "four_corner":
            # Refinement concentrated ON the cross point, in A only: the level jump sits exactly where
            # all four domains meet, and both of A's interfaces have to carry it into B and C, and then
            # around to D.
            def lev(e):
                x, y = e.get_Eulerian_midpoint()[0], e.get_Eulerian_midpoint()[1]
                d = max(abs(x - 0.5), abs(y - 0.5))
                return 3 if d < 0.2 else (1 if d < 0.35 else 0)
            self += RefineAccordingToElement(lev) @ "A"
        elif self.kind == "four_diagonal":
            # The DIAGONAL pair driven, to different levels: A and D are each pulled by two neighbours
            # that disagree with each other about how fine the interface should be.
            self += RefineAccordingToElement(lambda e: 3) @ "B"
            self += RefineAccordingToElement(lambda e: 2) @ "C"
        elif self.kind == "four_away":
            # Refinement in A but AWAY from every interface. Nothing may propagate: this is the case
            # that separates "the neighbours follow where they must" from "the neighbours follow
            # always", which the other two cannot distinguish.
            def lev(e):
                x, y = e.get_Eulerian_midpoint()[0], e.get_Eulerian_midpoint()[1]
                return 3 if (x < 0.2 and y > 0.8) else 0
            self += RefineAccordingToElement(lev) @ "A"
        else:
            raise ValueError("unknown four-domain kind: " + str(self.kind))


def solve_four_domain_case(kind, outdir=None):
    import numpy as np
    prob = FourDomainProblem(kind=kind)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.max_refinement_level = 3
        p.initialise()
        repairs_at_init = p._interface_conformity_repairs
        for _ in range(4):
            p.solve(spatial_adapt=1)
        # Wipe and re-solve on the final mesh, so the Newton history is a Jacobian oracle rather than
        # the tail of an already-converged solve (see solve_case).
        for d in _FOUR_DOMS:
            m = p.get_mesh(d)
            iu = m.get_nodal_field_indices()["u"]
            for n in m.nodes():
                if not n.is_pinned(iu):
                    n.set_value(iu, 0.0)
        p.solve()
        conv = p.get_last_residual_convergence()
        worst = 0.0
        levels = {}
        for d in _FOUR_DOMS:
            m = p.get_mesh(d)
            iu = m.get_nodal_field_indices()["u"]
            for n in m.nodes():
                worst = max(worst, abs(n.value(iu) - n.x(1)))
            levels[d] = sorted({e.refinement_level() for e in m.elements()})
        res = {
            "maxres": float(np.max(np.abs(np.asarray(p.get_residuals())))),
            "ndof": int(p.ndof()),
            "newton_steps": max(len(conv) - 1, 0),
            "newton_conv": [float(c) for c in conv],
            "nonconforming": int(p.check_interface_conformity(False, "four-domain")),
            "repairs_during_adapt": int(p._interface_conformity_repairs - repairs_at_init),
            "maxuerr": float(worst),
            "maxlevel": {d: max(levels[d]) for d in _FOUR_DOMS},
        }
        for d in _FOUR_DOMS:
            for name, val in p.get_mesh(d).evaluate_all_observables().items():
                res["obs_" + d + "_" + name] = float(val)
        return res
