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

# 3D counterpart of box_cases.py: the same equation systems on the box [-0.5,0.5]^3, discretised by
# MixedBoxMesh3D (see box_mesh_3d.py) in every geometrically possible combination of bricks, tetrahedra,
# wedges and pyramids. Shared verbatim between the serial tests (test_adaptive_3d_campaign.py) and the MPI
# harness (test_mpi_adaptivity_3d.py), for the same reason as in 2D.
#
# Boundary-condition pattern, mirroring the 2D campaign: Dirichlet on five walls and a NATURAL (zero-flux)
# condition on "top", which is also the wall carrying the refinement band. This matters twice over:
#   * it leaves live dofs on "top", so a boundary UnconstrainFieldsFromC1Space there is not a silent no-op;
#   * with u and v carrying the same conditions, the Green identity int(v) == int(u^2) is exact whenever the
#     two live in the same discrete space, which is the sharp oracle for the C1 constraint.
#
# Refinement envelope: base 2x2x2 cells with uniform level 1 plus a level-2 band on "top". Measured (see
# dev_docs/adaptive_refinement.md) to stay in the low thousands of elements and well under 1 GB, so it is
# safe for a 2-rank MPI run and leaves room for more than one refinement level, as required.

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.navier_stokes import StokesEquations
from pyoomph.equations.ALE import (LaplaceSmoothedMesh, ConstrainPositionsToC1Space,
                                   UnconstrainPositionsFromC1Space)

from box_mesh_3d import MixedBoxMesh3D, ALL_LAYOUTS, MIXED_LAYOUTS, PURE_LAYOUTS

_ALL_WALLS = ["left", "right", "front", "back", "bottom", "top"]
_SIDE_WALLS = ["left", "right", "front", "back", "bottom"]  # everything except "top"

MESH_KINDS = ALL_LAYOUTS
EQUATIONS = ["poisson1", "poisson2", "mixed12", "constrain12", "unconstrain12", "neumann",
             "stokes_th", "ale", "ale_posc1", "ale_posc1_unc"]
LEVELS = [(0, 0), (1, 1), (1, 2)]

J_EVAP = 0.1


class BoxProblem3D(Problem):
    def __init__(self, kind="hex", eq="poisson1", levels=(1, 2), N=2):
        super().__init__()
        self.kind, self.eq, self.levels, self.N = kind, eq, levels, N

    def define_problem(self):
        self += MixedBoxMesh3D(kind=self.kind, N=self.N)
        x, y, z = var(["coordinate_x", "coordinate_y", "coordinate_z"])
        if self.eq in ("poisson1", "poisson2"):
            eqs = PoissonEquation(source=1, space="C1" if self.eq == "poisson1" else "C2")
            eqs += DirichletBC(u=0) @ _SIDE_WALLS
            eqs += IntegralObservables(intu=var("u"), intu2=var("u") ** 2)
        elif self.eq in ("mixed12", "constrain12", "unconstrain12"):
            eqs = PoissonEquation(source=1, space="C2")
            eqs += PoissonEquation(source=var("u"), space="C1", name="v")
            eqs += DirichletBC(u=0, v=0) @ _SIDE_WALLS
            if self.eq != "mixed12":
                eqs += ConstrainFieldsToC1Space("u")
            if self.eq == "unconstrain12":
                eqs += UnconstrainFieldsFromC1Space("u") @ "top"
            eqs += IntegralObservables(intu=var("u"), intv=var("v"), intu2=var("u") ** 2)
        elif self.eq == "neumann":
            # Constant flux on a wall that is NOT refined ("right") and a spatially varying one on the
            # refined wall ("top"), so the flux integration meets face elements on hanging-node parents of
            # every shape present in the layout.
            eqs = PoissonEquation(source=1, space="C2")
            eqs += DirichletBC(u=0) @ ["left", "front", "bottom"]
            eqs += NeumannBC(u=1) @ "right"
            eqs += NeumannBC(u=x) @ "top"
            eqs += IntegralObservables(intu=var("u"), intu2=var("u") ** 2)
        elif self.eq == "stokes_th":
            # Stokes driven by the bulk force f = (-y, x, 0): the 3D analogue of the 2D box, a rotation about
            # the z axis. Taylor-Hood = C2 velocity + C1 pressure, i.e. mixed continuous spaces on one mesh.
            st = StokesEquations(mode="TH", dynamic_viscosity=1, bulkforce=vector(-y, x, 0))
            eqs = st
            eqs += DirichletBC(velocity_x=0, velocity_y=0, velocity_z=0) @ _ALL_WALLS
            eqs += st.create_pressure_fixation(value=0)
            eqs += IntegralObservables(intcurl=x * var("velocity_y") - y * var("velocity_x"),
                                       intu2=dot(var("velocity"), var("velocity")))
        elif self.eq in ("ale", "ale_posc1", "ale_posc1_unc"):
            st = StokesEquations(mode="TH", dynamic_viscosity=1)
            eqs = st
            eqs += LaplaceSmoothedMesh()
            eqs += DirichletBC(mesh_x=True, mesh_y=True, mesh_z=True) @ _SIDE_WALLS
            eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "top"
            eqs += DirichletBC(velocity_x=0, velocity_y=0, velocity_z=0) @ _SIDE_WALLS
            eqs += DirichletBC(velocity_x=0, velocity_y=0, velocity_z=J_EVAP) @ "top"
            eqs += DirichletBC(pressure=0) @ "bottom"
            # See the 2D counterpart in box_cases.py.
            if self.eq != "ale":
                eqs += ConstrainPositionsToC1Space()
            if self.eq == "ale_posc1_unc":
                eqs += UnconstrainPositionsFromC1Space() @ "top"
            eqs += IntegralObservables(volume=1, intuz=var("velocity_z"),
                                       intu2=dot(var("velocity"), var("velocity")))
        else:
            raise ValueError("unknown equation: " + str(self.eq))
        self += eqs @ "domain"
        lo, hi = self.levels
        if lo:
            self += RefineToLevel(lo) @ "domain"
        if hi and hi != lo:
            self += RefineToLevel(hi) @ "domain/top"


def solve_case(kind, eq, levels, N=2, outdir=None, linear_solver=None):
    # linear_solver overrides the default backend for this one case. Meant for the rare matrix where
    # the default is accurate enough to be a perfectly good answer but not accurate enough to certify
    # a Jacobian against a machine-zero tolerance -- see _EXACT_SOLVER_CASES in
    # test_adaptive_3d_campaign.py, which is the only caller that passes it. Left None everywhere else,
    # deliberately: which solver a case uses is part of what these runs measure.
    import numpy as np
    with BoxProblem3D(kind=kind, eq=eq, levels=tuple(levels), N=N) as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        if linear_solver is not None:
            p.set_linear_solver(linear_solver)
        p.max_refinement_level = max(max(levels), 1) + 1
        p.solve()
        m = p.get_mesh("domain")
        conv = p.get_last_residual_convergence()
        n_facets, n_bnd, n_int, max_inc = list(m.facet_adjacency_summary())
        res = {
            "maxres": float(np.max(np.abs(np.asarray(p.get_residuals())))),
            "ndof": int(p.ndof()),
            "newton_steps": max(len(conv) - 1, 0),
            "newton_conv": [float(c) for c in conv],
            # Conformity of the (refined) mesh: every facet is shared by at most two elements and is either
            # a boundary or an interior facet. A torn interface shows up here as a facet with incidence 1
            # that is not on the boundary. Cheap, and independent of the solve.
            "manifold": bool(max_inc == 2 and n_bnd + n_int == n_facets),
        }
        for name, val in m.evaluate_all_observables().items():
            res["obs_" + name] = float(val)
        return res


def case_id(kind, eq, levels):
    return "3d-%s-%s-%d%d" % (eq, kind, levels[0], levels[1])
