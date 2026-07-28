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

# Shared problem definitions for the MPI adaptivity tests. Imported BOTH by the pytest harness
# (tests/test_mpi_adaptivity.py, which runs them serially in-process to produce the reference) and by the
# worker that the harness launches under `mpirun -n N ... --distribute` (tests/mpi_worker.py). Keeping the
# definitions in one place is what makes the distributed-vs-serial comparison meaningful: both sides solve
# a bit-identical problem and differ only in how it is partitioned.
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
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.navier_stokes import StokesEquations
from pyoomph.meshes.mesh import MeshTemplate
from pyoomph.meshes.simplemeshes import RectangularQuadMesh

_BND = ["left", "right", "top", "bottom"]

MESH_KINDS = ["quad", "tri_left", "tri_crossed", "mixed"]
EQUATIONS = ["poisson1", "poisson2", "mixed12", "stokes_th", "stokes_cr"]
LEVELS = [(0, 0), (1, 1), (1, 3)]


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
        elif self.eq == "mixed12":
            # u on C2 driving v on C1: mixed continuous spaces on one mesh, so C1 owns a separate hang slot.
            eqs = PoissonEquation(source=1, space="C2")
            eqs += PoissonEquation(source=var("u"), space="C1", name="v")
            eqs += DirichletBC(u=0, v=0) @ _BND
            eqs += IntegralObservables(intu=var("u"), intv=var("v"), intu2=var("u") ** 2)
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
        lo, hi = self.levels
        if lo:
            self += RefineToLevel(lo) @ "domain"
        if hi and hi != lo:
            self += RefineToLevel(hi) @ "domain/top"


def solve_case(kind, eq, levels, N=4, outdir=None):
    """Solve one case and return the partition-independent measurements. The caller owns the Problem
    lifetime via the returned dict only -- the problem itself is torn down here."""
    import numpy as np
    prob = BoxProblem(kind=kind, eq=eq, levels=tuple(levels), N=N)
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.max_refinement_level = max(max(levels), 1) + 1
        p.solve()
        m = p.get_mesh("domain")
        res = {
            "maxres": float(np.max(np.abs(np.asarray(p.get_residuals())))),
            "ndof": int(p.ndof()),
        }
        for name, val in m.evaluate_all_observables().items():
            res["obs_" + name] = float(val)
        return res


def case_id(kind, eq, levels):
    return "%s-%s-%d%d" % (eq, kind, levels[0], levels[1])
