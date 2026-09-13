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

# Meshes an anisotropic order-2 geometry, which is what makes Gmsh's high-order optimizer give up.
#
# Run as a worker on purpose: the failure this guards against is not an exception but SIGABRT, and a
# test that provokes it in-process takes the whole pytest run down with it. See
# tests/test_gmsh_high_order_optimize.py.

import argparse
import sys

import gmsh

from pyoomph import Problem, DirichletBC
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.meshes.gmsh import GmshTemplate


class ThinTorus(GmshTemplate):
    """Major radius 1, minor radius 0.03, meshed at order 2.

    Every element around the tube is long and thin, and curving them to the torus leaves the
    optimizer with scaled Jacobians it cannot lift to Mesh.HighOrderThresholdMin - it ends on
    "Failed to reach critical value in pass 0 for measure(s): ScaledJac".
    """

    def __init__(self, high_order_optimize=None):
        super().__init__()
        self.kernel = "occ"           # chosen before the geometry object is built
        if high_order_optimize is not None:
            self.high_order_optimize = high_order_optimize

    def define_geometry(self):
        self.mesh_mode = "tetras"
        self.default_resolution = 0.5
        env = self._geom.env
        volume = env.addTorus(0, 0, 0, 1, 0.03)
        env.synchronize()
        self._store_name("domain", GmshTemplate.GmshFakeEntry(volume, (3, volume)))
        for _dim, tag in gmsh.model.getBoundary([(3, volume)], combined=False, oriented=False):
            self._store_name("surface", GmshTemplate.GmshFakeEntry(tag, (2, tag)))
        self._maxdim = 3


class _TorusProblem(Problem):
    def __init__(self, high_order_optimize):
        super().__init__()
        self._hoo = high_order_optimize

    def define_problem(self):
        self += ThinTorus(self._hoo)
        self += (PoissonEquation(source=1, space="C2") + DirichletBC(u=0) @ "surface") @ "domain"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--high-order-optimize", type=int, default=None)
    args = parser.parse_args()
    problem = _TorusProblem(args.high_order_optimize)
    problem.set_output_directory(args.outdir)
    problem.quiet()
    problem.initialise()
    print("NNODE %d" % problem.get_mesh("domain").nnode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
