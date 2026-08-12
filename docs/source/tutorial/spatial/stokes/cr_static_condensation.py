#  Lid-driven cavity with Crouzeix-Raviart elements, with and without static condensation of the
#  element-local degrees of freedom. The physics is beside the point here - what the script reports is
#  how large a system the linear solver is handed, and how long it takes to factorize it.
#
#      python3 cr_static_condensation.py                  # 2d, plain
#      python3 cr_static_condensation.py --condense       # 2d, bubbles and pressure gradients removed
#      python3 cr_static_condensation.py --3d --condense  # the same in 3d
#      python3 cr_static_condensation.py --quads --condense  # on quadrilaterals instead of triangles
#      python3 cr_static_condensation.py --N 30           # a finer mesh

import sys

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.navier_stokes import NavierStokesEquations, NoSlipBC
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, CuboidBrickMesh

CONDENSE = "--condense" in sys.argv
THREE_D = "--3d" in sys.argv
QUADS = "--quads" in sys.argv          # tensor-product elements instead of simplices
N = int(sys.argv[sys.argv.index("--N") + 1]) if "--N" in sys.argv else (8 if THREE_D else 24)


class TetCubeMesh(GmshTemplate):
    """The unit cube, meshed with tetrahedra, with the same boundary names as CuboidBrickMesh."""

    def __init__(self, N=8):
        super().__init__()
        self.N = N

    def define_geometry(self):
        self.mesh_mode = "tetras"
        self.default_resolution = 1.0 / self.N
        p = {(x, y, z): self.point(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)}
        # One plane surface per face, named as the brick mesh names them.
        faces = {"left": [(0, y, z) for y, z in ((0, 0), (1, 0), (1, 1), (0, 1))],
                 "right": [(1, y, z) for y, z in ((0, 0), (1, 0), (1, 1), (0, 1))],
                 "front": [(x, 0, z) for x, z in ((0, 0), (1, 0), (1, 1), (0, 1))],
                 "back": [(x, 1, z) for x, z in ((0, 0), (1, 0), (1, 1), (0, 1))],
                 "bottom": [(x, y, 0) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))],
                 "top": [(x, y, 1) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]}
        surfs = []
        for name, corners in faces.items():
            pts = [p[c] for c in corners]
            lines = [self.line(pts[i], pts[(i + 1) % 4]) for i in range(4)]
            surfs += self.plane_surface(*lines, name=name)
        self.volume(*surfs, name="domain")


class DrivenCavity(Problem):
    def __init__(self):
        super().__init__()
        self.N = N
        self.reynolds = 100  # only used to pick the viscosity, the cavity is driven at unit speed

    def define_problem(self):
        # Simplices by default, tensor-product elements with --quads. Both have exactly one
        # element-interior velocity node to condense: the added cubic bubble on a triangle or
        # tetrahedron, the centroid node of the "C2" element on a quadrilateral or brick.
        if THREE_D:
            self += CuboidBrickMesh(N=self.N, size=[1, 1, 1]) if QUADS else TetCubeMesh(N=self.N)
            lid, walls = "top", ["bottom", "left", "right", "front", "back"]
            driven = dict(velocity_x=1, velocity_y=0, velocity_z=0)
        else:
            self += RectangularQuadMesh(N=self.N, size=[1, 1],
                                        split_in_tris=False if QUADS else "crossed")
            lid, walls = "top", ["bottom", "left", "right"]
            driven = dict(velocity_x=1, velocity_y=0)

        # The predefined equations, with mode="CR": velocity on "C2TB", pressure on "DL". Every
        # velocity boundary is prescribed, so the pressure is only determined up to a constant and one
        # of its degrees of freedom has to be fixed.
        eqs = NavierStokesEquations(mass_density=self.reynolds, dynamic_viscosity=1,
                                    mode="CR").with_pressure_fixation()

        eqs += DirichletBC(**driven) @ lid
        for w in walls:
            eqs += NoSlipBC() @ w

        if CONDENSE:
            # The classical Crouzeix-Raviart elimination. Both halves are needed: neither the bubble
            # velocities nor the pressure gradients are invertible on their own, and the constant
            # pressure mode has to stay a global unknown - which is exactly what "DL_gradients" means.
            eqs += StaticCondensation(velocity="bubble", pressure="DL_gradients")

        self += eqs @ "domain"


if __name__ == "__main__":
    with DrivenCavity() as problem:
        problem.initialise()
        problem.solve()
        stats = problem._get_static_condensation_stats()
        n = problem.ndof()
        print(f"{'3d' if THREE_D else '2d'} cavity, N={N}")
        print(f"  degrees of freedom     : {n}")
        if stats.get("n_selected", 0):
            print(f"  condensed away         : {stats['n_selected']} in {stats['n_components']} "
                  f"blocks of at most {stats['component_size_max']}")
            print(f"  non-zeros              : {stats['full_nnz']} -> {stats['condensed_nnz']}")
            print(f"  seen by the solver     : {n - stats['n_selected']}")
        problem.output()
