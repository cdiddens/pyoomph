#  Free oscillations of a droplet in 3d, used here to count degrees of freedom rather than to do
#  physics: Navier-Stokes on tetrahedral MINI elements (velocity C1TB, pressure C1) on an ALE moving
#  mesh, with the two ways of reducing the size of the system switched on and off by the two flags
#  below. Run it as
#
#      python3 oscillating_droplet_dofs.py                 # plain
#      python3 oscillating_droplet_dofs.py --constrain     # tie the mesh bubbles to C1
#      python3 oscillating_droplet_dofs.py --condense      # condense the velocity bubbles
#      python3 oscillating_droplet_dofs.py --constrain --condense
#
#  and compare what it prints about the size of the system with what the solver reports per step.

import sys

from pyoomph import *
from pyoomph.expressions.units import *
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *
from pyoomph.generic.mpi import get_mpi_nproc
from pyoomph._pyoomph_core import set_detect_inverted_elements

CONSTRAIN_POSITIONS = "--constrain" in sys.argv
CONDENSE_BUBBLES = "--condense" in sys.argv


class BallMesh(GmshTemplate):
    """A full sphere of tetrahedra, built as a single OpenCASCADE ball."""

    def __init__(self, radius, resolution=0.22):
        super().__init__()
        self.kernel = "occ"  # must be set before define_geometry: the kernel is created around it
        self.radius, self.resolution = radius, resolution

    def define_geometry(self):
        self.mesh_mode = "tetras"
        self.default_resolution = self.resolution * self.nondim_size(self.radius)
        self.sphere(self.point(0, 0, 0), self.radius, surface_name="interface", name="droplet")


class OscillatingDroplet(Problem):
    def __init__(self):
        super().__init__()
        self.radius = 1 * milli * meter
        self.mass_density = 1000 * kilogram / meter**3
        self.dynamic_viscosity = 1 * milli * pascal * second
        self.surface_tension = 72 * milli * newton / meter
        self.resolution = 0.22

    def capillary_time(self):
        return square_root(self.mass_density * self.radius**3 / self.surface_tension)

    def define_problem(self):
        R, Tc = self.radius, self.capillary_time()
        self.set_scaling(spatial=R, temporal=Tc)
        self.set_scaling(velocity=R / Tc, pressure=self.surface_tension / R)

        self += BallMesh(R, self.resolution)

        # MINI: velocity on C1TB (linear plus one bubble per tetrahedron), pressure on C1.
        eqs = NavierStokesEquations(mass_density=self.mass_density,
                                    dynamic_viscosity=self.dynamic_viscosity, mode="mini")
        eqs += PseudoElasticMesh()
        eqs += NavierStokesFreeSurface(surface_tension=self.surface_tension) @ "interface"

        # An l=2 deformation to set the droplet oscillating, written as a solid harmonic so that it
        # stays regular at the centre and can be applied to the whole mesh.
        X = var("lagrangian")
        deform = 0.12 * (3 * X[2]**2 - dot(X, X)) / (2 * R**2)
        eqs += InitialCondition(mesh_x=X[0] * (1 + deform), mesh_y=X[1] * (1 + deform),
                                mesh_z=X[2] * (1 + deform))

        if CONSTRAIN_POSITIONS:
            # Removes the mesh-position bubble of every tetrahedron from the system. This is a
            # change of the discretization of the mesh motion, not a pure saving - see the text.
            eqs += ConstrainPositionsToC1Space()

        if CONDENSE_BUBBLES:
            # Exact, but serial only: an MPI run is refused rather than silently ignored, so only
            # ask for it when there is a single process.
            if get_mpi_nproc() > 1:
                print("More than one process: leaving static condensation off, it is serial only.")
            else:
                eqs += StaticCondensation(velocity="bubble")

        eqs += MeshFileOutput()
        self += eqs @ "droplet"


def report(problem):
    """Print what the linear solver will actually be handed."""
    mesh = problem.get_mesh("droplet")
    ndof = problem.ndof()
    stats = problem._get_static_condensation_stats()
    condensed = stats.get("n_selected", 0)
    print(f"tetrahedra              : {mesh.nelement()}")
    print(f"degrees of freedom      : {ndof}")
    if condensed:
        print(f"of these condensed away : {condensed} "
              f"in {stats['n_components']} blocks of at most {stats['component_size_max']}")
        print(f"non-zeros in the matrix : {stats['full_nnz']} -> {stats['condensed_nnz']}")
        print(f"seen by the solver      : {ndof - condensed}")


if __name__ == "__main__":
    set_detect_inverted_elements(True)
    with OscillatingDroplet() as problem:
        problem.initialise()
        Tc = problem.capillary_time()
        # One step is enough to have the condensation plan built and the solver report its timings.
        problem.run(0.1 * Tc, startstep=0.02 * Tc, maxstep=0.02 * Tc, outstep=False,
                    temporal_error=1.0)
        report(problem)
