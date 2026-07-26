#  von Kármán vortex street: incompressible flow past a cylinder in a channel.
#
#  This is the classic Schäfer & Turek (1996) 2D benchmark ("flow around a cylinder").
#  A parabolic inflow drives an incompressible Navier-Stokes flow past a circular
#  cylinder mounted slightly off the channel centerline. At Re = 100 the wake is
#  linearly unstable and sheds a periodic train of counter-rotating vortices - the
#  von Kármán vortex street.
#
#  Setup (all SI):
#     channel        : L = 2.2 m  x  H = 0.41 m
#     cylinder       : centre (0.2, 0.2) m, diameter D = 0.1 m  (=> off-centre => sheds)
#     fluid          : rho = 1 kg/m^3, mu = 1e-3 Pa s
#     inflow         : parabolic, peak U_m = 1.5 m/s  => mean U = 1.0 m/s
#     Reynolds number: Re = rho * U * D / mu = 100
#     Strouhal number: St ~ 0.30  => shedding frequency f ~ 3 Hz (period ~ 0.33 s)
#
#  Discretisation:
#     Taylor-Hood (P2/P1) triangles. The mesh is graded via a gmsh size callback:
#     fine on the cylinder surface, fine in a widening wake band downstream, coarse
#     towards the inlet/outlet far field. This yields ~2.0e4 degrees of freedom.
#
#  Outputs (in ./von_karman_vortex_street/):
#     - VTU/PVD files (velocity + pressure) for ParaView, via MeshFileOutput
#     - forces.txt : time series of the drag and lift force per unit span on the
#       cylinder, computed by integrating the fluid traction. The lift oscillation
#       is the fingerprint of the vortex street (and gives the Strouhal number).

from pyoomph import *
from pyoomph.equations.navier_stokes import *
from pyoomph.expressions import *
from pyoomph.expressions.units import *
from pyoomph.meshes.gmsh import GmshSizeCallback
import math

# --- Geometry constants (metres). Used by both the mesh and the size callback. ---
L, H = 2.2, 0.41            # channel length and height
cx, cy, r = 0.2, 0.2, 0.05  # cylinder centre and radius (diameter D = 0.1)


class ChannelSizeField(GmshSizeCallback):
    """Target element size (in metres) as a function of position.

    default_size is called with (dim, tag, x, y, z); the spatial scale is 1 m, so the
    coordinates arrive as plain metre values. Fine near the cylinder, fine in a
    widening wake band behind it, coarse in the far field. Tuned for ~2.0e4 dofs.
    """
    def default_size(self, dim, tag, x, y, z):
        dc = math.hypot(x - cx, y - cy) - r      # distance from the cylinder surface
        s = 0.0046 + 0.26 * max(dc, 0.0)         # grow away from the cylinder
        if x > cx:                                # downstream wake refinement
            band = 0.10 + 0.05 * (x - cx)         # band widens with distance
            if abs(y - cy) < band:
                s = min(s, 0.011 + 0.024 * max(x - (cx + 0.4), 0.0))
        return min(s, 0.044)                       # cap the coarsest element size


class ChannelWithCylinderMesh(GmshTemplate):
    def define_geometry(self):
        self.mesh_mode = "tris"
        self.default_resolution = 0.044
        self._mesh_size_callback = ChannelSizeField()   # graded refinement (fine cylinder+wake)
        # Channel corners (dimensional -> nondimensionalised against the spatial scale)
        p00 = self.point(0, 0)
        p10 = self.point(L * meter, 0)
        p11 = self.point(L * meter, H * meter)
        p01 = self.point(0, H * meter)
        self.line(p00, p10, name="bottom")
        self.line(p10, p11, name="outlet")
        self.line(p11, p01, name="top")
        self.line(p01, p00, name="inlet")
        # Cylinder as four named arcs cut out as a hole
        self.create_circle_lines((cx * meter, cy * meter), r * meter,
                                 mesh_size=0.0046, line_name="cylinder")
        self.plane_surface("bottom", "outlet", "top", "inlet",
                           name="channel", holes=[["cylinder"]])


class CylinderForces(InterfaceEquations):
    """Integrate the fluid traction over the cylinder to get the drag/lift per span.

    The traction needs the *bulk* velocity gradient (on the interface, grad() would be
    the surface gradient), so the velocity is read from the parent (bulk) domain. The
    outward bulk normal points into the cylinder, so we negate to report the force the
    fluid exerts on the cylinder (drag positive downstream).
    """
    required_parent_type = NavierStokesEquations

    def define_additional_functions(self):
        u = var("velocity", domain=self.get_parent_domain())
        mu = self.get_parent_equations().dynamic_viscosity
        p = var("pressure")
        stress = -p * identity_matrix() + 2 * mu * sym(grad(u))
        n = var("normal")
        traction = matproduct(stress, n)
        dx = self.get_dx()
        self.add_integral_function("drag", -dot(traction, vector(1, 0)) * dx)
        self.add_integral_function("lift", -dot(traction, vector(0, 1)) * dx)


class VonKarmanVortexStreet(Problem):
    def __init__(self):
        super().__init__()
        self.rho = 1.0 * kilogram / meter**3       # density
        self.mu = 1.0e-3 * pascal * second         # dynamic viscosity
        self.U_mean = 1.0 * meter / second         # mean inflow speed (Re = 100)
        self.U_peak = 1.5 * self.U_mean            # peak of the parabolic inflow

    def define_problem(self):
        # Consistent dimensional nondimensionalisation. spatial = 1 m keeps the gmsh
        # coordinates numerically equal to metres, which the size callback relies on.
        self.set_scaling(spatial=1 * meter, temporal=1 * second,
                         velocity=self.U_mean, pressure=self.rho * self.U_mean**2)

        self.add_mesh(ChannelWithCylinderMesh())

        eqs = NavierStokesEquations(dynamic_viscosity=self.mu, mass_density=self.rho, mode="TH")
        eqs += MeshFileOutput()                    # VTU/PVD for ParaView

        # Parabolic inflow: u_x(y) = 4 U_peak y (H - y) / H^2, u_y = 0
        y = var("coordinate_y")
        Hm = H * meter
        u_in = 4 * self.U_peak * y * (Hm - y) / Hm**2
        eqs += DirichletBC(velocity_x=u_in, velocity_y=0) @ "inlet"
        # No-slip on the channel walls and on the cylinder
        eqs += NoSlipBC() @ ["top", "bottom", "cylinder"]
        # Stress-free ("do nothing") outflow: only pin the transverse component.
        # Leaving velocity_x free fixes the pressure nullspace naturally.
        eqs += DirichletBC(velocity_y=0) @ "outlet"

        # Drag/lift monitoring on the cylinder -> forces.txt
        eqs += CylinderForces() @ "cylinder"
        eqs += IntegralObservableOutput(filename="forces")

        self.add_equations(eqs @ "channel")


if __name__ == "__main__":
    with VonKarmanVortexStreet() as problem:
        # Shedding period ~ 0.33 s; dt = 0.01 s resolves it with ~33 steps/period.
        # The wake instability grows from the off-centre cylinder; a clean street is
        # established after a few seconds. Run to 10 s (~30 shedding cycles).
        problem.run(10 * second, timestep=0.01 * second, outstep=0.05 * second)
