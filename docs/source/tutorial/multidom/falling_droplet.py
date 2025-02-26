#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================


from pyoomph import *  # Import dimensional Stokes from before
from pyoomph.equations.navier_stokes import *
from pyoomph.expressions.units import *


# Mesh: Two concentrical hemi-circles => axisymmetric => concentric spheres
class StokesLawMesh(GmshTemplate):
    def define_geometry(self):
        self.default_resolution = 0.05  # make it a bit finer
        self.mesh_mode = "tris"
        p = self.get_problem()  # get the problem to obtain parameters
        Rs = p.droplet_radius  # bind droplet radius
        Ro = p.outer_radius  # and outer radius
        self.far_size = self.default_resolution * float(Ro / Rs)  # Make the far field coarser
        p00 = self.point(0, 0)  # center
        pSnorth = self.point(0, Rs)  # points along the sphere
        pSeast = self.point(Rs, 0)
        pSsouth = self.point(0, -Rs)
        pOnorth = self.point(0, Ro, size=self.far_size)  # points of the far field
        pOeast = self.point(Ro, 0, size=self.far_size)
        pOsouth = self.point(0, -Ro, size=self.far_size)
        self.line(pOsouth, pSsouth, name="axisymm_lower")  # axisymmetric lines, we have two since we
        self.line(pOnorth, pSnorth, name="axisymm_upper")  # want to fix p=0 at a single p-DoF at axisymm_upper
        self.circle_arc(pOsouth, pOeast, center=p00, name="far_field")  # far field hemi-circle
        self.circle_arc(pOnorth, pOeast, center=p00, name="far_field")
        self.circle_arc(pSsouth, pSeast, center=p00, name="droplet_outside")  # sphere hemi-circle
        self.circle_arc(pSnorth, pSeast, center=p00, name="droplet_outside")
        self.plane_surface("axisymm_lower", "axisymm_upper", "far_field", "droplet_outside",
                           name="outer")  # liquid domain

        # Modification (besides renaming some interfaces): Add a droplet mesh
        self.line(pSnorth,pSsouth,name="droplet_axisymm")
        self.plane_surface("droplet_axisymm", "droplet_outside",name="droplet")  # droplet domain


class DragContribution(InterfaceEquations):
    required_parent_type = NavierStokesEquations

    def __init__(self, lagr_mult, direction=vector(0, -1)):
        super(DragContribution, self).__init__()
        self.lagr_mult = lagr_mult  # Store the destination Lagrange multiplier U
        self.direction = direction  # and the e_z direction

    def define_residuals(self):
        u = var("velocity",
                domain=self.get_parent_domain())  # Important: we want to calculate grad with respect to the bulk
        strain = 2 * self.get_parent_equations().dynamic_viscosity * sym(grad(u))  # get mu from the parent equations
        p = var("pressure")
        stress = -p * identity_matrix() + strain  # T=-p*1 + mu*(grad(u)+grad(u)^t))
        n = var("normal")  # interface normal pointing outwards
        traction = matproduct(stress, n)  # traction vector by projection
        ltest = testfunction(self.lagr_mult)  # test function V of the Lagrange multiplier U
        self.add_residual(weak(dot(traction, self.direction), ltest,
                               dimensional_dx=True))  # Integrate dimensionally over the traction


class NewtonEquation(ODEEquations):
    def __init__(self,M,F_buo):
        super(NewtonEquation, self).__init__()
        self.M, self.F_buo = M, F_buo

    def define_fields(self):
        self.define_ode_variable("UStokes",scale="velocity",testscale=1/scale_factor("force"))

    def define_residuals(self):
        U,V=var_and_test("UStokes")
        self.add_residual(weak(self.M*partial_t(U)-self.F_buo,V))


class FallingDropletProblem(Problem):
    def __init__(self):
        super(FallingDropletProblem, self).__init__()
        self.outer_radius = 50 * milli * meter  # radius of the far boundary
        self.gravity = 9.81 * meter / second ** 2  # gravitational acceleration

        # Modifications: Also define viscosity of the droplet
        self.droplet_radius = 0.25 * milli * meter  # radius of the spherical object
        self.droplet_density = 1200 * kilogram / meter ** 3  # density of the sphere
        self.outer_density = 1000 * kilogram / meter ** 3  # density of the liquid
        self.droplet_viscosity = 5 * milli * pascal * second  # viscosity
        self.outer_viscosity = 1 * milli * pascal * second  # viscosity
        self.surface_tension=50*milli*newton/meter # Also define a surface tension (although it does not matter on a static mesh)


    def get_theoretical_velocity(self):  # get the analytical terminal velocity (of a solid particle, not a droplet)
        return 2 / 9 * (
                    self.droplet_density - self.outer_density) / self.outer_viscosity * self.gravity * self.droplet_radius ** 2

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")  # axisymmetric
        self.set_scaling(spatial=self.droplet_radius)  # use the radius as spatial scale

        # Use the theoretical value as scaling for the velocity
        UStokes_ana = self.get_theoretical_velocity()
        self.set_scaling(velocity=UStokes_ana)
        self.set_scaling(pressure=scale_factor("velocity") * self.outer_viscosity / scale_factor("spatial"))
        # Buoyancy force
        F_buo = (self.droplet_density - self.outer_density) * self.gravity * 4 / 3 * pi * self.droplet_radius ** 3
        self.set_scaling(force=F_buo)  # define the scale "force" by the value of the gravity force

        self.set_scaling(temporal=self.droplet_radius/UStokes_ana)

        self.add_mesh(StokesLawMesh())  # Mesh

        U = var("UStokes", domain="globals")  # bind U from the domain "globals"
        inertia_correction=self.outer_density*vector([0,1])*partial_t(U)
        eqs = NavierStokesEquations(dynamic_viscosity=self.outer_viscosity,mass_density=self.outer_density,bulkforce=inertia_correction)  # Stokes equation and output
        eqs += MeshFileOutput()

        eqs += DirichletBC(velocity_x=0) @ "axisymm_lower"  # No flow through the axis of symmetry
        eqs += DirichletBC(velocity_x=0) @ "axisymm_upper"  # No flow through the axis of symmetry

        # Define the Lagrange multiplier U
        U_eqs=NewtonEquation(4/3*pi*self.droplet_radius**3*self.droplet_density,F_buo)
        U_eqs+=ODEFileOutput()
        self.add_equations(U_eqs @ "globals")  # add it to an ODE domain named "globals"


        # Add the traction integral, i.e. the drag force to U
        eqs += DragContribution(U) @ "droplet_outside"  # The constraint is now fully assembled

        # Far field condition
        R = self.droplet_radius
        r, z = var(["coordinate_x", "coordinate_y"])
        d = subexpression(square_root(r ** 2 + z ** 2))  # precalcuate d in the generated C code for faster computation
        ur_far = 3 * R ** 3 / 4 * r * z * U / d ** 5 - 3 * R / 4 * r * z * U / d ** 3  # u_r as function of U
        uz_far = R ** 3 / 4 * (3 * U * z ** 2 / d ** 5 - U / d ** 3) + U - 3 * R / 4 * (
                    U / d + U * z ** 2 / d ** 3)  # u_z as function of U

        # Since U is an unknown, DirichletBC should not be used here. Instead, we enforce the velocity components to the far field by Lagrange multipliers
        eqs += EnforcedBC(velocity_x=var("velocity_x") - ur_far, velocity_y=var("velocity_y") - uz_far) @ "far_field"
        eqs += DirichletBC(pressure=0) @ "droplet_outside/axisymm_upper"  # fix one pressure degree

        self.add_equations(eqs @ "outer")

        ## MODIFICATIONS:
        # Do not add a no slip
        # eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "droplet_outside"  # and no-slip on the object

        # droplet equations
        inertia_correction = self.droplet_density * vector([0, 1]) * partial_t(U)
        deqs=MeshFileOutput()
        deqs+=NavierStokesEquations(dynamic_viscosity=self.droplet_viscosity,mass_density=self.droplet_density,bulkforce=inertia_correction)
        deqs+=DirichletBC(velocity_x=0)@"droplet_axisymm"
        deqs+=NavierStokesFreeSurface(surface_tension=self.surface_tension,static_interface=True)@"droplet_outside"
        deqs+=ConnectVelocityAtInterface()@"droplet_outside"
        deqs += DragContribution(U) @ "droplet_outside"  # Also add the tractions from the inside to the drag
        self.add_equations(deqs@"droplet")


if __name__ == "__main__":
    with FallingDropletProblem() as problem:
        problem.run(0.5*second,startstep=0.05*second,outstep=True)  # solve and output




