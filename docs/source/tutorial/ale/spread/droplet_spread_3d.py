#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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


from droplet_spread_sliplength import * # Import the previous example
from pyoomph.meshes.simplemeshes import SphericalOctantMesh # import a 3d mesh, octant of a sphere

class DropletSpreading3d(Problem):
    def __init__(self):
        super(DropletSpreading3d, self).__init__()
        # The equilibirum contact angle will vary with the position along the substrate
        x,y=var(["coordinate_x","coordinate_y"])
        # some equilibrium contact angle expression
        self.contact_angle = (45+80*(minimum(x,2)-1)**2-30*(minimum(y,2)-1)**2) * degree

    def define_problem(self):
        # Eighth part of a sphere, rename the outer interface to "interface" and the z=0 plane to "substrate"
        mesh = SphericalOctantMesh(radius=1, interface_names={"shell":"interface","plane_z0":"substrate"})
        self.add_mesh(mesh)

        eqs = NavierStokesEquations(mass_density=0.01, dynamic_viscosity=1)  # flow
        # PseudoElasticMesh is a bit more expensive to calculate, but is more stable in terms of larger deformations than LaplaceSmoothedMesh
        eqs += PseudoElasticMesh()
        eqs += RefineToLevel(2)  # Since the SphericalOctanctMesh is really coarse

        # No flow through the boundaries and
        eqs += DirichletBC(mesh_x=0, velocity_x=0) @ "plane_x0"
        eqs += DirichletBC(mesh_y=0, velocity_y=0) @ "plane_y0"
        eqs += DirichletBC(mesh_z=0, velocity_z=0) @ "substrate"

        # free surface at the interface, equilibrium contact angle at the contact with the substrate
        n_free=var("normal",domain="domain/interface") # normal of the free surface
        n_substrate=vector(0,0,1) # normal of the substrate
        t_substrate=n_free-dot(n_free,n_substrate)*n_substrate # projection of the free surface normal on the substrate
        t_substrate=t_substrate/square_root(dot(t_substrate,t_substrate)) # normalized => tangent along the substrate locally outward
        N = cos(self.contact_angle)*t_substrate - sin(self.contact_angle)*n_substrate # Assemble N vector
        eqs += (FreeSurface(sigma=1) + EquilibriumContactAngle(N) @ "substrate") @ "interface"

        eqs += MeshFileOutput()  # output

        self.add_equations(eqs @ "domain")  # adding it to the system


if __name__ == "__main__":
    with DropletSpreading3d() as problem:
        problem.run(50, outstep=True, startstep=0.05,temporal_error=1,maxstep=2)
