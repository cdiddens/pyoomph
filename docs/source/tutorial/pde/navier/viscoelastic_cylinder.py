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

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.viscoelastic import *


class ConfinedCylinderMesh(GmshTemplate):
    """Upper half of the channel: a graded O-grid of quadrilaterals on the cylinder, triangles outside"""

    def define_geometry(self):
        self.mesh_mode = "tris"
        pr = cast(ConfinedCylinderProblem, self.get_problem())
        self.default_resolution = pr.far_resolution        
        centre = self.point(0, 0)
        angles = [0.0, pi / 2, pi]                         # only the upper half of the cylinder
        inner = [self.point(pr.R * cos(a), pr.R * sin(a),size=pr.near_resolution) for a in angles]
        outer = [self.point(pr.Ro * cos(a), pr.Ro * sin(a),size=0.5*pr.near_resolution) for a in angles]
        wall = [self.circle_arc(inner[i], inner[i + 1], center=centre, name="cylinder") for i in range(2)]
        ring = [self.circle_arc(outer[i], outer[i + 1], center=centre) for i in range(2)]
        # The radial lines at angle 0 and pi lie on the symmetry plane
        radial = [self.line(inner[0], outer[0], name="symmetry"),
                  self.line(inner[1], outer[1]),
                  self.line(inner[2], outer[2], name="symmetry")]

        # Two transfinite sectors, recombined into quadrilaterals and graded towards the wall
        sectors = [self.plane_surface(wall[i], radial[i + 1], ring[i], radial[i], name="fluid") for i in range(2)]
        self.make_lines_transfinite(*radial, numnodes=pr.n_radial, mode="Progression", coeff=pr.layer_growth)
        self.make_lines_transfinite(*wall, *ring, numnodes=pr.n_circumferential)
        for i, sector in enumerate(sectors):
            self.make_surface_transfinite(sector, corners=[inner[i], inner[i + 1], outer[i + 1], outer[i]])
        self.set_recombined_surfaces(sectors)

        # The outer boundary is a single loop, closed by the outer ring of the O-grid. The top wall is
        # split by an extra point directly above the cylinder, carrying a much smaller target size:
        # gmsh grades away from it, so the elements are fine in the gap and coarse far up- and
        # downstream, where nothing happens
        L, H = pr.channel_length, pr.channel_height        
        top_centre = self.point(0, H, size=pr.near_resolution)
        box = self.create_lines(outer[0], "symmetry", self.point(3,0,size=0.5*pr.near_resolution), "symmetry", self.point(L, 0), "outlet", self.point(L, H), "top",
                                top_centre, "top", self.point(-L, H), "inlet",
                                self.point(-L, 0), "symmetry", self.point(-3,0,size=pr.near_resolution), "symmetry", outer[2])
        self.plane_surface(*box, ring[1], ring[0], name="fluid")


class ConfinedCylinderProblem(Problem):
    def __init__(self):
        super().__init__()
        self.channel_length = 20 # upstream and downstream length
        self.channel_height = 2 # half-height of the channel
        self.R, self.Ro = 1, 1.6 # cylinder radius and outer radius of the O-grid
        # Resolution is set by hand rather than by an error estimator. 
        self.far_resolution, self.near_resolution = 1.5, 0.09
        # The O-grid: the polymer stress forms a thin boundary layer on the cylinder
        self.n_circumferential, self.n_radial, self.layer_growth = 80, 20, 1.25
        # Wi enters as a global parameter so that we can continue in it later on
        self.Wi = self.define_global_parameter(Wi=0.1)
        # solvent fraction of the viscosity        
        self.beta = 0.59
        # Viscoelastic model 
        self.model=OldroydB()

    def define_problem(self):
        self += ConfinedCylinderMesh()
        # Creeping flow, so Stokes rather than Navier-Stokes. Its viscosity is the SOLVENT one
        stokes = StokesEquations(dynamic_viscosity=self.beta, mode="TH")
        
        
        # SUPG is not decoration here. The constitutive equation has no diffusion at all - its only
        # spatial operator is the advection u.grad(Psi) - and the polymer stress grows exponentially
        # just behind the rear stagnation point. Plain Galerkin answers that with a node-to-node
        # sawtooth all along the wake, which leaves the drag almost untouched (it is an integral over
        # the cylinder) but ruins any profile plotted through the wake. The reference stabilises too,
        # with DEVSS-G/DG
        viscoelastic = ViscoelasticEquations(model=self.model, relaxation_time=self.Wi,
                                             polymer_viscosity=1 - self.beta, stabilization="SUPG")

        
        eqs = MeshFileOutput() + stokes + viscoelastic
        # Fully developed inflow: a parabola with mean velocity 1 and the matching Oldroyd-B stress.
        # ViscoelasticInflowBC differentiates the profile itself to get the local shear rate and
        # pins the log-conformation tensor to the viscometric solution of the model in use. On the
        # symmetry line that shear rate vanishes and the conformation tensor becomes isotropic,
        # which is the degenerate case of the matrix logarithm the condition goes through; it is
        # handled there
        inflow = vector(1.5 * (1 - (var("coordinate_y")/self.channel_height) ** 2 ), 0)
        eqs += DirichletBC(velocity=inflow) @ ["inlet","outlet"]
        eqs += ViscoelasticInflowBC(inflow) @ "inlet"
                        
        eqs += DirichletBC(velocity_y=0,log_conformation_xy=0) @ "symmetry"  # no penetration, free tangentially                
        eqs += NoSlipBC() @ ["cylinder","top"]
        
        # The velocity is prescribed on the entire boundary, so the pressure needs a datum
        eqs += AverageConstraint(pressure=0) 
        
        # Output quantities
        
        # Get the stress on the cylinder and integrate its x-component to get the drag coefficient.
        # The factor 2 restores the full cylinder, domain=".." makes sure bulk gradients are taken
        u=var("velocity",domain="..")
        p=var("pressure")
        stress = -p * identity_matrix() + 2 * stokes.dynamic_viscosity * sym(grad(u)) + var("polymer_stress",domain="..")
        eqs += IntegralObservables(drag=-2 * dot(var("normal"),stress@vector(1,0)) ) @ "cylinder"
        eqs += IntegralObservableOutput("cylinder_drag") @ "cylinder"
        
        # These are what Claus & Phillips contour in their Fig. 12: the Cauchy stress is made traceless,
        # T0 = sigma - 1/2*tr(sigma)*I, and then projected onto the streamline direction and its normal.
        # The pressure drops out of T0 identically - in an incompressible plane flow tr(sigma) is
        # -2p + tr(tau_p) - so only the solvent rate of strain and the polymer stress survive.
        u=var("velocity")
        T0 = 2 * stokes.dynamic_viscosity * sym(grad(u)) + var("polymer_stress")
        T0 = T0 - trace(T0) / 2 * identity_matrix(3)        
        u_mag = subexpression(square_root(dot(u, u)))
        par = u/u_mag
        perp = vector(-par[1], par[0])
        # The projection itself is solved on another residual so that we we do not have to solve for S1 and S2 while the flow is being solved. 
        # They are only needed for plotting, see solve_auxiliary_residual("output_projection") below
        eqs+=ProjectExpression(S1=dot(perp, matproduct(T0, par)), S2=dot(par, matproduct(T0, par)),destination="output_projection")
                        
        self += eqs @ "fluid"



if __name__ == "__main__":
    with ConfinedCylinderProblem() as problem:        
        problem.solve()
        for Wi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            # go_to_param continues towards the target and halves its step whenever Newton fails,            
            problem.go_to_param(Wi=Wi)            
            if Wi in (0.1, 0.5, 0.7):        # the three rows of their Fig. 12
                # S1 and S2 live on their own residual, so they are pinned and cost nothing while the
                # flow is being solved; But for the output, we want to include it
                problem.solve_auxiliary_residual("output_projection")
                problem.output(increase_time_for_PVD=True)
                