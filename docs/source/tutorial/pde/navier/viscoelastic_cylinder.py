#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
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
from pyoomph.equations.navier_stokes import StokesEquations, NoSlipBC
from pyoomph.equations.generic import SpatialErrorEstimator
# The viscoelastic equations and the Oldroyd-B model, plus two helpers for the inflow condition
from pyoomph.equations.viscoelastic import (ViscoelasticEquations, OldroydB,
                                            symmetric_2x2_matrix_log, oldroyd_b_shear_conformation)
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.output.plotting import MatplotlibPlotter
from math import pi, cos, sin

# Everything is nondimensionalised with the total viscosity, the cylinder radius and the mean inlet
# velocity, so beta is the solvent fraction of the viscosity and Wi is numerically the relaxation time
BETA = 0.59

#: Drag coefficients of Claus & Phillips, J. Non-Newtonian Fluid Mech. 200 (2013), Table 3
REFERENCE_DRAG = {0.1: 130.364, 0.2: 126.626, 0.3: 123.192, 0.4: 120.593,
                  0.5: 118.826, 0.6: 117.776, 0.7: 117.316}


class ConfinedCylinderMesh(GmshTemplate):
    """Upper half of the channel: a graded O-grid of quadrilaterals on the cylinder, triangles outside"""

    def define_geometry(self):
        self.mesh_mode = "tris"
        pr = self.get_problem()
        self.default_resolution = 0.5
        R, Ro = 1.0, 1.6                                   # cylinder radius and outer radius of the O-grid
        centre = self.point(0, 0)
        angles = [0.0, pi / 2, pi]                         # only the upper half of the cylinder
        inner = [self.point(R * cos(a), R * sin(a)) for a in angles]
        outer = [self.point(Ro * cos(a), Ro * sin(a)) for a in angles]
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

        # The outer boundary is a single loop, closed by the outer ring of the O-grid
        L, H = pr.channel_length, 2.0
        box = self.create_lines(outer[0], "symmetry", self.point(L, 0), "outlet", self.point(L, H), "top",
                                self.point(-L, H), "inlet", self.point(-L, 0), "symmetry", outer[2])
        self.plane_surface(*box, ring[1], ring[0], name="fluid")


class CylinderDrag(InterfaceEquations):
    """Drag coefficient of the cylinder, K=F_x/(eta_0*<u>), as an integral observable"""

    required_parent_type = StokesEquations

    def define_additional_functions(self):
        bulk = self.get_parent_domain().get_equations()
        stokes = bulk.get_equation_of_type(StokesEquations)
        viscoelastic = bulk.get_equation_of_type(ViscoelasticEquations)
        u, p = var("velocity", domain=".."), var("pressure", domain="..")
        # Total stress: pressure, solvent and polymer contributions
        stress = -p * identity_matrix(3) + 2 * stokes.dynamic_viscosity * sym(grad(u))
        if viscoelastic is not None:
            stress = stress + viscoelastic.get_polymer_stress(domain="..")
        traction = matproduct(stress, var("normal"))
        # var("normal") points out of the fluid, hence the minus; the factor 2 restores the full cylinder
        self.add_integral_function("drag", -2 * traction[0] * self.get_dx())


class ConfinedCylinderProblem(Problem):
    def __init__(self):
        super().__init__()
        self.channel_length = 20.0                          # upstream and downstream length
        # The O-grid resolution decides the accuracy: the error estimator will not refine these
        # structured quadrilaterals, so the stress boundary layer has to be resolved by hand
        self.n_circumferential, self.n_radial, self.layer_growth = 60, 18, 1.30
        self.max_refinement_level = 4

    def define_problem(self):
        self += ConfinedCylinderMesh()
        # Creeping flow, so Stokes rather than Navier-Stokes. Its viscosity is the SOLVENT one
        stokes = StokesEquations(dynamic_viscosity=BETA, mode="TH")
        eqs = stokes + MeshFileOutput()

        # Wi enters as a global parameter so that we can continue in it later on
        self.Wi = self.define_global_parameter(Wi=0.1)
        eqs += ViscoelasticEquations(model=OldroydB(), relaxation_time=self.Wi, polymer_viscosity=1 - BETA)

        # Fully developed inflow: a parabola with mean velocity 1 and the matching Oldroyd-B stress.
        # The shear rate du/dy vanishes on the symmetry line, where the conformation tensor becomes
        # isotropic - symmetric_2x2_matrix_log handles that degenerate case
        inflow = 1.5 * (1 - var("coordinate_y") ** 2 / 4)
        psi = symmetric_2x2_matrix_log(oldroyd_b_shear_conformation(self.Wi * (-0.75 * var("coordinate_y"))))
        eqs += DirichletBC(log_conformation_xx=psi[0, 0], log_conformation_xy=psi[0, 1],
                           log_conformation_yy=psi[1, 1]) @ "inlet"

        eqs += DirichletBC(velocity_x=inflow, velocity_y=0) @ "inlet"
        eqs += DirichletBC(velocity_x=inflow, velocity_y=0) @ "outlet"
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "top"
        eqs += DirichletBC(velocity_y=0) @ "symmetry"        # no penetration, free tangentially
        eqs += NoSlipBC() @ "cylinder"
        # The velocity is prescribed on the entire boundary, so the pressure needs a datum
        eqs += stokes.create_pressure_fixation(value=0) @ "inlet"

        eqs += SpatialErrorEstimator(velocity=1, group="flow")
        # Component by component: SpatialErrorEstimator takes grad() of what it is given
        eqs += SpatialErrorEstimator(log_conformation_xx=1, log_conformation_xy=1,
                                     log_conformation_yy=1, group="stress")
        eqs += CylinderDrag() @ "cylinder"
        self += eqs @ "fluid"

    def drag(self):
        return float(self.get_mesh("fluid/cylinder").evaluate_observable("drag"))


class StressPlotter(MatplotlibPlotter):
    """The polymer stress tau_xx around the cylinder and in the wake"""

    def define_plot(self):
        self.background_color = "white"
        self.set_view(-2.5, -2.05, 9.0, 2.05)            # the cylinder and its wake
        # Note vmin/vmax only widen the range, they cannot clip it: the colour scale is set by
        # the peak stress in the thin layer on the cylinder, so the wake looks faint by comparison
        cb = self.add_colorbar("polymer stress $\\tau_{xx}$", cmap="viridis", position="top center",
                               vmin=0, length=0.7, thickness=0.05)
        cb.textcolor, cb.textsize = "black", 14
        # Only the upper half is solved, so each plot is drawn twice: once as it is and once
        # mirrored about the symmetry line
        self.add_plot("fluid/polymer_stress_xx", colorbar=cb, transform=[None, "mirror_y"])
        self.add_plot("fluid/cylinder", linecolor="black", linewidths=1.5, transform=[None, "mirror_y"])


if __name__ == "__main__":
    with ConfinedCylinderProblem() as problem:
        problem.initialise()
        print("  Wi      K (pyoomph)   K (Claus & Phillips)")
        # Continuation in Wi: starting cold at a larger Wi would overshoot into a conformation
        # tensor so stretched that exp(Psi) overflows on the very first Newton step
        for Wi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            problem.Wi.value = Wi
            problem.solve()
            print("  %.1f     %9.4f       %9.3f" % (Wi, problem.drag(), REFERENCE_DRAG[Wi]))
        # A snapshot of the birefringent strand at the largest Wi reached
        problem.plotter = [StressPlotter(problem, filetrunk="viscoelastic_stress", fileext=["pdf", "png"])]
        problem.output()
