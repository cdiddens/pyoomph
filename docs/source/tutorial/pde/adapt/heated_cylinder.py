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

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.navier_stokes import NavierStokesEquations, NoSlipBC
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.output.plotting import MatplotlibPlotter

#: The tracer is a trace species: a thousand times weaker than the temperature.
C0 = 1e-3


class CylinderMesh(GmshTemplate):
    """A cylinder in a channel, with a structured O-grid of quadrilaterals at the wall and
    triangles everywhere else.

    The boundary layer is built as four transfinite sectors of an annulus. Making the radial lines
    transfinite with a "Progression" coefficient grades them, so the layers are thin at the wall and
    coarsen outwards, and recombining the sectors turns them into quadrilaterals that are aligned
    with the wall. The far field stays triangular, so the domain is genuinely mixed.
    """

    def define_geometry(self):
        self.mesh_mode = "tris"
        pr = cast(HeatedCylinderProblem, self.get_problem())
        self.default_resolution = pr.resolution
        centre = self.point(0, 0)
        angles = [0, pi/2, pi, 3*pi/2]
        inner = [self.point(pr.radius*cos(a), pr.radius*sin(a)) for a in angles]
        outer = [self.point(pr.boundary_layer_radius*cos(a), pr.boundary_layer_radius*sin(a))
                 for a in angles]
        wall = [self.circle_arc(inner[i], inner[(i+1) % 4], center=centre, name="cylinder")
                for i in range(4)]
        ring = [self.circle_arc(outer[i], outer[(i+1) % 4], center=centre) for i in range(4)]
        radial = [self.line(inner[i], outer[i]) for i in range(4)]

        sectors = [self.plane_surface(wall[i], radial[(i+1) % 4], ring[i], radial[i], name="fluid")
                   for i in range(4)]
        # Graded normal to the wall, uniform along it.
        self.make_lines_transfinite(*radial, numnodes=pr.n_radial, mode="Progression",
                                    coeff=pr.layer_growth)
        self.make_lines_transfinite(*wall, *ring, numnodes=pr.n_circumferential)
        for i, sector in enumerate(sectors):
            # The corners have to be given explicitly. Without them make_surface_transfinite works
            # out its own node counts from the point sizes and overrides the grading set above.
            self.make_surface_transfinite(sector, corners=[inner[i], inner[(i+1) % 4],
                                                           outer[(i+1) % 4], outer[i]])
        self.set_recombined_surfaces(sectors)

        corners = [self.point(-pr.L_up, -pr.half_height), self.point(pr.L_down, -pr.half_height),
                   self.point(pr.L_down, pr.half_height), self.point(-pr.L_up, pr.half_height)]
        box = self.create_lines(corners[0], "bottom", corners[1], "outlet",
                                corners[2], "top", corners[3], "inlet", corners[0])
        self.plane_surface(*box, holes=[ring], name="fluid")


class AdvectionDiffusion(Equations):
    """One passive scalar carried by the flow."""

    def __init__(self, name, peclet):
        super().__init__()
        self.name, self.peclet = name, peclet

    def define_fields(self):
        self.define_scalar_field(self.name, "C2")

    def define_residuals(self):
        c, q = var_and_test(self.name)
        self.add_residual(weak(dot(var("velocity"), grad(c)), q)
                          + weak(grad(c)/self.peclet, grad(q)))


class HeatedCylinderProblem(Problem):
    def __init__(self, criterion="grouped", budget=80000):
        super().__init__()
        self.Re = 40.0
        self.Pe_temperature, self.Pe_tracer = 200.0, 1000.0
        self.radius, self.boundary_layer_radius = 0.5, 0.9
        #: O-grid: nodes around a quadrant, nodes across the layer, and its radial growth ratio.
        self.n_circumferential, self.n_radial, self.layer_growth = 10, 4, 1.25
        self.L_up, self.L_down, self.half_height = 5.0, 18.0, 5.0
        self.resolution = 0.9
        #: Where the tracer filament enters, and how wide it is.
        self.y_source, self.source_width = 2.0, 0.15
        #: "joint" puts every field in one error group, "grouped" gives each its own.
        self.criterion = criterion
        self.desired_ndof = budget
        self.max_refinement_level = 8
        self.initial_adaption_steps = 0

    def define_problem(self):
        self += CylinderMesh()
        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=1/self.Re, mode="TH")
        eqs += MeshFileOutput()
        eqs += AdvectionDiffusion("temperature", self.Pe_temperature)
        eqs += AdvectionDiffusion("tracer", self.Pe_tracer)

        # The tracer does not come off the cylinder: it enters as a thin filament well above it and
        # is carried straight downstream, through a region where nothing else happens. That is what
        # makes the error criteria want different parts of the mesh.
        y = var("coordinate_y")
        filament = C0*exp(-((y-self.y_source)/self.source_width)**2)
        eqs += DirichletBC(velocity_x=1, velocity_y=0, temperature=0) @ "inlet"
        eqs += DirichletBC(tracer=filament) @ "inlet"
        eqs += DirichletBC(velocity_y=0, temperature=0, tracer=0) @ ["top", "bottom"]
        eqs += NoSlipBC() @ "cylinder"
        eqs += DirichletBC(temperature=1) @ "cylinder"
        # The outlet carries everything out: no condition at all, i.e. traction free.

        if self.criterion == "joint":
            # One group: the three fields' errors are summed before being normalised, so the tracer
            # -- a thousand times weaker than the temperature -- contributes essentially nothing.
            eqs += SpatialErrorEstimator(velocity=1, temperature=1, tracer=1)
        else:
            # One group each: every field is divided by its OWN recovered-flux norm and therefore
            # judged on its own scale, and the groups are combined by taking the maximum.
            eqs += SpatialErrorEstimator(velocity=1, group="flow")
            eqs += SpatialErrorEstimator(temperature=1, group="heat")
            eqs += SpatialErrorEstimator(tracer=1, group="trace",weight=100)

        # grad() on an interface is the SURFACE gradient, whose normal component vanishes by
        # construction. domain=".." takes the gradient in the bulk, which is what a wall flux is.
        eqs += IntegralObservables(
            heat=dot(grad(var("temperature", domain="..")), var("normal")))@"cylinder"
        self += eqs @ "fluid"

    def nusselt(self):
        """Mean Nusselt number: the wall heat flux per unit circumference. The diameter is 1, so
        this is Nu directly."""
        return float(self.get_mesh("fluid/cylinder").evaluate_observable("heat"))/pi

    def element_kinds(self):
        """How many quadrilaterals and triangles the mixed mesh actually has."""
        mesh = self.get_mesh("fluid")
        quads = sum(1 for e in mesh.elements() if e.nvertex_node() == 4)
        return quads, mesh.nelement()-quads

    def region_counts(self):
        """Elements on the tracer filament and around the cylinder."""
        mesh = self.get_mesh("fluid")
        cx = numpy.array([numpy.mean([e.node_pt(i).x(0) for i in range(e.nnode())])
                          for e in mesh.elements()])
        cy = numpy.array([numpy.mean([e.node_pt(i).x(1) for i in range(e.nnode())])
                          for e in mesh.elements()])
        filament = int(((cx > 0) & (numpy.abs(cy-self.y_source) < 0.4)).sum())
        cylinder = int((numpy.hypot(cx, cy) < 1.2).sum())
        return filament, cylinder


class CylinderMeshPlotter(MatplotlibPlotter):
    """The adapted mesh, drawn from the element outlines."""

    label = ""
    view = (-1.5, -3.0, 12.0, 3.5)

    def define_plot(self):
        self.dpi = 200
        self.background_color = "white"
        self.set_view(*self.view)
        self.add_plot("fluid", mode="outlines", linecolor="#20406e", linewidths=0.25)
        self.add_text(self.label, position="top center", textsize=20, color="black")


class CylinderFieldPlotter(MatplotlibPlotter):
    """One scalar field over the whole wake region.

    No velocity arrows or streamlines here, deliberately. Both are sampled on a regular grid, which
    needs matplotlib's TriFinder, and that rejects the non-conforming triangulation a deeply refined
    mesh produces -- the coarse side of a 2:1 interface overlaps its finer neighbours, which
    tricontourf tolerates but the trapezoid-map search structure does not.
    """

    field = "temperature"
    label = ""
    #: Multiplies the data for display, so the trace species can be shown in units of its own
    #: inlet concentration rather than as a row of zeros.
    factor = 1.0
    vmax = 1.0
    view = (-1.5, -1.2, 12.0, 3.2)
    #: Where the colour bar sits. The temperature panels put it at the top, so that the two bars of
    #: a column do not end up next to each other.
    cbposition = "bottom center"

    def define_plot(self):
        self.dpi = 200
        self.background_color = "white"
        self.set_view(*self.view)
        colorbar = self.add_colorbar(self.label, cmap="viridis", factor=self.factor,
                                     vmin=0.0, vmax=self.vmax,
                                     position=self.cbposition, length=0.75, thickness=0.055)
        # The bar sits on top of the dark end of the colour map, so label and ticks have to be white
        # to be legible at all.
        colorbar.textcolor = "white"
        colorbar.textsize = 20
        colorbar.ticsize = 16
        if self.cbposition.startswith("top"):
            # For a top-placed bar the title sits ABOVE it, outside the space the bar reports for
            # itself, so with the default margin it runs off the top of the figure.
            colorbar.ymargin = 0.15
        if self.cbposition.startswith("bottom"):
            # The default margin leaves room for the bar but not for the tick labels hanging below
            # it, which then get clipped at the edge of the figure. Lifting it clear of the edge
            # puts it across the cylinder, so it is shifted to the right of that as well.
            colorbar.ymargin = 0.19
            colorbar.xpos = 0.30
            colorbar.length = 0.62
        self.add_plot("fluid/"+self.field, colorbar=colorbar)
        self.add_plot("fluid/cylinder", linecolor="black", linewidths=1.5)


if __name__ == "__main__":
    print("%-9s %6s %6s %6s %8s %8s %8s" %
          ("criterion", "ndof", "quads", "tris", "filament", "cylinder", "Nu"))
    for criterion in ["joint", "grouped"]:
        with HeatedCylinderProblem(criterion=criterion) as problem:
            problem.set_output_directory("heated_cylinder_"+criterion)
            problem.quiet(True)            
            problem.plotter = [CylinderMeshPlotter(problem, filetrunk="mesh", fileext="pdf")]
            problem.plotter[-1].label = criterion
            # The two scalars, each with the velocity streamlines on top. The tracer is shown in
            # units of its own inlet concentration, i.e. scaled by 1/C0.
            for field, factor, vmax, cbpos in [("temperature", 1.0, 1.0, "top center"),
                                               ("tracer", 1/C0, 0.5, "bottom center")]:
                problem.plotter.append(CylinderFieldPlotter(problem, filetrunk=field, fileext="pdf"))
                problem.plotter[-1].field = field
                problem.plotter[-1].factor = factor
                problem.plotter[-1].vmax = vmax
                problem.plotter[-1].cbposition = cbpos
                problem.plotter[-1].label = field+" ("+criterion+")"
            problem.solve(spatial_adapt=12)
            problem.output()
            quads, tris = problem.element_kinds()
            filament, cylinder = problem.region_counts()
            print("%-9s %6d %6d %6d %8d %8d %8.3f"
                  % (criterion, problem.ndof(), quads, tris, filament, cylinder, problem.nusselt()))
