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

import matplotlib.colors
import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions.units import degree
from pyoomph.equations.navier_stokes import StokesEquations, NoSlipBC
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.output.plotting import MatplotlibPlotter


class WedgeMesh(GmshTemplate):
    """A circular sector: two straight walls meeting at the apex, closed by a driven arc."""

    def define_geometry(self):
        self.mesh_mode = "tris"
        pr = cast(MoffattProblem, self.get_problem())
        alpha = pr.half_angle
        self.default_resolution = pr.resolution
        apex = self.point(0, 0, size=pr.resolution*0.25)
        lo = self.point(cos(alpha), -sin(alpha))
        hi = self.point(cos(alpha), sin(alpha))
        self.line(apex, lo, name="lower_wall")
        self.line(apex, hi, name="upper_wall")
        self.circle_arc(lo, hi, center=apex, name="driven")
        self.plane_surface("lower_wall", "driven", "upper_wall", name="wedge")


class MoffattProblem(Problem):
    def __init__(self, scale_compensation=0, budget=20000):
        super().__init__()
        self.half_angle = 30*degree
        self.resolution = 0.08
        #: Exponent k in the estimator flux sym(grad(u))/r**k. 0 is the plain energy-norm estimator.
        self.scale_compensation = scale_compensation
        # Adapt towards a problem size instead of an error tolerance. Everything else about the
        # adaptation - which elements, and when to stop - follows from this.
        self.desired_ndof = budget
        self.max_refinement_level = 16
        self.initial_adaption_steps = 0

    def define_problem(self):
        self += WedgeMesh()
        x, y = var(["coordinate_x", "coordinate_y"])

        eqs = StokesEquations(dynamic_viscosity=1, mode="TH")
        eqs += NoSlipBC() @ ["lower_wall","upper_wall"]
        
        # The arc rotates. The drive is tapered smoothly to zero where the arc meets the walls, so
        # that the apex is the only singular point in the domain and the adaptivity has nothing else
        # to chase.
        taper = (1-(y/sin(self.half_angle))**2)**2
        eqs += DirichletBC(velocity_x=-taper*y, velocity_y=taper*x) @ "driven"
        # Velocity is prescribed on every boundary, so the pressure is only defined up to a constant.
        eqs += AverageConstraint(pressure=0)

        # The error criterion. With k=0 this is the ordinary energy-norm estimator, which measures
        # the jump in sym(grad(u)) -- and which will be shown to ignore the corner entirely. Dividing by a
        # power of the radius compensates for the eddies decaying geometrically towards the apex, so
        # that every decade of r can compete for the budget on equal terms.
        r = square_root(x**2+y**2+1e-18)
        flux = sym(grad(nondim("velocity"), nondim=True))/r**self.scale_compensation
        # normalize_relative=0 makes the errors absolute: they then genuinely shrink as the mesh is
        # refined, which is what lets them be compared between the runs below.
        eqs += SpatialErrorEstimator(flux, normalize_relative=0)

        eqs += MeshFileOutput()
        self += eqs @ "wedge"
        
    
    def reach(self):
        """How far into the corner the mesh actually reaches: the smallest nodal radius."""
        mesh = self.get_mesh("wedge")
        radii = [numpy.hypot(n.x(0), n.x(1)) for n in mesh.nodes()]
        return min(r for r in radii if r > 0)


    def elements_per_decade(self):
        mesh = self.get_mesh("wedge")
        centres = [numpy.mean([[e.node_pt(i).x(0), e.node_pt(i).x(1)] for i in range(e.nnode())], axis=0)
                for e in mesh.elements()]
        radii = numpy.array([numpy.hypot(c[0], c[1]) for c in centres])
        bounds = [(0, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
        return [int(((radii >= lo) & (radii < hi)).sum()) for lo, hi in bounds]



class MoffattStreamPlotter(MatplotlibPlotter):
    """Velocity magnitude and streamlines in a box of size ``zoom`` around the apex.

    The colour scale has to be logarithmic: the velocity spans several decades within a single
    picture, so a linear scale shows the outer part of the view and leaves the rest uniformly black.
    The streamlines are unaffected either way, since they follow the direction of the flow and ignore
    its magnitude entirely.
    """

    zoom = 1.0
    label = ""

    def local_velocity_range(self):
        """The velocity range actually present in the view, so each zoom gets its own decades."""
        mesh = self.get_problem().get_mesh("wedge")
        indices = mesh.get_nodal_field_indices()
        ix, iy = indices["velocity_x"], indices["velocity_y"]
        speeds = [numpy.hypot(n.value(ix), n.value(iy)) for n in mesh.nodes()
                  if numpy.hypot(n.x(0), n.x(1)) <= self.zoom]
        vmax = max(speeds)
        return 1e-4*vmax, vmax

    def define_plot(self):
        self.dpi=200
        problem = cast(MoffattProblem, self.get_problem())
        
        half_height = 1.05*self.zoom*tan(problem.half_angle)
        self.background_color = "white"
        self.set_view(-0.02*self.zoom, -half_height, 1.02*self.zoom, half_height)

        vmin, vmax = self.local_velocity_range()
        # pyoomph passes the norm straight to matplotlib, so any matplotlib norm works here -- but a
        # custom one must carry its own vmin/vmax, since there is nothing left for pyoomph to infer
        # them from.
        colorbar = self.add_colorbar("velocity", cmap="viridis",
                                     norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                                     position="bottom center", length=0.55, thickness=0.035)
        
        self.add_plot("wedge/velocity", colorbar=colorbar)
        self.add_text(self.label, position="top center", textsize=22)

        streams = self.add_plot("wedge/velocity", mode="streamlines",
                                linecolor="white", linewidths=1.0)
        # The interpolation grid is spanned over the view, so a deep zoom needs no special treatment
        # beyond enough points to resolve the eddy sitting in it.
        streams.density = 2.2
        streams.numx = streams.numy = 400
        self.add_plot("wedge/lower_wall", linecolor="black", linewidths=1.6)
        self.add_plot("wedge/upper_wall", linecolor="black", linewidths=1.6)


class MoffattMeshPlotter(MatplotlibPlotter):
    """The adapted mesh itself, drawn from the element outlines."""

    label = ""

    def define_plot(self):
        self.dpi=200
        problem = cast(MoffattProblem, self.get_problem())        
        half_height = 1.05*tan(problem.half_angle)
        self.background_color = "white"
        self.set_view(-0.02, -half_height, 1.02, half_height)
        # "outlines" asks each element for its own outline, which is the only reliable way to draw a
        # mesh: the nodes of an element are not in any particular geometric order, so joining the
        # first few of them by hand produces stray edges on some element types.
        self.add_plot("wedge", mode="outlines", linecolor="#20406e", linewidths=0.15)
        self.add_text(self.label, position="top center", textsize=22)



if __name__ == "__main__":
    budget = 80000
    print("%-28s %8s %10s %s" % ("criterion", "ndof", "reach", "elements per decade of r"))
    for k, label in [(0, "sym(grad(u))       (k=0)"), (3, "sym(grad(u))/r**3 (k=3)")]:
        with MoffattProblem(scale_compensation=k, budget=budget) as problem:
            problem.set_output_directory("moffatt_k%d" % k)
            # Add the mesh plotter, also for k=3, add the stream plotters for the three zoom levels. 
            problem.plotter=[MoffattMeshPlotter(problem, filetrunk="mesh", fileext="pdf")]
            problem.plotter[-1].label = label
            if k == 3:
                for i, zoom in enumerate([1.0, 0.1, 0.01]):
                    problem.plotter.append(MoffattStreamPlotter(filetrunk="stream_%d" % i, fileext="pdf"))
                    problem.plotter[-1].zoom = zoom
                    problem.plotter[-1].label = "zoom %.2f" % (1/zoom)
                    
            problem.quiet(True)
            problem.solve(spatial_adapt=30)
            problem.output()
            print("%-28s %8d %10.2e %s"
                  % (label, problem.ndof(), problem.reach(), problem.elements_per_decade()))
            