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
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *
from pyoomph.meshes.zeta import *
# The topological changes live in their own module: an interface equation that detects the event
# and a Gmsh template that knows how to rebuild the mesh once it has happened.
from pyoomph.equations.topological_changes import *
# The plan of a pinch-off is worked out by shapely, an optional dependency of pyoomph
# (pip install pyoomph[topology]). It is not required in this file, but the run will stop with an error if it is not installed, 
# so we import it here to make that clear.
import shapely  # noqa: F401

import numpy


class PinchingJetMesh(TopologicalChangesGmshTemplate):
    def define_geometry(self):
        self.mesh_mode = "tris"
        pr = cast("RayleighPlateauPinchOffProblem", self.get_problem())
        
        self.gmsh_options["Mesh.MeshSizeMin"] = pr.hmin
        self.gmsh_options["Mesh.MeshSizeMax"] = pr.hmax
        self.gmsh_options["Mesh.MeshSizeFromCurvature"] = 10
        # Gmsh parallelises its 2d meshing and the resulting mesh differs in the last bits from run to run
        # Deactivating threading it makes it deterministic        
        self.gmsh_options["General.NumThreads"] = 1

        if self.is_first_time():
            # The initial mesh: one full wavelength of the perturbed column, r=1+a*cos(k*z).
            zs = numpy.linspace(0.0, pr.L, 61)
            pts = [self.point(1 + pr.a * numpy.cos(pr.k * z), z) for z in zs]
            interfaces = [self.spline(pts, name="interface")]
            axes = [self.create_lines(pts[0], "bottom", self.point(0, 0),
                                      "axisymm", self.point(0, pr.L), "top", pts[-1])[1]]
        else:
            # This branch is entered for an ordinary quality remesh and for pinch-off events
            # Since the topology might change, we have to take a detour here:
            rb = self.get_reconnected_boundaries("liquid/interface", "liquid/axisymm")
            # One interface chain per connected fluid fragment, sorted by ascending y, and one axis
            # segment per fragment. Before the pinch there is one of each, afterwards three.
            interfaces = [self.spline_from_chain(chain, "interface") for chain in rb.interface_chains]
            axes = self.lines_from_axis_segments(rb.axis_segments, "axisymm")
            # The symmetry planes at z=0 and z=L. Every chain end is either "fixed" (it ends on a
            # boundary that was there before, i.e. on one of the two symmetry planes) or "axis" (it
            # ends on the axis of symmetry, where the interface closes itself off with a cap).            
            for chain in rb.interface_chains:
                for index, kind in ((0, chain.end_types[0]), (-1, chain.end_types[1])):
                    if kind == "fixed":
                        x, y = float(chain.points[index][0]), float(chain.points[index][1])
                        self.line(self.point(0.0, y), self.point(x, y),
                                  name="bottom" if y < 0.5 * pr.L else "top")


        self.plane_surface("bottom", "axisymm", "top", "interface", name="liquid")

        # The mesh size fields of the previous section, verbatim except that they now act on a list
        # of interface curves and a list of axis curves instead of on a single one of each
        at_interface = self.add_mesh_size_field("MathEval", F="x/" + str(pr.min_elements_per_radius))
        restr_interface = self.add_mesh_size_field("Restrict", InField=at_interface, CurvesList=interfaces)
        to_interface = self.add_mesh_size_field("Distance", CurvesList=interfaces, Sampling=400)
        at_axis = self.add_mesh_size_field("MathEval",
                                           F="F" + str(to_interface) + "/" + str(0.75 * pr.min_elements_per_radius))
        restr_axis = self.add_mesh_size_field("Restrict", InField=at_axis, CurvesList=axes)
        combined = self.add_mesh_size_field("Min", FieldsList=[restr_interface, restr_axis])
        self.set_mesh_size_background_field(combined)


class RayleighPlateauPinchOffProblem(Problem):
    # Nondimensionalised by the unperturbed radius R0, the density and the surface tension, so
    # lengths are in R0, time is in the inertio-capillary time and the viscosity is the Ohnesorge
    # number - exactly as in the previous section.
    def __init__(self):
        super().__init__()
        self.Oh = 0.1                        # Ohnesorge number
        self.k = 0.697                       # the inviscid fastest-growing wavenumber, k*R0 < 1
        self.L = 2 * numpy.pi / self.k       # one FULL wavelength, so the neck is in the interior
        self.a = 0.5                         # amplitude of the perturbation
        self.min_elements_per_radius = 2.5   # elements across the local radius
        self.hmin = 0.04                     # finest allowed element
        self.hmax = 0.35                     # coarsest allowed element
        # The neck radius at which we declare the column broken
        self.rmin = 2 * self.hmin
        self.post_pinch_steps = 6            # how far to continue past the event

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(PinchingJetMesh())

        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=self.Oh)
        eqs += HyperelasticSmoothedMesh()
        eqs += MeshFileOutput()
        eqs += RemeshWhen(RemeshingOptions())
        
        eqs += AxisymmetryBC() @ "axisymm"
        eqs += DirichletBC(mesh_y=True, velocity_y=0) @ ["top", "bottom"]
        eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"

        # This single line is what allows the interface to change its topology: a neck whose minimal
        # radius drops below rmin pinches off.
        eqs += AxisymmetricReconnection(rmin=self.rmin) @ "interface"

        # The volume is our accuracy check: the surgery is volume-conserving by construction, so
        # whatever the volume does at the event must be no worse than at an ordinary remesh.
        eqs += IntegralObservables(volume=1)
        eqs += ExtremumObservables(min_r=var("mesh_x")) @ "interface"

        # Accurate interpolation of the interface fields upon remeshing, as in the previous section.        
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="y+") @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axisymm"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "top"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "bottom"

        self.add_equations(eqs @ "liquid")

    def get_minimum_radius(self):
        return float(self.get_mesh("liquid/interface").evaluate_minimum("min_r", dimensional=False,
                                                                        as_float=True))

    def get_volume(self):
        return float(self.get_mesh("liquid").evaluate_observable("volume"))

    def get_number_of_fragments(self):
        # How many connected pieces the interface consists of, i.e. how many drops there are.
        # The interface mesh is a set of line elements, and the data cache can sort them into
        # connected segments for us.
        data = self.get_cached_mesh_data("liquid/interface",tesselate_tri=False,nondimensional=True)        
        return len(data.get_interface_line_segments()[0])

    def run_until_broken(self, dt_factor=0.15, maxstep=0.25, post_dt=0.01, max_steps=200):
        # As in the previous section, we tie the time step to the minimum radius: the inertial
        # collapse follows r ~ (t0-t)^(2/3), so a step that is a fixed fraction of r can never jump
        # over the event. After the break-up, however, the minimum radius is the tip of one of the
        # fresh caps, i.e. exactly zero, and is no longer the scale that has to be resolved - so we
        # switch to a fixed, modest step there.
        t, pinched_at = self.get_current_time(dimensional=False, as_float=True), None
        fragments = self.get_number_of_fragments()
        for step in range(max_steps):
            t += post_dt if pinched_at is not None else min(maxstep, dt_factor * self.get_minimum_radius())
            self.run(t, outstep=False, maxstep=maxstep, temporal_error=1, do_not_set_IC=True)
            yield self.get_current_time(dimensional=False, as_float=True)

            # Every event, not only the first one: this column produces a satellite, and a satellite
            # can pinch again. The restart below is owed to each of them.
            if self.get_number_of_fragments() != fragments:
                fragments = self.get_number_of_fragments()
                if pinched_at is None:
                    pinched_at = step
                # Let the first step after pinch-off become a  BDF1 instead of BDF2, because the history of the new fragment is only accurate to first order.
                self.timestepper.set_num_unsteady_steps_done(0)
            if pinched_at is not None and step - pinched_at >= self.post_pinch_steps:
                return


if __name__ == "__main__":
    with RayleighPlateauPinchOffProblem() as problem:
        problem.DTSF_max_increase_factor = 1.25
        problem.DTSF_min_decrease_factor = 0.75
        problem.initialise()

        history = problem.create_text_file_output("pinchoff.txt",
                                                  header=["t", "r_min", "fragments", "volume"])
        problem.output()
        history.add_row(0.0, problem.get_minimum_radius(), 1, problem.get_volume())

        for t in problem.run_until_broken():
            problem.output()
            history.add_row(t, problem.get_minimum_radius(), problem.get_number_of_fragments(),
                            problem.get_volume())
