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
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *
from pyoomph.equations.viscoelastic import *
from pyoomph.meshes.zeta import *
import numpy


class FilamentMesh(GmshTemplate):
    """One filament of length L, bounded by the axis on the left and the free surface on the right.

    The same template is used for the initial mesh and for every remesh: only the source of the
    interface points differs, everything else - including the mesh size fields - is shared.
    """

    def define_geometry(self):
        self.mesh_mode = "tris"
        pr = cast("BeadsOnStringProblem", self.get_problem())
        # Never finer than the thinnest thread we intend to resolve, never coarser than the
        # unperturbed radius, and fine enough to follow the curvature of the beads
        self.set_gmsh_parameter("Mesh.MeshSizeMin", pr.thinnest_thread / pr.elements_per_radius)
        self.set_gmsh_parameter("Mesh.MeshSizeMax", 1.0 / pr.max_elements_per_radius)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 8 * pr.elements_per_radius)

        p_axis_bot, p_axis_top = self.point(0, 0), self.point(0, pr.L)
        if self.is_first_time():
            zs = numpy.linspace(0, float(pr.L), pr.interface_spline_points)
            pts = [self.point(pr.initial_radius(z), z) for z in zs]
        else:
            # Remeshing: rebuild the interface from its current shape
            segments = self.get_boundary_coordinates("liquid/interface", sort_along_axis="y+")
            pts = [self.point(x, y) for x, y in segments[0]]
        interface = self.spline(pts, name="interface")

        axis = self.create_lines(pts[0], "bottom", p_axis_bot, "axisymm", p_axis_top, "top", pts[-1])[1]
        self.plane_surface("bottom", "axisymm", "top", "interface", name="liquid")

        # The mesh size must follow the *local* radius: once the thread has thinned to a few
        # percent of R0, a mesh that is uniform on the scale of the beads has no elements across
        # it at all. Same construction as in the Rayleigh-Plateau example: the interface takes its
        # size from its own distance to the axis, the axis from its distance to the interface.
        size_at_interface = self.add_mesh_size_field("MathEval", F="x/" + str(pr.elements_per_radius))
        restricted_to_interface = self.add_mesh_size_field("Restrict", InField=size_at_interface,
                                                           CurvesList=[interface])
        distance_to_interface = self.add_mesh_size_field("Distance", CurvesList=[interface], Sampling=1000)
        size_at_axis = self.add_mesh_size_field(
            "MathEval", F="F" + str(distance_to_interface) + "/" + str(pr.elements_per_radius * 0.75))
        restricted_to_axis = self.add_mesh_size_field("Restrict", InField=size_at_axis, CurvesList=[axis])
        self.set_mesh_size_background_field(
            self.add_mesh_size_field("Min", FieldsList=[restricted_to_interface, restricted_to_axis]))


class BeadsOnStringProblem(Problem):
    r"""Capillary thinning of an Oldroyd-B filament, i.e. the COMSOL model of the same name.

    Everything is nondimensionalised with the unperturbed radius :math:`R_0`, the surface tension
    :math:`\sigma`, the density :math:`\rho` and the inertio-capillary time
    :math:`\tau=\sqrt{\rho R_0^3/\sigma}`, which leaves

        Oh = eta_0/sqrt(rho*sigma*R0)   the Ohnesorge number,
        De = lambda/tau                 the dimensionless relaxation time,
        beta = eta_s/eta_0              the solvent fraction of the viscosity,

    so that the dimensionless solvent and polymer viscosities are beta*Oh and (1-beta)*Oh.
    """

    def __init__(self):
        super().__init__()
        self.Oh = 3.16               # Ohnesorge number
        self.De = 94.9               # Deborah number, i.e. the dimensionless relaxation time
        self.beta = 0.25             # solvent viscosity ratio
        self.perturbation = 0.05     # amplitude of the initial radius perturbation
        self.wavenumber = 0.5        # r(z,0) = 1 + perturbation*cos(wavenumber*z)
        self.L = 8 * pi              # domain length, i.e. two wavelengths
        self.elements_per_radius = 3      # resolution across the *local* radius
        self.max_elements_per_radius = 5  # upper bound on the resolution, i.e. lower bound on h
        self.thinnest_thread = 0.02       # thinnest thread the mesh is allowed to resolve
        self.interface_spline_points = 81
        # Run control, all overridable from the command line with e.g. -P tend=100
        self.tend, self.outstep, self.maxstep = 50.0, 5.0, 1.0        
        

    def initial_radius(self, z):
        return 1 + self.perturbation * cos(self.wavenumber * z)

    def define_problem(self):
        self.add_mesh(FilamentMesh())
        self.set_coordinate_system("axisymmetric")

        # The Navier-Stokes viscosity is the *solvent* one; the polymer contributes its stress
        # through the viscoelastic equations
        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=self.beta * self.Oh)
        eqs += ViscoelasticEquations(model=OldroydB(), relaxation_time=self.De,
                                     polymer_viscosity=(1 - self.beta) * self.Oh)
        eqs += HyperelasticSmoothedMesh()
        eqs += MeshFileOutput()
        eqs += RemeshWhen(RemeshingOptions())

        eqs += AxisymmetryBC() @ "axisymm"
        # The reference imposes periodicity at the two ends. The initial perturbation has a maximum
        # at both of them, so the solution is mirror-symmetric there and a symmetry plane - free
        # slip for the flow, a mesh that may only slide radially - imposes the same thing.
        eqs += DirichletBC(mesh_y=True, velocity_y=0) @ ["top", "bottom"]
        # No traction condition of its own is needed for the polymer: the momentum residual is
        # assembled in weak form, so the free surface sees the total stress, solvent plus polymer.
        eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"

        eqs += ExtremumObservables(min_r=var("mesh_x")) @ "interface"
        eqs += TextFileOutput() @ "interface"

        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "bottom"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "top"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axisymm"
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="y+") @ "interface"

        self.add_equations(eqs @ "liquid")
        # output file of the minimum
        self.minimum_out = self.create_text_file_output("minimum.txt", header=["t", "r_min", "z_min"])
        
        
    def minimum_radius_and_position(self):
        return self.get_mesh("liquid/interface").evaluate_minimum("min_r", return_x=True)[1]
            
    def output(self, stage= "", quiet= None, **kwargs):
        super().output(stage, quiet, **kwargs)
        self.minimum_out.add_row(self.get_current_time(), *self.minimum_radius_and_position())    


if __name__ == "__main__":
    with BeadsOnStringProblem() as problem:
        problem.initialise()                
        problem.run(problem.tend, outstep=problem.outstep, maxstep=problem.maxstep, temporal_error=1)        