#  Runnable companion to dev_docs/axisymmetric_topological_changes.md, section 1.
#
#  The smallest thing that pinches: one Rayleigh-Plateau wavelength of a liquid column, driven by
#  surface tension alone, with an AxisymmetricReconnection on the free surface. Run it and it prints a
#  line per step and one "Topological change: pinch at ..." when the column breaks; the mesh is then
#  rebuilt by the `else:` branch of define_geometry below, which is also the branch every ordinary
#  quality remesh takes.
#
#      python3 dev_docs/examples/axisymm_reconnection_minimal.py
#
#  Needs shapely (pip install pyoomph[topology]) and gmsh. ~1 min, ~15000 dofs at the end.
#  The assertions in tests/test_rayleigh_plateau_pinchoff.py are made on this same scenario, at the
#  same resolution, through tests/axisymm_physics_worker.py.

import numpy

from pyoomph import Problem, DirichletBC, var
from pyoomph.equations.ALE import HyperelasticSmoothedMesh
from pyoomph.equations.generic import (AxisymmetryBC, ExtremumObservables, IntegralObservables,
                                       RemeshWhen, RemeshingOptions)
from pyoomph.equations.navier_stokes import NavierStokesEquations, NavierStokesFreeSurface
from pyoomph.equations.topological_changes import (AxisymmetricReconnection,
                                                   TopologicalChangesGmshTemplate)
from pyoomph.meshes.zeta import (AssignZetaCoordinatesByArclength,
                                 AssignZetaCoordinatesByEulerianCoordinate)


class JetMesh(TopologicalChangesGmshTemplate):
    def define_geometry(self):
        self.mesh_mode = "tris"
        pr = self.get_problem()
        # Never finer than the neck we intend to resolve, never coarser than the unperturbed radius.
        self.set_gmsh_parameter("Mesh.MeshSizeMin", pr.hmin)
        self.set_gmsh_parameter("Mesh.MeshSizeMax", 0.35)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 10)
        # Single-threaded meshing, so that successive runs are reproducible. Gmsh parallelises its 2d
        # meshing and the mesh differs in the last bits from run to run; on a transient that ends in a
        # capillary singularity those bits separate within a few dozen steps.
        self.set_gmsh_parameter("General.NumThreads", 1)

        if self.is_first_time():
            zs = numpy.linspace(0.0, pr.L, 61)
            pts = [self.point(1 + pr.amplitude * numpy.cos(pr.k * z), z) for z in zs]
            interfaces = [self.spline(pts, name="interface")]
            axes = [self.create_lines(pts[0], "bottom", self.point(0, 0), "axisymm",
                                      self.point(0, pr.L), "top", pts[-1])[1]]
        else:
            # ONE code path for a reconnection and for an ordinary quality remesh: with no plan
            # pending, get_reconnected_boundaries() describes the current geometry instead.
            rb = self.get_reconnected_boundaries("liquid/interface", "liquid/axisymm")
            interfaces = [self.spline_from_chain(ch, "interface") for ch in rb.interface_chains]
            axes = self.lines_from_axis_segments(rb.axis_segments, "axisymm")
            # The two symmetry planes. After the pinch each fragment owns one of them, i.e. one
            # "fixed" chain end and one fresh "axis" cap - so this is a loop, not two literals.
            for ch in rb.interface_chains:
                for end, kind in ((0, ch.end_types[0]), (-1, ch.end_types[1])):
                    if kind == "fixed":
                        x, y = float(ch.points[end][0]), float(ch.points[end][1])
                        self.line(self.point(0.0, y), self.point(x, y),
                                  name="bottom" if y < 0.5 * pr.L else "top")

        self.plane_surface("bottom", "axisymm", "top", "interface", name="liquid")

        # Resolve the local radius: the interface takes its mesh size from its own distance to the
        # axis, the axis from its distance to the interface. Without this a neck a few percent of R0
        # across has no element in it at all.
        at_interface = self.add_mesh_size_field("MathEval", F="x/2.5")
        restr_i = self.add_mesh_size_field("Restrict", InField=at_interface, CurvesList=interfaces)
        to_interface = self.add_mesh_size_field("Distance", CurvesList=interfaces, Sampling=400)
        at_axis = self.add_mesh_size_field("MathEval", F="F" + str(to_interface) + "/1.875")
        restr_a = self.add_mesh_size_field("Restrict", InField=at_axis, CurvesList=axes)
        self.set_mesh_size_background_field(
            self.add_mesh_size_field("Min", FieldsList=[restr_i, restr_a]))


class RayleighPlateauProblem(Problem):
    """Nondimensionalised with R0, rho and sigma, so the viscosity is the Ohnesorge number."""

    def __init__(self):
        super().__init__()
        self.Oh = 0.1
        self.k = 0.697                     # the inviscid fastest-growing mode
        self.L = 2 * numpy.pi / self.k     # one wavelength: bulges at both ends, neck in the middle
        self.amplitude = 0.5
        self.hmin = 0.04
        self.rmin = 0.08                   # two elements across the neck when it pinches

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.add_mesh(JetMesh())
        eqs = NavierStokesEquations(mass_density=1, dynamic_viscosity=self.Oh)
        eqs += HyperelasticSmoothedMesh()
        eqs += RemeshWhen(RemeshingOptions())
        eqs += IntegralObservables(volume=1)
        eqs += AxisymmetryBC() @ "axisymm"
        eqs += DirichletBC(mesh_y=True, velocity_y=0) @ ["top", "bottom"]
        eqs += NavierStokesFreeSurface(surface_tension=1) @ "interface"
        eqs += AxisymmetricReconnection(rmin=self.rmin) @ "interface"
        eqs += ExtremumObservables(r=var("mesh_x")) @ "interface"
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="y+") @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axisymm"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "top"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "bottom"
        self.add_equations(eqs @ "liquid")


def _fragments(problem):
    """How many connected pieces the interface has, i.e. how many drops there are."""
    from pyoomph.meshes.meshdatacache import MeshDataCache
    data = MeshDataCache(tesselate_tri=False, nondimensional=True).get_data(
        problem.get_mesh("liquid/interface"))
    return len(data.get_interface_line_segments()[0])


if __name__ == "__main__":
    with RayleighPlateauProblem() as problem:
        problem.initialise()
        t, pinched = 0.0, False
        for step in range(120):
            # dt proportional to the neck radius: the inertial collapse is r ~ (t0-t)^(2/3), so a step
            # that is a fixed fraction of r cannot jump over the event.
            r = float(problem.get_mesh("liquid/interface").evaluate_minimum(
                "r", dimensional=False, as_float=True))
            t += min(0.25, max(2e-3, 0.15 * r))
            problem.run(t, outstep=False, maxstep=0.25, temporal_error=1, do_not_set_IC=True)
            volume = float(problem.get_mesh("liquid").evaluate_observable("volume"))
            n = _fragments(problem)
            print("t = %8.5f   r_min = %7.5f   volume = %.8f   fragments = %d"
                  % (t, r, volume, n), flush=True)
            if n > 1 and not pinched:
                pinched = True
                # Restart the time stepper, once, at the event. A node the surgery created has no
                # history: the transfer gives it whatever the old mesh held at that place, which for
                # a fresh cap is the middle of a neck that was collapsing at the largest velocity in
                # the domain, and BDF2 extrapolates through that on the next step. The alternative,
                # problem.timestepper.set_num_unsteady_steps_done(0) - one step with BDF1 weights -
                # is far more accurate but less robust on a marginally resolved cap retraction; see
                # section 8.4 of dev_docs/axisymmetric_topological_changes.md.
                problem.assign_initial_values_impulsive()
