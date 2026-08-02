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
# The viscoelastic equations and the Oldroyd-B model, plus two helpers for the inflow condition
from pyoomph.equations.viscoelastic import (ViscoelasticEquations, OldroydB,
                                            symmetric_2x2_matrix_log, oldroyd_b_shear_conformation)
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.output.plotting import MatplotlibPlotter
from math import pi, cos, sin
import numpy

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
        self.default_resolution = pr.far_resolution
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

        # The outer boundary is a single loop, closed by the outer ring of the O-grid. The top wall is
        # split by an extra point directly above the cylinder, carrying a much smaller target size:
        # gmsh grades away from it, so the elements are fine in the gap and coarse far up- and
        # downstream, where nothing happens
        L, H = pr.channel_length, 2.0
        top_centre = self.point(0, H, size=pr.near_resolution)
        box = self.create_lines(outer[0], "symmetry", self.point(L, 0), "outlet", self.point(L, H), "top",
                                top_centre, "top", self.point(-L, H), "inlet",
                                self.point(-L, 0), "symmetry", outer[2])
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
        # Resolution is set by hand rather than by an error estimator. A Z2 estimator is of little use
        # here: far up- and downstream the flow is fully developed, so the constitutive equation has
        # no spatial derivatives left and is satisfied exactly at every node whatever the mesh, yet
        # the estimator still sees a nonzero recovered-flux error there and spends most of the budget
        # on it. These two numbers put the elements where the physics is instead
        self.far_resolution, self.near_resolution = 1.5, 0.09
        # The O-grid: the polymer stress forms a thin boundary layer on the cylinder, and this is
        # what resolves it
        self.n_circumferential, self.n_radial, self.layer_growth = 80, 20, 1.25

    def define_problem(self):
        self += ConfinedCylinderMesh()
        # Creeping flow, so Stokes rather than Navier-Stokes. Its viscosity is the SOLVENT one
        stokes = StokesEquations(dynamic_viscosity=BETA, mode="TH")
        eqs = stokes + MeshFileOutput()

        # Wi enters as a global parameter so that we can continue in it later on
        self.Wi = self.define_global_parameter(Wi=0.1)
        # SUPG is not decoration here. The constitutive equation has no diffusion at all - its only
        # spatial operator is the advection u.grad(Psi) - and the polymer stress grows exponentially
        # just behind the rear stagnation point. Plain Galerkin answers that with a node-to-node
        # sawtooth all along the wake, which leaves the drag almost untouched (it is an integral over
        # the cylinder) but ruins any profile plotted through the wake. The reference stabilises too,
        # with DEVSS-G/DG
        viscoelastic = ViscoelasticEquations(model=OldroydB(), relaxation_time=self.Wi,
                                             polymer_viscosity=1 - BETA, stabilization="SUPG")
        eqs += viscoelastic

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
        # ... and the shear component of the conformation tensor vanishes on the symmetry line too,
        # being odd under y -> -y. This is not optional: the constitutive equation has no diffusion,
        # so nothing else damps an odd mode in that component, and leaving it free lets a node-to-node
        # sawtooth grow along the whole wake
        eqs += DirichletBC(log_conformation_xy=0) @ "symmetry"
        eqs += NoSlipBC() @ "cylinder"
        # The velocity is prescribed on the entire boundary, so the pressure needs a datum
        eqs += stokes.create_pressure_fixation(value=0) @ "inlet"

        eqs += FlowDirectedStresses()
        eqs += CylinderDrag() @ "cylinder"
        self += eqs @ "fluid"

    def drag(self):
        return float(self.get_mesh("fluid/cylinder").evaluate_observable("drag"))


class FlowDirectedStresses(Equations):
    """
    The flow-directed shear and normal stress of Bollada & Phillips, as output fields.

    These are what Claus & Phillips contour in their Fig. 12: the Cauchy stress is made traceless,
    T0 = sigma - 1/2*tr(sigma)*I, and then projected onto the streamline direction and its normal.
    The pressure drops out of T0 identically - in an incompressible plane flow tr(sigma) is
    -2p + tr(tau_p) - so only the solvent rate of strain and the polymer stress survive.

    It has to be an Equations rather than a plain expression built in define_problem, because
    get_polymer_stress() needs the code generator's scope to resolve the fields it is made of.
    """

    def define_residuals(self):
        combined = self._get_combined_element()
        stokes = combined.get_equation_of_type(StokesEquations)
        viscoelastic = combined.get_equation_of_type(ViscoelasticEquations)
        u = var("velocity")
        T0 = 2 * stokes.dynamic_viscosity * sym(grad(u)) + viscoelastic.get_polymer_stress()
        T0 = T0 - 0.5 * (T0[0, 0] + T0[1, 1]) * identity_matrix(3)
        # Unit vectors along and across the streamline. The regularisation keeps them finite at the
        # stagnation points fore and aft of the cylinder, where the velocity vanishes
        speed = square_root(dot(u, u) + 1e-12)
        par = vector(u[0] / speed, u[1] / speed)
        perp = vector(-par[1], par[0])
        self.add_local_function("S1", dot(perp, matproduct(T0, par)))
        self.add_local_function("S2", dot(par, matproduct(T0, par)))


class FlowStressPlotter(MatplotlibPlotter):
    """One panel of Fig. 12: a flow-directed stress around the cylinder"""

    field = "S1"
    title = "$S_1$"
    #: Fixed across the rows so that the three Weissenberg numbers are comparable, and matching the
    #: ranges of the reference figure. vmin/vmax can only WIDEN the range, never clip it, which is
    #: exactly what is wanted here: every panel ends up on (at least) this common scale
    vmin, vmax = -11.0, 15.0

    def define_plot(self):
        self.background_color = "white"
        self.set_view(-3.0, 0.0, 6.0, 2.05)      # the half that is actually solved
        # "jet" to match the discrete rainbow scale of the reference figure. vmin/vmax would only
        # widen the range, never clip it, so the scale is left to the data - which is the point of
        # the comparison, since the reference quotes its own extrema panel by panel
        cb = self.add_colorbar(self.title, cmap="jet", position="top center",
                               vmin=self.vmin, vmax=self.vmax, length=0.6, thickness=0.05)
        cb.textcolor, cb.textsize = "black", 13
        self.add_plot("fluid/" + self.field, colorbar=cb)
        self.add_plot("fluid/cylinder", linecolor="black", linewidths=1.2)


def stress_profile(problem, Wi):
    """
    tau_xx along the path used in Fig. 6 of Claus & Phillips: up the centreline, over the cylinder
    surface, and away down the wake.

    The polymer stress is not a degree of freedom - the unknown is Psi = log(C) - so it is rebuilt
    here from the nodal values, exactly as the equations do internally: C = exp(Psi) and then
    tau_xx = eta_p/lambda*(C_xx - 1).
    """
    mesh = problem.get_mesh("fluid")
    idx = mesh.get_nodal_field_indices()
    ixx, ixy, iyy = idx["log_conformation_xx"], idx["log_conformation_xy"], idx["log_conformation_yy"]
    xs, taus = [], []
    for node in mesh.nodes():
        x, y = node.x(0), node.x(1)
        on_cylinder = abs(numpy.hypot(x, y) - 1.0) < 1e-9
        on_centreline = abs(y) < 1e-9 and abs(x) >= 1.0
        if not (on_cylinder or on_centreline):
            continue
        psi = numpy.array([[node.value(ixx), node.value(ixy)],
                           [node.value(ixy), node.value(iyy)]])
        w, Q = numpy.linalg.eigh(psi)
        C = Q @ numpy.diag(numpy.exp(w)) @ Q.T
        xs.append(x)
        taus.append((1 - BETA) / Wi * (C[0, 0] - 1.0))
    order = numpy.argsort(xs)
    return numpy.array(xs)[order], numpy.array(taus)[order]


def plot_stress_profiles(profiles, filename):
    """Fig. 6 of Claus & Phillips, with their axes, for direct visual comparison"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = matplotlib.colormaps["viridis"]
    for i, (Wi, (x, tau)) in enumerate(sorted(profiles.items())):
        ax.plot(x, tau, color=cmap(i / max(1, len(profiles) - 1)), lw=1.4, label="Wi=%.1f" % Wi)
    ax.set_xlim(-3, 5)                     # the ranges of their Fig. 6
    ax.set_ylim(0, 130)
    ax.set_xlabel("$X$")
    ax.set_ylabel(r"$\tau_{xx}$")
    ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(filename + "." + ext, dpi=150)
    plt.close(fig)


def assemble_panels(panels, filename, titles):
    """Stack the individual S1/S2 panels into the 3x2 grid of Fig. 12"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 1.3 * len(panels)))
    for row, (Wi, files) in enumerate(sorted(panels.items())):
        for col in range(2):
            ax = axes[row][col]
            ax.imshow(mpimg.imread(files[col]))
            ax.set_axis_off()
            if row == 0:
                ax.set_title(titles[col], fontsize=12)
            if col == 0:
                ax.text(-0.02, 0.5, "Wi = %.1f" % Wi, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(filename + "." + ext, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    with ConfinedCylinderProblem() as problem:
        problem.initialise()
        problem.solve()              # the solution at the initial Wi, which go_to_param starts from
        print("  Wi      K (pyoomph)   K (Claus & Phillips)")
        profiles, panels = {}, {}
        for Wi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            # go_to_param continues towards the target and halves its step whenever Newton fails,
            # which is more robust than stepping by hand: a cold start at larger Wi would overshoot
            # into a conformation tensor so stretched that exp(Psi) overflows on the first step
            problem.go_to_param(Wi=Wi)
            profiles[Wi] = stress_profile(problem, Wi)
            reference = REFERENCE_DRAG.get(Wi)
            print("  %.1f     %9.4f       %s" % (Wi, problem.drag(),
                                                 "%9.3f" % reference if reference else "        -"))
            if Wi in (0.1, 0.5, 0.7):        # the three rows of their Fig. 12
                files = []
                for field, title, lo, hi in (("S1", "$S_1$", -11.0, 15.0), ("S2", "$S_2$", -8.0, 55.0)):
                    plotter = FlowStressPlotter(problem, filetrunk="%s_Wi%02d" % (field, round(10 * Wi)))
                    plotter.field, plotter.title = field, title
                    plotter.vmin, plotter.vmax = lo, hi
                    problem.plotter = [plotter]
                    problem.output()
                    files.append(problem.get_output_directory("_plots/" + plotter.file_trunk + ".png"))
                panels[Wi] = files
        plot_stress_profiles(profiles, "viscoelastic_stress")
        assemble_panels(panels, "viscoelastic_flowstress", ["$S_1$ (flow-directed shear)",
                                                            "$S_2$ (flow-directed normal)"])
