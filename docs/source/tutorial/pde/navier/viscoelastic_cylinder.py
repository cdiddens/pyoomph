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
from pyoomph.equations.generic import ProjectExpression
from pyoomph.equations.viscoelastic import *
from pyoomph.meshes.gmsh import GmshTemplate
from pyoomph.output.plotting import MatplotlibPlotter
import numpy
import os



REFERENCE_DRAG = {0.1: 130.364, 0.2: 126.626, 0.3: 123.192, 0.4: 120.593,
                  0.5: 118.826, 0.6: 117.776, 0.7: 117.316}


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
        # They are only needed for plotting, see solve_auxiliary_residual("plotting_projection") below
        eqs+=ProjectExpression(S1=dot(perp, matproduct(T0, par)), S2=dot(par, matproduct(T0, par)),destination="plotting_projection")
                        
        self += eqs @ "fluid"

    def drag(self):
        return float(self.get_mesh("fluid/cylinder").evaluate_observable("drag"))



class FlowStressPlotter(MatplotlibPlotter):
    """One panel of Fig. 12: a flow-directed stress around the cylinder"""

    field = "S1"
    title = "$S_1$"
    #: Fixed across the rows so that the three Weissenberg numbers are comparable, and matching the
    #: ranges of the reference figure. vmin/vmax can only WIDEN the range, never clip it, which is
    #: exactly what is wanted here: every panel ends up on (at least) this common scale
    vmin, vmax = -11.0, 15.0
    #: Where widening is not wanted, i.e. for S2, whose maximum on the cylinder runs away with Wi and
    #: would stretch the scale until nothing else on it is distinguishable. Cutting it off costs
    #: nothing that the figure shows anyway, and the colorbar grows an arrow to say so
    clamp_max: float | None = None

    def define_plot(self):
        self.background_color = "white"
        # The domain is the upper half only, so the strip below y=0 is empty: the view is extended
        # into it to give the colorbar somewhere to sit that is not on top of the cylinder
        self.set_view(-3.0, -0.62, 6.0, 2.05)
        # "jet" to match the discrete rainbow scale of the reference figure.
        # No title on the bar itself: the assembled figure labels the columns. Everything is drawn
        # oversized because each panel is shrunk by about a factor of three into the 3x2 grid, which
        # is what made the original colorbar illegible
        cb = self.add_colorbar("", cmap="jet", position="bottom center",
                               vmin=self.vmin, vmax=self.vmax, clamp_max=self.clamp_max,
                               length=0.85, thickness=0.09)
        cb.textcolor, cb.textsize, cb.ticsize = "black", 26, 26
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
        taus.append((1 - problem.beta) / Wi * (C[0, 0] - 1.0))
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


def assemble_panels(panels, filename, titles, panel_width="8cm",
                    row_sep="1pt", column_sep="1pt", dpi=200):
    """
    Stack the individual S1/S2 panels into the 3x2 grid of Fig. 12, with LaTeX.

    The panels go in as the PDFs matplotlib wrote, not as rasterised images: they stay vector all
    the way into the assembled PDF, so the colorbar labels are still text and the figure can be
    scaled afterwards without the panels turning to mush. Reading them back into matplotlib to
    imshow() them, which is what this did before, throws that away at the first step.

    A tikz matrix does the layout. It sizes its cells from their contents, so the row spacing does
    not have to be guessed from the panel aspect ratio - which is the other reason not to do this in
    matplotlib, where the panels sit in fixed-aspect axes with their own padding and the gaps come
    out large whatever tight_layout is asked for. row_sep/column_sep are the whole spacing story here.

    Needs pdflatex (with the standalone class) and pdftocairo, both of which a TeX Live plus poppler
    installation provides.
    """
    import shutil
    import subprocess
    import tempfile

    for tool in ("pdflatex", "pdftocairo"):
        if shutil.which(tool) is None:
            raise RuntimeError("assemble_panels needs " + tool + ", which is not on the PATH. "
                               "The individual panels are written either way, in " +
                               os.path.dirname(list(panels.values())[0][0]))

    rows = []
    for Wi, files in sorted(panels.items()):
        cells = ["\\rotatebox{90}{$\\mathrm{Wi} = %.1f$}" % Wi]
        cells += ["\\includegraphics[width=%s]{%s}" % (panel_width, os.path.abspath(f)) for f in files]
        rows.append(" & ".join(cells) + " \\\\")
    header = " & ".join([""] + list(titles)) + " \\\\"

    document = "\n".join([
        "\\documentclass[tikz,border=1pt]{standalone}",
        "\\usepackage{graphicx}",
        "\\usetikzlibrary{matrix}",
        "\\begin{document}",
        "\\begin{tikzpicture}",
        "\\matrix[matrix of nodes, row sep=%s, column sep=%s," % (row_sep, column_sep),
        "        nodes={inner sep=0pt, outer sep=0pt, anchor=center}] (panels) {",
        header, "\n".join(rows),
        "};",
        "\\end{tikzpicture}",
        "\\end{document}", ""])

    # In a scratch directory, so that a failed run leaves the previous figure alone and none of
    # LaTeX's half a dozen auxiliary files end up next to the script
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "assembled.tex")
        with open(source, "w") as f:
            f.write(document)
        for command in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source],
                        ["pdftocairo", "-png", "-r", str(dpi), "-singlefile",
                         os.path.join(tmp, "assembled.pdf"), os.path.abspath(filename)]):
            run = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            if run.returncode != 0:
                raise RuntimeError(command[0] + " failed while assembling " + filename + ":\n"
                                   + (run.stdout or "") + (run.stderr or ""))
        shutil.copyfile(os.path.join(tmp, "assembled.pdf"), os.path.abspath(filename) + ".pdf")


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
                # S1 and S2 live on their own residual, so they are pinned and cost nothing while the
                # flow is being solved; this is where they are actually wanted
                problem.solve_auxiliary_residual("plotting_projection")
                files = []
                # S2 is capped rather than left to the data: on the cylinder it reaches ~55 at
                # Wi=0.7 and everything below 20 would be squeezed into the bottom third of the
                # scale. The arrow on the colorbar is what says that the cap is not the maximum
                for field, title, lo, hi, cap in (("S1", "$S_1$", -11.0, 15.0, None),
                                                  ("S2", "$S_2$", -8.0, 35.0, 35.0)):
                    plotter = FlowStressPlotter(problem, filetrunk="%s_Wi%02d" % (field, round(10 * Wi)),
                                                fileext="pdf")   # vector panels for assemble_panels
                    plotter.field, plotter.title = field, title
                    plotter.vmin, plotter.vmax, plotter.clamp_max = lo, hi, cap
                    problem.plotter = [plotter]
                    problem.output_at_increased_time()
                    files.append(problem.get_output_directory("_plots/" + plotter.file_trunk + ".pdf"))
                panels[Wi] = files
        plot_stress_profiles(profiles, "viscoelastic_stress")
        assemble_panels(panels, "viscoelastic_flowstress", ["$S_1$ (flow-directed shear)",
                                                            "$S_2$ (flow-directed normal)"])
