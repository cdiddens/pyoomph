#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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

# Outputting a tagged point writes its FIELDS, not just its state dump - and puts everything back.
#
# The redirection is the interesting part. Problem._change_output_directory tells each output where to
# write by storing the new location RELATIVE to the problem's base directory; _MeshFileOutput was the
# one outputter that never overrode the hook, so VTUs ignored it entirely (which also meant
# PeriodicOrbit.output_orbit was writing orbit VTUs on top of the ordinary ones). The assertions below
# would all pass on a text-only problem, so the problem here has a MeshFileOutput on purpose.

"""Does the tagged-point output write fields, and put everything back afterwards?"""
import os, sys
from pyoomph import Problem, Equations, InitialCondition, DirichletBC, MeshFileOutput
from pyoomph.expressions import var_and_test, var, grad, partial_t
from pyoomph.equations.generic import IntegralObservables
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.utils.bifurcation_gui import BifurcationGUI
from pyoomph.utils.bifurcation_gui.controller import _FixedViewLimits
from pyoomph.output.plotting import MatplotlibPlotter


class Plotter(MatplotlibPlotter):
    """A plotter on the problem is the case that used to make the tag output die outright."""
    def define_plot(self):
        self.background_color = "white"
        self.set_view(-0.1, -0.1, 1.1, 1.1)
        cb = self.add_colorbar("u", position="top right")
        self.add_plot("domain/u", colorbar=cb)

class Diff(Equations):
    def __init__(self, mu): super().__init__(); self.mu=mu
    def define_fields(self): self.define_scalar_field("u","C2")
    def define_residuals(self):
        u,v=var_and_test("u")
        self.add_weak(partial_t(u),v); self.add_weak(grad(u),grad(v)); self.add_weak(-self.mu*u,v)

class Prob(Problem):
    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(size=[1,1],N=[4,4]))
        eqs=Diff(self.get_global_parameter("mu"))+InitialCondition(u=0)
        for b in ("left","right","top","bottom"): eqs+=DirichletBC(u=0)@b
        eqs+=MeshFileOutput()
        eqs+=IntegralObservables(_vol=1,_ui=var("u"))
        eqs+=IntegralObservables(u_avg=lambda _vol,_ui: _ui/_vol)
        self+=eqs@"domain"


# Run under a __main__ guard: this file lives in tests/ and the suite is invoked as
# `python -m pytest *.py` (see citools/nightly_develop.sh), which hands pytest every .py in
# the directory by name -- and a file named on the command line is imported regardless of
# python_files. Without the guard the whole problem below is built and solved at COLLECTION
# time, and pyoomph's own argv parsing then reads pytest's first filename argument as the
# output directory, so it tries to mkdir over a source file and collection dies. Every other
# worker in this directory is guarded the same way.
def main():
    with Prob() as problem:
        problem.set_output_directory(sys.argv[1])
        problem.set_linear_solver("superlu")
        problem.get_global_parameter("mu").value=1.0
        problem.plotter=Plotter()
        problem.quiet()
        gui=BifurcationGUI(problem,"mu"); gui.neigen=1
        c=gui.controller; c.view=_FixedViewLimits(xlim=(-5,40),ylim=(-5,5))
        c.start(0.5)
        for _ in range(2): c.step()

        c.tag_selected_point(1)                 # tag the current point
        pts=[p for b in c.branches for p in b if p.tag>=0]
        print("tagged points:",len(pts))
        before_dir=problem.get_output_directory()
        before_step=problem._output_step
        before_point=c.current_point
        n=c.output_tagged_points()
        print("outputs written:",n)
        odir=problem.get_output_directory(os.path.join(c.data_subdir,"output","tag01"))
        files=[f for dp,_,fs in os.walk(odir) for f in fs] if os.path.isdir(odir) else []
        print("tag01 dir:",os.path.isdir(odir)," files:",sorted(files)[:6])
        assert files, "the tag directory must contain the problem's output"
        assert any(f.endswith(".png") for f in files), "the plotter's image must land there too: "+str(files)
        print("outdir restored :",problem.get_output_directory()==before_dir)
        print("outstep restored:",problem._output_step==before_step)
        print("point restored  :",c.current_point is before_point)
        assert problem.get_output_directory()==before_dir and problem._output_step==before_step
        assert c.current_point is before_point
        # No state dump may have leaked into the diagram's own store.
        states=os.listdir(problem.get_output_directory(os.path.join(c.data_subdir,"_states")))
        print("state dumps:",len(states))
        print("TAGOUT OK")
        print("PYOOMPH_WORKER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
