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
from pyoomph.equations.navier_stokes import NavierStokesEquations, NoSlipBC
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.tracers import TracerParticles, TracerSeedGrid
from pyoomph.output.plotting import MatplotlibPlotter
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class TracerPlotter(MatplotlibPlotter):
    def define_plot(self):
        self.background_color = "darkgrey"
        self.set_view(-0.05, -0.05, 4.05, 1.35)
        cb = self.add_colorbar("velocity", position="bottom right")
        self.add_plot("domain/velocity", colorbar=cb)
        # The tracer collection is addressed by its name, exactly like a field. The trail is drawn
        # from the rolling position history, so it needs `history_time` to have been set below.
        tr = self.add_plot("domain/tracers")
        tr.trail = True
        tr.color = "white"
        tr.size = 5


class WavyChannel(Problem):
    """Channel flow whose top wall oscillates, so that the mesh keeps moving under the particles.

    The point of the example is that the tracers do not care: in a bulk domain the mesh velocity
    cancels out of the advection exactly, so what they follow is the flow and nothing else.
    """

    def __init__(self):
        super().__init__()
        self.amplitude = 0.2
        self.omega = 2 * pi

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(size=[4, 1], N=[40, 10]))

        eqs = MeshFileOutput()
        eqs += NavierStokesEquations(dynamic_viscosity=1, mass_density=1)
        # A moving mesh, driven by the wall motion below.
        eqs += LaplaceSmoothedMesh()

        # Poiseuille inflow, free outflow, no slip on the walls.
        y = var("coordinate_y")
        eqs += DirichletBC(velocity_x=4 * y * (1 - y), velocity_y=0) @ "left"
        eqs += DirichletBC(mesh_x=True, mesh_y=True) @ "left"
        eqs += DirichletBC(mesh_x=True) @ "right"
        eqs += NoSlipBC() @ "bottom" + DirichletBC(mesh_y=0) @ "bottom"
        # The top wall travels up and down. Its normal velocity has to appear in the flow boundary
        # condition too, otherwise the fluid would not know the wall is moving.
        wall = 1 + self.amplitude * sin(self.omega * var("time")) * sin(pi * var("coordinate_x"))
        eqs += DirichletBC(mesh_y=wall, velocity_x=0,
                           velocity_y=partial_t(wall, ALE=False)) @ "top"

        # Tracers. `history_time` is what makes the trails in the plot possible; `payloads` gives
        # each particle a scalar integrated along its own path - here simply the time it has spent
        # in the domain, which the outlet then removes along with the particle.
        eqs += TracerParticles(var("velocity"),
                               seed=TracerSeedGrid(0.15),
                               history_time=0.4,
                               payloads={"residence": 1},
                               statistics=True)

        self += eqs @ "domain"
        self += TracerPlotter(self)


if __name__ == "__main__":
    with WavyChannel() as problem:
        problem.run(2, outstep=0.02, startstep=0.01, maxstep=0.02, temporal_error=1)
