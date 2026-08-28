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

"""Restarting from a state file that was written after a remesh keeps the template's CLASS.

A state file records the ``.msh`` of the mesh that was current when it was written, and a restart has
to be built from that file rather than from ``define_geometry`` - the geometry in the file is a
remeshed one, which ``define_geometry`` cannot reproduce from an initial condition. The template that
loads it used to be a plain ``GmshTemplate``, so every subclass lost its class across such a restart,
and with it everything that dispatches on the class: an ``AxisymmetricReconnection`` on the restarted
problem refuses a bulk template that is not a ``TopologicalChangesGmshTemplate``, so a run with
topological changes could not be continued at all.

The template is now built from the original one instead - same class, same attributes the user gave
it, geometry from the file - and it is what the next remesh recreates the geometry with.

Two processes, because the writer's Problem has to be gone before the reader's is made (a second
Problem per process segfaults in the JIT loader), and because the reader must not share the writer's
output directory: initialising it would clear the very ``.msh`` the state file points at.
"""

import os
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh not found")

_SCRIPT = r'''
import sys
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.generic import RemeshWhen, RemeshingOptions


class MyMesh(GmshTemplate):
    """A template with a constructor argument of its own, i.e. one that cannot be re-instantiated
    by the reload without knowing what the user passed."""

    def __init__(self, marker):
        super().__init__()
        self.marker = marker
        self.default_resolution = 0.15
        self.mesh_mode = "tris"

    def define_geometry(self):
        if self.is_first_time():
            p = [self.point(0, 0), self.point(1, 0), self.point(1, 1), self.point(0, 1)]
            self.create_lines(p[0], "bottom", p[1], "right", p[2], "top", p[3], "left", p[0])
        else:
            segs = self.get_boundary_coordinates("domain/top", sort_along_axis="x+")
            pts = [self.point(x, y) for x, y in segs[0]]
            self.spline(pts, name="top")
            self.create_lines(pts[-1], "right", self.point(1, 0), "bottom", self.point(0, 0),
                              "left", pts[0])
        self.plane_surface("bottom", "right", "top", "left", name="domain")


class P(Problem):
    def define_problem(self):
        self.add_mesh(MyMesh(marker="a-user-attribute"))
        eqs = PoissonEquation(source=1) + LaplaceSmoothedMesh()
        eqs += RemeshWhen(RemeshingOptions(max_expansion=1.05, min_expansion=0.95))
        eqs += DirichletBC(u=0) @ ["bottom", "left", "right"]
        # A top wall that creeps upwards, so that the mesh really is remeshed on the way.
        eqs += DirichletBC(u=0, mesh_y=1 + 0.35 * var("time") * var("coordinate_x")) @ "top"
        eqs += DirichletBC(mesh_x=True, mesh_y=True) @ ["bottom", "left", "right"]
        self.add_equations(eqs @ "domain")


mode, outdir, dump = sys.argv[1], sys.argv[2], sys.argv[3]
with P() as problem:
    problem.set_c_compiler("tcc")
    problem.set_output_directory(outdir)
    if mode == "write":
        problem.run(1.0, outstep=0.25, startstep=0.25, temporal_error=None)
        print("REPORT class=%s marker=%s nelement=%d" % (
            type(problem.get_mesh("domain")._templatemesh).__name__,
            problem.get_mesh("domain")._templatemesh.marker,
            problem.get_mesh("domain").nelement()))
        problem.save_state(dump)
    else:
        problem.initialise()
        problem.load_state(dump)
        t = problem.get_mesh("domain")._templatemesh
        print("REPORT class=%s marker=%s nelement=%d" % (
            type(t).__name__, getattr(t, "marker", "<LOST>"), problem.get_mesh("domain").nelement()))
        # And the restored template must still be the one the next remesh goes through.
        problem.run(1.6, outstep=False, startstep=0.1, temporal_error=None)
        print("REPORT class=%s marker=%s nelement=%d" % (
            type(problem.get_mesh("domain")._templatemesh).__name__,
            problem.get_mesh("domain")._templatemesh.marker,
            problem.get_mesh("domain").nelement()))
'''


def _run(script, mode, outdir, dump, tmp_path):
    proc = subprocess.run([sys.executable, str(script), mode, outdir, dump],
                          cwd=str(tmp_path), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, "the %s run failed:\n%s\n%s" % (
        mode, proc.stdout[-3000:], proc.stderr[-3000:])
    return [line for line in proc.stdout.splitlines() if line.startswith("REPORT ")]


@pytest.fixture(scope="module")
def restarted(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("state_template")
    script = tmp_path / "run.py"
    script.write_text(_SCRIPT)
    dump = str(tmp_path / "ref" / "final.dump")
    wrote = _run(script, "write", "ref", dump, tmp_path)
    assert os.path.isfile(str(tmp_path / dump)) or os.path.isfile(dump)
    # A separate output directory: initialising the reader would clear "ref" and with it the .msh
    # the state file refers to.
    read = _run(script, "read", "out2", dump, tmp_path)
    return wrote, read


def _fields(line):
    return dict(kv.split("=", 1) for kv in line.split()[1:])


def test_the_writer_really_remeshed(restarted):
    wrote, _ = restarted
    assert len(wrote) == 1
    assert _fields(wrote[0])["class"] == "MyMesh"


def test_the_restart_keeps_the_template_class(restarted):
    _, read = restarted
    assert len(read) == 2, "the restarted run did not get past the remesh"
    assert _fields(read[0])["class"] == "MyMesh", \
        "restarting from a post-remesh state file fell back to a plain GmshTemplate"


def test_the_restart_keeps_the_users_own_attributes(restarted):
    _, read = restarted
    assert _fields(read[0])["marker"] == "a-user-attribute"


def test_the_restart_gets_the_remeshed_mesh_not_the_initial_one(restarted):
    wrote, read = restarted
    assert int(_fields(read[0])["nelement"]) == int(_fields(wrote[0])["nelement"]), \
        "the restored mesh is not the one the state file was written from"


def test_the_restored_template_is_what_the_next_remesh_uses(restarted):
    _, read = restarted
    assert _fields(read[1])["class"] == "MyMesh"
    assert _fields(read[1])["marker"] == "a-user-attribute"
