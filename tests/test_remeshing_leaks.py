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

# A remeshed Problem must stay garbage-collectible.
#
# Nothing in pyoomph frees a Problem eagerly unless the script uses "with" or calls release(); the
# ordinary script relies on Problem.__del__ running while the interpreter is still up. That only
# happens if the Problem is collectible, and remeshing kept breaking exactly that: the mesh a
# Problem holds is pinned by an nb::keep_alive record, which is a strong reference that Python's
# cyclic collector cannot see, so any reference path leading from a mesh back to its Problem closes
# a cycle with one invisible edge - which gc can never break. The whole Problem (meshes, nodes,
# elements, equations) then survives to interpreter shutdown and nanobind reports
# "nanobind: leaked N instances!" on the way out.
#
# Two such paths have been found and fixed so far: superseded meshes keeping their _templatemesh
# (_destroy_superseded_mesh(), generic/problem.py), and Remesher2d storing the Problem in a plain
# attribute (RemesherBase.problem is a live lookup now, meshes/remesher.py). The class of bug is
# easy to reintroduce with any new back-reference, hence this test.
#
# Run in a subprocess: the symptom is a message nanobind prints during interpreter finalization,
# long after any assertion inside the test could observe it.

import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent("""
    from pyoomph import *
    from pyoomph.meshes.remesher import Remesher2d

    class CircleSegment(GmshTemplate):
        def define_geometry(self):
            self.default_resolution = 0.2
            p00 = self.point(0, 0)
            if not self.is_remeshing():
                p10, p01 = self.point(1, 0), self.point(0, 1)
                self.circle_arc(p10, p01, center=p00, name="interface")
            else:
                # Rebuild the arc from where the nodes of the mesh being replaced actually are.
                coords = self.get_boundary_coordinates("domain/interface", sort_along_axis="x+")
                pts = [self.point(x, y) for x, y in coords[0]]
                self.spline(pts, name="interface")
                p10, p01 = pts[-1], pts[0]
            self.create_lines(p10, "substrate", p00, "axis", p01)
            self.plane_surface("substrate", "axis", "interface", name="domain")

    p = Problem()
    mesh = CircleSegment()
    %s
    p += mesh
    p.set_output_directory("leakcheck")
    p.quiet()
    p += (ElementSpace("C2") + Equations() @ "interface") @ "domain"
    p.initialise()
    p.force_remesh()
    print("REMESHED", flush=True)
""")


@pytest.mark.parametrize("remesher", [
    pytest.param("", id="via_recreation"),                      # the default remesher of any GmshTemplate
    pytest.param("mesh.remesher = Remesher2d(mesh)", id="remesher2d"),
])
def test_remeshed_problem_is_not_leaked(tmp_path, remesher):
    """A script that remeshes and then just ends must not leak its Problem.

    Deliberately no "with" block and no release() call: those tear the Problem down by hand and
    would hide the very cycle this checks for."""
    pytest.importorskip("gmsh", reason="remeshing needs gmsh")
    script = tmp_path / "remesh_case.py"
    script.write_text(_SCRIPT % remesher)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        "exited %d\n--- stdout ---\n%s\n--- stderr tail ---\n%s"
        % (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:]))
    assert "REMESHED" in proc.stdout, "the script never got as far as remeshing"
    output = proc.stdout + proc.stderr
    assert "nanobind: leaked" not in output, (
        "the remeshed Problem was still alive at interpreter shutdown -- some back-reference from "
        "a mesh (or its template/remesher/code generator) to the Problem closes a cycle through "
        "the invisible nb::keep_alive edge again:\n%s"
        % "\n".join(l for l in output.splitlines() if "leaked" in l or "nanobind" in l))
