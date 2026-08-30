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
# Why this compares two runs rather than asserting "leaked" is simply absent: the absolute count is
# not portable. "from pyoomph import *" (which the ordinary user script uses) binds pyoomph's
# nanobind singletons into __main__, and on some interpreter builds - macOS in particular - nanobind's
# leak check runs during _pyoomph_core finalization BEFORE __main__ is cleared, so those singletons
# are always reported leaked, remesh or not (a constant baseline, independent of problem size). On
# Linux the finalization order clears them first and the baseline is zero. Either way the question that
# matters is the same: does remeshing leave anything alive that an explicit release() would have freed?
# So the baseline is the identical script with p.release() (which tears the Problem's C++ side down
# regardless of any Python cycle); the plain run must not leak MORE than that. A reintroduced cycle
# keeps the whole Problem graph alive in the plain run only, pushing its count above the baseline.
#
# Run in a subprocess: the symptom is a message nanobind prints during interpreter finalization,
# long after any assertion inside the test could observe it.

import re
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
    %s
""")

# Substituted for the second %s. The plain run leaves the Problem for __del__/gc to reach at shutdown
# (the case under test); the baseline run tears it down by hand, so whatever it still reports is the
# platform's unavoidable "from pyoomph import *" shutdown noise, not a live Problem.
_PLAIN = ""
_RELEASED = "import gc as _gc; p.release(); _gc.collect()"


def _run(tmp_path, remesher, teardown, tag):
    script = tmp_path / ("remesh_%s.py" % tag)
    script.write_text(_SCRIPT % (remesher, teardown))
    return subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=600)


def _leaked_instances(output):
    """The count from 'nanobind: leaked N instances!', or 0 if the line is absent."""
    matches = re.findall(r"nanobind: leaked (\d+) instances", output)
    return max((int(m) for m in matches), default=0)


@pytest.mark.parametrize("remesher", [
    pytest.param("", id="via_recreation"),                      # the default remesher of any GmshTemplate
    pytest.param("mesh.remesher = Remesher2d(mesh)", id="remesher2d"),
])
def test_remeshed_problem_is_not_leaked(tmp_path, remesher):
    """A script that remeshes and then just ends must not leak its Problem.

    The plain run deliberately uses no "with" block and no release() call: those tear the Problem down
    by hand and would hide the very cycle this checks for. The released run is the baseline it is
    measured against (see the module comment)."""
    pytest.importorskip("gmsh", reason="remeshing needs gmsh")

    plain = _run(tmp_path, remesher, _PLAIN, "plain")
    released = _run(tmp_path, remesher, _RELEASED, "released")

    for proc, name in ((plain, "plain"), (released, "released")):
        assert proc.returncode == 0, (
            "%s run exited %d\n--- stdout ---\n%s\n--- stderr tail ---\n%s"
            % (name, proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:]))
        assert "REMESHED" in proc.stdout, "the %s run never got as far as remeshing" % name

    plain_leaked = _leaked_instances(plain.stdout + plain.stderr)
    base_leaked = _leaked_instances(released.stdout + released.stderr)

    assert plain_leaked <= base_leaked, (
        "the remeshed Problem was still alive at interpreter shutdown: the plain run leaked %d nanobind "
        "instances, %d more than the %d that survive even when the Problem is released by hand -- some "
        "back-reference from a mesh (or its template/remesher/code generator) to the Problem closes a "
        "cycle through the invisible nb::keep_alive edge again:\n%s"
        % (plain_leaked, plain_leaked - base_leaked, base_leaked,
           "\n".join(l for l in (plain.stdout + plain.stderr).splitlines()
                     if "leaked" in l or "nanobind" in l)))
