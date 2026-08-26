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

# Three defects of the mesh-to-mesh transfer that ran after remeshing, all of them silent: the run
# went on with wrong values rather than saying anything.
#
# (a) The bulk fallback for a boundary that has no interface mesh on either side - "corners to
#     another domain" - passed the OLD mesh's boundary index to a call that applies it to the NEW
#     mesh's nodes (Mesh::node_is_in_scope). Boundary indices are handed out per mesh in the order
#     the template's nodes are visited, so the two need not agree; when they do not, the boundary
#     whose new index nobody claimed is never transferred at all and keeps whatever it was built
#     with. Zero, in a freshly generated mesh.
#
# (b) Mesh::nodal_interpolate_along_boundary built its map of interface-added dofs from the BULK
#     function tables. A bulk code has numfields == numfields_basebulk, so the map came out empty and
#     every interface-only field - a surfactant concentration, a Lagrange multiplier - was dropped;
#     the codimension-2 call looked at the codim-1 interface's table instead of the codim-2 mesh's,
#     so that mesh's own dofs were dropped in the same way.
#
# (c) The codimension-2 pass runs last and wrote the BULK fields of the corner nodes too, replacing
#     the values the per-boundary passes had properly interpolated with a two-nearest-node blend.
#     For a corner that stays where it was that is invisible - the nearest old node is the old corner
#     - which is why the case below moves it.
#
# One Problem per process (a second one segfaults in the JIT loader), so each case runs in its own
# subprocess, as in tests/test_mpi_remeshing.py.

import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "boundary_interpolation_worker.py")

pytestmark = pytest.mark.skipif(shutil.which("gmsh") is None, reason="gmsh not found")

#: The transferred fields are quadratics, which C2 represents exactly, so a correct transfer is exact
#: to round-off. Loose enough for the projection solve, far tighter than any of the three defects.
EXACT = 1e-9


def _run(case, tmp_path):
    outdir = str(tmp_path / case)
    proc = subprocess.run([sys.executable, _WORKER, "--case", case, "--outdir", outdir],
                          cwd=_HERE, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, \
        "the %s worker failed:\n--- stdout tail ---\n%s\n--- stderr tail ---\n%s" % (
            case, proc.stdout[-3000:], proc.stderr[-3000:])
    return proc


def _lines(proc, prefix):
    out = []
    for line in proc.stdout.splitlines():
        if line.startswith(prefix + " "):
            out.append(dict(f.split("=", 1) for f in line.split()[1:]))
    return out


@pytest.fixture(scope="module")
def two_domain(tmp_path_factory):
    return _run("two_domain", tmp_path_factory.mktemp("two_domain"))


@pytest.fixture(scope="module")
def interface_field(tmp_path_factory):
    return _run("interface_field", tmp_path_factory.mktemp("interface_field"))


def test_the_case_really_renumbers_the_boundaries(two_domain):
    """Guards the two tests below against passing for the wrong reason.

    Both of them are about a bulk fallback that used the wrong mesh's boundary index. If the remeshed
    template happened to number the boundaries exactly as the old one did, the wrong index would be
    the right one and neither test would be measuring anything.
    """
    got = _lines(two_domain, "PYOOMPH_INDICES")
    assert got, "the worker did not report the boundary numbering:\n%s" % two_domain.stdout[-2000:]
    assert int(got[0]["mismatched"]) > 0, (
        "old and new mesh number the boundaries of 'left' identically, so this case cannot tell "
        "whether the fallback uses the destination's index")


def test_the_projection_is_exact_before_the_remesh(two_domain):
    """The premise of the tolerance: what is being transferred is exact to begin with."""
    for fields in _lines(two_domain, "PYOOMPH_PRE"):
        assert float(fields["worst"]) < EXACT, \
            "the projected field is not exact on %s already before the remesh: %s" % (
                fields["domain"], fields["worst"])


def test_every_boundary_is_transferred_although_the_indices_were_permuted(two_domain):
    """Defect (a). The per-boundary error is reported per boundary so a failure names the victim.

    The boundary left without a pass of its own is the one whose NEW index equals the OLD index of
    the boundary that has an interface mesh, i.e. which of them it is depends on the numbering. All
    of them are therefore checked, and the per-boundary numbers say which one it was.
    """
    per_boundary = _lines(two_domain, "PYOOMPH_BND")
    assert per_boundary, "the worker reported no boundaries:\n%s" % two_domain.stdout[-2000:]
    bad = [(f["domain"] + "/" + f["name"], float(f["worst"]))
           for f in per_boundary if float(f["worst"]) >= EXACT]
    assert not bad, (
        "these boundaries did not receive the transferred field (worst |u-exact| per boundary): %s"
        % ", ".join("%s=%.4g" % b for b in bad))


def test_the_moved_corner_keeps_its_interpolated_value(two_domain):
    """Defect (c). The corner moved, so the old corner's value is the wrong answer there."""
    corners = _lines(two_domain, "PYOOMPH_CORNER")
    assert len(corners) == 1, \
        "expected exactly one codimension-2 corner, got %d:\n%s" % (
            len(corners), two_domain.stdout[-2000:])
    err = float(corners[0]["err"])
    assert err < EXACT, (
        "the corner node is off by %.4g -- the codimension-2 pass replaced the value the "
        "per-boundary pass interpolated there by the nearest old node's" % err)


def test_the_whole_mesh_is_transferred_exactly(two_domain):
    """Both defects at once, as the single number a user would look at."""
    for fields in _lines(two_domain, "PYOOMPH_POST"):
        assert float(fields["worst"]) < EXACT, \
            "domain %s is off by %s after the remesh" % (fields["domain"], fields["worst"])


def test_an_interface_only_field_survives_the_legacy_boundary_transfer(interface_field):
    """Defect (b), on the codimension-1 call: the surfactant used not to be transferred at all.

    It ends up nearest-node blended rather than interpolated - that is what this path does - which
    reproduces the linear profile wherever the two matches bracket the new node, so the tolerance is
    loose. What it is really separating is "transferred" from "left at zero".
    """
    pre = _lines(interface_field, "PYOOMPH_SURF")
    got = {f["tag"]: f for f in pre}
    assert set(got) == {"pre", "post"}, \
        "the worker did not report the interface field twice:\n%s" % interface_field.stdout[-2000:]
    assert float(got["pre"]["worst"]) < EXACT, \
        "the interface field was not projected exactly before the remesh: %s" % got["pre"]["worst"]
    assert float(got["post"]["min"]) > 0.4, (
        "an interface node came out of the remesh at %s, i.e. it never received a value at all "
        "(the profile runs from 0.5 to 2.2)" % got["post"]["min"])
    assert float(got["post"]["worst"]) < 0.05, \
        "the interface field is off by %s after the remesh" % got["post"]["worst"]


def test_the_codim2_meshs_own_dof_survives(interface_field):
    """Defect (b) again, one level down: a dof of the codimension-2 mesh itself."""
    got = _lines(interface_field, "PYOOMPH_CL")
    pre = [f for f in got if f["tag"] == "pre"]
    post = [f for f in got if f["tag"] == "post"]
    assert len(pre) == 2 and len(post) == 2, \
        "expected one contact-line dof per side, before and after:\n%s" % interface_field.stdout[-2000:]
    for fields in pre:
        assert float(fields["value"]) == pytest.approx(3.25, abs=EXACT)
    for fields in post:
        assert float(fields["value"]) == pytest.approx(3.25, abs=1e-8), (
            "the codimension-2 mesh's own dof came out of the remesh at %s instead of 3.25 on the "
            "%s side" % (fields["value"], fields["side"]))
