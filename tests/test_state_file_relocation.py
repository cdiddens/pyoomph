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

# A state file of a GmshTemplate problem cannot simply be copied somewhere else.
#
# It names its .msh RELATIVE to the directory the dump sits in, and a load resolves that against the
# directory of the file being loaded - so a copy that lands at a different depth points at nothing,
# and meshio dies with "File .../../_gmsh/Mesh.msh not found". Exporting a tagged point out of the
# bifurcation GUI does exactly that kind of copy, hence states.copy_state_file, which brings the mesh
# along and rewrites the stored path.
#
# Each Problem gets its own process: a second Problem in one interpreter segfaults in the JIT loader.

import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

_PROBLEM = """
    from pyoomph import *
    from pyoomph.expressions import *
    from pyoomph.meshes.gmsh import GmshTemplate

    class Mesh(GmshTemplate):
        def define_geometry(self):
            self.default_resolution=0.3
            p00,p10=self.point(0,0),self.point(1,0)
            p11,p01=self.point(1,1),self.point(0,1)
            self.create_lines(p00,"bottom",p10,"right",p11,"top",p01,"left",p00)
            self.plane_surface("bottom","right","top","left",name="domain")

    class Poisson(Equations):
        def define_fields(self): self.define_scalar_field("u","C2")
        def define_residuals(self):
            u,v=var_and_test("u")
            self.add_weak(grad(u),grad(v)).add_weak(-1,v)

    class Prob(Problem):
        def define_problem(self):
            self+=Mesh()
            eqs=Poisson()
            for b in ("left","right","top","bottom"): eqs+=DirichletBC(u=0)@b
            self+=eqs@"domain"
"""

_SAVE = textwrap.dedent(_PROBLEM + """
    import os,shutil,sys
    from pyoomph.output.states import copy_state_file,get_state_file_mesh_files
    outdir=sys.argv[1]
    with Prob() as p:
        p.set_output_directory(outdir); p.quiet()
        p.solve()
        src=os.path.join(outdir,"_states","s.dump")
        os.makedirs(os.path.dirname(src),exist_ok=True)
        p.save_state(src)
        print("REFERENCES",[os.path.basename(f) for f in get_state_file_mesh_files(src)])
        print("NDOF",p.ndof())
        deep=os.path.join(outdir,"output","tag01")
        os.makedirs(deep,exist_ok=True)
        print("BROUGHT",[os.path.basename(f) for f in copy_state_file(src,os.path.join(deep,"state.dump"))])
        plain=os.path.join(outdir,"output","plain")
        os.makedirs(plain,exist_ok=True)
        shutil.copy2(src,os.path.join(plain,"state.dump"))
""")

_LOAD = textwrap.dedent(_PROBLEM + """
    import sys
    with Prob() as p:
        p.set_output_directory(sys.argv[1]); p.quiet()
        p.initialise()
        p.load_state(sys.argv[2])
        print("NDOF",p.ndof())
        print("LOAD OK")
""")


_needs_gmsh = pytest.mark.skipif(importlib.util.find_spec("gmsh") is None,
                                 reason="gmsh is not available")


def _run(script, *args, expect_success=True):
    r = subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True)
    if expect_success:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


@_needs_gmsh
def test_relocated_state_file_brings_its_mesh(tmp_path):
    out = str(tmp_path / "run")
    saved = _run(_SAVE, out)
    assert "REFERENCES ['Mesh.msh']" in saved.stdout, saved.stdout
    assert "BROUGHT ['Mesh.msh', 'Mesh.geo_unrolled']" in saved.stdout, saved.stdout
    ndof = [l for l in saved.stdout.splitlines() if l.startswith("NDOF")][0]

    tagdir = os.path.join(out, "output", "tag01")
    assert sorted(os.listdir(tagdir)) == ["Mesh.geo_unrolled", "Mesh.msh", "state.dump"]

    # The relocated copy loads, and gives back the problem it was written from.
    ok = _run(_LOAD, str(tmp_path / "load_ok"), os.path.join(tagdir, "state.dump"))
    assert "LOAD OK" in ok.stdout, ok.stdout
    assert ndof in ok.stdout, ok.stdout

    # ... and the plain copy, at the same new depth, does not: that is the bug being fixed, not an
    # incidental difference between the two.
    bad = _run(_LOAD, str(tmp_path / "load_bad"),
               os.path.join(out, "output", "plain", "state.dump"), expect_success=False)
    assert bad.returncode != 0
    assert "Mesh.msh" in bad.stderr and "not found" in bad.stderr, bad.stderr


@_needs_gmsh
def test_two_different_meshes_of_the_same_name_do_not_overwrite(tmp_path):
    """Two states from different runs, both calling their mesh Mesh.msh, exported side by side."""
    from pyoomph.output.states import copy_state_file, get_state_file_mesh_files
    out = str(tmp_path / "run")
    _run(_SAVE, out)
    src = os.path.join(out, "_states", "s.dump")
    msh = get_state_file_mesh_files(src)[0]

    # A second run's directory, with a mesh of the same name but different content.
    other = tmp_path / "other"
    (other / "_states").mkdir(parents=True)
    (other / "_gmsh").mkdir()
    shutil.copy2(src, other / "_states" / "s.dump")
    shutil.copy2(msh, other / "_gmsh" / "Mesh.msh")
    with open(other / "_gmsh" / "Mesh.msh", "a") as f:
        f.write("// a different mesh\n")

    dest = tmp_path / "export"
    dest.mkdir()
    copy_state_file(src, str(dest / "tag01.dump"))
    copy_state_file(str(other / "_states" / "s.dump"), str(dest / "tag02.dump"))
    names = sorted(f for f in os.listdir(dest) if f.endswith(".msh"))
    assert names == ["Mesh.msh", "Mesh_1.msh"], names
    assert get_state_file_mesh_files(str(dest / "tag01.dump")) == [str(dest / "Mesh.msh")]
    assert get_state_file_mesh_files(str(dest / "tag02.dump")) == [str(dest / "Mesh_1.msh")]
