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

# What a state file says about itself, and what happens when it says something unexpected.
#
# A state file starts with a magic string, a format version and the sharding of its mesh data, and
# ends with a footer. All three are checked on load, and they have to be, because none of the failures
# they catch is loud on its own: a file from another program desynchronizes on the first array and
# dies somewhere deep inside numpy, a newer file reads plausible-looking garbage, and one part of a
# (future) sharded set would quietly restore a fraction of the mesh.

import os

import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.output.states import DumpFile


class PoissonEqs(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) - weak(1, v))


class SmallProblem(Problem):
    def define_problem(self):
        self += RectangularQuadMesh(N=3, size=[1, 1])
        self += (PoissonEqs() + DirichletBC(u=0) @ "left") @ "domain"
        self.write_states = False


def _write_state(tmp_path, name="state.dump"):
    fname = str(tmp_path / name)
    with SmallProblem() as problem:
        problem.set_output_directory(str(tmp_path / "out"))
        problem.solve()
        problem.save_state(fname)
    return fname


def _read_header(fname):
    """The header as it sits in the file, without loading the problem."""
    dump = DumpFile(fname, False)
    try:
        return [dump.read_string_data(), dump.read_string_data(), dump.read_string_data()]
    finally:
        dump.close()


def test_header_identifies_the_file(tmp_path):
    magic, version, sharding = _read_header(_write_state(tmp_path))
    assert magic == "pyoomph_dump"
    assert version == SmallProblem()._dump_version
    assert sharding == "global", "a single file holding the whole problem must say so"


def test_footer_is_present(tmp_path):
    dump = DumpFile(_write_state(tmp_path), False)
    try:
        assert dump.check_footer("EOF_pyoomph")
    finally:
        dump.close()


def test_a_foreign_file_is_rejected(tmp_path):
    # Not a state file at all. The footer check must decide that from the last bytes rather than
    # reading whatever length those bytes happen to encode - which used to end in a MemoryError.
    foreign = str(tmp_path / "foreign.dump")
    with open(foreign, "wb") as f:
        f.write(os.urandom(4096))
    with SmallProblem() as problem:
        problem.set_output_directory(str(tmp_path / "out2"))
        problem.solve()
        with pytest.raises(RuntimeError, match="Unsupported state file"):
            problem.load_state(foreign)


def test_a_truncated_file_is_rejected(tmp_path):
    fname = _write_state(tmp_path)
    with open(fname, "rb") as f:
        data = f.read()
    truncated = str(tmp_path / "truncated.dump")
    with open(truncated, "wb") as f:
        f.write(data[: len(data) // 2])
    with SmallProblem() as problem:
        problem.set_output_directory(str(tmp_path / "out3"))
        problem.solve()
        with pytest.raises(RuntimeError, match="Unsupported state file"):
            problem.load_state(truncated)


def test_a_newer_file_is_rejected(tmp_path):
    fname = _write_state(tmp_path)
    with SmallProblem() as problem:
        problem.set_output_directory(str(tmp_path / "out4"))
        problem.solve()
        problem._dump_version = "0.0.1"  # pretend this build is older than the file
        with pytest.raises(Exception):
            problem.load_state(fname)


def test_a_sharded_file_is_refused_with_a_clear_message(tmp_path):
    # Nothing writes sharded files yet; the flag exists so that the day one appears, a reader that
    # cannot handle it says so instead of restoring a fraction of the mesh.
    fname = str(tmp_path / "sharded.dump")
    with SmallProblem() as problem:
        problem.set_output_directory(str(tmp_path / "out5"))
        problem.solve()
        original = DumpFile.__init__

        def patched(self, name, save, compression_level=None):
            original(self, name, save, compression_level=compression_level)
            self.sharding = "sharded"

        DumpFile.__init__ = patched
        try:
            problem.save_state(fname)
        finally:
            DumpFile.__init__ = original
        assert _read_header(fname)[2] == "sharded"
        with pytest.raises(RuntimeError, match="sharded"):
            problem.load_state(fname)


def test_version_comparison_is_componentwise(tmp_path):
    # "0.10.0" < "0.2.0" as strings, which would pick the wrong format branch the first time a
    # component reaches double digits
    dump = DumpFile(str(tmp_path / "scratch.dump"), True)
    try:
        dump.version = "0.10.0"
        assert dump.version_at_least(0, 2, 0)
        assert dump.version_at_least(0, 10)
        assert not dump.version_at_least(0, 11)
        dump.version = "0.1.0"
        assert dump.version_at_least(0, 1)
        assert not dump.version_at_least(0, 1, 1)
    finally:
        dump.close()
