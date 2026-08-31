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

# Two Problems ALIVE AT THE SAME TIME, each with its own equations.
#
# test_multiple_problems.py covers Problems used one after another; this covers the overlapping case,
# where the second is constructed and solved while the first is still holding its compiled code.
#
# What used to go wrong: every Problem defaults to _outdir = basename(__main__.__file__) with
# _ccode_dir = "_ccode", and the generated file is named after the domain, so two Problems with a
# domain called "domain" both compile to <outdir>/_ccode/domain.so. dlopen dedupes by inode and only
# bumps a refcount, so the second Problem was handed the FIRST one's already-mapped image -- while
# its own linker had just overwritten that very file underneath the live mapping. The result was a
# second Problem silently computing the first one's equations (source=2 giving the source=1 answer),
# and, where the two libraries differed enough in layout, a SIGSEGV inside dlsym.
#
# Three things now stand between that and the user, and there is a test for each below:
#   1. Problem._claim_unique_ccode_dir moves the second Problem's code aside on collision.
#   2. Problem::load_jit_code refuses a library some other live Problem already loaded, by name.
#   3. The JIT cache and the linker install their .so via os.replace, so a live mapping is never
#      rewritten in place.
#
# Run in subprocesses: one failure mode is a SIGSEGV, which would take the pytest process with it.

import subprocess
import sys
import textwrap

import pytest

_PREAMBLE = textwrap.dedent("""
    from pyoomph.generic.ccompiler import BaseCCompiler
    print("CHILD COMPILERS", BaseCCompiler.available_compilers(), flush=True)
    from pyoomph import Problem, DirichletBC
    from pyoomph.equations.poisson import PoissonEquation
    from pyoomph.meshes.simplemeshes import RectangularQuadMesh

    class Poisson(Problem):
        def __init__(self, source):
            super().__init__()
            self._src = source
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=6))
            eqs = PoissonEquation(source=self._src)
            for b in ["left", "right", "top", "bottom"]:
                eqs += DirichletBC(u=0) @ b
            self.add_equations(eqs @ "domain")

    def peak(p):
        return max(abs(x) for x in p.get_current_dofs()[0])

    def build(source, outdir):
        p = Poisson(source)
        p.set_output_directory(outdir)
        p.quiet()
        p.initialise()
        p.solve()
        return p
""")


def _compilers_here():
    """What this process would JIT with, for the failure message.

    The child gets its own answer, and the two need not agree: on the Windows wheel job of 30th
    August 2026 these cases died in the CHILD with "Cannot open include file: math.h", i.e. it had
    picked the MSVC toolchain (SystemCCompiler, quality 5, which outranks tccbox's 4) while the
    pytest process compiling the rest of the suite had not - and nothing in the log said what either
    of them chose. Reported rather than asserted on: a machine is entitled to any of them.
    """
    try:
        from pyoomph.generic.ccompiler import BaseCCompiler
        return repr(BaseCCompiler.available_compilers())
    except Exception as e:
        return "unavailable (%s)" % e


def _run(tmp_path, body, timeout=900):
    script = tmp_path / "case.py"
    script.write_text(_PREAMBLE + textwrap.dedent(body))
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, (
        "exited %d (a negative value is the killing signal -- -11 is SIGSEGV)\n"
        "compilers available to pytest: %s\n"
        "--- stdout ---\n%s\n--- stderr tail ---\n%s"
        % (proc.returncode, _compilers_here(), proc.stdout[-3000:], proc.stderr[-4000:]))
    return proc.stdout


@pytest.mark.slow
def test_two_live_problems_keep_their_own_equations(tmp_path):
    """Separate output directories: each Problem solves the equations it was given.

    Poisson with a constant source is linear in that source, so doubling it doubles the solution
    exactly -- which makes "B got A's compiled code" a numerically unmistakable failure rather than a
    matter of tolerance. Re-solving A afterwards catches the opposite mix-up.
    """
    out = _run(tmp_path, """
        a = build(1, "out_a")
        ref = peak(a)
        b = build(2, "out_b")          # second Problem, first still alive
        got_b = peak(b)
        a.solve()                      # A again, with B still alive
        got_a = peak(a)
        print("REF %.15g" % ref, flush=True)
        print("B %.15g" % got_b, flush=True)
        print("A %.15g" % got_a, flush=True)
        print("DONE", flush=True)
    """)
    vals = {l.split()[0]: float(l.split()[1]) for l in out.splitlines()
            if l.startswith(("REF ", "B ", "A "))}
    assert "DONE" in out, out
    assert vals["REF"] > 0
    assert abs(vals["A"] - vals["REF"]) < 1e-12 * max(1.0, vals["REF"]), \
        "Problem A changed once Problem B existed: %r" % vals
    assert abs(vals["B"] - 2 * vals["REF"]) < 1e-10 * max(1.0, vals["REF"]), \
        "Problem B did not get its own source term: %r" % vals


@pytest.mark.slow
def test_colliding_code_paths_are_moved_aside(tmp_path):
    """Same output directory AND same domain name: the second Problem's code must move aside.

    This is the configuration two Problems fall into by default, since _outdir defaults to the script
    name for every one of them.
    """
    out = _run(tmp_path, """
        a = build(1, "shared")
        ref = peak(a)
        b = build(2, "shared")         # identical path for both
        got_b = peak(b)
        a.solve()
        print("REF %.15g" % ref, flush=True)
        print("B %.15g" % got_b, flush=True)
        print("A %.15g" % peak(a), flush=True)
        print("DONE", flush=True)
    """)
    vals = {l.split()[0]: float(l.split()[1]) for l in out.splitlines()
            if l.startswith(("REF ", "B ", "A "))}
    assert "DONE" in out, out
    assert abs(vals["B"] - 2 * vals["REF"]) < 1e-10 * max(1.0, vals["REF"]), \
        "the second Problem ran the first one's compiled equations: %r" % vals
    assert abs(vals["A"] - vals["REF"]) < 1e-12 * max(1.0, vals["REF"]), vals
    assert (tmp_path / "shared" / "_ccode").is_dir()
    assert (tmp_path / "shared" / "_ccode_1").is_dir(), \
        "the colliding Problem did not get its own code directory"


@pytest.mark.slow
def test_pinned_colliding_code_dir_is_refused(tmp_path):
    """With the code directory pinned by the caller, the collision must be an error, not a guess.

    _claim_unique_ccode_dir deliberately stands aside once the directory was chosen explicitly (as
    redefine_problem(code_dir=...) does), so this exercises the backstop in Problem::load_jit_code.
    """
    out = _run(tmp_path, """
        a = build(1, "shared")
        b = Poisson(2)
        b.set_output_directory("shared")
        b._ccode_dir_is_unique = True      # as if the caller had pinned it
        b.quiet()
        try:
            b.initialise()
            print("NO ERROR", flush=True)
        except RuntimeError as e:
            print("RAISED", "same shared library" in str(e), "different Problems" in str(e), flush=True)
        print("DONE", flush=True)
    """)
    assert "DONE" in out, out
    assert "RAISED True True" in out, \
        "the pinned collision was not refused with an actionable message: %r" % out


def test_the_default_compiler_falls_back_when_the_system_one_cannot_compile(monkeypatch):
    """A broken system toolchain must cost speed, not the run.

    get_default_c_compiler() used to return a hardcoded "system", which on Windows means MSVC
    whatever else is installed. The Windows job of the full-suite run of 30th August 2026 got an
    MSVC whose vcvars returned an INCLUDE without the Windows SDK - every JIT compile died on
    "Cannot open include file: 'math.h'" while a working tccbox sat there unconsulted. Not
    reproducible on the next runner, which is exactly why the fallback is worth having rather than
    the image being blamed.

    Faked here rather than waited for: check_avail() is what a missing SDK makes false.
    """
    import warnings

    import pyoomph
    from pyoomph.generic.ccompiler import BaseCCompiler, SystemCCompiler

    monkeypatch.setattr(SystemCCompiler, "check_avail", staticmethod(lambda: False))
    monkeypatch.setattr(pyoomph, "_resolved_default_c_compiler", None)
    assert "tccbox" in BaseCCompiler.available_compilers(), \
        "the fallback this test is about needs tccbox installed (pyoomph depends on it)"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = pyoomph.get_default_c_compiler()
    assert got == "tccbox", got
    assert any("cannot compile on this machine" in str(w.message) for w in caught), \
        [str(w.message) for w in caught]

    # ... and the answer is memoised, since check_avail() compiles and links a program.
    monkeypatch.setattr(SystemCCompiler, "check_avail", staticmethod(lambda: True))
    assert pyoomph.get_default_c_compiler() == "tccbox"
