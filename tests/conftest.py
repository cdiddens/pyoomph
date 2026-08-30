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

# Splits the suite into a fast default run and a full run.
#
#   python -m pytest *.py              # fast: skips the tests marked "slow"
#   python -m pytest *.py --full       # everything, including the adaptive-mesh campaign and MPI
#   PYOOMPH_FULL_TESTS=1 python -m pytest *.py    # same, for CI where passing a flag is awkward
#
# The "slow" tests are the ones that sweep a large matrix or launch mpirun: the 3D adaptive-mesh campaign
# and both MPI modules. They are the right thing to run before merging a branch and the wrong thing to run
# after every edit. Everything else -- including the 2D campaign, which is only ~30 s -- stays in the fast
# run, so a quick check still covers the 2D physics end to end.
#
# Nothing is ever deleted or permanently excluded: --full runs the entire suite.
#
# There is a SECOND, independent axis: "campaign". It marks the mixed-adaptive-mesh validation campaign
# (the four test_adaptive_*_campaign / test_mpi_adaptivity* modules), which sweeps a large matrix of
# discretisations to prove the refinement engine correct. The wheel-building workflow deselects it with
#
#   pytest tests -m "not campaign"     # see test-command in pyproject.toml
#
# because cibuildwheel repeats the test command once per Python version per platform, so a suite that takes
# minutes locally costs a multiple of that there. What CI keeps is the set of tests that predates the
# campaign -- 177 tests, ~4 minutes -- which is what a wheel needs to certify: that the built extension
# imports, compiles elements and solves. Proving the refinement engine correct is the job of a branch merge,
# not of every wheel.
#
# The two axes are orthogonal on purpose. "slow" is about what YOU wait for while working; "campaign" is
# about what CI pays for per wheel. A default local run still executes the 2D campaign, and --full still
# runs absolutely everything.

import os

import pytest

_FULL_ENV = "PYOOMPH_FULL_TESTS"


def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", default=False,
                     help="also run the tests marked 'slow' (the 3D adaptive-mesh campaign and the MPI "
                          "modules). Use before merging a branch.")


def pytest_configure(config):
    # Every MPI module launches its worker under mpirun and reads results back as
    # "PYOOMPH_MPI_RESULT <json>" lines that EVERY rank prints, then asserts it got one per rank.
    # pyoomph's default MPI console mode is "condensed", which lets only rank 0's stdout reach the
    # terminal, so the harness saw a single line and every one of the 160 MPI tests failed with
    # "case ... reported from 1 of 2 ranks". "off" (no filtering at all) is the mode that fits: "all"
    # does emit every rank, but tags each line "[rank N] ", which no longer starts with the marker
    # the harness matches on.
    # Set here rather than in the 16 launchers: they all copy os.environ into the child's env, so one
    # assignment covers them and any module added later. Forced rather than setdefault() -- a
    # developer with PYOOMPH_MPI_OUTPUT exported in their shell would otherwise silently break the
    # whole MPI suite, and no test wants to vary this.
    os.environ["PYOOMPH_MPI_OUTPUT"] = "off"
    config.addinivalue_line(
        "markers",
        "slow: large sweep or mpirun-based; skipped unless --full (or PYOOMPH_FULL_TESTS=1) is given")
    config.addinivalue_line(
        "markers",
        "campaign: part of the mixed-adaptive-mesh validation campaign; deselected in the wheel builds "
        "with -m 'not campaign' (see pyproject.toml), never skipped by default locally")


def _full_requested(config):
    return config.getoption("--full") or os.environ.get(_FULL_ENV, "") not in ("", "0", "false", "False")


def pytest_collection_modifyitems(config, items):
    if _full_requested(config):
        return
    skip = pytest.mark.skip(reason="slow: run with --full (or PYOOMPH_FULL_TESTS=1), e.g. before merging")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _output_below_tmp_path(tmp_path, monkeypatch):
    """Runs every test in its own temporary directory, so that no test writes into the repository.

    A Problem that is never given an output directory falls back to the basename of __main__.__file__
    (pyoomph/generic/problem.py), which under "python -m pytest" is pytest's own entry module: the 26
    modules that set no output directory all wrote into one "__main__" directory beside whatever the
    CWD happened to be. Modules that DO name a directory mostly name a relative one, which lands in
    the same place. None of it shows up in "git status", because pyoomph drops a .gitignore into each
    output directory it creates -- so this accumulated in the repository root unnoticed until someone
    looked with ls.

    Redirecting the CWD rather than passing tmp_path into every Problem keeps the descriptive
    directory names each module chose (they are what makes a failed run's output findable under
    pytest's basetemp) and needs no cooperation from modules added later. It is safe because nothing
    under tests/ resolves a relative path itself: the mesh templates generate their .geo in code
    rather than loading a file, the state-file dumps are all absolute, and every subprocess launch
    passes an explicit cwd= (the MPI launchers use the directory of their worker script). It also
    costs no compile time, because the JIT cache is keyed on the generated code text and lives under
    ~/.cache, not in the output directory (pyoomph/generic/jit_cache.py).
    """
    monkeypatch.chdir(tmp_path)


def has_complex_target_eigensolver():
    """Is an eigensolver that can TARGET a complex eigenvalue available in this installation?

    Used by tests that need a Hopf pair or an azimuthal/normal-mode eigenfunction. The ARPACK-based
    backends ("scipy", "pardiso", "accelerate") cannot do it at all - they raise on any target - so
    such a test used to be skipped without slepc4py. The built-in "spectra" backend answers the same
    question without PETSc, which is what makes these tests run on a plain wheel, so either will do.
    """
    from pyoomph import _pyoomph_core
    if getattr(_pyoomph_core, "has_spectra", False):
        return True
    try:
        import slepc4py  # type:ignore  # noqa: F401
        return True
    except Exception:
        return False
