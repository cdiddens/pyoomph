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
