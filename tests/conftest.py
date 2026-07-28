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


def _full_requested(config):
    return config.getoption("--full") or os.environ.get(_FULL_ENV, "") not in ("", "0", "false", "False")


def pytest_collection_modifyitems(config, items):
    if _full_requested(config):
        return
    skip = pytest.mark.skip(reason="slow: run with --full (or PYOOMPH_FULL_TESTS=1), e.g. before merging")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
