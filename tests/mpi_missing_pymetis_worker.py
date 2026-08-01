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

# Worker for tests/test_mpi_missing_pymetis.py -- launched under `mpirun ... --distribute`.
# PyMetis is hidden from the import machinery (rather than actually uninstalled) so that the test
# runs on a machine that has it.

import argparse
import importlib.abc
import sys
import traceback


class _BlockPymetis(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pymetis":
            raise ImportError("pymetis hidden by the test")
        return None


# Everything that runs lives in main(), like the other mpi_*_worker.py files. The suite is
# invoked as `pytest *.py`, so pytest imports every file in this directory during collection --
# and argparse at module scope then exits with "the following arguments are required: --outdir",
# which surfaces as an INTERNALERROR that collects zero tests and aborts the entire run.
#
# pyoomph is imported inside main() rather than at the top of the file, because the import has to
# happen after _BlockPymetis is on sys.meta_path: hiding pymetis from pyoomph is the whole point
# of this worker, and an import at module scope would have already found it.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--skip-preflight", action="store_true",
                        help="bypass Problem.distribute()'s collective check, so the failure happens "
                             "inside the rank-0-only METIS callback")
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    sys.modules.pop("pymetis", None)
    sys.meta_path.insert(0, _BlockPymetis())

    from pyoomph import Problem, DirichletBC
    from pyoomph.equations.poisson import PoissonEquation
    from pyoomph.meshes.simplemeshes import RectangularQuadMesh
    from pyoomph.generic.mpi import get_mpi_rank

    class Pois(Problem):
        def define_problem(self):
            self.add_mesh(RectangularQuadMesh(N=8))
            eqs = PoissonEquation(source=1)
            for b in ["left", "right", "top", "bottom"]:
                eqs += DirichletBC(u=0) @ b
            self.add_equations(eqs @ "domain")

    p = Pois()
    if args.skip_preflight:
        p.distribute = lambda: super(Problem, p).distribute()
    try:
        with p:
            p.set_output_directory(args.outdir)
            p.quiet()
            p.initialise()
        print("PYOOMPH_MPI_RESULT rank=%d completed" % get_mpi_rank())
    except BaseException as e:  # noqa: BLE001
        print("PYOOMPH_MPI_RESULT rank=%d raised %s: %s" % (get_mpi_rank(), type(e).__name__, e))
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(3)


if __name__ == "__main__":
    main()
