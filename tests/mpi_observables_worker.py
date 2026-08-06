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

# Worker for tests/test_mpi_observables.py -- launched under mpirun.
# Two things that describe the WHOLE problem and used to be answered per rank: the extremum of an
# observable, and a text file of scalar rows.

import argparse
import sys
import traceback

from pyoomph import Problem, ElementSpace, Equations, var
from pyoomph.equations.generic import ProjectExpression, ExtremumObservables
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.expressions.units import meter
from pyoomph.generic.mpi import get_mpi_rank


class Observed(Problem):
    def define_problem(self):
        self += RectangularQuadMesh(N=8, size=[1, 1])
        # Deliberately functions of the coordinates rather than of a solved field: a projection
        # solve is not bit-reproducible between a serial run and an mpirun (different solver path),
        # and this test is about the reduction, not about the solver. The extrema are well inside the
        # domain, so which rank owns them depends on the partition.
        x, y = var("coordinate_x"), var("coordinate_y")
        eqs = ElementSpace("C2") + ProjectExpression(u=0)
        eqs += ExtremumObservables(u_dimless=(x - 0.3) ** 2 + (y - 0.7) ** 2,
                                   u_metric=((x - 0.3) ** 2 + (y - 0.7) ** 2) * meter)
        # The same on one boundary. At four ranks a rank can hold no element of it at all, which is
        # what the dimensional value has to survive: the unit cannot be read off a local element
        # there, so it comes from the registered expression instead.
        eqs += (Equations() + ExtremumObservables(edge=(x - 0.4) ** 2)) @ "top"
        self += eqs @ "domain"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    p = Observed()
    try:
        with p:
            p.set_output_directory(args.outdir)
            p.quiet()
            p.initialise()
            dom = p.get_mesh("domain")
            lo, lo_x = dom.evaluate_minimum("u_dimless", dimensional=False, return_x=True)
            hi, hi_x = dom.evaluate_maximum("u_dimless", dimensional=False, return_x=True)
            # Dimensional, i.e. the path that needs the unit; as_float only strips it afterwards.
            metric = dom.evaluate_maximum("u_metric", as_float=True)
            edge = p.get_mesh("domain/top").evaluate_minimum("edge", dimensional=False)

            # Every rank writes the same rows; only one file, with one header, may result.
            out = p.create_text_file_output("rows.txt", header=["i", "value"])
            for i in range(3):
                out.add_row(i, 10.0 * i)
            out.close()
        print("PYOOMPH_MPI_RESULT rank=%d lo=%.12g hi=%.12g lox=%.12g loy=%.12g hix=%.12g hiy=%.12g "
              "metric=%.12g edge=%.12g" % (get_mpi_rank(), lo, hi, lo_x[0], lo_x[1], hi_x[0], hi_x[1],
                                           metric, edge))
    except BaseException as e:  # noqa: BLE001
        print("PYOOMPH_MPI_RESULT rank=%d raised %s: %s" % (
            get_mpi_rank(), type(e).__name__, " | ".join(str(e).splitlines())))
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(3)


if __name__ == "__main__":
    main()
