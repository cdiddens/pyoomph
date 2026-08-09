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

# Worker for tests/test_mpi_orbit_output.py -- launched under mpirun.
#
# The Stuart-Landau oscillator in Cartesian form has the exact limit cycle x=cos(t), y=sin(t) with
# T=2*pi, so the orbit solve starts from an all-but-converged guess and the answer is known
# analytically. What is under test is not that number, though, but that orbit tracking plus
# PeriodicOrbit.output_orbit() survive an mpirun at all.

import argparse
import json
import os
import sys
import traceback

import numpy

from pyoomph import Problem, ODEEquations
from pyoomph.expressions import var, testfunction, partial_t
from pyoomph.output.generic import ODEFileOutput
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc, mpi_barrier


class StuartLandauSystem(ODEEquations):
    """r'=r(1-r^2), phi'=1 in Cartesian coordinates: a limit cycle at r=1 with period 2*pi."""

    def define_fields(self):
        self.define_ode_variable("x", "y")

    def define_residuals(self):
        x, y = var(["x", "y"])
        r2 = x ** 2 + y ** 2
        self.add_residual((partial_t(x) - (x - y - x * r2)) * testfunction(x))
        self.add_residual((partial_t(y) - (x + y - y * r2)) * testfunction(y))


class StuartLandauProblem(Problem):
    def define_problem(self):
        ode = StuartLandauSystem()
        # The output whose file handle only rank 0 ever owns; output_orbit() below moves the output
        # directory, which is what used to close it on every rank.
        ode += ODEFileOutput()
        self.add_equations(ode @ "osc")


def orbit_output_case(N=24, outdir=None):
    prob = StuartLandauProblem()
    with prob as p:
        if outdir is not None:
            p.set_output_directory(outdir)
        p.quiet()
        # Orbit tracking needs the symbolic Hessian; the C++ handler refuses to start without it.
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.initialise()
        # The exact cycle, sampled: dof order is (x,y), as define_ode_variable declares them.
        t = numpy.linspace(0.0, 2 * numpy.pi, N, endpoint=False)
        guess = numpy.column_stack([numpy.cos(t), numpy.sin(t)])
        p.set_current_dofs(guess[0])
        orbit = p.activate_periodic_orbit_handler(2 * numpy.pi, history_dofs=guess[1:],
                                                  mode="bspline", order=3, GL_order=3)
        p.solve()
        # THE regression: output_orbit() switches the output directory for the duration of the
        # write, and the ODE outputter used to close a file handle that exists on rank 0 alone.
        # Ranks > 0 raised AttributeError while rank 0 went on into the next collective, so the
        # run hung instead of failing (docs/.../orbit/manual_orbit.py).
        orbit.output_orbit("orbit_probe.txt")
        subdir = p.get_output_directory("orbit_probe.txt")
        # Only rank 0 writes these files, so counting their rows without a barrier first measures
        # how far rank 0 happens to have got (rank 1 saw 22 of 24 rows while writing this test).
        mpi_barrier()
        res = {
            "ndof": int(p.ndof()),
            "distributed": bool(p.is_distributed()),
            "T": float(orbit.get_T()),
            "orbit_dir_exists": bool(os.path.isdir(subdir)),
            # Written by rank 0 only, so this is also a check that the OTHER ranks did not write
            # into the same file behind its back.
            "orbit_rows": _count_rows(os.path.join(subdir, "osc.txt")),
        }
        # Continuing to write into the ORIGINAL file after the excursion is the other half of
        # change_output_directory(): the handle has to be reopened, not left closed.
        p.deactivate_bifurcation_tracking()
        p.output()
        mpi_barrier()
        res["base_rows_after"] = _count_rows(p.get_output_directory("osc.txt"))
        return res


def _count_rows(fname):
    if not os.path.isfile(fname):
        return -1
    with open(fname) as f:
        return sum(1 for line in f if line.strip() and not line.startswith("#"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=24)
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc()}
    try:
        payload.update(orbit_output_case(N=args.size, outdir=args.outdir))
    except Exception as e:
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
