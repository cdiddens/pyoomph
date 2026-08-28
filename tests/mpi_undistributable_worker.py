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

# Worker for tests/test_mpi_undistributable.py -- launched under `mpirun ... --distribute`.
#
# Three problems that get the same --distribute request: one that cannot be partitioned (a pure ODE,
# i.e. a single element), one that easily can, and one whose single element becomes four in the
# initial uniform refinement -- which happens before the distribution and therefore counts.

import argparse
import json
import traceback

from pyoomph import Problem, InitialCondition, DirichletBC, RefineToLevel, SpatialErrorEstimator
from pyoomph.equations.harmonic_oscillator import HarmonicOscillator
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.expressions import var
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class OscillatorOnly(Problem):
    """One ODE and nothing else: the global mesh has a single element, whatever nproc is."""

    def define_problem(self):
        eqs = HarmonicOscillator(omega=1, name="y") + InitialCondition(y=1 - var("time"))
        self.add_equations(eqs @ "harmonic_oscillator")


class PoissonAndOscillator(Problem):
    """A partitionable mesh, plus the same ODE alongside it."""

    def __init__(self, N=8):
        super().__init__()
        self.N = N

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N))
        eqs = PoissonEquation(name="u", source=1)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        self.add_equations(eqs @ "domain")
        self.add_equations((HarmonicOscillator(omega=1, name="y") +
                            InitialCondition(y=1 - var("time"))) @ "harmonic_oscillator")


class RefinedSingleElement(Problem):
    """One quad element that RefineToLevel turns into four, before anything is distributed.

    The element count that decides whether the problem can be partitioned therefore has to be read
    after the initial uniform refinement -- reading it a few lines earlier would refuse a problem
    oomph-lib can distribute perfectly well.
    """

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=1))
        eqs = PoissonEquation(name="u", source=1)
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        eqs += SpatialErrorEstimator(u=1)
        eqs += RefineToLevel(1)
        self.add_equations(eqs @ "domain")


def _run(p, outdir, adaptive=False, has_ode=True):
    with p:
        p.set_output_directory(outdir)
        p.quiet()
        if not adaptive:
            # Adaptivity is exercised by the other MPI tests; here it would only add a second reason
            # for the two runs to differ.
            p.max_refinement_level = 0
        # Fixed steps, so the trajectory does not depend on how the error estimate came out.
        p.run(1.0, startstep=0.05, outstep=False, temporal_error=None)
        res = {
            "distributed": bool(p.is_distributed()),
            "ndof": int(p.ndof()),
            "final_time": float(p.get_current_time(dimensional=False)),
        }
        if has_ode:
            res["y"] = float(p.get_ode("harmonic_oscillator").get_value("y", as_float=True))
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["ode", "pde", "refined"])
    ap.add_argument("--outdir", required=True)
    args, _ = ap.parse_known_args()

    payload = {"rank": get_mpi_rank(), "nproc": get_mpi_nproc(), "case": args.case}
    try:
        p = {"ode": OscillatorOnly, "pde": PoissonAndOscillator, "refined": RefinedSingleElement}[args.case]()
        payload.update(_run(p, args.outdir, adaptive=(args.case == "refined"),
                            has_ode=(args.case != "refined")))
    except Exception as e:  # noqa: BLE001
        payload["error"] = type(e).__name__ + ": " + str(e)
        payload["traceback"] = traceback.format_exc()[-2000:]
    print("PYOOMPH_MPI_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
