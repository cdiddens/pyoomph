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

# Worker for test_mpi_tracers.py.
#
# Poiseuille flow given as an analytic expression, so the answer is the exact straight line
# x(t) = x0 + (1 - y0^2) t to round-off, whatever the partitioning is. That matters: a migration bug
# that loses or duplicates a particle, or that restarts it in the wrong element, cannot hide behind a
# tolerance here.
#
# The particles are seeded on a horizontal line spanning the whole domain and then travel most of its
# length, so on any sensible partition every one of them crosses at least one partition boundary.

import argparse
import json
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.equations.tracers import (TracerParticles, TracerSeedPoints, TracerSeedGrid,
                                       TracerPeriodicBoundaryCondition)
from pyoomph.meshes.simplemeshes import RectangularQuadMesh
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


LX, LY, NX, NY = 4.0, 2.0, 16, 8


def _seed_positions(corner_only):
    if corner_only:
        # Everything in one corner, so at least one process starts (and stays) with no particles at
        # all. A collective that only the processes holding particles enter deadlocks here.
        return [[0.2 + 0.05 * i, -0.9 + 0.05 * j] for i in range(4) for j in range(4)]
    return [[0.3, y] for y in (-0.75, -0.4, 0.0, 0.35, 0.7)]


class TracerProblem(Problem):
    def __init__(self, corner_only=False, payloads=True, periodic=False):
        super().__init__()
        self.corner_only = corner_only
        self.use_payloads = payloads
        self.periodic = periodic

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(size=[LX, LY], lower_left=[0, -1], N=[NX, NY]))
        eqs = PoissonEquation(source=0) + DirichletBC(u=0) @ "left" + DirichletBC(u=1) @ "right"
        eqs += TracerParticles(vector(1 - var("coordinate_y") ** 2, 0),
                               seed=TracerSeedPoints(_seed_positions(self.corner_only)),
                               payloads={"residence": 1} if self.use_payloads else None,
                               history_time=0.5,
                               rtol=1e-11, atol=1e-13)
        if self.periodic:
            # The hard case for a distributed run: the periodic image of a point at the outlet is
            # at the inlet, which is nowhere near the process the particle leaves from - not even a
            # halo of it, so no migration exchange can reach it.
            eqs += TracerPeriodicBoundaryCondition(vector(-LX, 0)) @ "right"
        self += eqs @ "domain"

    def tracers(self):
        return self.get_mesh("domain").get_tracers()


def run_case(mode, outdir, statefile=None, nsteps=55, dt=0.05):
    if mode == "periodic":
        # Long enough that the fastest particles wrap, and then some.
        nsteps = max(nsteps, 140)
    p = TracerProblem(corner_only=(mode == "corner"), periodic=(mode == "periodic"))
    p.set_output_directory(outdir)
    p.quiet()
    if get_mpi_nproc() > 1:
        p.distribute()
    p.initialise()

    def local_histories(collection):
        # Local, not gathered: there is no collective for the history, so the ranks' dicts are
        # merged by the test. Keyed by identity, which is what makes the merge partition-independent.
        out = {}
        for tid in collection.get_ids():
            h = collection.get_history(int(tid))
            out[str(int(tid))] = [float(v) for v in numpy.asarray(h).ravel()]
        return out

    hist_at_state = {}
    tr = p.tracers()
    if mode == "load":
        assert statefile is not None
        p.load_state(statefile)
        tr = p.tracers()
        hist_at_state = local_histories(tr)
        # A restarted run gets fewer steps: the saved particles are already most of the way down the
        # channel, and running the full length again would simply push them out of the outlet - which
        # is correct behaviour, but would compare two different particle sets.
        nsteps = min(nsteps, 8)

    # Captured AFTER any load, so the analytic check below is against where the particles actually
    # started this run from.
    start = tr.gather_positions().copy()
    start_ids = list(tr.gather_ids())

    t0 = float(p.get_current_time(as_float=True, dimensional=False))
    # Track the local count over the run: it changing at all is the proof that particles really did
    # cross a partition boundary, without which the analytic assertion below would pass for a
    # completely serial implementation that simply never migrated anything.
    nlocal_seen = {int(tr.nlocal())}
    # The two disjoint halves of a periodic wrap: the image was in this process's own mesh, or it
    # was not and another process took the particle over. The second is what says the collective
    # reinjection round was actually needed, which the positions alone cannot distinguish.
    nwrapped, nreinjected = 0, 0
    if mode == "plot":
        # Half the channel: the particles are then spread over the partitions rather than bunched up
        # at the outlet, so the drawing rank really does hold only some of them and the gather has
        # something to do. The test asserts that (nlocal_at_plot < nglobal on some rank).
        nsteps = nsteps // 2
    for _ in range(nsteps):
        p.solve(timestep=dt)
        nlocal_seen.add(int(tr.nlocal()))
        nwrapped += int(tr.get_wraps_last_step())
        nreinjected += int(tr.get_reinjections_last_step())
    elapsed = float(p.get_current_time(as_float=True, dimensional=False)) - t0

    if mode == "save":
        p.save_state(statefile)
        hist_at_state = local_histories(tr)

    plotted = None
    if mode == "plot":
        # What a tracer plot would draw. MatplotLibTracers reads the collection of the drawing rank,
        # which is a fraction of the cloud on a distributed mesh, so it now gathers instead - and the
        # gather has to be reached by every rank, i.e. it goes through the plot's request scope.
        from pyoomph.output.plotting import MatplotlibPlotter

        drawn = {}

        class _RecordingPlotter(MatplotlibPlotter):
            def define_plot(self):
                part = self.add_plot("domain/tracers")
                part.trail = True
                part.invisible = True  # record what it holds; drawing it needs a figure we do not want
                col = part.mesh.get_tracers(part.tracer_name)
                from pyoomph.meshes.meshdatamerge import needs_merging, gather_global_tracers
                if needs_merging(part.mesh):
                    col = gather_global_tracers(part.mesh, part.tracer_name, with_history=True)
                pos = numpy.asarray(col.get_positions(), dtype=float)
                drawn["n"] = int(len(pos))
                drawn["x_sum"] = float(numpy.sum(pos[:, 0])) if len(pos) else 0.0
                drawn["y_sum"] = float(numpy.sum(pos[:, 1])) if len(pos) else 0.0
                drawn["ids"] = sorted(int(i) for i in col.get_ids())
                drawn["trail_samples"] = int(sum(len(numpy.asarray(col.get_history(int(i))))
                                                 for i in col.get_ids()
                                                 if col.get_history(int(i)) is not None))

        plotter = _RecordingPlotter()
        plotter._problem = p
        plotter._named_problems[""] = p
        # What this rank holds by itself, i.e. what the plot used to draw
        local_at_plot = int(tr.nlocal())
        plotter.plot()
        plotted = drawn if get_mpi_rank() == 0 else {}
        plotted["nlocal_at_plot"] = local_at_plot
        plotted["mesh_distributed"] = bool(p.get_mesh("domain").is_mesh_distributed())

    end = tr.gather_positions()
    ids = list(tr.gather_ids())
    pay = tr.gather_payloads()

    # Exact answer, from the STARTING positions the same gather reports, so this does not depend on
    # the seeding order either.
    err = 0.0
    if len(end) == len(start) and len(end):
        expect_x = start[:, 0] + (1.0 - start[:, 1] ** 2) * elapsed
        if mode == "periodic":
            expect_x = numpy.mod(expect_x, LX)
        err = float(max(numpy.max(numpy.abs(end[:, 0] - expect_x)),
                        numpy.max(numpy.abs(end[:, 1] - start[:, 1]))))


    return {
        "rank": get_mpi_rank(),
        "nproc": get_mpi_nproc(),
        "nlocal": int(tr.nlocal()),
        "nlocal_changed": len(nlocal_seen) > 1,
        "nglobal": int(tr.nglobal()),
        "nstart": len(start_ids),
        "ids": [int(i) for i in ids],
        "positions": [[float(v) for v in row] for row in end],
        "payloads": [float(row[0]) for row in pay] if len(pay) else [],
        "elapsed": elapsed,
        "analytic_error": err,
        "history_at_state": hist_at_state,
        "nwrapped": nwrapped,
        "nreinjected": nreinjected,
        "plotted": plotted,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "corner", "save", "load", "periodic", "plot"])
    ap.add_argument("--outdir", default="mpi_tracer_out")
    ap.add_argument("--statefile", default=None)
    ap.add_argument("--nsteps", type=int, default=55)
    args, _unknown = ap.parse_known_args()
    try:
        res = run_case(args.mode, args.outdir, statefile=args.statefile, nsteps=args.nsteps)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        return 1
    print("PYOOMPH_MPI_RESULT " + json.dumps(res))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
