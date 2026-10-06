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

# Worker for test_mpi_augmentation_layout.py. Prints one PYOOMPH_MPI_RESULT line per rank. NOT a
# test module itself (the name deliberately does not start with "test_").
#
# The DISTRIBUTED branch of the dof layout a Python augmentation now installs, i.e. the half of
# DofAugmentations' move onto AugmentedDofDistributionHelper that serial tests cannot reach. Under
# --distribute the layout is supposed to be: rank d owns its base rows, then its rows of each vector
# block, with each scalar contributing one row on rank 0 alone, and the whole thing tiling
# [0, n_aug) in rank order.
#
# This goes through the LOW-LEVEL bindings -- _create_dof_augmentation / _add_augmented_dofs /
# _reset_augmented_dof_vector_to_nonaugmented -- and not through Problem.set_custom_assembler, which
# still refuses nproc>1 for the rest of that pipeline. No custom assembler is installed and nothing
# is assembled while the augmentation is up, so none of B1's unfinished business is in the way: the
# only thing under test is the dof bookkeeping.
#
# What is reported per rank:
#   - the base and augmented row layouts, so the test can check they tile and that the base one is
#     still visible (Problem::augmented_dof_distribution_helper()).
#   - "aug_local_total": how many augmented dofs this rank actually owns. Scalars land on rank 0, so
#     the ranks must NOT all report the same number -- that is what distinguishes a genuinely
#     distributed layout from the replicated one.
#   - "split_lengths": what split() hands back, which must be this rank's OWNED rows of each block.
#   - "block_values": the block contents, to check the registered guess was scattered to the right
#     rows rather than copied wholesale onto every rank.

import argparse
import json
import os
import sys
import traceback

import numpy

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.generic.mpi import get_mpi_rank, get_mpi_nproc


class Poisson(Equations):
    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) + weak(3 * exp(u), v))


class AugProblem(Problem):
    def __init__(self, N=6):
        super().__init__()
        self.N = N

    def define_problem(self):
        self += RectangularQuadMesh(N=self.N, size=[1, 1], name="domain")
        eqs = Poisson()
        for b in ["left", "right", "top", "bottom"]:
            eqs += DirichletBC(u=0) @ b
        eqs += IntegralObservables(usqr=var("u") ** 2)
        self += eqs @ "domain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--distribute", action="store_true")
    args, _ = ap.parse_known_args()

    payload: dict = {"rank": get_mpi_rank(), "nproc": max(get_mpi_nproc(), 1)}
    try:
        with AugProblem(args.N) as p:
            p.set_output_directory(args.outdir)
            p.initialise()
            p.solve()
            payload["usqr"] = float(p.get_mesh("domain").evaluate_all_observables()["usqr"])

            base_n, base_loc, base_first, base_dist = p._get_dof_distribution_info()
            payload["base"] = [int(base_n), int(base_loc), int(base_first), bool(base_dist)]

            # A fold-shaped layout: one base-sized vector block plus one scalar. The guess is
            # position-dependent so that a block filled from the wrong rows is visible in its values.
            guess = numpy.arange(base_n, dtype=numpy.float64)
            aug = p._create_dof_augmentation()
            aug.add_vector(guess)
            aug.add_scalar(7.25)
            p._add_augmented_dofs(aug)

            a_n, a_loc, a_first, a_dist = p._get_dof_distribution_info()
            payload["aug"] = [int(a_n), int(a_loc), int(a_first), bool(a_dist)]
            b_n, b_loc, b_first, b_dist = p._get_base_dof_distribution_info()
            payload["aug_base"] = [int(b_n), int(b_loc), int(b_first), bool(b_dist)]
            payload["aug_local_total"] = int(a_loc)
            payload["ndof"] = int(p.ndof())

            # The naive->augmented translation table. Written to a file rather than into the result
            # line: it is one entry per augmented dof and a long line does not arrive atomically on
            # the shared stdout pipe.
            table = p._get_augmented_eqn_table()
            numpy.savez(os.path.join(args.outdir, "eqntable_rank%d.npz" % get_mpi_rank()),
                        table=numpy.asarray(table, dtype=numpy.int64),
                        base_first=numpy.array([b_first]), base_nloc=numpy.array([b_loc]),
                        base_n=numpy.array([b_n]), aug_n=numpy.array([a_n]),
                        aug_first=numpy.array([a_first]), aug_nloc=numpy.array([a_loc]))
            payload["table_len"] = int(len(table))

            blocks = aug.split(startindex=1)
            payload["split_lengths"] = [len(b) for b in blocks]
            payload["block_values"] = [float(v) for v in blocks[0]]
            payload["scalar_value"] = float(blocks[1][0])

            p._reset_augmented_dof_vector_to_nonaugmented()
            r_n, r_loc, r_first, r_dist = p._get_dof_distribution_info()
            payload["restored"] = [int(r_n), int(r_loc), int(r_first), bool(r_dist)]

            # The base state must be untouched by all of that.
            payload["usqr_after"] = float(p.get_mesh("domain").evaluate_all_observables()["usqr"])
            p.solve()
            payload["usqr_resolved"] = float(p.get_mesh("domain").evaluate_all_observables()["usqr"])
        payload["ok"] = True
    except Exception as e:
        payload["ok"] = False
        payload["error"] = repr(e)
        payload["traceback"] = traceback.format_exc()[-2500:]
    # sys.__stdout__, not print(): pyoomph's MPI console mutes stdout on every rank but 0 by default,
    # which would make the per-rank assertions pass by being vacuous.
    out = sys.__stdout__ or sys.stdout
    out.write("PYOOMPH_MPI_RESULT " + json.dumps(payload) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
