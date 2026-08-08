# A parallel path for the custom assembler

Status: **plan only, nothing implemented.** File and line references are to `develop` at `7813cad`.

The custom assembler is the Python-side augmented-system machinery: `CustomAssemblyBase`
(`pyoomph/generic/assembly.py:38`) and the `AugmentedAssemblyHandler` family in
`pyoomph/generic/bifurcation_tools.py:543` - the fold/pitchfork/Hopf/normal-mode/eigenbranch trackers
and the deflation handler. It works only in serial. Under `mpirun` it throws from deep inside C++,
and behind that throw sit three more layers that would each fail in turn.

This document is the plan for making it run under MPI, in stages, and says what each stage buys.

## 0. The starting state, exactly

`docs/source/tutorial/temporal/stability/deflated_solve.py` (a one-dof ODE, so no MPI benefit
whatsoever - it is just the smallest reproduction) under `mpirun -n 2`:

    RuntimeError: src/problem.cpp:4631: This likely does not work in parallel

raised from `Problem::sparse_assemble_row_or_column_compressed_base_problem`
(`src/problem.cpp:4622`) via `assemble_multiassembly` (`src/problem.cpp:5014`) and
`MultiAssembleRequest.assemble` (`pyoomph/generic/bifurcation_tools.py:527`).

A throwaway prototype (each rank assembling the full element range, the residual/Jacobian handoff
sliced to the caller's distribution, and a gather/scatter around the deflation solve) got past all
three of the layers below and then **hung** - it was still running Newton steps at a 600 s timeout.
So the list in §2 is what is known to be wrong, not necessarily all of it; see B6.

## 1. What the machinery does, and where the assumption of "one rank" is baked in

    tracker.get_residuals_and_jacobian()          bifurcation_tools.py:742 (FoldTracker, e.g.)
      -> MultiAssembleRequest.assemble()          bifurcation_tools.py:527
         -> Problem::assemble_multiassembly       problem.cpp:5014
            installs CustomMultiAssembleHandler, then
            -> sparse_assemble_row_or_column_compressed_base_problem   problem.cpp:4622
         <- ONE complete global CSR + global dense vectors, as numpy/scipy
      builds the bordered system in Python (scipy.sparse.block_array)
    Problem::get_jacobian / get_residuals         problem.cpp:1663 / 1594 / 1631
      -> get_custom_residuals_jacobian (Python)   problem.py:2430
      <- the bordered system, copied into oomph's DoubleVector/CRDoubleMatrix
    -> oomph SuperLUSolver -> pyoomph LA backend (PETSc/pardiso/scipy)

Three properties of that pipeline are what "serial only" means concretely:

* the multi-assembly returns the **whole** system to every caller, indexed 0..ndof-1;
* the augmented dofs (an eigenvector `V`, a parameter, Lagrange multipliers) are appended to
  `Dof_pt` by `add_augmented_dofs` (`problem.cpp:4969`), which rebuilds the dof distribution
  **non-distributed** (`problem.cpp:5002`), and `DofAugmentations::split` (`problem.cpp:5569`) reads
  the raw `Dof_pt` array;
* the trackers do their algebra on global numpy vectors: `numpy.dot(V, V0)`,
  `numpy.linalg.norm(V)`, `J@V`, `scipy.sparse.block_array([[J, None, col(dRdP)], ...])`.

Two MPI regimes have to be kept apart throughout:

* **(a) replicated** - `mpirun -n N` without `--distribute`. Every rank holds the whole mesh and the
  whole dof vector. oomph-lib still splits the *element loop* across ranks
  (`First_el_for_assembly`/`Last_el_plus_one_for_assembly`) and the linear algebra is genuinely
  row-distributed: `SuperLUSolver::solve(Problem*)` builds
  `LinearAlgebraDistribution(comm, ndof, true)` (`src/thirdparty/oomph-lib/include/linear_solver.cc:859`).
* **(b) distributed** - `--distribute`. The mesh is partitioned, elements are halo/non-halo, and
  `Dof_pt` holds only this rank's dofs. Every global-vector assumption above is then false as well.

Regime (a) is where the win is and where the work should start. Bifurcation tracking on a real 2D/3D
problem is dominated by the elemental assembly of `J`, `dJ/dp` and the Hessian-vector products, and
in (a) the O(ndof) memory per rank is already implied by the replicated mesh.

## 2. The blockers, each verified against the reproduction

**B1 - the assembly refuses to run.** `problem.cpp:4631` (nproc > 1) and `:4634` (distributed).
The element range would otherwise be this rank's slice, and nothing in the path reduces over the
ranks, so lifting the throw without a reduction hands each rank a partial residual and Jacobian.

**B2 - the residual/Jacobian handoff ignores the distribution it is given.** The custom branches of
`get_residuals` (`problem.cpp:1594`), `get_derivative_wrt_global_parameter` (`:1631`) and
`get_jacobian` (`:1663`) copy `info.residuals.size()` entries into a `DoubleVector` and call
`jacobian.build(n_global, ...)`, but the caller has already built both on the solver's distribution
(see the quote in §1). `CRDoubleMatrix::build(ncol, ...)` uses `this->nrow_local()`
(`src/thirdparty/oomph-lib/include/matrices.cc:1694`), so on the one-dof case rank 1 owns no rows and
gets `nnz = row_start[0] = 0` with a one-entry value array - which surfaced as PETSc's
`ValueError: size(J) is 1, expected 0`. The `DoubleVector` writes are worse: they run past the end of
the local storage with nothing to catch them. **This is a live memory-corruption bug independent of
everything else here**, and it is the one part of the plan worth doing on its own merits.

**B3 - the augmented dof distribution disagrees with the solver's.** `add_augmented_dofs` builds the
dof distribution with `distributed=false` (`problem.cpp:5002`), while the solver picks a distributed
one for the same ndof. In (a) that is survivable because the dof *values* are replicated; in (b) it
is not.

**B4 - no LA backend supports the deflation solve under MPI.** `DeflationAssemblyHandler` is the only
assembler with `has_custom_solve_routine` (`bifurcation_tools.py:1560`): its `custom_solve_routine`
(`:1563`) wraps three solves of the same Jacobian into a Sherman-Morrison correction, using global
dot products. PETSc refuses it outright, in the serial *and* the distributed path
(`pyoomph/solvers/petsc.py:651`, `:742`) - which also means deflation cannot use PETSc at all, not
just under MPI. Pardiso's distributed path gathers the system onto rank 0 but then passes the
**local** `b` into the routine (`pyoomph/solvers/pardiso.py:1127`, against the gathered `b_global`
used on the line below it) - a latent wrong-answer bug. scipy/SuperLU is serial-only.

**B5 - the Python algebra is global.** Every reduction in the trackers (`numpy.dot`,
`numpy.linalg.norm`), every `J@V`, and `split()` reading `Dof_pt` assume one rank owns everything.
This costs nothing in regime (a) - the vectors really are replicated - and is the bulk of the work
for regime (b).

**B6 - unexplained hang.** The prototype completed several Newton solves and then stopped making
progress. Candidates, in order of suspicion: a rank-dependent decision inside the deflation loop
(`Problem.iterate_over_multiple_solutions_by_deflation`, `problem.py:6822` - it retries from random
perturbations, and any RNG that is not identically seeded on every rank sends the ranks down
different control flow), the Newton failure/`max_newton_iterations` recovery path taking different
branches per rank, and PETSc's `_ksp_solve_checked` rebuild retry. **Diagnose this before committing
to stage 2** - it may well be an entirely separate bug from the assembly, and it decides whether the
"identical Python on every rank" assumption that stages 1-3 rest on actually holds.

## 3. Design decision: which parallelism, in which regime

Three ways to make the pipeline parallel, evaluated for regime (a):

**Option A - replicated LA.** Every rank assembles the whole system and solves it redundantly with a
*serial* backend. Rejected: it needs a per-rank serial solver, and the MPI default backend is
PETSc/MUMPS on `COMM_WORLD` (`petsc.py:625` creates the Mat on the world communicator), so this
requires either `COMM_SELF` PETSc objects or forcing a different backend under MPI. It also buys no
speedup at all and extends to (b) not at all.

**Option B - fully row-distributed.** The multi-assembly returns each rank's row block, the trackers
work on local blocks with allreduces for every inner product, and the border rows/columns live on a
designated rank. This is the endpoint for regime (b) and the only design that reduces memory per
rank. It touches every tracker in `bifurcation_tools.py` and needs a distributed bordered solve.

**Option C - parallel assembly, replicated algebra, distributed solve.** Keep the element loop split
across ranks (real parallel work), reduce the CSR/residual over the communicator, `Allgather` so
every rank holds the identical global system, leave all the Python untouched, and slice to the
caller's distribution on the way into the linear solver. O(ndof) memory per rank, and the assembly -
the expensive part for these systems - is the part that parallelises.

**Recommendation: C for regime (a), then B for regime (b).** C is a legitimate endpoint for (a), not
a stopgap: in (a) the mesh is replicated anyway, so replicated vectors cost nothing extra, and the
Python trackers - which are the user-extensible part, the whole point of `CustomBifurcationTracker` -
keep working unchanged. B is a much larger change and only pays for itself once `--distribute` is in
play.

## 4. Stages

### Stage 0 - refuse honestly (small, do first)

Add a `_require_single_rank(...)`-style guard next to `_require_non_distributed`
(`pyoomph/generic/problem.py:1515`), and call it from `Problem.set_custom_assembler`
(`problem.py:2415`) so the failure names the feature at the point the user switched it on, instead of
arriving from `problem.cpp:4631` five frames deep - or, worse, not arriving at all (B2 corrupts
memory silently). Also fix the pardiso `b`/`b_global` mix-up (B4), which is a two-character fix and
wrong today regardless of this plan.

Buys: a clear message; a known-good baseline for the tutorial harness under `mpirun`.

### Stage 1 - make the handoff distribution-correct (B2)

In the three custom branches (`problem.cpp:1594`, `:1631`, `:1663`): read `first_row()`/`nrow_local()`
off the incoming `DoubleVector`/`CRDoubleMatrix`, copy only that row block, and rebase the CSR row
starts (column indices stay global). Build a non-distributed distribution only when the caller handed
in something unbuilt. In serial the block is the whole thing, so nothing changes there.

Buys: removes the out-of-bounds writes. Prerequisite for every later stage. Prototyped already
(kept out of the tree); it turns the PETSc `size(J)` error into a working distributed solve of a
correctly assembled system.

### Stage 2 - parallel multi-assembly for regime (a) (B1, Option C)

In `sparse_assemble_row_or_column_compressed_base_problem`:

1. keep the `First_el_for_assembly` element slice (do **not** make every rank walk every element);
2. after the local map-based accumulation, reduce across the communicator. The cheap correct version
   is an `Allreduce` on the dense residual vectors plus a merge of the per-row `std::map`s; the fast
   version, once it works, is to build the frozen sparsity pattern (`build_frozen_sparsity`,
   `problem.cpp:3492` - it is a purely local walk over the replicated mesh, so every rank derives the
   *same* pattern with no communication) and then `Allreduce` the value arrays in place, which turns
   the whole reduction into one collective per matrix;
3. leave the load-balance bookkeeping (`Elemental_assembly_time`,
   `recompute_load_balanced_assembly`) alone - the element split is real here, so the timings mean
   what they normally mean;
4. keep the `Problem_has_been_distributed` throw (that is stage 4).

Buys: the assembly - the dominant cost - actually runs `nproc`-way parallel, and every tracker works
under `mpirun -n N` with no Python change. Validate by asserting the assembled CSR is bit-identical
to the serial one at 1, 2 and 4 ranks (same element set, so only summation order differs - compare
with a tolerance, and say so).

### Stage 3 - the deflation solve under MPI (B4)

Deflation needs `solve_Jx_b` several times on the same factorisation plus global dot products. With
the system replicated (stage 2) and the solve row-distributed, the adapter is: `Allgatherv` the local
RHS block, run the routine on global vectors with a `solve_Jx_b` that scatters into the KSP and
gathers the result back, then write this rank's block into `b`. Roughly 30 lines in
`petsc.py::solve_distributed`, plus the same treatment in `solve_serial` so deflation works with
PETSc in serial at all. The gather/scatter branch must be chosen from the caller ("am I the
distributed entry point?"), never from `len(b) == n` - on a one-dof, two-rank problem rank 0 owns
every row and rank 1 owns none, so a local test sends the ranks into different branches and hangs the
first collective.

Buys: the two deflation tutorials run under `mpirun`. Lowest value of all the stages (deflation is
used by two tutorial scripts and nothing else) - but it is also where B6 will have to be settled.

### Stage 4 - regime (b), `--distribute` (B3, B5, Option B)

Only worth starting once stages 1-3 are in and the `--distribute` remeshing/adaptivity work it sits
on is settled. Sketch: augmented dofs get a genuine distribution (the border rows on the last rank,
`add_augmented_dofs` building a distributed `LinearAlgebraDistribution`); `split()` returns local
blocks; the trackers get a small reduction API (`self.dot(a,b)`, `self.norm(a)`) so that
`numpy.dot`/`numpy.linalg.norm` become allreduces and user-written trackers have one obvious way to
stay correct; `block_array` assembles local blocks with global column indices. The `--distribute`
guard stays until all of that is done.

## 5. Validation

* `docs/source/tutorial/temporal/stability/deflated_solve.py` and `deflated_continuation.py`: the
  reported solution set must match the serial run (serial today: `x = [0, 1, -1]` at `r=1`).
* One tracker tutorial per family (fold, Hopf, normal-mode) at 1 vs 2 vs 4 ranks: same critical
  parameter to solver tolerance, same eigenvector up to sign/normalisation.
* A unit test at the C++ boundary: assemble the same augmented system in serial and at N ranks and
  compare the CSR - this is what would have caught B2 immediately.
* Everything MPI goes behind the existing `slow`/`--full` marking, as `tests/test_mpi_eigenvalues.py`
  does.

## 6. Open questions

* B6 - is the hang a control-flow divergence between ranks, and if so, does anything else in the
  custom-assembler path make rank-dependent decisions? Nothing in the design above survives if the
  ranks can take different Python branches.
* Is Option C's `Allgather` acceptable at the sizes these trackers are actually run at? An augmented
  system is ~2x the base ndof and the collective is O(ndof) per Newton step per matrix; at the
  200k-dof ceiling that is a few MB per rank per step, which is fine, but it should be measured
  in-process (§ "Benchmarking" in `CLAUDE.md`) rather than assumed.
* Should the trackers' reduction API (stage 4) be introduced early, as no-ops in regime (a), so that
  user-written trackers written today keep working when (b) lands?
