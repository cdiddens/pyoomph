# Structural assembly: precomputed CSR sparsity, value-only re-assembly and solver reuse

Branch: `structural_assembly`. Status: **investigation + plan only, nothing implemented yet.**
All measurements in §2 were taken on `main` (commit `d03a562`) on this machine; all file/line
references are to the tree state at the time of writing.

**Goal.** Today every Jacobian assembly rebuilds the CSR structure (row starts + column indices)
from scratch, and every linear solve therefore re-does its symbolic phase (Pardiso phase 11, PETSc
`MatCreateAIJ` + preallocation). Neither is necessary: as long as the equation numbering is
unchanged, the sparsity pattern of the Jacobian is fixed. If we compute that pattern **once**, we
can (a) assemble straight into a preallocated value array, (b) hand solvers a "values only have
changed" promise so they can skip reordering/symbolic factorisation, and (c) get the
"zeros on the diagonal" that some PETSc backends require *for free*, instead of the current manual
`MatShift(0.0)` hack.

**MPI is a requirement, not an afterthought.** Every feature below must work, and be tested, under
`--distribute` with `mpirun -n N`, on the same footing as the serial path. The distributed assembly
is a *separate* oomph routine (`parallel_sparse_assemble`) with its own containers and an
index+value exchange between ranks, so it needs its own treatment — but it is also where a frozen
pattern pays off most, because the column-index half of every exchange becomes redundant. §7 is the
MPI design; the MPI work is folded into each phase rather than deferred to the end.

---

## 1. How pyoomph assembles the Jacobian today

### 1.1 The call chain

```
Problem::newton_solve                              oomph problem.cc:9078...
  └── LinearSolver::solve(problem, dx)             (SuperLUSolver)
        ├── problem_pt->get_jacobian(residuals, CR_jacobian)     problem.cc:4068
        │     └── sparse_assemble_row_or_column_compressed(...)  problem.cc:4501 (virtual)
        │           └── pyoomph::Problem::sparse_assemble_row_or_column_compressed
        │                                                         src/problem.cpp:2755
        │                 ├── ..._for_periodic_orbit()            src/problem.cpp:2510
        │                 └── oomph::Problem::sparse_assemble_row_or_column_compressed
        │                       → one of five container strategies (problem.cc:4601/4947/5362/5715/6078)
        ├── factorise()   → superlu(op_flag=1, ...)   linear_solver.cc:2303
        └── backsub()     → superlu(op_flag=2, ...)   linear_solver.cc:2610
```

`superlu()` is **not** SuperLU here. pyoomph defines C-linkage symbols with SuperLU's exact
signatures (`src/nanobind/solver.cpp:145`) so that oomph-lib's solver calls land in
`GeneralSolverCallback::solve_la_system_serial(op_flag, n, nnz, nrhs, values, rowind, colptr, b,
ldb, transpose)`, which is a nanobind trampoline into the Python solver classes
(`pyoomph/solvers/pardiso.py:474`, `pyoomph/solvers/petsc.py:249`, `mumps.py`, ...). The CSR arrays
arrive as zero-copy numpy views onto oomph-lib's `CRDoubleMatrix` buffers.

`op_flag` semantics (from oomph's SuperLU wrapper): **1 = factorise**, **2 = back-substitute**,
3 = clean up (never reached in pyoomph, since the shim leaves `f_factors` at NULL).

### 1.2 Where the structure is (re)built

All five oomph strategies do the same thing at different memory/speed trade-offs:

1. loop over elements, call `assembly_handler_pt->get_all_vectors_and_matrices()` → dense
   `nvar × nvar` elemental Jacobian;
2. for every `(i,j)` with `fabs(value) > Numerical_zero_for_sparse_assembly`, accumulate
   `container[eqn_number(e,i)][eqn_number(e,j)] += value` — into a `std::map`, a list, a vector of
   pairs, two parallel vectors, or two plain arrays;
3. compress: count entries per row, `new int[nnz]`, `new double[nnz]`, walk the sorted container and
   emit `column_index` / `row_start` / `value`.

Step 3 allocates **fresh** `row_start` and `column_index` arrays on every single assembly.
`CRDoubleMatrix::build_without_copy()` takes ownership and frees them again after the solve.

`Numerical_zero_for_sparse_assembly` is `0.0` (oomph `problem.cc:217`) and the test is a strict
`>`, so **exact zeros are dropped**. This is the crux: the emitted pattern is *value dependent*.

pyoomph has three additional, structurally identical assembly routines of its own:

| routine | file | purpose | container |
|---|---|---|---|
| `..._for_periodic_orbit` | `src/problem.cpp:2510` | augmented periodic-orbit system | `std::map` |
| `..._base_problem` | `src/problem.cpp:2813` | unaugmented block while bifurcation/arclength augmentation is active; also the engine behind `assemble_multiassembly` | `std::map` |
| `assemble_hessian_tensor` | `src/problem.cpp:2224` | rank-3 Hessian | `SparseRank3Tensor` |

And when the problem is distributed, `get_jacobian` takes an entirely different branch
(oomph `problem.cc:4160`) into `Problem::parallel_sparse_assemble` (oomph `problem.cc:6574`) —
see §7.

### 1.2b An abandoned first attempt already exists

`Problem::update_jacobian_csr_structure()` (`src/problem.cpp:2774`, declared
`src/problem.hpp:328`) is a stub for precisely this idea — it was meant to fill
`global_eqs_to_jacobian_buffer_index` (`src/problem.hpp:315`) so that "a later assembly pass could
write directly into a preallocated buffer instead of building the sparsity pattern from scratch each
time". The body is commented out and it `throw_runtime_error("Implement and check performance")`s.
Phase 2 replaces it; the `std::map<unsigned,unsigned>`-per-row data structure it proposed is *not*
what we want (see §3.1 for the flat-array alternative).

### 1.3 The symbolic information the code generator already has

`src/codegen.cpp:8060-8111` emits, per generated element code and per residual/Jacobian combination,

```c
bool ***contributes_to_jacobian;   // [res_jac_index][field_i][field_j]     jitbridge.h:478
bool  **contributes_to_residual;   // [res_jac_index][field_i]              jitbridge.h:477
char **contribution_names;         // field names, contribution_entries_size of them
```

i.e. an **exact symbolic field-coupling table**: "does ∂(residual of field i)/∂(field j) contain any
term at all?". `Problem::assemble_defined_field_list()` (`src/problem.cpp:3160-3280`) already
consumes it, but only to build a *problem-global* union
(`jacobian_contributing_fields[res][row_field][col_field]`) used to detect and pin fields with empty
Jacobian rows/columns. Nobody uses it for sparsity.

`BulkElementBase::get_dof_names()` (`src/elements.cpp:5626`, interface variant `:13819`) shows the
machinery to map *local dof index → (field, space, node)* already exists via
`eleminfo.nodal_local_eqn` / `eleminfo.pos_local_eqn` and the functable field info. That is the
missing link needed to apply the per-code coupling table at element level.

### 1.4 Existing precedent inside pyoomph

`SparseRank3Tensor::finalize_for_vector_product()` (`src/hessian_tensor.hpp:97`) already implements
exactly the pattern we want, one level up: it freezes a CSR structure once, returns
`(col_indices, row_start)`, and afterwards `right_vector_mult()` returns **only a value array** that
is stuffed into a pre-built `csr_matrix` whose `indptr`/`indices` never change. See
`pyoomph/generic/assembly.py:216-219` and `:347`. `FixedMeshMaxQuadraticNonlinearAssembly` is the
Python-level "structure is constant, only values change" assembler. The proposal below is the same
idea applied to the Jacobian itself, in C++.

---

## 2. Measured baseline

Machine: this dev box, `system` compiler, `optimize_for_max_speed()`, serial, MKL Pardiso available.
Scripts live in the session scratchpad; they will be promoted to `tests/benchmarks/` in Phase 0.

### 2.1 Benchmark A — 3D lid-driven cavity, Taylor–Hood Navier–Stokes, `CuboidBrickMesh(N=8)`

`ndof = 10 853`, `nnz = 1 800 539` (166 per row), 512 elements, `nvar = 89` per element,
2 688 761 raw scatter entries.

| step | time |
|---|---|
| residuals only | 55 ms |
| residuals + Jacobian | **424 ms** |
| residuals + Jacobian + mass matrix (`assemble_eigenproblem_matrices`) | 467 ms |
| MKL Pardiso phase 11 (analysis / reordering) | **179 ms** |
| MKL Pardiso phase 22 (numerical factorisation) | 111 ms |
| MKL Pardiso phase 33 (triangular solves) | 4.7 ms |
| MKL Pardiso phase 12 (= 11 + 22, what `factor()` does today) | 246 ms |
| scipy `splu` (for scale) | 3.10 s |
| Python-side `csr_matrix(...).copy()` of J | 2.4 ms |
| PETSc `createAIJ` + `assemble` | 20.4 ms |
| PETSc `createAIJ` + `shift(0.0)` + `assemble` | 16.8 ms |

**A full Newton step costs 424 + 246 + 5 ≈ 675 ms, of which 179 ms (26 %) is a symbolic
factorisation that recomputes an unchanged result.**

Container choice barely matters — the whole `Sparse_assembly_method` knob is a rounding error:

| method | res+Jac |
|---|---|
| `vectors_of_pairs` (default) | 425 ms |
| `two_arrays` | 420 ms |
| `two_vectors` | 429 ms |
| `lists` | 504 ms |
| `maps` | 506 ms |

### 2.2 Benchmark B — 2D lid-driven cavity, `RectangularQuadMesh(N=60)`

`ndof = 32 042`, `nnz = 1 240 174` (38.7 per row).
residuals 39 ms · residuals+Jacobian 129 ms · +mass matrix 148 ms · scipy `splu` 2.12 s.

### 2.3 The pattern is *not* stable today

3D NS, `N=4`, `ndof = 1153`:

| state | nnz |
|---|---|
| at `U = 0` | 138 716 |
| after `solve()` | 138 817 |

572 entries appear, 36 disappear. `J0.indices == J1.indices` is **False**. So "just reuse the
arrays and hope" is not an option, and it explains why `pardisoSolver.update_matrix_values()`
(`pyoomph/solvers/pardiso.py:305`) — which bails out on any length mismatch — cannot be relied on.
Any reuse scheme **must** be built on a value-independent pattern.

### 2.4 A value-independent pattern costs almost nothing

"Structural pattern" = union over elements of all `(eqn_number(e,i), eqn_number(e,j))` pairs,
computed from element connectivity alone, ignoring values.

| case | ndof | numerical nnz | structural nnz | ratio | rows without a diagonal entry | numerical ⊄ structural |
|---|---|---|---|---|---|---|
| NS 3D, `N=8` | 10 853 | 1 800 539 | 1 816 855 | **1.009** | 0 | 0 |
| NS 2D, `N=40` | 14 162 | 542 104 | 556 738 | **1.027** | 0 | 0 |
| NS + advection–diffusion 2D, `N=40` | 20 642 | 840 730 | 1 132 540 | **1.347** | 0 | 0 |

Two things fall out of this table:

* For single-physics problems the structural pattern is within 1–3 % of the numerical one — the
  extra explicit zeros are essentially free.
* The structural pattern **always contains the full diagonal** (every dof belongs to at least one
  element, and `(i,i)` is in that element's block). The 728 rows of Benchmark A that currently have
  *no* structural diagonal entry (the pressure–pressure block of Taylor–Hood) — precisely what
  `petsc_mat.shift(0.0)` at `pyoomph/solvers/petsc.py:281` is papering over — are covered for free.
* For weakly coupled multiphysics the pure connectivity pattern over-allocates (1.35×) because it
  includes field blocks that are symbolically zero (`c` vs `pressure` here).

Applying the field-coupling mask (what `contributes_to_jacobian` gives us) to the coupled case:

| pattern | nnz | ratio to numerical | rows without diagonal |
|---|---|---|---|
| connectivity only (Tier A) | 1 132 540 | 1.347 | 0 |
| field-pair masked (Tier B) | 840 730 | **1.0000** | 1680 |

Tier B reproduces the numerical pattern *exactly* — but then the pressure diagonal is empty again,
so **Tier B needs the explicit "force diagonal entries" option**. The two features the task
description asks for are therefore the same feature, seen from two sides.

---

## 3. Design

### 3.1 Object: `JacobianSparsity`

New C++ class (proposed `src/sparsity.hpp` / `.cpp`), owned by `pyoomph::Problem`:

```cpp
class JacobianSparsity {
  unsigned ndof;
  std::vector<int> row_start;      // ndof+1
  std::vector<int> column_index;   // nnz
  std::vector<int> scatter_index;  // per element, per (i,j): position in the value array
  std::vector<int> element_offset; // where each element's block starts in scatter_index
  unsigned long generation;        // bumped on every rebuild; published to solvers
  bool force_diagonal;
  PatternTier tier;                // Connectivity | FieldPair | Exact
};
```

Build (once per equation numbering):

1. loop over elements, collect `eqn_number(e,i)` for `i < ndof(e)`, discard `< 0`;
2. optionally mask pairs by the element code's `contributes_to_jacobian[res][field(i)][field(j)]`;
3. optionally add `(i,i)` for every row if `force_diagonal`;
4. sort/unique per row → `row_start`, `column_index`;
5. second pass over elements, binary-search each `(i,j)` in its row → `scatter_index`.

Assembly then becomes

```cpp
std::fill(value.begin(), value.end(), 0.0);
for each element e:
    get_all_vectors_and_matrices(e, el_res, el_jac);   // unchanged
    const int *idx = &scatter_index[element_offset[e]];
    for (k = 0; k < nvar*nvar; ++k) value[idx[k]] += el_jac_flat[k];
```

with no map, no sort, no compression, no allocation. `scatter_index` costs
`4 bytes × raw_entries` — 10.8 MB for Benchmark A, versus 14 MB for the value array itself.
Acceptable, and it must be made optional (`Connectivity` tier without `scatter_index` falls back to
a binary search per entry, ~2× slower scatter but no extra memory).

### 3.2 Invalidation

The pattern is valid exactly as long as the element→equation-number map is. Invalidate in
`pyoomph::Problem::assign_eqn_numbers()` (`src/problem.cpp:1048`) — which already runs after mesh
adaptation, remeshing, (un)pinning, Dirichlet changes and augmented-dof changes. Bump `generation`
there. The Python side already has the matching hooks:
`GenericLinearSystemSolver._before_assigning_equation_numbers()` and
`GenericEigenSolver._before_assigning_equation_numbers()`, called from
`pyoomph/generic/problem.py:2414-2415`, plus `CustomAssemblyBase.actions_after_equation_numbering()`.

Things that do **not** invalidate it: Newton iterations, time steps, parameter continuation, moving
(but not re-generated) meshes, changes of global parameters.

### 3.3 Telling solvers "only values changed"

Extending `GeneralSolverCallback::solve_la_system_serial` with an extra argument would break every
existing Python solver (it is a nanobind trampoline with a fixed arity, and third-party solvers
subclass it). Instead, publish a **read-only generation counter on the problem**:

```python
problem.jacobian_structure_id   # unsigned, changes iff the pattern changed
```

Solvers opt in:

* **Pardiso** (`pyoomph/solvers/pardiso.py:474`): if `structure_id` and `nnz` are unchanged, copy
  the new values into the existing `a` array and run **phase 22** instead of phase 12.
  Expected saving on Benchmark A: 179 ms per Newton step (26 % of the step).
  Note this is a *different* mechanism from the existing `try_to_reuse_solver` path, which reuses the
  whole *numerical* factorisation as an iterative-refinement preconditioner and falls back when it
  stops converging. The two compose: phase-11 reuse is always safe, phase-22 reuse is a gamble.
* **PETSc** (`pyoomph/solvers/petsc.py:249`): keep the `Mat`, `MatSetValuesCSR` (or better, hold the
  `values` view and `MatAssemblyBegin/End`) instead of `createAIJ` + `shift(0.0)` + destroy, and
  keep the `KSP` with `setReusePreconditioner` where appropriate. Measured `createAIJ` cost on
  Benchmark A is 20 ms — small next to Pardiso's 179 ms, but for iterative solves the real prize is
  keeping the preconditioner.
* **MUMPS** (`pyoomph/solvers/mumps.py`): analogous (job 1 once, job 2 per step).

### 3.4 "Add zeros on the diagonal"

Public option, proposed name `problem.force_jacobian_diagonal_entries = True` (C++
`set_force_jacobian_diagonal_entries`). Semantics: the assembled CSR contains an entry for every
`(i,i)`, `0 ≤ i < ndof`, even if its value is exactly zero. Implemented in the pattern builder
(step 3 above), so it costs nothing at assembly time.

This replaces `petsc_mat.shift(0.0)` at `pyoomph/solvers/petsc.py:281` and `:341`, and the
commented-out `set_diagonal_zero_entries` stub at `src/problem.hpp:502`. With Tier A the option is a
no-op (the diagonal is already there); with Tier B it is required. Default: **on** when a pattern is
in use, because it is what PETSc/MUMPS/hypre expect and the cost is ≤ ndof extra entries
(0.2 % on the coupled benchmark).

### 3.5 Pattern tiers

| tier | source | over-allocation | diagonal | effort |
|---|---|---|---|---|
| **A — connectivity** | element `eqn_number` lists only | 1.0–1.35× | free | low |
| **B — field-pair** | + per-code `contributes_to_jacobian` masked by local-dof→field map | ~1.00× | needs `force_diagonal` | medium |
| **C — exact symbolic** | codegen emits a per-element-type `nvar × nvar` boolean mask | 1.00× | needs `force_diagonal` | high |

Tier C is listed for completeness; Tier B already hit 1.0000× on the coupled benchmark, so C is
unlikely to be worth it. The pattern must always be a **superset** of the numerical pattern — that
invariant is checkable and will be a test (§6).

### 3.6 A cheap Tier-A shortcut worth validating first

`Numerical_zero_for_sparse_assembly` is a protected `oomph::Problem` member and the filter is
`if (fabs(value) > Numerical_zero_for_sparse_assembly)`. Setting it to a **negative** value makes the
test always true, which turns *every existing* assembly routine — oomph's five, plus pyoomph's
periodic-orbit and base-problem ones — into a Tier-A structural assembly, with no changes to any
assembly loop. That immediately buys: a stable pattern, a full diagonal, and hence solver reuse
(§3.3), at the price of assembling 1–35 % more entries. It does **not** buy the fast scatter (§3.1).

This is the right thing to land first: it makes the payoff measurable before committing to the
larger refactor, and it is trivially revertible.

---

## 4. What this does *not* fix

The measurements say assembly (424 ms) is bigger than the solve (246 ms) on Benchmark A, so
eliminating the symbolic phase is only half the story. Where does the 424 ms go? Residual-only
assembly is 55 ms, so ~370 ms is elemental Jacobian evaluation *plus* scatter/compression. The
per-raw-entry cost is suspiciously identical for very different equations (91 ns for 3D NS,
89 ns for 3D C2 Poisson, whose elemental Jacobian is far cheaper per entry), which points at the
scatter dominating — but `perf` and `gdb` attach are both blocked on this box (`perf_event_paranoid`,
`ptrace_scope`), so this is **not yet established**. Splitting that number is Phase 0 and gates how
much §3.1 is worth: if the scatter is 300 ms, the preallocated-scatter route saves more than the
solver reuse does; if it is 50 ms, Phase 1 is most of the win and Phase 2 is optional.

---

## 5. Phased plan

### Phase 0 — instrumentation and a benchmark harness *(prerequisite)*

* `tests/benchmarks/bench_assembly.py`: parametrised (2D/3D, NS / Poisson / coupled multiphysics,
  mesh size), reporting residual / Jacobian / mass-matrix assembly, per-Pardiso-phase timings, and a
  full Newton step breakdown. Promote the scratchpad scripts. **Must accept `--distribute` and run
  under `mpirun`**, reporting per-rank and max-over-ranks assembly times (oomph already has
  `Doc_imbalance_in_parallel_assembly`, `problem.cc:7013`).
* Add C++-side timers inside `sparse_assemble_row_or_column_compressed_base_problem` (elemental
  evaluation / scatter / compression) behind a `problem.report_assembly_timings` flag, and mirror
  them into the oomph routine we end up owning. **This resolves §4.**
* Record baselines for the three benchmark cases (serial and `-n 2`/`-n 4`) into the doc.

### Phase 1 — value-independent pattern + solver structure reuse *(the cheap win)*

* `problem.keep_structural_zeros` → sets `Numerical_zero_for_sparse_assembly` negative (§3.6).
  Works unchanged on the distributed path, which uses the same filter (oomph `problem.cc:6899`).
* `problem.force_jacobian_diagonal_entries` (no-op under Tier A, wired up properly in Phase 2).
* `problem.jacobian_structure_id`, bumped in `pyoomph::Problem::assign_eqn_numbers`. It is
  automatically consistent across ranks because `assign_eqn_numbers` is collective — but that must
  be asserted, not assumed (§7.3).
* Pardiso: phase-11 reuse when `structure_id` and `nnz` are unchanged, in both `solve_serial` and
  `solve_distributed` (`pyoomph/solvers/pardiso.py:474`, `:543`).
* PETSc: keep `Mat`/`KSP`, value-only update, drop `shift(0.0)` — likewise in both.
* Benchmark: expect −26 % per Newton step on Benchmark A, minus the extra assembly cost of the
  1.7 % larger pattern; plus the distributed runs.

### Phase 2 — precomputed CSR pattern and direct scatter

* `JacobianSparsity` (§3.1), Tier A, built in `assign_eqn_numbers`. Replaces the abandoned
  `update_jacobian_csr_structure()` stub (§1.2b).
* A pyoomph-owned `sparse_assemble_row_or_column_compressed` that uses it, replacing the delegation
  to oomph for the serial non-augmented case. Keep the oomph path reachable via
  `sparse_assembly_method = "..."` as a fallback and as the differential-test oracle.
* The handed-out `row_start`/`column_index` arrays must still be `new int[]` copies, because
  `CRDoubleMatrix::build_without_copy()` takes ownership and frees them (oomph `problem.cc:4155`).
  Two `memcpy`s of ~7 MB (~2 ms on Benchmark A) — measure before optimising this away with a
  persistent `CRDoubleMatrix`.
* Extend to `..._base_problem` (`src/problem.cpp:2813`) and `..._for_periodic_orbit`
  (`src/problem.cpp:2510`), which are `std::map`-based and hence the slowest paths in the codebase.

### Phase 2b — distributed pattern and value-only exchange *(MPI, same phase, own step)*

* `pyoomph::Problem::parallel_sparse_assemble` override holding a `DistributedJacobianSparsity`
  (§7.2): cached `my_eqns`, per-rank send plan, frozen local column indices and scatter map.
* Per assembly, exchange **values only**; send the column indices once per `structure_id`.
* Falls back to the oomph routine whenever the pattern is invalid, so this is always revertible at
  runtime.

### Phase 3 — Tier B (field-pair pruning)

* Per-element-code map: local dof index → contribution/field index. Follow `get_dof_names()`
  (`src/elements.cpp:5626`, `:13819`); cache per `(code, element type)`, not per element.
* Mask the pattern with `ft->contributes_to_jacobian[res][j][k]`.
* Make `force_jacobian_diagonal_entries` default to on here.
* Guard: the mask must be per *residual/Jacobian combination* (`_solved_residual`,
  `src/problem.hpp:293`) — switching the active residual changes the coupling table and therefore
  the pattern, so it must bump `structure_id`.

### Phase 4 — eigenproblems and bifurcation tracking

Only after Phases 1–3 are validated on the plain Newton path.

* **Eigenproblems.** `assemble_eigenproblem_matrices` (`src/problem.cpp:1091`) →
  `get_eigenproblem_matrices` (oomph `problem.cc:8641`) → same sparse-assembly routine with
  `n_matrix = 2`. J and M share the element connectivity, so they can share **one** pattern
  (M's numerical pattern is a strict subset; storing M on J's pattern costs explicit zeros but makes
  `J - σM` a pure `axpy` on the value arrays — attractive for shift-and-invert sweeps). Measured
  cost of the second matrix today: 43 ms on Benchmark A. SLEPc/ARPACK shift-invert would then only
  need one symbolic factorisation for a whole parameter sweep.
* **Bifurcation tracking.** `MyFoldHandler` / `MyPitchForkHandler` / `MyHopfHandler` /
  `AzimuthalSymmetryBreakingHandler` / `PeriodicOrbitHandler` (`src/bifurcation.hpp:53-532`) each
  define their own `eqn_number(elem, i)` over the *augmented* dof vector. The pattern is still fixed
  for a given handler + numbering, but it changes when the handler is installed/removed
  (`add_augmented_dofs`, `src/problem.cpp:3051`) — so that must bump `structure_id` too. The
  augmented border rows/columns are dense-ish and must be added to the pattern explicitly.
* **`assemble_multiassembly`** (`src/problem.cpp:3096`, used by
  `pyoomph/generic/bifurcation_tools.py:527`): assembles several matrices in one pass through the
  `std::map`-based base-problem routine. Biggest single beneficiary of Phase 2, since all its
  matrices share one pattern.
* **`FixedMeshMaxQuadraticNonlinearAssembly`** (`pyoomph/generic/assembly.py:99`) should be
  revisited afterwards: with a shared pattern its `matJ + w0*matM + 0.5*H` becomes an elementwise
  `data` operation instead of scipy sparse additions that re-derive the union pattern every step.

### Out of scope (for now)

* Threaded/openmp assembly. A precomputed scatter index makes a colour- or lock-free parallel
  scatter feasible, but that is a follow-up.
* Making bifurcation tracking / `assemble_multiassembly` work under MPI at all — pyoomph's
  `..._base_problem` throws for `nproc > 1` today (`src/problem.cpp:2822-2825`). That is a
  pre-existing gap, not one this work introduces; the plan must not make it worse, and §7.4 records
  what would be needed.

---

## 6. Correctness gates

Every phase must keep these green before it lands:

1. **Superset invariant.** For a set of representative problems, `structural_pattern ⊇
   numerical_pattern` at several dof states (U=0, after 1 Newton step, converged). Verified in
   §2.4 for the three benchmark cases with `missed = 0`; turn it into a test.
2. **Bit-identical solutions.** Newton trajectories (residual norm per iteration, converged dof
   vector) must match the pre-change values to round-off with the pattern route enabled and
   disabled. Explicit zeros must not change any result.
3. **Differential test against the oomph path.** With the same problem, compare
   `sparse_assembly_method="two_arrays"` (oomph) against the pyoomph pattern route:
   same values on the common pattern, extra entries all exactly 0.0.
4. **Invalidation coverage.** After each of: mesh adaptation, remeshing, `pin`/`unpin`, Dirichlet
   changes, switching `_solved_residual`, installing/removing a bifurcation handler, adding
   augmented dofs — `structure_id` must have changed and the rebuilt pattern must still satisfy (1).
   This is the highest-risk area: a *missed* invalidation is a silent wrong-answer bug, not a crash.
5. **Existing suites.** `tests/` in full (the adaptive 2D/3D campaigns exercise exactly the
   renumbering paths that can break invalidation), then the tutorial pipeline once, batched at the
   end of a phase — not per fix.
6. **MPI gates** — every one of 1–5 also under `mpirun -n 2` and `-n 4` with `--distribute`, plus:
   * cross-rank agreement and serial agreement on the gathered residual / global observables, the
     two oracles `tests/test_mpi_adaptivity.py` already implements (see its header);
   * `jacobian_structure_id` identical on all ranks after every renumbering
     (an `MPI_Allreduce(MIN/MAX)` assertion under `PARANOID`);
   * the assembled distributed Jacobian equals the serial one (gather and compare on the common
     pattern; extra entries exactly 0.0).

---

## 7. MPI / distributed design

### 7.1 What the distributed path does today

When the problem is distributed, `get_jacobian` (oomph `problem.cc:4160`) calls
`Problem::parallel_sparse_assemble` (oomph `problem.cc:6574`) instead of
`sparse_assemble_row_or_column_compressed`. pyoomph does **not** override it, so this is pure
oomph-lib today. It works in four stages:

1. **`my_eqns`** (`get_my_eqns`, called at `problem.cc:6684`): the sorted set of *global* equations
   this rank contributes to, gathered from its non-halo elements. Note this is **not** the set of
   equations this rank *owns* — a non-halo element can touch a halo node, whose equation lives on
   another rank. Purely a function of the element→eqn map: **structural**.
2. **Local accumulation** into `matrix_col_indices[m][local_row][k]` / `matrix_values[...]`, sized
   by `Sparse_assemble_with_arrays_previous_allocation` and grown by
   `Sparse_assemble_with_arrays_allocation_increment`. Row lookup is a **binary search into
   `my_eqns` per elemental dof** (`problem.cc:6801-6873`), and insertion is a **linear scan of the
   row** (`problem.cc:6932-6968`) — both would disappear with a precomputed scatter index. The value
   filter is the same `fabs(value) > Numerical_zero_for_sparse_assembly` (`problem.cc:6899`), so
   §3.6 applies here verbatim.
3. **Send plan**: `n_eqn_for_proc`, `first_eqn_element_for_proc` (`problem.cc:7086-7105`) and
   `nnz_for_proc` (`problem.cc:7108`). The first two are structural; `nnz_for_proc` becomes
   structural as soon as the pattern is frozen.
4. **Exchange**: each rank ships the rows it does not own to their owner — row starts, **column
   indices** and values — then the owner merges them into its local CSR block.

### 7.2 What a frozen pattern buys under MPI

* **The column-index payload of stage 4 is sent once instead of every assembly.** Per entry the
  exchange currently costs 4 bytes of `unsigned` index + 8 bytes of `double`; freezing the pattern
  drops it to 8. That is a **1/3 reduction in message volume** on the assembly critical path, and it
  removes the per-assembly `nnz_for_proc` handshake.
* Stages 1 and 3 (binary searches, send-plan construction, allocation growth) collapse into table
  lookups.
* The owner-side merge becomes a fixed `value[perm[k]] += recv[k]` scatter instead of a
  pattern-dependent merge.
* Downstream, `structure_id` lets the distributed solvers skip their symbolic phases exactly as in
  the serial case: `pyoomph/solvers/pardiso.py:543` (which currently gathers the whole matrix to
  rank 0 and solves there — so it benefits from serial phase-11 reuse) and
  `pyoomph/solvers/petsc.py:319` (`createAIJ` per rank each solve → keep the `Mat`).

Proposed object, mirroring `JacobianSparsity`:

```cpp
class DistributedJacobianSparsity {
  std::vector<unsigned> my_eqns;              // stage 1, cached
  std::vector<int>      scatter_index;        // elemental (i,j) -> slot in the local value array
  std::vector<int>      element_offset;
  std::vector<int>      local_row_start, local_col_index;   // frozen local block
  std::vector<int>      n_eqn_for_proc, first_eqn_for_proc; // stage 3, cached
  std::vector<int>      recv_permutation;     // incoming value k -> slot in the owned CSR block
  unsigned long         generation;
};
```

### 7.3 MPI-specific invalidation and hazards

* `assign_eqn_numbers` is collective, so bumping `structure_id` there is automatically consistent —
  but a *silently* divergent `structure_id` would make ranks disagree about whether to re-send
  indices and deadlock. Assert equality with an `MPI_Allreduce` under `PARANOID`.
* `Must_recompute_load_balance_for_assembly` / `recompute_load_balanced_assembly()`
  (`problem.cc:7066-7081`) **changes `el_lo`/`el_hi` between assemblies** on non-distributed MPI
  runs. That changes which rank contributes which elemental block, hence `my_eqns` and the scatter
  map — it must invalidate the distributed pattern. This is the subtlest hazard in the whole plan.
* Halo/haloed node list drift after distributed adaptive refinement is a **known open defect** for
  custom 3D simplex meshes (see the `mpi_distributed_adaptivity_gap` notes). The pattern must be
  rebuilt after every adapt anyway, so this work neither fixes nor worsens it — but MPI benchmarks
  should avoid that configuration until it is fixed, or the failure will be misattributed here.
* When MPI misbehaves, check the vendored oomph `PARANOID` blocks first — several of them abort on
  conditions pyoomph legitimately produces (lesson from the mixed-adapt validation campaign).

### 7.4 Bifurcation tracking under MPI

`..._base_problem` (`src/problem.cpp:2813`) and `..._for_periodic_orbit` (`:2510`) both hard-throw
for `nproc > 1`. Making them distributed means giving the augmented rows/columns an owner and
adding them to the exchange — a genuinely separate project. Recorded here so the Phase 4 work does
not silently assume MPI support it does not have.

## 8. Open questions

* Does the residual-only assembly path (`get_residuals`, 55 ms on Benchmark A) share enough with the
  Jacobian path to also benefit? It has no structure at all, so probably not, but 55 ms of a 675 ms
  Newton step is worth a look once §4 is resolved.
* Is `scatter_index` (10.8 MB on Benchmark A, scaling with raw entries) acceptable for the large 3D
  problems people actually run, or does it need the binary-search fallback as the default?
* Should the pattern be exposed to Python (`problem.get_jacobian_sparsity()` → `(indptr, indices)`)?
  It would let `CustomAssemblyBase` implementors and the eigen matrix manipulators
  (`pyoomph/solvers/generic.py:160-350`, which currently do `csr` surgery row by row) work on a
  fixed pattern too.
* Do any *numerical* entries fall outside the connectivity pattern in exotic setups — external data
  coupling across domains, `add_interior_facet_terms`, tracer/ODE couplings, `DofAugmentations`?
  Gate (1) run over a broad problem set is the way to find out.
* Under MPI, is it worth freezing the pattern of the *local* block only (cheap, self-contained) as an
  intermediate step, before also freezing the exchange (§7.2)? The local binary searches and linear
  row scans may already be the bulk of the distributed assembly cost — Phase 0's per-rank timers
  will say.
* Does `Sparse_assemble_with_arrays_previous_allocation` (oomph `problem.h:708` region, reset at
  `problem.cc:2151`/`:2481`) interact badly with `keep_structural_zeros`? It sizes rows from the
  previous assembly; a first assembly at `U = 0` under the old value filter would under-allocate for
  every later one. With structural zeros kept, allocations stabilise immediately — likely a small
  side benefit, worth confirming.
