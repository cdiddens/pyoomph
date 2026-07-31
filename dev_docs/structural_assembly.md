# Structural assembly: precomputed CSR sparsity, value-only re-assembly and solver reuse

Branch: `structural_assembly`. Status: **all planned phases implemented, tested and benchmarked
(§6) — 0, 1, 2, 2b (distributed), 3 and 4.** All measurements in §2 were taken on `main`
(commit `d03a562`) on this machine; §6 records what the implemented work actually moved. All
file/line references are to the tree state at the time of writing.

All tests pass: the `test_moving_mesh_distributed` regression of §7c is resolved (§7c).

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

### 3.7 The threshold must be per matrix, not per problem

A single problem-wide threshold is the wrong granularity, because a sparse assembly pass can build
**several matrices at once** and they do not want the same policy:

* `get_eigenproblem_matrices` (oomph `problem.cc:8641`) assembles with `n_matrix = 2`:
  index 0 = the Jacobian, index 1 = the mass matrix.
* `assemble_multiassembly` (`src/problem.cpp`) packs an arbitrary number of matrices.

Measured on the 2D cavity, `N=8` — the mass matrix is **~3× sparser** than the Jacobian, because only
fields carrying a time derivative contribute to it at all:

| | J nnz | M nnz | M/J |
|---|---|---|---|
| both value-filtered | 17 560 | 6 050 | 0.34 |
| structural zeros on both | 18 178 | 18 178 | 1.00 |
| **structural on J only** | 18 178 | 6 050 | 0.33 |

Forcing J's connectivity pattern onto M inflates it threefold for no benefit: what a stable pattern
buys is symbolic-factorisation reuse, and the operator being factorised is J. So the fix is a
per-matrix threshold — implemented as a new `virtual double
numerical_zero_for_sparse_assembly(matrix_index)` on `oomph::Problem` (defaulting to the existing
member, so oomph behaviour is unchanged) called at the seven filter sites, with pyoomph overriding
it. Recorded in `src/thirdparty/INFO_oomph-lib` per the vendoring convention.

Two consequences worth stating explicitly:

* **The mixed policy is still safe for eigenproblems.** A shift-and-invert solve factorises
  `J − σM`, whose pattern is the union of the two — and M's entries all lie inside J's structural
  pattern (both come from the same elemental blocks), so that union *is* J's pattern. It stays
  value-independent and reusable even though M is stored more tightly. Tested.
* **The Hessian must be excluded.** `assemble_hessian_tensor` used the same threshold, and it is a
  rank-3 tensor: structural zeros there would store every `(i,j,k)` triple of an element — `nvar³`,
  about 700 k entries per element for 3D Taylor-Hood, against `nvar²` for a matrix. It keeps the raw
  value filter, with a test guarding against it being wired into the policy later.

---

## 4. What this does *not* fix — RESOLVED by Phase 0

The measurements say assembly (424 ms) is bigger than the solve (246 ms) on Benchmark A, so
eliminating the symbolic phase is only half the story. Where does the 424 ms go?

The first draft of this document guessed "mostly the scatter", from the observation that the cost
per raw scatter entry is nearly identical for very different equations (91 ns for 3D NS, 89 ns for
3D C2 Poisson, whose elemental Jacobian is far cheaper per entry). **That guess was wrong.**
`Problem::benchmark_elemental_assembly()` (Phase 0) times the element loop with the scatter removed,
and says:

| case | elemental Jacobian | full assembly | scatter + compression |
|---|---|---|---|
| NS 3D `N=8` | 252 ms | 407 ms | 155 ms (**38 %**) |
| NS 2D `N=60` | 74 ms | 118 ms | 44 ms (38 %) |
| NS+AD 2D `N=40` | 42 ms | 75 ms | 33 ms (44 %) |
| Poisson 3D C2 `N=8` | 63 ms | 73 ms | 10 ms (**13 %**) |

So the elemental evaluation is the *larger* half everywhere, and overwhelmingly so for a cheap
scalar equation. This caps Phase 2: removing the scatter entirely saves ~38 % of assembly, i.e.
~155 ms of a 662 ms Newton step (**~23 %**) on Benchmark A — worth doing, but not more than Phase 1
delivers, and it will do nothing for Poisson-like problems. It also means that **making the
elemental JIT code faster is a bigger lever than anything in this document**, and deserves its own
investigation.

---

## 5. Phased plan

### Phase 0 — instrumentation and a benchmark harness — **DONE**

* `Problem::benchmark_elemental_assembly()` (`src/problem.cpp`, bound as
  `problem._benchmark_elemental_assembly`): runs the element loop with the scatter removed. Chosen
  over adding timers inside the assembly routines because it needs no patch to vendored oomph-lib
  and no flag threaded through five container variants. **Resolved §4.**
* `tests/benchmarks/bench_assembly.py`: parametrised (2D/3D, NS / Poisson / coupled multiphysics,
  mesh size), reports the elemental/scatter split, the cost of structural zeros, and end-to-end
  Newton solves; runs under `mpirun` with `--distribute`, one line per rank.

### Phase 1 — value-independent pattern + solver structure reuse — **DONE**

* `problem.keep_structural_zeros` → makes the assembly's zero threshold negative (§3.6), **for the
  Jacobian only** (§3.7). Works unchanged on the distributed path, which uses the same filter
  (oomph `problem.cc:6899`).
* `problem.keep_structural_zeros_in_mass_matrix` (default off) extends it to the secondary matrices
  of a multi-matrix assembly.
* `problem.jacobian_structure_id`, bumped in `pyoomph::Problem::assign_eqn_numbers` — *and*
  lazily re-validated against the assembly handler, dof counts and active residual, so that a state
  change nobody hooked cannot hand a solver a stale pattern. That belt-and-braces design earned its
  keep: it is what makes fold tracking invalidate correctly, with no change at any of the eight
  handler-installation sites. Returns 0 (= "unusable") whenever `keep_structural_zeros` is off, so
  every consumer switches itself off for existing scripts.
* Pardiso: phase 22 instead of phase 12 when the pattern is unchanged
  (`pyoomph/solvers/pardiso.py`). The pattern is *verified* by comparing the index arrays, not
  merely trusted — and that check immediately caught a real bug: **oomph-lib emits unsorted column
  indices per row**, which `pardisoSolver` sorts in its constructor, so copying the freshly
  assembled values into the sorted value array would have scrambled them.
  `solve_distributed` is left alone: `PardisoSolver.__init__` refuses `nproc > 1` outright, so that
  method is unreachable.
* PETSc: keep the `Mat` (values via `setValuesCSR`) and the `KSP` built on it, so PETSc sees
  `SAME_NONZERO_PATTERN`; serial and distributed. `_force_zero_diagonal` replaces the unconditional
  `MatShift(0.0)`.
* **`MatShift(0.0)` turns out to be a no-op in PETSc 3.22** — verified directly: a 3×3 matrix with a
  missing diagonal keeps `nz_used = 4` across `shift(0.0)`, with or without
  `MAT_FORCE_DIAGONAL_ENTRIES`. So the "manually inserted zeros on the diagonal" this project set
  out to replace *never actually inserted anything*. `keep_structural_zeros` does: PETSc's
  `MatLUFactorSymbolic_SeqAIJ` error "Matrix is missing diagonal entry 0" disappears with it on.
  (Plain `pc_type lu` with PETSc's own factoriser then fails on Taylor-Hood for the honest reason —
  a zero pivot on the pressure block, which needs pivoting; MUMPS handles it. Both failures are
  pre-existing and reproduce identically on `main`.)

### Phase 2 — precomputed CSR pattern and direct scatter — **DONE**

`FrozenSparsity` (one per matrix of a multi-matrix assembly) holds the CSR pattern plus a compact,
slot-sorted scatter map; `assemble_with_frozen_sparsity()` fills the output arrays straight from it,
with no container, no per-row search, no sort and no compression pass. It declines (returning false,
having touched nothing) whenever the route does not fit — distributed runs, augmented systems,
compressed-column output, or any element the code generator cannot describe — so oomph-lib's
assembly remains the fallback.

Measured, assembly only:

| case | container | frozen | | scatter share |
|---|---|---|---|---|
| NS 3D `N=8` | 435 ms | 271 ms | **−38 %** | 176 → 9.6 ms (**18×**) |
| NS 2D `N=60` | 132 ms | 87 ms | **−35 %** | 57 → 12 ms (4.9×) |
| NS+AD 2D `N=40` | 88 ms | 51 ms | **−42 %** | 41 → 4.5 ms (9.2×) |
| Poisson 3D C2 | 75 ms | 67 ms | −11 % | 13 → 4.7 ms (2.8×) |

End-to-end Newton solve (3D Taylor-Hood NS, ndof 10 853, pardiso, identical converged solutions):
3.031 s → 2.381 s with Phases 1+3 → **1.723 s** with Phase 2, i.e. **−43 % cumulative**.

That beats the ≤ −23 % this document predicted from the Phase 0 breakdown, because removing ~95 % of
the scatter compounds with the symbolic-factorisation reuse across every Newton step.

Two things worth recording:

* **A false negative nearly buried this.** The first measurements said the frozen path was 11–16 %
  *slower*. It was never running: `build_frozen_sparsity` bailed out on the Dirichlet-condition
  elements, whose `current_res_jac` is −1 and whose mask was therefore NULL — so every assembly paid
  for a full (two-pass, sorting) pattern build and then fell back. Those elements contribute nothing,
  so the fix is to return an all-zero mask rather than "no description". Lesson: the fast path needs
  a positive signal that it engaged, not just a flag that permits it.
* **The frozen path emits canonically sorted CSR**, where the container assembly emits insertion
  order. Free bonus: `sort_indices()` in the Pardiso reuse check becomes a no-op.

Safety: the compact scatter never visits the positions the pattern omits, so nothing would notice a
mask that under-reported. `verify_frozen_sparsity` (on by default) counts the nonzeros of each
elemental block and of the part actually scattered and refuses to continue if they differ. Counting,
not summing — a magnitude comparison over two different orderings reports spurious mismatches from
rounding alone, which it duly did on the first attempt.

#### Original plan

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

### Phase 2b — distributed pattern and value-only exchange — **DONE**

**Result, measured with the same harness on both sides** (2D cavity `N=40`, ndof 14 162, `--distribute`,
`_assemble_residual_jacobian` timed over 12 assemblies after a warm-up):

| ranks | oomph-lib | frozen | | vs. serial |
|---|---|---|---|---|
| 1 (serial frozen path, for reference) | — | 36.4 ms | | 1.00× |
| 2 | 49.1 ms | **20.7 ms** | −58 % | **1.76×** |
| 4 | 23.3 ms | **12.0 ms** | −49 % | **3.03×** |

Better than the −33 % this section predicted, and it settles the other finding below: distributed
assembly used to be *slower than serial* at two ranks (49.1 vs 36.4 ms — the overhead ate the whole
parallel gain). It now pays for itself from two ranks up, and scales.

**End to end**, re-solving the same converged problem (one assembly plus one MUMPS solve per Newton
step, six repeats, identical solutions to all digits): **−12 % at 2 ranks** (148.8 → 131.2 ms) and
**−41 % at 4 ranks** (274.0 → 161.3 ms). The solve dominates the step at 2 ranks, which is why the
assembly's −57 % shows up as −12 % there; at 4 ranks assembly is a much larger share. These absolute
numbers are on an oversubscribed machine and should be read as ratios at a fixed rank count, not
across rank counts.

The reason it beat the estimate is that freezing does not merely remove the exchange *traffic*, it
removes the owner-side *merge*. oomph-lib merges each incoming entry into the row built so far by
**linearly rescanning that row** — quadratic in the row length, on every assembly, with chunked
reallocation and copying whenever the estimate runs out. Frozen, that whole loop is one precomputed
permutation and a scatter-add. Two further consequences fell out for free:

* the send buffers are **contiguous slices**, so nothing has to be gathered before sending. `my_eqns`
  is sorted and the target distribution gives each rank a contiguous global range, so the rows
  destined for rank *p* are a single run — which is the same property oomph-lib's own
  `first_eqn_element_for_proc` relies on, just never exploited for the payload;
* the final column indices come out **sorted within each row**, where oomph-lib emits them in
  first-seen order. That is what PETSc and the direct solvers want anyway, and it makes the
  Phase 1 index-array comparison in `_can_reuse_structure` stable.

**Verified against oomph-lib directly**, not just at the solution level: `--mode compare-distributed`
in `tests/mpi_structural_worker.py` assembles the same converged state through both routes and
compares the local CSRs as `{column: value}` maps per row (not element-wise — the column *order*
legitimately differs). At 2, 3 and 4 ranks in 2D and 3D: identical nnz, no entry present in one and
absent from the other, worst value difference **3.5e-18** (the two routes sum an entry's
cross-rank contributions in a different order), residuals identical. A solution-level check would
not have been enough — Newton converges to the right answer from a slightly wrong Jacobian, so a
defective merge permutation would have passed.

**What it does not cover**, each falling back to oomph-lib:

* a **replicated problem whose elements are merely split across ranks** (`--distribute` absent but
  `nproc > 1`). oomph-lib re-tunes `First_el_for_assembly` from measured per-element timings as it
  goes, so the element range is *not* a function of the equation numbering and a frozen plan could
  silently describe the wrong slice of the mesh. Deliberately excluded rather than guarded;
* residual-only assembly (`ParallelResidualsHandler`), augmented systems, and any element the code
  generator cannot describe symbolically.

**Everything from the freshness check onwards is collective**, so both things that could diverge
between ranks are put to a vote rather than assumed: whether the plan needs rebuilding (`nrow_local`,
`first_row` and `nelement` are per-rank quantities) and whether the build succeeded. There is a
second, internal vote inside the build, immediately before the first collective — every bail-out up
to that point (a missing symbolic mask, a distribution that is not contiguous by rank) is decided
per rank, and a rank that bailed while the others entered `MPI_Alltoall` would hang them. Half the
ranks in the frozen exchange and half in oomph-lib's is a deadlock, not a wrong answer, which is why
this is voted on rather than checked afterwards. The two extra `MPI_Allreduce`s of one `int` cost
nothing measurable against a 20 ms assembly.

**The two-matrix case works too** — a Jacobian and a mass matrix in one pass, each on its own
pattern (the mass matrix's is about three times tighter, from `contributes_to_mass_matrix`), each
with its own exchange plan and merge permutation. Checked through
`assemble_eigenproblem_matrices()` at 2 and 4 ranks: both matrices reproduce oomph-lib's exactly.
Note that this had to be checked below the Python eigen layer, which cannot handle a distributed
problem at all — `get_J_M_n_and_type()` wraps a *row-local* CSR in a `csr_matrix` of *global* shape
and raises. That is pre-existing and unrelated: it fails identically with the frozen route off.
(Also note that the arrays `assemble_eigenproblem_matrices()` returns are views into
`eigen_{Mass,Jacobian}MatrixPt`, which the next call deletes and reallocates. Comparing two calls
without copying first compares freed memory against live memory and reports pure noise — it briefly
looked like a serious defect in this very code.)

The knob is `problem.use_frozen_distributed_sparsity` (default on) and
`problem._get_distributed_frozen_rebuild_count()` is the positive signal that it engaged — the
lesson from the Phase 2 false negative, where a fast path that never ran was measured as "slower".

The original analysis follows.

#### Original analysis

**Measured first, as for Phase 2.** 2D lid-driven cavity `N=40`, ndof 14 162, `--distribute`, timing
the C++ assembly directly (`_assemble_residual_jacobian`; note `assemble_jacobian` cannot be used on a
distributed problem — it wraps the local block in a scipy matrix of *global* shape and raises):

| ranks | elemental | total assembly | scatter + exchange |
|---|---|---|---|
| 1 (serial, frozen — Phase 2) | 35.0 ms | 37.1 ms | **2.2 ms (6 %)** |
| 2 (oomph `parallel_sparse_assemble`) | 20.0 ms | 39.8 ms | **19.9 ms (50 %)** |
| 4 (oomph `parallel_sparse_assemble`) | 13.4 ms | 24.9 ms | **11.7 ms (47 %)** |

Two things follow. Half of a distributed assembly is scatter and exchange, against 6 % once the
serial path is frozen — so the headroom here is larger than anywhere else left in the plan. And at
two ranks the total is *worse than serial* (39.8 vs 37.1 ms) even though the elemental work has
halved: the overhead eats the entire parallel gain. Distributed assembly does not currently pay for
itself below about four ranks on a problem this size.

**Which half of that 50 % is it?** §8 asked whether the local binary searches and row scans are the
bulk, with the inter-rank exchange secondary. Instrumenting the local stage of
`parallel_sparse_assemble` directly (temporary timer around everything up to "End of vector
assembly") answers it — **the opposite way round**. Per assembly at 2 ranks:

| stage | time |
|---|---|
| elemental evaluation | 20.0 ms |
| local scatter (`my_eqns` bisection + per-row scan + regrow) | **6.7 ms** |
| exchange + owner-side merge | **~12.5 ms** |

So the exchange is about **two thirds** of the overhead, not the local scatter. That reprioritises
Phase 2b and changes what it is worth:

* Freezing the local scatter alone (the direct reuse of Phase 2's machinery, and by far the easier
  half) caps out at roughly **−17 %** of distributed assembly.
* The exchange is the real target. Freezing the pattern makes the column indices, the send plan and
  the owner-side merge permutation all constant, so the per-assembly traffic drops to values plus
  residuals — a third less payload, and the packing/unpacking and allocation around it becomes a
  cached permutation instead of work redone every time.

Both halves are therefore needed for 2b to be worth its risk; the local half on its own is not.
Note also that the diagnostic oomph-lib already has for this
(`enable_doc_imbalance_in_parallel_assembly`, bound as
`problem._enable_doc_imbalance_in_parallel_assembly`) prints through `oomph_info`, which pyoomph
redirects — so it produced nothing here and a direct timer was needed.


* `pyoomph::Problem::parallel_sparse_assemble` override holding a `DistributedJacobianSparsity`
  (§7.2): cached `my_eqns`, per-rank send plan, frozen local column indices and scatter map.
* Per assembly, exchange **values only**; send the column indices once per `structure_id`.
* Falls back to the oomph routine whenever the pattern is invalid, so this is always revertible at
  runtime.

### Phase 3 — Tier B (field-pair pruning) — **DONE**

Result, measured after all four steps below (2D cavity `N=8`, and the same + advection–diffusion):

| | J nnz, filtered | Tier A (connectivity) | **Tier B (field mask)** |
|---|---|---|---|
| cavity | 17 560 (unstable, 80 empty diagonals) | 18 178 | **17 640** |
| coupled | 28 186 (unstable, 80 empty diagonals) | 38 716 | **28 266** |

Tier B is the numerical pattern plus exactly the 80 forced diagonals, stable and with a complete
diagonal. On assembly cost the coupled case goes from **+34.7 % nnz / +16.5 % time** (Tier A) to
**+0.2 % / −0.2 %**, i.e. the structural pattern is now essentially free there too. Mass matrix:
6 050 / 9 950, its own pattern from `contributes_to_mass_matrix`, now value-independent as well —
so §3.7's compromise (tight *or* stable) is gone. Newton solves unchanged: pardiso −21 %,
petsc_mumps −22 %, bit-identical solutions.


Goal: bring the coupled case from 1.35× back to 1.00× nnz (§2.4 measured that the field-pair mask
reproduces the numerical pattern *exactly*), which is what would let `keep_structural_zeros` default
to on. Steps:

1. **The symbolic tables.** ✔ **DONE.** `contributes_to_jacobian[res][field_i][field_j]` already
   existed (`src/jitbridge.h`, emitted in `src/codegen.cpp`). Its mass-matrix counterpart,
   `contributes_to_mass_matrix`, is now emitted alongside it: codegen was *already* computing
   `mass_part = diff(diffpart, __partial_t_mass_matrix)` at `src/codegen.cpp` (the
   `ADD_TO_MASS_MATRIX` emission point) and discarding the fact of its non-zeroness. Recording it
   costs one call, mirroring `mark_jacobian_contribution_for_code`.
   Verified on NS + advection–diffusion: 11 Jacobian field pairs, **3** mass-matrix pairs —
   `(c,c)`, `(velocity_x,velocity_x)`, `(velocity_y,velocity_y)`, i.e. exactly the `∂t` terms, with
   pressure correctly absent. 3/11 = 0.27 against the measured M/J nnz ratio of 0.257.
2. **Per-element local dof → contribution index map.** Follow `get_dof_names()`
   (`src/elements.cpp:5626`, interface variant `:13819`), which already walks
   `eleminfo.nodal_local_eqn` / `pos_local_eqn` and the functable field info to attribute each local
   dof to a field. Cache per `(code, element type)`, not per element. This is the bulk of the work.
3. **Getting the mask into the scatter.** ✔ **DONE.** The per-matrix threshold hook of §3.7 is not
   enough — a mask is per `(i,j)`, not per matrix. The same oomph patch was extended with

   ```cpp
   //FOR PYOOMPH
   virtual const bool* sparsity_mask_for_element(const unsigned& matrix_index,
                                                 GeneralisedElement* const& elem_pt,
                                                 const unsigned& nvar) const { return 0; }
   ```

   fetched **once per element per matrix** (hoisted above the `i`/`j` loops, so no virtual call per
   entry), with the filter becoming

   ```cpp
   if ((mask[m] && mask[m][i * nvar + j]) ||
       std::fabs(value) > numerical_zero_for_sparse_assembly(m))
   ```

   Note the **`||`**, not a replacement: the mask may only ever *add* entries, never remove them. A
   wrong or stale mask then costs storage, not correctness — and a mask that under-reports cannot
   silently truncate the Jacobian, which is the one failure mode that would be a wrong answer rather
   than a slowdown.

   Applied in the `maps`, `vectors_of_pairs`, `two_vectors` and `two_arrays` variants and in
   `parallel_sparse_assemble`. **Not** in `lists`: it filters a second time during compression, when
   merging duplicate column indices, where the elemental `(i,j)` is out of scope — feeding it explicit
   zeros there makes it emit the same column index twice. pyoomph throws if that method is combined
   with pruning rather than silently producing an unusable pattern.

   **Bug found here, worth remembering.** The hook's contract lets the implementation return a pointer
   into a scratch buffer instead of allocating. With one shared buffer that is wrong: a multi-matrix
   assembly fetches the masks for *all* matrices of an element before using any of them, so the
   Jacobian's pointer was left aliasing the mass matrix's (much sparser) mask — the Jacobian silently
   lost its forced diagonals and the whole feature appeared not to work on the eigenproblem path while
   working perfectly on the Newton path (`n_matrix == 1`). One scratch buffer *per matrix index*, and
   the contract in oomph's `problem.h` now says so explicitly.
4. **`force_jacobian_diagonal_entries`** ✔ **DONE**, defaulting to on: the field-pair mask leaves the
   pressure–pressure diagonal empty, which is exactly what PETSc's factorisers reject. It is applied
   inside the mask build, so it costs nothing and only ever adds up to `ndof` entries.
5. Guard: the mask is per *residual/Jacobian combination* (`_solved_residual`,
   `src/problem.hpp:293`) — switching the active residual changes the coupling table and therefore
   the pattern. Already covered: `structure_id` re-validates against `_solved_residual`, with a test.

The mass matrix now gets its own tight structural pattern from `contributes_to_mass_matrix`,
replacing the value filter of §3.7 — so M is value-independent *and* ~3× sparser than J, rather than
having to choose. `keep_structural_zeros_in_mass_matrix` consequently only does anything under
Tier A, where it remains the only way to give M a stable pattern.

**Not yet done:** `keep_structural_zeros` still defaults to **off**. It is now measurably free
(≤ +0.2 % nnz and within noise on time for every benchmarked case, with bit-identical solutions), so
defaulting it on is the natural next step — but it should wait until the tutorial pipeline has run
and until remeshing invalidation and interface-heavy problems are covered, since flipping it also
turns on solver symbolic reuse for every user.

### Phase 4 — eigenproblems and bifurcation tracking — **multi-assembly DONE**

`sparse_assemble_row_or_column_compressed_base_problem` — the routine behind
`assemble_multiassembly`, and so behind every Python-level bifurcation tracker — now has a frozen
fast path. It was the worst assembly in the codebase: a `std::map` per row, with 30–86 % of its time
outside the elemental evaluation (the range is wide because `_benchmark_elemental_assembly` measures a
single-matrix Jacobian while this pass evaluates five quantities per element).

The key structural fact is that **all of its matrices share one pattern**: the Jacobian, its parameter
derivative and a Hessian-vector product are all derivatives of the same residual w.r.t. the dofs, so
one frozen pattern serves all of them and the scatter writes the same slots into several value arrays.
The mask is taken from matrix 0 for every matrix — exact for the Jacobian-derived ones, a superset for
any mass-matrix-derived one, which is the safe direction and is checked per element anyway.

Bratu fold tracking, `R + J + dRdp + dJdp + dJdU` in one pass:

| | map-based | frozen | |
|---|---|---|---|
| multi-assembly, `N=30` (ndof 6963) | 60.2 ms | 42.0 ms | **−30 %** |
| multi-assembly, `N=50` (ndof 19603) | 182.1 ms | 119.4 ms | **−34 %** |
| full fold solve, `N=40` | 0.763 s | 0.546 s | **−28 %** |

with the tracked fold identical to 10 digits.

**A false alarm worth recording, because it will recur.** The first comparison said the
Hessian-vector product was exactly *negated* by the fast path — `max|A + B| = 3.5e-17`, all 1521
entries wrong. It was the test, not the code: it contracted the Hessian with
`get_real_eigenvector_guess()`, and an eigenvector is only defined up to sign, so the two runs were
handed opposite vectors. The product is linear in that vector, hence the exact negation, while every
quantity not depending on it agreed to round-off. Anything comparing Hessian-vector products across
two runs must contract with a **fixed** vector.

Still open in Phase 4: whether a values-only `J − σM` update is worth a dedicated path for
shift-and-invert sweeps.

### Phase 4 (original plan) — eigenproblems and bifurcation tracking

Only after Phases 1–3 are validated on the plain Newton path.

* **Eigenproblems.** `assemble_eigenproblem_matrices` (`src/problem.cpp:1091`) →
  `get_eigenproblem_matrices` (oomph `problem.cc:8641`) → same sparse-assembly routine with
  `n_matrix = 2`. **Do not give M the Jacobian's pattern** — §3.7 measured the cost (3× inflation)
  and the per-matrix threshold now prevents it. The `J − σM` operator still gets a stable,
  value-independent pattern for free, because M's entries lie inside J's structural pattern, so
  SLEPc/ARPACK shift-invert needs only one symbolic factorisation for a whole parameter sweep.
  What remains open is whether the *values-only* `J − σM` update is worth a dedicated path
  (`keep_structural_zeros_in_mass_matrix = True` would make it a pure `axpy` on the value arrays, at
  the price of the 3× inflation — measure both).
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

### Phase 5 — backend reuse contract *(new, from the §7d review)*

Ordered by value:

1. **Multi-pattern cache** (§7d A5). Replace the single `FrozenSparsity` per matrix index with a
   small keyed cache on `(structure_id, matrix_index)`. Without it, any setup that alternates between
   two residuals — a PETSc preconditioner matrix built by `assemble_matrix`, a multi-residual problem
   — rebuilds the pattern on every assembly and is *slower* than no cache at all. This is a
   correctness-of-claim issue, not just an optimisation.
2. **Per-solver diagonal requirement** (§7b). `requires_explicit_diagonal()` on the solver, answered
   by `PETSCSolver` from the configured factorisation, replacing the global flag.
3. **Mac Accelerate symbolic reuse** (§7d A4). `refactorize()` and a cached CSR already exist; hook
   them to `jacobian_structure_id` as Pardiso is hooked. Needs a Mac to test.
4. **Compose the Pardiso tiers** (§7d A3). Benchmark symbolic reuse together with
   `try_to_reuse_solver`'s numeric-as-preconditioner mode on a time-stepping / continuation workload,
   where the Jacobian changes slowly.
5. `assemble_matrix` should drop `shift(0.0)` (§7d A6 — it does nothing) and route through the same
   diagonal policy as the main path.

### Out of scope (for now)

* Threaded/openmp assembly. A precomputed scatter index makes a colour- or lock-free parallel
  scatter feasible, but that is a follow-up.
* Making bifurcation tracking / `assemble_multiassembly` work under MPI at all — pyoomph's
  `..._base_problem` throws for `nproc > 1` today (`src/problem.cpp:2822-2825`). That is a
  pre-existing gap, not one this work introduces; the plan must not make it worse, and §7.4 records
  what would be needed.

---

## 5b. Results of Phases 0–1

### What structural zeros cost at assembly time

| case | ndof | nnz | Δnnz | Δ assembly time |
|---|---|---|---|---|
| NS 3D `N=8` | 10 853 | 1 800 539 → 1 816 855 | +0.9 % | +0.2 % |
| NS 2D `N=60` | 32 042 | 1 240 184 → 1 272 938 | +2.6 % | +1.7 % |
| Poisson 3D C2 `N=8` | 4 624 | 253 500 → 253 500 | **0 %** | −0.2 % |
| NS+AD 2D `N=40` | 20 642 | 840 730 → 1 132 540 | **+34.7 %** | **+16.5 %** |

Free for single-physics (Poisson's elemental block is already dense, so the structural pattern *is*
the numerical one), and expensive only for weakly coupled multiphysics — which is exactly the case
Phase 3's field-pair pruning is for.

**Caveat found here, not predicted — and since fixed.** The first cut used one problem-wide
threshold, so the mass matrix was given the Jacobian's pattern too and `res+jac+mass` degraded far
more than `res+jac`: for NS 3D, 469 → 574 ms (+22 %) against +0.2 % for the Jacobian alone. §3.7 is
the fix (a per-matrix threshold). After it:

| case | `res+jac+mass` filtered | structural | |
|---|---|---|---|
| NS 3D `N=8` | 474 ms | 480 ms | +1.2 % (was +22 %) |
| NS+AD 2D `N=40` | 96.6 ms | 112.1 ms | +16 % (was +50 %) |

The residual +16 % on the coupled case is the Jacobian's own connectivity blow-up (+34.7 % nnz), not
the mass matrix — that is what Phase 3's field-pair pruning is for.

### End-to-end Newton solves

3D Taylor–Hood NS, `N=8`, ndof 10 853, converged to `|R| = 7.6e-15`, solutions bit-identical:

| solver | filtered | structural | |
|---|---|---|---|
| `pardiso` | 2.941 s | 2.347 s | **−20 %** |
| `petsc_mumps` | 3.758 s | 2.923 s | **−22 %** |

MPI, 2D lid-driven cavity `N=24` (ndof 5042), `petsc_mumps`, `--distribute`. Every rank reported the
same `|R|` and the same integral observables (`ke`, `vx`) to 12 significant digits in all
configurations, and the same `jacobian_structure_id`:

| ranks | filtered | structural | |
|---|---|---|---|
| 1 | 0.328 s | 0.259 s | −21 % |
| 2 | 0.247 s | 0.195 s | −21 % |
| 4 | 0.320 s | 0.155 s | −52 % |

3D distributed (`N=6`, ndof 4335) also agrees to 12 digits across ranks, with smaller gains
(−1 % at `-n 2`, −14 % at `-n 4`): a single solve from scratch is only ~3 Newton steps, and the
first one cannot reuse anything.

Note `-n 1 --distribute` still takes the *serial* solver path — `get_jacobian` branches on
`Communicator_pt->nproc() == 1`, so `solve_distributed` is only exercised from two ranks up.

### Whole-pipeline comparison against `main`

The numbers above are microbenchmarks of assembly and of a single Newton solve. To see what the work
is worth on a broad, realistic mix, the tutorial suite was run on both branches:
`citools/test_all_tutorial_scripts.py --quick-test --no-petsc`, 126 scripts, each branch built from
scratch and run twice (the first run warms the JIT cache, the second is the measurement).

| | `main` (d03a562) | `structural_assembly` (7e40bf7) | |
|---|---|---|---|
| Wall clock | 4:46.35 (286.4 s) | 4:21.36 (261.4 s) | **−8.7 %** |
| User CPU | 312.2 s | 280.9 s | **−10.0 %** |
| System CPU | 52.2 s | 50.0 s | −4.3 % |
| Peak RSS | 812 992 kB | 810 836 kB | −0.3 % |

Both pass all 126 scripts. The warm-up runs agree on the direction (293.4 s vs 275.5 s).

**Read that −8.7 % in context, in both directions.** `--quick-test` stops each script after its first
successful Newton solve, so the suite is deliberately dominated by JIT-compiling the generated element
code, process startup, mesh generation and output writing — assembly and linear solves are a small
slice of it. A ~9 % wall-clock gain on *that* workload is therefore a lower bound on what a real
production run sees; the −43 % measured on an actual Newton solve is the upper end. Where a given
simulation lands between the two depends entirely on how much of its time is spent solving.

**Peak memory is unchanged**, which was the open question: the frozen scatter map costs about 8 bytes
per raw elemental entry (two ints per stored position), and at tutorial problem sizes that is lost in
the noise. It is not free at scale — for the 3D Taylor-Hood benchmark (2.7 M raw entries) it is
~21 MB per matrix — so it remains worth watching on genuinely large 3D problems, where
`use_frozen_sparsity=False` is the escape hatch.

### Not yet done from Phase 1 — **both since closed**

* `problem.force_jacobian_diagonal_entries` was deferred as a no-op knob. It is now a tri-state
  (`on` / `off` / `auto`) defaulting to **off**, with `auto` answered per solve by the solver itself
  through `requires_explicit_diagonal()` — see §7b.
* Invalidation after **remeshing** had no test. It has one now (`test_structural_assembly.py`).

---

## 6. Correctness gates

Implemented as `tests/test_structural_assembly.py` (13 tests, <4 s, in the fast run).
Every phase must keep these green before it lands:

1. **Superset invariant.** `structural_pattern ⊇ numerical_pattern` at several dof states (U=0,
   converged), and equal to the element-connectivity pattern recomputed independently in Python —
   so the test does not merely compare the implementation against itself.
   ✔ `test_structural_pattern_is_value_independent`, `..._is_a_superset_with_identical_values`.
2. **Bit-identical solutions.** Converged dof vectors must match with the pattern route enabled and
   disabled; explicit zeros must not change any result. ✔ `test_newton_solution_is_unchanged`,
   `test_arclength_continuation_is_unchanged_and_keeps_the_pattern`.
3. **Differential test against the oomph path.** Same values on the common pattern, extra entries
   all exactly 0.0. ✔ `test_structural_pattern_is_a_superset_with_identical_values`.
4. **Invalidation coverage.** After each of: mesh adaptation, remeshing, `pin`/`unpin`, Dirichlet
   changes, switching `_solved_residual`, installing/removing a bifurcation handler, adding
   augmented dofs — `structure_id` must have changed and the rebuilt pattern must still satisfy (1).
   Conversely it must *not* change across Newton steps or arclength continuation steps, or the reuse
   never fires. This is the highest-risk area: a *missed* invalidation is a silent wrong-answer bug,
   not a crash. ✔ `test_structure_id_*` (6 tests), covering renumbering, augmentation by fold
   tracking, and switching the active residual. Still uncovered: remeshing.
5. **Existing suites.** ✔ `tests/` passes: 536 passed / 314 skipped / 3 pre-existing xfails (the
   documented curved-boundary over-marking), and `--full` is green.
   ✔ **Tutorial pipeline**, `citools/test_all_tutorial_scripts.py --quick-test --no-petsc`, run
   twice: once with the shipped default (structural pattern off — the regression gate for the
   oomph-lib patch, whose new filter and mask-fetch lines execute on every assembly regardless), and
   once with `keep_structural_zeros` forced on for every problem via a `sitecustomize.py` on
   `PYTHONPATH`. **All 126 scripts pass both ways.** 119 of the 126 logs verifiably report
   `keep_structural_zeros=True prune=True forcediag=True`; the other 7 are scripts that never
   initialise a `Problem` (material-definition examples) or that only launch further subprocesses.
   This is the broadest evidence available that the feature is safe on real problems — moving
   meshes, interfaces, eigenproblems, bifurcation tracking and continuation all appear in that set.
7. **Per-matrix policy.** The mass matrix must keep its own pattern, its values must be unaffected by
   the Jacobian's policy, its entries must lie inside the Jacobian's structural pattern, and the
   Hessian tensor must not be dragged in (§3.7). ✔ `test_mass_matrix_*`,
   `test_hessian_tensor_is_not_given_structural_zeros`.
6. **MPI gates** — every one of 1–5 also under `mpirun -n 2` and `-n 4` with `--distribute`, plus:
   * cross-rank agreement and serial agreement on the gathered residual / global observables, the
     two oracles `tests/test_mpi_adaptivity.py` already implements (see its header);
     ✔ `tests/test_mpi_structural_assembly.py` (+ `tests/mpi_structural_worker.py`): 4 tests, 2D and
     3D at `-n 2` and `-n 4`, each checking cross-rank agreement, agreement between the filtered and
     structural runs, and agreement with an in-process serial reference. The worker solves *twice*
     on purpose — the first solve can never reuse a factorisation, so a single-solve test would pass
     with the reuse path dead.
   * `jacobian_structure_id` identical on all ranks after every renumbering.
     ✔ asserted with `MPI_Allreduce(MIN/MAX)` under `PARANOID` in
     `pyoomph::Problem::assign_eqn_numbers`. Deliberately *not* in `get_jacobian_structure_id()`:
     that one is called from Python and may legitimately be read on a single rank, where a
     collective call would be the very deadlock it is meant to prevent;
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

## 7c. `test_moving_mesh_distributed` — **RESOLVED**

One test, one case (`ale-tri_crossed-12-level`). Root cause found, fixed, and the whole MPI suite is
green (67 passed).

### What it was

`keep_structural_zeros` turns a **structurally absent** diagonal into a diagonal entry that is
**present and exactly zero**. On this case that happens for 158 rows, and they are not arbitrary:

| dofs gaining a stored-zero diagonal | count |
| --- | --- |
| `<added interface dof>` — Lagrange multipliers of the interface coupling | 124 |
| `pressure__C1__*` | 35 |

Both are **saddle-point constraint rows** whose self-coupling is zero *by construction*, not by
coincidence at this state. So the field-coupling mask over-reports a coupling that can never be
nonzero.

MUMPS chooses its elimination order from the **structure**, in the analysis phase, before it sees a
single value. Those stored zeros invite it to plan an elimination that then hits a zero pivot at
factorisation time — `INFOG(28)` reports **8 null pivots encountered**. With null-pivot detection off
(`ICNTL(24) = 0`, the MUMPS default) it divides by them and returns a silently wrong solution. Newton
then *diverges from a nearly converged state*: 3.6e-07 → 2e-05 → 0.03 → 4350 → 1e18, which is the
give-away — a correct Jacobian cannot amplify like that from 1e-7.

### The fix

`PETSCSolver.use_mumps()` now sets `ICNTL(24) = 1` by default (and the SLEPc spectral transform does
the same). It costs nothing on a matrix with no null pivots and only ever replaces a garbage answer
with a usable one. Set through `_SetDefaultPetscOption`, so anyone who configures it explicitly keeps
control. Flip side worth knowing: a genuinely singular system now yields a pseudo-solution, so it
shows up as a Newton that fails to converge rather than one that explodes.

### Why it took so long, and what to do differently

Every earlier attribution was wrong, and the reason is worth recording:

* the failure is a **knife-edge**, and whether MUMPS's ordering reaches a null pivot shifts with tiny
  perturbations. That made it look nondeterministic and produced apparent *batches* of passes and
  failures;
* one such perturbation is absurd but real: putting an **empty directory on `PYTHONPATH`** flipped
  the result reproducibly, 10 out of 10 interleaved runs. Not because pyoomph reads `PYTHONPATH` —
  it does not — but because `problem.py` logs `str(sys.path)`, so a longer path changes an allocation
  and shifts the heap. Setting an unrelated variable of the same length (`FOOBAR=...`) did *not* flip
  it, which is what ruled out "environment size" and pointed at the log line;
* consequently **no single run means anything**, which is exactly how the two retracted explanations
  came about. I nearly recorded a third ("it's `PYTHONPATH`") before interleaving disproved it.

What finally worked was refusing to bisect until there was a *deterministic* failing configuration,
and then bisecting the feature flags inside it, injecting them through the case module rather than a
`sitecustomize` on `PYTHONPATH` — because the harness itself was one of the perturbations. That gave
`keep_structural_zeros` = fail 3/3, off = pass 3/3, and the frozen-assembly flags irrelevant, in one
clean table. From there the diagnosis is three measurements: the two matrices are identical
(all 8 235 / 11 260 extra entries exactly `0.0`), the diagonal census, and `INFOG(28)`.

### Loose end

The mask **over-reports**: a correct field-coupling analysis would not add `(λ, λ)` or `(p, p)` for
these rows at all, since those blocks are structurally absent. §3's safety argument — "a mask that
over-reports merely stores explicit zeros" — is now known to be too optimistic: over-reporting on the
diagonal of a saddle-point system is what created this. Tightening it would also cut nnz. Worth doing,
but it is an optimisation now rather than a correctness fix.

## 7b. The forced diagonal belongs to the solver — **DONE**

Needing a stored diagonal is a property of the *factorisation*, not of the problem, so
`problem.force_jacobian_diagonal_entries` was the wrong owner. It is now decided by the active linear
solver, with the flag surviving as an override:

* `GenericLinearSystemSolver.requires_explicit_diagonal()` — default `False`.
* `PETSCSolver` answers it **from the options database**, not from its class. Every `*pc_type` option
  is scanned (so a factorisation under a fieldsplit is seen), and a
  `*pc_factor_mat_solver_type` naming an external package (MUMPS, SuperLU, PaStiX, CHOLMOD, UMFPACK,
  MKL Pardiso, STRUMPACK…) answers `False`, because those build their own structure from the CSR.
  PETSc's own LU/ILU/Cholesky/ICC answers `True`. Deciding from the options rather than the class is
  what makes `petsc_mumps` and a hand-configured `-pc_type lu -pc_factor_mat_solver_type mumps` agree.
* Pardiso, SuperLU and scipy inherit the `False` default.
* `Problem._sync_diagonal_requirement_from_solver()`, called from `actions_before_newton_solve`, pushes
  the answer down before each solve — re-asked every time, because PETSc options can change at any
  point. Changing the answer invalidates the pattern, so a solver that *starts* needing a diagonal
  cannot keep being handed one assembled without it.
* The flag is now tri-state: reading it gives the effective value, assigning overrides the solver and
  stops it being consulted, and `_set_force_jacobian_diagonal_entries_auto()` hands control back.

Verified, each in a clean process:

| solver | `requires_explicit_diagonal()` | rows without a diagonal | outcome |
|---|---|---|---|
| `pardiso` | False | 48 | converged |
| `petsc_mumps` | False | 48 | converged |
| `petsc` (PETSc's own LU) | **True** | **0** | fails on a zero pivot |

The last row is the point. PETSc's LU used to reject the matrix outright with
`MatLUFactorSymbolic_SeqAIJ … Matrix is missing diagonal entry 0`; that message is now **gone** — the
structural obstacle is removed and the diagonal arrives from the assembly. What remains is the
pre-existing numerical limitation of non-pivoting LU on a Taylor-Hood saddle-point system, which
reproduces identically on `main`. Erring towards `False` is deliberate: an unnecessary `True` costs a
stored zero on every diagonal and perturbs pivoting, whereas a wrong `False` surfaces as PETSc's own
explicit complaint, which the user answers with `force_jacobian_diagonal_entries = True`.

**Caveat worth knowing.** PETSc's options database is global and sticky within a process, so a solver
configured earlier changes what a later one reports. Constructing `petsc_mumps` and then `petsc` in one
process leaves `pc_factor_mat_solver_type=mumps` in the database, and the second solver then both
reports `False` *and* actually runs MUMPS. That is PETSc's design, not something this layer can fix;
it is why the table above was measured one solver per process.

`PETSCSolver._force_zero_diagonal()` is now a documented no-op: the diagonal comes from the assembly,
and the `MatShift(0.0)` it used to call never inserted anything (§7d A6).

---

## 7d. Review of the reuse contract with each backend

Six concerns raised in review, investigated. Two are real gaps that change the plan (**A5**, **A4**);
the rest are answered.

### A1. Does any backend use the CSR index arrays as scratch, destroying the pattern?

**No evidence of it, and the design fails safe either way.**

* **PETSc.** `createAIJ(csr=...)` *copies* into PETSc's own AIJ storage, so nothing we hold is aliased.
  The zero-copy alternative, `MatCreateSeqAIJWithArrays`, *would* alias — it was tried before and
  abandoned because it silently broke hypre/BoomerAMG (see the comment at
  `pyoomph/solvers/petsc.py`). It must stay abandoned: with a frozen pattern the aliasing hazard is
  worse, not better, since the arrays now outlive a single solve.
* **MKL Pardiso.** `a`, `ia`, `ja` are documented inputs; MKL keeps its permutations in its own
  storage. pyoomph additionally hands Pardiso a private `.copy()`.
* **oomph-lib.** `CRDoubleMatrix::sort_entries()` would reorder in place, but nothing in pyoomph or
  oomph calls it — checked.

The important structural point is that this does not rest on the audit being exhaustive: the Pardiso
reuse path *verifies* `ia`/`ja` by comparison before reusing (§5b), so anything that did modify them
produces a full refactorisation, not a wrong answer. Any future backend reuse must keep that property.

**And for MPIAIJ (the distributed path)?** Same answer, more strongly — tested on 2 ranks by
scribbling over the input arrays after `createAIJ` and re-reading the matrix:

```
rank 0 aliased=False (diag [1 1 1] -> [1 1 1])  stored_cols=[0 1 2]
rank 1 aliased=False (diag [2 2 2] -> [2 2 2])  stored_cols=[3 4 5]
```

It cannot be otherwise: an MPIAIJ matrix splits the *global* column indices it is handed into
diagonal and off-diagonal blocks with its own local numbering, so it must transform them, and a
transform implies a copy. `MatMPIAIJSetPreallocationCSR` documents the arrays as not retained.

Two further MPIAIJ-specific points that had to be checked because the reuse path depends on them:

* **`setValuesCSR` on MPIAIJ is ownership-range relative**, not global-row-indexed. Handing it an `I`
  array of length `nrow_local + 1` writes *this rank's* rows, which is what
  `PETSCSolver.solve_distributed` does. Verified on 2 ranks: rank 1's update landed on global rows
  3-5 and left rank 0's values untouched. Had it been global-row-indexed, every rank above 0 would
  have silently written into rank 0's rows via the stash — a wrong answer, not a crash, so this was
  worth confirming rather than assuming.
* **Latent dtype inconsistency, pre-existing — fixed.** The distributed *creation* call passed the raw
  arrays (`csr=(row_start, col_index, values)`) while the serial one converted with
  `.astype(PETSc.IntType, copy=False)`, so on a 64-bit-index (or complex-scalar) PETSc build the
  distributed path would have handed int32/float64 where int64/complex was expected. Phase 1 added the
  conversion to the update call, which left creation and update disagreeing with each other.
  `assemble_preconditioner()` had the same gap. Both now convert; every `csr=` in the file does.
  On a matching build each conversion is a no-op returning the same view, so this costs nothing here —
  it only shows up on a PETSc configured differently from this machine's, which is exactly why it had
  survived.

### A2. Is MUMPS/PETSc symbolic reuse actually happening? — **verified**

Yes. With `-info`, a Mat kept across two solves with only its values changed emits
`MatLUFactorSymbolic_SeqAIJ` **once**, not twice: PETSc compares the matrix's nonzero state, sees
`SAME_NONZERO_PATTERN`, and calls only `MatLUFactorNumeric`. This is what the measured op2 drop
(294 → 134 ms) was; it is now confirmed rather than inferred.

### A3. Pardiso's iterative resolve mode

`PardisoSolver.try_to_reuse_solver` (off by default) sets `iparm[3] = 63` — MKL's preconditioned
CGS with the *previous numerical* factorisation as preconditioner, tolerance 1e-6 — plus
`iparm[7]` iterative-refinement steps, and falls back to a fresh factorisation when it stalls or the
residual check fails. That is a **third, independent tier** of reuse:

| tier | reuses | validity |
|---|---|---|
| symbolic (Phase 1) | fill-reducing ordering, elimination tree | always, while the pattern holds |
| numeric-as-preconditioner (`try_to_reuse_solver`) | the LU factors themselves | only while the values stay close; checked a posteriori |

**Composed and measured — and the answer is that the numeric tier is not worth having.**

The two are now composed: when a numeric reuse stalls, the fallback is a phase-22 refactorisation
rather than a full rebuild, because a failed *numerical* reuse says nothing about the sparsity. That
makes `try_to_reuse_solver` strictly better than it was. It does not make it a good idea.

Six time steps of 3D Taylor-Hood NS (the slowly-varying-Jacobian case the numeric tier is for),
observables identical to 10 digits throughout:

| | `N=6` | `N=9` | full (ph12) | numeric (ph22) | CGS reuse |
|---|---|---|---|---|---|
| no reuse at all | 1.804 s | 7.652 s | 10 | 0 | 0 |
| **symbolic only** | **1.191 s** | **5.19–5.28 s** | 2 | 8 | 0 |
| both tiers | 1.191 s | 5.85–5.91 s | 1 | 1–2 | 9 |
| both, pre-composition fallback | — | 5.888 s | 2 | 0 | 9 |

Symbolic reuse alone is worth **−32 %**. Adding the numeric tier is a wash at `N=6` and a
**+13 % loss** at `N=9`, reproduced three times: once the problem is big enough, up to 30 CGS
iterations plus a sparse mat-vec for the residual check cost more than simply redoing the numbers.
The composition recovers a little of that (one full factorisation instead of two) but cannot rescue it.

So `try_to_reuse_solver` stays **off by default**, and the recommendation is to leave it off — its
premise, that reusing numerical factors beats recomputing them, stopped holding once phase 11 was no
longer part of "recomputing them". Phase 1 made the cheap thing cheap enough that the clever thing
lost.

### A4. scipy and Mac Accelerate backends

* **scipy** (`pyoomph/solvers/scipy.py`): `splu` exposes no symbolic-reuse API, so there is nothing to
  reuse at the factorisation level. It still benefits from the frozen *assembly*, and from the frozen
  path emitting canonically sorted CSR. Low priority.
* **Mac Accelerate** (`src/mac_accelerate.{hpp,cpp}`): **done and verified on real hardware.**
  `MacAccelerateSparseSolver::refactorize_values_only()` keeps the symbolic factorization and calls
  `SparseRefactor`, refilling the values through a cached CSR→CSC permutation;
  `MacAccelerateLinearSolver` calls it while `jacobian_structure_id` is unchanged, and it verifies the
  index arrays before acting, exactly as the Pardiso path does. A failed `SparseRefactor` releases the
  factorization and returns false rather than throwing, so the caller falls back to a full
  factorization — reusing a symbolic factorization fixes the pivoting, and new values may not tolerate
  it where a fresh one would have chosen differently.

  **None of this could be compiled here.** It is inside `#ifdef __APPLE__`, so a Linux build does not
  even syntax-check it. `.github/workflows/test_mac_accelerate.yml` (manual dispatch, `macos-14`,
  arm64) builds pyoomph and runs `citools/check_accelerate_reuse.py`, which checks three things: that
  the answers match an independent solver, that the reuse *actually happened*
  (`num_symbolic_factorizations()` stops growing while `num_numeric_refactorizations()` climbs — not
  timings, which is how the Phase 2 false negative survived so long), and that renumbering forces a
  new symbolic factorization. GitHub only shows the dispatch button for workflows on the default
  branch, so the file has to reach `main` before it can be run.

  **Result of run `30524219210`** (branch `structural_assembly`, macos-14/arm64) — all checks passed:

  | check | outcome |
  | --- | --- |
  | reuse agrees with no-reuse | `max\|du\| = 0.0` — bit-identical |
  | reuse agrees with an independent superlu solve | `max\|du\| = 5.6e-17` |
  | reuse on: symbolic / numeric-only factorizations | **1 / 2** |
  | reuse off: symbolic / numeric-only factorizations | **3 / 0** |
  | re-solve on an unchanged pattern | no new symbolic (1 → 1), 4 numeric-only |
  | after renumbering | a new symbolic is taken (1 → 2) |

  So the saving is real (2 of 3 factorizations became numeric-only) *and* it is given up exactly when
  it must be. The one thing this does not measure is wall-clock: `SparseRefactor` is assumed cheaper
  than `SparseFactor`, and a shared CI runner is too noisy to prove it. That is the same assumption the
  Pardiso path makes, where it was measured locally and held.

### A5. A preconditioner matrix would thrash the single frozen pattern — **real gap**

`PETSCSolver.assemble_matrix(which_one)` builds a second matrix from a *different residual* for
preconditioner construction. Switching the active residual changes the field couplings and the
pinning, so `get_jacobian_structure_id()` correctly reports a different pattern — but
`Problem::frozen_sparsity` holds **one** pattern per matrix index and rebuilds whenever the generation
changes. Alternating between the Jacobian and a preconditioner matrix therefore rebuilds the pattern
on *every* assembly, which is strictly worse than not caching at all (a rebuild is two passes over the
mesh plus a sort).

Nothing is corrupted — the handed-out arrays are copies — but the performance claim collapses for
exactly the setups that most need a good preconditioner.

**Fixed**, and the fix turned out to need two halves — the second only came to light when asked
whether the cache also covers mass matrices assembled through the eigenproblem assembly:

1. **A keyed cache.** `Problem::frozen_sparsity_cache` now holds several `FrozenSparsity` entries
   keyed on `(pattern id, matrix index)`, LRU-evicted, capacity 8 (raised automatically if a single
   assembly needs more, and never evicting a slot the current assembly is using).
2. **A pattern id that is a *function* of the configuration, not a counter.** This was the real
   blocker. `get_jacobian_structure_id()` used to increment whenever the watched state differed from a
   snapshot, so alternating between two configurations produced ids 5, 6, 7, 8 … that never repeated —
   *no* cache could hit, whatever its capacity, and neither could Pardiso's symbolic factorisation or
   a retained PETSc `Mat`. The id is now looked up in a map keyed on
   `(assembly-handler type, ndof, n_unaugmented_dofs, active residual)`, so returning to a
   configuration returns the id it had before. Ids are still never recycled across an invalidation:
   the map is cleared but the counter keeps climbing, so an id a solver held from before a renumbering
   cannot accidentally match afterwards.

   Keyed on the handler's **type**, not its address: the eigenproblem assembly `new`s and deletes its
   `EigenProblemHandler` on every call, so an address key missed every time. That cost a measurement
   to notice — the address key took the alternation benchmark from 12 rebuilds to 9, not to 3.

Measured on four Newton/eigenproblem alternations, which involve exactly three distinct patterns
(Newton's Jacobian, the eigenproblem's Jacobian, the mass matrix):

| | pattern rebuilds |
|---|---|
| counter id, single slot per matrix index | 12 |
| stable id keyed on handler *address* | 9 |
| **stable id keyed on handler *type*, keyed cache** | **3** (the floor) |

So yes — mass matrices assembled through `assemble_eigenproblem_matrices` are cached, and now survive
alternation with the Newton assembly rather than being rebuilt each time.

Still to do: `assemble_matrix` should stop calling `shift(0.0)` (A6 — it does nothing) and route
through the same `_force_zero_diagonal` policy as the main path.

### A6. Is `shift(0.0)` a no-op *together with* `NEW_NONZERO_ALLOCATION_ERR = False`? — **still a no-op**

Tested all four combinations on a 3x3 matrix with a deliberately missing diagonal, checking the actual
stored column indices and whether PETSc's LU accepts the matrix — not just `nz_used`, which was the
weaker evidence used the first time:

| `NEW_NONZERO_ALLOCATION_ERR` | `shift(0.0)` | row 0 diagonal stored | PETSc LU |
|---|---|---|---|
| False | False | no | "Matrix is missing diagonal entry 0" |
| False | True | no | same |
| True | False | no | same |
| **True** | **True** | **no** | **same** |

So the option does not rescue it: the two together still insert nothing. `MatShift` with a zero shift
returns early regardless of allocation policy. The conclusion of §5b stands, now on stronger evidence.

---

## 7e. Freezing the bifurcation trackers

**Status: infrastructure and the fold spec implemented, OFF BY DEFAULT
(`problem._use_frozen_sparsity_for_bifurcation_tracking`). One unexplained failure, below.**

### What the first attempt showed

`AugmentedBlockSpec` + the `AugmentedSparsityProvider` no-op mixin are in `src/bifurcation.hpp`,
`MyFoldHandler::get_sparsity_pattern` describes the fold layout, and
`Problem::augmented_sparsity_mask_for_element` tiles the raw masks into the augmented one. It works:
on a PDE fold tracking (Kuramoto-Sivashinsky, ndof 15 229 augmented) the frozen pattern is built
(`_get_frozen_sparsity_nnz` 0 -> 735 462) and the augmented assembly drops from **114.6 ms to 63.1 ms
(-45 %)**, finding the same fold point to 12 digits.

**The cost, and where it went.** The first version stored **549 693 -> 735 462 nonzeros, +34 %** against
the value-filtered pattern. Breaking that down by block showed the augmented *structure* costs
essentially nothing -- `base x base`, `eig x eig` and `param x eig` were bit-identical -- and that
**96 % of the excess sat in one block**, `eig x base`, the Hessian-vector product:

| block | value-filtered | Jacobian-patterned | with `contributes_to_hessian` |
| --- | --- | --- | --- |
| base x base | 237 540 | 237 540 | 237 540 |
| **eig x base** | **59 385** | **237 540** | **59 385** |
| eig x eig | 237 540 | 237 540 | 237 540 |
| base/eig x param | 3 807 each | 7 614 each | 7 614 each |
| **total** | **549 693** | 735 462 (+34 %) | **557 307 (+1.4 %)** |

Declaring Hessian blocks `Jacobian`-patterned is *true but loose*: `d2R_i/du_j du_k != 0` implies
`dR_i/du_j` is not identically zero, so it is a subset -- but every LINEAR term of the residual is in
J and in no Hessian at all. On Kuramoto-Sivashinsky the biharmonic and Laplacian carry most of J and
none of the Hessian, so the contracted block is **four times sparser**.

The fix is the move Phase 3 already made for the mass matrix: codegen now emits
`contributes_to_hessian[res][fi][fj]` alongside the other two, marked where the second derivative is
generated (both `(res,f)` and `(res,f2)`, since `d2R/df df2` is symmetric and a contraction sums over
one of the two indices). `AugmentedBlockSpec` gains `Hessian`/`HessianT`, and the fold's `eig x base`
block uses it. That leaves the augmented frozen pattern within **1.4 %** of the numerically exact one,
all of it the two border columns, which are dense per element and not worth chasing.

**Measured, with the tighter table:** augmented assembly **-45 %** (113.3 -> 62.5 ms), a re-solve
including the factorisation **-31 %** (171.6 -> 118.1 ms), and a full fold-tracking solve **-24 %**
(0.70 -> 0.53 s), with the same fold point to 12 digits.

### Hopf

`MyHopfHandler::get_sparsity_pattern` describes the five-group layout
`[ base | Phi | Psi | parameter | Omega ]`:

|  | base | Phi | Psi | param | Omega |
| --- | --- | --- | --- | --- | --- |
| **base rows** | J | - | - | dense | - |
| **Phi rows** | H | J | M | dense | dense |
| **Psi rows** | H | M | J | dense | dense |
| **param row** | - | dense | - | - | - |
| **Omega row** | - | - | dense | - | - |

Both eigenvector-row base blocks are Hessians (`dJdU_Eig` and `dMdU_Eig`); `contributes_to_hessian`
covers the mass-matrix one too, because it is marked whenever *either* the Jacobian or the mass part
of the second derivative is non-zero. Neither scalar row has a diagonal, deliberately -- `C.Phi - 1 = 0`
involves neither the parameter nor Omega, and declaring those `Dense` would manufacture the stored
zero diagonal of §7c.

Verified on the Lorenz Hopf: the frozen pattern engages (`_get_frozen_sparsity_nnz` 0 -> 59), every
structural block is **identical** to the value-filtered one, and the Hopf point is unchanged
(`rho = 24.7368421053`). The only differences are the three `Dense` border columns, 1 -> 3 each,
because only one of the three Lorenz equations depends on rho.

### Before writing the azimuthal or pitchfork spec: the vocabulary is not yet enough

`AugmentedBlockSpec::Kind` names *which matrix* a block's pattern comes from, but not *which residual
index*. The builder reads a single `ft->current_res_jac` and indexes `contributes_to_jacobian[resind]`
with it, which is correct for fold and Hopf: every block of those comes from one residual.

It is not correct for the other two. `AzimuthalSymmetryBreakingResidualContributionList` holds
**three** indices at once -- `{base (axisymmetric), real azimuthal, imag azimuthal}` -- and
`PitchForkResidualContributionList` holds two (`{base, mass-matrix residual}`). A spec written in
today's vocabulary would describe every block with whichever residual happened to be active, silently.

Neither handler is affected today: neither inherits `AugmentedSparsityProvider`, so no provider is
found, the builder returns NULL and both fall back exactly as before. Their `ndof` (`3*raw+2`,
`2*raw+2`) can never equal `raw` either, so the augmented dispatch always fires and always declines --
there is no path by which they could pick up a *wrong* mask rather than none.

**Done ahead of the specs**, rather than retrofitted around one. Each block entry now carries a
residual index (`-1` = "the one currently being assembled", which is what fold and Hopf use
throughout), `field_coupling_mask_for_element(matrix, residual, ...)` is the shared core that both the
plain and the augmented path call, and the builder collects the distinct **(matrix, residual)** pairs
its spec refers to *before* filling any of them -- so the buffer indices stay valid while several are
held at once, which a per-matrix scheme could not guarantee once a block mixes residuals.

Writing the azimuthal or pitchfork spec is now a matter of naming the residual per block; nothing in
the machinery has to change.

### Without an analytic Hessian, refuse rather than approximate

The first version fell back to the Jacobian pattern for Hessian blocks when no analytic Hessian had
been generated. That is wrong-headed: the Jacobian bounds an *analytic* Hessian, but nothing bounds
what a *finite-differenced* one writes, and the handler finite-differences precisely when codegen has
emitted no Hessian. The builder now returns NULL in that case, so the whole augmented pattern is
declined and assembly falls back to oomph-lib's. It costs the speedup for problems that have not
called `setup_for_stability_analysis(analytic_hessian=True)`, and nothing else.

### Pitchfork and azimuthal

Both are now described, and both needed the vocabulary extended beyond what fold and Hopf used.

**Pitchfork**, `[ base | parameter | Y | Sigma ]`. Its base block is the base Jacobian *plus*, when
`improved_pitchfork_tracking_on_unstructured_meshes` is on, a Hessian of the symmetry residual --
so a block needs to be **several patterns OR'd together**, not one. `AugmentedBlockSpec` therefore
holds a list of `Term{kind, residual}` per block rather than a single kind. Note also that the
residual map only exists in that improved mode; without it there is a single residual, the active
one, which is what `-1` already means.

**Azimuthal**, `[ base | real eig | imag eig | parameter | Omega ]` (or `[ base | eig | parameter ]`
without an imaginary part). This is the case the (matrix, residual) keying was built for: the
eigenvector blocks mix all three residuals -- `jacobian_real(m,n) - Omega*M_imag(m,n)` is the real
azimuthal Jacobian OR'd with the imaginary azimuthal mass matrix -- and nothing about it could be
said in a one-kind-per-block vocabulary.

Two things it taught, both about *declining*:

* **A negative residual index means "this element has no such contribution", not "undescribable".**
  Returning false there abandoned the pattern for the whole MESH, because `build_frozen_sparsity`
  gives up as soon as one element cannot be described. On Rayleigh-Benard that was every element:
  33 012 declines and no frozen path at all -- while the script still passed, because falling back is
  silent. Those blocks are simply empty, and saying so is a perfectly good description.
* **The axis boundary conditions write an identity** over the rows of dofs in
  `base_dofs_forced_zero` / `eigen_dofs_forced_zero`. Those entries come from no residual at all, so
  no coupling table can predict them, and the verification caught them immediately once the spec was
  actually being used. Hence a `Diagonal` kind: the diagonal of a square block only.

**`PeriodicOrbitHandler` is deliberately NOT described, and a spec for it would do nothing.** Periodic
orbits never form the augmented elemental block at all -- the routine says so:

```
// Periodic orbits would have very huge elemental Jacobians, so we must assemble them with block jacobians
```

They have their own assembly, `sparse_assemble_row_or_column_compressed_for_periodic_orbit`, which is
`std::map`-per-row. So `get_sparsity_pattern` would never be consulted; freezing them means a new fast
path in that routine, which is the Phase 2 item ("extend to ..._for_periodic_orbit") rather than a
tracker spec.

**It is worth doing.** Measured on `tests/benchmarks/bench_periodic_orbit_1d.py`, a 1D Brusselator
(validated against theory: Hopf at `b = 1 + a^2 = 2`, `omega = a = 1`, found at `b = 1.99999996`,
`omega = 1`):

| N | NT | orbit ndof | nnz | density | assembly | orbit solve | assembly share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 20 | 1 805 | 57 786 | 1.77 % | 7.3 ms | 0.01 s | 64 % |
| 40 | 30 | 5 023 | 164 286 | 0.65 % | 25.3 ms | 0.03 s | 74 % |
| 40 | 60 | 9 883 | 328 086 | 0.34 % | 55.6 ms | 0.08 s | 72 % |
| 80 | 60 | 19 643 | 654 966 | 0.17 % | 113.4 ms | 0.16 s | 70 % |

The assembly is **64-74 % of an orbit solve**, far above the 30-50 % typical elsewhere, and the table
shows why: the orbit matrix is so sparse that the factorisation is cheap and the map-based assembly
dominates. Freezing it therefore targets ~70 % of the runtime.

Note this could not be judged from the tutorials. All three that use periodic orbits drive them from a
3-dof ODE where the entire solve is sub-millisecond; extrapolating from those suggested it did not
matter, which is why the PDE case was written.

### Three defects this exposed, all worth remembering

1. **The mass table was unreachable.** `sparsity_mask_for_element` refuses `matrix_index > 0` unless
   an `EigenProblemHandler` is active. That guard is about the second matrix of a multi-matrix
   *assembly*; using the mass *coupling table* as a pattern source is meaningful whatever is
   assembling. Until a `MASK_MASS` sentinel was added, every Hopf spec silently fell back and the
   frozen path never engaged -- and a first "every block identical" reading was worthless, because
   both sides of the comparison had fallen back. **Check that the fast path engaged before believing
   a comparison**; this is the Phase 2 false negative in a new costume.
2. **Finite-differenced Hessians have no table.** The plan said the FD path "fills the same block
   positions, so the spec is independent of it -- but that must be verified rather than assumed". It
   is not independent: with no analytic Hessian, codegen emits none, `contributes_to_hessian` is
   empty, and the block is masked away while the handler fills it by finite differences. The fold
   normal form makes it concrete -- `d2R/dx2 = 2` is the entire Hessian block. The builder now checks
   `hessian_generated` and falls back to the Jacobian pattern, which is the superset the Hessian is a
   subset of.
3. **Sentinel values as array indices.** The `resind < 0` shortcut sized its scratch buffer with
   `resize(matrix_index + 1)`, and `matrix_index` can now be `(unsigned)-5`. That asks for ~4 billion
   entries and surfaced as `std::bad_alloc` in an ordinary continuation step, nowhere near the
   sparsity code.

### The failure that was unexplained, and what it was

`Advanced_Linear_Dynamics/hanging_droplet.py` tripped the per-element verification at element 898
(`nvar 7`, `raw 3`) with the parameter row against the eigenvector columns missing -- a block the spec
declares `Dense` and the tiling demonstrably writes. Instrumenting the *build* rather than the failure
settled it: the mask for that element was entirely zero (`npairs=0`).

The cause is the "no active residual" shortcut in `sparsity_mask_for_element`, which returns an
all-zero mask and sat **before** the augmented dispatch. That is correct for the raw block -- such an
element contributes nothing to the base Jacobian -- but a bifurcation handler still writes the border
of the augmented block for *every* element, because the normalisation row is a property of the
handler, not of the element's residual. Hoisting the dispatch above the shortcut fixed it.

The lesson is the same one as Phase 2: the contradiction was only resolvable by instrumenting where
the pattern is BUILT, not where it is used. Reasoning about the spec and the tiling -- both of which
were correct -- could not have found it.

### Superseded: the failure as first recorded

`Advanced_Linear_Dynamics/hanging_droplet.py` (a fold tracker on a moving mesh) trips the per-element
verification with the switch on:

```
element 898, nvar 7, raw_ndof 3 -- positions missing: (3,4), (3,5), (3,6)
```

nvar 7 with raw 3 is the fold layout `[base(3) | param(1) | eig(3)]`, so those three are the
**parameter row against the eigenvector columns** -- which the spec declares `Dense`, and which the
tiling demonstrably writes. Printing the spec the builder actually receives confirms it
(`(1,2)=Dense`, starts `0 3 4 7`), yet re-fetching the mask at the point of failure returns 0 there.
Not resolved. The guard is doing its job -- a refusal, not a truncated Jacobian -- but until the
contradiction is explained the switch must stay off.

Worth noting the failure only appears now because the frozen path never engaged for tracking before,
so this is an existing behaviour being exposed rather than a new defect; whether the fault is in the
spec or in something the augmented pattern reuses is exactly what is unresolved.

### The original plan follows.



### Where the two tracker families stand today

There are two, and they are frozen to very different degrees.

**The C++ handlers** (`src/bifurcation.{cpp,hpp}`: `MyFoldHandler`, `MyHopfHandler`,
`MyPitchForkHandler`, `AzimuthalSymmetryBreakingHandler`, `PeriodicOrbitHandler`) are oomph-lib
`AssemblyHandler`s. They present an *augmented elemental block* — `2·raw+1` for fold, `3·raw+2` for
Hopf, `raw·nT+1` for a periodic orbit — and fill it by hand, then let oomph's generic sparse assembly
scatter it. **Nothing about them is frozen.** `Problem::sparsity_mask_for_element` refuses at

```cpp
const std::vector<int> &cidx = be->get_local_dof_contribution_indices();
if (cidx.size() != nvar) return NULL;   // nvar = handler->ndof(elem) = augmented size
```

because the element's field description covers `raw_ndof` dofs and the handler is asking about a
block several times larger. Confirmed by measurement: under fold tracking
`_get_frozen_sparsity_nnz(0)` is 0, against 1 for the same problem without tracking.

**The Python trackers** (`pyoomph/generic/bifurcation_tools.py`: `FoldTracker`, `HopfTracker`,
`PitchForkTracker`, `NormalModeBifurcationTracker`) are the other way round. Their *constituent*
matrices already go through the frozen multi-assembly (Phase 4) — `R().J().dRdp().dJdp().dJdU(V)` is
one frozen pass. What is not frozen is the **bordering**, which is redone in scipy on every step:

```python
Jaug = scipy.sparse.block_array(
    [[J,    None,     col(dRdP)],
     [HV,   J,        col(dJdP@V)],
     [None, row(...), None]]).tocsr()
```

`block_array(...).tocsr()` rebuilds the whole augmented CSR — a COO build, a sort and a duplicate
merge — every Newton step, from blocks whose patterns are already known and fixed.

### The proposal

A no-op `get_sparsity_pattern` on a base class, overridden per tracker, describing the augmented
block in terms of the patterns it is *made of* rather than as an opaque dense block.

The vocabulary needs exactly six entries, because every sub-block of every tracker is one of them:

| kind | meaning |
| --- | --- |
| `Empty` | structurally absent |
| `Jacobian` / `JacobianT` | the base Jacobian pattern, or its transpose |
| `MassMatrix` / `MassMatrixT` | the base mass-matrix pattern, or its transpose |
| `Dense` | a border row/column (the parameter column, the normalisation row) |

Hessian-contracted blocks (`dJdU·Φ`, `dMdU·Φ`) do **not** need their own kind: contracting the
Hessian over one index cannot produce an `(i,j)` that the Jacobian lacks, because
`∂²R_i/∂u_j∂u_k ≠ 0` implies `∂R_i/∂u_j ≢ 0`. So they are `Jacobian`/`MassMatrix` — **or their
transposes**, and that is exactly where "left or right eigenvector" enters: a left-eigenvector
tracker assembles `Hᵀ·V`, whose pattern is `Jᵀ`. This is the same distinction that Phase 4 got wrong
and had to fix by symmetrising; here it can be stated exactly instead of over-approximated.

Read off the code, `MyFoldHandler` in `Full_augmented` (layout `[base | param | eigenvector]`,
from `jacobian(raw+1+n, ...)` and `jacobian(n, raw_ndof)`) is:

|  | base cols | param col | eig cols |
| --- | --- | --- | --- |
| **base rows** | `Jacobian` | `Dense` | `Empty` |
| **param row** | `Empty` | `Empty` | `Dense` |
| **eig rows** | `Jacobian` (from `dJduPhiH`) | `Dense` | `Jacobian` |

Hopf adds the imaginary eigenvector group and the `MassMatrix` blocks; the periodic-orbit handler is
`nT` diagonal `Jacobian` blocks plus coupling blocks and one border row/column.

### C++ side

1. `virtual bool get_sparsity_pattern(GeneralisedElement*, AugmentedBlockSpec&) const { return false; }`
   on `oomph::AssemblyHandler` — vendored, one line, no-op, so every handler that does not override it
   behaves exactly as now.
2. `AugmentedBlockSpec` carries the group layout (how the augmented dofs partition into `raw`-sized
   groups plus scalar border dofs) and the `n_groups × n_groups` matrix of kinds above.
3. `Problem::sparsity_mask_for_element` gains a branch *before* the `cidx.size() != nvar` refusal: if
   the active handler fills a spec, build the `nvar × nvar` mask by tiling the base masks
   (`sparsity_mask_for_element(0, ...)` for `Jacobian`, `(1, ...)` for `MassMatrix`, transposed where
   asked, all-ones for `Dense`, all-zeros for `Empty`).
4. `get_jacobian_structure_id()` already keys on the assembly handler's *type*, which is necessary but
   no longer sufficient: `MyFoldHandler` changes its own block layout with `Solve_which_system`, and
   `MyHopfHandler` likewise. The key must include whatever the spec depends on.

From there `build_frozen_sparsity` and `assemble_with_frozen_sparsity` work unchanged, and the
trackers get the full frozen path.

### Python side

1. `AugmentedAssemblyHandler.get_sparsity_pattern()` — no-op returning `None`.
2. Each tracker returns the same block description, in the same vocabulary, for its `block_array`
   layout. It is a direct transcription of the literal already written there.
3. A frozen bordered assembly replaces `block_array(...).tocsr()`: build the augmented `indptr`/
   `indices` once per structure id, together with a scatter map from each constituent block's value
   array into the augmented one, then per step copy values only.

**One trap on this side.** The border blocks are built as `csr_matrix(dense_array.reshape(-1,1))`,
which *drops the zeros of a dense vector*. So the border column's pattern today depends on the values
of `dRdP` — it is not stable, and a frozen version must declare those blocks structurally dense
rather than inheriting whatever `csr_matrix` happened to keep.

### Risks

* **Under-reporting is the dangerous direction**, as everywhere in this work. It is guarded: the
  per-element verification refuses rather than truncates, and in oomph-lib's ordinary assembly the
  mask can only ever add. A wrong spec is therefore loud, not silent.
* **§7c will bite harder.** The augmented pattern introduces stored zeros on the border rows and
  columns, and the normalisation row's diagonal is structurally zero by construction. §7c showed what
  stored zero diagonals do to MUMPS's analysis — `ICNTL(24)=1` is already on, and §7d's measurement
  that the remaining legitimate zero diagonals *still* trip it with the option off says this needs
  checking on a real tracking run, not assuming.
* **`Dense` really is dense**, so the border column adds `raw_ndof` stored entries per augmented
  dof. For a fold that is one column and one row; for a periodic orbit with large `nT` it is worth
  measuring before assuming it pays.
* The C++ handlers also run a **finite-difference path** when no analytic Hessian is set. That path
  fills the same block positions, so the spec is independent of it — but that must be verified rather
  than assumed, since it is exactly the kind of assumption that produced the Phase 4 bug.

### Suggested order

1. C++ `MyFoldHandler` alone, behind the no-op default, with the verification on. Smallest augmented
   layout, and the one with a ready-made regression case in the tutorials.
2. Measure. If a fold tracking step does not improve materially, stop — the rest of the family shares
   the same cost structure and there is no point paying the complexity.
3. `MyHopfHandler` (adds the mass-matrix and transpose kinds), then the remaining C++ handlers.
4. The Python bordering, which is a separate mechanism and can be judged on its own.

## 7f. Freezing the periodic-orbit assembly

The orbit handler was the last tracker left, and the most valuable one: on the 1D Brusselator
(`tests/benchmarks/bench_periodic_orbit_1d.py`) the assembly is 64-74% of an orbit solve, because the
orbit matrix is so sparse (0.17% at N=80/NT=60) that the factorisation is cheap next to it.

### Two things I had wrong

I had recorded that "periodic orbits never form the augmented elemental block, so a spec would never
be consulted", on the strength of the comment in
`sparse_assemble_row_or_column_compressed_for_periodic_orbit`:

> Periodic orbits would have very huge elemental Jacobians, so we must assemble them with block jacobians

That comment describes the band-matrix path, which is `#define`d **out** (`//#define
PYOOMPH_PERIODIC_ORBIT_BAND_MATRIX`) and never implemented -- its body is
`//throw_runtime_error("TODO: Fill it in")`. The path that actually runs builds a dense
`nvar x nvar` `oomph::DenseMatrix` with `nvar = raw*nT+1` and scans all of it into a
`std::map<unsigned,double>` per row. So the augmented block is formed after all, and
`AugmentedBlockSpec` applies exactly as it does to Fold/Hopf/Pitchfork/azimuthal.

The reason a spec would nonetheless never have been consulted was something else entirely: the
dispatcher tried the orbit routine *before* `assemble_with_frozen_sparsity`. Reordering it is the
whole enabling change; the frozen path declines on its own when it cannot apply, so the orbit routine
stays as the fallback.

### The spec: nT raw-sized groups plus the period

The augmented unknowns are nT copies of the raw dof set followed by the scalar period T, so the block
is an (nT+1)x(nT+1) grid. Within a coupled pair of time nodes the pattern is just the base Jacobian
and mass matrix, so the only thing the spec has to describe is **which time nodes couple** -- and that
is what differs between the five discretisations. All of it is read off the same data the assembly
loops use rather than re-derived:

| mode | coupling |
|---|---|
| collocation (default) | pairs within a time element of `time_mesh`, via `TimeNode::get_index()` |
| B-spline | pairs supported on a common basis element, via `get_integration_info()`'s `indices` |
| Floquet | `t -> t` and `t -> t+1` (J and M), last row flushed and replaced by the wrap identity |
| central, BDF2 | `t -> t` (J and M), plus the `dU/ds` stencil `FD_ds_inds[t]` -- **mass matrix only** |

### Two subtleties, both caught by measurement rather than by reading

**Rows and columns do not run over the same node set.** A collocation time element with `nnode()`
nodes carries only `nnode()-1` collocation points, and the equation of collocation point `inode` is
written into the row of node `inode`. So only the *first* `nnode()-1` nodes of an element are equation
rows, while all of them appear as columns. Declaring the last node as a row too is legal -- it only
stores zeros -- but it cost +25% nnz, and the give-away was that blocks (3,0), (6,3), (9,6), (12,9)
came out **100%** structural zeros, i.e. exactly the element-boundary nodes at order 3. The period
column has the same asymmetry, so it is gated on whether a time node carries an equation at all.

**The FD stencil multiplies the mass matrix alone.** The first version of that branch guessed
`t, t+-1, FD_ds_inds[t]` with both J and M. It was correct but cost +44% nnz for central and +89% for
BDF2. The nodal modes have no `t -> t+1` term at all -- the spatial operator is evaluated at the node
itself -- and the stencil carries only M.

`AugmentedBlockSpec::Diagonal` also had to stop requiring `gr == gc`: the wrap-around
`u(nT-1) - u(0) = 0` that closes the orbit is an identity landing on the diagonal of an *off-diagonal*
block. That is the second consumer of `Diagonal`, after the azimuthal axis boundary conditions.

### Verified

Frozen vs unfrozen assembled at the **identical state** (toggling the flag between two assemblies in
one run, rather than comparing two independent runs -- those drift, and a 7.5e-03 difference that
looked like a defect was just that). Brusselator N=20, NT=12:

| mode | order | engaged | nnz off | nnz on | d nnz | max abs dJ |
|---|---|---|---|---|---|---|
| collocation | 3 | yes | 33126 | 33127 | +0.00% | 0 |
| collocation | 2 | yes | 25398 | 25399 | +0.00% | 0 |
| floquet | 0 | yes | 17670 | 17671 | +0.01% | 0 |
| central | 2 | yes | 17424 | 17425 | +0.01% | 0 |
| BDF2 | 2 | yes | 17424 | 17507 | +0.48% | 0 |
| bspline | 3 | yes | 56064 | 56065 | +0.00% | 0 |

Bit-identical in every mode. The one extra entry is the (T,T) diagonal, declared deliberately so the
period row has a stored diagonal (see the null-pivot discussion in 7d). BDF2's +0.48% is the wrap
identity, which that mode does not use.

Speedup, collocation, frozen vs unfrozen:

| N / NT | orbit ndof | nnz | assembly | orbit solve |
|---|---|---|---|---|
| 20 / 12 | 1067 | 33k | 3.29 -> 0.95 ms (-71%) | -69% |
| 40 / 30 | 5023 | 164k | 24.8 -> 5.8 ms (-77%) | -71% |
| 80 / 60 | 19643 | 655k | 116.1 -> 29.0 ms (-75%) | -66% |

`docs/.../orbit/hopf_switch.py` produces identical output with the flag forced on and off.
`langford_floquet.py` fails identically both ways with `TypeError: must be real number, not complex`
-- a pre-existing Python-side bug in that tutorial, unrelated to this work.

A stray `std::cout << "Sparse assembly for periodic orbit:"` that printed on every fallback orbit
assembly, even under `quiet()`, was removed while here.

## 7g. What the stored zeros in a frozen pattern actually are

A frozen pattern must be a superset of the numerically nonzero entries or it could not be reused, so
it always carries some stored zeros. `_get_frozen_sparsity_fill_stats()` measures how many; across the
tutorial suite it is 6.0% of 8.6e9 slots, median **0.000%** per script -- most patterns are exactly
tight -- with a few scripts far above (rivulet 39.8%, rising_bubble 29.1%, eigendynamics 28.4%).

`_get_frozen_fill_breakdown()` (only filled when `PYOOMPH_FROZEN_FILL_BREAKDOWN` is set) attributes
every scattered entry to its (row class, column class) pair. That is what separates the two causes,
which want opposite treatment.

### Cause 1: unattributed dofs -- a real defect, fixed

On the DG convection-diffusion tutorial 33% of the pattern was stored zeros, and every one of them
sat in a row or column of an `<unattributed>` dof of the `_internal_facets_` element, while the
properly attributed `domain/c` block was exactly tight. An interface element works on dofs belonging
to other elements and the attribution walk could not see them, so they kept the -1 sentinel, which the
mask must read as "coupled to everything". Fixed by adopting the source element's own attribution
through the equation map that already exists (rivulet 39.8% -> 37.0%, eigendynamics 28.4% -> 27.2%).

### Cause 2: a field that is identically zero -- inherent, and must NOT be pruned

What remains is concentrated in blocks like

    domain/velocity_phi x domain/velocity_y     87.4% zero
    domain/velocity_phi x domain/coordinate_y   90.0% zero
    liquid/velocity_normal x liquid/coordinate_x 67.1% zero

My first reading of this was that it is azimuthal-mode-dependent coupling, and that folding the mode
number into `jacobian_structure_id` would let the pattern be tightened per mode. **That was wrong.**
Measuring the base state settles it:

    domain/velocity_phi        691 dofs   max|value| = 0.000e+00   100% exactly zero
    liquid/velocity_normal     591 dofs   max|value| = 0.000e+00   100% exactly zero

The blocks are empty because the base state has no swirl and no normal flow, so `velocity_phi` and
`velocity_normal` are identically zero and appear as a multiplicative factor in exactly those Jacobian
terms. The coupling is structurally real; it is the VALUE that vanishes.

So these stored zeros are the correct price of a value-independent pattern and must be left alone.
Pruning them would be actively unsafe: the moment the solution develops swirl or normal flow -- which
in a stability analysis is precisely what is being looked for, since the perturbation lives in those
components -- the entries become nonzero, and a pattern pruned on the current values would truncate
the Jacobian (or, with the verification on, refuse to assemble at all).

The general rule this illustrates: a block that is 100% zero because a coupling was never written is
a table defect worth fixing; a block that is mostly zero because a field currently happens to vanish
is not, and the two are indistinguishable without the per-block breakdown.

### Possible future feature: let the user declare a field zero in the base state

The framework cannot know that `velocity_phi` or `velocity_normal` stays zero -- nothing in the
residuals forbids it, and in the perturbation problem those components are exactly what is nonzero.
The USER often does know, and knows it for a structural reason rather than by accident. In the rivulet
case the slip-length condition at the bottom prevents any non-zero `velocity_normal` in the base
solution at all; in a stability analysis of a non-swirling flow the same holds for `velocity_phi`.

So there is room for an opt-in declaration -- something like "this field is identically zero in the
base state" -- which would let the BASE-state block drop the rows and columns of the declared fields
from its pattern. Measured above, that is 87-90% of those blocks, and they are the bulk of what is
left after the attribution fix, so the payoff is real: rivulet and eigendynamics would go from ~37%
and ~27% stored zeros to close to the median of 0%.

Three things it would have to respect, all of which the current design already provides for:

* **Base state only.** The declaration applies to the base Jacobian, not to the perturbation or
  eigenproblem matrices, where those components carry the whole solution. Patterns are already keyed
  per matrix, so this is a per-matrix property, not a global one.
* **Part of the pattern key.** `jacobian_structure_id` would have to include the declaration, or a
  pattern built with it could be handed to an assembly running without it.
* **Opt-in, never inferred.** Deriving it from observed values would be exactly the unsafe pruning
  described above -- a field that happens to be zero at the current point is not a field that must be.
  The declaration has to be a statement about the problem, made by whoever set it up.

What makes this safe to offer at all is that the per-element verification already exists: a wrong
declaration does not silently truncate the Jacobian, it refuses to assemble and names the positions
that escaped the pattern. That turns a subtle correctness hazard into a loud, immediate error.

Not implemented; recorded here because the measurement that motivates it is above and would otherwise
have to be redone.

## 7h. A reused KSP freezes MUMPS' *analysis*, not just its symbolic factorisation — **FIXED**

Reported from a user script (an axisymmetric two-phase ALE continuation, `WaterEthanol_Air_QS`
family): runs fine under `pardiso`, and under `--petsc_mumps` collapses to `Maximum residuals inf`
partway through an arclength continuation, from which it never recovers — the step keeps halving
until oomph aborts with `DESIRED ARC-LENGTH STEP … HAS FALLEN BELOW MINIMUM TOLERANCE`.

### What it was

Nothing to do with the matrix. Instrumenting `PETSCSolver.solve_serial` gives the whole story in one
line per solve:

| solve | `INFOG(12)` (off-diagonal pivots) | `INFOG(1)` | KSP reason | ‖x‖ |
| --- | --- | --- | --- | --- |
| first, ndof 12474 | 288 | 0 | 4 (converged) | 2.5 |
| a few continuation steps later | 1648 → 1682 → 1776 | 0 | 4 | fine |
| the one that breaks | 2638 | **−9** | **−11** (`DIVERGED_PC_FAILED`) | **inf** |
| every solve after it | 2638 (frozen) | −9 | −11 | inf, in ~30 µs |

`INFOG(1) = -9` is "the main internal real workarray is too small". §5b's KSP reuse is what caused
it. Keeping the KSP alive across solves is the whole point of `_setup_solver_if_needed`, but PETSc
then sees `SAME_NONZERO_PATTERN` and skips `MatLUFactorSymbolic` — which for MUMPS means skipping the
**analysis phase**, and `use_mumps()` sets `ICNTL(6) = 5`, so that analysis is *value-dependent*: it
fixes a maximum-transversal column permutation and a scaling from the matrix it first saw. A
continuation walks the values away from that matrix, the frozen ordering stops putting large entries
on the diagonal, numerical pivoting grows an order of magnitude, and the fill-in finally exceeds the
workspace the stale analysis had predicted.

Three A/B runs of the same script, each carried to the same point, isolate it:

| arm | failures | `INFOG(12)` | in-process KSP time (25 solves) |
| --- | --- | --- | --- |
| as shipped (Mat + KSP reused) | fails, unrecoverably | 288 → 2638 | — |
| Mat reused, **KSP rebuilt each solve** | none | ~250, flat | 6.53 s |
| Mat + KSP reused, `ICNTL(14) = 200` | none | ~2900 | 2.43 s |
| no reuse at all (`reuse_matrix_structure = False`) | none | ~238, flat | 7.61 s |

The middle row is the diagnosis: rebuilding *only* the KSP, on the very same `Mat` object, is enough
to fix it. So it is the analysis being frozen, not the matrix being reused.

The last column is why the fix is not "stop reusing". The stale analysis is *worse numerically* and
still **2.7× faster** than redoing the analysis every solve, because the analysis phase dominates. The
reuse is worth keeping.

### The fix

`PETSCSolver._ksp_solve_checked()` replaces the bare `ksp.solve()` in both `solve_serial` and
`solve_distributed`:

1. Read `getConvergedReason()`. **Nothing did this before**, which is the reason the failure presented
   as an `inf` residual rather than an error: on a PC failure PETSc returns immediately and leaves the
   solution vector untouched, and pyoomph handed that straight to Newton.
2. Act only on `DIVERGED_PCSETUP_FAILED` (−11) and `DIVERGED_NANORINF` (−9) — the two that mean the
   returned vector is not a solution of anything. Destroy the KSP and rebuild it (a fresh PC means a
   fresh analysis for the values actually on hand) and solve again. If MUMPS reported one of the
   workspace codes (`INFOG(1) ∈ {−8, −9, −11, −12, −14, −15, −17, −19, −20}`), double `ICNTL(14)`
   first and keep the larger value for the rest of the run.
3. If the retry also fails, raise, naming the reason and `INFOG(1)`.
   `raise_on_failed_solve = False` downgrades that to a warning.
4. Every *other* negative reason warns and hands the iterate on unchanged, which is what every
   configuration did before the check existed. `DIVERGED_MAX_IT` is the one that matters: an iterative
   KSP stopping on its iteration limit has produced a real, merely inaccurate, answer that Newton is
   often happy with. Retrying there would discard a hypre/GAMG setup and redo the same solve to reach
   the same place, and raising would break any hand-configured iterative solver that has always been
   allowed to return early.

Nothing MUMPS-specific is reached unless MUMPS is both present and in use: `_mumps_infog()` gates on
`PETSc.Sys.hasExternalPackage("mumps")` and then on `pc.getFactorSolverType() == "mumps"`, and every
MUMPS branch hangs off a non-None answer from it. `getFactorSolverType()` is what makes the second gate
free of side effects — it answers None for a PC with no factor matrix at all (jacobi, gamg, hypre,
none), where `getFactorMatrix()` raises PETSc error 56, and it names the package for the ones that do,
so SuperLU or UMFPACK is turned away rather than being asked for an `INFOG` it does not have.

Collective under MPI: `KSPSolve` reduces the PC failure across the communicator, so every rank sees
the same reason and takes the same rebuild branch.

Checked on four configurations besides the reported script: PETSc's own LU and SuperLU (both
`_mumps_infog(1) → None`, converged, no MUMPS call made), GMRES+Jacobi capped at one iteration
(`DIVERGED_MAX_IT` every solve: warned, no retry, no raise, Newton still creeps down 3.1e-3 → 2.3e-3
and then hits oomph's own max-iteration abort exactly as it does without the patch), and
`mpi_structural_worker` on `petsc_mumps` at 1 and 2 ranks. A synthetic raise from inside the callback
was also confirmed to unwind cleanly through oomph's C++ Newton loop, be catchable in Python, and
leave the `Problem` usable for further solves.

On the reported script this turns the run into one recovered failure —
`KSPConvergedReason -11, MUMPS INFOG(1) -9 … retrying with -mat_mumps_icntl_14 40` — after which the
continuation completes and reproduces the `pardiso` trajectory (the arclength derivative agrees to 15
digits). Total in-process KSP time 1.58 s, i.e. the fast reuse path is kept.

### The general point

Reusing a factorisation object across solves silently changes *which* factorisation you get, not just
how long it takes to get it. Anything the first call decided from values — MUMPS' transversal and
scaling here — is now decided once, for a matrix that is no longer the one being solved. Backends
whose symbolic phase is purely structural (Pardiso's phase 11) do not have this failure mode, which is
exactly why `pardiso` was unaffected. Worth remembering before extending the reuse to another backend.

### Pardiso could not report its own errors either — **FIXED**

`run_pardiso()` passed MKL's `error` argument as `byref(c_int(ERR))`. That builds a throwaway ctypes
object from the Python int, so MKL wrote its code into a temporary that was discarded on the next
line, and the `if ERR != 0` below it compared the untouched Python int against zero. Pardiso has
therefore never been able to report any of its own errors, for as long as the call has existed — same
class of bug as the KSP reason nobody read, different backend.

Now a named `c_int` whose `.value` is checked, against MKL's documented table (`-1` input
inconsistent … `-15` internal error), raising with the phase named (`reordering and symbolic
factorisation`, `numerical factorisation`, `solve and iterative refinement`, …). The release phase is
the deliberate exception: `clear()` runs from `__del__`, where raising only produces an "Exception
ignored in `__del__`" and there is nothing to salvage anyway, so it warns.

Verified by forcing a real MKL failure (`error -1` from a corrupted `n`), which is now reported with
the right phase and message; a healthy factor/solve stays silent, and the reported script under
`pardiso` runs unchanged with no error reported anywhere. Worth knowing for anyone testing this: a
structurally singular matrix does **not** provoke an error, because MKL replaces the tiny pivot and
returns success with a huge solution — that is MKL's documented behaviour, not a gap in the check.

## 7i. The same question for SLEPc + MUMPS — escalation added, the rest does not apply

Asked directly: should the eigensolver get the same treatment? Investigated; the answer is a third of
it, and the reasons the other two thirds do not apply are worth recording.

**The stale analysis cannot happen here.** `SlepcEigenSolver.solve()` builds `J`, `M` and the `EPS`
from scratch on every call and destroys all three at the end, so the ST's KSP, its PC and the MUMPS
instance inside it are new each time and MUMPS always re-runs its analysis on the matrix actually
being factorised. There is nothing cached to go stale, and nothing to discard and rebuild. §7h's
retry-with-a-fresh-KSP would be pure overhead.

**The silent garbage cannot happen here either.** SLEPc's `STSetDefaultKSP`
(`src/sys/classes/st/interface/stsles.c`) calls `KSPSetErrorIfNotConverged(st->ksp, PETSC_TRUE)`, so a
failed solve inside the spectral transform raises out of `EPSSolve` rather than returning an untouched
vector — the opposite of the bare `KSPSolve` in the linear solver. `STMatSolve` itself never checks the
reason, but it does not need to. And PETSc's MUMPS wrapper already names the failure:

```
[0] MUMPS error in numerical factorization: INFOG(1)=-19, INFO(2)=933361
    (see users manual https://mumps-solver.org/... "Error and warning diagnostics")
```

So no convergence check was added: the library performs it, and better.

**The escalation is worth having.** A shift-and-invert factorises `J − σM`, commonly denser than `J`
alone, so it exhausts MUMPS' workspace more readily than the linear solver does — and pyoomph never
sets `st_mat_mumps_icntl_14` unless `use_mumps()` was handed an explicit `mumps_param14`, which asks
the user to have foreseen the whole thing. `_eps_solve_with_workspace_retry()` doubles it and rebuilds
the EPS once. Rebuilding, not re-solving: a PETSc option is consumed when the object it configures is
set up, so a raised `ICNTL(14)` is only seen by a KSP/PC that has not been set up yet, and the spectral
transform's has. The EPS construction moved into a closure for exactly this.

**`INFOG(1) = -19` is not in the escalation list**, in either solver — a correction to §7h as first
written, which had it there. It means the factorisation exceeded `ICNTL(23)`, the hard cap on working
memory in MB, and doubling `ICNTL(14)` against a cap asks for more of something already forbidden. The
lever is `ICNTL(23)`, which is the user's to set (pyoomph never does), so both solvers now say so and
re-raise instead of spending a retry that cannot work.

Verified on four configurations, on the complex PETSc build, `LineMesh(N=20000)`, 80 002 dofs:

| provocation | INFOG(1) | outcome |
| --- | --- | --- |
| none | 0 | unchanged; same eigenvalues as before the patch |
| `st_mat_mumps_icntl_14 = -95` | −9 family | `retrying with -st_mat_mumps_icntl_14 40`, then the correct spectrum |
| `st_mat_mumps_icntl_23 = 1` | −19 | names the cap, re-raises, no retry spent |
| an ST KSP that cannot converge | n/a | re-raised untouched (PETSc error 91), exactly as before |

The MUMPS plumbing both solvers share — the INFOG accessor with its two gates, the `ICNTL(14)` doubling,
and the two error-code lists — is now module-level in `pyoomph/solvers/petsc.py` rather than duplicated.

## 8. Open questions

### Closed by the work

* ~~Under MPI, is it worth freezing the *local* block only as an intermediate step, before also
  freezing the exchange?~~ Both are done (Phase 2b), and the premise was wrong: the local scatter is
  only about a third of the distributed overhead, the exchange and owner-side merge are the rest.
* ~~Do numerical entries fall outside the connectivity pattern in exotic setups?~~ Not answered
  exhaustively, but the question is now defended rather than open: `verify_frozen_sparsity` (on by
  default) counts, per element per matrix, the nonzeros of the whole elemental block against those
  the scatter actually took, and refuses the assembly if anything was dropped. So an unmodelled
  coupling produces a loud error naming the element, not a silently truncated Jacobian.

### Still open

* ~~**The residual-only assembly path**~~ — **done, and the premise was half wrong.** Measured
  serially it is **96–101 % elemental JIT evaluation** (2D: 17.15 ms total, 16.41 elemental; 3D:
  178.9 total, 180.7 elemental — the difference is noise), so there is nothing there to win outside
  the JIT and the idea that it could reuse the elemental loop and dof-gather is refuted.

  Under MPI it was a different story, and that half is now fixed. oomph-lib routes `get_residuals()`
  through the whole of `parallel_sparse_assemble()` with zero matrices — recomputing `my_eqns`,
  building the array-of-arrays, exchanging equation numbers and merging by bisection per row, all to
  sum a vector, once per Newton step. Overhead: 2.85 ms of 11.16 (26 %) at 2 ranks and 3.28 of 7.32
  (**45 %**) at 4, and roughly *constant* in absolute terms as ranks grow. Frozen, the residual
  assembly is **−20 %** at both 2 and 4 ranks, and the end-to-end distributed Newton step goes to
  −18 % / −27 % (Phase 2b alone was −12 % / −41 %).

  Worth noting for anyone re-treading this: the Newton loop evaluates the residual **twice at every
  state** — once standalone for the convergence check, once again inside the next `get_jacobian`.
  That looks like free money and is not. The bundled one is nearly free (same elemental pass), so the
  only way to remove the duplication is to restructure the loop to assemble R+J before the
  convergence test, which wastes a Jacobian on the final step. For *n* Newton steps that trades
  *n*×R against one marginal J: a win in 2D (R = 17 ms, marginal J = 20 ms) but a clear loss in 3D,
  where the marginal J is 895 ms against a 179 ms residual and it would need *n* ≥ 6 to break even.
* **Memory footprint of the scatter map on large 3D problems.** The compact slot-sorted form
  (`scatter_source`/`scatter_slot`) is much smaller than the per-position table it replaced, and
  Phase 2b adds a per-rank plan (`merge_perm`, `final_col`, `local_col`, the scatter map) of order a
  few times nnz in `int`s. Never measured on a problem large enough to care. The binary-search
  fallback exists if it turns out to matter.
* **Exposing the pattern to Python** (`problem.get_jacobian_sparsity()` → `(indptr, indices)`), so
  `CustomAssemblyBase` implementors and the eigen matrix manipulators
  (`pyoomph/solvers/generic.py:160-350`, which do `csr` surgery row by row) can work on a fixed
  pattern too. Nice-to-have; nothing depends on it.
* **`Sparse_assemble_with_arrays_previous_allocation`** (oomph `problem.h:708` region, reset at
  `problem.cc:2151`/`:2481`) sizes rows from the previous assembly. With structural zeros kept,
  allocations should stabilise immediately — likely a small side benefit, still unconfirmed. Note
  the frozen paths bypass this allocator entirely, so it only matters where they fall back.

### Found on the way, not part of this work

* **Distributed eigensolving is broken in the Python layer**, independent of anything here.
  `GenericEigenSolver.get_J_M_n_and_type()` (`pyoomph/solvers/generic.py:449`) wraps the *row-local*
  CSR that `assemble_eigenproblem_matrices()` returns in a `csr_matrix` of *global* shape, and scipy
  rejects it (`index pointer size (584) should be (1227)`). It fails identically with the frozen
  route on or off. The C++ side is fine — the two-matrix distributed assembly was verified below
  this layer (§ Phase 2b) — so this is a wrapper bug, not an assembly one.
* Also worth knowing when testing that layer: the arrays `assemble_eigenproblem_matrices()` returns
  are **views** into `eigen_{Mass,Jacobian}MatrixPt`, which the next call deletes and reallocates.
