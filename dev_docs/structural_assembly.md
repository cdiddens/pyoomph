# Structural assembly: precomputed CSR sparsity, value-only re-assembly and solver reuse

Status: **implemented, tested and benchmarked** — serial, distributed, and for every bifurcation
tracker including periodic orbits. The solver-side half of the story (what each backend guarantees when
a factorisation is reused, and what stored zeros do to MUMPS) lives in
[linear_solvers.md](linear_solvers.md).

**The idea.** Every Jacobian assembly used to rebuild the CSR structure (row starts + column indices)
from scratch, and every linear solve therefore re-did its symbolic phase. Neither is necessary: as long
as the equation numbering is unchanged, the sparsity pattern of the Jacobian is fixed. Compute it
**once** and you can (a) assemble straight into a preallocated value array, (b) hand solvers a
"values-only" promise so they skip reordering/symbolic factorisation, and (c) get the "zeros on the
diagonal" some PETSc backends require for free.

**MPI was a requirement, not an afterthought.** The distributed assembly is a *separate* oomph routine
(`parallel_sparse_assemble`) with its own containers and an index+value exchange between ranks — but it
is also where a frozen pattern pays off most, because the column-index half of every exchange becomes
redundant.

---

## 1. How the Jacobian is assembled, and where the structure is rebuilt

```
Problem::newton_solve
  └── LinearSolver::solve(problem, dx)                    (SuperLUSolver)
        ├── problem_pt->get_jacobian(residuals, CR_jacobian)
        │     └── sparse_assemble_row_or_column_compressed(...)      (virtual)
        │           └── pyoomph::Problem::sparse_assemble_row_or_column_compressed
        │                 ├── ..._for_periodic_orbit()
        │                 └── oomph::Problem::sparse_assemble_row_or_column_compressed
        │                       → one of five container strategies
        ├── factorise()   → superlu(op_flag=1, ...)
        └── backsub()     → superlu(op_flag=2, ...)
```

All five oomph strategies do the same thing at different memory/speed trade-offs: loop over elements
and get a dense `nvar × nvar` elemental Jacobian; for every `(i,j)` with
`fabs(value) > Numerical_zero_for_sparse_assembly`, accumulate into a container; then compress —
count entries per row, `new int[nnz]`, `new double[nnz]`, walk the sorted container and emit
`column_index` / `row_start` / `value`. **The last step allocates fresh index arrays on every single
assembly**, and `CRDoubleMatrix::build_without_copy()` frees them again after the solve.

`Numerical_zero_for_sparse_assembly` is `0.0` and the test is a strict `>`, so **exact zeros are
dropped**. That is the crux: the emitted pattern is *value dependent*.

pyoomph has three structurally identical routines of its own — `..._for_periodic_orbit` (the augmented
periodic-orbit system), `..._base_problem` (the unaugmented block while a bifurcation/arclength
augmentation is active, and the engine behind `assemble_multiassembly`), and `assemble_hessian_tensor`
(rank-3). When the problem is distributed, `get_jacobian` takes an entirely different branch into
`Problem::parallel_sparse_assemble` — see §5.

An abandoned first attempt at this already existed: `Problem::update_jacobian_csr_structure()` was a
stub for precisely this idea, with its body commented out and a
`throw_runtime_error("Implement and check performance")`. Its `std::map<unsigned,unsigned>`-per-row
data structure is *not* what was wanted.

**The symbolic information was already there.** `src/codegen.cpp` emits, per generated element code and
per residual/Jacobian combination, `contributes_to_jacobian[res][field_i][field_j]` — an exact symbolic
field-coupling table answering "does ∂(residual of field i)/∂(field j) contain any term at all?".
`Problem::assemble_defined_field_list()` already consumed it, but only to build a *problem-global* union
used to detect fields with empty Jacobian rows/columns. Nobody used it for sparsity.
`BulkElementBase::get_dof_names()` shows the machinery to map *local dof index → (field, space, node)*
already exists, which is the missing link to apply the table at element level.

The index of that table is the *contribution class* of a field: the domain it is DEFINED on plus its
name (`block_contribution_class_name()`), so that a bulk field and the interface's view of it share one
class — they are one dof. One refinement was added later for hybridized DG: on an interior facet the two
sides are distinct classes, `<domain>/<field>` and `<domain>/<field>@opposite`, but **only** for
element-private (DG/DL/D0) spaces, where the two sides' Data provably cannot be the same. That is what
lets the table state that the near-side and far-side copies do not couple, which in turn is what makes
static condensation decompose an HDG system per element ([static_condensation.md](static_condensation.md)
§4). A continuous field keeps one merged class, because its facet dofs really are shared and a split
would declare a live entry structurally absent. `Problem::assemble_defined_field_list()` strips the
suffix again, so the problem-global view and `_jacobian_structure.txt` stay side-agnostic.

And a precedent existed one level up: `SparseRank3Tensor::finalize_for_vector_product()` already freezes
a CSR structure once and afterwards returns **only a value array**, stuffed into a pre-built
`csr_matrix` whose `indptr`/`indices` never change.

---

## 2. Measured baseline

3D lid-driven cavity, Taylor–Hood NS, `CuboidBrickMesh(N=8)`: `ndof = 10 853`, `nnz = 1 800 539` (166
per row), 512 elements, `nvar = 89`, 2 688 761 raw scatter entries.

| step | time |
|---|---|
| residuals only | 55 ms |
| residuals + Jacobian | **424 ms** |
| MKL Pardiso phase 11 (analysis / reordering) | **179 ms** |
| MKL Pardiso phase 22 (numerical factorisation) | 111 ms |
| MKL Pardiso phase 33 (triangular solves) | 4.7 ms |
| PETSc `createAIJ` + `assemble` | 20.4 ms |

**A full Newton step costs 424 + 246 + 5 ≈ 675 ms, of which 179 ms (26 %) is a symbolic factorisation
that recomputes an unchanged result.** Container choice barely matters — the whole
`Sparse_assembly_method` knob is a rounding error (420–506 ms across all five).

### 2.1 The pattern is not stable today

3D NS, `N=4`, `ndof = 1153`: 138 716 nnz at `U = 0`, 138 817 after `solve()` — 572 entries appear, 36
disappear, `J0.indices == J1.indices` is **False**. So "just reuse the arrays and hope" is not an
option. Any reuse scheme **must** be built on a value-independent pattern.

### 2.2 A value-independent pattern costs almost nothing

"Structural pattern" = union over elements of all `(eqn_number(e,i), eqn_number(e,j))` pairs, from
connectivity alone.

| case | ndof | numerical nnz | structural nnz | ratio | numerical ⊄ structural |
|---|---|---|---|---|---|
| NS 3D, `N=8` | 10 853 | 1 800 539 | 1 816 855 | **1.009** | 0 |
| NS 2D, `N=40` | 14 162 | 542 104 | 556 738 | **1.027** | 0 |
| NS + advection–diffusion 2D, `N=40` | 20 642 | 840 730 | 1 132 540 | **1.347** | 0 |

For single-physics the structural pattern is within 1–3 % of the numerical one. It **always contains
the full diagonal** (every dof belongs to at least one element, and `(i,i)` is in that element's block).
For weakly coupled multiphysics pure connectivity over-allocates, because it includes field blocks that
are symbolically zero — and applying the field-coupling mask to that case gives 840 730 nnz, **exactly**
the numerical pattern, but then the pressure diagonal is empty again. So the two features this work set
out to provide — a stable pattern and forced diagonal entries — turn out to be the same feature seen
from two sides.

### 2.3 Where the assembly time actually goes

The first guess was "mostly the scatter", from the observation that the cost per raw scatter entry is
nearly identical for very different equations. **That guess was wrong.**
`Problem::benchmark_elemental_assembly()` times the element loop with the scatter removed:

| case | elemental Jacobian | full assembly | scatter + compression |
|---|---|---|---|
| NS 3D `N=8` | 252 ms | 407 ms | 155 ms (**38 %**) |
| NS 2D `N=60` | 74 ms | 118 ms | 44 ms (38 %) |
| NS+AD 2D `N=40` | 42 ms | 75 ms | 33 ms (44 %) |
| Poisson 3D C2 `N=8` | 63 ms | 73 ms | 10 ms (**13 %**) |

The elemental evaluation is the *larger* half everywhere, and overwhelmingly so for a cheap scalar
equation. That caps what removing the scatter can buy, and it means **making the elemental JIT code
faster is a bigger lever than anything here** — see [code_generation.md](code_generation.md).

---

## 3. The design

### 3.1 `FrozenSparsity`

One per matrix of a multi-matrix assembly. Holds the CSR pattern plus a compact, slot-sorted scatter
map; `assemble_with_frozen_sparsity()` fills the output arrays straight from it, with no container, no
per-row search, no sort and no compression pass:

```cpp
std::fill(value.begin(), value.end(), 0.0);
for each element e:
    get_all_vectors_and_matrices(e, el_res, el_jac);   // unchanged
    const int *idx = &scatter_index[element_offset[e]];
    for (k = 0; k < nvar*nvar; ++k) value[idx[k]] += el_jac_flat[k];
```

It **declines** (returning false, having touched nothing) whenever the route does not fit — augmented
systems it has no spec for, compressed-column output, or any element the code generator cannot describe
— so oomph-lib's assembly remains the fallback.

**The pattern must always be a superset of the numerical one.** That invariant is what makes reuse
sound, and it is checked rather than argued: `verify_frozen_sparsity` (on by default) counts the
nonzeros of each elemental block and of the part actually scattered and refuses to continue if they
differ. *Counting*, not summing — a magnitude comparison over two different orderings reports spurious
mismatches from rounding alone, which it duly did on the first attempt.

### 3.2 Pattern tiers

| tier | source | over-allocation | diagonal |
|---|---|---|---|
| **A — connectivity** | element `eqn_number` lists only | 1.0–1.35× | free |
| **B — field-pair** | + per-code `contributes_to_jacobian` masked by local-dof→field map | ~1.00× | needs `force_diagonal` |

Tier B was measured at 17 640 nnz against a numerical 17 560 for the 2D cavity, and 28 266 against
28 186 for the coupled case — the numerical pattern plus exactly the 80 forced diagonals. On assembly
cost the coupled case goes from **+34.7 % nnz / +16.5 % time** (Tier A) to **+0.2 % / −0.2 %**.

A Tier C (codegen emitting a per-element-type `nvar × nvar` boolean mask) was considered and is
unlikely to be worth it, since B already reaches 1.00×.

### 3.3 Getting the mask into the assembly

Two vendored oomph-lib hooks, both `//FOR PYOOMPH` and recorded in `src/thirdparty/INFO_oomph-lib`:

```cpp
virtual double numerical_zero_for_sparse_assembly(const unsigned& matrix_index) const;
virtual const bool* sparsity_mask_for_element(const unsigned& matrix_index,
                                              GeneralisedElement* const& elem_pt,
                                              const unsigned& nvar) const { return 0; }
```

The mask is fetched **once per element per matrix** (hoisted above the `i`/`j` loops, so no virtual call
per entry), and the filter becomes

```cpp
if ((mask[m] && mask[m][i * nvar + j]) || std::fabs(value) > numerical_zero_for_sparse_assembly(m))
```

Note the **`||`**, not a replacement: the mask may only ever *add* entries, never remove them. A wrong
or stale mask then costs storage, not correctness — and a mask that under-reports cannot silently
truncate the Jacobian, which is the one failure mode that would be a wrong answer rather than a
slowdown.

Applied in the `maps`, `vectors_of_pairs`, `two_vectors` and `two_arrays` variants and in
`parallel_sparse_assemble`. **Not** in `lists`: it filters a second time during compression, when
merging duplicate column indices, where the elemental `(i,j)` is out of scope — feeding it explicit
zeros there makes it emit the same column index twice. pyoomph throws if that method is combined with
pruning rather than silently producing an unusable pattern.

> **Bug worth remembering.** The hook's contract lets the implementation return a pointer into a scratch
> buffer instead of allocating. With one shared buffer that is wrong: a multi-matrix assembly fetches the
> masks for *all* matrices of an element before using any of them, so the Jacobian's pointer was left
> aliasing the mass matrix's (much sparser) mask — the Jacobian silently lost its forced diagonals, and
> the whole feature appeared not to work on the eigenproblem path while working perfectly on the Newton
> path (`n_matrix == 1`). One scratch buffer *per matrix index*, and the contract in oomph's `problem.h`
> now says so explicitly.

### 3.4 The threshold and the mask must be per matrix, not per problem

A sparse assembly pass can build **several matrices at once** and they do not want the same policy.
Measured on the 2D cavity — the mass matrix is ~3× sparser than the Jacobian, because only fields
carrying a time derivative contribute to it at all:

| | J nnz | M nnz | M/J |
|---|---|---|---|
| both value-filtered | 17 560 | 6 050 | 0.34 |
| structural zeros on both | 18 178 | 18 178 | 1.00 |
| **structural on J only** | 18 178 | 6 050 | 0.33 |

Forcing J's connectivity pattern onto M inflates it threefold for no benefit. (Tier B removes the
compromise entirely: `contributes_to_mass_matrix` gives M its own tight *and* value-independent pattern.
Codegen was already computing `mass_part = diff(diffpart, __partial_t_mass_matrix)` and discarding the
fact of its non-zeroness; recording it costs one call. Verified on NS + advection–diffusion: 11 Jacobian
field pairs, **3** mass-matrix pairs — exactly the `∂t` terms, with pressure correctly absent; 3/11 =
0.27 against the measured M/J nnz ratio of 0.257.)

Two consequences worth stating:

* **The mixed policy is safe for eigenproblems.** A shift-and-invert solve factorises `J − σM`, whose
  pattern is the union of the two — and M's entries all lie inside J's structural pattern (both come
  from the same elemental blocks), so that union *is* J's pattern. It stays value-independent and
  reusable even though M is stored more tightly.
* **The Hessian must be excluded.** `assemble_hessian_tensor` used the same threshold, and it is a
  rank-3 tensor: structural zeros there would store every `(i,j,k)` triple of an element — `nvar³`,
  about 700 k entries per element for 3D Taylor-Hood, against `nvar²` for a matrix. It keeps the raw
  value filter, with a test guarding against it being wired into the policy later.

### 3.5 Invalidation, and a pattern id that is a *function* rather than a counter

The pattern is valid exactly as long as the element→equation-number map is. `jacobian_structure_id` is
bumped in `pyoomph::Problem::assign_eqn_numbers` — which already runs after mesh adaptation, remeshing,
(un)pinning, Dirichlet changes and augmented-dof changes — *and* lazily re-validated against the
assembly handler, dof counts and active residual, so that a state change nobody hooked cannot hand a
solver a stale pattern. That belt-and-braces design earned its keep: it is what makes fold tracking
invalidate correctly, with no change at any of the eight handler-installation sites.

Things that do **not** invalidate it: Newton iterations, time steps, parameter continuation, moving (but
not re-generated) meshes, changes of global parameters.

**The id is looked up in a map, not incremented.** It used to increment whenever the watched state
differed from a snapshot, so alternating between two configurations produced ids 5, 6, 7, 8 … that never
repeated — *no* cache could hit, whatever its capacity, and neither could Pardiso's symbolic
factorisation or a retained PETSc `Mat`. It is now keyed on
`(assembly-handler type, ndof, n_unaugmented_dofs, active residual)`, so returning to a configuration
returns the id it had before. Ids are still never recycled across an invalidation: the map is cleared
but the counter keeps climbing, so an id a solver held from before a renumbering cannot accidentally
match afterwards.

Keyed on the handler's **type**, not its address: the eigenproblem assembly `new`s and deletes its
`EigenProblemHandler` on every call, so an address key missed every time. That cost a measurement to
notice — the address key took the alternation benchmark from 12 rebuilds to 9, not to 3.

`Problem::frozen_sparsity_cache` correspondingly holds several `FrozenSparsity` entries keyed on
`(pattern id, matrix index)`, LRU-evicted, capacity 8 (raised automatically if a single assembly needs
more, never evicting a slot the current assembly is using). Without both halves, any setup that
alternates between two residuals — a PETSc preconditioner matrix built by `assemble_matrix`, a
multi-residual problem — rebuilt the pattern on *every* assembly, which is strictly worse than not
caching at all. Measured on four Newton/eigenproblem alternations, which involve exactly three distinct
patterns:

| | pattern rebuilds |
|---|---|
| counter id, single slot per matrix index | 12 |
| stable id keyed on handler *address* | 9 |
| **stable id keyed on handler *type*, keyed cache** | **3** (the floor) |

---

## 4. What it delivers

**Assembly only:**

| case | container | frozen | | scatter share |
|---|---|---|---|---|
| NS 3D `N=8` | 435 ms | 271 ms | **−38 %** | 176 → 9.6 ms (**18×**) |
| NS 2D `N=60` | 132 ms | 87 ms | **−35 %** | 57 → 12 ms (4.9×) |
| NS+AD 2D `N=40` | 88 ms | 51 ms | **−42 %** | 41 → 4.5 ms (9.2×) |
| Poisson 3D C2 | 75 ms | 67 ms | −11 % | 13 → 4.7 ms (2.8×) |

**End-to-end Newton solve** (3D Taylor-Hood NS, ndof 10 853, pardiso, identical converged solutions):
3.031 s → 2.381 s with the structural pattern and solver reuse → **1.723 s** with the frozen scatter,
i.e. **−43 % cumulative**. That beats the ≤ −23 % §2.3 predicted, because removing ~95 % of the scatter
compounds with the symbolic-factorisation reuse across every Newton step.

Structural zeros alone (the cheap first step — `Numerical_zero_for_sparse_assembly` made negative, which
turns *every existing* assembly routine into a Tier-A structural assembly with no change to any assembly
loop) are worth −20 % on pardiso and −22 % on petsc_mumps, bit-identical solutions.

**Whole pipeline against `main`:** `citools/test_all_tutorial_scripts.py --quick-test --no-petsc`, 126
scripts, each branch built from scratch and run twice (the first run warms the JIT cache):

| | before | after | |
|---|---|---|---|
| Wall clock | 286.4 s | 261.4 s | **−8.7 %** |
| User CPU | 312.2 s | 280.9 s | **−10.0 %** |
| Peak RSS | 812 992 kB | 810 836 kB | −0.3 % |

**Read that −8.7 % in context, in both directions.** `--quick-test` stops each script after its first
successful Newton solve, so the suite is deliberately dominated by JIT-compiling the generated element
code, process startup, mesh generation and output writing — assembly and linear solves are a small slice
of it. A ~9 % gain on *that* workload is a lower bound on what a production run sees; the −43 % on an
actual Newton solve is the upper end.

**Peak memory is unchanged**, which was the open question: the frozen scatter map costs about 8 bytes
per raw elemental entry, and at tutorial problem sizes that is lost in the noise. It is not free at
scale — for the 3D Taylor-Hood benchmark (2.7 M raw entries) it is ~21 MB per matrix — so it remains
worth watching on genuinely large 3D problems, where `use_frozen_sparsity=False` is the escape hatch.

### 4.1 Two things worth recording

* **A false negative nearly buried this.** The first measurements said the frozen path was 11–16 %
  *slower*. It was never running: `build_frozen_sparsity` bailed out on the Dirichlet-condition
  elements, whose `current_res_jac` is −1 and whose mask was therefore NULL — so every assembly paid for
  a full (two-pass, sorting) pattern build and then fell back. Those elements contribute nothing, so the
  fix is to return an all-zero mask rather than "no description". **The lesson: the fast path needs a
  positive signal that it engaged, not just a flag that permits it.** Hence
  `_get_frozen_sparsity_nnz()` and `_get_distributed_frozen_rebuild_count()`. This recurs — see §6.3.
* **The frozen path emits canonically sorted CSR**, where the container assembly emits insertion order.
  Free bonus: `sort_indices()` in the Pardiso reuse check becomes a no-op.

### 4.2 Still off by default

`keep_structural_zeros` defaults to **off**. It is measurably free (≤ +0.2 % nnz and within noise on
time for every benchmarked case, with bit-identical solutions), so defaulting it on is the natural next
step — but flipping it also turns on solver symbolic reuse for every user, and
[linear_solvers.md](linear_solvers.md) §4 is the reason to be careful about that: stored zeros on a
saddle-point diagonal change what MUMPS' analysis plans.

---

## 5. The distributed path

`parallel_sparse_assemble` works in four stages: build `my_eqns` (the sorted set of *global* equations
this rank contributes to, gathered from its non-halo elements — note this is **not** the set it owns, a
non-halo element can touch a halo node); accumulate locally, with a **binary search into `my_eqns` per
elemental dof** and a **linear scan of the row** to insert; build the send plan; and exchange — each
rank ships the rows it does not own to their owner, row starts, **column indices** and values, and the
owner merges them into its local CSR block. Stages 1 and 3 are purely structural; the column-index half
of stage 4 becomes redundant as soon as the pattern is frozen.

**Where the time actually went** (2D cavity `N=40`, ndof 14 162, `--distribute`):

| ranks | elemental | total assembly | scatter + exchange |
|---|---|---|---|
| 1 (serial, frozen) | 35.0 ms | 37.1 ms | **2.2 ms (6 %)** |
| 2 (oomph) | 20.0 ms | 39.8 ms | **19.9 ms (50 %)** |
| 4 (oomph) | 13.4 ms | 24.9 ms | **11.7 ms (47 %)** |

Half a distributed assembly was scatter and exchange, against 6 % once the serial path is frozen — and
at two ranks the total was *worse than serial* even though the elemental work had halved: **the overhead
ate the entire parallel gain.** Instrumenting the local stage directly settled which half it was, the
opposite way round from the guess: at 2 ranks, local scatter 6.7 ms against exchange + owner-side merge
~12.5 ms. So freezing the local scatter alone caps out at roughly −17 %; the exchange is the real
target, and both halves were needed for the work to be worth its risk.

> The diagnostic oomph-lib already has for this (`enable_doc_imbalance_in_parallel_assembly`) prints
> through `oomph_info`, which pyoomph redirects — so it produced nothing and a direct timer was needed.

**Result:**

| ranks | oomph-lib | frozen | | vs. serial |
|---|---|---|---|---|
| 1 (serial frozen, reference) | — | 36.4 ms | | 1.00× |
| 2 | 49.1 ms | **20.7 ms** | −58 % | **1.76×** |
| 4 | 23.3 ms | **12.0 ms** | −49 % | **3.03×** |

Distributed assembly now pays for itself from two ranks up, and scales. End to end (one assembly plus
one MUMPS solve per Newton step, identical solutions): **−12 % at 2 ranks** and **−41 % at 4**. The solve
dominates the step at 2 ranks, which is why the assembly's −57 % shows up as −12 % there.

It beat the estimate because freezing does not merely remove the exchange *traffic*, it removes the
owner-side **merge**: oomph-lib merges each incoming entry into the row built so far by *linearly
rescanning that row* — quadratic in the row length, on every assembly, with chunked reallocation. Frozen,
that whole loop is one precomputed permutation and a scatter-add. Two further consequences fell out for
free: the send buffers are **contiguous slices** (`my_eqns` is sorted and the target distribution gives
each rank a contiguous global range, so the rows destined for rank *p* are a single run — the same
property oomph's own `first_eqn_element_for_proc` relies on, just never exploited for the payload); and
the final column indices come out **sorted within each row**, which is what PETSc and the direct solvers
want anyway.

**Verified against oomph-lib directly**, not just at the solution level: `--mode compare-distributed` in
`tests/mpi_structural_worker.py` assembles the same converged state through both routes and compares the
local CSRs as `{column: value}` maps per row (not element-wise — the column *order* legitimately
differs). At 2, 3 and 4 ranks in 2D and 3D: identical nnz, no entry present in one and absent from the
other, worst value difference **3.5e-18**. A solution-level check would not have been enough — Newton
converges to the right answer from a slightly wrong Jacobian, so a defective merge permutation would
have passed.

**Everything from the freshness check onwards is collective**, so both things that could diverge between
ranks are put to a vote rather than assumed: whether the plan needs rebuilding (`nrow_local`, `first_row`
and `nelement` are per-rank quantities) and whether the build succeeded. There is a second, internal vote
inside the build, immediately before the first collective — every bail-out up to that point is decided
per rank, and a rank that bailed while the others entered `MPI_Alltoall` would hang them. **Half the
ranks in the frozen exchange and half in oomph-lib's is a deadlock, not a wrong answer**, which is why
this is voted on rather than checked afterwards. Two extra `MPI_Allreduce`s of one `int`, costing
nothing measurable against a 20 ms assembly.

**REPLICATED runs** (`--distribute` absent but `nproc > 1`, every rank holding the whole mesh) are covered
too, and were the subtlest thing here. They differ only in which elements a rank evaluates: a slice of the
element list rather than the non-halo ones. That slice is *not* a function of the equation numbering —
oomph-lib re-tunes `First_el_for_assembly` from measured per-element timings whenever its own routine runs
(`recompute_load_balanced_assembly`, at the tail of `parallel_sparse_assemble`) — so a plan built for one
slice would assemble a different part of the mesh through the scatter map of another: a wrong Jacobian
*and* a wrong residual, which Newton then converges to a wrong state from rather than failing on. The
range is therefore recorded in the plan and compared as part of its collective freshness vote, the same
way `nrow_local` and `nelement` are. The first version excluded the mode outright instead; the exclusion
was the safe move but not a necessary one, and it was what kept static condensation out of that mode
([static_condensation.md](static_condensation.md) §9.9).

`tests/test_mpi_structural_assembly.py::test_frozen_replicated_assembly_matches_oomph` compares the local
CSR against oomph-lib's own at 2/3/4 ranks in 2d and 3d, and asserts that the re-tune actually happened in
the run — otherwise the comparison would pass without ever exercising the staleness it exists to catch.

**What it does not cover**, each falling back to oomph-lib: augmented systems on a distributed problem,
and any element the code generator cannot describe.

**Residual-only assembly** was also frozen, and the premise for doing so was half wrong. Measured
serially it is **96–101 % elemental JIT evaluation**, so there is nothing there to win. Under MPI it was
a different story: oomph routes `get_residuals()` through the whole of `parallel_sparse_assemble()` with
zero matrices — recomputing `my_eqns`, building the array-of-arrays, exchanging equation numbers and
merging by bisection per row, all to sum a vector, once per Newton step. Overhead 26 % at 2 ranks and
**45 %** at 4, roughly *constant* in absolute terms as ranks grow. Frozen, residual assembly is −20 % at
both, and the end-to-end distributed Newton step goes to −18 % / −27 %.

> Worth noting for anyone re-treading this: the Newton loop evaluates the residual **twice at every
> state** — once standalone for the convergence check, once again inside the next `get_jacobian`. That
> looks like free money and is not. The bundled one is nearly free (same elemental pass), so the only way
> to remove the duplication is to restructure the loop to assemble R+J before the convergence test, which
> wastes a Jacobian on the final step. For *n* Newton steps that trades *n*×R against one marginal J: a
> win in 2D (R = 17 ms, marginal J = 20 ms) but a clear loss in 3D, where the marginal J is 895 ms
> against a 179 ms residual and it would need *n* ≥ 6 to break even.

`jacobian_structure_id` equality across ranks is asserted with `MPI_Allreduce(MIN/MAX)` under `PARANOID`
in `assign_eqn_numbers`. Deliberately **not** in `get_jacobian_structure_id()`: that one is called from
Python and may legitimately be read on a single rank, where a collective would be the very deadlock it is
meant to prevent.

---

## 6. Augmented systems

### 6.1 Multi-assembly

`sparse_assemble_row_or_column_compressed_base_problem` — the routine behind `assemble_multiassembly`,
and so behind every Python-level bifurcation tracker — was the worst assembly in the codebase: a
`std::map` per row, with 30–86 % of its time outside the elemental evaluation. The key structural fact is
that **all of its matrices share one pattern**: the Jacobian, its parameter derivative and a
Hessian-vector product are all derivatives of the same residual w.r.t. the dofs, so one frozen pattern
serves all of them and the scatter writes the same slots into several value arrays. The mask is taken
from matrix 0 for every matrix — exact for the Jacobian-derived ones, a superset for any
mass-matrix-derived one, which is the safe direction and is checked per element anyway.

Bratu fold tracking, `R + J + dRdp + dJdp + dJdU` in one pass: multi-assembly −30 % at ndof 6963 and
−34 % at 19603, full fold solve −28 %, with the tracked fold identical to 10 digits.

> **A false alarm worth recording, because it will recur.** The first comparison said the Hessian-vector
> product was exactly *negated* by the fast path — `max|A + B| = 3.5e-17`, all 1521 entries wrong. It was
> the test, not the code: it contracted the Hessian with `get_real_eigenvector_guess()`, and an
> eigenvector is only defined up to sign, so the two runs were handed opposite vectors. The product is
> linear in that vector, hence the exact negation. **Anything comparing Hessian-vector products across
> two runs must contract with a fixed vector.**

### 6.2 The C++ trackers: `AugmentedBlockSpec`

The C++ handlers present an *augmented elemental block* — `2·raw+1` for fold, `3·raw+2` for Hopf,
`raw·nT+1` for a periodic orbit — and fill it by hand. `Problem::sparsity_mask_for_element` used to
refuse them outright, because the element's field description covers `raw_ndof` dofs and the handler asks
about a block several times larger.

`AugmentedBlockSpec` describes the augmented block in terms of the patterns it is *made of*: a group
layout (how the augmented dofs partition into `raw`-sized groups plus scalar border dofs) and, per block,
a list of `Term{kind, residual}` where kind is one of `Empty`, `Jacobian`/`JacobianT`,
`MassMatrix`/`MassMatrixT`, `Hessian`/`HessianT`, `Dense` or `Diagonal`.
`Problem::augmented_sparsity_mask_for_element` tiles the raw masks into the augmented one, and from there
`build_frozen_sparsity` works unchanged.

Measured on Kuramoto-Sivashinsky fold tracking (ndof 15 229 augmented): augmented assembly **−45 %**
(113.3 → 62.5 ms), a re-solve including the factorisation **−31 %**, a full fold-tracking solve **−24 %**,
same fold point to 12 digits. Hopf on Lorenz: every structural block **identical** to the value-filtered
one, Hopf point unchanged.

Four things this cost, each of which changed the design:

* **`Hessian` had to be its own kind.** Declaring Hessian blocks `Jacobian`-patterned is *true but
  loose*: `d²R_i/du_j du_k ≠ 0` implies `dR_i/du_j` is not identically zero, so it is a subset — but
  every LINEAR term of the residual is in J and in no Hessian at all. On Kuramoto-Sivashinsky the
  biharmonic and Laplacian carry most of J and none of the Hessian, so the contracted block is four times
  sparser, and 96 % of a +34 % excess sat in that one block. Codegen now emits
  `contributes_to_hessian[res][fi][fj]`, marked where the second derivative is generated (both `(res,f)`
  and `(res,f2)`, since `d²R/df df2` is symmetric and a contraction sums over one index). That brings the
  augmented pattern within **1.4 %** of the numerically exact one, all of it the two border columns.
* **A block needs several patterns OR'd together, and a *residual index* per term.** Pitchfork's base
  block is the base Jacobian *plus*, in `improved_pitchfork_tracking_on_unstructured_meshes` mode, a
  Hessian of the symmetry residual. And `AzimuthalSymmetryBreakingResidualContributionList` holds **three**
  residual indices at once — `jacobian_real(m,n) - Omega*M_imag(m,n)` is the real azimuthal Jacobian OR'd
  with the imaginary azimuthal mass matrix, which cannot be said in a one-kind-per-block vocabulary. Each
  block entry therefore carries a residual index (`-1` = "the one currently being assembled", which is
  what fold and Hopf use throughout), and the builder collects the distinct **(matrix, residual)** pairs
  its spec refers to *before* filling any of them — so buffer indices stay valid while several are held
  at once.
* **A negative residual index means "this element has no such contribution", not "undescribable".**
  Returning false there abandoned the pattern for the whole MESH, because `build_frozen_sparsity` gives
  up as soon as one element cannot be described. On Rayleigh-Bénard that was every element: 33 012
  declines and no frozen path at all — while the script still passed, because falling back is silent.
* **`Diagonal` had to exist, and then had to stop requiring `gr == gc`.** The axis boundary conditions
  write an identity over the rows in `base_dofs_forced_zero`/`eigen_dofs_forced_zero`; those entries come
  from no residual at all, so no coupling table can predict them. And the periodic orbit's wrap-around
  `u(nT-1) - u(0) = 0` is an identity landing on the diagonal of an *off-diagonal* block.

**Without an analytic Hessian, refuse rather than approximate.** The first version fell back to the
Jacobian pattern for Hessian blocks when no analytic Hessian had been generated. That is wrong-headed:
the Jacobian bounds an *analytic* Hessian, but nothing bounds what a *finite-differenced* one writes, and
the handler finite-differences precisely when codegen has emitted no Hessian. The builder returns NULL in
that case, so the whole augmented pattern is declined.

**Neither scalar row has a diagonal, deliberately** — `C.Phi - 1 = 0` involves neither the parameter nor
Omega, and declaring those `Dense` would manufacture the stored zero diagonal that
[linear_solvers.md](linear_solvers.md) §4 is about.

### 6.3 Periodic orbits

The last tracker, and the most valuable: on the 1D Brusselator
(`tests/benchmarks/bench_periodic_orbit_1d.py`) the assembly is **64–74 % of an orbit solve**, far above
the 30–50 % typical elsewhere, because the orbit matrix is so sparse (0.17 % at N=80/NT=60) that the
factorisation is cheap next to it.

> This could not be judged from the tutorials. All three that use periodic orbits drive them from a 3-dof
> ODE where the entire solve is sub-millisecond; extrapolating from those suggested it did not matter,
> which is why the PDE case was written.

Two things were recorded wrongly first. "Periodic orbits never form the augmented elemental block, so a
spec would never be consulted" rested on the comment *"Periodic orbits would have very huge elemental
Jacobians, so we must assemble them with block jacobians"* — which describes the band-matrix path, and
that path is `#define`d **out** and never implemented (its body is
`//throw_runtime_error("TODO: Fill it in")`). The path that runs builds a dense `nvar × nvar`
`oomph::DenseMatrix` with `nvar = raw*nT+1`, so the augmented block is formed after all. The real reason a
spec would never have been consulted was that the dispatcher tried the orbit routine *before*
`assemble_with_frozen_sparsity`; reordering it is the whole enabling change.

The augmented unknowns are nT copies of the raw dof set followed by the scalar period T, so the block is
an (nT+1)×(nT+1) grid. Within a coupled pair of time nodes the pattern is the base Jacobian and mass
matrix, so the only thing the spec describes is **which time nodes couple**, read off the same data the
assembly loops use:

| mode | coupling |
|---|---|
| collocation (default) | pairs within a time element of `time_mesh`, via `TimeNode::get_index()` |
| B-spline | pairs supported on a common basis element, via `get_integration_info()`'s `indices` |
| Floquet | `t → t` and `t → t+1` (J and M), last row flushed and replaced by the wrap identity |
| central, BDF2 | `t → t` (J and M), plus the `dU/ds` stencil `FD_ds_inds[t]` — **mass matrix only** |

Two subtleties, both caught by measurement rather than by reading:

* **Rows and columns do not run over the same node set.** A collocation time element with `nnode()` nodes
  carries only `nnode()-1` collocation points, and the equation of collocation point `inode` is written
  into the row of node `inode`. So only the *first* `nnode()-1` nodes are equation rows, while all of
  them appear as columns. Declaring the last node as a row too is legal — it only stores zeros — but it
  cost +25 % nnz, and the give-away was that blocks (3,0), (6,3), (9,6), (12,9) came out **100 %**
  structural zeros, i.e. exactly the element-boundary nodes at order 3.
* **The FD stencil multiplies the mass matrix alone.** The first version guessed `t, t±1, FD_ds_inds[t]`
  with both J and M. Correct, but it cost +44 % nnz for central and +89 % for BDF2. The nodal modes have
  no `t → t+1` term at all — the spatial operator is evaluated at the node itself — and the stencil
  carries only M.

Verified frozen vs unfrozen at the **identical state** (toggling the flag between two assemblies in one
run, rather than comparing two independent runs — those drift, and a 7.5e-03 difference that looked like
a defect was just that). Brusselator N=20, NT=12: bit-identical `max abs dJ = 0` in all six modes, nnz
within +0.01 % except BDF2's +0.48 % (the wrap identity, which that mode does not use). The one extra
entry everywhere is the (T,T) diagonal, declared deliberately so the period row has a stored diagonal.

Speedup, collocation: assembly −71 % / −77 % / −75 % and orbit solve −69 % / −71 % / −66 % at orbit ndof
1067 / 5023 / 19643.

### 6.4 Three defects the tracker work exposed

1. **The mass table was unreachable.** `sparsity_mask_for_element` refused `matrix_index > 0` unless an
   `EigenProblemHandler` was active. That guard is about the second matrix of a multi-matrix *assembly*;
   using the mass *coupling table* as a pattern source is meaningful whatever is assembling. Until a
   `MASK_MASS` sentinel was added, every Hopf spec silently fell back and the frozen path never engaged —
   and a first "every block identical" reading was worthless, because both sides of the comparison had
   fallen back. **Check that the fast path engaged before believing a comparison**; this is §4.1's false
   negative in a new costume.
2. **Finite-differenced Hessians have no table.** The plan said the FD path "fills the same block
   positions, so the spec is independent of it — but that must be verified rather than assumed". It is
   not independent: with no analytic Hessian, codegen emits none, `contributes_to_hessian` is empty, and
   the block is masked away while the handler fills it by finite differences.
3. **Sentinel values as array indices.** The `resind < 0` shortcut sized its scratch buffer with
   `resize(matrix_index + 1)`, and `matrix_index` can be `(unsigned)-5`. That asks for ~4 billion entries
   and surfaced as `std::bad_alloc` in an ordinary continuation step, nowhere near the sparsity code.

And one failure that resisted for a while: a fold tracker on a moving mesh tripped the per-element
verification at element 898 with the parameter row against the eigenvector columns missing — a block the
spec declares `Dense` and the tiling demonstrably writes. **Instrumenting the *build* rather than the
failure settled it**: the mask for that element was entirely zero. The cause is the "no active residual"
shortcut in `sparsity_mask_for_element`, which returns an all-zero mask and sat *before* the augmented
dispatch. That is correct for the raw block — such an element contributes nothing to the base Jacobian —
but a bifurcation handler still writes the border of the augmented block for *every* element, because the
normalisation row is a property of the handler, not of the element's residual. Hoisting the dispatch
above the shortcut fixed it. The lesson is the same as §4.1: the contradiction was only resolvable by
instrumenting where the pattern is **built**, not where it is used. Reasoning about the spec and the
tiling — both of which were correct — could not have found it.

### 6.5 The Python trackers' bordering is still not frozen

The Python trackers' *constituent* matrices already go through the frozen multi-assembly, but the
**bordering** is redone in scipy on every step:

```python
Jaug = scipy.sparse.block_array([[J, None, col(dRdP)], [HV, J, col(dJdP@V)], [None, row(...), None]]).tocsr()
```

`block_array(...).tocsr()` rebuilds the whole augmented CSR — a COO build, a sort and a duplicate merge —
every Newton step, from blocks whose patterns are already known and fixed. A frozen bordered assembly
would build the augmented `indptr`/`indices` once per structure id together with a scatter map from each
constituent block's value array, then copy values only.

**One trap on that side.** The border blocks are built as `csr_matrix(dense_array.reshape(-1,1))`, which
*drops the zeros of a dense vector*. So the border column's pattern today depends on the values of
`dRdP` — it is not stable, and a frozen version must declare those blocks structurally dense rather than
inheriting whatever `csr_matrix` happened to keep.

---

## 7. What the stored zeros actually are

A frozen pattern must be a superset of the numerically nonzero entries, so it always carries some stored
zeros. `_get_frozen_sparsity_fill_stats()` measures how many; across the tutorial suite it is 6.0 % of
8.6e9 slots, median **0.000 %** per script — most patterns are exactly tight — with a few far above
(rivulet 39.8 %, rising_bubble 29.1 %, eigendynamics 28.4 %).

`_get_frozen_fill_breakdown()` (filled only when `PYOOMPH_FROZEN_FILL_BREAKDOWN` is set) attributes every
scattered entry to its (row class, column class) pair. That is what separates the two causes, which want
opposite treatment.

**Cause 1: unattributed dofs — a real defect, fixed.** On the DG convection-diffusion tutorial 33 % of the
pattern was stored zeros, and every one sat in a row or column of an `<unattributed>` dof of the
`_internal_facets_` element, while the properly attributed block was exactly tight. An interface element
works on dofs belonging to other elements and the attribution walk could not see them, so they kept the
-1 sentinel, which the mask must read as "coupled to everything". Fixed by adopting the source element's
own attribution through the equation map that already exists (rivulet 39.8 → 37.0 %, eigendynamics
28.4 → 27.2 %).

**Cause 2: a field that is identically zero — inherent, and must NOT be pruned.** What remains is
concentrated in blocks like `domain/velocity_phi × domain/velocity_y` (87.4 % zero) and
`liquid/velocity_normal × liquid/coordinate_x` (67.1 %). The first reading was that this is
azimuthal-mode-dependent coupling, and that folding the mode number into `jacobian_structure_id` would let
the pattern be tightened per mode. **That was wrong.** Measuring the base state settles it:

```
domain/velocity_phi        691 dofs   max|value| = 0.000e+00   100% exactly zero
liquid/velocity_normal     591 dofs   max|value| = 0.000e+00   100% exactly zero
```

The blocks are empty because the base state has no swirl and no normal flow, so those fields are
identically zero and appear as a multiplicative factor in exactly those Jacobian terms. **The coupling is
structurally real; it is the VALUE that vanishes.** Pruning would be actively unsafe: the moment the
solution develops swirl or normal flow — which in a stability analysis is precisely what is being looked
for, since the perturbation lives in those components — the entries become nonzero and a pattern pruned
on the current values would truncate the Jacobian.

**The general rule:** a block that is 100 % zero because a coupling was never written is a table defect
worth fixing; a block that is mostly zero because a field currently happens to vanish is not, and the two
are indistinguishable without the per-block breakdown.

**A possible feature this suggests.** The framework cannot know that `velocity_phi` stays zero — nothing
in the residuals forbids it. The *user* often does, and for a structural reason (in the rivulet case the
slip-length condition prevents any non-zero `velocity_normal` in the base solution at all). So there is
room for an opt-in "this field is identically zero in the base state" declaration, letting the base-state
block drop those rows and columns: 87–90 % of those blocks, and the bulk of what is left after the
attribution fix. Three things it would have to respect, all of which the design already provides for:
**base state only** (patterns are already per-matrix), **part of the pattern key** (or a pattern built
with it could be handed to an assembly running without it), and **opt-in, never inferred** — deriving it
from observed values is exactly the unsafe pruning above. What makes it safe to offer at all is that the
per-element verification already exists: a wrong declaration does not silently truncate the Jacobian, it
refuses to assemble and names the positions that escaped the pattern.

---

## 8. Correctness gates

`tests/test_structural_assembly.py` (13 tests, <4 s, in the fast run) plus
`tests/test_mpi_structural_assembly.py` + `tests/mpi_structural_worker.py` (4 tests, 2D and 3D at `-n 2`
and `-n 4`).

1. **Superset invariant.** `structural_pattern ⊇ numerical_pattern` at several dof states, and equal to
   the element-connectivity pattern recomputed independently in Python — so the test does not merely
   compare the implementation against itself.
2. **Bit-identical solutions**, with the pattern route enabled and disabled; explicit zeros must not
   change any result.
3. **Differential test against the oomph path**: same values on the common pattern, extra entries all
   exactly 0.0.
4. **Invalidation coverage.** After each of mesh adaptation, remeshing, `pin`/`unpin`, Dirichlet changes,
   switching `_solved_residual`, installing/removing a bifurcation handler, adding augmented dofs —
   `structure_id` must have changed and the rebuilt pattern must still satisfy (1). Conversely it must
   *not* change across Newton steps or arclength continuation steps, or the reuse never fires. **This is
   the highest-risk area: a missed invalidation is a silent wrong-answer bug, not a crash.**
5. **Per-matrix policy.** The mass matrix keeps its own pattern, its values are unaffected by the
   Jacobian's policy, its entries lie inside the Jacobian's structural pattern, and the Hessian tensor is
   not dragged in (§3.4).
6. **MPI**: all of the above at 2 and 4 ranks, plus cross-rank agreement on the gathered residual and
   global observables, `jacobian_structure_id` identical on all ranks after every renumbering, and the
   assembled distributed Jacobian equal to the serial one. The worker solves **twice** on purpose — the
   first solve can never reuse a factorisation, so a single-solve test would pass with the reuse path
   dead.
7. **Tutorial pipeline**, `--quick-test --no-petsc`, run twice: once with the shipped default (the
   regression gate for the oomph-lib patch, whose new filter and mask-fetch lines execute on every
   assembly regardless), and once with `keep_structural_zeros` forced on for every problem via a
   `sitecustomize.py`. All 126 scripts pass both ways, and 119 of the 126 logs verifiably report
   `keep_structural_zeros=True prune=True forcediag=True` (the other 7 never initialise a `Problem`).
   This is the broadest evidence available that the feature is safe on real problems — moving meshes,
   interfaces, eigenproblems, bifurcation tracking and continuation all appear in that set.

---

## 9. Open

* **Default `keep_structural_zeros` on** (§4.2), once the interaction with MUMPS' analysis is settled.
* **The Python trackers' bordering** (§6.5).
* **A values-only `J − σM` update** for shift-and-invert sweeps: `keep_structural_zeros_in_mass_matrix`
  would make it a pure `axpy` on the value arrays at the price of a 3× inflated M — measure both.
* **Memory footprint of the scatter map on large 3D problems.** The compact slot-sorted form is much
  smaller than the per-position table it replaced, and the distributed plan adds a per-rank
  `merge_perm`/`final_col`/`local_col` of order a few times nnz in `int`s. Never measured on a problem
  large enough to care; the binary-search fallback exists if it turns out to matter.
* **Exposing the pattern to Python** (`problem.get_jacobian_sparsity()` → `(indptr, indices)`), so
  `CustomAssemblyBase` implementors and the eigen matrix manipulators — which do `csr` surgery row by row
  — can work on a fixed pattern too. Nice-to-have; nothing depends on it.
* **Threaded assembly.** A precomputed scatter index makes a colour- or lock-free parallel scatter
  feasible. Follow-up.
* **Extending the frozen path to `..._for_periodic_orbit`'s remaining fallback** and to the augmented
  systems on a *distributed* problem — see [mpi_augmented_systems.md](mpi_augmented_systems.md).

**Found on the way, not part of this work:** distributed eigensolving is broken in the Python layer.
`GenericEigenSolver.get_J_M_n_and_type()` wraps the *row-local* CSR that
`assemble_eigenproblem_matrices()` returns in a `csr_matrix` of *global* shape, and scipy rejects it. It
fails identically with the frozen route on or off; the C++ side is fine, as §5 verified below this layer.
Also worth knowing when testing that layer: the arrays `assemble_eigenproblem_matrices()` returns are
**views** into `eigen_{Mass,Jacobian}MatrixPt`, which the next call deletes and reallocates — comparing
two calls without copying first compares freed memory against live memory and reports pure noise. It
briefly looked like a serious defect in this very code.
