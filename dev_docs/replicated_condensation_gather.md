# Static condensation replicated: gathering a component whose rows straddle the row split

Status: **planned, not implemented — and no longer needed for the case it was written for.** The
Crouzeix–Raviart selection of §1 is served replicated by numbering the dofs differently instead; see
[dof_ordering.md](dof_ordering.md) and §2.2 below. What remains for this plan is a selection that
renumbering *cannot* make contiguous, e.g. interior-penalty DG.

Everything below is reasoned against the code, not measured; the cost figures are arithmetic on the
flagship case, not benchmarks. Companion to [static_condensation.md](static_condensation.md), whose
§9.9 this plan would rewrite.

## 1. What is missing

Replicated MPI (`mpirun` without `--distribute`) is served today — but only for selections whose dofs
are element **internal**. The flagship **Crouzeix–Raviart** selection is refused, and
`citools/test_all_tutorial_scripts.py:290` skips `cr_static_condensation.py` on the replicated pass
because of it.

Replicated, every rank holds the whole mesh, every `Data` and the whole dof vector; only the element
loop and the Jacobian's **rows** are split, the latter uniformly and **contiguously**
(`LinearAlgebraDistribution(comm, nrow, true)`, `problem.cc:417`). A component can currently only be
eliminated on the rank that owns *all* of its L rows. `Problem::condensation_row_cuts()`
(`src/problem.cpp:4532`) moves the cut points off the blocks, which works while a block's equations sit
close together. oomph numbers **every nodal value before any internal one**, so a CR block (bubble
velocity nodal, DL pressure modes internal) has its two halves hundreds of equations apart: no
contiguous split keeps one whole, `condensation_row_cuts` takes its "block longer than a rank's share"
escape (`src/problem.cpp:4623`), and the straddling guard (`src/problem.cpp:4838`) refuses collectively.

The goal is that a mixed nodal/internal selection condenses under plain `mpirun` and reproduces the
serial answer. **Distributed-mode straddling stays refused**: there it genuinely means the selection is
not element-local, and serving it would be a distributed dense solve per component.

## 2. Approach: elect a component owner, gather its L rows

Replicated only, drop "one rank owns all L rows of a component". Each component elects an **owner**, and
the ranks owning its other L rows ship those rows to it each assembly. The owner then forms
`X_C = A_LL⁻¹A_LE` and `y_C` exactly as it does today, and everything downstream — the `X`/`y` exchange
to the E-row owners (tag 84), the reconstruction `MPI_Allgatherv` — is reused unchanged.

What makes this sound replicated and not distributed: every rank holds every `Data` and every element,
so *any* rank may hold `X_C`, reconstruct an increment and (through the existing allgather) have it
applied everywhere. The only thing distributed is the row block, and the frozen distributed assembly
already guarantees each rank's owned rows are **complete**, so the row owner holds precisely what the
component owner lacks.

Two refinements keep the change small:

* **The sender densifies into component coordinates.** Instead of shipping a row's pattern and having
  the owner merge it into slot machinery, the sender ships, per foreign L row, a dense
  `nL + nE + 1` vector: `A[row, L_eqns[0..nL)]`, `A[row, E_eqns[0..nE)]`, `r[row]`, in the component's
  own ordering. The owner `memcpy`s it into row *i* of `LL` and `RHS`. Slot provenance then attaches to
  a **row**, not to an entry, so `LL_slots`/`LE_slots` stay monomorphic and the inner
  `nL×(nL+nE)` gather loop keeps its shape.
* **Only *touched* components are published.** The globalisation allgather carries membership only for
  components with at least one cross-rank L–L edge, plus the edges. For HDG there are none, so it is a
  zero-length `MPI_Allgatherv` and "element-internal selections cost no extra communication replicated"
  stays literally true — and assertable.

### 2.1 Why the component structure is taken from the pattern, not from the mask

`condensation_row_cuts()` computes components from the symbolic mask with no communication at all, and
reusing that would make the whole plan build rank-local. It must not be done: **the mask may only add
entries to the assembled pattern, never remove them** (`src/problem.cpp:7553` — the pattern is the union
of the mask and whatever was numerically nonzero). A mask-derived decomposition can therefore be a
strict *subset* of the true one, i.e. a component silently split in two and a Schur complement wrong by
an amount nobody measures. The mask stays what it is: good enough to place the cuts, not to decide what
may be inverted.

### 2.2 Rejected: renumber the equations instead — SUPERSEDED, see [dof_ordering.md](dof_ordering.md)

*This section was wrong, and the renumbering it rejects has since been built and shipped. Kept for
the argument, which is sound about the thing it actually describes.*

The obvious question is why not renumber so a component's dofs are contiguous and let
`condensation_row_cuts()` do the whole job — zero new condensation machinery. It means permuting
`Data::Eqn_number` and `Problem::Dof_pt` after `assign_eqn_numbers`, i.e. changing an oomph-core
invariant behind a pyoomph flag, and then proving *forever* that everything storing a dof index
(continuation's `Dof_current`/`Dof_derivative`, arclength, bifurcation tracking, `n_unaugmented_dofs`,
`dirichlet_info.get_global_pinned_equation_set()`, `describe_dofs`) is built after the permutation.
Global risk for a gain that exists only in replicated MPI.

**What that misses is the word “after”.** The permutation does not have to happen after
`assign_eqn_numbers` returns; it happens **inside** it, from a new virtual
(`Problem::reorder_global_eqn_numbers`) called immediately after `Mesh::assign_global_eqn_numbers` and
before anything reads the result. Every consumer on that list is built below that point, so there is
nothing to prove forever: a permutation applied there is simply what the numbering *is*. The same
placement serves a distributed run, where the numbers are still rank-local and
`synchronise_eqn_numbers` shifts them afterwards, so each rank's range stays contiguous.

With `problem.dof_ordering = ElementBlockOrdering("domain/velocity_*", "domain/pressure")` the
Crouzeix–Raviart case of §1 condenses replicated at 2 and 4 ranks and reproduces the serial answer
(`tests/test_mpi_dof_ordering_rowsplit.py`). **This plan is therefore no longer needed for CR.** What
it is still needed for is a selection that renumbering cannot make contiguous — an interior-penalty DG
one, where the component genuinely percolates the mesh — and §5's globalised size guard is what would
keep that case from attempting a dense LU of the whole mesh. Re-read §1 and §8 in that light.

### 2.3 Rejected: re-assemble the component's elements locally

§9.2 of [static_condensation.md](static_condensation.md) rejected halo re-assembly distributed because
its completeness condition cannot be checked. Replicated the completeness objection vanishes — every
rank holds every element — but a stronger one takes its place: **the matrix condensation is handed is
not the sum of the elemental Jacobians.** `remove_dirichlets_by_matrix_manipulation`
(`src/problem.cpp:1499`, called at `:1745`, deliberately *before* the elimination) replaces pinned rows
by the identity and zeroes every pinned dof's **column**. Re-assembling elementally reconstructs entries
the true system has zeroed; that they multiply a zero increment is one more "nearly always true" claim,
and hanging-node constraint transformation would have to be reimplemented in a second place besides.

## 3. Plan build (`build_distributed_condensation_plan`, `src/problem.cpp:4667`)

Replicated branch; the distributed path keeps its current code throughout.

1. **Exchange 1 disappears.** The selection is already identical on every rank (every rank holds every
   `Data`, and `condense_element_private_dofs`'s reverse scan sees the same elements — what
   `condensation_row_cuts()` already relies on), so the tag-80/81 alltoall asking a column's owner "is
   this condensed?" is answered locally: build `is_cond` from the **whole** selection instead of
   restricting to owned rows at `src/problem.cpp:4740`.
   Keep a **voted** build-time assertion that every L equation this rank owns is in its own `l_index` —
   replicated the selections *should* agree, but the rank-local reverse scan is exactly why Exchange 1
   exists, and a disagreement here is a hang, not a wrong answer.
2. **Cross-rank edges instead of the refusal.** The union–find over owned L rows runs as now; where an
   owned L row has an L column this rank does not own, record `(row, col)` instead of pushing the
   equation into `straddling`.
3. **Globalise the decomposition.** One `MPI_Allgather` of a count plus one `MPI_Allgatherv` carrying,
   per rank, `(L_eqn, local representative)` for the members of *touched* components and the cross-rank
   edges. Every rank replays the local unions and the edges into a union–find over the global L set and
   obtains an identical partition. Canonicalise: component id = `min(member eqn)`, members ascending.
   Union-find path-compression order differs per rank; the partition does not.
4. **Owner election: `sp.rank_of_row(min L eqn)`.** Not "the rank with the most L rows": the owner must
   hold the component's smallest L equation, because Exchange 3 is packed in ascending
   smallest-L-equation order and every existing sort key in the file
   (`comps[a].L_eqns[0] < comps[b].L_eqns[0]`, `src/problem.cpp:5042`) relies on the owner being able to
   evaluate it. For CR it elects the same rank anyway — the nodal half is numbered below the internal
   half — and for untouched components it reproduces today's `owner_rank = my_rank` exactly.
5. **Globalise the size guard.** `src/problem.cpp:4874` tests `owned[c].nL()`, the *local* half of a
   component. Once the straddling refusal is lifted, an interior-penalty DG selection run replicated
   forms one global component of O(ndof) whose local halves are each ≈ ndof/nproc: the guard would pass
   on every rank and the kernel would attempt a dense LU of the whole mesh. Today that case is caught by
   the refusal being removed. **This is the one change that must not be forgotten.**
6. **Globalise `row_ok`/`col_ok`** (`src/problem.cpp:4812`, and the comment at `:4886` that says in so
   many words that they are local *because* straddling was already refused). Left local, a legitimate
   straddling selection is refused as "structurally singular" whenever an L dof's only defining equation
   lives on another rank.
7. **Exchange 2 re-routed.** The `(retained row, condensed column)` pairs (`src/problem.cpp:4936`) go to
   the **component owner** rather than the column's row owner — computable locally after step 3, and
   identical to today for untouched components. The receive-side validation (`:4952`) relaxes from "is
   `c` one of my own L rows" to "is `c` in the L set of a component I own", still as a voted `err`.
8. **New tag-85 structural exchange.** For every L row it owns whose component is owned elsewhere, a
   rank sends the row's retained columns to the owner, which merges them into the "by column" half of
   `E_C` — the half it can no longer see for itself. Through `condensation_alltoall_ints`
   (`src/problem.cpp:4283`), which does an `MPI_Alltoall` of counts first and is therefore safe by
   construction.
9. **Exchange 3 extended, carefully.** Descriptors must also reach ranks that own only an L row of the
   component. That silently breaks the operator-exchange plan unless both sides use the *same*
   predicate: `xy_send_comp` is built from "p owns an E row" (`comp_peers`), while `xy_recv_comp` is
   built from "I hold a record of this component" (`src/problem.cpp:5057`). Today they coincide. Fix by
   moving `my_E` (`:5069`) **above** the xy plan and having the receiver push into `comps` only when its
   `my_E` is non-empty; a descriptor received solely because of a foreign L row builds the `l_send_*`
   layout and is then discarded. A no-op today, and what keeps it a no-op tomorrow.
10. **Layouts.** Per peer: components ascending by min L equation, rows within a component ascending by
    equation. Both sides enumerate from the same globally known descriptor, so no further negotiation.
    Derive send and receive membership from one predicate — a row travels from `p` to the owner iff
    `sp.rank_of_row(L_eqns[i]) == p && p != owner` — evaluated by the sender over its owned L rows and
    by the owner over `comp.L_eqns`.

## 4. Struct changes (`src/problem.hpp`)

`CondensationComponent` (`:559`):

| field | change |
|---|---|
| `owner_rank` (`:593`) | semantics: "the rank owning the smallest L equation, which gathers the rest". The current doc comment becomes false and must be rewritten. |
| `L_eqns` (`:561`) | on the owner, the full global ascending list, not just its owned rows. |
| `L_values` (`:563`) | must be nL-long on the owner **including foreign rows**: `reconstruct_condensed_dofs_distributed` dereferences `comp.L_values[k].first` at `src/problem.cpp:5932` in the `dx_is_global` branch too (the `PARANOID` equation check). Fill it from the same global selection scan that builds `all_L_eqn` (`:5178`), carrying `(eqn, Data*, v)` instead of `(eqn, double*)`. |
| `LL_slots`, `LE_slots` (`:568`) | shape unchanged; **all `-1` in a foreign row**. Also a memory-safety fix: `full_slot()` (`:5216`) indexes `row_start[grow - first_row]` unguarded, so calling it for a foreign row would read out of bounds. |
| **new** `std::vector<int> L_buf_pos` | nL on the owner of a straddling component: `-1` where the row is local, else the offset of that row's `nL+nE+1` block in the plan's concatenated receive buffer. **Empty** for non-straddling and remote components, so the kernel tests `L_buf_pos.empty()` once per component and the HDG path stays the code it is today. |

`CondensationPlan` (`:606`), replicated-only, mirroring `xy_send_*`/`xy_recv_*`:

```cpp
std::vector<int> l_send_start;      // nproc+1, slices the row-parallel arrays below
std::vector<int> l_send_row_off;    // per shipped row + sentinel: offset into l_send_slots
std::vector<int> l_send_slots;      // nL+nE full-value-array slots per shipped row, -1 where absent
std::vector<int> l_send_resid_row;  // per shipped row: local row index, for the r entry
std::vector<int> l_recv_start;      // nproc+1, slices l_recv_comp / l_recv_li
std::vector<int> l_recv_comp;       // component index
std::vector<int> l_recv_li;         // index into that component's L_eqns
unsigned long l_recv_buf_size = 0;  // doubles, one concatenated buffer over all peers
unsigned long n_straddling_owned = 0, n_l_rows_sent = 0, n_l_rows_recv = 0; // engagement counters
```

**`CondensationPlan::clear()` (`:655`) must reset every one of them** — a missed reset here is a stale
plan and therefore a hang, not a crash.

## 5. Kernel (`apply_static_condensation_distributed`, `src/problem.cpp:5584`)

New order:

1. pack the L-row sends — reads `r`, so **before** any zeroing — and post the tag-86 `Isend`/`Irecv`
   (sizes precomputed on both sides at plan time);
2. the pass-through copy while they fly;
3. `MPI_Waitall`;
4. factorise, with each L row taken either from `full_val`/`r` as today or, when
   `L_buf_pos[i] >= 0`, `memcpy`ed out of the receive buffer — one branch per L row:

```cpp
for (i = 0; i < nl; i++)
  if (!comp.L_buf_pos.empty() && comp.L_buf_pos[i] >= 0)
  { const double *b = lrecv.data() + comp.L_buf_pos[i];
    // b[0..nl) -> LL row i;  b[nl..nl+ne) -> RHS row i;  b[nl+ne] -> RHS[i*nrhs+ne]
  }
  else  /* today's gather from full_val and r */;
```

5. **`collective_throw` for the pivot guard**, in its current position relative to the operator
   exchange — after all tag-86 traffic has completed, before any tag-84 traffic is posted;
6. tag-84 `X`/`y` exchange and the Schur update of owned E rows, unchanged;
7. `r_L = 0` **over `plan.condensed_eqns`** and the identity diagonal, then `build_without_copy`.

Step 7 is a simplification, not new machinery: `plan.condensed_eqns` (`:5309`) is already exactly this
rank's owned L rows, and `L_diagonal_cond_slots` (`:5310`) and `is_condensed` (`:5314`) are already built
per owned row. Only the `r_L = 0` loop (`:5776`) is per owned *component*; replacing it with a sweep over
`condensed_eqns` moves the identity/zeroing to the row owner with a net negative diff, and no `my_L`
field is needed. Note that `build_without_copy` frees `jacobian.value()` (`:5808`), so the tag-86
`Waitall` must — and does — complete long before it.

## 6. Unchanged

Reconstruction: the owner computes `dx_L = y − X dx_E` for **all** nL rows, reading every E increment
straight out of the full-length replicated `dx` (`plan.dx_is_global`), and the `(equation, increment)`
pairs go through the existing `MPI_Allgatherv` (`:5946`) to be applied by every rank — which is what
keeps the ranks bit-identical. No duplicates arise: a row owner that is not the component owner is
skipped at `:5909`.

`condensation_row_cuts()` keeps snapping. Its worst-case displacement is one merged block —
a few dozen rows against a share of `ndof/nproc` — and it is the only thing that keeps element-internal
selections free of the new traffic. Optionally (small, independent) its "one over-long merged block →
give up" escape (`:4623`) can become "skip over-long blocks while snapping", which can only reduce the
number of straddling components.

## 7. What this costs

For CR essentially every component straddles, and — since the DL block is contiguous at the *top* of the
equation range — **its internal half is owned by the last rank**. With the smallest-equation owner rule,
ownership follows the nodal half and is spread evenly, but rank `nproc-1` is the sender for nearly every
component, so tag-86 is a fan-out from one rank. Arithmetic for N = 20, 4 ranks, ~800 straddling
components, `nL = 4`, `nE ≈ 30`, densified row `nL+nE+1 = 35` doubles: ≈ 1600 rows × 35 × 8 B ≈ **450 kB
per Newton step**, from one rank. Tolerable, independent of `nproc`, and asymmetric — worth saying out
loud rather than discovering in a benchmark. If it ever matters, the fix is to bias the cuts so the
internal block is split in proportion to where its nodal partners live, not to change the gather.

For comparison, allgathering the straddling L rows so every participant computes `X`,`y` redundantly
(which would delete the operator exchange) costs ≈ 3.6 MB received *by each rank* at 4 ranks, plus
nproc× the dense LU work, and still needs the global decomposition, an allgathered Exchange 2 and a
dedup rule for the reconstruction — i.e. an election by the back door. Rejected.

Whether condensation *pays* replicated is a separate, open question: the distributed mode is not
benchmarked either (§10 of [static_condensation.md](static_condensation.md)). This change is about
correctness and availability and claims no speed-up until one is measured.

## 8. Staging

**C1 — "the distributed plan names the component owner, not the L-row owner"** (pure refactor, no
behaviour change). Rewrite the `owner_rank` comment and derive it as `sp.rank_of_row(L_eqns[0])`
(identical to today's value); move `my_E` above the xy plan and build `xy_recv_comp` from
`owner_rank == p && p != my_rank && !my_E.empty()`; replace the per-component `r_L = 0` loop with a sweep
over `condensed_eqns`; guard `full_slot()` against a non-owned row. *Verify:* the whole existing MPI gate
at 2 and 4 ranks, `--full`; the stats dict byte-identical to before. **This commit must move no number.**

**C2 — "a replicated component may span ranks: the global decomposition"** (still refuses, but for
globalised reasons). Steps 1–6 of §3; new stats `n_straddling_owned` and a global
`n_touched_components`. *Verify:* replicated HDG unchanged **and** newly asserting
`n_touched_components == 0` (the promise that the allgather is zero-length there); the mixed
nodal/internal refusal still passes; everything distributed unchanged. **New:** run `--mode straddle`
*replicated* at 2 and 4 ranks under the bounded timeout and assert it is refused **by the size guard**,
quoting the component size — the test that proves the globalised guard exists *before* the refusal is
lifted.

**C3 — "the component owner gathers the L rows it does not own"** (the behaviour change). Steps 7–10 of
§3, §4, §5; the replicated refusal goes. Tests below.

**C4 — cut snapping over long blocks** (optional, §6). *Verify:* HDG replicated row cuts unchanged
(asserted at `tests/test_mpi_static_condensation.py:355`); CR `n_straddling_owned` unchanged or lower.

**C5 — docs.** Rewrite §9.9 of [static_condensation.md](static_condensation.md) (its third paragraph
`:1075` asserts the refusal as design), and update §9.1's table, §9.3 steps 2/4/7, §9.4's "two new
refusals" (one becomes distributed-only), §9.5's step list and §9.8's first bullet. Carry the rejected
alternatives of §2.1–2.3 across.

## 9. Tests

`tests/test_mpi_static_condensation.py` + `tests/mpi_condensation_worker.py`, all `slow` (`--full`),
each rank's numbers scraped from `PYOOMPH_MPI_RESULT` lines under a bounded timeout — a one-sided
refusal shows up only as that timeout.

* **Flip** `test_replicated_mixed_nodal_and_internal_selection_is_refused` (`:367`) into
  `test_replicated_cr_condensed_solve_matches_serial[2,4]`, against the existing serial **uncondensed**
  reference through `_assert_same_state`. The `internal_sum`/`internal_sqsum` checksums do the real
  work: they are exactly the eliminated DL modes, which an integral moment of the retained velocity
  would barely notice.
* **Engagement, in pairs** — a "distributed" plan in which nothing straddled would pass every
  equivalence test while exercising nothing:
  ```python
  assert sum(r["n_straddling_owned"] for r in per_rank) > 0
  assert sum(r["n_l_rows_sent"] for r in per_rank) > 0
  assert sum(r["n_l_rows_sent"] ...) == sum(r["n_l_rows_recv"] ...)   # a mismatch is a HANG
  assert r["plan_rebuilds"] == 1                                       # two solves, one plan
  ```
  and the negative on the replicated HDG test: `sum(n_l_rows_sent) == 0`.
* **New:** replicated CR transient (3 BDF2 steps: the gather runs per step and the plan survives);
  replicated CR with `--interface 1` (requirement 2 — an interface element writing into the condensed
  dofs' own rows — composed with those rows being split across ranks); and a replicated
  `--condense 0` vs `--condense 1` comparison on the same partition, mirroring
  `test_distributed_condensed_matches_distributed_uncondensed` (`:227`).
* Expose the counters in `src/nanobind/problem.cpp` next to `n_operator_sends`/`n_operator_recvs`
  (`:1060`), forward them in the worker's stats list (`tests/mpi_condensation_worker.py:178`) and add
  them to `_assert_ranks_agree`'s per-rank keys (`:148`).
* Regression, unchanged: `tests/test_static_condensation.py`, `tests/test_structural_assembly.py`,
  `tests/test_mpi_structural_assembly.py --full`.
* Manually, from a fresh scratch directory each time:
  `mpirun -n 4 python3 docs/source/tutorial/spatial/stokes/cr_static_condensation.py` without
  `--distribute`, against the serial run of the same script, and once more with `--distribute`.

Build with `./build_for_develop.sh` after every C++ stage; never `ninja` alone.

## 10. Stale claims to correct on the way (already wrong today)

`e601593` served both MPI modes but left several statements behind:

| where | says |
|---|---|
| `pyoomph/generic/problem.py:1495` | "a replicated MPI run … is refused too" |
| `pyoomph/equations/generic.py:2237` | "An MPI run … refused with an error" (contradicts `:2195` of the same docstring) |
| `src/nanobind/problem.cpp:986` (mirrored into `pyoomph/_pyoomph_core.pyi`) | lists "an MPI run that was never distributed" among the refused combinations |
| `dev_docs/README.md:45` | "serial and experimental" |
| `dev_docs/static_condensation.md` §9.4 | names the hook `prefer_dof_distribution_for_linear_solver` and `align_matrix_distribution_for_condensation`; both were replaced by `preferred_linear_solver_distribution` (`src/thirdparty/INFO_oomph-lib:631`) |

Once C3 lands, the two places that document the CR restriction also go:
`citools/test_all_tutorial_scripts.py:290` (the replicated skip) and the `.. note::` at
`docs/source/tutorial/spatial/stokes/cr_condensation.rst:121`.
