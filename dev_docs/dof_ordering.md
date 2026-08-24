# Choosing the global dof numbering

Status: **implemented**, serial and both MPI modes, and **measured** (§6: 15x on a BoomerAMG
elasticity solve, though not for the reason the layouts were built). Companion to
[static_condensation.md](static_condensation.md) and
[replicated_condensation_gather.md](replicated_condensation_gather.md), whose §2.2 this supersedes.

## 1. What oomph gives, and why it is sometimes wrong

`Mesh::assign_global_eqn_numbers` (`mesh.cc`) numbers **all nodal values of a mesh, then all
element-internal ones**, and within a node `Data::assign_eqn_numbers` walks the values in one go. So
the default order is already **node-major**, which is worth knowing before writing a layout: a nodal
block layout in the natural field order is the *identity* on a single-mesh nodal problem, and a test
that only checks "the answer did not change" cannot tell that from a no-op.

Two consumers want something else:

* **Block preconditioners.** Hypre's BoomerAMG coarsens a vector system well only when a node's
  unknowns are adjacent and strided. Node-major already gives that for a plain bulk field, but not
  when an element-internal field belongs in the block, not when several meshes interleave, and not
  when the fields are wanted in a particular order.
* **Static condensation.** It needs the dofs it eliminates from one element adjacent, so that a
  replicated MPI row split can cut *between* the blocks. Nodal-before-internal is exactly wrong here:
  a Crouzeix-Raviart block pairs a bubble velocity (nodal) with the DL pressure gradient modes
  (internal), and the two halves end up hundreds of equations apart.

A third thing breaks a strided layout independently of the numbering: `apply_Dirichlet_BCs_by_dof_removing`
(the default) means a constrained value is not a dof at all, so boundary nodes' blocks are short. Set
it to `False` for a constant block size — see §6.

## 2. Where the permutation happens, and why only there

`Problem::reorder_global_eqn_numbers(Vector<double*>&)` is a new virtual in the vendored oomph-lib
(`problem.h`, logged in `src/thirdparty/INFO_oomph-lib`), called from `Problem::assign_eqn_numbers`
at **one** point: immediately after `Mesh_pt->assign_global_eqn_numbers(...)` and before the block
that either calls `synchronise_eqn_numbers()` or builds `Dof_distribution_pt`.

Everything that reads the numbering is built *below* that point — `Mesh::assign_local_eqn_numbers`
with the elemental info, `InterfaceMesh::update_equation_remapping`, the Dirichlet pinned-equation
set, the dof distribution, pyoomph's sparsity generation id. A permutation applied there is therefore
simply *what the numbering is*, and nothing that caches a dof index can survive it.

That is the whole safety argument, and it is why `replicated_condensation_gather.md` §2.2 was right
to reject renumbering and is nonetheless superseded: that section assumed the permutation would be
applied **after** `assign_eqn_numbers()` had returned, which would indeed have meant proving forever
that every consumer of a dof index is rebuilt afterwards.

**One implementation serves both MPI modes.** Distributed, the numbers at the hook are still
rank-local (`0..my_n-1`, halo data being `Is_pinned`) and `synchronise_eqn_numbers()` shifts them by
the rank base immediately afterwards, so a **rank-local** permutation leaves each rank's global range
contiguous — which the distributed assembly and the condensation row ownership both require. Serial
and replicated, the numbers are already global.

**`ndof()` is not usable inside the hook.** It reads `Dof_distribution_pt->nrow()`, and the
distribution is rebuilt below — so inside the hook it still reports the *previous* numbering's size,
zero on the first call. Anything sizing a per-dof buffer there must use `dof_pt.size()`. This bit
once: `get_dof_to_global_field_index_mapping()` sizes that way, and the layout silently declined
because the buffer came out the wrong length.

## 3. The Python API

```python
problem.dof_ordering = NodalBlockOrdering("domain/velocity_x", "domain/velocity_y", "domain/pressure")
```

or a list, applied in order:

```python
problem.dof_ordering = [
    NodalBlockOrdering("bulk/velocity_*", "bulk/pressure"),
    ElementBlockOrdering("bulk/pressure_dg"),
    NodalBlockOrdering("bulk/top/lambda"),
]
```

* `NodalBlockOrdering` keeps the named fields of one **node** adjacent, in the argument order.
* `ElementBlockOrdering` keeps them adjacent per **element**; a dof reachable from several elements is
  claimed by the first that reports it, and a cell-interior bubble node belongs to exactly one.
* Fields are **glob patterns over `problem._get_global_field_names()`** — the same vocabulary as
  `petsc_fieldsplit`. That covers interface-only fields (named by the interface's full path,
  `"domain/top/lambda"`) and nodal positions (`"domain/coordinate_x"`, *not* `mesh_x`). It does not
  distinguish a field's boundary-restricted dofs from its bulk ones, which is the right granularity
  for a layout.
* A dof is claimed by the **first** layout naming its field, which is how several meshes compose.
* Dofs no layout names keep their original relative order and **trail** the ordered ones.
* A pattern matching nothing **raises**. Silently ignoring it would return a layout that is not the
  one asked for while reporting success.

Implementation: `Problem::build_dof_permutation_from_specs` (`src/problem.cpp`). It resolves the
patterns once per field, walks `Mesh::visit_global_dofs` to attach each dof to its group (a node
pointer or an element pointer), collapses each group to the smallest equation number in it, and
stable-sorts by `(layout, group min, field rank, original equation)`. Collapsing to the group minimum
is what makes an already-satisfied layout come out as the identity rather than an equivalent
reshuffle.

## 4. Keeping the MPI row split off the blocks

Replicated (`mpirun` without `--distribute`), the Jacobian's rows are split uniformly and
contiguously by nobody's choice, so cut points land inside blocks — one per rank boundary, essentially
always. `Problem::preferred_linear_solver_distribution` (a pyoomph virtual consulted by
`create_new_linear_algebra_distribution` and `SuperLUSolver::solve`) hands back a split whose cuts have
been moved forward off the blocks.

Two claimants, and **static condensation wins whenever it is on**: its requirement is correctness (a
component must be invertible on the rank owning its rows), a layout's is only that a preconditioner
sees whole blocks. They cannot both be served in general — their blocks are different partitions of
the same rows. The snapping itself is one helper, `snap_cuts_to_blocks`, shared by
`condensation_row_cuts()` and `dof_ordering_row_cuts()`; it returns empty when some block is longer
than a rank's share, which cannot be stepped over by moving a cut at all.

Distributed, nothing is asked for: each rank's dofs are already contiguous and the permutation was
rank-local, so no block straddles a rank.

`Block_dof_pt_start` / `Block_dof_arrangement_used` in the vendored oomph-lib was an earlier attempt at
this and is **dead** — every consumer is commented out. The Python property that exposed it,
`Problem.nodal_block_dof_arrangement_used`, now raises and names the replacement.

## 5. Crouzeix-Raviart condensation, replicated

Measured on a 4x4 split-in-tris cavity, `StaticCondensation(velocity="bubble", pressure=[1,2])`,
`mpirun -n 4` without `--distribute`:

| layout | outcome |
|---|---|
| none | refused: *"a connected block of selected degrees of freedom is split across MPI ranks"* |
| `ElementBlockOrdering("domain/velocity_*", "domain/pressure")` | condensed, cuts `[0,67,130,192,257]`, checksum matching the serial reference to 1.7e-13 |

Gated by `tests/test_mpi_dof_ordering_rowsplit.py` at 2 and 4 ranks, including the refusal itself so
that the positive test is known to be testing something.

This does **not** retire [replicated_condensation_gather.md](replicated_condensation_gather.md). That
plan serves selections which *cannot* be made contiguous by renumbering — an interior-penalty DG one,
where the component genuinely percolates the mesh. It does retire §2.2's rejection of renumbering, and
its §1 and §8 staging should be re-read in that light.

## 6. Measured: what actually helps BoomerAMG

Script-level experiment, kept in `Scratchpad/amg_bench/` (`problem.py`, `bench.py`, `RESULTS.md`).
3D linear elasticity, C1 vector field on a 2×1×1 brick, clamped at `left` **plus a symmetry plane
`u_y=0` on `front`** (which matters — see below). CG, `rtol 1e-8`, unpreconditioned norm, PC =
hypre/BoomerAMG, N=20, 54243 dofs. All arms converge to the same tip deflection to eight digits.

The metric is the **iteration count**: deterministic, and immune to the machine load that makes
wall-clock A/B comparisons on this machine untrustworthy. Times are in-process around `ksp.solve()`
only, with the arms interleaved; across three reps the iteration counts were identical and the times
within ~1%.

| arm | what | ndof | iters | best t [s] | s/iter |
|---|---|---|---|---|---|
| A | default numbering, Dirichlet **removed** (the default) | 52080 | 43 | 44.11 | 1.026 |
| B | default numbering, Dirichlet **kept** | 54243 | 43 | 44.35 | 1.031 |
| C | nodal layout + kept + **`MatSetBlockSize(3)`** | 54243 | **14** | **2.91** | 0.208 |
| D | C + `-pc_hypre_boomeramg_nodal_coarsen 4` | 54243 | 93 | 3.66 | 0.039 |
| E | `MatSetBlockSize(3)` but Dirichlet **removed** | 52080 | 88 | 6.52 | 0.074 |

**C is 15× faster than A**, and not only through the iteration count: each iteration is also 5×
cheaper, which is the signature of scalar AMG building an expensive and poor hierarchy on a vector
system — it cannot see that three unknowns belong to one point, so the coupling between the components
wrecks the coarsening.

### 6.1 Attribution — and the uncomfortable part

**The whole gain is `MatSetBlockSize`, which pyoomph does not call anywhere.** Neither of this
document's two mechanisms produces it:

* `NodalBlockOrdering` is worth **nothing** here, because on this problem it is the *identity* — oomph
  is already node-major (§1). That is not a disappointment so much as a confirmation of §1: the layout
  earns its keep where oomph is not already right, and a plain single-mesh vector field is not such a
  case.
* Keeping the Dirichlet dofs is worth **nothing on its own** (B = A, 43 iterations either way).

**What keeping them buys is the right to declare the block size at all, and that is worth 6× in
iterations.** Arm E declares `bs=3` with the dofs removed: 88 iterations against C's 14, and 2.2×
slower. PETSc accepts it silently, because `ndof` still happens to divide by 3 — the blocks simply no
longer line up with the nodes.

**That comparison only works because of the symmetry BC.** Without it, the clamped face pins all three
components of each node, dof removal deletes whole 3-blocks, the stride survives, and E matches C
exactly (15 iterations, 2.99 s) — i.e. a benchmark whose only Dirichlet condition constrains every
component of a node "proves" that keeping the dofs is pointless. It is a roller/symmetry condition,
pinning one component and not the other two, that actually breaks a strided layout. Worth knowing
before repeating this measurement.

**`nodal_coarsen` is a pessimisation** (D: 93 iterations against 14). Its iterations are the cheapest
of any arm, so it does build a small hierarchy — just a much worse one. Not worth enabling by default.

### 6.2 The unit Dirichlet diagonal does not hurt AMG

The worry that motivated §6's original open item was that `remove_dirichlets_by_matrix_manipulation`
writes a literal `1.0` on the constrained diagonal while the neighbouring entries scale with the
modulus. Sweeping E over twelve orders of magnitude, at N=14 (19575 dofs), iteration counts:

| E | A | B | C | D | E-arm |
|---|---|---|---|---|---|
| 1e-6 | 32 | 32 | 14 | 67 | 63 |
| 1 | 32 | 32 | 14 | 65 | 63 |
| 1e6 | 32 | 32 | 14 | 66 | 63 |

Flat. The reason is structural: the manipulation zeroes the constrained row **and its column**, so the
row is completely decoupled and AMG sees an isolated 1×1 block. A strength-of-connection measure is
per-row and relative, so a row with no off-diagonals has nothing to be out of scale *with*.

Magnitude-matching that diagonal therefore buys nothing for BoomerAMG — at least for a decoupled
identity row and a hybrid-Gauss-Seidel smoother. It would still matter to anything that scales
globally by the diagonal, and the condition number genuinely does change; but the original motivation
does not survive measurement, so it is recorded here rather than left on the open list.

## 7. Open

* **Declare the block size from `PETSCSolver`.** §6 says this is where the entire measured gain is,
  and it is a script trick today. The solver should ask the active layout for a constant block size
  and call `MatSetBlockSize` itself. A constant stride exists only when
  `apply_Dirichlet_BCs_by_dof_removing = False` *and* every node carries the same field set, so the
  layout has to be able to answer "no".
* **`setNearNullSpace` is untouched.** BoomerAMG never receives rigid-body modes; for elasticity that
  is the next lever after the block size.
* **The measurement covers one problem.** Serial, C1 elements, one preconditioner, a single vector
  field. Nothing is claimed for Stokes or any multi-field problem, where a constant block size does
  not exist in the first place, and nothing is claimed under MPI.
* **The permutation's own cost is unmeasured.** One O(n log n) sort and two O(nnode+nelement) walks
  inside a call whose elemental pass is ~96% of the total, so it should not register — but that is
  arithmetic, not a benchmark.
