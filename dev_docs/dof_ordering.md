# Choosing the global dof numbering

Status: **implemented**, serial and both MPI modes. Companion to
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

## 6. Open

* **Constant block size for AMG.** Nothing yet calls `MatSetBlockSize` or `setNearNullSpace`; the
  matrix is plain AIJ. A constant stride additionally needs `apply_Dirichlet_BCs_by_dof_removing =
  False`, and even then only holds where every node carries the same field set.
* **The Dirichlet diagonal is still a literal `1.0`** (`remove_dirichlets_by_matrix_manipulation`).
  Keeping the dofs now forces the diagonal into the pattern (that was a genuine zero-row bug, see
  `tests/test_dirichlet_matrix_manipulation.py`), but the value is not yet magnitude-matched to its
  neighbours, which is what AMG cares about. Any non-zero value gives an exactly zero increment, so
  this is free to change.
* **No benchmark.** Nothing here claims a speed-up. The permutation's own cost is one O(n log n) sort
  and two O(nnode+nelement) walks inside a call whose elemental pass is ~96% of the total, but that
  has not been measured, and whether a nodal-block layout actually helps BoomerAMG on a pyoomph
  problem is untested.
