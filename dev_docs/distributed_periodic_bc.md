# Periodic boundary conditions under `--distribute`

## 1. What a periodic node actually is

Periodicity in pyoomph is **pointer aliasing**, not a constraint equation and not a Lagrange
multiplier. `BoundaryNodeBase::make_node_periodic`
(`src/thirdparty/oomph-lib/include/nodes.cc:2951`) throws away the copy node's own storage and points
it at the master's:

```cpp
Copied_node_pt = copied_node_pt;
node_pt->delete_value_storage();
node_pt->Value      = copied_node_pt->Value;
node_pt->Eqn_number = copied_node_pt->Eqn_number;
copied_node_pt->add_copy(node_pt);
```

`BoundaryNode<NODE_TYPE>::assign_eqn_numbers` (`nodes.h:2828`) then skips a copy entirely, so the copy
contributes no equations and reads the master's numbers purely because the array is physically the
same one. Two consequences that everything below follows from:

* **A copy owns no data at all.** Anything that reaches the master reaches the copy for free.
* **Nodal positions are NOT aliased.** `make_periodic` only touches `Value`/`Eqn_number`; the warning
  at `nodes.h:2885` spells this out. So `variable_position_pt()` needs no special treatment anywhere,
  and a periodic BC on a moving mesh is a separate (pre-existing, serial) caveat.

Periodicity is declared with **`PeriodicBC`** (`pyoomph/equations/generic.py:1771`), which matches the
two boundaries by KD-tree after the mesh is built and defers corner nodes -- a node that is a slave on
one seam and a master on another -- to `MeshFromTemplateBase._link_periodic_corner_nodes`
(`pyoomph/meshes/mesh.py:1223`). It runs before distribution, while the mesh is still replicated, so
the matching itself never had a problem.

There is a second, lower-level route: `MeshTemplate::add_periodic_node_pair` / `link_periodic_nodes`
(`src/meshtemplate.cpp:1676`, `:2000`), which links nodes by template index before the oomph-lib nodes
exist. `LineMesh` and `RectangularQuadMesh` used to expose it through a `periodic` argument; that
argument was removed, because the route cannot patch the corner mid-side nodes of a doubly periodic
mesh (it fails in `MeshTemplateElementCollection::set_element_code` with "Cannot find the
corresponding L2 node for periodicity", `src/meshtemplate.cpp:1577`) and it never populated
`Mesh::copied_masters`. It is still available to hand-written mesh templates, and everything below
applies to it too -- it ends in the same `oomph::Node::make_periodic`.

## 2. Why it did not work

oomph-lib's distribution machinery does not know that copied nodes exist -- `mesh.cc`, `problem.cc`,
`refineable_mesh.cc` and `partitioning.cc` contain no occurrence of the word "periodic". Master and
copy are two independent `Data*` keys in every halo map while being one array. pyoomph's answer was to
refuse the combination outright, in `Mesh::ensure_halos_for_periodic_boundaries`:

> Distributed parallel with copied nodes (i.e. PeriodicBC) does not work with nodal degrees of
> freedom. Either use pure DG or implement a periodic boundary condition by Lagrange multipliers

Four things actually went wrong:

1. **The shared equation numbers got bumped twice.** `Problem::synchronise_eqn_numbers`
   (`problem.cc:17136`) walks `mesh_pt()->nnode()` and does
   `nod_pt->eqn_number(ival) += my_eqn_num_base`. `Data::eqn_number(i)` hands out a `long&` **into the
   shared array**, and master and copy are both in `Node_pt`, so the same `long` got the offset twice.
   Invisible on rank 0 (base 0) and silently out of range on every other rank. This one fires on the
   owner rank whatever the partition looks like.
2. **Halo classification gave the pair two different verdicts.** `processor_in_charge` is
   `max(domains touching the node)`, computed per node. The two sides of a periodic seam are at
   opposite ends of the domain and have completely different bulk neighbours, so one of the pair could
   end up halo and the other not -- while sharing one `Eqn_number` array.
3. **The halo exchange then wrote through the alias.** `synchronise_dofs` (`problem.cc:16976`) and
   `copy_haloed_eqn_numbers_helper` (`problem.cc:17469`) write received data straight into
   `Value`/`Eqn_number`. Receiving into the copy overwrites the local master and vice versa.
4. **A pruned master silently un-periodifies its copies.** `~Data` (`nodes.cc:483`) calls
   `clear_copied_pointers()` (`nodes.h:2332`), which allocates fresh storage for each surviving copy
   and sets `Copied_node_pt = 0`. No warning. The copy becomes an ordinary node, starts contributing
   its own equations at the next `assign_eqn_numbers`, and the run converges to something that is not
   periodic.

## 3. The fix

### 3.1 Keep the copies out of the halo bookkeeping altogether

This is the load-bearing change, in `Mesh::setup_shared_node_scheme` and
`Mesh::classify_halo_and_haloed_nodes` (`src/thirdparty/oomph-lib/include/mesh.cc`): a node with
`is_a_copy()` is skipped when building `Shared_node_pt`, `Halo_node_pt`, `Haloed_node_pt` and the
`processors_associated_with_data` / `processor_in_charge` maps. After the classification the copies
are given their master's halo status, so that code asking a node whether it is a halo still gets a
sane answer, but they appear in no list.

This is safe because a copy owns no data: whatever the master receives *is* the copy's data. It is
also *necessary*, and that is the part worth remembering:

> The two members of a periodic pair sit at opposite ends of the domain. No partitioning puts them in
> the same halo layer, so the halo/haloed/shared schemes structurally **cannot** pair them up.

`is_a_copy()` is a local property of the node and identical on every rank, so both sides of every
lookup scheme skip exactly the same nodes and the ordering invariants -- which are what the whole
scheme rests on -- still hold.

**Rejected alternative, and why.** The first attempt agreed one owner per pair in `Mesh::distribute`
(where the mesh is still replicated on every rank and `element_domain` has already been broadcast, so
the union over both sides is computable without communicating) and had
`classify_halo_and_haloed_nodes` honour it. That is a globally consistent verdict, and it still does
not work: the forced owner is a rank that holds *the other* member of the pair, so the node is not in
the shared node scheme with it and the "overlooked halo node" reconciliation throws

```
Failed to find node that is shared node 48 (with processor 0)
 in shared node lookup scheme with processor 1 which is in charge.
```

(`mesh.cc:4079`). Patching that reconciliation to cope only moved the failure to configurations with
three or more ranks, where the intermediate processor is a third rank that also does not have the
node. The lesson is that the halo scheme is built out of *element adjacency*, and no amount of
relabelling makes a periodic pair adjacent.

### 3.2 Do not bump a copy's equation numbers

`Problem::synchronise_eqn_numbers` skips the value bump for `is_a_copy()` nodes, and (for symmetry --
it is never true today) the position bump for `position_is_a_copy()` ones. Needed independently of
3.1: both nodes stay in `Node_pt` either way. Verified by ablation -- restoring the stock loop while
keeping everything else makes five of the twelve tests fail, all of them with PETSc's

```
Argument out of range -- Column too large: col 1440 max 991
```

which is what a doubly offset equation number looks like by the time it reaches the matrix.

### 3.3 Keep the master alive on every rank

`Mesh::ensure_halos_for_periodic_boundaries` (`src/mesh.cpp:1198`, called from
`Problem.actions_before_distribute`) already walked the boundaries, found the copy nodes and called
`set_must_be_kept_as_halo()` on a boundary element from each side. That is what stops failure mode 4,
and it is what 3.1 relies on: a rank holding the copy must also hold the master, because the master is
the only node in the halo scheme. One element per side is enough -- it only has to keep the node alive
and reachable -- which is why both searches in it stop at the first hit. The refusal and two
unconditional `std::cout` debug lines were removed from it; the flagging stayed -- and stubbing the
function out fails 6 of the tests, so 3.1 really does depend on it.

## 4. What is still refused, and why

**Periodic boundaries on a moving (ALE) mesh.** `make_periodic` aliases the values and not the
positions, so a copy on a moving mesh carries position degrees of freedom of its own -- and those are
exactly what 3.1 takes away, since the copy is in no halo list to carry them. They would be numbered
on every rank that holds the copy instead of on one. `Problem._require_no_distributed_periodic_
position_dofs`, called straight after `super().distribute()` (which is where
`Problem::distribute()` assigns the equation numbers, so there is nothing to read before that),
reduces `Mesh::has_periodic_position_dofs()` over the ranks and stops the run. Note the combination is
questionable even serially: oomph-lib does not make the two sides' coordinates coincide (see the
warning at `nodes.h:2885`) and points at Lagrange multipliers for that.

**Adapting a distributed mesh that has periodic nodes.** `Mesh::distribute` ends with
`setup_tree_forest()` (`mesh.cc:5568`), which rebuilds tree neighbours by matching shared nodes. A
periodic master and its copy are distinct `Node*`, so the `TreeRoot::Neighbour_periodic` links that
`BulkElementBase::connect_periodic_tree` (`src/elements_adapt.cpp:46`) installed do not survive.
Refining afterwards would create ordinary, non-periodic nodes along the seam via
`RefineableQElement<2>::node_created_by_neighbour`, and the solution would quietly stop being periodic.

`Problem._require_no_distributed_periodic_refinement` (`pyoomph/generic/problem.py`) stops the run
instead. It is checked *after* the adaption, on the number of elements it actually touched, because
`initialise()` runs its initial adaption loop on every problem whether or not it is adaptive -- a check
beforehand would refuse every distributed periodic run over an adapt that was going to be a no-op.
Both of its branches are collective: the periodic nodes can all sit on one partition, and a
`RuntimeError` raised on that rank alone would leave the others in the next collective.

Refinement *before* distribution is unaffected, which is where the initial uniform refinement happens.

Lifting this needs a `reconnect_periodic_trees()` that re-establishes the periodic tree neighbours
after `setup_tree_forest()` (the pairing is available from the copy/master node relation, so it need
not go back through `PeriodicBC`'s KD-tree), plus a story for the copy nodes that refinement itself
mints on the seam.

## 5. Other open items

* **Remeshing and load balancing.** `PeriodicBC.before_finalization` is not re-run after a remesh
  (`MeshFromTemplateBase._construct_after_remesh` does not call it) and `Mesh::copied_masters` is not
  carried across by `Mesh::_setup_information_from_old_mesh`. Untested with `--distribute`.
* **The mesh-template route cannot do a doubly periodic mesh at all**, serially or otherwise:
  `MeshTemplateElementCollection::set_element_code` fails with "Cannot find the corresponding L2 node
  for periodicity" (`src/meshtemplate.cpp:1577`) on the corner mid-side nodes. Unrelated to
  distribution; it is why the `periodic` argument of `LineMesh`/`RectangularQuadMesh` was dropped in
  favour of `PeriodicBC`.
* **Triangles, tets, wedges and pyramids cannot refine across a periodic seam** at all
  (`RefineableTElement<2>::node_created_by_neighbour` is a stub returning `is_periodic = false`), which
  is a serial limitation that the refusal in section 4 happens to also cover.
* **`resize_halo_nodes` never sees a copy any more**, since copies are in no halo list. That is the
  right outcome -- `BoundaryNode::resize` is a no-op on a copy and the master is resized through its
  own halo entry -- but it has not been exercised by a periodic boundary that also carries interface
  fields.

## 6. Tests

`tests/test_mpi_periodic.py` + `tests/mpi_periodic_worker.py` + `tests/periodic_cases.py`, run with

```
python3 -m pytest tests/test_mpi_periodic.py -q --full
```

Five cases at 2 and 4 ranks, all built with `PeriodicBC`: a 1D `LineMesh` (a single node per seam,
binary-tree periodic connection), an x-periodic `RectangularQuadMesh`, a doubly periodic one (the only
case that reaches the deferred corner-node pass), and two non-periodic controls at the same
discretisation. Plus the two
refusals of section 4 -- an adaptive periodic mesh and a periodic moving mesh -- which are asserted to
fail, on every rank and for the stated reason.

The oracle that matters is **`ndof`**, and it is exact: failure mode 4 is otherwise entirely silent,
and shows up only as extra degrees of freedom. The integral observables (`Mesh::evaluate_integral_
function` skips halo elements and `MPI_Allreduce`-sums, so they are true global integrals) catch a
consistent but wrong field -- in particular the double bump of section 2.1, which rank 0 cannot see.
`seam_jump` compares the two sides of the seam directly and is exactly zero while the aliasing holds.
