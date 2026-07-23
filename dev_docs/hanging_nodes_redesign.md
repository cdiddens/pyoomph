# Hanging-node redesign (branch `new_hanging`)

Status: **design + in-progress implementation**. Author: development notes.

This document describes how oomph-lib represents hanging nodes on adaptive
(non-conforming) meshes, how pyoomph currently piggybacks on that machinery, and
the plan to re-express hanging in pyoomph around a small number of *named*
`HangInfo` pointers (geometric / C2(TB) / C1(TB), plus interface- and
constraint-specific ones) so that adaptive refinement, interface degrees of
freedom, and local dof constraints (`ConstrainFieldsToC1Space`,
`ConstrainPositionsToC1Space`) can all coexist — and eventually work under MPI.

All file/line references are to the state of the tree at the time of writing.

---

## 1. Background: oomph-lib's hanging-node machinery

### 1.1 `HangInfo` — one constraint row

`oomph::HangInfo` (`src/thirdparty/oomph-lib/include/nodes.h:741`) stores a single
hanging quantity as a linear combination of *master* nodes:

```
q = sum_m  Master_weights[m] * q(Master_nodes_pt[m])
```

with members `Node** Master_nodes_pt`, `double* Master_weights`, `unsigned
Nmaster`. Master nodes are, by construction, never themselves hanging. Copy
construction/assignment are deleted, so a `HangInfo` is always heap-owned by
exactly one owner.

### 1.2 `Node::Hanging_pt` — per-value array, index offset by one

`Node` holds `HangInfo** Hanging_pt` (`nodes.h:942`), an array of size
`nvalue()+1`. The signed logical index `i` maps to physical slot `i+1`:

* `i == -1` -> slot `0` -> **geometric / positional** hanging (the node position).
* `i >= 0`  -> slot `i+1` -> hanging of continuously-interpolated **value** `i`.

Accessors `hanging_pt()`, `hanging_pt(i)`, `is_hanging()`, `is_hanging(i)`
(`nodes.h:1228-1332`) simply read `Hanging_pt[0]` / `Hanging_pt[i+1]`. **These are
non-virtual and inlined.** So are `Node::value(i)` / `Node::position(i)`
(`nodes.h:1477`, `nodes.h:1539`). This is the single most important architectural
constraint (see section 4).

Interpolation:
* `Node::value(i)` (`nodes.cc:2408`): if not hanging, `raw_value(i)`, else
  `sum_m master(m)->raw_value(i) * w_m` using **value slot `i+1`**.
* `Node::position(i)` (`nodes.cc:2528`): if not (geometrically) hanging, `x(i)`,
  else `sum_m master(m)->x(i) * w_m` using the **geometric slot 0** only.

### 1.3 The "same as geometric" sharing convention

`set_hanging_pt(hang, -1)` (`nodes.cc:2068`) records which value slots currently
*alias* `Hanging_pt[0]`, installs the new geometric `HangInfo`, then re-points
every aliasing value slot back to slot 0 and `constrain()`s it. By default all
values therefore "follow" the geometry; the documented exception is Taylor-Hood
pressure, given its own slot via `further_setup_hanging_nodes()`.

### 1.4 Ownership / deletion rule (critical)

Both `set_nonhanging()` (`nodes.cc:2281`) and `~Node()` (`nodes.cc:1695`) free a
value slot **iff `Hanging_pt[ival] != Hanging_pt[0]`**. They recognise exactly
**one** shared pointer — the geometric one at slot 0. Any *other* `HangInfo`
object referenced by two or more value slots is `delete`d once per slot ->
**double free**. `set_nonhanging` is non-virtual; the destructor behaviour is
fixed.

### 1.5 Local equation numbering

* Values: `RefineableElement::Local_hang_eqn` is `std::map<Node*,int>[]`
  (`refineable_elements.h:167`), indexed by value id, keyed by *master* node.
  `assign_hanging_local_eqn_numbers` (`refineable_elements.cc:312`) loops values
  `j in [0, ncont)`, and for each `is_hanging(j)` node walks its masters, giving
  each master's `j`-th value a fresh local dof appended after `ndof()`, deduped
  via a per-value `done` map, reusing `nodal_local_eqn` when a master is also a
  local node. The constraint is enforced by **condensation** (the slave gets no
  equation; its influence folds into the master columns weighted by `w_m`), not
  Lagrange multipliers.
* Positions: `RefineableSolidElement::Local_position_hang_eqn` is
  `std::map<Node*, DenseMatrix<int>>` (`refineable_elements.h:880`), per master a
  `(n_position_type x dim)` block, set up by
  `assign_solid_hanging_local_eqn_numbers` and consumed by
  `fill_in_jacobian_from_solid_position_by_fd`. This is the "negative index"
  path, accessed via argument-less `is_hanging()` / `hanging_pt()`.

---

## 2. How pyoomph uses this today

pyoomph JIT-generates residuals/Jacobians and does its own hang condensation, but
it deliberately keeps the hang data **inside `Node::Hanging_pt`** and reads it
back:

* Each space carries a `hangindex`, set in codegen (`codegen.cpp:7759-7772`):
  * Position, C2TB, C2 -> `hangindex = -1` (share geometric slot 0).
  * C1TB, C1 -> `-1` when the coordinate space is linear, else a **separate value
    index** `= nC2TB_basebulk + nC2_basebulk` (its own slot).

  So pyoomph already runs a de-facto **two-pointer** scheme: geometric+C2 fused,
  plus a distinct C1 pointer for quadratic-coordinate elements. Generalising to
  an explicit three-pointer model (geom / C2(TB) / C1(TB)) is a formalisation of
  this, not a new mechanism.

* At assembly, `fill_hang_info_with_equations_{for_pos,basebulk,interface}`
  (`elements.cpp:402,444,681`) read `node->hanging_pt(hangindex)` and
  `node->hanging_pt()`, then look up `local_hang_eqn(master, value_index)` /
  `local_position_hang_eqn(master)` to fill JIT `hanginfo` buffers, remapping
  local eqns so slaves redistribute to masters
  (`fill_hang_info_with_equations`, `elements.cpp:1129`).

* **Interface dofs**: `InterfaceElementBase` adds nodal values *beyond*
  `ncont_interpolated_values()` and maintains its own `Local_interface_hang_eqn`
  (`elements.hpp:2240`). pyoomph overrides `fill_in_jacobian_from_nodal_by_fd`
  (`elements.hpp:446-453`) because oomph's version would index `Local_hang_eqn`
  out of bounds for these.

* **Local dof constraints** (`ConstrainFieldsToC1Space`,
  `ConstrainPositionsToC1Space`) already synthesise a hanging scheme in
  `fill_additional_hang_buffer_data` (`elements.cpp:497`) — but they **throw if
  combined with genuine oomph hanging** (`elements.cpp:422,473`). Unifying these
  two worlds is the end goal.

* **Genuine tree hanging exists only for Q-elements** (oomph's
  `RefineableQElement`). pyoomph's `RefineableTElement<1..3>::setup_hanging_nodes`
  / `setup_hang_for_value` / `quad_hang_helper` are `throw("Implement")` stubs
  (`refineable_telements.cpp:188,1254,1442`); triangular/tet meshes adapt by
  **remeshing**, not hanging nodes.

---

## 3. Goal

Replace the implicit per-value `hangindex` scheme with up to **three canonical
`HangInfo` pointers per node**:

1. geometric / position,
2. C2 / C2TB fields,
3. C1 / C1TB fields,

plus **extra `HangInfo`s for interface-added dofs** (again split C2(TB) vs
C1(TB)), and eventually **more** for per-field / per-interface dof constraints —
all working on adaptive meshes and eventually under MPI.

---

## 4. Feasibility verdict and chosen architecture

The methods that decide hanging behaviour (`is_hanging`, `hanging_pt`, `value`,
`position`) are **non-virtual and inlined**, and oomph-lib's refinement
(`RefineableQElement::setup_hanging_nodes`), equation numbering
(`assign_hanging_local_eqn_numbers`), error estimation, output, mesh-to-mesh
projection, and **MPI hang synchronisation** (`synchronise_hanging_nodes`, the
`missing_masters` machinery) all read `Node::Hanging_pt` directly or via these
non-virtual accessors. Overriding them on a pyoomph-derived `Node` would not
change what oomph-lib sees.

**Therefore: keep `Node::Hanging_pt` as the single source of truth that oomph-lib
reads. Introduce the three (or more) named `HangInfo` pointers as a pyoomph-owned
*management / naming layer* whose job is to decide which `HangInfo` object each
`Hanging_pt` slot points at, and to own their lifetime safely.**

This reuses oomph's numbering, FD-Jacobian fallback, error estimator, and MPI
sync unchanged. The override surface is the *setup / lifetime* hooks, not the
accessors:

* `BulkElementBase::further_setup_hanging_nodes()` (`elements.hpp:670`, today
  empty) — the canonical oomph hook to assign non-geometric slots (C2 elements
  already delegate here).
* pyoomph's `RefineableTElement::setup_hang_for_value` / `quad_hang_helper` (the
  stubs) — only if genuine T-element hanging is ever in scope.
* `NodeWithFieldIndices::resize` (already overridden, `nodes.hpp:82`) — the one
  virtual seam in the Node lifetime we control.
* pyoomph's existing `fill_hang_info_with_equations*` — where the named grouping
  is consumed.

### 4.1 Materialisation rule (avoids the double-free trap of 1.4)

The three pointers are pyoomph's *logical* grouping. When projected into
`Hanging_pt`:

* All C2/C2TB value slots alias slot 0 (the existing, free-safe geometric-alias
  trick), matching `hangindex == -1`.
* The C1/C1TB group is materialised so that **no non-geometric `HangInfo` object
  is referenced by more than one `Hanging_pt` slot** — i.e. either a single
  representative C1 slot with the invariant of 5.3 preserved, or distinct
  per-value copies. Never install the *same* C1 `HangInfo*` into two value slots,
  or oomph's dtor / `set_nonhanging` double-frees it.

Invariant to preserve (see 5.3): **every value index of a hanging space must have
a non-null `Hanging_pt` slot**, so oomph's `assign_hanging_local_eqn_numbers`
populates `Local_hang_eqn[value_index]` for every field pyoomph later reads.

---

## 5. Issues and risks (ranked)

1. **Non-virtual accessors (decisive)** — dictates the "keep `Hanging_pt`
   populated" architecture (section 4).
2. **HangInfo ownership / double-free (high)** — section 1.4 / 4.1. The named
   pointers are a logical grouping; the slots hold oomph-owned objects following
   the geometric-alias rule. Do not share a non-geometric `HangInfo*` across
   slots.
3. **`Local_hang_eqn` indexing coupling (high)** — pyoomph reads
   `local_hang_eqn(master, real_value_index)` per field but reads hang *status*
   from the single `hangindex` slot; this only works if every real value index of
   a hanging space reports `is_hanging(j)==true` (invariant in 4.1). Otherwise
   `assign_hanging_local_eqn_numbers` must also be overridden.
4. **Interface-added dofs beyond `ncont_interpolated_values()` (high)** — live in
   pyoomph's own `Local_interface_hang_eqn` + FD override, not oomph's
   `Local_hang_eqn`. The extra interface `HangInfo`s plug into that parallel
   path.
5. **Unifying dof-constraints with genuine hanging (the actual goal,
   medium-high)** — today mutually exclusive via a `throw`. Make both produce
   ordinary `HangInfo`s in the same slots. The subtle case is a node that is
   *both* geometrically hanging from refinement *and* locally constrained to C1:
   the two linear maps must be **composed** (substitute one into the other) into a
   single `HangInfo`. Needs dedicated tests on a refined mesh straddling a
   constraint region.
6. **`resize` interplay (medium)** — `Node::resize` (`nodes.cc:2167`) reallocates
   `Hanging_pt`, defaulting new slots to the geometric pointer. Interface dofs are
   added by resizing, so extra-`HangInfo` bookkeeping must be re-established after
   every resize (extend the existing `NodeWithFieldIndices::resize`).
7. **MPI (medium, design-shaping)** — `synchronise_hanging_nodes` /
   `missing_masters` serialise hang trees from `Hanging_pt`. Keeping the truth
   there keeps this working. Extra (interface/constraint) `HangInfo`s referencing
   off-processor masters need matching sync via pyoomph's
   `additional_synchronise_hanging_nodes` (`mesh.hpp:392`). **Do not invent a
   hang representation only pyoomph understands.**
8. **T-element hanging is unimplemented (scope flag)** — if "adaptive meshes"
   means quad/brick, oomph refinement is inherited. If it means triangles/tets
   via tree refinement, the stubs are a separate, substantial project (oomph
   triangles do not natively hang).

---

## 6. Implementation phases

1. **Formalise the current two-pointer reality into an explicit named model**: a
   small pyoomph-owned struct grouping the geometric / C2(TB) / C1(TB) `HangInfo`
   for a node (plus interface extensions), *projected into* `Hanging_pt` on every
   (un)refinement / resize following the free-safe alias rules of 4.1. Land as a
   **no-behaviour-change refactor** validated against existing Q-element
   adaptivity tests.
2. **Route setup through `further_setup_hanging_nodes()`** so the groups are
   assigned in one auditable place instead of the implicit codegen `hangindex`.
3. **Generalise interface-added dofs** to carry their own C2/C1 `HangInfo`s via
   the existing `Local_interface_hang_eqn` path; add refined-interface tests.
4. **Unify dof-constraints with genuine hanging** (5.5): compose constraint maps
   with refinement maps into single `HangInfo`s, remove the two `throw`s, add
   straddling-region tests.
5. **MPI**: verify `synchronise_hanging_nodes` +
   `additional_synchronise_hanging_nodes` cover the new `HangInfo`s; add a
   distributed adaptivity test.
6. *(Optional / large)* implement genuine T-element tree hanging if simplex
   adaptivity via hanging (not remeshing) is in scope.

The biggest latent bug source is HangInfo ownership (2); the genuinely new
engineering is (4) and (5).
