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

   ### 5.5.1 What a local per-element patch cannot do (investigated on `new_hanging`)

   A first attempt tried to keep the composition purely local: in
   `fill_hang_info_with_equations_{for_pos,basebulk}`, route a C1-constrained
   field through the C1 hang instead of its native (C2) hang, drop the three
   throws, and rely on `fill_additional_hang_buffer_data`'s existing one-level
   chaining. Reproducer: `tests/test_constrained_adaptivity.py` (Poisson C2 field
   `u` constrained to C1, plus a `_dummyC1` C1 space, on a two-level adaptively
   refined `CircularMesh`). Oracle: the problem is linear, so the post-solve
   residual must be ~0.

   Result: single-level / uniformly-refined constraint regions **do** converge to
   machine zero, but the genuinely non-conforming case **diverges**. Runtime
   instrumentation pinned the cause precisely:

   * For a constrained mid-edge node on a T-junction, `hanging_pt(C1.hangindex)`
     returns the **quadratic (3-master) geometric hang**, not a linear C1 hang —
     e.g. masters with weights `(-0.125, 0.375, 0.75)`. The `0.75` master is the
     coarse edge's mid-node, whose value the constraint **pins** (`local_eqn=-1`).
     Its contribution is silently dropped, leaving only `0.375-0.125` of the
     value. So "just use the C1 hang" retrieves the wrong object for non-vertex
     nodes.
   * The correct expansion of that pinned-constrained master (mid-node `cm`
     `= 0.5(c1+c2)`) needs `cm`'s **home coarse element's** C1 corners. From the
     neighbouring fine element `cm` is a C1 *vertex*, so `get_c1_masters(cm)`
     returns NULL and its constraint cannot be expressed there at all. The needed
     information is cross-element.
   * Equivalently: pinning `cm` **severs the sensitivity chain** `M1 -> cm ->
     (c1,c2)` that the hang depends on; the constraint's redistribution of `cm`
     lives only in `cm`'s own element's hangbuffer, invisible to `M1`'s element.

   Conclusion: correctness requires the constraint to be materialised as a
   **flattened, globally-visible** map on the node (`cm`'s constraint =
   `{c1:0.5, c2:0.5}` with real, non-hanging, non-pinned leaf masters), computed
   once and read by every element that uses `cm` as a master. Genuine hangs that
   reference a constrained master are then **flattened against it** so all masters
   are real dofs.

   ### 5.5.2 Implemented solution (landed on `new_hanging`)

   The flattening above is implemented directly in pyoomph's JIT hangbuffer fill,
   without touching oomph-lib's `Node::Hanging_pt`, pin state, or equation
   numbering:

   * **Per-node stored expansion.** `NodeWithFieldIndicesBase::c1_constraint_corners`
     holds each constrained node's immediate C1-corner expansion (the C1 vertices
     of the element where it is a non-vertex node, equal weights). It is computed
     in `BulkElementBase::setup_additional_dof_constraints` — i.e. in the element
     that *can* see those corners — so it is available even when the node is later
     reached as a hang master from a neighbouring element (where it is a C1 vertex
     and its own corners are not locally derivable). Recomputed after every adapt.
   * **Recursive flatten.** `flatten_hang_for_value` / `flatten_hang_for_position`
     expand a dof into a weighted sum over real free leaf dofs: a constrained node
     → its stored C1 corners (recursed); a genuinely hanging node →
     `hanging_pt(v)` masters (recursed); otherwise a real leaf, whose local eqn is
     `nodal_local_eqn` if it is one of this element's nodes, else the
     hang-registered `local_hang_eqn` / `local_position_hang_eqn`.
   * **Why no new equation numbers are needed.** The leaf vertices reached by the
     flattening are exactly the coarse edge/face vertices that oomph-lib already
     registered in `Local_hang_eqn` as masters of the genuinely-hanging non-vertex
     nodes on the same edge/face. The one native master the constraint *drops* is
     the coarse mid-node it pins.
   * **Fill.** `fill_hang_info_with_equations_{basebulk,for_pos}` build each
     hangbuffer entry from the flattened map (deduped; capped at `MAX_HANG`, with a
     clear error on overflow). The old one-level corner-average in
     `fill_additional_hang_buffer_data` (base version) is retired; the three
     protective throws are removed.

   **Validated** with the linear residual oracle (residual → machine zero certifies
   the analytic Jacobian): `ConstrainFieldsToC1Space` (full and `where`-restricted),
   `ConstrainPositionsToC1Space` (on a Laplace-smoothed moving mesh), and a 3D
   `CuboidBrickMesh`, each with two-level adaptive refinement plus a Neumann face
   element, all converge; non-constrained adaptivity (2D/3D), Stokes and
   linear-response regressions still pass. See `tests/test_constrained_adaptivity.py`.
   The flattening is dimension-agnostic, so 3D bricks worked with no extra changes.

   **Interface-dof constraints** (`INTERFACE_DOF_CONSTRAIN_TO_C1`, i.e.
   `ConstrainFieldsToC1Space` applied to a surface/interface field) now work on
   refined interfaces too — but the fix was *not* in the C++ hang fill. Two Python
   wiring bugs had made the feature a silent no-op:
   * `ConstrainFieldsToC1Space.before_assigning_equations_preorder` called
     `n.set_additional_dof_constraint(index, mode)`, but the binding signature is
     `(mode, index)` (see `src/nanobind/mesh.cpp` and `ConstrainPositionsToC1Space`).
     The swap only happened to be harmless for a bulk field at value index 0
     (mode==index==0) and mislabeled everything else — a bulk field at index ≥ 1
     became an interface constraint, and an interface field became a bulk
     constraint on the wrong index. Fixed to `(mode, index)`.
   * `Problem.reapply_boundary_conditions` applied/cleared additional dof
     constraints only on the top-level bulk meshes in `_meshdict`, never recursing
     into the nested `_interfacemeshes`, so an interface element's
     `setup_additional_dof_constraints` (which pins the constrained interface dof
     and sets `has_additional_dof_constraints`) was never called. Now recurses.

   With those two fixes the existing `InterfaceElementBase::fill_additional_hang_buffer_data`
   chaining assembles a correct Jacobian: a surface field constrained to C1 on a
   two-/three-/four-level adaptively refined interface converges to the linear
   residual oracle (~1e-14), reduces the dof count, and matches serial under
   `mpirun -n 2 --petsc_mumps`.

   **Still not covered**: `interpolate_hang_values` uses the pre-flatten value
   interpolation (affects stored dummy/pinned values for output/restart, not the
   solve).
   * **MPI**: **validated.** (A pre-existing crash in the gather-to-root Pardiso
     path — `pyoomph/solvers/pardiso.py:513` recomputed `nnz_local = len(data)`
     from the always-`None` oomph solver-state handle — was fixed first; it hit
     plain non-constrained adaptive Poisson too, so it was unrelated to the
     hanging-node work.) With that fix, constrained + two-level-adaptive Poisson
     with a Neumann face element solves under `mpirun -n {2,3}` and the reduced
     integral observable matches the serial result to all printed digits, both
     with the fixed Pardiso gather solver and with `--petsc_mumps` (true
     distributed MUMPS). This works because the flattening keeps the truth in
     oomph-lib's `Node::Hanging_pt` / `Local_hang_eqn` and stores only
     element-local corner nodes, so oomph's `synchronise_hanging_nodes` handles
     the halo masters. Note: run distributed with `--petsc_mumps`, not
     `--distribute` with Pardiso (the latter only gathers to root).
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
4. **Unify dof-constraints with genuine hanging** (5.5): **DONE** for bulk field
   and position constraints — see 5.5.2. Each C1 constraint is materialised as a
   per-node stored corner expansion and flattened (composed with the refinement
   hang) into real free leaf dofs at hangbuffer-fill time; the three `throw`s are
   removed and `tests/test_constrained_adaptivity.py` asserts the residual oracle.
   Remaining: interface-dof constraints on refined interfaces, and
   `interpolate_hang_values` value consistency.
5. **MPI**: verify `synchronise_hanging_nodes` +
   `additional_synchronise_hanging_nodes` cover the new `HangInfo`s; add a
   distributed adaptivity test.
6. *(Optional / large)* implement genuine T-element tree hanging if simplex
   adaptivity via hanging (not remeshing) is in scope.

The biggest latent bug source is HangInfo ownership (2); the genuinely new
engineering is (4) and (5).
