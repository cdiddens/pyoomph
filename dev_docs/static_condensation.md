# Static condensation: eliminating element-local dofs from the linear system

Status: **implemented, tested and benchmarked — serial and distributed, and experimental.** The API is
in place — the `StaticCondensation` equation class, with `condense_dofs`, `condense_element_private_dofs`
and `use_static_condensation` underneath it; a flagged Newton solve takes the same number of steps and
lands on the same solution as with the switch off, to the accuracy of the linear solves themselves; and
the factorisation of a Crouzeix–Raviart Navier–Stokes system gets **51–65 % faster**, in 2D and in 3D.
A `--distribute`d run at 2 and 4 ranks reproduces the serial answer and the serial iteration count (§9).
Jacobian reuse, the line search, a replicated (undistributed) MPI run and a selection whose block is
split across ranks are **refused with a message**, not silently ignored — collectively, under MPI;
eigenvalue problems and arclength continuation quietly keep the full system.
`tests/test_static_condensation.py` and `tests/test_mpi_static_condensation.py` are the gates.

---

## 1. The idea, and why pyoomph needed its own version of it

The dominant cost of a Newton step is not the assembly but the factorisation of the Jacobian. Degrees
of freedom that live inside a single element — a Crouzeix–Raviart bubble velocity, the gradient modes of
a discontinuous (DL) pressure, a projected D0/DG field — couple to nothing outside their own element
patch. They can be eliminated by a small dense Schur complement before the matrix reaches the solver
and reconstructed from the retained increment afterwards, and the solver never sees them.

NGSolve exposes this as `BilinearForm(condense=True)`. Its model is:

* dofs are classified **per space**: `LOCAL_DOF`/`HIDDEN_DOF` are condensable, `INTERFACE_DOF`/
  `WIREBASKET_DOF` are not;
* only the element Schur complement `S = A_EE − A_EL·A_LL⁻¹·A_LE` is scattered into a **full-size,
  non-renumbered** global matrix (condensed rows structurally zero, the solve masked with
  `FreeDofs(coupling=True)`);
* three dense per-element operators (`harmonic_extension` = −A_LL⁻¹A_LE, its transpose, and
  `inner_solve` = A_LL⁻¹) condense the residual and reconstruct the internal dofs after each solve;
* in a Newton iteration everything is rebuilt per linearisation, and the residual is assembled full and
  condensed afterwards.

pyoomph keeps the *shape* of that model — full-size matrix, identity rows, per-linearisation
operators — and changes two things, because two of the user's requirements do not fit it:

1. **Selection is per dof, not per space.** The classical CR elimination has to take the bubble
   velocities **and** the pressure gradient modes together, while leaving the constant pressure mode
   global. A per-space classification cannot say that. It is not a nicety: pressure-only condensation of
   CR is **structurally singular**, because the continuity rows are `weak(div(u), p_test)`
   ([navier_stokes.py:419](../pyoomph/equations/navier_stokes.py#L419)) and contain no pressure at all,
   so A_LL ≡ 0 whatever the values are.
2. **Condensed dofs may be referenced from outside their owning element**, and the result must still be
   exact. See the next section.

---

## 2. Architecture: global elimination *after* assembly

A per-element pre-scatter Schur complement (NGSolve's approach) is exact only if the owning element
holds the complete row **and** column of every condensed dof. In pyoomph that stops being true as soon
as

* an interface element adopts a bulk element's internal Data
  (`InterfaceElementBase::add_required_external_data`, [src/elements.cpp:12946](../src/elements.cpp)) —
  the free-surface case; or
* an interior-facet DG term writes into a neighbour's bubble row
  ([navier_stokes.py:453–454](../pyoomph/equations/navier_stokes.py#L453)).

In both, an element that does not own the dof writes entries of its row and of its column. So:

> **Assemble the full Jacobian and residual unchanged — frozen-sparsity fast path included — and then
> eliminate the selected dofs from the assembled CSR as a pure post-pass.**

Concretely, with `L` the selected dofs and `E` the rest:

1. The structural L×L coupling decomposes into small connected components (element patches; CR: one 4×4
   component per element in 2D, 6×6 in 3D). Union–find on the frozen pattern finds them; a size guard
   refuses selections that percolate.
2. Per component `C`: gather `A_LL_C` densely, factorise it once, solve for all right-hand sides at once
   to get `X_C = A_LL_C⁻¹A_{L_C,E_C}` and `y_C = A_LL_C⁻¹r_{L_C}`; scatter `−A_{E,L_C}·X_C` into
   precomputed fill slots of a condensed CSR and `r_E −= A_{E,L_C}·y_C`; make the `L` rows identity rows
   with `r_L = 0`, so the solver returns `dx_L = 0`. Keep `X_C`, `y_C`.
3. After the Newton dof update (oomph does `*Dof_pt[l] -= Relaxation_factor*dx[l]`), reconstruct
   `dx_{L_C} = y_C − X_C·dx_{E_C}` and apply `*value_pt -= Relaxation_factor*dx_L[k]` per condensed
   `(Data*, value)`.

What this buys, beyond correctness in the two cases above: `ndof`, the global equation numbering, dumps
and continuation state are **unchanged**, so bifurcation handlers, eigen and Hessian assemblies,
multi-assembly and arclength continuation automatically get the full system and stay correct without
knowing that condensation exists. Residual-only calls (`get_residuals`, i.e. the Newton convergence
check) also stay full, which is what makes "identical iteration counts" a meaningful equivalence test.

Everything is driven by a precomputed **`CondensationPlan`**, valid for as long as
`get_jacobian_structure_id()` ([src/problem.cpp:2436](../src/problem.cpp)) is, and cleared by
`invalidate_jacobian_structure()` ([src/problem.hpp:720](../src/problem.hpp)). v1 **requires** the
frozen fast path ([structural_assembly.md](structural_assembly.md)): the plan is expressed as positions
in its value array, so if that path declines, condensation declines with it.

**MPI-readiness was designed in, not deferred, and it held.** Components are self-contained objects
keyed by global equation numbers, replicable on any rank (a halo element resolves identically to its
owner); all runtime access goes through precomputed slot lists; and the reconstruction is formulated in
terms of `dx_E` rather than of a global increment vector, so the distributed version recovers it from
halo-synchronised values, `(u_old − u_new)/Relaxation_factor`, and never exchanges `dx`. §9 is what
became of that, including the one place the plan did not survive contact (the linear solver's row
distribution, §9.4).

---

## 3. Selecting the dofs

### The interface: an equation, on the domain it applies to

Everything else in pyoomph is said in the equation tree, and so is this. `StaticCondensation`
([pyoomph/equations/generic.py](../pyoomph/equations/generic.py)) is the interface users are meant to
use; it contributes no residual and does nothing but state a selection on the domain it is added to.

```python
eqs = NavierStokesEquations(mode="CR", dynamic_viscosity=1, mass_density=100)
eqs += StaticCondensation(velocity="bubble", pressure=[1, 2])   # the classical CR elimination
eqs += StaticCondensation("q")             # a whole elemental (DL/D0/DG) field, positionally
eqs += StaticCondensation()                # every element-private dof OF THIS DOMAIN
self.add_equations(eqs @ "domain")
```

A keyword value is `"all"`/`True` (the whole field), `"internal"`, `"bubble"`, or a list of value
indices of the elemental Data; a positional string means `"all"`. Malformed specs raise at construction
(`ValueError`), where the mistake was written, rather than when the mesh is first resolved. Adding the
class to an interface or ODE domain is refused with a message — condensation eliminates element-local
dofs of a *bulk* domain, and an interface element's internal data belongs to a facet.

Adding it anywhere in the tree also **switches `use_static_condensation` on**. The problem-level switch
stays exactly what it was, and an explicit assignment to it — in either direction — always wins, so
`self.use_static_condensation = False` is a kill switch that disables the feature wholesale without
touching the equations (`Problem::auto_enable_static_condensation` refuses to act once
`use_static_condensation_is_explicit`, which any assignment sets, even one that changes nothing).
Instances on different domains compose, since selections union.

### The plumbing: rules at problem level

Underneath, rules are the durable objects. A rule names a mesh, a field and which *part* of that field
is meant, so it survives adaptation and renumbering; the concrete `(oomph::Data*, value index)` pairs it
denotes are derived from it and thrown away by `Problem::invalidate_jacobian_structure()`. Storing the
pairs instead would leave dangling Data pointers after the first `adapt()`.

```python
problem.condense_dofs("domain/velocity", part="bubble")   # cell-interior bubble nodes (C1TB/C2TB)
problem.condense_dofs("domain/pressure", values=[1, 2])   # DL gradient modes; the constant stays global
problem.condense_dofs("domain/q")                         # a whole elemental (DL/D0/DG) field
problem.condense_element_private_dofs()                   # auto-detect: internal Data nobody else reads
problem.condense_element_private_dofs(domain="domain")    # ...restricted to one domain
problem.use_static_condensation = True                    # master switch, default False
problem.static_condensation_max_component_size = 64       # refusal threshold (default)
```

These declare a rule and nothing else: unlike the equation class, they do **not** switch condensation
on. Selections union; overlapping rules are harmless. `condense_dofs` splits the path on the last `/`,
so the field may sit on a subdomain, and a vector field may be named by its base name (`velocity`),
which selects all of its `_x`/`_y`/`_z` components. An element-private rule with a mesh restricts only
the *selection* to that domain; the reverse external-data scan that decides what "private" means stays
problem-wide, because an interface element of any domain may adopt this domain's internal Data.

### Lifecycle: when a rule is (re-)stated, and why re-stating it must be free

A `StaticCondensation` instance registers its rules from **`on_apply_boundary_conditions(mesh)`**, the
same hook the rest of the "act on the mesh, contribute no weak form" family (`PythonDirichletBC`,
`PinWhere`, `UnpinDofs`) uses. It is the right one because it (a) hands the mesh over directly, (b) first
fires from `Problem.setup_pinning()` during initialisation, i.e. as soon as the mesh exists, and (c)
fires again from `reapply_boundary_conditions()` after adaptation, remeshing and reading a state file.
`before_finalization` additionally rejects an interface domain at code-generation time, so a misplaced
instance is reported even on a domain whose mesh ends up holding no element.

It fires *often* — several times per solve — and the C++ rule list must not be edited each time:
`add_static_condensation_rule`/`clear` bump `static_condensation_rules_revision`, which is part of the
Jacobian structure id, so restating an unchanged selection would throw away the condensation plan and
the solver's symbolic factorisation on every solve. A pure performance bug, invisible in any answer.

The idempotency is therefore **not** in `add_static_condensation_rule` (which stays a plain append: it
is the low-level call, and two identical rules are harmless there) but one level up, in a Python-side
registry on `Problem`:

* every declarer — one `condense_dofs()` call, or one `StaticCondensation` instance on one domain —
  owns an entry keyed by itself, holding the **domain path** and the field specs;
* `_sync_static_condensation_rules()` resolves the paths to meshes, forms a signature of
  `(id(mesh), specs)` and compares it with what was last pushed down. Equal means **return**: no clear,
  no add, no revision bump. Different means clear-and-restate everything, which costs one rebuild of the
  (unchanged) full pattern and is setup-time work. It runs on every registration, unchanged entry
  included — that comparison is a handful of dict lookups, and running it unconditionally is what makes
  a re-registration *repair* a rule whose mesh has been replaced;
* the registered mesh objects are kept referenced, so an `id()` in a live signature cannot be recycled.

Keying on the **path** rather than on the mesh object is what makes this survive remeshing, and it fixes
a latent bug in the problem-level API as well: remeshing does not adapt a mesh, it builds a new one and
`_destroy_superseded_mesh()` destroys the old, so a C++ rule that named it holds a freed pointer that
`update_static_condensation_selection()` would then walk (a use-after-free that happens not to crash on
small examples). The equation's hook restates its rules on the way through
`reapply_boundary_conditions()`; `Problem.actions_after_remeshing()` calls the synchronisation once more
for rules nothing else would restate, i.e. the problem-level ones. Adaptive refinement does not replace
the mesh object, so it costs nothing — the signature is unchanged and the selection is re-resolved by
the existing invalidation.

`_clear_static_condensation_rules()` is overridden in Python to clear the registry too, or a cleared
rule would come back at the next synchronisation. A `StaticCondensation` in the tree does re-register
itself when the boundary conditions are next applied: the equation tree *is* the declaration, and
`use_static_condensation = False` is how the feature is switched off.

Implementation points worth keeping:

* **Bubble nodes are found by occurrence, not by index.** A cell-interior bubble node occurs in exactly
  one element of the mesh and lies on no boundary. That is robust across element types and automatically
  excludes the tet-C2TB *face* bubbles, which two elements share and which are therefore not
  element-local at all.
* **"Element-private" is measured against a reverse external-data scan** over every element of every
  mesh, interface meshes included. Internal Data adopted as external data elsewhere is excluded — the
  free-surface case that rules out a pre-scatter Schur complement. An explicitly named rule is *not*
  filtered this way: the user asked for those dofs, and the post-assembly elimination handles them
  correctly (§7 measures exactly that).
* **Pinned and constrained values are never selected.** A condensed dof has to be a genuine unknown, and
  dropping them at resolution time means no later stage has to distinguish them. Boundary conditions
  therefore silently shrink a selection; that is intended.
* **Rank-local and communication-free by construction.** Every decision is taken from data the rank
  already holds, and the reported equation list is sorted, so nothing downstream depends on hash-map
  iteration order (the mistake [replicated_mpi_correctness.md](replicated_mpi_correctness.md) §3
  documents).

Introspection: `_get_static_condensation_dofs()` (sorted global equation numbers) and
`_get_static_condensation_stats()` (`n_rules`, `n_selected`, `n_data`, `n_valid_eqns`, `per_rule`, plus
the plan statistics of §4 once one exists).

Verified counts on a 2D CR cavity (`RectangularQuadMesh(N=3, split_in_tris="left")`, 18 triangles, one
pressure value fixed): velocity bubbles 36 = 2/element, `pressure values=[1,2]` 36 = 2/element, whole
`pressure` 53 = 3·18−1, `element_private` 53 — and 44 once an interface term on `"top"` adopts the three
adjacent elements' pressure Data, i.e. 9 adopted values correctly excluded. All counts scale exactly ×4
under `refine_uniformly()`, with the rules re-resolved automatically. `StaticCondensation(velocity=
"bubble", pressure=[1,2])` selects the same 72 equation numbers as the two `condense_dofs` calls, and a
`StaticCondensation()` on one of two D0 domains selects 4 of the 8 element-private dofs, the
problem-wide rule all 8.

---

## 4. The elimination plan

`Problem::build_condensation_plan()` derives, from the **frozen full sparsity pattern alone**,
everything the numeric kernel will ever need. It reads no matrix value and nothing it produces depends
on one.

```cpp
class CondensationComponent {
  std::vector<int> L_eqns;                                   // condensed dofs of this block, ascending
  std::vector<std::pair<oomph::Data*,unsigned>> L_values;    // parallel: where to write the increment back
  std::vector<int> E_eqns;                                   // retained dofs it couples to, ascending
  std::vector<int> LL_slots, LE_slots, EL_slots;             // gather positions in the FULL value array, -1 where absent
  std::vector<int> fill_slots;                               // nE x nE scatter positions in the CONDENSED value array
  std::vector<double> X, y;  bool ops_valid;                 // A_LL^-1 A_LE and A_LL^-1 r_L
};
class CondensationPlan {
  unsigned long generation; unsigned ndof;
  std::vector<CondensationComponent> components;
  std::vector<int> cond_row_start, cond_column_index;        // full-size, non-renumbered condensed CSR
  std::vector<int> passthrough_full_slot, passthrough_cond_slot;
  std::vector<int> condensed_eqns, L_diagonal_cond_slots;
  std::vector<char> is_condensed;  unsigned long full_nnz;
};
```

Built in one pass over the pattern:

1. the L set, ordered by global equation number (so a rebuild of an unchanged pattern is bit-identical);
2. **union–find on the symmetrised L×L coupling** — an entry `(r,c)` couples `r` and `c` whichever way
   round the matrix stores it, because A_LL is inverted as a block and reducibility is a property of the
   undirected graph. Components are numbered by their smallest member;
3. the **size guard** and
4. the **structural pre-check** (below);
5. `E_C` = every retained dof coupled to the component **by column** (it appears in one of the
   component's equations) **or by row** (its own equation contains one of the component's unknowns).
   Both directions are needed and they are genuinely different sets — that is the whole reason this
   design condenses after assembly rather than per element;
6. gather slots by binary search into the (sorted) full rows, once, here;
7. the condensed pattern: a retained row keeps its retained columns and gains the whole `E_C × E_C`
   block of every component it takes part in; a condensed row keeps its diagonal alone;
8. the pass-through map (every `(E,E)` entry the full pattern already had), the fill slots and the
   identity-row slots.

`E_C × E_C` is a **superset** of the actual fill-in footprint of `−A_{E,L_C}·A_{L_C,L_C}⁻¹·A_{L_C,E}`,
since the row set and the column set are unioned rather than kept apart. That is the safe direction:
extra stored zeros, never a truncated matrix. The two overlap the pass-through entries wherever the full
pattern already had them, which is why the kernel must **accumulate** into the condensed array rather
than assign.

### Guard semantics

Both guards **throw** rather than silently declining. Not eliminating what the user asked for would turn
into a performance mystery, not an error.

* **Size guard.** Any component larger than `static_condensation_max_component_size` (default 64) is
  refused, naming the size, the limit and up to five sample dofs. Each component is inverted densely, so
  a selection whose coupling percolates through the mesh is not a condensation but a second, worse,
  direct solve. Sample dofs carry their equation number, because `get_dof_names()` is **local to an
  element** — without it, five samples from five elements read as five copies of `pressure__DL__1`.
* **Structural invertibility.** Every L row must have a structurally present L column and every L column
  must appear in some L row. This is what catches selecting a CR/DL pressure on its own: those rows are
  identically zero whatever the values are. The message says so, and says what to do instead.

Everything else is a quiet `false` with the plan cleared, meaning "nothing to do, or the route does not
apply": no dofs selected, `keep_structural_zeros` off (a value-dependent pattern cannot be planned
from), an augmented system, a distributed run, or a frozen pattern that could not be built.

### Cache lifecycle

**One** cached plan, keyed by the pattern id, against the frozen sparsity's eight LRU slots. The
alternation that forced a keyed cache there (an eigen assembly, a preconditioner matrix built from
another residual) cannot reach this path at all: condensation only ever engages on the default Newton
assembly of the unaugmented problem, so there is a single pattern in play. It is cleared by
`invalidate_jacobian_structure()` and by any change to the rules.

`JacobianStructureKey` gained two fields, `use_static_condensation` and a monotonic
`static_condensation_rules_revision`. The full pattern does not depend on either — but what a solver is
handed under that id once the switch is on is the **condensed** matrix, so reusing a symbolic
factorisation across a toggle or across an edit of the rules would factorise one pattern and
back-substitute on another. The revision counter also means that adding a rule invalidates the id
without any call site having to remember to; the price is one rebuild of the (unchanged) full pattern
per rule change, which is setup-time work.

### Measured structure

2D CR Stokes, `RectangularQuadMesh(N=4, split_in_tris="left")`, 32 elements, ndof 258, selecting bubble
velocities plus DL pressure values `[1,2]`:

| | 32 elements | after `refine_uniformly()` |
|---|---|---|
| components | **32 = one per element**, every one of size 4 | **128**, still size 4 |
| full nnz | 4 812 | 23 884 |
| condensed nnz | **2 316** | **12 140** |
| of which pass-through / identity rows / genuinely new | 2 156 / 128 / 32 | 11 500 / 512 / 128 |

Size 4 is exactly the classical CR block: two bubble velocity values and the two DL gradient modes. `nE`
runs 3…13 per component, the small values being boundary elements whose velocities are pinned. The 32
new entries are the pressure–pressure couplings the constant modes acquire — the field-coupling pruning
removes the p–p block from the pattern, and the Schur complement puts some of it back.

Every one of those numbers — component count, the `(nL,nE)` list, condensed nnz, pass-through count,
fill slots, new entries — is **recomputed independently in scipy** from the assembled Jacobian
(`scipy.sparse.csgraph.connected_components` on the symmetrised L×L block, the E sets from the CSR and
its CSC transpose) and agrees exactly, on both meshes. The referee first checks
`_get_frozen_sparsity_nnz() == nnz(J)`, since a comparison in which the frozen path silently fell back
would be worthless ([structural_assembly.md](structural_assembly.md) §4.1).

### HDG: the case condensation is *for*

Hybridizable DG is the one method where condensing is not an optimisation but the definition — the
global system is meant to be the trace system alone. It did not work until the contribution classes
learned about facet sides, and the reason is worth keeping.

The element code was always right: in an interior-facet element, the residual of the near-side bulk
field differentiates w.r.t. the facet unknown and the near side only, never w.r.t. the far side. But
`block_contribution_class_name()` named a class after the domain the field is *defined* on, and the
opposite side is a dummy `FiniteElementCode` carrying the same domain path — so both sides landed in one
class, `contributes_to_jacobian[u][u]` conflated `u⁺`–`u⁺` with `u⁺`–`u⁻`, and the pruned pattern kept
an edge across every facet. The union-find then chained the whole mesh into one component: measured on a
τ-stabilised 1D HDG Poisson, `n_components = 1` with a block of `2·N`.

Opposite-side classes now carry an `@opposite` suffix, for roles -2/-3 and **only for element-private
(DG/DL/D0) spaces**. That restriction is a soundness requirement, not tidiness: a continuous field's
facet dofs are literally the same unknown on both sides, so they are reachable through both class
columns, and a one-sided facet term would declare a real entry structurally absent. Element-owned Data
cannot be shared by two elements, so for DG the two sides' dof sets are provably disjoint.

The same 1D HDG Poisson now gives `n_components = N`, every block of size 2 (the element's two `D1`
values), and the solver is handed exactly the interior facets — 9 unknowns for 10 elements, 39 for 40 —
with the L2 error identical to the uncondensed run digit for digit.

Two things follow that are easy to get wrong:

* **Interior-penalty DG still percolates, and must.** Its `jump(u)·jump(v)` really does couple the two
  sides, so the table says so and the size guard refuses the selection. Adding such a term back to the
  HDG problem returns `n_components` to 1 — the check that the table follows the equations rather than
  having been hardcoded sparse. `test_size_guard_fires_on_a_genuinely_percolating_selection` is that
  guarantee in the suite.
* **`condense_element_private_dofs()` will not select the bulk field.** The facet elements adopt it as
  external data, which that rule deliberately excludes (§3). HDG needs the explicit
  `condense_dofs("domain/u")` / `StaticCondensation("u")`, which is not filtered.

The problem level does not see any of this: `assemble_defined_field_list()` strips the suffix, so
`defined_fields` and the grids in `_jacobian_structure.txt` are unchanged, and the file merely gains a
line naming the fields that were split per element. A far-side class typically has an empty Jacobian
*row*, and exposing it globally would make the empty-row analysis pin a field that does not exist.

---

## 5. The numeric kernel

`Problem::apply_static_condensation(residuals, jacobian)` turns the freshly assembled full system into
the condensed one, in place, at the end of `Problem::get_jacobian()`. It is a pure consumer of the plan:
no column search, no allocation per component, no knowledge of elements or meshes.

1. **Positive engagement check.** The plan is a list of positions in a particular value array, so the
   matrix is first compared against the frozen pattern it was derived from — row count, nnz, and then
   `row_start` and `column_index` outright. A silent mismatch would not crash; it would condense the
   wrong entries. The comparison costs one `memcmp` against work that is already O(nnz).
2. **Pass-through copy** of every `(E,E)` entry into the zero-initialised condensed value array.
3. **Per component**: gather `A_LL` (nL×nL), `A_LE` and `r_L` as one nL×(nE+1) right-hand side, and
   `A_EL` (nE×nL) through the plan's slot lists, absent slots reading 0. Factorise `A_LL` once, solve
   for all nE+1 right-hand sides in one sweep, keep `X` and `y` in the component for the reconstruction.
   Then `cond_val[fill_slots[a,b]] -= Σ_l A_EL(a,l)X(l,b)` and `r[E_a] -= Σ_l A_EL(a,l)y(l)`, and
   `r[L_i] = 0`. **Accumulating** into the fill slots is mandatory: they overlap the pass-through
   entries wherever the full pattern already had that coupling, and two components sharing a retained
   dof write to the same slots. Zeroing `r_L` inside the loop is safe because an `E` set only ever holds
   retained dofs. The components are summed rather than combined because `A_LL` is block diagonal over
   them by construction.
4. **Identity rows**: `1.0` on the diagonal of each eliminated row, its only entry.
5. **Rebuild.** `CRDoubleMatrix::build_without_copy()` **takes ownership** of the value, column-index and
   row-start arrays and deletes them when the matrix is cleared — which is why
   `assemble_with_frozen_sparsity()` hands out a fresh copy of its immutable pattern on every assembly,
   and why the kernel does the same with the plan's condensed pattern. It also frees the arrays the
   matrix currently holds, so nothing may read the full values after this point. The value array is held
   in a `unique_ptr` until then, so a pivot failure does not leak it.

### Not oomph-lib's dense LU

`DenseDoubleMatrix::ludecompose()` is deliberately unused. `DenseLU::factorise()`
([linear_solver.cc:139](../src/thirdparty/oomph-lib/include/linear_solver.cc#L139)) throws only when an
entire row is exactly zero, and otherwise substitutes `1e-20` for a vanishing pivot — the Numerical
Recipes trick — so a singular `A_LL` returns an increment of order 1e20 rather than an error. That is
precisely what the classic "condense the DL constant mode as well" mistake produces, so the guard has to
be a *relative* pivot test inside the factorisation, `|pivot| <= 1e-12·max|A_LL|`, and testing the pivots
means owning the factorisation. Two lesser reasons point the same way: `DenseLU` news its LU factors and
its pivot array on every `factorise`, and it back-substitutes one `DoubleVector` at a time where a
component wants nE+1 right-hand sides against one set of factors. The kernel's LU is ~40 lines with
partial pivoting; its scratch buffers are `Problem` members, so a solve settles into doing no allocation
at all. The threshold is relative to the *whole block*, not to the row: a component mixes bubble-velocity
and pressure-gradient equations whose natural scales differ by orders of magnitude.

### Where it sits relative to the Dirichlet manipulation

**Condensation runs after `remove_dirichlets_by_matrix_manipulation()`, and the order is forced.**
Condensation is an exact algebraic elimination of whatever linear system it is handed, so it has to be
handed the *final* one, and the Dirichlet manipulation is still rewriting that system: it replaces a
Dirichlet row by the identity, clears that dof's column in every other row and zeroes its residual.
Condensing first would eliminate a Dirichlet-constrained dof through its **raw** equation and fold that
residual into the retained ones — and the manipulation could no longer reach the eliminated row
afterwards anyway, since it is no longer in the matrix. In the order implemented, a Dirichlet dof caught
by the selection arrives at the kernel already decoupled and comes back out as `dx = 0`, which is what
the full system gives too. (The selection cannot easily *contain* a Dirichlet dof in practice — bubble
and DL-gradient dofs are not boundary dofs — so that sub-case is argued rather than measured. If it ever
happens with a pruned pattern that has no diagonal for the field, `A_LL` acquires a zero row and the
pivot guard refuses it, which is the right outcome if not the clearest message.)

---

## 6. The Newton path, and what does or does not condense

### The one vendored change

```cpp
// oomph problem.h, next to the other actions_* virtuals
virtual void actions_after_newton_dof_update(const DoubleVector& dx) {}
```

called from `Problem::newton_solve()` between `synchronise_all_dofs()` and `actions_after_newton_step()`,
i.e. after `*Dof_pt[l] -= Relaxation_factor*dx_pt[l]` and before anything looks at the new state.
`actions_after_newton_step()` cannot serve: it exposes no increment, and the retained increment exists in
unmixed form nowhere else — the dofs already carry it scaled by `Relaxation_factor`, and one step later
it is overwritten. The contract is deliberately narrow (raw `dx`, dofs updated, halos synchronised) so
that the distributed stage can keep it unchanged. Marked `//FOR PYOOMPH` and described in
`src/thirdparty/INFO_oomph-lib`.

**Only `newton_solve()` got the call, and that is enough**: `steady_newton_solve()`,
`newton_solve(max_adapt)`, all three `unsteady_newton_solve` overloads, `adaptive_unsteady_newton_solve`
and `doubly_adaptive_unsteady_newton_solve_helper` have no dof-update loop of their own and funnel into
it — `Relaxation_factor` occurs in exactly one place in the whole of `problem.cc`.
`newton_solve_continuation()` does have its own update loop and was left alone on purpose (see the table
below).

`pyoomph::Problem::actions_after_newton_dof_update` applies
`dx_L = y_C − X_C dx_{E_C}`, then `*value_pt -= Relaxation_factor * dx_L`. Notes:

* **The first line is `if (!last_jacobian_was_condensed) return;`** — with the feature off the hook is
  one bool test per Newton step.
* **The same `Relaxation_factor`** as the retained dofs, or the eliminated ones would describe a
  different point along the Newton direction than the rest of the solution.
* **The operators are consumed.** `ops_valid` is cleared after use and reaching a component twice
  without a reassembly in between throws. Each linearisation's operators belong to one Newton step;
  applying them twice would silently double an increment.
* **One gather function.** Recovering `dx_{E_C}` from `dx` is isolated in a single lambda, because that
  is the only thing the distributed reconstruction replaces (§9.6). This whole routine is the serial
  branch; distributed, `actions_after_newton_dof_update` hands over on its fourth line.
* The equation numbers of the `L_values` are re-checked under `PARANOID` only.

### Engagement matrix

`static_condensation_engages_now()` reads `use_static_condensation && (inside_flagged_newton_solve ||
_debug_force_condensed_assembly) && <the gates>`. `inside_flagged_newton_solve` is set by an RAII guard
in thin `pyoomph::Problem` wrappers around every Newton entry point, which do nothing else.

| route | behaviour |
|---|---|
| `newton_solve()`, `newton_solve(max_adapt)`, `steady_newton_solve(max_adapt)` | **condenses** |
| `unsteady_newton_solve(...)` (all three overloads) | **condenses** |
| `adaptive_unsteady_newton_solve(...)`, `doubly_adaptive_unsteady_newton_solve(...)` | **condenses** |
| `newton_solve_continuation()` / `arc_length_step()` | full — has its own dof update, which the hook does not see |
| a bare `assemble_jacobian()` / `get_jacobian()` from Python | full — nothing would reconstruct afterwards |
| `get_residuals()`, the Newton convergence check | full — always, by construction |
| eigenproblem, Hessian, multi-assembly, bifurcation tracking | full — non-default assembly handler |
| an augmented (`n_unaugmented_dofs != 0`) system | full |
| `use_static_condensation = True` with no rules | full, silently: nothing was selected |
| a **distributed** (`--distribute`) run | **condenses**, per §9 |
| **MPI without `distribute()`** (`nproc > 1`, replicated) | **throws** |
| a **component split across ranks** (distributed) | **throws**, collectively |
| **Jacobian reuse** (`jacobian_reuse_is_enabled()`) | **throws** |
| **globally convergent Newton** (line search) | **throws** |

The throws are combinations the user explicitly asked for that cannot be served; the "full" rows
are routes that legitimately want the full system, so they say nothing. `Use_globally_convergent_newton_method`
is private in `oomph::Problem` and has no getter, so `pyoomph::Problem` mirrors it in shadowing
`enable_/disable_globally_convergent_newton_method()`; every route into the flag — the nanobind binding
included — goes through a `pyoomph::Problem`, so the mirror cannot drift. Note that from Python
`solve(globally_convergent_newton=True)` is the way to reach it: `solve()` passes `False` on every call,
so setting the flag beforehand does not survive.

The wrappers shadow oomph-lib's non-virtual functions. That works because the nanobind bindings
([src/nanobind/problem.cpp](../src/nanobind/problem.cpp), the `*newton_solve*` block) name them as `pyoomph::Problem::`
members — verified in the object file, where all seven bound entry points resolve to `pyoomph::Problem::…`
and call the `oomph::Problem::…` original. oomph-lib's own internal calls resolve to the base versions,
which is right: they are already inside the outermost wrapper's guard. The guard saves and restores
rather than clearing, so a nested solve (a Python `actions_*` hook that solves again) returns to the
outer state, and being RAII it also resets when a Newton solve throws — which is not hypothetical, an
unconverged solve throws `NewtonSolverError` straight through.

Test levers: `_debug_force_condensed_assembly` (condense every Jacobian assembly, without a solve, so
the algebra can be refereed at a fixed state) and `_last_jacobian_was_condensed()` — the only direct
evidence that the path engaged rather than quietly declining. The latter is written by `get_jacobian()`
alone, so after an eigenproblem assembly it still reports on the preceding Newton step; check the size of
the matrix instead there.

---

## 7. What was verified

### The algebra, against scipy at a fixed state

2D CR Stokes (`RectangularQuadMesh(N=4, split_in_tris="left")`, 32 triangles, `mode="CR"`, bubble
velocities + DL values `[1,2]`, one pressure fixed; ndof 257, full nnz 4796, condensed nnz 2303, 32
components of size 4). scipy recomputes everything — components, `E` sets, `X`, `y`, the Schur
complements — from the full Jacobian and the selection alone.

| case | `J̃` vs `A_EE − A_EL A_LL⁻¹A_LE` | `r̃` | identity rows / cols, `r̃_L` | `dx̃_E` vs `dx_E` | `y_C − X_C dx̃_{E_C}` vs `dx_L` |
|---|---|---|---|---|---|
| plain CR | 2.4e-16 | 1.4e-16 | exact 0 | 3.4e-13 | 2.3e-13 |
| + interface writing the condensed **column** | 2.4e-16 | 1.4e-16 | exact 0 | 3.3e-13 | 2.4e-13 |
| + interface writing the condensed **row** too | 2.4e-16 | 1.4e-16 | exact 0 | 2.0e-13 | 2.5e-13 |
| Dirichlet by matrix manipulation | 2.4e-16 | 1.4e-16 | exact 0 | 2.2e-13 | 1.9e-13 |

(Relative deviations; the `dx` figures are the accuracy of two independent sparse solves, not of the
elimination.) The interface cases are **requirement 2 live**: a `lam` field on `"top"` whose equation
reads the bulk DL pressure adopts that pressure Data, so the interface element writes 12 entries into
`J[lam, L]` and, with the reverse coupling added, 12 more into `J[L, lam]` — the condensed dofs' own
rows, written by an element that does not own them. A per-element pre-scatter Schur complement would be
wrong in exactly those entries.

### Newton equivalence

Same 2D CR cavity, ndof 257. Each case run with the switch off and on, comparing **every** nodal value
including pinned ones, **every** element-internal value (the condensed DL gradient modes *and* the
constant mode that stays global) and the dof vector, at every step:

| case | Newton steps, off = on | max abs deviation, all values | residual histories |
|---|---|---|---|
| steady Stokes | 1 = 1 | 2.6e-12 | agree to 2.9e-13 |
| Navier–Stokes, Re = 200 | 5 = 5 | 1.0e-14 | 3.1e-14 |
| 5 BDF2 steps, Re = 100 | 3,3,3,2,2 each side | 1.1e-13 | 1.1e-15 |
| relaxation factor 0.7 | 17 = 17 | 3.6e-15 | 1.0e-15 |
| `refine_uniformly()` between two solves (257 → 1089 dofs) | 1, 1 | 9.2e-12 | 3.6e-13 |
| interface adopting the DL pressure (requirement 2) | 1 = 1 | 7.2e-12 | 7.5e-13 |
| adaptive-`dt` solve (`temporal_error=1e-3`, 4 steps) | 3,3,3,4; same `dt` sequence | 6.4e-14 | 1.5e-14 |
| steady solve + 3 arclength steps in Re | 4 = 4 | 9.9e-14 | 4.2e-14 |

`_last_jacobian_was_condensed()` is `True` after every one of those solves in the condensed arm and
`False` in the other, so engagement is measured rather than assumed. In the continuation case it is
`True` for the initial solve and `False` for all three arclength steps — the policy of §6, observed.

**These numbers are with a direct solver that solves to round-off.** With the machine's default
(pardiso) the iteration counts and the deviations of the *converged* states still agree (worst case
9.1e-8 on the refined mesh), but the intermediate residuals do not: pardiso leaves a residual of ~1e-6 on
this system, so the second entry of the history is the *solver's* error, which naturally differs between
two different-but-equivalent matrices, and the default Newton tolerance of 1e-8 then lets the converged
states differ by ~1e-8. The elimination is exact per step; the comparison only shows that once the linear
solves are. See §8, where pardiso does considerably worse than that on a larger CR system.

### Flagship cases

Three cases beyond the CR cavity of the test suite, each run with the switch off and on in the same
process. These were scratch scripts (`flagship1_th_projection.py`, `flagship2_free_surface.py`,
`flagship3_cr3d.py`) rather than tests — they are slower than the suite should be, and what they
establish is written down here:

**(a) Taylor–Hood + a projected field.** Lid-driven cavity at Re = 100, 288 triangles, `mode="TH"`, plus
the viscous dissipation `2μ(D:D)` projected onto a discontinuous space and condensed with
`condense_element_private_dofs()` — no rule written, no space named.

| | selected | components | full nnz → condensed | fill-in | steps | max abs Δ, all dofs |
|---|---|---|---|---|---|---|
| DL projection, one-way | 864 = 3/element | 288 × size 3 | 43 844 → 32 864 (75.0 %) | **0** | 6 = 6 | 5.1e-13 |
| D0 projection, one-way | 288 = 1/element | 288 × size 1 | 35 372 → 32 288 (91.3 %) | **0** | 6 = 6 | 5.7e-14 |
| DL, feeding back into momentum | 864 | 288 × size 3 | 53 096 → 32 864 (61.9 %) | **0** | 6 = 6 | 4.5e-13 |

The fill-in is zero in all three, and that is structural rather than lucky: a component here is one
element's internal Data, so its `E` set lies inside that same element, and every intra-element coupling
is in the full pattern already. It is also why the one-way case is free — `A_EL ≡ 0` makes the Schur
complement `A_EE` itself — and the third row exists to have a case where it is not.

**(b) A moving free surface on CR.** A modulated liquid layer relaxing under surface tension: ALE mesh
coordinates as unknowns, a kinematic-BC Lagrange multiplier, `NavierStokesFreeSurface`, 128 CR triangles,
ten BDF2 steps of dt = 0.02.

| | ndof | components | full nnz → condensed | fill-in | steps/step | max abs Δ (dofs / nodal / internal) |
|---|---|---|---|---|---|---|
| plain free surface | 1577 | 128 × size 4 | 41 792 → 33 412 (79.9 %) | 7 516 | 3 each, both arms | 2.0e-13 / 2.4e-15 / 2.0e-13 |
| + interface traction probe | 1610 | 128 × size 4 | 43 394 → 36 646 (84.4 %) | 9 436 | 3 each, both arms | 2.6e-13 / 1.2e-13 / 2.6e-13 |

Worth knowing: **a plain free surface does not touch the condensed dofs at all.** Its interface elements
adopt no bulk element-internal Data — the kinematic condition sees velocities and mesh positions and the
dynamic one is a natural boundary condition that never mentions the pressure — so 0 of the 512 selected
dofs are external data of an interface element. The requirement-2 situation needs an interface equation
that *reads* a bulk elemental field; the traction probe (`t = n·2μD·n − p` as an interface unknown) is
one, and it adopts 64 of the 512 (4 per top element: both bubble velocities and both pressure gradient
modes), because the bulk pressure Data is adopted wholesale and `grad(u)` at the face involves the
cell-interior bubble node, whose shape function vanishes on the face but whose gradient does not. The
fill-in is much larger here than on the fixed-mesh cavity because the mesh coordinates are unknowns too,
so a component's `E` set contains the element's position dofs as well.

**(c) Crouzeix–Raviart in 3D.** Tetrahedra (Kuhn split, `tests/box_mesh_3d.py`), bubble velocities plus
DL values `[1,2,3]`:

| | elements | ndof | selected | components | full nnz → condensed | fill-in | steps | max abs Δ |
|---|---|---|---|---|---|---|---|---|
| Stokes, N=2 | 48 | 632 | 288 = 6/element | 48 × size 6 | 26 511 → 14 468 (54.6 %) | 47 | 2 = 2 | 6.2e-11 |
| Navier–Stokes Re=50, N=3 | 162 | 2318 | 972 = 6/element | 162 × size 6 | 128 463 → 76 520 (59.6 %) | 161 | 5 = 5 | 3.6e-14 |

The 3D CR block is 6 = three bubble velocity components + three DL gradient modes, one component per
element, exactly as intended — in particular the C2TB tetrahedron's **face** bubbles, which two elements
share, are correctly not selected; had they been, the components would have merged across faces.

### Gates and non-interference

Measured: a bare `assemble_jacobian()` with the switch on and rules declared comes back with the full
nonzero count and `_last_jacobian_was_condensed() == False`; a Newton solve forced to fail throws and the
next bare assembly is full again, i.e. the RAII guard reset on the way out of the exception; the
eigenvalues of a problem solved with condensation equal those of one solved without, and the
eigenproblem's own Jacobian comes back with the full nnz; arclength continuation tracks the same branch
with the switch on as with it off, condensing the initial solve and nothing after it; six repeated
condensed assemblies in one process are bit-stable and rebuild the plan zero times, i.e.
`build_without_copy`'s ownership transfer is balanced. `tests/test_static_condensation.py` is 37 passed
in ~12 s (one of them marked `slow`, so 36 in a default run), and `tests/test_structural_assembly.py` is
unchanged at 42 passed.

### The equation-tree interface

Section 7 of the test file covers it: the same two rules stated as
`StaticCondensation(velocity="bubble", pressure=[1,2])` select the same equation numbers and produce the
same plan (18 components of size 4) as the two `condense_dofs` calls; a steady Navier–Stokes solve at
Re = 100 and a solve across a `refine_uniformly()` agree with the uncondensed arm to 1e-9 in every dof
and every element-internal value, at the same iteration count; the switch comes on by itself and an
explicit `use_static_condensation = False` set before initialisation survives every re-registration
(with the rules still declared, so it is the switch that is being tested and not an empty selection);
four transient solves in a row leave `jacobian_structure_id` and the plan-rebuild count untouched, which
is the idempotency requirement; a remesh (`force_remesh()` on a Gmsh domain) leaves both the
equation-tree and the problem-level rules pointing at the *new* mesh, and the solve after it still
condenses; and `StaticCondensation()` on one of two independent D0 domains condenses that domain only.

The flagship 2D CR cavity of §8 rewritten in the equation-tree style reproduces its structure exactly
(N = 20, Re = 100: ndof 7041, 3200 condensed = 45 %, nnz 168 636 → 88 063 = 52 %, fill-in 799, one plan
build, six Newton steps both arms, `max|Δdof|` 2.1e-13 against the uncondensed run) — as it must, since
only the declaration moved.

Refusals, verbatim (abridged):

> Static condensation: the selected degrees of freedom cannot be eliminated, because the block to be
> inverted is structurally singular. The equations of these selected dofs contain no selected unknown at
> all (an empty ROW of the block): domain/pressure__DL__1 (eqn 163), … A dof can only be eliminated
> together with an equation that determines it. The classic case is selecting a discontinuous (DL)
> pressure on its own: the continuity equation contains no pressure, so it has to be condensed jointly
> with the bubble velocities that it does determine — and the constant pressure mode has to stay in the
> global system.

> Static condensation: the selected degrees of freedom do not decompose into small element-local blocks.
> One connected component holds 96 mutually coupled dofs, above the limit
> static_condensation_max_component_size = 64. […] Either restrict the selection to dofs that really are
> element-local (a DG field coupled across facets is not), or raise
> problem.static_condensation_max_component_size if the size is genuinely intended.

> Static condensation: the block of selected dofs is numerically singular and cannot be eliminated. The
> elimination of a connected component of 5 selected dofs broke down at step 2: no pivot above 1e-12
> times the size of the block was left. […] The classic case is taking the CONSTANT mode of a
> discontinuous (DL) pressure along with the bubble velocities: the bubble has no coupling to it (the
> integral of div(u_bubble) against a constant test function vanishes), so value 0 has to stay a global
> unknown and only the gradient modes may be condensed — values=[1,2] in 2D, [1,2,3] in 3D.

The second is an interior-penalty DG Poisson (`space="D1"`) with the whole field selected: the facet
terms couple every element to its neighbours, so all 96 dofs of a 32-element mesh are one component. The
same guard fires on the CR case at `max_component_size = 3` (component size 4). The third is reached
through the real path with values `[0,1,2]` — it passes the structural pre-check, because the pattern
does contain a pressure→bubble-velocity coupling, and breaks down at the third pivot of every 5×5 block.

One finding on the side, unrelated to condensation but hit while validating the Dirichlet ordering: with
`apply_Dirichlet_BCs_by_dof_removing = False`, `remove_dirichlets_by_matrix_manipulation()` writes the
identity into *existing* entries only, so a Dirichlet dof whose diagonal the field-coupling pruning
removed keeps an all-zero row and the matrix is exactly singular. It is invisible in practice because the
solvers that need this route ask for `force_jacobian_diagonal_entries` anyway.

---

## 8. Does it pay?

Yes, substantially — and the answer is not the trivial one, because the eliminated dofs do not come for
free: the Schur complements make the retained rows denser and the kernel costs assembly time.

**How it was measured** (scratch script `bench_condensation.py`). In process, never as the wall time
of a script ([structural_assembly.md](structural_assembly.md)'s lesson). The factorisation and the
back-substitution are timed by subclassing the linear solver and bracketing its own `solve_serial()`
(`op_flag` 1 and 2), so those numbers are the solver's real cost and nothing else; the Jacobian setup is
oomph's `Jacobian_setup_time`, collected per step in `actions_after_newton_step()`, which brackets the
elemental assembly, the Dirichlet handling **and** the condensation kernel. The two arms are
**interleaved** (A, B, A, B, …), three rounds, the first discarded as a warm-up, the best of the rest
reported; nothing else ran on the machine. Engagement was asserted before anything was believed
(`_get_frozen_sparsity_nnz() > 0` and `_last_jacobian_was_condensed()`), and the two arms' converged
states were compared (the `max|Δdof|` column) so that a comparison of two *different* answers cannot be
mistaken for a speed-up.

**2D lid-driven cavity, CR Navier–Stokes at Re = 100, scipy SuperLU:**

| N | ndof | condensed | nnz full → condensed | fill-in | steps | factorisation | back-subst. | Jacobian setup | total `solve()` | max\|Δdof\| |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 7 041 | 3 200 (45 %) | 168 636 → 88 063 (52 %) | 799 | 5 | 0.316 → 0.133 s **−58 %** | −52 % | +13 % | 0.475 → 0.299 s **−37 %** | 2.2e-13 |
| 40 | 28 481 | 12 800 (45 %) | 701 756 → 369 503 (53 %) | 3 199 | 4 | 2.95 → 1.30 s **−56 %** | −57 % | +15 % | 3.56 → 1.89 s **−47 %** | 7.6e-13 |
| 60 | 64 321 | 28 800 (45 %) | 1 599 676 → 844 543 (53 %) | 7 199 | 4 | 11.97 → 4.72 s **−61 %** | −59 % | +21 % | 13.27 → 6.10 s **−54 %** | 4.2e-12 |
| 80 | 114 561 | 51 200 (45 %) | 2 862 396 → 1 513 183 (53 %) | 12 799 | 4 | 31.95 → 11.33 s **−65 %** | −60 % | +24 % | 34.16 → 13.68 s **−60 %** | 5.2e-12 |

**3D box, CR Navier–Stokes at Re = 50 on tetrahedra, scipy SuperLU:**

| N | elements | ndof | condensed | nnz full → condensed | steps | factorisation | Jacobian setup | total `solve()` | max\|Δdof\| |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 162 | 2 318 | 972 (42 %) | 128 463 → 76 520 (60 %) | 4 | 0.311 → 0.122 s **−61 %** | +7 % | 0.567 → 0.386 s **−32 %** | 3.6e-14 |
| 4 | 384 | 5 732 | 2 304 (40 %) | 364 479 → 224 900 (62 %) | 4 | 2.31 → 1.04 s **−55 %** | +8 % | 2.98 → 1.73 s **−42 %** | 4.6e-14 |
| 5 | 750 | 11 486 | 4 500 (39 %) | 792 015 → 497 984 (63 %) | 4 | 11.79 → 5.80 s **−51 %** | +9 % | 13.17 → 7.22 s **−45 %** | 1.6e-13 |

Reading these:

* **The factorisation more than tracks the nnz reduction.** Removing 45 % of the dofs removes 47 % of the
  nonzeros — the fill-in is a rounding error here, 12 799 entries against 1.35 million removed — and buys
  56–65 % of the factorisation time. It is superlinear because a direct factorisation is, and it *grows*
  with the problem size in 2D, which is the right direction: the bigger the run, the more it is worth.
* **The kernel is cheap and the assembly pays for it.** The Jacobian setup grows by 7–24 %, i.e. by
  0.07 s per Newton step at N = 80, against 5.15 s per step saved in the factorisation — a ratio of about
  1 : 70. Timing the kernel directly (the same assembly with and without `_debug_force_condensed_assembly`,
  5 repeats at the converged state) gives −1 % to +8 % of one assembly, i.e. it sits in the noise of a
  single assembly and only the setup-time column resolves it at all.
* **Back-substitution halves too**, but it was never the cost.

**With pardiso the picture is different, and it is pardiso's doing.** Its static pivoting copes badly
with this CR system: backward errors of 1e-2…1e-1 with `repair_bad_solves` off, at Re = 100 and N ≥ 40 the
**uncondensed** arm does not converge within 10 Newton steps at all, and where both arms do converge the
condensed one needs *more* steps (5 vs 3 at N = 20, Re = 10) because the condensed matrix is a different
matrix and pardiso's error on it is different. Per Newton step it is still much faster —
31.4 → 10.1 ms of factorisation, −68 % — but the extra iterations eat the gain and the total comes out
roughly neutral (+13 % in one run). This is the same phenomenon as the residual-history caveat in §7,
one level up: **a condensed matrix is not the same matrix, and a solver whose accuracy is marginal on the
original will not behave identically on it.** Judge condensation with a solver that solves the system.

**Where it will pay less.** The projection cases of §7(a) reduce the nnz by 8–38 % with zero fill-in, so
they can only help. A selection with a large `E` set relative to `L` gets much less: on the moving-mesh
free surface of §7(b), the fill-in is 7 500 entries against 8 400 removed, so the matrix only shrinks to
80 % of its size for the same kernel cost. That case was not benchmarked — ten steps of it take a third
of a second either way, which is below the noise floor — but the structural numbers already say the
answer will be small. The rule of thumb the measurements support: condensation pays when the eliminated
dofs are a large *fraction* of the system and their `E` sets are small, which is exactly the CR case it
was built for.

---

## 9. Distributed condensation

Status: **implemented and tested at 2 and 4 ranks** (§9.7), not benchmarked. The design below was
written before the code and is kept in step with it; the one place where the implementation forced a
change is called out as such, at the end of §9.4.

The serial design was built so that this stage would be an extension rather than a rewrite, and that
held: components are keyed by global equation numbers, every runtime access goes through precomputed
slot lists, and the reconstruction is formulated in terms of `dx_E` rather than of a global increment
vector. What the distributed stage adds is *who holds what*, one structural exchange to answer that, and
one value exchange per assembly.

### 9.1 What each rank has, and what follows from it

Two facts about pyoomph's distributed assembly decide the whole design
([structural_assembly.md](structural_assembly.md) §5):

1. **A rank's OWNED rows are complete.** `parallel_sparse_assemble` — oomph-lib's and pyoomph's frozen
   replacement alike — ships every row a rank contributed to but does not own to its owner, and the owner
   merges. So for a row `r` this rank owns, it holds *every* entry of `r`, including contributions from
   elements that live on other ranks.
2. **Row ownership is dof ownership.** For a distributed problem oomph numbers each rank's own dofs
   contiguously (`Dof_distribution_pt->build(comm, my_eqn_num_base, my_n_eqn)`,
   [problem.cc:17242](../src/thirdparty/oomph-lib/include/problem.cc)), so the row range a rank owns is
   exactly the set of `Data` values it owns non-halo.

From (2), the L dofs of a component — the internal `Data` of one element, or its cell-interior bubble
node — are owned by the rank on which that element is non-halo. From (1), that rank holds `A_LL`,
`A_LE` and `r_L` **complete**, which is everything the Schur complement needs. And a *neighbour* rank
that owns one of the component's E rows holds that row complete too, including its `A_EL` entries in the
component's L columns — but it does not hold the L rows and cannot form `X_C` itself.

So the work splits without ambiguity:

| | who | needs |
|---|---|---|
| factorise `A_LL`, form `X_C = A_LL⁻¹A_LE` and `y_C = A_LL⁻¹r_L` | the **owner** | nothing from anyone |
| `r_E −= A_EL·y_C`, `cond[E_a,E_b] −= A_EL·X_C` for an owned row `E_a` | **whoever owns that E row** | `X_C`, `y_C`, and the E ordering |
| identity rows, `r_L = 0`, and writing the reconstructed `dx_L` back | the **owner** | — |

### 9.2 Mechanism: exchange, not replication

Two mechanisms were on the table.

**(a) Exchange.** The owner factorises and sends `X_C` (`nL×nE` doubles) and `y_C` (`nL`) to each rank
owning an affected E row, mirroring the value exchange of `assemble_distributed_with_frozen_sparsity`.

**(b) Halo re-assembly.** Every rank holding the component's patch — owner or halo — re-evaluates the
elemental blocks locally and factorises identically, so nothing is communicated.

**(a) is implemented.** (b) is attractive on paper and its completeness condition is exactly what cannot
be checked cheaply: it needs *every* element contributing to the component's L rows to be present in the
rank's halo layer, interface elements that adopt the dof included — and requirement 2 of §2 (an interface
element writing into a condensed dof's own row) is precisely the case where that is in question. Getting
it wrong does not throw; it silently factorises a truncated `A_LL` on some ranks and produces a Schur
complement that is wrong by an amount nobody measures. (a) has a known-good template in the same file,
its buffers are contiguous, and its cost is small: one `Isend`/`Irecv` per peer per assembly carrying
`nL·(nE+1)` doubles per shared component — for 2D CR, 4·(nE+1) ≈ 50 doubles for each element whose patch
straddles the partition boundary, i.e. proportional to the *interface*, not to the mesh.

(b) remains a possible optimisation for selections known to be interior to one element, where the
completeness condition is trivially satisfied. It is not implemented and would need the condition to be
verified rather than assumed.

### 9.3 The plan build: three structural exchanges, then a vote

`build_distributed_condensation_plan()` runs once per Jacobian structure id, off the **distributed**
frozen pattern (`DistributedFrozenSparsity::Mat::final_row_start/final_col`, i.e. the merged owned rows).

0. **Rank-local.** Resolve the selection (already rank-local and deterministic by design, §3) and keep
   only the values whose equation this rank owns. The selection scan sees halo copies too and therefore
   also names dofs owned by other ranks; those are that rank's business, not this one's.
1. **Exchange 1 — whose columns are condensed?** A rank needs to know, for every *foreign* column
   appearing in its owned rows, whether it is condensed. It could decide that from its own halo copy of
   the `Data`, and would nearly always be right — but `condense_element_private_dofs()` measures
   "private" against a *rank-local* reverse external-data scan, which a neighbour may see differently.
   So the owner is asked: each rank sends its sorted list of foreign columns to their owners and gets one
   flag back per column. Authoritative, and proportional to the partition boundary.
2. **Components, and the cross-rank refusal.** Union–find on the owned L rows, exactly as serially. If an
   owned L row has an L column this rank does *not* own, the component straddles ranks: refuse. That test
   catches every such component from at least one side (the other rank sees the mirror-image entry from
   its own L row, or, if the coupling is one-directional, the rank owning the row sees it), and the
   refusal is collective, so it cannot deadlock. The size and structural-invertibility guards of §4 run
   unchanged on the owned block; `col_ok` is computable locally precisely *because* a component that
   reached across a rank boundary has already been refused.
3. **Exchange 2 — the "by row" half of E.** `E_C` is the union of the retained dofs appearing in the
   component's rows (owner-visible) and of the retained rows whose own equation contains one of its
   unknowns. The second half lives wherever those rows live, so every rank scans its owned retained rows
   for condensed columns and sends the `(row, condensed column)` pairs to the column's owner, which
   merges them into `E_C`. Without this the Schur complement would miss exactly the couplings that made
   this a post-assembly elimination in the first place.
4. **Exchange 3 — the component descriptors.** The owner now has the final `E_C`, sorted. It ships
   `(L_eqns, E_eqns)` to every rank owning at least one E row. That rank stores the component as a
   *remote* component: no `L_values`, no `A_LL`, but the full E ordering it needs to index `X_C`, and
   `my_E`, the indices of the E rows it owns.
   Note that a rank learns of its participation from this message and not from its own scan: the
   by-column direction (its row is a column of one of the owner's L rows) is not visible to it at all.
5. **The local condensed pattern**, built exactly as serially but over the owned rows only: a retained
   owned row keeps its retained columns and gains the whole `E_C` of every component it takes part in
   (owned or remote); a condensed owned row keeps its diagonal. Columns are global, so a fill entry in a
   column this rank never had is unremarkable.
6. **Slot lists.** `LL_slots`/`LE_slots` on the owner; `EL_slots` and `fill_slots` sized `|my_E|×nL` and
   `|my_E|×nE` on every participant; pass-through and identity-row slots as before.
7. **The exchange plan.** Per peer, the list of components whose `X_C`/`y_C` travel, ordered by the
   component's smallest L equation — a globally unique, deterministic key that both sides can sort by
   without agreeing on anything further. Sender and receiver therefore lay out the same buffer.
8. **One collective vote** on success, and one on refusal (below).

**The ordering of the components in the plan array is by smallest L equation**, with remote components
interleaved among the owned ones. That is what makes the per-peer send and receive lists line up.

### 9.4 Collective refusal

Every guard that throws serially — size, structural singularity, the numeric pivot — becomes a
**collective vote**: each rank forms a message or the empty string, an `MPI_Allreduce(MAX)` over
"rank that has a message, else −1" picks one, that message is broadcast, and **every** rank throws it,
prefixed with the rank it came from. A one-sided throw would leave the others in the next collective for
ever, which is the lesson [structural_assembly.md](structural_assembly.md) §5 records in its own words:
half the ranks in one code path and half in another is a deadlock, not a wrong answer.

The same applies to the quiet declines. `static_condensation_engages_now()` returns a globally uniform
answer because every input to it is globally uniform (the switch, the flagged-solve guard, the assembly
handler type, `n_unaugmented_dofs`, and `jacobian_structure_id`, which is `MPI_Allreduce`-checked for
equality under `PARANOID`), and because the plan build itself votes. That is what makes
`last_jacobian_was_condensed` — and hence whether the reconstruction hook does anything — the same on
every rank.

Two new refusals are distributed-only:

* **A component straddling ranks** (step 2 above). The message names the dofs and says why: the block to
  be inverted is not held by any one rank.
* **The matrix distribution is not the dof distribution** — and this is the one point where the design
  above did not survive contact, so it is worth stating in full.

  §9.1's second fact is that row ownership *is* dof ownership. That is true of `Dof_distribution_pt`.
  It is **not** true of the distribution the Jacobian actually arrives in. Two separate places choose it,
  and by default both choose something else:

  * `Problem::create_new_linear_algebra_distribution()` picks between `Dof_distribution_pt` and a
    uniform one and switches to uniform as soon as one rank's dof count exceeds the uniform share by
    10 % ([problem.cc:454](../src/thirdparty/oomph-lib/include/problem.cc));
  * and a distributed **solve** never reaches that function at all. `SuperLUSolver::solve(Problem*,
    DoubleVector&)` — which is the route pyoomph's `petsc_mumps` and every other distributed backend
    take, through `superlu_dist_distributed_matrix()` — builds `LinearAlgebraDistribution(comm, ndof,
    true)`, i.e. an unconditionally **uniform** split, and hands `get_jacobian()` a matrix already
    carrying it.

  Under a uniform split the rank owning a row generally does not hold that degree of freedom: it cannot
  write the reconstructed value back, and a component's L rows are no longer even guaranteed to be on
  one rank. The whole construction collapses, and it collapses *silently* — which is how it was found,
  as the refusal below firing on the very first 2-rank run.

  So both places are steered rather than merely checked. The first is resolved in favour of the dof
  distribution by `align_matrix_distribution_for_condensation()`, called from the Newton-entry guard
  whenever condensation is on. The second needed a second vendored hook,
  `Problem::prefer_dof_distribution_for_linear_solver()` (`//FOR PYOOMPH`, and
  `src/thirdparty/INFO_oomph-lib`): a public virtual returning `false` in oomph-lib, overridden in
  `pyoomph::Problem` to return `use_static_condensation`, and consulted in `SuperLUSolver::solve`'s
  distributed branch to build the row distribution from `problem_pt->dof_distribution_pt()` instead.
  It is opt-in, so no other run changes at all; and where it is on, it only removes the
  `dx.redistribute(Dof_distribution_pt)` that `newton_solve()` performs immediately afterwards anyway.

  What remains is a genuine check: the plan builder compares the pattern's `first_row`/`nrow_local`
  against `Dof_distribution_pt` and refuses collectively if they differ, which now means the user chose
  a uniform distribution explicitly. Overriding *that* behind their back would be the wrong kind of
  helpful.

### 9.5 The kernel, per assembly

1. Positive engagement check against the distributed frozen pattern: `nrow_local`, `nnz`, then `memcmp`
   of `row_start` and `column_index` — the same check as serially, on the local block.
2. Pass-through copy into the condensed local value array.
3. **Owner:** gather `A_LL`, `A_LE`, `r_L` from owned rows; factorise; keep `X_C`, `y_C`.
4. **Vote on the pivot guard** before any of the exchange happens, so a breakdown throws on all ranks
   instead of hanging the rest in `MPI_Waitall`.
5. Pack `X_C`, `y_C` per peer and post the sends and receives.
6. **Everyone:** for each owned E row of each component, gather `A_EL` from that row and apply
   `r_E −= A_EL y` and `cond[E_a,E_b] −= A_EL X`. Owned components are done while the messages fly.
7. **Owner:** `r_L = 0`, identity diagonal.
8. Rebuild the local `CRDoubleMatrix` on the condensed pattern with `build_without_copy`, which keeps the
   distribution.

### 9.6 Reconstruction, and why it needs one extra synchronisation

The hook fires after `*Dof_pt[l] -= Relaxation_factor*dx[l]` and after `synchronise_all_dofs()`.

**`dx_E` is recovered without exchanging `dx`**, as the serial design promised. For an E dof this rank
owns, `dx` has it directly (exact). For a foreign E dof, the value trick of §2 applies:
`dx_E = (u_stored − u_now)/Relaxation_factor`, where `u_stored` was captured at *kernel* time from this
rank's halo copy and `u_now` is read from the same copy after the synchronisation. That is sound because
oomph's `synchronise_dofs()` copies haloed **node values and haloed elements' internal data**
([problem.cc:16826](../src/thirdparty/oomph-lib/include/problem.cc)), so the halo copy equals the owner's
value at both instants, and the difference is therefore exactly `Relaxation_factor·dx_E`. The hybrid —
`dx` where it is available, the difference only across the boundary — keeps the cancellation error
(`~ε|u|/Relaxation_factor`, absolute, i.e. ~1e-16 on an O(1) field) off every interior dof, and leaves
the serial path using `dx` alone, untouched.

Every foreign E dof *is* reachable on the owner: it is a dof of one of the owner's non-halo elements (an
E dof only arises as a column of an L row or as a row containing an L column, and both come from elements
that contain the L dof itself), so a halo copy of it exists locally. The plan resolves it to a `double*`
once, at build time.

**What does need communication is the result.** Only the owner writes the reconstructed `dx_L`, so every
*halo* copy of a condensed dof is stale the moment the hook returns — and a neighbour's non-halo
interface or facet element may read exactly that copy in the next residual assembly. The hook therefore
ends with one `synchronise_all_dofs()` when anything was reconstructed. It is collective and safe
precisely because `last_jacobian_was_condensed` is globally uniform (§9.4). This is a deliberate
deviation from "replicate the reconstruction on every rank holding the `Data`": replication would need
`X_C`/`y_C` shipped to ranks that own no E row at all, and the condition "a rank holding a halo copy of
the patch owns an E row" is exactly the kind of thing that is nearly always true and occasionally not.
One extra halo exchange per Newton step is a small, *measurable* price for not having to be right about
that.

### 9.7 What was verified

The gate is `tests/test_mpi_static_condensation.py` + `tests/mpi_condensation_worker.py`, 11 tests
(marked `slow`, so `--full` is needed), each launching `mpirun -n {2,4} ... --distribute` under a bounded
timeout — a timeout being the only way a one-sided refusal shows up at all. The reference is the
**serial, uncondensed** run of the same problem, computed in-process from the same worker module, so a
globally consistent but wrong field cannot pass by agreeing with itself.

The comparison is by integral moments (`∫u`, `∫x·u`, `∫|u|²`, `∫|∇u|²`, `∫p²`, `∫x·p`) plus the sum and
sum-of-squares of every **element-internal** value over the non-halo elements, MPI-reduced. Moments
rather than a dof-by-dof diff because a distributed dof vector has no partition-independent ordering;
the element-internal checksums because the DL gradient modes are exactly what is eliminated, and an
integral observable of the retained velocity field would barely notice a reconstruction that had left
them at their previous values.

2D CR lid-driven cavity, Re = 100, `RectangularQuadMesh(split_in_tris="left")`, bubble velocities plus
DL values `[1,2]`, `petsc_mumps`:

| | ndof | components | Newton steps (two solves) | worst relative deviation from serial |
|---|---|---|---|---|
| N = 8, 2 ranks | 1 089 | 128 | 6, 2 — identical to serial | 1.9e-14 |
| N = 8, 4 ranks | 1 089 | 128 | 6, 2 — identical | 4.4e-14 |
| **N = 20, 2 ranks** | **7 041** | 800 | 6, 2 — identical | **1.0e-13** |
| **N = 20, 4 ranks** | **7 041** | 800 | 6, 2 — identical | **2.4e-13** |

The worst figure is the *sum* of the 3 200 eliminated internal values in every case — a cancelling sum,
hence the least favourable of the ten quantities; the integral moments agree to 1e-16…1e-13. Three BDF2
time steps agree the same way, at the same step counts (4, 4, 4), and the interface case of §7 —
requirement 2, an interface element writing into the condensed dofs' own rows, now possibly from a
different rank — agrees at 2 ranks with the serial run to the same tolerance.

**The cross-rank paths are asserted to have engaged**, not assumed — §4.1 of
[structural_assembly.md](structural_assembly.md) in a new costume, since a "distributed" plan in which
every component happened to be interior to its rank would pass every equivalence test above while
exercising nothing. At N = 8 on 4 ranks: 128 components split 33/31/32/32 by owner, **35 of them shared**
(35 operator sends against 35 receives — checked as a pair, because a mismatch is a rank waiting for a
message nobody sends), and **136 E dofs recovered from values** rather than from `dx`. At N = 20 on 4
ranks, 800 components split 200/200/201/199, 91 shared and 364 value-recovered; at 2 ranks, 41 and 176. The plan is built **once** for two solves and
survives a time step, which is the other thing worth asserting: it costs three communication rounds.

Both refusals are checked at 2 and 4 ranks for *returning at all*, and for every rank reporting the
identical message with the deciding rank named in it: the structurally singular selection (the whole DL
pressure) and the component split across ranks (an interior-penalty DG field, with the size guard raised
out of the way so that it is the cross-rank guard that speaks). A replicated `mpirun` without
`--distribute` is refused too.

Serially, `tests/test_static_condensation.py` (37) and `tests/test_structural_assembly.py` (42) are
unchanged, and `tests/test_mpi_structural_assembly.py --full` (17) still passes — the vendored
`prefer_dof_distribution_for_linear_solver` hook is off for it, which is what that run confirms.

### 9.8 What is not covered

* **Only genuinely distributed problems** (`--distribute`). `nproc > 1` without distribution is a
  replicated problem whose elements are merely split across ranks; the frozen distributed assembly
  declines it for its own reasons ([structural_assembly.md](structural_assembly.md) §5) and so does
  condensation, which needs that assembly's pattern. It is a refusal with a message, not a silent
  fallback.
* **Components must be rank-local.** A DG selection whose coupling crosses a partition boundary is
  refused. Serving it would mean a distributed dense solve per component, which is not what condensation
  is for.
* Everything serially refused stays refused (Jacobian reuse, the line search, continuation, eigenvalue
  problems).

---

## 10. Limitations, and what v3 would add

**Jacobian reuse and the line search are refused**, for the reasons in §6. Both are real restrictions:
`solve(globally_convergent_newton=True)` on a problem with condensation on is an error, not a fallback.

**Continuation is deliberately uncondensed**, so an arclength sweep gets no benefit at all. Giving it one
means teaching `newton_solve_continuation()`'s own dof-update loop about the reconstruction, i.e. a second
vendored hook, and dealing with the augmented system's numbering. Not attempted.

**Eigenvalue problems assemble full.** A condensed generalised eigenproblem is only the same spectrum if
`M_LL = 0` — detectable, since `contributes_to_mass_matrix` says exactly that — but for CR the bubble
velocities carry mass, so the general case needs a nonlinear (rational) eigenproblem and was not
attempted. Assembling full is always correct, just not faster.

**Bubble positions on moving meshes are unvalidated.** Selecting bubble *field values* on a moving mesh
is fine and is what flagship (b) does; selecting a bubble node's *position* dofs is representable but has
never been run.

**Pardiso is a poor partner for CR.** Not a condensation limitation, but it dominates any measurement
made with it: see §8.

**The stats are only as fresh as the plan.** `_get_static_condensation_stats()` deliberately does not
build a plan (a diagnostic must not change what it measures), so `has_plan` is False until a solve or an
explicit `_build_condensation_plan()`. Note that on a distributed problem `_build_condensation_plan()`
*is* collective, so calling that one on a single rank hangs — which is the price of the diagnostic being
allowed to force a build at all.

**Distributed: not benchmarked.** §9 establishes that the distributed answer is the serial one at the
serial iteration count, and nothing more. What condensation buys under MPI — a smaller matrix for the
distributed factorisation, against one operator exchange and one extra halo synchronisation per Newton
step — has not been measured, and the tests are deliberately far too small to say anything about it
(1 089 and 7 041 dofs). The serial measurements of §8 are the reason to expect it to pay; they are not
evidence that it does.

**Distributed: the extra `synchronise_all_dofs()` is unconditional.** §9.6 argues why it is needed in
general. It is not needed when no condensed `Data` is ever referenced from another rank — which is the
common case, since a bulk element's internal `Data` is normally read by nothing but that element. The
reverse external-data scan already computed for `condense_element_private_dofs()` would answer it, and
turning that into a plan-time flag would remove one collective per Newton step. Not attempted.

### The optional escape hatch

If a selection ever cannot be expressed as a rule, the fallback is a vendored `Data::set_condensed(i, bool)`
flag, which the resolution pass would simply OR into the selection. It is deliberately not implemented:
a flag on `Data` does not survive adaptation any better than a stored `(Data*, value)` pair does, so it
would be a per-run marker the user has to reapply, and every case met so far (CR, DL/D0/DG fields,
element-private auto-detection) is expressible as a rule.

---

## 11. Where things are

| | |
|---|---|
| [src/problem.hpp](../src/problem.hpp) | `StaticCondensationRule`, `CondensationComponent`, `CondensationPlan`, the flags, the Newton-entry-point wrappers, `align_matrix_distribution_for_condensation`, `prefer_dof_distribution_for_linear_solver` |
| [src/problem.cpp](../src/problem.cpp) | `update_static_condensation_selection`, `build_condensation_plan`, `apply_static_condensation`, `actions_after_newton_dof_update`, `static_condensation_engages_now` |
| [src/problem.cpp](../src/problem.cpp), §9 half | `build_distributed_condensation_plan`, `apply_static_condensation_distributed`, `reconstruct_condensed_dofs_distributed`, `condensation_collective_throw` |
| [src/nanobind/problem.cpp](../src/nanobind/problem.cpp) | the bindings and their docstrings |
| [pyoomph/equations/generic.py](../pyoomph/equations/generic.py) | `StaticCondensation`, the user-facing interface: spec parsing, the registration hook, the interface/ODE refusals |
| [pyoomph/generic/problem.py](../pyoomph/generic/problem.py) | `condense_dofs`, `condense_element_private_dofs`, and the rule registry (`_declare_static_condensation_rules`, `_sync_static_condensation_rules`, `_clear_static_condensation_rules`) |
| [src/thirdparty/oomph-lib/include/problem.h](../src/thirdparty/oomph-lib/include/problem.h) / `problem.cc` | the vendored `actions_after_newton_dof_update` hook (`//FOR PYOOMPH`, and `src/thirdparty/INFO_oomph-lib`) |
| [src/thirdparty/oomph-lib/include/problem.h](../src/thirdparty/oomph-lib/include/problem.h) / `linear_solver.cc` | the vendored `prefer_dof_distribution_for_linear_solver` hook (§9.4), same two places |
| [tests/test_static_condensation.py](../tests/test_static_condensation.py) | selection, plan, refusals, the scipy referee, Newton equivalence, non-interference, the equation-tree interface |
| [tests/test_mpi_static_condensation.py](../tests/test_mpi_static_condensation.py) + [tests/mpi_condensation_worker.py](../tests/mpi_condensation_worker.py) | the distributed gate: equivalence with serial at 2 and 4 ranks, the collective refusals, and the assertions that the cross-rank paths engaged at all |

> Worth knowing when writing scratch scripts for this: several `Problem` objects can live in one
> interpreter as long as each one is used inside its `with` block, which is what the test file relies on.
> Creating a second one *without* closing the first segfaults in the JIT loader
> (`CCompiler::get_init_func` → `dlsym`) — pre-existing and unrelated to condensation, but it is why the
> stage-3 and stage-4 harnesses used one process per case.
