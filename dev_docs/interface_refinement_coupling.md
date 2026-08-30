# Conforming refinement across coupled domain interfaces

Status: **implemented**, serially and under MPI. §1–§11 are the design; **§13 records what building it
turned up that the design did not predict** — including two defects the design's own reasoning would
have produced, and one place (§2) where it is simply wrong about distribution. **§14 is the mixed
element-space sweep**: it holds for every space pair on a flat or macro-element-curved interface, and it
does not hold once the interface geometry is a genuine unknown.

---

## 1. The problem

Two bulk domains built from the same `MeshTemplate` — say `domainA` and `domainB` — can share a
geometric interface. Their nodes at that interface are *distinct objects*: continuity is imposed
weakly, by Lagrange multipliers (`ConnectFieldsAtInterface`) or by position constraints
(`ConnectMeshAtInterface`). The two sides are wired together by an *opposite-interface connection*
declared on the template, which at runtime pairs up the interface elements of `domainA/if` with
those of `domainB/if`.

That pairing is exact and one-to-one. `InterfaceMesh::connect_interface_elements_by_kdtree`
([src/mesh.cpp:4884](src/mesh.cpp#L4884)) indexes every vertex position of side B in a KD-tree, then
looks up side A's vertices and demands an element of B whose vertex-index *set* matches exactly:

```
if (!nodes_to_elemB.count(indices)) { throw_runtime_error("Cannot locate opposite element"); }
```

Refinement breaks this. oomph-lib adapts meshes **individually** — `Problem._adapt_with_interfacial_errors`
loops `for name, errors in errs.items(): mesh.adapt_by_elemental_errors(errors)`
([pyoomph/generic/problem.py:1814-1822](pyoomph/generic/problem.py#L1814-L1822)) — and nothing makes
the two sides arrive at the same decision. As soon as one side refines a facet the other does not,
the vertex sets no longer correspond and the run dies in the matcher, or (worse, in the cases where
the matcher happens to still find *a* partner) silently couples the wrong elements.

### What exists today

Two partial mitigations, both in Python, both insufficient:

1. **`InterfaceMesh._override_bulk_errors_where_necessary`**
   ([pyoomph/meshes/mesh.py:1432](pyoomph/meshes/mesh.py#L1432)) pushes an interface element's error
   onto the adjacent bulk element and, when an opposite side is present, onto the opposite bulk
   element too, via the `get_opposite_bulk_element()` pointer.

2. **The block at [pyoomph/generic/problem.py:1753-1783](pyoomph/generic/problem.py#L1753-L1783)**,
   comment `# Ensure same refinement at connected interfaces`. One pass, no iteration: for each
   interface element, if my bulk element is above `min_permitted_error` and the opposite one is not,
   bump the opposite one to `0.5*(min+max)` and vice versa.

Thirty lines later the file still says
`# TODO: Ensure same refinement at connected interfaces` ([:1787](pyoomph/generic/problem.py#L1787)) —
the author's own verdict on how far that got.

It gets *refinement* roughly right and *unrefinement* wrong, does not iterate (so an A–B–C chain is
not closed), does not survive MPI (`get_opposite_bulk_element()` may return an element this rank does
not own, or nothing at all), and does not account for the extra refinement that
`enforce_refinement_balance` performs **after** `adapt()` on simplex-family meshes
([src/mesh3d.cpp:251](src/mesh3d.cpp#L251)).

The practical status is visible in the tutorials. [docs/source/tutorial/multidom/simple_fsi.py:46-48](docs/source/tutorial/multidom/simple_fsi.py#L46-L48):

```python
leqs += SpatialErrorEstimator(velocity=1)
leqs += RefineToLevel()@"liquid_solid"
seqs += RefineToLevel()@"liquid_solid"
```

The interface is forced to maximum refinement on *both* sides so that the two sides cannot disagree.
That is the user-visible cost of the missing feature: you cannot have an adaptive coupled interface,
only a uniformly-maximally-refined one. Removing those two lines is the acceptance test for this work.

---

## 2. The invariant to enforce

> **Interface conformity.** For every declared opposite-interface connection
> (`meshA`, boundary `bA`) ↔ (`meshB`, boundary `bB`) with offset vector `t`: the set of boundary
> facets of `meshA` on `bA`, translated by `t`, equals the set of boundary facets of `meshB` on `bB`,
> facet for facet.

This is precisely the precondition of `connect_interface_elements_by_kdtree`, stated on the bulk
meshes rather than on the interface meshes. Three things follow, and they matter:

- **It is a statement about facets, not about refinement levels.** Two domains may carry different
  `_initial_uniform_refinement_level`, so equal `refinement_level()` is neither necessary nor
  sufficient. Facet identity is the thing the matcher actually needs.
- **It is checkable and repairable without any interface element existing.** Boundary facets come
  from `nboundary_element(bind)` + `Face_index_at_boundary`, which `Mesh::generate_interface_elements`
  itself uses ([src/mesh.cpp:903-916](src/mesh.cpp#L903-L916)) and which `adapt` keeps current via
  `setup_boundary_element_info`. **This is the answer to "interface elements are removed during
  adaptation": the new machinery never looks at an interface mesh at all.**
- **It implies matching hanging-node patterns along the interface.** A node on an interface facet
  hangs because a *neighbouring interface facet* is finer; conformity forces that neighbour to be
  equally fine on both sides, so the hang pattern within the interface surface agrees. (Claim to be
  validated by test, §9, not assumed.)

---

## 3. Architecture

One new C++ component, `src/refinement_coupling.{hpp,cpp}`:

```cpp
struct CoupledInterface {
  Mesh *meshA, *meshB;
  unsigned bindA, bindB;         // boundary indices, resolved lazily (a side may be absent)
  std::vector<double> offset;    // A + offset == B, for periodic/translated pairs
  bool enforce;                  // "auto" => on iff a connection was declared
};

class InterfaceRefinementCoupler {          // owned by pyoomph::Problem
  std::vector<CoupledInterface> pairs;
  // -- the primitive everything else is built on --
  FacetTable collect(Mesh*, unsigned bind, const std::vector<double>& off) const;
  void       exchange(FacetTable&) const;   // MPI Allgatherv, no-op in serial
  // -- the three users --
  unsigned harmonise_error_overrides();     // §6, pre-adapt
  unsigned harmonise_adapt_selection();     // §7, between selection and execution
  unsigned repair_after_adapt();            // §8, the guarantee
  unsigned check(bool throw_on_mismatch);   // §10, diagnostic
};
```

Registration mirrors `MeshTemplateOppositeInterfaceConnection._connect_elements`
([pyoomph/meshes/mesh.py:441-460](pyoomph/meshes/mesh.py#L441-L460)) but keys on **(bulk mesh,
boundary index)**, which are permanent, instead of on interface mesh objects, which are destroyed and
rebuilt on every adapt. It is refreshed in `actions_after_adapt` alongside `_connect_opposite_elements`.

### The facet key

`FacetRec` per boundary facet:

| field | meaning |
|---|---|
| `key` | quantised facet centroid, `llround(x*1e8)` per component, offset applied |
| `elem` | local element index (may be a halo copy) |
| `face` | face index at boundary |
| `level` | `refinement_level()` of the owning bulk element |
| `err` | current merged error / error override |
| `flags` | `to_be_refined`, `father_sons_to_be_unrefined`, `at_max_level`, `is_halo` |

Quantisation scale `1e8` matches both `enforce_refinement_balance`
([src/mesh3d.cpp:262](src/mesh3d.cpp#L262)) and the KD-tree's `epsilon = 1e-8`
([src/kdtree.hpp:57-58](src/kdtree.hpp#L57-L58)), so a pair the coupler considers matched is exactly a
pair the matcher will later find. Eulerian coordinates, deliberately: the existing matcher already
runs on Eulerian coordinates after every adapt and copes with moving meshes, so there is no evidence
a Lagrangian key is needed. If ALE tolerance ever bites, §11 has the fallback.

A shape-neutral "vertices of face *f* of element *e*" accessor is needed. The per-element face
boundary-tag machinery (`setup_boundary_element_info_from_face_tags`) already enumerates faces
generically for every family; the accessor should be factored out of it rather than written afresh.

---

## 4. Where this plugs into the adapt cycle

Today (Python-driven):

```
custom_adapt.__enter__  ->  actions_before_adapt: every interface mesh clear_before_adapt()
   ...for each bulk mesh: adapt_by_elemental_errors(errors)
        TemplatedMeshBase::adapt:  update_elemental_errors (py hook)
                                   synchronise_elemental_errors      (MPI)
                                   TreeBasedRefineableMeshBase::adapt = select + adapt_mesh
                                   enforce_refinement_balance        (simplex families)
                                   post_adapt_setup_hanging_nodes
custom_adapt.__exit__   ->  actions_after_adapt: rebuild_after_adapt, _connect_opposite_elements <-- throws here
```

Proposed (C++-driven, one `Problem::adapt_all_meshes()`):

```
 A  compute error overrides                        (§5, moved to C++)
 B  harmonise overrides across coupled interfaces  (§6, new, iterated)
 C  per mesh: SELECT elements for refine/unrefine  (§7, split out of oomph's adapt)
 D  harmonise the SELECTION across interfaces      (§7, new, iterated)   <-- the exact step
 E  per mesh: EXECUTE  (adapt_mesh)
 F  fixed point { per mesh: one 2:1 balance round ; per pair: one conformity repair round }  (§8)
 G  per mesh: post_adapt_setup_hanging_nodes
 H  check conformity                               (§10)
```

The important structural change is that **G must move out of `TemplatedMeshBase::adapt` and run after
F**, and that F is a *joint* fixed point over all meshes rather than a per-mesh loop. `adapt()` keeps
working as a standalone entry point for the single-mesh case by calling the same pieces.

---

## 5. Moving the error overrides into C++

Prerequisite for §6, and requested independently. The state already lives in C++:
`BulkElementBase::elemental_error_max_override` ([src/elements.hpp:925](src/elements.hpp#L925)),
`max_permitted_error()` / `min_permitted_error()` / `max_refinement_level()` on the oomph base. Only
the *orchestration* and the *criteria* are in Python.

**Criteria → C++ directives.** Register a small list of directives per domain, evaluated in C++ over
every element (halo copies included, so they are consistent by construction and need no
synchronisation at all):

| Python API (unchanged) | becomes |
|---|---|
| `RefineToLevel(level)` ([generic.py:257](pyoomph/equations/generic.py#L257)) | `RefineToLevelDirective` |
| `RefineMaxElementSize(s)` ([generic.py:303](pyoomph/equations/generic.py#L303)) | `RefineMaxElementSizeDirective` |
| `RefineAccordingToElement(f)` ([generic.py:326](pyoomph/equations/generic.py#L326)) | stays Python — it *is* a callback |
| user `calculate_error_overrides` | stays Python, called via a virtual hook |

The Python classes keep their public signatures; `calculate_error_overrides` becomes a thin
registration call in `after_compilation`.

**Orchestration → `Problem::compute_elemental_error_overrides()`**, a C++ port of
[problem.py:1668-1751](pyoomph/generic/problem.py#L1668-L1751):

1. reset overrides on all meshes, all tree depths;
2. seed from the Z2 estimate per mesh;
3. eigen-error contribution — **stays in Python for now** (it swaps eigenvectors into the dof
   vector and back; porting it is a separate job with its own risks);
4. deepest-tree-depth-first, apply directives + the Python hook;
5. deepest-first, `override_bulk_errors_where_necessary()` — C++ port of
   [mesh.py:1432-1494](pyoomph/meshes/mesh.py#L1432-L1494), including
   `enlarge_elemental_error_max_override_to_only_nodal_connected_elems`, which is already C++
   ([src/mesh.cpp:536](src/mesh.cpp#L536));
6. cross-domain harmonisation (§6);
7. synchronise across MPI copies.

### A required change in step 7

`TemplatedMeshBase::synchronise_elemental_errors` ([src/mesh.hpp:625](src/mesh.hpp#L625)) currently
does **owner wins**: each haloed element's value is copied onto its halo counterparts. That was right
for the defect it was written for, and it is wrong here.

Step 5 computes an override *from* an interface element and writes it *to* the bulk element on the
**opposite domain**. The two domains are partitioned independently, so the rank holding the interface
element frequently does not own the opposite bulk element — it holds a halo copy, retained by the
`set_must_be_kept_as_halo` marking in `actions_before_distribute`
([problem.py:3194-3199](pyoomph/generic/problem.py#L3194-L3199)). Owner-wins would then throw the
override away, on every rank, silently.

Change it to a **max-reduce over all copies**: halo → haloed, max at the owner, haloed → halo. This is
semantically exact (the field is called `..._max_override`; the operation is a max), it is a strict
superset of the current behaviour for the the rank-local error-override defect it was written for (adaptive_refinement.md §8.2) (the rank with the larger
override still wins), and it is idempotent.

---

## 6. Pre-adapt harmonisation of the overrides (step B)

Cheap, approximate, and *not* where correctness comes from. Its job is to stop the two sides from
disagreeing in the first place, so that §8 has nothing to repair.

Per coupled pair, per shared facet key:

- `refineA := errA > maxerrA && levelA < maxlevelA`. If `refineA != refineB`, raise the other side's
  error to `100 * maxerr`; if that side cannot refine (at max level), instead *lower* the first side
  to `0.5*(min+max)` — "hold, do not refine" — and count it as an overruled refinement.
- `keepA := errA >= minerrA` (not eligible for unrefinement). If `keepA && !keepB`, raise `errB` to
  `0.5*(min+max)_B`.

Iterate to a global fixed point (`MPI_Allreduce` on a change counter). Only ever *raises* errors, so
it terminates.

### Why this cannot be the whole answer

Take a 3D brick refined into 8 sons, 4 of which touch the interface. oomph unrefines a father only if
**all** of its sons want it ([refineable_mesh.cc:432-461](src/thirdparty/oomph-lib/include/refineable_mesh.cc#L432-L461)).
Suppose on side A one of the four *interior* sons vetoes, and on side B none does. The four
interface-touching sons agree on both sides, so no facet-level harmonisation sees anything wrong —
and yet A keeps its 8 sons while B merges its 8 into one. Conformity broken, by a veto that never
appears on the interface.

Error-space harmonisation at facet granularity is therefore structurally incapable of guaranteeing
the invariant. Two consequences: §8 is mandatory, and §7 is worth the effort.

---

## 7. Harmonising the *decision* instead of the error (steps C/D)

oomph's `TreeBasedRefineableMeshBase::adapt(errors)`
([refineable_mesh.cc:307](src/thirdparty/oomph-lib/include/refineable_mesh.cc#L307)) is two phases in
one function: ~150 lines that translate errors into per-element flags, then a call to the public
`adapt_mesh(doc_info)` ([refineable_mesh.h:513](src/thirdparty/oomph-lib/include/refineable_mesh.h#L513))
that executes them. The selection phase touches only public API — `select_for_refinement`,
`deselect_for_refinement`, `select_sons_for_unrefinement`, `deselect_sons_for_unrefinement`,
`refinement_is_enabled`, `refinement_level`, `tree_pt`.

**Split it.** Patch the vendored oomph to factor the selection out into a virtual
`select_elements_for_refinement_and_unrefinement(errors)`, leaving `adapt()` as the two-line
composition. Minimal, behaviour-preserving, upstreamable — and much better than copying 150 lines of
vendored logic into pyoomph, where it would silently drift the next time oomph is refreshed.

Then insert step D between them and harmonise the *flags*, which is exact:

- **Unrefinement, first, to a fixed point.** A's element unrefines iff its father has
  `sons_to_be_unrefined()`. If A's father is selected and B's partner's father is not, **deselect A's
  father**. Only ever deselects — we cannot manufacture an unrefinement, since that needs unanimity
  among sons we do not control. Monotone downward, terminates.
- **Refinement, second, to a fixed point.** If A's element is selected and B's partner is not, select
  B's — unless B cannot refine, in which case deselect A's and record the overrule. Monotone upward,
  terminates.

The order matters and there is no oscillation between the two: selecting a refinement never creates a
new unrefinement selection, and deselecting an unrefinement never creates a new refinement selection.

The fixed point must be global — chains A–B–C need more than one round — with `MPI_Allreduce` on the
change count so every rank runs the same number of rounds and enters the collectives inside
`adapt_mesh` together. Exactly the failure mode fixed in `enforce_refinement_balance`
([src/mesh3d.cpp:325-340](src/mesh3d.cpp#L325-L340)); the same discipline applies here.

Also replicate oomph's `total_n_refine`/`total_n_unrefine` Allreduce gate before calling `adapt_mesh`
([refineable_mesh.cc:497-545](src/thirdparty/oomph-lib/include/refineable_mesh.cc#L497-L545)) — after
harmonisation, not before, since harmonisation changes the counts.

Note the vendored `son_type() == OcTreeNames::LDB` "first son in charge" assumption
([refineable_mesh.cc:446](src/thirdparty/oomph-lib/include/refineable_mesh.cc#L446)) — check it holds
for pyoomph's `DynamicTree` sons on wedges and pyramids before relying on the unrefinement flags.

---

## 8. The post-adapt repair loop (step F) — where the guarantee comes from

Even with §7 exact, execution is not the end: `enforce_refinement_balance` runs *after* `adapt_mesh`
and refines further elements to restore 2:1 balance on simplex-family meshes. Some of those are at
the interface, and their partners were not refined. So balancing and conformity have to be solved in
**one** fixed point, not in sequence:

```
repeat
    changed = 0
    for each mesh:  changed += one round of enforce_refinement_balance
    for each pair:  changed += one round of conform_by_refining_the_coarse_side
    MPI_Allreduce(changed, MAX)
until changed == 0
```

`conform_by_refining_the_coarse_side`: collect and exchange both sides' facet tables; a facet present
on A but subdivided on B (its key absent from B while keys strictly inside it are present) selects
A's owning element for refinement. Selection is unioned globally by centroid key and re-resolved
locally, so halo copies are refined too — the pattern already proven in
[src/mesh3d.cpp:325-364](src/mesh3d.cpp#L325-L364).

Refinement only, never unrefinement, so it terminates: levels are bounded by `max_refinement_level()`.

This loop subsumes today's per-mesh `enforce_refinement_balance` call inside `adapt`.
`post_adapt_setup_hanging_nodes` moves after it.

**Repairing is lossy, which is why §6 and §7 exist.** If A unrefines a patch that B keeps, the repair
re-refines A — and the son values are re-interpolated from the merged father. The fine-scale
information is gone. A correct-but-churning implementation would also make adapt non-idempotent and
could oscillate across successive adapts. §7 prevents the round trip; §8 catches whatever §7 could not
foresee.

---

## 9. Level caps and unsatisfiable couplings

If `meshA.max_refinement_level < meshB.max_refinement_level`, B can reach a fineness A cannot match
and the repair loop cannot converge. Detect at registration and resolve explicitly:

- clamp the effective max level **on both sides** to the minimum of the two, and say so once; or
- honour a per-connection `max_refinement_level` override.

Silently spinning in the repair loop is the one outcome to rule out — bound the loop and throw with
the offending facet positions if it is hit. Likewise, detect at registration when the two sides'
level-0 facets are not in bijection at all (a triangular face against a quadrilateral one) and throw
there, where the message can be useful, rather than in the KD-tree matcher.

---

## 10. Diagnostics

Extend the pattern that caught the rank-local error-override defect (adaptive_refinement.md §8.2). `InterfaceRefinementCoupler::check()` compares the two facet
tables and reports non-matching facets by position, count, and level, on both sides.

- `Problem.check_interface_conformity(throw_on_mismatch=False)` — collective, callable from Python.
- Fold into the existing `PYOOMPH_CHECK_HALO_CONSISTENCY` env var rather than adding a second one:
  the two checks answer the same question ("do the pieces that must agree still agree?") and users
  should not have to know which to set. Same `0 / 1=warn / 2=throw` levels, same
  `MPI_Allreduce`-the-verdict discipline so throwing mode fails the whole job instead of one rank
  while the others block ([src/mesh.cpp](src/mesh.cpp), `check_halo_element_consistency`).
- Run the check unconditionally under `--full` / `PYOOMPH_FULL_TESTS`.

---

## 11. Known risks

| risk | mitigation |
|---|---|
| Eulerian keys drift under ALE, since `ConnectMeshAtInterface` only equates positions to solver tolerance | The existing matcher already runs on Eulerian coordinates post-adapt and copes. Fallback: key on Lagrangian coordinates (`InterfaceMesh` already maintains a Lagrangian KD-tree — note `invalidate_lagrangian_kdtree`), or on a topological key (level-0 facet id + son path), which is exact and motion-invariant but needs facet-descendant tracking. |
| Lagrange multipliers on *hanging* interface nodes — `pin_redundant_lagrange_multipliers` ([generic.py:144-154](pyoomph/equations/generic.py#L144-L154)) has never been exercised with a hanging interface | Dedicated test (§12 case f). This may turn out to be the hard part, independent of everything above. |
| Global `Allgatherv` of all interface facets is O(N_interface) per rank | Fine at current scale. Optimisation: build a persistent facet-key → owning-rank routing table at setup, refresh on adapt, exchange point-to-point. |
| Duplicating vendored oomph selection logic drifts on the next oomph refresh | Patch oomph to split `adapt()`; do not copy (§7). |
| `_internal_facets_` / DG | Already rejected: `"TODO: Adaption with internal facets"` ([problem.py:1761](pyoomph/generic/problem.py#L1761)) and `"Cannot adapt yet when having discontinuous fields added at an interface"` ([src/mesh.cpp:4843](src/mesh.cpp#L4843)). Out of scope; keep the throws. |
| Remeshing invalidates the correspondence | Geometric keys need no rebuild — one more reason to prefer them over topological keys for v1. |
| The eigen-error path (`_adapt_eigenindex`) stays in Python and writes overrides after the C++ pass | Order it before step B; assert in C++ that nothing writes overrides after harmonisation. |

---

## 12. Tests

`tests/two_domain_cases.py` is the single definition, shared with the MPI harness exactly as
`box_cases.py` / `box_cases_3d.py` are — that sharing is what kept the serial and distributed
campaigns from drifting apart. Two boxes sharing a face, every element family, 2D and 3D, serial and
`mpirun --distribute`, covering: the error estimator active in one domain only; `RefineToLevel` and
`RefineMaxElementSize` on one side only; a criterion stated **on the interface** rather than on the
bulk (the case where a halo-holding rank has no interface element to read); refine-then-smooth so one
side wants to unrefine and the other does not; `ConnectMeshAtInterface` + moving mesh with a hanging
node on the interface; a chain A–B–C, to prove the fixed point closes; and differing
`max_refinement_level` and `_initial_uniform_refinement_level` per domain.

Assertions run after **every** adapt, not only at the end: facet sets equal per pair, the KD-tree
matcher does not throw, and under MPI the gathered residual, global `ndof` and MPI-reduced observables
match the serial reference.

**Every fix was negative-tested**: disable the coupler by env var, rebuild, confirm each test actually
fails. A test that passes with the fix disabled is measuring nothing — §13.9's first `callback` test
was exactly that.

Regression: `docs/source/tutorial/multidom/simple_fsi.py` with its `RefineToLevel()@"liquid_solid"`
workaround **removed** must run and agree with the reference. The feature is only done when the
workarounds are gone.

---

## 13. What building it actually turned up

Ten things the design above did not anticipate. Two are ordering constraints; the rest are defects the
design's own reasoning would have produced.

### 13.1 `distribute()` refuses a non-uniformly refined mesh (ordering)

oomph's `Problem::distribute()` throws *"at least one of your meshes is no longer uniformly refined"* —
it has to preserve the tree forest. The repair refines part of a domain, which is by definition
non-uniform, so repairing an initial mismatch **before** distribution breaks distribution instead.

Levelling everyone *up* to the maximum would keep the meshes uniform, and was the first thing tried. It
is wrong: a domain asked for level 1 silently gets level 2, and `RefineToLevel(1)` then marks those
level-2 elements "may not unrefine", so the excess never goes away again.

What works: uniform-refine coupled domains only as far as they agree, distribute, then apply the
remainder — where partial refinement is allowed and the repair can do its job.
`Problem._defer_uneven_initial_refinement` / `_apply_deferred_initial_refinement`, held in place by
`test_uneven_initial_levels_survive_distribution`.

### 13.2 Side A and side B are not the same two domains on every rank (defect)

The one that cost the most to find, and the most general lesson here.

`_collect_coupled_interfaces` reads the connections out of the mesh template, where they are put by the
C++ auto-detection. That detection walks pointer-keyed containers, and **heap addresses differ between
processes**, so two ranks can discover the same connection with the two sides in opposite order:

```
rank 0: [('upper', 'interface', 'lower', 'interface')]
rank 1: [('lower', 'interface', 'upper', 'interface')]
```

The facet sets are unioned across ranks *per side*, so "side A" became a mixture of facets from both
domains. Globally consistent-looking, entirely fictitious. The symptom was that a case passed alone and
failed when a *triangle* case had run before it in the same process — an allocation-pattern dependence
with no plausible mechanism until the two orders were printed side by side.

Fixed by orienting each pair canonically by `(domain name, boundary name)` and sorting the connection
list the same way. The general rule, which applies to anything else added here: **a collective decision
may not depend on discovery order.** Not on pointer order, not on hash order, not on partition order.

### 13.3 Conformity is necessary but not sufficient under MPI (design gap)

§2 claims facet conformity is exactly the matcher's precondition. That is true serially and false under
distribution: `connect_interface_elements_by_kdtree` is **rank-local**, and the two domains share no
nodes, so the partitioner sees two disconnected components and cuts them independently. A rank therefore
routinely holds one side of a facet pair and not the other. Globally conforming, locally unpairable.

That is a halo-coverage property, supplied by the `set_must_be_kept_as_halo` marking in
`actions_before_distribute` — a different mechanism with a different failure mode, and the matcher's
error message cannot tell the two apart. `check_interface_conformity` therefore counts and reports them
**separately**: "no counterpart on the opposite side" vs "a counterpart, but not on the process that
holds them". Without that split, §13.2 would have been diagnosed as a refinement bug and fixed in the
wrong place; with it, the first diagnostic run said "the halo layer does not cover the opposite domain"
and pointed straight at the orientation.

### 13.4 The geometry-free coarse-side test held up

§8's test — a facet absent from the other side's facet set but with all its corners in the other side's
vertex set was subdivided there — needed no adjustment for any of the four families, for quads meeting
triangles across the interface, or for moving meshes. Nesting does all the work: a facet's corners
survive as corners of its children. No point-in-facet predicate was ever needed.

### 13.5 Moving the error overrides into C++ (§5): two parts of three

Two of the three parts of §5 landed; the third did not, and the reason is worth recording.

**The declarative criteria moved.** `RefineToLevel` and `RefineMaxElementSize` are now registered once,
at compile time, as C++ refinement directives (`pyoomph::Mesh::apply_refinement_directives`) instead of
looping over `mesh.elements()` from Python on every adapt. The values are identical. What changes is
*coverage*: a C++ pass runs over every element the process holds, **halo copies included**, so a halo
copy reaches the same verdict as the element it copies, by construction, with nothing left to
synchronise. The Python versions produced the same numbers but produced them as one more rank-local
override for the halo exchange to repair afterwards. `RefineAccordingToElement` stays in Python — it is
a user callback, and that is not something to move.

**The synchronisation changed from owner-wins to max-reduce** (§5, "A required change in step 7"), and
the argument there held up exactly as written. Now two passes: halo → owner (owner takes the max) →
halo. The quantity is `elemental_error_max_override`; a max is both the semantically right reduction and
idempotent.

**`_override_bulk_errors_where_necessary` did not move.** It is the propagation, not a criterion, and it
is entangled with Python-side state (`_opposite_interface_mesh`, `_parent`, `_interface_name`) that would
have to be mirrored in C++ first. Its rank-locality problem — an override computed on the rank holding
the interface element but written to a bulk element owned elsewhere — is already covered by the
max-reduce above, so the port is now a refactor rather than a fix, and was not worth the regression risk
in the same change.

### 13.6 The two-phase adapt (§7) was worth it

The split of oomph's `adapt()` went in as §7 described: a `select_elements_for_refinement_and_unrefinement`
/ `execute_selected_adaptation` pair in the vendored `refineable_mesh.{h,cc}`, with `adapt()` reduced to
the two-line composition, and the same three-way split mirrored on `TemplatedMeshBase`
(`adapt_select` / `adapt_execute` / `adapt_finalise`). The flag reconciliation runs in the gap, as
`harmonise_adapt_selection`: unrefinement deselection to a fixed point first, then refinement selection,
which cannot chase each other for the monotonicity reason in §7.

Two things worth recording.

**The counts had to be recomputed.** oomph counts `n_refine`/`n_unrefine` as it sets the flags, and those
counts drive the collective "is this worth doing at all" gate inside the execute half. Reconciling the
flags in between invalidates them — and a stale zero there does not merely mis-report, it skips the
adaptation entirely. `TemplatedMeshBase::recount_pending_adaptation()` re-derives both from the flags as
they stand.

**The payoff is measurable, and only one criterion measures it.** With
`PYOOMPH_DISABLE_ADAPT_RECONCILIATION=1` the post-adapt repair has to refine 5-7 elements back on the
`estimator` cases; with reconciliation, zero. The criteria that reach their target level at *initialise*
(`level`, `size`, `callback`) never disagree during adapt at all, so they need no repair either way and
cannot detect this -- which is exactly why `test_adapt_selection_is_reconciled_before_acting` uses the
estimator case and says so. Both routes reach a conforming mesh, so nothing else in the suite can tell
them apart; what the repair route loses is the fine-scale solution on the patch it merges and then
re-refines.

The old Python heuristic (`# Ensure same refinement at connected interfaces`) is gone with it, along with
the `# TODO: Ensure same refinement at connected interfaces` thirty lines below it that had been the
honest summary of the situation since the branch began.

**Added later, then reverted: skipping an empty adaptation.** Knowing the decision before acting on it
answers a second question for free — *is this adaptation going to change anything at all?* — and the
answer is very often no. oomph only leaves its own adaption loop once an `adapt()` has reported 0/0, so
with `spatial_adapt>0` the last adaptation of every solve is a no-op by construction, and a mesh sitting
at `max_refinement_level` with errors still above the refinement tolerance never reports anything else.
Acting on that anyway costs a full `actions_before_adapt`/`actions_after_adapt` cycle — every interface
mesh torn down and rebuilt, the global mesh reassembled — plus an `assign_eqn_numbers()`, which calls
`invalidate_jacobian_structure()` unconditionally and so discards the frozen sparsity pattern for a
numbering that has not changed. `Problem._adapt_with_interfacial_errors` therefore summed
`Mesh._adapt_pending_counts()` over the meshes and the ranks and skipped the whole block when it came
out zero, with `Mesh._adapt_abandon()` clearing the stale `nrefined()`/`nunrefined()` statistics.

The first version of that skipped one thing too many. `_adapt_execute()` and `_adapt_finalise()`
**reorder the nodes even when they refine nothing** — into the order the elements walk them, rather than
the order the mesh generator created them in — and precisely because the no-op adaptation is universal,
that reordering was what made every run agree on the node order. oomph-lib does it deliberately, in the
branch of `execute_selected_adaptation()` that decides the adaptation is not worth carrying out, and
says why: *"to establish a standard ordering regardless of the sequence of mesh refinements — this is
required to allow dump/restart on refined meshes"*. Skipping it left the order depending on the route:
`load_state()`, a real refinement and a distribution rebuild reorder, a plain run does not. Anything
comparing two runs then compared permuted states — restarts stopped being bit-identical (every value
still exact, only its position moved), distributed runs stopped matching their serial reference, and
`linear_response_drum.py` got the vertices first and the midside nodes afterwards out of
`get_cached_mesh_data`, which broke a spline through them. That was 10 test failures and both tutorial
passes of nightly 20260816.

The reordering is therefore done on its own, by `Mesh::reorder_nodes_if_needed()`
(`Mesh._reorder_nodes_if_needed` in Python), which puts the node vector in the canonical order and
reports whether that moved anything. It is idempotent, so it moves something only the first time it is
reached — in practice the initial adaptation — and the caller renumbers only in that case. Every later
no-op adaptation finds the order already canonical and keeps its interface meshes and its sparsity
pattern. Note that restoring only the teardown and renumbering around an empty refinement does *not*
work: the order comes from the two stages themselves, which is why the reordering had to be separated
out rather than the wrapper put back.

One case still takes the long way round: a **coupled interface on a distributed problem**. There the
node order is not all an executed adaptation leaves behind — with the interface geometry itself an
unknown (`ConnectMeshAtInterface`), a rank that skips the teardown ends up with four dofs the serial run
does not have (`ale-tri_left` in `tests/test_mpi_interface_coupling.py`, 2942 against 2938,
reproducibly), and reordering does not repair it. What the post-adapt repair does for that case has to
be understood before the skip can be extended to it. Plain distributed problems are unaffected and do
skip. `tests/test_state_file_restart.py` is what catches a regression in the node order.

The restructuring around the gap stands: select (and, when there is something to reconcile,
`harmonise_adapt_selection`) still runs *before* `custom_adapt.__enter__`, and the uncoupled case still
goes through the same three stages rather than through `adapt_by_elemental_errors()` — that call is
exactly the three back to back, so there is nothing to lose, and one path means the reconciliation is not
tied to being coupled. Covered by `test_a_noop_adaptation_leaves_the_mesh_and_the_node_order_alone` and
its neighbours in `tests/test_adaptivity.py`.

### 13.7 Differing `max_refinement_level` (§9), as actually resolved

§9 offered two options and recommended clamping both sides to the minimum. That is the wrong one: it
takes refinement away from a domain in its *interior*, where nothing requires it. What is implemented
instead is narrower and better.

The reconciliation carries a `can_refine` bit per facet (`IFACET_CAN_REFINE`), so a facet whose partner
sits at its own `max_refinement_level` is never selected for refinement. Both sides reach the
mirror-image conclusion from the same globally-reduced flags, so they agree without an extra exchange.
The result is that **the shallower cap governs the interface and only the interface**. Measured, with
`lower` capped at 3 and `upper` at 1, driven by an adapt-time criterion:

```
lower interior levels = [0, 1, 3]      lower AT the interface = [0, 1]
                                       upper AT the interface = [0, 1]      nonconforming = 0
```

**But `RefineToLevel` does not go through `adapt`.** It sets `_initial_uniform_refinement_level`, and
`refine_uniformly` ignores `max_refinement_level` entirely -- so one domain can be driven past a cap the
other cannot follow, and no reconciliation or repair can rescue it. The repair loop stops when it has
nothing left to *select*, which is not the same as having succeeded, and that difference used to fall
through to `connect_interface_elements_by_kdtree` as `Cannot locate opposite node at x=(...)`.

§9's "silently spinning in the repair loop is the one outcome to rule out" was therefore only half done:
the loop was bounded, but its giving-up was not distinguished from its succeeding.
`enforce_interface_conformity` now ends with a facet-mismatch count (deliberately excluding the MPI
"partner not on this process" case, which is a halo problem refinement cannot fix) and throws with the
offending facets and the two plausible causes. Pinned by
`test_unsatisfiable_cap_is_diagnosed_not_left_to_the_matcher` and
`test_lower_max_refinement_level_governs_the_interface`.

### 13.8 Four domains at a cross point, and why the junction is well-posed

`RectangularQuadMesh` split four ways (A|B over C|D) is a topology the two-domain matrix cannot reach,
and it exercises two things:

* the coupling graph is a **cycle** (A-B-D-C-A), not a chain, so the reconciliation has to close a loop.
  D shares no interface with A -- they touch only at the cross point -- yet refinement raised in A
  reaches D, by travelling round. Measured: `four_corner` drives A alone and D ends at level 3, with
  `repairs_during_adapt == 0`, i.e. the reconciliation gets there before the repair does. Under
  `mpirun -n 2` the `ndof` matches the serial run exactly (481 / 1923 / 253).
* the cross point itself, which is where a naive reading says the system must be over-constrained -- and
  it is not, for a reason worth writing down because it is not obvious and is easy to break.

**Why the cross point is not over-constrained.** An interface field is a nodal value on the *bulk* node,
allocated by `Mesh::resolve_interface_dof_id(name)` -- keyed on the NAME, once per bulk mesh. Domain A
owns two interfaces here (`A_B` and `A_C`); `ConnectFieldsAtInterface` names its multiplier
`lagr_mult_prefix + inner + "_" + outer`, which is `_lagr_conn_u_u` on both. They therefore resolve to
the *same* slot: A's cross-point node carries ONE multiplier serving both interfaces, and its residual
there is the sum of the two contributions. Seven multipliers for seven independent conditions.

Verified by forcing the names apart (`lagr_mult_prefix="_lm_<iface>_"` per interface), which gives A's
cross node two slots:

| | ndof | smallest singular value | condition number | rank |
|---|---|---|---|---|
| shared name (default) | 13 | 1.2e-02 | 7.5e+01 | **13/13** |
| distinct names | 14 | 7.8e-18 | 1.1e+17 | **13/14** |

So the shared name is load-bearing, not incidental.

**The trap.** The rank-deficient case still produces the correct answer -- `max|u-y| = 0`, Newton
converging in one step to 5.6e-17. The null space lives entirely in the multipliers, so the primal
solution is still determined and a direct solver simply returns some λ from the affine family. Neither
the residual nor the one-Newton-step Jacobian oracle notices. A user who gives a different
`lagr_mult_prefix` per interface at a multi-domain junction therefore gets a Jacobian that is singular to
machine precision while everything looks healthy, and pays for it later in an eigensolve or an iterative
solver. This is pre-existing `ConnectFieldsAtInterface` behaviour, not introduced here;
`with_removed_overconstraining(*corners)` is the existing escape hatch.

For the record: `pin_redundant_lagrange_multipliers` is NOT what makes the junction work. It pins a
multiplier only when every variable it constrains is already pinned, and at an interior cross point none
of them is.

### 13.9 Conformity is stated on facets, and some elements have none

Found on a real case rather than in the matrix: an evaporating droplet coupled to a gas domain
(`DropletOnSubstrate`, pure triangles), with a Z2 estimator on the velocity in the **droplet only**, so
every bit of refinement the gas ever does is forced on it from the other side.

§2 says the invariant is about facets, and everything built on it — the flag reconciliation (§7), the
repair loop (§8), the checker (§10) — reads `nboundary_element()`/`face_index_at_boundary()`. An
element that touches the interface at a single **vertex** appears in none of those. It is not a
boundary element, it contributes no key to either side's facet set, and no test above can even
mention it. So the gas refined its facet-carrying elements to follow the droplet, and left their
vertex-only neighbours exactly where they were:

```
gas, after 5 adapts:   levels present            0 1 2 3 4      nonconforming = 0
                       vertex-only elements >= 2 levels behind their interface: 34
                       worst: level 0 at the contact line, against level 4 facets
```

A refined band one element thick, dropping four levels at a stroke — and every oracle in this document
reported success, because every one of them is a statement about facets.

**The rule.** 2:1, stated at the vertex: an element that shares an interface vertex with a boundary
facet may not be more than one level coarser than the finest facet at that vertex. It lives in the
same fixed point as the facet repair and each mesh's own `enforce_refinement_balance`
(`select_vertex_connected_too_coarse`, `src/refinement_coupling.cpp`), which it has to, because such an
element can carry a facet of a *different* coupled interface.

Three things make it well-behaved, and they are the reason it can be a repair rather than a
reconciliation:

* **it never creates a facet.** A corner element has no edge/face on the interface, so neither has any
  of its sons — the son at the shared vertex is again corner-only. It therefore cannot cascade along
  the interface and cannot invalidate the facet sets the repair has just agreed on.
* **it is not lossy.** §8 warns that repairing is lossy, and it is — for an element merged away and
  then refined back. These elements have never been refined at all, so their sons interpolate a father
  that still holds the current solution. There is no round trip to lose anything in. That is why the
  count is reported separately (`Problem._interface_vertex_balance_refinements`) instead of being added
  to `_interface_conformity_repairs`, whose whole point is that a non-zero value is bad news.
* **it is bounded**, by the finest facet level and by `max_refinement_level()`.

**Why the error path did not already do this.** It knows about exactly this set of elements:
`Mesh::enlarge_elemental_error_max_override_to_only_nodal_connected_elems` spreads a boundary element's
override to them, "to force refinement rather than leave a 2:1 hang on the boundary". But it can only
fire when the refinement is *driven by an error at that interface*. A refinement forced by the opposite
domain never passes through an error at all — it is a flag reconciliation and a facet repair — so
nothing spreads it.

**Why the §12 matrix did not catch it, which is the part worth remembering.** It was not that the cases
could not reproduce it. They reproduce it perfectly: `tri_left` driven to level 3 in `lower` alone leaves
`upper` with vertex-only triangles at levels 0, 1, 2 and 3 against level-3 facets, a jump of 3. The
matrix simply had no oracle that could see it — every measurement in `solve_case` was a residual, a dof
count, an integral, or `nonconforming`, and a mesh can be badly graded while all four are perfect. The
missing test was not a missing case. `two_domain_cases.max_vertex_level_jump` is now that oracle,
`test_vertex_connected_elements_follow_the_forced_interface` asserts it, and its negative twin
(`PYOOMPH_DISABLE_INTERFACE_VERTEX_BALANCE=1`) measures 3 where the fix measures 1.

The general lesson, and it is the same shape as §13.3: **an invariant stated on one kind of mesh entity
says nothing about the others.** Facet conformity was necessary, and it was never sufficient — not
under MPI (§13.3, a halo property) and not at a vertex (here, a grading property). Both times the
symptom was a check that reported success.

### 13.10 Still open

* **`_override_bulk_errors_where_necessary` is still Python**, per §13.5. It is the propagation rather
  than a criterion, its rank-locality problem is already covered by the max-reduce, and moving it needs
  `_opposite_interface_mesh` / `_parent` / `_interface_name` mirrored in C++ first. A refactor now, not a
  fix.
* **`_internal_facets_` (DG) adaptivity** remains rejected outright, as it was before this work.
* **A failing case poisons later cases in the same worker process.** Seen while bisecting §13.2: after a
  case raised, the next one in the same `mpi_worker.py` process failed too. Pre-existing (the worker has
  always looped over cases in one process) and not investigated. It matters for reading MPI test output:
  the *first* failing case is the real one.

---

## 14. Mixed element spaces across the interface: what the space matrix measures

Everything above was built and validated with the two domains carrying the **same** element space. Nothing
in the machinery requires that — it reads only `nvertex_node()` / `get_vertex_nodes_of_face()`, which are
corner-only and identical for C1 and C2 — and §12's `connect12` has always run a C2/C1 pair. But that pair
is one cell of a matrix nobody had swept, on a flat static interface, where it cannot fail.

This section is the sweep: every ordered pair from `{C1, C1TB, C2, C2TB}` that a family can carry, across
quad / tri_left / tri_crossed / mixed / curved / 3d-brick / 3d-tet, serially and under `--distribute`, plus
the four-domain junction. 108 serial cases + 19 distributed, **one process per case** (§13.10: a failing
case poisons the ones after it in the same process, and under MPI it would leave the other ranks blocked).

`tests/two_domain_cases.py` grew the axes and the oracles for it: an arbitrary `connect:<lo>/<up>` /
`move:<lo>/<up>` selector, a `curved` kind, `box_hex`/`box_tet` 3d kinds, four-domain space selectors
(`four_corner:C1,C2,C2,C1`), and three measurements described below.

### 14.1 The oracle that was missing: `max_vertex_gap`

Every step of the chain identifies "the same place on the interface" by a **quantised Eulerian position** —
`node_key()` at `1e8`, the KD-tree's `1e-8`, the permutation search's `1e-14`. All of it rests on one
quantity: the two sides' interface vertices coincide. Nothing measured that. `nonconforming` answers a
different question ("did the two sides refine the same facets"), and a mesh can pass it while the two
sides have put the shared facet's new vertex in two different places.

`interface_vertex_gap` measures it directly, after **every** adapt rather than at the end — a divergence
introduced by one refinement is washed out by the next solve, and the matcher runs in between.

Two details it needed. It reads the vertex nodes of the **interface mesh's elements**, not "every vertex
node flagged as being on the boundary": in 3d tets those differ, see §14.6. And it keys on the position
rather than on `id(node)` — nanobind hands out a fresh Python wrapper per call and a freed one's id is
reused, so an id-keyed dict silently collapsed 9 interface vertices to 3.

### 14.2 On a flat interface every space pair works. That is why nothing had caught this.

42 of 44 `connect` pairs pass with `max_vertex_gap == 0` exactly, and the two failures are §14.5, which is
not a keying problem. Under `mpirun -n 2 --distribute`, all 19 pairs tried reproduce the serial `ndof`
exactly.

A **curved** interface passes too, as long as the facets carry a `MeshTemplateCurvedEntity`: 19 of 20, and
every interface node on both sides sits on the arc to `1.1e-16`. The macro element places the C1 side's
refinement-created vertices on the curve, exactly where the C2 side's pre-existing midside node already
was. So curvature by itself is not the problem.

### 14.3 What does break: mixed ORDER, once the geometry is not straight-sided

The `move` family is the configuration that produces it — a moving mesh in both domains with the interface
shape prescribed on both sides as a curve, which is the geometry a free surface produces:

* the C2 side has its interface midside nodes **on the curve, off the chord**; a refinement promotes one of
  them to a vertex, still on the curve;
* the C1 side has no such node and creates its new vertex at the **chord midpoint**.

```
refinement_coupling.cpp:866: Cannot make the two sides of a coupled interface match.
  side A facet (0,0.5,0) (0.0625,0.538268,0)  [too fine]
```
`0.538268` is on the curve; the C1 side is at `0.535355`. 22 of 44 cases fail, and the 20 that fail here
(the other 2 are §14.5) are **exactly** the mixed-order pairs:

| lower ↔ upper | result |
|---|---|
| C1↔C1, C1↔C1TB, C1TB↔C1TB, C2↔C2, C2↔C2TB, C2TB↔C2TB (same order) | pass, gap 0 |
| any of {C1,C1TB} against any of {C2,C2TB} | **fail, all 20** |

Both the bubble spaces behave exactly like their non-bubble partner: the bubble is interior and never
reaches a facet. Order is the only thing that matters.

The error message is misleading here — it blames differing `max_refinement_level` or non-bijective facets,
and neither is the case. The two sides refined the *same* facet and disagree about where its midpoint is.

**Two configurations that do NOT reproduce it**, both worth recording so they are not tried again:

* driving the moving mesh from an **outer** boundary. `ConnectMeshAtInterface` tests its constraint with
  the negotiated (lower) space, so the higher side's interface midside nodes are not constrained by it and
  follow their own domain's smoothing; the interface then stays flat to machine precision. The
  pre-existing `ale` case does the same thing, which is why the suite has never had a case whose interface
  actually moves.
* prescribing the shape on **one** side while coupling the positions. That is over-constrained: the
  multiplier is redundant on the pinned side and the free side never follows at all, so it fails for
  *equal* spaces too and measures nothing about spaces.

### 14.4 Junctions: mixed spaces at a cross point are silently WRONG

The four-domain cross point of §13.8, with the four domains given different spaces. Exact solution `u = y`,
so the nodal error is a direct oracle.

| spaces (A,B,C,D) | ndof | max\|u − y\| |
|---|---|---|
| C1,C1,C1,C1 | 138 | 1.3e-15 |
| C2,C2,C2,C2 | 481 | 1.8e-15 |
| C1,C1,C1,**C2** | 179 | **1.3e-03** |
| C2,C1,C1,C2 | 319 | **1.4e-03** |
| C1,C1,C2,C2 | 252 | **2.0e-03** |
| four_away C2,C1,C1,C2 | 217 | **1.2e-02** |

No crash, `nonconforming == 0`, Newton converged in one step. A single domain differing from the other
three is enough. The two-domain mixed pairs are exact (1e-15), so this is the junction, not mixed spaces
as such.

> **Correction.** The mechanism given below is **wrong**, and §15.4 has the measurement that shows it.
> The two interfaces do not need to negotiate different spaces for this to happen -- `C1,C1,C1,C2` fails
> while every multiplier in it is C1. What actually goes wrong is that one multiplier enforces the SUM of
> two coupling conditions, which is only harmless while the redundancy that makes it so still exists.
> The table of errors above stands; the explanation of it does not.

**The mechanism is §13.8's shared multiplier name.** An interface field is a nodal value on the bulk node,
allocated by `Mesh::resolve_interface_dof_id(name)` — keyed on the NAME, once per bulk mesh. Domain A owns
two interfaces and `ConnectFieldsAtInterface` names both multipliers `_lagr_conn_u_u`, so A's cross-point
node carries ONE slot serving both. §13.8 established that this sharing is load-bearing: with all domains
in one space it is what keeps the system non-singular (rank 13/13 against 13/14 with distinct names).

With mixed spaces the two interfaces negotiate **different** multiplier spaces, and one slot cannot be
both. Forcing the names apart confirms it:

| | max\|u − y\| |
|---|---|
| shared name (default) | 1.354e-03 |
| `use_highest_space=True` | 4.273e-04 |
| distinct `lagr_mult_prefix` per interface | **1.110e-16** |

So neither option is right: the shared name gives a wrong solution, distinct names give the right solution
on a Jacobian §13.8 measured as rank-deficient (condition number 1e17, the null space living entirely in
the multipliers). `use_highest_space` is not a workaround — it only shrinks the error. This needs a real
fix: the slot has to be keyed on (name, negotiated space) or the junction has to be resolved explicitly.

### 14.5 `C2` against `C1TB` segfaults, and the meet of the space lattice is wrong

Not adaptivity-specific: plain uniform refinement to level 1 is enough, and it happens on every triangular
kind (flat and curved).

`src/elements_interface.cpp`, `interpolate_newly_constructed_additional_dof(..., space="C1TB")`:

```cpp
psi.resize(father->get_eleminfo()->nnode_of_space[SPACE_INDEX_C1TB]);
father->shape_at_s_C1TB(sfather, psi);
...
unsigned fnode_index = father->get_nodal_space_index_to_element_index_map()[SPACE_INDEX_C1TB][lf];
pyoomph::BoundaryNode *bn = dynamic_cast<...>(father->node_pt(fnode_index));   // <-- SIGSEGV
```

`BulkElementTri2dC2::Nodal_Space_Index_To_Element_Index_Map[SPACE_INDEX_C1TB]` is **`{}`**
(`src/elements_2d.cpp`) — a C2 triangle has no C1TB space. The index runs off an empty vector and
`node_pt()` is handed garbage.

It gets there because `get_interface_field_connection_space("C2", "C1TB")` returns `"C1TB"`
(`pyoomph/equations/generic.py`): its `space_order` is a **total** order in which C1TB sits below C2. But
C1TB is not a subspace of C2 — a C2 triangle cannot host it. The correct meet is **C1**, the dual of the
join rule `find_dominant_element_space` already has (`C2 ∪ C1TB → C2TB`, "only space that can hold both").
Quads are unaffected: their C1TB row is non-empty, so `C2-C1TB-mixed` survives.

In **3d tets** the same negotiation is caught rather than crashing, by the guard in
`pyoomph/generic/codegen.py`: *"You tried to define a C1TB field at an interface attached to 3d bulk domain
with element space C1TB ... 3d tetrahedral elements of C1TB do not provide the face bubble node ... adjust
the facet space to C1"*. Note it fires for **C1TB against C1TB** as well — equal spaces — so in 3d, C1TB is
simply not usable with `ConnectFieldsAtInterface` at all. Four of the sixteen 3d-tet pairs are lost to it.

### 14.6 Two more things the sweep turned up, both latent

**`Quad2dFaceOrientation::analyze` writes out-of-range opposite-node indices.** It ends with

```cpp
node_index = node_index_map(orientation, nnode_1d);
```

using **this** element's `nnode_1d`. A C2 brick face (9 nodes) against a C1 one (4) is therefore left
holding indices up to 8 into a 4-node element. The 1d line elements handle exactly this mismatch
explicitly (`elements_concrete.hpp`, *"opposite is C1: no midside node"* → index `-1`), and so do the
triangular faces (`resize(6, -1)`); the quad face does not. The entries are latent — `box_hex` with
C2/C1 runs, converges, and reports `nonconforming = 0` — until something dereferences one, at which point
`opposite_node_pt()` segfaults inside the nanobind caster. `interface_node_pairing` skips that combination
for exactly this reason and says so.

**In 3d tets, vertex nodes are marked as being on the interface while carrying no facet there** — 3 of them
with a C1 lower domain, 14 with a C2 one. They come from refining an element that touches the interface
only along an EDGE: the new node lands on that edge, on the interface plane, and is marked. Conformity is
stated on facets and is silent about it, and the matcher only looks up facet vertices, so nothing here
depends on it — but it is a boundary-MEMBERSHIP question (see `boundary_node_membership.md`) that should be
settled rather than assumed. Counted as `stray_boundary_vertices`; zero in every 2d kind.

### 14.7 Where that leaves the original question

*Is a C1 bulk domain meeting a C2 one supported, with adaptivity?* On a flat interface, and on a curved one
carrying a curved entity: yes, in 2d and 3d, serially and distributed, for every space pair except the two
defects above. As soon as the interface geometry is a genuine unknown — the free-surface / FSI case — no:
the two sides put the same facet's new vertex in different places, and every identity in the coupling
machinery is a quantised Eulerian position.

The fix for that is a correspondence that does not depend on the current geometry. Lagrangian coordinates
are **not** available as a basis: remeshing a deformable solid interpolates `xi`, so an FSI pair's two
sides need not agree on it. What remains is either a topological key — (level-0 facet identity from the
`MeshTemplate`, plus the dyadic local coordinate within it, composed at every split the way
`inherit_macro_element_from_father` already composes `Macro_element_vertex_s`) — or making the two sides
share one geometry at the interface by constraining the higher side's interface positions to the lower
space. That decision, and the junction fix in §14.4, are the next piece of work.

---

## 15. The fix: one interface space, and a topological identity for its nodes

§14 found four defects. Three of them are one sentence violated three times —

> A coupled interface is one object with one space: the multiplier space is the **meet** of the two bulk
> spaces, capped by what each bulk element family can carry on a facet.

— and the fourth, §14.3, is not about spaces at all but about **identity**: the machinery asked *where*
a node is when it should have asked *which* node it is.

### 15.1 The meet of the space lattice (§14.5, §14.4's 3d twin)

`{C1, C1TB, C2, C2TB}` is a lattice, not a chain: `C1 < C1TB < C2TB` and `C1 < C2 < C2TB`, with **C1TB
and C2 incomparable** — a C2 element has no bubble node, a C1TB element has no midside ones.
`find_dominant_element_space` has always known the join (`C2 ∪ C1TB → C2TB`, *"only space that can hold
both"*); `get_interface_field_connection_space` walked a total order instead and returned "the one further
down the list", which is not the meet. `find_subordinate_element_space` in
`pyoomph/expressions/generic.py` is the dual, and the negotiation now uses it.

**Then the meet is capped by what the inner domain can carry on a facet**, because the multiplier is a
nodal value on that domain's bulk nodes. `largest_facet_space(parent_space, parent_dim)` states the rule:
a bubble space needs a bubble *node*, so C1TB/D1TB requires a `C1TB` or `C2TB` parent, and in 3d only
`C2TB` (a tet facet needs a *face* bubble, which a C1TB tet does not have — it carries only a cell one).

Two consequences beyond the segfault. `C2` against `C1TB` on a triangle now negotiates **C1** and runs
(previously: an empty `Nodal_Space_Index_To_Element_Index_Map` row indexed off its end). And **C1TB
against C1TB on 3d tets** now runs too — it used to throw for *equal* spaces, because the negotiation
asked the facet for a space the tet cannot put there.

The guard in `_internal_define_scalar_field` that used to be the only check is kept as a backstop and
corrected: it permitted C1TB on a 2d C2 parent, which is exactly the case that segfaulted. And a real
backstop now exists in C++, at the one place the binding is made — `InterfaceElementBase::add_interface_dofs`
refuses any space whose map row is empty on the bulk element, which covers the ~20 sites that index one.
**A wrong space must never again be a segfault.**

### 15.2 The identity: what a node IS, not where it is (§14.3)

Every step of the chain — `node_key()` in `refinement_coupling.cpp`, the KD-tree in
`connect_interface_elements_by_kdtree`, the permutation search in `analyze_opposite_orientation` — used a
quantised Eulerian position. That rests on the two sides' interface vertices coinciding, and at a
mixed-order interface they do not: refinement promotes the C2 side's off-chord midside node to a vertex
while the C1 side creates one on the chord.

`pyoomph::Node::interface_topological_id` is a 128-bit digest of what the two domains actually share,
which is the `MeshTemplate`. It is the cross-domain extension of `refinement_generating_key`, which had
already made this move *within* a mesh and says why: matching by position is only as good as the positions
are. Two ingredients:

* **level 0** — a generated node is stamped with the identity of its template node. A corner is its own
  index; an **intermediate** node (edge mid-point, face or cell centre) is the C1 combination of the
  entity's corners, which `MeshTemplate::intermediate_node_map` already records, keyed by the corner set
  precisely because *"that set — not the node's position — is its identity"*.
* **refinement** — a new node is the C1 interpolation of its father's vertex nodes at a dyadic local
  coordinate, from `get_nodal_s_in_father` (implemented for every refineable family, wedges and pyramids
  included) and `shape_at_s_C1`.

Filled in by `Mesh::assign_interface_topological_ids()`, one sweep in order of increasing refinement
level. No MPI: pre-distribution ids travel with the nodes (the same argument that carries
`global_root_index`), and later ones are born of a refinement this rank performed, halo elements included.

**The identity must be flattened to the template basis, and that is the part the design got wrong.**
Digesting the *immediate* parents is not canonical. A point a quarter of the way along a level-0 edge is
reached by a C2 domain as "the midside node of a level-1 element" — weights 3/4, 1/4 over two level-0
corners — and by a C1 domain as "the midpoint of a level-1 edge" — weights 1/2, 1/2 over a level-0 corner
and the level-0 midpoint. Both describe the same point, and the two digests differed. The symptom was
sharp and misleading: uniform refinement to any level conformed perfectly, and only *asymmetric*
refinement past level 1 failed, because that is the first time a node is reached by two different routes.
`Node::interface_topological_expansion` therefore carries the full (template index, weight) expansion,
merged and sorted, and the digest is taken of that. The weights are dyadic, hence exact in double.

**Bubble nodes are the exception, and they must not throw.** A centroid bubble's C1 weights are thirds
or sixths — not dyadic, not even exact in double, and nothing a refinement ever produces. Those get an
*opaque* identity: deterministic, but outside the comparable set. That is safe because only C1 **corners**
ever enter an expansion or a facet key, and a bubble is never one. The first version threw on a non-dyadic
weight and took every C2TB/C1TB pair with it.

### 15.3 Where it is used, and where it deliberately is not

| site | change |
|---|---|
| `refinement_coupling.cpp` `node_key()` | reads the digest; the facet sets, `facet_is_too_coarse`, the `Allgatherv`, the flag reconciliation and the vertex-balance closure are untouched |
| `InterfaceMesh::connect_interface_elements_topologically` | the same vertex-set pairing on an exact `std::map` instead of a KD-tree with an epsilon |
| `InterfaceElementBase::vertex_match_distance2` | the permutation search returns 0/1 on identity rather than a squared distance — with two genuinely different surfaces the nearest miss is not a match |

The decision is made **per connection, once**, by `pair_is_topological()`: both sides must carry a
complete set of ids, and the connection must be **coincident**. A periodic or translated pair relates two
*different* template facets, which no topological identity can bridge, so those keep the positions and
their offset — as does a user-supplied `matchfunc` and any mesh not built from a `MeshTemplate`. A mesh
read in at second order is the other limitation: its midside nodes never passed through
`add_intermediate_node_unique`, so `intermediate_node_map` has never seen them and their entity is not
recoverable; they fall back to an index identity, which is consistent within the mesh but cannot match a
neighbour's refinement node.

`PYOOMPH_DISABLE_TOPOLOGICAL_INTERFACE_KEYS=1` reverts to the positions everywhere. The setting is
re-read once per mesh per adaptation rather than cached forever, so an in-process `monkeypatch` reaches
it — without that the negative tests silently measured the wrong build.

### 15.4 The junction (§14.4): diagnosed, not fixed

§14.4 attributed the wrong answer at a mixed-space cross point to one multiplier slot being asked to be
two different spaces. **The measurements do not support that.** Assembling the Jacobian on a 2x2
four-domain mesh and taking its SVD:

| spaces A,B,C,D | shared multiplier name (today) | distinct name per interface |
|---|---|---|
| C1,C1,C1,C1 | ndof 13, rank **13**, cond 7.5e1, err 1.1e-16 | ndof 14, rank **13**, cond 1.1e17, err 1.1e-16 |
| C1,C1,C1,C2 | ndof 17, rank **17**, cond 4.1e2, **err 2.6e-2** | ndof 18, rank **18**, cond 2.2e3, err 1.1e-16 |
| C2,C1,C1,C2 | ndof 21, rank **21**, cond 4.0e2, **err 4.2e-2** | ndof 22, rank **22**, cond 1.0e3, err 2.2e-16 |

The shared-name system is **full rank and wrong** — not singular, but a well-posed system imposing a
different condition. A domain owning two interfaces has one multiplier on the node where they meet, so
its residual enforces the **sum** of the two coupling conditions rather than each. With equal spaces the
remaining multipliers still force them individually — that redundancy is real, and merging is what keeps
the cross point full rank. With a mixed space the redundancy **disappears** (distinct names go from rank
13/14 to 18/18) and the merged system is no longer equivalent.

Note `C1,C1,C1,C2` fails while *every* multiplier in it negotiates C1, so no space conflict exists at
all — and A, which owns both slots, touches only B and C, all three C1. It is **D**, which meets A at the
cross point and nowhere else, that breaks it. Any criterion stated over a domain's own partners misses it.

`Problem._check_coupled_interface_junctions()` therefore **refuses** the configuration: it reconstructs
each junction by connecting the boundaries that meet pairwise, and throws when a domain's interfaces share
a multiplier name there while the domains at that junction do not share an element space. A distinct
`lagr_mult_prefix` per interface is correct in every configuration measured, at the price of the singular
Jacobian above should the junction later become homogeneous; `with_removed_overconstraining()` is the
other escape hatch. Making that trade automatic needs the redundancy counted from the junction topology —
where *k* domains meet with *m* pairwise couplings only *k−1* conditions are independent — which is its
own piece of work. `use_highest_space=True` is **not** a workaround: measured, it only shrinks the error
from 1.354e-03 to 4.273e-04.

### 15.5 The quad-face out-of-bounds (§14.6)

`Quad2dFaceOrientation::analyze` built `opposite_node_index` at **this** element's `nnode_1d`, so a C2
brick face (9 nodes) held indices up to 8 into a 4-node C1 face. It now builds the map at the opposite
side's `nnode_1d` and writes `-1` where this side has a node the other does not — what the 1d line
elements and the triangular faces have always done.

### 15.6 What mixed order still costs, and it is not a bug

With the identity fixed, the two sides pair up correctly — but they still integrate the coupling over
**different surfaces**, because a C2 side's interface is a quadratic curve where a C1 side's is the chord.
On a *curved* interface that shows up directly in the answer. Measured on `curved`/`curved_tri` with the
exact solution `u = y`:

```
same order (C1-C1, C1-C1TB, C2-C2, C2TB-C2TB, ...)   max|u-y| ~ 1e-14
mixed order (C1-C2, C2-C1, C1TB-C2, C2TB-C1, ...)    max|u-y| ~ 1e-4 ... 1e-3
```

That is an O(h²) consistency error, not a defect of the coupling machinery, and it is the price of the
decision recorded in §15.7. On a flat interface it vanishes, which is why §14.2 saw nothing.

A separate, pre-existing effect in the same family: a coupling whose **outer** side is richer than the
multiplier leaves that side's extra facet dofs carrying an unconstrained reaction. Measured on 3d tets,
`C1` against `C2TB` gives `max|u-y| = 4.09e-3` on a flat interface, and `C1TB` against `C2TB` now joins it
(it used to throw). Not introduced here, and not addressed here.

### 15.7 What the sweeps say now

The §14 matrix, re-run against the fix. Same harness, same one-process-per-case discipline.

| sweep | before (§14) | after |
|---|---|---|
| `connect`, flat, 2d, all space pairs | 42 ok, **2 segfault** | **44 ok** |
| `curved` / `curved_tri`, curved entity | 19 ok, **1 segfault** | **20 ok** |
| `box_hex` / `box_tet`, 3d | 16 ok, **4 throw** (every C1TB pair) | **20 ok** |
| `move`, curved+moving interface | 22 ok, **22 fail** (20 conformity throws + 2 segfaults) | **44 ok** |
| `connect` under `mpirun -n 2 --distribute` | 19 ok (C2/C1TB excluded: it segfaulted) | **20 ok**, `ndof` identical to serial |
| `move` under `mpirun -n 2 --distribute` | — | 19 ok, `ndof` identical to serial |

The 20 `move` cases that used to throw are exactly the mixed-order pairs, and they are exactly the ones
that fail again under `PYOOMPH_DISABLE_TOPOLOGICAL_INTERFACE_KEYS=1`
(`test_mixed_order_needs_the_topological_identity`).

One distributed case does not converge: `move:C2TB/C2TB` on `tri_left` under two ranks fails in the Newton
solver. It is **not** a coupling failure -- it converges serially, the interface machinery gets all the way
to the solve, and it fails identically with the topological identity switched off. A C2TB-triangle moving
mesh with a prescribed curved interface is simply a hard problem for that partition; recorded here rather
than hidden, and excluded from nothing.

### 15.8 Rejected, with the reason

**Constraining the higher side's interface geometry to the meet**, via the existing
`ConstrainPositionsToC1Space` / `additional_dof_constraints` machinery. It would have made the two sides
share one surface, removed §15.6's error at source, and left every Eulerian key working untouched — at the
price of a C2 domain no longer being able to curve at a coupled interface. That price was judged too high.
The machinery is there, and is re-applied after every adaptation through `reapply_boundary_conditions`,
should the trade ever look different.

**Lagrangian coordinates as the identity.** Remeshing a deformable solid interpolates `xi`, so an FSI
pair's two sides need not agree on it.
