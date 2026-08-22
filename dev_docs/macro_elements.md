# Generalised MacroElements: curved boundaries for every element type

Status: **complete.** Every element shape pyoomph has — quad, triangle, brick, tetrahedron, wedge,
pyramid — places refinement-generated nodes on curved boundaries exactly, in 2d and 3d, from hand-built
templates and from gmsh, serially and under `--distribute`, including mixed forests.
`GenericMacroElement` is ~300 lines and contains one shape-dependent function. Moving meshes were
investigated and deliberately left unchanged (§7).

Two questions drove the work:

1. How do we generalise the MacroElement treatment so that *all* shapes place their refinement-generated
   nodes on curved boundaries?
2. The boundary coordinate was a single scalar (a polar angle for a circle). An angle has a branch cut,
   so a closed loop is discontinuous somewhere. Can it be given an extra dimension — a circle point
   described by its normal `(nx, ny)` rather than by `atan2` — so that no seam exists?

The answer to (2) turned out to be the smaller change, and it partly falls out of (1) — but not in the
way originally expected: see §5.

---

## 1. What was wrong

pyoomph describes curved geometry with `MeshTemplateCurvedEntity` (`src/meshtemplate.hpp`): a two-way
map between a *parametric* coordinate and a Cartesian position. An entity is attached to a **facet** —
an edge in 2d, a face in 3d — via `MeshTemplate::add_facet_to_boundary(...)`, and each of the facet's
vertex nodes is converted to the entity's parametric coordinate and cached in
`MeshTemplateFacet::parametrics`. At element-construction time `MeshTemplate::factory_element` notices a
node on a curved facet, allocates a `MacroElement`, and calls `set_macro_elem_pt()` +
`map_nodes_on_macro_element()`. From there the intention was that oomph-lib does the rest:
`FiniteElement::get_x()` routes through `get_x_from_macro_element()` whenever `Macro_elem_pt != 0`, and
refinement builds new nodes with `father_el_pt->get_x(t, s, x)`.

That was the design. It was not what happened. Only quads and bricks built a working macro element at
all; triangles built a dead one; tets, wedges and pyramids threw. And even for quads, **new nodes were
placed by plain FE interpolation** — a quarter disc's boundary error appeared at the first refinement
(5.4e-4) and did not shrink with further refinement (7.1e-4 at levels 2 and 3), because subdividing a
polygon gives a finer polygon. With C1 fields a new node landed at the chord midpoint of a 45° arc,
`1 − cos(22.5°) = 0.0761`, and the measured error was 0.07612.

Curved quad boundaries *appeared* to work only because `Problem.map_nodes_on_macro_elements()` re-snaps
every node of every element that still has a macro element, as a separate global pass — invoked in the
initial-adaption loop and in remeshing, **and nowhere in the runtime `adapt()` path**:

```
after initialise (48 elements)                 worst |r-1| = 2.2e-16
after one runtime refinement (192 elements)    worst |r-1| = 2.2e-06
after explicit map_nodes_on_macro_elements()   worst |r-1| = 2.2e-16
```

Triangles were worse than "cannot refine": the macro element was inert from the moment it was created.
A C2 curved triangular mesh has a mid-edge node per rim facet, placed at the chord midpoint and never
snapped, because `map_nodes_on_macro_element` returned early for T-elements. Measured on an 8-segment
disc: 7.6e-2, exactly `1 − cos(22.5°)`. C1 was exact only because it has no node that could be wrong.
Refining then threw `"MACRO ELEM"` — and **any adaptively refined gmsh triangle mesh with a circular or
spline boundary died**, since the gmsh reader does attach `CurvedEntityCircleArc` to 1d facets. The only
workaround was `with_macro_element=False`, i.e. give up the curved boundary.

### 1.1 The two structural obstacles

**(i) oomph-lib's blending is Q-only.** `MacroElement::macro_map` is implemented in `QMacroElement<2>`
and `<3>` as a transfinite (Coons/Gordon–Hall) blend of the reference square's or cube's boundary
parametrisations. The construction is intrinsically tensor-product: it names N/S/W/E and L/R/D/U/B/F, and
corrects the interior by `(1-η)·diff_S + η·diff_N + (1-ξ)·diff_W + ξ·diff_E`. There is no meaningful
specialisation to a triangle, tetrahedron, wedge or pyramid — which is why `TMacroElement<2>` existed
only as a stub whose `macro_map` threw. The sub-element region bookkeeping is Q-only in the same way:
`QElementBase` tracks which part of the macro element a refined son occupies with an axis-aligned box
(`s_macro_ll`/`s_macro_ur`), and a red-refined triangle son is not an axis-aligned sub-box of its father,
while a tetrahedral son of a pyramid is not even the same shape as its father.

**(ii) The solid build discards the macro positions.** Every pyoomph quad and brick derives from
`oomph::RefineableSolidQElement<2>/<3>` because pyoomph always supports moving meshes. That class's
`build()` calls `RefineableQElement<2>::build()` — which *does* place new nodes via the macro map — and
then overwrites every position with the FE value:

```cpp
// x_fe is the FE representation -- this is all we can work with in a solid mechanics problem.
// If you wish to reposition nodes on curvilinear boundaries [...] you'll have to do this yourself!
elastic_node_pt->x(i) = x_fe[i];
```

This is deliberate on oomph-lib's side. `further_build()` is not a usable hook to undo it — it runs
*inside* `RefineableQElement<2>::build`, before the overwrite. Only an override of `build()` itself can
repair it.

There is a second FE-overwrite, in `RefineableQElement<2>::quad_hang_helper` ("Fine adjust the
coordinates (macro map will pick up boundary accurately but will lead to different element edges)").
That one is **correct and must be preserved** — see §6.1.

### 1.2 Two incidental defects, and one silent one

* **Leftover debug output** on the whole 3d path (`MeshTemplateQMacroElement3::macro_element_boundary`
  printing `STARTING FACE i`, `CurvedEntityCylinderArc` printing on every parametric conversion,
  `CurvedEntitySpherePart` printing its frame in the constructor), consistent with that path never
  having been finished. Removed. `Mesh::set_lagrangian_nodal_coordinates` had the same problem and was
  noisy on every moving mesh.
* **Curved entities were held as raw borrowed pointers.** `add_facet_to_boundary` stored the
  `MeshTemplateCurvedEntity*` with no ownership and no nanobind `keep_alive`, so a Python user who
  constructed the entity as a local inside `define_geometry` got a **segfault** as soon as it was
  collected. The shipped meshes avoided it only by accident, by keeping a list alive. Fixed with a
  `keep_alive` on the binding.
* **A reflection where an unwrap was meant** — the one that mattered, and it was silent.
  `CurvedEntityCircleArc::apply_periodicity` had

  ```cpp
  if (parametric[0][0] > 0)  parametric[0][0] = -M_PI + (parametric[0][0] - M_PI);   //  p - 2*pi   correct
  else                       parametric[0][0] =  M_PI - (parametric[0][0] + M_PI);   //  -p         a reflection
  ```

  The second line returns `-p`, which is not congruent to `p` modulo 2π at all — it is the mirror image
  across the *x* axis. The entity then reported the wrong point for that facet corner, the Coons blend's
  corner values stopped agreeing, and the mesh came out **silently wrong**: no error, a boundary node up
  to 3.4e-2 off a unit circle. Which of the two failure modes an arc hit depended on the order in which
  the mesh author happened to declare the facet's two nodes — a free choice describing identical
  geometry:

  ```
  30 deg arc, facet order   a0 = 155..165           a0 = 170..180
  start-to-end              throws                  exact
  end-to-start              builds, 1.9e-2..3.4e-2  throws
  ```

  Fixed by replacing both that heuristic and `CurvedEntityCylinderArc`'s unconditional throw with one
  shared `unwrap_periodic_component()`: shift every facet node onto the branch nearest the first node's.
  Correct at every orientation, in both node orders, and — unlike the two-node code it replaces — for the
  three- and four-node facets a 3d face has. The Python `CurvedEntityCircle` example had the same class
  of bug (a per-value `numpy.mod`, which moves the cut instead of removing it) and got the same fix.

---

## 2. The generalisation: C1 vertex shapes as generalised barycentrics

`BulkElementBase::shape_at_s_C1(s, psi)` is pure virtual and implemented by every concrete element,
wedges and pyramids included. For each shape it returns the vertex shape functions `λ_v(s)`, and for
each shape these have the three properties the blend needs:

* `Σ_v λ_v(s) = 1` everywhere (partition of unity);
* `λ_v = 1` at vertex *v*, 0 at every other vertex;
* `λ_v ≡ 0` on every facet that does not contain *v*.

For a simplex these *are* the barycentric coordinates; for a quad/brick they are the bi/trilinear shapes,
for a wedge the (triangle barycentric)×(linear) product, for a pyramid the standard rational shapes. This
is the unification: **treat `shape_at_s_C1` as a generalised barycentric coordinate system and write
everything in terms of it.** The straight-sided reference map is then `x_lin(s) = Σ_v λ_v(s) X_v`.

### 2.1 Blended deviation — which is exactly the Coons / Gordon–Hall blend

For a sub-entity *S* (a facet or an edge) with vertex set `V(S)`, define the weight
`w_S(s) = Σ_{v∈V(S)} λ_v(s)`; the restriction `σ_{S,v}(s) = λ_v(s)/w_S(s)`, which is a partition of unity
*on S* and by the third property is the correct facet-local coordinate whenever *s* lies on *S*; the
straight image `L_S(σ) = Σ σ_v X_v`; the curved image `C_S(σ)` (the entity at the σ-blend of *S*'s stored
parametrics); and the deviation `d_S = C_S − L_S`, which vanishes at every vertex of *S*. Then

> **x(s) = Σ_v λ_v X_v + Σ_{F curved} w_F · d_F(σ_F) − Σ_E (m_E − 1) · w_E · d_E(σ_E)**

where the last sum runs over edges *E* contained in `m_E ≥ 2` curved facets (3d only; empty in 2d,
because two edges of a 2d element meet only at a vertex, where `d` already vanishes).

**This is not a new or weaker blend — it is the classical transfinite interpolant, rewritten so that the
shape does not appear.** For a quad it telescopes line for line into `QMacroElement<2>::macro_map` (the
Boolean sum `P₁ ⊕ P₂ = P₁ + P₂ − P₁P₂`); for a brick with all six faces curved into
`QMacroElement<3>::macro_map`. So **existing curved quad meshes do not move at all**, interior included.
For a triangle each vertex lies on two edges, so it collapses to `x = Σ_E w_E C_E(σ_E) − L`; for a
tetrahedron to the brick's closed form. Wedges and pyramids need no separate derivation — they are
covered by the general `(m_E − 1)` statement, which is what the implementation evaluates.

Correctness, independent of the equivalences: on a curved facet *F* (`w_F = 1`, `σ_F` the actual facet
coordinate) the first two terms give exactly `C_F`, and every other curved facet contributes at a point
where its σ has collapsed onto the shared edge, so its deviation is that edge's — which the correction
subtracts exactly. At any vertex all deviations vanish, so vertices never move.

**The equivalence was executed, not asserted.** A Python reference implementation of the deviation form
was compared against oomph's actual `QMacroElement<2>::macro_map` on a 41×41 grid over the *whole*
reference square, through the new `Element.get_macro_element_position_at_s()`:

```
arc      curved facets   max |coons - deviation|   interior bending
0..45          1                4.0e-16                7.6e-02
170..200       1                1.1e-15                3.4e-02
150..210       1                9.0e-16                1.3e-01
0..45          2                4.0e-16                5.0e-01
```

"Interior bending" is how far the macro map departs from the straight-sided bilinear map — three to five
orders of magnitude larger than the discrepancy, so the agreement is a real statement about two different
*curved* maps, not two flat ones agreeing trivially. The refactor was landed as its own commit before any
behaviour changed, and the fast suite was byte-for-byte unchanged across it.

### 2.2 Partially curved 3d elements — where this is strictly better

A brick with one curved face *B* and five interior faces is the common 3d case, and it is where the old
3d path was wrong. `MeshTemplateQMacroElement3::macro_element_boundary` returned the **flat** bilinear
corner interpolation for a face with no entity attached. But the four edges of *B* lie on the curved
surface, so a side face *W* meeting *B* along edge `E` claimed `E` was straight while *B* claimed it was
curved; Gordon–Hall then blended two contradictory boundary descriptions and *B* was no longer reproduced
exactly. (In 2d this cannot happen: an edge's boundary is two corner nodes, on which everybody agrees.)

The deviation form has no such failure mode. On the side face one gets `x|_W = L_W + w_B|_W · d_E(σ_E)`,
i.e. *W* becomes the **ruled surface** between the curved edge and the opposite straight edge. That
expression depends only on *W*'s vertices and on `E`, so the neighbouring element across *W* computes the
identical surface: the mapped geometry stays watertight with no extra machinery.

Verified on a tetrahedron whose four vertices all lie on a sphere — and *any two faces of a tet share an
edge*, so it is the sharpest available test of the inclusion–exclusion term. Sampling the macro map on a
21-point barycentric grid over each face:

```
curved faces   max |r-1| on the declared-curved faces   on the others
      1                    2.2e-16                        1.1e-01
      2                    3.3e-16                        1.1e-01
      3                    3.3e-16                        9.7e-02
      4                    5.6e-16                          --
```

The right-hand column matters as much as the left: a face that was *not* declared curved must not be
dragged onto the sphere — it is a ruled surface carrying whatever curved edges it inherits.

### 2.3 Curved edges are a mesh-level fact, not an element-level one

The watertightness argument has a precondition: both elements sharing *W* must know that `E` is curved.
Element *A* knows because it owns a curved face containing `E`. Element *B* usually knows for the same
reason — **but not necessarily**: *B* can touch the curved surface along `E` alone, without having a face
on it.

This was initially dismissed on the grounds that every element touching the sphere in
`SphericalOctantMesh` and the tet ball touches it in a face. That is a property of hand-built structured
meshes, not of curved geometry. An unstructured gmsh tetrahedral ball breaks it as a matter of course:

```
nodes on the shell   elements   had a macro element
        0               44             no
        1               12             no
        2               25             no      <-- these are the problem
        3               45             yes
```

Those 25 had no curved facet, so they got no macro element, and when refined they placed the midpoint of
that shared edge on the chord while the element on the other side placed the same node on the sphere.
Whichever built the node first won.

`MeshTemplate` now builds `curved_edge_map`, keyed by sorted template node-index pair, from the edges of
every curved facet; `factory_element` gives an element a macro element carrying those edges as
**two-vertex curved sub-entities** whenever it touches one without owning the facet. `GenericMacroElement`
needed nothing at all: a sub-entity of two vertices is already what the blend expects, and `w_E d_E` on an
otherwise straight element is precisely the ruled surface that makes the two sides agree. With the
registry, 70 of the 126 elements carry a macro element and the shell is exact at every refinement level.

Two details worth keeping: edges already inside one of the element's own curved facets are **skipped**, or
the facet deviation and the edge deviation would both be applied and the edge counted twice; and only
*genuine* edges are registered, from a per-shape table (`macro_edges`) — a quadrilateral facet's diagonal
joins two points of the surface but is not an edge of anything, and curving it would bulge the element's
interior rather than its boundary. `GenericMacroElement::rebuild_edge_corrections` is where an
inconsistency check belongs if two entities ever claim the same edge.

### 2.4 Son-region tracking

The son's region is carried as its vertex coordinates in the *root macro element's* reference domain,
propagated father → son by one line of the same primitive:

```
S_macro_vertex_son[k] = Σ_v ψ^{C1,father}(sv_k) · S_macro_vertex_father[v]
```

where `sv_k` is son vertex *k* in father-local coordinates — a vector that already exists for every shape
from the adaptivity work (`son_vertices_in_father`). Then
`get_x_from_macro_element` evaluates `s_macro = Σ_k ψ^{C1,this}(s) · S_macro_vertex[k]` and calls
`macro_map`.

This is **the only formulation that survives a mixed forest**, where a tetrahedral son has a pyramidal
father: the father's C1 shape is evaluated at the son's vertex coordinates, so the differing vertex counts
never appear in the same expression. oomph's axis-aligned `s_macro_ll`/`s_macro_ur` cannot express it at
all, which is the concrete reason the box had to be generalised rather than extended.

The Q family nonetheless **keeps** the box: oomph's own build already maintains it correctly, the affine
map into it is exactly what the general vertex form reproduces, and `QElementBase` already provides a
final overrider for `get_x_from_macro_element`. Adding a second one reachable from the same class would
have been ambiguous under virtual inheritance and would have forced a disambiguating override into every
Q class for no gain. So `get_macro_element_coordinate_at_s` branches once: box for Q, vertex coordinates
for everything else.

### 2.5 What collapsed into `GenericMacroElement`

New `src/macroelements.{hpp,cpp}`; `src/Tmacroelements.hpp` deleted. It stores the element kind, the
vertex **positions by value** (§8.1 says why not pointers), the curved facets, and the 3d edge
corrections.

* **`find_permutation` disappears.** It existed only to reconcile the macro element's canonical facet
  node order with the facet's own order, and the brick version brute-forced all 4! permutations. Each
  weight is now carried together with the vertex it belongs to, so ordering ambiguity cannot arise. This
  also removes the only place that would have needed a 3!-permutation variant for triangular faces.
* **`MeshTemplateQMacroElement2` / `MeshTemplateTMacroElement2` / `MeshTemplateQMacroElement3` and
  `oomph::TMacroElement<DIM>` all collapse into one class**, and the four near-duplicate bodies in
  `factory_element` into one loop.
* **`MeshTemplateDomain::macro_element_boundary` and its `dynamic_cast` chain disappear** — the macro
  element evaluates itself. `MeshTemplateDomain` survives only because `oomph::MacroElement`'s
  constructor demands a `Domain*`.
* The kind is stored rather than a back-pointer to the root element: evaluating `λ_v(s)` needs only the
  shape, so a free function `c1_shape(kind, s, psi)` avoids a lifetime coupling.

**The pyramid was the test of the whole design, and it passed unchanged.** Its C1 shape functions are
*rational* — `psi_0 = (w−s0)(w−s1)/w` with `w = 1−s2` — because the whole quadrilateral base collapses to
a point at the apex. They are still a partition of unity, still 1 at their own vertex and 0 at the others,
and still vanish on non-adjacent facets, which is all §2.1 asks. So the blend needed no pyramid-specific
term, only an entry in `macro_c1_shape`. One caveat: the shipped `PyramidElementShapeC1::shape` divides by
`1−s2` unguarded, which is fine for its callers but not for a macro map evaluated at arbitrary points, so
the macro version takes the apex limit explicitly. And refining a pyramid yields **six pyramids and four
tetrahedra**, so a son inherits a macro element from a father *of a different shape* — which works only
because of §2.4.

---

## 3. Where it plugs in

1. `BulkElementBase::get_x_from_macro_element` — the §2.4 implementation. Note the inheritance hazard:
   `QElementBase` also overrides this virtual, so the safe form is a non-virtual
   `get_x_from_generic_macro_element(...)` plus a two-line `override` in each concrete class that needs it
   (only the simplex classes, whose bases leave the virtual broken).
2. `BulkElementBase::map_nodes_on_macro_element` — no `qelem`/`telem` split, `local_coordinate_of_node`
   + §2.4 for every shape.
3. `BulkElementQuad2dC1/C2::build` and `BulkElementBrick3dC1/C2::build` — after
   `RefineableSolidQElement<N>::build`, re-apply the macro position to the nodes this build created (§1.1
   obstacle (ii)). Gated on `moving_nodes`, see §7.
4. The T-family, wedge and pyramid builders already place new nodes through `father_el_pt->get_x(t, s, x)`
   and are *not* `RefineableSolidQElement`s, so nothing overwrites the result and they need **no**
   post-build repair. The Q family is the awkward one, not the simplex family. The four remaining throws
   (wedge, `build_as_pyramid_son` which also serves tet sons of a pyramid, and `build_as_brick_son`) all
   became the same two lines, `inherit_macro_element_from_father(father, sv)`, since each builder already
   computes the son's vertices in father coordinates for its own node mapping. The brick son needed
   `collect_son_vertex_coords_in_father`, which recovers the eight vertices by matching each reference
   vertex to the son node sitting there.
5. `factory_element` — one loop over `el->nfacets()` for all shapes; `construct_facet` was already
   implemented for every shape in both C1 and C2.

---

## 4. gmsh in 3d: `map_to_sphere` has to be opt-in

`GmshTemplate._curved_entities2d` was declared with a `# TODO: Set those` and never populated, so a gmsh
3d mesh had no curved geometry at all. The consumer side already existed; populating the dict is the whole
of the wiring.

What it cannot be is **automatic**. Gmsh's built-in kernel does not produce an exact sphere from a ruled
surface, even when the bounding curves are great-circle arcs — the surface it meshes is its own ruled
interpolant. Attaching a spherical entity to every ruled surface would impose a geometry gmsh never
meshed. So `ruled_surface(..., map_to_sphere=True)` is opt-in, and it is the user asserting "this surface
is meant to be a sphere", which pyoomph then makes exactly true (including projecting the surface's own
nodes onto it, which moves them slightly).

Centre and radius are recovered from the bounding curves with a deliberate bias toward refusing:

* only points *certainly* on the surface are used — the two endpoints of each bounding curve. A
  `CircleArc`'s middle point is its arc centre, which lies on the sphere only if the arc is a great
  circle, so it is collected separately as a *candidate centre*;
* a candidate centre is accepted only if every boundary point is equidistant from it. For the usual
  construction — a patch bounded by great-circle arcs about the sphere's centre — that is exact, so no
  fitting happens at all;
* failing that, a least-squares sphere fit, but **only from five distinct boundary points on**. Four
  points in general position lie on exactly one sphere, so a fit through four always succeeds and the
  verification could never reject anything; requiring a fifth is what makes agreement evidence;
* otherwise it raises, and says that `map_to_sphere=(cx,cy,cz)` states the centre explicitly.

Measured on an octant of a tetrahedral ball: `map_to_sphere=True` gives shell error `0`, `2.2e-16`,
`2.2e-16` at 0, 1 and 2 refinements; without it, `3.3e-16`, `1.5e-2`, `1.6e-2` — the nodes gmsh itself
places are on the sphere, everything refinement adds is not.

---

## 5. The parametric coordinate as an opaque vector

`MeshTemplateCurvedEntity::get_parametric_dimension()` (how many doubles are stored) is now separated
from `get_intrinsic_dimension()` (manifold dimension: 1 curve, 2 surface), and
`get_parametric_dimension() >= get_intrinsic_dimension()` is the normal case for closed geometry. The
storage side was already dimension-agnostic; the blend side hardcoded `std::vector<double> parametric(1)`
in 2d and `(2)` in 3d, and that was the whole mechanical change.

### 5.1 The rule: a redundant chart when the natural one is *degenerate*, not merely when it wraps

This is the conclusion that matters, and it is the opposite of what was planned.

**The sphere gets normals.** `(θ, φ)` with `θ = acos(z)`, `φ = atan2(y, x)` has a coordinate *degeneracy*
at the pole, not merely a branch cut: at θ = 0 every φ names the same point, so no amount of unwrapping
makes a facet containing or straddling the pole blend correctly. It is also what forced
`CurvedEntitySpherePart`'s 90°-opening-angle restriction. `position_to_parametric(x) = (x − c)/|x − c|`
removes the seam, the pole degeneracy, the tangent/cotangent frame construction and the opening-angle
limit in one go. This was needed before anything 3d could be tested at all.

**The circle does not, and rewriting it would make it worse.** `CurvedEntityCircleArc` parametrises by
angle and maps through `cos`/`sin`, so blending the angle linearly **is already exact slerp** — uniform in
arclength, exact at every weight. Moving to normals would mean either nlerp or implementing slerp on
normals to get back to precisely where it started. Measured, the cost of nlerp (worst case over all
weights, unit circle):

```
facet width   tangential shift   as % of edge length   as % of the sagitta
   15 deg        2.9e-04                0.11%                 3.4%
   45 deg        8.0e-03                1.05%                10.5%
   90 deg        7.1e-02                5.03%                24.3%
  120 deg        1.9e-01               11.03%                38.2%
```

(At weight ½ — the bisector, which is what edge refinement of a facet actually asks for — the two agree
exactly; the numbers come from the off-centre weights that deeper tree levels produce. And the error is
*tangential*: both blends put the node exactly on the circle, they differ only in where along the arc it
lands.) The circle's branch cut is handled exactly by `unwrap_periodic_component` (§1.2), verified across
the full orientation sweep in both node orders. A redundant chart buys nothing there.

The one case the normal representation cannot express is a facet spanning exactly 180° or more (antipodal
normals sum to zero). It must be rejected at facet-construction time with a message saying to split the
facet — a strictly milder restriction than the old one, which triggered at 16° purely because of where the
arc sat relative to the seam.

### 5.2 `blend_parametric`, and why it is load-bearing

A parametric coordinate is an opaque vector whose meaning belongs to the entity, so the rule for combining
two of them belongs to the entity too. `blend_parametric(weights, params, result)` defaults to the
weighted sum — correct for a flat, non-redundant chart: an angle, an arclength, a spline parameter — and
`CurvedEntitySpherePart` overrides it to renormalise, because the average of two unit normals is not a
unit normal. `GenericMacroElement::subentity_deviation` calls it instead of summing inline, at arbitrary
weights, not only at a facet's vertices.

Distinct from `apply_periodicity()`, which runs once when a facet is built and merely chooses which
representatives of a periodic coordinate to store. Both are kept and the header says which is which.

The hook is exposed to Python, which is the point of it: a user can define an entity whose chart is
redundant. `test_user_entity_can_own_its_blending_rule` builds exactly that — a circle charted by a
2-component unit normal — and asserts **both** halves, because the interesting claim is not that it works
but that the hook does something: with the override the rim is exact at 1.1e-16; with the default weighted
sum, 7.6e-2, the chord sagitta of a 45° facet, i.e. no better than no curved treatment at all.

`facet_map` (blend-then-map as one overridable call) and `project_onto` (closest-point projection) were
designed and **dropped as speculative** — neither has a caller, and adding unused virtuals to a class users
subclass is a cost with no return. Projection is worth remembering as a design if a spline whose
parametrisation is only approximately arclength ever needs it, but it must be *blend first, then project*:
on a closed or folded curve the chord midpoint can project onto the wrong branch, so only the blend
guarantees the right one. `get_intrinsic_dimension()` *was* added despite having no caller in the core,
because it makes an otherwise puzzling asymmetry legible — a sphere patch reporting
`get_parametric_dimension() == 3` looks like a bug until something says it is a 2-manifold on purpose.

### 5.3 What the blend path costs, and the guidance that follows

The macro map is **not** in the assembly loop — it is reached only from `map_nodes_on_macro_element` (once
per node, at mesh build and after each adaption) and from `FiniteElement::get_x` while refinement builds
new nodes. So the frequency is O(nodes created), not O(quadrature points × Newton steps), and
`pyoomph/meshes/interpolator.py` already strips macro elements before `locate_zeta`, which is the one other
route that would have made it hot.

Measured on a triangular disc refined to 8192 elements (`refine` = four uniform refinements, `resnap` = one
full `map_nodes_on_macro_elements()` pass):

```
                                     refine     resnap    boundary error
no macro element at all              0.221 s    0.006 s      4.8e-03
built-in C++ entity                  0.240 s    0.026 s      2.2e-16
Python entity, no blend() override   0.333 s    0.171 s      2.2e-16
Python entity, with blend()          0.431 s    0.311 s      2.2e-16
```

The C++ path is not a concern (~8% on refinement). An attempt to remove the remaining per-call allocation
by hoisting the blend scratch into member buffers recovered ~8% of the 26 ms and was **reverted**: it
introduces a mutable member whose failure mode, if a re-entrant Python callback clobbered it mid-use, is
silently wrong geometry.

A Python entity costs roughly an order of magnitude more, mostly not new. But **overriding `blend()`
roughly doubles that**, because it is a second callback per evaluation — and the third row above leaves
`blend()` alone and normalises inside `parametric_to_pos` instead, at equal exactness and half the
overhead. Hence the guidance, now also in the header where an entity author will meet it: **fold a
correction into `parametric_to_position` when it can be expressed pointwise; override `blend_parametric`
only when the rule genuinely needs the other samples** — unwrapping a periodic coordinate relative to its
neighbours, or a true slerp.

---

## 6. Interactions

### 6.1 Hanging nodes

A hanging node's position is determined by its masters. If one on a curved boundary were snapped to the
curve, the two sides of the edge would no longer agree and `check_integrity` would fire. oomph gets this
right for quads by overwriting hanging-node positions with the coarse neighbour's `interpolated_x`, and
that overwrite must survive.

`map_nodes_on_macro_element` snaps unconditionally, so it needs a "is this node hanging" guard. In 2d the
guard **never fires**, for a structural reason worth stating: *a node interior to a boundary facet belongs
to exactly one element*, because a boundary facet has no neighbour across it, so nothing coarser can
constrain it — and in 2d the facets of a curved boundary *are* its edges, so boundary nodes cannot hang.
Measured on a deliberately non-conforming disc: 24 hanging nodes (quad) and 64 (tri), none on the rim, rim
error 0. The guard stops being vacuous in 3d, where two boundary *faces* share an edge and a node on that
edge can hang when the two faces differ in refinement level.

### 6.2 Coupled interfaces

Two coupled domains sharing a curved interface place the shared nodes identically, because both build
their macro elements from the same `MeshTemplate` facets and the same entity. That was an argument, and it
was worth checking, because the two sides run independent refinement decisions through independently
constructed macro elements and "same inputs" only implies "same outputs" if nothing in between introduces
an ordering or accumulation difference. Measured on two concentric annuli with only the inner one told to
refine, so `InterfaceRefinementCoupler` has to carry the requirement across: interface exact to 1.1e-16
from both sides, and the two domains' interface node sets **identical**, not merely close. The
straight-sided control gives 1.3e-2.

### 6.3 Remeshing and interpolation

`pyoomph/meshes/interpolator.py` strips macro elements before `locate_zeta` ("really troublesome") because
`locate_zeta` inverts `get_x`, and inverting the macro map is expensive and non-robust. That stays; the
generic macro map does not make it worse. Inverting the macro map is explicitly not attempted.

### 6.4 What this does not attempt

Curved *interior* elements (only facets carrying an entity are curved; the interior deviation decays
linearly — no high-order P2/P3 geometry); curved 3d edges attachable **independently** of faces (a 3d
feature edge takes its geometry from the faces, and §2.3's registry is where an explicit attachment would
go); and making `map_nodes_on_macro_elements` unnecessary — it is kept as an invariant check.

---

## 7. Moving meshes: investigated, closed without a code change

This is the one place the design needed a decision rather than a derivation, and it is why oomph's solid
build discards macro positions in the first place. When a domain has `functable->moving_nodes`, the
Eulerian position **is an unknown**, and forcing a new node onto the curve is wrong in general: the mesh
has already moved away from the template geometry, and the curve describes where the boundary *was*.

Today `reapply_macro_element_positions` is a no-op when `moving_nodes` is set, and separately
`remove_macro_elements_after_initial_adaption = "auto"` drops the macro elements once the coordinates are
free. The measurable gap is a drift to **7.13e-4** after two runtime refinements where a static mesh is
exact. Four options were weighed — leave as today; snap only where the mesh position is pinned; always
snap; an explicit per-boundary declaration. **Decision: leave as today.** The drift is second-order and
only affects a case (a curved boundary that has not moved, on an adaptively refined ALE mesh) that no
shipped example exercises, whereas every candidate fix changes ALE results for meshes that are currently
correct.

Two things about the scope of this, because both are easy to read the wrong way round:

* **The trigger is `moving_nodes`, not pinning.** Both guards key on "are the coordinates dofs", never on
  whether a node's position is pinned, and the drift measures the same with and without a position
  Dirichlet condition on the boundary (5.4e-4 either way after one uniform refinement of the §6.1 quarter
  disc). Whether the boundary is held is what decides if the drift is *unwanted*, not whether it happens.
* **Initial refinement is not affected.** It runs inside `initialise()`, before the strip and before the
  closing `map_nodes_on_macro_elements()`, so a curved boundary refined at startup is exact with or
  without an ALE equation attached — `RefineToLevel(3)` on the quarter disc puts all 33 rim nodes on the
  circle to 3e-16 — and is then free to move. Only refinement *during* the run is at issue.

### 7.1 The route the plan proposed does exist, and two wires are missing

*Corrected 2026-08-22. This section previously reported that `interpolated_xi` returns zeros at every
node, and concluded that ξ is not the quantity pyoomph's ALE reads and that the route was therefore a
dead end. That measurement no longer holds — it predates the fix that gave refined simplex nodes their
interpolated ξ — and the conclusion drawn from it was wrong. Re-measured below.*

The plan was that on a moving mesh the macro element should drive the **Lagrangian** coordinate ξ, via
oomph's `Undeformed_macro_elem_pt` and `enable_use_of_undeformed_macro_element_for_new_lagrangian_coords`.
That *is* the mechanism pyoomph uses. `PseudoElasticMesh` builds its residual from `X = var("lagrangian")`
(`pyoomph/equations/ALE.py`), `var("lagrangian")` expands to the `lagrangian_x/y/z` fields, and the shape
buffer fills those from `raw_lagrangian_position_gen` (`src/elements_shapeinfo.cpp`) — i.e. from
`oomph::SolidNode::xi`. On the quarter disc of §6.1 with a `PseudoElasticMesh` attached, ξ is populated
and equal to x at initialisation (`max|ξ| = 1.000000`, `max|x − ξ| = 0`). Wiring
`Undeformed_macro_elem_pt` therefore drives exactly the quantity the ALE residual reads.

What is missing is two wires, both on the pyoomph side:

1. **`Undeformed_macro_elem_pt` is never set.** `SolidFiniteElement::set_macro_elem_pt` exists precisely
   to default it to the same macro element, but `QSolidElementBase::set_macro_elem_pt` overrides it and
   chains to `FiniteElement::set_macro_elem_pt`, which sets `Macro_elem_pt` alone.
2. **The flag never reaches a son.** `RefineableSolidElement::further_build()` exists precisely to pass
   `Use_undeformed_macro_element_for_new_lagrangian_coords` down from the father, but
   `BulkElementBase::further_build()` overrides it without chaining, so every son starts `false`.

So a new node takes ξ from `xi_fe`, the FE interpolation of the father's ξ — which leaves ξ exactly as far
off the curve as x. That is why `DirichletBC(mesh=var("lagrangian"))` on the rim reproduces the drift
rather than curing it: both sides of the condition are chord-bound.

Prototyped on the quarter disc (§6.1 geometry, `PseudoElasticMesh`, one runtime uniform refinement) by
setting the undeformed macro element to the same macro element, propagating the flag in `further_build`,
and retaining the macro elements past `initialise()`:

```
                                                      rim |x|-R    rim |xi|-R
today                                                  5.4e-4        5.4e-4
xi-from-macro, rim pinned rigidly                      5.4e-4        2.2e-16
xi-from-macro + DirichletBC(mesh=var("lagrangian"))    2.2e-16       2.2e-16
```

The middle row is the informative one: the two halves are independent. The ξ route fixes the *reference*
geometry; `DirichletBC(mesh=var("lagrangian"))` is what carries it into x, and it survives the solve
(`max|x − ξ|` back to 1.6e-16 in the interior afterwards).

**§7's decision still stands, for §7.2's reason rather than this one.** Three things would have to be
settled before this becomes a feature:

* **Q family only.** `get_x_and_xi` is `QSolidElementBase`'s and uses its `s_macro_ll/ur` box. The
  simplex, wedge and pyramid families have no equivalent; they would need their own ξ assignment where
  `inherit_macro_element_from_father` already runs.
* **The macro element has to survive `initialise()`,** which `remove_macro_elements_after_initial_adaption
  = "auto"` strips on an ALE mesh. Retaining it re-exposes what §7.3 describes. The clean version keeps it
  for ξ only and never lets it touch x — which is how `macro_element_may_set_positions()` is already
  structured.
* **§7.2's parametrisation objection survives in full.** Pinning x to ξ fixes a node's position *along*
  the arc, not just its distance from the centre, so a contact line sliding on a curved wall is dragged
  back. As a boundary the mesh author explicitly declares, that is the intent; as a default it is the bug
  §7.2 describes. Which is §7.2's own conclusion — an explicit per-boundary declaration — except that the
  mechanism turns out to be reachable today and `DirichletBC(mesh=var("lagrangian"))` is already the
  declaration.

### 7.2 "Snap only where pinned" is the wrong criterion

Two objections, both fatal to the option that was initially recommended.

**Dirichlet conditions come and go during a run.** A boundary can be pinned for part of a simulation and
free for the rest, so "is this position pinned?" is a snapshot of the current state, not a property of the
boundary. Deciding at node-creation time is coherent and matches oomph's own view (its solid build comments
that "pinning doesn't mean 'pin in place' or 'pin to the curvilinear boundary'"), but a node created while
the boundary was free is then permanently off the curve on a boundary fixed for the rest of the run.
Re-snapping whenever pinned fixes that, at the price of moving the nodes of an already-solved mesh
discontinuously the moment a condition is switched on. The tension is unresolvable because "pinned" is a
proxy for the thing actually meant: *this boundary's geometry is prescribed by the template for all time*.
That is a statement about the model, not the boundary conditions, and only the mesh author can make it —
**so the right design is an explicit per-boundary declaration.**

**A fixed curved boundary meeting a moving one.** In the corner element the two share a node, and that node
slides *along* the curved boundary — a contact line moving on a curved wall. The macro element caches each
facet's parametric coordinates when the template is built, and (§8.1) the vertex positions by value, so
both go stale as soon as that corner moves. The consequence is worse than "slightly stale": **the macro
element pins the parametrisation, not just the geometry.** Demonstrated by rotating a circular boundary
along itself, so every node stays exactly on the circle and only its position along it changes:

```
after the solve                       r in [1.000000, 1.000000]   (exactly on the circle)
after map_nodes_on_macro_elements()   r in [1.000000, 1.000000]   shifted by up to 0.0927 rad
```

Nothing was geometrically wrong and the macro map moved the nodes anyway, back toward the parameters they
had when the template was built. In a corner element that is precisely the contact line being dragged back.

Any future implementation must therefore recompute a facet's parametric coordinates from the *current* node
positions. That is possible — a curved entity is a global object and `position_to_parametric` is exactly
that query — but it reintroduces the need for live node access, which §8.1 removed. **The two constraints
have to be satisfied together, and that is the real cost.**

### 7.3 What protects the current code, and how thin it is

Exactly one thing: `remove_macro_elements_after_initial_adaption` defaults to `"auto"`. With the macro
elements retained, calling the equally public `map_nodes_on_macro_elements()` after the mesh has moved
silently undoes the motion:

```
boundary driven outward to r = 1.2, macro elements retained
after map_nodes_on_macro_elements()   r in [1.0000, 1.0000]
```

That is not a bug in the current default — it is what the mapping pass is *for* — but the combination is
reachable through documented API, so the tutorial's moving-mesh warning says so.

---

## 8. MPI: two defects that only `--distribute` could reach

Neither of the two options originally weighed (build macro elements on every rank, or accept FE placement
on halos and let synchronisation copy the owner's positions) was the issue. The macro elements were already
present on every rank — the mesh is built replicated and then thinned — and the failures were in what a
macro element *holds* and what it *sets*. Nothing had ever run this combination: distributed adaptivity was
covered, curved boundaries were covered, the two together were not.

### 8.1 A macro element must not hold node pointers

The first `--distribute` run on a curved mesh segfaulted in `GenericMacroElement::vertex_position`, reached
from `Mesh::distribute` → `classify_halo_and_haloed_nodes` → `synchronise_nonhanging_nodes` →
`FiniteElement::get_x`. `Mesh::distribute()` deletes the elements and nodes a rank does not own; a macro
element belongs to the **root** element and is shared by every son, so a surviving son can be left holding a
macro element whose root vertices have been freed.

Not introduced by this work — the implementation it replaced stored
`std::vector<std::vector<pyoomph::Node *>> default_facet_nodes` and dereferenced it from the same call site.
It had simply never been exercised.

`GenericMacroElement` stores vertex **positions by value**, and the header records why the obvious choice is
wrong, since it is the sort of thing a later reader would try to "fix" back: node pointers would let the map
follow the nodes through history levels and, on a moving mesh, through the solve — but the macro element is
only ever consulted for geometry that is fixed (§7), so there is no motion to follow.

### 8.2 `map_nodes_on_macro_element` was only setting the present time level

With the crash gone, the mesh came out geometrically corrupt: *"Max. error in quadtree neighbour finding:
0.447214 is too big"*. Serial and replicated-MPI runs of the same problem were exact (2.5e-16), so it was
distribution-specific.

The cause was a `// TODO: Time loop` that predated this work: it snapped `x(id)` — history level 0 only —
leaving `t >= 1` on the straight interpolant that `factory_element` had written there.
`synchronise_nonhanging_nodes` compares `get_x(t, s)` against `x(t, ·)` **at every history level** while
distributing, saw conforming nodes disagree, and "repaired" them.

Every history level is now set; the macro geometry does not depend on time, so one evaluation serves them
all. This also fixes something visible without MPI: a curved mesh used to start with `x(0)` on the curve and
`x(1)` on the chord, i.e. presenting a mesh at rest as though it had been moving into its initial state.

### 8.3 Two adjacent findings

* **A `Problem` must be released before the next one distributes.** An un-released `Problem` keeps its
  distributed state alive and the next `distribute()` in that process dies. A worker running several cases
  in one process without a `with` block always fails on the *second* case, whichever it is.
* **`Problem.load_balance()` called by hand** after `initialise()` dies in
  `Mesh::generate_interface_elements` with *"bulkmesh was not set"* and then segfaults. **Not** a
  curved-boundary defect — it reproduces identically with `with_curved_entities=False`, i.e. with no macro
  element in the problem at all. Something about the manual entry point leaves an interface mesh without its
  bulk mesh after redistribution. The flag-driven path through the initial adaption
  (`call_load_balance_in_initial_adaption = True`) works, and curved boundaries come through it exact (`0.0`
  on both ranks, all five geometries).

---

## 9. Public API and documentation

`MeshTemplate.create_curved_entity` — the *supported* way to build a curved entity from a hand-written
template — only ever accepted `"circle_arc"`. Every sphere in this branch, tests and `SphericalOctantMesh`
included, reached past it into `_pyoomph.CurvedEntitySpherePart` directly. That is a fair signal: if the
tests cannot use the public API, neither can a user. It now accepts `"sphere_part"` and `"cylinder_arc"`,
both taking node indices or coordinates like `"circle_arc"` does, both registered in `_macrobounds` so the
entity stays alive; `SphericalOctantMesh` was switched over to it.

`CylinderMesh` has a curved mantle (exact at every refinement level, against 7.6e-2 before) and takes
`with_curved_entities` (default `True`), consistent with `CircularMesh` and `SphericalOctantMesh`.

Two sections in `docs/source/tutorial/spatial/mesh/unitsandmacro.rst`, which already claimed curved
boundaries "resemble the very same smooth boundary curve also upon refinement" — aspirational when written,
now true. *Curved boundaries in general* states the coverage and makes the cost of *not* using one concrete
with the spherical octant's 23% volume deficit, deliberately, because the intuition this work repeatedly ran
into is that refinement will fix a coarse curved boundary and it will not. It also carries the moving-mesh
warning, so §7's decision is visible to users rather than buried here. *Writing your own curved entity*
documents the Python subclass interface, the opaque/redundant parametric coordinate with the sphere's unit
normal as the worked reason, and the §5.3 guidance.

---

## 10. Tests

`tests/test_curved_boundaries.py` (61 cases) and `tests/test_mpi_curved_boundaries.py` +
`mpi_curved_worker.py` (all five geometries — quad disc, triangular disc, brick spherical octant,
tetrahedral ball, gmsh tetrahedral ball — on 2 and 4 ranks, exact to machine precision on every rank).

The acceptance criterion throughout is **geometric exactness**: after *n* uniform refinements, every node on
a curved boundary satisfies the entity's implicit equation to machine precision. Covered: per shape in 2d
and 3d; an orientation sweep of a single facet over `[0°, 360°)` in both node orders (the direct test of the
seam); a full closed loop, including facets deliberately ordered to straddle the seam; mixed forests
(pyramid-rooted, refined until tet sons of pyramid fathers exist); error-estimator-driven `adapt()` *without*
any call to `map_nodes_on_macro_elements()`; strongly non-conforming refinement (§6.1); coupled interfaces
(§6.2); partial curvature and watertightness (§2.2), sampled over the reference *interior* via
`get_macro_element_position_at_s()`; load balancing; and the regressions —
`with_macro_element=False` still straight-sided, entity lifetime no longer segfaulting, no stray `std::cout`.

Two harness notes. **The throws were unrecoverable**: `throw_runtime_error("MACRO ELEM")` fired *during*
`RefineableTElement<2>::build`, leaving a partially built son tree, and catching the `RuntimeError` was not
enough — the next teardown of that `Problem` walked the half-built tree and **aborted the interpreter**
(verified both ways: the same script survives without a `with` block, where release is deferred to
interpreter shutdown and skipped). The triangular cases therefore ran in a child process, and that isolation
is kept even though the throws are gone: it costs under a second per case and is the right shape for any
future case that can crash rather than fail. **And a test hook had to be added**:
`Element.get_macro_element_coordinate_at_s()` stops at the macro-element coordinate although its docstring
claimed it "Returns the position given by the element's macro element mapping";
`get_macro_element_position_at_s()` (which does) was added and the other docstring corrected.

One measurement worth quoting when someone asks whether refinement will fix a coarse curved boundary: it
does not. Every "before" number in this document — 7.6e-2, 7.13e-4, 1.5e-2 — is flat in the refinement
level, because subdividing a polygon gives a finer polygon.

## 11. Open

* **Moving meshes** (§7), which is a policy decision plus the two coupled constraints of §7.2.
* **`Problem.load_balance()` by hand** (§8.3), pre-existing and unrelated.
* Curved 3d edges attachable independently of faces (§6.4), if a feature edge ever needs geometry its
  faces do not carry.
