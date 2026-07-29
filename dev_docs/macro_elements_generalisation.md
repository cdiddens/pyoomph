# Generalised MacroElements: curved boundaries for every element type

Written 2026-07-29 on branch `macro_elements`, after the interface-refinement-coupling work
(`interface_refinement_coupling.md`) landed on `main`.

Status: **complete.** S0–S4, S6, S7 and gmsh-3d are done; S5 was investigated and deliberately closed without a code change (§18). Every element shape is covered, in 2d and 3d, from hand-built templates and from gmsh, serially and under MPI. §1 and §2 are measured, not assumed — every claim about current
behaviour below was reproduced, and the numbers are quoted. §3–§5 are the design. §11 is the
staging. **§13 records what S0 actually turned up**, including a defect worse than anything §1 and
§2 predicted and a measurement that materially weakens the case for part of S2 — read it before
starting S1.

Two questions drive this document:

1. How do we generalise the MacroElement treatment so that *all* pyoomph element shapes (quad, tri,
   brick, tetra, wedge, pyramid) place their refinement-generated nodes on curved boundaries?
2. The boundary coordinate is currently a single scalar (a polar angle for a circle). An angle has a
   branch cut, so a closed loop is discontinuous somewhere. Can the boundary coordinate be given an
   extra dimension — a circle point described by its normal `(nx, ny)` rather than by `atan2` — so
   that no seam exists?

The answer to (2) turns out to be the smaller of the two changes, and it partly falls out of (1).

---

## 1. What is actually implemented today

### 1.1 The intended mechanism

pyoomph describes curved geometry with `MeshTemplateCurvedEntity`
([src/meshtemplate.hpp:92](src/meshtemplate.hpp#L92)): a two-way map between a *parametric*
coordinate and a Cartesian position, plus an `apply_periodicity()` hook. Concrete entities are
`CurvedEntityCircleArc` (parameter = polar angle), `CurvedEntityCylinderArc` (angle, axial
position), `CurvedEntitySpherePart` (θ, φ) and `CurvedEntityCatmullRomSpline` (spline parameter),
and users can subclass `MeshTemplateCurvedEntity` from Python
([pyoomph/meshes/curved_entities.py:35](pyoomph/meshes/curved_entities.py#L35)).

An entity is attached to a *facet* — an edge in 2d, a face in 3d — via
`MeshTemplate::add_facet_to_boundary(name, node_indices, vertex_indices, curved_entity)`
([src/meshtemplate.cpp:2090](src/meshtemplate.cpp#L2090)). When the facet is created, each of its
vertex nodes is converted to the entity's parametric coordinate and the result is cached in
`MeshTemplateFacet::parametrics` ([src/meshtemplate.cpp:75-100](src/meshtemplate.cpp#L75-L100)).

At element-construction time, `MeshTemplate::factory_element`
([src/meshtemplate.cpp:2404-2563](src/meshtemplate.cpp#L2404-L2563)) notices that a node sits on a
curved facet, allocates an oomph-lib `MacroElement`, attaches every curved facet to it, calls
`set_macro_elem_pt()` and then `map_nodes_on_macro_element()`.

From there on the intention is that oomph-lib does the rest: `FiniteElement::get_x()` routes through
`get_x_from_macro_element()` whenever `Macro_elem_pt != 0`
([src/thirdparty/oomph-lib/include/elements.h:1885-1897](src/thirdparty/oomph-lib/include/elements.h#L1885-L1897)),
and refinement builds new nodes with `father_el_pt->get_x(t, s, x)`. So new nodes should land on the
curve.

That is the design. It is not what happens.

### 1.2 Per-family status

| shape | macro element built? | initial mesh mapped? | refinement | notes |
|---|---|---|---|---|
| quad (`BulkElementQuad2dC1/C2`) | yes, `MeshTemplateQMacroElement2` | yes | **silently FE-interpolated** | repaired only by a separate global re-snap pass, §1.4 |
| tri (`BulkElementTri2dC1/C2`) | yes, `MeshTemplateTMacroElement2` | **no-op** | **hard throw** | `throw_runtime_error("MACRO ELEM")` |
| brick (`BulkElementBrick3dC1/C2`) | yes, `MeshTemplateQMacroElement3` | yes | **silently FE-interpolated** | never exercised; debug `std::cout` still in the code |
| tetra | — | — | throw | `factory_element` throws first |
| wedge | — | — | throw | idem |
| pyramid | — | — | throw | idem |

The throw sites:

- [src/refineable_telements.cpp:743](src/refineable_telements.cpp#L743) — `throw_runtime_error("MACRO ELEM")`, 2d triangles.
- [src/refineable_telements.cpp:2453](src/refineable_telements.cpp#L2453) — tetrahedra.
- [src/elements.cpp:7506](src/elements.cpp#L7506) — pyramids.
- [src/wedges_and_pyramids.cpp:367](src/wedges_and_pyramids.cpp#L367) — wedges.
- [src/meshtemplate.cpp:2562](src/meshtemplate.cpp#L2562) — `"MacroElements not implement for this element type"` for anything that is not a quad, tri or brick.

The triangular macro element is dead code in three independent ways, any one of which would be
enough to disable it:

- `MeshTemplateDomain::macro_element_boundary` dispatches on `dynamic_cast` and has **no branch for
  `MeshTemplateTMacroElement2`** ([src/meshtemplate.cpp:337-350](src/meshtemplate.cpp#L337-L350)) —
  it returns leaving `f` untouched.
- `oomph::TMacroElement<2>::macro_map` throws `"Not implemented"`
  ([src/Tmacroelements.hpp:89-92](src/Tmacroelements.hpp#L89-L92)); the whole class is described in
  its own header comment as a "rather dummy class".
- `BulkElementBase::map_nodes_on_macro_element` has `if (telem) { /* TODO */ return; }`
  ([src/elements.cpp:2337-2342](src/elements.cpp#L2337-L2342)).

3d curved geometry is not reachable from any shipped mesh: `SphericalOctantMesh` has its curved
entity behind `if False: # TODO: This does not work yet`
([pyoomph/meshes/simplemeshes.py:647-654](pyoomph/meshes/simplemeshes.py#L647-L654)), `CylinderMesh`
carries `# TODO: Add curved entities!`
([pyoomph/meshes/simplemeshes.py:659](pyoomph/meshes/simplemeshes.py#L659)), and the gmsh reader
declares `self._curved_entities2d ... # TODO: Set those`
([pyoomph/meshes/gmsh.py:349](pyoomph/meshes/gmsh.py#L349)) and never populates it.

### 1.3 What the measurements say

**Experiment A — a quad quarter disc.** `CircularMesh(radius=1, segments=["NE"])`, Poisson with C2
fields, `max |r-1|` over all nodes on `circumference`:

```
nref 0   5 boundary nodes    max |r-1| = 0
nref 1   9 boundary nodes    max |r-1| = 5.43e-04
nref 2  17 boundary nodes    max |r-1| = 7.13e-04
nref 3  33 boundary nodes    max |r-1| = 7.13e-04
```

The error appears at the first refinement and does not shrink. Splitting by node type: at `nref 1`
the *vertex* nodes are exact and the *mid-side* nodes carry the whole error; at `nref 2` the vertex
error equals the previous generation's mid-side error. The vertices are exact only because they are
reused father mid-side nodes — nothing new is placed correctly.

The deviating node at `nref 1` sits at `(0.9795213, 0.1986242)`. Evaluating the father's quadratic
Lagrange interpolation through its three boundary nodes (0°, 22.5°, 45°) at `η = -0.5` gives
`(0.9795213, 0.1986242)` to all printed digits. With C1 fields the new node lands at the chord
midpoint of a 45° arc, `1 - cos(22.5°) = 0.0761`, and the measured error is `0.07612`. **New nodes
are placed by plain FE interpolation. The macro element is not consulted.**

**Experiment B — where the macro element does get used.** Same mesh, driven through the code paths
rather than by hand:

```
after initialise (48 elements)                 worst |r-1| = 2.2e-16
after one runtime refinement (192 elements)    worst |r-1| = 2.2e-06
after explicit map_nodes_on_macro_elements()   worst |r-1| = 2.2e-16
```

So curved quad boundaries do work — but only because
`Problem.map_nodes_on_macro_elements()`
([pyoomph/generic/problem.py:3275-3286](pyoomph/generic/problem.py#L3275-L3286)) re-snaps *every
node of every element that still has a macro element* onto `macro_map`, as a separate global pass.
That pass is invoked in the initial-adaption loop
([pyoomph/generic/problem.py:2994-3019](pyoomph/generic/problem.py#L2994-L3019)) and in remeshing
([pyoomph/generic/problem.py:6346-6358](pyoomph/generic/problem.py#L6346-L6358)) — **and nowhere in
the runtime `adapt()` path.** Error-estimator-driven adaptation during a time loop therefore leaves
new boundary nodes on the polygonal interpolant.

**Experiment C — triangles.** A disc meshed with 8 `add_tri_2d_C1` elements, each outer edge given a
`CurvedEntityCircleArc`, one uniform refinement:

```
RuntimeError: src/refineable_telements.cpp:743: MACRO ELEM
```

The same happens with a realistic mesh — `GmshTemplate.create_circle_lines(...)` +
`plane_surface(...)` in `mesh_mode="tris"`, one refinement — because the gmsh reader *does* attach
`CurvedEntityCircleArc` to 1d facets ([pyoomph/meshes/gmsh.py:844](pyoomph/meshes/gmsh.py#L844),
[:1544](pyoomph/meshes/gmsh.py#L1544)). **Any adaptively refined gmsh triangle mesh with a circular
or spline boundary dies.** The workaround is the `with_macro_element=False` argument on
`circle_arc()`/`spline()`, i.e. give up the curved boundary — the analogue of the
`RefineToLevel()@"interface"` workaround that motivated the interface-coupling work.

### 1.4 The two structural obstacles

**(i) oomph-lib's blending is Q-only.** `MacroElement::macro_map` is implemented in
`QMacroElement<2>` and `QMacroElement<3>` as a transfinite (Coons/Gordon–Hall) blend of the
reference square's or cube's four/six boundary parametrisations
([src/thirdparty/oomph-lib/include/macro_element.cc:123-204](src/thirdparty/oomph-lib/include/macro_element.cc#L123-L204),
[:440](src/thirdparty/oomph-lib/include/macro_element.cc#L440)). The construction is intrinsically
tensor-product: it names N/S/W/E and L/R/D/U/B/F, and it corrects the interior by
`(1-η)·diff_S + η·diff_N + (1-ξ)·diff_W + ξ·diff_E`. There is no meaningful specialisation of it to
a triangle, a tetrahedron, a wedge or a pyramid, which is why `TMacroElement<2>` exists only as a
stub.

The *sub-element region* bookkeeping is Q-only in the same way. `QElementBase` tracks which part of
the macro element a refined son occupies with an axis-aligned box, `s_macro_ll` / `s_macro_ur`
([src/thirdparty/oomph-lib/include/Qelements.h:186-245](src/thirdparty/oomph-lib/include/Qelements.h#L186-L245)),
and `get_x_from_macro_element` maps `s` affinely into that box
([:249-295](src/thirdparty/oomph-lib/include/Qelements.h#L249-L295)). A red-refined triangle son is
not an axis-aligned sub-box of its father, and a tetrahedral son of a pyramid is not even the same
shape as its father, so the box cannot represent them.

**(ii) The solid build discards the macro positions.** Every pyoomph quad and brick derives from
`oomph::RefineableSolidQElement<2>` / `<3>`
([src/elements.hpp:1358](src/elements.hpp#L1358), [:1427](src/elements.hpp#L1427),
[:1766](src/elements.hpp#L1766)) because pyoomph always supports moving meshes. That class's
`build()` calls `RefineableQElement<2>::build()` — which does place new nodes via the macro map —
and then loops over every node and overwrites the position with the FE value:

```cpp
// x_fe is the FE representation -- this is all we can
// work with in a solid mechanics problem. If you wish
// to reposition nodes on curvilinear boundaries of
// a domain to their exact positions on those boundaries
// you'll have to do this yourself! [...]
elastic_node_pt->x(i) = x_fe[i];
```
([src/thirdparty/oomph-lib/include/refineable_quad_element.h:402-432](src/thirdparty/oomph-lib/include/refineable_quad_element.h#L402-L432),
identically at [refineable_brick_element.h:440](src/thirdparty/oomph-lib/include/refineable_brick_element.h#L440)).

This is deliberate on oomph-lib's side and it is the reason for Experiment A. `further_build()` is
not a usable hook to undo it: it runs at
[refineable_quad_element.cc:1429](src/thirdparty/oomph-lib/include/refineable_quad_element.cc#L1429),
*inside* `RefineableQElement<2>::build`, i.e. before the overwrite. Only an override of `build()`
itself can repair it — pyoomph already overrides `build()` for bricks for an unrelated reason
([src/elements.cpp:7826](src/elements.cpp#L7826), [:7840](src/elements.cpp#L7840)), so the pattern
exists.

There is a second FE-overwrite, in `RefineableQElement<2>::quad_hang_helper`:

```cpp
// Fine adjust the coordinates (macro map will pick up boundary
// accurately but will lead to different element edges)
local_node_pt->x(0) = x_in_neighb[0];
```
([refineable_quad_element.cc:1813-1824](src/thirdparty/oomph-lib/include/refineable_quad_element.cc#L1813-L1824)).
That one is *correct* and must be preserved — see §6.1.

### 1.5 Two incidental defects found on the way

*(Both fixed in S0, along with a third and worse one that only the harness found — §13.1.)*

- **Leftover debug output.** `MeshTemplateQMacroElement3::macro_element_boundary` prints
  `STARTING FACE i` and `COMPARING PARAMS ...` unconditionally
  ([src/meshtemplate.cpp:268](src/meshtemplate.cpp#L268),
  [:308](src/meshtemplate.cpp#L308)); `CurvedEntityCylinderArc` prints on every parametric
  conversion ([src/meshtemplate.hpp:255](src/meshtemplate.hpp#L255),
  [:269](src/meshtemplate.hpp#L269)); `CurvedEntitySpherePart` prints its frame in the constructor
  and a round-trip test in `position_to_parametric`
  ([src/meshtemplate.hpp:333-335](src/meshtemplate.hpp#L333-L335),
  [:373-378](src/meshtemplate.hpp#L373-L378)). All three are on the 3d path, which is consistent
  with it never having been finished.

- **Curved entities are held as raw borrowed pointers.** `add_facet_to_boundary` stores the
  `MeshTemplateCurvedEntity*` with no ownership and no nanobind `keep_alive`. A Python user who
  writes

  ```python
  ce = _pyoomph.CurvedEntityCircleArc(centre, start, end)
  self.add_facet_to_boundary("arc", [n0, n1], [n0, n1], ce)
  ```

  inside `define_geometry` gets a **segfault** in
  `MeshTemplateQMacroElement2::macro_element_boundary` as soon as `ce` is collected. The shipped
  meshes avoid it only by accident, by keeping a list alive
  (`self._curved_entities` at [simplemeshes.py:366](pyoomph/meshes/simplemeshes.py#L366),
  `self._curved_entities1d` at [gmsh.py:348](pyoomph/meshes/gmsh.py#L348)). Reproduced; fix is a
  `nb::keep_alive<1, 5>()` on the `add_facet_to_boundary` binding
  ([src/nanobind/mesh.cpp:1504](src/nanobind/mesh.cpp#L1504)) or shared ownership on the C++ side.

---

## 2. The second problem: one scalar is not enough

### 2.1 The seam

> **Superseded in part by §13.1–13.2.** The measurements below stand, but their *cause* was a plain
> arithmetic bug (a reflection where an unwrap was meant), not the limitation this section infers
> from them — and the sweep that found it also found a silent-wrong-geometry mode this section
> misses entirely. Both are fixed; the 1d seam is no longer an argument for S2. Read §13.2 for what
> is left of the case.

`CurvedEntityCircleArc::position_to_parametric` is `atan2`
([src/meshtemplate.hpp:149-152](src/meshtemplate.hpp#L149-L152)), so the parameter lives on
`(-π, π]` and the entity has a branch cut along the negative *x* axis. The macro element blends the
two endpoint parameters linearly
([src/meshtemplate.cpp:164](src/meshtemplate.cpp#L164)); if the facet straddles the cut, the blend
of `+π-ε` and `-π+ε` is `≈ 0` — the *opposite* point of the circle.

`apply_periodicity()` exists to patch this
([src/meshtemplate.hpp:153-175](src/meshtemplate.hpp#L153-L175)), by mirroring whichever endpoint
has the larger `|angle|`. It handles the two-node case only when the endpoints are asymmetric about
the cut, and otherwise throws. Measured, with a single quad whose north edge is a circular arc:

```
arc  10..40 deg   worst |r-1| = 1.1e-16      ok
arc 170..200 deg  worst |r-1| = 1.1e-16      ok  (asymmetric about the cut)
arc 172..188 deg  RuntimeError: src/meshtemplate.hpp:172: Handle periodic case here: 0.955556  -0.955556
arc 179..181 deg  RuntimeError: src/meshtemplate.hpp:172: Handle periodic case here: 0.994444  -0.994444
```

Whether a mesh builds therefore depends on where the seam happens to fall relative to a facet.
`CurvedEntityCylinderArc::apply_periodicity` does not even try — it throws unconditionally on any
wrap ([src/meshtemplate.hpp:271-277](src/meshtemplate.hpp#L271-L277)). The Python
`CurvedEntityCircle.ensure_periodicity` applies `numpy.mod` to both endpoints
([pyoomph/meshes/curved_entities.py:51-57](pyoomph/meshes/curved_entities.py#L51-L57)), which moves
the seam rather than removing it.

`CurvedEntitySpherePart` has the same disease twice over: `(θ, φ)` with `θ = acos(z)` and
`φ = atan2(y, x)` ([src/meshtemplate.hpp:350-379](src/meshtemplate.hpp#L350-L379)) has a seam in φ
*and* a coordinate degeneracy at θ = 0 where φ is undefined. Its constructor already refuses
patches wider than 90° for related reasons.

The root cause is not the patching logic. It is that a *closed* 1-manifold has no global chart:
**any** single scalar parametrisation of a circle is discontinuous somewhere. The user's suggested
fix — describe the point by its normal instead — is the standard way out: embed the 1-manifold in
`R²` (or the 2-manifold in `R³`) where it *is* globally, smoothly coordinatised, and let the entity
project back.

### 2.2 What already supports it, and what does not

Encouragingly, the *storage* side is already dimension-agnostic.
`MeshTemplateCurvedEntity::get_parametric_dimension()` returns a per-entity `dim`
([src/meshtemplate.hpp:114](src/meshtemplate.hpp#L114)) and `MeshTemplateFacet`'s constructor sizes
each stored parametric vector by it
([src/meshtemplate.cpp:81](src/meshtemplate.cpp#L81)). Nothing forces `dim` to equal the facet's
topological dimension.

The *blend* side does not. `MeshTemplateQMacroElement2::macro_element_boundary` hardcodes

```cpp
std::vector<double> parametric(1);
```
([src/meshtemplate.cpp:160](src/meshtemplate.cpp#L160), same at
[:206](src/meshtemplate.cpp#L206)), and the 3d version hardcodes `parametric(2)`
([src/meshtemplate.cpp:281](src/meshtemplate.cpp#L281)). Verified by writing a Python
`MeshTemplateCurvedEntity` subclass with `super().__init__(2)` that stores the unit normal: its
`pos_to_parametric` is called with a length-2 array (correct), its `parametric_to_pos` is called
from the macro map with a length-1 array, and the script fails with
`IndexError: index 1 is out of bounds for axis 0 with size 1`.

So aspect (2) reduces to: **size the blend buffer from `get_parametric_dimension()`, blend every
component, and let the entity decide what "blend" means.**

---

## 3. Design A — a shape-generic macro map

The proposal is to stop using oomph-lib's `QMacroElement` blending and implement one pyoomph macro
map that works for every shape, driven by a primitive pyoomph already has for all six shapes.

### 3.1 The primitive: C1 vertex shape functions as generalised barycentrics

`BulkElementBase::shape_at_s_C1(s, psi)` is pure virtual
([src/elements.hpp:731](src/elements.hpp#L731)) and implemented by every concrete element,
including wedges and pyramids. For each shape it returns the vertex (corner-node) shape functions
`λ_v(s)`, and for each shape these have the three properties we need:

- `Σ_v λ_v(s) = 1` everywhere (partition of unity);
- `λ_v` = 1 at vertex *v*, 0 at every other vertex;
- `λ_v ≡ 0` on every facet that does not contain *v*.

For a simplex these *are* the barycentric coordinates. For a quad/brick they are the
bi/trilinear shapes, for a wedge the (triangle barycentric)×(linear) product, for a pyramid the
standard rational shapes. The third property holds in all cases. This is the unification: **treat
`shape_at_s_C1` as a generalised barycentric coordinate system and write everything in terms of
it.** `build_as_pyramid_son` already uses it exactly this way
([src/elements.cpp:7537-7541](src/elements.cpp#L7537-L7541)).

The straight-sided reference map is then just `x_lin(s) = Σ_v λ_v(s) · X_v`.

### 3.2 Blended deviation — which is exactly the Coons / Gordon–Hall blend

For a sub-entity *S* (a facet or an edge) with vertex set `V(S)`, define

- weight `w_S(s) = Σ_{v∈V(S)} λ_v(s)`;
- restriction `σ_S,v(s) = λ_v(s) / w_S(s)` for `v ∈ V(S)` — a partition of unity *on S*, and by the
  third property above it is the correct facet-local coordinate whenever *s* lies on *S*;
- straight image `L_S(σ) = Σ_{v∈V(S)} σ_v X_v`;
- curved image `C_S(σ)` — the entity evaluated at the σ-blend of *S*'s stored parametrics (§4);
- deviation `d_S = C_S − L_S`, which vanishes at every vertex of *S* by construction.

The macro map is

> **x(s) = Σ_v λ_v X_v  +  Σ_{F curved} w_F · d_F(σ_F)  −  Σ_E (m_E − 1) · w_E · d_E(σ_E)**

where the last sum runs over edges *E* contained in `m_E ≥ 2` curved facets (3d only; empty in 2d,
because two edges of a 2d element meet only at a vertex, where `d` already vanishes).

**This is not a new or weaker blend — it is the classical transfinite interpolant, rewritten so
that the shape does not appear.** The interior is genuinely curved, exactly as a Coons patch is.

*Quad.* Write `ξ = (s₀+1)/2`, `η = (s₁+1)/2`, so `w_S = 1−η`, `w_N = η`, `w_W = 1−ξ`, `w_E = ξ`, and
`Σ_v λ_v X_v = f_rect` (the bilinear corner interpolant). The bilinear interpolant is linear between
its own edge restrictions in either direction, so `(1−η)L_S + ηL_N = f_rect = (1−ξ)L_W + ξL_E`.
Substituting,

```
x = f_rect + Σ_E w_E (C_E − L_E)
  = f_rect + [(1−η)C_S + ηC_N + (1−ξ)C_W + ξC_E] − 2·f_rect
  = (1−η)C_S + ηC_N + (1−ξ)C_W + ξC_E − f_rect
```

which is line for line
[`QMacroElement<2>::macro_map`](src/thirdparty/oomph-lib/include/macro_element.cc#L200-L203) —
the Boolean sum `P₁ ⊕ P₂ = P₁ + P₂ − P₁P₂`. **Existing curved quad meshes do not move at all**, in
the interior or on the boundary.

*Brick, all six faces curved.* The same telescoping applies twice. `Σ_F w_F L_F = 3L` (each of the
three opposite-face pairs blends back to the trilinear interpolant `L`) and `Σ_E w_E L_E = 3L` (each
of the three groups of four parallel edges does the same), and every edge has `m_E = 2`, so

```
x = L + Σ_F w_F C_F − Σ_E w_E C_E
```

which is `ΣPᵢ − ΣPᵢPⱼ + P₁P₂P₃`, i.e. exactly
[`QMacroElement<3>::macro_map`](src/thirdparty/oomph-lib/include/macro_element.cc#L440).

*Triangle.* Each vertex lies on two edges, so `Σ_E w_E L_E = 2L` and the formula collapses to
`x = Σ_E w_E C_E(σ_E) − L` — the same Boolean-sum shape as the quad, with three terms instead of
four. *Tetrahedron:* three faces and three edges meet at each vertex, giving the brick's closed form
`x = L + Σ_F w_F C_F − Σ_E w_E C_E`. Wedges and pyramids need no separate derivation: they are
covered by the general `(m_E − 1)` statement, which is what the implementation evaluates. The closed
forms are stated only because they are easier to check by hand.

Correctness, independent of the equivalences:

- **On a curved facet F** (`w_F = 1`, `σ_F` = the actual facet coordinate) the first two terms give
  exactly `C_F`. Every *other* curved facet `F'` contributes at a point where `σ_{F'}` has collapsed
  onto `F ∩ F'`, so `d_{F'}` is that shared edge's deviation — and the correction term subtracts
  exactly that. Net `x = C_F`.
- **At any vertex** all deviations vanish, so `x = X_v`. Vertices never move.

### 3.2.1 Partially curved 3d elements — where this is strictly better

A brick with one curved face *B* and five interior faces is the common 3d case, and it is the one
where the existing 3d path is wrong. `MeshTemplateQMacroElement3::macro_element_boundary` returns
the **flat** bilinear corner interpolation for a face with no entity attached
([src/meshtemplate.cpp:272-278](src/meshtemplate.cpp#L272-L278)). But the four edges of *B* lie on
the curved surface, so a side face *W* meeting *B* along edge `E` claims `E` is straight while *B*
claims it is curved. Gordon–Hall then blends two contradictory boundary descriptions and *B* is no
longer reproduced exactly. (In 2d this cannot happen: an edge's boundary is two corner nodes, on
which everybody agrees. That is why the 2d path has never shown the symptom.)

The deviation form has no such failure mode, because an uncurved facet contributes nothing and the
curved facet's deviation is what propagates. On the side face *W* one gets

```
x|_W = L_W + w_B|_W · d_E(σ_E)
```

i.e. *W* becomes the **ruled surface** between the curved edge `E` and the opposite straight edge —
which is the geometrically right answer, and the weight `w_B|_W` is the linear function on *W* that
is 1 on `E` and 0 on the opposite edge. That expression depends only on *W*'s vertices and on `E`,
so the neighbouring element across *W* computes the identical surface: the mapped geometry stays
watertight without any extra machinery.

### 3.2.2 Edge geometry must be a mesh-level fact, not an element-level one

The watertightness argument above has one precondition: both elements sharing *W* must know that `E`
is curved. Element *A* knows because it owns a curved face containing `E`. Element *B* usually knows
for the same reason — but not necessarily: *B* can touch the curved surface along `E` alone, without
having a face on it. Then *A* would rule *W* and *B* would leave it flat, and the two would disagree.

Facets are already deduplicated mesh-wide in `MeshTemplate::facetmap`
([src/meshtemplate.cpp:2082](src/meshtemplate.cpp#L2082)); edges are not. The fix is small and
should be part of S3: build an **edge → curved entity registry** at template level, derived once
from the curved facets (every edge of every curved facet inherits that facet's entity and the
parametrics of its two endpoints), and have `GenericMacroElement` take `C_E` from that registry
rather than from whichever facet of *this* element happens to carry it. This also gives the natural
home for a future explicitly-attached 3d feature edge (§7), and it is where an inconsistency check
belongs: if two different entities claim the same edge, warn with both names rather than silently
picking one.

In 2d the registry is unnecessary — edges *are* the facets and are already deduplicated.

### 3.3 Son-region tracking, generalised

Replace `s_macro_ll` / `s_macro_ur` (an axis-aligned box) with the son's vertex coordinates in the
*root macro element's* reference domain:

```cpp
std::vector<std::vector<double>> S_macro_vertex;   // nvertex x dim, on BulkElementBase
```

Propagation father → son is one line of the same primitive:

```
S_macro_vertex_son[k] = Σ_v ψ^{C1,father}(sv_k) · S_macro_vertex_father[v]
```

where `sv_k` is son vertex *k* in father-local coordinates. That vector already exists for every
shape: `RefineableTElement<2>/<3>::son_vertices_in_father`
([src/refineable_telements.hpp:435](src/refineable_telements.hpp#L435),
[src/refineable_telements.cpp:2768](src/refineable_telements.cpp#L2768)),
`RefineableWedgeElement::son_vertices_in_father`
([src/wedges_and_pyramids.cpp:211](src/wedges_and_pyramids.cpp#L211)),
`RefineablePyramidElement::son_vertices_in_father`
([src/wedges_and_pyramids.cpp:615](src/wedges_and_pyramids.cpp#L615)), and for quads/bricks the
`s_lo`/`s_hi` box that oomph already computes.

Then

```
get_x_from_macro_element(t, s, x):
    s_macro = Σ_k ψ^{C1,this}(s) · S_macro_vertex[k]
    Macro_elem_pt->macro_map(t, s_macro, x)
```

This reduces *exactly* to oomph's box formula for quads and bricks (the multilinear shape over a
sub-box is the affine box map), and it is the only formulation that survives a **mixed forest**,
where a tetrahedral son has a pyramidal father — the father's C1 shape is evaluated at the son's
vertex coordinates, and the differing vertex counts never appear in the same expression. The
pyramid apex singularity needs the same guard `build_as_pyramid_son` already applies
([src/elements.cpp:7531-7534](src/elements.cpp#L7531-L7534)).

### 3.4 Class layout

New `src/macroelements.{hpp,cpp}`; `src/Tmacroelements.hpp` is deleted.

```cpp
namespace pyoomph {

  // One curved facet of one macro element.
  struct MacroCurvedFacet {
    MeshTemplateCurvedEntity *entity;           // borrowed (see §1.5)
    std::vector<unsigned>     local_vertices;   // element-local C1 vertex indices lying on the facet
    std::vector<unsigned>     parametric_index; // parallel: index into MeshTemplateFacet::parametrics
    const std::vector<std::vector<double>> *parametrics;
  };

  class GenericMacroElement : public oomph::MacroElement {
    ElementGeometryKind             kind;             // quad/tri/brick/tet/wedge/pyramid
    std::vector<oomph::Node *>      vertex_nodes;     // in C1 shape order, positions read live per t
    std::vector<MacroCurvedFacet>   curved_facets;
    std::vector<MacroEdgeCorrection> edge_corrections; // 3d only, built once at construction
  public:
    void macro_map(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &r) override;
    // assemble_macro_to_eulerian_jacobian{,2}: central finite differences of macro_map
    // output / output_macro_element_boundaries: generic sampling of macro_map
  };
}
```

Deliberate consequences:

- **`find_permutation` disappears.** It exists only to reconcile the macro element's canonical facet
  node order with the facet's own order, and the brick version brute-forces all 4! permutations
  ([src/meshtemplate.cpp:227-256](src/meshtemplate.cpp#L227-L256)). In the new scheme each weight is
  carried together with the vertex it belongs to, so `parametric_index[i]` is a direct node-identity
  lookup and ordering ambiguity cannot arise. This also removes the only place that would have
  needed a 3!-permutation variant for triangular faces of tets and wedges.
- **`MeshTemplateQMacroElement2` / `MeshTemplateTMacroElement2` / `MeshTemplateQMacroElement3` and
  `oomph::TMacroElement<DIM>` all collapse into `GenericMacroElement`.** The four near-duplicate
  bodies in `factory_element` ([src/meshtemplate.cpp:2408-2562](src/meshtemplate.cpp#L2408-L2562))
  collapse into one loop.
- **`MeshTemplateDomain::macro_element_boundary` and its `dynamic_cast` chain disappear** — the
  macro element evaluates itself, no round-trip through `Domain` is needed. `MeshTemplateDomain`
  survives only because `oomph::MacroElement`'s constructor demands a `Domain*`.
- `kind` rather than a back-pointer to the root element: evaluating `λ_v(s)` needs only the shape
  kind, so a free function `c1_shape(kind, s, psi)` avoids a lifetime coupling between the macro
  element and an element object.

### 3.5 Where it plugs in

1. `BulkElementBase::get_x_from_macro_element(t, s, x)` — the generic §3.3 implementation. Note the
   inheritance hazard: `QElementBase` also overrides this virtual, so for quad/brick classes that
   inherit both `BulkElementBase` and `QElement<..>` virtually, the most-derived class must
   disambiguate. The safe form is a non-virtual
   `BulkElementBase::get_x_from_generic_macro_element(...)` plus a two-line `override` in each
   concrete `BulkElement*` class — consistent with how those classes are already written.
2. `BulkElementBase::map_nodes_on_macro_element` ([src/elements.cpp:2310](src/elements.cpp#L2310)) —
   drop the `qelem`/`telem` split, use `local_coordinate_of_node` + §3.3 for every shape. Also add
   the hanging-node guard of §6.1.
3. `BulkElementQuad2dC1/C2::build` and `BulkElementBrick3dC1/C2::build` — after calling
   `RefineableSolidQElement<N>::build`, re-apply the macro position to the nodes that were created
   by this build (§5 governs when).
4. Delete the four throws listed in §1.2. The T-family, wedge and pyramid builders already place new
   nodes through `father_el_pt->get_x(t, s, x)`
   ([src/refineable_telements.cpp:1028](src/refineable_telements.cpp#L1028),
   [src/elements.cpp:7648](src/elements.cpp#L7648), [:7797](src/elements.cpp#L7797)), so once
   `get_x_from_macro_element` works for them they need **no** post-build repair — they are not
   `RefineableSolidQElement`s and nothing overwrites the result. The Q-family is the awkward one,
   not the simplex family.
5. `factory_element` — one loop over `el->nfacets()` for all shapes. `construct_facet` is already
   implemented for line, quad, tri, brick, tetra, wedge and pyramid in both C1 and C2
   ([src/meshtemplate.cpp:388-1300](src/meshtemplate.cpp#L388-L1300)), so no new facet topology is
   needed.

---

## 4. Design B — the boundary coordinate as an opaque vector

### 4.1 Separate "how many numbers" from "how many dimensions"

`MeshTemplateCurvedEntity` gains a clear split:

```cpp
virtual unsigned get_parametric_dimension() const;  // how many doubles are stored  (existing `dim`)
virtual unsigned get_intrinsic_dimension()  const;  // manifold dimension: 1 curve, 2 surface
```

`get_parametric_dimension() >= get_intrinsic_dimension()` is now allowed and is the normal case for
closed geometry. Nothing in the storage path changes (§2.2); the macro map sizes its buffer from
`get_parametric_dimension()` instead of hardcoding 1 or 2.

### 4.2 The entity owns the blend

Componentwise linear interpolation is only correct for a flat, non-redundant chart. Add

```cpp
// Combine the stored parametric coordinates of a facet's vertices with the given (partition-of-unity)
// weights. Default: the weighted sum, i.e. exactly today's behaviour.
virtual void blend_parametric(const std::vector<double> &weights,
                              const std::vector<const std::vector<double>*> &params,
                              std::vector<double> &result);

// Convenience: blend then map. Default = parametric_to_position(blend_parametric(...)).
virtual void facet_map(const unsigned &t,
                       const std::vector<double> &weights,
                       const std::vector<const std::vector<double>*> &params,
                       std::vector<double> &position);
```

`blend_parametric` takes arbitrary weights (not just the corners), because refinement evaluates at
arbitrary positions along a facet. The default implementation reproduces the current arithmetic
exactly, so every existing entity — including Python subclasses — keeps working unchanged.

`apply_periodicity()` becomes optional and, for the rewritten entities, unnecessary. Keep it for
backwards compatibility with user subclasses; deprecate it in the docs.

### 4.3 The rewritten entities

| entity | parametric coordinate | intrinsic dim | blend | seam? |
|---|---|---|---|---|
| `CurvedEntityCircleArc` | unit normal `(nx, ny)` | 1 | slerp for 2 points, normalised sum otherwise | none |
| `CurvedEntitySpherePart` | unit normal `(nx, ny, nz)` | 2 | normalised weighted sum | none |
| `CurvedEntityCylinderArc` | `(nx, ny, nz, z_axial)` | 2 | normalise the radial part, linear in `z` | none |
| `CurvedEntityCatmullRomSpline` | scalar `t` (+ optional period) | 1 | period-aware unwrap, else linear | only if closed |

For the circle: `position_to_parametric(x) = (x − c)/|x − c|`, `parametric_to_position(n) = c + R·n̂`.
Both are total functions on `R² \ {c}` — no branch, no case distinction, and `apply_periodicity`
becomes a no-op. At weight ½ the normalised sum of two unit normals is exactly the arc bisector,
which is the case refinement actually needs; slerp additionally makes the parameter distribution
uniform in arclength at every other weight, and is cheap for the two-point case that 1d facets
always are.

The one case the normal representation cannot express is a facet spanning exactly 180° (antipodal
normals sum to zero) or more. That must be detected at facet-construction time and rejected with a
clear message. It is a strictly milder restriction than today's — the current failure mode triggers
at 16° (measured, §2.1) purely because of where the arc sits relative to the seam.

For the sphere the gain is larger: the normal representation removes the pole degeneracy *and* the
seam *and* the tangent/cotangent frame construction, and it lifts the "less than 90° opening angle"
restriction that `CurvedEntitySpherePart`'s constructor currently enforces
([src/meshtemplate.hpp:309-312](src/meshtemplate.hpp#L309-L312)) up to the same
less-than-a-hemisphere-per-facet limit.

The proof of concept in §2.2 shows this works from Python today apart from the hardcoded buffer
size — which is the whole change.

### 4.4 An optional projection hook

For quadrics, "blend the normals and normalise" is identical to "take the straight-sided point and
project onto the surface". Projection needs no parametrisation at all, so it is worth exposing:

```cpp
// Move x onto the entity. Return false if the entity has no cheap projection.
virtual bool project_onto(const unsigned &t, std::vector<double> &x) { return false; }
```

`CurvedEntityCatmullRomSpline` can implement it directly — its `position_to_parametric` already does
a sampled closest-point search
([src/meshtemplate.hpp:391-411](src/meshtemplate.hpp#L391-L411)). Projection is **not** proposed as
the primary mechanism, because it ignores the facet's endpoints: on a closed or folded curve the
chord midpoint can project onto the wrong branch. The recommended composition is *blend first, then
optionally project* — the blend guarantees the right branch, the projection cleans up entities whose
parametrisation is only approximately arclength.

### 4.5 Backwards compatibility

- Existing C++ entities: unchanged signatures, default `blend_parametric` = today's arithmetic.
- Existing Python subclasses (`pos_to_parametric` / `parametric_to_pos` / `ensure_periodicity`):
  unchanged; they simply keep `parametric_dimension == intrinsic_dimension`.
- Serialisation: `get_information_string` / `load_from_strings` must carry the new representation.
  Note `load_from_strings` is currently `throw_runtime_error("IMPLEM")`
  ([src/meshtemplate.cpp:2571](src/meshtemplate.cpp#L2571)) while `GmshTemplate.write_curved_entities`
  writes the file ([pyoomph/meshes/gmsh.py:1616](pyoomph/meshes/gmsh.py#L1616)) — i.e. the write side
  exists and the read side does not. Changing the representation is therefore free on the read side
  and needs a version tag on the write side.

---

## 5. Moving meshes: what the macro element should drive

This is the one place where the design needs a decision rather than a derivation, and it is why
oomph-lib's solid build discards macro positions in the first place.

When a domain has `functable->moving_nodes` (i.e. `coordinates_as_dofs`, see
[src/codegen.cpp:7790](src/codegen.cpp#L7790) and `Problem::has_moving_nodes`
[src/problem.hpp:158](src/problem.hpp#L158)), the Eulerian position **is an unknown**. Forcing a
newly created node onto the curve is then wrong in general: the mesh has already moved away from the
template geometry, and the curve describes where the boundary *was*, not where it *is*.

pyoomph already encodes this distinction, in Python:
`Problem.remove_macro_elements_after_initial_adaption = "auto"` drops the macro elements after the
initial adaption **iff** the coordinates are free
([pyoomph/generic/problem.py:554](pyoomph/generic/problem.py#L554),
[:3288-3299](pyoomph/generic/problem.py#L3288-L3299)), and `GmshTemplate.circle_arc`'s docstring
already says "on moving meshes only the initial refinement". The proposed rule keeps that semantics
and makes it explicit:

- **Static mesh** (`moving_nodes == false`): the macro map defines `x`. New nodes are snapped at
  creation, for every shape, in the runtime `adapt()` path as well as the initial one. This is the
  behaviour users expect and currently do not get (§1.3, Experiment B).
- **Moving mesh** (`moving_nodes == true`): the macro map defines the **Lagrangian** coordinate `ξ`
  via `Undeformed_macro_elem_pt` and
  `enable_use_of_undeformed_macro_element_for_new_lagrangian_coords()`
  ([src/thirdparty/oomph-lib/include/refineable_elements.h:990](src/thirdparty/oomph-lib/include/refineable_elements.h#L990)).
  `x` is snapped only while the mesh is still in its initial configuration, exactly as today. The
  `set_undeformed_macro_element` binding already exists
  ([src/nanobind/mesh.cpp:916](src/nanobind/mesh.cpp#L916)) and is currently unused from Python.

This matters for the simplex family in particular: the refined-triangle Lagrangian-coordinate defect
recorded in the project notes (refined tri/tet nodes lacking interpolated `ξ`) is the same corner of
the code, and the undeformed-macro-element route is the principled fix for curved moving meshes.

**Open question for the user:** should a curved static boundary continue to be re-snapped globally
by `map_nodes_on_macro_elements()` after the change, or should that pass become a no-op / diagnostic
once nodes are placed correctly at creation? Keeping it is harmless and is a good invariant check;
removing it is cleaner. The plan below keeps it and adds an assertion that it moves nothing.

---

## 6. Interaction with existing machinery

### 6.1 Hanging nodes

A hanging node's position is determined by its masters. If a hanging node on a curved boundary were
snapped to the curve, the two sides of the edge would no longer agree and `check_integrity` would
fire. oomph-lib gets this right for quads by overwriting hanging-node positions with the coarse
neighbour's `interpolated_x`
([refineable_quad_element.cc:1813-1824](src/thirdparty/oomph-lib/include/refineable_quad_element.cc#L1813-L1824)),
and that overwrite must survive this work.

The corollary is a guard that does **not** exist today: `map_nodes_on_macro_element`
([src/elements.cpp:2320-2333](src/elements.cpp#L2320-L2333)) loops over *all* `nnode()` and snaps
unconditionally, hanging or not. On a uniformly refined mesh there are no hanging nodes, which is
why this has not bitten — every measurement in §1.3 was uniform. On an error-estimator-refined
curved mesh it should. **Marked as suspected, not confirmed**: the attempt to produce a curved mesh
with a hanging boundary node did not refine non-uniformly, so this is a test to write (§9, T7), not
a defect to claim.

The generic hanging-node engine from the `mixed_adapt` work is the right place to ask "is this node
hanging" for the simplex and mixed families.

### 6.2 MPI

Macro elements are per-element and hold node pointers plus borrowed entity pointers. Under
`--distribute` a halo element must reach the same position for a shared node as its non-halo
counterpart, which requires the macro element (or an equivalent) to exist on both ranks.
`missing_masters.hpp` already contains oomph's machinery for reconstructing
`MacroElementNodeUpdate` elements across ranks
([src/missing_masters.hpp:709-760](src/missing_masters.hpp#L709-L760)) and it warns loudly when it
has to. Two options, to be decided at S6:

- construct the macro elements on every rank from the (replicated) `MeshTemplate`, which is how the
  initial mesh is already built on all ranks; or
- accept FE placement on halo elements and rely on the subsequent halo synchronisation to copy the
  non-halo position.

The second is much cheaper and is probably sufficient, but it must be verified against the
`mixed_adapt_validation` MPI harness, which is exactly the kind of drift that campaign was built to
catch.

### 6.3 Remeshing and interpolation

`pyoomph/meshes/interpolator.py` strips macro elements before `locate_zeta`
("really troublesome", [interpolator.py:79-84](pyoomph/meshes/interpolator.py#L79-L84)) because
`locate_zeta` inverts `get_x`, and inverting the macro map is expensive and non-robust. That stays;
the generic macro map does not make it worse. `Remesher.use_macro_elements`
([pyoomph/meshes/remesher.py:121](pyoomph/meshes/remesher.py#L121)) carries the template's flag
through remeshing and needs no change.

### 6.4 Interface refinement coupling

Two coupled domains sharing a curved interface must place the shared interface nodes identically.
Since both sides build their own macro elements from the same `MeshTemplate` facets and the same
entity object, and the placement depends only on the facet's vertices and the entity, agreement is
automatic — *provided* both sides attach the curved entity to the interface facet. The conformity
invariant of `interface_refinement_coupling.md` §2 guarantees the facets themselves match. Worth an
explicit test (§9, T8) rather than an assumption.

### 6.5 Mixed forests

`RefineablePyramidElement::Mixed_forest_active` routes brick/wedge/pyramid/tet builds through
pyoomph's own son builders ([src/elements.cpp:7828](src/elements.cpp#L7828)). Those already call
`father_el_pt->get_x` and already carry the vertex-shape machinery, so they inherit curved-boundary
support from §3.3 with no additional work beyond removing the throws — the shape-agnostic
formulation was chosen precisely so that a tet son of a pyramid father is not a special case.

---

## 7. What this does *not* attempt

- **Curved interior elements.** Only facets carrying an entity are curved; the interior deviation
  decays linearly. No attempt at high-order (P2/P3) geometry representation.
- **Curved 3d *edges* independent of faces.** An entity can be attached to a facet only. A 3d
  feature edge shared by two curved faces takes its geometry from those faces. Attaching a 1d entity
  to a 3d edge is a plausible later extension; the `MacroEdgeCorrection` structure of §3.4 is where
  it would go.
- **Inverting the macro map.** `locate_zeta` continues to strip macro elements (§6.3).
- **Making `map_nodes_on_macro_elements` unnecessary.** It is kept as an invariant check (§5).

---

## 8. Risks

1. **The virtual-inheritance clash on `get_x_from_macro_element`** (§3.5, item 1). `BulkElementBase`
   and `QElementBase` both want to override the same `FiniteElement` virtual through virtual
   inheritance. Mitigation is mechanical (per-class `override` forwarding), but it touches every
   quad/brick class and will produce confusing compiler errors if approached the other way.
2. ~~The interior deviation differs from oomph's Coons blend.~~ **Retracted** — §3.2 proves the
   deviation form *is* the Coons blend for quads and Gordon–Hall for bricks, so existing curved
   quad meshes do not move by even a rounding error and no second code path is needed. What remains
   is the narrower risk that the equivalence is asserted rather than executed: S1 must verify it
   numerically (§10.1) before the old `QMacroElement` path is deleted, not after. The one place the
   two genuinely differ is partially-curved 3d bricks, where the old path is wrong (§3.2.1) — a 3d
   path nothing currently reaches (§1.2), so there is nothing to regress.
3. **Snapping in the runtime `adapt()` path is a behaviour change** for every existing static curved
   mesh — positions that used to drift now do not. That is the point of the work, but it will move
   results.
4. **Pyramid shape functions are rational** and singular at the apex. The `macro_map` evaluates
   `λ_v` at arbitrary `s`, so the apex guard must be applied there too, not only in the son builder.
5. **Facets wider than a hemisphere** are rejected by the normal representation (§4.3). A user with a
   very coarse curved mesh may hit a new error where they previously got a silently wrong mesh.
   The error message must say what to do (split the facet).
6. **MPI** (§6.2) is the least-explored area and the one where the `mixed_adapt` campaign's history
   says surprises live.

---

## 9. Test plan

Geometric exactness is the acceptance criterion throughout: after *n* uniform refinements, every
node that lies on a curved boundary must satisfy the entity's implicit equation to machine
precision. All of these are cheap and belong in `tests/`.

- **T1 — 2d, per shape.** Quarter/full disc as quads and as triangles, C1 and C2 fields, 0–4 uniform
  refinements: `max |r − R| < 1e-14`. Today: quads 7.1e-4, triangles throw.
- **T2 — seam sweep.** A single curved facet spanning `[a, a+Δ]` for `a` on a 5° grid over
  `[0°, 360°)` and `Δ ∈ {10°, 45°, 90°, 170°}`: build and refine must succeed and be exact for every
  `a`. Today: fails for `a` near the ±π seam (§2.1). This is the direct test of aspect (2).
- **T3 — full closed loop.** A disc whose entire circumference is one closed entity, refined; plus
  the same with the facets deliberately ordered so that one straddles the seam.
- **T4 — 3d, per shape.** Spherical octant as bricks, as tets, and mixed (wedge/pyramid), against a
  `CurvedEntitySpherePart`; cylinder mantle against `CurvedEntityCylinderArc`, including a facet
  through the φ seam and one containing the pole.
- **T5 — mixed forest.** A pyramid-rooted forest with a curved boundary face, refined until tet sons
  of pyramid fathers exist (the `test_pyramid_refinement.py` harness already builds these).
- **T6 — runtime adapt.** Error-estimator-driven `adapt()` on a curved mesh, *without* any call to
  `map_nodes_on_macro_elements()`: boundary nodes must be exact. Today: 2.2e-6 (§1.3, Experiment B).
- **T7 — hanging nodes on a curved boundary.** Non-uniform refinement so that a boundary edge carries
  a hanging node; `check_integrity()` must pass and the hanging node must sit on its masters'
  interpolant, *not* on the curve (§6.1).
- **T8 — coupled interface.** Two domains sharing a curved interface, adaptively refined, exercising
  `InterfaceRefinementCoupler`: shared node positions must agree exactly.
- **T9 — MPI.** T1/T4/T6 under `--distribute` on 2 and 4 ranks; node positions must be identical to
  serial.
- **T10 — moving mesh.** A curved static boundary with `coordinates_as_dofs`: `ξ` follows the
  undeformed macro element, `x` is not forced after the first solve (§5).
- **T11 — regressions.** `with_macro_element=False` still produces a straight-sided mesh; entity
  lifetime (§1.5) no longer segfaults when the Python object is dropped; no stray `std::cout`.
- **T12 — the tutorials.** The full 126-script pipeline, once, at the end — not per stage.
- **T13 — partially curved 3d and watertightness.** A brick and a tet with exactly one curved face,
  plus a neighbour that touches the curved surface only along an edge (§3.2.2): the curved face must
  be reproduced exactly, the adjacent interior faces must be ruled, and the two elements sharing an
  interior face must agree on it to machine precision. Also the equivalence check of §10.1, sampled
  over the *interior* of the reference domain.

---

## 10. What to measure before writing code — **done, results below**

### 10.1 The equivalence of §3.2, executed

A reference implementation of the deviation form (generalised barycentrics, in Python) was compared
against oomph's actual `QMacroElement<2>::macro_map` on a 41×41 grid over the **whole** reference
square, sampled through the new `Element.get_macro_element_position_at_s()`:

```
arc      curved facets   max |coons - deviation|   interior bending
0..45          1                4.0e-16                7.6e-02
170..200       1                1.1e-15                3.4e-02
150..210       1                9.0e-16                1.3e-01
355..25        1                4.4e-16                3.4e-02
0..45          2                4.0e-16                5.0e-01
170..200       2                1.1e-15                5.0e-01
```

"Interior bending" is how far the macro map departs from the straight-sided bilinear map, i.e. how
much of the reference domain's interior is actually curved. It is three to five orders of magnitude
larger than the discrepancy, so the agreement is a real statement about two different curved maps,
not two flat ones agreeing trivially.

**Conclusion: risk 2 is retired.** One code path, no interior diffs, and this comparison becomes
unit test T13 rather than a one-off measurement.

### 10.2 slerp vs normalised-sum, measured

If a circle is parametrised by its unit normal (§4.3), both blends put the node *exactly* on the
circle — the boundary is exact either way. They differ only in where along the arc it lands, so the
error is tangential (uneven spacing) rather than normal (wrong geometry). Worst case over all
weights, on a unit circle:

```
facet width   tangential shift   as % of edge length   as % of the sagitta
   15 deg        2.9e-04                0.11%                 3.4%
   30 deg        2.3e-03                0.45%                 6.9%
   45 deg        8.0e-03                1.05%                10.5%
   90 deg        7.1e-02                5.03%                24.3%
  120 deg        1.9e-01               11.03%                38.2%
```

At weight ½ — the bisector, which is what edge refinement of a facet actually asks for — the two
agree exactly; the numbers above come from the off-centre weights that deeper tree levels produce.

**Conclusion: keep the §4.3 plan as written.** Up to ~45° facets normalised-sum costs about 1% of
an element edge, which is tolerable, but exact slerp for the two-point case is three lines and 1d
facets are *always* two-point, so there is no reason to accept even that. Normalised-sum stays for
the three- and four-point facets of a 3d face, where no closed-form slerp exists and where §3.2.1's
ruled-surface behaviour dominates the error budget anyway.

---

## 11. Staging

Each stage ends with its tests green and is committed separately.

- **S0 — harness and honesty.** ✅ **Done** (`tests/test_curved_boundaries.py`, 31 passed /
  6 xfailed; full fast suite 447 passed). T1/T2/T6 recorded as strict-xfail against the measured
  present-day numbers; §1.5's two incidental defects fixed, plus a third and worse one found while
  writing the harness; both §10 measurements run. **See §13 — it changes S2.**
- **S1 — the generic 2d macro map.** ✅ **Done**, in two commits: S1a the refactor (one
  `GenericMacroElement`, `Tmacroelements.hpp` and `find_permutation` and the Domain dispatch all
  deleted), S1b the behaviour (`Macro_element_vertex_s`, generic `map_nodes_on_macro_element`, the
  post-build re-snap for quads and bricks, the `"MACRO ELEM"` throw removed). All six strict xfails
  flipped to XPASS and the markers came off; curved module 39 passed, fast suite 455 passed.
  **The gmsh-triangle failure of §1.3 is fixed: that mesh now refines with a boundary error of
  exactly 0.0 where it previously died.** §14 records what S1 turned up.
- **S2 — the parametric coordinate.** ✅ **Done**, though not as written: buffer sizing landed in S1a
  and the normal-based sphere in S3a, `blend_parametric` and `get_intrinsic_dimension` landed here,
  and rewriting `CurvedEntityCircleArc` on normals was **deliberately not done** — §20.2 explains why
  it would have made that entity worse. `facet_map` and `project_onto` were dropped as speculative.
- **S3 — 3d Q and simplex.** ✅ **Done**, in three commits: S3a the normal-parametrised sphere plus
  `SphericalOctantMesh`'s shell (bricks), S3b tetrahedra, S3c the shared-edge inclusion–exclusion.
  T4 and T13 green; curved module 55 passed, fast suite 471 passed. **Not** done here, and moved to
  their own stage: populating `GmshTemplate._curved_entities2d`, and the edge→entity registry of
  §3.2.2 — see §15.3 for why the registry turned out not to be needed yet.
- **S4 — wedge, pyramid, mixed forests.** ✅ **Done.** The last four throws removed (wedge, pyramid,
  mixed brick, and the tet-of-pyramid path). T5 green; curved module 61 passed, fast suite 477.
- **S5 — moving meshes.** ⏹ **Investigated, closed without a code change** — the mechanism §5 named
  turned out not to be the one pyoomph uses, and the behaviour change it implied was declined in
  favour of keeping ALE semantics stable. §18 records the measurements and the decision. T10 is
  therefore not written: there is no new behaviour to pin.
- **S6 — MPI.** ✅ **Done**, and it found two defects that only this combination could reach (§19).
  `tests/test_mpi_curved_boundaries.py` + `mpi_curved_worker.py`, all five geometries on 2 and 4
  ranks. T8 (coupled interfaces with curved boundaries) is *not* done — see §19.4.
- **S7 — docs and tutorials.** ✅ **Done** (§21). Two new sections in
  `docs/source/tutorial/spatial/mesh/unitsandmacro.rst`; `create_curved_entity` extended to spheres
  and cylinders; `CylinderMesh` given its curved mantle. No tutorial carried a
  `with_macro_element=False` workaround, so there was none to remove.

---

## 12. Summary of the two answers

**Aspect 1 — how to generalise to all shapes.** Stop specialising the macro element per shape and
specialise the *coordinate system* instead. `shape_at_s_C1` gives every one of the six shapes a
partition of unity whose members vanish on non-adjacent facets; that is all a transfinite blend
needs. Written in those coordinates, the Coons/Gordon–Hall Boolean sum stops mentioning N/S/W/E or
L/R/D/U/B/F and becomes a single formula (§3.2) valid for quad, tri, brick, tet, wedge and pyramid
alike — provably identical to oomph's blend where oomph has one, so the interior is curved exactly
as before and no existing mesh moves. The son-region box becomes one formula (§3.3), six classes
become one, and `find_permutation` becomes unnecessary. The genuinely shape-specific part — "where
is son *k*'s vertex in father coordinates" — already exists for every shape from the
mixed-adaptivity work. As a by-product the partially-curved 3d case, which the present blend gets
wrong (§3.2.1), comes out right.

**Aspect 2 — the closed-loop discontinuity.** Yes, and the storage layer already permits it: make
the parametric coordinate an opaque vector whose length is the entity's business, let the entity own
the blending rule, and represent closed geometry by its normal. A circle becomes `(nx, ny)` with
normalised blending, a sphere `(nx, ny, nz)`, and the branch cut — together with
`apply_periodicity`, its two `throw_runtime_error("Handle periodic case here")` sites and the
sphere's 90°-opening-angle restriction — ceases to exist. The only code change required for the
mechanism itself is replacing two hardcoded buffer sizes
([src/meshtemplate.cpp:160](src/meshtemplate.cpp#L160),
[:281](src/meshtemplate.cpp#L281)) with `get_parametric_dimension()`.

*(§13.2 and §13.3 qualify the second paragraph: the 1d seam turned out to be a plain arithmetic bug
and is now fixed independently of S2. What is left for the normal representation is surfaces, and
simplification.)*

---

## 13. What S0 actually turned up

Four things, in descending order of how much they change the plan.

### 13.1 The seam failure was a reflection, not a limitation — and it was silent

Writing the T2 orientation sweep produced failures the plan did not predict, at orientations §2.1
said were fine. The cause was in the second branch of `CurvedEntityCircleArc::apply_periodicity`:

```cpp
if (parametric[0][0] > 0)  parametric[0][0] = -M_PI + (parametric[0][0] - M_PI);   //  p - 2*pi   correct
else                       parametric[0][0] =  M_PI - (parametric[0][0] + M_PI);   //  -p         a reflection
```

The first line unwraps by a period. The second returns `-p`, which is not congruent to `p` modulo
2π at all — it is the mirror image across the *x* axis. The entity then reported the wrong point
for that facet corner, the Coons blend's corner values stopped agreeing (§1.4 shows why the blend
is only exact when they do), and the mesh came out **silently wrong**: no error, no warning, a
boundary node up to `3.4e-2` off a unit circle.

Which of the two failure modes an arc hit depended on the order in which the mesh author happened
to declare the facet's two nodes — a free choice describing identical geometry:

```
30 deg arc, facet order   a0 = 155..165           a0 = 170..180
start-to-end              throws                  exact
end-to-start              builds, 1.9e-2..3.4e-2  throws
```

So for *every* arc straddling the cut, one of the two equally valid ways of writing it down was
wrong, and half the time wrong without saying so. That is materially worse than §2.1's "it throws
in a narrow band", and it is the kind of defect the acceptance criterion of the harness — machine
precision, not "looks plausible" — exists to catch.

Fixed in S0 by replacing both that heuristic and `CurvedEntityCylinderArc`'s unconditional throw
with one shared `unwrap_periodic_component()`: shift every facet node onto the branch nearest the
first node's. It is correct at every orientation, in both node orders, and — unlike the two-node
code it replaces — for the three- and four-node facets a 3d face has. The Python
`CurvedEntityCircle` example had the same class of bug (a per-value `numpy.mod`, which moves the
cut instead of removing it) and got the same fix.

### 13.2 Which weakens the case for S2, and the plan should say so

With per-facet unwrapping in place, T2 passes at every orientation in both node orders. **The 1d
seam is no longer a reason to change the representation.** The honest remaining case for the
normal-based parametrisation of §4.3 is narrower than §2.1 implies:

- **Surfaces, where it is still a correctness matter.** `(θ, φ)` on a sphere has a coordinate
  *degeneracy* at the pole, not merely a branch cut: at θ = 0 every φ names the same point, so no
  amount of unwrapping makes a facet containing or straddling the pole blend correctly. This is
  also what forces `CurvedEntitySpherePart`'s 90°-opening-angle restriction
  ([src/meshtemplate.hpp:309](src/meshtemplate.hpp#L309)). Normals remove both.
- **Facets spanning half the period or more**, where "nearest branch" is a tie and unwrapping is
  ambiguous. Normals fail here too (antipodal normals sum to zero, §4.3), but they fail *loudly*
  and at a facet width no reasonable mesh reaches.
- **Simplification.** Unwrapping is an atlas of per-facet local charts that works because facets
  are small. Normals are a single global chart, so `apply_periodicity`, the per-component period
  bookkeeping and the whole question disappear rather than being handled.

None of that is a blocker, so **S2 should be re-scoped**: it is no longer "fix the seam" but
"remove the pole degeneracy and retire a class of bug", and it can move after S3 (3d) if the 3d
work wants the entity interface settled first. The `blend_parametric` hook of §4.2 remains the
right shape, because unwrapping is exactly a blend rule the entity should own.

### 13.3 The triangular gap is worse than "cannot refine"

§1.2 recorded that triangles throw on refinement. They are also wrong *before* refinement: a
C2 curved triangular mesh has a mid-edge node per rim facet, placed at the chord midpoint by
`convert_for_C2_space` and never snapped onto the arc, because
`BulkElementBase::map_nodes_on_macro_element` returns early for T-elements
([src/elements.cpp:2337](src/elements.cpp#L2337)). Measured on an 8-segment disc: `7.6e-2`, exactly
`1 − cos(22.5°)`. C1 is exact only because it has no node that could be wrong.

So the macro element does *nothing whatsoever* for triangles — it is not "built but unusable during
refinement", it is inert from the moment it is created. Recorded as
`test_curved_tri_template_mesh_is_exact[C2]`.

### 13.4 The throw is unrecoverable, which the harness has to work around

`throw_runtime_error("MACRO ELEM")` fires *during* `RefineableTElement<2>::build`, leaving a
partially built son tree. Catching the `RuntimeError` is not enough: the next teardown of that
`Problem` walks the half-built tree and **aborts the interpreter**. Verified both ways — the same
script survives without a `with` block (release deferred to interpreter shutdown, which is skipped)
and aborts with one.

Consequence for the harness: the triangular cases run in a child process, so one unimplemented
feature cannot decide whether the rest of the suite gets to run. S1 removes the throw and with it
the need for the isolation, but the isolation is kept — it costs under a second per case and it is
the right shape for any future case that can crash rather than fail.

### 13.5 A test hook that did not exist, and a docstring that lied

§10.1 needed to sample the macro map over the reference *interior*, which nothing exposed:
`Element.get_macro_element_coordinate_at_s()` stops at the macro-element coordinate, although its
docstring claimed it "Returns the position given by the element's macro element mapping". Added
`get_macro_element_position_at_s()` (which does), corrected the other docstring, and T13 will use
it as its primary oracle.

---

## 14. What S1 turned up

### 14.1 The equivalence held, and it is now the refactor's safety net

S1a swapped the blend before S1b changed any behaviour, precisely so the swap could be judged on its
own. §10.1's comparison run against the new `GenericMacroElement` gives 2e-16 to 5e-16 over the whole
reference square at every orientation, against a Python reference that had been measured against
oomph's `QMacroElement<2>::macro_map` at 1.1e-15 before the swap. Transitively the new blend is the
old one, and the fast suite was byte-for-byte unchanged across S1a (447 passed either side). Risk 2
never materialised because it was never real.

### 14.2 Hanging nodes on a curved boundary do not exist in 2d — and the reason generalises

§6.1 flagged that `map_nodes_on_macro_element` snaps every node unconditionally, hanging ones
included, and asked for a test. The test found the guard never fires, for a structural reason worth
stating: **a node interior to a boundary facet belongs to exactly one element**, because a boundary
facet has no neighbour across it, so nothing coarser can ever constrain it. In 2d the facets of a
curved boundary *are* its edges, so boundary nodes simply cannot hang. Measured on a deliberately
non-conforming disc — 24 hanging nodes (quad) and 64 (tri), none of them on the rim, rim error 0.

The guard is still right, and it stops being vacuous in 3d: there two boundary *faces* share an edge,
so a node on that shared edge can hang when the two faces differ in refinement level. That is the
case S3 has to test, and §6.1's concern should be read as a 3d one.

`test_non_uniform_refinement_keeps_curved_boundary_exact` therefore asserts what is actually true and
useful in 2d — that a strongly non-conforming mesh leaves the curved boundary exact — and records why
the hanging case is absent rather than leaving it to be rediscovered.

### 14.3 The Q family did not need `Macro_element_vertex_s` after all

The plan (§3.3) proposed replacing `s_macro_ll`/`s_macro_ur` everywhere. In the event the Q family
keeps them: oomph's own build already maintains the box correctly, the affine map into it is exactly
what the general vertex form would reproduce, and `QElementBase` already provides a final overrider
for `get_x_from_macro_element`. Adding a second one reachable from the same class would have been
ambiguous (risk 1) and would have forced a disambiguating override into every Q class for no gain.

So `get_macro_element_coordinate_at_s` branches once: box for Q, vertex coordinates for everything
else. Risk 1 evaporates with it — only the simplex classes, whose bases leave the virtual broken,
need the two-line forwarder. The general form is still what makes the simplex and mixed cases work,
and is still what S3/S4 will use; it just does not have to displace something that already works.

### 14.4 What is deferred, explicitly

- `Brick3d` is carried over so the 3d template-level path does not regress, and bricks get the same
  post-build re-snap as quads. But a 3d element with **more than one curved facet now throws**: two
  curved facets sharing an edge each contribute that edge's deviation and the blend needs §3.2's
  inclusion–exclusion term. Refusing is better than silently doubling a deviation, and S3 removes it.
- Tetrahedra, wedges and pyramids still throw in `factory_element` — S3/S4.
- Moving meshes are unchanged: `reapply_macro_element_positions` is gated on `moving_nodes`, so an
  ALE mesh still gets the macro element only through the initial-adaption pass, exactly as before.
  S5 gives it the undeformed-macro-element treatment of §5.

---

## 15. What S3 turned up

### 15.1 The inclusion–exclusion term is right, and the test had to be chosen carefully to show it

A tetrahedron whose four vertices all lie on the sphere can have any number of its faces declared
curved, and *any two faces of a tet share an edge* — so it is the sharpest available test of §3.2's
correction. Sampling the macro map on a 21-point barycentric grid over each face:

```
curved faces   max |r-1| on the declared-curved faces   on the others
      1                    2.2e-16                        1.1e-01
      2                    3.3e-16                        1.1e-01
      3                    3.3e-16                        9.7e-02
      4                    5.6e-16                          --
```

Exact in every configuration. The right-hand column matters as much as the left: a face that was not
declared curved must *not* be dragged onto the sphere — it is a ruled surface carrying whatever
curved edges it inherits (§3.2.1), which is what makes a partially curved element watertight against
its neighbour.

Getting there needed one correction of method. Measuring *node positions* after refinement showed
errors of 1.1e-1 that had nothing to do with the blend — see §15.2. The test therefore samples the
macro map directly, which is the thing under test, and leaves node bookkeeping to its own test.

### 15.2 A node can be marked as being on a boundary it is not on

`RefineableTElement<2>::get_boundaries` gives a new mid-edge node the boundaries **shared by both end
nodes** ([src/refineable_telements.cpp:458-467](src/refineable_telements.cpp#L458-L467)), which the
3d builder uses too. Two nodes can share a boundary label without the edge between them lying on that
boundary, and then the new node is labelled for a boundary it is nowhere near.

This is not new and has nothing to do with macro elements — it reproduces with straight-sided
elements and no curved entity anywhere. Measured on the degenerate case, a *single* tet with all four
faces on one boundary:

```
refinements   nodes marked "shell"   of those, geometrically interior   pinned by DirichletBC
     1                 10                          0                            0
     2                 35                          1                            0
     3                165                         35                            0
```

Two things bound its consequences. It needs a genuinely ambiguous geometry: the same measurement on
the tet-ball octant — three tets, one face each on the shell, i.e. a mesh someone might actually
write — gives **0** spurious nodes at every level. And where it does happen, the mislabelled nodes
are **not pinned**: their solution values come out as ordinary interior values (3.6e-3, 5.8e-3, not
the boundary value 0). Dirichlet conditions are applied through interface elements built from element
*faces* (`nboundary_element` + `Face_index_at_boundary`), not from node labels, so the assembled
problem is unaffected.

What it does affect is anything that iterates nodes *by boundary index* — user post-processing,
boundary-coordinate assignment, and tests like the ones in this file. **Characterised properly in
§23**, which supersedes the "needs a degenerate geometry" reading here: the trigger is narrower than
it looks and the worst case is larger.

### 15.3 ~~The edge→entity registry of §3.2.2 is not needed yet~~ — wrong, see §17.2

*Retracted.* This section argued the registry could be deferred because every element that touches
the sphere in `SphericalOctantMesh` and the tet ball touches it in a face. That is true of those two
meshes and false in general: it is a property of hand-built structured meshes, not of the geometry.
An unstructured gmsh tetrahedral ball has 25 of 126 elements touching the sphere along an **edge**
only, and they were silently placing that edge's new nodes on the chord. §17.2 has the measurement
and the fix. The inconsistency check this section mentions is still where it says, in
`GenericMacroElement::rebuild_edge_corrections`.

### 15.4 The sphere is now the reason to finish S2, not the seam

S3a needed the normal-parametrised sphere before anything 3d could be tested at all, so the part of
S2 that is a genuine correctness matter (§13.2) is already in. What remains of S2 —
`blend_parametric`, the intrinsic/parametric dimension split, and the optional projection hook — is
now purely about giving user-defined entities a clean interface, since the built-in entities that
needed it have it. It has no dependents left and can be scheduled freely.

---

## 16. What S4 turned up

### 16.1 The pyramid was the test of the whole design, and it passed unchanged

Every earlier stage could in principle have been done by special-casing. The pyramid could not, and
it is where the two central choices earn their keep.

Its C1 shape functions are *rational* — `psi_0 = (w-s0)(w-s1)/w` with `w = 1-s2` — because the whole
quadrilateral base collapses to a point at the apex. They are still a partition of unity, still 1 at
their own vertex and 0 at the others, and still vanish on the facets not containing their vertex,
which is all §3.2 ever asked of them. So the blend needed no pyramid-specific term: only an entry in
`macro_c1_shape`. One caveat had to be handled — the shipped `PyramidElementShapeC1::shape` divides
by `1-s2` unguarded, which is fine for its callers but not for a macro map evaluated at arbitrary
points of the reference domain, so the macro version takes the apex limit explicitly.

And refining a pyramid yields **six pyramids and four tetrahedra** — 10 elements after one
refinement, 92 after two, both measured. A son therefore inherits a macro element from a father *of a
different shape*. That works only because §3.3 carries the son's region as vertex coordinates
converted through the father's own shape functions: the two shapes never appear in the same
expression, so nothing has to reconcile 5 vertices with 4. oomph's axis-aligned `s_macro_ll`/
`s_macro_ur` cannot express it at all, which is the concrete reason the box had to be generalised
rather than extended.

Wedge and pyramid, one curved facet each, are exact at 0, 1 and 2 refinements (`0`, `0`, `1.1e-16`).

### 16.2 Four throws, three call sites, one mechanism

The remaining refusals were at
[wedges_and_pyramids.cpp:367](src/wedges_and_pyramids.cpp#L367) (wedge),
[elements.cpp:7590](src/elements.cpp#L7590) (`build_as_pyramid_son`, which also serves tet sons of a
pyramid) and [elements.cpp:7772](src/elements.cpp#L7772) (`build_as_brick_son`, the mixed-forest
brick). All three became the same two lines — `inherit_macro_element_from_father(father, sv)` — since
each builder already computes `sv`, the son's vertices in the father's coordinates, for its own node
mapping.

The brick son was the only one that needed anything extra: it maps node-by-node rather than from a
vertex list, so `collect_son_vertex_coords_in_father` recovers the eight vertices by matching each
reference vertex to the son node sitting there and asking the builder where that node lies in the
father. Generic over shapes, and the natural hook if another builder ever needs the same.

### 16.3 Coverage now, and what is left

Every element shape pyoomph has — quad, triangle, brick, tetrahedron, wedge, pyramid — places
refinement-generated nodes on curved boundaries exactly, in 2d and 3d, including mixed forests.
`GenericMacroElement` is ~300 lines and contains one shape-dependent function.

What remains is not about shapes:

- **S2 residual** (§15.4): `blend_parametric`, the intrinsic/parametric dimension split, the optional
  projection hook. Purely an interface for user-defined entities now; the built-ins that needed
  normals have them.
- **gmsh 3d**: `GmshTemplate._curved_entities2d` is still never populated, so 3d curved boundaries
  are reachable from hand-built templates and `SphericalOctantMesh` but not yet from gmsh. This is
  now the largest user-visible gap and deserves its own stage.
- **S5 moving meshes**, **S6 MPI**, **S7 docs** as planned.
- The boundary-inheritance over-marking of §15.2, on its own merits.

---

## 17. gmsh in 3d

### 17.1 `map_to_sphere` has to be opt-in

`GmshTemplate._curved_entities2d` was declared with a `# TODO: Set those` and never populated, so a
gmsh 3d mesh had no curved geometry at all. The consumer side already existed
([gmsh.py:1450](pyoomph/meshes/gmsh.py#L1450)) and needed no change: populating the dict is the whole
of the wiring.

What it cannot be is automatic. **Gmsh's built-in kernel does not produce an exact sphere from a
ruled surface**, even when the bounding curves are great-circle arcs — the surface it meshes is its
own ruled interpolant, not the sphere. Attaching a spherical entity to every ruled surface would
therefore impose a geometry gmsh never meshed. So `ruled_surface(..., map_to_sphere=True)` is opt-in,
and it is the user asserting "this surface is meant to be a sphere", which pyoomph then makes exactly
true (including projecting the surface's own nodes onto it, which moves them slightly).

The centre and radius are recovered from the bounding curves, with a deliberate bias toward refusing:

- Only points *certainly* on the surface are used — the two endpoints of each bounding curve. A
  `CircleArc`'s middle point is its arc centre, which lies on the sphere only if the arc is a great
  circle, so it is collected separately as a *candidate centre* instead.
- A candidate centre is accepted only if every boundary point is equidistant from it. For the usual
  construction — a patch bounded by great-circle arcs about the sphere's centre — the arc centre is
  that point exactly, so no fitting happens at all.
- Failing that, a least-squares sphere fit, but **only from five distinct boundary points on**. Four
  points in general position lie on exactly one sphere, so a fit through four always succeeds and the
  verification could never reject anything; requiring a fifth is what makes agreement evidence.
- Otherwise it raises, and says that `map_to_sphere=(cx,cy,cz)` will state the centre explicitly.

Measured on an octant of a ball meshed with tetrahedra: `map_to_sphere=True` gives a shell error of
`0`, `2.2e-16`, `2.2e-16` at 0, 1 and 2 refinements; without it, `3.3e-16`, `1.5e-2`, `1.6e-2` — the
nodes gmsh itself places are on the sphere, everything refinement adds is not.

### 17.2 An unstructured mesh needs the curved-edge registry, and §15.3 was wrong to defer it

The first `map_to_sphere=True` run improved the refined shell from `1.5e-2` to `1.1e-2` — better, and
nowhere near exact. Classifying the coarse mesh's 126 elements by how much of the sphere they touch:

```
nodes on the shell   elements   had a macro element
        0               44             no
        1               12             no
        2               25             no      <-- these are the problem
        3               45             yes
```

45 elements own a face on the sphere and get a macro element. **25 touch it along an edge only.**
Those have no curved facet, so before this they got no macro element at all, and when refined they
placed the midpoint of that shared edge by straight interpolation — on the chord — while the element
on the other side of it placed the same node on the sphere. Whichever built the node first won.

This is exactly the case §3.2.2 predicted and §15.3 dismissed, and the dismissal was the wrong
inference: "every element touching the sphere owns a face on it" is a property of the two structured
meshes that had been tried, not of curved boundaries. An unstructured mesh breaks it as a matter of
course.

The fix is §3.2.2's registry, and the pieces were already in place. `MeshTemplate` now builds
`curved_edge_map`, keyed by sorted template node-index pair, from the edges of every curved facet;
`factory_element` gives an element a macro element carrying those edges as **two-vertex curved
sub-entities** whenever it touches one without owning the facet. `GenericMacroElement` needed nothing
at all: a sub-entity of two vertices is already what the blend expects, and `w_E d_E` on an otherwise
straight element is precisely the ruled surface that makes the two sides agree.

Two details worth keeping:

- Edges already inside one of the element's own curved facets are skipped, or the facet deviation and
  the edge deviation would both be applied and the edge would be counted twice.
- Only *genuine* edges are registered, from a per-shape table (`macro_edges`). A quadrilateral facet's
  diagonal joins two points of the surface but is not an edge of anything; curving it would bulge an
  element's interior rather than its boundary.

With the registry, 70 of the 126 elements carry a macro element — the 45 with faces plus exactly the
25 with edges — and the shell is exact at every refinement level.

---

## 18. S5 (moving meshes): investigated, closed without a code change

Decided on 2026-07-29. Recorded here rather than silently dropped, because the plan promised
something that turned out to rest on a false premise, and because the alternative was a change to ALE
behaviour that should not happen by accident.

### 18.1 The `Undeformed_macro_elem_pt` route does not apply

§5 proposed that on a moving mesh the macro element should drive the **Lagrangian** coordinate ξ, via
oomph's `Undeformed_macro_elem_pt` and
`enable_use_of_undeformed_macro_element_for_new_lagrangian_coords`. The mechanism exists and is
plumbed for the Q family: `QSolidElementBase::get_x_and_xi`
([Qelements.h:379-414](src/thirdparty/oomph-lib/include/Qelements.h#L379-L414)) takes ξ from the
undeformed macro element when one is set.

It is not the mechanism pyoomph uses. `pyoomph::Node` *is* an `oomph::SolidNode`
([nodes.hpp:119](src/nodes.hpp#L119)), and `MeshTemplateElementCollection::lagrangian_dimension()`
defaults to the nodal dimension, so ξ storage is allocated — yet in a Poisson +
`PseudoElasticMesh` problem `interpolated_xi` returns zeros at every node, including a boundary node
whose Eulerian position is `(0, 1)`. The element's `nlagrangian()` comes from the generated code's
`lagr_dim`, and pyoomph's `var("lagrangian")` — which `PseudoElasticMesh` really does use, via
`grad(..., lagrangian=True)` — reaches the reference configuration by its own route, not through
oomph's `xi`.

So there is no ξ defect here to fix, and wiring `Undeformed_macro_elem_pt` would have driven a
quantity nothing reads. Anyone reviving this must first establish where pyoomph's ALE reference
configuration actually lives; the plan's §5 assumed an answer it did not check.

### 18.2 What is really left is a policy question about x

The measurable gap is in the Eulerian position: a moving mesh with a curved boundary drifts to
**7.13e-4** after two runtime refinements, where a static one is exact.

The cause is S1's own gate. `reapply_macro_element_positions` is a no-op when `moving_nodes` is set
(§5 asked for exactly that, to keep ALE semantics unchanged until this stage), and separately
`remove_macro_elements_after_initial_adaption="auto"` drops the macro elements once the coordinates
are free. Both were checked, and it is the gate that matters: forcing the flag to `False` keeps 32
macro elements alive through the refinement and the error is still 7.13e-4.

Whether that is a bug depends on what the curved boundary *means*, which the code cannot know:

- On a **free surface** — `droplet_spread_3d`'s shell — the sphere is only the initial shape. Snapping
  new nodes back onto it would drag the interface toward a shape the solve has already left. Today's
  behaviour is correct.
- On a **curved rigid wall** whose position is pinned, the curve is the geometry for all time, and
  today's behaviour degrades it a little on every adapt.

Four options were weighed: leave as today; snap only where the mesh position is pinned; always snap;
or add an explicit per-boundary flag. **Decision: leave as today.** The drift is second-order and only
affects a case (a pinned curved boundary on an adaptively refined ALE mesh) that no shipped example
exercises, whereas every candidate fix changes ALE results for meshes that are currently correct.
Stability of ALE semantics is worth more than the last 7e-4 here.

If it is ever revisited, "snap only where pinned" is the one to build: it leaves free surfaces alone
by construction. Note that it cannot be done in `build()` — position boundary conditions are not
applied yet at that point — so it belongs in a post-adapt pass alongside
`Problem.map_nodes_on_macro_elements()`.

### 18.3 One thing fixed on the way

`Mesh::set_lagrangian_nodal_coordinates` printed `"Setting Lagrangian nodal coordinates for all nodes
in mesh"` unconditionally on every call — the same class of leftover as the three removed in S0, and
noisy on any moving mesh since it runs on every macro-element mapping pass. Removed.

---

## 19. S6 (MPI): two defects that only `--distribute` could reach

§6.2 offered two options — build macro elements on every rank, or accept FE placement on halos and
let halo synchronisation copy the owner's positions — and guessed the second would be enough. Neither
was the issue. The macro elements were already present on every rank (the mesh is built replicated and
then thinned), and the failures were in what a macro element *holds* and what it *sets*.

Nothing had ever run this combination: distributed adaptivity was covered, curved boundaries were
covered, the two together were not.

### 19.1 A macro element must not hold node pointers

The first `--distribute` run on a curved mesh segfaulted, in
`GenericMacroElement::vertex_position` reached from
`Mesh::distribute` → `classify_halo_and_haloed_nodes` → `synchronise_nonhanging_nodes` →
`FiniteElement::get_x`.

`Mesh::distribute()` deletes the elements and nodes a rank does not own. A macro element belongs to
the **root** element and is shared by every son, so a son that survives on this rank can be left
holding a macro element whose root vertices have been freed — and the halo classification pass then
calls `get_x()` on it.

This was **not introduced by this work**: the implementation it replaced stored
`std::vector<std::vector<pyoomph::Node *>> default_facet_nodes` and dereferenced it at
`meshtemplate.cpp:155` from the same call site. It had simply never been exercised.

`GenericMacroElement` now stores vertex **positions by value**. The header records why the obvious
choice is wrong, since it is the sort of thing a later reader would try to "fix" back: node pointers
would let the map follow the nodes through history levels and, on a moving mesh, through the solve —
but the macro element is only ever consulted for geometry that is fixed (on a moving mesh it
deliberately does not drive the Eulerian position, §18.2), so there is no motion to follow and
nothing is lost.

### 19.2 `map_nodes_on_macro_element` was only setting the present time level

With the crash gone, the mesh came out geometrically corrupt: *"Max. error in quadtree neighbour
finding: 0.447214 is too big"*. Serial and replicated-MPI runs of the same problem were exact
(`2.5e-16`), so it was distribution-specific.

The cause was a `// TODO: Time loop` that had been sitting in `map_nodes_on_macro_element` since
before this work: it snapped `x(id)` — history level 0 only — leaving `t >= 1` on the straight
interpolant that `factory_element` had written there. `synchronise_nonhanging_nodes` compares
`get_x(t, s)` against `x(t, ·)` **at every history level** while distributing, saw conforming nodes
disagree, and "repaired" them.

Now every history level is set. The macro geometry does not depend on time, so one evaluation serves
them all. This also fixes something visible without MPI: a curved mesh used to start with `x(0)` on
the curve and `x(1)` on the chord, i.e. presenting a mesh at rest as though it had been moving into
its initial state.

### 19.3 A Problem must be released before the next one distributes

Not a curved-boundary defect, but it cost time and belongs in the record. The worker initially ran its
five cases in one process without a `with` block; the *second* case always failed, whichever it was.
An un-released `Problem` keeps its distributed state alive and the next `distribute()` in that process
dies. `box_cases.solve_case` already used `with` for the same reason. The worker says so at the point
where it matters.

### 19.4 What S6 did not cover — **both closed in §22**

- ~~T8, coupled interfaces with curved boundaries.~~ Measured in §22.1.
- ~~Curved boundaries under load balancing.~~ Measured in §22.2.

Measured: all five geometries (quad disc, triangular disc, spherical octant of bricks, tetrahedral
ball, gmsh tetrahedral ball) exact to machine precision on every rank at 2 and 4 ranks.

---

## 20. S2 as actually built

By the time S2 came round, most of it had already happened elsewhere: buffer sizing from
`get_parametric_dimension()` fell out of S1a, and the normal-parametrised sphere was pulled into S3a
because the pole degeneracy blocked all 3d testing (§13.2, §15.4). What was left was the interface —
and one item that turned out to be a bad idea.

### 20.1 `blend_parametric`, and why it is load-bearing rather than decorative

A parametric coordinate is an opaque vector whose meaning belongs to the entity, so the rule for
combining two of them belongs to the entity too. `MeshTemplateCurvedEntity::blend_parametric(weights,
params, result)` defaults to the weighted sum — correct for a flat, non-redundant chart: an angle, an
arclength, a spline parameter — and `CurvedEntitySpherePart` overrides it to renormalise, because the
average of two unit normals is not a unit normal.

`GenericMacroElement::subentity_deviation` now calls it instead of summing inline. It is called at
arbitrary weights, not only at a facet's vertices, since refinement evaluates the blend anywhere on
the facet. This is distinct from `apply_periodicity()`, which runs once when a facet is built and
merely chooses which representatives of a periodic coordinate to store; both are kept and the header
says which is which.

The hook is exposed to Python (`blend`), which is the point of it: a user can now define an entity
whose chart is redundant. `test_user_entity_can_own_its_blending_rule` builds exactly that — a circle
charted by a 2-component unit normal — and asserts **both** halves, because the interesting claim is
not that it works but that the hook is doing something: with the override the rim is exact at
`1.1e-16`, and with the default weighted sum left in place it is `7.6e-2`, the chord sagitta of a 45°
facet, i.e. no better than no curved treatment at all.

### 20.2 `CurvedEntityCircleArc` should NOT be rewritten on normals

§4.3's table said to move the circle to a unit normal alongside the sphere. That would make it worse,
and the reason is worth recording so it is not "fixed" later.

`CurvedEntityCircleArc` parametrises by angle and maps through `cos`/`sin`, so blending the angle
linearly **is already exact slerp** — uniform in arclength, exact at every weight. Moving to normals
would mean either nlerp, which §10.2 measured at up to 1% of an element edge for a 45° facet, or
implementing slerp on normals to get back to precisely where it started.

The sphere is a different case and the distinction is the whole point: there the angular chart has a
genuine *degeneracy* at the pole, not merely a branch cut, so no repair of the blend could fix it and
the redundant chart was a correctness matter. For a circle, the branch cut is handled exactly by
`unwrap_periodic_component` (S0), verified across the full orientation sweep in both node orders. A
redundant chart buys nothing there.

The rule this leaves: **use a redundant chart when the natural one is degenerate, not merely when it
wraps.**

### 20.3 Dropped as speculative

`facet_map` (blend-then-map as one overridable call) and `project_onto` (closest-point projection)
were both in §4.4. Neither has a caller: `blend_parametric` plus `parametric_to_position` already
covers everything the macro element does, and projection was motivated by splines whose
parametrisation is only approximately arclength — a real concern, but one no current entity or test
exercises. Adding unused virtuals to a class users subclass is a cost with no return, so they are
left in §4.4 as designs rather than shipped as API. `get_intrinsic_dimension()` *was* added despite
having no caller in the core, because it makes an otherwise puzzling asymmetry legible: a sphere
patch reporting `get_parametric_dimension() == 3` looks like a bug until something says it is a
2-manifold on purpose.

### 20.4 What the blend path costs

Asked after the fact, and worth recording because the answer changes how a user should write an
entity. The macro map is *not* in the assembly loop — it is reached only from
`map_nodes_on_macro_element` (once per node, at mesh build and after each adaption) and from
`FiniteElement::get_x` while refinement builds new nodes. So the frequency is O(nodes created), not
O(quadrature points × Newton steps), and `pyoomph/meshes/interpolator.py` already strips macro
elements before `locate_zeta`, which is the one other route that would have made it hot.

Measured on a triangular disc refined to 8192 elements — `refine` is four uniform refinements,
`resnap` one full `map_nodes_on_macro_elements()` pass:

```
                                     refine     resnap    boundary error
no macro element at all              0.221 s    0.006 s      4.8e-03
built-in C++ entity                  0.240 s    0.026 s      2.2e-16
Python entity, no blend() override   0.333 s    0.171 s      2.2e-16
Python entity, with blend()          0.431 s    0.311 s      2.2e-16
```

Three things follow.

**The C++ path is not a concern.** Curved geometry costs ~8% on refinement and 20 ms per re-snap pass
at this size. An attempt to remove the remaining per-call allocation (hoisting the blend scratch into
member buffers) recovered about 8% of the 26 ms and was **reverted**: it introduces a mutable member
whose failure mode, if a re-entrant Python callback ever clobbered it mid-use, is silently wrong
geometry. Not a trade worth making for that.

**A Python entity costs roughly an order of magnitude more**, and that is mostly *not* new — the
`parametric_to_position` callback has always been there. Still linear in mesh size, so a 10^5-element
mesh pays ~2 s per re-snap pass.

**Overriding `blend()` roughly doubles the Python cost**, because it is a second callback per
evaluation. The third row above is the useful one: that entity leaves `blend()` alone and normalises
inside `parametric_to_pos` instead — and is *equally exact*, at half the overhead. So the guidance,
now also in the header where someone writing an entity will meet it: fold a correction into
`parametric_to_position` when it can be expressed pointwise; override `blend_parametric` only when the
rule genuinely needs the other samples, such as unwrapping a periodic coordinate relative to its
neighbours, or a true slerp.

---

## 21. S7 (docs), and the API gap it exposed

### 21.1 The public factory could not build anything this work added

Writing the documentation surfaced a gap the implementation stages had all worked around without
noticing: `MeshTemplate.create_curved_entity` — the *supported* way to build a curved entity from a
hand-written template — only ever accepted `"circle_arc"`. Every sphere in this branch, including the
ones in `tests/test_curved_boundaries.py` and in `SphericalOctantMesh`, reached past it into
`_pyoomph.CurvedEntitySpherePart` directly. That is a fair signal: if the tests cannot use the public
API, neither can a user.

It now accepts `"sphere_part"` (a point on the sphere plus the `center`) and `"cylinder_arc"` (start,
end, plus the `center`, with the axis following from the three points), both taking node indices or
coordinates like `"circle_arc"` does, and both registered in `_macrobounds` so the entity stays alive.
`SphericalOctantMesh` was switched over to it, so the shipped meshes exercise the same path a user
would take.

### 21.2 `CylinderMesh` has a curved mantle

The `# TODO: Add curved entities!` above it is resolved. Measured: the mantle is exact at every
refinement level, against `7.6e-2` before — a number that, as everywhere else in this work, does not
improve with refinement, because subdividing a polygon gives a finer polygon.

Consistent with `CircularMesh` and `SphericalOctantMesh`, it takes `with_curved_entities` (default
`True`) for the old behaviour.

### 21.3 What the documentation says

Two new sections in `docs/source/tutorial/spatial/mesh/unitsandmacro.rst`, which already covered the
2d quadrilateral case and already claimed curved boundaries "resemble the very same smooth boundary
curve also upon refinement" — a promise that was aspirational when written and is now true.

*Curved boundaries in general* states the coverage (all six element types, 2d and 3d, hand-built and
gmsh, serial and distributed), lists the entity types, and makes the cost of *not* using one concrete
with the spherical octant's 23% volume deficit — deliberately, because the intuition this work
repeatedly ran into is that refinement will fix a coarse curved boundary, and it will not. It also
carries a warning about moving meshes, so §18's decision is visible to users rather than buried here.

*Writing your own curved entity* documents the Python subclass interface, and in particular that the
parametric coordinate is opaque and may be redundant, with the sphere's unit normal as the worked
reason. It gives the §20.4 guidance in the form a user needs it: prefer to absorb a correction into
`parametric_to_pos`, which runs anyway, and override `blend` only when the rule needs the other
samples.

---

## 22. Closing §19.4

### 22.1 A curved interface shared by two coupled domains

Two concentric annular domains sharing a circular interface, with only the inner one told to refine so
that `InterfaceRefinementCoupler` has to carry the requirement across. Each side then places the
interface's new nodes through its *own* macro elements, and they have to land in the same places or
`connect_interface_elements_by_kdtree` has nothing to pair up.

They do. Measured: interface exact to `1.1e-16` from both sides, and the two domains' interface node
sets are **identical** — not merely close, the same rounded coordinates. The straight-sided control
gives `1.3e-2`, so the test is discriminating.

§19.4 called this "an argument, not a measurement", the argument being that both sides attach the same
entity to the same facets. The argument was right, but it was worth checking: the two sides run
independent refinement decisions through independently constructed macro elements, and "same inputs"
only implies "same outputs" if nothing in between introduces an ordering or accumulation difference.
Now it is `test_curved_shared_interface_agrees_from_both_sides`.

### 22.2 Load balancing, and a pre-existing defect next to it

`call_load_balance_in_initial_adaption = True` redistributes the mesh during the initial adaption, and
curved boundaries come through it exact (`0.0` on both ranks, all five geometries). Covered by
`test_curved_boundaries_survive_load_balancing`.

Calling `Problem.load_balance()` **by hand** after `initialise()` is a different matter: it dies in
`Mesh::generate_interface_elements` with *"bulkmesh was not set"*
([src/mesh.cpp:4958](src/mesh.cpp#L4958)) and then segfaults. That is **not** a curved-boundary
defect — it reproduces identically with `with_curved_entities=False`, i.e. with no macro element in
the problem at all. It is recorded here because this is where it was found, not because it belongs to
this work: something about the manual entry point leaves an interface mesh without its bulk mesh after
redistribution. Anyone wanting it should start from the fact that the flag-driven path through the
initial adaption works.

---

## 23. §15.2 characterised: boundary-node membership

Pre-existing and unrelated to curved boundaries — it reproduces with straight-sided elements and no
macro element anywhere. Followed up because it was found here and left vaguely stated.

### 23.1 The rule, the trigger, and the size of it

A node is on a boundary iff it belongs to one of that boundary's facets. When refining, oomph
approximates that: a new mid-edge node inherits the boundaries **shared by both its end nodes**
([src/refineable_telements.cpp:458-467](src/refineable_telements.cpp#L458-L467)). Two nodes can share
a boundary label without the edge between them lying on that boundary.

§15.2 read this as needing a degenerate geometry. That was wrong in both directions. The actual
trigger is precise:

> An element with **two or more faces on the same boundary**. Its remaining faces then have all their
> vertices on that boundary, so every edge of them is mislabelled — and each such edge seeds more of
> them at the next refinement.

Measured, as (nodes marked) → (of those, on no facet of that boundary):

```
                                          nref=1        nref=2        nref=3
tet, 2 of 4 faces on the boundary        1 of 10      10 of 35      84 of 165   (51%)
tet, 4 of 4 faces on the boundary        0 of 10       1 of 35      35 of 165   (21%)
```

So where it does bite it is not marginal. But on every mesh anyone would actually write it does not
bite at all — measured **0 spurious nodes** across a triangular disc, a spherical octant of bricks, a
tetrahedral ball, a rectangular quad/tri mesh, a full circular disc and an unstructured gmsh
tetrahedral ball, each at one, two and three refinements, up to 64512 elements. The reason is
structural: an element normally meets a given boundary in a single face, so no interior edge has both
ends on it. Two faces of one element on the *same* boundary needs the boundary to wrap around the
element — a one-element-thick shell or slot. Reachable, but not what the shipped meshes do.

Both halves are pinned: `test_boundary_node_membership_matches_the_facets` guards the realistic
meshes, and `test_boundary_node_membership_is_wrong_when_a_boundary_wraps_an_element` is a strict
xfail carrying the numbers above, so a fix cannot land unnoticed.

### 23.2 Not repaired, and why

The consequences are bounded. Dirichlet conditions are applied through interface elements built from
element *faces*, not node labels, so the assembled problem is unaffected — verified in §15.2 by
solving and checking that the mislabelled nodes are not pinned and carry ordinary interior values.
What is affected is code that iterates nodes *by boundary index*: user post-processing, boundary
coordinates, and tests.

The fix is not the inheritance rule. An element cannot tell whether its face is on a boundary from its
own nodes — that is the circularity the rule exists to work around — so it would have to consult the
mesh's boundary-element information, which is exactly what `setup_boundary_element_info` rebuilds
*after* an adapt. The natural shape is therefore a post-adapt repair: for each boundary, recompute the
node set from its facets and drop anything else, which is precisely what the test above already does
in Python.

That was deliberately **not** implemented here. It changes which nodes belong to which boundary —
mesh topology bookkeeping that boundary coordinates, hanging-node setup and the distributed halo node
lists all read — at the end of a large branch, in exchange for fixing zero occurrences in any mesh the
suite contains. The risk is concentrated exactly where this branch already found its subtlest defects
(§19), and the payoff is not measurable on realistic input. It is a good change to make deliberately,
with its own validation pass; it is a bad change to slip in here.
