# Generalised MacroElements: curved boundaries for every element type

Written 2026-07-29 on branch `macro_elements`, after the interface-refinement-coupling work
(`interface_refinement_coupling.md`) landed on `main`.

Status: **S0, S1 and S3 done; S2 (re-scoped) and S4 next.** §1 and §2 are measured, not assumed — every claim about current
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
- **S2 — the parametric coordinate.** `get_intrinsic_dimension`, `blend_parametric`, `facet_map`,
  buffer sizing from `get_parametric_dimension()`; rewrite `CurvedEntityCircleArc` and the Python
  `CurvedEntityCircle` on normals; deprecate `apply_periodicity`. T2, T3 green.
- **S3 — 3d Q and simplex.** ✅ **Done**, in three commits: S3a the normal-parametrised sphere plus
  `SphericalOctantMesh`'s shell (bricks), S3b tetrahedra, S3c the shared-edge inclusion–exclusion.
  T4 and T13 green; curved module 55 passed, fast suite 471 passed. **Not** done here, and moved to
  their own stage: populating `GmshTemplate._curved_entities2d`, and the edge→entity registry of
  §3.2.2 — see §15.3 for why the registry turned out not to be needed yet.
- **S4 — wedge, pyramid, mixed forests.** Remove the last two throws. T5 green.
- **S5 — moving meshes.** The `moving_nodes` split of §5, `Undeformed_macro_elem_pt` wiring, revisit
  `remove_macro_elements_after_initial_adaption`. T10 green.
- **S6 — MPI and coupled interfaces.** §6.2's decision, validated against the `mixed_adapt` MPI
  harness. T8, T9 green.
- **S7 — docs and tutorials.** Curved-entity documentation, `CylinderMesh` curved entities, remove
  the `with_macro_element=False` workarounds from any tutorial that carries them. T12.

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
boundary-coordinate assignment, and tests like the ones in this file. Worth fixing on its own merits
(the honest rule is "on the boundary iff the containing facet is"), but it is not this work's, and
nothing here depends on it.

### 15.3 The edge→entity registry of §3.2.2 is not needed yet

§3.2.2 argued that watertightness needs edge geometry to be a mesh-level fact, because an element can
touch a curved surface along an edge alone, without owning a face on it — and would then leave a
shared face flat while its neighbour ruled it. That case did not arise in any mesh built here: in
both `SphericalOctantMesh` and the tet ball, every element that touches the sphere touches it in a
face, so both sides of every interior face agree. The registry is therefore deferred rather than
built, with the reasoning left in §3.2.2 for when a mesh does produce the case. The
inconsistency check it was also meant to host is implemented directly in
`GenericMacroElement::rebuild_edge_corrections`, which refuses two different entities claiming the
same edge of one element.

### 15.4 The sphere is now the reason to finish S2, not the seam

S3a needed the normal-parametrised sphere before anything 3d could be tested at all, so the part of
S2 that is a genuine correctness matter (§13.2) is already in. What remains of S2 —
`blend_parametric`, the intrinsic/parametric dimension split, and the optional projection hook — is
now purely about giving user-defined entities a clean interface, since the built-in entities that
needed it have it. It has no dependents left and can be scheduled freely.
