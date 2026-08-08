# Mesh construction: boundary layers, element winding, and inverted elements

Two subjects that meet in the Gmsh path: how to build a mesh that stays *solvable* after refinement,
and what to do when a moving mesh folds. The second is partly a plan — §5 marks where implementation
stops.

---

## 1. Boundary-layer meshes: three constructions, one survivor

Written out of building the heated-cylinder tutorial, which needed a mixed quadrilateral/triangle mesh
— quads aligned with a wall, triangles in the far field — that could then be *adapted*. All three
constructions were measured.

A boundary layer is resolved far more economically by quadrilaterals aligned with the wall than by
triangles: the element can be long along the wall and thin across it, so the normal direction gets its
resolution without paying for it tangentially. pyoomph supports both in one domain — several Gmsh
surfaces may share a single domain name, and `set_recombined_surfaces` recombines only the ones you
name. Whether the result can be *adapted* is a separate question, and the one that decided the outcome:
the mixed hanging-node machinery itself is fine (see [adaptive_refinement.md](adaptive_refinement.md)),
but whether the elements a given construction produces stay solvable once refined varies.

### 1.1 Plain recombined annulus

An annulus between the cylinder and a slightly larger circle, meshed as one surface with
`mesh_mode="tris"` overridden by `set_recombined_surfaces`, with the outer box as a second surface
carrying the same domain name. Measured: 264 quads at radius 0.54–0.95, 5640 triangles from 1.02
outwards. It works and it adapts. Its weakness is that the layers are of uniform thickness — Gmsh fills
the annulus as it sees fit, so there is no grading towards the wall and no control over the first layer.

### 1.2 Gmsh's `BoundaryLayer` mesh size field — best mesh, does not survive adaptation

Gmsh can grow the layers itself. `"BoundaryLayer"` was already in the accepted field list of
`add_mesh_size_field`, but registering one needs `gmsh.model.mesh.field.setAsBoundaryLayer`, which
pyoomph did not expose — a boundary-layer field is not a background mesh size field (it describes a
region to *extrude* from the wall, not a size to sample), so Gmsh treats the two separately. Hence
`set_mesh_size_boundary_layer_field`:

```python
layer = self.add_mesh_size_field("BoundaryLayer", CurvesList=wall,
                                 Size=0.006, Ratio=1.25, Thickness=0.30, Quads=1)
self.set_mesh_size_boundary_layer_field(layer)
```

This gives the nicest initial mesh of the three, in the least code: 704 properly graded wall-normal
quads (r 0.502–0.727), thin at the wall and growing outwards, blending into 3316 triangles. The geometry
needs no annulus at all — one surface with the cylinder as a hole.

**And it does not survive adaptation:**

| adaptation steps | result |
|---|---|
| 0 | fine — Nu 6.6529, in line with the reference |
| 1 | fine, but only after raising `max_newton_iterations` |
| ≥2 | Newton fails to converge |

It is not a structural or hanging-node failure: the residual falls to ~1e-6 and then stalls, degrading
progressively as the stretched layers are refined. Raising the iteration cap to 30 bought exactly one
more adaptation; `globally_convergent_newton=True` bought nothing.

**Why the highly stretched elements Gmsh's extrusion produces become so much harder to solve once
refined has not been chased down. It is worth chasing** — the field is the least-effort way to get a
good boundary layer and would be useful for non-adaptive problems today.

### 1.3 Transfinite O-grid — what the tutorial uses

Four transfinite sectors of an annulus, graded radially and recombined:

```python
sectors = [self.plane_surface(wall[i], radial[(i+1) % 4], ring[i], radial[i], name="fluid")
           for i in range(4)]
self.make_lines_transfinite(*radial, numnodes=n_radial, mode="Progression", coeff=layer_growth)
self.make_lines_transfinite(*wall, *ring, numnodes=n_circumferential)
for i, sector in enumerate(sectors):
    self.make_surface_transfinite(sector, corners=[inner[i], inner[(i+1) % 4],
                                                   outer[(i+1) % 4], outer[i]])
self.set_recombined_surfaces(sectors)
```

Structured, aligned, graded by the `"Progression"` coefficient on the radial lines, and — the point — it
adapts happily through twelve refinement steps, reproducibly. More code than the boundary-layer field,
and the grading is yours to choose rather than Gmsh's.

**The corners must be passed explicitly.** `make_surface_transfinite` only takes its automatic path when
`corners` is empty, and that path re-derives node counts from the point sizes and *overrides* whatever
transfinite settings the curves already carry. Omitting them produces

```
Surface 2 cannot be meshed using the transfinite algorithm
(non-matching number of nodes on opposite sides 0 != 1 or 27 != -1)
```

which reads like a geometry problem and is not one.

### 1.4 Two latent bugs in the transfinite helpers

Both found while building the O-grid, both latent because every existing call site passes explicit
values, both fixed:

* **`make_lines_transfinite` sized only its first line.** `numnodes` and `coeff` were assigned back onto
  the *arguments* inside the per-line loop, so after the first line `numnodes` was no longer `"auto"` and
  every later line silently inherited the first one's count. The two sides of a 4×1 rectangle came out
  `[40, 40]` instead of `[40, 10]`.
* **`make_surface_transfinite` shared one surface's corners with all the others**, the same way: the
  auto-detected corners were assigned back onto the `corners` argument, so the second and later surfaces
  of a single call received the *first* surface's corners.

### 1.5 Recommendation

For a wall-bounded problem that will be **adapted**, use the transfinite O-grid. For one that will not,
the `BoundaryLayer` field is less code and grades better.

---

## 2. Two plotting findings from the same tutorial

### 2.1 Velocity overlays fail on deeply refined meshes

On the heated cylinder's *joint*-criterion mesh, both `mode="streamlines"` and `mode="arrows"` die with
matplotlib's `RuntimeError: Triangulation is invalid`, raised from
`LinearTriInterpolator.__init__ → get_trifinder() → TrapezoidMapTriFinder._initialize()`. Both sample
the field on a regular grid, and anything that does needs a `TriFinder`.

The triangulation is **not** geometrically broken, which is worth recording because it is the first thing
one suspects: 20 388 points, 28 636 triangles, **zero duplicate points and zero zero-area triangles**
(smallest area 2.7e-5). What the trapezoid map rejects is *overlapping* triangles — the signature of
hanging nodes, where the coarse side of a 2:1 interface spans two finer edges. `tricontourf` walks the
triangles directly and does not care; the search structure does.

Consequences worth knowing:

* **It is mesh-dependent, not a blanket incompatibility.** The Moffatt page draws streamlines happily on
  an adapted triangular mesh that also has hanging nodes. The cylinder's joint-criterion mesh, which
  refines the O-grid hardest, is the one that fails.
* **`crash_on_invalid_triangulation = False` does not rescue the plot.** It is a deliberate option
  (default `True`), but the part that fails sets `_has_invalid_triangulation` and `save()` then returns
  early — so the *whole figure* is dropped, not just the overlay.
* **Scalar colour plots are unaffected**, so a field comparison can still be made.

If a per-part degradation is ever wanted — drop the overlay, keep the figure — that is a small change
where `_has_invalid_triangulation` is consumed, and it is arguably what
`crash_on_invalid_triangulation = False` should already mean. It was not made because in a *comparison*
figure an overlay that silently appears on one panel and not the other reads as a physical difference
rather than a plotting artefact.

### 2.2 Colour bars over a dark field

`add_colorbar` places the bar by its own extent and does not account for the text around it, so at the
default `ymargin` of 0.05 a **top**-placed bar has its title *above* it, outside the reported extent,
running off the top of the figure; and a **bottom**-placed bar has its tick labels below it, running off
the bottom. Both clip inside the saved PDF, so no amount of `pdfcrop` recovers them. The fix is
per-position margins (0.15 top / 0.19 bottom in the tutorial), plus `xpos`/`length` on the bottom bar to
keep it clear of the geometry the extra margin pushed it into.

One trap in the figure pipeline rather than the plotter: converting dense line art to PNG with
ImageMagick's `-colors 4` merged the black annotation text into the nearest palette entry, which was the
mesh blue — the labels looked like a plotter setting had been ignored when they were black all along.
Sampling the pixels settled it: RGB (36,62,100) at 4 colours, (4,4,4) at 48.

---

## 3. Inverted elements: what the detector gives you

Detection (`BulkElementBase::detect_inverted_elements`) is global, off by default, switched on with
`set_detect_inverted_elements(True)`. When on, every integration point of every element whose mapping is
square gets its signed `det(dx/ds)` tested, and a non-positive one raises
`oomph::InvertedElementError`. Two places catch it, both backported alongside the exception and
described in `src/thirdparty/INFO_oomph-lib`: the adaptive timestepping loop, which rejects the step and
halves `dt`, and the arclength continuation loop, which rejects and scales `Ds` by 2/3.

**So the response to an inversion today is always "take a smaller step"**, and §3.2 is a case where that
cannot work no matter how small the step gets.

One thing the flag is not: it is global, while `RemeshWhen` is per-domain. Any per-domain option that
arms it therefore turns detection on everywhere and everyone pays the ~2 % assembly cost. Making it
per-domain would mean threading a flag down to the element, which is not obviously worth it — but it
needs saying in the user-facing documentation rather than being discovered.

### 3.1 The test case

`dev_docs/examples/inverted_element_notch.py`. Unit square, 10×10 quads, `LaplaceSmoothedMesh`, plus a
diffusing scalar. The deformation is prescribed entirely through the boundary: a Gaussian notch of
linearly growing depth pushed into the top edge, with the other three edges free to slide along
themselves. Three properties make it the right test rather than merely a mesh that breaks:

* **The domain stays valid.** At the moment of folding the notch is only about 0.13 deep. What folds is
  the harmonic extension into a non-convex shape, not the shape — Radó–Kneser–Choquet guarantees
  injectivity of a harmonic extension onto a *convex* target and says nothing here. So the fold is a
  discretisation artefact that a remesh genuinely repairs, which a collapsing domain would not be.
* **The scalar field is load-bearing.** Without a non-geometric unknown there is no temporal error norm
  to drive adaptivity with, and nothing for a remesh to interpolate.
* **The flags are independent** (`detect`, `adaptive`, `remesh`, `tight`), so the four behaviours below
  are the same problem seen four ways rather than four problems.

The script reports `min det(dx/ds)` over the 3×3 Gauss points, computed in Python from the nodal
positions. This **deliberately duplicates the C++ test rather than reusing it**, so the two can disagree
— which is exactly what settled §4. A first version used the signed area of the four corner nodes
instead and put the fold at t = 0.344; that is wrong, because the mid-side nodes of a biquadratic quad
fold well before the corner quadrilateral does. The integration-point figure agrees with the C++
detector to six digits.

### 3.2 Rejection cannot recover

**No detection:** the fold is at t ≈ 0.1565 and the run proceeds to t = 1 without complaint, `min detJ`
marching steadily further negative. This is the pre-detector behaviour and the reason the detector
exists: `J` as pyoomph computes it is `sqrt(det(g_ab))`, non-negative by construction, so an inside-out
element integrates perfectly happily.

**Detection plus adaptive dt:** 189 rejections, `dt` driven from 6e-3 to the 1e-12 floor, and `t` frozen
at 0.156540 — the fold time to six digits. The run dies.

```
### step     t=0.156540  min detJ=-9.8388e-13  inverted elems=1
### step     t=0.156540  min detJ=-2.4712e-12  inverted elems=2
Tried to reduce dt to 7.27596e-13 which is less than the minimum dt (1e-12).
```

**This is not a tuning failure and no threshold fixes it.** The deformation is prescribed as a function
of `t`, so the fold sits at a fixed *time*. Halving `dt` only makes `t` approach that time more slowly;
the mesh at t = 0.1565 + ε is folded for every ε > 0. The adaptive loop is being asked to solve a problem
it structurally cannot: it can only change how far it steps, and the obstacle is not a step length.
**That is the whole argument for remeshing as the response** — not an optimisation here, but the only
available one.

### 3.3 Transient inversions, which shape the design

With detection on but `dt` fixed, the run aborts at t = 0.12 — *before* the fold, at a time when the
converged mesh is perfectly valid. The inversion is raised on an intermediate Newton iterate: the first
residual assembly of the step has the top boundary already at its new prescribed position while the
interior is still at the old one, and that transient configuration has a squashed layer under the notch
that is briefly inside out.

Adaptive `dt` self-corrects this (a smaller step means the first iterate is closer to the answer), which
is why 412 of the rejections in the successful run of §4.3 came from iterates that were never solutions.
**A trigger that remeshes on the *first* inversion would therefore remesh hundreds of times**, each one
an interpolation that costs accuracy, to cure something a smaller step already cures.

---

## 4. `GmshTemplate.fix_2d_orientation` — implemented

Running `remesh,detect,adaptive` before this fix died shortly after the first remesh, with all 128
elements of the *freshly remeshed* mesh reported as inverted and `dt` driven to the floor.

### 4.1 Root cause

Gmsh orients its elements after the surface normal, i.e. after the winding of the curve loop. A loop
assembled programmatically has no reason to come out one way rather than the other. Reading the signed
area of the loops that `Remesher2d` wrote over one run:

```
REMESH_000000  Curve Loop(1) = {1, 3, -4, -2}   signed area = -0.97469
REMESH_000001  Curve Loop(1) = {1, 3, -4,  2}   signed area = -0.95296
REMESH_000002  Curve Loop(1) = {1, -2, 4, -3}   signed area = +0.93485
REMESH_000003  Curve Loop(1) = {1, -3, 4, -2}   signed area = +0.92040
```

which matches the observed flip-flopping exactly: all elements inverted after remeshes 0 and 1, none
after 2 and 3.

**The Gmsh→pyoomph permutations were never the problem.** `perm = [0, 4, 1, 7, 8, 5, 3, 6, 2]` maps a
counter-clockwise Gmsh quad9 onto pyoomph's tensor ordering orientation-preservingly; it faithfully
reproduces a clockwise one as clockwise. This had presumably been latent since the remesher was written,
and was harmless for the same reason §3.2 is: `sqrt(det(g_ab))` does not care. It stopped being harmless
the moment something tested the *signed* determinant.

### 4.2 The fix

`GmshTemplate.fix_2d_orientation` (default `True`) relabels clockwise elements as they are constructed,
through `_orient_2d` at each of the five `add_*_2d_*` call sites.

Reversing a quad is a transpose of its `(s0,s1)` index pair. Reversing a triangle swaps corners 1 and 2,
which also swaps the mid-side nodes `mid(0,1)` and `mid(0,2)` while `mid(1,2)` stays put — pyoomph's
`MeshTemplateElementTriC2` uses the same convention as Gmsh's `triangle6`, which
`MeshTemplateElementTriC1::convert_for_C2_space` pins down.

Two cases need nothing of their own: Scott-Vogelius splits keep the orientation of their parent, so
flipping the parent covers all three sub-triangles; and a `mirror_mesh` half is corrected as a side
effect, mirroring being orientation-reversing. Meshes in a 3d nodal space are skipped — a surface has no
orientation to fix, the same case the C++ check skips for not having a square mapping.

A counter, `num_flipped_2d_elements`, reports what the last construction touched. Deliberately no print:
it would otherwise fire on every remesh of every script.

### 4.3 Verification

`dev_docs/examples/gmsh_orientation_matrix.py` runs two geometries × four element types × three arms, 24
cases. The geometries are a non-convex L, so the winding is not something Gmsh can silently normalise
away, and a disk, which covers the curved-boundary/macro-element path. The clockwise arm is forced with
`plane_surface(reversed_order=True)`.

```
L    tris2  reverse=False fix=True   flipped=  0  area=0.75000000 OK  int(u)=0.01325301 -   detector=ok
L    tris2  reverse=True  fix=False  flipped=  0  area=0.75000000 OK  int(u)=nan        -   detector=INVERTED
L    tris2  reverse=True  fix=True   flipped= 58  area=0.75000000 OK  int(u)=0.01325301 OK  detector=ok
```

Four things are separated here, and the middle arm is what stops the test passing vacuously: correctly
wound meshes are flipped zero times, so nothing existing changes; the clockwise control really is
rejected by the detector, so the failure mode is real; and the repaired arm's integrated area stays exact
**and** its Poisson solution matches the counter-clockwise arm to 1e-8. That pair is what rules out a
wrong mid-side permutation — a mid-node landing on the wrong edge moves both, and neither alone would
catch it, since area alone would not distinguish a wrong reversal from a right one on a symmetric
element.

On the original problem: the notch case with `remesh,detect,adaptive` now runs 207 steps to t = 1.0 with
**no inverted element at any step**, remeshing 15 times and absorbing 412 inversion rejections on the way
(§3.3 is what those 412 are).

**What it does not change.** Interface normals were the obvious way this could have silently altered
results, face elements being plausible candidates to inherit the bulk element's winding. The divergence
theorem settles it: `∫ n·x ds` over the L must equal `2·area = 1.5`, and it does in all three arms, the
inside-out control included. So normals do not follow the bulk winding, and the fix is a pure
relabelling: no existing result moves.

The orientation matrix is worth promoting from `dev_docs/examples/` into `tests/` at some point; it runs
in well under a minute and is the kind of thing that silently regresses.

---

## 5. Remeshing as the response to an inversion — designed, not implemented

### 5.1 Where a remesh may fire, and where it may not

This is the constraint everything else follows from. Three candidate places, only the last usable:

**Inside the assembly**, where the exception is raised. Mid element-loop, with the assembler holding
pointers into the mesh. Not a candidate.

**In `actions_after_newton_solve`** — where `force_remesh` is called *today*. This looks like the natural
place and is not, because oomph-lib calls it from inside `newton_solve()`, hence from inside
`adaptive_unsteady_newton_solve`, which brackets the whole thing with a flat-index dof snapshot:

```cpp
for (unsigned i = 0; i < n_dof_local; i++) dofs_current[i] = dof(i);   // before the loop
...
for (unsigned i = 0; i < ni; i++) dof(i) = dofs_current[i];            // on rejection
```

A remesh in between changes the dof count and their ordering, so the restore writes stale values into
unrelated dofs, or runs off the end of the mesh. **This is a live hazard for the existing quality-based
`RemeshWhen`, not only for anything new.** It has not been observed biting: in the test case the
inversion exception aborts before that hook is reached, and the temporal error never rose enough to
provoke an error-based rejection (a run with the tolerance tightened to 1e-7 still produced none, the
deformation being linear in `t` and therefore integrated exactly by BDF2). Recorded as read off the
source, not as a reproduction.

**In Python, after the C++ call returns.** Safe, and the only place where the state is coherent.

There is a second, independent reason the remesh cannot happen at the moment of detection: when the
exception is raised the mesh has *already folded*, so its outline may be self-intersecting and Gmsh has
nothing sensible to mesh. What we want to remesh is the last accepted configuration — and the C++
rejection path restores exactly that, dofs and time both. That restore is another thing only the C++ loop
can do, and it must happen before any remesh.

**So: detect inside, reject inside, remesh outside.**

### 5.2 The plan

`RemeshingOptions(on_inverted_element=True)` arms `set_detect_inverted_elements(True)` and registers the
domain, in the same way `on_invalid_triangulation` already registers one. The rejection stays in C++
where the state can be restored; the remesh happens in `Problem.solve()`, after the C++ call has
returned, followed by a retry of the step.

**When to escalate: not on the first inversion** (§3.3). Halving `dt` is the *correct* first response to
a transient iterate fold and is already implemented; what distinguishes a real fold is that halving stops
helping. So: keep the existing catch, count *consecutive* inversion-caused rejections, and escalate only
past a threshold. `k` consecutive rejections is exactly a `dt` reduction of `2^-k`, so the count is a
proxy for "shrinking the step is not helping". Suggested default `k = 3`. **The retry after a remesh must
use the originally requested `dt`**, not the reduced one — the reduction is precisely what was just
judged useless.

Mechanism: C++ `adaptive_unsteady_newton_solve` counts consecutive inversion rejections and, past the
threshold and only when a remesh trigger is armed, throws a distinct recoverable exception *after* the
restore block has run rather than halving again; Python's `Problem.solve()` catches it, calls
`force_remesh` on the reporting domains and retries at the original `dt`, with a retry cap so a domain
that stays folded fails loudly instead of looping. **The fixed-`dt` path needs none of the C++ work** —
the exception already propagates to Python today, so it is the same handler with the escalation logic
skipped, and it is where the whole Python handler can be built and tested.

**Fold the §5.1 hazard into the same change:** `actions_after_newton_solve` should set a pending flag
instead of calling `force_remesh`, with the Python wrapper performing it once the C++ call has returned.
That closes it for `RemeshWhen` generally. Check the interaction with
`doubly_adaptive_unsteady_newton_solve`, which does spatial adaptation after the temporal loop and so has
a second thing happening between "step accepted" and "back in Python".

**MPI:** the escalation decision has to be collective, because `force_remesh` is.
`_agree_on_domains_to_remesh` already does this for the domain set, but "did any rank see an inversion"
needs the same treatment — an inversion is inherently local to whichever rank owns the folded element,
and a rank that saw none must not sail past the escalation point while the others remesh.

### 5.3 Open decisions

* **Threshold semantics.** Consecutive-rejection count (proposed, default 3), or an explicit "`dt` fell
  below this fraction of the requested `dt`"? The count is simpler to implement; the ratio is more
  legible in a log.
* **Scope of the first change.** Whether moving the existing quality-based trigger off the
  inside-the-C++-loop path belongs in the same commit as the new option, or lands separately so the new
  option arrives in isolation.
