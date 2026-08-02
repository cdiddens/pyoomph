# Boundary-layer meshes: three ways to build one, and which survives adaptation

Written 2026-08-02 on branch `develop`, out of building the heated-cylinder tutorial
(`docs/source/tutorial/pde/adapt/cylinder.rst`). That page needed a mixed quadrilateral/triangle
mesh — quads aligned with a wall, triangles in the far field — that could then be *adapted*. Three
constructions were tried. Two produce a good mesh; only one of them still works after the second
refinement.

This is a findings document, not a design. Everything in it was measured.

---

## 1. Why a mixed mesh at all

A boundary layer is resolved far more economically by quadrilaterals aligned with the wall than by
triangles: the element can be long along the wall and thin across it, so the normal direction gets
its resolution without paying for it tangentially. The far field has no such preferred direction and
is much easier to fill with triangles. pyoomph supports both in one domain — several Gmsh surfaces
may share a single domain name, and `set_recombined_surfaces`
([gmsh.py:1306](pyoomph/meshes/gmsh.py#L1306)) recombines only the ones you name.

Whether the resulting mesh can be *adapted* is a separate question, and the one that decided the
outcome here. The mixed hanging-node machinery itself is fine — that was validated in
`mixed_adapt_validation.md`. What varies is whether the elements a given construction produces stay
solvable once they are refined.

---

## 2. The three constructions

### 2.1 Plain recombined annulus

An annulus between the cylinder and a slightly larger circle, meshed as one surface with
`mesh_mode="tris"` overridden by `set_recombined_surfaces`, with the outer box as a second surface
carrying the same domain name:

```python
annulus = self.plane_surface(*ring, holes=[wall], name="fluid")
self.set_recombined_surfaces(annulus)
...
self.plane_surface(*box, holes=[ring], name="fluid")
```

Measured: 264 quads at radius 0.54–0.95, 5640 triangles from 1.02 outwards. It works, it adapts, and
it was what the tutorial used first. Its weakness is that the layers are of uniform thickness — Gmsh
fills the annulus as it sees fit, so there is no grading towards the wall and no control over the
first layer.

### 2.2 Gmsh's `BoundaryLayer` mesh size field

Gmsh can grow the layers itself. `"BoundaryLayer"` was already in the accepted field list of
`add_mesh_size_field` ([gmsh.py:2188](pyoomph/meshes/gmsh.py#L2188)), but registering one needs
`gmsh.model.mesh.field.setAsBoundaryLayer`, which pyoomph did not expose — a boundary-layer field is
not a background mesh size field (it describes a region to *extrude* from the wall, not a size to
sample), so Gmsh treats the two separately. Hence
`set_mesh_size_boundary_layer_field` ([gmsh.py:2211](pyoomph/meshes/gmsh.py#L2211)):

```python
layer = self.add_mesh_size_field("BoundaryLayer", CurvesList=wall,
                                 Size=0.006, Ratio=1.25, Thickness=0.30, Quads=1)
self.set_mesh_size_boundary_layer_field(layer)
```

This gives the nicest initial mesh of the three, in the least code: 704 properly graded wall-normal
quads (r 0.502–0.727), thin at the wall and growing outwards, blending into 3316 triangles. The
geometry needs no annulus at all — one surface with the cylinder as a hole.

**And it does not survive adaptation.** Measured on the cylinder problem:

| adaptation steps | result |
|---|---|
| 0 | fine — Nu 6.6529, in line with the reference |
| 1 | fine, but only after raising `max_newton_iterations` |
| ≥2 | Newton fails to converge |

It is not a structural or hanging-node failure: the residual falls to ~1e-6 and then stalls, and it
degrades progressively as the stretched layers are refined. Raising the iteration cap to 30 bought
exactly one more adaptation; `globally_convergent_newton=True` bought nothing. Since the tutorial is
*about* adaptivity, that disqualified it.

Why the highly stretched elements Gmsh's extrusion produces become so much harder to solve once
refined has not been chased down. It is worth chasing: the field is the least-effort way to get a
good boundary layer, and it would be useful for non-adaptive problems today.

### 2.3 Transfinite O-grid — what the tutorial uses

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

Structured, aligned, graded by the `"Progression"` coefficient on the radial lines, and — the point —
it adapts happily through twelve refinement steps, reproducibly. More code than the boundary-layer
field, and the grading is yours to choose rather than Gmsh's.

**The corners must be passed explicitly.** `make_surface_transfinite`
([gmsh.py:689](pyoomph/meshes/gmsh.py#L689)) only takes its automatic path when `corners` is empty,
and that path re-derives node counts from the point sizes and *overrides* whatever transfinite
settings the curves already carry. Omitting them here produces

```
Surface 2 cannot be meshed using the transfinite algorithm
(non-matching number of nodes on opposite sides 0 != 1 or 27 != -1)
```

which reads like a geometry problem and is not one. That override is now stated in the function's
own comments.

---

## 3. Two bugs in the transfinite helpers

Both were found while building the O-grid, both were latent — every existing call site passes
explicit values — and both are fixed.

**`make_lines_transfinite` sized only its first line.** `numnodes` and `coeff` were assigned back
onto the *arguments* inside the per-line loop, so after the first line `numnodes` was no longer
`"auto"` and every later line silently inherited the first one's count. The two sides of a 4×1
rectangle came out `[40, 40]` instead of `[40, 10]`. Now per-line locals.

**`make_surface_transfinite` shared one surface's corners with all the others.** The auto-detected
corners were likewise assigned back onto the `corners` argument, so the second and later surfaces of
a single call received the *first* surface's corners. Now per-surface.

Also removed: a leftover `print(i, l, linfos[i][1])` that fired for every curve of every
auto-cornered surface, and a stray `#exit()`.

---

## 4. Plotting: velocity overlays fail on deeply refined meshes

Separate finding, same tutorial. On the heated cylinder's *joint*-criterion mesh, both

```python
self.add_plot("fluid/velocity", mode="streamlines", ...)
self.add_plot("fluid/velocity", mode="arrows", ...)
```

die with matplotlib's `RuntimeError: Triangulation is invalid`, raised from

```
LinearTriInterpolator.__init__ → get_trifinder() → TrapezoidMapTriFinder._initialize()
```

at [plotting.py:949](pyoomph/output/plotting.py#L949) (streams) and
[:1023](pyoomph/output/plotting.py#L1023) (arrows). Both sample the field on a regular grid, and
anything that does needs a `TriFinder`.

The triangulation is **not** geometrically broken, which is worth recording because it is the first
thing one suspects: 20 388 points, 28 636 triangles, **zero duplicate points and zero zero-area
triangles** (smallest area 2.7e-5). What the trapezoid map rejects is *overlapping* triangles — the
signature of hanging nodes, where the coarse side of a 2:1 interface spans two finer edges.
`tricontourf` walks the triangles directly and does not care; the search structure does.

Consequences worth knowing:

- **It is mesh-dependent, not a blanket incompatibility.** The Moffatt page draws streamlines happily
  on an adapted triangular mesh that also has hanging nodes. The cylinder's joint-criterion mesh,
  which refines the O-grid hardest, is the one that fails.
- **`crash_on_invalid_triangulation = False` does not rescue the plot.** It is a deliberate option
  ([plotting.py:2371](pyoomph/output/plotting.py#L2371), default `True`), but the part that fails
  sets `_has_invalid_triangulation`, and `save()` then returns early — so the *whole figure* is
  dropped, not just the overlay. In the cylinder run this produced no field images at all for one of
  the two criteria.
- **Scalar colour plots are unaffected**, so a field comparison can still be made; the tutorial's
  2×2 figure is drawn without any vector overlay for exactly this reason, and says so.

If a per-part degradation is ever wanted — drop the overlay, keep the figure — that is a small change
where `_has_invalid_triangulation` is consumed, and it is arguably what
`crash_on_invalid_triangulation = False` should already mean. It was not made here because in a
*comparison* figure an overlay that silently appears on one panel and not the other reads as a
physical difference rather than a plotting artefact.

---

## 5. Colour bars over a dark field

Minor, but it cost a few iterations. `add_colorbar` places the bar by its own extent and does not
account for the text around it, so at the default `ymargin` of 0.05:

- a **top**-placed bar has its title *above* it, outside the reported extent, and the title runs off
  the top of the figure;
- a **bottom**-placed bar has its tick labels *below* it, and they run off the bottom.

Both clip inside the saved PDF, so no amount of `pdfcrop` recovers them. The fix is per-position
margins (`ymargin` 0.15 top / 0.19 bottom in the tutorial), plus `xpos`/`length` on the bottom bar to
keep it clear of the geometry the extra margin pushed it into. `textcolor`, `textsize` and `ticsize`
are settable on the returned colorbar and need to be white over a viridis background.

One trap in the figure pipeline rather than the plotter: converting dense line art to PNG with
ImageMagick's `-colors 4` merged the black annotation text into the nearest palette entry, which was
the mesh blue — the labels looked like a plotter setting had been ignored when they were black all
along. Sampling the pixels settled it: RGB (36,62,100) at 4 colours, (4,4,4) at 48.

---

## 6. Recommendation

For a wall-bounded problem that will be **adapted**, use the transfinite O-grid (§2.3). For one that
will not, the `BoundaryLayer` field (§2.2) is less code and grades better.

The open question worth someone's time is §2.2's failure: if the boundary-layer field's elements
could be made to survive refinement, it would be the better answer in both cases.
