# pyoomph — meshes

Companion to [`AGENTS.md`](../AGENTS.md). Ready-made mesh templates, hand-built
`MeshTemplate`s, unstructured `GmshTemplate` geometry, multi-domain meshes, and the
moving-mesh / remeshing machinery.

## Ready-made templates

Ready-made templates (pass to `Problem.add_mesh(...)`). `size`, `N` and `lower_left`
accept either a scalar (same in every direction) or a per-axis list, e.g.
`RectangularQuadMesh(size=[R, H], N=[20, 40], lower_left=[0, 0], name="domain")`.
**The domain-name keyword differs**: `LineMesh` and `RectangularQuadMesh` take `name=`,
while `CircularMesh`, `CylinderMesh`, `CuboidBrickMesh`, `SphericalOctantMesh` and
`PointMesh` take `domain_name=`. All default to `"domain"`.

| Class | Key kwargs | Domain/boundary names |
|---|---|---|
| `LineMesh(N=10, size=1.0, minimum=0.0)` | 1D interval | domain `"domain"`, boundaries `"left"`/`"right"` |
| `RectangularQuadMesh(size=1.0, N=10, lower_left=[0,0], split_in_tris=False)` | 2D rectangle | boundaries `"left"/"right"/"top"/"bottom"` |
| `CircularMesh(radius=1, segments="all", inner_factor=0.4, domain_name=, outer_interface="circumference")` | 2D disk | outer boundary `"circumference"`. `segments=["NE","SE"]` gives a **half**-disc (quadrant keys `"NW"/"NE"/"SW"/"SE"`), which is how you mesh half a droplet for an axisymmetric run; name the straight cut edges with `straight_interface_name=`. It has **no resolution/`N` argument** — it is one macro-element per segment, refined afterwards with `RefineToLevel`/`spatial_adapt`, or replaced by a `GmshTemplate` if you want a set element size. |
| `CuboidBrickMesh(size=1.0, N=10, lower_left=[0,0,0], domain_name=)` | 3D brick | `"left"/"right"/"front"/"back"/"top"/"bottom"` |
| `CylinderMesh(radius=1, height=1, domain_name=, ...)` | 3D | `"mantle"`, `"top"`, `"bottom"` |
| `SphericalOctantMesh(radius=1, domain_name=, ...)` | 3D | `"shell"`, `"plane_x0"`, `"plane_y0"` |
| `PointMesh(...)` | 0D single point (host for ODE-like equations that still want a spatial "location") | |

## Unstructured geometry with `GmshTemplate`

For unstructured/complex 2D/3D geometry, subclass `GmshTemplate` and override
`define_geometry()` using pygmsh-style primitives (`point`, `line`, `spline`,
`circle_arc`, `plane_surface`, ...).

## Custom meshes

Custom meshes: subclass `MeshTemplate`, override `define_geometry()`, use
`new_domain(name)`, `nondim_size(length)` (divide a dimensional length by the problem's
spatial scale — node coordinates are given nondimensionally, so `set_scaling` must come
before `add_mesh`), `add_node_unique(x[, y[, z]])` (returns a node index, and returns the
*same* index for a coordinate already added — which is how two domains come to share
nodes), then add elements by index on the domain object:
`add_line_1d_C1(n0,n1)`, `add_tri_2d_C1(n0,n1,n2)`, `add_quad_2d_C1(n0..n3)`,
`add_tetra_3d_C1(n0..n3)`, `add_brick_3d_C1(n0..n7)`, `add_point_element(n0)`. Quads and
bricks take their nodes in **lexicographic (tensor-product) order**, not cyclically around
the element: `(i,j), (i+1,j), (i,j+1), (i+1,j+1)`. Mark boundaries with
`add_facet_to_boundary(name, [nodes...])`, once per element facet (two end nodes for a 2D
edge); marking a facet once is enough for it to become a boundary of *both* domains that
share it, which is what turns it into a real interface. Second-order geometry has
`_C2` variants taking the mid-side nodes as well.

Once two domains share an interface, an `InterfaceEquations` attached to **one** side can
reach the other with `get_opposite_side_of_interface()`. Before writing such a coupling by
hand, read [`units.md`](units.md) § *On interfaces and boundaries* — the units of a term
paired with a **bulk** test function are not what the strong form suggests, and that
section carries the worked Lagrange-multiplier flux.

**Multi-domain geometry with `GmshTemplate`** is usually easier: create the shared line
once, then name it in *both* `plane_surface(...)` calls. The two surfaces then use the same
line, the mesh is conforming across it, and pyoomph makes it a real interface that an
`InterfaceEquations` can reach across:

```python
self.create_lines(p_bl, "bottom", p_br, "side_lo", p_ir, "interface", p_il, "side_lo", p_bl)
self.create_lines(p_il, "side_hi", p_tl, "top", p_tr, "side_hi", p_ir)
self.plane_surface("bottom", "side_lo", "interface", name="lower")
self.plane_surface("interface", "side_hi", "top",   name="upper")
```
`create_lines` takes an alternating chain `point, name, point, name, point, ...`; the loop
of a `plane_surface` need not be given in order.

Domain/boundary names defined by the mesh are exactly the strings used with `@`.
Interfaces between two domains (or `eqs @ "boundary_name"`) become their own
`InterfaceMesh` automatically.


## Moving meshes, free surfaces and remeshing

A moving (ALE) mesh is not a different kind of problem — it is one more equation on the
same domain. Add a mesh-motion equation from `pyoomph/equations/ALE.py` to the bulk, and
the nodal positions become unknowns like any other field:

```python
eqs  = NavierStokesEquations(dynamic_viscosity=mu, mass_density=rho)
eqs += LaplaceSmoothedMesh()                      # or PseudoElasticMesh(), HyperelasticSmoothedMesh()
eqs += NavierStokesFreeSurface(surface_tension=sigma) @ "top"   # the free surface itself
eqs += DirichletBC(mesh_y=0) @ "bottom"           # pin the mesh where it must not move
self.add_equations(eqs @ "liquid")
```

- The mesh position is the field `"mesh"` (`var("mesh")`, components `mesh_x`/`mesh_y`/
  `mesh_z`); `var("lagrangian")` is the undeformed reference position. Pin mesh components
  with an ordinary `DirichletBC(mesh_x=...)`; the value `True` means "pin at the current
  value", e.g. `DirichletBC(mesh_x=0, mesh_y=True) @ "left"` lets the left boundary slide
  vertically but not horizontally. A prescribed motion is just an expression:
  `DirichletBC(mesh_x=1+0.5*var("lagrangian")[1]*var("time")) @ "right"`.
- `partial_t(f)` defaults to `ALE="auto"`, i.e. it already subtracts the mesh velocity on a
  moving mesh. Do not hand-roll the convective correction.
- The free surface itself is an `InterfaceEquations` on a boundary
  (`NavierStokesFreeSurface`, `MultiComponentNavierStokesInterface`); contact lines are a
  further `@"boundary/corner"` interface (`NavierStokesContactAngle`,
  `DynamicContactLineEquations`).
- `ConnectMeshAtInterface(lagr_mult_prefix="_lagr_conn_", use_highest_space=False)` ties
  the meshes of two domains that share an interface — **required whenever a moving mesh
  spans more than one domain**, since each domain owns its own nodal positions there and
  they otherwise drift apart silently. Attach it to one side only. Its field-level
  counterparts are `ConnectFieldsAtInterface(fields)` (any fields, by name, or a
  `{inner: outer}` dict when the names differ) and `ConnectVelocityAtInterface(...)`
  (which also takes a `mass_transfer_rate=` so the velocities may differ by an evaporative
  flux). `ConnectVelocityAtInterface` transfers the interfacial **traction** as well as
  matching the velocity — provided both domains share the same velocity and pressure scale
  *and* test scale; with mismatched scales the traction balance is silently wrong. On an
  `FSIConnection` interface add neither, since it supplies both. On a
  `MultiComponentNavierStokesInterface` between two moving-mesh domains
  `ConnectMeshAtInterface` **is still required**, but `ConnectVelocityAtInterface` is not —
  that class carries its own velocity coupling and traction transfer;
  `EnforceVolumeByPressure(volume=...)` fixes an enclosed volume via an internal pressure.

**Remeshing (2D only).** When the mesh eventually distorts too far, pyoomph can rebuild it
from the current deformed boundary and interpolate the solution onto the new mesh. Two
things are needed, and forgetting the first is the usual failure:

```python
from pyoomph.meshes.remesher import Remesher2d   # RemeshWhen/RemeshingOptions/RemeshMeshSize
                                                 # come from pyoomph.equations.generic
mesh = MyGmshMeshTemplate()
mesh.remesher = Remesher2d(mesh)         # <- on the TEMPLATE, without this RemeshWhen does nothing
self.add_mesh(mesh)
...
eqs += RemeshWhen(RemeshingOptions(max_expansion=2, min_expansion=0.3, min_quality_decrease=0.2))
eqs += RemeshMeshSize(size=0.02) @ "right/top"   # target element size, NONDIMENSIONAL
```

`RemeshWhen` also takes its thresholds directly as kwargs, plus
`on_invalid_triangulation=True`/`on_inverted_element=True` to remesh as an emergency
measure. `RemeshMeshSize(size=...)` wants a **nondimensional** size (in units of the
`spatial` scale); a dimensional one raises a `TypeError` from `setup_remeshing_size`.
Remeshing interpolates the solution onto the new mesh, so it is not exactly conservative —
expect a small drift in a conserved quantity and monitor it with an `IntegralObservables`
entry. Remeshing is *not* the same as h-adaptivity: for adaptivity use
`SpatialErrorEstimator(...)` plus `solve(spatial_adapt=n)`/`run(..., spatial_adapt=n)`, and
`RefineToLevel`/`RefineMaxElementSize` to constrain it. The two combine.

See [`advanced.md`](advanced.md) §4 for how `activate_coordinates_as_dofs`
works underneath and for the interpolation guarantees across a remesh, and
[`examples.md`](examples.md) recipe 3 for a complete free-surface script.

