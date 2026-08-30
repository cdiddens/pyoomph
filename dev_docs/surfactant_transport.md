# Surfactant transport: why the conservative form is the default

Reference for `pyoomph/equations/surfactants.py`. Everything below was measured, not argued.

## 1. The testbed

Prescribed interface motion, so that the only error measured is the transport scheme's. The mesh
velocity `w` and the fluid velocity `u` are given **affine** fields on a circle (2D Cartesian) or a
sphere (axisymmetric):

```
w = a_m*x + om_m*(-y, x)        u = a_f*x + om_f*(-y, x)
```

- `a_m - a_f` is a *normal* slip, i.e. exactly what evaporation does: `(u-w)·n = j/rho`.
- `om_m - om_f` is a *tangential* slip: the nodes slide along the interface relative to the liquid,
  which is what every Laplace- or pseudo-elastically smoothed moving mesh does permanently.

Because `w` is affine and identical for every node, the discrete interface is at all times an exact
affine image of the initial one, so exact discrete reference solutions exist.

One trap that cost an afternoon: `CircularMesh(with_curved_entities=True)` + `refine_uniformly()`
places new boundary nodes on the piecewise-quadratic arc, **not** on the exact circle, and further
refinement does not reduce that (~5e-4, flat in h). The discrete normal then stops converging and an
h-study measures the mesh generator. `tests/test_surfactant_transport.py` snaps the interface nodes
onto the exact sphere after refining, for that reason.

## 2. What the old form costs

`weak(partial_t(G), v) + D*weak(grad(G), grad(v)) + weak(div(G*ui), v)` with `ui` the L2-projected
`(I-nn)u + (w·n)n`. Relative drift of `IntegralObservables(mass=var("surfconc_X"))` at t=1, BDF2,
32 boundary elements:

| case | dt=0.05 | dt=0.025 | dt=0.0125 | dt=0.00625 |
|---|---|---|---|---|
| static mesh, fluid rotates | −1.1e−11 | −1.3e−11 | −2.0e−11 | −2.2e−11 |
| **mesh slides tangentially, geometry unchanged** | −6.8e−04 | −1.7e−04 | −4.2e−05 | −1.1e−05 |
| uniform dilatation | +3.4e−04 | +8.4e−05 | +2.1e−05 | +5.3e−06 |
| normal slip (mass transfer) | +3.4e−04 | +8.4e−05 | +2.1e−05 | +4.8e−06 |
| everything at once | −8.7e−05 | −2.0e−05 | −5.0e−06 | −1.5e−06 |
| axisymmetric shrinking sphere | +1.1e−03 | +2.9e−04 | +7.1e−05 | — |

Second order in dt, i.e. it tracks the time-stepping order and no mesh refinement removes it. Row 2
is the sharp one: the interface is a circle at every instant and its length does not change, so there
is no dilatation to get wrong — only the nodes slide, and that alone costs 7e-4.

Cause: the discrete rate of change of the surface metric is not the discrete `div_s(w)`, and the ALE
correction inside `partial_t` is non-conservative advection. This is the geometric conservation law,
the same defect `GCL=True` already fixed for the bulk composition equations.

## 3. The conservative form

```python
time_derivative_of_integral(weak(G, v), scheme="BDF2_degr")
- weak(G*(u-w), grad(v))
+ D*weak(grad(G), grad(v))
```

With `v = 1` the first term is a telescoping difference of the discrete amount and the other two
vanish, so conservation is a property of the discrete system rather than of the time step. Measured
1e−14 in every row of the table above, in both coordinate systems, static and moving mesh, and the
residue is purely the Newton tolerance: at 160 steps it reads −1.76e−08 with the default
`newton_solver_tolerance=1e-8` and −4.2e−14 at 1e−12.

It is also never less accurate. Pointwise L2 error of Γ at t=1:

| case | legacy | conservative | conservative, exact normal |
|---|---|---|---|
| uniform dilatation | 4.8e−04 (O(dt²)) | **1.1e−11** | 1.1e−11 |
| mass transfer | 7.2e−03 | 7.3e−04 | 1.1e−11 |
| everything | 1.6e−03 | 1.2e−03 | 1.0e−03 |

For pure normal motion it is *exact* — the discrete statement is "Γ times the local metric is
constant", which holds nodewise — where the old form is merely second order.

### Two things deliberately not done

**No tangential projector on `(u−w)`.** `grad()` on an interface is the surface gradient, so
`grad(v)` is already exactly orthogonal to the element normal and the normal part of `u−w` cannot
contribute. Measured: with and without the projector agree to five digits.

**The normal is not smoothed.** Substituting an L2-projected C2 normal for the element's own makes
the mass-transfer case *ten times worse* (7.2e−3 vs 7.3e−4). Under mass transfer `u−w` is almost
purely normal with magnitude j/ρ, and a smoothed normal is no longer orthogonal to the element
tangent, so the projection leaves a spurious tangential slip of order (j/ρ)×(smoothing error) — which
is larger than the element's own geometric error. Note that the legacy form's velocity projection is
doing the same damage: it scores 7.2e−3, exactly the level of a smoothed normal. Removing it makes
that form worse still (1.8e−2), because it differentiates the normal; the conservative form does not
differentiate it at all, which is why it can simply drop it.

## 4. Positivity

Not decided by the transport form. On an axisymmetric apex-convergence problem (tangential flow
converging on the pole, where the ring area 2πr → 0, sweeping a tanh front onto the axis) all three
forms undershoot identically to three digits; the undershoot sits on the apex node itself, whose ring
weight is essentially zero so that it follows its neighbours freely. It appears as soon as the
compressed front is thinner than an element and is absent when it is resolved.

Remedies, at nref=3, T=2, 240 steps, exact peak 24.29:

| | min Γ | peak | peak error | mass drift |
|---|---|---|---|---|
| unstabilized | −1.50 | 26.85 | +10.6 % | −5.3e−14 |
| `stabilization="limited"`, dc 0.1 | −0.61 | 23.94 | **−1.4 %** | −6.1e−14 |
| `stabilization="artificial"`, C 0.1 | **+1.8e−02** | 16.81 | −30.8 % | −5.3e−14 |
| `variable="log"` | **+5.4e−06** | 21.73 | −10.5 % | −6.3e−14 |

- `"limited"` keeps the peak best but does not guarantee positivity, and Newton diverges above
  `dc_factor ≈ 0.3` — the same threshold `ScalarTransportStabilization.dc_diffusivity` documents for
  the bulk term.
- `"artificial"` is positive here but is *not monotone in C* (C=0.25 and C=0.05 both go negative
  where C=0.5 and C=0.1 do not): a smearing knob, not a bound.
- `variable="log"` is the only one that guarantees Γ > 0, and it costs a third of what artificial
  diffusion costs for a stronger guarantee. It needs a bounded dynamic range: with Γ spanning 1e−5
  Newton diverged even at dt/4; floored at 0.02 it runs at dt/2.

**None of them can break the conservation**, because all of them are written against `grad(v)` and
therefore vanish for the constant test function, and the log transform changes what Γ depends on but
not the structure of the two terms the constant test function sees. Measured 1e−14 for all four.

Note that on a one-dimensional interface — any 2D or axisymmetric problem — streamline diffusion
*is* isotropic diffusion (measured identical to five digits), and a crosswind term is identically
zero. `ScalarTransportStabilization`'s DC term defaults to `dc_form="crosswind"` and would therefore
contribute nothing at all on an interface curve.

## 5. The ends of the interface

Integrating the advection by parts creates `−∮ Γ(u−w)·m v dl` at the interface's own boundary.
**Omitting it is the natural zero-total-flux condition**, which is both what an insoluble surfactant
needs and what makes the conservation exact.

The legacy form differs here: it does not integrate by parts, so `∫ div_s(Γ u_P) dS` retains a
genuine advective outflow at a contact line. `DynamicContactLineEquations` Lagrange-enforces
`u_P = mesh_velocity()` at the contact line to cancel exactly that; in the conservative form there is
nothing to cancel, and that constraint is switched off via
`MultiComponentNavierStokesInterface.uses_projected_surfactant_velocity()`.

At an end point on the symmetry axis nothing is needed either way: `AxisymmetricCoordinateSystem.
integral_dx` has no `edim==0` branch, so a point domain carries 2πr and the term is identically zero
at r=0. Verified: an imposed `SurfactantEndFlux` at the axis changes the amount by 1.6e−14, while the
same flux at the equator changes it by exactly q·2πr.

## 6. Bit-compatibility of `form="legacy"`

On an evaporating axisymmetric droplet with an insoluble surfactant, the generated C code of the
interface element is **byte-identical** between the pre-refactor implementation and
`form="legacy"` (`domain__interface.c`, `domain.c`, `domain__interface__substrate.c` all diff clean),
and the amount drift agrees to all printed digits (−4.0389e−05).

## 7. Upwind DG on the interface: the bound-preserving form

`form="dg_upwind"`. Piecewise-constant `Γ` per interface element plus an upwind numerical flux on the
facets between neighbouring interface elements. Implicit in time this is the ALE finite-volume scheme
`(Γ|K|)^{n+1} − (Γ|K|)^n + Δt Σ F̂ = 0`, whose system is an M-matrix — so `Γ ≥ 0` unconditionally,
which is the guarantee §4 says nothing else here gives.

### That this works at all was the open question

DG on a *co-dimension-1* domain is written but had never been exercised: every one of the 125 tests in
`tests/test_internal_facet_fields.py` and both DG tutorials are bulk-only, and
`dev_docs/internal_facet_fields.md` says "any bulk mesh". Measured:

- `InterfaceMesh::fill_internal_facet_buffers` (`src/mesh.cpp:5975`) pairs the two interface elements
  sharing a vertex node. A closed curve of N elements gives N facets, an open one N−1 — the two free
  ends are correctly excluded, so `SurfactantEndFlux` remains the mechanism there.
- `var("normal")` on such a facet is the **in-surface conormal**, not the interface normal: measured
  `|m·x̂|` on a unit circle falls from 1.9e-3 to 3.3e-4 under one refinement, i.e. it converges to the
  curve tangent. The interface's own normal stays reachable as `var("normal", domain="..")`.
- **The 2:1 worry does not apply.** A facet of a curve is a point, and a point is shared by exactly two
  elements whatever their sizes, so `opposite_already_at_index = -1` is correct rather than a gap. On a
  Z2-adapted interface with an element-size ratio of 4, the facet count is still exactly right.
- **A field owned by the interface is not visible on its own facets.** Unlike a bulk DG field, which a
  facet element carries as external data, an interface-owned field lives in that interface element's
  internal data. `var("G")` unrestricted and `jump(G, at_facet=True)` both fail at code generation;
  `var("G", domain="+")`, `jump(G)`, `avg(G)` work. So the flux must be written
  `F = (a·m){{G}} + |a·m|/2 [[G]]` rather than in the `jump(...,at_facet=True)` shape the bulk
  convection-diffusion tutorial uses.

### Measured

Conservation, prescribed-motion testbed, all four cases of §2: **1e-15 to 1e-13**. The flux is
single-valued, so it cancels pairwise under the constant test function exactly as the continuous
conservative form does.

Apex problem (§4), `nref=3`, T=2, 240 steps, exact `∫Γ dS` = 9.4248 and `∫Γ² dS` = 91.34:

| | ∫(negative part), worst over the run | drift | ∫Γ² |
|---|---|---|---|
| CG conservative | 1.26e−01 | −1.9e−16 | 91.14 |
| **DG upwind, `D0`** | **0.0** | +9.4e−16 | 52.36 |

`D0` is exactly non-negative and 43 % low in `∫Γ²` — first order, as expected. On a shrinking mesh
(`a_m=−0.3`) the picture is the same: CG 2.7e−4 negative, DG exactly 0, both conserving to 5e−14.

A `DL` row measured here initially looked attractive (non-negative, much sharper). It was wrong: see
§9 — that run had started from a uniform `Γ`, because a DL initial condition silently keeps only the
constant mode. `DL` is refused by the constructor for that reason.

### Limits, all guarded with a message

- **3D.** `src/mesh.cpp:5988` throws for a 2-d interface. `_check_dg_is_available` catches it first.
- **`--distribute`.** `src/problem.cpp:4343` refuses any skeleton whose bulk is an `InterfaceMesh`;
  there is no halo key for facets between face elements built on the fly. Refused up front.
- **Stiffness.** Coupled to a real Navier–Stokes interface the form needs a smaller time step than the
  continuous ones — on an evaporating sessile droplet, Δt/2 diverges and Δt/5 converges, after which
  the amount is conserved to 2.9e−15. This is the scheme being first-order and non-smooth, not a
  Jacobian defect: it was confirmed by bisecting on the time step alone, with the scaling held fixed.
- **A contact line reads a discontinuous Γ through `domain=".."`** — fixed, see §11. It is only
  *unrestricted* access that fails.

## 8. Surface diffusion on a discontinuous space

The element term `D*weak(grad(G),grad(v))` couples nothing across a facet, and at order 0 it is
**identically zero**: a `D0` surfactant accepted a surface diffusivity and ignored it completely.
Measured before the fix, on a static circle with `Γ₀ = 1 + ½cos2θ`, `D = 0.05`, `t = 0.5` (the mode-2
variance must decay as `exp(-2Dk²t)` = 0.818731):

| | variance ratio |
|---|---|
| CG conservative | 0.818731 |
| DG upwind `D0`, before | **1.000000** — no diffusion at all |

The fix is the same symmetric interior penalty `PoissonEquation` uses in the bulk
(`pyoomph/equations/poisson.py:80-85`), with `stab = 1` at order 0. At order 0 both consistency terms
vanish on their own — `grad` of an elementwise constant is zero — and what is left,
`D[[Γ]][[v]]/h̄`, is exactly the two-point finite-volume flux. So the penalty there is *consistent*,
not a stabilization, which is why its coefficient is fixed rather than tunable. It also keeps the
system an M-matrix, so switching diffusion on does not cost the boundedness the upwind flux buys.

After:

| | variance ratio | error |
|---|---|---|
| CG conservative | 0.818731 | exact |
| DG upwind `D0` | 0.820830 | +0.26 % |
| DG upwind `D1` | 0.816745 | −0.24 % |

## 9. FIXED: a DL initial condition kept only the constant mode

Found while testing the above, and **not specific to surfactants or to interfaces**. Same field, same
`InitialCondition(f = 1 + ½(X²−Y²))`, measured variance right after `set_initial_condition`:

| space | bulk variance | interface variance |
|---|---|---|
| C1 | 0.105124 | 0.606629 |
| D0 | 0.093657 | 0.618353 |
| **DL** | **0.000000** | **0.000000** |
| D1 | 0.105124 | 0.606629 |

### Two bugs, not one

`Mesh::setup_initial_conditions` did have a DL branch — a midpoint value plus a finite difference
along each local direction — and it was wrong twice over.

**The Lagrangian sample buffer was never filled.** The copy loop was bounded by `xlagr.size()`, i.e.
`get_Lagrangian_midpoint_from_local_coordinate().size()`, which is the *element's* `nlagrangian()`
(`functable->lagr_dim`) — zero whenever the equations do not use Lagrangian coordinates, while the
nodes still carry `xi`. The nodal path is bounded by `nodept->nlagrangian()` instead and was fine.
So an IC written in `lagrangian_x` evaluated at the origin for **every** element: not "the slopes were
lost" but "every element got `f(0,0)`", which is why the field came out uniform with mean exactly 1.
The same trap sits in `interpolated_xi`, which loops over the element's `nlagrangian()` too — the fix
interpolates `xi` from the nodes with the element shape functions.

**The gradient modes were in the wrong basis.** The DL values are amplitudes in the `shape_at_s_DL`
basis, not a value plus d/d*s* slopes. With an *Eulerian* linear IC — which DL represents exactly —
DL gave 4.5997 where C1 and D1 both give 3.7297.

### The fix

Sample the IC on a lattice of local points and least-squares fit it onto the DL basis, reusing
`ElementModeFit` and `sample_local_coordinates` — the same fitter the adaptation and remeshing
transfer already use, and already pinned by
`tests/test_mesh_point_locator.py::test_interface_dl_reproduces_a_linear_field_across_adaptation`.
Fitting in the real basis fixes the second bug by construction, and taking the samples from the nodes
fixes the first.

Measured after, on a linear IC:

| | C1 | D0 | **DL** | D1 |
|---|---|---|---|---|
| Eulerian, interface | 15.211216 | 15.013211 | **15.211216** | 15.211216 |
| Lagrangian, interface | 14.083476 | 13.893007 | **14.083476** | 14.083476 |
| Eulerian, bulk | 3.729734 | 3.574942 | **3.728170** | 3.729734 |

DL now reproduces a linear field *exactly* on the interface. The bulk residual of 0.04 % is real and
not a bug: `CircularMesh` bulk elements are curved, so linear-in-*x* is not in the DL (linear-in-*s*)
space at all, and the least-squares fit is the best available. The original nonlinear case that
exposed this goes from 0.000000 to 0.106556 (bulk) and 0.642626 (interface).

`SurfactantTransportEquations` accepts `space="DL"` again. `D0` stays the default, because DL still
needs a limiter to be bounded (section 7).

## 10. FIXED: an interface skeleton segfaulted on destruction

Building an interface `_internal_facets_` skeleton used to exit with SIGSEGV. Backtrace:

```
pyoomph::BulkElementBase::free_element_info ()          src/elements.cpp:1156
pyoomph::BulkElementBase::~BulkElementBase ()           src/elements.cpp:1193
pyoomph::InterfaceElementPoint0d::~InterfaceElementPoint0d ()
pyoomph::InterfaceMesh::~InterfaceMesh ()               src/mesh.cpp:5924
nanobind::detail::inst_dealloc ()
subtype_dealloc (self=<FiniteElementCodeGenerator ... _name='_internal_facets_' ...>)
```

`~InterfaceMesh` deletes the elements it owns in `opposite_interior_facets`; `free_element_info` then
dereferences `this->jitcode->get_func_table()`, and by that point the `FiniteElementCodeGenerator`
that owns the `DynamicJITCode` is itself being deallocated — the skeleton mesh is destroyed *from
inside* that codegen's dealloc (frame 15).

It is a destruction-order interaction, not a defect in the DG path:

- a plain script running one DG problem exits 0;
- so does one running two DG problems, and one running a continuous problem followed by a DG one;
- `tests/test_salt_transport.py` builds 34 problems under pytest and exits 0, so it is not "many
  problems under pytest" either.

What differs is that the skeleton hangs off an *interface* codegen rather than a bulk one, so the
mesh and the JIT code it points at are released in the opposite order. The bulk skeleton never hits
it because the bulk mesh outlives its skeleton.

### The cause, and the fix

`free_element_info` read its loop bounds back out of the function table:

```cpp
for (unsigned int j = eleminfo.nodal_dim + jitcode->get_func_table()->lagr_dim;
     j < this->jitcode->get_func_table()->info_Pos.numfields; j++)
```

`DynamicJITCode` is a **nanobind-owned Python object** and the element holds a raw pointer to it, so
nothing keeps it alive for the element's destructor. At interpreter shutdown the code generator's
instance dict is cleared before the C++ mesh it owns is destroyed, and that releases the code first.

A Python-side keepalive on the mesh does **not** work, and it is worth knowing why: CPython's
`subtype_dealloc` clears the instance `__dict__` *before* calling the base `tp_dealloc`, so an
attribute on the mesh is already gone by the time `~InterfaceMesh` runs. Tried and measured — still
SIGSEGV.

The fix removes the dependency instead of trying to order it, and stores nothing per element:
`BulkElementBase::owned_nodal_coord_range()` derives the range from what the element already knows.

```cpp
const unsigned begin  = eleminfo.nodal_dim + this->nlagrangian();
const unsigned n_zeta = this->as_interface_element() ? this->dim() : 0;
return {begin, begin + this->dim() + n_zeta};
```

`nodal_dim` is already in `eleminfo`, and `nlagrangian()` is `GeomObject` state that
`fill_element_info` sets from `lagr_dim`. The generated `Pos` layout is
`[Eulerian, Lagrangian, local coordinates, zeta]`, which the generated C shows directly:

```
info_Pos.numfields=6;   coordinate_x coordinate_y lagrangian_x lagrangian_y
                        local_coordinate_1 zeta_coordinate_1
```

Two things keep that from being a guess:

* **`fill_element_info` checks the derivation** against `info_Pos.numfields`, once, where the table is
  still guaranteed alive, and throws naming `owned_nodal_coord_range()` if it disagrees. A change to
  the generated layout is then caught at the source rather than silently freeing the wrong slots.
  Exercised on 1d line, 2d quad, 2d triangle, 3d brick and 3d tet, each with an interface, plus the
  co-dimension-2 point domains that `SurfactantEndFlux` and the DG facets live on.
* **`as_interface_element()` is virtual, so it only tells the truth while the object still is one.**
  By `~BulkElementBase` the `InterfaceElementBase` sub-object is gone and it reports NULL, which would
  silently leak the zeta buffers. So `~InterfaceElementBase` calls `free_element_info()` itself; the
  later call from `~BulkElementBase` finds `eleminfo.alloced` already false and returns. This is the
  subtlety that made a first attempt cache the two numbers as members instead - which worked, but put
  8 bytes on every element for something derivable.

Why the bulk skeleton never hit it: a bulk mesh is also held by `Problem._meshdict`, so it outlives
its code generator; the interface skeleton's mesh is destroyed from inside that generator's dealloc.

Verified: the reproducer exits 0 whether the problem dies at scope exit or at interpreter shutdown,
and all eight DG tests are back on — `tests/test_surfactant_transport.py` is 31 passed, exit 0.

## 11. The surface tension is evaluated on the interface, not at its end point

`NavierStokesContactAngle`, `NavierStokesFreeSurfaceBalancedEnd` and
`MultiComponentNavierStokesInterfaceBalancedEnd` all took `surface_tension` and used it directly on the
co-dimension-2 domain they live on. For a continuous field that is the same number; for a
discontinuous one it is not readable at all, because the field lives in the interface element's
internal data. A `D0` surfactant therefore failed with `Cannot expand the field 'surfconc_…'` the
moment a contact angle was imposed.

**A `D*` field of an interface *is* readable from its end point — through `domain=".."`.** Measured,
reading a field set to `1+x+y` at the equator of a quarter circle, where the continuous answer is 2:

| space | unrestricted | `domain=".."` |
|---|---|---|
| C2 | 2.000000 | 2.000000 |
| D0 | **fails** | 2.168306 (the elementwise value there — correct for D0) |
| D1 | 2.015024 | 2.015024 |

So the fix is one line in each of the three: wrap the surface tension in
`evaluate_in_domain(sigma, "..")`. It also closes the standing `# TODO: Use surface tension projection
if present` in `NavierStokesContactAngle`, in the sense that the term is now consistent with what the
interface itself assembles.

**The test-scale correction does not apply here, and it was checked rather than reasoned.** The trap
`MultiComponentNavierStokesInterfaceBalancedEnd` documents is real but narrower than it looks: a test
scale gains a factor `1/spatial` per domain level, so a *field whose scale is stored symbolically in
terms of a test scale* comes out one length too large when re-expanded a level down. `_surf_tension`
is exactly such a field — its scale is `spatial/test_scale(velocity)` — which is why the projected
branch multiplies by `test_scale_factor("velocity")/test_scale_factor("velocity", domain="..")`.

What is moved here is only the *value* expression. The test function stays on the point domain, so
`weak(sigma, dot(m, u_test))` keeps the local measure and the local velocity test scale; and the only
scaled quantity inside `interface_props.surface_tension` is `var("surfconc_X")`, carried by the
*named* scale `surface_concentration`, which is not a test scale and does not pick up the per-level
factor. Measured on a dimensional problem (R₀ = 1 mm, σ = σ₀ − RTΓ, a real `NavierStokesContactAngle`),
comparing the full dof vector after six steps with only `navier_stokes.py` reverted:

```
n = 78   max|new-old| = 3.2e-12   relative = 1.7e-12   (Newton tolerance was 1e-10)
```

A stray factor of the spatial scale would be a relative difference of order 1, not 1e-12. Had the
*test function* been moved with `domain=".."` instead, the correction would have been required.

**Behaviour-preserving for continuous fields**, checked rather than assumed:
`docs/source/tutorial/ale/spread/droplet_spread_marangoni_and_gravity.py --quick-test` produces
byte-identical output before and after, the log timestamp aside. With the fix, an evaporating sessile
droplet carrying a `D0` surfactant and a real `NavierStokesContactAngle` runs and conserves the
surfactant to 6.4e−14.

## 12. Spatial scaling of a term whose test function belongs to another domain

**A test scale gains a factor 1/spatial per domain level.** That is the rule
`MultiComponentNavierStokesInterfaceBalancedEnd` documents for the velocity, and it applies to every
term written on a domain other than the field's own — here, the end-point flux and the DG facet flux.

It is invisible to a nondimensional test. Every test in this file except section 6 has every scale at
1, so a spurious power of `L` reads as a factor of one. Worse, `add_interior_facet_residual` does
**not** run the residual unit check (unlike `add_residual`), so a facet term has nothing standing
behind it but an explicit scale-invariance test.

Checked by solving the *same physical problem* nondimensionalised three different ways:

| | spatial = 1 mm | spatial = 10 mm | spatial = 1 m |
|---|---|---|---|
| `SurfactantEndFlux`, ΔN | −0.250000 | −0.250000 | −0.250000 |
| conservative advection, `∫Γ²` | 7061.63940616 | 7061.63940616 | 7061.63940616 |
| `dg_upwind` facet flux, `∫Γ²` | 6640.99466565 | 6640.99466565 | 6640.99466565 |

Both are exactly scale-invariant, to every digit. The reason is that the two effects cancel: the
end point's integration measure is short by one length (in Cartesian it is the dimensionless 1), and
the interface test function is long by exactly that one length. So the naive term is right.

Two consequences worth stating outright:

- **`SurfactantEndFlux` takes a flux per unit end-point length** — mol/(m s) — even in 2d Cartesian,
  where the "length" is a point. A rate in mol/s is rejected by the unit check of a dimensional
  problem and would pass silently in a nondimensional one. Now in the docstring, and pinned by
  `test_an_imposed_end_flux_does_not_depend_on_the_spatial_scale`.
- **A `test_scale_factor(f)/test_scale_factor(f, domain="..")` correction is wrong here**, though it
  is exactly what `BalancedEnd` needs. Added to the end flux "for safety" it made the answer scale as
  1/L: −250 at spatial = 1 mm against −0.25 at 1 m. `BalancedEnd` needs it because it re-expands a
  scale that was *stored symbolically* (`_surf_tension`'s scale is `spatial/test_scale(velocity)`);
  a plain field's test scale needs nothing. Reverted.

Also confirmed by this: the extra time-step restriction `dg_upwind` needs on a real coupled problem
(section 7: dt/2 diverges, dt/5 converges) is genuine stiffness and not a scaling artifact — the same
run is bit-identical at spatial scales spanning four orders of magnitude.
