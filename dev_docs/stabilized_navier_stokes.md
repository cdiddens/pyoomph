# Residual-based stabilization of the Navier-Stokes equations

Status: **in the tree** as `pyoomph/equations/stabilized_ns.py`, with a small hook added to
`pyoomph/equations/navier_stokes.py` (§8). Not covered by `tests/`; the physics test suite lives in
`pyoomph_runs/SUPG/stab/` (`poiseuille.py`, `kovasznay.py`, `cavity.py`, `droplet.py`, `axisym.py`,
`backflow.py`/`bfscan.py`, `units.py`, `bench_subexpr.py`, and a `README.md` carrying the full result
tables). No C++ was touched.

`pyoomph/equations/SUPG.py` was the earlier, independent attempt (`ElementSizeForSUPG`,
`GenericStabilizationMethod`, `PSPG`, `ASGS`). It added the stabilization *alongside* the flow
equations instead of subclassing them, and its strong residual omitted the viscous term (§2); its
`ASGS.get_tau1` carried the dimensional-units bug of §5 in a dead line. **It has been removed**
(§9).

---

## 1. What was built

`StabilizedNavierStokes` subclasses `NavierStokesEquations` rather than `Equations`. That is the
single most important design decision: `NavierStokesFreeSurface`, `NavierStokesSlipLength`,
`NavierStokesContactAngle`, `ConnectVelocityAtInterface` and the azimuthal-stability machinery all
declare `required_parent_type = StokesEquations`, so subclassing keeps every one of them working,
and the ALE/GCL handling in the base `define_residuals` comes for free. The subclass overrides only

* `define_stress_tensor` — to offer the Laplace form `mu grad(u)` next to the stress form,
* `define_residuals` — `super()` first, then the stabilization terms.

Equal-order spaces need no new code: the base class's `mode="C1"` and `mode="C2"` are already
C1/C1 and C2/C2.

Switchable: `space`, `viscous_form` (`stress`/`laplace`), `stabilization`
(`SUPG`/`PSPG`/`LSIC`/`GLS`/`ASGS`/`VMS` and combinations), `tau_formula`
(`shakib`/`codina`/`tezduyar`), `tauC_formula`, `include_viscous_in_residual`, `C_I`/`c_t`/`c_C`,
`stab_factor`. Extra interface equations: `ImposedTraction`, `BackflowStabilization`,
`LSICTractionDiagnostic`, `StabilizationBoundaryFlux`.

---

## 2. Second derivatives are what makes this consistent — and they are correct

The strong momentum residual is

```
R_M = rho (d_t u + (u - u_mesh).grad u) + grad p - div(2 mu D(u)) - f - rho g
```

`div(2 mu D(u))` is `mu (div(grad(u)) + grad(div(u)))` for constant `mu`, which needs second
derivatives of the shape functions. Those arrived with the "Allow second order spatial derivatives"
commits. Before them the term had to be dropped, which is what the removed `equations/SUPG.py`'s
`GenericStabilizationMethod.get_momentum_residual` did.

**Dropping it is not harmless.** Kovasznay flow at Re = 40, C2/C2 with SUPG+PSPG+LSIC:

| | rel L2 err u @ N=48 | order | rel L2 err p @ N=48 | order |
|---|---|---|---|---|
| full `R_M` | 1.6e-05 | 3.4 | 8.3e-05 | 2.3 |
| `-div(2 mu D)` dropped | 2.1e-03 | 1.6 | 8.1e-02 | 0.7 |

A factor of 1000 in the pressure. On C1 velocities the term is elementwise zero, so the omission is
free there — which is exactly why it survives unnoticed until someone selects a quadratic velocity
space.

**The second derivatives are correct, including in axisymmetry at the axis.** This is the part worth
recording as a validation of the core. Axisymmetric Hagen-Poiseuille, `u_z = U0(1-r^2/R^2)`, has
`R_M == 0` *including* the explicit `1/r` and `1/r^2` terms of the axisymmetric Laplacian, and it is
exactly representable in C2. Every consistent scheme (PSPG, SUPG+PSPG+LSIC, GLS, VMS on C2/C2)
returns roundoff (6e-09 … 2e-08), and the first element column at `r < h` is no worse than the
column at the wall. So `div(grad(u))` and `grad(div(u))` reproduce the axisymmetric operator
including the terms that only cancel numerically near `r = 0`.

The **free surface at the axis on equal-order elements is a different matter.** Splitting the static
axisymmetric hemisphere's pressure error into a near-axis and a far column:

| | L2 pressure error | at the axis | away from it |
|---|---|---|---|
| C1/C1 Galerkin (unstable) | 3.97e-02 | **1.14e-01** | 3.45e-02 |
| C1/C1 PSPG | 2.75e-03 | 2.63e-03 | 2.76e-03 |
| C2/C1 Galerkin | 1.95e-07 | 2.18e-07 | 1.94e-07 |

The checkerboard of the inf-sup-unstable pair **concentrates at the axis**, 3.3x worse there than
elsewhere. PSPG removes the concentration completely — 43x smaller at the axis and no remaining
distinction between the two regions. So the answer to "does an axisymmetric free surface misbehave at
the axis in C1/C1" is yes, and PSPG is what fixes it. On C2 spaces the axis is unremarkable either
way.

---

## 3. What the tests establish about the physics

Full tables in the suite's `README.md`. In brief:

* **Poiseuille consistency.** `u` quadratic, `p` linear, both exactly in C2/C1 and C2/C2, and it
  solves the *Navier*-Stokes equations because `u.grad u == 0`. Every consistent variant returns
  roundoff; the metric's floor is 1e-08 because the squared error bottoms out at 1e-16.
  Unstabilized C2/C2 is so ill-conditioned that PARDISO reports a backward error of 3e+03 and Newton
  never converges — the equal-order pair without PSPG is unsolvable, not merely inaccurate.
* **Lid-driven cavity, Re = 1000, vs Ghia et al. (1982).** PSPG cuts a normalized checkerboard
  indicator from 1.167 to 0.033, i.e. down to the Taylor-Hood level; LSIC then cuts the L2 norm
  of div(u) from 4.28e-01 to 2.66e-01. Under refinement, unstabilized C1/C1 *stops converging* (Newton diverges at
  64x64) while stabilized C1/C1 converges to Ghia at roughly half the effective resolution of
  Taylor-Hood.
* **Free surface.** Static hemisphere: `p = 2 sigma/R` to 7 digits on C2 spaces; volume conserved to
  1e-14. Oscillating droplet: Lamb (2D) and Rayleigh (axisymmetric) n=2 periods recovered to
  ~1 %. Unstabilized C1/C1 fails outright on the 2D free surface and PSPG rescues it.
* **Taylor-Hood should not be stabilized** unless convection demands it. It is inf-sup stable
  already, and the added terms cost ~2x in pressure error and ~4x in the L2 norm of div(u).
  Isolated: LSIC alone is harmless, PSPG alone costs a little, SUPG alone quadruples it.

---

## 4. Neumann boundaries: the ten points

This was the question the exercise was set to answer; the suite's `README.md` has the measurements.
Condensed:

1. Residual-based stabilization needs **no companion surface term**. The perturbed test function
   multiplies `R_M`, which vanishes for the exact solution, so the natural BC is untouched.
2. …provided `R_M` is complete. An incomplete `R_M` hurts **most at an open boundary**: the same
   inconsistency that only pollutes the pressure at a Dirichlet outflow corrupts the velocity
   itself there (9.0e-03 vs 1.0e-08).
3. `stress` vs `laplace` decides what "do nothing" means. Stress form + zero traction is *not*
   satisfied by Poiseuille (1.7 % velocity error); Laplace form is exact. For a free surface only
   `stress` is admissible. Independent of stabilization.
4. LSIC adds `tau_C rho (div u)` to the effective normal traction — 0.2–0.3 % of the physical
   traction on the oscillating droplet, zero without LSIC. Generally: **any bulk term written against
   `grad(w)` writes its own footprint into the natural BC.**
5. PSPG keeps *global* mass conservation exactly (`q = 1` kills `grad q`) but not local. Free-surface
   volume drift is unchanged by it.
6. The SUPG weight `tau a.grad w` does not vanish at a Neumann boundary the way it does at a no-slip
   wall, so the stabilization is fully active exactly where the solution is least constrained.
7. `grad()` on an interface is the **surface** gradient. A strong residual assembled on a boundary
   must be wrapped in `evaluate_in_domain(..., parent_domain)` or the normal derivatives are silently
   missing.
8. **Backflow through an open boundary is a surface instability and no bulk term addresses it.**
   Without `BackflowStabilization` a pulsatile backward-facing-step flow dies at every Reynolds
   number from 1000 to 5000, always with the boundary energy influx exploding (4–17 against a healthy
   0.038); with `beta = 1` it survives and the peak kinetic energy changes by 1e-03. The term is not
   consistent, so use the smallest `beta` that works.
9. PSPG does not remove the constant-pressure nullspace; it still has to be pinned.
10. On a moving mesh both the SUPG streamline direction and the cell Reynolds number must use
    `u - u_mesh`, and `h` the current deformed size.

Point 3 in particular is worth knowing independently of stabilization: pyoomph's
`StokesEquations` uses the stress form (`define_stress_tensor` returns `2 mu sym(grad u) - p I` and
tests it against `grad(u_test)`), so **every "do nothing" outflow in pyoomph imposes
`n.(2 mu D) - p n = 0`, whose tangential part is not zero for a developed profile.** A truncated
outflow that is expected to pass a Poiseuille profile through unchanged needs either the Laplace form
or an explicitly imposed traction.

---

## 5. Two dimensional-units traps

Both were found only by writing the same problem in mm / ms / mPa.s with non-unity scalings
(`units.py`); both are invisible in a nondimensional problem.

**`timestepper_weight(1,0,"BDF1")` is the *nondimensional* 1/dt.** Any tau of the form

```python
tau = 1/sqrt((c_t*timestepper_weight(1,0,"BDF1"))**2 + (2*U/h)**2 + (C_I*nu/h**2)**2)
```

mixes `1/s^2` with a pure number, and pyoomph rejects it outright
("Adding/subtracting different units \[second^(-2)\] and \[1\]"). It has to be divided by
`scale_factor("temporal")`. The removed `equations/SUPG.py`'s `ASGS.get_tau1` contained exactly this
construction — `self.alpha*timestepper_weight(1,0,"BDF1") + 4*mu/(h**2*rho)` — in a dead line:
`tau1` was overwritten immediately after, so the transient term never reached the residual and the
bug never surfaced. It went out with the file.

Using the BDF1 weight rather than an explicit `1/dt` is otherwise the right call: pyoomph zeroes the
weights in a steady solve, so the transient term of tau switches itself off instead of dividing by an
infinite dt.

**Regularization constants need a scale.** `sqrt(u.u + eps**2)` with a bare `eps` is the same unit
error. It must be `eps * scale_factor("velocity")`, i.e. the constant is *relative*.

Note also that `evaluate_observable` returns a **dimensional** `Expression`, so `float()` on it
raises; only ratios of observables are pure numbers. This trips up every error-norm harness.

---

## 6. `subexpression()` placement, measured

`bench_subexpr.py`, interleaved A/B/A/B, median of 7 `assemble_jacobian()` calls, 4722 dofs:

| | C1/C1 SUPG+PSPG+LSIC | VMS on C1/C1 | GLS on C2/C2 |
|---|---|---|---|
| all wrapped (default) | 18.1 ms / 49 kB | 20.3 ms / 59 kB | 29.2 ms / 57 kB |
| `R_M` unwrapped | 18.2 ms / 50 kB | 19.7 ms / 73 kB | 29.0 ms / 61 kB |
| `tau_M` unwrapped | 27.2 ms / 57 kB | — | — |
| `\|u\|` unwrapped | 18.1 ms / 49 kB | — | — |
| none wrapped | 26.5 ms / 61 kB | 29.4 ms / 94 kB | 51.9 ms / 81 kB |

* **`tau_M` is the only wrap that buys runtime**, and it buys a lot: 1.5x on its own, and it accounts
  for essentially all of the 1.45–1.78x penalty of unwrapping everything. It is the deepest tree — a
  square root of a sum of squares containing `|u|` — and it multiplies three or four separate weak
  terms, so without the temporary its *derivative* is expanded once per term.
* **Wrapping `R_M` buys no runtime** (18.1 vs 18.2 ms; 3 % faster without on VMS, i.e. noise), even
  though `R_M` also appears in three or four terms and squared in the VMS Reynolds term. It does
  shrink the generated C by ~20 % on VMS (59 vs 73 kB), so it is worth keeping for compile time, but
  the runtime intuition ("it appears N times, so wrap it") is simply wrong here.
* **`|u|` is redundant** once `tau_M` is wrapped — it lives inside it.
* The penalty grows with the tree: 1.46x for C1/C1 SUPG+PSPG+LSIC, 1.78x for GLS on C2/C2 where
  second derivatives appear in both `R_M` and the test operator.

The wrap sites are class attributes `_wrap_R` / `_wrap_tau` / `_wrap_U` (a code-generation knob, not
physics) so the benchmark can flip them without touching the residual code.

---

## 7. Loose ends

* **The module has no `tests/` coverage.** Everything is validated by the scripts in
  `pyoomph_runs/SUPG/stab/`, which are not pytest and take minutes each. The Poiseuille consistency
  check would make a good fast unit test: the exact solution lies in the C2 space, so every
  consistent variant must return roundoff on a 32x8 mesh in a couple of seconds.
* **`tau` constants are on the diffusive side.** `C_I = 4` gives `tau -> h^2/(4 nu)` in the Stokes
  limit — what the textbook formula "4 nu/h^2" literally says, but 3x the classical
  Hughes-Franca-Balestra value for Q1/Q1. Kovasznay, C1/C1 + PSPG at N = 48:

  | | rel L2 err u | order | rel L2 err p |
  |---|---|---|---|
  | `C_I = 4` | 3.51e-03 | 1.69 | 6.55e-02 |
  | `C_I = 12` | 1.97e-03 | 1.90 | 4.97e-02 |
  | `C_I = 36` | 1.28e-03 | **2.00** | 3.52e-02 |

  Only `C_I = 36` reaches the expected O(h^2) velocity convergence. The cavity points the same way
  (the primary vortex is damped ~10 % at the default). If any of this is promoted into `pyoomph/`,
  the default should be revisited.
* **`tau` is isotropic** (`h = V^(1/d)` via `cartesian_element_length_h`). On stretched elements the
  metric-tensor form `tau = (u.G.u + C_I nu^2 G:G + (c_t/dt)^2)^(-1/2)` is much better. Not
  implemented; it needs the inverse Jacobian of the element map exposed symbolically.
  Note `cartesian_element_length_h`, not `element_length_h`: the latter is the *revolved* volume in
  axisymmetry, so tau would grow like `r^(1/3)` away from the axis. It is also the *live* size, i.e.
  differentiated through the mesh dofs on an ALE mesh — see the frozen-element-size discussion in §9.
* **ASGS diverges on the free-surface case.** It differs from GLS only in the sign of the viscous
  perturbation (adjoint rather than operator) and is the less robust of the two here.
* **Variable viscosity** is behind `constant_viscosity=False`, which makes GiNaC differentiate
  through `mu` in `div(2 mu D)`. Untested.
* **JIT cache Tier-2 shadow-mode mismatches.** These scripts emit
  `*** JIT cache Tier-2 shadow-mode MISMATCH for domain__left/domain__right` throughout — the same
  pre-codegen fingerprint producing different generated code. Codegen ran in full so the results are
  unaffected, but it is a reproducible case for whoever works on
  `get_precodegen_fingerprint_text()` in `src/codegen.cpp`: sweeping many variants of one equation
  class in a single process, differing only in constructor flags, is what triggers it.
* **Same-script concurrency.** Two instances of the same script share the JIT cache directory and
  race on the `.so` files, which segfaults. Not specific to this work, but easy to hit when sweeping.

---

## 8. The hook in `navier_stokes.py`

`StokesEquations.get_stabilization_traction(normal, bulk_domain)` returns `Expression(0)`;
`StabilizedNavierStokes` overrides it. Five interface equations call it and subtract the result:
`NavierStokesNormalTraction`, `NavierStokesFreeSurface` and `StokesFlowRadialFarField` in full,
`NavierStokesSlipLength` and `NavierStokesPrescribedNormalVelocity` with only the tangential part
(their normal direction is constrained by whatever enforces no penetration, which absorbs the
footprint). `ImposedTraction` in the new module does the same.

The base returning zero is what keeps this **fully backward compatible**: for an unstabilized
`StokesEquations` the added term is identically zero and GiNaC drops it, so the generated code is
unchanged. `tests/test_stokes.py` and the `free_surface` and `droplet_spread_marangoni_and_gravity`
tutorials were re-run as a check.

Two design points worth recording:

* **`bulk_domain` is passed explicitly.** The method is called from an interface, where `grad()` is
  the *surface* gradient, so the strong residual has to be wrapped in `evaluate_in_domain`. Making
  the caller hand over the parent domain is what forces that to be right; a signature without it
  would produce a plausible-looking expression missing every normal derivative.
* **The correction defaults to off**, `natural_bc_correction=False`. It is a trade-off, not a free
  improvement. On C1/C1 Poiseuille with the exact traction prescribed at the outflow (so the
  parabola is not representable and the footprint is genuinely nonzero):

  | | rel L2 err u | rel L2 err p | p err @outflow |
  |---|---|---|---|
  | correction off, N=8/16/32 | 5.61e-02 / 1.46e-02 / 3.72e-03 | 4.57e-02 / 1.21e-02 / 3.18e-03 | 0.881 / 0.440 / 0.219 |
  | LSIC correction, N=8/16/32 | 5.61e-02 / 1.46e-02 / 3.72e-03 | 5.34e-02 / 1.62e-02 / 5.31e-03 | 0.702 / 0.348 / 0.173 |

  The error *at* the boundary drops by a uniform 21 %, but the global pressure convergence falls
  from O(h^1.9) to O(h^1.6). The velocity is unaffected to four digits, and the SUPG part of the
  correction contributes almost nothing on top of the LSIC part. On a static free surface the whole
  thing is a wash (2.74e-03 vs 2.82e-03). So: switch it on when the traction on a boundary is the
  quantity of interest — an imposed load, a measured force, a coupling to another domain — and leave
  it off when the interior field is.

  The PSPG footprint (`(tau/rho) q R.n` in the continuity equation) is deliberately *not*
  correctable through this hook. With `q = 1` the PSPG term drops out of the discrete continuity
  equation altogether, which is exactly why global mass conservation is exact; adding a boundary
  flux would break it. `StabilizationBoundaryFlux` offers it separately, for experiments.

`create_pressure_fixation` also gained `"C2"`: the equal-order C2/C2 pressure is nodal like
Taylor-Hood's, and `PressureFixationTaylorHood` pins `node_pt(0)`, a vertex node that carries every
continuous space. Equal-order pairs still need it — PSPG leaves the constant pressure mode in the
nullspace, since `grad(q)` annihilates constants. Verified on an enclosed lid-driven cavity for
C2/C1, C1/C1 and C2/C2.

---

---

## 9. Removal of `pyoomph.equations.SUPG`, and the frozen element size

The old module is gone. An AST scan over `pyoomph/`, `docs/source/tutorial/`, `tests/` and the
`pyoomph_runs/` tree found that of its five classes, `GenericStabilizationMethod`, `PSPG`, `ASGS`
and `ElementSizeFromInitialCartesianSize` had **no users at all** — `stabilized_ns` supersedes them
— and only `ElementSizeForSUPG` was used, by exactly two places:

* `pyoomph/equations/multi_component.py`, behind the `useCompoSUPG` / `useSUPG` flags. Those flags,
  `CompositionAdvectionDiffusionEquations.get_supg_tau()` and the SUPG residual term are removed.
  The feature was incomplete anyway: it raised `RuntimeError("TODO")` when combined with
  `integrate_advection_by_parts`, and `RuntimeError("SUPG does not work yet for ternary or higher
  systems")` for anything past a binary mixture.
* `docs/source/tutorial/pde/convdiffu/convdiffu_SUPG.py`, which now takes the element size from
  `var("cartesian_element_length_h")` instead. On that tutorial's `LineMesh(N=100, size=100)` both
  give exactly `h = 1`, so the tutorial's physics is unchanged; `supg.rst` was updated accordingly.

Two further references were spurious: `docs/source/tutorial/dg/convection_diffusion.py` and
`AGENTS_ADVANCED.md` both did `from pyoomph.equations.SUPG import *` "for is_DG_space", but
`is_DG_space` lives in `pyoomph/expressions/generic.py` and only arrived through the star-import
chain. Both now import what they actually use. (Watch out for this pattern in general: the deleted
star-import was also the only source of `vector` in the two tutorials, which is why they now carry
an explicit `from pyoomph.expressions import *`.)

### The frozen element size

`ElementSizeForSUPG` was **not** merely a slower `var("cartesian_element_length_h")`, even though
the note in `supg.rst` used to say so. It projected the Cartesian element measure onto a `D0` field
with `discontinuous_refinement_exponent=1` and then **pinned** it in
`before_assigning_equations_postorder`. On a moving mesh that makes the element size a *frozen*
quantity: it holds the size the element had when the equations were assigned, and contributes no
Jacobian entries with respect to the nodal positions.

`var("cartesian_element_length_h")`, which `StabilizedNavierStokes.element_h()` uses, is the live
size. On an ALE mesh it is differentiated through the mesh degrees of freedom, so `tau` couples the
momentum and mesh blocks of the Jacobian:

* **For:** it is the consistent linearization, so Newton keeps its quadratic convergence, and `tau`
  actually tracks a deforming element instead of drifting away from it. On a strongly deforming
  free surface the frozen size is simply wrong after a while, and there is no mechanism to refresh
  it short of a remesh.
* **Against:** extra Jacobian coupling. The velocity/pressure rows acquire entries in the mesh
  columns that a Galerkin discretization would not have, which matters for a block preconditioner
  or a fieldsplit that assumes the flow block is self-contained, and it makes the assembled
  Jacobian denser.

The free-surface tests here (oscillating droplet in 2D and axisymmetry, static hemisphere,
Rayleigh/Lamb periods to ~1 %) all used the live size without trouble, so the live size is the
right default. But the choice is a genuine trade-off rather than an oversight in the old module,
and it is the one thing `ElementSizeForSUPG` did that is not trivially reproducible. If it is ever
wanted back, the clean form is a `frozen_element_size` option on `StabilizedNavierStokes` that
swaps `element_h()` for a pinned `D0` projection — a dozen lines, and worth doing the moment a
block preconditioner wants the flow rows free of mesh columns.
