# Residual-based stabilization of the scalar transport equations

Status: **in the tree** as `pyoomph/equations/stabilization.py`, wired into
`pyoomph/equations/advection_diffusion.py` and `pyoomph/equations/multi_component.py`. Covered by
`tests/test_stabilized_transport.py`. One line of C++ was touched, the return type of `div`
(§2), and two pre-existing defects were fixed along the way (§7).

This is the scalar-transport counterpart of `stabilized_navier_stokes.md`, and it is worth reading
that one first: the design, the `tau` formulas, the units traps and the `subexpression()` placement
are the same, and the shared pieces now live in one module rather than two.

---

## 1. What was built

One new module holding three things:

* the **free functions** `element_h`, `inv_dt`, `regularized_magnitude`, `tau_advective_diffusive`
  and `divergence`, shared with `stabilized_ns.py`, which was refactored to delegate to them;
* `ScalarTransportStabilization`, a settings bundle, so each equation class gains exactly *one*
  keyword argument instead of a dozen;
* `ScalarTransportEquations`, the base class of the three transport equations, which supplies
  everything except the strong residual.

The three equations each supply five hooks — `stabilized_fieldnames`, `stabilization_wind`,
`stabilization_diffusivity`, `stabilization_residual_scale`, `strong_residual` — and call
`add_stabilization_residuals()` at the end of `define_residuals`.

Terms: `SUPG`, `GLSDIFF`/`ASGSDIFF` (the diffusive part of the perturbation operator, with the
operator's own sign and with the adjoint sign respectively) and `DC` (discontinuity capturing,
crosswind or isotropic). No VMS analogue — for a scalar advected by a *given* velocity there is no
fine-scale quadratic term — and no LSIC analogue, since there is no constraint equation.

**It is a base class and not a mixin.** `Equations` is a nanobind type and nanobind permits exactly
one base, so `class Foo(Mixin, Equations)` fails at class-creation time with
`nb_type_init(): invalid number of bases!`. That is worth knowing before designing any other
cross-cutting addition to the equation classes.

## 2. `div()` of a gradient was non-commutative — fixed in `src/expressions.cpp`

Before this work, `div(grad(c))` for a scalar `c` could not be combined with a numeric constant:

```python
div(grad(c)) + 1        # add::eval(): sum of non-commutative objects has non-zero numeric term
div(grad(c))**2 + 1     # same
div(D*grad(c)) + 1      # same, for any coefficient
div(u) + 1              # fine -- a plain vector *field* was unaffected
div(grad(c)) + c        # fine -- only the *numeric* term was refused
trace(grad(grad(c))) + 1  # fine -- the identical quantity, written differently
```

That is exactly the sum a discontinuity capturing term needs, since it regularizes as
`sqrt(R^2 + eps^2)`, so the scalar Laplacian was unusable in any nonlinear expression.

**Cause.** `REGISTER_FUNCTION(div, eval_func(div_eval))` gave no return type, and GiNaC then infers
one from the first argument. `div` is *held* whenever its argument still contains an unexpanded
placeholder (`need_to_hold`, which lists `grad`), and `grad` is registered noncommutative — so the
held `div` inherited it. `div(u)` on a plain vector field never holds, which is why it was fine.
Reproduced in 12 lines against stock GiNaC, independent of pyoomph.

**Fix.** `div` is now registered `set_return_type(commutative)`, like `dot`, `trace` and
`determinant` already were. `div` lowers the rank by one, so commutative is the honest tag for the
vector argument. For a tensor argument the result is a vector and the tag is a white lie; all it
costs is GiNaC's guard against adding a bare number to an *isolated* tensor divergence, and even
that survives as soon as the vector is summed with a `grad` — measured, `divc(gradf(x)) + gradf(x)`
still reports noncommutative and still refuses `+ 1`.

The `trace(grad(...))` rewrite that this module used before the fix has been removed; the strong
residuals write `div(...)` directly again. The two are the same operator, which was worth confirming
independently: on a nonlinear flux `J = (1+c^2) grad(c) + (c, 2c)`,
`int (div(J) - trace(grad(J)))^2 / int div(J)^2` is **exactly 0.0** in Cartesian *and* axisymmetric
coordinates, metric terms included.

## 3. Correcting `stabilized_navier_stokes.md` §2: the C1 second derivative is not always zero

That document says "on C1 velocities the term is elementwise zero, so the omission is free there".
That holds on an *affine* element only. A Q1 field on a **bilinearly mapped** quad has nonzero
physical second derivatives, and most multi-component meshes are quads. Measured, with
`sqrt(<lap^2>/<|grad c|^2>)` on a Q1 field:

| mesh | value |
|---|---|
| rectangular (affine) quads | 1.5e-16 |
| bilinearly distorted quads | 3.9e-01 |

`compo_space` defaults to `"C1"` in `CompositionFlowEquations`, so neither the diffusive part of the
strong residual nor `GLSDIFF` is free there once the mesh is distorted — which on a moving mesh it
always is.

## 4. What the tests establish

**Consistency.** A manufactured solution that lies in the space *and* satisfies the PDE strongly
(so `R == 0` pointwise) must come back unchanged. Relative maximum dof difference against the
unstabilized solve, C2, 8x4 mesh:

| | `advection_by_parts=False` | `True` | `"skew"` |
|---|---|---|---|
| SUPG | 2.0e-10 | 3.9e-09 | 1.8e-09 |
| GLS | 2.1e-10 | 4.0e-09 | 1.9e-09 |
| ASGS | 6.0e-08 | 5.2e-07 | 2.3e-07 |

On C1 the same numbers are 3.5e-15, i.e. roundoff. On C2 the floor is set by the *evaluation* of the
strong residual, which contains second derivatives and cancels down to ~1e-7 relative. **ASGS
amplifies that floor by ~100x** — it differs from GLS only in the sign of the diffusive
perturbation, which makes it anti-dissipative there. `stabilized_navier_stokes.md` §7 records the
same asymmetry for the momentum version ("ASGS diverges on the free-surface case"). Prefer GLS.

**The default is off, bitwise.** `stab_factor=0` (with `dc_factor=0`) reproduces the unstabilized
dofs *exactly*, for every term including `DC`. Bitwise rather than "close", because that is what
catches a term added unconditionally or a prefactor that does not reach every contribution — the
removed `pyoomph/equations/SUPG.py` carried exactly such a bug in a dead line for years.

**It does what it is for.** 1d transport at `D = 1e-3` on 20 C1 elements, Dirichlet 0 and 1:

| | min | max (interior) | overshoot |
|---|---|---|---|
| Galerkin | **-1.409** | 0.815 | 1.4 |
| SUPG / GLS / SUPG+DC | 0.000 | 0.019 | 0 |

The unstabilized solution undershoots to -1.41 on a field whose exact solution is in [0,1].

**The strong residual mirrors what is assembled.** With a non-solenoidal wind `a = (x, 0)` the
conservative and convective forms differ by exactly `int c*div(a) = int c`, verified to 1e-10, and
`conservative_residual="auto"` picks the branch matching `advection_by_parts`.

**Dimensional units.** The whole thing is compiled once more in mm/ms with non-unity scalings. This
is the §5 trap of the flow document — `timestepper_weight` is the *nondimensional* `1/dt`, and a bare
`eps` under a square root is a unit error — and the discontinuity capturing term adds two more
instances of it, in the regularizations of `|grad c|` and `|R|`.

## 5. The interface physics is untouched, and that is measured, not argued

The requirement was that switching a stabilization on must not perturb the Marangoni stress, the
kinematic boundary condition, mass transfer, latent heat, surfactant transport or the contact-angle
conditions. The structural reason is that **no stabilization term is ever written against
`testfunction("velocity")`, `testfunction("mesh")` or any interface field** — every one of them is an
element-interior integral against the test function of the stabilized field only.

Measured directly on a two-domain evaporating-droplet problem (liquid `CompositionFlowEquations` +
gas `CompositionDiffusionEquations` + `MultiComponentNavierStokesInterface` with a mass-transfer
model, non-uniform composition/temperature/velocity so that `R != 0`), splitting the residual by dof
type with `get_dof_description()`:

| switch | rows that move |
|---|---|
| `compo_stabilization="SUPG"` | `liquid/massfrac_ethanol` and its three boundary variants — nothing else |
| `thermal_stabilization="SUPG"` | `liquid/temperature` — nothing else |
| `ns_stabilization="SUPG+PSPG"` | `liquid/pressure`, `liquid/velocity_y` and their boundary variants — nothing else |

Untouched in every case: `_kin_bc`, `_lagr_conn_*`, `masstrans_*`, `surfconc_*`, the mesh rows and
the entire gas domain.

The one term that *does* change the natural boundary condition is `DC`: it is diffusion-like and
written against `grad(v)`, so it deposits `rho_hat nu_dc (n.grad c)` on the boundary. It is off by
default, its footprint is exposed as `get_stabilization_flux`, and subtracting it is opt-in through
`natural_bc_correction` — the same trade-off, and the same default, as `natural_bc_correction` on the
flow side.

**The default is bitwise identical to before.** The full residual of the two-domain problem above,
with all three switches at their defaults, is `numpy.array_equal`-identical to the same problem on
`develop`.

## 6. Closing the gap on the flow side

`MultiComponentNavierStokesInterface` is a standalone `InterfaceEquations`, not a subclass of
`NavierStokesFreeSurface`, and it never called `get_stabilization_traction` — so a flow stabilization
with `natural_bc_correction` on would have balanced surface tension against the wrong stress there,
while the plain free surface got it right. It now subtracts the traction in the same place and with
the same sign as `navier_stokes.py:799`, and the outer phase's traction against `outside_test` in the
two-phase block. `BalanceGravityAtFarField` had the same gap and is the analogue of
`NavierStokesNormalTraction`; it now subtracts the full vector, since that boundary is do-nothing
tangentially.

`MultiComponentNavierStokesInterfaceBalancedEnd` needs nothing: `NavierStokesFreeSurfaceBalancedEnd`
does not call the hook either, so the two were already consistent.

The two-phase correction is **reasoned, not measured** — it only activates with
`natural_bc_correction=True` on a stabilized outer phase, which nothing in the tutorials does.

## 7. Two things found along the way, not fixed here

**The GCL branch of `CompositionAdvectionDiffusionEquations` dropped `dt_factor` — fixed.** It used a
plain `add_dweak_dt(rho_factor*f, f_test, scheme=...)` with no `self.dt_factor`, while the two
sibling branches a few lines below both applied it, so `compo_dt_factor` was silently ignored under
`GCL=True` alone. It is now `dt_factor * time_derivative_of_integral(weak(rho*f, v))`, i.e. the
factor multiplies the derivative of the whole integral from *outside*. Not folded into the integrand:
`add_dweak_dt` builds a BDF combination of past evaluations, so a `dt_factor` sitting inside would be
taken at each history step's own value rather than scaling the term.

**This changes results for anyone who combined `GCL=True` with `compo_dt_factor != 1`.** With the
default `compo_dt_factor=1` nothing moves, which is every tutorial in the tree.

**Several `Problem`s in one output directory share the JIT cache and reuse each other's code.**
Sweeping variants that differ only in constructor flags — exactly what a stabilization comparison
does — silently gives every variant the first one's compiled equations, so they all look identical.
Giving each `Problem` its own `set_output_directory` fixes it; the tests do that and say why. This is
the same phenomenon `stabilized_navier_stokes.md` §7 records as the Tier-2 shadow-mode mismatch.

Also worth knowing when diffing two runs: `get_dof_description()` can return the mass-transfer names
in a different *order* between runs, because the mass-transfer model iterates a Python `set` of
component names. Fix `PYTHONHASHSEED` or match rows by name, not by index.

## 8. Loose ends

* **`C_I = 4` is unmeasured for scalar transport.** It is inherited from `StabilizedNavierStokes` for
  consistency, where §7 of that document measures `C_I = 36` as the only value reaching O(h^2). A
  convergence study on skew advection at high Peclet would settle it. Shipping the inherited value
  rather than guessing a different one.
* **No exact-1d `tau`.** The classical `tau = (h/2|a|)(coth Pe - 1/Pe)`, which
  `docs/source/tutorial/pde/convdiffu/convdiffu_SUPG.py` derives by hand, is deliberately not offered:
  it is 0/0 as `Pe -> 0` and needs a series expansion the symbolic layer does not have.
* **No `constant_diffusivity` escape hatch**, unlike `constant_viscosity` on the flow side. GiNaC
  differentiates `div(D*grad(c))` and `div(Jdiff)` correctly through a composition- and
  temperature-dependent `rho` and `D`, and in a mixture `rho` genuinely varies — dropping `grad(rho)`
  would make the stabilization *not* vanish for the exact solution, i.e. buy code size at the price of
  consistency. `include_diffusion_in_residual` remains, for wedge/pyramid meshes and 0d domains where
  second derivatives are unavailable at all.
* **`div(Jdiff)` tree size on a ternary-or-higher mixture** has not been measured. The accessor sums
  over all components and optionally over thermophoresis, and its divergence with variable `rho` and
  `D_ij` will be large. `_wrap_R` is on by default and is the mitigation.
* **`AdvectionDiffusionFluxInterface` cannot subtract the flux** — it is an `Equations`, not an
  `InterfaceEquations`, so it cannot reach the parent domain. Documented in its docstring.
* **The live element size now couples the transport rows to the mesh columns** on an ALE mesh, a
  coupling those rows did not have before. Relevant to anyone using a fieldsplit. The
  `frozen_element_size` idea of `stabilized_navier_stokes.md` §9 would serve both modules.
