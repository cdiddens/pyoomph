# Electrostatics, electrokinetics and electrohydrodynamics

Status: **in the tree** as `pyoomph/equations/electrostatics.py` (field solvers, electrolytes,
electrostatic interface conditions) and `pyoomph/equations/electrohydrodynamics.py` (coupling into
the flow). Covered by `tests/test_electrostatics.py` and `tests/test_ehd.py`. Four pre-existing
files were touched, each in one place: `pyoomph/expressions/phys_consts.py`,
`pyoomph/expressions/units.py`, `pyoomph/equations/poisson.py` (a pure extraction), `pyoomph/equations/navier_stokes.py` (one new
keyword, `extra_stress`) and `pyoomph/equations/stabilization.py` (two new hooks). The last two both
generate byte-identical C when unused, checked by md5 rather than asserted.

Not built: magnetics, full transient Maxwell, AC electrokinetics. Reasons in §8. The per-species
SUPG limitation this document originally recorded has since been fixed, see §6.

## 1. Why the potential formulation, and nothing else

The natural first instinct for Gauss's law is the mixed form, solving for `D` and `phi` together.
pyoomph has no `H(div)` or `H(curl)` spaces — no Raviart-Thomas, no Nédélec — so that is not
available, and adding them for this was never on the table. Everything here therefore solves

    -div(eps*grad(phi)) = rho_e

with ordinary continuous Lagrange elements, and exposes `E = -grad(phi)`.

The same constraint is what keeps magnetics out (§8): a vector potential in 3D needs `H(curl)` to
avoid spurious modes.

`E` is provided as `var("electric_field")` through `define_field_by_substitution`, with an optional
real `L2` projection (`ElectricFieldProjection`) for output and error estimators.

**One trap, and it is invisible without dimensional scales.** Substituted fields are expanded as

    var(name)  ->  get_scaling(name) * <substitution>

exactly as a real field is `scale * nondim`. A substitution written *dimensionally* is therefore off
by one factor of its own scale. That is undetectable in any nondimensional test, because the scale is
1 there. The fix is to write the substitution nondimensionally by dividing by the same
`scale_factor` the expansion multiplies back in, so the two cancel identically whether or not anyone
ever registered that scale. `tests/test_electrostatics.py::test_electric_field_matches_gradient`
runs with a potential scale of 2 V and a length scale of 10 µm precisely so that the factor is not 1.

## 2. Not a subclass of `PoissonEquation`

`PoissonEquation` already implements `-div(coeff*grad(u)) = f`, which *is* Gauss's law, and
subclassing it would have brought the DG variant, the Nitsche weak-Dirichlet hook and the far-field
condition for free. It was still rejected, for one reason: it fixes `testscale=1/scale_factor(name)`
inside `define_fields`, whereas the entire cross-domain story here rests on a test scale built from a
**shared** permittivity scale (§3).

Reuse was recovered where it mattered without inheritance: the body of
`PoissonFarFieldMonopoleCondition` is now the module-level helper
`poisson.farfield_monopole_residual(...)`, called by both it and `ElectricFarFieldCondition`. A pure
extraction, no behaviour change. Incidentally that helper had **no test coverage at all** before —
nothing in the tree used the far-field condition — and now has one.

## 3. Scaling, and the trap it guards against

The potential test scale is

    test_scale(phi) = spatial**2 / (permittivity_scale * scale(phi))

built from `scale_factor("permittivity")` (or an explicit expression), **never** from the local
`self.permittivity`. This is the single most important line in the module.

Reason: `ElectricPotentialConnection` transmits the normal displacement across an interface through a
Lagrange multiplier that is added with `+l` to one side's test row and `-l` to the other's. Because
the two outward normals are anti-parallel that *is* `n.D_in = n.D_out`, with each side's own
permittivity picked up from its own bulk residual — **neither permittivity appears in the interface
equation**. But the two rows are weighted by each side's test scale, so if a gas (eps_0) and an
electrolyte (78 eps_0) build their test scale from their own permittivity, the multiplier transmits
78 times too much on one side. Nothing diverges; the answer is simply wrong.

`docs/source/tutorial/multidom/conduction.rst` documents the same trap for the potential *scale*.
Both are now guarded: `check_consistent_scaling` compares `scale_factor` and `test_scale_factor` of
the potential on the two sides and raises naming both, and
`test_inconsistent_permittivity_scale_is_rejected` checks that it does.

`set_electrostatic_scaling(problem, ...)` exists so that the natural way to set the scales is also the
correct one; it defaults the potential to the thermal voltage `R*T/F`.

**Residuals are dimensionless, including interface ones — but the two get there differently.** A
*bulk* field's test function used on an interface inherits the parent test scale with an extra
`1/spatial` per level, which is what already makes `weak(sigma, div(u_test))` dimensionless in
`NavierStokesFreeSurface` even though `sigma/P` alone has units of metre. A field defined **on** the
interface, such as the surface charge, inherits nothing and its test scale has to do the whole job by
itself: `test_scale(q) = temporal/charge_scale`, so that `d_t(q)*test_scale` is 1. Getting this wrong
is not subtle — `src/codegen.cpp` rejects a non-dimensionless residual with the offending unit named
— but it is easy to reason about backwards.

**Scales are cosmetic only within reason.** `test_two_dielectrics_scales_are_cosmetic` spans
25 mV to 300 V. A scale six orders off the physics (1 µV for a 2 V problem) makes the nondimensional
dofs ~1e6, the initial residual ~1e9, and Newton stalls above tolerance. That is a real property of
any nondimensionalisation, not something the test should paper over, so the parametrisation is
bounded and says why.

## 4. Sign conventions

Derived once, tested once, and everything else keys off them.

The bulk residual is `+weak(eps*grad(phi), grad(v))`, whose integration by parts leaves
`+<n.D, v>` with `n` the domain's **outward** normal. A pillbox on a wall with a field-free exterior
gives `n.D = -sigma_s`. So the interface term is

    -weak(sigma_s, phi_test)

i.e. **the same sign as the volumetric `-weak(rho_e, phi_test)`** — that is the mnemonic worth
remembering. In 1D with the wall at x=0 it gives `dphi/dx|_0 = -sigma_s/eps`, so a positively charged
wall has a positive zeta potential, and matching against Debye-Hückel reproduces
`sigma_s = eps*zeta/lambda_D`. `test_surface_charge_sign` runs both signs, because the magnitude
alone would not notice a flip.

Choosing the connection multiplier as `l = n.D_outside` is what makes the surface-charge term land on
the parent row with exactly this sign, so **removing the opposite domain degenerates
`ElectricPotentialConnection` continuously into `SurfaceChargeBC`**. Putting the charge on the other
row, or splitting it, gives the same `phi` but a different `l`, and the two classes would then need
opposite signs.

Momentum: stress enters as `+weak(sigma, grad(v))`, body forces as `-weak(f, v)`, so an *imposed*
traction `t` enters as `-weak(t, u_test)` — consistent with `NavierStokesNormalTraction`, whose
positive argument is an external pressure.

## 5. The three EHD routes, and the one that is silently wrong

For constant `eps`, `div(sigma_M) = rho_e*E` holds **exactly**, not approximately: `E = -grad(phi)`
is a gradient, so `(E.grad)E == grad|E|^2/2` identically and the two non-Gauss terms of the Maxwell
stress cancel. The two routes

    weak(sigma_M, grad(v))     MaxwellStressEquations
    -weak(rho_e*E, v)          ElectricBodyForceEquations

therefore differ **only** by the surface integral `<n.sigma_M, v>` on the boundary. Consequences:

* With the velocity pinned on the whole boundary the integral is eliminated and the two give the same
  velocity **and the same pressure**. Measured: max dof difference < 1e-8 relative
  (`test_stress_and_body_force_agree_when_velocity_is_pinned`).

  This corrects the folklore that the two differ by a pressure offset of `eps|E|^2/2`. They do not.
  That offset belongs to a formulation that keeps only *part* of the Maxwell stress — e.g. writing
  `f = rho_e*E` while dropping the `-|E|^2 grad(eps)/2` term for a non-uniform permittivity. With the
  full body force, both routes solve `-div(2 mu D) + grad(p) = rho_e*E` and `p` is the same `p`.
  The planning document for this work asserted the offset; the test disproved it.

* On a boundary carrying a traction — a do-nothing outlet, and above all a **free surface** — the
  integral survives, so the body-force route is missing the *entire* electric traction, normal and
  tangential. It is silent: the solve converges and the interface shape is simply wrong.
  `test_maxwell_stress_interface_restores_the_traction` measures the gap (>1e-3 relative on the dof
  vector) and shows that `MaxwellStressInterface` closes it to 1e-7, i.e. to solver tolerance.

`MaxwellStressInterface` is a separate class rather than an argument of `NavierStokesFreeSurface`
because the latter's `additional_normal_traction` is a **scalar**, and the tangential electric
traction is exactly what drives the Taylor circulation of a leaky-dielectric drop. Both write into
the same velocity test row, so they compose by addition.

Its `mode` says *which side's* Maxwell stress this equation supplies, which depends on how each bulk
is already coupled. With `MaxwellStressEquations` in **both** bulks nothing is needed at all — each
natural boundary condition already carries its own `n.sigma_M` and the jump balances by itself.

**The Maxwell footprint is physical, the stabilization footprint is not.** `get_electric_traction` is
structurally the analogue of `StokesEquations.get_stabilization_traction`, but the latter is
proportional to a residual that vanishes for the exact solution and is *subtracted* by every traction
BC, whereas this one must be *kept*. That is why `NavierStokesFreeSurface` needed no change. The
cost is that a user computing wall drag from `-p*I + 2*mu*sym(grad(u))` is now missing `n.sigma_M`;
hence the public `get_maxwell_stress(domain="..")` accessor, and the docstring that says so.

### The default is that it just works

`ElectricPotentialEquations` (and therefore every subclass — Poisson-Boltzmann, Debye-Hückel, Ohmic,
and the potential half of `PoissonNernstPlanck`) applies the Maxwell stress to a co-located
(Navier-)Stokes by itself, `add_maxwell_stress_to_momentum=True`. This is the
`ViscoelasticEquations.add_polymer_stress_to_momentum` pattern: the flow equations assemble their
momentum residual as `weak(stress, grad(v))` with the stress containing `-p*identity`, so an extra
stress is simply added with the same sign into the same row.

Consequences, all deliberate:

* An EHD problem is `NavierStokesEquations(...) + ElectricPotentialEquations(...)`. Nothing else.
* On a domain with **no** flow the flag does nothing, so pure electrostatics is unaffected. (It has
  to be checked rather than assumed: asking for a velocity test function on a domain without one is
  an error, not a no-op.)
* Combining it with an explicit `MaxwellStressEquations`, an `ElectricBodyForceEquations`, or a
  `StokesEquations(extra_stress=...)` would count the force twice, so all three clashes raise with
  the flag to set. `MaxwellStressEquations` is consequently now a specialist tool — for a different
  permittivity or potential than the field equations use, an explicit `time_scheme`, or a potential
  solved on another domain — and its docstring says so.
* `MaxwellStressInterface`'s common mode shifts: a bulk with flow now carries its own traction, so a
  free surface against a passive phase wants `"opposite_only"`, and a two-fluid interface with flow
  on both sides needs no interface term at all.

Verified dof-by-dof against the explicit `MaxwellStressEquations` (<1e-10 relative), which the tests
above have in turn tied to the body-force route.

`extra_stress` on `StokesEquations` exists because the pre-existing `stress_tensor` argument
*replaces* the Newtonian expression rather than adding to it, so it could not be used for this
without restating the solvent part. One `+=` line, `None` default, byte-identical generated C
(verified by md5 of `domain.c` before and after), and 113 existing flow tests unaffected.

## 6. Electrolytes

Four models, all interchangeable at an interface because `ElectricPotentialConnection` only requires
the base class:

| | |
|---|---|
| `PoissonNernstPlanck` | resolved Debye layer, N species with valence and mobility |
| `PoissonBoltzmannEquations` | equilibrium closure, nonlinear |
| `DebyeHuckelEquations` | its linearization; usable **without declaring any ions**, just a Debye length |
| `OhmicConductionEquations` | leaky dielectric, current-conservation form (charge only on interfaces) |

The sharpest available cross-check is that steady PNP against a reservoir **is** Poisson-Boltzmann:
in 1D `div(J_i)=0` makes each flux constant and a blocking wall makes that constant zero, so `J_i=0`
and the Boltzmann distribution follows. The two share no code — one solves a transport equation, the
other an algebraic closure. Measured mismatch at psi_0 = 4: 4.7e-3 at N=150, 6.9e-4 at N=300,
9.1e-5 at N=600, i.e. **orders 2.78 and 2.93**. It is discretisation, not physics: `c_i` is a C2
interpolant while `exp(-phi/VT)` of a C2 `phi` is not, so the two cannot coincide on a finite mesh.
The tests assert the convergence order, not an absolute floor, because the floor is a property of the
mesh.

Nonlinear PB is validated against Gouy-Chapman (`tanh(psi/4) = tanh(psi_0/4)*exp(-x/lambda)`) at
psi_0 = 4, where the linearized model is off by tens of percent.

### The leaky dielectric has two closures, and they need different bulk operators

This one only surfaced during implementation and is worth stating plainly, because choosing the wrong
pairing is silent.

**A bulk operator's natural boundary condition is a jump in whatever flux it conserves.** So:

* *Gauss-driven* (the usual EHD-drop closure). Bulk = `ElectricPotentialEquations` with a
  `conductivity` attached, i.e. it still solves Gauss's law; its natural BC is therefore a jump in
  the **displacement**, which is exactly the surface charge. `SurfaceChargeConservation` supplies
  that charge as a dynamic unknown, driven by the Ohmic current jump it builds from the conductivity.

* *Current-driven* (steady DC conduction). Bulk = `OhmicConductionEquations`, solving
  `div(sigma_c grad phi) = 0`; its natural BC is a jump in the **current**, so
  `ElectricPotentialConnection` there means current continuity and the surface charge is a derived
  algebraic quantity, not the thing that closes the potential.

Passing `surface_charge_density=` to a connection sitting on an *Ohmic* bulk therefore imposes a
spurious **current source**, not a charge. Both classes' docstrings say this, and the base
`ElectricPotentialEquations` grew an optional `conductivity` argument (which does not enter its own
equation) so that the Gauss-driven pairing is expressible at all.

Validated against the Maxwell-Wagner interfacial charge of two leaky layers in series. The sharp
case is `eps_1/sigma_1 == eps_2/sigma_2`: matched charge relaxation times mean the interfacial charge
must be **exactly** zero, which no wrong prefactor or sign survives.

### Stabilization is per-species

`ScalarTransportEquations.stabilization_wind(self)` takes no fieldname, but the Nernst-Planck wind
is per species, `a_i = u - z_i m_i F grad(phi)` — a cation and an anion drift in *opposite*
directions in the same field. Worse, expanding the conservative migration term gives an advective
part **plus** `-z_i m_i F c_i lap(phi)`, a term linear in `c_i` itself, i.e. a reaction rate. In a
thin Debye layer that rate is of order `D/lambda_D^2` and dominates every other rate in `tau`.
Sizing every species from the fluid velocity and omitting the reaction therefore left `tau` far too
large exactly where the stabilization is being asked to work — and `tau` too large is not a mild
error, see the Hele-Shaw measurement in `stabilized_scalar_transport.md`.

Both are now supplied, through two new hooks on the base class:

* `stabilization_wind_for_field(fieldname)`, defaulting to `stabilization_wind()`.
* `stabilization_reaction_rate(fieldname)`, defaulting to 0.

`tau_advective_diffusive` already had a `reaction` argument; `ScalarTransportEquations.tau` simply
never passed it. It does now, along with a new `c_r` and `reaction_eps` on the settings object. The
rate is passed as a *regularized magnitude*: `"codina"` and `"tezduyar"` add it linearly, so a
negative rate — which is exactly what the co-ion has — would inflate `tau` instead of reducing it,
and `absolute()` has a 0/0 derivative at the origin, which is where a rate proportional to the
solution sits on a uniform initial condition. That is the trap `dc_diffusivity` already records.

**Backward compatibility was the whole difficulty, and it is established by measurement, not by
inspection.** `fieldname` is threaded through `_stabilization_is_advective`, `convective_velocity`
and `stabilization_velocity_magnitude` as an *optional* argument whose `None` case routes through
`stabilization_wind()` rather than `stabilization_wind_for_field()`, so an equation overriding only
the old hook produces the identical expression tree. Nothing in the tree overrides those three (the
`convective_velocity` in `stabilized_ns.py` belongs to `NavierStokesEquations`, an unrelated
hierarchy), so widening their signatures is safe. Verified two ways: the generated `domain.c` for a
`GLS+DC` advection-diffusion is **byte-identical** before and after (md5 `23b8368c…`), and
`test_stabilized_transport.py` — whose invariant 2 is precisely "bitwise identical" — passes
unchanged.

`add_stabilization_residuals` now decides advectiveness and builds the wind *per field* rather than
once, since with a per-species wind one field can be advected while another is not. For every
equation whose fields share a wind that is the same decision and the same expression for all of
them, which is what the byte-identical check confirms.

Consequence worth knowing: a stabilized Nernst-Planck is **advective even at `wind=0`**, because
the migration drift is not zero. So a quiescent stabilized electrolyte does reference
`scale_factor("velocity")`, which a quiescent stabilized advection-diffusion does not. The natural
value is the migration drift `D/lambda_D`.

Consistency is tested against a manufactured PNP solution that lies in C2 **and** satisfies the
equations strongly, so the strong residual is identically zero and SUPG/GLS/ASGS must all return
the unstabilized answer. Construction: pick `phi` quadratic, so `phi''` is constant and Gauss forces
`c_+ - c_-` to the constant `-eps*phi''/F`; any common quadratic may be added to both species on
top. The Nernst-Planck sources are whatever is left over, supplied through `reactions`, which the
strong residual subtracts again. Measured: the manufactured solution is reproduced to 1e-14 and no
dof moves by more than 1e-8 when stabilization is switched on.

### Corrected while in there

`stab_factor` was documented as "a global prefactor on tau **and** nu_dc". It only scales `tau`;
`nu_dc` has its own `dc_factor`, and `test_stab_factor_zero_is_bitwise_identical` has always passed
both. The docstring now says so.

## 7. Constants and units

`k_Boltzmann` was the CODATA-2014 value `1.38064852e-23` while `gas_constant` and `N_Avogadro` on the
lines either side of it were already the exact post-2019 SI values, so `k_B*N_A != R` by 3.5e-8
relative. That is a **bug, not a convention**: one line was not updated. It matters here because
Poisson-Boltzmann is written interchangeably with `e/(k_B T)` and `F/(R T)`, so the inconsistency put
an unexplained floor under any test comparing the two forms. Blast radius when changed:
`grep -rn k_Boltzmann pyoomph/ tests/ docs/source/` returned **exactly one hit, the definition
itself**. Now `1.380649e-23`, and `k_Boltzmann*N_Avogadro == gas_constant` exactly, asserted in a test.

Added: `faraday_constant`, and the functions `thermal_voltage(T)`, `debye_length(eps, I, T)`,
`bjerrum_length(eps, T)` — functions rather than constants because they depend on a temperature that
is usually a field.

The charge unit was spelled `coloumb`. It is now `coulomb`, and the misspelling is **gone**, not
aliased. That is a breaking rename of a public name, made on the owner's call after checking the
blast radius: `grep -rn coloumb` over the whole tree (excluding build artefacts) found it only in its
own definition, in `elementary_charge`, and in the then-new tests — no tutorial, no demo, no other
module, no `.pyi`.

`ohm` did not exist as a unit at all -- `siemens = ampere/volt` was there, but not its reciprocal --
so resistance and resistivity, both routine in electrochemistry, printed as raw base units. Added.

Simplified output units were registered (`C`, `S`, `S/m`, `F/m`, `V/m`, `C/m^2`, `C/m^3`).
for everything this module produces: `C`, `S`, `S/m`, `F/m`, `F/m^2`, `V/m`, `C/m^2`, `C/m^3`,
`Ohm`, `Ohm m`, `C/mol`, `C m`, `S m^2/mol`, `H`, `H/m`. A survey of what a user of these modules
actually writes down was the starting point, not a guess: before this, an areal capacitance (the
Stern layer, the Lippmann relation) printed as `s^4 A^2/(kg m^4)`, and `faraday_constant` -- the
module's own constant -- printed as `Ms A/mol`.

Two rules govern what may be added:

* **Signature collision.** `unit_to_string` matches on the exact base-unit signature and takes the
  first hit in insertion order, so a new entry can only relabel a quantity whose signature is
  identical to it and to no earlier entry. All of the above are pairwise distinct and distinct from
  the existing `Pa/Pas/N/N·m/W/V/F`, and every electric one contains `ampere`, which occurs in no
  non-electric entry.
* **The label must start with a *derived* symbol.** The prefix is written in front of the whole
  numerator and binds to the first symbol *together with its exponent*, but finding a simplified
  name short-circuits the exponent search and leaves the exponent at 1. So a label beginning with an
  exponentiated base unit is silently wrong by three orders per prefix step — `"m^2/(V s)"` for ion
  mobility would print 1e-8 m²/(V·s) as "10 nm^2/(V s)", i.e. 1e-17. This is the same trap the farad
  comment in `units.py` records, seen from the other side. Ion mobility is therefore **deliberately
  not registered** and keeps its base-unit spelling, where the generic path does handle the exponent
  correctly.

Tested by round-tripping all 17 named electric units against 8 magnitudes each — the printed number
times the printed (prefixed) unit must reproduce the input — plus negative controls that `1/s`,
`Pas`, `N/m`, `Mg/m^3` and `aNm` are unaffected. A Maxwell stress printing as `Pa` is a pleasant
confirmation that the signature really is a pressure.

Still absent: `tesla` and `weber`, the two remaining SI electromagnetic named units. One line each if
magnetics is ever picked up.

## 8. Not built, and why

* **Magnetics.** Deferred by choice to keep this shippable, not by obstacle — magnetostatics through
  a scalar potential `-div(mu*grad(psi_m)) = 0` needs no new spaces and would slot in beside the
  electrostatic classes. 2D/axisymmetric MHD through a single-component vector potential likewise.
  Genuinely 3D MHD does not: it needs `H(curl)`.
* **Full transient Maxwell.** Same obstacle, and the electroquasistatic limit is what fluid dynamics
  needs.
* **AC electrokinetics / impedance.** Would follow `helmholtz.py`'s `_Re`/`_Im` split-field pattern.
* **Bulk charge relaxation** in `OhmicConductionEquations`. `tau_e = eps/sigma_c` is ns to µs against
  ms-to-s flow times: 6 to 9 orders of stiffness. The flag exists and raises `NotImplementedError`
  with that explanation rather than silently producing something that will not converge.
  `SurfaceChargeConservation` keeps its `d_t q`, which is the physically meaningful transient — the
  interfacial RC time, not the bulk one. Tested: the flag raises with that explanation.
* **`formulation="log"`** for Nernst-Planck (the Slotboom variable). Designed into the constructor
  signature but not implemented; retrofitting it would change the field names. It is the same trick,
  for the same reason, as the log-conformation viscoelastic model: it enforces positivity
  structurally and tames the exponential.

## 9. Pitfalls a user will hit

1. **`grad()` on an interface is the surface gradient.** Every accessor in both modules takes
   `domain=`, and every interface class passes `".."` or `"|.."`. The one deliberate exception is
   `ElectroosmoticSlip`, where the surface gradient *is* `E_t` — commented so nobody "fixes" it.
2. **Debye layer resolution.** `lambda_D` is 1-100 nm against a millimetric drop: 1e5 to 1e6 scale
   separation, out of reach in 2D/3D. Resolved PNP is a 1D/thin-2D validation tool; use the
   thin-double-layer models beyond that. `debye_length_ratio` is reported as a named numerical factor
   so the ratio is visible rather than assumed.
3. **Three distinct nullspaces**, each exactly singular. (a) The constant potential mode when no
   Dirichlet condition exists anywhere — remove it with **one** integral constraint across the
   *connected* domains; two is over-constrained, and this is the one that surprises people. (b) Ion
   totals under blocking walls — `NernstPlanckEquations.with_fixed_amounts`. (c) The Stokes pressure
   mode — the existing `with_pressure_fixation`.
4. **Newton and nonlinear PB.** `exp(zF phi/RT)` overflows above about 700, and Newton *transiently*
   visits potentials the solution never reaches. `exponent_limit` clamps the argument, changing the
   equation off the solution but not on it; it defaults to `None` so nobody gets a silently modified
   equation. Prefer continuation, or starting from the Debye-Hückel solution. The globally convergent
   Newton is not an option — its line search collapses to a zero step.
5. **Negative concentrations.** Nothing in a Galerkin discretisation keeps `c_i > 0`; an
   under-resolved layer goes negative and then `log`/`sqrt` diagnostics NaN.
6. **`Equations` is a nanobind type accepting exactly one base**, so no mixins. Every hierarchy here
   is single inheritance by construction.
