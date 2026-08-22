# Electrostatics, electrokinetics and electrohydrodynamics

Status: **in the tree** as `pyoomph/equations/electrostatics.py` (field solvers, electrolytes,
electrostatic interface conditions) and `pyoomph/equations/electrohydrodynamics.py` (coupling into
the flow), covered by `tests/test_electrostatics.py` and `tests/test_ehd.py`.

Pre-existing files touched: `pyoomph/expressions/phys_consts.py` and `units.py` (constants and
electric units, §7), `pyoomph/equations/poisson.py` (a pure extraction of the far-field residual),
`pyoomph/equations/navier_stokes.py` (one new keyword, `extra_stress`),
`pyoomph/equations/stabilization.py` (two new hooks, §6) and `pyoomph/materials/` (electric
properties and ions, §8). The `navier_stokes` and `stabilization` changes both generate
byte-identical C when unused, checked by md5 rather than asserted.

**Two sections say what is missing, and they mean different things.** §9 is deliberate scope: what
was decided against, and why the decision still stands. §10 is unfinished work: the log formulation
for Nernst-Planck, the parts of the public API that no test names, and the coordinate systems and
parallel modes that have never been run. Read §10 before trusting anything here outside 1D and 2D
Cartesian.

Corrections this document has already needed, kept because the pattern is instructive: it claimed a
pressure offset between the two EHD routes that does not exist (§5), that `formulation="log"` was
reserved in a constructor signature when it was not (§10.1), and it recorded a per-species SUPG
limitation that has since been fixed (§6).

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
  On a moving mesh it does so conservatively; see section 11. The Ohmic jump can be switched off with
  `bulk_currents=0` when the charge is driven by adsorption or the motion is prescribed instead.

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

## 8. The material library

`relative_permittivity` and `electric_conductivity` are now in the `possible_properties` whitelist of
`BaseLiquidProperties`, `BaseGasProperties` and `BaseSolidProperties`, and `IonProperties` sits
beside `SurfactantProperties` as "a pure liquid property that additionally does something special" —
so an ion's molar mass, diffusivity and mass fraction go through machinery that already exists, and
what it adds is the charge it carries.

**The properties are annotated, not assigned.** `self.relative_permittivity: ExpressionOrNum` with no
value, exactly as `dynamic_viscosity` has always been. Both `make_static` and
`set_by_weighted_average` iterate `possible_properties` and test `hasattr`, so assigning `None` in
`__init__` would make every material *look* like it has a permittivity and break both. A test pins
this: glycerol lists the property and does not have it.

`get_absolute_permittivity(temperature=None)` is a **method, not a property**, because `make_static`
does `setattr` over `possible_properties` and a read-only property in that set would raise on the way
through. It takes a temperature because water's permittivity varies by a third over the liquid range
(87.7 at 0 °C, 78.3 at 25 °C, 55.7 at 100 °C, Malmberg–Maryott), so a Debye length asked for at a
definite temperature must not use a symbolic field. `ElectricPotentialEquations` grew a matching
`temperature=` argument for the same reason: without one, taking the permittivity from a material
makes the equation depend on `var("temperature")`, which then has to exist as a field — a real
usability cliff for an isothermal EHD problem, and the first thing that went wrong when driving PNP
from the library.

`add_salt(cation, anion, concentration)` dissolves the pair in the stoichiometry that makes the
solution electroneutral, `|z_-|` cations to `z_+` anions, so CaCl2 gives one Ca(2+) per two Cl(-).
`get_net_charge_number()` is the check, and it is tested for the asymmetric case rather than assumed.
`electric_conductivity_from_ions()` is the Nernst–Einstein closure — deriving the leaky-dielectric
conductivity from the same table the resolved model uses is what keeps the two from silently
disagreeing about the material. Measured against the textbook for 1 mM NaCl: λ_D = 9.61 nm,
σ_c = 12.6 mS/m.

**The ions are library materials, not constructor arguments.** `pyoomph/materials/ions.py` registers
the 28 common ones with the same `@MaterialProperties.register()` decorator every other material
uses, and `get_ion("Na+")` fetches one exactly as `get_pure_liquid`/`get_surfactant` do — a fresh
instance per call, so dissolving it in one liquid cannot reach another. `add_ion`/`add_salt` accept
a name and go through the same lookup, so `water.add_salt("Na+", "Cl-", 1*milli*mol/liter)` is the
whole setup; a name that is not registered still needs an explicit `charge_number` and
`diffusivity`, which `new_ion` then puts into the same table.

The datum stored per ion is the **limiting molar conductivity** at 25 °C, and the diffusivity is
derived from it by Nernst-Einstein. One number per ion is deliberate: keeping the tabulated `D_i^0`
as a second, independent datum is what would let the transport model and the conductivity closure
disagree about the same ion. The conversion is done at the temperature it is asked for, i.e. at
`var("temperature")` unless `get_diffusivity`/`ions_from_material` is given one — the same
convention as every other property in the library, and an isothermal problem answers it the way it
already answers it for the viscosity or the permittivity:

    self.define_named_var(temperature=25*celsius)

A test pins all 28 against the CRC table at 25 °C, which is also what catches the one unit trap here:
`S cm²/mol` is `siemens*(centi*meter)**2/mol`, and the dimensionally impeccable
`siemens*centi*meter**2/mol` is a factor of 100 out.

**The temperature dependence is the solvent's, not Nernst-Einstein's.** Holding `λ^0` fixed and
letting Nernst-Einstein supply the temperature gives `D ∝ T`, i.e. +37% between 0 and 100 °C where
the measured change is a factor of five, and — worse, because it looks like a result rather than an
omission — it makes an electrolyte's conductivity *exactly* temperature independent, since the `T`
in `D` cancels the `1/T` in `σ = F²/(RT) Σ z²Dc`. The library therefore carries `λ^0` to another
temperature (and to another solvent) by the fractional Walden rule `λ^0 μ^n = const`, so that
`D_i ∝ T/μ^n` — Stokes-Einstein for `n = 1`. Measured against the multi-temperature tables over
0–45 °C:

| | constant λ⁰ | linear 2 %/K | Walden n=1 | fitted n |
|---|---|---|---|---|
| Na⁺ | −47% @0 °C | ≤5% | ≤4% | **0.4%** (n=0.94) |
| K⁺ | −45% | ≤10% | ≤8% | **3.1%** (0.90) |
| Cl⁻ | −46% | ≤8% | ≤5% | **3.5%** (0.95) |
| H⁺ | −36% | ≤1.4% | **−21%** | **2.3%** (0.63) |

Two things fall out of that table. Arrhenius fits (`λ^0 = A exp(-Ea/RT)`, not shown) are 7–12% and
were dropped — they are fitting the viscosity's non-Arrhenius shape with an Arrhenius form. And H⁺
is not a badly-fitted ion, it is a different mechanism: Grotthuss proton transfer is far less
sensitive to the solvent viscosity than dragging an ion through it, which is exactly what an
exponent of 0.63 against everyone else's ~0.94 says. Only the five ions with unambiguous
multi-temperature data carry a fitted exponent; the rest keep 1, which is a statement about missing
data rather than about them being better Stokes spheres than sodium. Above ~60 °C all of it degrades
and none of it should be believed.

Three implementation points, each of which was a bug first:

* **The correction belongs to the solvent, not to the ion.** An `IonProperties` does not know what
  it is dissolved in, so `BaseLiquidProperties.get_ion_diffusivity` is where the rule is applied,
  and `ions_from_material` and `electric_conductivity_from_ions` both go through it — otherwise the
  transport and the conductivity end up using differently corrected numbers for the same ion. The
  same rule then also transfers the aqueous `λ^0` to a non-aqueous solvent, which is crude but far
  better than the alternative: Na⁺ in glycerol comes out 737× slower than in water instead of
  identical to it.
* **`temperature=None` means symbolic, and it is not the same as being handed `var("temperature")`.**
  Substituting a temperature *field* into water's viscosity correlation fails outright — the field
  lands inside the exponent of `10^(247.8/(T-140))` and the unit machinery cannot extract a unit
  from that. So the `None` path uses the material's viscosity expression as it stands, which is fine
  because by code-generation time the field has a scale. This is the convention
  `get_absolute_permittivity` already used, and the reason it used it.
* **The exponents are `rational_num`, never floats.** A float exponent on a quantity that still
  carries units is the other place GiNaC's unit handling gives up. A test walks the registered ions
  and rejects any float.

The reference viscosity is water at 25 °C *as pyoomph's own correlation gives it* (0.890439 mPa·s),
not the 0.890 the tables were measured at: the correction has to be exactly 1 for an aqueous
solution at the table temperature, and a test pins the constant to the correlation.

**Salts are recipes, and a mixture can carry them.** `pyoomph/materials/ions.py` also registers the
common salts and strong acids as `SaltProperties`, which is a `PureSolidProperties` (a salt *is* a
solid) naming two ions. `get_salt("NaCl")` pulls those ions out of the ion library when the salt is
constructed, so a salt cannot name an ion that does not exist, and the stoichiometry is **derived
from the two charge numbers** rather than parsed out of the name: `nu+ = |z-|/g`, `nu- = z+/g`. For
every registered salt that reproduces the formula the name already carries — Na2SO4 comes out 2:1
because sulfate is divalent — and a test checks the derived molar masses against the textbook ones.

Multiplying a salt by a concentration gives a `DissolvedSpeciesComponent`, which `Mixture` accepts
alongside the solvent fractions:

    mix = Mixture(water + 20*percent*glycerol + 1*milli*molar*get_salt("NaCl"))

The dissolved species deliberately do **not** live in the fraction list. Solvent fractions must sum
to unity; a concentration is not one of those, and at 1 mM a salt is 6e-5 of the solution by mass, so
pretending it displaces some of the water would be a larger error than ignoring it.
`DissolvedSpeciesComponent.mass_fraction_in` is there for when that assumption needs checking. The
same applies to a bare ion: `c*ion` dissolves it, while `fraction*ion` keeps the mixture-component
meaning an ion inherits from `PureLiquidProperties` — the units are what tell the two apart, and a
salt accepts only the former.

Two traps this hit, both now tested:

* **A one-component "mixture" is the object you passed in.** `get_mixture_properties(water)` returns
  that same `water`, so dissolving into it put the ions into the caller's material, and the next
  `Mixture` built from the same object inherited them — seen as a KCl solution reporting the Ca2+ of
  an unrelated mixture built one line earlier. `Mixture` now gives the material its own ion table
  before dissolving. `SaltProperties.dissolve_in` likewise hands over copies of its ions.
* **A mixture has no `name`, only `components`.** Three error paths interpolated `self.name` and so
  raised `AttributeError` instead of their message, exactly when a mixture was involved.
  `MaterialProperties.describe()` is what they use now.

A mixture's permittivity is still not averaged automatically — linear mixing is a poor rule for it —
but glycerol now carries one (42.5), so `set_by_weighted_average("relative_permittivity")` is an
available answer, and the error message says so.

**Not solving for the potential at all** is `dev_docs/salt_transport.md`: one field per salt with
the ambipolar diffusivity, the ion concentrations as substitutions under these same names, and the
interface condition that keeps a salt in an evaporating liquid. That is the model to reach for when
the double layer is thin and no field is applied; the two must not be put on one domain.

**Ion names are not field names.** A chemist writes `"Na+"` and `"Cl-"`, and `add_salt` should accept
that, but a pyoomph field name may only contain letters, digits and underscores. `ion_fieldname_stem`
maps `+` → `_p` and `-` → `_m` and anything else invalid to `_`, so `"Na+"` is solved for as
`c_Na_p`. That is the name a `DirichletBC` has to use; `NernstPlanckEquations.fieldname_of` is the
way to ask rather than to guess.

The equations take a material wherever they took literals: `PoissonNernstPlanck(fluid_props=water)`
reads the ions, their valences, their diffusivities, the bulk concentrations *and* the permittivity.
The end-to-end test drives a diffuse layer entirely from the library and checks the diffuse charge
against the Grahame equation.

Interface properties gained `surface_charge_density`, `zeta_potential`, `stern_layer_capacitance`,
`double_layer_capacitance` and `surface_conductance`; `SurfaceChargeBC`, `SternLayer` and
`ElectroosmoticSlip` accept an `interface_props=` and read them. `zeta_potential` defaults to `None`
rather than 0 on purpose — `None` means "this interface is not described by a thin double layer at
all", which is a different statement from "its zeta potential happens to be zero", and
`ElectroosmoticSlip` refuses the former.

One caveat worth stating: adding these to `possible_properties` means `set_by_weighted_average()`
with no argument will mass-average a mixture's permittivity if every component defines one. Linear
mixing is not a good rule for permittivity (Bruggeman and Looyenga exist for a reason), but it is the
same coarse assumption the framework already makes for its other properties, and it only fires when
explicitly asked for.

## 9. Deliberately out of scope

Distinct from §10: these are decisions, not omissions, and each would be a separate piece of work
rather than the completion of this one.

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
  interfacial RC time, not the bulk one. Tested: the flag raises with that explanation. Since
  section 11 that `d_t q` is by default the derivative of the whole surface integral rather than a
  pointwise `partial_t`, which does not change the statement above but does change what it discretises.
* **`formulation="log"`** for Nernst-Planck. Not implemented and, contrary to what this document
  said until the materials work, **not reserved in the constructor signature either** — that claim
  was wrong when it was written. See §11 for what it would take.

## 10. Open issues

Ordered by how likely each is to bite someone, not by effort.

### 10.1 Positivity: the log formulation for Nernst-Planck

**The problem.** Nothing in a Galerkin discretisation keeps `c_i > 0`. The equilibrium profile is
`c_i = c_inf*exp(-z_i*F*phi/(R*T))`, which at `zeta = 4*RT/F` spans a factor `e^4 ≈ 55` across a few
nanometres; a C2 interpolant of that overshoots below zero as soon as the layer is under-resolved.
Once a concentration goes negative:

* the ionic strength and the Debye length go complex in any diagnostic that takes their square root;
* `tau` picks up a NaN through the same route;
* the reaction rate `z_i m_i F rho_e/eps` changes sign, so the stabilization pushes the wrong way;
* Newton usually diverges rather than recovers, because the Jacobian entry through `c_i*grad(phi)`
  has the wrong sign over the negative region.

**The fix.** Solve for `psi_i = log(c_i/c_ref)` instead — the Slotboom or log-concentration variable.
Then `c_i = c_ref*exp(psi_i) > 0` structurally, for any nodal value, on any mesh. This is the same
trick and the same justification as the log-conformation representation in
`viscoelastic_log_conformation.md`: an exponential field whose positivity is a physical invariant is
represented by its logarithm so that the invariant survives discretisation.

**What it would take.** The flux is unchanged in form,

    J_i = c_i*u - D_i*grad(c_i) - z_i*m_i*F*c_i*grad(phi)
        = c_ref*exp(psi_i) * ( u - D_i*grad(psi_i) - z_i*m_i*F*grad(phi) )

so `c_i` factors out of the whole bracket, and the equation becomes

    exp(psi_i)*( d_t(psi_i) + u.grad(psi_i) ) + div( exp(psi_i)*(...) ) = R_i/c_ref

Concretely:

1. `define_fields` defines `psi_<ion>` rather than `c_<ion>`, and supplies `c_<ion>` through
   `define_field_by_substitution` so that every consumer — `get_charge_density`, the observables, a
   `DirichletBC` written in concentrations, the output — keeps working unchanged. That substitution
   must be written **nondimensionally**, see §1.
2. `get_flux`, `strong_residual`, `stabilization_wind_for_field` and `stabilization_reaction_rate`
   all rewrite in terms of `psi`. The wind gains the `-D_i*grad(psi_i)` term, i.e. **diffusion
   becomes advective in the log variable** — which is the whole reason the formulation is
   well behaved, and also means `tau` must be resized. That is not a detail: get it wrong and the
   stabilization is inconsistent, which the manufactured-solution test would catch.
3. Dirichlet conditions and initial conditions need translating, `c -> log(c/c_ref)`, which means a
   user writing `DirichletBC(c_Na_p=cinf)` has to be intercepted or told. Intercepting is nicer and
   is what `set_Dirichlet_condition` on the substituted field would have to do.
4. `c_ref` is `scale_factor("ion_concentration")`, so the log is of a dimensionless quantity.

**Why it was not done.** It is a genuine second discretisation of the same physics, not a flag: two
code paths through `strong_residual` and four through the stabilization hooks, each needing its own
manufactured-solution test. Doing it inside the electrokinetics work would have meant shipping it
untested. It is also **not** a free improvement — the log variable makes the equations more
nonlinear (Newton on `exp(psi)` has a worse basin than on `c`), so it trades a positivity failure for
a convergence one, and the honest comparison needs both paths side by side on the same
under-resolved layer.

**Retrofitting cost.** Low, and lower than this document previously claimed. The field *names* change
(`c_Na_p` to `psi_Na_p`), but since `c_<ion>` would remain available by substitution, nothing that
reads a concentration breaks. The claim that the signature already reserved a `formulation` argument
was simply false — it never did.

### 10.2 Untested surface area

The audit below is by counting references from `tests/test_electrostatics.py` and `tests/test_ehd.py`,
so "untested" here means *no test names it at all*, not "lightly tested".

| symbol | state |
|---|---|
| `ElectroosmoticSlip` | tested, `test_ehd.py`. The Helmholtz-Smoluchowski plug flow in a straight channel with open ends is exact (1e-11 on both a quadrilateral and a triangle mesh), and is checked for the sign of `zeta*E`, for using only the tangential field under an oblique one, for the permittivity lookup, for `wall_velocity`, for `impose_no_penetration=False`, for the default Maxwell-stress coupling not disturbing it, and dimensionally in SI units against 0.347 mm/s for silica-water. Both a quadrilateral and a triangle mesh are run: writing this test is what turned up [the Gauss knot typo](#a-quadrature-typo-that-used-to-cap-2d-quadrilateral-c2-accuracy-at-1e-9-fixed), and keeping the pair guards against it returning. Still untested: curved walls, a non-uniform zeta, and any regime where the Dukhin number matters. |
| `lippmann_surface_tension`, `surface_charge_surface_tension`, `debye_huckel_surface_tension` | untested. They are one-line expressions, but the *sign* of the Lippmann relation (charging always lowers the tension) is exactly the kind of thing that should not be trusted to inspection. |
| `ElectricFieldProjection` | untested. Verified once by hand — the output columns are correct and do not duplicate the local-expression ones — but that check is not in the suite. |
| `IonFluxBC` | still untested *as a class*, but its sign convention is now pinned indirectly: the ad-/desorption bulk coupling of `SurfaceChargeConservation` uses the identical boundary term, and `tests/test_surface_charge_transport.py` asserts that what the surface gains the bulk loses. Writing that test is what turned up [the backwards sign in this class's docstring](#114-two-records-corrected-while-in-there). The stabilization footprint is still not exercised. |
| `with_fixed_amounts` | untested. It removes a genuinely singular nullspace, so a wrong sign there shows up as a solver failure rather than a wrong answer, but it is untested all the same. |
| `electric_body_force`, `ions_from_material` | untested directly; each is used through a class that is tested. (`helmholtz_smoluchowski_velocity` is now covered through `ElectroosmoticSlip`.) |
| `SternLayer` | constructor only. `ThinDielectricLayer`, its base, has the series-capacitance test. |

#### A quadrature typo that used to cap 2D quadrilateral C2 accuracy at 1e-9 (fixed)

Found while writing the electroosmotic plug-flow test, **fixed**, and recorded here because this is
where it surfaced -- it was never an electrohydrodynamics issue.

`src/thirdparty/oomph-lib/include/integral.cc`, `Gauss<2,3>::Knot`, the 3x3 rule used by every 2D
quadrilateral element, had five of its nine entries written as

    0.774596662941483        instead of        0.774596669241483

-- two digits transposed, a difference of 6.3e-9, and on the **positive** knot only. The rule
therefore kept the right total weight but stopped being symmetric, which is why nothing caught it:
an asymmetric quadrature is invisible to any test whose reference value is computed on the same mesh.
What it did instead was leave a fixed defect in the assembly. The integral of a mid-side shape
function derivative over an element came out as 7.0e-9 where it is identically zero, so a field
lying exactly in the C2 space no longer produced a zero residual, and the answer was off by ~1e-9
**however fine the mesh** -- a defect of the rule, not a discretisation error.

Measured on a plain `PoissonEquation` with a linear Dirichlet profile, whose exact solution is in
the space, before and after:

| mesh / space | before | after |
|---|---|---|
| quads, `C2` | 8.5e-10 (identical at N=4x2 and N=16x8, and under `superlu` / `umfpack` / `pardiso`) | 4.0e-15 |
| quads, `C1` | 4e-16 | 4e-16 |
| triangles, `C2` | 7e-15 | 7e-15 |

and on the electroosmotic plug flow of section 10.2, quad mesh, N=8x4: nodal spread 1.8e-8 before,
2.6e-14 after; residual of the exact plug state 6.1e-9 before, 2.6e-15 after.

The fix writes the knot once as a named constant and negates it, rather than spelling out both signs:
negation is exact in IEEE arithmetic, so the rule is now symmetric to the last bit. `Gauss<1,3>` and
`Gauss<3,3>` were already correct, as were the quadrilateral rules at the other orders; an audit of
every `Gauss<D,N>` knot and weight table against exact Legendre roots put all the others at 4e-15 or
better, so this was one table.

`tests/test_quadrature.py` is the regression guard, and tests the rules directly rather than through
any physics -- odd monomials over a symmetric domain must integrate to exactly zero, and a Poisson
solution lying in the space must be reproduced to machine precision, both across lines, quads,
triangles and bricks. Verified to fail on the pre-fix build: three of its sixteen cases go red, all
three of them the quadrilateral ones.

### 10.3 Coordinate systems, adaptivity, MPI

* **Axisymmetry is smoke-tested, not validated.**
  `test_ehd.py::test_maxwell_stress_assembles_in_axisymmetry` drives an axisymmetric box with the
  Maxwell stress and checks only that it assembles, produces a flow and stays finite — which
  establishes that `dyadic` and `identity_matrix` survive a coordinate system where the tensor is
  3x3 over a 2D mesh. There is no analytic reference behind it, and the test says so. The Taylor
  leaky-dielectric drop — small-deformation `D = (9/16)*Ca_E*Phi_T/(2+R)^2`, with the sign of the
  discriminating function distinguishing prolate from oblate — is the test that would validate the
  axisymmetric Maxwell traction *and* the whole leaky-dielectric interface at once. It was planned
  and is not written.
* **`radialsymmetric`** is exercised, by the point-charge far-field test.
* **Spatial adaptivity** is untested for these equations. `NernstPlanckEquations` registers error
  estimators on the ion gradients, which is the right criterion for a diffuse layer, but no test
  refines anything. Interface-coupled adaptivity in particular (the gas/liquid pairing under
  `enforce_interface_conformity`) has never been run.
* **MPI** is untested. Nothing in these modules is obviously distribution-sensitive — there is no
  custom assembly and no node matching — but `ElectricPotentialConnection` relies on the
  opposite-side interface pairing, which is exactly the machinery that has needed attention before
  (see `distributed_remeshing.md`, `mpi_augmented_systems.md`).
* **Transients** were barely covered — one ion-conservation check over a few steps, and
  `SurfaceChargeConservation`'s `d_t q` only ever solved in steady state, where it degenerates to
  current continuity. `tests/test_surface_charge_transport.py` closes most of that: 27 tests over a
  prescribed moving interface and a shrinking film, transient throughout. What is still steady-only is
  the *coupled* transient, i.e. `d_t q` driven by an actual Ohmic current jump rather than by
  prescribed motion or adsorption.

### 10.4 Not implemented, but named in the design

* **`FloatingElectrode`** — a conductor at an unknown uniform potential carrying a prescribed total
  charge. The construction is settled (a `GlobalLagrangeMultiplier` plus an interface multiplier, the
  `with_pressure_integral_constraint` pattern) but no code exists.
* **Charge regulation** — a surface charge that responds to the local pH through site dissociation,
  `sigma_s(phi) = -e*Gamma/(1 + (K/cH)*exp(-e*phi/(kB*T)))`. That expression is the *equilibrium limit*
  of an adsorption/desorption rate pair, so section 11.3 supplies the dynamic version of it:
  `SurfaceChargeConservation(adsorption=...)` takes a rate, per ion or as a lumped charge flux, and a
  Langmuir pair relaxes to exactly this isotherm. What is still missing is the named class for the
  equilibrium form and, more usefully, the Grahame conversion between a zeta potential and a surface
  charge, which the design specified as a `zeta_model` argument and which does not exist.

### 10.5 Meshing

Nothing here helps a user build a mesh that resolves a Debye layer. `debye_length_ratio` is reported
as a named numerical factor so the scale separation is at least visible, but the graded-mesh recipe
that would make a resolved 2D calculation practical is not written, and `dev_docs/mesh_construction.md`
covers boundary-layer meshes generically rather than for this case. Until that exists, resolved PNP
in 2D is only honest for `lambda_D/L` above roughly 1e-3.

### 10.6 Documentation

There is no tutorial chapter. The gas/liquid pairing and the EHD routes exist only as tests, which
are written to pin behaviour rather than to teach, and the API documentation is autodoc over
docstrings. A chapter covering a capacitor, a charged wall in both PB and DH, the gas/liquid pairing
and an EHD drop is the missing piece for anyone who did not write these modules.

## 11. Conservative transport on a moving mesh

Added after the question "how does the surface charge behave under evaporation?", whose honest answer
at the time was: *it is conserved in the model and not in the code*. Everything below was measured,
in `tests/test_surface_charge_transport.py`.

### 11.1 The surface charge

`SurfaceChargeConservation._advection()` always built the physically right velocity —
`u_s = (u - (u.n)n) + (w.n)n`, i.e. the charge follows the *interface* normally and the *liquid*
tangentially, which is exactly what evaporation needs. The assembly around it was not. It was

    weak(partial_t(q, ALE="auto"), q_test) + weak(div(q*u_s), q_test)

which is the form `surfactant_transport.md` had already measured and rejected. `form="conservative"`
is now the default:

    time_derivative_of_integral(weak(q, q_test), scheme) - weak(q*(u - dt_factor*w), grad(q_test))

Relative drift of the total charge at t=1, 20 steps unless stated, nref=2:

| case | legacy | conservative |
|---|---|---|
| uniform dilatation | 1.1e-03 | -7e-15 |
| **evaporation (normal slip)** | 6.8e-04 | -7e-15 |
| tangential mesh slide only | -6.8e-04 | -5e-15 |
| everything at once | -6.0e-04 | 3e-13 |

**The evaporation row is worse than it looks, and this is the sharpest reason for the change.** Under
a normal slip the legacy error does *not* converge in the time step at all:

| nref | 20 steps | 40 | 80 | 160 |
|---|---|---|---|---|
| 1 | -7.5e-04 | -1.6e-03 | -1.8e-03 | -1.9e-03 |
| 2 | +6.8e-04 | -1.7e-04 | -3.8e-04 | -4.4e-04 |
| 3 | +1.0e-03 | +1.7e-04 | -4.4e-05 | -9.8e-05 |

It converges to a nonzero limit, and that limit is second order in **h**. The cause is that
`div(q*u_s)` differentiates the element normal, which the conservative form never touches — the same
mechanism `surfactant_transport.md` section 3 records for a smoothed normal, and the reason the
surfactant module's `"strong"` form scores worse than its `"legacy"` one. Note the sign change
between 20 and 40 steps: a two-point convergence study lands on a ratio of 4 there by coincidence and
reports "second order". The test asserts the tail, not two points.

**Three deliberate omissions**, all inherited with their reasons from `surfactant_transport.md`: no
tangential projector on `u-w` (`grad` on an interface is already the surface gradient), no smoothed
normal (ten times worse under mass transfer), and no contour term at the ends — that omission *is*
the zero-end-flux condition, and `SurfaceChargeEndFlux` exists for when it is not wanted.

**`dt_factor` multiplies the mesh velocity too.** The GCL transient strongly supplies
`dt_factor*(d_t q|_E + div_s(q w))`, while the equation wants `dt_factor*d_t q|_E + div_s(q u)`, so
the flux term carries `u - dt_factor*w`. They coincide at `dt_factor=1`; at `dt_factor=0`
(`quasi_static`) this collapses to the correct steady `div_s(q u)`, where a bare `(u-w)` would have
solved `div_s(q(u-w))=0` instead. Conservation is unaffected either way, since the flux term vanishes
for the constant test function — it is *which equation* that differs.

### 11.2 The ions, and the other transport equations

`NernstPlanckEquations` had no conservative branch. `GCL="auto"` is now the default, meaning **on
whenever the mesh moves** and off on a static mesh. It is not a flat `True` because integrating the
advection by parts also changes the natural boundary condition, and on a static mesh with through-flow
that is a change for nothing; on a moving mesh it is the entire point.

A 1d film thinning to `exp(-0.5)` of its height with the liquid at rest, so nothing can leave:

| | 20 steps | 40 | 80 | 160 |
|---|---|---|---|---|
| `GCL=False` | -3.9324e-01 | -3.9341e-01 | -3.9346e-01 | -3.9347e-01 |
| `GCL=True` | -4e-14 | | -6e-14 | |

The non-conservative number is not a discretization error that a smaller step would reduce: it is
exactly `1 - L/L0`. The concentration never changes at all and the receding boundary carries the ions
away, precisely as `salt_transport.md` section 3 describes for the salt. That table applies verbatim
here, and it is now quoted in the `GCL` docstring, because **an `IonFluxBC` written for the old form
becomes a double count under GCL** — the one way to be silently wrong with this change.

`AdvectionDiffusionEquations` and `TemperatureConductionEquation` /
`TemperatureAdvectionConductionEquation` got the same `GCL` flag, opt-in (`False`) since flipping them
would move published tutorial numbers for no requested gain. Both conserve to 1e-14 with it and lose
the shrink ratio without it.

Two things specific to the temperature:

- **The conserved quantity is `rho*cp*T`**, so `rho` and `cp` sit *inside* the integral. If either
  depends on `T`, the GCL form differentiates the enthalpy density where the standard form multiplies
  `rho*cp` onto `d_t T` — a different model, not a different discretization. Identical for constant
  properties. Said in the docstring rather than left to be discovered.
- **The test overrides the conductivity as well**, for an unrelated reason worth recording: conserving
  the enthalpy of a domain shrinking to 0.61 of its size means `T` rises to 495 K, which is outside
  water's conductivity correlation, and Newton diverges at step 5. That is the material data leaving
  its range, not the scheme.

### 11.3 Ad- and desorption

`SurfaceChargeConservation(adsorption=...)` takes either an expression — a net charge flux in
C/(m^2 s), positive towards the interface — or a `{ion_name: molar_rate}` mapping, which contributes
`sum(z_i F R_i)` to the charge and takes exactly `R_i` out of a co-located `NernstPlanckEquations`.
Materials can carry either, as `surface_charge_adsorption_rate` and `ion_adsorption_rate` on
`BaseInterfaceProperties`, and `SurfaceChargeConservation` now takes `interface_props=` like
`SurfaceChargeBC`, `SternLayer` and `ElectroosmoticSlip` already did. It also finally reads
`surface_conductance`, which had been sitting on the material unread since it was added.

The per-ion test pins the whole sign chain in one assertion, at three valences: the surface gains
`z*F*R`, the bulk loses exactly `R` moles, and bulk-plus-surface charge does not move. Getting the
bulk sign backwards passes the first and fails the other two — which is precisely what the wrong
`IonFluxBC` docstring (section 11.4) would have produced.

`interface_props.surface_charge_density` is *also* what `SurfaceChargeBC` imposes as a fixed charge,
so using both classes on one interface counts it twice. Documented; not detectable cheaply.

### 11.4 Two records corrected while in there

- **`IonFluxBC`'s sign was documented backwards.** It adds `+weak(flux, c_test)` while the docstring
  claimed a positive value is an *influx*. The bulk assembles `int(d_t c v) - int(J.grad v)`, so the
  omitted boundary term is `-oint(n.J)v` and `+weak(g, c_test)` imposes `n.J = g`, an **outflux**.
  `NeumannBC`, which the docstring appealed to, uses the same convention once its value is read as a
  flux rather than as a gradient. Nothing caught it because, as section 10.2 already recorded, no test
  imposes a nonzero ion flux. The docstring now also says *which* flux `g` is, since that depends on
  `advection_by_parts`.
- **`NernstPlanckEquations` claimed its natural boundary condition is the zero *total* normal flux.**
  With the default `advection_by_parts=False` only the terms written against `grad(c_test)` leave a
  boundary term, so it is the zero diffusive-plus-migration flux and the advective part is
  unconstrained. True as stated only under `advection_by_parts` / `GCL`.
- **`NernstPlanckEquations.strong_residual` always wrote `dot(wind, grad(c))`**, i.e. under
  `advection_by_parts=True` the stabilization was built on a different equation than the one being
  assembled. It now consults `stab_cfg.conservative_residual` like the composition equations do.
  `SaltTransportEquations` had the identical hole and got the identical fix.

### 11.5 Still open

- `time_derivative_of_integral` breaks the telescoping at an adaptation or a remesh, unavoidably: the
  history integral is over the pre-adaptation mesh, so the conservation error at that step is the
  projection error. Inherent to the formulation, not to this change. The tests do not adapt.
- Its mass matrix comes solely from the `GiNaC_mass_matrix_marker` term, because a finite difference
  of integrals is invisible to the `__partial_t_mass_matrix` probe. Whether that survives the
  azimuthal expansion is pinned by no test — here or for the surfactants.
- MPI is untouched by the change and untested for it, as for everything else in section 10.3.
- `CompositionAdvectionDiffusionEquations`' GCL branch passes plain `"BDF2"` to
  `time_derivative_of_integral`, because it shares one `scheme` argument with its `time_scheme`
  wrapper and `time_scheme()` rejects the `_degr` names. So it runs first order in the first step.
  Left alone deliberately: fixing it means splitting that argument, which moves published numbers.
  The new flags all carry a separate `gcl_scheme`/`scheme` defaulting to `"BDF2_degr"`.

## 12. Pitfalls a user will hit

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
