# pyoomph — units, scaling and the dimensionless-residual rule

Companion to [`AGENTS.md`](../AGENTS.md). The single most common reason a pyoomph script
fails at setup is a residual that is not dimensionless. This file is the long form of the
rule that `AGENTS.md` states in short.

## Units and constants

`from pyoomph.expressions.units import *` — a **separate import**: units are not in
`from pyoomph import *` nor in `from pyoomph.expressions import *`. Base units `meter`,
`second`, `kilogram`, `kelvin`, `mol`, `ampere`; **the full SI prefix set** — `atto`,
`femto`, `pico`, `nano`, `micro`, `milli`, `centi`, `deci`, `deca`, `hecto`, `kilo`,
`mega`, `giga`, `tera`, `peta`, `exa`, `zepto`/`zetta`, `yocto`/`yotta`; derived units
`newton`, `pascal`, `bar`, `joule`, `watt`, `volt`, `coulomb`, `farad`, `ohm`, `siemens`,
`henry`, `hertz`, `gram`, `liter`, `molar`, `angstrom`, `minute`, `hour`, `day`, `atm`,
`torr`, `mmHg`, `celsius`, `degree`, `percent`.

**Physical constants** live in `pyoomph.expressions.phys_consts` (a separate import, and
it re-exports the units too): `epsilon_0`, `mu_0`, `c_of_light`, `k_Boltzmann`,
`elementary_charge`, `N_Avogadro`, `faraday_constant`, `gas_constant`,
`thermal_voltage(T)`, `debye_length(...)`, `bjerrum_length(...)`. Do not redefine these
by hand. **`celsius` carries the 273.15 K offset** — `20*celsius` really is `293.15*kelvin`
— so use it for an absolute temperature and plain `kelvin` for a temperature *difference*.
Multiply numeric literals by these to get dimensional `Expression`s, e.g.
`5*milli*meter` or `0.1*milli*meter/second`. There is **no `mm`/`cm`/`km` shorthand** —
compose them from a prefix and a base unit.

**Write the physics dimensionally and let pyoomph nondimensionalize it.** This is the
default and strongly preferred way — only go nondimensional when the user asks for it.
`problem.set_scaling(...)` declares the scale of each quantity:

```python
self.set_scaling(spatial=1*milli*meter, temporal=1*second, T=1*kelvin,
                 velocity=1*milli*meter/second)
```

Every field then has a scale (`scale_factor("T")`) and a test-function scale
(`test_scale_factor("T")`). Inside a reusable `Equations` class, express `scale`/`testscale`
through `scale_factor(...)` rather than hard-coded numbers, so the class stays usable at
any set of scales — the idiom is
`testscale=scale_factor("temporal")/scale_factor("T")`, which makes the residual
dimensionless. `nondim(name)` gives the nondimensional counterpart of `var(name)`.

## The one rule that governs every residual

**`weak(a, b)` integrates over the *nondimensionalized* domain.** The measure `dΩ` carries
no units by default (`weak(..., dimensional_dx=True)` opts into the physical `m^d`), so
every residual contribution must satisfy

> `a * b` is dimensionless — nothing else.

`b` is a test function, whose scale you choose with `testscale=`. So the recipe is: pick
`testscale` as whatever makes `a*b` dimensionless. For a heat equation whose leading term
is `weak(partial_t(T), Ttest)`, that is `testscale=scale_factor("temporal")/scale_factor("T")`
— exactly the skeleton at the top of this file.

Two consequences that catch people out, because they make units of a *coefficient* differ
from the textbook strong form:

- A gradient pairing `weak(coeff*grad(u), grad(utest))` with `testscale=1/scale_factor(u)`
  contributes `coeff / L²`. **So `coeff` must carry `length²`**, not the diffusivity's usual
  units. This is why `PoissonEquation(coefficient=1*meter**2)` is right and
  `coefficient=1` is a hard error in a dimensional problem.
- A source pairing `weak(f, utest)` with the same testscale contributes `f / u`. **So the
  source carries the units of `u` itself**, not of `u/length²`.

Both terms have to be dimensionless *individually*, so with that testscale the two are
pinned: the equation actually solved is `-coefficient·Δu = source` with `coefficient` an
**area** and `source` in the units of `u`. A problem posed as `-D·Δu = f` with a physical
diffusivity (say `D` in m²/s and `f` in K/s) is the same equation multiplied through by a
constant — multiply both sides by whatever turns `D` into an area, here one second, and
pass `coefficient=D*second`, `source=f*second`.

A stationary problem needs no `temporal=` scale: only terms that actually appear in the
residual are checked, and `scale_factor("temporal")` is referenced only by equations that
have a time derivative.

## On interfaces and boundaries

Interface residuals must be **strictly dimensionless too** — the rule never relaxes. What
changes is only whose test function you are pairing with:

- A field **declared on the interface** (a Lagrange multiplier, a surface concentration)
  behaves exactly as in the bulk: choose `testscale` as `1/units(a)` so that `a*b` is
  dimensionless. No spatial factor anywhere.
- The test function of a **bulk** field, used inside an interface equation, already carries
  an extra `1/scale_factor("spatial")`. That is precisely what makes the natural boundary
  term work with the *same* `coeff` as the bulk equation: integrating
  `-div(coeff*grad(u))` by parts leaves `weak(coeff*dot(grad(u),n), utest)`, and that is
  dimensionless whenever the bulk `weak(coeff*grad(u), grad(utest))` is.

The practical consequence: pair a bulk test function with something in the units of
`coeff*grad(u)`, **not** with a raw physical flux that has the material coefficient baked
in. For heat conduction posed as `PoissonEquation(coefficient=k*C)` (with `C` the constant
that turns `k` into an area, see above), a multiplier `q` holding the *physical* flux in
W/m² enters as

```python
self.add_residual(weak(C*q, T_test))                      # flux out of this side
self.add_residual(weak(-C*q, T_opposite_test))            # and into the other
self.add_residual(weak(T - T_opposite - R_contact*q, q_test))   # the constraint itself
```
because `C*q == coeff*grad(T)` dimensionally. Writing `coeff*q` (leftover `W`) or dividing
by `scale_factor("spatial")` (leftover `meter^(-1)`) are the two usual mistakes, and both
are caught at setup. This is also why `NeumannBC(u=-g)` on a `PoissonEquation` imposes
`coeff*grad(u).n = g`.

If you get it wrong, pyoomph refuses at setup with

```
The added residual contribution is not dimensionless.
It still carries the base unit: meter
All terms agree on the unit meter^(-2), i.e. it is consistent with itself but not dimensionless.
```

and then prints every scale in play plus the offending term, expanded. Read that list: it
names the field, the scale and the test scale it used, so the fix is usually one factor.
A residual that is *self-consistent but not dimensionless*, as above, means a coefficient
is short exactly that power of length. Note that a wrong-but-consistent choice can also
slip through when a scale absorbs it — a unit that comes out as `kelvin/meter**2` where you
expected `kelvin` means the source or coefficient units were off even though it ran.

**Pitfalls that bite hard and surface far from the cause:**

- **Non-integer exponents must be exact rationals, never Python floats.** Write
  `x**rational_num(19,20)`, not `x**0.95`. GiNaC's unit handling gives up on a float
  exponent applied to a quantity that still carries units, and the error shows up
  somewhere else entirely. Put the decimal in a trailing comment. The same applies to
  `**rational_num(1,2)` for a square root of a dimensional quantity.
- **A missing `set_scaling` entry silently means a scale of 1** in the corresponding SI
  base unit, which for e.g. a micrometre-scale problem gives a badly conditioned system
  rather than an error. Set a scale for every field whose natural magnitude is far from 1.
- **Don't mix dimensional and nondimensional expressions.** Any quantity fed into a
  substituted field or a material property must be consistently one or the other; a bare
  float where an `Expression` with units is expected is a dimension error waiting to
  happen.
- `problem.get_scaling(...)` and multiplying/dividing by a unit are how you convert a
  computed nondimensional number back to a physical one for output.

