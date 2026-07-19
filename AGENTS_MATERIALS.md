# pyoomph — materials system reference for AI coding assistants

Companion to [`AGENTS.md`](AGENTS.md). This covers `pyoomph.materials` — the fluid/
solid/interface property system consumed by the multi-component equation classes in
`pyoomph/equations/multi_component.py`, `NSCH.py`, `low_order_NSCH.py`, `darcy.py`,
and `contact_angle.py` (see `AGENTS.md`'s "Built-in physics" section for those
equation classes themselves; this file is about the material/property objects you
pass into them). For a full worked example combining materials with flow equations,
see recipe-style usage in [`AGENTS_EXAMPLES.md`](AGENTS_EXAMPLES.md).

## Mental model

A "material" is a Python object describing physical properties (density, viscosity,
vapor pressure, ...) as symbolic `Expression`s (so they may depend on `var("temperature")`,
composition fields, etc.), registered into a global class-level registry keyed by
state-of-matter + name (pure) or component set (mixtures). You either:

1. **Look up** a predefined material shipped in `pyoomph.materials.default_materials`, or
2. **Register a new one** by subclassing the appropriate base class and decorating it
   with `@MaterialProperties.register()`.

Mixtures are built by "adding" weighted pure components (`0.5*get_pure_liquid("ethanol") + get_pure_liquid("water")`)
and finalizing with `Mixture(...)`; interfaces are built with the `|` operator between
two bulk-phase material objects (or `get_interface_properties(...)`), and fall back to
a sensible default if no specific interface class was registered for that pair.

## Looking up predefined materials

```python
import pyoomph.materials.default_materials  # side-effect import: registers the library
from pyoomph.materials import *

water = get_pure_liquid("water")
air = get_pure_gas("air")
mix = Mixture(get_pure_liquid("water") + 0.5*get_pure_liquid("ethanol"))  # mass-fraction by default
```

Factory functions (`pyoomph/materials/generic.py`), all raising a descriptive
`RuntimeError` if nothing matches:

| Function | Returns |
|---|---|
| `get_pure_liquid(name)` | `PureLiquidProperties` |
| `get_pure_gas(name)` | `PureGasProperties` |
| `get_pure_solid(name)` | `PureSolidProperties` |
| `get_surfactant(name)` | `SurfactantProperties` |
| `get_mixture_properties(*pure_components)` | `MixtureLiquidProperties`/`MixtureGasProperties` |
| `get_interface_properties(phaseA, phaseB, surfactants=None)` | a `BaseInterfaceProperties` subclass instance |
| `Mixture(mixture_definition, temperature=None, quantity="mass_fraction", pressure=1*atm)` | finalizes a mixture definition into a `MixtureLiquidProperties`/`MixtureGasProperties` |
| `new_pure_liquid(name, mass_density=..., dynamic_viscosity=..., surface_tension=..., molar_mass=..., ...)` | quick inline pure-liquid registration, no subclassing needed |
| `new_pure_gas(name, ...)` | same, for gases |

There is no single `Material("water")` factory — always use the specific
`get_pure_liquid`/`get_pure_gas`/`get_pure_solid`/`get_surfactant` function.

**Predefined materials in `pyoomph/materials/default_materials.py`:**

| Name(s) | State | Notes |
|---|---|---|
| `"water"`, `"glycerol"`, `"ethanol"`, `"12hexanediol"` | pure liquid | |
| `{"water","glycerol"}`, `{"water","ethanol"}`, `{"12hexanediol","water"}` | liquid mixture | composition-dependent viscosity/surface tension/diffusivity fits |
| `"air"`, `"water"`, `"ethanol"` | pure gas | gas `"water"` is a separate registry entry from liquid `"water"` |
| `{"water","air"}`, `{"ethanol","air"}`, `{"ethanol","water","air"}` | gas mixture | |
| `"aluminium"`, `"paper"`, `"borosilicate"`, `"stainless_steel"` | pure solid | |

No liquid-gas/liquid-solid *interface* classes are pre-registered — an unregistered
pair automatically falls back to `DefaultLiquidGasInterface` (surface tension taken
from the liquid's `default_surface_tension["gas"]`; standard mass-transfer model).

`quantity=` for `Mixture(...)`/`.finalise(...)` accepts `"mass_fraction"`/`"wt"`,
`"mole_fraction"`, `"volume_fraction"`, or `"relative_humidity"`/`"RH"` (for gas
mixtures against a given liquid's vapor pressure).

## Defining a new material

Subclass the matching base and register it. Exactly one component may be omitted
its explicit fraction when building a mixture (its fraction is inferred as
`1 - sum(others)`).

| Base class | Use for | Must set in `__init__` |
|---|---|---|
| `PureLiquidProperties` | a new pure liquid | class attr `name`; `self.molar_mass`, `self.mass_density`, `self.dynamic_viscosity`; optionally `self.default_surface_tension["gas"]`, `self.thermal_conductivity`, `self.specific_heat_capacity`, `self.latent_heat_of_evaporation`, vapor pressure (`self.set_vapor_pressure_by_Antoine_coeffs(A,B,C)` or `self.vapor_pressure=...`), UNIFAC groups (`self.set_unifac_groups({...})`) |
| `PureGasProperties` | a new pure gas | class attr `name`; `self.molar_mass`; density typically via `self.set_mass_density_from_ideal_gas_law()` |
| `PureSolidProperties` | a new pure solid | class attr `name`; `self.mass_density` |
| `SurfactantProperties` (subclass of `PureLiquidProperties`) | a surfactant (soluble or insoluble) | same as `PureLiquidProperties`, plus `self.surface_diffusivity` |
| `MixtureLiquidProperties` | how a *specific* liquid mixture behaves | class attr `components: Set[str]` (must exactly match the set of pure-liquid names being mixed, else lookup fails with `KeyError`); `__init__(self, pure_props)` receives the pure-component instances as a dict |
| `MixtureGasProperties` | how a specific gas mixture behaves | same pattern as `MixtureLiquidProperties` |

Inside a `MixtureLiquidProperties`/`MixtureGasProperties` subclass, useful helpers on
`self`: `self.pure_properties["name"]` (the pure-component instance),
`self.get_mass_fraction_field(name)` (→ `var("massfrac_"+name)`),
`self.get_mole_fraction_field(name)`, `self.set_by_weighted_average("propname")`
(derive a bulk property as the mass-fraction-weighted average of the pure
components' same-named property), `self.set_diffusion_coefficient(D)` (or
per-pair overloads), `self.set_activity_coefficients_by_unifac(model)`.

Example — registering a new pure liquid and a mixture with hand-fit
composition-dependent properties (trimmed from `docs/source/tutorial/mcflow/matlib/`):

```python
from pyoomph.materials import *

@MaterialProperties.register()
class PureLiquidWater(PureLiquidProperties):
    name = "water"
    def __init__(self):
        super().__init__()
        self.molar_mass = 18.01528*gram/mol
        self.mass_density = 998*kilogram/meter**3
        self.dynamic_viscosity = 1*milli*pascal*second
        self.specific_heat_capacity = 4.187*kilo*joule/(kilogram*kelvin)
        self.thermal_conductivity = 0.597*watt/(meter*kelvin)
        self.latent_heat_of_evaporation = 2437.69081321*kilo*joule/kilogram

        TKelvin = var("temperature")/kelvin
        self.default_surface_tension["gas"] = 0.07275*(1.0-0.002*(TKelvin-291.0))*newton/meter
        self.set_vapor_pressure_by_Antoine_coeffs(8.07131, 1730.63, 233.426)  # mmHg/celsius convention
        self.set_unifac_groups({"H2O": 1})


@MaterialProperties.register()
class MixtureLiquidGlycerolWater(MixtureLiquidProperties):
    components = {"water", "glycerol"}
    passive_field = "water"  # the field NOT solved for explicitly (inferred from mass conservation)

    def __init__(self, pure_properties):
        super().__init__(pure_properties)
        self.set_by_weighted_average("mass_density")
        self.set_by_weighted_average("thermal_conductivity")
        self.set_by_weighted_average("specific_heat_capacity")

        yG = self.get_mass_fraction_field("glycerol")
        TCelsius = subexpression(var("temperature")/kelvin - 273.15)
        # ... composition/temperature-dependent viscosity/surface-tension/diffusivity fits ...
        self.set_diffusion_coefficient(1.024e-11*(-0.721*yG+0.7368)/(0.49311e-2*yG+0.7368e-2)*meter**2/second)
        self.set_activity_coefficients_by_unifac("AIOMFAC")
```

`subexpression(expr)` wraps a sub-expression so the generated C code computes it once
and reuses the value — use it for any moderately expensive shared piece of a
composition/temperature fit (as above), not just in materials code.

## Interfaces between phases

| Class | Use for | Class attrs used for registry lookup |
|---|---|---|
| `LiquidGasInterfaceProperties` | liquid-vapor interface | `liquid_components`, `gas_components`, `surfactants` |
| `LiquidSolidInterfaceProperties` | liquid-solid (wetting) interface | `liquid_components`, `solid_components`, `surfactants` |
| `LiquidLiquidInterfaceProperties` | liquid-liquid interface | `componentsA`, `componentsB`, `surfactants` |
| `DefaultLiquidGasInterface` | fallback when no specific liquid-gas interface is registered | copies `default_surface_tension["gas"]` from the liquid |

Get one via `phaseA | phaseB` or `get_interface_properties(phaseA, phaseB, surfactants=None)`.
Both look up `MaterialProperties.library["interfaces"][...]` keyed by the frozenset
of component names (falling back to a wildcard match on the non-liquid side, then to
`DefaultLiquidGasInterface` for liquid-gas only — there is no built-in default for
liquid-solid/liquid-liquid). All interface classes expose `.surface_tension`
(an `Expression`, may depend on temperature/composition/surfactant coverage),
`set_mass_transfer_model(mdl)`/`get_mass_transfer_model()`, and
`set_latent_heat_of(name, L)`/`get_latent_heat_of(name)`.

## Surfactants

A surfactant is "just a pure liquid" (`SurfactantProperties(PureLiquidProperties)`)
that can additionally adsorb/desorb at an interface. Adsorption isotherms
(`pyoomph/materials/surfactant_isotherms.py`), all subclassing `SurfactantIsotherm`:

| Class | Extra ctor args | Model |
|---|---|---|
| `HenryIsotherm(surfactant_name, k_ads=, k_des=, K=None)` | — | linear: Γ = K·C |
| `LangmuirIsotherm(surfactant_name, GammaInfty, k_ads=, k_des=, K=None)` | max surface concentration `GammaInfty` | saturating: Γ = Γ∞KC/(1+KC) |
| `VolmerIsotherm(surfactant_name, GammaInfty, k_ads=, k_des=)` | `GammaInfty` | excluded-area correction |
| `FrumkinIsotherm(surfactant_name, GammaInfty, beta=, k_ads=, k_des=)` | `GammaInfty`, interaction `beta` | Langmuir + intermolecular interaction |
| `VanDerWaalsIsotherm(surfactant_name, GammaInfty, beta=, k_ads=, k_des=)` | `GammaInfty`, `beta` | van der Waals equation of state analogue |

Usage: build the isotherm, then `isotherm.apply_on_interface(interface_props, pure_surface_tension=interface_props.surface_tension, min_surface_tension=...)`
— this overwrites `interface_props.surface_tension` (subtracting the surface pressure)
and sets `interface_props.surfactant_adsorption_rate[name]` (net ad-/desorption flux),
which the flow-interface equations (`MultiComponentNavierStokesInterface`) then pick up
automatically. Full example (`docs/source/tutorial/mcflow/surfact/soluble_surfactants.py`, trimmed):

```python
from pyoomph import *
from pyoomph.materials import *
import pyoomph.materials.default_materials
from pyoomph.expressions.units import *
from pyoomph.expressions.phys_consts import gas_constant

@MaterialProperties.register()
class MySolubleSurfactant(SurfactantProperties):
    name = "my_soluble_surfactant"
    def __init__(self):
        super().__init__()
        self.molar_mass = 100*gram/mol
        self.surface_diffusivity = 0.5e-9*meter**2/second

@MaterialProperties.register()
class MixLiquidWaterMySolubleSurfactant(MixtureLiquidProperties):
    components = {"water", "my_soluble_surfactant"}
    def __init__(self, pure_props):
        super().__init__(pure_props)
        self.mass_density = self.pure_properties["water"].mass_density
        self.dynamic_viscosity = self.pure_properties["water"].dynamic_viscosity
        self.default_surface_tension["gas"] = self.pure_properties["water"].default_surface_tension["gas"]
        self.set_diffusion_coefficient(1e-9*meter**2/second)  # assume small effect on bulk properties

from pyoomph.materials.surfactant_isotherms import *

@MaterialProperties.register()
class InterfaceWaterMySolubleSurfactantVSGas(DefaultLiquidGasInterface):
    liquid_components = {"water", "my_soluble_surfactant"}
    surfactants = {"my_soluble_surfactant"}

    def __init__(self, phaseA, phaseB, surfactants):
        super().__init__(phaseA, phaseB, surfactants)
        isotherm = LangmuirIsotherm("my_soluble_surfactant", k_ads=5e-6*meter/second,
                                     k_des=9.5/second, GammaInfty=5*micro*mol/meter**2)
        isotherm.apply_on_interface(self, pure_surface_tension=self.surface_tension,
                                     min_surface_tension=20*milli*newton/meter)

if __name__ == "__main__":
    liquid = Mixture(get_pure_liquid("water") + 0.001*get_pure_liquid("my_soluble_surfactant"))
    gas = get_pure_gas("air")
    interface = get_interface_properties(liquid, gas, surfactants={"my_soluble_surfactant": 1*micro*mol/meter**2})
```

For a soluble surfactant, a nonzero (possibly tiny) concentration must appear both in
the liquid mixture *and* in the `surfactants` dict passed to `get_interface_properties`
— an *insoluble* surfactant only lives on the interface (no bulk mixture term needed).

## Mass transfer (evaporation/condensation) models

Class hierarchy in `pyoomph/materials/mass_transfer.py`:

```
MassTransferModelBase
└── ProjectedMassTransferModelBase
    ├── PrescribedMassTransfer(**rates)
    └── FluidPropMassTransferModel(props_inside, props_outside)
        ├── DifferenceDrivenMassTransferModel(...)                      # a.k.a. StandardMassTransferModelLiquidGas
        │   └── HertzKnudsenSchrageMassTransferModel(..., sticking_coefficient=0.1)
        │   └── LLEMassTransferModel(..., unifac_model=None, ...)       # liquid-liquid equilibrium driven
        └── LagrangeMultiplierMassTransferModel(...)
```

`LiquidGasInterfaceProperties` auto-assigns `StandardMassTransferModelLiquidGas` (a
Raoult's-law-driven rate model, default rate coefficient `100 kg/(m^2 s)`) unless
overridden. Attach explicitly with `interface_props.set_mass_transfer_model(model)`,
or pass `masstransfer_model=` directly to `MultiComponentNavierStokesInterface(...)`
(`None` = use whatever the interface object carries, `False` = disable mass transfer
entirely, or an explicit model instance).

To implement a fully custom model, subclass `MassTransferModelBase` and override
`identify_transfer_components() -> Set[str]` and `get_mass_transfer_rate_of(name) -> Expression`
(plus `setup_scaling`/`define_fields`/`define_residuals` if extra internal fields are
needed); for the common "driven by a bulk-vs-interface difference" shape, subclassing
`DifferenceDrivenMassTransferModel` and overriding only
`get_driving_nondimensional_difference_for(name)` is simpler.

## Activity coefficients / UNIFAC

For non-ideal liquid mixtures, vapor pressure of each component depends on an
activity coefficient from a group-contribution model. Workflow:

1. Register UNIFAC functional groups on each pure liquid:
   `pure_liquid.set_unifac_groups({"CH2": 2, "OH": 1, ...}, only_for=None)`
   (`only_for` restricts to specific models if the group decomposition differs
   between them, e.g. `only_for="Original"`).
2. On the mixture: `mixture.set_activity_coefficients_by_unifac(model, set_vapor_pressures=True)`.

`model` is one of the exact registered strings **`"Original"`, `"Dortmund"`,
`"AIOMFAC"`** (in `pyoomph/materials/UNIFAC/{original,dortmund,aiomfac}.py`) — note
some tutorial prose says `"UNIFAC"` for the original model, which is stale; the
correct string verified from source is `"Original"`. For mixtures of 3+ components,
this internally builds a `CustomMultiReturnExpression` (finite-difference Jacobian)
for performance — such an expression cannot be used inside bifurcation tracking (see
`AGENTS_ADVANCED.md`).

## Using materials in flow equations (cross-reference)

Once you have a `fluid_props`/`interface_props` object, hand it to the multi-component
equation classes from `AGENTS.md`'s built-in physics catalog:

```python
from pyoomph.equations.multi_component import *

eqs = CompositionFlowEquations(fluid_props, compo_space="C1", ...)      # bulk: flow + species (+ optional heat)
eqs += MultiComponentNavierStokesInterface(interface_props, ...) @ "some_boundary"  # free interface w/ mass transfer, Marangoni, surfactants
```

See `AGENTS_EXAMPLES.md` (or `docs/source/tutorial/mcflow/marangoni_instability.py`
in this repo) for a complete Hele-Shaw Marangoni-instability problem built this way:
two coupled domains (`"liquid"`, `"gas"`) with a `MultiComponentNavierStokesInterface`
between them, driven purely by looked-up/mixed materials.
