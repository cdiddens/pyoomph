#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
#
#  @section LICENSE
#
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================

"""
A liquid given by concentration in Mixture(...), e.g. a soluble surfactant at 1 mM.

Unlike a dissolved salt, such a species is a real component: it counts towards the mass fractions.
The only question is what volume the concentration refers to, and there are two answers that differ
by the solute's own mass fraction:

  * ``concentration_basis="base_mixture"`` (the default) is how a solution is made -- mix the base,
    measure *its* volume, add the solute by that volume. The statement it makes is a mass balance,
    "per rho_base kg of base there are c*M kg of solute", and it claims nothing about the volume the
    solute takes up. Tested as that balance rather than as the formula the implementation uses.
  * ``concentration_basis="solution"`` is moles per volume of the finished solution, i.e. what the
    solver itself reports: CompositionAdvectionDiffusionEquations defines
    molarconc_<n> = massfrac_<n>*mass_density/M_n, and every surfactant isotherm is written against
    that field. So this basis is pinned by requiring molarconc_<n> at t=0 to be exactly what was
    typed -- which needs the mixture's own density, hence a fixed point.

Both are pinned here, together with the fact that they agree in the dilute limit and part company
where they should. Symbolic only: no Problem, no mesh, no JIT.
"""

import pytest

from pyoomph.expressions import var
from pyoomph.expressions.units import *
from pyoomph.materials import *
import pyoomph.materials.default_materials
import pyoomph.materials.ions
from pyoomph.materials.generic import (ConcentrationMixtureDefinitionComponent,
                                       DissolvedSpeciesComponent, LiquidMixtureDefinitionComponent)

T = 20 * celsius
#: A solute with a molar mass large enough that the two bases actually differ where they should.
M_SOLUTE = 180 * gram / mol


_SOLUTE = new_pure_liquid("conc_test_solute", mass_density=1200 * kilogram / meter ** 3,
                          molar_mass=M_SOLUTE)


@MaterialProperties.register()
class _MixWaterSolute(MixtureLiquidProperties):
    components = {"water", "conc_test_solute"}
    passive_field = "water"

    def __init__(self, pure_props):
        super().__init__(pure_props)
        self.set_by_weighted_average("mass_density")
        self.dynamic_viscosity = self.pure_properties["water"].dynamic_viscosity
        self.set_diffusion_coefficient(1e-9 * meter ** 2 / second)


@MaterialProperties.register()
class _MixWaterGlycerolSolute(MixtureLiquidProperties):
    components = {"water", "glycerol", "conc_test_solute"}
    passive_field = "water"

    def __init__(self, pure_props):
        super().__init__(pure_props)
        # Explicitly, not by weighted average: glycerol's own density correlation makes the averaged
        # expression something the unit collector complains loudly about, which has nothing to do
        # with what is tested here.
        self.mass_density = sum(var("massfrac_" + c) * self.pure_properties[c].mass_density
                                for c in sorted(self.components))
        self.dynamic_viscosity = self.pure_properties["water"].dynamic_viscosity
        self.set_diffusion_coefficient(1e-9 * meter ** 2 / second)


def water():
    return get_pure_liquid("water")


def glycerol():
    return get_pure_liquid("glycerol")


def solute():
    return get_pure_liquid("conc_test_solute")


def molarconc(props, name):
    """molarconc_<name> at the initial condition.

    The same expression CompositionAdvectionDiffusionEquations.define_fields substitutes for that
    field (pyoomph/equations/multi_component.py), so this is what the isotherms would read at t=0.
    """
    M = props.pure_properties[name].molar_mass
    expr = var("massfrac_" + name) * props.mass_density / M
    return float(props.evaluate_at_condition(expr, props.initial_condition) / (mol / meter ** 3))


def mixture(*, c, basis="base_mixture", with_glycerol=False, **kwargs):
    mdef = water() + c * solute()
    if with_glycerol:
        mdef = water() + 20 * percent * glycerol() + c * solute()
    return Mixture(mdef, temperature=T, concentration_basis=basis, **kwargs)


def base_density(props):
    """The density of the base mixture alone, i.e. at the composition without the solute."""
    ic = {k: v for k, v in props.initial_condition.items() if not k.startswith("massfrac_")}
    solvents = {c for c in props.components if c != "conc_test_solute"}
    total = sum(props.initial_condition["massfrac_" + c] for c in solvents)
    ic.update({"massfrac_" + c: props.initial_condition["massfrac_" + c] / total for c in solvents})
    ic["massfrac_conc_test_solute"] = 0.0
    return float(props.evaluate_at_condition(props.mass_density, ic) / (kilogram / meter ** 3))


# ---------------------------------------------------------------- the two bases


@pytest.mark.parametrize("c", [1 * milli * molar, 100 * milli * molar, 1 * molar])
def test_solution_basis_reproduces_the_concentration_exactly(c):
    """The whole point of the "solution" basis: molarconc_ at t=0 is what was typed."""
    props = mixture(c=c, basis="solution")
    assert molarconc(props, "conc_test_solute") == pytest.approx(float(c / (mol / meter ** 3)),
                                                                 rel=1e-12)


@pytest.mark.parametrize("c", [1 * milli * molar, 100 * milli * molar, 1 * molar])
def test_base_mixture_basis_is_a_mass_balance(c):
    """Per rho_base kg of base mixture there are c*M kg of solute -- no more is claimed."""
    props = mixture(c=c, basis="base_mixture")
    w = props.initial_condition["massfrac_conc_test_solute"]
    added = w * base_density(props) / (1 - w)
    assert added == pytest.approx(float(c * M_SOLUTE / (kilogram / meter ** 3)), rel=1e-12)


def test_the_two_bases_agree_when_dilute_and_differ_when_not():
    dilute = [mixture(c=1 * milli * molar, basis=b).initial_condition["massfrac_conc_test_solute"]
              for b in ("base_mixture", "solution")]
    assert dilute[0] == pytest.approx(dilute[1], rel=1e-3)
    strong = [mixture(c=1 * molar, basis=b).initial_condition["massfrac_conc_test_solute"]
              for b in ("base_mixture", "solution")]
    # 1 M of a 180 g/mol solute is 15 wt%: the two bases must be visibly, not subtly, different.
    assert abs(strong[0] / strong[1] - 1) > 0.1


def test_the_default_is_the_base_mixture():
    assert (Mixture(water() + 1 * molar * solute(), temperature=T).initial_condition
            == mixture(c=1 * molar, basis="base_mixture").initial_condition)


def test_mass_concentration_needs_no_molar_mass():
    """5 g/l is 5 kg per m^3 of whichever volume the basis names, molar mass or not."""
    props = mixture(c=5 * gram / litre, basis="base_mixture")
    w = props.initial_condition["massfrac_conc_test_solute"]
    assert w * base_density(props) / (1 - w) == pytest.approx(5.0, rel=1e-12)
    props = mixture(c=5 * gram / litre, basis="solution")
    rho = float(props.evaluate_at_condition(props.mass_density, props.initial_condition)
                / (kilogram / meter ** 3))
    assert props.initial_condition["massfrac_conc_test_solute"] * rho == pytest.approx(5.0, rel=1e-12)


# ---------------------------------------------------------------- the base survives


@pytest.mark.parametrize("basis", ["base_mixture", "solution"])
def test_the_base_composition_is_preserved(basis):
    """"20 % glycerol" means 20 % of the base, and the solute is added on top of it."""
    props = mixture(c=50 * milli * molar, basis=basis, with_glycerol=True)
    ic = props.initial_condition
    wg, ww = ic["massfrac_glycerol"], ic["massfrac_water"]
    assert wg / (wg + ww) == pytest.approx(0.2, rel=1e-13)
    assert sum(ic["massfrac_" + c] for c in props.components) == pytest.approx(1.0, abs=1e-14)


def test_the_mass_fraction_form_gives_the_very_same_mixture():
    """Feeding the resulting fraction back in by hand has to reproduce the composition."""
    props = mixture(c=100 * milli * molar, basis="solution")
    w = props.initial_condition["massfrac_conc_test_solute"]
    byhand = Mixture(water() + w * solute(), temperature=T)
    for c in props.components:
        assert byhand.initial_condition["massfrac_" + c] == pytest.approx(
            props.initial_condition["massfrac_" + c], rel=1e-14)


def test_mole_fractions_describe_the_base_mixture():
    """quantity= converts the base; the concentration species is not one of those fractions."""
    props = Mixture(water() + 0.2 * glycerol() + 50 * milli * molar * solute(),
                    temperature=T, quantity="mole_fraction", concentration_basis="solution")
    ic = props.initial_condition
    Mw = float(get_pure_liquid("water").molar_mass / (gram / mol))
    Mg = float(glycerol().molar_mass / (gram / mol))
    expect = 0.2 * Mg / (0.2 * Mg + 0.8 * Mw)
    wg, ww = ic["massfrac_glycerol"], ic["massfrac_water"]
    assert wg / (wg + ww) == pytest.approx(expect, rel=1e-13)
    assert molarconc(props, "conc_test_solute") == pytest.approx(50.0, rel=1e-12)


# ---------------------------------------------------------------- the dispatch


def test_a_unit_decides_what_the_factor_means():
    assert isinstance(0.001 * solute(), LiquidMixtureDefinitionComponent)
    assert not isinstance(0.001 * solute(), ConcentrationMixtureDefinitionComponent)
    assert isinstance(1 * milli * molar * solute(), ConcentrationMixtureDefinitionComponent)
    assert isinstance(5 * gram / litre * solute(), ConcentrationMixtureDefinitionComponent)
    assert isinstance(20 * percent * glycerol(), LiquidMixtureDefinitionComponent)
    assert not isinstance(20 * percent * glycerol(), ConcentrationMixtureDefinitionComponent)


def test_the_ion_and_salt_route_is_untouched():
    """Salts and ions keep their dilute DissolvedSpeciesComponent, which is not a component at all."""
    assert isinstance(1 * milli * molar * get_salt("NaCl"), DissolvedSpeciesComponent)
    assert isinstance(1 * milli * molar * get_ion("Na+"), DissolvedSpeciesComponent)
    assert isinstance(0.001 * get_ion("Na+"), LiquidMixtureDefinitionComponent)


def test_scaling_a_concentration_component_stays_a_concentration():
    """MixtureDefinitionComponent.__mul__ mutates and returns None; the subclass must not."""
    doubled = 2 * (1 * molar * solute())
    assert isinstance(doubled, ConcentrationMixtureDefinitionComponent)
    assert float(doubled.concentration / (mol / meter ** 3)) == pytest.approx(2000.0)


# ---------------------------------------------------------------- refusals


def test_a_meaningless_unit_is_refused_at_the_multiplication():
    with pytest.raises(ValueError, match="molar concentration"):
        1 * milli * meter * solute()


def test_a_concentration_needs_a_base_mixture():
    with pytest.raises(RuntimeError, match="needs a base mixture"):
        Mixture(1 * milli * molar * solute(), temperature=T)


def test_a_concentration_needs_a_temperature():
    with pytest.raises(RuntimeError, match="temperature="):
        Mixture(water() + 1 * milli * molar * solute())


def test_more_solute_than_solution_is_refused():
    with pytest.raises(ValueError, match="whole solution"):
        mixture(c=100 * molar, basis="solution")


def test_a_component_cannot_be_given_both_ways():
    with pytest.raises(ValueError, match="both as a fraction and as a concentration"):
        Mixture(water() + 0.5 * solute() + 1 * milli * molar * solute(), temperature=T)


def test_an_unknown_basis_is_refused():
    with pytest.raises(ValueError, match="concentration_basis"):
        mixture(c=1 * milli * molar, basis="per_litre_of_tea")


def test_a_gas_cannot_take_a_concentration():
    with pytest.raises(Exception):
        Mixture(get_pure_gas("air") + 1 * milli * molar * get_pure_gas("water"), temperature=T)


# ---------------------------------------------------------------- alongside a salt


def test_a_dilute_salt_leaves_the_concentration_alone():
    props = Mixture(water() + 1 * milli * molar * solute() + 100 * milli * molar * get_salt("NaCl"),
                    temperature=T, concentration_basis="solution")
    assert molarconc(props, "conc_test_solute") == pytest.approx(1.0, rel=1e-12)
    assert float(props.get_bulk_concentration("Na+") / (mol / meter ** 3)) == pytest.approx(100.0)


def test_a_salt_that_is_a_component_dilutes_the_concentration_by_its_own_volume():
    r"""salt_treatment="component" gives the salt a share of the volume, so what was 1 mM per litre
    of solution becomes c*(1-c_s*V_phi,s). That is right rather than a defect: at fixed volume the
    salt displaces solution. Re-solving to force the concentration back would move the salt molarity
    instead, and one of the two has to give."""
    c_salt = 1 * molar
    salt = get_salt("NaCl")
    props = Mixture(water() + 1 * milli * molar * solute() + c_salt * salt,
                    temperature=T, concentration_basis="solution", salt_treatment="component")
    shift = 1 - float(c_salt * salt.get_apparent_molar_volume())
    assert molarconc(props, "conc_test_solute") == pytest.approx(shift, rel=1e-12)


# ---------------------------------------------------------------- nothing else moved


def test_plain_fraction_mixtures_are_unchanged():
    assert Mixture(water() + 0.2 * glycerol()).initial_condition == {"massfrac_glycerol": 0.2,
                                                                     "massfrac_water": 0.8}
    gas = Mixture(get_pure_gas("air") + 0.01 * get_pure_gas("water")).initial_condition
    assert gas == {"massfrac_water": 0.01, "massfrac_air": 0.99}
