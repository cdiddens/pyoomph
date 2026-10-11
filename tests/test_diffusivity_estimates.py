#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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
Diffusivities estimated from activity coefficients.

The estimate is a product of four matrices, three of which are exact:

    [D] = (1/Mbar) [C] [B]^-1 [Gamma] [A]

Only [B] (Maxwell-Stefan) and [Gamma] (thermodynamic factor) carry any modelling; [A], [C] and Mbar
are reference-frame and mole-to-mass conversions that follow from the definitions alone. That split
is what makes this testable without a single tabulated diffusivity:

  1. THE EXACT PARTS are pinned by two identities. A binary must collapse to D = Dms*Gamma, and equal
     Maxwell-Stefan diffusivities in an ideal mixture must give an isotropic matrix, because
     (1/Mbar)[C][A] = [I]. Either identity fails the moment [A] is transposed or M_j and M_n are
     swapped, and both hold to machine precision when they are right.
  2. THE THERMODYNAMIC FACTOR is computed two ways -- symbolic differentiation of the UNIFAC
     expression, and a central difference of the multi-return one -- which have no code in common
     beyond the activity model itself. They must agree, and they must both eliminate the passive mole
     fraction: a Gamma built without that elimination is a different (wrong) quantity, so there is a
     test that it really differs.
  3. THE RESULT COMPILES. Multi-return activity coefficients cannot be differentiated symbolically
     (the attempt now raises, see tests/test_custom_callbacks.py; before that error existed it
     produced a C file gcc rejects, invisible to every purely symbolic test here). So one test
     actually builds and solves a problem with an estimated diffusivity matrix in its residual.

The frozen number Gamma = 0.270976 for water/1,2-hexanediol at a mass fraction of 0.4 and 25 degC is
the sign convention and the elimination in one value; it is a UNIFAC/AIOMFAC output, not a
measurement.
"""

import math
import warnings

import numpy
import pytest

from pyoomph import Problem, LineMesh, InitialCondition
from pyoomph.equations.multi_component import CompositionAdvectionDiffusionEquations
from pyoomph.expressions import var, ExpressionOrNum
from pyoomph.expressions.units import *
from pyoomph.materials import *
from pyoomph.materials.diffusivity_estimates import (fickian_mass_diffusivity_matrix,
                                                     get_molar_volume_at_boiling_point,
                                                     hydrodynamic_radius,
                                                     thermodynamic_factor_matrix,
                                                     thermodynamic_factor_matrix_symbolic,
                                                     wilke_chang_diffusivity)
import pyoomph.materials.default_materials  # noqa: F401  (registers the material library)
import pyoomph._pyoomph_core as _pyoomph

#: An arbitrary but realistic Maxwell-Stefan diffusivity, used wherever the test does not want the
#: correlation to be part of what is being checked.
D_MS: ExpressionOrNum = 1.3e-9 * meter**2 / second

#: The thermodynamic factor of 1,2-hexanediol in water at a mass fraction of 0.4 and 25 degC, from
#: AIOMFAC. Well below one: this mixture is close to demixing there.
GAMMA_HEXANEDIOL_AT_04 = 0.270976


@MaterialProperties.register()
class MixtureLiquidWaterEthanolGlycerolForTests(MixtureLiquidProperties):
    """A ternary, since the material library has none registered. Water is the passive component."""
    components = {"water", "ethanol", "glycerol"}
    passive_field = "water"

    def __init__(self, pure_properties):
        super().__init__(pure_properties)
        self.set_by_weighted_average("mass_density")
        self.set_by_weighted_average("dynamic_viscosity")


def _binary(use_multi_return=3) -> MixtureLiquidProperties:
    mix = Mixture(get_pure_liquid("water") + 0.4 * get_pure_liquid("12hexanediol"))
    assert isinstance(mix, MixtureLiquidProperties)
    mix.set_activity_coefficients_by_unifac("AIOMFAC", use_multi_return=use_multi_return)
    return mix


def _ternary(use_multi_return=3, activity=True) -> MixtureLiquidProperties:
    mix = Mixture(get_pure_liquid("water") + 0.2 * get_pure_liquid("ethanol")
                  + 0.2 * get_pure_liquid("glycerol"))
    assert isinstance(mix, MixtureLiquidProperties)
    if activity:
        mix.set_activity_coefficients_by_unifac("AIOMFAC", use_multi_return=use_multi_return)
    return mix


def _at(mix, expr, cond, unit: ExpressionOrNum = 1) -> float:
    return float(mix.evaluate_at_condition(expr, cond) / unit)


def _condition(temperature: ExpressionOrNum = 25 * celsius, **massfracs: float) -> dict[str, ExpressionOrNum]:
    cond: dict[str, ExpressionOrNum] = {"massfrac_" + n: v for n, v in massfracs.items()}
    cond["temperature"] = temperature
    return cond


def test_a_binary_reduces_to_the_maxwell_stefan_diffusivity_times_gamma():
    # The only case in which the mass- and mole-frame Fickian diffusivities coincide, so the whole
    # [C], [A] and Mbar apparatus has to cancel exactly. It does not cancel if [A] is transposed.
    mix = _binary()
    D = fickian_mass_diffusivity_matrix(
        mix, "given", maxwell_stefan_diffusivities={("water", "12hexanediol"): D_MS},
        use_subexpressions=False)
    Gamma = thermodynamic_factor_matrix(mix, ["12hexanediol"])
    assert set(D.keys()) == {("12hexanediol", "12hexanediol")}
    for w in (0.05, 0.2, 0.4, 0.7, 0.95):
        cond = _condition(**{"12hexanediol": w})
        got = _at(mix, D[("12hexanediol", "12hexanediol")], cond, meter**2 / second)
        expected = 1.3e-9 * _at(mix, Gamma[0][0], cond)
        assert got == pytest.approx(expected, rel=1e-10)


def test_equal_diffusivities_in_an_ideal_mixture_give_an_isotropic_matrix():
    # (1/Mbar)[C][A] = [I]: with no thermodynamic factor and one Maxwell-Stefan diffusivity for every
    # pair, every species must diffuse with that same diffusivity and nothing may couple.
    mix = _ternary(activity=False)
    pairs = {("water", "ethanol"): D_MS, ("water", "glycerol"): D_MS,
             ("ethanol", "glycerol"): D_MS}
    D = fickian_mass_diffusivity_matrix(mix, "given", maxwell_stefan_diffusivities=pairs,
                                        thermodynamic_factor=False, use_subexpressions=False)
    assert set(D.keys()) == {(a, b) for a in ("ethanol", "glycerol") for b in ("ethanol", "glycerol")}
    for we in (0.0, 0.1, 0.35, 0.6):
        for wg in (0.0, 0.15, 0.3):
            cond = _condition(ethanol=we, glycerol=wg)
            for (a, b), expr in D.items():
                expected = 1.3e-9 if a == b else 0.0
                assert _at(mix, expr, cond, meter**2 / second) == pytest.approx(expected, abs=1.3e-21)


def test_the_symbolic_and_finite_difference_thermodynamic_factors_agree():
    # Two implementations that share nothing but the activity model: one differentiates the UNIFAC
    # expression, the other finite-differences the multi-return callback.
    sym = thermodynamic_factor_matrix(_ternary(use_multi_return=False), ["ethanol", "glycerol"])
    fd = thermodynamic_factor_matrix(_ternary(use_multi_return=True), ["ethanol", "glycerol"])
    mix = _ternary(use_multi_return=False)
    for we, wg in ((0.1, 0.1), (0.3, 0.2), (0.05, 0.5), (0.0, 0.3)):
        cond = _condition(ethanol=we, glycerol=wg)
        for i in range(2):
            for j in range(2):
                assert _at(mix, fd[i][j], cond) == pytest.approx(_at(mix, sym[i][j], cond),
                                                                 rel=1e-6, abs=1e-9)


def test_the_thermodynamic_factor_matches_its_frozen_reference():
    mix = _binary(use_multi_return=False)
    Gamma = thermodynamic_factor_matrix(mix, ["12hexanediol"])
    cond = _condition(**{"12hexanediol": 0.4})
    assert _at(mix, Gamma[0][0], cond) == pytest.approx(GAMMA_HEXANEDIOL_AT_04, rel=1e-5)


def test_the_passive_mole_fraction_has_to_be_eliminated():
    # Guards the previous two tests: without the elimination, d ln(gamma)/dx is taken at constant
    # passive mole fraction, which is a different quantity. Both branches would agree on it, so only
    # a test that it DIFFERS pins the convention.
    mix = _binary(use_multi_return=False)
    E = _pyoomph.Expression
    from pyoomph.expressions import log
    X = _pyoomph.GiNaC_new_symbol("__unconstrained_x")
    lng = _pyoomph.GiNaC_subs(E(log(mix.activity_coefficients["12hexanediol"])),
                              E(var("molefrac_12hexanediol")), E(X))
    d = _pyoomph.GiNaC_subs(_pyoomph.GiNaC_diff(lng, X), E(X), E(var("molefrac_12hexanediol")))
    unconstrained = 1 + var("molefrac_12hexanediol") * d
    constrained = thermodynamic_factor_matrix_symbolic(mix, ["12hexanediol"])[0][0]
    cond = _condition(**{"12hexanediol": 0.4})
    assert _at(mix, unconstrained, cond) != pytest.approx(_at(mix, constrained, cond), rel=1e-3)


def test_the_matrix_is_positive_where_the_binary_is_stable():
    mix = _binary()
    D = fickian_mass_diffusivity_matrix(mix, "given",
                                        maxwell_stefan_diffusivities={("water", "12hexanediol"): D_MS},
                                        use_subexpressions=False)
    for w in (0.0, 0.05, 0.1, 0.15, 0.2):
        cond = _condition(**{"12hexanediol": w})
        assert _at(mix, D[("12hexanediol", "12hexanediol")], cond, meter**2 / second) > 0


def test_the_ternary_matrix_is_positive_definite_where_the_mixture_is_dilute():
    mix = _ternary()
    D = fickian_mass_diffusivity_matrix(mix, "given",
                                        maxwell_stefan_diffusivities={("water", "ethanol"): D_MS,
                                                                      ("water", "glycerol"): D_MS,
                                                                      ("ethanol", "glycerol"): D_MS},
                                        use_subexpressions=False)
    indep = ["ethanol", "glycerol"]
    for we, wg in ((0.0, 0.0), (0.05, 0.05), (0.1, 0.1), (0.15, 0.05)):
        cond = _condition(ethanol=we, glycerol=wg)
        M = numpy.array([[_at(mix, D[(a, b)], cond, meter**2 / second) for b in indep] for a in indep])
        assert numpy.min(numpy.real(numpy.linalg.eigvals(M))) > 0


def test_the_molar_volume_fallback_warns():
    # The fallback is molar_mass/mass_density, i.e. the room-temperature liquid volume, which is too
    # small and silently overestimates the diffusivity by up to ~17%.
    fake = new_pure_liquid("testliquid_without_molar_volume", molar_mass=60 * gram / mol,
                           mass_density=800 * kilogram / meter**3, override=True)
    assert fake.molar_volume_for_Wilke_Chang_eq is None
    V, estimated = get_molar_volume_at_boiling_point(fake)
    assert estimated
    assert float(V / ((centi * meter) ** 3 / mol)) == pytest.approx(75.0, rel=1e-12)

    @MaterialProperties.register(override=True)
    class MixtureWithoutMolarVolume(MixtureLiquidProperties):
        components = {"water", "testliquid_without_molar_volume"}
        passive_field = "water"

        def __init__(self, pure_properties):
            super().__init__(pure_properties)
            self.set_by_weighted_average("mass_density")
            self.set_by_weighted_average("dynamic_viscosity")

    mix = Mixture(get_pure_liquid("water") + 0.2 * get_pure_liquid("testliquid_without_molar_volume"))
    with pytest.warns(UserWarning, match="molar volume"):
        fickian_mass_diffusivity_matrix(mix, "vignes", thermodynamic_factor=False,
                                        use_subexpressions=False)


def test_asking_for_a_symbolic_factor_of_a_multi_return_model_fails_early():
    # Not at code generation time, hours later, with a C compiler error.
    mix = _ternary(use_multi_return=True)
    with pytest.raises(RuntimeError, match="multi-return"):
        thermodynamic_factor_matrix(mix, ["ethanol", "glycerol"], mode="symbolic")


def test_an_ideal_mixture_warns_that_it_is_ideal():
    mix = _ternary(activity=False)
    with pytest.warns(UserWarning, match="activity"):
        fickian_mass_diffusivity_matrix(mix, "given",
                                        maxwell_stefan_diffusivities={("water", "ethanol"): D_MS,
                                                                      ("water", "glycerol"): D_MS,
                                                                      ("ethanol", "glycerol"): D_MS},
                                        use_subexpressions=False)


def test_an_already_set_diffusivity_is_kept_unless_asked_otherwise():
    mix = _binary()
    mix.set_diffusion_coefficient("12hexanediol", D_MS)
    with pytest.warns(UserWarning, match="already set"):
        written = mix.set_estimate_diffusivities()
    assert written == {}
    assert mix.get_diffusion_coefficient("12hexanediol") is D_MS
    written = mix.set_estimate_diffusivities(overwrite=True)
    assert set(written.keys()) == {("12hexanediol", "12hexanediol")}
    assert mix.get_diffusion_coefficient("12hexanediol") is not D_MS


def test_wilke_chang_reproduces_its_own_arithmetic_and_a_known_diffusivity():
    # Ethanol at infinite dilution in water at 25 degC, measured as about 1.24e-9 m^2/s. The
    # correlation claims 10-20%, and this is what guards the cm^2/s, cP and g/mol conversions from
    # cancelling wrongly against each other.
    water, ethanol = get_pure_liquid("water"), get_pure_liquid("ethanol")
    mix = _binary()  # only needed to evaluate the temperature-dependent water viscosity
    cond = _condition(**{"12hexanediol": 0.0})
    D0 = wilke_chang_diffusivity(ethanol, water, water.dynamic_viscosity)
    assert _at(mix, D0, cond, meter**2 / second) == pytest.approx(1.24e-9, rel=0.25)

    # The reverse pair against the correlation's own arithmetic rather than a measurement: Wilke-Chang
    # is known to be poor when water is the solute, and this is a unit check, not a physics claim.
    #   7.4e-8*sqrt(1.5*46.07)*298.15/(1.2*18.9^0.6) cm^2/s
    hand = 7.4e-8 * math.sqrt(1.5 * 46.07) * 298.15 / (1.2 * 18.9**0.6) * 1e-4
    Dwe = wilke_chang_diffusivity(water, ethanol, ethanol.dynamic_viscosity, temperature=298.15 * kelvin)
    assert _at(mix, Dwe, cond, meter**2 / second) == pytest.approx(hand, rel=1e-3)


def test_the_spinodal_scan_reports_what_it_finds():
    # The diagnostic to reach for when a solve with an estimated matrix diverges: it says where the
    # matrix stops being positive definite. Water/1,2-hexanediol comes close -- Gamma drops to 0.27,
    # a factor of four slower than the Maxwell-Stefan diffusivity on its own -- but AIOMFAC keeps it
    # positive over the whole range, so an empty report is the right answer here.
    from pyoomph.materials.diffusivity_estimates import scan_thermodynamic_factor
    mix = _binary()
    assert scan_thermodynamic_factor(mix, npoints=21, temperature=25 * celsius) == []
    Gamma = thermodynamic_factor_matrix(mix, ["12hexanediol"])
    sampled = [_at(mix, Gamma[0][0], _condition(**{"12hexanediol": w}))
               for w in numpy.linspace(0, 1, 21)]
    assert min(sampled) < 0.3 and min(sampled) > 0

    # And it does report one that is not positive definite.
    reported = scan_thermodynamic_factor(_ternary(activity=False), npoints=4,
                                         matrix={("ethanol", "ethanol"): -1.0,
                                                 ("ethanol", "glycerol"): 0.0,
                                                 ("glycerol", "ethanol"): 0.0,
                                                 ("glycerol", "glycerol"): 1.0})
    assert reported and all(value == pytest.approx(-1.0) for _, value in reported)


def test_the_hydrodynamic_radius_is_of_molecular_size():
    for name, expected in (("water", 1.9e-10), ("glycerol", 3.4e-10)):
        r = float(hydrodynamic_radius(get_pure_liquid(name)) / meter)
        assert r == pytest.approx(expected, rel=0.1)


class _EstimatedDiffusionProblem(Problem):
    """A 1D problem whose diffusivity matrix is the estimate, so that it has to survive code
    generation -- including the activity coefficients that sit inside the thermodynamic factor."""

    def __init__(self, ternary: bool):
        super().__init__()
        self.ternary = ternary
        self.fieldnames: list[str] = []

    def define_problem(self):
        # The ternary keeps the multi-return default, so its thermodynamic factor is the finite
        # difference; the binary is small enough to differentiate symbolically.
        mix = _ternary() if self.ternary else _binary(use_multi_return=False)
        mix.set_estimate_diffusivities(overwrite=True)
        self.set_scaling(spatial=1 * milli * meter, temporal=1 * second)
        self.define_named_var(temperature=25 * celsius)
        mix.set_reference_scaling_to_problem(self, temperature=25 * celsius)
        self += LineMesh(N=8, size=1 * milli * meter, name="dom")
        eqs = CompositionAdvectionDiffusionEquations(mix, wind=0)
        ramp = 0.05 * var("coordinate_x") / (1 * milli * meter)
        self.fieldnames = sorted(mix.components - {mix.passive_field})
        ic = {"massfrac_" + c: 0.1 + (ramp if i == 0 else 0)
              for i, c in enumerate(self.fieldnames)}
        eqs += InitialCondition(**ic)
        self += eqs @ "dom"


@pytest.mark.parametrize("ternary", [True, False],
                         ids=["ternary_finite_difference_gamma", "binary_symbolic_gamma"])
def test_the_estimated_diffusivities_compile_and_solve(tmp_path, ternary):
    # The point of this test is code generation, not the numbers: a symbolically differentiated
    # multi-return gamma produces a C file that gcc rejects, and a Vignes interpolation written as a
    # power with a field exponent segfaults GiNaC's collect_common_factors. Neither shows up in any
    # of the tests above, which never leave Python.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _EstimatedDiffusionProblem(ternary=ternary) as problem:
            problem.set_output_directory(str(tmp_path / ("out_" + str(ternary))))
            problem.run(3 * second, numouts=1, startstep=1 * second, temporal_error=None)
            mesh = problem.get_mesh("dom")
            lo = [mesh.get_maximum_value_of_field("massfrac_" + n, minimum_instead=True,
                                                  dimensional=False) for n in problem.fieldnames]
            hi = [mesh.get_maximum_value_of_field("massfrac_" + n, dimensional=False)
                  for n in problem.fieldnames]
    assert all(v > 0 for v in lo)
    assert sum(hi) < 1
