#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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
AIOMFAC with ions: the activity coefficients of a solution that contains a salt.

Checked against AIOMFAC itself. The reference numbers below were produced by building the AIOMFAC
Fortran source (https://github.com/andizuend/AIOMFAC, commit 492b091, version 3.13) with gfortran and
running its own input format; they are frozen here so that the tests need no Fortran, and
citools/generate_aiomfac_parameters.py is what regenerates the parameters they depend on.

What the three sections check:

  1. THE PARAMETERS are the ones AIOMFAC has, including the ones a previous import got wrong -- two
     ions were attached to the wrong species, and the ion molar masses were in g/mol among neutrals
     in kg/mol.
  2. THE NUMBERS match AIOMFAC for water + NaCl and for water + glycerol + NaCl, in both the solvent
     activity coefficients and the molality-based ionic ones.
  3. THE THREE BACK-ENDS agree: the symbolic (GiNaC) expressions, the numpy evaluation of the
     multi-return expression, and the C it generates. The middle- and long-range maths is written
     once and rendered three ways, and this is what says the rendering is faithful.
"""

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy
import pytest

from pyoomph.expressions import var
from pyoomph.expressions.units import *
from pyoomph.materials import *
from pyoomph.materials.activity import ActivityModel
from pyoomph.materials.activity_electrolyte import (AIOMFACElectrolyteMixture,
                                                    AIOMFACElectrolyteMultiReturnExpression,
                                                    CCodeExpressionGenerator, CExpr,
                                                    FloatExpressionGenerator)
import pyoomph.materials.default_materials  # noqa: F401  (registers the material library)
import pyoomph.materials.ions  # noqa: F401  (registers the ions and salts)

#: Glycerol as AIOMFAC sees it, and as pyoomph's own library defines it.
GLYCEROL_GROUPS = {"CH2(hydroxy)": 2, "CH(hydroxy)": 1, "OH(new)": 3}
#: From the material library rather than typed out: a mole fraction computed with a molar mass that
#: differs in the sixth digit shows up as a difference in the sixth digit of an activity coefficient,
#: which would look like a disagreement between two back-ends that in fact agree.
M_WATER = float(get_pure_liquid("water").molar_mass / (kilogram / mol))
M_GLYCEROL = float(get_pure_liquid("glycerol").molar_mass / (kilogram / mol))


def _aiomfac_model():
    m = ActivityModel.get_activity_model_by_name("AIOMFAC")
    return m


# =================================================================================================
#  1. The parameters
# =================================================================================================

def test_ion_subgroups_are_the_ones_aiomfac_says_they_are():
    # A previous import had 246 as CH3COO- and 247 as SCN-. They are IO3- and OH-, which is not a
    # naming quibble: the whole middle-range table is indexed by these ids, so a wrong name means
    # silently computing with another ion's parameters.
    m = _aiomfac_model()
    by_index = {sg.index: sg.name for sg in m.subgroups.values() if sg.index is not None}
    assert by_index[202] == "Na+" and by_index[242] == "Cl-"
    assert by_index[246] == "IO3-" and by_index[247] == "OH-"
    assert by_index[261] == "SO4--"
    assert "SCN-" not in m.subgroups and "CH3COO-" not in m.subgroups


def test_molar_masses_are_all_in_kg_per_mol():
    # The ions used to be in g/mol while every neutral subgroup was in kg/mol -- harmless only for as
    # long as nothing read them, which stopped being true when the middle-range part arrived.
    m = _aiomfac_model()
    assert m.subgroups["Na+"].molar_mass == pytest.approx(0.02299, rel=1e-6)
    assert m.subgroups["Cl-"].molar_mass == pytest.approx(0.035453, rel=1e-6)
    assert m.subgroups["H2O"].molar_mass == pytest.approx(0.01801528, rel=1e-6)
    for sg in m.subgroups.values():
        if sg.molar_mass is not None:
            assert sg.molar_mass < 1.0, sg.name + " looks like g/mol"


def test_ions_carry_their_charge():
    m = _aiomfac_model()
    assert m.subgroups["Na+"].charge == 1 and m.subgroups["Ca++"].charge == 2
    assert m.subgroups["Cl-"].charge == -1 and m.subgroups["SO4--"].charge == -2
    assert m.subgroups["H2O"].charge == 0 and not m.subgroups["H2O"].is_ion()
    assert m.subgroups["Na+"].is_ion()


def test_the_model_refuses_a_mixture_it_has_no_parameters_for():
    # AIOMFAC's tables carry a marker where a pair was never fitted. Treating that as zero would be
    # an ideal mixture where AIOMFAC itself stops with an error, so pyoomph stops too.
    m = _aiomfac_model()
    assert len(m.undetermined_interactions) > 0
    assert (4, 71) in m.undetermined_interactions


# =================================================================================================
#  2. The numbers, against AIOMFAC itself
# =================================================================================================

#: Water + NaCl at 298.15 K, from AIOMFAC 3.13. Per molality: mole fraction of water on the
#: all-species basis, then the activity coefficients of water, Na+ and Cl-.
AQUEOUS_NACL = {
    0.1: (9.96410E-01, 1.00024E+00, 7.76755E-01, 7.74964E-01),
    0.5: (9.82304E-01, 1.00130E+00, 6.79902E-01, 6.72185E-01),
    1.0: (9.65223E-01, 1.00166E+00, 6.60712E-01, 6.46000E-01),
    2.0: (9.32783E-01, 9.98125E-01, 6.86209E-01, 6.56784E-01),
    3.0: (9.02453E-01, 9.88550E-01, 7.45733E-01, 6.99485E-01),
    5.0: (8.47348E-01, 9.52613E-01, 9.21236E-01, 8.32401E-01),
}

#: Water + glycerol + NaCl at 298.15 K: (mass fraction glycerol, mass fraction NaCl, molality,
#: x_water, gamma_water, x_glycerol, gamma_glycerol, gamma_Na, gamma_Cl).
GLYCEROL_NACL = [
    (0.20, 0.01, 1.72835E-01, 9.45781E-01, 9.96456E-01, 4.68385E-02, 5.83924E-01, 1.64975E+00, 1.11453E+00),
    (0.20, 0.05, 9.00563E-01, 9.14691E-01, 9.86427E-01, 4.77148E-02, 7.31028E-01, 1.51520E+00, 9.70730E-01),
    (0.40, 0.02, 3.49198E-01, 8.64926E-01, 9.77957E-01, 1.16687E-01, 6.82116E-01, 3.51742E+00, 1.57315E+00),
    (0.50, 0.10, 1.90119E+00, 7.14975E-01, 9.00883E-01, 1.74828E-01, 9.29490E-01, 6.15467E+00, 1.90899E+00),
    (0.10, 0.15, 3.01953E+00, 8.70031E-01, 9.61636E-01, 2.26925E-02, 1.40735E+00, 1.18671E+00, 8.72816E-01),
]


def _aqueous_nacl_mixture():
    return AIOMFACElectrolyteMixture(_aiomfac_model(), {"water": {"H2O": 1}}, ["Na+", "Cl-"])


def _glycerol_nacl_mixture():
    return AIOMFACElectrolyteMixture(_aiomfac_model(),
                                     {"water": {"H2O": 1}, "glycerol": GLYCEROL_GROUPS},
                                     ["Na+", "Cl-"])


@pytest.mark.parametrize("molality", sorted(AQUEOUS_NACL))
def test_aqueous_nacl_matches_aiomfac(molality):
    # The solvent is pure water, so its salt-free mole fraction is 1 and pyoomph's activity
    # coefficient is the water activity outright. AIOMFAC reports gamma against the all-species mole
    # fraction, hence the product on the right: the two conventions differ by exactly that factor,
    # and getting it wrong is invisible without a salt.
    x_water, gamma_w, gamma_na, gamma_cl = AQUEOUS_NACL[molality]
    mix = _aqueous_nacl_mixture()
    gen = FloatExpressionGenerator(298.15)
    res = mix.activity_coefficients(gen, {"water": 1.0}, {"Na+": molality, "Cl-": molality}, 298.15)
    assert res["water"] == pytest.approx(gamma_w * x_water, rel=1e-5)
    assert res["Na+"] == pytest.approx(gamma_na, rel=1e-5)
    assert res["Cl-"] == pytest.approx(gamma_cl, rel=1e-5)


@pytest.mark.parametrize("case", range(len(GLYCEROL_NACL)))
def test_glycerol_water_nacl_matches_aiomfac(case):
    # The case that exercises the middle-range machinery properly: more than one main group, so the
    # mass-weighted main group masses and their mean actually do something. Point 4 has gamma(Na+)
    # above 6, i.e. far from any regime where a mistake would hide.
    wg, ws, molality, xw, gw, xg, gg, gna, gcl = GLYCEROL_NACL[case]
    wsolv = 1.0 - ws
    nw, ng = (wsolv - wg) / M_WATER, wg / M_GLYCEROL
    x_sf = {"water": nw / (nw + ng), "glycerol": ng / (nw + ng)}
    mix = _glycerol_nacl_mixture()
    gen = FloatExpressionGenerator(298.15)
    res = mix.activity_coefficients(gen, x_sf, {"Na+": molality, "Cl-": molality}, 298.15)
    # Compared as activities, which is what the two conventions have in common.
    assert res["water"] * x_sf["water"] == pytest.approx(gw * xw, rel=1e-4)
    assert res["glycerol"] * x_sf["glycerol"] == pytest.approx(gg * xg, rel=1e-4)
    assert res["Na+"] == pytest.approx(gna, rel=1e-4)
    assert res["Cl-"] == pytest.approx(gcl, rel=1e-4)


def test_the_mean_ionic_coefficient_has_the_right_shape():
    # gamma_pm of aqueous NaCl falls to a minimum near 1 molal and climbs again -- a shape a wrong
    # implementation does not produce by accident. Literature: 0.778, 0.657, 0.714 at 0.1, 1 and 3.
    mix = _aqueous_nacl_mixture()
    gen = FloatExpressionGenerator(298.15)
    got = {}
    for m in (0.1, 1.0, 3.0):
        res = mix.activity_coefficients(gen, {"water": 1.0}, {"Na+": m, "Cl-": m}, 298.15)
        got[m] = mix.mean_ionic_activity_coefficient(res, "Na+", "Cl-", 1, 1)
    assert got[0.1] == pytest.approx(0.778, rel=2e-2)
    assert got[1.0] == pytest.approx(0.657, rel=3e-2)
    assert got[3.0] == pytest.approx(0.714, rel=3e-2)
    assert got[1.0] < got[0.1] and got[1.0] < got[3.0]


def test_the_dilute_limit_follows_the_debye_hueckel_slope():
    # As the salt vanishes the middle range goes with it and only the long-range term survives, so
    # ln(gamma_pm) must approach -A|z+z-|sqrt(I). This is the one part of the model with no fitted
    # parameters at all, and the limit is what it exists to reproduce.
    mix = _aqueous_nacl_mixture()
    gen = FloatExpressionGenerator(298.15)
    for m in (1e-4, 1e-5):
        res = mix.activity_coefficients(gen, {"water": 1.0}, {"Na+": m, "Cl-": m}, 298.15)
        gpm = mix.mean_ionic_activity_coefficient(res, "Na+", "Cl-", 1, 1)
        # The Debye-Hueckel limiting law on the molality scale, A = 1.172 kg^0.5/mol^0.5 at 298 K.
        assert math.log(gpm) == pytest.approx(-1.172 * math.sqrt(m), rel=0.05)


# =================================================================================================
#  3. The three back-ends, and the material library
# =================================================================================================

def _float_reference(x_sf, molal, T=298.15):
    mix = _glycerol_nacl_mixture()
    return mix.activity_coefficients(FloatExpressionGenerator(T), x_sf, molal, T)


def test_the_multi_return_evaluation_agrees_with_the_direct_one():
    # Same maths, same generator, but reached through the multi-return machinery: this is what
    # catches an argument that is ordered differently on the two paths.
    mix = _glycerol_nacl_mixture()
    mr = AIOMFACElectrolyteMultiReturnExpression(mix)
    assert mr.argument_order == ["glycerol", "water", "Na+", "Cl-"]
    args = numpy.array([0.1, 0.9, 0.5, 0.5, 298.15])
    res = numpy.zeros(4)
    mr.eval(0, args, res, numpy.zeros(4 * 5))
    ref = _float_reference({"water": 0.9, "glycerol": 0.1}, {"Na+": 0.5, "Cl-": 0.5})
    for i, name in enumerate(mr.result_order):
        assert res[i] == pytest.approx(ref[name], rel=1e-12)


def test_the_multi_return_jacobian_is_a_sensible_finite_difference():
    mix = _glycerol_nacl_mixture()
    mr = AIOMFACElectrolyteMultiReturnExpression(mix)
    args = numpy.array([0.1, 0.9, 0.5, 0.5, 298.15])
    res, der = numpy.zeros(4), numpy.zeros(4 * 5)
    mr.eval(1, args, res, der)
    # More salt lowers the water activity coefficient, which is the entire point of the exercise.
    i_water, j_na = mr.result_order.index("water"), mr.argument_order.index("Na+")
    assert der[i_water * 5 + j_na] < 0.0
    # And a central difference of the same quantity agrees with it.
    h = 1e-6
    up = numpy.zeros(4); dn = numpy.zeros(4)
    a2 = args.copy(); a2[j_na] += h
    mr.eval(0, a2, up, numpy.zeros(4 * 5))
    a2[j_na] -= 2 * h
    mr.eval(0, a2, dn, numpy.zeros(4 * 5))
    assert der[i_water * 5 + j_na] == pytest.approx((up[i_water] - dn[i_water]) / (2 * h), rel=1e-4)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs a C compiler to check generated code")
def test_the_generated_c_computes_the_same_numbers():
    # The C path is the one that actually runs in a simulation, and it is the one nothing else would
    # notice was wrong. Compiled here and compared against the numpy evaluation of the same code.
    mix = _glycerol_nacl_mixture()
    gen = CCodeExpressionGenerator(temperature="T_in_K")
    x = {"water": CExpr("x_w"), "glycerol": CExpr("x_g")}
    molal = {"Na+": CExpr("m_na"), "Cl-": CExpr("m_cl")}
    res = mix.activity_coefficients(gen, x, molal, gen.get_temperature_in_kelvin())
    names = ["water", "glycerol", "Na+", "Cl-"]
    body = "\n    ".join(gen.lines)
    prints = "\n    ".join('printf("%%.17g\\n", %s);' % res[n].code for n in names)
    src = ('#include <stdio.h>\n#include <stdlib.h>\n#include <math.h>\nint main(int argc,char**argv){\n'
           '    const double T_in_K=298.15;\n'
           '    double x_w=atof(argv[1]), x_g=atof(argv[2]), m_na=atof(argv[3]), m_cl=atof(argv[4]);\n'
           '    %s\n    %s\n    return 0;\n}\n' % (body, prints))
    with tempfile.TemporaryDirectory() as d:
        csrc, exe = Path(d) / "aiomfac.c", Path(d) / "aiomfac"
        csrc.write_text(src)
        subprocess.run(["gcc", "-O2", "-o", str(exe), str(csrc), "-lm"], check=True)
        for xw, xg, ms in ((0.9, 0.1, 0.5), (0.7, 0.3, 2.0), (0.5, 0.5, 0.1)):
            out = subprocess.run([str(exe), str(xw), str(xg), str(ms), str(ms)],
                                 capture_output=True, text=True, check=True).stdout.split()
            ref = _float_reference({"water": xw, "glycerol": xg}, {"Na+": ms, "Cl-": ms})
            for name, value in zip(names, out):
                assert float(value) == pytest.approx(ref[name], rel=1e-10)


def test_the_symbolic_expressions_agree_with_the_numbers():
    # The third back-end: GiNaC expressions built through the material, evaluated at a composition.
    T = 298.15 * kelvin
    mix = Mixture(get_pure_liquid("water") + 20 * percent * get_pure_liquid("glycerol")
                  + 1 * molar * get_salt("NaCl"))
    mix.set_activity_coefficients_by_unifac("AIOMFAC", use_multi_return=False)
    rho = mix.get_reference_mass_density(T)
    molality = 0.7
    wsf_g = 0.2
    nw, ng = (1 - wsf_g) / M_WATER, wsf_g / M_GLYCEROL
    x_sf = {"water": nw / (nw + ng), "glycerol": ng / (nw + ng)}
    c = molality * (mol / kilogram) * rho
    ic = {"massfrac_water": 1 - wsf_g, "massfrac_glycerol": wsf_g, "temperature": T,
          "c_Na_p": c, "c_Cl_m": c}
    ref = _float_reference(x_sf, {"Na+": molality, "Cl-": molality})
    # Machine precision, not a physics tolerance: the same expression tree evaluated two ways. The
    # molality round trip -- molality to concentration in the test, back to molality in the
    # expression -- is exact, and the mole fractions agree to 1e-16, so anything above roundoff here
    # means the two back-ends are not computing the same thing.
    for name in ("water", "glycerol"):
        got = float(mix.evaluate_at_condition(mix.activity_coefficients[name], ic))
        assert got == pytest.approx(ref[name], rel=1e-12)
    for ion, sub in (("Na+", "Na+"), ("Cl-", "Cl-")):
        got = float(mix.evaluate_at_condition(mix.ion_activity_coefficients[ion], ic))
        assert got == pytest.approx(ref[sub], rel=1e-12)


def test_salt_lowers_the_vapour_pressure_of_the_solvent():
    # The reason any of this exists: an evaporating brine has a lower water activity, so it dries
    # more slowly, and that now follows from the material rather than having to be imposed.
    T = 298.15 * kelvin
    salted = Mixture(get_pure_liquid("water") + 1 * molar * get_salt("NaCl"))
    salted.set_activity_coefficients_by_unifac("AIOMFAC", use_multi_return=False)
    rho = salted.get_reference_mass_density(T)
    a = {}
    for molality in (0.0, 1.0, 5.0):
        c = molality * (mol / kilogram) * rho
        ic = {"temperature": T, "c_Na_p": c, "c_Cl_m": c}
        a[molality] = float(salted.evaluate_at_condition(salted.activity_coefficients["water"], ic))
    assert a[0.0] == pytest.approx(1.0, rel=1e-9)          # pure water is its own reference
    assert a[1.0] == pytest.approx(0.966822, rel=1e-4)     # AIOMFAC: 1.00166 * 0.965223
    assert a[5.0] == pytest.approx(0.807195, rel=1e-4)
    assert a[5.0] < a[1.0] < a[0.0]


def test_a_model_without_electrolyte_parameters_says_so():
    mix = Mixture(get_pure_liquid("water") + 20 * percent * get_pure_liquid("ethanol")
                  + 1 * milli * molar * get_salt("NaCl"))
    with pytest.raises(RuntimeError, match="knows nothing about ions"):
        mix.set_activity_coefficients_by_unifac("Original")


def test_an_ion_aiomfac_has_no_parameters_for_is_refused():
    # Zinc is in pyoomph's ion library because it has a diffusivity and a charge; AIOMFAC has no
    # middle-range parameters for it, and inventing them would be worse than refusing.
    mix = Mixture(get_pure_liquid("water") + 20 * percent * get_pure_liquid("glycerol")
                  + 1 * milli * molar * get_salt("ZnSO4"))
    with pytest.raises(RuntimeError, match="no electrolyte parameters"):
        mix.set_activity_coefficients_by_unifac("AIOMFAC")


def test_an_unsalted_mixture_is_unchanged_by_all_of_this():
    # The regenerated parameters must not move an existing result. These are the values the previous
    # tables gave, which they reproduce to eight digits -- the difference is that the old table had
    # been round-tripped through single precision (697.20001221 for 697.2).
    for a, b, xb, expect in (("water", "glycerol", 0.1, (0.9903253366, 0.5986314428)),
                             ("water", "glycerol", 0.5, (0.8390852716, 0.9503364580)),
                             ("water", "ethanol", 0.1, (1.0327959411, 2.8155014015)),
                             ("water", "ethanol", 0.5, (1.3868865397, 1.2126399947))):
        mix = Mixture(get_pure_liquid(a) + xb * get_pure_liquid(b), quantity="mole_fraction",
                      temperature=20 * celsius)
        for name, ref in zip((a, b), expect):
            got = float(mix.evaluate_at_condition(mix.activity_coefficients[name], "IC",
                                                  temperature=20 * celsius))
            assert got == pytest.approx(ref, rel=1e-7)
