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
The standard library of ionic species, registered the same way every other material is.

Import this module to make the common ions available by name::

    import pyoomph.materials.ions   # registers them
    from pyoomph.materials import get_pure_liquid, get_ion, get_salt, Mixture

    water = get_pure_liquid("water")
    water.add_salt("Na+", "Cl-", 1*milli*mol/liter)

:py:func:`~pyoomph.materials.generic.get_ion` gets one on its own, e.g. to override a property
before dissolving it. Anything not listed here is declared with
:py:func:`~pyoomph.materials.generic.new_ion`, which registers it in exactly the same table.

**The salts are here too**, registered the same way and fetched with
:py:func:`~pyoomph.materials.generic.get_salt`. A salt is a recipe -- two ions, pulled from the table
above when it is constructed, plus the stoichiometry that electroneutrality forces -- so the way to
use one is to multiply it by a concentration and hand it to :py:func:`Mixture`::

    mix = Mixture(water + 20*percent*glycerol + 1*milli*molar*get_salt("NaCl"))

The salt is dissolved in the finished mixture and takes no part in the mass fractions: it is a
concentration, and at 1 mM it is 6e-5 of the solution by mass. A single ion works the same way, and
then it is on you to keep the set electroneutral.

**The datum per ion is the limiting molar conductivity** :math:`\\lambda_i^0` at 25 °C and infinite
dilution (CRC Handbook, Vanysek, "Ionic conductivity and diffusion at infinite dilution"), and the
diffusivity follows from it by Nernst-Einstein, :math:`D_i=\\lambda_i^0RT/(z_i^2F^2)`. One number per
ion means the two cannot drift apart; taking the tabulated :math:`D_i^0` as a second, independent
datum is what would let them.

The conversion happens **at whatever temperature it is asked for**, i.e. at ``var("temperature")``
unless :py:meth:`~pyoomph.materials.generic.IonProperties.get_diffusivity` (or
:py:func:`~pyoomph.equations.electrostatics.ions_from_material`) is given one. An isothermal problem
therefore has to say what its temperature is, the same way it already does for the other material
properties::

    self.define_named_var(temperature=25*celsius)

**The temperature dependence comes from the solvent viscosity**, by the fractional Walden rule
:math:`\\lambda_i^0\\mu^n=` const, so that :math:`D_i\\propto T/\\mu^n`. Nernst-Einstein on its own
would give :math:`D_i\\propto T` at constant :math:`\\lambda^0`, i.e. +37% between 0 and 100 °C where
the truth is a factor of five, and it would make the conductivity of an electrolyte exactly
temperature independent, which is the one thing every conductivity meter is built to correct for.
Applying it is the solvent's job, since the ion does not know what it is dissolved in --
:py:meth:`~pyoomph.materials.generic.BaseLiquidProperties.get_ion_diffusivity` is where it happens,
and everything downstream goes through there.

The exponent is 1 (plain Walden, i.e. Stokes drag) unless the ion below sets another. Fitted against
the measured conductivities over 0-45 °C, that gets Na+ to 0.4%, K+ to 3.1% and Cl- to 3.5%, against
the 45-47% a constant :math:`\\lambda^0` is out by at 0 °C. **H+ is the real exception**: it moves by
Grotthuss proton transfer rather than Stokes drag, its fitted exponent is 0.63, and the plain rule is
21% low for it at 0 °C. Above roughly 60 °C all of these degrade -- at 100 °C even the fitted
exponents are 10-20% out, and nothing here should be believed there.

The exponents are fitted from the standard multi-temperature tables (Robinson & Stokes) for the five
ions where those are unambiguous; **every other ion keeps 1.0**, which is a statement about missing
data and not about that ion being a better Stokes sphere than sodium.

The values are for **infinite dilution**. At the millimolar concentrations these models are usually
run at, the Kohlrausch correction is a percent or so; at molar concentrations it is tens of percent
and none of these numbers should be believed.
"""

from __future__ import annotations

from .generic import *
from ..expressions import *
from ..expressions.units import *
from ..expressions.phys_consts import *  # gas_constant, faraday_constant
from ..typings import *

#: The temperature the tabulated limiting molar conductivities refer to. Nothing here evaluates
#: anything at it -- it is the temperature at which these numbers are the measured ones.
table_temperature = 298.15 * kelvin

_lambda_unit = siemens * (centi * meter)**2 / mol   # S cm^2/mol, and cm^2 is not centi*m^2


#: Partial molar volume at infinite dilution and 25 degC, in cm^3/mol, on the conventional scale that
#: sets V(H+) = 0 (Millero, Chem. Rev. 71 (1971) 147). Salts are additive over these to within the
#: accuracy of the tables -- NaCl 16.62 = -1.21 + 17.83, CaCl2 17.81 against a measured 17.85, MgSO4
#: -7.19 against -7.28 -- so this is stored per ion and combined by stoichiometry, the same way the
#: ambipolar diffusivity is. Ions without an entry are refused when a salt is asked to be a
#: composition field, since there is no harmless default for a volume.
#:
#: Negative entries are electrostriction, not typos: a small, highly charged ion draws the
#: surrounding water in tighter than its own volume displaces.
PARTIAL_MOLAR_VOLUME = {
    "H+": 0.0, "Li+": -0.88, "Na+": -1.21, "K+": 9.02, "NH4+": 17.86, "Cs+": 21.34,
    "Mg2+": -21.17, "Ca2+": -17.85, "Ba2+": -12.47,
    "F-": -1.16, "Cl-": 17.83, "Br-": 24.71, "I-": 36.22, "OH-": -4.04,
    "NO3-": 29.00, "ClO4-": 44.12, "HCO3-": 23.40, "CO3 2-": -6.70, "SO4 2-": 13.98,
    "CH3COO-": 40.46,
}


#: How an ion of this library is spelled as an AIOMFAC subgroup. Only the ones AIOMFAC actually has
#: middle-range parameters for are listed; the rest have no entry, and an activity model asked about
#: them says so rather than guessing. AIOMFAC writes a double charge as "++" where this library
#: writes "2+", which is the only reason most of these entries exist.
AIOMFAC_SUBGROUP_OF_ION = {
    "H+": "H+", "Li+": "Li+", "Na+": "Na+", "K+": "K+", "NH4+": "NH4+",
    "Mg2+": "Mg++", "Ca2+": "Ca++",
    "OH-": "OH-", "Cl-": "Cl-", "Br-": "Br-", "I-": "I-", "NO3-": "NO3-", "HCO3-": "HCO3-",
    "SO4 2-": "SO4--", "CO3 2-": "CO3--",
}


class LibraryIon(IonProperties):
    """
    Base class of the tabulated ions: turns the three class-level data into an
    :py:class:`~pyoomph.materials.generic.IonProperties`.

    Subclasses set :py:attr:`name`, :py:attr:`z`, :py:attr:`limiting_molar_conductivity_25C` and
    :py:attr:`molar_mass_amu` as class variables and are registered with
    :py:meth:`~pyoomph.materials.generic.MaterialProperties.register`.
    """

    #: Charge number :math:`z_i`.
    z: int
    #: Limiting molar conductivity in S cm^2/mol at 25 °C, i.e. as the tables print it. Note that
    #: tables often list the *equivalent* conductivity :math:`\lambda^0/|z|` instead; these are the
    #: values for the ion itself, so sulfate is 160.0 and not 80.0.
    limiting_molar_conductivity_25C: float
    #: Molar mass in g/mol.
    molar_mass_amu: float
    #: Exponent of the fractional Walden rule, see :py:attr:`IonProperties.walden_exponent`. The
    #: default 1 is the plain rule, i.e. Stokes drag; the ions below that carry another value have it
    #: fitted to their measured conductivity over 0-45 °C. Always an exact rational -- a float
    #: exponent on a quantity that still carries units trips GiNaC up.
    walden_exponent_fit: ExpressionOrNum = 1

    def __init__(self):
        super().__init__()
        self.charge_number = self.z
        self.molar_mass = self.molar_mass_amu * gram / mol
        self.limiting_molar_conductivity = self.limiting_molar_conductivity_25C * _lambda_unit
        # diffusivity is deliberately left unset: IonProperties.get_diffusivity converts the
        # conductivity by Nernst-Einstein at the temperature it is asked for. Storing a second,
        # independent number here is what would let the two drift apart.
        self.walden_exponent = self.walden_exponent_fit
        self.aiomfac_subgroup = AIOMFAC_SUBGROUP_OF_ION.get(self.name)
        vol = PARTIAL_MOLAR_VOLUME.get(self.name)
        if vol is not None:
            self.partial_molar_volume = vol * (centi * meter)**3 / mol
        # A dissolved ion has neither of these on its own, but it goes through the pure-liquid
        # machinery, which expects them. They only matter if the ions also carry mass in a flow
        # model, and then they are the user's to set.
        self.mass_density = 1000 * kilogram / meter**3
        self.dynamic_viscosity = 1 * milli * pascal * second


# ---------------------------------------------------------------------------------------------
# Cations
# ---------------------------------------------------------------------------------------------

@MaterialProperties.register()
class IonHydrogen(LibraryIon):
    # Grotthuss hopping, not Stokes drag: H+ and OH- are the two outliers in the table by a factor
    # of five, which is why an acid conducts so much better than its salt. The same mechanism shows
    # up in the Walden exponent, which is 0.63 rather than ~0.94 -- proton transfer is far less
    # sensitive to the solvent viscosity than dragging an ion through it. Taking the plain rule for
    # H+ is 21% low at 0 °C.
    name = "H+"
    z = +1
    limiting_molar_conductivity_25C = 349.65
    molar_mass_amu = 1.008
    walden_exponent_fit = rational_num(63,100)   # 0.63


@MaterialProperties.register()
class IonLithium(LibraryIon):
    name = "Li+"
    z = +1
    limiting_molar_conductivity_25C = 38.66
    molar_mass_amu = 6.94


@MaterialProperties.register()
class IonSodium(LibraryIon):
    name = "Na+"
    z = +1
    limiting_molar_conductivity_25C = 50.08
    molar_mass_amu = 22.990
    walden_exponent_fit = rational_num(47,50)   # 0.94


@MaterialProperties.register()
class IonPotassium(LibraryIon):
    name = "K+"
    z = +1
    limiting_molar_conductivity_25C = 73.48
    molar_mass_amu = 39.098
    walden_exponent_fit = rational_num(9,10)   # 0.90


@MaterialProperties.register()
class IonAmmonium(LibraryIon):
    name = "NH4+"
    z = +1
    limiting_molar_conductivity_25C = 73.5
    molar_mass_amu = 18.038


@MaterialProperties.register()
class IonCaesium(LibraryIon):
    name = "Cs+"
    z = +1
    limiting_molar_conductivity_25C = 77.2
    molar_mass_amu = 132.905


@MaterialProperties.register()
class IonSilver(LibraryIon):
    name = "Ag+"
    z = +1
    limiting_molar_conductivity_25C = 61.9
    molar_mass_amu = 107.868


@MaterialProperties.register()
class IonMagnesium(LibraryIon):
    name = "Mg2+"
    z = +2
    limiting_molar_conductivity_25C = 106.0
    molar_mass_amu = 24.305


@MaterialProperties.register()
class IonCalcium(LibraryIon):
    name = "Ca2+"
    z = +2
    limiting_molar_conductivity_25C = 119.0
    molar_mass_amu = 40.078


@MaterialProperties.register()
class IonBarium(LibraryIon):
    name = "Ba2+"
    z = +2
    limiting_molar_conductivity_25C = 127.2
    molar_mass_amu = 137.327


@MaterialProperties.register()
class IonZinc(LibraryIon):
    name = "Zn2+"
    z = +2
    limiting_molar_conductivity_25C = 105.6
    molar_mass_amu = 65.38


@MaterialProperties.register()
class IonCopper(LibraryIon):
    name = "Cu2+"
    z = +2
    limiting_molar_conductivity_25C = 107.2
    molar_mass_amu = 63.546


@MaterialProperties.register()
class IonIron2(LibraryIon):
    name = "Fe2+"
    z = +2
    limiting_molar_conductivity_25C = 108.0
    molar_mass_amu = 55.845


@MaterialProperties.register()
class IonIron3(LibraryIon):
    name = "Fe3+"
    z = +3
    limiting_molar_conductivity_25C = 204.0
    molar_mass_amu = 55.845


@MaterialProperties.register()
class IonAluminium(LibraryIon):
    name = "Al3+"
    z = +3
    limiting_molar_conductivity_25C = 183.0
    molar_mass_amu = 26.982


# ---------------------------------------------------------------------------------------------
# Anions
# ---------------------------------------------------------------------------------------------

@MaterialProperties.register()
class IonHydroxide(LibraryIon):
    # The other Grotthuss carrier, see IonHydrogen -- but the exponent below rests on two
    # temperatures only, so it is not the evidence H+'s 0.63 is.
    name = "OH-"
    z = -1
    limiting_molar_conductivity_25C = 198.0
    molar_mass_amu = 17.007
    walden_exponent_fit = rational_num(47,50)   # 0.94


@MaterialProperties.register()
class IonFluoride(LibraryIon):
    name = "F-"
    z = -1
    limiting_molar_conductivity_25C = 55.4
    molar_mass_amu = 18.998


@MaterialProperties.register()
class IonChloride(LibraryIon):
    name = "Cl-"
    z = -1
    limiting_molar_conductivity_25C = 76.31
    molar_mass_amu = 35.453
    walden_exponent_fit = rational_num(19,20)   # 0.95


@MaterialProperties.register()
class IonBromide(LibraryIon):
    name = "Br-"
    z = -1
    limiting_molar_conductivity_25C = 78.1
    molar_mass_amu = 79.904


@MaterialProperties.register()
class IonIodide(LibraryIon):
    name = "I-"
    z = -1
    limiting_molar_conductivity_25C = 76.8
    molar_mass_amu = 126.904


@MaterialProperties.register()
class IonNitrate(LibraryIon):
    name = "NO3-"
    z = -1
    limiting_molar_conductivity_25C = 71.42
    molar_mass_amu = 62.004


@MaterialProperties.register()
class IonPerchlorate(LibraryIon):
    name = "ClO4-"
    z = -1
    limiting_molar_conductivity_25C = 67.3
    molar_mass_amu = 99.451


@MaterialProperties.register()
class IonBicarbonate(LibraryIon):
    name = "HCO3-"
    z = -1
    limiting_molar_conductivity_25C = 44.5
    molar_mass_amu = 61.017


@MaterialProperties.register()
class IonCarbonate(LibraryIon):
    name = "CO3 2-"
    z = -2
    limiting_molar_conductivity_25C = 138.6
    molar_mass_amu = 60.009


@MaterialProperties.register()
class IonSulfate(LibraryIon):
    name = "SO4 2-"
    z = -2
    limiting_molar_conductivity_25C = 160.0
    molar_mass_amu = 96.06


@MaterialProperties.register()
class IonAcetate(LibraryIon):
    name = "CH3COO-"
    z = -1
    limiting_molar_conductivity_25C = 40.9
    molar_mass_amu = 59.044


@MaterialProperties.register()
class IonDihydrogenPhosphate(LibraryIon):
    name = "H2PO4-"
    z = -1
    limiting_molar_conductivity_25C = 36.0
    molar_mass_amu = 96.987


@MaterialProperties.register()
class IonHydrogenPhosphate(LibraryIon):
    name = "HPO4 2-"
    z = -2
    limiting_molar_conductivity_25C = 114.0
    molar_mass_amu = 95.979


# ---------------------------------------------------------------------------------------------
# Salts
#
# Each one only names its two ions: the stoichiometry follows from their charge numbers (see
# SaltProperties), and the molar mass follows from that, so nothing here can disagree with the ion
# table above. All of these are strong electrolytes at the concentrations these models are run at,
# i.e. taken as fully dissociated.
#
# surface_tension_increment is dsigma/dc at 25 degC (Weissenborn & Pugh, J. Colloid Interface Sci.
# 184 (1996) 550; Ozdemir et al., Chem. Eng. Sci. 64 (2009) 2609), given only where the tables have
# it -- a salt without one contributes no Marangoni stress rather than a guessed one. It is positive
# because ions are pushed *away* from the surface: an ion there would have to give up part of its
# hydration shell and is repelled by its own image charge in the low-permittivity vapour, so the
# surface is a little purer than the bulk and costs more to make. The strong acids are the exception
# and are negative, since the proton does sit at the surface. The values are the linear coefficient
# valid from ~0.1 M up; below ~1 mM real solutions show the small Jones-Ray dip, which is not here.
# ---------------------------------------------------------------------------------------------

@MaterialProperties.register()
class SaltSodiumChloride(SaltProperties):
    name = "NaCl"
    cation_name = "Na+"
    anion_name = "Cl-"
    surface_tension_increment = 1.64 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltPotassiumChloride(SaltProperties):
    name = "KCl"
    cation_name = "K+"
    anion_name = "Cl-"
    surface_tension_increment = 1.6 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltLithiumChloride(SaltProperties):
    name = "LiCl"
    cation_name = "Li+"
    anion_name = "Cl-"
    surface_tension_increment = 1.63 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltCaesiumChloride(SaltProperties):
    name = "CsCl"
    cation_name = "Cs+"
    anion_name = "Cl-"
    surface_tension_increment = 1.54 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltAmmoniumChloride(SaltProperties):
    name = "NH4Cl"
    cation_name = "NH4+"
    anion_name = "Cl-"
    surface_tension_increment = 1.39 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltCalciumChloride(SaltProperties):
    name = "CaCl2"
    cation_name = "Ca2+"
    anion_name = "Cl-"
    surface_tension_increment = 3.66 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltMagnesiumChloride(SaltProperties):
    name = "MgCl2"
    cation_name = "Mg2+"
    anion_name = "Cl-"
    surface_tension_increment = 3.16 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltBariumChloride(SaltProperties):
    name = "BaCl2"
    cation_name = "Ba2+"
    anion_name = "Cl-"
    surface_tension_increment = 3.15 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltZincChloride(SaltProperties):
    name = "ZnCl2"
    cation_name = "Zn2+"
    anion_name = "Cl-"


@MaterialProperties.register()
class SaltCopperChloride(SaltProperties):
    name = "CuCl2"
    cation_name = "Cu2+"
    anion_name = "Cl-"


@MaterialProperties.register()
class SaltIronIIChloride(SaltProperties):
    name = "FeCl2"
    cation_name = "Fe2+"
    anion_name = "Cl-"


@MaterialProperties.register()
class SaltIronIIIChloride(SaltProperties):
    name = "FeCl3"
    cation_name = "Fe3+"
    anion_name = "Cl-"


@MaterialProperties.register()
class SaltAluminiumChloride(SaltProperties):
    name = "AlCl3"
    cation_name = "Al3+"
    anion_name = "Cl-"


@MaterialProperties.register()
class SaltSodiumBromide(SaltProperties):
    name = "NaBr"
    cation_name = "Na+"
    anion_name = "Br-"
    surface_tension_increment = 1.53 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltPotassiumBromide(SaltProperties):
    name = "KBr"
    cation_name = "K+"
    anion_name = "Br-"
    surface_tension_increment = 1.48 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltSodiumIodide(SaltProperties):
    name = "NaI"
    cation_name = "Na+"
    anion_name = "I-"
    surface_tension_increment = 1.21 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltPotassiumIodide(SaltProperties):
    name = "KI"
    cation_name = "K+"
    anion_name = "I-"
    surface_tension_increment = 1.15 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltSodiumFluoride(SaltProperties):
    name = "NaF"
    cation_name = "Na+"
    anion_name = "F-"
    surface_tension_increment = 2.02 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltSodiumSulfate(SaltProperties):
    name = "Na2SO4"
    cation_name = "Na+"
    anion_name = "SO4 2-"
    surface_tension_increment = 2.7 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltPotassiumSulfate(SaltProperties):
    name = "K2SO4"
    cation_name = "K+"
    anion_name = "SO4 2-"
    surface_tension_increment = 2.58 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltLithiumSulfate(SaltProperties):
    name = "Li2SO4"
    cation_name = "Li+"
    anion_name = "SO4 2-"


@MaterialProperties.register()
class SaltMagnesiumSulfate(SaltProperties):
    name = "MgSO4"
    cation_name = "Mg2+"
    anion_name = "SO4 2-"
    surface_tension_increment = 3.22 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltCopperSulfate(SaltProperties):
    name = "CuSO4"
    cation_name = "Cu2+"
    anion_name = "SO4 2-"


@MaterialProperties.register()
class SaltZincSulfate(SaltProperties):
    name = "ZnSO4"
    cation_name = "Zn2+"
    anion_name = "SO4 2-"
    surface_tension_increment = 3.12 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltIronIISulfate(SaltProperties):
    name = "FeSO4"
    cation_name = "Fe2+"
    anion_name = "SO4 2-"


@MaterialProperties.register()
class SaltSodiumNitrate(SaltProperties):
    name = "NaNO3"
    cation_name = "Na+"
    anion_name = "NO3-"
    surface_tension_increment = 1.16 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltPotassiumNitrate(SaltProperties):
    name = "KNO3"
    cation_name = "K+"
    anion_name = "NO3-"
    surface_tension_increment = 1.06 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltAmmoniumNitrate(SaltProperties):
    name = "NH4NO3"
    cation_name = "NH4+"
    anion_name = "NO3-"
    surface_tension_increment = 0.85 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltSilverNitrate(SaltProperties):
    name = "AgNO3"
    cation_name = "Ag+"
    anion_name = "NO3-"


@MaterialProperties.register()
class SaltCalciumNitrate(SaltProperties):
    name = "Ca(NO3)2"
    cation_name = "Ca2+"
    anion_name = "NO3-"


@MaterialProperties.register()
class SaltMagnesiumNitrate(SaltProperties):
    name = "Mg(NO3)2"
    cation_name = "Mg2+"
    anion_name = "NO3-"


@MaterialProperties.register()
class SaltSodiumPerchlorate(SaltProperties):
    name = "NaClO4"
    cation_name = "Na+"
    anion_name = "ClO4-"


@MaterialProperties.register()
class SaltLithiumPerchlorate(SaltProperties):
    name = "LiClO4"
    cation_name = "Li+"
    anion_name = "ClO4-"


@MaterialProperties.register()
class SaltSodiumBicarbonate(SaltProperties):
    name = "NaHCO3"
    cation_name = "Na+"
    anion_name = "HCO3-"
    surface_tension_increment = 1.6 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltSodiumCarbonate(SaltProperties):
    name = "Na2CO3"
    cation_name = "Na+"
    anion_name = "CO3 2-"
    surface_tension_increment = 2.85 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltPotassiumCarbonate(SaltProperties):
    name = "K2CO3"
    cation_name = "K+"
    anion_name = "CO3 2-"


@MaterialProperties.register()
class SaltSodiumAcetate(SaltProperties):
    name = "CH3COONa"
    cation_name = "Na+"
    anion_name = "CH3COO-"


@MaterialProperties.register()
class SaltSodiumDihydrogenPhosphate(SaltProperties):
    name = "NaH2PO4"
    cation_name = "Na+"
    anion_name = "H2PO4-"


@MaterialProperties.register()
class SaltPotassiumDihydrogenPhosphate(SaltProperties):
    name = "KH2PO4"
    cation_name = "K+"
    anion_name = "H2PO4-"


@MaterialProperties.register()
class SaltDisodiumHydrogenPhosphate(SaltProperties):
    name = "Na2HPO4"
    cation_name = "Na+"
    anion_name = "HPO4 2-"


@MaterialProperties.register()
class SaltDipotassiumHydrogenPhosphate(SaltProperties):
    name = "K2HPO4"
    cation_name = "K+"
    anion_name = "HPO4 2-"


@MaterialProperties.register()
class SaltSodiumHydroxide(SaltProperties):
    name = "NaOH"
    cation_name = "Na+"
    anion_name = "OH-"
    surface_tension_increment = 1.8 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltPotassiumHydroxide(SaltProperties):
    name = "KOH"
    cation_name = "K+"
    anion_name = "OH-"
    surface_tension_increment = 1.75 * milli * newton / meter / molar


@MaterialProperties.register()
class SaltLithiumHydroxide(SaltProperties):
    name = "LiOH"
    cation_name = "Li+"
    anion_name = "OH-"
    surface_tension_increment = 1.8 * milli * newton / meter / molar


# The strong acids and bases dissociate the same way and are the same object here. H2SO4 is the one
# idealization worth naming: only its first proton is strong, and treating it as 2 H+ + SO4(2-) is
# right at the dilute end and increasingly wrong towards molar, where HSO4- is a real species.

@MaterialProperties.register()
class AcidHydrochloric(SaltProperties):
    name = "HCl"
    cation_name = "H+"
    anion_name = "Cl-"
    surface_tension_increment = -0.3 * milli * newton / meter / molar


@MaterialProperties.register()
class AcidHydrobromic(SaltProperties):
    name = "HBr"
    cation_name = "H+"
    anion_name = "Br-"
    surface_tension_increment = -0.35 * milli * newton / meter / molar


@MaterialProperties.register()
class AcidNitric(SaltProperties):
    name = "HNO3"
    cation_name = "H+"
    anion_name = "NO3-"
    surface_tension_increment = -0.8 * milli * newton / meter / molar


@MaterialProperties.register()
class AcidSulfuric(SaltProperties):
    name = "H2SO4"
    cation_name = "H+"
    anion_name = "SO4 2-"
    surface_tension_increment = 0.8 * milli * newton / meter / molar


@MaterialProperties.register()
class AcidPerchloric(SaltProperties):
    name = "HClO4"
    cation_name = "H+"
    anion_name = "ClO4-"


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
