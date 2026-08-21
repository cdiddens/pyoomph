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
    from pyoomph.materials import get_pure_liquid, get_ion

    water = get_pure_liquid("water")
    water.add_salt("Na+", "Cl-", 1*milli*mol/liter)

:py:func:`~pyoomph.materials.generic.get_ion` gets one on its own, e.g. to override a property
before dissolving it. Anything not listed here is declared with
:py:func:`~pyoomph.materials.generic.new_ion`, which registers it in exactly the same table.

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


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
