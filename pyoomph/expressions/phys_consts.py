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
 
 
from .units import *
from .generic import var, rational_num, Expression, ExpressionOrNum


#: Universal gas constant
gas_constant=8.31446261815324*joule/(kelvin*mol) 
#: Boltzmann constant. The exact post-2019-SI value: it used to be the CODATA-2014 value
#: 1.38064852e-23, which made k_Boltzmann*N_Avogadro differ from gas_constant by 3.5e-8 relative,
#: while gas_constant and N_Avogadro on either side of it were already the exact post-2019 ones.
#: Poisson-Boltzmann and Nernst-Planck are written interchangeably with e/(k_B*T) and F/(R*T), so
#: that inconsistency put an unexplained floor under any test comparing the two forms.
k_Boltzmann= 1.380649e-23 * joule/kelvin
#: Avogadro's number
N_Avogadro=6.02214076e23/mol

epsilon_0=8.854187817e-12*farad/meter # vacuum permittivity
c_of_light=2.99792458e8*meter/second # speed of light in vacuum
mu_0=4*pi*1e-7*henry/meter # vacuum permeability

elementary_charge=1.602176634e-19*coulomb # elementary charge

#: Faraday constant, F = e*N_A. Exact in the post-2019 SI, since both factors are exact.
faraday_constant=elementary_charge*N_Avogadro


# The following are functions rather than module-level constants because they depend on the
# temperature, which in pyoomph is usually a field (var("temperature")) rather than a number.

def thermal_voltage(temperature:ExpressionOrNum=var("temperature"))->Expression:
    """
    The thermal voltage :math:`R T/F` (about 25.693 mV at 298.15 K).

    This is the natural potential scale of every electrokinetic problem and the argument of every
    Boltzmann factor in :py:mod:`pyoomph.equations.electrostatics`.

    Args:
        temperature: The temperature, by default the field ``var("temperature")``.

    Returns:
        The thermal voltage.
    """
    return gas_constant*temperature/faraday_constant


def debye_length(permittivity:ExpressionOrNum,ionic_strength:ExpressionOrNum,temperature:ExpressionOrNum=var("temperature"))->Expression:
    """
    The Debye screening length :math:`\\sqrt{\\varepsilon R T/(2F^2 I)}`.

    Args:
        permittivity: The absolute permittivity of the solvent.
        ionic_strength: The *molar* ionic strength :math:`I=\\frac{1}{2}\\sum_i z_i^2 c_i`.
        temperature: The temperature, by default the field ``var("temperature")``.

    Returns:
        The Debye length.
    """
    return (permittivity*gas_constant*temperature/(2*faraday_constant**2*ionic_strength))**rational_num(1,2)


def bjerrum_length(permittivity:ExpressionOrNum,temperature:ExpressionOrNum=var("temperature"))->Expression:
    """
    The Bjerrum length :math:`e^2/(4\\pi\\varepsilon k_B T)`, i.e. the distance at which the
    electrostatic interaction of two elementary charges equals the thermal energy.

    Args:
        permittivity: The absolute permittivity of the solvent.
        temperature: The temperature, by default the field ``var("temperature")``.

    Returns:
        The Bjerrum length.
    """
    return elementary_charge**2/(4*pi*permittivity*k_Boltzmann*temperature)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
