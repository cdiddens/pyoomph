#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 
 
from .units import *

gas_constant=8.31446261815324*joule/(kelvin*mol)
k_Boltzmann= 1.38064852e-23 * joule/kelvin

N_Avogadro=6.02214076e23/mol

epsilon_0=8.854187817e-12*farad/meter # vacuum permittivity
c_of_light=2.99792458e8*meter/second # speed of light in vacuum
mu_0=4*pi*1e-7*henry/meter # vacuum permeability

elementary_charge=1.602176634e-19*coloumb # elementary charge