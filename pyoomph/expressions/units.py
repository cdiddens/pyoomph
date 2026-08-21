from __future__ import annotations
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
Module containing physical units (meter, second, etc) and some constants for use in expressions.
"""
 
from .. import _pyoomph_core as _pyoomph

import numpy 

from ..typings import *
import math

from .generic import ExpressionOrNum,Expression

meter = _pyoomph.GiNaC_unit("meter", "meter")
second = _pyoomph.GiNaC_unit("second", "second")
kilogram = _pyoomph.GiNaC_unit("kilogram", "kilogram")
kelvin = _pyoomph.GiNaC_unit("kelvin", "kelvin")
mol = _pyoomph.GiNaC_unit("mol", "mol")
ampere = _pyoomph.GiNaC_unit("ampere", "ampere")


#################

def _power_of_ten(d:ExpressionOrNum)->Expression:
    return _pyoomph.GiNaC_rational_number(10, 1) ** d


yotta = _power_of_ten(24)
zetta = _power_of_ten(21)
exa = _power_of_ten(18)
peta = _power_of_ten(15)
tera = _power_of_ten(12)
giga = _power_of_ten(9)
mega = _power_of_ten(6)
kilo = _power_of_ten(3)
hecto = _power_of_ten(2)
deca = _power_of_ten(1)

deci = _power_of_ten(-1)
centi = _power_of_ten(-2)
milli = _power_of_ten(-3)
micro = _power_of_ten(-6)
nano = _power_of_ten(-9)
pico = _power_of_ten(-12)
femto = _power_of_ten(-15)
atto = _power_of_ten(-18)
zepto = _power_of_ten(-21)
yocto = _power_of_ten(-24)

#####################

pi = 2 * _pyoomph.GiNaC_asin(1)
degree = pi / 180

percent = 1 / 100
gram = _power_of_ten(-3) * kilogram
minute = 60 * second
hour = 60 * minute
day = 24 * hour
litre = _power_of_ten(-3) * meter ** 3
liter = litre
#: Molar concentration, i.e. mol/litre. "1 mM NaCl" is 1*milli*molar.
molar = mol / litre

hertz = 1 / second
newton = kilogram * meter / second ** 2
pascal = newton / meter ** 2
joule = newton * meter
watt = joule / second
stokes = _power_of_ten(-4) * meter ** 2 / second

#####################
angstrom = _power_of_ten(2) * pico * meter
bar = _power_of_ten(5) * pascal
atm = 1013.25 * milli * bar
barye = pascal / 10
mmHg = 133.322387415 * pascal
torr = atm / 760
dyne = _power_of_ten(-5) * newton
darcy = 9.86923e-13 * meter ** 2

volt = watt / ampere
coulomb = ampere * second
farad= coulomb / volt
henry = volt * second / ampere
siemens = ampere / volt
#: Electrical resistance. Note that only its reciprocal, :py:data:`siemens`, used to be defined.
ohm = volt / ampere

## TODO Celsius conversion function
class CelsiusClass:
    cf = 273.15

    def __mul__(self, other:ExpressionOrNum)->_pyoomph.Expression:
        return (other + CelsiusClass.cf) * kelvin

    def __rmul__(self, other:ExpressionOrNum)->_pyoomph.Expression:
        return (other + CelsiusClass.cf) * kelvin

    def __rdiv__(self, other:ExpressionOrNum)->_pyoomph.Expression:
        return (other / kelvin - CelsiusClass.cf)

    def __div__(self, other:ExpressionOrNum)->_pyoomph.Expression:
        return (other / kelvin - CelsiusClass.cf)


celsius = CelsiusClass()

#: Written between the symbols of a compound unit, e.g. "kg m^2/s^3". Output headers are tab-joined
#: AND read back tab-separated (LoadedTextDataFile), so a space here cannot break a column - it did
#: while the reader still split the header on arbitrary whitespace.
UNIT_SEPARATOR=" "

__simplified_units:dict[str,dict[str,tuple[int,int]]] = {}

@overload
def unit_to_string(inp:ExpressionOrNum,estimate_prefix:Literal[True]=...)->tuple[str,float,float]: ...

@overload
def unit_to_string(inp:ExpressionOrNum,estimate_prefix:Literal[False])->str: ...

def unit_to_string(inp:ExpressionOrNum,estimate_prefix:bool=True) -> str | tuple[str, float, float]:
    __prefixes:dict[float,str]={1e-18:"a",1e-15:"f",1e-12:"p",1e-9:"n",1e-6:"u",1e-3:"m",1:"",
                                1e3:"k",1e6:"M",1e9:"G",1e12:"T",1e15:"P",1e18:"E"}
    __shorts = {"meter": "m", "second": "s", "kilogram": "kg", "kelvin": "K", "mol": "mol", "ampere": "A"}
    __sort_numer=["kilogram","mol","kelvin","second","ampere","meter"] # meter must be at the end!
    __sort_denom = ["kilogram", "mol", "meter","kelvin", "second", "ampere" ]

    if not isinstance(inp,Expression):
        inp=Expression(inp)
    factor, unit, _, success = _pyoomph.GiNaC_collect_units(inp) 

    if not success:
        raise ValueError("Cannot extract the unit from "+str(inp))
    contribs = _pyoomph.GiNaC_sep_base_units(unit)

    numer_mass=__shorts["kilogram"]
    prefix = ""
    factorf=float(factor)

    prefix_factor=1.0 # a __prefixes key, i.e. a power of ten
    factor_bound_factor=10

    # The prefix is written in front of the whole numerator ("m"+"m^2" -> "mm^2"), so it binds to the
    # first unit TOGETHER WITH ITS EXPONENT: mm^2 is (mm)^2 = 1e-6 m^2, not 1e-3 m^2. Treating the
    # exponent as 1 made the printed quantity differ from the input - 1e-6 m^2 came out as "1 um^2",
    # which is 1e-12 m^2, wrong by six orders of magnitude. So the prefix p has to be chosen from
    # magnitude^(1/n) and the value divided by p^n.
    # A derived name is a single symbol, so a prefix on it always has exponent 1: "pF" is (pF)^1. Look
    # the name up FIRST, because deriving the exponent from the base units gives the wrong answer for
    # any derived unit whose leading base exponent is not 1 - farad is s^4*A^2/(kg*m^2), so 1e-12 F came
    # out as "1 mF", a billion times too large. The units registered before farad all happened to lead
    # with kilogram^1, which is why this never showed.
    simplified_name=None
    for _rep,_sig in __simplified_units.items():
        if _sig==contribs:
            simplified_name=_rep
            break

    prefix_exponent=1.0
    for _un in ([] if simplified_name is not None else __sort_numer):
        _c=contribs.get(_un)
        if _c is not None and _c[0]>0:
            prefix_exponent=float(_c[0])/float(_c[1])
            break
    else:
        # No numerator at all: the prefix moves into the denominator (1/Mm), where the exponent as
        # PRINTED is the positive one.
        for _un in ([] if simplified_name is not None else __sort_denom):
            _c=contribs.get(_un)
            if _c is not None and _c[0]<0:
                prefix_exponent=-float(_c[0])/float(_c[1])
                break

    def _choose_prefix(magnitude:float,exponent:float)->"tuple[str,float]":
        """The prefix p and its numeric value, such that magnitude/p^exponent reads nicely."""
        if magnitude<=0 or exponent==0 or not math.isfinite(magnitude):
            return "",1.0
        # Compare in the unit's own root, since it is p^exponent that has to absorb the magnitude.
        root=magnitude**(1.0/exponent)
        # A hair of slack, because the root of an exact power is not exact in binary: (1e-6)^(1/3) comes
        # out just above 0.01 and a cubic metre volume would miss "mm^3" by one ulp.
        for k in sorted(__prefixes.keys()):
            if root <= factor_bound_factor * k * (1.0+1e-9):
                return __prefixes[k],k
        return "",1.0

    # The prefix follows the MAGNITUDE. Comparing the signed value made every negative quantity match
    # the smallest prefix on the first iteration, so -0.5/second came out as "1/Gs" (numerically right,
    # -5e8/Gs, but unreadable) - and eigenvalues of stable modes are exactly the negative case. Zero has
    # no meaningful prefix either; it matched "n" for the same reason.
    magnitude=abs(factorf)
    if estimate_prefix:
        prefix,prefix_factor=_choose_prefix(magnitude,prefix_exponent)

    if simplified_name is not None:
        if estimate_prefix:
            return prefix+simplified_name,factorf,prefix_factor**(-prefix_exponent)
        else:
            return simplified_name

    if estimate_prefix:
        if prefix!="":
            if "kilogram" in contribs:
                if contribs["kilogram"][0]>0:
                    factor*=kilo
                    factorf=float(factor)
                    prefix,prefix_factor=_choose_prefix(abs(factorf),prefix_exponent)
                    numer_mass="g"


    def contrib_part(sign:int) -> "list[str]":
        """The symbols on one side of the fraction, e.g. ``["kg", "m^2"]``, in a fixed order."""
        parts:list[str]=[]
        sort=__sort_numer if sign==1 else __sort_denom
        for un in sort:
            if not (un in contribs.keys()):
                continue

            c=contribs[un]

            if c[0]*sign > 0:
                ustr=__shorts[un]
                if sign>0 and un=="kilogram":
                    ustr=numer_mass
                if c[0]*sign != 1 or c[1] != 1:
                    ustr += "^"
                    if c[1] != 1:
                        ustr += "(" + str(c[0]*sign) + "/" + str(c[1]) + ")"
                    else:
                        ustr += str(c[0]*sign)
                parts.append(ustr)
        return parts

    numer_parts=contrib_part(1)
    denom_parts=contrib_part(-1)
    numer=UNIT_SEPARATOR.join(numer_parts)
    denom=UNIT_SEPARATOR.join(denom_parts)

    def with_denominator(head:str,den:str)->str:
        """``head/den``, bracketing a denominator of several factors.

        Without the brackets "kg/m s^2" reads as kg*s^2/m, i.e. the opposite of what it means - and run
        together as it used to be, "kg/ms^2" looks like kg per millisecond squared.
        """
        return head+"/"+(("("+den+")") if len(denom_parts)>1 else den)

    if denom!="":
        if numer=="":
            numer="1"
            # The prefix moves into the denominator, so it has to be inverted: 1/(nm) is 1 Gm^-1.
            # Every entry of __prefixes needs one here, or a value that picks an unlisted prefix
            # raises a KeyError instead of being printed.
            __invprefix:dict[str,str]={"":"", "a":"E","f":"P","p":"T","n":"G","u":"M","m":"k",
                                       "k":"m","M":"u","G":"n","T":"p","P":"f","E":"a"}
            resstr=with_denominator(numer,__invprefix[prefix]+denom)
        else:
            resstr=with_denominator(prefix+numer,denom)
    else:
        if numer=="":
            resstr=""
            prefix_factor=1
        else:
            resstr=prefix+numer

    if estimate_prefix:
        return resstr,factorf,prefix_factor**(-prefix_exponent)
    else:
        return resstr


__simplified_units["Pa"] = _pyoomph.GiNaC_sep_base_units(pascal)
__simplified_units["Pas"] = _pyoomph.GiNaC_sep_base_units(pascal*second)
__simplified_units["N"] = _pyoomph.GiNaC_sep_base_units(newton)
__simplified_units["N/m"] = _pyoomph.GiNaC_sep_base_units(newton/meter)
__simplified_units["Nm"] = _pyoomph.GiNaC_sep_base_units(newton*meter)
# W, V and F share no base-unit signature with anything above, so the order they are added in does not
# matter. Two that are NOT here on purpose: "Hz" has the same base units as any rate, so it would
# relabel every growth rate and eigenvalue as a frequency; and "J" is dimensionally identical to "Nm",
# which is already registered, so energy and torque cannot be told apart by units at all.
__simplified_units["W"] = _pyoomph.GiNaC_sep_base_units(watt)
__simplified_units["V"] = _pyoomph.GiNaC_sep_base_units(volt)
__simplified_units["F"] = _pyoomph.GiNaC_sep_base_units(farad)
# The electric units below. unit_to_string matches on the exact base-unit signature and takes the
# first hit in insertion order, so a new entry can only ever relabel a quantity whose signature is
# identical to it and to no earlier entry. These seven are pairwise distinct and distinct from
# everything above, and every one of them contains ampere, which occurs in no non-electric entry --
# so nothing mechanical or thermal can be caught by them. Appended last, which also preserves the
# prefix estimation described further up.
__simplified_units["C"] = _pyoomph.GiNaC_sep_base_units(coulomb)
__simplified_units["S"] = _pyoomph.GiNaC_sep_base_units(siemens)
__simplified_units["S/m"] = _pyoomph.GiNaC_sep_base_units(siemens/meter)
__simplified_units["F/m"] = _pyoomph.GiNaC_sep_base_units(farad/meter)
__simplified_units["V/m"] = _pyoomph.GiNaC_sep_base_units(volt/meter)
__simplified_units["C/m^2"] = _pyoomph.GiNaC_sep_base_units(coulomb/meter**2)
__simplified_units["C/m^3"] = _pyoomph.GiNaC_sep_base_units(coulomb/meter**3)
__simplified_units["Ohm"] = _pyoomph.GiNaC_sep_base_units(ohm)
__simplified_units["Ohm m"] = _pyoomph.GiNaC_sep_base_units(ohm*meter)
__simplified_units["F/m^2"] = _pyoomph.GiNaC_sep_base_units(farad/meter**2)
__simplified_units["C/mol"] = _pyoomph.GiNaC_sep_base_units(coulomb/mol)
__simplified_units["C m"] = _pyoomph.GiNaC_sep_base_units(coulomb*meter)
__simplified_units["S m^2/mol"] = _pyoomph.GiNaC_sep_base_units(siemens*meter**2/mol)
__simplified_units["H"] = _pyoomph.GiNaC_sep_base_units(henry)
__simplified_units["H/m"] = _pyoomph.GiNaC_sep_base_units(henry/meter)
# A name registered here MUST start with a derived symbol, i.e. one whose prefix binds with exponent
# 1 -- the lookup above short-circuits the exponent search whenever a simplified name is found, so a
# label like "m^2/(V s)" would be printed with a prefix meant for exponent 1 while reading as
# exponent 2, i.e. wrong by three orders per prefix step. That is the same trap the farad comment
# further up records, seen from the other side. Ion mobility (m^2/(V s)) is therefore deliberately
# NOT registered and keeps its base-unit spelling.




class ArrayWithUnits:
    def __init__(self,array:Sequence[ExpressionOrNum] | NPFloatArray,unit:ExpressionOrNum | None=None):
        super(ArrayWithUnits, self).__init__()
        if unit is None:
            if isinstance(array,ArrayWithUnits):
                unit=array.unit
                array=array.values
            elif isinstance(array,(numpy.ndarray,list,tuple)):
                for k in array:
                    v,u=assert_dimensional_value(k)
                    if v!=0:
                        unit=u
                        break
                else:
                    unit=1
                ndarr:list[float]=[]
                for k in array:
                    try:
                        ndarr.append(float(k/unit))
                    except:
                        raise RuntimeError("Cannot cast all values to the common unit of "+str(unit))
                array=numpy.array(ndarr) #type:ignore
            else:
                raise ValueError("Cannot cast this to an ArrayWithUnits")
        self.values:Sequence[ExpressionOrNum] | NPFloatArray=array
        self.unit:ExpressionOrNum=unit

    def __getitem__(self, item:int)->ExpressionOrNum:
        return self.values[item]*self.unit

    def __setitem__(self, item:int,value:ExpressionOrNum):
        try:
            float(value/self.unit)
        except:
            raise ValueError("Cannot set the value "+str(value)+" to a ArrayWithUnits with unit "+str(self.unit))

        return self.values[item]*self.unit

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return "<ArrayWithUnits, unit: "+str(self.unit)+", values="+repr(self.values)+">"

    #def __


# Will check for a value like 1.44*meter/second, but not anything like 400*x*y*meter
def assert_dimensional_value(dim_val:ExpressionOrNum,required_unit:ExpressionOrNum | None=None):
    if isinstance(dim_val,(float,int)):
        return dim_val,1
    if isinstance(dim_val,_pyoomph.GiNaC_GlobalParam):
        # Not accepted by collect_units itself; wrapped it splits into its current value and unit 1,
        # which is the same freeze-the-number-now that the plain float branch above does.
        dim_val=0+dim_val
    factor, unit, rest, success = _pyoomph.GiNaC_collect_units(dim_val)
    if not success:
        raise ValueError(str(dim_val)+" is not a simple dimensional value, i.e. a product of a numerical value and a unit")
    try:
        factor*=float(rest)
    except:
        raise ValueError(str(dim_val) + " is not a simple dimensional value, i.e. a product of a numerical value and a unit")
    if required_unit is not None:
        try:
            float(unit/required_unit)
        except:
            raise ValueError("Expected a dimensional quantity with unit "+str(required_unit)+", but got "+str(dim_val)+ " instead")
    return float(factor),unit


def _dimensional_numpy_space(start:ExpressionOrNum,stop:ExpressionOrNum,npfunc:Any,**npkwargs:Any):
    start_wo, start_unit = assert_dimensional_value(start)
    stop_wo, stop_unit = assert_dimensional_value(stop)
    if start_wo == 0:
        unit = stop_unit
    elif stop_wo == 0:
        unit = stop_unit
    else:
        try:
            t=float(start_unit / stop_unit)
            stop_wo*=t
            unit=start_unit
        except:
            raise RuntimeError(
                "start and stop do not have the same physical unit: " + str(start_unit) + " vs " + str(stop_unit))
    vals=npfunc(start_wo,stop_wo,**npkwargs)
    return ArrayWithUnits(vals, unit)

def dimensional_linspace(start:ExpressionOrNum,stop:ExpressionOrNum,num:int=50,endpoint:bool=True):
    return _dimensional_numpy_space(start,stop,numpy.linspace,num=num,endpoint=endpoint) #type:ignore

def dimensional_geomspace(start:ExpressionOrNum,stop:ExpressionOrNum,num:int=50,endpoint:bool=True):
    return _dimensional_numpy_space(start, stop, numpy.geomspace, num=num, endpoint=endpoint)#type:ignore


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
