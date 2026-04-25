#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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
 
 
"""
Module containing physical units (meter, second, etc) and some constants for use in expressions.
"""
 
import _pyoomph

import numpy 

from ..typings import *

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
coloumb = ampere * second
farad= coloumb / volt
henry = volt * second / ampere
siemens = ampere / volt

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

__simplified_units:Dict[str,Dict[str,Tuple[int,int]]] = {}

@overload
def unit_to_string(inp:ExpressionOrNum,estimate_prefix:Literal[True]=...)->Tuple[str,float,float]: ...

@overload
def unit_to_string(inp:ExpressionOrNum,estimate_prefix:Literal[False])->str: ...

def unit_to_string(inp:ExpressionOrNum,estimate_prefix:bool=True) -> Union[str, Tuple[str, float, float]]:
    __prefixes:Dict[float,str]={1e-9:"n",1e-6:"u",1e-3:"m",1:"",1e3:"k",1e6:"M",1e9:"G"}
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

    prefix_factor=1
    factor_bound_factor=10
    if estimate_prefix:
        for k in sorted(__prefixes.keys()):
            #print("IN",inp,estimate_prefix, k,__prefixes[k],factorf,factor_bound_factor,prefix_factor)
            if factorf <= factor_bound_factor * k:
                prefix = __prefixes[k]
                prefix_factor=k
                break

    for rep,s in __simplified_units.items():
        if s==contribs:
            if estimate_prefix:
                return prefix+rep,factorf,1/prefix_factor
            else:
                return prefix+rep

    if estimate_prefix:
        if prefix!="":
            if "kilogram" in contribs:
                if contribs["kilogram"][0]>0:
                    factor*=kilo
                    factorf=float(factor)
                    for k in sorted(__prefixes.keys()):
                        if factorf <= factor_bound_factor * k:
                            prefix = __prefixes[k]
                            prefix_factor = k
                            break
                    numer_mass="g"


    def contrib_part(sign:int) -> str:
        resstr = ""
        sort=__sort_numer if sign==1 else __sort_denom
        for un in sort:
            if not (un in contribs.keys()):
                continue

            c=contribs[un]

            if c[0]*sign > 0:
                ustr=__shorts[un]
                if sign>0 and un=="kilogram":
                    ustr=numer_mass
                resstr += ustr
                if c[0]*sign != 1 or c[1] != 1:
                    resstr += "^"
                    if c[1] != 1:
                        resstr += "(" + str(c[0]*sign) + "/" + str(c[1]) + ")"
                    else:
                        resstr += str(c[0]*sign)
        return resstr

    numer=contrib_part(1)
    denom = contrib_part(-1)

    if denom!="":
        if numer=="":
            numer="1"
            __invprefix:Dict[str,str]={"":"", "n":"G","u":"M","m":"k","k":"m","M":"u","G":"p"}
            resstr=numer+"/"+__invprefix[prefix]+denom
        else:
            resstr=prefix+numer+"/"+denom
    else:
        if numer=="":
            resstr=""
            prefix_factor=1
        else:
            resstr=prefix+numer

    if estimate_prefix:
        return resstr,factorf,1/prefix_factor
    else:
        return resstr


__simplified_units["Pa"] = _pyoomph.GiNaC_sep_base_units(pascal)
__simplified_units["Pas"] = _pyoomph.GiNaC_sep_base_units(pascal*second)
__simplified_units["N"] = _pyoomph.GiNaC_sep_base_units(newton)
__simplified_units["N/m"] = _pyoomph.GiNaC_sep_base_units(newton/meter)
__simplified_units["Nm"] = _pyoomph.GiNaC_sep_base_units(newton*meter)




class ArrayWithUnits:
    def __init__(self,array:Union[Sequence[ExpressionOrNum],NPFloatArray],unit:Optional[ExpressionOrNum]=None):
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
                ndarr:List[float]=[]
                for k in array:
                    try:
                        ndarr.append(float(k/unit))
                    except:
                        raise RuntimeError("Cannot cast all values to the common unit of "+str(unit))
                array=numpy.array(ndarr) #type:ignore
            else:
                raise ValueError("Cannot cast this to an ArrayWithUnits")
        self.values=array
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
def assert_dimensional_value(dim_val:ExpressionOrNum,required_unit:Optional[ExpressionOrNum]=None):
    if isinstance(dim_val,(float,int)):
        return dim_val,1
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


