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
 
from .. import _pyoomph_core as _pyoomph
import os
import copy
import math
import itertools
from pathlib import Path
from collections import OrderedDict as OrderDict

from ..expressions import var,grad,subexpression,exp,log,rational_num,square_root,is_zero
from ..expressions import ExpressionOrNum,ExpressionNumOrNone,Expression
from ..expressions.units import *
from ..expressions.phys_consts import gas_constant,faraday_constant,epsilon_0,debye_length
from .activity import *


from ..typings import *

if TYPE_CHECKING:
    from ..generic.problem import Problem
    from .mass_transfer import MassTransferModelBase

import math

MixQuantityDefinition:TypeAlias=Literal["mass_fraction","wt","mole_fraction","volume_fraction","relative_humidity","RH"]
AnyMaterialProperties:TypeAlias="MaterialProperties | BaseLiquidProperties | BaseGasProperties | BaseSolidProperties | PureSolidProperties | PureLiquidProperties | PureGasProperties | MixtureLiquidProperties | MixtureGasProperties"
OutputPropertiesType=dict[str,Callable[["MaterialProperties"], ExpressionNumOrNone] | None]
DefaultSurfaceTensionType=dict[Literal["gas","solid","liquid"],ExpressionNumOrNone]
PropertySampleRangeType:TypeAlias=ExpressionOrNum|ArrayWithUnits|list[ExpressionOrNum]

AnyFluidProperties:TypeAlias="PureLiquidProperties | PureGasProperties | MixtureLiquidProperties | MixtureGasProperties"
AnyLiquidProperties:TypeAlias="PureLiquidProperties | MixtureLiquidProperties"
AnyGasProperties:TypeAlias="PureGasProperties | MixtureGasProperties"
AnyFluidFluidInterface:TypeAlias="LiquidGasInterfaceProperties | LiquidLiquidInterfaceProperties"

_TypeMaterialProperties=TypeVar("_TypeMaterialProperties",bound="type[MaterialProperties] | type[BaseInterfaceProperties]")


def assert_liquid_properties(props:"MaterialProperties")->AnyLiquidProperties:
    if isinstance(props,(PureLiquidProperties,MixtureLiquidProperties)):
        return props
    else:
        raise RuntimeError("Expected liquid properties, but got "+str(props))

def assert_gas_properties(props:"MaterialProperties")->AnyGasProperties:
    if isinstance(props,(PureGasProperties,MixtureGasProperties)):
        return props
    else:
        raise RuntimeError("Expected gas properties, but got "+str(props))

def assert_fluid_properties(props:"MaterialProperties")->AnyFluidProperties:
    if isinstance(props,(PureGasProperties,MixtureGasProperties,PureLiquidProperties,MixtureLiquidProperties)):
        return props
    else:
        raise RuntimeError("Expected fluid properties, but got "+str(props))

def assert_liquid_gas_interface(interf:"BaseInterfaceProperties")->"LiquidGasInterfaceProperties":
    if isinstance(interf,LiquidGasInterfaceProperties):
        return interf
    else:
        raise RuntimeError("Expected fluid properties, but got "+str(interf))

class BaseInterfaceProperties:
    """
    Base class for interface properties. 
    """
    def _sort_phases(self,sideA:AnyMaterialProperties,sideB:AnyMaterialProperties)->tuple[AnyMaterialProperties,AnyMaterialProperties]:
        return sideA,sideB
    def __init__(self,sideA:"AnyMaterialProperties | MixtureDefinitionComponents",sideB:"AnyMaterialProperties | MixtureDefinitionComponents"):
        if isinstance(sideA,MixtureDefinitionComponents):
            sideA=Mixture(sideA)
        if isinstance(sideB,MixtureDefinitionComponents):
            sideB=Mixture(sideB)
        self._phaseA,self._phaseB=self._sort_phases(sideA,sideB)
        #: The surface tension of this interface
        self.surface_tension:ExpressionOrNum
        #: The mass transfer model to use for this interface
        self._mass_transfer_model:MassTransferModelBase | None=None
        self._surfactant_table:dict[str,Any]={}
        self._latent_heats:dict[str,ExpressionOrNum]={}
        #: Free surface charge density sitting on this interface.
        self.surface_charge_density:ExpressionOrNum=0
        #: Zeta potential of the diffuse layer, for thin-double-layer models such as
        #: :py:class:`~pyoomph.equations.electrohydrodynamics.ElectroosmoticSlip`. ``None`` means
        #: this interface is not described that way.
        self.zeta_potential:ExpressionNumOrNone=None
        #: Areal capacitance of the compact (Stern) layer, see
        #: :py:class:`~pyoomph.equations.electrostatics.SternLayer`.
        self.stern_layer_capacitance:ExpressionNumOrNone=None
        #: Areal capacitance of the diffuse layer, used by the Lippmann relation.
        self.double_layer_capacitance:ExpressionNumOrNone=None
        #: Excess surface conductance, i.e. the numerator of the Dukhin number.
        self.surface_conductance:ExpressionNumOrNone=None

    def set_latent_heat_of(self,name:str,lat_heat:ExpressionOrNum):
        self._latent_heats[name]=lat_heat

    def get_latent_heat_of(self,name:str)->ExpressionOrNum:
        res=self._latent_heats.get(name)
        if res is None:
            raise RuntimeError("No latent heat set for "+name)
        return res

    def set_mass_transfer_model(self,mdl:"MassTransferModelBase | None") -> "MassTransferModelBase | None":
        """
        Sets the mass transfer model.
        """
        self._mass_transfer_model=mdl
        return mdl # For convenience, return the model so that it can be used in the same line as the function call.
######

def _as_component_set(compos:"str | set[str] | frozenset[str] | None")->frozenset[str]:
    """The component names of an interface side, as the hashable set the library is keyed by.

    A single name may be given as a plain string, and an absent side (no gas components, no
    surfactants) as None, which becomes the empty set."""
    if compos is None:
        return frozenset()
    if isinstance(compos,str):
        return frozenset({compos})
    return frozenset(compos)


class MaterialProperties:
    """
    Base class for all material properties. This class should not be instantiated directly, but rather one of its subclasses should be used.
    However, this class allows to register new material properties and interfaces using the ``@MaterialProperties.register()`` decorator (see :py:meth:`register`) on the definition of the subclass.        
    """
    #: Unique name of the material. Names should be unique within the same state of matter (e.g. liquid, gas, solid), and the same material name for the same material should be used for different states of matter.
    name:str
    #: Whether the material is pure or mixed. If the material is mixed, the components of the mixture should be specified in the :py:attr:`~pyoomph.materials.generic.MaterialProperties.components` attribute. This should be treated as read-only property.
    is_pure:bool | None
    #: State of matter of the material. This should be treated as read-only property.
    state_of_matter:str | None
    #: In case of a mixture, the components of the mixture. Should be set as class-variable in the subclass.
    components:set[str]=set()
    
    library:dict[str,dict[str,Any]]={"gas":{"pure":{},"mixed":{}},"solid":{"pure":{},"mixed":{}},"liquid":{"pure":{},"mixed":{}},"interfaces":{"liquid_gas":{},"liquid_solid":{},"_defaults":{},"liquid_liquid":{}}}
    _output_properties:OutputPropertiesType={}
    
    
    @classmethod
    def register(cls, *, override:bool=False):
        """
        Decorated your material classes with this to register them in the material library. This allows to use the functions :py:func:`~pyoomph.materials.generic.get_pure_gas`, :py:func:`~pyoomph.materials.generic.get_pure_liquid`, :py:func:`~pyoomph.materials.generic.Mixture`, :py:func:`~pyoomph.materials.generic.get_interface_properties`, etc. to retrieve the properties of the materials.
        """
        def decorator(subclass:_TypeMaterialProperties)->_TypeMaterialProperties:
            if issubclass(subclass,BaseInterfaceProperties):
                if issubclass(subclass,LiquidGasInterfaceProperties):
                    table=cls.library["interfaces"]["liquid_gas"]
                    if subclass.liquid_components is None:
                        raise  RuntimeError("To register liquid-gas interfaces, you must set the information liquid_components")
                    liq_set=_as_component_set(subclass.liquid_components)
                    gas_set=_as_component_set(subclass.gas_components)
                    surf_set=_as_component_set(subclass.surfactants)
                    entry=(liq_set,gas_set,surf_set)

                    if entry in table.keys() and not override:
                        raise RuntimeError("There is already an liquid-gas interface property defined for "+str(entry)+". Please use override=True to register and override.")
                    table[entry]=subclass
                elif  issubclass(subclass,LiquidSolidInterfaceProperties):
                    table = cls.library["interfaces"]["liquid_solid"]
                    if subclass.liquid_components is None:
                        raise RuntimeError(
                            "To register liquid-solid interfaces, you must set the information liquid_components")
                    liq_set = _as_component_set(subclass.liquid_components)
                    solid_set = _as_component_set(subclass.solid_components)
                    surf_set = _as_component_set(subclass.surfactants)
                    entry = (liq_set, solid_set, surf_set)
                    if entry in table.keys() and not override:
                        raise RuntimeError("There is already an liquid-solid interface property defined for " + str(
                            entry) + ". Please use override=True to register and override.")
                    table[entry] = subclass
                elif  issubclass(subclass,LiquidLiquidInterfaceProperties):
                    table = cls.library["interfaces"]["liquid_liquid"]
                    if subclass.componentsA is None or subclass.componentsB is None:
                        raise RuntimeError(
                            "To register liquid-liquid interfaces, you must set the information componentsA and componentsB")
                    setA = _as_component_set(subclass.componentsA)
                    setB = _as_component_set(subclass.componentsB)
                    surf_set = _as_component_set(subclass.surfactants)
                    # A liquid-liquid interface is keyed by the unordered pair of its two sides,
                    # not by three named sides like the others
                    ll_entry = (frozenset({setA, setB}), surf_set)

                    if ll_entry in table.keys() and not override:
                        raise RuntimeError("There is already an liquid-liquid interface property defined for " + str(
                            ll_entry) + ". Please use override=True to register and override.")
                    table[ll_entry] = subclass
                else:
                    raise RuntimeError("TODO: Register other interfaces Interface")
                return subclass
            # From here on, subclass is guaranteed to be a MaterialProperties subclass (interfaces are handled and returned above).
            # A local, explicitly typed alias is used since pyright cannot narrow the _TypeMaterialProperties TypeVar itself here.
            mat_subclass=cast("type[MaterialProperties]",subclass)
            if not hasattr(mat_subclass, 'state_of_matter') or mat_subclass.state_of_matter is None:
                raise RuntimeError("Cannot register material '"+subclass.__name__+"', since it does not have a state_of_matter. Please define "+subclass.__name__+".state_of_matter=...")
            if not hasattr(mat_subclass,"is_pure") or mat_subclass.is_pure is None:
                raise RuntimeError("Bulk material properites must have is_pure set to True or False")
            if mat_subclass.is_pure:
                if not hasattr(mat_subclass, 'name'):
                    raise RuntimeError("Cannot register pure "+str(mat_subclass.state_of_matter)+" material '"+subclass.__name__+"', since it does not have a name. Please define "+subclass.__name__+".name=\"...\"")

                if mat_subclass.name in cls.library[mat_subclass.state_of_matter]["pure"].keys():
                    if not override:
                        raise RuntimeError("You tried to register the pure "+mat_subclass.state_of_matter+" material named '"+mat_subclass.name+"', but there is already one defined. Please either use another name or add override=True to the arguments of @MaterialProperties.register(override=True)")
                cls.library[mat_subclass.state_of_matter]["pure"][mat_subclass.name]=subclass
            else:
                if not hasattr(mat_subclass, 'components'):
                    raise RuntimeError("Cannot register mixed "+str(mat_subclass.state_of_matter)+" material '"+subclass.__name__+"', since it does not have a list of pure components. Please define "+subclass.__name__+".components={...}")
                if type(mat_subclass.components)!=set:
                    raise RuntimeError("Cannot register mixed "+str(mat_subclass.state_of_matter)+" material '"+subclass.__name__+"', since the list of pure components needs to be a set. Please define "+subclass.__name__+".components={...}")

                frz=frozenset(mat_subclass.components)
                if frz in cls.library[mat_subclass.state_of_matter]["mixed"].keys():
                    if not override:
                        raise RuntimeError("You tried to register the mixed "+str(mat_subclass.state_of_matter)+" material with components '"+str(sorted(mat_subclass.components))+"', but there is already one defined. Please add override=True to the arguments of @MaterialProperties.register(override=True)")
                cls.library[mat_subclass.state_of_matter]["mixed"][frz]=subclass

    #      cls.subclasses[message_type] = subclass
            # cast: issubclass() above narrowed the TypeVar away, although what is returned is the
            # very class that was passed in
            return cast(_TypeMaterialProperties,subclass)
        return decorator


    def generate_field_substs(self,cond:dict[str,ExpressionOrNum])->tuple[dict[str,Expression],dict[str,Expression]]:
        fields:dict[str,Expression]={}
        defined_massfracs:dict[str,ExpressionOrNum]={}
        for lhs,rhs in cond.items():
            if rhs is None:
                continue
            if lhs.startswith("massfrac_"):
                fields[lhs]=_pyoomph.Expression(rhs)
                defined_massfracs[lhs[9:]]=fields[lhs]
            elif lhs.startswith("molefrac_"):
                raise RuntimeError("Please specify via massfrac_<componame>, not molefrac_<componame> ... ")
            else:
                fields[lhs]=_pyoomph.Expression(rhs)
        missing_massfracs=self.components.difference(defined_massfracs.keys())
        if len(missing_massfracs)==1:
            remsum=1-sum(v for v in defined_massfracs.values())
            missname=(list(missing_massfracs))[0]
            defined_massfracs[missname]=remsum
            missing_massfracs.remove(missname)
            if ("massfrac_"+missname) not in fields.keys():
                if isinstance(remsum,_pyoomph.Expression):
                    fields["massfrac_"+missname]=0+remsum
                else:
                    fields["massfrac_" + missname] = _pyoomph.Expression(remsum)

        if len(missing_massfracs)==0:
            #Calc the mole fracs from the mass fracs
            if isinstance(self,BaseMixedProperties):
                denomsum:ExpressionOrNum=sum([defined_massfracs[k]/self.pure_properties[k].molar_mass for k in defined_massfracs.keys()] )
                for k in defined_massfracs.keys():
                    e=(defined_massfracs[k]/self.pure_properties[k].molar_mass)/denomsum
                    if not isinstance(e,Expression):
                        e=Expression(e)
                    fields["molefrac_"+k]=e

        #print(fields)
        return fields,{}
    
    def evaluate_at_condition(self,expr:ExpressionOrNum | str,cond:dict[str, ExpressionOrNum] | Literal["initial", "IC", "initial_condition"]={},*,temperature:ExpressionNumOrNone=None,**kwargs:ExpressionNumOrNone) -> Expression:
        """
        Evaluates a property at the given condition (temperature, mass fractions, etc.). The mass fractions should be given as ``massfrac_<component_name>``, where ``<component_name>`` is the name of the component. The mole fractions should be given as ``molefrac_<component_name>``. Other typical conditions are ``temperature`` and ``absolute_pressure``.

        Args:
            expr: Either a property name like ``"mass_density"`` or an expression to evaluate.
            cond: Condition to evaluate. Can be e.g. ``{"massfrac_water":0.5,"temperature":300*kelvin}`` or use the :py:attr:`initial_condition` of the material properties (shortcuts ``"initial"``, ``"IC"`` or ``"initial_condition"``).
            temperature: Temperature to evaluate the property at. If not given, the temperature from the condition will be used.

        Returns:
            The property evaluated at the given condition.
        """
        if isinstance(expr,str):
            if hasattr(self,expr):
                expr=getattr(self,expr)
            else:
                raise ValueError("No property "+expr+" defined")
        if not isinstance(expr,_pyoomph.Expression):
            expr=_pyoomph.Expression(expr)
        if isinstance(cond,str):
            actual_cond=self.initial_condition
        else:
            actual_cond=cond
        mycond=actual_cond.copy()
        for i,j in kwargs.items():
            if j is not None:
                mycond[i]=j
        if temperature is not None:
            mycond["temperature"]=temperature
        fields,nondims=self.generate_field_substs(mycond)
        remkeys:set[str]=set()
        for n,f in fields.items():
            if f is None:
                remkeys.add(n)
                continue
            if not isinstance(f,_pyoomph.Expression): #type:ignore
                fields[n]=_pyoomph.Expression(f)
        for k in remkeys:
            fields.pop(k)

        remkeys = set()
        for n,f in nondims.items():
            if f is None:
                remkeys.add(n)
                continue
            if not isinstance(f,_pyoomph.Expression): #type:ignore
                nondims[n]=_pyoomph.Expression(f)
        for k in remkeys:
            nondims.pop(k)
        #print("SUBS FIELDS", expr, "FIELDS", fields, "NONDIM", nondims, "COND", cond)
        #print("RET ",_pyoomph.GiNaC_subsfields(expr,fields,nondims,{})) #TODO Global params
#		ext()
        return _pyoomph.GiNaC_subsfields(expr,cast(dict[str,_pyoomph.Expression],fields),cast(dict[str,_pyoomph.Expression],nondims),{}) #type:ignore #TODO Global params

    def simplify_property_expressions(self,*property_names:str,**variables:ExpressionOrNum):
        for name in property_names:
            if hasattr(self,name):
                setattr(self,name,self.evaluate_at_condition(getattr(self,name),variables))
            else:
                raise RuntimeError(str(self)+" has no property "+str(name))

    def __init__(self):
        #: The initial condition of the material. Will be set automatically when using e.g. the :py:func:`Mixture` function to assemble a mixture of pure components.
        self.initial_condition:dict[str,ExpressionOrNum]={}
        #: The mass density of the material. 
        self.mass_density:ExpressionOrNum#=None
        #: The specific heat capacity of the material.
        self.specific_heat_capacity:ExpressionOrNum# = None
        #: The thermal conductivity of the material.
        self.thermal_conductivity:ExpressionOrNum# = None
        #: The molecular weight of the material, used to calculate the mole fractions from the mass fractions.
        self.molar_mass:ExpressionOrNum#=None


    def describe(self)->str:
        """A name for error messages. A mixture has no ``name``, only ``components``, and reaching
        for the missing attribute inside an error path turns a clear message into an
        AttributeError."""
        name=getattr(self,"name",None)
        if name is not None:
            return str(name)
        comps=getattr(self,"components",None)
        return "mixture of "+", ".join(sorted(comps)) if comps else "<unnamed material>"

    def __mul__(self,other:float | int | Expression)->"MixtureDefinitionComponent":
        return MixtureDefinitionComponent(self,other)

    def __rmul__(self,other:float | int | Expression)->"MixtureDefinitionComponent":
        return MixtureDefinitionComponent(self,other)

    def __or__(self,other:AnyMaterialProperties)->"BaseInterfaceProperties | LiquidLiquidInterfaceProperties | LiquidGasInterfaceProperties | LiquidSolidInterfaceProperties":
        if isinstance(other,MaterialProperties): #type:ignore
            return get_interface_properties(self,other)
        elif isinstance(other,MixtureDefinitionComponents) or isinstance(other,MixtureDefinitionComponent):
            raise RuntimeError("Please finalize a mixture of pure components with a Mixture(...) call: "+str(other))

    def evaluate_at_multiple_params(self,expr:ExpressionOrNum | str,_sort:str="len",**kwargs:PropertySampleRangeType)->tuple[list[ExpressionOrNum],ExpressionOrNum,OrderedDict[str,ArrayWithUnits],dict[str,ExpressionOrNum]]:
        if isinstance(expr,str):
            if hasattr(self,expr):
                expr=getattr(self,expr)
            else:
                raise RuntimeError("Cannot find the property "+str(expr)+" in "+str(self))
        expr=cast(ExpressionOrNum,expr)
        vari_ranges:list[ArrayWithUnits]=[]
        vari_names:list[str]=[]
        first_cond:dict[str,ExpressionOrNum]={}
        second_cond:dict[str,ExpressionOrNum]={}
        consts:dict[str,ExpressionOrNum]={}
        for k,v in kwargs.items():
            if isinstance(v,ArrayWithUnits):
                if len(v)==0:
                    continue
                vari_ranges.append(v)
                first_cond[k] = v[0]
                if len(v)>1:
                    second_cond[k]=v[1]
                vari_names.append(k)
            elif isinstance(v,(list,tuple,numpy.ndarray)):
                if len(v)==0:
                    continue
                vari_ranges.append(ArrayWithUnits(v))
                first_cond[k]=cast(ExpressionOrNum,v[0])
                if len(v)>1:
                    second_cond[k] = cast(ExpressionOrNum,v[1])
                vari_names.append(k)
            else:
                #vari_ranges.append([v])
                consts[k]=v
                first_cond[k] = v

        sorti:list[int] | list[str]
        if _sort=="len" or _sort=="len_rev":
            sorti=[len(l) for l in vari_ranges]
        elif _sort=="name" or _sort=="name_rev":
            sorti=[n for n in vari_names]
        else:
            raise ValueError("_sort may only have the values len, len_rev, name, name_rev")
        inds:list[int]=[i for i,_ in sorted(enumerate(sorti), key = lambda x: x[1])] # type: ignore[arg-type,return-value] # len or name, both orderable, but not as one type
        if _sort=="len_rev" or _sort=="name_rev":
            inds=list(reversed(inds))
        vari_names=[vari_names[i] for i in inds]
        vari_ranges=[vari_ranges[i] for i in inds]

        result:list[ExpressionOrNum]=[]
        #Simplify the condition
        first_res=self.evaluate_at_condition(expr, cond=first_cond)

        numval,unit=assert_dimensional_value(first_res)
        if is_zero(numval):
            sec_cond={k:second_cond.get(k,first_cond[k]) for k in first_cond.keys()}
            second_res = self.evaluate_at_condition(expr, cond=sec_cond)
            numval,unit=assert_dimensional_value(second_res)
            if is_zero(numval):
                raise RuntimeError("Problem: Cannot get the unit of "+str(expr))
        dimless_expr:ExpressionOrNum=expr/unit

        for vrs in itertools.product(*vari_ranges): #type:ignore
            cond:dict[str,ExpressionOrNum]={vari_names[i]:vrs[i] for i in range(len(vari_names))} #type:ignore
            cond.update(consts) 
            result.append(float(self.evaluate_at_condition(dimless_expr, cond=cond)))

        rangs:"OrderedDict[str,ArrayWithUnits]"=OrderDict()
        for n,rang in zip(vari_names,vari_ranges):
            rangs[n]=rang
        return result,unit,rangs,consts

    def sample_all_properties_to_text_files(self,dirname:str,_sort:str="len",_newlines:bool=True,**kwargs:ExpressionOrNum):
        """
        This function will sample all properties of this material to text files. You can either pass single values, e.g. ``massfrac_water=0.5,temperature=300*kelvin``, or ranges, e.g. ``massfrac_water=numpy.linspace(0,1,100),temperature=[300*kelvin,400*kelvin]``. The function will then sample all properties at these conditions and write them to text files in the given directory.

        Args:
            dirname: Directory to create the text files
            _sort: How to sort the output files. Can be ``"len"`` or ``"name"`` to sort by the length of the ranges or the names of the variables, or ``"len_rev"`` or ``"name_rev"`` to sort in reverse order.
            _newlines: Add a new line after each set of values in the text files.
        """
        if not os.path.exists(dirname):
            Path(dirname).mkdir(parents=True, exist_ok=True)
        for k,v in self._output_properties.items():
            expr:"str | ExpressionOrNum"
            if v is None:
                expr=k # the property is the field of that name
            elif callable(v):
                computed=v(self)
                if computed is None:
                    continue
                expr=computed
            else:
                expr=v
            print("Sampling property "+k+" to text file...")
            self.sample_property_to_text_file(os.path.join(dirname,k+".txt"),expr,_name=k,_sort=_sort,_newlines=_newlines,**kwargs)


    def sample_property_to_text_file(self,fname:str,expr:str | ExpressionOrNum,_name:str | None=None,_sort:str="len",_newlines:bool=True,**kwargs:PropertySampleRangeType):
        """
        This function will sample a single property of this material to a text file. You can either pass single values, e.g. ``massfrac_water=0.5,temperature=300*kelvin``, or ranges, e.g. ``massfrac_water=numpy.linspace(0,1,100),temperature=[300*kelvin,400*kelvin]``. It will sample the property at these conditions and write them to a text file.

        Args:
            fname: Text file name to write.
            expr: Property to sample. Can be a string with the name of the property or an expression.
            _sort: How to sort the output files. Can be ``"len"`` or ``"name"`` to sort by the length of the ranges or the names of the variables, or ``"len_rev"`` or ``"name_rev"`` to sort in reverse order.
            _newlines: Add a new line after each set of values in the text files.
        """
        res,unit,inds,consts=self.evaluate_at_multiple_params(expr,_sort=_sort,**kwargs)
        u2str:Callable[[ExpressionOrNum],str] =lambda u : "["+unit_to_string(u,estimate_prefix=False)+"]" if u!=1 else ""
        #if len(inds)>1:
        #    raise RuntimeError("Cannot sample a property along more than one range to file")
        with open(fname,"wt") as f:
            if _name is None:
                if isinstance(expr,str):
                    _name=expr
            f.write("# ")
            for iname,rang in inds.items():
                f.write(iname+u2str(rang.unit)+"\t")
            if _name is not None:
                f.write(_name+u2str(unit)+"\t")
            else:
                f.write("<no name set>")
            if len(consts):
                f.write(" @ "+", ".join([str(n)+"="+str(v) for n,v in consts.items()]))
            f.write("\n")
            f.flush()
            numinds=[v.values for v in inds.values()]
            nlmod=1
            totl=1
            for n in numinds:
                totl*=len(n)
            if _newlines:
                if len(numinds)<2:
                    _newlines=False
                else:
                    nlmod=len(numinds[-1])
            for i,(vrs,val) in enumerate(zip(itertools.product(*numinds),res)):
                f.write("\t".join(map(str,vrs))+"\t"+str(val)+"\n")
                if _newlines and (i+1)%nlmod==0 and i+1<totl:
                    f.write("\n")
            f.flush()


#######################


class MixtureDefinitionComponent:
    def __init__(self,compo:MaterialProperties,quant:ExpressionNumOrNone):
        self.compo=compo
        self.quant=quant

    def __mul__(self,other:float):
        if self.quant is None:
            raise RuntimeError("This should not happen")
        self.quant*=other

    def __rmul__(self,other:float):
        if self.quant is None:
            raise RuntimeError("This should not happen")
        self.quant*=other

#    def __radd__(self,other:Union["MixtureDefinitionComponent",MaterialProperties])->"MixtureDefinitionComponents":
#        if isinstance(other,MixtureDefinitionComponent):
#            return MixtureDefinitionComponents([self,other])
#        elif isinstance(other,MaterialProperties): #type:ignore
#            return self+MixtureDefinitionComponent(other,None)

 #   def __add__(self,other:Union["MixtureDefinitionComponent",MaterialProperties])->"MixtureDefinitionComponents":
 #       return self.__radd__(other)

class LiquidMixtureDefinitionComponent(MixtureDefinitionComponent):
    def __init__(self, compo: MaterialProperties, quant: ExpressionNumOrNone):
        super().__init__(compo, quant)

    def __radd__(self,other:"MixtureDefinitionComponent | MaterialProperties | DissolvedSpeciesComponent")->"LiquidMixtureDefinitionComponents | LiquidMixtureDefinitionComponent":
        if other==0:
            return self # This allows to use e.g. sum(massfrac[c]*component[c] for c in ...)
        elif isinstance(other,DissolvedSpeciesComponent):
            return LiquidMixtureDefinitionComponents([self],dissolved=[other])
        elif isinstance(other,LiquidMixtureDefinitionComponent):
            return LiquidMixtureDefinitionComponents([self,other])
        elif isinstance(other,PureLiquidProperties):
            return self+LiquidMixtureDefinitionComponent(other,None)
        else:
            raise RuntimeError("Tried to mix a liquid with something else:"+str(self)+" and "+str(other))

    def __add__(self,other:"MixtureDefinitionComponent | MaterialProperties | DissolvedSpeciesComponent")->"LiquidMixtureDefinitionComponents | LiquidMixtureDefinitionComponent":
        return self.__radd__(other)

    def get_compo(self)->"PureLiquidProperties":
        assert isinstance(self.compo,PureLiquidProperties)
        return self.compo

class GasMixtureDefinitionComponent(MixtureDefinitionComponent):
    def __init__(self, compo: MaterialProperties, quant: ExpressionNumOrNone):
        super().__init__(compo, quant)

    def __radd__(self,other:"MixtureDefinitionComponent | MaterialProperties")->"GasMixtureDefinitionComponents":
        if isinstance(other,GasMixtureDefinitionComponent):
            return GasMixtureDefinitionComponents([self,other])
        elif isinstance(other,PureGasProperties): 
            return self+GasMixtureDefinitionComponent(other,None)
        else:
            raise RuntimeError("Tried to mix a gas with something else:"+str(self)+" and "+str(other))

    def __add__(self,other:"MixtureDefinitionComponent | MaterialProperties")->"GasMixtureDefinitionComponents":
        return self.__radd__(other)

    def get_compo(self)->"PureGasProperties":
        assert isinstance(self.compo,PureGasProperties)
        return self.compo

class MixtureDefinitionComponents():
    def __init__(self,lst:list[MixtureDefinitionComponent],
                 dissolved:"list[DissolvedSpeciesComponent] | None"=None):
        self.lst=lst
        #: Salts and ions given by concentration. They ride alongside rather than inside ``lst``,
        #: because ``lst`` holds fractions that must sum to unity and a concentration is not one of
        #: those -- see :py:class:`DissolvedSpeciesComponent`.
        self.dissolved:"list[DissolvedSpeciesComponent]"=dissolved if dissolved is not None else []

    def __add__(self,other:"MixtureDefinitionComponents | MixtureDefinitionComponent | MaterialProperties | DissolvedSpeciesComponent")->"MixtureDefinitionComponents":
        if isinstance(other,DissolvedSpeciesComponent):
            return type(self)(self.lst,dissolved=self.dissolved+[other])
        elif isinstance(other,MixtureDefinitionComponents):
            return MixtureDefinitionComponents(self.lst+other.lst,dissolved=self.dissolved+other.dissolved)
        elif isinstance(other,MixtureDefinitionComponent):
            return MixtureDefinitionComponents(self.lst+[other],dissolved=self.dissolved)
        elif isinstance(other,MaterialProperties): #type:ignore
            return self+MixtureDefinitionComponent(other,None)

    def __repr__(self) -> str:
        return "%s(%r%s)" % (self.__class__, self.lst,
                             ", dissolved=%r" % self.dissolved if self.dissolved else "")

    def finalise(self,quantity:MixQuantityDefinition="mass_fraction",temperature:ExpressionNumOrNone=None,pressure:ExpressionNumOrNone=1*atm) -> tuple[set[MaterialProperties], dict[str, ExpressionOrNum]]:
        #if len(self.lst)==1:
        #    return {self.lst[0].compo},1
        if quantity=="RH":
            quantity="relative_humidity"
        elif quantity=="wt":
            quantity="mass_fraction"
        comps = set([e.compo for e in self.lst])

        if (temperature is not None) and not (isinstance(temperature,(float,int))):
            _,_=assert_dimensional_value(temperature,required_unit=kelvin)

        total:ExpressionOrNum=0
        hasNone=None
        for e in self.lst:
            if e.quant is None:
                if hasNone is not None:
                    raise ValueError("Found at least 2 contributions to the mixture which do not have a factor. You may add several <factor>*<component>, but only in one term, the factor may be omitted. This factor is then determined by 1 minus the others")
                hasNone=e
            else:
                total=total+e.quant

        if quantity=="relative_humidity":
            gasprops=get_mixture_properties(*comps)
            if gasprops.state_of_matter!="gas":
                raise RuntimeError("relative_humidity works only for gases")
            for e in self.lst:
                if e==hasNone:
                    continue
                else:
                    pure_liquid=get_pure_liquid(e.compo.name)
                    if pure_liquid.vapor_pressure is None:
                        raise RuntimeError("Relative humidity calculations requires the vapor_pressure of pure liquid "+e.compo.name+" to be set")
                    if pressure is None:
                        raise RuntimeError("Must pressure=...")
                    if temperature is None:
                        raise RuntimeError("Must temperature=...")
                    cnds={"temperature":temperature,"absolute_pressure":pressure}                    
                    Pvap_rel_expr=(pure_liquid.evaluate_at_condition(pure_liquid.vapor_pressure,cnds))/pressure
                    try:
                        Pvap_rel=float(Pvap_rel_expr)
                    except:
                        raise RuntimeError("Cannot case the relative vapor pressure to a float, most likely since you have not set any temperature when specifying the Mixture(...,temperature=...):\n"+str(Pvap_rel_expr))
                    assert e.quant is not None
                    e.quant*=Pvap_rel
            quantity="mole_fraction"
            total = 0
            for e in self.lst:
                if e==hasNone:
                    continue
                else:
                    assert e.quant is not None
                    total = total + e.quant


        eps=1e-6
        must_sum_to_unity=(quantity=="mass_fraction" or quantity=="mole_fraction" or quantity=="volume_fraction")


        totalf=float(total)
        if must_sum_to_unity and totalf>1+eps:
            raise ValueError("The total fractions of the mixture exceed unity: "+quantity+"  "+str(self.lst))
        if hasNone is not None:
            if must_sum_to_unity:
                hasNone.quant=1-totalf
        elif must_sum_to_unity and totalf<1-eps:
            raise ValueError("The total fractions of the mixture are less than unity")

        init:dict[str,ExpressionOrNum]
        if quantity=="mass_fraction":
            init = {c.name: 0.0 for c in comps}
            for e in self.lst:
                assert e.quant is not None
                init[e.compo.name] += e.quant
        elif quantity=="mole_fraction":
            init = {c.name: 0.0 for c in comps}
            for e in self.lst:
                assert e.quant is not None
                init[e.compo.name] += e.quant
            props = get_mixture_properties(*comps)
            assert isinstance(props,(MixtureGasProperties,MixtureLiquidProperties))
            molar_denom=sum([props.pure_properties[c].molar_mass*init[c] for c in sorted(props.components)])
            for c in sorted(props.components):
                init[c]*=props.pure_properties[c].molar_mass/molar_denom
                init[c]=float(init[c])
        elif quantity=="volume_fraction":
            init = {c.name: 0.0 for c in comps}
            for e in self.lst:
                assert e.quant is not None
                init[e.compo.name] += e.quant
            props = get_mixture_properties(*comps)
            assert isinstance(props,(MixtureGasProperties,MixtureLiquidProperties))
            rhos = {c: props.pure_properties[c].evaluate_at_condition(props.pure_properties[c].mass_density,temperature=temperature) for c in sorted(props.components)}
            for _, rho in rhos.items():
                assert_dimensional_value(rho)
            denom = sum([rhos[c] * init[c] for c in sorted(props.components)])
            for c in sorted(props.components):
                init[c]*=rhos[c]/denom
                init[c]=float(init[c])
        else:
            raise ValueError("quantity=... may only take 'mass_fraction'/'wt', 'mole_fraction', 'volume_fraction' and 'relative_humidity'/'RH'.")

        return comps,init


class LiquidMixtureDefinitionComponents(MixtureDefinitionComponents):
    def __init__(self, lst: list[MixtureDefinitionComponent],
                 dissolved:"list[DissolvedSpeciesComponent] | None"=None):
        super().__init__(lst,dissolved=dissolved)
        for a in lst:
            if not isinstance(a,LiquidMixtureDefinitionComponent):
                RuntimeError("You tried to mix a gas with something else: "+str(self)+" contains "+str(a))

    def __add__(self,other:"MixtureDefinitionComponents | MixtureDefinitionComponent | MaterialProperties | DissolvedSpeciesComponent")->"LiquidMixtureDefinitionComponents":
        if isinstance(other,DissolvedSpeciesComponent):
            return LiquidMixtureDefinitionComponents(self.lst,dissolved=self.dissolved+[other])
        elif isinstance(other,LiquidMixtureDefinitionComponents):
            return LiquidMixtureDefinitionComponents(self.lst+other.lst,dissolved=self.dissolved+other.dissolved)
        elif isinstance(other,LiquidMixtureDefinitionComponent):
            return LiquidMixtureDefinitionComponents(self.lst+[other],dissolved=self.dissolved)
        elif isinstance(other,PureLiquidProperties):
            return self+LiquidMixtureDefinitionComponent(other,None)
        else:
            raise RuntimeError("You tried to mix a liquid with something else: "+str(self)+" and "+str(other))

class GasMixtureDefinitionComponents(MixtureDefinitionComponents):
    def __init__(self, lst: list[MixtureDefinitionComponent],
                 dissolved:"list[DissolvedSpeciesComponent] | None"=None):
        super().__init__(lst)
        if dissolved:
            raise RuntimeError("A gas cannot carry dissolved ions: "+str(dissolved))
        for a in lst:
            if not isinstance(a,GasMixtureDefinitionComponent):
                RuntimeError("You tried to mix a gas with something else: "+str(self)+" contains "+str(a))

    def __add__(self,other:"MixtureDefinitionComponents | MixtureDefinitionComponent | MaterialProperties | DissolvedSpeciesComponent")->"GasMixtureDefinitionComponents":
        if isinstance(other,GasMixtureDefinitionComponents):
            return GasMixtureDefinitionComponents(self.lst+other.lst)
        elif isinstance(other,GasMixtureDefinitionComponent):
            return GasMixtureDefinitionComponents(self.lst+[other])
        elif isinstance(other,PureGasProperties):
            return self+GasMixtureDefinitionComponent(other,None)
        else:
            raise RuntimeError("You tried to mix a gas with something else: "+str(self)+" and "+str(other))

#######################
class BaseLiquidProperties(MaterialProperties):
    """
    A base class for defining liquid materials.
    """
    state_of_matter="liquid"
    passive_field:str | None=None
    required_adv_diff_fields:set[str]=set()
    possible_properties:set[str]={"mass_density","dynamic_viscosity","default_surface_tension","relative_permittivity","electric_conductivity"}
    _output_properties:OutputPropertiesType={"mass_density":None,"dynamic_viscosity":None,"relative_permittivity":None,"electric_conductivity":None,"default_surface_tension_gas":lambda self : cast(DefaultSurfaceTensionType,self.default_surface_tension).get("gas")} #type:ignore
    def __init__(self):
        super(BaseLiquidProperties, self).__init__()
        #: Default surface tension of the liquid. This is a dictionary with the keys ``"gas"``, ``"solid"``, and ``"liquid"``. The value for each key is the surface tension of the liquid with the respective other phase. 
        self.default_surface_tension:DefaultSurfaceTensionType={"gas":None}
        #: The dynamic viscosity of the liquid.
        self.dynamic_viscosity:ExpressionOrNum#=None
        # Annotated, not assigned: make_static and set_by_weighted_average both iterate
        # possible_properties and test hasattr, so a material that never sets these must not appear
        # to have them. Assigning None here would make hasattr true and break both.
        #: Relative permittivity (dielectric constant), dimensionless.
        self.relative_permittivity:ExpressionOrNum
        #: Ohmic conductivity of the liquid.
        self.electric_conductivity:ExpressionOrNum
        #: Activity coefficients per component. Filled by set_activity_coefficients_by_unifac; a
        #: pure liquid only has non-unity ones when something is dissolved in it.
        self.activity_coefficients:dict[str,ExpressionOrNum]={}
        #: The same for the dissolved ions, molality-based and referenced to infinite dilution in
        #: pure water. Only AIOMFAC fills this.
        self.ion_activity_coefficients:dict[str,ExpressionOrNum]={}
        self._ion_table:dict[str,"IonProperties"]={}
        self._bulk_concentrations:dict[str,ExpressionOrNum]={}
        # Which *salts* were dissolved, as opposed to which ions ended up in the table. The
        # electroneutral transport model solves one field per salt, so it needs the pairing back,
        # and the ion table cannot give it: NaCl + Na2SO4 leaves three ions whose split into salts
        # is not recoverable from the concentrations alone.
        self._salt_table:dict[str,"DissolvedSalt"]={}

    # A method, not a property: make_static does setattr over possible_properties, and a read-only
    # property listed there would raise on the way through.
    def get_absolute_permittivity(self,temperature:ExpressionNumOrNone=None)->ExpressionOrNum:
        r""":math:`\varepsilon=\varepsilon_\mathrm{r}\varepsilon_0`, i.e. what the electrostatic
        equations actually want.

        With a ``temperature`` the expression is evaluated at it, as
        :py:meth:`get_reference_dynamic_viscosity` does -- the permittivity of water varies by a
        third over the liquid range, so a Debye length asked for at a definite temperature has to
        use the permittivity at that temperature and not a symbolic field.
        """
        if not hasattr(self,"relative_permittivity"):
            extra=""
            if getattr(self,"components",None):
                # Deliberately not averaged automatically: linear mixing is a poor rule for the
                # permittivity (Bruggeman and Looyenga exist for a reason), so a mixture gets one
                # only when someone asks for it, or sets its own correlation.
                extra=("\nFor a mixture, either set it, or ask for the mass-weighted average "+
                       "explicitly with set_by_weighted_average('relative_permittivity') if every "+
                       "component defines one.")
            raise RuntimeError("The material '"+self.describe()+"' does not define a "+
                               "relative_permittivity"+extra)
        eps=self.relative_permittivity*epsilon_0
        if temperature is None:
            return eps
        ics=self.initial_condition.copy()
        ics["temperature"]=temperature
        return self.evaluate_at_condition(eps,ics)

    def add_ion(self,ion:"IonProperties | str",concentration:ExpressionOrNum,*,
                charge_number:int | None=None,diffusivity:ExpressionNumOrNone=None,
                walden_exponent:"ExpressionNumOrNone | Literal[False]"=False)->"IonProperties":
        r"""
        Dissolve an ionic species in this liquid at a given bulk concentration.

        The concentration is the reservoir value :math:`c_i^\infty`, i.e. what
        :py:class:`~pyoomph.equations.electrostatics.PoissonBoltzmannEquations` screens with and
        what :py:class:`~pyoomph.equations.electrostatics.NernstPlanckEquations` uses as the initial
        condition. Nothing here checks electroneutrality of the set -- see
        :py:meth:`get_net_charge_number`, which is what :py:meth:`add_salt` guarantees.

        Args:
            ion: An :py:class:`IonProperties`, or the name of a registered one, which is looked up
                with :py:func:`get_ion` -- import :py:mod:`pyoomph.materials.ions` for the standard
                library. A name that is not registered is created on the spot, which then needs
                ``charge_number`` and ``diffusivity``.
            concentration: The bulk concentration.
            charge_number: Overrides the ion's own charge number.
            diffusivity: Overrides the ion's own diffusivity, which is then the value in *this*
                solvent at 25 degC rather than the tabulated aqueous one, so pass
                ``walden_exponent=None`` with it if you do not want it carried over by the Walden
                rule as well.
            walden_exponent: Overrides :py:attr:`IonProperties.walden_exponent`; ``None`` switches
                the solvent/temperature correction off.
        """
        if isinstance(ion,str):
            found=_lookup_registered_ion(ion)
            if found is not None:
                ion=found
            else:
                if charge_number is None or diffusivity is None:
                    raise ValueError("'"+ion+"' is not a registered ion, so add_ion needs both a "+
                                     "charge_number and a diffusivity to create it. The standard "+
                                     "ions are registered by importing pyoomph.materials.ions.")
                ion=new_ion(ion,charge_number=charge_number,diffusivity=diffusivity)
        if charge_number is not None:
            ion.charge_number=charge_number
        if diffusivity is not None:
            ion.diffusivity=diffusivity
        if walden_exponent is not False:   # None is a meaningful value here, so False means "unset"
            ion.walden_exponent=walden_exponent
        if ion.charge_number==0:
            raise ValueError("Ion '"+ion.name+"' has charge_number 0, which makes it a neutral solute")
        self._ion_table[ion.name]=ion
        self._bulk_concentrations[ion.name]=concentration
        return ion

    @overload
    def add_salt(self,salt:"SaltProperties | str",concentration:ExpressionOrNum,/)->None: ...

    @overload
    def add_salt(self,cation:"IonProperties | str",anion:"IonProperties | str",
                 concentration:ExpressionOrNum,**kwargs:Any)->None: ...

    def add_salt(self,*args:Any,**kwargs:Any)->None:
        r"""
        Dissolve a salt, i.e. the pair of ions that "1 mM KCl" means, in the stoichiometry that
        makes the solution electroneutral: :math:`|z_-|` cations to :math:`z_+` anions.

        Two ways to say it, told apart by how many arguments are given::

            water.add_salt(get_salt("NaCl"), 1*milli*molar)   # or just "NaCl"
            water.add_salt("Na+", "Cl-", 1*milli*molar)

        ``concentration`` is the concentration of the *salt*, so for the 1:1 case it is also the
        concentration of each ion, while for CaCl2 it gives one Ca(2+) per two Cl(-).

        Ions and salts may be given by name, looked up with :py:func:`get_ion` and :py:func:`get_salt`;
        import :py:mod:`pyoomph.materials.ions` for the standard library. In the three-argument form,
        keyword arguments prefixed ``cation_`` and ``anion_`` are passed on to the respective
        :py:meth:`add_ion`, e.g. ``anion_diffusivity=...`` to override one of them.
        """
        if len(args)==2 and not kwargs:
            salt,concentration=args
            if isinstance(salt,str):
                if salt not in _registered_salts():
                    extra=(" '"+salt+"' is an ion, and a salt needs both of them:"+
                           " add_salt(cation, anion, concentration).") if _lookup_registered_ion(salt) is not None else ""
                    raise TypeError("add_salt(x, concentration) expects a salt, but '"+salt+
                                    "' is not a registered one."+extra)
                salt=get_salt(salt)
            if not isinstance(salt,SaltProperties):
                raise TypeError("add_salt(x, concentration) expects a salt, but got "+repr(salt)+
                                ". Two ions are given as add_salt(cation, anion, concentration).")
            salt.dissolve_in(self,concentration)
            return
        if len(args)!=3:
            raise TypeError("add_salt takes either (salt, concentration) or (cation, anion, "+
                            "concentration), but got "+str(len(args))+" positional arguments")
        cation,anion,concentration=args
        self._add_salt_from_ions(cation,anion,concentration,**kwargs)

    def _add_salt_from_ions(self,cation:"IonProperties | str",anion:"IonProperties | str",
                            concentration:ExpressionOrNum,*,_salt:"SaltProperties | None"=None,
                            **kwargs:Any)->None:
        """The two-ion form of :py:meth:`add_salt`. ``_salt`` is the library entry it came from, if
        any, which is what carries the name and the surface tension increment."""
        cation_kwargs={k[len("cation_"):]:kwargs.pop(k) for k in list(kwargs) if k.startswith("cation_")}
        anion_kwargs={k[len("anion_"):]:kwargs.pop(k) for k in list(kwargs) if k.startswith("anion_")}
        if kwargs:
            raise TypeError("unexpected arguments "+str(sorted(kwargs)))
        # add_ion overwrites the ion's bulk concentration, so what was there before this salt has to
        # be kept: two salts may share an ion (NaCl + Na2SO4), and then it carries the sum.
        before=dict(self._bulk_concentrations)
        c=self.add_ion(cation,concentration,**cation_kwargs)
        a=self.add_ion(anion,concentration,**anion_kwargs)
        # Swapping the two would still give an electroneutral solution, but the stoichiometry it
        # reports would be that of a different salt, so say so instead.
        if c.charge_number<0 or a.charge_number>0:
            raise ValueError("add_salt expects the cation first and the anion second, but got '"+
                             c.name+"' (z="+str(c.charge_number)+") and '"+a.name+"' (z="+
                             str(a.charge_number)+")")
        salt=DissolvedSalt(c,a,concentration,salt=_salt)
        self._bulk_concentrations[c.name]=before.get(c.name,0)+salt.cation_stoichiometry*concentration
        self._bulk_concentrations[a.name]=before.get(a.name,0)+salt.anion_stoichiometry*concentration
        if salt.name in self._salt_table:
            raise ValueError("'"+salt.name+"' is already dissolved in '"+self.describe()+
                             "'. Dissolve it once, at the total concentration.")
        self._salt_table[salt.name]=salt
        # As initial conditions as well, under the names the transport equations give the fields --
        # the salt's own, and both ions'. Every material scale is computed by evaluating an
        # expression "at the IC", and the surface tension is now one of the expressions that mentions
        # a concentration field. Which one it mentions depends on the model, so record all of them;
        # CompositionInitialCondition picks out only the mass fractions it needs, so these are inert
        # for anything that does not ask.
        from ..equations.electrostatics import ion_fieldname_stem
        self.initial_condition["c_"+salt.fieldname_stem()]=concentration
        for ion in (c,a):
            self.initial_condition["c_"+ion_fieldname_stem(ion.name)]=self._bulk_concentrations[ion.name]

    def get_ions(self)->"dict[str,IonProperties]":
        """The dissolved ions, in a deterministic order -- the field order downstream depends on it."""
        return {n:self._ion_table[n] for n in sorted(self._ion_table)}

    def get_salts(self)->"dict[str,DissolvedSalt]":
        """The dissolved salts, in a deterministic order.

        A salt is what the electroneutral transport model solves for, one field each; the ions it
        contributes are then substitutions. An ion added on its own with :py:meth:`add_ion` is not
        in here, since it is not half of a known pair.
        """
        return {n:self._salt_table[n] for n in sorted(self._salt_table)}

    def get_ion_molality(self,ion_name:str,temperature:ExpressionNumOrNone=None)->ExpressionOrNum:
        r"""
        The molality of a dissolved ion, :math:`m_i=c_i/\rho`, in mol per kg of solvent.

        The concentration is the field the salt transport solves for and the density is this
        material's own -- which, under the dilute-solute convention, is the mass of *solvent* per
        unit volume, so the ratio is a molality without further correction.
        """
        if ion_name not in self._ion_table:
            raise RuntimeError("No ion '"+ion_name+"' dissolved in '"+self.describe()+"'")
        from ..equations.electrostatics import ion_fieldname_stem
        rho=self.mass_density if temperature is None else self.get_reference_mass_density(temperature)
        return var("c_"+ion_fieldname_stem(ion_name))/rho

    def _aiomfac_ion_subgroups(self,modelname:str)->dict[str,str]:
        """``{ion name: AIOMFAC subgroup name}`` for the dissolved ions, or a clear refusal."""
        if modelname!="AIOMFAC":
            raise RuntimeError("The activity model '"+modelname+"' knows nothing about ions, and "+
                               "'"+self.describe()+"' has "+", ".join(sorted(self.get_ions()))+
                               " dissolved. AIOMFAC is the only model here with an electrolyte "+
                               "extension; use it, or dissolve nothing.")
        res:dict[str,str]={}
        unknown:list[str]=[]
        for name,ion in self.get_ions().items():
            sub=getattr(ion,"aiomfac_subgroup",None)
            if sub is None:
                unknown.append(name)
            else:
                res[name]=sub
        if unknown:
            raise RuntimeError("AIOMFAC has no electrolyte parameters for "+", ".join(unknown)+
                               ", so it cannot give an activity coefficient for this solution. The "+
                               "ions it can do are listed in pyoomph.materials.ions as "+
                               "AIOMFAC_SUBGROUP_OF_ION.")
        return res

    def _set_activity_coefficients_with_ions(self,modelname:str,use_multi_return:bool | int=3)->None:
        """Activity coefficients of a salted mixture, i.e. the full AIOMFAC with its middle- and
        long-range parts. Sets both the solvents' coefficients and the ions'."""
        from .activity_electrolyte import AIOMFACElectrolyteMixture,AIOMFACElectrolyteMultiReturnExpression
        server=ActivityModel.get_activity_model_by_name(modelname)
        assert isinstance(server,UNIFACLikeActivityModel)
        ion_subgroups=self._aiomfac_ion_subgroups(modelname)
        salts=self.get_salts()
        # The solvents only: a salt is described by its ions here, not as a UNIFAC molecule, whether
        # or not it also happens to be a composition field.
        solvent_components=[cn for cn in sorted(self.components) if cn not in salts]
        molecule_subgroups:dict[str,dict[str,int]]={}
        for cn in solvent_components:
            comp=self.get_pure_component(cn)
            assert isinstance(comp,PureLiquidProperties)
            groups=comp._UNIFAC_groups.get(modelname,{})
            if len(groups)==0:
                raise RuntimeError("Component "+cn+" has no UNIFAC groups defined for model "+modelname)
            molecule_subgroups[cn]=dict(groups)
        self._unifac_electrolyte=AIOMFACElectrolyteMixture(server,molecule_subgroups,
                                                           list(ion_subgroups.values()))
        gen=UNIFACPyoomphExpressionGenerator()
        basis_factor:ExpressionOrNum=1
        if getattr(self,"_salts_are_components",False):
            # Everything follows from the mass fractions, with no density in the way: the molality is
            # per kg of *solvent*, so it divides by what is left after the salts.
            n={cn:var("massfrac_"+cn)/self.get_pure_component(cn).molar_mass
               for cn in solvent_components}
            n_solvent=subexpression(sum(n.values()))
            x:dict[str,ExpressionOrNum]={cn:n[cn]/n_solvent for cn in solvent_components}
            w_salt_total=subexpression(sum(var("massfrac_"+sn) for sn in sorted(salts)))
            molal={}
            for name,sub in ion_subgroups.items():
                per_kg:ExpressionOrNum=0
                for sn,salt in salts.items():
                    nu=salt.stoichiometry_of(name)
                    if nu:
                        per_kg=per_kg+nu*var("massfrac_"+sn)/(salt.molar_mass*(1-w_salt_total))
                molal[sub]=per_kg/(mol/kilogram)
            # molefrac_* now counts a salt as one particle per formula unit where AIOMFAC counts its
            # nu ions, so the coefficient that goes with pyoomph's mole fraction is larger by the
            # ratio of the two totals. See dev_docs/salt_transport.md.
            n_salt=subexpression(sum(var("massfrac_"+sn)/salts[sn].molar_mass for sn in sorted(salts)))
            basis_factor=1+n_salt/n_solvent
        else:
            # A pure solvent's salt-free mole fraction is one, and saying so keeps the expression free
            # of a molefrac_ field that a pure liquid does not otherwise define.
            x={cn:(1 if self.is_pure else var("molefrac_"+cn)) for cn in solvent_components}
            # Nondimensional: the AIOMFAC correlations are written for a molality in mol/kg and a
            # temperature in Kelvin, so what goes in must be the bare numbers.
            molal={sub:self.get_ion_molality(name)/(mol/kilogram)
                   for name,sub in ion_subgroups.items()}
        nspecies=len(solvent_components)+len(ion_subgroups)
        if use_multi_return==True or ((use_multi_return is not False) and nspecies>=use_multi_return):
            # The same maths, evaluated in generated C with a finite-difference Jacobian rather than
            # differentiated symbolically. Worth it as soon as there are a few species, which a
            # salted mixture reaches immediately: two ions already count.
            self._unifac_multi_return=AIOMFACElectrolyteMultiReturnExpression(self._unifac_electrolyte)
            coeffs=self._unifac_multi_return.get_activity_coefficients(x,molal,var("temperature"))
        else:
            coeffs=self._unifac_electrolyte.activity_coefficients(gen,x,molal,var("temperature")/kelvin)
        for cn in solvent_components:
            self.activity_coefficients[cn]=cast(Expression,coeffs[cn]*basis_factor)
        for name,sub in ion_subgroups.items():
            self.ion_activity_coefficients[name]=cast(Expression,coeffs[sub])

    def get_ion_activity_coefficient(self,ion_name:str)->ExpressionOrNum:
        r"""
        The molality-based activity coefficient of a dissolved ion, with infinite dilution in pure
        water as its reference -- the convention AIOMFAC reports and the literature tabulates.

        Available once :py:meth:`set_activity_coefficients_by_unifac` has been called on a material
        that carries ions.
        """
        if ion_name not in self.ion_activity_coefficients:
            raise RuntimeError("No activity coefficient for the ion '"+ion_name+"' in '"+
                               self.describe()+"'. Call set_activity_coefficients_by_unifac"+
                               "('AIOMFAC') after dissolving the salts.")
        return self.ion_activity_coefficients[ion_name]

    def get_mean_ionic_activity_coefficient(self,salt_name:str)->ExpressionOrNum:
        r"""
        :math:`\gamma_\pm=(\gamma_+^{\nu_+}\gamma_-^{\nu_-})^{1/(\nu_++\nu_-)}` of one dissolved
        salt, which is the combination that can actually be measured -- a single ion's coefficient
        cannot.
        """
        salts=self.get_salts()
        if salt_name not in salts:
            raise RuntimeError("No salt '"+salt_name+"' dissolved in '"+self.describe()+"'")
        s=salts[salt_name]
        gp=self.get_ion_activity_coefficient(s.cation.name)
        gm=self.get_ion_activity_coefficient(s.anion.name)
        nu=s.cation_stoichiometry+s.anion_stoichiometry
        return (gp**s.cation_stoichiometry*gm**s.anion_stoichiometry)**rational_num(1,nu)

    def treat_salts_as_components(self)->None:
        r"""
        Promote every dissolved salt from a dilute solute to a real component of the mixture.

        The salt gains a mass fraction that sums to unity with the solvents, a mole fraction, and a
        share of the volume, so the composition equations transport it the way they transport
        anything else -- including, at an evaporating interface, the :math:`j_i=0` case of a term
        they already write. That is the whole point: none of the machinery the dilute treatment
        needs applies here.

        **In place**, and idempotent. The alternative -- returning a new material -- would leave the
        interface properties, the mass transfer model and the vapour pressures pointing at the old
        one, and two objects disagreeing about what ``massfrac_water`` means is exactly the kind of
        split that produces a plausible wrong answer.

        The density becomes volume-additive,

        .. math:: \frac{1}{\rho}=\frac{w_\mathrm{solv}}{\rho_\mathrm{solv}}+\sum_s\frac{w_sV_{\phi,s}}{M_s}

        with :math:`\rho_\mathrm{solv}` the material's own correlation evaluated at the *renormalised*
        salt-free composition -- at the raw mass fractions it would think the solvent had been
        diluted by the salt, and misstate e.g. the glycerol-to-water ratio by the salt's mass
        fraction.
        """
        if getattr(self,"_salts_are_components",False):
            return
        salts=self.get_salts()
        if not salts:
            raise RuntimeError("Nothing is dissolved in '"+self.describe()+"', so there is no salt "+
                               "to make a component of.")
        # Everything that can refuse does so before anything is changed: a half-upgraded material
        # would be worse than one that never started.
        volumes={n:s.get_apparent_molar_volume() for n,s in salts.items()}
        masses={n:s.molar_mass for n,s in salts.items()}
        solvents=sorted(self.components)
        if self.passive_field in salts:
            raise RuntimeError("The passive field cannot be a salt")
        solvent_density=self.mass_density
        # At the initial composition, while the material still consists of solvents alone: the
        # initial mass fractions are worked out from it further down, and by then asking for a
        # density would mean asking about a composition that includes a salt it does not yet have.
        # Substituted structurally rather than through evaluate_at_condition, for the same reason as
        # below and for one more: a mixture built without a temperature leaves the density a function
        # of var("temperature"), which is a perfectly good initial condition but not something the
        # unit collector can be asked about.
        rho_solvent_at_IC=Expression(solvent_density)
        for field,value in self.initial_condition.items():
            rho_solvent_at_IC=_pyoomph.GiNaC_subs(rho_solvent_at_IC,var(field),Expression(value))

        for n,s in salts.items():
            self.pure_properties[n]=_salt_pseudo_component(n,masses[n],volumes[n])
        self.components=set(solvents)|set(salts)
        self.required_adv_diff_fields=self.components-{cast(str,self.passive_field)}

        w_salt={n:var("massfrac_"+n) for n in sorted(salts)}
        w_salt_total=subexpression(sum(w_salt.values()))
        # GiNaC_subs, not evaluate_at_condition: substituting through the latter runs the unit
        # collector over the whole expression, which cannot get a unit out of a subexpression that
        # contains a temperature field -- and every density correlation here is one of those. This
        # replaces a symbol by an expression and leaves the rest of the tree alone.
        rho_solvent=solvent_density
        for c in solvents:
            rho_solvent=_pyoomph.GiNaC_subs(0+rho_solvent,var("massfrac_"+c),
                                            var("massfrac_"+c)/(1-w_salt_total))
        self.mass_density=1/((1-w_salt_total)/rho_solvent
                             +sum(w_salt[n]*volumes[n]/masses[n] for n in sorted(salts)))

        for n,s in salts.items():
            # The ambipolar diffusivity, i.e. the rate the ion pair travels at: what the salt field
            # meant in the dilute treatment, and what its mass fraction means here.
            self.set_diffusion_coefficient(n,s.get_diffusivity(self))

        self._set_salt_initial_conditions(salts,volumes,masses,rho_solvent_at_IC)
        self._salts_are_components=True
        if getattr(self,"_unifac_model",None) is not None:
            # The mole fraction basis has changed, so the activity coefficients and the vapour
            # pressures built on it have to be rebuilt.
            self.set_activity_coefficients_by_unifac(cast(str,self._unifac_model))

    def _set_salt_initial_conditions(self,salts:"dict[str,DissolvedSalt]",
                                     volumes:dict[str,ExpressionOrNum],
                                     masses:dict[str,ExpressionOrNum],
                                     rho_solvent:ExpressionOrNum)->None:
        r"""Turn the concentrations the salts were dissolved at into mass fractions.

        Exact under volume additivity: a cubic metre holding :math:`c_s` moles of each salt gives
        them :math:`\sum c_sV_{\phi,s}` of its volume and the solvent the rest, so the mass of the
        whole is :math:`\rho_\mathrm{solv}(1-\sum c_sV_{\phi,s})+\sum c_sM_s`.
        """
        ic=self.initial_condition
        rho_solv=rho_solvent
        conc={n:s.concentration for n,s in salts.items()}
        salt_volume=sum(conc[n]*volumes[n] for n in sorted(salts))
        salt_mass=sum(conc[n]*masses[n] for n in sorted(salts))
        total=rho_solv*(1-salt_volume)+salt_mass
        w_salt_total:ExpressionOrNum=0
        for n in sorted(salts):
            w=conc[n]*masses[n]/total
            ic["massfrac_"+n]=w
            w_salt_total=w_salt_total+w
        # The solvents were a complete composition on their own; they now share with the salt.
        for c in sorted(self.components):
            if c in salts:
                continue
            ic["massfrac_"+c]=ic.get("massfrac_"+c,0)*(1-w_salt_total)

    def get_salt_surface_tension_shift(self,as_field:bool=True,field_prefix:str="c_")->ExpressionOrNum:
        r"""
        :math:`\sum_s (\mathrm{d}\sigma/\mathrm{d}c)_s\,c_s`, i.e. how much the dissolved salts raise
        the surface tension of this liquid.

        With ``as_field`` (the default) each :math:`c_s` is the *field* the transport equations solve
        for, which is what makes an evaporating, salt-enriched surface pull on its surroundings.
        Without it, each is the uniform bulk concentration, which is the right fallback when nothing
        solves for the salt: the absolute surface tension is then still correct, there is just no
        gradient to drive anything.

        Salts are surface-depleted, so this is positive and the enriched region has the *higher*
        surface tension -- Marangoni flow runs towards it, the opposite of the surfactant case.
        """
        res:ExpressionOrNum=0
        salts=self.get_salts()
        for s in salts.values():
            if is_zero(0+s.surface_tension_increment):
                continue
            if not as_field:
                res=res+s.surface_tension_increment*s.concentration
                continue
            # Written against an *ion* concentration rather than the salt's own field, because that
            # is the name both electrolyte models have: a substitution under the electroneutral one,
            # a solved dof under Nernst-Planck. So the same surface tension law drives the same
            # Marangoni stress either way, which is the entire reason the names are shared.
            ion,nu=s._identifying_ion(salts)
            from ..equations.electrostatics import ion_fieldname_stem
            res=res+s.surface_tension_increment/nu*var(field_prefix+ion_fieldname_stem(ion.name))
        return res

    def get_bulk_concentration(self,ion_name:str)->ExpressionOrNum:
        """The reservoir concentration of one dissolved ion."""
        if ion_name not in self._bulk_concentrations:
            raise RuntimeError("No ion '"+ion_name+"' dissolved in '"+self.describe()+"'")
        return self._bulk_concentrations[ion_name]

    def get_ion_diffusivity(self,ion_name:str,
                            temperature:ExpressionNumOrNone=None)->ExpressionOrNum:
        r"""
        The diffusivity of one dissolved ion *in this solvent, at this temperature*.

        This is where the tabulated aqueous value gets corrected: the ion owns the Walden rule and
        its exponent (:py:meth:`IonProperties.get_diffusivity_in_solvent`), and the solvent is the
        only one that knows the viscosity to put into it. Everything that consumes a diffusivity --
        :py:func:`~pyoomph.equations.electrostatics.ions_from_material` and
        :py:meth:`electric_conductivity_from_ions` -- goes through here, so the transport and the
        conductivity cannot end up using differently corrected numbers.

        ``temperature`` follows the same convention as :py:meth:`get_absolute_permittivity`: ``None``
        leaves it as ``var("temperature")`` and uses the material's viscosity expression as it
        stands -- for a mixture that expression depends on the composition too, so the ion's
        diffusivity then follows the local mass fractions as well, which is what a solvent that
        thickens as it evaporates should do. A given temperature evaluates both at it and at the
        material's initial composition.
        That is not just symmetry -- substituting a temperature *field* into water's viscosity
        correlation fails outright, because the field sits inside the exponent of
        :math:`10^{247.8/(T-140)}` and the unit machinery cannot get a unit out of that. The raw
        expression is fine, since by the time it reaches the code generator the field has a scale.
        """
        if ion_name not in self._ion_table:
            raise RuntimeError("No ion '"+ion_name+"' dissolved in '"+self.describe()+"'")
        if temperature is None:
            return self._ion_table[ion_name].get_diffusivity_in_solvent(self.dynamic_viscosity)
        return self._ion_table[ion_name].get_diffusivity_in_solvent(
            self.get_reference_dynamic_viscosity(temperature),temperature)

    def get_net_charge_number(self)->ExpressionOrNum:
        r""":math:`\sum_i z_i c_i^\infty`, which must vanish for an electroneutral reservoir."""
        return sum(i.charge_number*self._bulk_concentrations[n] for n,i in self.get_ions().items())

    def get_ionic_strength(self)->ExpressionOrNum:
        r"""The reservoir molar ionic strength :math:`I=\frac{1}{2}\sum_i z_i^2 c_i^\infty`."""
        return sum(i.charge_number**2*self._bulk_concentrations[n]
                   for n,i in self.get_ions().items())/2

    def get_debye_length(self,temperature:ExpressionNumOrNone=None)->Expression:
        r"""
        The screening length :math:`\lambda_\mathrm{D}` of the dissolved ions in this solvent.

        ``temperature`` defaults to ``None``, i.e. the field ``var("temperature")`` with a symbolic
        permittivity; pass a definite temperature to get a number out.
        """
        T=temperature if temperature is not None else var("temperature")
        return debye_length(self.get_absolute_permittivity(temperature),self.get_ionic_strength(),T)

    def electric_conductivity_from_ions(self,temperature:ExpressionNumOrNone=None)->ExpressionOrNum:
        r"""
        The Nernst-Einstein conductivity
        :math:`\sigma_\mathrm{c}=\frac{F^2}{RT}\sum_i z_i^2 D_i c_i^\infty`.

        Deriving it from the same ion table that the resolved model uses is what keeps a
        leaky-dielectric run and a Nernst-Planck run from silently disagreeing about the material.

        The diffusivities come from :py:meth:`get_ion_diffusivity`, so this inherits the Walden
        correction and rises with temperature at roughly the 2%/K a conductivity meter compensates
        for. Without it the two temperature dependences cancel exactly and the conductivity would
        come out constant, which it is not.
        """
        T=var("temperature") if temperature is None else temperature
        return faraday_constant**2/(gas_constant*T)*sum(
            i.charge_number**2*self.get_ion_diffusivity(n,temperature)*self._bulk_concentrations[n]
            for n,i in self.get_ions().items())

    def set_electric_conductivity_from_ions(self,temperature:ExpressionNumOrNone=None)->None:
        """Sets :py:attr:`electric_conductivity` to :py:meth:`electric_conductivity_from_ions`."""
        self.electric_conductivity=self.electric_conductivity_from_ions(temperature)

    def get_reference_dynamic_viscosity(self,temperature:ExpressionOrNum | None=None) -> Expression:
        ics=self.initial_condition.copy()
        if temperature is not None:
            ics["temperature"]=temperature
        return self.evaluate_at_condition(self.dynamic_viscosity,ics)

    def get_reference_mass_density(self,temperature:ExpressionOrNum | None=None) -> Expression:
        ics=self.initial_condition.copy()
        if temperature is not None:
            ics["temperature"]=temperature
        return self.evaluate_at_condition(self.mass_density,ics)


    def set_reference_scaling_to_problem(self, problem: "Problem", temperature: ExpressionOrNum | None = None, **kwargs: ExpressionOrNum):
            """
            Set the reference scaling to nondimensionalize a dimensional problem. 

            Args:
                problem: The problem for which the reference scaling is being set.
                temperature: The temperature to be used for nondimensionalization. If not provided, the initial condition temperature will be used.
                **kwargs: Additional parameters to be used for scaling.

            Raises:
                RuntimeError: If at least two of the scales 'temporal', 'spatial', and 'velocity' are not set in the problem before.
            """
            ics = self.initial_condition.copy()
            if temperature is not None:
                ics["temperature"] = temperature
            for k, v in kwargs.items():
                ics[k] = v

            TEMPS = problem.get_scaling("temperature", none_if_not_set=True)
            if TEMPS is None:
                problem.set_scaling(temperature=kelvin)
            rho0 = self.evaluate_at_condition(self.mass_density, ics)
            assert_dimensional_value(rho0)
            mu0 = self.evaluate_at_condition(self.dynamic_viscosity, ics)
            assert_dimensional_value(mu0)
            US = problem.get_scaling("velocity", none_if_not_set=True)
            XS = problem.get_scaling("spatial", none_if_not_set=True)
            TS = problem.get_scaling("temporal", none_if_not_set=True)
            if US and XS and TS:
                pass
            elif XS and TS:
                US = XS / TS  # type:ignore
                problem.set_scaling(velocity=US)
            elif XS and US:
                TS = XS / US  # type:ignore
                problem.set_scaling(temporal=TS)
            elif US and TS:
                XS = US * TS  # type:ignore
                problem.set_scaling(spatial=XS)
            else:
                raise RuntimeError("Please set at least two of the scales 'temporal', 'spatial' and 'velocity' first")
            if problem.get_scaling("pressure", none_if_not_set=True) is None:
                PS = mu0 * US / XS
                problem.set_scaling(pressure=PS)
            if problem.get_scaling("mass_density", none_if_not_set=True) is None:
                problem.set_scaling(mass_density=rho0)
            if hasattr(self, "thermal_conductivity") and self.thermal_conductivity is not None and hasattr(self, "specific_heat_capacity") and self.specific_heat_capacity is not None:
                lambda0 = self.evaluate_at_condition(self.thermal_conductivity, ics)
                assert_dimensional_value(lambda0)
                cp0 = self.evaluate_at_condition(self.specific_heat_capacity, ics)
                assert_dimensional_value(cp0)
                problem.set_scaling(thermal_conductivity=lambda0)
                problem.set_scaling(rho_cp=rho0 * cp0)

    def get_vapor_mass_concentration(self,component:str,relative_humidity_for_far_field:ExpressionNumOrNone=None,temperature:ExpressionNumOrNone=None,at_mixture_composition:bool | dict[str, ExpressionOrNum]=True):
        """
        Calculates the saturation vapor concentration :math:`c_{sat}` for the given component in [kg/m^3].
        If relative_humidity_for_far_field is set, it does not apply Raoult's law, but uses the relative humidity to calculate the vapor concentration in the far field
        
        Args:
            component: Name of the component.
            relative_humidity_for_far_field: Relative humidity in the far field. If set, it is used to calculate the vapor concentration of the pure vapor in the far field.
            temperature: Temperature at which to calculate the vapor concentration. If not set, the temperature from the initial condition is used.
            at_mixture_composition: If set to ``True``, the vapor concentration is calculated at the mixture initial composition. If set to a dictionary, the vapor concentration is calculated at the given composition.
        """
        # get_pure_component/get_vapor_pressure_for are only implemented by the concrete subclasses of
        # BaseLiquidProperties (PureLiquidProperties and MixtureLiquidProperties), not by this base class itself.
        assert isinstance(self,(PureLiquidProperties,MixtureLiquidProperties))
        gas_constant=8.3144598*joule/(mol*kelvin)
        pure_component=self.get_pure_component(component)
        if pure_component is None:
            raise RuntimeError("Component '"+component+"' is not present in "+str(self))
        M=pure_component.molar_mass
        temperature_set=temperature
        if temperature is None:
            temperature=var("temperature")
        if relative_humidity_for_far_field is not None:
            pvap_pure=self.get_vapor_pressure_for(component,pure=True)
            if pvap_pure is None:
                raise RuntimeError("No vapor pressure set for the pure component '"+component+"', cannot calculate the vapor concentration from the relative humidity")
            psat:ExpressionOrNum=relative_humidity_for_far_field*pvap_pure
        else:
            psat_of_component=self.get_vapor_pressure_for(component)
            if psat_of_component is None:
                raise RuntimeError("No vapor pressure set for component '"+component+"' in "+str(self))
            psat=psat_of_component

        csat=psat/(temperature * gas_constant)*M
        if temperature_set is not None:
            csat=self.evaluate_at_condition(csat,{},temperature=temperature_set)
        if at_mixture_composition:
            if at_mixture_composition is True:
                csat=self.evaluate_at_condition(csat,self.initial_condition)
            else:
                csat=self.evaluate_at_condition(csat,at_mixture_composition)
        return csat


class BaseGasProperties(MaterialProperties):
    """
    A base class for defining gaseous materials.
    """    
    state_of_matter="gas"
    passive_field:str | None=None
    required_adv_diff_fields:set[str]=set()
    possible_properties:set[str]={"mass_density","dynamic_viscosity","relative_permittivity","electric_conductivity"}
    _output_properties:OutputPropertiesType = {"mass_density": None, "dynamic_viscosity": None, "relative_permittivity": None}
    def __init__(self):
        super(BaseGasProperties, self).__init__()
        self.mass_density:ExpressionOrNum
        #: The dynamic viscosity of the gas.
        self.dynamic_viscosity:ExpressionOrNum
        #: Relative permittivity, dimensionless. About 1.0006 for air at ambient conditions, i.e.
        #: 1 is almost always good enough. Annotated rather than assigned, see BaseLiquidProperties.
        self.relative_permittivity:ExpressionOrNum
        #: Ohmic conductivity of the gas.
        self.electric_conductivity:ExpressionOrNum

class BaseSolidProperties(MaterialProperties):
    """
    A base class for defining solid materials.
    """
    state_of_matter="solid"
    passive_field:str | None=None
    required_adv_diff_fields:set[str]=set()
    possible_properties:set[str]={"mass_density","relative_permittivity","electric_conductivity"}
    _output_properties:OutputPropertiesType = {"mass_density": None}


class BaseMixedProperties:
    """
    A base class used for defining mixtures of pure components.
    """
    name:str
    components:set[str] = set()
    # Both come from the MaterialProperties base that the concrete mixtures inherit from as well;
    # restated because this mixin fills them in below.
    _output_properties:OutputPropertiesType
    required_adv_diff_fields:set[str]
    def __init__(self,pure_props:dict[str,MaterialProperties]):

        self.pure_properties=pure_props
        self.is_static=False #You can make a mixture static, i.e. remove all mass fractions fields from it
        assert hasattr(self,"passive_field")
        #: The passive field of the mixture. This is the field for which a advective-diffusive equation is not solved, since we can calculate it from the mass fractions of the other components.
        self.passive_field:str | None=getattr(self,"passive_field")
        if self.passive_field is None:	#Select one passive field
            for a in reversed(sorted(self.components)):
                self.passive_field=a
                break
        assert hasattr(self,"_output_properties")
        self._output_properties=self._output_properties.copy()
        def make_diffusion_coeff_lambda(k1:str,k2:str)->Callable[[MaterialProperties],ExpressionNumOrNone]:
            return lambda self: self.get_diffusion_coefficient(k1, k2) #type:ignore
        # sorted: components is a set, and its iteration order would otherwise decide the order in
        # which these properties are sampled to files
        for k1 in sorted(self.components):
            if k1==self.passive_field:
                continue
            for k2 in sorted(self.components):
                if k2 == self.passive_field:
                    continue
                self._output_properties["diffusivity_"+k1+"__"+k2]=make_diffusion_coeff_lambda(k1,k2)

        self.required_adv_diff_fields=self.components-{self.passive_field}
        if self.components!=set(self.pure_properties.keys()):
            raise ValueError("Cannot create a mixture with the components "+str(sorted(self.components))+" by passing the wrong pure component properties: "+str(self.pure_properties))
        self._diffusion_table:dict[tuple[str,str],ExpressionOrNum]={}


    def set_passive_field(self,passive_component:str):
        self.passive_field=passive_component
        self.required_adv_diff_fields=self.components-{self.passive_field}
    
    @overload
    def get_pure_component(self,name:str,raise_error:Literal[False]=...)->MaterialProperties | None: ...
    @overload
    def get_pure_component(self,name:str,raise_error:Literal[True])->MaterialProperties: ...

    def get_pure_component(self,name:str,raise_error:bool=False)->MaterialProperties | None:
        """
        Returns the pure component properties for the specified component.

        Args:
            name: Name of the pure component.
            raise_error: Raise an error if the component is not present in the mixture. Otherwise, ``None`` is returned.

        Returns:
            The pure properties of the component.
        """
        if name in self.pure_properties.keys():
            return self.pure_properties[name]
        elif raise_error:
            raise RuntimeError("Component '"+str(name)+"' is not present in the mixture")
        else:
            return None
    

    @overload
    def set_diffusion_coefficient(self,arg1:ExpressionOrNum,arg2:Literal[None]=...,arg3:Literal[None]=...)->None: ...
    @overload
    def set_diffusion_coefficient(self,arg1:str,arg2:ExpressionOrNum,arg3:Literal[None]=...)->None: ...
    @overload
    def set_diffusion_coefficient(self,arg1:str,arg2:str,arg3:ExpressionOrNum)->None: ...

    def set_diffusion_coefficient(self, arg1: ExpressionOrNum | str, arg2: ExpressionNumOrNone | str = None, arg3: ExpressionNumOrNone = None):
        """
        Set the diffusion coefficient for the specified component in the mixture.

        Parameters:
            arg1: Either the diagonal diffusion coefficient for all components or the name of the component to set the diffusion coefficient.
            arg2: Either the diagonal diffusion coeffient for the component given as first argument or the name of the second component or ``None`` for off-diagonal diffusion.
            arg3: The diffusion coefficient for off-diagonal diffusion.         
        """
        if arg3 is None and (arg2 is not None):
            assert isinstance(arg1, str)
            assert isinstance(arg2, (Expression, int, float))
            name1 = arg1
            name2 = arg1
            coeff = arg2
        elif arg3 is not None and arg2 is not None:
            assert isinstance(arg1, str)
            assert isinstance(arg2, str)
            assert isinstance(arg3, (Expression, int, float))
            name1 = arg1
            name2 = arg2
            coeff = arg3
        elif arg2 is None and arg3 is None:
            assert isinstance(arg1, (Expression, int, float))
            for c in sorted(self.components):
                self.set_diffusion_coefficient(c, arg1, None)
            return
        else:
            raise RuntimeError("set_diffusion_coefficient needs to be called with either <component name>, <diffusivity> for diagonal diffusion or <component name1>, <component name2>, <diffusivity> for off-diagonal diffusion")
        
        fs = (name1, name2,)
        if name1 not in self.components:
            raise RuntimeError("Cannot set the diffusivity for " + str(fs) + " since " + name1 + " is not present in the mixture")
        if name2 not in self.components:
            raise RuntimeError("Cannot set the diffusivity for " + str(fs) + " since " + name2 + " is not present in the mixture")
        
        self._diffusion_table[fs] = coeff

    def get_diffusion_coefficient(self,n1:str,n2:str | None=None,default:ExpressionNumOrNone | None=None)->ExpressionNumOrNone:
        """
        Returns the diffusion coefficient between two components. If only one component is given, the diagonal element is returned.

        Args:
            n1: Component name for the diffusive flux.
            n2: Potential second component name for an off-diagonal diffusion coefficient. If None, the diagonal element is returned. 
            default: Default value to return if the diffusion coefficient is not set.

        Returns:
            The diffusion coefficient.
        """
        if n2 is None:
            n2=n1
        fs=(n1,n2,)
        return self._diffusion_table.get(fs,default)

    # Sets factor D_T in front of J=-J_massdiff - rho D_T grad(T)
    def set_thermophoresis_coefficient(self,for_component:str | Iterable[str],coeff:ExpressionOrNum):        
        if isinstance(for_component, (list, tuple,set)): #Usually not so meaningful...
            for a in for_component:
                self.set_thermophoresis_coefficient(a,coeff)
        else:
            assert isinstance(for_component,str)
            self._diffusion_table[(for_component,"temperature",)]=coeff

    def get_diffusive_mass_flux_for(self,n:str)->ExpressionOrNum:
        """
        Returns the diffusive mass flux for one component according to Fick's law.
        """
        res:ExpressionOrNum
        if n==self.passive_field:
            res=0
            for c in sorted(self.components):
                if c!=self.passive_field:
                    res-=self.get_diffusive_mass_flux_for(c)
            return res

        res = 0
        for fn2 in sorted(self.components):
            f2 = var("massfrac_"+fn2)
            D = self.get_diffusion_coefficient(n, fn2,default=0)
            assert D is not None
            res = res + D * grad(f2)
        DT=self.get_diffusion_coefficient(n,"temperature")
        if DT is not None:
            res+=DT*grad(var("temperature"))
        assert hasattr(self,"mass_density") 
        rho=getattr(self,"mass_density")
        assert isinstance(rho,(Expression,float,int))
        res = -rho * res
        return res

    def get_mass_fraction_field(self,name:str,**kwargs:Any)->Expression:
        """
        Returns the mass fraction field for the given component.
        """
        if not self.pure_properties[name]:
            raise ValueError("Mass fraction '"+name+"' is not in the components: "+str(sorted(self.components)))
        return var("massfrac_"+name,**kwargs)

    def get_mole_fraction_field(self,name:str,**kwargs:Any)->Expression:
        """
        Returns the mole fraction field for the given component.
        """
        if not self.pure_properties[name]:
            raise ValueError("Mass fraction '"+name+"' is not in the components: "+str(sorted(self.components)))
        return var("molefrac_"+name,**kwargs)


    def make_static(self,cond:dict[str, ExpressionOrNum] | None=None,temperature:ExpressionNumOrNone=None):	#TODO Make a copy!
        """
        This will make the mixture static, i.e. all mass fraction fields will be replaced by their values from the given condition. This is useful for to remove advection-diffusion equations if the composition stays homogeneous.

        Args:
            cond: Optional condition, otherwise the :py:attr:`~pyoomph.materials.generic.MaterialProperties.initial_condition` is used.
            temperature: Optional temperature        
        """
        assert isinstance(self,MaterialProperties)     
        assert isinstance(self,(BaseLiquidProperties,BaseGasProperties,BaseSolidProperties))   
        cond=cond.copy() if cond is not None else self.initial_condition
        if temperature is not None:
            cond["temperature"]=temperature        
        fields,nondims=self.generate_field_substs(cond)
        for p in self.possible_properties:
            if hasattr(self,p):
                dct:ExpressionOrNum | dict[str, ExpressionOrNum] = getattr(self, p)
                if isinstance(dct,dict):
                    for nn,pp in dct.items():
                        if not isinstance(pp,Expression):
                            pp=Expression(pp)
                        dct[nn]=_pyoomph.GiNaC_subsfields(pp, fields, nondims, {})  # TODO Global params
                    setattr(self, p,dct)
                else:
                    setattr(self,p,_pyoomph.GiNaC_subsfields(getattr(self,p),fields,nondims,{})) #TODO Global params
        self.is_static=True
        return self


    def set_by_weighted_average(self,what:str | None=None,fraction_type:str="mass_fraction"):
        """
        Calculate a property by just taking the weighted average of the properties of all pure components.
        Args:
            what: Property or expression to be calculated. If None, all properties that are present in all pure components are calculated.
            fraction_type: Which fraction to weight the average with. Can be ``"mass_fraction"`` (default) or ``"mole_fraction"``.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        assert isinstance(self,(BaseLiquidProperties,BaseGasProperties,BaseSolidProperties))   
        if what is None:
            good=True
            for p in self.possible_properties:
                for props in self.pure_properties.values():
                    if not hasattr(props,p):
                        good=False
                        break
                if good:
                    self.set_by_weighted_average(p)
        else:
            if fraction_type!="mass_fraction" and fraction_type!="mole_fraction":
                raise ValueError("Can only use fraction_type='mass_fraction' or 'mole_fraction' at the moment")
            res:ExpressionOrNum=0
            for c,v in self.pure_properties.items():
                if not hasattr(v,what) or (getattr(v,what) is None):
                    raise ValueError("Mixture component "+c+" has no property "+what+" defined to take the average for the mixture")
                pure_prop=getattr(v,what)
                fraction:ExpressionOrNum
                if c==self.passive_field:
                    fraction = 1
                    for c2 in self.pure_properties.keys():
                        if c2 != self.passive_field:
                            if fraction_type=="mole_fraction":
                                fraction=self.get_mole_fraction_field(c2)
                            else:
                                fraction-=self.get_mass_fraction_field(c2)
                else:
                    if fraction_type=="mole_fraction":
                        fraction = self.get_mole_fraction_field(c)
                    else:
                        fraction=self.get_mass_fraction_field(c)

                res+=fraction*pure_prop

            setattr(self,what,res)


#####################
class PureLiquidProperties(BaseLiquidProperties):
    """
    Properties of a pure liquid.
    """
    is_pure:bool | None=True

    def make_static(self,*args:Any,**kwargs:Any):
        return self

    def __init__(self):
        super().__init__()
        self.initial_condition["massfrac_"+self.name]=1.0
        #: Vapor pressure of the pure liquid
        self.vapor_pressure:ExpressionNumOrNone=None
        #: The salt-free vapor pressure, kept aside when a dissolved salt multiplies
        #: :py:attr:`vapor_pressure` by an activity coefficient. ``None`` while the two agree.
        self._pure_vapor_pressure:ExpressionNumOrNone=None
        self.passive_field=self.name
        #: The components are used for mixtures. Here it is just the set with only the name of the liquid as only element.
        self.components = set({self.name})
        self._UNIFAC_groups:dict[str,dict[str,int]]={}
        #: Latent heat of evaporation of the pure liquid
        self.latent_heat_of_evaporation:ExpressionNumOrNone=None
        self._output_properties=self._output_properties.copy()
        self._output_properties["vapor_pressure_"+self.name]=lambda props : self.get_vapor_pressure_for(self.name)


    
    def set_activity_coefficients_by_unifac(self,model:str,set_vapor_pressures:bool=True,
                                            use_multi_return:bool | int=3):
        """
        Activity coefficients of this liquid *with whatever is dissolved in it*.

        A pure liquid is its own reference, so without ions every coefficient is one and there is
        nothing to compute; with a salt there is, and it is the case that matters most -- brine.
        Only AIOMFAC can do it, see :py:meth:`MixtureLiquidProperties.set_activity_coefficients_by_unifac`.
        """
        if not self.get_ions():
            raise RuntimeError("'"+str(self.name)+"' is a pure liquid with nothing dissolved in it, "+
                               "so its activity coefficient is 1 by definition. Dissolve a salt "+
                               "first, or use a mixture.")
        self._unifac_model=model
        self._set_activity_coefficients_with_ions(model,use_multi_return)
        if set_vapor_pressures and self.vapor_pressure is not None:
            # Raoult with a salt-free mole fraction of one: the whole effect is in the coefficient.
            # The salt-free value is kept, since pure=True must stay answerable afterwards: a far
            # field concentration from a relative humidity is built from it, and without this the
            # brine's own lowered vapour pressure would be taken as the saturation reference.
            self._pure_vapor_pressure=self.vapor_pressure
            self.vapor_pressure=self.activity_coefficients[self.name]*self.vapor_pressure

    def set_unifac_groups(self,grps:dict[str,int],only_for:set[str] | str | None=None):
        """
        Sets the UNIFAC groups for the pure liquid, which are relevant for the activity coefficients in mixtures.

        Args:
            grps: Dictionary of UNIFAC groups and their amounts.
            only_for: Set groups only for specific group interaction models. Default is None, which sets the groups for the models ``{"AIOMFAC","Original","Dortmund"}``. 
        """
        if only_for is None:
            only_for={"AIOMFAC","Original","Dortmund"}
        elif isinstance(only_for,str):
            only_for={only_for}
        for g in only_for:
            if not (g in self._UNIFAC_groups.keys()):
                self._UNIFAC_groups[g]={}
            for grp,amount in grps.items():
                self._UNIFAC_groups[g][grp]=amount

    def set_vapor_pressure_by_Antoine_coeffs(self,A:float,B:float,C:float,convention_P:Expression=mmHg,convention_T:Expression | CelsiusClass=celsius):
        """
        Sets the vapor pressure by the Antoine equation.

        Args:
            A: Antoine coefficient A
            B: Antoine coefficient B
            C: Antoine coefficient C
            convention_P: Pressure unit for the Antoine coefficients. Default is mmHg.
            convention_T: Temperature unit for the Antoine coefficients. Default is celsius.
        """
        
        APa = A + math.log10(float(convention_P / pascal))
        
        if convention_T==kelvin:
            CKelvin=C
        elif convention_T==celsius:
            CKelvin=C-273.15
        else:
            raise RuntimeError("Only kelvin and celsius are supported for temperature unit in Antoine equation")
        TKelvin = var("temperature")/kelvin
        self.vapor_pressure=10 ** (APa - B / (CKelvin + TKelvin))* pascal

    def get_pure_component(self,name:str):
        """
        Just returns itself if the name matches. Otherwise None.
        """
        if self.name==name:
            return self
        else:
            return None

    def get_vapor_pressure_for(self,name:str,pure:bool=False) -> ExpressionNumOrNone:
        """
        Returns the vapor pressure of the pure liquid.

        Args:
            name: Name of the liquid, i.e. this one.
            pure: Return the salt-free value. Only differs once a dissolved salt has lowered the
                vapor pressure by an activity coefficient.
        """        
        if self.name==name:
            if pure and self._pure_vapor_pressure is not None:
                return self._pure_vapor_pressure
            return self.vapor_pressure
        else:
            return None

    def get_latent_heat_of_evaporation(self,name:str) -> ExpressionNumOrNone:
        """
        Returns the latent heat of evaporation for the pure liquid.
        """
        if name==self.name:
            return self.latent_heat_of_evaporation
        else:
            return None

    def __mul__(self,other:float | int | Expression)->"LiquidMixtureDefinitionComponent":
        return LiquidMixtureDefinitionComponent(self,other)

    def __rmul__(self,other:float | int | Expression)->"LiquidMixtureDefinitionComponent":
        return LiquidMixtureDefinitionComponent(self,other)


#A surfactant is by definition just a pure liquid property, can therefore be mixed with other liquids
class SurfactantProperties(PureLiquidProperties):
    """
    A surfactant is by definition a pure liquid property in pyoomph and can therefore be mixed with other liquids. However, it also can be adsorbed, desorbed and transported at an interface.
    """
    def __init__(self):
        super(SurfactantProperties, self).__init__()
        #: The default surface diffusivity of the surfactant
        self.surface_diffusivity=None


#An ion is by definition just a pure liquid property, can therefore be dissolved in other liquids
class IonProperties(PureLiquidProperties):
    """
    A dissolved ionic species.

    Like :py:class:`SurfactantProperties`, this is by definition a pure liquid property and can
    therefore be a component of a liquid mixture -- so its molar mass, diffusivity and mass fraction
    go through the machinery that is already there. What it adds is the charge it carries.

    The common ones are in :py:mod:`pyoomph.materials.ions` and are fetched by name with
    :py:func:`get_ion`; declare your own with :py:func:`~pyoomph.materials.generic.new_ion`.
    Dissolve one in a solvent with :py:meth:`BaseLiquidProperties.add_ion`, which is what
    :py:class:`~pyoomph.equations.electrostatics.NernstPlanckEquations` reads when it is given a
    ``fluid_props``.
    """
    def __init__(self):
        super(IonProperties, self).__init__()
        #: The charge number :math:`z_i`, e.g. +1 for Na+ and -2 for SO4(2-).
        self.charge_number:int=0
        #: Limiting molar conductivity at infinite dilution, if known. Only used as an alternative
        #: route to the diffusivity via the Nernst-Einstein relation.
        self.limiting_molar_conductivity:ExpressionNumOrNone=None
        #: Diffusivity at infinite dilution in *water at 25 degC*, i.e. what the tables give. The
        #: value in the actual solvent, at the actual temperature, is
        #: :py:meth:`get_diffusivity_in_solvent`.
        self.diffusivity:ExpressionNumOrNone=None
        #: Name of the corresponding AIOMFAC ion subgroup, if the ion has one. AIOMFAC spells some
        #: of them differently ("Ca++" for "Ca2+") and has parameters for only a subset, so this is
        #: what says whether an activity model can say anything about this ion at all.
        self.aiomfac_subgroup:str | None=None
        #: The exponent :math:`n` of the (fractional) Walden rule :math:`\lambda_i^0\mu^n=`const,
        #: which is how :py:meth:`get_diffusivity_in_solvent` carries the tabulated value to another
        #: temperature or another solvent. ``1`` is the plain rule, i.e. Stokes drag, and setting it
        #: to ``None`` switches the correction off, leaving the tabulated value everywhere. Use
        #: :py:func:`~pyoomph.expressions.generic.rational_num` for a fitted value rather than a
        #: float: a float exponent on a quantity that still carries units trips GiNaC up.
        self.walden_exponent:ExpressionNumOrNone=1
        #: The ion's partial molar volume at infinite dilution, on the conventional scale that sets
        #: it to zero for H+. What a dissolved salt takes up of the solution's volume, and therefore
        #: what lets a salt be a real composition field rather than a dilute rider. ``None`` where
        #: the tables have no value, and then a salt built from it is refused rather than guessed.
        self.partial_molar_volume:ExpressionNumOrNone=None

    #: The viscosity the tabulated conductivities and diffusivities refer to: water at 25 degC, as
    #: pyoomph's own water correlation gives it. Taking the measured 0.890 mPa*s instead would leave
    #: a permanent half-percent correction on an aqueous solution at the table temperature, where
    #: the correction has to be exactly 1. A test pins the two together.
    walden_reference_viscosity:ExpressionOrNum=0.890439*milli*pascal*second

    def get_diffusivity(self,temperature:ExpressionOrNum=var("temperature"))->ExpressionOrNum:
        r"""
        The diffusivity in water, from :py:attr:`limiting_molar_conductivity` by Nernst-Einstein
        (:math:`D_i=\lambda_i^0 RT/(z_i^2F^2)`) if it was not given directly.

        Note that this knows nothing about the solvent the ion is actually dissolved in -- see
        :py:meth:`get_diffusivity_in_solvent`, which is what
        :py:meth:`BaseLiquidProperties.get_ion_diffusivity` and therefore the equations use.
        """
        if self.diffusivity is not None:
            return self.diffusivity
        if self.limiting_molar_conductivity is None:
            raise RuntimeError("Ion '"+self.name+"' has neither a diffusivity nor a "+
                               "limiting_molar_conductivity set")
        return self.limiting_molar_conductivity*gas_constant*temperature \
               /(self.charge_number**2*faraday_constant**2)

    def get_diffusivity_in_solvent(self,solvent_viscosity:ExpressionOrNum,
                                   temperature:ExpressionOrNum=var("temperature"))->ExpressionOrNum:
        r"""
        The diffusivity in a solvent of the given viscosity, by the (fractional) Walden rule

        .. math:: \lambda_i^0(\mu)=\lambda_i^0\big|_\mathrm{ref}
                  \left(\frac{\mu_\mathrm{ref}}{\mu}\right)^{n}

        so that :math:`D_i\propto T/\mu^n` -- Stokes-Einstein for :math:`n=1`. Both the temperature
        dependence and the solvent dependence come from this one rule, since it is the solvent
        viscosity that carries essentially all of either.

        The rule is good to a few percent over 0-50 degC for the ions whose exponent was fitted (see
        :py:mod:`pyoomph.materials.ions`) and degrades above roughly 60 degC. Across solvents it is
        a rough estimate rather than a correlation, but it beats the alternative of returning the
        aqueous number for an ion dissolved in glycerol, which is three orders of magnitude out.
        """
        D=self.get_diffusivity(temperature)
        if self.walden_exponent is None:
            return D
        return D*(self.walden_reference_viscosity/solvent_viscosity)**self.walden_exponent

    def get_mobility(self,temperature:ExpressionOrNum=var("temperature"))->ExpressionOrNum:
        r"""The molar mobility :math:`m_i=D_i/(RT)` in water, i.e. the Einstein relation."""
        return self.get_diffusivity(temperature)/(gas_constant*temperature)

    def _times(self,other:ExpressionOrNum)->"DissolvedSpeciesComponent | LiquidMixtureDefinitionComponent":
        """``c*ion`` means two different things and the units say which.

        An ion is a pure liquid property, so a dimensionless factor keeps the mixture-component
        meaning it inherits -- a mass fraction that has to sum to unity with the others. A
        concentration cannot mean that, and means the ion is dissolved in whatever it is added to.
        """
        try:
            float(other)
        except Exception:
            return DissolvedSpeciesComponent(self,other)   # validates the unit itself
        return LiquidMixtureDefinitionComponent(self,other)

    def __mul__(self,other:ExpressionOrNum)->"DissolvedSpeciesComponent | LiquidMixtureDefinitionComponent":  #type:ignore[override]
        return self._times(other)

    def __rmul__(self,other:ExpressionOrNum)->"DissolvedSpeciesComponent | LiquidMixtureDefinitionComponent":  #type:ignore[override]
        return self._times(other)


class PureGasProperties(BaseGasProperties):
    """
    Provides properties of a pure gas.    
    """
    is_pure:bool | None=True
    def make_static(self,*args:Any,**kwargs:Any):
        return self
    def __init__(self):
        super().__init__()
        self.initial_condition["massfrac_"+self.name]=1.0
        #: Dynamic viscosity of the gas
        self.dynamic_viscosity:ExpressionOrNum
        self.passive_field = self.name
        self.components = set({self.name})

        #: Can be set to e.g. numerical values in (cm^3) e.g. according to, Fuller, E. N. and Giddings, J. C. 1965. J. Gas Chromatogr., 3, 222 or Fuller, E. N., Ensley, K. and Giddings, J. C. 1969. J. Phys. Chem., 75, 3679 or Fuller, E. N., Schettler, P. D. and Giddings, J. C. 1966. Ind. Eng. Chem., 58, 18
        self.diffusion_volume_for_Fuller_eq=None # 
        

    def mass_density_from_ideal_gas_law(self,pressure:ExpressionOrNum=var("absolute_pressure"),temperature:ExpressionOrNum=var("temperature")) -> Expression:
        """
        Returns the mass density by assuming the ideal gas law.
        Args:
            pressure: Either a constant pressure or, by default, a potentially varying pressure given by ``var("absolute_pressure")``.
            temperature: Either a constant temperature or, by default, a potentially varying temperature given by ``var("temperature")``.

        Returns:
            The mass density according to the ideal gas law.
        """
        gas_constant=8.3144598*joule/(mol*kelvin)
        spec_gas_const=gas_constant/self.molar_mass
        return pressure / (spec_gas_const * temperature)

    def set_mass_density_from_ideal_gas_law(self):
        """
        Sets the mass density by using :py:meth:`mass_density_from_ideal_gas_law`.
        """
        self.mass_density=self.mass_density_from_ideal_gas_law()

    def get_pure_component(self,name:str):
        if self.name==name:
            return self
        else:
            return None

    def __mul__(self,other:float | int | Expression)->"GasMixtureDefinitionComponent":
        return GasMixtureDefinitionComponent(self,other)

    def __rmul__(self,other:float | int | Expression)->"GasMixtureDefinitionComponent":
        return GasMixtureDefinitionComponent(self,other)

class PureSolidProperties(BaseSolidProperties):
    """
    Defines properties of a pure solid.    
    """
    is_pure:bool | None=True
    def __init__(self):
        super().__init__()
        self.initial_condition["massfrac_"+self.name]=1.0
        self.components = set({self.name})

    def get_pure_component(self,name:str):
        if self.name==name:
            return self
        else:
            return None

def salt_stoichiometry(cation_charge:int,anion_charge:int)->tuple[int,int]:
    r"""
    The number of cations and anions per formula unit, :math:`\nu_+=|z_-|/g` and :math:`\nu_-=z_+/g`
    with :math:`g=\gcd(z_+,|z_-|)` -- the only ratio for which :math:`\nu_+z_++\nu_-z_-=0`.
    """
    if cation_charge<=0 or anion_charge>=0:
        raise ValueError("A salt needs a positive and a negative charge number, got "+
                         str(cation_charge)+" and "+str(anion_charge))
    g=math.gcd(cation_charge,-anion_charge)
    return -anion_charge//g, cation_charge//g


def ambipolar_diffusivity(cation_charge:int,anion_charge:int,cation_diffusivity:ExpressionOrNum,
                          anion_diffusivity:ExpressionOrNum)->ExpressionOrNum:
    r"""
    The diffusivity of a dissolved binary salt,

    .. math:: D_\mathrm{s}=\frac{(z_+-z_-)D_+D_-}{z_+D_+-z_-D_-}

    i.e. the rate at which the *pair* moves. The two ions cannot separate: any lead one takes builds
    a charge separation whose field drags it back, so a fast anion is held up by its slow cation and
    the salt moves at neither ion's speed. This is the diffusivity of the electroneutral model, and
    it is what a Nernst-Planck solution reduces to outside the double layer.

    Derived from the ion table rather than tabulated separately, and it agrees with the measured salt
    diffusivities at 25 degC to better than half a percent -- NaCl 1.610 (measured 1.610), KCl 1.994
    (1.990), CaCl2 1.335 (1.335), Na2SO4 1.230 (1.230), HCl 3.336 (3.340), in 1e-9 m^2/s. HCl is the
    telling one: its ions differ in diffusivity by a factor of 4.6, and the salt still moves at a
    single well-defined rate about a third of the proton's.
    """
    return (cation_charge-anion_charge)*cation_diffusivity*anion_diffusivity \
           /(cation_charge*cation_diffusivity-anion_charge*anion_diffusivity)


def pure_liquid_as_mixture(pure:"PureLiquidProperties")->"MixtureLiquidProperties":
    """A one-component mixture holding the same liquid, carrying whatever is dissolved in it.

    A salt that is a composition field makes the solution a mixture, and a
    :py:class:`PureLiquidProperties` cannot host components. Rather than refuse the most ordinary
    case there is -- brine -- this wraps the solvent so that it can. The pure liquid itself is left
    untouched and becomes the mixture's single component.
    """
    cls=type("SaltedSolvent",(MixtureLiquidProperties,),
             {"components":{pure.name},"passive_field":pure.name})
    res=cast("MixtureLiquidProperties",cls({pure.name:pure}))
    # A one-component mixture has nothing to average, so the properties are the solvent's own.
    for prop in ("mass_density","dynamic_viscosity","relative_permittivity","electric_conductivity",
                 "thermal_conductivity","specific_heat_capacity"):
        if hasattr(pure,prop):
            setattr(res,prop,getattr(pure,prop))
    res.default_surface_tension=dict(pure.default_surface_tension)
    res.initial_condition=dict(pure.initial_condition)
    res._ion_table=dict(pure._ion_table)
    res._bulk_concentrations=dict(pure._bulk_concentrations)
    res._salt_table=dict(pure._salt_table)
    res.set_vapor_pressure_by_raoults_law()
    return res


def _salt_pseudo_component(name:str,molar_mass:ExpressionOrNum,
                           apparent_molar_volume:ExpressionOrNum)->"PureLiquidProperties":
    """The stand-in a salt needs to appear among a mixture's pure components.

    Not a material anybody should look up: it exists so that the mixture machinery -- mole fractions,
    the passive field, the output properties -- has something to ask for a molar mass. Its density is
    the one that reproduces the salt's apparent molar volume, and it is left unset where that volume
    is negative, since no liquid has a negative density and nothing here needs one: the mixture
    density is built from the volumes directly rather than by averaging.
    """
    cls=type("SaltComponent",(PureLiquidProperties,),{"name":name})
    res=cast("PureLiquidProperties",cls())
    res.molar_mass=molar_mass
    if not is_zero(0+apparent_molar_volume) and float(apparent_molar_volume/(meter**3/mol))>0:
        res.mass_density=molar_mass/apparent_molar_volume
    return res


def _apparent_molar_volume(cation:"IonProperties",anion:"IonProperties",nu_cation:int,
                           nu_anion:int,salt_name:str)->ExpressionOrNum:
    missing=[i.name for i in (cation,anion) if i.partial_molar_volume is None]
    if missing:
        raise RuntimeError("No partial molar volume for "+", ".join(missing)+", so the volume '"+
                           salt_name+"' takes up in solution is unknown and it cannot be treated as "+
                           "a composition field. Set partial_molar_volume on the ion, or keep the "+
                           "salt dilute.")
    return (nu_cation*cast(ExpressionOrNum,cation.partial_molar_volume)
            +nu_anion*cast(ExpressionOrNum,anion.partial_molar_volume))


class DissolvedSalt:
    """
    One salt dissolved in a liquid: which ions, in what ratio, how much of it.

    This is what :py:meth:`BaseLiquidProperties.get_salts` hands out and what the electroneutral
    transport model solves one field for. It is not a material -- the material is
    :py:class:`SaltProperties`, which this points back at when the salt came from the library.
    """
    def __init__(self,cation:"IonProperties",anion:"IonProperties",concentration:ExpressionOrNum,
                 salt:"SaltProperties | None"=None):
        self.cation=cation
        self.anion=anion
        self.concentration=concentration
        #: The library entry this came from, if any. ``None`` for a salt given as two loose ions,
        #: which then has no name of its own and no surface tension increment.
        self.salt=salt
        self.cation_stoichiometry,self.anion_stoichiometry= \
            salt_stoichiometry(cation.charge_number,anion.charge_number)
        #: Name of the salt, e.g. ``"NaCl"``, or ``"Na+/Cl-"`` when it was given as two loose ions.
        self.name=salt.name if salt is not None else cation.name+"/"+anion.name

    def stoichiometry_of(self,ion_name:str)->int:
        """How many of that ion one formula unit provides, 0 if it is not part of this salt."""
        return (self.cation_stoichiometry if ion_name==self.cation.name else 0) \
               +(self.anion_stoichiometry if ion_name==self.anion.name else 0)

    def _identifying_ion(self,salts:"dict[str,DissolvedSalt]")->"tuple[IonProperties,int]":
        """One ion of this salt that no other dissolved salt contributes to, and its stoichiometry.

        Concentrations are additive over the salts, so an ion that two salts share cannot say how
        much of *this* salt is present. The anion is tried first: sharing a cation is the common case
        (NaCl and Na2SO4 in the same solution), sharing both is not something the salt-level data can
        resolve at all.
        """
        others=[o for o in salts.values() if o is not self]
        for ion,nu in ((self.anion,self.anion_stoichiometry),(self.cation,self.cation_stoichiometry)):
            if all(o.stoichiometry_of(ion.name)==0 for o in others):
                return ion,nu
        raise RuntimeError("'"+self.name+"' shares both of its ions with the other dissolved salts, "+
                           "so no ion concentration says how much of it is present. Set the surface "+
                           "tension of the interface yourself, or dissolve them separately.")

    def fieldname_stem(self)->str:
        """What the transport equations call this salt's concentration field, e.g. ``NaCl``.

        A local import, because equations/ imports materials/ and not the other way round. This is
        the one place a material has to know what the equations will name a field, and it has to be
        the same function the ion fields go through or the two would sanitize differently.
        """
        from ..equations.electrostatics import ion_fieldname_stem
        return ion_fieldname_stem(self.name)

    @property
    def molar_mass(self)->ExpressionOrNum:
        """Molar mass of one formula unit, from the ions."""
        return (self.cation_stoichiometry*self.cation.molar_mass
                +self.anion_stoichiometry*self.anion.molar_mass)

    def get_apparent_molar_volume(self)->ExpressionOrNum:
        r""":math:`V_\phi`, the volume one mole of this salt adds to the solution, see
        :py:meth:`SaltProperties.get_apparent_molar_volume`."""
        return _apparent_molar_volume(self.cation,self.anion,self.cation_stoichiometry,
                                      self.anion_stoichiometry,self.name)

    @property
    def surface_tension_increment(self)->ExpressionOrNum:
        r""":math:`\mathrm{d}\sigma/\mathrm{d}c` of the solution, 0 if unknown."""
        return getattr(self.salt,"surface_tension_increment",0) if self.salt is not None else 0

    def get_diffusivity(self,liquid:"BaseLiquidProperties",
                        temperature:ExpressionNumOrNone=None)->ExpressionOrNum:
        """The salt's :py:func:`ambipolar_diffusivity` *in this liquid*, i.e. through
        :py:meth:`BaseLiquidProperties.get_ion_diffusivity`, so it carries the solvent correction."""
        return ambipolar_diffusivity(self.cation.charge_number,self.anion.charge_number,
                                     liquid.get_ion_diffusivity(self.cation.name,temperature),
                                     liquid.get_ion_diffusivity(self.anion.name,temperature))

    def __repr__(self)->str:
        return "DissolvedSalt("+self.name+", "+str(self.concentration)+")"


class SaltProperties(PureSolidProperties):
    r"""
    A salt, i.e. the pair of ions that "1 mM NaCl" means, together with the stoichiometry that makes
    the solution electroneutral.

    A salt is a solid, and a recipe: what it is *for* is to be dissolved, which is what multiplying
    it by a concentration does::

        mix = Mixture(water + 20*percent*glycerol + 1*milli*molar*get_salt("NaCl"))

    Subclasses name their two ions and are registered with
    :py:meth:`MaterialProperties.register`; :py:mod:`pyoomph.materials.ions` ships the common ones
    and :py:func:`get_salt` fetches them. The ions themselves are pulled from the ion library when
    the salt is constructed, so a salt cannot name an ion that does not exist.

    The stoichiometry is *derived* from the two charge numbers rather than parsed out of the name:
    :math:`\nu_+=|z_-|/g` and :math:`\nu_-=z_+/g` with :math:`g=\gcd(z_+,|z_-|)`, which is the only
    ratio that is electroneutral. For every standard salt this reproduces the formula the name
    already carries -- Na2SO4 comes out as 2:1 because sulfate is divalent, not because the name
    contains a 2.
    """
    #: Name of the cation in the ion library, e.g. ``"Na+"``.
    cation_name:str
    #: Name of the anion in the ion library, e.g. ``"Cl-"``.
    anion_name:str
    #: :math:`\mathrm{d}\sigma/\mathrm{d}c`, how much the surface tension of the solution rises per
    #: unit salt concentration. Salts are surface-*depleted* -- an ion near the surface loses part of
    #: its hydration shell and is pushed back by its image charge -- so this is positive for a salt
    #: and negative for the strong acids, whose protons do sit at the surface. Zero unless the salt
    #: below sets it, so that nothing invents a Marangoni stress out of missing data.
    surface_tension_increment:ExpressionOrNum=0

    def __init__(self):
        super().__init__()
        #: The cation, an :py:class:`IonProperties` from the library.
        self.cation=get_ion(self.cation_name)
        #: The anion.
        self.anion=get_ion(self.anion_name)
        if self.cation.charge_number<0 or self.anion.charge_number>0:
            raise ValueError("Salt '"+self.name+"' names '"+self.cation_name+"' as its cation and '"+
                             self.anion_name+"' as its anion, but their charge numbers are "+
                             str(self.cation.charge_number)+" and "+str(self.anion.charge_number))
        #: Number of cations per formula unit, e.g. 2 for Na2SO4.
        #: Number of anions per formula unit, e.g. 2 for CaCl2.
        self.cation_stoichiometry,self.anion_stoichiometry= \
            salt_stoichiometry(self.cation.charge_number,self.anion.charge_number)
        self.molar_mass=self.cation_stoichiometry*self.cation.molar_mass \
                        +self.anion_stoichiometry*self.anion.molar_mass

    def get_apparent_molar_volume(self)->ExpressionOrNum:
        r"""
        :math:`V_\phi=\nu_+V_+^\circ+\nu_-V_-^\circ`, the volume one mole of this salt adds to the
        solution.

        Additive over the ions to the accuracy of the tables -- NaCl's 16.62 cm^3/mol is
        -1.21 + 17.83 -- so it is stored per ion and combined here, the same way the ambipolar
        diffusivity is. It is what turns a salt concentration into a volume, hence into a mass
        fraction and a density.

        Negative values are ordinary rather than an error: a small, highly charged ion pulls the
        surrounding water in tighter than it displaces (electrostriction), so Na2SO4 adds 11.6
        cm^3/mol and MgSO4 takes 7.2 away.
        """
        return _apparent_molar_volume(self.cation,self.anion,self.cation_stoichiometry,
                                      self.anion_stoichiometry,self.name)

    def get_ambipolar_diffusivity(self,temperature:ExpressionOrNum=var("temperature"))->ExpressionOrNum:
        """The salt's diffusivity in water, see :py:func:`ambipolar_diffusivity`. The value in an
        actual solvent is :py:meth:`DissolvedSalt.get_diffusivity`."""
        return ambipolar_diffusivity(self.cation.charge_number,self.anion.charge_number,
                                     self.cation.get_diffusivity(temperature),
                                     self.anion.get_diffusivity(temperature))

    def dissolve_in(self,liquid:"BaseLiquidProperties",concentration:ExpressionOrNum)->None:
        """
        Dissolve this salt in a liquid at the given concentration *of the salt*, i.e. what
        :py:meth:`BaseLiquidProperties.add_salt` means by it.

        The ions are handed over as copies, so dissolving one salt object in two liquids does not
        give the two liquids the same ion objects -- the same isolation ``get_ion`` gives by handing
        out a fresh instance per call.
        """
        liquid._add_salt_from_ions(copy.copy(self.cation),copy.copy(self.anion),concentration,
                                   _salt=self)

    def __mul__(self,other:ExpressionOrNum)->"DissolvedSpeciesComponent":  #type:ignore[override]
        return DissolvedSpeciesComponent(self,other)

    def __rmul__(self,other:ExpressionOrNum)->"DissolvedSpeciesComponent":  #type:ignore[override]
        return DissolvedSpeciesComponent(self,other)

    def __repr__(self)->str:
        return "Salt("+self.name+": "+str(self.cation_stoichiometry)+" "+self.cation.name+" + "+ \
               str(self.anion_stoichiometry)+" "+self.anion.name+")"


class DissolvedSpeciesComponent:
    r"""
    A salt or a single ion at a given concentration, waiting for a solvent.

    This is what ``1*milli*molar*get_salt("NaCl")`` is, and it is deliberately *not* a
    :py:class:`MixtureDefinitionComponent`: the solvent components of a mixture are fractions that
    must sum to unity, while a dissolved species is a concentration and stays out of that
    bookkeeping entirely. Physically that is the dilute-solute assumption -- 1 mM NaCl is 6e-5 by
    mass, and pretending it displaces some of the water would be a bigger error than ignoring it.
    :py:meth:`mass_fraction_in` is there to check that assumption when it matters.
    """
    def __init__(self,species:"SaltProperties | IonProperties",concentration:ExpressionOrNum):
        # assert_dimensional_value lets a plain number through without ever looking at the required
        # unit, so a dimensionless factor has to be caught here or "0.2*NaCl" would silently become
        # a concentration of 0.2 mol/m^3.
        if isinstance(concentration,(float,int)):
            raise ValueError("Expected a dimensional quantity with unit "+str(mol/meter**3)+
                             " for how much "+species.name+" is dissolved, but got the plain "+
                             "number "+str(concentration)+". A salt is given by concentration, "+
                             "e.g. 1*milli*molar*"+species.name+".")
        _,_=assert_dimensional_value(concentration,required_unit=mol/meter**3)
        self.species=species
        self.concentration=concentration

    def dissolve_in(self,liquid:"BaseLiquidProperties")->None:
        """Put this species into the liquid's ion table."""
        if isinstance(self.species,SaltProperties):
            self.species.dissolve_in(liquid,self.concentration)
        else:
            liquid.add_ion(copy.copy(self.species),self.concentration)

    def mass_fraction_in(self,liquid:"BaseLiquidProperties",
                         temperature:ExpressionNumOrNone=None)->ExpressionOrNum:
        r""":math:`c M/\rho`, i.e. how much of the solution's mass this species actually is -- the
        quantity the dilute-solute assumption above says is negligible."""
        return self.concentration*self.species.molar_mass \
               /liquid.get_reference_mass_density(temperature)

    def __radd__(self,other:"MixtureDefinitionComponent | MixtureDefinitionComponents | MaterialProperties | DissolvedSpeciesComponent | Literal[0]")->"LiquidMixtureDefinitionComponents":
        if other==0:   # sum(...) starts at 0
            return LiquidMixtureDefinitionComponents([],dissolved=[self])
        if isinstance(other,DissolvedSpeciesComponent):
            return LiquidMixtureDefinitionComponents([],dissolved=[other,self])
        if isinstance(other,LiquidMixtureDefinitionComponents):
            return LiquidMixtureDefinitionComponents(other.lst,dissolved=other.dissolved+[self])
        if isinstance(other,LiquidMixtureDefinitionComponent):
            return LiquidMixtureDefinitionComponents([other],dissolved=[self])
        if isinstance(other,PureLiquidProperties):
            return LiquidMixtureDefinitionComponents([LiquidMixtureDefinitionComponent(other,None)],
                                                     dissolved=[self])
        raise RuntimeError("Cannot dissolve "+str(self)+" in "+str(other)+
                           ": only liquids and liquid mixtures can carry dissolved species")

    def __add__(self,other:"MixtureDefinitionComponent | MixtureDefinitionComponents | MaterialProperties | DissolvedSpeciesComponent")->"LiquidMixtureDefinitionComponents":
        res=self.__radd__(other)
        # __radd__ appends self last, which is right when self is on the right of the +. Here it is
        # on the left, and the order of the dissolved species is what get_ions() sorts anyway.
        return res

    def __repr__(self)->str:
        return "DissolvedSpecies("+str(self.concentration)+" of "+self.species.name+")"


class UNIFACPyoomphExpressionGenerator(UNIFACExpressionGeneratorBase):
    def get_molefrac_var(self,name:str) -> Expression:
        return var("molefrac_"+name)
    def get_temperature_in_kelvin(self) -> Expression:
        return var("temperature")/kelvin
    def pow(self,a:ExpressionOrNum,b:ExpressionOrNum) -> ExpressionOrNum:
        # A GlobalParameter has no ** of its own, and the exponent here can be one (e.g. continuing
        # in modified_volume_fraction_exponent). Wrapped it stays live in the expression.
        if isinstance(a,_pyoomph.GiNaC_GlobalParam):
            a=0+a
        if isinstance(b,_pyoomph.GiNaC_GlobalParam):
            b=0+b
        return a**b
    def subexpression(self,expr:ExpressionOrNum) -> ExpressionOrNum:
        #return expr
        return subexpression(expr)
    def ln(self,arg:ExpressionOrNum) -> ExpressionOrNum:
        return log(arg)
    def exp(self,arg:ExpressionOrNum) -> ExpressionOrNum:
        return exp(arg)


class MixtureLiquidProperties(BaseLiquidProperties,BaseMixedProperties):
    """
    Class to define liquid mixtures.

    Args:
        pure_props: Pure component properties, will be passed when mixing the gaseous mixture with the :py:func:`Mixture` function.
    """
    is_pure:bool | None=False
    # Narrower than the mixin's plain MaterialProperties: a liquid mixture is made of pure liquids
    pure_properties:dict[str,PureLiquidProperties] # type: ignore[assignment]

    def __init__(self,pure_props:dict[str,MaterialProperties]):
        BaseLiquidProperties.__init__(self)
        BaseMixedProperties.__init__(self,pure_props=pure_props)
        
        #: A dict holding the vapor pressures given by the name of each pure component. By default, it will be set to ideal Raoult's law.
        self.vapor_pressure_for:dict[str,ExpressionOrNum]={}
        self.set_vapor_pressure_by_raoults_law()
        self._latent_heat_of_evaporation:dict[str,ExpressionOrNum]={}

        self._output_properties=self._output_properties.copy()
        def make_lambda_for_vapor_pressure(k:str)->Callable[[MaterialProperties],ExpressionNumOrNone]:
            return lambda props: self.get_vapor_pressure_for(k)
        for k in sorted(self.components):
            self._output_properties["vapor_pressure_" + k] = make_lambda_for_vapor_pressure(k)
        def make_lambda_for_activity_coeff(k:str)->Callable[[MaterialProperties],ExpressionNumOrNone]:
            return lambda props: self.activity_coefficients.get(k)
        for k in sorted(self.components):
            self._output_properties["activity_coefficient_" + k] = make_lambda_for_activity_coeff(k)

        self._reaction_rates:dict[str,ExpressionOrNum]={}

        self._unifac_multi_return:UNIFACMultiReturnExpression | None=None
        self._unifac_model:str | None=None # Used UNIFAC parameter table

    def add_reaction_rate(self,dest:str,rate:ExpressionOrNum,**source_factors:float):
        if not dest in self.components:
            raise RuntimeError("Cannot define a reaction rate for component '"+str(dest)+"' since it is not in the mixture")
        old=self._reaction_rates.get(dest,0)
        self._reaction_rates[dest]=old+rate
        for source,factor in source_factors.items():
            self.add_reaction_rate(source,-factor*rate)        

    def clear_reaction_rate(self,dest:str):
        if dest in self._reaction_rates.keys():
            del self._reaction_rates[dest]

    def get_reaction_rate(self,field:str)->ExpressionOrNum:
        if field not in self._reaction_rates.keys():
            return 0
        else:
            return self._reaction_rates[field]

    def check_reaction_rates_for_consistency(self):
        addition:ExpressionOrNum=0
        for rate in self._reaction_rates.values():
            addition+=rate
        if not is_zero(addition):
            raise RuntimeError("The sum of all reaction rates is not zero, but "+str(addition))



    def set_latent_heat_of_evaporation(self,name:str,Lambda:ExpressionOrNum):
        """
        Sets the latent heat of a single component. By default, we just use the latent heat from the pure component.
        """
        self._latent_heat_of_evaporation[name]=Lambda

    def get_latent_heat_of_evaporation(self, name:str)->ExpressionNumOrNone:
        """
        Returns the latent heat of evaporation for a given component. Falls back to the pure component if not changed specifically via :py:meth:`set_latent_heat_of_evaporation`.
        """
        res=self._latent_heat_of_evaporation.get(name,None)
        if res:
            return res
        elif name in self.pure_properties.keys():
            pc=self.pure_properties[name]
            assert isinstance(pc,PureLiquidProperties)
            return pc.get_latent_heat_of_evaporation(name)
        else:
            raise RuntimeError("Cannot get the latent heat of evaporation for the absent component "+str(name))

    def get_vapor_pressure_for(self,name:str,pure:bool=False)->ExpressionNumOrNone:
        """Returns the vapor pressure of a component in the mixture. 

        Args:
            name: Name of the pure component in this mixture.
            pure: If set, it returns the vapor pressure of the pure component, i.e. in absence of all other components in this mixture.

        Returns:
            ExpressionNumOrNone: _description_
        """
        if not pure:
            return self.vapor_pressure_for.get(name,None)
        else:
            pc=self.pure_properties[name]
            assert isinstance(pc,PureLiquidProperties)
            return pc.get_vapor_pressure_for(name,pure=True)


    # use_multi_return uses multi-return expression instead of subexpressions
    # if it is and int, it will use multi-return for mixtures with #components>=use_multi_return
    # multi-return expressions are considerably faster in generating C code. However, they use finite differences for the Jacobain
    def set_activity_coefficients_by_unifac(self,model:str,set_vapor_pressures:bool=True,use_multi_return:bool | int=3):
        """
        Sets the activity coefficients by a UNIFAC model.

        Args:
            model: A particular UNIFAC model to use. By default, pyoomph has ``"AIOMFAC"``, ``"Original"`` UNIFAC and ``"Dortmund"`` modified UNIFAC implemented.
            set_vapor_pressures: Also set the vapor pressures using non-ideal Raoult's law.
            use_multi_return: Either a bool or a maximum number of components when to use multi-return expressions. By default, it uses multi-return for mixtures with 3 or more components. multi-return expressions are faster for code generation, but use finite differences for the Jacobian. They cannot be used in all contexts, e.g. for bifurcation tracking.
        """
        if isinstance(model,str): #type:ignore
            modelname=model
        else:
            raise RuntimeError("Cannot do this right now")
        self._unifac_model=model
        if self.get_ions():
            self._set_activity_coefficients_with_ions(modelname,use_multi_return)
            if set_vapor_pressures:
                self.set_vapor_pressure_by_raoults_law()
            return
        if use_multi_return==True or ((use_multi_return is not False) and len(self.components)>=use_multi_return):
            self._unifac_multi_return=UNIFACMultiReturnExpression(self,model)
            for cn in sorted(self.components):
                self.activity_coefficients[cn]=cast(Expression,self._unifac_multi_return.get_activity_coefficient(cn))
        else:
            server=ActivityModel.get_activity_model_by_name(modelname)

            unifac_components:dict[str,UNIFACMolecule]={cn:UNIFACMolecule(cn,server) for cn in sorted(self.components)}
            for cn in sorted(self.components):
                comp=self.pure_properties[cn]
                assert isinstance(comp,PureLiquidProperties)
                subgroups=comp._UNIFAC_groups[modelname] 
                if len(subgroups)==0:
                    raise RuntimeError("Component "+cn+" has no UNIFAC groups defined for model "+modelname)
                for sgn,amount in subgroups.items():
                    unifac_components[cn].add_subgroup(sgn,amount)
            unifac_mix=UNIFACMixture(*unifac_components.values())
            unifac_mix.set_expression_generator(UNIFACPyoomphExpressionGenerator())
            for cn in sorted(self.components):
                self.activity_coefficients[cn]=cast(Expression,unifac_mix.get_activity_coefficient_expression(cn))

        if set_vapor_pressures:
            self.set_vapor_pressure_by_raoults_law()

    def set_vapor_pressure_by_raoults_law(self):
        """
        Set the vapor pressures based on Raoult's law. Potentially set activity coefficients are considered.
        """
        for c in sorted(self.components):
            cpure=self.pure_properties[c]
            assert isinstance(cpure,PureLiquidProperties)
            p_pure=cpure.get_vapor_pressure_for(c)
            if p_pure is not None:
                gamma=self.activity_coefficients.get(c,1)
                self.vapor_pressure_for[c]=gamma*var("molefrac_"+c)*p_pure


class MixtureGasProperties(BaseGasProperties,BaseMixedProperties):
    """
    Class to define gas mixtures.

    Args:
        pure_props: Pure component properties, will be passed when mixing the gaseous mixture with the :py:func:`Mixture` function.
    """
    is_pure:bool | None=False
    # Narrower than the mixin's plain MaterialProperties: a gas mixture is made of pure gases
    pure_properties:dict[str,PureGasProperties] # type: ignore[assignment]

    def __init__(self,pure_props:dict[str,MaterialProperties]):
        BaseGasProperties.__init__(self)
        BaseMixedProperties.__init__(self,pure_props=pure_props)

    def mass_density_from_ideal_gas_law(self,pressure:ExpressionOrNum=var("absolute_pressure"),temperature:ExpressionOrNum=var("temperature")):
        """
        Returns the mass density by assuming the ideal gas law.
        Args:
            pressure: Either a constant pressure or, by default, a potentially varying pressure given by ``var("absolute_pressure")``.
            temperature: Either a constant temperature or, by default, a potentially varying temperature given by ``var("temperature")``.

        Returns:
            The mass density according to the ideal gas law.
        """
        gas_constant=8.3144598*joule/(mol*kelvin)
        molar_mass:ExpressionOrNum=0
        for n,pc in self.pure_properties.items():
            molar_mass+=var("molefrac_"+n)*pc.molar_mass
        return pressure *molar_mass/ (gas_constant * temperature)

    def set_mass_density_from_ideal_gas_law(self):
        """
        Sets the mass density by using :py:meth:`mass_density_from_ideal_gas_law`.
        """
        self.mass_density=self.mass_density_from_ideal_gas_law()


    
    def set_diffusion_coefficient_by_Fuller_eq(self, for_dilute_gas:str, dominant_gas:str | None=None):
        """
        Sets the diffusion coefficient by the Fuller equation. This is a simple approximation for the Fickian diffusion in gas mixtures. The equation is based on the diffusion volumes of the gases. See:
        
            * Fuller, E. N. and Giddings, J. C. 1965. J. Gas Chromatogr., 3: 222
            * Fuller, E. N., Ensley, K. and Giddings, J. C. 1969. J. Phys. Chem., 75: 3679
            * Fuller, E. N., Schettler, P. D. and Giddings, J. C. 1966. Ind. Eng. Chem., 58: 18
        """
        if for_dilute_gas not in self.components:
            raise RuntimeError("Cannot apply the Fuller equation for a non-present gas component " + str(for_dilute_gas))
        if len(self.components) == 2 and dominant_gas is None:
            lst = list(self.components)
            dominant_gas = lst[1] if for_dilute_gas == lst[0] else lst[0]
        elif len(self.components) > 2 and dominant_gas is None:
            raise RuntimeError(
                "Please provide th dominant_gas for ternary or higher gas systems to approximate the Fickian diffusion by the Fuller equation")
        elif dominant_gas not in self.components:
            raise RuntimeError("Cannot apply the Fuller equation for a non-present gas component " + str(dominant_gas))
        c1=self.pure_properties[for_dilute_gas]
        c2=self.pure_properties[for_dilute_gas]
        assert isinstance(c1,PureGasProperties)
        assert isinstance(c2,PureGasProperties)
        v1 = c1.diffusion_volume_for_Fuller_eq
        v2 = c2.diffusion_volume_for_Fuller_eq
        if v1 is None or v2 is None:
            raise RuntimeError(
                "Please set the diffusion_volume_for_Fuller_eq properties of the pure gas components for the Fuller equation")
        TK = var("temperature") / kelvin
        pAtm = var("absolute_pressure") / atm
        M1 = self.pure_properties[for_dilute_gas].molar_mass / (gram / mol)
        M2 = self.pure_properties[dominant_gas].molar_mass / (gram / mol)
        D = 1e-3 * TK ** (rational_num(7, 4)) * square_root(1 / M1 + 1 / M2) / (
                    (pAtm) * (v1 ** (1 / 3) + v2 ** (1 / 3)) ** 2)
        D = D * (centi * meter) ** 2 / second #type:ignore
        Dexpr=cast(Expression,D)
        if len(self.components) == 2:
            self.set_diffusion_coefficient(Dexpr)
        else:
            if self.passive_field != dominant_gas:
                raise RuntimeError("How to do it here in a good way?")
            self.set_diffusion_coefficient(for_dilute_gas, D)


##################



class LiquidGasInterfaceProperties(BaseInterfaceProperties):
    """
    A class representing the properties of a liquid-gas interface.
    
    Args:
        phaseA: Usually the liquid phase properties.
        phaseB: Usually the gas phase properties.
        surfactant_dict: A dictionary of surfactants and their initial concentrations.
    """
    typus="liquid_gas"
    #: The components of the liquid phase
    liquid_components:str | set[str] | None = None
    #: The components of the gas phase
    gas_components:str | set[str] | None = None
    #: The surfactants at the interface
    surfactants:set[str] | str | None = None
    def _sort_phases(self,sideA:AnyMaterialProperties,sideB:AnyMaterialProperties)->tuple[AnyMaterialProperties,AnyMaterialProperties]:
        if sideA.state_of_matter=="liquid" and sideB.state_of_matter=="gas":
            return sideA,sideB
        elif sideA.state_of_matter=="gas" and sideB.state_of_matter=="liquid":
            return sideB,sideA
        else:
            raise RuntimeError("This liquid-gas interface does not have a liquid and a gas side")

    #: Whether the salts dissolved in the liquid raise the surface tension of this interface, see
    #: :py:meth:`BaseLiquidProperties.get_salt_surface_tension_shift`. Switch it off for an interface
    #: whose own correlation already accounts for them.
    apply_salt_surface_tension_shift:bool=True
    # A property, so that it does not matter which subclass assigns the surface tension or when: the
    # salt contribution is added on the way out, once, to whatever is stored. Assigning still works
    # exactly as before, and reading before anything was assigned still raises (so hasattr is still
    # False), because the getter reaches for an attribute that is not there yet.
    @property
    def surface_tension(self)->ExpressionOrNum:  #type:ignore[override]
        r""":math:`\sigma` of this interface, including what the dissolved salts add to it."""
        sigma=self._surface_tension_without_salts
        if self.apply_salt_surface_tension_shift:
            sigma=sigma+self._liquid_phase.get_salt_surface_tension_shift()
        return sigma

    @surface_tension.setter
    def surface_tension(self,value:ExpressionOrNum)->None:
        self._surface_tension_without_salts=value

    def __init__(self,phaseA:AnyMaterialProperties,phaseB:AnyMaterialProperties,surfactant_dict:dict[SurfactantProperties,ExpressionOrNum]):
        from .mass_transfer import StandardMassTransferModelLiquidGas
        super(LiquidGasInterfaceProperties, self).__init__(phaseA,phaseB)
        assert isinstance(self._phaseA,(PureLiquidProperties,MixtureLiquidProperties))
        assert isinstance(self._phaseB,(PureGasProperties,MixtureGasProperties))
        self._liquid_phase:PureLiquidProperties | MixtureLiquidProperties=self._phaseA        
        self._gas_phase:PureGasProperties | MixtureGasProperties = self._phaseB
        self._surfactants=surfactant_dict.copy() if surfactant_dict is not None else {}
        if "gas" in self._liquid_phase.default_surface_tension.keys():
            sigm=self._liquid_phase.default_surface_tension.get("gas")
            if sigm is not None:
                self.surface_tension=sigm
        self._mass_transfer_model=StandardMassTransferModelLiquidGas(self._liquid_phase,self._gas_phase)
        #: The rate of surfactant adsorption and desorption, merged in a single expression per surfactant
        self.surfactant_adsorption_rate:dict[str,ExpressionOrNum]={}
        self._surface_diffusivity:dict[str,ExpressionOrNum]={}

    def get_surface_diffusivity(self,surfactant_name:str) -> ExpressionNumOrNone:
        """
        Returns the surface diffusivity of a surfactant.
        """
        if surfactant_name in self._surface_diffusivity:
            return self._surface_diffusivity[surfactant_name]

        for sp,_ in self._surfactants.items():
            if sp.name==surfactant_name:
                return sp.surface_diffusivity

        raise RuntimeError("Cannot get the surface_diffusivity of surfactant "+str(surfactant_name))

    def set_surface_diffusivity(self,surfactant_name:str,expr:ExpressionOrNum):
        """
        Sets the surface diffusivity of a surfactant.
        """
        if not surfactant_name in {S.name for S in self._surfactants.keys()}:
            raise RuntimeError("Cannot set the surface diffusivity of a non-present surfactant "+str(surfactant_name))
        self._surface_diffusivity[surfactant_name]=expr


    def get_liquid_properties(self) -> AnyLiquidProperties:
        """
        Returns the liquid properties.
        """
        return self._liquid_phase

    def get_gas_properties(self) -> AnyGasProperties:
        """
        Returns the gas properties.
        """
        return self._gas_phase

    def evaluate_at_initial_surfactant_concentrations(self,expr:ExpressionOrNum) -> ExpressionOrNum:
        """
        Evaluates an expression, e.g. the surface tension, at the initial surfactant concentrations.
        """
        if not isinstance(expr,Expression):
            return expr
        fields:dict[str,Expression]={}
        nondims:dict[str,Expression]={}
        for surf,conc in self._surfactants.items():
            if not isinstance(conc,_pyoomph.Expression):
                conc=_pyoomph.Expression(conc)
            fields["surfconc_"+surf.name]=conc
        fields["velocity"]=_pyoomph.Expression(0)
        fields["velocity_x"] = _pyoomph.Expression(0)
        fields["velocity_y"] = _pyoomph.Expression(0)
        fields["velocity_z"] = _pyoomph.Expression(0)
        return _pyoomph.GiNaC_subsfields(expr, fields, nondims, {})

    def get_mass_transfer_model(self) -> "MassTransferModelBase | None":
        """
        Returns the mass transfer model.
        """
        return self._mass_transfer_model

    def get_latent_heat_of(self,name:str) -> ExpressionOrNum:
        """
        Returns the latent heat of evaporation for a component.
        """
        res=self._latent_heats.get(name)
        if res is None:
            res=self._liquid_phase.get_latent_heat_of_evaporation(name)
            if res is None:
                raise RuntimeError("No latent heat set for "+name)
        return res

class DefaultLiquidGasInterface(LiquidGasInterfaceProperties):
    """
    Default liquid-gas interface properties, which just uses the default surface tension of the liquid phase against gas.

    Args:
        phaseA: The liquid phase properties.
        phaseB: The gas phase properties.
        surfactant_dict: A dictionary of surfactants and their initial concentrations.
    """
    def __init__(self,phaseA:AnyMaterialProperties,phaseB:AnyMaterialProperties,surfactant_dict:dict[SurfactantProperties,ExpressionOrNum]):
        super(DefaultLiquidGasInterface, self).__init__(phaseA,phaseB,surfactant_dict=surfactant_dict)
        if self._liquid_phase.default_surface_tension.get("gas") is None:
            raise RuntimeError("Either specify the interface properties of the liquid-gas interface of liquid:"+str(self._liquid_phase)+" vs. gas:"+str(self._gas_phase)+" or at least set the default surface tension against gas in the liquid properties")
        if self._liquid_phase.default_surface_tension["gas"] is None:
            raise RuntimeError("interface properties of the liquid-gas interface of liquid:"+str(self._liquid_phase)+" vs. gas:"+str(self._gas_phase)+". That's okay, if you at least provide a default surface tension against gas in the liquid phase")
        self.surface_tension=0+self._liquid_phase.default_surface_tension["gas"]

MaterialProperties.library["interfaces"]["_defaults"]["liquid_gas"]=DefaultLiquidGasInterface


class LiquidSolidInterfaceProperties(BaseInterfaceProperties):
    typus="liquid_solid"
    liquid_components:str | set[str] | None = None
    solid_components:str | set[str] | None = None
    surfactants:set[str] | str | None = None
    def _sort_phases(self,sideA:AnyMaterialProperties,sideB:AnyMaterialProperties)->tuple[AnyMaterialProperties,AnyMaterialProperties]:
        if sideA.state_of_matter=="liquid" and sideB.state_of_matter=="solid":
            return sideA,sideB
        elif sideA.state_of_matter=="solid" and sideB.state_of_matter=="liquid":
            return sideB,sideA
        else:
            raise RuntimeError("The liquid-solid interface does not have a liquid and a solid bulk side")

    def get_liquid_properties(self) -> PureLiquidProperties | MixtureLiquidProperties:
        return self._liquid_phase

    def get_solid_properties(self) -> PureSolidProperties:
        return self._solid_phase

    def __init__(self,phaseA:AnyMaterialProperties,phaseB:AnyMaterialProperties,surfactant_dict:dict[SurfactantProperties,ExpressionOrNum]):
        super(LiquidSolidInterfaceProperties, self).__init__(phaseA,phaseB)
        assert isinstance(self._phaseA,(PureLiquidProperties,MixtureLiquidProperties))
        assert isinstance(self._phaseB,PureSolidProperties)
        self._liquid_phase:PureLiquidProperties | MixtureLiquidProperties=self._phaseA        
        self._solid_phase:PureSolidProperties = self._phaseB        
        self._surfactants=surfactant_dict.copy() if surfactant_dict is not None else {}
        self.equilibrium_temperature=None
        self.latent_heat_of_fusion=None
        self.surfactant_adsorption_rate:dict[str,ExpressionOrNum]={}
        self._surface_diffusivity:dict[str,ExpressionOrNum]={}

    def get_surface_diffusivity(self,surfactant_name:str) -> ExpressionNumOrNone:
        if surfactant_name in self._surface_diffusivity:
            return self._surface_diffusivity[surfactant_name]
        return None

    def set_surface_diffusivity(self,surfactant_name:str,expr:ExpressionNumOrNone):
        if not surfactant_name in {S.name for S in self._surfactants.keys()}:
            raise RuntimeError("Cannot set the surface diffusivity of a non-present surfactant "+str(surfactant_name))
        if expr is None:
            if surfactant_name in self._surface_diffusivity.keys():
                del self._surface_diffusivity[surfactant_name]
        else:
            self._surface_diffusivity[surfactant_name]=expr

    def evaluate_at_initial_surfactant_concentrations(self,expr:ExpressionOrNum) -> ExpressionOrNum:
        if not isinstance(expr,Expression):
            return expr
        fields:dict[str,Expression]={}
        nondims:dict[str,Expression]={}
        for surf,conc in self._surfactants.items():
            if not isinstance(conc,_pyoomph.Expression):
                conc=_pyoomph.Expression(conc)
            fields["surfconc_"+surf.name]=conc
        fields["velocity"]=_pyoomph.Expression(0)
        fields["velocity_x"] = _pyoomph.Expression(0)
        fields["velocity_y"] = _pyoomph.Expression(0)
        fields["velocity_z"] = _pyoomph.Expression(0)
        return _pyoomph.GiNaC_subsfields(expr, fields, nondims, {})

class LiquidLiquidInterfaceProperties(BaseInterfaceProperties):
    typus="liquid_liquid"
    surfactants:set[str] | str | None = None
    componentsA:set[str] | str | None = set()
    componentsB:set[str] | str | None = set()

    def get_fraction_in_rich_phase(self,varname:str,rich_component:str | None=None,in_bulk:bool=False):
        if rich_component is None:
            rich_component=varname
        if self._phaseA.initial_condition[rich_component]>self._phaseB.initial_condition[rich_component]: #type:ignore
            return var(varname,domain=".." if in_bulk else ".")
        elif self._phaseA.initial_condition[rich_component]<self._phaseB.initial_condition[rich_component]: #type:ignore
            return var(varname,domain="|.." if in_bulk else "|.")
        else:
            raise RuntimeError("Cannot distinguish phases")
        
    def get_fraction_in_poor_phase(self,varname:str,poor_component:str | None=None,in_bulk:bool=False):
        if poor_component is None:
            poor_component=varname
        if self._phaseA.initial_condition[poor_component]<self._phaseB.initial_condition[poor_component]: #type:ignore
            return var(varname,domain=".." if in_bulk else ".")
        elif self._phaseA.initial_condition[poor_component]>self._phaseB.initial_condition[poor_component]: #type:ignore
            return var(varname,domain="|.." if in_bulk else "|.")
        else:
            raise RuntimeError("Cannot distinguish phases")

    def __init__(self,phaseA:AnyMaterialProperties,phaseB:AnyMaterialProperties,surfactant_dict:dict[SurfactantProperties,ExpressionOrNum]):
        super(LiquidLiquidInterfaceProperties, self).__init__(phaseA,phaseB)
        self._surfactants=surfactant_dict.copy() if surfactant_dict is not None else {}
        self.surfactant_adsorption_rate:dict[str,ExpressionOrNum]={}
        self._mass_transfer_model=None

    def get_mass_transfer_model(self) -> "MassTransferModelBase | None":
        return self._mass_transfer_model

    def evaluate_at_initial_surfactant_concentrations(self,expr:ExpressionOrNum) -> ExpressionOrNum:
        if not isinstance(expr,Expression):
            return expr
        fields:dict[str,Expression]={}
        nondims:dict[str,Expression]={}
        for surf,conc in self._surfactants.items():
            if not isinstance(conc,_pyoomph.Expression):
                conc=_pyoomph.Expression(conc)
            fields["surfconc_"+surf.name]=conc
        fields["velocity"]=_pyoomph.Expression(0)
        fields["velocity_x"] = _pyoomph.Expression(0)
        fields["velocity_y"] = _pyoomph.Expression(0)
        fields["velocity_z"] = _pyoomph.Expression(0)
        return _pyoomph.GiNaC_subsfields(expr, fields, nondims, {})
##################

#Can take multiple names
@overload
def get_pure_material(state_of_matter:str,name:str,return_class:Literal[False]=...)->MaterialProperties: ...

@overload
def get_pure_material(state_of_matter:str,name:str,return_class:Literal[True])->type[MaterialProperties]: ...

@overload
def get_pure_material(state_of_matter:str,name:list[str],return_class:Literal[False]=...)->tuple[MaterialProperties,...]: ...

@overload
def get_pure_material(state_of_matter:str,name:list[str],return_class:Literal[True])->tuple[type[MaterialProperties],...]: ...


def get_pure_material(state_of_matter:str,name:str | list[str],return_class:bool=False)->MaterialProperties | type[MaterialProperties] | tuple[MaterialProperties, ...] | tuple[type[MaterialProperties], ...]:
    if isinstance(name,(list,tuple)):
        res:list[MaterialProperties]=[]
        for n in name:
            res.append(cast(MaterialProperties,get_pure_material(state_of_matter,n,return_class))) #type:ignore
        return tuple(res)
    else:
        if not name in MaterialProperties.library[state_of_matter]["pure"].keys():
            table=MaterialProperties.library[state_of_matter]["pure"]
            # The ions are pure liquids as well, but listing all of them here buries the handful of
            # solvents the user is actually looking for.
            plain=sorted(n for n,c in table.items()
                         if not (isinstance(c,type) and issubclass(c,(IonProperties,SaltProperties))))
            print("Available pure " + state_of_matter + " components: " + str(plain))
            if len(plain)<len(table):
                print("(plus " + str(len(table)-len(plain)) + " registered ions/salts, see "
                      + "get_ion() and get_salt())")
            raise RuntimeError(
                "Cannot find any materials named '" + name + "' and in state '" + state_of_matter + "'. Make sure to import the corresponding python file, where these component is defined or define it yourself. For examples, please have a look at " + os.path.realpath(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), "default_materials.py")))
        if return_class:
            return MaterialProperties.library[state_of_matter]["pure"][name]
        else:
            return MaterialProperties.library[state_of_matter]["pure"][name]()
    

#Can take multiple names
@overload
def get_pure_liquid(name:str,return_class:Literal[False]=...)->PureLiquidProperties: ...

@overload
def get_pure_liquid(name:str,return_class:Literal[True])->type[PureLiquidProperties]: ...

@overload
def get_pure_liquid(name:list[str],return_class:Literal[False]=...)->tuple[PureLiquidProperties,...]: ...

@overload
def get_pure_liquid(name:list[str],return_class:Literal[True])->tuple[type[PureLiquidProperties],...]: ...

def get_pure_liquid(name:str | list[str],return_class:bool=False)->PureLiquidProperties | type[PureLiquidProperties] | tuple[PureLiquidProperties, ...] | tuple[type[PureLiquidProperties], ...]:
    """
    Returns the pure liquid properties for the given name(s) from the material library. Property classes must be decorated with the decorator :py:meth:`MaterialProperties.register` before this works.

    Args:
        name: Name of the pure liquid component(s) to be returned.
        return_class: Return the class instead of an instance of the class.

    Returns:
        The generated pure liquid properties as object(s) or class(es).
    """
    res=get_pure_material("liquid",name,return_class) #type:ignore
    return res #type:ignore

@overload
def get_surfactant(name:str,return_class:Literal[False]=...)->SurfactantProperties: ...

@overload
def get_surfactant(name:str,return_class:Literal[True])->type[SurfactantProperties]: ...

@overload
def get_surfactant(name:list[str],return_class:Literal[False]=...)->tuple[SurfactantProperties,...]: ...

@overload
def get_surfactant(name:list[str],return_class:Literal[True])->tuple[type[SurfactantProperties],...]: ...

def get_surfactant(name:str | list[str],return_class:bool=False)->SurfactantProperties | type[SurfactantProperties] | tuple[SurfactantProperties, ...] | tuple[type[SurfactantProperties], ...]:
    """
    Returns the surfactant properties for the given name(s) from the material library. Property classes must be decorated with the decorator :py:meth:`MaterialProperties.register` before this works.

    Args:
        name: Name of the surfactant properties to be returned.
        return_class: Return the class instead of an instance of the class.

    Returns:
        The generated surfactant properties as object(s) or class(es).
    """
    res=get_pure_liquid(name,return_class) #type:ignore
    if not isinstance(res,(tuple)):
        if not isinstance(res,SurfactantProperties):
            raise RuntimeError(str(name)+" is not a surfactant, but a normal pure liquid")
        return res
    else:
        for r in res: #type:ignore
            if not isinstance(r,SurfactantProperties):
                raise RuntimeError(str(r)+" is not a surfactant, but a normal pure liquid") #type:ignore
        return res #type:ignore

def _registered_ions()->"dict[str,type[IonProperties]]":
    """Everything in the pure-liquid table that is an ion, keyed by name."""
    return {n:c for n,c in MaterialProperties.library["liquid"]["pure"].items()
            if isinstance(c,type) and issubclass(c,IonProperties)}


def _lookup_registered_ion(name:str)->"IonProperties | None":
    """The registered ion of that name, or None if no material of that name is registered at all.

    Separate from :py:func:`get_ion` because :py:meth:`BaseLiquidProperties.add_ion` has to tell
    "not registered" (create it from the arguments) from "registered, but not an ion" (a mistake).
    """
    table=MaterialProperties.library["liquid"]["pure"]
    if name not in table:
        return None
    cls=table[name]
    if not (isinstance(cls,type) and issubclass(cls,IonProperties)):
        raise RuntimeError("'"+name+"' is a pure liquid, not an ion")
    return cls()


@overload
def get_ion(name:str,return_class:Literal[False]=...)->IonProperties: ...

@overload
def get_ion(name:str,return_class:Literal[True])->type[IonProperties]: ...

@overload
def get_ion(name:list[str],return_class:Literal[False]=...)->tuple[IonProperties,...]: ...

@overload
def get_ion(name:list[str],return_class:Literal[True])->tuple[type[IonProperties],...]: ...

def get_ion(name:str | list[str],return_class:bool=False)->IonProperties | type[IonProperties] | tuple[IonProperties, ...] | tuple[type[IonProperties], ...]:
    """
    Returns the ionic species for the given name(s) from the material library, in the same way
    :py:func:`get_pure_liquid` and :py:func:`get_surfactant` do. Ion classes must be decorated with
    :py:meth:`MaterialProperties.register` before this works -- import
    :py:mod:`pyoomph.materials.ions` for the standard library, or declare your own with
    :py:func:`new_ion`.

    Like all these getters, this hands out a **new instance** per call, so dissolving it in one
    liquid does not affect another.

    Args:
        name: Name of the ion(s), e.g. ``"Na+"``.
        return_class: Return the class instead of an instance of the class.

    Returns:
        The ion properties as object(s) or class(es).
    """
    if isinstance(name,(list,tuple)):
        return tuple(cast("IonProperties",get_ion(n,return_class)) for n in name) #type:ignore
    _lookup_registered_ion(name)  # raises the specific error if the name is a non-ion material
    known=_registered_ions()
    if name not in known:
        raise RuntimeError("Cannot find any ion named '"+name+"'. Registered ions: "+
                           str(sorted(known.keys()))+". The standard ions are registered by "+
                           "importing pyoomph.materials.ions; a new one is declared with new_ion().")
    cls=known[name]
    return cls if return_class else cls()


def _registered_salts()->"dict[str,type[SaltProperties]]":
    """Everything in the pure-solid table that is a salt, keyed by name."""
    return {n:c for n,c in MaterialProperties.library["solid"]["pure"].items()
            if isinstance(c,type) and issubclass(c,SaltProperties)}


@overload
def get_salt(name:str,return_class:Literal[False]=...)->SaltProperties: ...

@overload
def get_salt(name:str,return_class:Literal[True])->type[SaltProperties]: ...

@overload
def get_salt(name:list[str],return_class:Literal[False]=...)->tuple[SaltProperties,...]: ...

@overload
def get_salt(name:list[str],return_class:Literal[True])->tuple[type[SaltProperties],...]: ...

def get_salt(name:str | list[str],return_class:bool=False)->SaltProperties | type[SaltProperties] | tuple[SaltProperties, ...] | tuple[type[SaltProperties], ...]:
    """
    Returns the salt of the given name(s) from the material library, in the same way
    :py:func:`get_pure_liquid` and :py:func:`get_ion` do. Import :py:mod:`pyoomph.materials.ions`
    for the standard library, or register your own :py:class:`SaltProperties` subclass.

    Constructing a salt pulls its two ions out of the ion library, so what you get back already
    knows their valences, diffusivities and molar masses. Multiply it by a concentration to dissolve
    it::

        mix = Mixture(water + 20*percent*glycerol + 1*milli*molar*get_salt("NaCl"))

    Args:
        name: Name of the salt(s), e.g. ``"NaCl"``.
        return_class: Return the class instead of an instance of the class.

    Returns:
        The salt properties as object(s) or class(es).
    """
    if isinstance(name,(list,tuple)):
        return tuple(cast("SaltProperties",get_salt(n,return_class)) for n in name) #type:ignore
    known=_registered_salts()
    if name not in known:
        raise RuntimeError("Cannot find any salt named '"+name+"'. Registered salts: "+
                           str(sorted(known.keys()))+". The standard salts are registered by "+
                           "importing pyoomph.materials.ions.")
    cls=known[name]
    return cls if return_class else cls()


@overload
def get_pure_gas(name:str,return_class:Literal[False]=...)->PureGasProperties: ...

@overload
def get_pure_gas(name:str,return_class:Literal[True])->type[PureGasProperties]: ...

@overload
def get_pure_gas(name:list[str],return_class:Literal[False]=...)->tuple[PureGasProperties,...]: ...

@overload
def get_pure_gas(name:list[str],return_class:Literal[True])->tuple[type[PureGasProperties],...]: ...

def get_pure_gas(name:str | list[str],return_class:bool=False)->PureGasProperties | type[PureGasProperties] | tuple[PureGasProperties, ...] | tuple[type[PureGasProperties], ...]:
    """
    Returns the pure gas properties for the given name(s) from the material library. Property classes must be decorated with the decorator :py:meth:`MaterialProperties.register` before this works.

    Args:
        name: Name of the pure gas component(s) to be returned.
        return_class: Return the class instead of an instance of the class.

    Returns:
        The generated pure gas properties as object(s) or class(es).
    """
    return get_pure_material("gas",name,return_class) #type:ignore


@overload
def get_pure_solid(name:str,return_class:Literal[False]=...)->PureSolidProperties: ...

@overload
def get_pure_solid(name:str,return_class:Literal[True])->type[PureSolidProperties]: ...

@overload
def get_pure_solid(name:list[str],return_class:Literal[False]=...)->tuple[PureSolidProperties,...]: ...

@overload
def get_pure_solid(name:list[str],return_class:Literal[True])->tuple[type[PureSolidProperties],...]: ...

def get_pure_solid(name:str | list[str],return_class:bool=False)->PureSolidProperties | type[PureSolidProperties] | tuple[PureSolidProperties, ...] | tuple[type[PureSolidProperties], ...]:
    """
    Returns the pure solid properties for the given name(s) from the material library. Property classes must be decorated with the decorator :py:meth:`MaterialProperties.register` before this works.

    Args:
        name: Name of the pure solid component(s) to be returned.
        return_class: Return the class instead of an instance of the class.

    Returns:
        The generated pure solid properties as object(s) or class(es).
    """
    return get_pure_material("solid",name,return_class) #type:ignore


#Takes a list of components
def get_mixture_properties(*purecompos:MaterialProperties,**kwargs:Any)->MaterialProperties | MixtureGasProperties | MixtureLiquidProperties:
    if len(purecompos)==1:
        return purecompos[0]	#Pure material
    som=None
    comps:set[str]=set()
    pureprops:dict[str,MaterialProperties]={}
    for c in purecompos:
        if som is None:
            som=c.state_of_matter
        else:
            if som!=c.state_of_matter:
                raise ValueError("Tried to mix components of different states: "+str(som)+" and "+str(c.state_of_matter))
        if c.name in pureprops.keys():
            if pureprops[c.name]!=c:
                raise ValueError("Tried to mix components with the same name, but different properties: "+c.name)
        pureprops[c.name]=c
        comps.add(c.name)
    frz=frozenset(comps)
    if som is None:
        raise RuntimeError("Should not happen")
    cls=MaterialProperties.library[som]["mixed"].get(frz)
    if cls is None:
        raise KeyError("Mixture properties of mixture from " + str(sorted(comps)) + " in state " + som + " not defined")
    if kwargs.get("return_class",False):
        return cls
    else:
        return cls(pureprops)


@overload
def get_interface_properties(phaseA:PureLiquidProperties | MixtureLiquidProperties,phaseB:PureGasProperties | MixtureGasProperties,surfactants:str | SurfactantProperties | dict[str, ExpressionOrNum] | dict[SurfactantProperties, ExpressionOrNum] | None=None)->LiquidGasInterfaceProperties: ...

@overload
def get_interface_properties(phaseA:PureLiquidProperties | MixtureLiquidProperties,phaseB:PureSolidProperties,surfactants:str | SurfactantProperties | dict[str, ExpressionOrNum] | dict[SurfactantProperties, ExpressionOrNum] | None=None)->LiquidSolidInterfaceProperties: ...

@overload
def get_interface_properties(phaseA:MaterialProperties | MixtureDefinitionComponents,phaseB:MaterialProperties | MixtureDefinitionComponents,surfactants:str | SurfactantProperties | dict[str, ExpressionOrNum] | dict[SurfactantProperties, ExpressionOrNum] | None=None)->BaseInterfaceProperties: ...

def get_interface_properties(phaseA:MaterialProperties | MixtureDefinitionComponents,phaseB:MaterialProperties | MixtureDefinitionComponents,surfactants:str | SurfactantProperties | dict[str, ExpressionOrNum] | dict[SurfactantProperties, ExpressionOrNum] | None=None)->BaseInterfaceProperties | LiquidGasInterfaceProperties:
    """
    Returns the interface properties for the two given phases (and potentially surfactants at the interface) from the material library. Property classes must be decorated with the decorator :py:meth:`MaterialProperties.register` before this works.

    Args:
        phaseA: Inner phase material
        phaseB: Outer phase material
        surfactants: Potential surfactants on the interface.

    Returns:
        The interface properties from the material library.
    """
    typus=None
    surfactantsN:dict[SurfactantProperties,ExpressionOrNum]={}
    if surfactants is None:
        #TODO: Auto extract the surfactants from the liquid!
        pass
    elif not isinstance(surfactants,dict):
        if isinstance(surfactants,str):
            surfactantsN={get_surfactant(surfactants): 0}
        elif isinstance(surfactants,SurfactantProperties): #type:ignore
            surfactantsN={surfactants : 0}
    else:
        for surfactant,amount in surfactants.items():
            if isinstance(surfactant,str):
                surfactant=get_surfactant(surfactant)
            surfactantsN[surfactant]=amount


    if isinstance(phaseA,MixtureDefinitionComponents):
        phaseA=Mixture(phaseA)
    if isinstance(phaseB,MixtureDefinitionComponents):
        phaseB=Mixture(phaseB)

    
    if phaseA.state_of_matter=="liquid" and phaseB.state_of_matter=="gas":
        typus = "liquid_gas"
        liquid,gas=phaseA,phaseB
        solid=None
    elif phaseB.state_of_matter=="liquid" and phaseA.state_of_matter=="gas":
        typus = "liquid_gas"
        liquid,gas=phaseB,phaseA
        solid=None
    elif phaseB.state_of_matter=="liquid" and phaseA.state_of_matter=="solid":
        typus = "liquid_solid"
        liquid,solid=phaseB,phaseA
        gas=None
    elif phaseA.state_of_matter=="liquid" and phaseB.state_of_matter=="solid":
        typus = "liquid_solid"
        liquid,solid=phaseA,phaseB
        gas=None
    elif phaseA.state_of_matter=="liquid" and phaseB.state_of_matter=="liquid":
        typus = "liquid_liquid"
        solid=None
        gas=None       
        liquid=None 
    else:
        raise RuntimeError("Implement interface selection for states of matter "+str(phaseA.state_of_matter)+" and "+str(phaseB.state_of_matter))

    if typus=="liquid_gas":
        assert isinstance(liquid,(PureLiquidProperties,MixtureLiquidProperties))
        assert isinstance(gas,(PureGasProperties,MixtureGasProperties))
        lcomps=frozenset(liquid.components)
        gcomps=frozenset(gas.components)
        scomps=frozenset({s.name for s in surfactantsN.keys()}) #TODO Surfactants
        key=(lcomps,gcomps,scomps,)
        if key  in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][key](liquid,gas,surfactantsN)
        key = (lcomps, frozenset(cast(set[str],set())), frozenset(scomps))
        if key in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][key](liquid, gas, surfactantsN)
        if len(scomps)>0:
            raise RuntimeError("Cannot find a liquid-gas interface definition between liquid "+str(sorted(lcomps))+" and gas "+str(sorted(gcomps))+" with surfactants "+str(sorted(scomps)))
        #key=(lcomps,frozenset(set()),frozenset(set()))
        #if key  in MaterialProperties.library["interfaces"][typus].keys():
        #    return MaterialProperties.library["interfaces"][typus][key](liquid,gas,surfactants)
    elif typus=="liquid_solid":
        assert isinstance(liquid,(PureLiquidProperties,MixtureLiquidProperties))
        assert isinstance(solid,PureSolidProperties)
        lcomps = frozenset(liquid.components)
        solcomps = frozenset(solid.components)
        scomps=frozenset({s.name for s in surfactantsN.keys()}) #TODO Surfactants
        #print("IN LS",lcomps,solcomps,scomps)        
        #print(MaterialProperties.library["interfaces"][typus])
        key = (lcomps, solcomps, scomps,)
        #print(key,"IN",MaterialProperties.library["interfaces"][typus].keys())
        if key in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][key](liquid, solid, surfactantsN)
        key = (lcomps, frozenset(cast(set[str],set())), frozenset(solcomps))
        if key in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][key](liquid, solid, surfactantsN)
        key = (lcomps, frozenset(cast(set[str],set())), frozenset(cast(set[str],set())))
        if key in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][key](liquid, solid, surfactantsN)
    elif typus=="liquid_liquid":
        assert isinstance(phaseA,(PureLiquidProperties,MixtureLiquidProperties))
        assert isinstance(phaseB,(PureLiquidProperties,MixtureLiquidProperties))
        Acomps = frozenset(phaseA.components)
        Bcomps = frozenset(phaseB.components)
        scomps = frozenset({s.name for s in surfactantsN.keys()})  # TODO Surfactants
        # As in register(): a liquid-liquid interface is keyed by the unordered pair of its sides
        ll_key = (frozenset({Acomps, Bcomps}), scomps,)
        if ll_key in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][ll_key](phaseA, phaseB, surfactantsN)
    else:
        raise RuntimeError("Implement")
    #print("PHASE A",phaseA)
    #print("PHASE B",phaseB)
    #exit()
    #MaterialProperties.library["interfaces"][typus]

    if typus in MaterialProperties.library["interfaces"]["_defaults"].keys():
        return MaterialProperties.library["interfaces"]["_defaults"][typus](phaseA,phaseB,surfactantsN)
    else:
        n1=phaseA.name if phaseA.is_pure else "["+", ".join(sorted(phaseA.components))+"]"
        n2 = phaseB.name if phaseB.is_pure else "[" + ", ".join(sorted(phaseB.components)) + "]"
        if len(surfactantsN)==0:
            raise RuntimeError("Cannot find an interface of type "+typus+" for "+n1+" | "+n2)
        else:
            raise RuntimeError("Cannot find an interface of type "+typus+" for "+n1+" | "+n2+" and the surfactants "+str({s.name for s in surfactantsN}))

@overload
def Mixture(mdef:LiquidMixtureDefinitionComponents | LiquidMixtureDefinitionComponent | PureLiquidProperties,temperature:ExpressionNumOrNone=...,quantity:MixQuantityDefinition=...,pressure:ExpressionOrNum=...)->AnyLiquidProperties: ...

@overload
def Mixture(mdef:GasMixtureDefinitionComponents | GasMixtureDefinitionComponent | PureGasProperties,temperature:ExpressionNumOrNone=...,quantity:MixQuantityDefinition=...,pressure:ExpressionOrNum=...)->AnyGasProperties: ...

@overload
def Mixture(mdef:MixtureDefinitionComponents | MixtureDefinitionComponent | AnyMaterialProperties,temperature:ExpressionNumOrNone=...,quantity:MixQuantityDefinition=...,pressure:ExpressionOrNum=...)->MaterialProperties: ...

def Mixture(mdef:MixtureDefinitionComponents | MixtureDefinitionComponent | AnyMaterialProperties,temperature:ExpressionNumOrNone=None,quantity:MixQuantityDefinition="mass_fraction",pressure:ExpressionOrNum=1*atm,salt_treatment:Literal["dilute","component"]="dilute")->AnyMaterialProperties:
    """
    Returns a gas or liquid mixture from the given mixture definition components or a single material properties object.

    Args:
        mdef: Either a pure substance or a mixture like ``get_pure_liquid("water")+0.5*get_pure_liquid("ethanol")``.
            Salts and ions may be added by *concentration* -- ``+ 1*milli*molar*get_salt("NaCl")`` --
            in which case they are dissolved in the finished mixture rather than counted among the
            fractions, see :py:class:`DissolvedSpeciesComponent`.
        salt_treatment: ``"dilute"`` (the default) leaves a dissolved salt out of the mass fractions,
            which is right to a few percent up to about half molar. ``"component"`` makes it a
            component of its own, see
            :py:meth:`BaseLiquidProperties.treat_salts_as_components`; a single solvent then becomes
            a one-component mixture, since a pure liquid cannot host components.
        temperature: The temperature of the mixture. Used for potential initial conditions and required if you want to use e.g. volume fractions or relative humidity as mixture quantity.
        quantity: Specifies the quantity definition of the mixture. Can be either ``"mass_fraction"``, ``"volume_fraction"``, ``"molar_fraction"``, or ``"relative_humidity"``.
        pressure: Absolute pressure. Necessary for particular conversions.

    Returns:
        The properties of the mixture from the material library.
    """
    if isinstance(mdef,MixtureDefinitionComponents):
        if not mdef.lst:
            raise RuntimeError("A dissolved species needs a solvent: "+str(mdef.dissolved))
        res,init=mdef.finalise(quantity,temperature=temperature,pressure=pressure)
        props=get_mixture_properties(*tuple(res))
        for e,k in init.items():
            props.initial_condition["massfrac_"+e]=k
        if temperature is not None:
            props.initial_condition["temperature"]=temperature
        # Salts and ions last: they are concentrations, so they take no part in the fractions above
        # and only need the solvent to exist.
        if mdef.dissolved and salt_treatment=="component" and getattr(props,"is_pure",False):
            # A salt that is a composition field makes even a single solvent a mixture.
            props=pure_liquid_as_mixture(cast(PureLiquidProperties,props))
        if mdef.dissolved:
            if any(props is c for c in res):
                # A one-component "mixture" is the very object that was passed in, so dissolving
                # here would put the ions into the caller's own `water` -- and the next Mixture
                # built from that same object would inherit them. Seen as a KCl solution reporting
                # the Ca2+ of an unrelated mixture built one line earlier.
                props=_with_its_own_ion_table(props)
            for d in mdef.dissolved:
                d.dissolve_in(cast(BaseLiquidProperties,props))
            if salt_treatment=="component":
                cast(BaseLiquidProperties,props).treat_salts_as_components()
        elif salt_treatment=="component":
            raise RuntimeError("salt_treatment='component' but nothing is dissolved in this mixture.")
        return props
    elif isinstance(mdef,DissolvedSpeciesComponent):
        raise RuntimeError("A dissolved species needs a solvent: "+str(mdef)+
                           ". Write e.g. Mixture(water+"+str(mdef.concentration)+"*"+
                           mdef.species.name+") instead.")
    elif isinstance(mdef,LiquidMixtureDefinitionComponent): 
        return Mixture(LiquidMixtureDefinitionComponents([mdef]),temperature=temperature,quantity=quantity,pressure=pressure)
    elif isinstance(mdef,GasMixtureDefinitionComponent): 
        return Mixture(GasMixtureDefinitionComponents([mdef]),temperature=temperature,quantity=quantity,pressure=pressure)
    elif isinstance(mdef,LiquidMixtureDefinitionComponent):
        return Mixture(mdef.get_compo(),temperature=temperature,quantity=quantity,pressure=pressure)
    elif isinstance(mdef,GasMixtureDefinitionComponent):
        return Mixture(mdef.get_compo(),temperature=temperature,quantity=quantity,pressure=pressure)
    elif isinstance(mdef,PureLiquidProperties):
        return Mixture(mdef*1,temperature=temperature,pressure=pressure)
    elif isinstance(mdef,PureGasProperties):
        return Mixture(mdef*1,temperature=temperature,pressure=pressure)    
    else:
        raise RuntimeError("Handle this case"+str(mdef))



def _with_its_own_ion_table(props:"AnyMaterialProperties")->"AnyMaterialProperties":
    """A copy of a material that does not share the containers a dissolved species writes into.

    Shallow otherwise: everything else about the material is read-only as far as dissolving is
    concerned, and a deep copy of a material full of GiNaC expressions is neither cheap nor needed.
    """
    res=copy.copy(props)
    for attr in ("_ion_table","_bulk_concentrations","_salt_table","initial_condition"):
        if hasattr(res,attr):
            setattr(res,attr,copy.copy(getattr(res,attr)))
    return res


def new_pure_liquid(name:str,mass_density:ExpressionOrNum=1000*kilogram/meter**3,dynamic_viscosity:ExpressionOrNum=1*milli*pascal*second,surface_tension:ExpressionOrNum=70*milli*newton/meter,molar_mass:ExpressionOrNum=50*gram/mol,override:bool=False,thermal_conductivity:ExpressionNumOrNone=None,specific_heat_capacity:ExpressionNumOrNone=None,latent_heat:ExpressionNumOrNone=None,vapor_pressure:ExpressionNumOrNone=None) -> PureLiquidProperties:
    """
    Shortcut to create new pure liquid with the specified properties.

    Args:
        name: The name of the pure liquid material.
        mass_density: The mass density of the pure liquid material.
        dynamic_viscosity: The dynamic viscosity of the pure liquid material.
        surface_tension: The surface tension of the pure liquid material.
        molar_mass: The molar mass of the pure liquid material. 
        override: Whether to override existing material properties with the same name.
        thermal_conductivity: The thermal conductivity of the pure liquid material. 
        specific_heat_capacity: The specific heat capacity of the pure liquid material.
        latent_heat: The latent heat of evaporation of the pure liquid material.
        vapor_pressure: The vapor pressure of the pure liquid material. 

    Returns:
        An instance of the new added pure liquid material, which is also registered in the material library.
    """
    _name=name
    @MaterialProperties.register(override=override)
    class CustomPureLiquid(PureLiquidProperties):   #type:ignore     
        name=_name
        def __init__(self):
            super().__init__()
            self.molar_mass=molar_mass
            self.mass_density=mass_density
            self.dynamic_viscosity=dynamic_viscosity
            self.default_surface_tension["gas"]=surface_tension
            if thermal_conductivity is not None:
                self.thermal_conductivity=thermal_conductivity
            if specific_heat_capacity is not None:
                self.specific_heat_capacity=specific_heat_capacity
            if latent_heat is not None:
                self.latent_heat_of_evaporation=latent_heat
            if vapor_pressure is not None:
                self.vapor_pressure=vapor_pressure        
    return get_pure_liquid(_name)


def new_ion(name:str,charge_number:int,diffusivity:ExpressionNumOrNone=None,*,
            limiting_molar_conductivity:ExpressionNumOrNone=None,
            molar_mass:ExpressionOrNum=50*gram/mol,
            mass_density:ExpressionOrNum=1000*kilogram/meter**3,
            dynamic_viscosity:ExpressionOrNum=1*milli*pascal*second,
            override:bool=False)->IonProperties:
    r"""
    Shortcut to create and register a new ionic species, i.e. to put one into the same table that
    :py:mod:`pyoomph.materials.ions` fills and :py:func:`get_ion` reads.

    Give either a ``diffusivity`` or a ``limiting_molar_conductivity``; the latter is converted by
    Nernst-Einstein, :math:`D_i=\lambda_i^0RT/(z_i^2F^2)`, at
    :py:meth:`IonProperties.get_diffusivity`'s temperature -- which is ``var("temperature")`` unless
    told otherwise, so an isothermal problem has to define one, e.g. with
    ``self.define_named_var(temperature=25*celsius)``.

    Args:
        name: Name of the ion, e.g. ``"Na+"``. Also the stem of its concentration field.
        charge_number: The charge number :math:`z_i`, e.g. +1 for Na+ and -2 for SO4(2-).
        diffusivity: Diffusivity at infinite dilution.
        limiting_molar_conductivity: Alternative to the diffusivity.
        molar_mass: Molar mass, only relevant if the ions also carry mass in a flow model.
        mass_density: Mass density, inherited from the pure-liquid machinery and rarely meaningful
            for a dissolved ion.
        dynamic_viscosity: Likewise.
        override: Whether to override an existing material of the same name.

    Returns:
        The registered ion.
    """
    if charge_number==0:
        raise ValueError("An ion with charge_number 0 is a neutral solute, not an ion")
    if diffusivity is None and limiting_molar_conductivity is None:
        raise ValueError("new_ion needs either a diffusivity or a limiting_molar_conductivity")
    _name,_z,_D,_lam=name,charge_number,diffusivity,limiting_molar_conductivity
    @MaterialProperties.register(override=override)
    class CustomIon(IonProperties):   #type:ignore
        name=_name
        def __init__(self):
            super().__init__()
            self.charge_number=_z
            self.diffusivity=_D
            self.limiting_molar_conductivity=_lam
            self.molar_mass=molar_mass
            self.mass_density=mass_density
            self.dynamic_viscosity=dynamic_viscosity
    res=get_pure_liquid(_name)
    assert isinstance(res,IonProperties)
    return res


def new_pure_gas(name:str,mass_density:ExpressionOrNum=1000*kilogram/meter**3,dynamic_viscosity:ExpressionOrNum=1*milli*pascal*second,molar_mass:ExpressionOrNum=50*gram/mol,override:bool=False,thermal_conductivity:ExpressionNumOrNone=None,specific_heat_capacity:ExpressionNumOrNone=None) -> PureGasProperties:
    """
    Shortcut to create new pure gas with the specified properties.

    Args:
        name: The name of the pure gas material.
        mass_density: The mass density of the pure gas material.
        dynamic_viscosity: The dynamic viscosity of the pure gas material.
        molar_mass: The molar mass of the pure gas material. 
        override: Whether to override existing material properties with the same name.
        thermal_conductivity: The thermal conductivity of the pure liquid material. 
        specific_heat_capacity: The specific heat capacity of the pure liquid material.

    Returns:
        An instance of the new added pure gas material, which is also registered in the material library.
    """
    _name=name
    @MaterialProperties.register(override=override)
    class CustomPureLiquid(PureGasProperties):   #type:ignore     
        name=_name
        def __init__(self):
            super().__init__()
            self.molar_mass=molar_mass
            self.mass_density=mass_density
            self.dynamic_viscosity=dynamic_viscosity
            if thermal_conductivity is not None:
                self.thermal_conductivity=thermal_conductivity
            if specific_heat_capacity is not None:
                self.specific_heat_capacity=specific_heat_capacity
    return get_pure_gas(_name)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
