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
 
import _pyoomph
import os
import itertools
from pathlib import Path
from collections import OrderedDict as OrderDict

from ..expressions import var,grad,subexpression,exp,log,rational_num,square_root,is_zero
from ..expressions import ExpressionOrNum,ExpressionNumOrNone,Expression
from ..expressions.units import *
from .activity import *


from ..typings import *

if TYPE_CHECKING:
    from ..generic.problem import Problem
    from .mass_transfer import MassTransferModelBase

import math

MixQuantityDefinition=Literal["mass_fraction","wt","mole_fraction","volume_fraction","relative_humidity","RH"]
AnyMaterialProperties=Union["MaterialProperties", "BaseLiquidProperties", "BaseGasProperties", "BaseSolidProperties", "PureSolidProperties", "PureLiquidProperties", "PureGasProperties", "MixtureLiquidProperties","MixtureGasProperties"]
OutputPropertiesType=Dict[str,Optional[Callable[["MaterialProperties"],ExpressionNumOrNone]]]
DefaultSurfaceTensionType=Dict[Literal["gas","solid","liquid"],ExpressionNumOrNone]
PropertySampleRangeType=Union[ExpressionOrNum,ArrayWithUnits,List[ExpressionOrNum]]

AnyFluidProperties=Union["PureLiquidProperties", "PureGasProperties", "MixtureLiquidProperties","MixtureGasProperties"]
AnyLiquidProperties=Union["PureLiquidProperties", "MixtureLiquidProperties"]
AnyGasProperties=Union["PureGasProperties", "MixtureGasProperties"]
AnyFluidFluidInterface=Union["LiquidGasInterfaceProperties","LiquidLiquidInterfaceProperties"]

_TypeMaterialProperties=TypeVar("_TypeMaterialProperties",bound=Union[Type["MaterialProperties"],Type["BaseInterfaceProperties"]])


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
    def _sort_phases(self,sideA:AnyMaterialProperties,sideB:AnyMaterialProperties)->Tuple[AnyMaterialProperties,AnyMaterialProperties]:
        return sideA,sideB
    def __init__(self,sideA:Union[AnyMaterialProperties,"MixtureDefinitionComponents"],sideB:Union[AnyMaterialProperties,"MixtureDefinitionComponents"]):
        from .mass_transfer import MassTransferModelBase
        if isinstance(sideA,MixtureDefinitionComponents):
            sideA=Mixture(sideA)
        if isinstance(sideB,MixtureDefinitionComponents):
            sideB=Mixture(sideB)
        self._phaseA,self._phaseB=self._sort_phases(sideA,sideB)
        #: The surface tension of this interface
        self.surface_tension:ExpressionOrNum
        #: The mass transfer model to use for this interface
        self._mass_transfer_model:Optional[MassTransferModelBase]=None
        self._surfactant_table={}
        self._latent_heats:Dict[str,ExpressionOrNum]={}

    def set_latent_heat_of(self,name:str,lat_heat:ExpressionOrNum):
        self._latent_heats[name]=lat_heat

    def get_latent_heat_of(self,name:str)->ExpressionOrNum:
        res=self._latent_heats.get(name)
        if res is None:
            raise RuntimeError("No latent heat set for "+name)
        return res

    def set_mass_transfer_model(self,mdl:Optional["MassTransferModelBase"]) -> None:
        """
        Sets the mass transfer model.
        """
        self._mass_transfer_model=mdl
######

class MaterialProperties:
    """
    Base class for all material properties. This class should not be instantiated directly, but rather one of its subclasses should be used.
    However, this class allows to register new material properties and interfaces using the ``@MaterialProperties.register()`` decorator (see :py:meth:`register`) on the definition of the subclass.        
    """
    #: Unique name of the material. Names should be unique within the same state of matter (e.g. liquid, gas, solid), and the same material name for the same material should be used for different states of matter.
    name:str
    #: Whether the material is pure or mixed. If the material is mixed, the components of the mixture should be specified in the :py:attr:`components` attribute. This should be treated as read-only property.
    is_pure:Optional[bool]
    #: State of matter of the material. This should be treated as read-only property.
    state_of_matter:Optional[str]
    #: In case of a mixture, the components of the mixture. Should be set as class-variable in the subclass.
    components:Set[str]=set()
    
    library:Dict[str,Dict[str,Any]]={"gas":{"pure":{},"mixed":{}},"solid":{"pure":{},"mixed":{}},"liquid":{"pure":{},"mixed":{}},"interfaces":{"liquid_gas":{},"liquid_solid":{},"_defaults":{},"liquid_liquid":{}}}
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
                    liq_compos=subclass.liquid_components
                    if liq_compos is None:
                        raise  RuntimeError("To register liquid-gas interfaces, you must set the information liquid_components")
                    elif isinstance(liq_compos,str):
                        liq_compos=frozenset({liq_compos})
                    else:
                        liq_compos=frozenset(liq_compos)
                    gas_compos=subclass.gas_components
                    if gas_compos is None:
                        gas_compos=cast(Set[str],set())
                    elif isinstance(gas_compos,str):
                        gas_compos={gas_compos}
                    gas_compos=frozenset(gas_compos)
                    surfacts=subclass.surfactants
                    if surfacts is None:
                        surfacts=cast(Set[str],set())
                    elif isinstance(surfacts,str):
                        surfacts = {surfacts}
                    surfacts=frozenset(surfacts)
                    entry=(liq_compos,gas_compos,surfacts)

                    if entry in table.keys() and not override:
                        raise RuntimeError("There is already an liquid-gas interface property defined for "+str(entry)+". Please use override=True to register and override.")
                    table[entry]=subclass
                elif  issubclass(subclass,LiquidSolidInterfaceProperties):
                    table = cls.library["interfaces"]["liquid_solid"]
                    liq_compos = subclass.liquid_components
                    if liq_compos is None:
                        raise RuntimeError(
                            "To register liquid-solid interfaces, you must set the information liquid_components")
                    else:
                        liq_compos = frozenset(liq_compos)
                    solid_compos = subclass.solid_components
                    if solid_compos is None:
                        solid_compos = cast(Set[str],set())
                    elif isinstance(solid_compos,str):
                        solid_compos={solid_compos}
                    solid_compos = frozenset(solid_compos)
                    surfacts = subclass.surfactants
                    if surfacts is None:
                        surfacts=cast(Set[str],set())
                    elif isinstance(surfacts,str):
                        surfacts = {surfacts}
                    surfacts=frozenset(surfacts)
                    entry = (liq_compos, solid_compos, surfacts)
                    if entry in table.keys() and not override:
                        raise RuntimeError("There is already an liquid-solid interface property defined for " + str(
                            entry) + ". Please use override=True to register and override.")
                    table[entry] = subclass
                elif  issubclass(subclass,LiquidLiquidInterfaceProperties):
                    table = cls.library["interfaces"]["liquid_liquid"]
                    compsA = subclass.componentsA
                    compsB = subclass.componentsB
                    if compsA is None or compsB is None:
                        raise RuntimeError(
                            "To register liquid-liquid interfaces, you must set the information componentsA and componentsB")
                    else:
                        compsA = frozenset(compsA)
                        compsB = frozenset(compsB)
                    surfacts = subclass.surfactants
                    if surfacts is None:
                        surfacts = cast(Set[str],set())
                    surfacts = frozenset(surfacts)
                    entry = (frozenset({compsA, compsB}), surfacts)

                    if entry in table.keys() and not override:
                        raise RuntimeError("There is already an liquid-liquid interface property defined for " + str(
                            entry) + ". Please use override=True to register and override.")
                    table[entry] = subclass
                else:
                    raise RuntimeError("TODO: Register other interfaces Interface")
                return subclass
            if not hasattr(subclass, 'state_of_matter') or subclass.state_of_matter is None:
                raise RuntimeError("Cannot register material '"+subclass.__name__+"', since it does not have a state_of_matter. Please define "+subclass.__name__+".state_of_matter=...")
            if not hasattr(subclass,"is_pure") or subclass.is_pure is None:
                raise RuntimeError("Bulk material properites must have is_pure set to True or False")
            if subclass.is_pure:
                if not hasattr(subclass, 'name'):
                    raise RuntimeError("Cannot register pure "+str(subclass.state_of_matter)+" material '"+subclass.__name__+"', since it does not have a name. Please define "+subclass.__name__+".name=\"...\"")

                if subclass.name in cls.library[subclass.state_of_matter]["pure"].keys():
                    if not override:
                        raise RuntimeError("You tried to register the pure "+subclass.state_of_matter+" material named '"+subclass.name+"', but there is already one defined. Please either use another name or add override=True to the arguments of @MaterialProperties.register(override=True)")
                cls.library[subclass.state_of_matter]["pure"][subclass.name]=subclass
            else:
                if not hasattr(subclass, 'components'):
                    raise RuntimeError("Cannot register mixed "+subclass.state_of_matter+" material '"+subclass.__name__+"', since it does not have a list of pure components. Please define "+subclass.__name__+".components={...}")
                if type(subclass.components)!=set:
                    raise RuntimeError("Cannot register mixed "+subclass.state_of_matter+" material '"+subclass.__name__+"', since the list of pure components needs to be a set. Please define "+subclass.__name__+".components={...}")

                frz=frozenset(subclass.components)
                if frz in cls.library[subclass.state_of_matter]["mixed"].keys():
                    if not override:
                        raise RuntimeError("You tried to register the mixed "+subclass.state_of_matter+" material with components '"+str(subclass.components)+"', but there is already one defined. Please add override=True to the arguments of @MaterialProperties.register(override=True)")
                cls.library[subclass.state_of_matter]["mixed"][frz]=subclass

    #      cls.subclasses[message_type] = subclass
            return subclass
        return decorator


    def generate_field_substs(self,cond:Dict[str,ExpressionOrNum])->Tuple[Dict[str,Expression],Dict[str,Expression]]:
        fields:Dict[str,Expression]={}
        defined_massfracs:Dict[str,ExpressionOrNum]={}
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
    
    def evaluate_at_condition(self,expr:Union[ExpressionOrNum,str],cond:Union[Dict[str,ExpressionOrNum],Literal["initial","IC","initial_condition"]]={},*,temperature:ExpressionNumOrNone=None,**kwargs:ExpressionNumOrNone) -> Expression:
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
        if cond in ["initial","IC","initial_condition"]:
            cond=self.initial_condition
        mycond=cond.copy()
        for i,j in kwargs.items():
            if j is not None:
                mycond[i]=j
        if temperature is not None:
            mycond["temperature"]=temperature
        fields,nondims=self.generate_field_substs(mycond)
        remkeys:Set[str]=set()
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
        return _pyoomph.GiNaC_subsfields(expr,cast(Dict[str,_pyoomph.Expression],fields),cast(Dict[str,_pyoomph.Expression],nondims),{}) #type:ignore #TODO Global params

    def simplify_property_expressions(self,*property_names:str,**variables:ExpressionOrNum):
        for name in property_names:
            if hasattr(self,name):
                setattr(self,name,self.evaluate_at_condition(getattr(self,name),variables))
            else:
                raise RuntimeError(str(self)+" has no property "+str(name))

    def __init__(self):
        #: The initial condition of the material. Will be set automatically when using e.g. the :py:func:`Mixture` function to assemble a mixture of pure components.
        self.initial_condition:Dict[str,ExpressionOrNum]={}
        #: The mass density of the material. 
        self.mass_density:ExpressionOrNum#=None
        #: The specific heat capacity of the material.
        self.specific_heat_capacity:ExpressionOrNum# = None
        #: The thermal conductivity of the material.
        self.thermal_conductivity:ExpressionOrNum# = None
        #: The molecular weight of the material, used to calculate the mole fractions from the mass fractions.
        self.molar_mass:ExpressionOrNum#=None


    def __mul__(self,other:Union[float,int,Expression])->"MixtureDefinitionComponent":
        return MixtureDefinitionComponent(self,other)

    def __rmul__(self,other:Union[float,int,Expression])->"MixtureDefinitionComponent":
        return MixtureDefinitionComponent(self,other)

    def __or__(self,other:AnyMaterialProperties)->Union[BaseInterfaceProperties,'LiquidLiquidInterfaceProperties','LiquidGasInterfaceProperties','LiquidSolidInterfaceProperties']:
        if isinstance(other,MaterialProperties): #type:ignore
            return get_interface_properties(self,other)
        elif isinstance(other,MixtureDefinitionComponents) or isinstance(other,MixtureDefinitionComponent):
            raise RuntimeError("Please finalize a mixture of pure components with a Mixture(...) call: "+str(other))

    def evaluate_at_multiple_params(self,expr:Union[ExpressionOrNum,str],_sort:str="len",**kwargs:PropertySampleRangeType)->Tuple[List[ExpressionOrNum],ExpressionOrNum,OrderedDict[str,ArrayWithUnits],Dict[str,ExpressionOrNum]]:
        if isinstance(expr,str):
            if hasattr(self,expr):
                expr=getattr(self,expr)
            else:
                raise RuntimeError("Cannot find the property "+str(expr)+" in "+str(self))
        expr=cast(ExpressionOrNum,expr)
        vari_ranges:List[ArrayWithUnits]=[]
        vari_names:List[str]=[]
        first_cond:Dict[str,ExpressionOrNum]={}
        second_cond:Dict[str,ExpressionOrNum]={}
        consts:Dict[str,ExpressionOrNum]={}
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

        if _sort=="len" or _sort=="len_rev":
            sorti=[len(l) for l in vari_ranges]
        elif _sort=="name" or _sort=="name_rev":
            sorti=[n for n in vari_names]
        else:
            raise ValueError("_sort may only have the values len, len_rev, name, name_rev")
        inds:List[int]=[i for i,_ in sorted(enumerate(sorti), key = lambda x: x[1])] 
        if _sort=="len_rev" or _sort=="name_rev":
            inds=list(reversed(inds))
        vari_names=[vari_names[i] for i in inds]
        vari_ranges=[vari_ranges[i] for i in inds]

        result:List[ExpressionOrNum]=[]
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
            cond:Dict[str,ExpressionOrNum]={vari_names[i]:vrs[i] for i in range(len(vari_names))} #type:ignore
            cond.update(consts) 
            result.append(float(self.evaluate_at_condition(dimless_expr, cond=cond)))

        rangs:Dict[str,ArrayWithUnits]=OrderDict()
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
            if v is None:
                v=k
            elif callable(v):
                v=v(self)
                if v is None:
                    continue
            print("Sampling property "+k+" to text file...")
            self.sample_property_to_text_file(os.path.join(dirname,k+".txt"),v,_name=k,_sort=_sort,_newlines=_newlines,**kwargs)


    def sample_property_to_text_file(self,fname:str,expr:Union[str,ExpressionOrNum],_name:Optional[str]=None,_sort:str="len",_newlines:bool=True,**kwargs:PropertySampleRangeType):
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

    def __radd__(self,other:Union["MixtureDefinitionComponent",MaterialProperties])->"LiquidMixtureDefinitionComponents":
        if other==0:
            return self # This allows to use e.g. sum(massfrac[c]*component[c] for c in ...)
        elif isinstance(other,LiquidMixtureDefinitionComponent):
            return LiquidMixtureDefinitionComponents([self,other])
        elif isinstance(other,PureLiquidProperties): 
            return self+LiquidMixtureDefinitionComponent(other,None)
        else:
            raise RuntimeError("Tried to mix a liquid with something else:"+str(self)+" and "+str(other))

    def __add__(self,other:Union["MixtureDefinitionComponent",MaterialProperties])->"LiquidMixtureDefinitionComponents":
        return self.__radd__(other)

    def get_compo(self)->"PureLiquidProperties":
        assert isinstance(self.compo,PureLiquidProperties)
        return self.compo

class GasMixtureDefinitionComponent(MixtureDefinitionComponent):
    def __init__(self, compo: MaterialProperties, quant: ExpressionNumOrNone):
        super().__init__(compo, quant)

    def __radd__(self,other:Union["MixtureDefinitionComponent",MaterialProperties])->"GasMixtureDefinitionComponents":
        if isinstance(other,GasMixtureDefinitionComponent):
            return GasMixtureDefinitionComponents([self,other])
        elif isinstance(other,PureGasProperties): 
            return self+GasMixtureDefinitionComponent(other,None)
        else:
            raise RuntimeError("Tried to mix a gas with something else:"+str(self)+" and "+str(other))

    def __add__(self,other:Union["MixtureDefinitionComponent",MaterialProperties])->"GasMixtureDefinitionComponents":
        return self.__radd__(other)

    def get_compo(self)->"PureGasProperties":
        assert isinstance(self.compo,PureGasProperties)
        return self.compo

class MixtureDefinitionComponents():
    def __init__(self,lst:List[MixtureDefinitionComponent]):
        self.lst=lst

    def __add__(self,other:Union["MixtureDefinitionComponents",MixtureDefinitionComponent,MaterialProperties])->"MixtureDefinitionComponents":
        if isinstance(other,MixtureDefinitionComponents):
            return MixtureDefinitionComponents(self.lst+other.lst)
        elif isinstance(other,MixtureDefinitionComponent):
            return MixtureDefinitionComponents(self.lst+[other])
        elif isinstance(other,MaterialProperties): #type:ignore
            return self+MixtureDefinitionComponent(other,None)

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__, self.lst)

    def finalise(self,quantity:MixQuantityDefinition="mass_fraction",temperature:ExpressionNumOrNone=None,pressure:ExpressionNumOrNone=1*atm) -> Tuple[Set[MaterialProperties], Dict[str, ExpressionOrNum]]:
        #if len(self.lst)==1:
        #    return {self.lst[0].compo},1
        if quantity=="RH":
            quantity="relative_humidity"
        elif quantity=="wt":
            quantity="mass_fraction"
        comps = set([e.compo for e in self.lst])

        if (temperature is not None) and not (isinstance(temperature,(float,int))):
            _,_=assert_dimensional_value(temperature,required_unit=kelvin)

        total=0
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
                    Pvap_rel=(pure_liquid.evaluate_at_condition(pure_liquid.vapor_pressure,cnds))/pressure
                    try:
                        Pvap_rel=float(Pvap_rel)
                    except:
                        raise RuntimeError("Cannot case the relative vapor pressure to a float, most likely since you have not set any temperature when specifying the Mixture(...,temperature=...):\n"+str(Pvap_rel))
                    assert e.quant is not None
                    e.quant*=Pvap_rel
            quantity="mole_fraction"
            total:ExpressionOrNum = 0
            for e in self.lst:
                if e==hasNone:
                    continue
                else:
                    assert e.quant is not None
                    total = total + e.quant


        eps=1e-6
        must_sum_to_unity=(quantity=="mass_fraction" or quantity=="mole_fraction" or quantity=="volume_fraction")


        total=float(total)
        if must_sum_to_unity and total>1+eps:
            raise ValueError("The total fractions of the mixture exceed unity: "+quantity+"  "+str(self.lst))
        if hasNone is not None:
            if must_sum_to_unity:
                hasNone.quant=1-total
        elif must_sum_to_unity and total<1-eps:
            raise ValueError("The total fractions of the mixture are less than unity")

        init:Dict[str,ExpressionOrNum]
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
            molar_denom=sum([props.pure_properties[c].molar_mass*init[c] for c in props.components])
            for c in props.components:
                init[c]*=props.pure_properties[c].molar_mass/molar_denom
                init[c]=float(init[c])
        elif quantity=="volume_fraction":
            init = {c.name: 0.0 for c in comps}
            for e in self.lst:
                assert e.quant is not None
                init[e.compo.name] += e.quant
            props = get_mixture_properties(*comps)
            assert isinstance(props,(MixtureGasProperties,MixtureLiquidProperties))
            rhos = {c: props.pure_properties[c].evaluate_at_condition(props.pure_properties[c].mass_density,temperature=temperature) for c in props.components}
            for _, rho in rhos.items():
                assert_dimensional_value(rho)
            denom = sum([rhos[c] * init[c] for c in props.components])
            for c in props.components:
                init[c]*=rhos[c]/denom
                init[c]=float(init[c])
        else:
            raise ValueError("quantity=... may only take 'mass_fraction'/'wt', 'mole_fraction', 'volume_fraction' and 'relative_humidity'/'RH'.")

        return comps,init


class LiquidMixtureDefinitionComponents(MixtureDefinitionComponents):
    def __init__(self, lst: List[MixtureDefinitionComponent]):
        super().__init__(lst)
        for a in lst:
            if not isinstance(a,LiquidMixtureDefinitionComponent):
                RuntimeError("You tried to mix a gas with something else: "+str(self)+" contains "+str(a))

    def __add__(self,other:Union["MixtureDefinitionComponents",MixtureDefinitionComponent,MaterialProperties])->"LiquidMixtureDefinitionComponents":
        if isinstance(other,LiquidMixtureDefinitionComponents):
            return LiquidMixtureDefinitionComponents(self.lst+other.lst)
        elif isinstance(other,LiquidMixtureDefinitionComponent):
            return LiquidMixtureDefinitionComponents(self.lst+[other])
        elif isinstance(other,PureLiquidProperties):
            return self+LiquidMixtureDefinitionComponent(other,None)
        else:
            raise RuntimeError("You tried to mix a liquid with something else: "+str(self)+" and "+str(other))

class GasMixtureDefinitionComponents(MixtureDefinitionComponents):
    def __init__(self, lst: List[MixtureDefinitionComponent]):
        super().__init__(lst)        
        for a in lst:
            if not isinstance(a,GasMixtureDefinitionComponent):
                RuntimeError("You tried to mix a gas with something else: "+str(self)+" contains "+str(a))

    def __add__(self,other:Union["MixtureDefinitionComponents",MixtureDefinitionComponent,MaterialProperties])->"GasMixtureDefinitionComponents":
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
    passive_field=None
    required_adv_diff_fields:Set[str]=set()
    possible_properties:Set[str]={"mass_density","dynamic_viscosity","default_surface_tension"}
    _output_properties:OutputPropertiesType={"mass_density":None,"dynamic_viscosity":None,"default_surface_tension_gas":lambda self : cast(DefaultSurfaceTensionType,self.default_surface_tension).get("gas")} #type:ignore
    def __init__(self):
        super(BaseLiquidProperties, self).__init__()
        #: Default surface tension of the liquid. This is a dictionary with the keys ``"gas"``, ``"solid"``, and ``"liquid"``. The value for each key is the surface tension of the liquid with the respective other phase. 
        self.default_surface_tension:DefaultSurfaceTensionType={"gas":None}
        #: The dynamic viscosity of the liquid.
        self.dynamic_viscosity:ExpressionOrNum#=None

    def get_reference_dynamic_viscosity(self,temperature:Optional[ExpressionOrNum]=None) -> Expression:
        ics=self.initial_condition.copy()
        if temperature is not None:
            ics["temperature"]=temperature
        return self.evaluate_at_condition(self.dynamic_viscosity,ics)

    def get_reference_mass_density(self,temperature:Optional[ExpressionOrNum]=None) -> Expression:
        ics=self.initial_condition.copy()
        if temperature is not None:
            ics["temperature"]=temperature
        return self.evaluate_at_condition(self.mass_density,ics)


    def set_reference_scaling_to_problem(self, problem: "Problem", temperature: Optional[ExpressionOrNum] = None, **kwargs: ExpressionOrNum):
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

    def get_vapor_mass_concentration(self,component:str,relative_humidity_for_far_field:ExpressionNumOrNone=None,temperature:ExpressionNumOrNone=None,at_mixture_composition:Union[bool,Dict[str,ExpressionOrNum]]=True):
        """
        Calculates the saturation vapor concentration :math:`c_{sat}` for the given component in [kg/m^3].
        If relative_humidity_for_far_field is set, it does not apply Raoult's law, but uses the relative humidity to calculate the vapor concentration in the far field
        
        Args:
            component: Name of the component.
            relative_humidity_for_far_field: Relative humidity in the far field. If set, it is used to calculate the vapor concentration of the pure vapor in the far field.
            temperature: Temperature at which to calculate the vapor concentration. If not set, the temperature from the initial condition is used.
            at_mixture_composition: If set to ``True``, the vapor concentration is calculated at the mixture initial composition. If set to a dictionary, the vapor concentration is calculated at the given composition.
        """
        gas_constant=8.3144598*joule/(mol*kelvin)
        M=self.get_pure_component(component).molar_mass
        temperature_set=temperature
        if temperature is None:
            temperature=var("temperature")
        if relative_humidity_for_far_field is not None:
            psat=relative_humidity_for_far_field*self.get_vapor_pressure_for(component,pure=True)
        else:
            psat=self.get_vapor_pressure_for(component)

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
    passive_field=None
    required_adv_diff_fields:Set[str]=set()
    possible_properties:Set[str]={"mass_density","dynamic_viscosity"}
    _output_properties = {"mass_density": None, "dynamic_viscosity": None}
    def __init__(self):
        super(BaseGasProperties, self).__init__()
        self.mass_density:ExpressionOrNum
        #: The dynamic viscosity of the gas.
        self.dynamic_viscosity:ExpressionOrNum

class BaseSolidProperties(MaterialProperties):
    """
    A base class for defining solid materials.
    """
    state_of_matter="solid"
    passive_field=None
    required_adv_diff_fields:Set[str]=set()
    possible_properties:Set[str]={"mass_density"}
    _output_properties = {"mass_density": None}


class BaseMixedProperties:
    """
    A base class used for defining mixtures of pure components.
    """
    name:str
    components:Set[str] = set()
    def __init__(self,pure_props:Dict[str,MaterialProperties]):

        self.pure_properties=pure_props
        self.is_static=False #You can make a mixture static, i.e. remove all mass fractions fields from it
        assert hasattr(self,"passive_field")
        #: The passive field of the mixture. This is the field for which a advective-diffusive equation is not solved, since we can calculate it from the mass fractions of the other components.
        self.passive_field:Optional[str]=getattr(self,"passive_field")
        if self.passive_field is None:	#Select one passive field
            for a in reversed(sorted(self.components)):
                self.passive_field=a
                break
        assert hasattr(self,"_output_properties")
        self._output_properties=cast(OutputPropertiesType,self._output_properties).copy()
        def make_diffusion_coeff_lambda(k1:str,k2:str)->Callable[[MaterialProperties],ExpressionNumOrNone]:
            return lambda self: self.get_diffusion_coefficient(k1, k2) #type:ignore
        for k1 in self.components:
            if k1==self.passive_field:
                continue
            for k2 in self.components:
                if k2 == self.passive_field:
                    continue
                self._output_properties["diffusivity_"+k1+"__"+k2]=make_diffusion_coeff_lambda(k1,k2)

        self.required_adv_diff_fields=self.components-{self.passive_field}
        if self.components!=set(self.pure_properties.keys()):
            raise ValueError("Cannot create a mixture with the components "+str(self.components)+" by passing the wrong pure component properties: "+str(self.pure_properties))
        self._diffusion_table:Dict[Tuple[str,str],ExpressionOrNum]={}


    def set_passive_field(self,passive_component:str):
        self.passive_field=passive_component
        self.required_adv_diff_fields=self.components-{self.passive_field}
    
    @overload
    def get_pure_component(self,name:str,raise_error:Literal[False]=...)->Optional[MaterialProperties]: ...
    @overload
    def get_pure_component(self,name:str,raise_error:Literal[True]=...)->MaterialProperties: ...

    def get_pure_component(self,name:str,raise_error:bool=False)->Optional[MaterialProperties]:
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

    def set_diffusion_coefficient(self, arg1: Union[ExpressionOrNum, str], arg2: Union[ExpressionNumOrNone, str] = None, arg3: ExpressionNumOrNone = None):
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
            for c in self.components:
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

    def get_diffusion_coefficient(self,n1:str,n2:Optional[str]=None,default:Optional[ExpressionNumOrNone]=None)->ExpressionNumOrNone:
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
    def set_thermophoresis_coefficient(self,for_component:Union[str,Iterable[str]],coeff:ExpressionOrNum):        
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
        if n==self.passive_field:
            res=0
            for c in self.components:
                if c!=self.passive_field:
                    res-=self.get_diffusive_mass_flux_for(c)
            return res

        res:ExpressionOrNum = 0
        for fn2 in self.components:
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
            raise ValueError("Mass fraction '"+name+"' is not in the components: "+str(self.components))
        return var("massfrac_"+name,**kwargs)

    def get_mole_fraction_field(self,name:str,**kwargs:Any)->Expression:
        """
        Returns the mole fraction field for the given component.
        """
        if not self.pure_properties[name]:
            raise ValueError("Mass fraction '"+name+"' is not in the components: "+str(self.components))
        return var("molefrac_"+name,**kwargs)


    def make_static(self,cond:Optional[Dict[str,ExpressionOrNum]]=None,temperature:ExpressionNumOrNone=None):	#TODO Make a copy!
        """
        This will make the mixture static, i.e. all mass fraction fields will be replaced by their values from the given condition. This is useful for to remove advection-diffusion equations if the composition stays homogeneous.

        Args:
            cond: Optional condition, otherwise the :py:attr:`initial_condition` is used.
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
                dct:Union[ExpressionOrNum,Dict[str,ExpressionOrNum]] = getattr(self, p)
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


    def set_by_weighted_average(self,what:Optional[str]=None,fraction_type:str="mass_fraction"):
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
                for c in self.pure_properties.values():
                    if not hasattr(c,p):
                        good=False
                        break
                if good:
                    self.set_by_weighted_average(p)
        else:
            if fraction_type!="mass_fraction" and fraction_type!="mole_fraction":
                raise ValueError("Can only use fraction_type='mass_fraction' or 'mole_fraction' at the moment")
            res=0
            for c,v in self.pure_properties.items():
                if not hasattr(v,what) or (getattr(v,what) is None):
                    raise ValueError("Mixture component "+c+" has no property "+what+" defined to take the average for the mixture")
                pure_prop=getattr(v,what)
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
    is_pure:bool=True

    def make_static(self,*args:Any,**kwargs:Any):
        return self

    def __init__(self):
        super().__init__()
        self.initial_condition["massfrac_"+self.name]=1.0
        #: Vapor pressure of the pure liquid
        self.vapor_pressure:ExpressionNumOrNone=None
        self.passive_field=self.name
        #: The components are used for mixtures. Here it is just the set with only the name of the liquid as only element.
        self.components = set({self.name})
        self._UNIFAC_groups:Dict[str,Dict[str,int]]={}
        #: Latent heat of evaporation of the pure liquid
        self.latent_heat_of_evaporation:ExpressionNumOrNone=None
        self._output_properties=self._output_properties.copy()
        self._output_properties["vapor_pressure_"+self.name]=lambda props : self.get_vapor_pressure_for(self.name)


    
    def set_unifac_groups(self,grps:Dict[str,int],only_for:Optional[Union[Set[str],str]]=None):
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

    def set_vapor_pressure_by_Antoine_coeffs(self,A:float,B:float,C:float,convention_P:Expression=mmHg,convention_T:Union[Expression,CelsiusClass]=celsius):
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
        """        
        if self.name==name:
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

    def __mul__(self,other:Union[float,int,Expression])->"LiquidMixtureDefinitionComponent":
        return LiquidMixtureDefinitionComponent(self,other)

    def __rmul__(self,other:Union[float,int,Expression])->"LiquidMixtureDefinitionComponent":
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


class PureGasProperties(BaseGasProperties):
    """
    Provides properties of a pure gas.    
    """
    is_pure:bool=True
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

    def __mul__(self,other:Union[float,int,Expression])->"GasMixtureDefinitionComponent":
        return GasMixtureDefinitionComponent(self,other)

    def __rmul__(self,other:Union[float,int,Expression])->"GasMixtureDefinitionComponent":
        return GasMixtureDefinitionComponent(self,other)

class PureSolidProperties(BaseSolidProperties):
    """
    Defines properties of a pure solid.    
    """
    is_pure:bool=True
    def __init__(self):
        super().__init__()
        self.initial_condition["massfrac_"+self.name]=1.0
        self.components = set({self.name})

    def get_pure_component(self,name:str):
        if self.name==name:
            return self
        else:
            return None

class UNIFACPyoomphExpressionGenerator(UNIFACExpressionGeneratorBase):
    def get_molefrac_var(self,name:str) -> Expression:
        return var("molefrac_"+name)
    def get_temperature_in_kelvin(self) -> Expression:
        return var("temperature")/kelvin
    def pow(self,a:ExpressionOrNum,b:ExpressionOrNum) -> ExpressionOrNum:
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
    is_pure:bool=False
    def __init__(self,pure_props:Dict[str,MaterialProperties]):
        BaseLiquidProperties.__init__(self)
        BaseMixedProperties.__init__(self,pure_props=pure_props)
        self.pure_properties=cast(Dict[str,PureLiquidProperties],self.pure_properties)
        
        #: A dict holding the vapor pressures given by the name of each pure component. By default, it will be set to ideal Raoult's law.
        self.vapor_pressure_for:Dict[str,ExpressionOrNum]={}
        #: A dict holding the activity coefficients given by the name of each pure component.
        self.activity_coefficients:Dict[str,ExpressionOrNum]={}
        self.set_vapor_pressure_by_raoults_law()
        self._latent_heat_of_evaporation:Dict[str,ExpressionOrNum]={}

        self._output_properties=self._output_properties.copy()
        def make_lambda_for_vapor_pressure(k:str)->Callable[[MaterialProperties],ExpressionNumOrNone]:
            return lambda props: self.get_vapor_pressure_for(k)
        for k in self.components:
            self._output_properties["vapor_pressure_" + k] = make_lambda_for_vapor_pressure(k)
        def make_lambda_for_activity_coeff(k:str)->Callable[[MaterialProperties],ExpressionNumOrNone]:
            return lambda props: self.activity_coefficients.get(k)
        for k in self.components:
            self._output_properties["activity_coefficient_" + k] = make_lambda_for_activity_coeff(k)

        self._reaction_rates:Dict[str,ExpressionOrNum]={}

        self._unifac_multi_return:Optional[UNIFACMultiReturnExpression]=None
        self._unifac_model:Optional[str]=None # Used UNIFAC parameter table

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
            return pc.get_vapor_pressure_for(name)


    # use_multi_return uses multi-return expression instead of subexpressions
    # if it is and int, it will use multi-return for mixtures with #components>=use_multi_return
    # multi-return expressions are considerably faster in generating C code. However, they use finite differences for the Jacobain
    def set_activity_coefficients_by_unifac(self,model:str,set_vapor_pressures:bool=True,use_multi_return:Union[bool,int]=3):
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
        if use_multi_return==True or ((use_multi_return is not False) and len(self.components)>=use_multi_return):
            self._unifac_multi_return=UNIFACMultiReturnExpression(self,model)
            for cn in self.components:
                self.activity_coefficients[cn]=cast(Expression,self._unifac_multi_return.get_activity_coefficient(cn))
        else:
            server=ActivityModel.get_activity_model_by_name(modelname)

            unifac_components:Dict[str,UNIFACMolecule]={cn:UNIFACMolecule(cn,server) for cn in self.components}
            for cn in self.components:
                comp=self.pure_properties[cn]
                assert isinstance(comp,PureLiquidProperties)
                subgroups=comp._UNIFAC_groups[modelname] 
                if len(subgroups)==0:
                    raise RuntimeError("Component "+cn+" has no UNIFAC groups defined for model "+modelname)
                for sgn,amount in subgroups.items():
                    unifac_components[cn].add_subgroup(sgn,amount)
            unifac_mix=UNIFACMixture(*unifac_components.values())
            unifac_mix.set_expression_generator(UNIFACPyoomphExpressionGenerator())
            for cn in self.components:
                self.activity_coefficients[cn]=cast(Expression,unifac_mix.get_activity_coefficient_expression(cn))

        if set_vapor_pressures:
            self.set_vapor_pressure_by_raoults_law()

    def set_vapor_pressure_by_raoults_law(self):
        """
        Set the vapor pressures based on Raoult's law. Potentially set activity coefficients are considered.
        """
        for c in self.components:
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
    is_pure:bool=False
    def __init__(self,pure_props:Dict[str,MaterialProperties]):
        BaseGasProperties.__init__(self)
        BaseMixedProperties.__init__(self,pure_props=pure_props)
        self.pure_properties=cast(Dict[str,PureGasProperties],self.pure_properties)

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
        molar_mass=0
        for n,pc in self.pure_properties.items():
            molar_mass+=var("molefrac_"+n)*pc.molar_mass
        return pressure *molar_mass/ (gas_constant * temperature)

    def set_mass_density_from_ideal_gas_law(self):
        """
        Sets the mass density by using :py:meth:`mass_density_from_ideal_gas_law`.
        """
        self.mass_density=self.mass_density_from_ideal_gas_law()


    
    def set_diffusion_coefficient_by_Fuller_eq(self, for_dilute_gas:str, dominant_gas:Optional[str]=None):
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
    liquid_components:Union[str,Set[str],None] = None
    #: The components of the gas phase
    gas_components:Union[str,Set[str],None] = None
    #: The surfactants at the interface
    surfactants:Union[Set[str],str,None] = None
    def _sort_phases(self,sideA:AnyMaterialProperties,sideB:AnyMaterialProperties)->Tuple[AnyMaterialProperties,AnyMaterialProperties]:
        if sideA.state_of_matter=="liquid" and sideB.state_of_matter=="gas":
            return sideA,sideB
        elif sideA.state_of_matter=="gas" and sideB.state_of_matter=="liquid":
            return sideB,sideA
        else:
            raise RuntimeError("This liquid-gas interface does not have a liquid and a gas side")

    def __init__(self,phaseA:AnyMaterialProperties,phaseB:AnyMaterialProperties,surfactant_dict:Dict[SurfactantProperties,ExpressionOrNum]):
        from .mass_transfer import StandardMassTransferModelLiquidGas
        super(LiquidGasInterfaceProperties, self).__init__(phaseA,phaseB)
        assert isinstance(self._phaseA,(PureLiquidProperties,MixtureLiquidProperties))
        assert isinstance(self._phaseB,(PureGasProperties,MixtureGasProperties))
        self._liquid_phase:Union[PureLiquidProperties,MixtureLiquidProperties]=self._phaseA        
        self._gas_phase:Union[PureGasProperties,MixtureGasProperties] = self._phaseB
        self._surfactants=surfactant_dict.copy() if surfactant_dict is not None else {}
        if "gas" in self._liquid_phase.default_surface_tension.keys():
            sigm=self._liquid_phase.default_surface_tension.get("gas")
            if sigm is not None:
                self.surface_tension=sigm
        self._mass_transfer_model=StandardMassTransferModelLiquidGas(self._liquid_phase,self._gas_phase)
        #: The rate of surfactant adsorption and desorption, merged in a single expression per surfactant
        self.surfactant_adsorption_rate:Dict[str,ExpressionOrNum]={}
        self._surface_diffusivity:Dict[str,ExpressionOrNum]={}

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
        fields={}
        nondims={}
        for surf,conc in self._surfactants.items():
            if not isinstance(conc,_pyoomph.Expression):
                conc=_pyoomph.Expression(conc)
            fields["surfconc_"+surf.name]=conc
        fields["velocity"]=_pyoomph.Expression(0)
        fields["velocity_x"] = _pyoomph.Expression(0)
        fields["velocity_y"] = _pyoomph.Expression(0)
        fields["velocity_z"] = _pyoomph.Expression(0)
        return _pyoomph.GiNaC_subsfields(expr, fields, nondims, {})

    def get_mass_transfer_model(self) -> Optional["MassTransferModelBase"]:
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
    def __init__(self,phaseA:AnyMaterialProperties,phaseB:AnyMaterialProperties,surfactant_dict:Dict[SurfactantProperties,ExpressionOrNum]):
        super(DefaultLiquidGasInterface, self).__init__(phaseA,phaseB,surfactant_dict=surfactant_dict)
        if self._liquid_phase.default_surface_tension.get("gas") is None:
            raise RuntimeError("Either specify the interface properties of the liquid-gas interface of liquid:"+str(self._liquid_phase)+" vs. gas:"+str(self._gas_phase)+" or at least set the default surface tension against gas in the liquid properties")
        if self._liquid_phase.default_surface_tension["gas"] is None:
            raise RuntimeError("interface properties of the liquid-gas interface of liquid:"+str(self._liquid_phase)+" vs. gas:"+str(self._gas_phase)+". That's okay, if you at least provide a default surface tension against gas in the liquid phase")
        self.surface_tension=0+self._liquid_phase.default_surface_tension["gas"]

MaterialProperties.library["interfaces"]["_defaults"]["liquid_gas"]=DefaultLiquidGasInterface


class LiquidSolidInterfaceProperties(BaseInterfaceProperties):
    typus="liquid_solid"
    liquid_components:Union[str,Set[str],None] = None
    solid_components:Union[str,Set[str],None] = None
    surfactants:Union[Set[str],str,None] = None
    def _sort_phases(self,sideA:AnyMaterialProperties,sideB:AnyMaterialProperties)->Tuple[AnyMaterialProperties,AnyMaterialProperties]:
        if sideA.state_of_matter=="liquid" and sideB.state_of_matter=="solid":
            return sideA,sideB
        elif sideA.state_of_matter=="solid" and sideB.state_of_matter=="liquid":
            return sideB,sideA
        else:
            raise RuntimeError("The liquid-solid interface does not have a liquid and a solid bulk side")

    def get_liquid_properties(self) -> Union[PureLiquidProperties, MixtureLiquidProperties]:
        return self._liquid_phase

    def get_solid_properties(self) -> PureSolidProperties:
        return self._solid_phase

    def __init__(self,phaseA:AnyMaterialProperties,phaseB:AnyMaterialProperties,surfactant_dict:Dict[SurfactantProperties,ExpressionOrNum]):
        super(LiquidSolidInterfaceProperties, self).__init__(phaseA,phaseB)
        assert isinstance(self._phaseA,(PureLiquidProperties,MixtureLiquidProperties))
        assert isinstance(self._phaseB,PureSolidProperties)
        self._liquid_phase:Union[PureLiquidProperties,MixtureLiquidProperties]=self._phaseA        
        self._solid_phase:PureSolidProperties = self._phaseB        
        self._surfactants=surfactant_dict.copy() if surfactant_dict is not None else {}
        self.surfactant_adsorption_rate={}
        self.equilibrium_temperature=None
        self.latent_heat_of_fusion=None
        self.surfactant_adsorption_rate:Dict[str,ExpressionOrNum]={}
        self._surface_diffusivity:Dict[str,ExpressionOrNum]={}

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
        fields={}
        nondims={}
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
    surfactants:Union[Set[str],str,None] = None
    componentsA:Union[Set[str],str,None] = set()
    componentsB:Union[Set[str],str,None] = set()

    def get_fraction_in_rich_phase(self,varname:str,rich_component:Optional[str]=None,in_bulk:bool=False):
        if rich_component is None:
            rich_component=varname
        if self._phaseA.initial_condition[rich_component]>self._phaseB.initial_condition[rich_component]:
            return var(varname,domain=".." if in_bulk else ".")
        elif self._phaseA.initial_condition[rich_component]<self._phaseB.initial_condition[rich_component]:
            return var(varname,domain="|.." if in_bulk else "|.")
        else:
            raise RuntimeError("Cannot distinguish phases")
        
    def get_fraction_in_poor_phase(self,varname:str,poor_component:Optional[str]=None,in_bulk:bool=False):
        if poor_component is None:
            poor_component=varname
        if self._phaseA.initial_condition[poor_component]<self._phaseB.initial_condition[poor_component]:
            return var(varname,domain=".." if in_bulk else ".")
        elif self._phaseA.initial_condition[poor_component]>self._phaseB.initial_condition[poor_component]:
            return var(varname,domain="|.." if in_bulk else "|.")
        else:
            raise RuntimeError("Cannot distinguish phases")

    def __init__(self,phaseA:AnyMaterialProperties,phaseB:AnyMaterialProperties,surfactant_dict:Dict[SurfactantProperties,ExpressionOrNum]):
        super(LiquidLiquidInterfaceProperties, self).__init__(phaseA,phaseB)
        self._surfactants=surfactant_dict.copy() if surfactant_dict is not None else {}
        self.surfactant_adsorption_rate={}
        self._mass_transfer_model=None

    def get_mass_transfer_model(self) -> Optional["MassTransferModelBase"]:
        return self._mass_transfer_model

    def evaluate_at_initial_surfactant_concentrations(self,expr:ExpressionOrNum) -> ExpressionOrNum:
        if not isinstance(expr,Expression):
            return expr
        fields={}
        nondims={}
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
def get_pure_material(state_of_matter:str,name:str,return_class:Literal[True])->Type[MaterialProperties]: ...

@overload
def get_pure_material(state_of_matter:str,name:List[str],return_class:Literal[False]=...)->Tuple[MaterialProperties,...]: ...

@overload
def get_pure_material(state_of_matter:str,name:List[str],return_class:Literal[True])->Tuple[Type[MaterialProperties],...]: ...


def get_pure_material(state_of_matter:str,name:Union[str,List[str]],return_class:bool=False)->Union[MaterialProperties,Type[MaterialProperties],Tuple[MaterialProperties,...],Tuple[Type[MaterialProperties],...]]:
    if isinstance(name,(list,tuple)):
        res:List[MaterialProperties]=[]
        for n in name:
            res.append(cast(MaterialProperties,get_pure_material(state_of_matter,n,return_class))) #type:ignore
        return tuple(res)
    else:
        if not name in MaterialProperties.library[state_of_matter]["pure"].keys():
            print("Available pure " + state_of_matter + " components: " + str(MaterialProperties.library[state_of_matter]["pure"].keys()))
            raise RuntimeError(
                "Cannot find any materials named '" + name + "' and in state '" + state_of_matter + "'. Make sure to import the corresponding python file, where these component is defined or define it yourself. For examples, please have a look at " + os.path.realpath(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "default_materials.py")))
        if return_class:
            return MaterialProperties.library[state_of_matter]["pure"][name]
        else:
            return MaterialProperties.library[state_of_matter]["pure"][name]()
    

#Can take multiple names
@overload
def get_pure_liquid(name:str,return_class:Literal[False]=...)->PureLiquidProperties: ...

@overload
def get_pure_liquid(name:str,return_class:Literal[True])->Type[PureLiquidProperties]: ...

@overload
def get_pure_liquid(name:List[str],return_class:Literal[False]=...)->Tuple[PureLiquidProperties,...]: ...

@overload
def get_pure_liquid(name:List[str],return_class:Literal[True])->Tuple[Type[PureLiquidProperties],...]: ...

def get_pure_liquid(name:Union[str,List[str]],return_class:bool=False)->Union[PureLiquidProperties,Type[PureLiquidProperties],Tuple[PureLiquidProperties,...],Tuple[Type[PureLiquidProperties],...]]:
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
def get_surfactant(name:str,return_class:Literal[True])->Type[SurfactantProperties]: ...

@overload
def get_surfactant(name:List[str],return_class:Literal[False]=...)->Tuple[SurfactantProperties,...]: ...

@overload
def get_surfactant(name:List[str],return_class:Literal[True])->Tuple[Type[SurfactantProperties],...]: ...

def get_surfactant(name:Union[str,List[str]],return_class:bool=False)->Union[SurfactantProperties,Type[SurfactantProperties],Tuple[SurfactantProperties,...],Tuple[Type[SurfactantProperties],...]]:
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

@overload
def get_pure_gas(name:str,return_class:Literal[False]=...)->PureGasProperties: ...

@overload
def get_pure_gas(name:str,return_class:Literal[True])->Type[PureGasProperties]: ...

@overload
def get_pure_gas(name:List[str],return_class:Literal[False]=...)->Tuple[PureGasProperties,...]: ...

@overload
def get_pure_gas(name:List[str],return_class:Literal[True])->Tuple[Type[PureGasProperties],...]: ...

def get_pure_gas(name:Union[str,List[str]],return_class:bool=False)->Union[PureGasProperties,Type[PureGasProperties],Tuple[PureGasProperties,...],Tuple[Type[PureGasProperties],...]]:
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
def get_pure_solid(name:str,return_class:Literal[True])->Type[PureSolidProperties]: ...

@overload
def get_pure_solid(name:List[str],return_class:Literal[False]=...)->Tuple[PureSolidProperties,...]: ...

@overload
def get_pure_solid(name:List[str],return_class:Literal[True])->Tuple[Type[PureSolidProperties],...]: ...

def get_pure_solid(name:Union[str,List[str]],return_class:bool=False)->Union[PureSolidProperties,Type[PureSolidProperties],Tuple[PureSolidProperties,...],Tuple[Type[PureSolidProperties],...]]:
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
def get_mixture_properties(*purecompos:MaterialProperties,**kwargs:Any)->Union[MaterialProperties,MixtureGasProperties,MixtureLiquidProperties]:
    if len(purecompos)==1:
        return purecompos[0]	#Pure material
    som=None
    comps:Set[str]=set()
    pureprops={}
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
        raise KeyError("Mixture properties of mixture from " + str(comps) + " in state " + som + " not defined")
    if kwargs.get("return_class",False):
        return cls
    else:
        return cls(pureprops)


@overload
def get_interface_properties(phaseA:Union[PureLiquidProperties,MixtureLiquidProperties],phaseB:Union[PureGasProperties,MixtureGasProperties],surfactants:Optional[Union[str,SurfactantProperties,Dict[str,ExpressionOrNum],Dict[SurfactantProperties,ExpressionOrNum]]]=None)->LiquidGasInterfaceProperties: ...

@overload
def get_interface_properties(phaseA:Union[PureLiquidProperties,MixtureLiquidProperties],phaseB:PureSolidProperties,surfactants:Optional[Union[str,SurfactantProperties,Dict[str,ExpressionOrNum],Dict[SurfactantProperties,ExpressionOrNum]]]=None)->LiquidSolidInterfaceProperties: ...

def get_interface_properties(phaseA:Union[MaterialProperties,MixtureDefinitionComponents],phaseB:Union[MaterialProperties,MixtureDefinitionComponents],surfactants:Optional[Union[str,SurfactantProperties,Dict[str,ExpressionOrNum],Dict[SurfactantProperties,ExpressionOrNum]]]=None)->Union[BaseInterfaceProperties,LiquidGasInterfaceProperties]:
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
    surfactantsN:Dict[SurfactantProperties,ExpressionOrNum]={}
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
        key = (lcomps, frozenset(cast(Set[str],set())), frozenset(scomps))
        if key in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][key](liquid, gas, surfactantsN)
        if len(scomps)>0:
            raise RuntimeError("Cannot find a liquid-gas interface definition between liquid "+str(set(lcomps))+" and gas "+str(set(gcomps))+" with surfactants "+str(set(scomps)))
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
        key = (lcomps, frozenset(cast(Set[str],set())), frozenset(solcomps))
        if key in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][key](liquid, solid, surfactantsN)
        key = (lcomps, frozenset(cast(Set[str],set())), frozenset(cast(Set[str],set())))
        if key in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][key](liquid, solid, surfactantsN)
    elif typus=="liquid_liquid":
        assert isinstance(phaseA,(PureLiquidProperties,MixtureLiquidProperties))
        assert isinstance(phaseB,(PureLiquidProperties,MixtureLiquidProperties))
        Acomps = frozenset(phaseA.components)
        Bcomps = frozenset(phaseB.components)
        scomps = frozenset(surfactantsN.keys())  # TODO Surfactants
        key = (frozenset({Acomps, Bcomps}), scomps,)
        if key in MaterialProperties.library["interfaces"][typus].keys():
            return MaterialProperties.library["interfaces"][typus][key](phaseA, phaseB, surfactantsN)
    else:
        raise RuntimeError("Implement")
    #print("PHASE A",phaseA)
    #print("PHASE B",phaseB)
    #exit()
    #MaterialProperties.library["interfaces"][typus]

    if typus in MaterialProperties.library["interfaces"]["_defaults"].keys():
        return MaterialProperties.library["interfaces"]["_defaults"][typus](phaseA,phaseB,surfactantsN)
    else:
        n1=phaseA.name if phaseA.is_pure else "["+", ".join(phaseA.components)+"]"
        n2 = phaseB.name if phaseB.is_pure else "[" + ", ".join(phaseB.components) + "]"
        if len(surfactantsN)==0:
            raise RuntimeError("Cannot find an interface of type "+typus+" for "+n1+" | "+n2)
        else:
            raise RuntimeError("Cannot find an interface of type "+typus+" for "+n1+" | "+n2+" and the surfactants "+str({s.name for s in surfactantsN}))

@overload
def Mixture(mdef:Union[LiquidMixtureDefinitionComponents,LiquidMixtureDefinitionComponent,PureLiquidProperties],temperature:ExpressionNumOrNone=...,quantity:MixQuantityDefinition=...,pressure:ExpressionOrNum=...)->AnyLiquidProperties: ...

@overload
def Mixture(mdef:Union[GasMixtureDefinitionComponents,GasMixtureDefinitionComponent,PureGasProperties],temperature:ExpressionNumOrNone=...,quantity:MixQuantityDefinition=...,pressure:ExpressionOrNum=...)->AnyGasProperties: ...

@overload
def Mixture(mdef:Union[MixtureDefinitionComponents,MixtureDefinitionComponent,AnyMaterialProperties],temperature:ExpressionNumOrNone=...,quantity:MixQuantityDefinition=...,pressure:ExpressionOrNum=...)->MaterialProperties: ...

def Mixture(mdef:Union[MixtureDefinitionComponents,MixtureDefinitionComponent,AnyMaterialProperties],temperature:ExpressionNumOrNone=None,quantity:MixQuantityDefinition="mass_fraction",pressure:ExpressionOrNum=1*atm)->AnyMaterialProperties:
    """
    Returns a gas or liquid mixture from the given mixture definition components or a single material properties object.

    Args:
        mdef: Either a pure substance or a mixture like ``get_pure_liquid("water")+0.5*get_pure_liquid("ethanol")``.
        temperature: The temperature of the mixture. Used for potential initial conditions and required if you want to use e.g. volume fractions or relative humidity as mixture quantity.
        quantity: Specifies the quantity definition of the mixture. Can be either ``"mass_fraction"``, ``"volume_fraction"``, ``"molar_fraction"``, or ``"relative_humidity"``.
        pressure: Absolute pressure. Necessary for particular conversions.

    Returns:
        The properties of the mixture from the material library.
    """
    if isinstance(mdef,MixtureDefinitionComponents):
        res,init=mdef.finalise(quantity,temperature=temperature,pressure=pressure)
        res=tuple(res)
        #print(res)

        props=get_mixture_properties(*res)
        for e,k in init.items():
            props.initial_condition["massfrac_"+e]=k
        if temperature is not None:
            props.initial_condition["temperature"]=temperature
        return props
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