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
 
from abc import abstractmethod
from token import OP

from ..typings import *
from ..expressions.cb import CustomMultiReturnExpression,Expression
from ..expressions.units import kelvin
from ..expressions import ExpressionNumOrNone,ExpressionOrNum, var

import numpy

if TYPE_CHECKING:
    from .generic import MixtureLiquidProperties,PureLiquidProperties

_TypeActivityModel=TypeVar("_TypeActivityModel",bound=Type["ActivityModel"])

class ActivityModel:
    """
    A generic class to predict activity coefficients of mixtures. 
    """
    registered_models:Dict[str,Type["ActivityModel"]]={}
    model_instances:Dict[str,Optional["ActivityModel"]]={}
    name:str
    @classmethod
    def register_activity_model(cls, *, override:bool=False):
        def decorator(subclass:_TypeActivityModel)->_TypeActivityModel:
            if subclass.name is None:
                raise RuntimeError("Please set a name for the acitivity coefficient model")
            if subclass.name in cls.registered_models.keys():
                if not override:
                    raise RuntimeError("Activity model with name "+subclass.name+" already registered. Use override=True to override")
                else:
                    cls.model_instances[subclass.name]=None #force Reinstance
            cls.registered_models[subclass.name]=subclass
            cls.model_instances[subclass.name]=None
            return subclass
        return decorator

    @classmethod
    def get_activity_model_by_name(cls,name:str)->"ActivityModel":
        if not (name in cls.registered_models.keys()):
            raise RuntimeError("No activity model with name "+name+" registered")
        if cls.model_instances[name] is None:
            cls.model_instances[name]=cls.registered_models[name]()
        return cls.model_instances[name] #type:ignore

    def __init__(self):
        pass

class UNIFACMainGroup:
    def __init__(self,name:str,index:Optional[int]=None):
        self.name=name
        self.index=index
        self.subgroups:Dict[str,"UNIFACSubGroup"]={}

class UNIFACSubGroup:
    def __init__(self,name:str,maingroup:UNIFACMainGroup,R:float,Q:float,index:Optional[int]=None):
        self.name=name
        self.maingroup=maingroup
        self.R=R
        self.Q=Q
        self.index=index
        self.maingroup.subgroups[self.name]=self

class UNIFACMolecule:
    def __init__(self,name:str,server:ActivityModel):
        self.name=name
        self.server=server
        self.groups:Dict[str,int]={}

    def add_subgroup(self,name_or_index:Union[str,int],count:int=1):
        assert isinstance(self.server,UNIFACLikeActivityModel)
        if isinstance(name_or_index,int):
            name_or_index=self.server.subgroup_by_index[name_or_index].name
        if not (name_or_index in self.server.subgroups.keys()):
            poss=list(map(str,self.server.subgroups.keys()))
            raise ValueError("Cannot find a functional subgroup "+str(name_or_index)+" in the table of subgroups of "+str(self.server.name)+", possible are:\n"+", ".join(poss))
        self.groups[name_or_index]=self.groups.get(name_or_index,0)+count


    def get_molecular_r(self) -> float:
        assert isinstance(self.server,UNIFACLikeActivityModel)
        res=0.0
        for n,count in self.groups.items():
            res+=self.server.subgroups[n].R*count
        return res

    def get_molecular_q(self) -> float:
        assert isinstance(self.server,UNIFACLikeActivityModel)
        res=0.0
        for n,count in self.groups.items():
            res+=self.server.subgroups[n].Q*count
        return res

class UNIFACExpressionGeneratorBase:
    def __init__(self):
        pass
    @abstractmethod
    def ln(self,arg): #type:ignore
        raise NotImplementedError()
    @abstractmethod
    def pow(self,a,b): #type:ignore
        raise NotImplementedError()
    @abstractmethod
    def get_molefrac_var(self,name): #type:ignore
        raise NotImplementedError()
    @abstractmethod
    def get_temperature_in_kelvin(self): #type:ignore
        raise NotImplementedError()
    @abstractmethod
    def subexpression(self,expr): #type:ignore
        return expr #type:ignore

class UNIFACMixture:
    def __init__(self,*components:UNIFACMolecule):
        if len(components)<=1:
            raise RuntimeError("Cannot make a UNIFAC mixture with one or less components")
        self.components=components
        self.server=components[0].server
        for c in self.components:
            if self.server!=c.server:
                raise RuntimeError("UNIFAC components defined on different servers!")

        self._generator:UNIFACExpressionGeneratorBase

    def set_expression_generator(self,generator:UNIFACExpressionGeneratorBase):
        if hasattr(self,"_generator") and self._generator==generator:
            return
        self._generator=generator
        self.update()

    def update(self):
        assert isinstance(self.server,UNIFACLikeActivityModel)
        #Precalculate some expressions
        V=0
        F=0
        VOtherExponent=0
        for c in self.components:
            r=c.get_molecular_r()
            q=c.get_molecular_q()
            x=self._generator.get_molefrac_var(c.name) #type:ignore
            V+=r*x #type:ignore
            F += q * x #type:ignore
            VOtherExponent+=(self._generator.pow(r,self.server.modified_volume_fraction_exponent))*x #type:ignore
        self._V=self._generator.subexpression(V) #type:ignore
        self._F = self._generator.subexpression(F) #type:ignore
        if self.server.modified_volume_fraction_exponent!=1: #type:ignore
            self._VOtherExponent=self._generator.subexpression(VOtherExponent) #type:ignore
        else:
            self._VOtherExponent=self._V #type:ignore

        self._allgroups:Dict[str,int]={}
        for c in self.components:
            for sg,sgcount in c.groups.items():
                self._allgroups[sg]=self._allgroups.get(sg,0)+sgcount #Total number of each subgroup
        self._nu:Dict[str,Dict[str,int]]={}
        for c in self.components:
            entry= {n:0 for n in self._allgroups.keys()}
            for sg,sgcount in c.groups.items():
                entry[sg]=sgcount
            self._nu[c.name]=entry
        self._group_Qs:Dict[str,float]={n: self.server.subgroups[n].Q for n in self._allgroups.keys()}
        self._sub_to_main:Dict[str,str]={}
        for sg in self._allgroups.keys():
            self._sub_to_main[sg]=self.server.subgroups[sg].maingroup.name
        self._As = {n: {m: self.server.interaction_table.get(self._sub_to_main[n], {}).get(self._sub_to_main[m], {}).get("A",0) for m in self._allgroups.keys()} for n in self._allgroups.keys()} 
        self._Bs = {n: {m: self.server.interaction_table.get(self._sub_to_main[n], {}).get(self._sub_to_main[m], {}).get("B", 0) for m in self._allgroups.keys()} for n in self._allgroups.keys()}
        self._Cs = {n: {m: self.server.interaction_table.get(self._sub_to_main[n], {}).get(self._sub_to_main[m], {}).get("C", 0) for m in self._allgroups.keys()} for n in self._allgroups.keys()}





    def get_ln_combinatorial_gamma(self,compo): #type:ignore
        r = compo.get_molecular_r() #type:ignore
        q=compo.get_molecular_q() #type:ignore        
        ln=self._generator.ln #type:ignore
        Z=self.server.coordination_number #type:ignore
        V = r / self._V #type:ignore
        F = q / self._F #type:ignore
        if self.server.modified_volume_fraction_exponent!=1: #type:ignore
            VoE = self._generator.pow(r, self.server.modified_volume_fraction_exponent) / self._VOtherExponent #type:ignore
        else:
            VoE=V #type:ignore
        V_over_F=self._generator.subexpression(V/F) #type:ignore
        return 1 - VoE + ln(VoE) -  Z/2 * q * (1 - V_over_F + ln(V_over_F)) #type:ignore

    def generate_ln_residual_thetas(self, compo, puremode): #type:ignore
        xj = []
        if puremode:
            for c in self.components:
                xj.append(1 if c == compo else 0) #type:ignore
        else:
            for c in self.components:
                xj.append(self._generator.get_molefrac_var(c.name)) #type:ignore

        denom = 0
        counts = {n:0 for n in self._allgroups.keys()}
        for j, c in enumerate(self.components):
            for sgn in self._allgroups.keys():
                if self._nu[c.name][sgn] > 0:
                    counts[sgn] += self._nu[c.name][sgn] * xj[j] #type:ignore
                    denom += self._nu[c.name][sgn] * xj[j] #type:ignore

        if not puremode:
            denom = self._generator.subexpression(denom) #type:ignore
        Xms =  {n:0 for n in self._allgroups.keys()}
        for sgn in self._allgroups.keys():
            if counts[sgn] != 0:
                Xms[sgn] = counts[sgn] / denom #type:ignore
        Qs = self._group_Qs
        Thetas = {n:0 for n in self._allgroups.keys()}
        Qdenom = 0
        for sgn in self._allgroups.keys():
            Qdenom += Xms[sgn] * Qs[sgn]
        if not puremode:
            Qdenom = self._generator.subexpression(Qdenom) #type:ignore
        for sgn in self._allgroups.keys():
            Thetas[sgn] = Xms[sgn] * Qs[sgn] / Qdenom #type:ignore
        return Thetas

    def get_ln_residual_gamma(self,compo): #type:ignore
        Theta = self.generate_ln_residual_thetas(compo, False) #type:ignore
        Theta_pure=self.generate_ln_residual_thetas(compo, True) #type:ignore

        TKelv = self._generator.get_temperature_in_kelvin() #type:ignore
        expo = self._generator.exp #type:ignore
        ln = self._generator.ln #type:ignore
        subexpr = self._generator.subexpression #type:ignore
        interaction_table = {}
        for i in self._allgroups.keys():
            interaction_table[i] = {}
            for j in self._allgroups.keys():
                interaction_table[i][j] = subexpr(
                    expo(-(self._As[i][j] / TKelv + self._Bs[i][j] + self._Cs[i][j] * TKelv))) #type:ignore

        denomM = {}
        denomMpure = {}
        for m in self._allgroups.keys():
            for n in self._allgroups.keys():
                denomM[m] = denomM.get(m,0)+ Theta[n] * interaction_table[n][m] #type:ignore
                denomMpure[m] = denomMpure.get(m,0) + Theta_pure[n] * interaction_table[n][m] #type:ignore
            denomM[m]=self._generator.subexpression(denomM[m]) #type:ignore
            denomMpure[m] = self._generator.subexpression(denomMpure[m]) #type:ignore

        def GammaKPart(k, theta, denoms): #type:ignore
            logarg=0
            linarg=0
            for m in self._allgroups.keys():
                logarg += theta[m] * interaction_table[m][k] #type:ignore
                linarg += theta[m] * interaction_table[k][m] / denoms[m] #type:ignore
            return ln(logarg) + linarg #type:ignore

        lgGammaR=0
        for k in self._allgroups.keys():
            if self._nu[compo.name][k] > 0: #type:ignore
                lgGammaR+= self._nu[compo.name][k] * self._group_Qs[k] * (GammaKPart(k, Theta_pure, denomMpure) - GammaKPart(k, Theta, denomM)) #type:ignore
        return lgGammaR #type:ignore

    def _get_component_by_name(self,compo:str) -> UNIFACMolecule:
        for c in self.components:
            if c.name == compo:
                return c
        else:
            raise RuntimeError("Component " + compo + " not in the mixture")

    def get_activity_coefficient_expression(self,compo:Union[str,UNIFACMolecule]): #type:ignore
        if self._generator is None:
            raise RuntimeError("Set an expression generator first")
        if isinstance(compo,str):
            compo=self._get_component_by_name(compo)

        gammaC=self._generator.subexpression(self.get_ln_combinatorial_gamma(compo)) #type:ignore
        gammaR=self._generator.subexpression(self.get_ln_residual_gamma(compo)) #type:ignore
        return self._generator.subexpression(self._generator.exp(gammaC+gammaR)) #type:ignore

        
    

class UNIFACLikeActivityModel(ActivityModel):
    """
    An activity model based on the UNIFAC model. This model is a generalization of the UNIFAC model, and can be used to define custom activity models based on the UNIFAC model. The model is defined by defining main groups and subgroups, and then setting the interaction parameters between the subgroups. The activity coefficient is then calculated using the UNIFAC model.    
    """
    def __init__(self):
        super(UNIFACLikeActivityModel, self).__init__()
        self.maingroups:Dict[str,UNIFACMainGroup]={}
        self.maingroup_by_index:Dict[int,UNIFACMainGroup]={}
        self.subgroups:Dict[str,UNIFACSubGroup]={}
        self.subgroup_by_index:Dict[int,UNIFACSubGroup]={}
        self._current_subgroup_definer:Any=None
        self.interaction_table:Dict[str,Dict[str,Dict[str,float]]]={}

        self.modified_volume_fraction_exponent=1
        self.coordination_number=10

    def define_main_group(self,name:str,index:Optional[int]=None):
        server=self
        class _MainGroupDefiner:
            def __init__(self,maingrp:UNIFACMainGroup):
                self.maingrp=maingrp
            def __enter__(self):
                server._current_subgroup_definer=self
                return self

            def __exit__(self, exc_type, exc_val, exc_tb): #type:ignore
                server._current_subgroup_definer=None
                return

            def sub_group(self,name:str,R:float,Q:float,index:Optional[int]=None,molar_mass:Optional[float]=None):
                if name in server.subgroups.keys():
                    raise RuntimeError("Subgroup "+name+" already defined in another main group "+server.subgroups[name].maingroup.name)
                res=UNIFACSubGroup(name,self.maingrp,R,Q,index)
                server.subgroups[name]=res
                if index is not None:
                    server.subgroup_by_index[index]=res
                return res
        grp=UNIFACMainGroup(name,index=index)
        self.maingroups[name]=grp
        if index is not None:
            self.maingroup_by_index[index]=grp
        return _MainGroupDefiner(grp)


    def define_sub_group(self,name:str,R:float,Q:float,index:int,molar_mass:Optional[float]=None):
        if self._current_subgroup_definer is None:
            raise RuntimeError("Can only do it in 'with self.define_main_group():' statements")
        self._current_subgroup_definer.sub_group(name,R,Q,index,molar_mass=molar_mass)

    def set_interaction(self,mainI:Union[int,str],mainJ:Union[int,str],*,Aij:Optional[float]=None,Aji:Optional[float]=None,Bij:Optional[float]=None,Bji:Optional[float]=None,Cij:Optional[float]=None,Cji:Optional[float]=None):
        def ensure_table(first:str,second:str):
            if not (first in self.interaction_table.keys()):
                self.interaction_table[first]={}
            if not (second in self.interaction_table[first].keys()):
                self.interaction_table[first][second] = {}
            return self.interaction_table[first][second]

        if isinstance(mainI,int):
            mainI=self.maingroup_by_index[mainI].name
        if isinstance(mainJ,int):
            mainJ=self.maingroup_by_index[mainJ].name
        if Aij is not None:
            ensure_table(mainI,mainJ)["A"]=Aij
        if Aji is not None:
            ensure_table(mainJ,mainI)["A"]=Aji
        if Bij is not None:
            ensure_table(mainI,mainJ)["B"]=Bij
        if Bji is not None:
            ensure_table(mainJ,mainI)["B"]=Bji
        if Cij is not None:
            ensure_table(mainI,mainJ)["C"]=Cij
        if Cji is not None:
            ensure_table(mainJ,mainI)["C"]=Cji

    

class ActivityServer:
    def __init__(self):
        pass




class UNIFACMultiReturnExpression(CustomMultiReturnExpression):
    def __init__(self,mix:"MixtureLiquidProperties",modelname:str,FD_epsilon=1e-9,constant_temperature:ExpressionNumOrNone=None):        
        super().__init__()
        
        from .generic import PureLiquidProperties
        self.mix=mix
        self.FD_epsilon=FD_epsilon
        self.constant_temperature=constant_temperature
        self._constant_temperature_in_K=float(self.constant_temperature/kelvin) if self.constant_temperature is not None else None
        self.argument_order=list(self.mix.required_adv_diff_fields)        
        self.argument_order.sort()        
        self.argument_order_with_passive=self.argument_order.copy()
        self.argument_order_with_passive.append(self.mix.passive_field)
        
        
        self.server=ActivityModel.get_activity_model_by_name(modelname)
        unifac_components:Dict[str,UNIFACMolecule]={cn:UNIFACMolecule(cn,self.server) for cn in mix.components}
        for cn in mix.components:
            comp=mix.pure_properties[cn]
            assert isinstance(comp,PureLiquidProperties)
            subgroups=comp._UNIFAC_groups[modelname] 
            if len(subgroups)==0:
                raise RuntimeError("Component "+cn+" has no UNIFAC groups defined for model "+modelname)
            for sgn,amount in subgroups.items():
                unifac_components[cn].add_subgroup(sgn,amount)
        self.unifac_mix=UNIFACMixture(*unifac_components.values())

        self.component_name_to_arg_index={n:self.argument_order_with_passive.index(n) for n in self.argument_order_with_passive}
               

        self._allgroups:Dict[str,int]={}
        for c in self.unifac_mix.components:
            for sg,sgcount in c.groups.items():
                self._allgroups[sg]=self._allgroups.get(sg,0)+sgcount #Total number of each subgroup
        self._nu:Dict[str,Dict[str,int]]={}
        for c in self.unifac_mix.components:
            entry= {n:0 for n in self._allgroups.keys()}
            for sg,sgcount in c.groups.items():
                entry[sg]=sgcount
            self._nu[c.name]=entry
        self._group_Qs:Dict[str,float]={n: self.server.subgroups[n].Q for n in self._allgroups.keys()}
        self._sub_to_main:Dict[str,str]={}
        for sg in self._allgroups.keys():
            self._sub_to_main[sg]=self.server.subgroups[sg].maingroup.name
        self._As = {n: {m: self.server.interaction_table.get(self._sub_to_main[n], {}).get(self._sub_to_main[m], {}).get("A",0) for m in self._allgroups.keys()} for n in self._allgroups.keys()} 
        self._Bs = {n: {m: self.server.interaction_table.get(self._sub_to_main[n], {}).get(self._sub_to_main[m], {}).get("B", 0) for m in self._allgroups.keys()} for n in self._allgroups.keys()}
        self._Cs = {n: {m: self.server.interaction_table.get(self._sub_to_main[n], {}).get(self._sub_to_main[m], {}).get("C", 0) for m in self._allgroups.keys()} for n in self._allgroups.keys()}
        self.unifac_mix._allgroups=self._allgroups
        self.unifac_mix._nu=self._nu
        self.unifac_mix._group_Qs=self._group_Qs

        self.rs=[0]*len(self.argument_order_with_passive)
        self.qs=[0]*len(self.argument_order_with_passive)
        self.thetas_pure={}
        for c in self.unifac_mix.components:
            i=self.component_name_to_arg_index[c.name]
            self.rs[i]=c.get_molecular_r()
            self.qs[i]=c.get_molecular_q()
            self.thetas_pure[c.name]=self.unifac_mix.generate_ln_residual_thetas(c, True)            
        
        self.molefraction_limit_epsilon=1e-9
    
    def get_num_returned_scalars(self, nargs: int) -> int:
        if nargs!=len(self.argument_order)+(1 if self._constant_temperature_in_K is None else 0):
            raise RuntimeError("Must be called with "+str(len(self.argument_order))+" arguments, namely the molar fractions in the order "+str(self.argument_order)+" and the temperature at the end if no constant temperature is specified. Got "+str(nargs)+" arguments instead")
        return len(self.argument_order_with_passive)
    
    def generate_c_code(self) -> str:
        #static void multi_ret_ccode_0(int flag, double *arg_list, double *result_list, double *derivative_matrix,int nargs,int nret)
        T_index=len(self.argument_order)    
        reduced_rs=numpy.array(self.rs)
        reduced_rs[:-1]-=reduced_rs[-1]
        reduced_qs=numpy.array(self.qs)
        reduced_qs[:-1]-=reduced_qs[-1]
        reduced_rs_other_exp=numpy.power(numpy.array(self.rs),self.server.modified_volume_fraction_exponent)
        reduced_rs_other_exp[:-1]-=reduced_rs_other_exp[-1]
        res= """
            const double T_in_K="""+(str(self._constant_temperature_in_K) if self._constant_temperature_in_K is not None else """arg_list["""+str(T_index)+"""]""")+""";
            // Make sure the mole fractions are normalized
            PYOOMPH_AQUIRE_ARRAY(double, molefracs,"""+ str(T_index)+""");
            double molar_sum=0.0;
            const double molefraction_limit_epsilon="""+str(self.molefraction_limit_epsilon)+""";
            for (int i=0;i<"""+str(T_index)+""";i++) 
            {
                if (arg_list[i]<molefraction_limit_epsilon) molefracs[i]=molefraction_limit_epsilon;
                else if (arg_list[i]>1.0-molefraction_limit_epsilon) molefracs[i]=1.0-molefraction_limit_epsilon;
                else molefracs[i]=arg_list[i];  
                molar_sum+=arg_list[i];
            }
            if (molar_sum>1.0-molefraction_limit_epsilon) {for (int i=0;i<"""+str(T_index)+""";i++) molefracs[i]*=(1.0-molefraction_limit_epsilon)/molar_sum;}
            """            
        res+="""
            const double V="""+str(self.rs[-1])+"+"+("+".join([str(reduced_rs[i])+"*molefracs["+str(i)+"]" for i in range(T_index)]))+""";
            const double F="""+str(self.qs[-1])+"+"+("+".join([str(reduced_qs[i])+"*molefracs["+str(i)+"]" for i in range(T_index)]))+""";
            const double F_over_V=F/V;
            """
        if self.server.modified_volume_fraction_exponent!=1:
            res+="""
            const double V_other_exp="""+(str(reduced_rs_other_exp[-1])+"+"+ "+".join([str(reduced_rs_other_exp[i])+"*molefracs["+str(i)+"]" for i in range(T_index)]))+""";
            const double F_over_V_other_exp=F/V_other_exp;
            """

        res+="""// ln_combinatorial        
            const double Z="""+str(self.server.coordination_number)+""";
            """            
        for i,c in enumerate(self.argument_order_with_passive):
            res+="""const double V_over_F_"""+str(i)+" = ("+str(self.rs[i]/self.qs[i])+""")*F_over_V ;                    
            const double r_over_V_other_exp_"""+str(i)+" = ("+str(numpy.power(self.rs[i], self.server.modified_volume_fraction_exponent))+""")/("""+("V_other_exp" if self.server.modified_volume_fraction_exponent!=1 else "V")+""") ;
            result_list["""+str(i)+"]=1.0 - r_over_V_other_exp_"""+str(i)+" + log(r_over_V_other_exp_"+str(i)+") -  Z/2.0 * ("+str(self.qs[i])+") * (1.0 - V_over_F_"+str(i)+" + log(V_over_F_"+str(i)+"""));
            """


        res+=""" // Get subgroup counts
            """

        # First get the Thetas at this composition
        counts_matrix = {n:[] for n in self._allgroups.keys()}
        for j, c in enumerate(self.argument_order_with_passive):
            for sgn in self._allgroups.keys():
                counts_matrix[sgn].append(self._nu[c][sgn])
        for sgn in self._allgroups.keys():
            counts_matrix[sgn]=numpy.array(counts_matrix[sgn])
            counts_matrix[sgn][:-1]-=counts_matrix[sgn][-1]
        for sgi,sgn in enumerate(self._allgroups.keys()):
            res+="double Theta_sg_"+str(sgi)+ " = "
            if counts_matrix[sgn][-1]!=0:
                res+=str(counts_matrix[sgn][-1])+" + "
            res+=("+".join(["("+str(counts_matrix[sgn][i])+")"+"*molefracs["+str(i)+"]" for i in range(T_index) if counts_matrix[sgn][i]!=0]))+""";
            """
        res+="double Theta_denom = "+ ("+".join("Theta_sg_"+str(sgi) for sgi in range(len(self._allgroups))))+""";
            """
        for sgi,sgn in enumerate(self._allgroups.keys()):
            res+="Theta_sg_"+str(sgi)+ " = Theta_sg_"+str(sgi)+" / Theta_denom * ("+str(self._group_Qs[sgn])+""") ;
            """
        res+="Theta_denom = "+ ("+".join("Theta_sg_"+str(sgi) for sgi in range(len(self._allgroups))))+""";
            """
        for sgi,sgn in enumerate(self._allgroups.keys()):
            res+="Theta_sg_"+str(sgi)+ " = Theta_sg_"+str(sgi)+""" / Theta_denom;
            """            

        res+=""" //Interaction table
            """        
        for ii,i in enumerate(self._allgroups.keys()):           
            for jj,j in enumerate(self._allgroups.keys()):
                res+="const double interact_"+str(ii)+"_"+str(jj)+" = exp(-( ("+str(self._As[i][j])+")/T_in_K "
                if self._Bs[i][j]!=0:
                     res+="+ ("+str(self._Bs[i][j])+")"
                if self._Cs[i][j]!=0:
                     res+="+ ("+str(self._Cs[i][j])+")*T_in_K"
                res+=""" ));                
            """                

        res+=""" //Interaction denominators
            """        
        for mi,m in enumerate(self._allgroups.keys()):
                res+="const double denomM_"+str(mi)+ " = "+ "+".join("Theta_sg_"+str(sgi)+"*interact_"+str(sgi)+"_"+str(mi) for sgi in range(len(self._allgroups)))+""";
            """
        for i,c in enumerate(self.argument_order_with_passive):                        
                for mi,m in enumerate(self._allgroups.keys()):
                    res+="const double denomMpure_"+str(i)+"_"+str(mi)+ " = "+ "+".join("("+str(self.thetas_pure[c][sgi])+ ")"+"*interact_"+str(sgii)+"_"+str(mi) for sgii,(sgi,sg) in enumerate(self._allgroups.items()) if self.thetas_pure[c][sgi]!=0)+""";
            """                                    
        res+=""" //Interaction Gammas
            """
        for ik,k in enumerate(self._allgroups.keys()):
            res+="const double Gamma_"+str(ik)+" = log("+( "+".join("Theta_sg_"+str(m)+"*interact_"+str(m)+"_"+str(ik) for m in range(len(self._allgroups)))) +")"
            res+=" + " + "+".join("Theta_sg_"+str(m)+"*interact_"+str(ik)+"_"+str(m)+"/denomM_"+str(m) for m in range(len(self._allgroups)))+""";
            """            

        for i,c in enumerate(self.argument_order_with_passive):
            for ik,k in enumerate(self._allgroups.keys()):
                if self._nu[c][k]>0:
                    res+="const double Gamma_pure_"+str(i)+"_"+str(ik)+" = log("+( "+".join("("+str(self.thetas_pure[c][sgm])+")*interact_"+str(m)+"_"+str(ik) for m,sgm in enumerate(self._allgroups.keys()) if self.thetas_pure[c][sgm]!=0 )) +")"
                    res+=" + " + "+".join("("+str(self.thetas_pure[c][sgm])+")*interact_"+str(ik)+"_"+str(m)+"/denomMpure_"+str(i)+"_"+str(m) for m,sgm in enumerate(self._allgroups.keys()) if self.thetas_pure[c][sgm]!=0)+""";
            """
        
        res+=""" // Add residual gamma
            """
        for i,c in enumerate(self.argument_order_with_passive):
            res+="result_list["+str(i)+"] = result_list["+str(i)+"] + "+ "+".join( "("+str(self._nu[c][k]*self._group_Qs[k])+")*(Gamma_pure_"+str(i)+"_"+str(ik)+" - "+"Gamma_"+str(ik)+ ")" for ik,k in enumerate(self._allgroups.keys()) if self._nu[c][k]>0  )+""";
            """            
      
      #  for k in self._allgroups.keys():
      #          if self._nu[c][k] > 0: #type:ignore
      #              lgGammaR+= self._nu[c][k] * self._group_Qs[k] * (GammaKPart(k, self.thetas_pure[c], denomMpure) - GammaKPart(k, Thetas, denomM)) #type:ignore

        res+=""" // Finalize
            """
        for i,c in enumerate(self.argument_order_with_passive):
            res+="""result_list["""+str(i)+"""] = exp(result_list["""+str(i)+"""]);
            """

        res+="""
            FILL_MULTI_RET_JACOBIAN_BY_FD("""+str(self.FD_epsilon)+ """)
            """
        return res
    
    def eval(self, flag: int, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray) -> None:
       # print("CALLED WITH",arg_list)
        V=0
        F=0
        VOtherExponent=0
        arg_list2=arg_list.copy()
        if self._constant_temperature_in_K is None:
            T_in_K=0+arg_list2[-1]
            arg_list2[-1]=1-sum(arg_list2[:-1])
        else:
            T_in_K=self._constant_temperature_in_K
            arg_list2=list(arg_list2)
            arg_list2.append(1-sum(arg_list2))
            arg_list2=numpy.array(arg_list2)
        
        for i,c in enumerate(self.argument_order_with_passive):
            V+=self.rs[i]*arg_list2[i] #type:ignore
            F += self.qs[i] * arg_list2[i] #type:ignore
            VOtherExponent+=(numpy.power(self.rs[i],self.server.modified_volume_fraction_exponent))*arg_list2[i] #type:ignore

        # ln_combinatorial
        ln_combinatorial=[]
        Z=self.server.coordination_number #type:ignore
        for i,c in enumerate(self.argument_order_with_passive):                                        
            Vi = self.rs[i] / V #type:ignore
            Fi = self.qs[i] / F #type:ignore        
            VoE = numpy.power(self.rs[i], self.server.modified_volume_fraction_exponent) / VOtherExponent #type:ignore        
            V_over_F=Vi/Fi #type:ignore
            ln_combinatorial.append(1 - VoE + numpy.log(VoE) -  Z/2 * self.qs[i] * (1 - V_over_F + numpy.log(V_over_F))) #type:ignore

        
        
        # First get the Thetas at this composition
        denom = 0
        counts = {n:0 for n in self._allgroups.keys()}
        for j, c in enumerate(self.argument_order_with_passive):
            for sgn in self._allgroups.keys():
                if self._nu[c][sgn] > 0:
                    counts[sgn] += self._nu[c][sgn] * arg_list2[j] #type:ignore
                    denom += self._nu[c][sgn] * arg_list2[j] #type:ignore
        
        Xms =  {n:0 for n in self._allgroups.keys()}
        for sgn in self._allgroups.keys():
            if counts[sgn] != 0:
                Xms[sgn] = counts[sgn] / denom #type:ignore
        
        Qs = self._group_Qs
        Thetas = {n:0 for n in self._allgroups.keys()}
        Qdenom = 0
        for sgn in self._allgroups.keys():
            Qdenom += Xms[sgn] * Qs[sgn]
        for sgn in self._allgroups.keys():
            Thetas[sgn] = Xms[sgn] * Qs[sgn] / Qdenom #type:ignore
        
        
        # ln_residual
        interaction_table = {}
        for i in self._allgroups.keys():
            interaction_table[i] = {}
            for j in self._allgroups.keys():
                interaction_table[i][j] = numpy.exp(-(self._As[i][j] / T_in_K + self._Bs[i][j] + self._Cs[i][j] * T_in_K))
        
        ln_residual=[]
        denomM = {}
        for m in self._allgroups.keys():
            for n in self._allgroups.keys():
                denomM[m] = denomM.get(m,0)+ Thetas[n] * interaction_table[n][m] #type:ignore
        for i,c in enumerate(self.argument_order_with_passive):                                               
            denomMpure = {}
            for m in self._allgroups.keys():
                for n in self._allgroups.keys():                    
                    denomMpure[m] = denomMpure.get(m,0) + self.thetas_pure[c][n] * interaction_table[n][m] #type:ignore            
            
            def GammaKPart(k, theta, denoms): #type:ignore
                logarg=0
                linarg=0
                for m in self._allgroups.keys():                    
                    logarg += theta[m] * interaction_table[m][k] #type:ignore
                    linarg += theta[m] * interaction_table[k][m] / denoms[m] #type:ignore
                return numpy.log(logarg) + linarg #type:ignore
            lgGammaR=0
            for k in self._allgroups.keys():
                if self._nu[c][k] > 0: #type:ignore
                    lgGammaR+= self._nu[c][k] * self._group_Qs[k] * (GammaKPart(k, self.thetas_pure[c], denomMpure) - GammaKPart(k, Thetas, denomM)) #type:ignore             
            ln_residual.append(lgGammaR)
            
        for i,(lnC,lnR) in enumerate(zip(ln_combinatorial,ln_residual)):
            result_list[i]=numpy.exp(lnC+lnR)

        if flag:
            self.fill_python_derivatives_by_FD(arg_list,result_list,derivative_matrix,fd_epsilion=self.FD_epsilon)
        
    
    def process_args_to_scalar_list(self, *args: ExpressionOrNum) -> List[ExpressionOrNum]:
        res=[a for a in args]
        res[-1]=res[-1]/kelvin
        return res
    

    def get_activity_coefficient(self,component:str,domain:Optional[str]=None):
        call_args=[var("molefrac_"+c,domain=domain) for c in self.argument_order]
        if self._constant_temperature_in_K is None:
            call_args.append(var("temperature",domain=domain))
        print("CALL ARGS",call_args)
        return self.__call__(*call_args)[self.argument_order_with_passive.index(component)]
