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
from ..materials.generic import MixtureLiquidProperties,MixtureGasProperties
from .. import GlobalLagrangeMultiplier, WeakContribution
from ..generic import Equations,InterfaceEquations
from ..expressions import * #Import grad et al

from ..typings import *

if TYPE_CHECKING:
   from ..generic import Problem

class AdvectionDiffusionEquations(Equations):
   r"""
      .. _AdvectionDiffusionEquations:



      Represents the advection-diffusion equation in the form:

      .. math::
      
        \partial_t u + \mathbf{b} \cdot \nabla u - \nabla \cdot (D \nabla u) = f

      
      where :math:`u` is the dependent scalar variable, :math:`D` is the diffusion coefficient, :math:`\mathbf{b}` is the advection velocity, and  :math:`f` is the source term.
      
      In the weak form, the equation reads:

      .. math::

         (\partial_t u, v) + (\mathbf{b} \cdot \nabla u, v) + (D \nabla u, \nabla v) - \langle \nabla u \cdot \mathbf{n} , v \rangle = (f, v)
      
      Args:
         fieldnames(Union[str,List[str]]): The name(s) of the dependent variable(s). Default is "advdiffu".
         diffusivity(ExpressionOrNum): The diffusion coefficient. Default is 1.
         space(FiniteElementSpaceEnum): The finite element space. Default is "C2", i.e. second order continuous Lagrangian elements.
         consider_scaling(bool): Whether to consider scaling. Default is True.
         fluid_props(Optional[Union[MixtureLiquidProperties,MixtureGasProperties]]): The fluid properties. Default is None.
         wind(ExpressionOrNum): The advection velocity. Default is var("velocity").
         dt_factor(ExpressionOrNum): Multiplicative time step factor. Default is 1.
         time_scheme(Optional[TimeSteppingScheme]): The time stepping scheme. Default is None.
         source(Union[ExpressionOrNum,Dict[str,ExpressionOrNum]]): The source term. Default is {}.
         advection_by_parts(Union[bool,Literal["skew"]]): Whether to integrate by partsthe weak form of the advective term. Default is False.
         velocity_name_for_scaling(str): The name of the velocity for scaling. Default is "velocity".
   """

   def __init__(self,fieldnames:Union[str,List[str]]="advdiffu",*,diffusivity:ExpressionOrNum=1,space:"FiniteElementSpaceEnum"="C2",consider_scaling:bool=True,fluid_props:Optional[Union[MixtureLiquidProperties,MixtureGasProperties]]=None,wind:ExpressionOrNum=var("velocity"),dt_factor:ExpressionOrNum=1,time_scheme:Optional[TimeSteppingScheme]=None,source:Union[ExpressionOrNum,Dict[str,ExpressionOrNum]]={},advection_by_parts:Union[bool,Literal["skew"]]=False,velocity_name_for_scaling="velocity"):
      super().__init__()
      self.dt_factor=dt_factor
      self.diffusivity=diffusivity      
      self.space:"FiniteElementSpaceEnum"=space
      self.wind=wind      
      self.velocity_name_for_scaling=velocity_name_for_scaling
      self.time_scheme:Optional[TimeSteppingScheme]=time_scheme
      self.advection_by_parts=advection_by_parts
      if isinstance(fieldnames,str):
         self.fieldnames=[fieldnames]
      else:
         self.fieldnames=fieldnames
      if not isinstance(source,dict):
         self.source={n:source for n in self.fieldnames}
      else:
         self.source=source
      self.consider_scaling=consider_scaling
      self.fluid_props=fluid_props
      self.component_names:Dict[str,str]={}
      if self.fluid_props is not None:
         self.fieldnames:List[str]=[]         
         for n in self.fluid_props.required_adv_diff_fields:
            self.component_names["massfrac_"+n]=n
            self.fieldnames.append("massfrac_"+n)
         #print(self.fluid_props.required_adv_diff_fields)
         #print(dir(self.fluid_props))
         #exit()
         #self.diffusivity=fluid_props.diffusivity
      self.spatial_error_estimators=True


   def define_fields(self):
      remaining:Expression=Expression(1)
      remaining_test:Optional[Expression]=None
      mydom=self.get_my_domain()
      for f in self.fieldnames:
         ts=scale_factor("spatial")/scale_factor(self.velocity_name_for_scaling)/scale_factor(f) if self.consider_scaling else 1
         self.define_scalar_field(f,self.space,testscale=ts)
         remaining-=var(f,domain=mydom)
         if remaining_test is None:
            remaining_test=-testfunction(f,domain=mydom)
      if self.fluid_props is not None:
         assert self.fluid_props.passive_field is not None and isinstance(self.fluid_props.passive_field,str)
         assert remaining_test is not None
         self.define_field_by_substitution("massfrac_"+self.fluid_props.passive_field,remaining,also_on_interface=True)
         self.define_testfunction_by_substitution("massfrac_"+self.fluid_props.passive_field,remaining_test,also_on_interface=True)
         sum_massfrac_by_molar_mass=0

         for n,c in self.fluid_props.pure_properties.items():
            sum_massfrac_by_molar_mass+=var("massfrac_"+n,domain=mydom)/c.molar_mass

            self.define_field_by_substitution("molefrac_"+n, (var("massfrac_"+n,domain=mydom)/c.molar_mass)/var("_sum_massfrac_by_molar_mass",domain=mydom),also_on_interface=True)
         sum_massfrac_by_molar_mass=subexpression(sum_massfrac_by_molar_mass)
         self.define_field_by_substitution("_sum_massfrac_by_molar_mass",sum_massfrac_by_molar_mass,also_on_interface=True)

   def get_diffusion_coefficient(self,f1:str,f2:Optional[str]=None) -> ExpressionNumOrNone:
      if f2 is None:
         f2=f1
      if self.fluid_props is not None:
         return self.fluid_props.get_diffusion_coefficient(f1,f2,default=0)
      if f1!=f2:
         raise RuntimeError("Implement mixed diffusion between "+str(f1)+" and "+str(f2) )
      return self.diffusivity

   def define_residuals(self):
      if self.time_scheme is None:
         ts:Callable[[Expression],Expression]=lambda x :x
      else:
         ts:Callable[[Expression],Expression]=lambda x: time_scheme(cast(TimeSteppingScheme,self.time_scheme),x)
      if self.fluid_props is not None:
         for fn in self.fieldnames:
            f, f_test = var_and_test(fn)
            self.add_residual(weak(ts(self.dt_factor*partial_t(f)-self.source.get(fn,0)),f_test))
            if self.advection_by_parts=="skew":
               self.add_residual(-weak( ts(self.wind* f),grad(f_test))/2)
               self.add_residual(weak(ts(dot(self.wind, grad(f))), f_test)/2)
            elif self.advection_by_parts:
               self.add_residual(-weak( ts(self.wind* f),grad(f_test)))
            else:
               self.add_residual(weak(ts(dot(self.wind, grad(f))), f_test))
            for fn2 in self.fieldnames:
               f2,_=var_and_test(fn2)
               diffuD=self.get_diffusion_coefficient(self.component_names[fn],self.component_names[fn2])
               assert diffuD is not None
               self.add_residual(weak(ts(diffuD*grad(f2)),grad(f_test)))
      else:
         for fn in self.fieldnames:
            f, f_test = var_and_test(fn)
            self.add_residual(weak(ts(self.dt_factor * partial_t(f)),f_test)-weak(ts(self.source.get(fn, 0)),f_test))
            if self.advection_by_parts=="skew":
               self.add_residual(-weak( ts(self.wind* f),grad(f_test))/2)
               self.add_residual(weak(ts(dot(self.wind, grad(f))), f_test)/2)
            elif self.advection_by_parts:
               self.add_residual(-weak(ts(self.wind*f), grad(f_test)))
            else:
               self.add_residual(weak(ts(dot(self.wind,grad(f))),f_test))
            diffuD=self.diffusivity
            self.add_residual(weak(ts(diffuD*grad(f)),grad(f_test)))


   # Use this to either fix the average or the total integral of the field, i.e. add eqs+=AdvectionDiffusionEquations(...).with_integral_constraint(...)
   def with_integral_constraint(self,problem:"Problem",*,average:Optional[Union[Dict[str,ExpressionOrNum],ExpressionOrNum]]=None,integral:Optional[Union[Dict[str,ExpressionOrNum],ExpressionOrNum]]=None,ode_domain_name:str="globals",lagrange_prefix:Union[str,Dict[str,str]]="lagr_intconstr_",set_zero_on_normal_mode_eigensolve:bool=True) -> Equations:
      eq_additions=self
      if average is None and integral is None:
         raise ValueError("Please either specify average= or integral=")
      if average is None:
         average={}
      elif isinstance(average,dict):
         average=average.copy()
      else:
         if len(self.fieldnames)==1:
            average={self.fieldnames[0]:average}
         else:
            raise RuntimeError("Cannot set all averages like this")
      if integral is None:
         integral={}
      elif isinstance(integral,dict):
         integral=integral.copy()
      else:
         integral = {self.fieldnames[0]: integral}

      possible_fields=self.fieldnames
      lagr_mults:Dict[str,ExpressionOrNum]={}
      lagr_names:Dict[str,str]={}
      for k in possible_fields:
         if k in average.keys() and k in integral.keys():
            raise ValueError("Cannot set simultaneously average and integral for the field "+str(k))
         if k in average.keys():
            lagr_names[k]=(lagrange_prefix+k) if isinstance(lagrange_prefix,str) else lagrange_prefix[k]
            lagr_mults[lagr_names[k]]=0
            eq_additions+=WeakContribution(var(k)-average[k],testfunction(lagr_names[k],domain=ode_domain_name))
            eq_additions+=WeakContribution(var(lagr_names[k],domain=ode_domain_name),testfunction(k))
         elif k in integral.keys():
            lagr_names[k]=(lagrange_prefix+k) if isinstance(lagrange_prefix,str) else lagrange_prefix[k]
            lagr_mults[lagr_names[k]]=-integral[k]
            eq_additions+=WeakContribution(var(k),testfunction(lagr_names[k],domain=ode_domain_name),dimensional_dx=True)
            eq_additions+=WeakContribution(var(lagr_names[k],domain=ode_domain_name),testfunction(k),dimensional_dx=True)

      ode_additions=GlobalLagrangeMultiplier(**lagr_mults,set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve)
      problem.add_equations(ode_additions@ode_domain_name)
      return eq_additions



class AdvectionDiffusionFluxInterface(Equations):
   """
      Represents the flux through the interface that naturally arises from the integration by parts of the diffusion term in the advection-diffusion equation.

      Args:
         **kwargs: name of the flux and its value.
   """
      
   def __init__(self, **kwargs:ExpressionOrNum):
      super(AdvectionDiffusionFluxInterface, self).__init__()
      self.fluxes=kwargs.copy()

   def define_residuals(self):
      for name,flux in self.fluxes.items():
         test=testfunction(name)
         self.add_residual(weak(flux,test))


      # In the weak form, the equation reads:

      # .. math::   
         
      #    (D (u - u_{\\infty}) \dfrac{(r-R) \cdot \mathbf{n}}{||(r-R)||^2}, v)

class AdvectionDiffusionInfinity(InterfaceEquations):
   """
      Represents the condition at infinity for the advection-diffusion equation in the form:

      .. math::
         u + R \\dfrac{\\partial u}{\\partial r} = u_{\\infty}

      where :math:`u` is the dependent variable, :math:`u_{\\infty}` is the value at infinity, and :math:`R` is the distance from the origin and :math:`r` is the radial coordinate.
      Hence, works only correctly in axisymmetric or 3D.
      For 2D, we use:

      .. math::
         u + R \\dfrac{\\partial u}{\\partial r} / \\log(R / L) = u_{\\infty}

      Due to the logarithmic solution behavior in 2D, the farfield length L must be provided here.

      This class requires the parent equations to be of type :ref:`AdvectionDiffusionEquations <AdvectionDiffusionEquations>`, meaning that if :ref:`AdvectionDiffusionEquations <AdvectionDiffusionEquations>` (or subclasses) are not defined in the parent domain, an error will be raised.

      Args:
         origin(ExpressionOrNum): The origin of the system. Default is vector([0]).
         farfield_length(Optional[ExpressionNumOrNone]): The farfield length (only required for 2D). Default is None.
         **kwargs: name of the field and its value.
   """
   required_parent_type = AdvectionDiffusionEquations
   def __init__(self,origin:ExpressionOrNum=vector([0]), farfield_length:ExpressionNumOrNone=None,**kwargs:ExpressionOrNum):
      super(AdvectionDiffusionInfinity, self).__init__()
      self.inftyvals={**kwargs}
      self.origin=origin
      self.farfield_length=farfield_length

   def define_residuals(self):
      n = self.get_normal()
      d = var("coordinate") - self.origin
      parents=self.get_parent_equations(AdvectionDiffusionEquations)
      assert parents is not None      
      if not isinstance(parents,(list,tuple)):
         parents=[parents]
      for fn,val in self.inftyvals.items():
         diffuD=None
         for p in parents:
            assert isinstance(p,AdvectionDiffusionEquations)
            if fn in p.fieldnames:
               diffuD = p.get_diffusion_coefficient(fn)
               break
         if diffuD is None:
            raise RuntimeError("Cannot find any diffusion coefficient for field "+fn)
         y, y_test = var_and_test(fn)
         R = square_root(dot(d, d))
         
         real_dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
         if real_dim==1:
            if self.farfield_length is None:
                raise RuntimeError("For 1D far-field monopole conditions, a farfield_length must be provided")
            dist_to_ff=dot(n,d)-self.farfield_length
            self.add_residual(weak(-diffuD * (y - val) /dist_to_ff,y_test))
         else:
            if real_dim==3:
                coordsys_dim_factor = 1
            elif real_dim==2:
                if self.farfield_length is None:
                    raise RuntimeError("For 2D far-field monopole conditions, a farfield_length must be provided")
                coordsys_dim_factor = -1/log(R/self.farfield_length)        
            self.add_residual(weak(diffuD * coordsys_dim_factor * (y - val) * dot(n, d) / (dot(d, d)) , y_test) )
            