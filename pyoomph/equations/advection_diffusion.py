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
from ..materials.generic import MixtureLiquidProperties,MixtureGasProperties
from .. import GlobalLagrangeMultiplier, WeakContribution
from ..generic import Equations,InterfaceEquations
from ..generic.codegen import sorted_field_kwargs
from ..expressions import * #Import grad et al
from .stabilization import ScalarTransportEquations,ScalarTransportStabilization

from ..typings import *

if TYPE_CHECKING:
   from ..generic import Problem

class AdvectionDiffusionEquations(ScalarTransportEquations):
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
         advection_by_parts(Union[bool,Literal["skew"]]): Whether to integrate by parts the weak form of the advective term. Default is False.
         GCL(bool): Write the transient term as the derivative of the whole integral and advect with the velocity *relative to the mesh*, i.e. the conservative ALE form. On a mesh that follows an evaporating or otherwise moving boundary this conserves the total amount to machine precision instead of to the order of the time stepping, and its natural boundary condition becomes zero flux *through the moving boundary* rather than zero diffusive flux. Implies ``advection_by_parts`` and is incompatible with ``"skew"``. Default is False.
         gcl_scheme: Time stepping scheme of the ``GCL`` transient, from the set a derivative of an integral understands - a different set from ``time_scheme``, which rejects the ``_degr`` names. The default ``"BDF2_degr"`` degrades to first order in the first step, where an initial condition has no history; plain ``"BDF2"`` makes the whole run first order.
         velocity_name_for_scaling(str): The name of the velocity for scaling. Default is "velocity".
         stabilization: Optional residual-based stabilization (SUPG etc.), see :py:class:`~pyoomph.equations.stabilization.ScalarTransportStabilization`. ``None`` (the default) adds nothing at all.
   """

   def __init__(self,fieldnames:str | list[str]="advdiffu",*,diffusivity:ExpressionOrNum=1,space:"FiniteElementSpaceEnum"="C2",consider_scaling:bool=True,fluid_props:MixtureLiquidProperties | MixtureGasProperties | None=None,wind:ExpressionOrNum=var("velocity"),dt_factor:ExpressionOrNum=1,time_scheme:TimeSteppingScheme | None=None,source:ExpressionOrNum | dict[str, ExpressionOrNum]={},advection_by_parts:bool | Literal["skew"]=False,GCL:bool=False,gcl_scheme:"IntegralTimeSteppingScheme"="BDF2_degr",velocity_name_for_scaling="velocity",stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None):
      super().__init__()
      if GCL:
         if advection_by_parts=="skew":
            raise ValueError("AdvectionDiffusionEquations cannot combine GCL=True with "+
                             "advection_by_parts='skew': the conservative ALE form needs the whole "+
                             "advective term written against grad(v), and a half-and-half skew form "+
                             "leaves the other half unaccounted for.")
         # The conservative form *is* the by-parts form: d/dt of the whole integral needs the
         # advection written against grad(v), or the two do not describe the same equation.
         advection_by_parts=True
      self.GCL=GCL
      self.gcl_scheme:"IntegralTimeSteppingScheme"=gcl_scheme
      self.dt_factor=dt_factor
      self.diffusivity=diffusivity      
      self.space:"FiniteElementSpaceEnum"=space
      self.wind=wind      
      self.velocity_name_for_scaling=velocity_name_for_scaling
      self.time_scheme:TimeSteppingScheme | None=time_scheme
      self.advection_by_parts=advection_by_parts
      self.fieldnames:list[str]=[fieldnames] if isinstance(fieldnames,str) else list(fieldnames)
      if not isinstance(source,dict):
         self.source={n:source for n in self.fieldnames}
      else:
         self.source=source
      self.consider_scaling=consider_scaling
      self.fluid_props=fluid_props
      self.component_names:dict[str,str]={}
      if self.fluid_props is not None:
         self.fieldnames=[]         
         # sorted, as in CompositionAdvectionDiffusionEquations: this list fixes the order in which
         # the mass fraction fields are defined, and required_adv_diff_fields is an unordered set
         for n in sorted(self.fluid_props.required_adv_diff_fields):
            self.component_names["massfrac_"+n]=n
            self.fieldnames.append("massfrac_"+n)
         #print(self.fluid_props.required_adv_diff_fields)
         #print(dir(self.fluid_props))
         #exit()
         #self.diffusivity=fluid_props.diffusivity
      self.spatial_error_estimators=True
      # The wind of this class is scaled by velocity_name_for_scaling, not necessarily by "velocity"
      self._init_stabilization(stabilization,velocity_scale=velocity_name_for_scaling)


   def define_fields(self):
      remaining:Expression=Expression(1)
      remaining_test:Expression | None=None
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

   def get_diffusion_coefficient(self,f1:str,f2:str | None=None) -> ExpressionNumOrNone:
      if f2 is None:
         f2=f1
      if self.fluid_props is not None:
         return self.fluid_props.get_diffusion_coefficient(f1,f2,default=0)
      if f1!=f2:
         raise RuntimeError("Implement mixed diffusion between "+str(f1)+" and "+str(f2) )
      return self.diffusivity

   # ---- hooks of the stabilization mixin -------------------------------------------------------

   def stabilized_fieldnames(self) -> list[str]:
      return self.fieldnames

   def stabilization_wind(self) -> ExpressionOrNum:
      return self.wind

   def stabilization_residual_scale(self,fieldname:str) -> ExpressionOrNum:
      return 1  # this equation has no density/capacity factor in front of the time derivative

   def stabilization_diffusivity(self,fieldname:str) -> ExpressionOrNum:
      # tau only needs a representative scalar diffusivity, i.e. the diagonal entry. Cross-diffusion
      # still enters the strong residual, just not the parameter.
      D=self.get_diffusion_coefficient(self.component_names.get(fieldname,fieldname))
      assert D is not None
      return D

   def strong_residual(self,fieldname:str) -> Expression:
      f=var(fieldname)
      # partial_t defaults to ALE="auto" and hence is already the *Eulerian* derivative, exactly as
      # assembled. The mesh velocity therefore appears only in convective_velocity(), which sets the
      # streamline direction and the cell Peclet number. Note that for dt_factor != 1 the two differ
      # by (dt_factor-1)*u_mesh.grad(f): the strong residual mirrors the assembly, not the physics.
      R:Expression=self.dt_factor*partial_t(f)
      conservative=self.stab_cfg.conservative_residual
      if conservative=="auto" and self.advection_by_parts=="skew":
         R=R+(dot(self.wind,grad(f))+div(self.wind*f))/2
      else:
         if conservative=="auto":
            # GCL included: its transient is strongly d_t f|_E + div(w f) and its flux term adds
            # div((u-w) f), which sum to the same conservative expression as the plain by-parts one.
            conservative=bool(self.advection_by_parts) or self.GCL
         R=R+(div(self.wind*f) if conservative else dot(self.wind,grad(f)))
      R=R-convert_to_expression(self.source.get(fieldname,0))
      if self.stab_cfg.include_diffusion_in_residual:
         if self.fluid_props is not None:
            for fn2 in self.fieldnames:
               D=self.get_diffusion_coefficient(self.component_names[fieldname],self.component_names[fn2])
               assert D is not None
               R=R-div(D*grad(var(fn2)))
         else:
            R=R-div(self.diffusivity*grad(f))
      return R

   def _transient_and_advection(self,fn:str,ts:"Callable[[Expression],Expression]")->None:
      """The transient term and the advective term, which the GCL form replaces as a pair."""
      f, f_test = var_and_test(fn)
      if self.GCL:
         # d/dt of the whole integral, so that the change of the element volume is taken by the same
         # finite difference that advances the field; testing with v=1 then telescopes exactly.
         # dt_factor multiplies from OUTSIDE - inside, the history terms would carry it at their own
         # time levels. It multiplies the mesh velocity as well, since the GCL transient strongly
         # supplies dt_factor*(d_t f|_E + div(w f)) while the equation wants dt_factor*d_t f|_E +
         # div(u f). They coincide at dt_factor=1.
         self.add_residual(self.dt_factor*time_derivative_of_integral(weak(f,f_test),
                                                                      scheme=self.gcl_scheme))
         adv=convert_to_expression(self.wind)-self.dt_factor*mesh_velocity()
         if not is_zero(adv):
            self.add_residual(-weak(ts(f*adv),grad(f_test)))
         return
      self.add_residual(weak(ts(self.dt_factor*partial_t(f)),f_test))
      if self.advection_by_parts=="skew":
         self.add_residual(-weak(ts(self.wind*f),grad(f_test))/2)
         self.add_residual(weak(ts(dot(self.wind,grad(f))),f_test)/2)
      elif self.advection_by_parts:
         self.add_residual(-weak(ts(self.wind*f),grad(f_test)))
      else:
         self.add_residual(weak(ts(dot(self.wind,grad(f))),f_test))

   def define_residuals(self):
      if self.time_scheme is None:
         ts:Callable[[Expression],Expression]=lambda x :x
      else:
         ts:Callable[[Expression],Expression]=lambda x: time_scheme(cast(TimeSteppingScheme,self.time_scheme),x)
      if self.fluid_props is not None:
         for fn in self.fieldnames:
            f, f_test = var_and_test(fn)
            self._transient_and_advection(fn,ts)
            self.add_residual(-weak(ts(convert_to_expression(self.source.get(fn,0))),f_test))
            for fn2 in self.fieldnames:
               f2,_=var_and_test(fn2)
               diffuD=self.get_diffusion_coefficient(self.component_names[fn],self.component_names[fn2])
               assert diffuD is not None
               self.add_residual(weak(ts(diffuD*grad(f2)),grad(f_test)))
      else:
         for fn in self.fieldnames:
            f, f_test = var_and_test(fn)
            self._transient_and_advection(fn,ts)
            self.add_residual(-weak(ts(Expression(self.source.get(fn, 0))),f_test))
            diffuD=self.diffusivity
            self.add_residual(weak(ts(diffuD*grad(f)),grad(f_test)))
      self.add_stabilization_residuals(ts)


   # Use this to either fix the average or the total integral of the field, i.e. add eqs+=AdvectionDiffusionEquations(...).with_integral_constraint(...)
   def with_integral_constraint(self,problem:"Problem",*,average:dict[str, ExpressionOrNum] | ExpressionOrNum | None=None,integral:dict[str, ExpressionOrNum] | ExpressionOrNum | None=None,ode_domain_name:str="globals",lagrange_prefix:str | dict[str, str]="lagr_intconstr_",set_zero_on_normal_mode_eigensolve:bool=True) -> Equations:
      eq_additions:Equations=self
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
      lagr_mults:dict[str,ExpressionOrNum]={}
      lagr_names:dict[str,str]={}
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

      ode_additions=GlobalLagrangeMultiplier(**lagr_mults,set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve) #type:ignore
      problem.add_equations(ode_additions@ode_domain_name)
      return eq_additions



class AdvectionDiffusionFluxInterface(Equations):
   """
      Represents the flux through the interface that naturally arises from the integration by parts of the diffusion term in the advection-diffusion equation.

      .. note::
         Unlike ``AdvectionDiffusionInfinity`` this is a plain
         ``Equations``, not an ``InterfaceEquations``, so it cannot reach the bulk equations and
         cannot subtract their ``get_stabilization_flux``. If a bulk stabilization with
         ``natural_bc_correction`` is used, the flux imposed here is the physical one *plus* that
         footprint.

      Args:
         **kwargs: name of the flux and its value.
   """

   def __init__(self, **kwargs:ExpressionOrNum):
      super(AdvectionDiffusionFluxInterface, self).__init__()
      self.fluxes=sorted_field_kwargs(kwargs)

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
      self.inftyvals=sorted_field_kwargs(kwargs)
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
         owner=None
         for p in parents:
            assert isinstance(p,AdvectionDiffusionEquations)
            if fn in p.fieldnames:
               diffuD = p.get_diffusion_coefficient(fn)
               owner = p
               break
         if diffuD is None:
            raise RuntimeError("Cannot find any diffusion coefficient for field "+fn)
         y, y_test = var_and_test(fn)
         R = square_root(dot(d, d))
         # A stabilized bulk deposits its own flux on this boundary; subtract it so that the
         # far-field condition imposes the physical one. Zero unless natural_bc_correction is set.
         assert owner is not None
         self.add_residual(-weak(owner.get_stabilization_flux(fn,n,self.get_parent_domain()),y_test))


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
            else:
                raise RuntimeError("Far-field monopole conditions are only implemented for real_dim in {1,2,3}, not "+str(real_dim))
            self.add_residual(weak(diffuD * coordsys_dim_factor * (y - val) * dot(n, d) / (dot(d, d)) , y_test) )
            


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
