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
 
 
from ..materials.generic import AnyLiquidProperties
from ..generic import Equations, InterfaceEquations
from ..expressions import *  # Import grad et al
from ..typings import *

if TYPE_CHECKING:
    from ..materials.generic import *

class LubricationEquations(Equations):
   """
   Represents the lubrication equations, which are second order partial differential equation given by:

      dh/dt + div(-h^3 / (3 * mu)  * grad(p) + h^2 / (2 * mu) * grad(sigma)) = 0
      p + div(sigma * grad(h)) - disjoining_pressure = 0
   
   where h is the height of the liquid, p is the pressure, mu is the dynamic viscosity of the liquid, sigma is the surface tension of the liquid, and disjoining_pressure is the disjoining pressure.
   div is the divergence operator and grad is the gradient operator.

   Args:
      space (FiniteElementSpaceEnum): Finite element space. Default is "C2".
      mu (ExpressionOrNum): Dynamic viscosity of the liquid. Default is 1.0.
      disjoining_pressure (Union[ExpressionNumOrNone,Dict[str,ExpressionOrNum]]): Disjoining pressure. Default is None.
      sigma (ExpressionOrNum): Surface tension of the liquid. Default is 1.0.
      fluid_props (AnyLiquidProperties): Liquid properties. Default is None.
      use_exact_pressure (bool): Use the exact pressure. Default is False.
      height_source (ExpressionOrNum): Source term for the height equation. Default is 0.
      dt_h_factor (ExpressionOrNum): Time factor for the height equation. Default is 1.
      mu0 (ExpressionNumOrNone): Dynamic viscosity scaling. Default is None.
      sigma0 (ExpressionNumOrNone): Surface tension scaling. Default is None.
      swap_test_functions (bool): Swap test functions between pressure and height. Default is False.
      scheme (TimeSteppingScheme): Time stepping scheme. Default is "BDF2".
      pfactor (ExpressionOrNum): Multiplicative factor for pressure. Default is 1.
      use_subexpressions (bool): Use subexpressions to speed up. Might be incompatible with some features. Default is True.
   """
   def __init__(self, *, space:FiniteElementSpaceEnum="C2", mu:ExpressionOrNum=1.0, disjoining_pressure:Union[ExpressionNumOrNone,Dict[str,ExpressionOrNum]]=None, sigma:ExpressionOrNum=1, fluid_props:Optional[AnyLiquidProperties]=None,use_exact_pressure:bool=False,height_source:ExpressionOrNum=0,dt_h_factor:ExpressionOrNum=1,mu0:ExpressionNumOrNone=None,sigma0:ExpressionNumOrNone=None,swap_test_functions:bool=False,scheme:TimeSteppingScheme="BDF2", pfactor:ExpressionOrNum=1,use_subexpressions:bool=True):
      super().__init__()
      self.space:FiniteElementSpaceEnum = space
      self.mu = mu
      self.sigma = sigma
      self._use_exact_pressure=use_exact_pressure
      self.height_source=height_source
      self.dt_h_factor=dt_h_factor
      self.scheme=scheme
      self.pfactor=pfactor
      self.use_subexpressions=use_subexpressions

      if fluid_props is not None:
         self.fluid_props = fluid_props
         sigm=fluid_props.default_surface_tension["gas"]
         assert sigm is not None
         self.sigma =sigm 
         
         self.mu = fluid_props.dynamic_viscosity
      else:
         self.fluid_props = None

      
      if isinstance(disjoining_pressure, dict):
         self.disjoining_pressure = self.build_disjoining_pressure(**disjoining_pressure)
      else:
         self.disjoining_pressure = disjoining_pressure

      self.mu0=mu0
      self.sigma0=sigma0
      self.swap_test_functions=swap_test_functions

   def __define_scaling(self):
      super().define_scaling()
      scaling:Dict[str,ExpressionOrNum]={}
      if self.sigma0 is not None:
         scaling["surface_tension"] = self.sigma0
      elif self.fluid_props is not None:
         scaling["surface_tension"] = self.fluid_props.evaluate_at_condition(self.sigma,self.fluid_props.initial_condition)
      if self.mu0 is not None:
         scaling["dynamic_viscosity"] = self.mu0
      elif self.fluid_props is not None:
         scaling["dynamic_viscosity"] = self.fluid_props.evaluate_at_condition(self.mu,self.fluid_props.initial_condition)
      scaling["pressure"] = self.get_scaling("surface_tension") * self.get_scaling("height") / self.get_scaling("spatial") ** 2  # Laplace pressure scale
      problem=self.get_current_code_generator()._problem 
      assert problem is not None
      if not ("temporal" in problem.scaling.keys()):
         problem.scaling["temporal"] = 3 * self.get_scaling("dynamic_viscosity") * self.get_scaling("spatial") ** 4 / (self.get_scaling("surface_tension") * self.get_scaling("height") ** 3)

   def build_disjoining_pressure(self, *, precursor_height:ExpressionOrNum, contact_angle:ExpressionOrNum, n:int=3, m:int=2)->Expression:
      h = var("height")
      invh = precursor_height / h
      B = self.sigma / precursor_height * (n - 1) * (m - 1) / (n - m) * (1 - cos(contact_angle))
      return -B * (invh ** n - invh ** m)

   def define_fields(self):
      TSH=scale_factor("temporal")/(scale_factor("height"))
      TSP=1/scale_factor("pressure")
      if self.swap_test_functions:
         TSH,TSP=TSP,TSH
      self.define_scalar_field("height", self.space,testscale=TSH)
      self.define_scalar_field("pressure", self.space,testscale=TSP)

   def define_residuals(self):
      h, h_test = var_and_test("height")
      p, p_test = var_and_test("pressure")
      p *= self.pfactor
      if self.swap_test_functions:
         h_test,p_test=p_test,h_test

      # Pressure Equation
      disjoin = 0
      if self.disjoining_pressure:
         if self.use_subexpressions:
            disjoin = subexpression(self.disjoining_pressure)
         else:
            disjoin = self.disjoining_pressure
      self.add_residual(weak(time_scheme(self.scheme, p - disjoin), p_test))
      if self._use_exact_pressure:
         self.add_residual(- weak(time_scheme(self.scheme,self.sigma * grad(h) / square_root(1 + dot(grad(h), grad(h)))), grad(p_test)) )
      else:
         self.add_residual( - weak(time_scheme(self.scheme,self.sigma * grad(h)), grad(p_test)))

      # Height equation
      self.add_residual(weak(time_scheme(self.scheme,self.dt_h_factor*partial_t(h)) , h_test))
      self.add_residual(weak(time_scheme(self.scheme,(1 / self.mu) * (h ** 3 / 3 * grad(p) - h ** 2 / 2 * grad(self.sigma))), grad(h_test)))
      self.add_residual(-weak(time_scheme(self.scheme,self.height_source),h_test))



class LubricationBoundary(InterfaceEquations):
   """
      Represents the Neumann boundary condition for the lubrication equations, given by:

         dot(n,sigma * grad(h)) = 0

      Optionally, the exact pressure can be used, transforming the boundary condition to:

         dot(n,sigma * grad(h)/sqrt(1+dot(grad(h),grad(h)))) = 0
      
      This class requires the parent equations to be of type LubricationEquations, meaning that if LubricationEquations (or subclasses) are not defined in the parent domain, an error will be raised.
   
      Args: 
         sigma (ExpressionNumOrNone): Surface tension. Default is None.
         use_exact_pressure (Optional[bool]): Use the exact pressure. Default is None.
   """

   required_parent_type = LubricationEquations
   def __init__(self,*,sigma:ExpressionNumOrNone=None,use_exact_pressure:Optional[bool]=None):
      super().__init__()
      self.sigma=sigma
      self._use_exact_pressure=use_exact_pressure

   def define_residuals(self):
      hbulk, _ = var_and_test("height", domain=self.get_parent_domain())
      _, p_test = var_and_test("pressure")
      n = var("normal")
      lubric=self.get_parent_equations()
      assert isinstance(lubric,LubricationEquations)
      sigma = self.sigma if self.sigma is not None else lubric.sigma
      gradhbulk=grad(hbulk)
      use_exact_pressure=self._use_exact_pressure if self._use_exact_pressure is not None else lubric._use_exact_pressure 
      if use_exact_pressure:
         self.add_residual(weak( dot(n, sigma * gradhbulk/square_root(1+dot(gradhbulk,gradhbulk))),p_test))
      else:
         self.add_residual(weak( dot(n, sigma * gradhbulk),p_test))

class LubricationBoundaryByLagrange(InterfaceEquations):
   """
      Represents an alternative way of imposing the Neumann boundary condition for the lubrication equations, via a Lagrange Multiplier restraining the pressure, given by:

         dot(n,grad(p)) = 0

      This class requires the parent equations to be of type LubricationEquations, meaning that if LubricationEquations (or subclasses) are not defined in the parent domain, an error will be raised.
   """
   required_parent_type = LubricationEquations
   def define_fields(self):
      lubric=self.get_parent_domain().get_equations().get_equation_of_type(LubricationEquations)
      if lubric.swap_test_functions:
         scal=1/(test_scale_factor("height")*scale_factor("spatial"))
      else:
         scal=1/(test_scale_factor("pressure")*scale_factor("spatial"))
      self.define_scalar_field("_lagr_pbulk0",lubric.space,testscale=1/scale_factor("pressure")*scale_factor("spatial"),scale=scal)

   def define_residuals(self):
      lubric=self.get_parent_domain().get_equations().get_equation_of_type(LubricationEquations)
      p,ptest=var_and_test("pressure",domain="..")
      h,htest=var_and_test("height",domain="..")
      if lubric.swap_test_functions:
         test=htest
      else:
         test=ptest
      l,ltest=var_and_test("_lagr_pbulk0")
      n=var("normal")
      self.add_residual(weak(dot(n,grad(p)),ltest))
      self.add_residual(weak(l,test))

class LubricationHeightAveragedBulkVelocity(Equations):
   """
      Represents the projection of the velocity field to the bulk, given by:

         u = -h**2/(3*mu)*grad(p)+h/(2*mu)*grad(sigma)

      where u is the velocity field, h is the height of the liquid, p is the pressure, mu is the dynamic viscosity of the liquid, and sigma is the surface tension of the liquid.      

      This class requires the parent equations to be of type LubricationEquations, meaning that if LubricationEquations (or subclasses) are not defined in the parent domain, an error will be raised.
   """
   def define_fields(self):
      self.define_vector_field("velocity","C1",testscale=1/scale_factor("velocity"))

   def define_residuals(self):
      p=var("pressure")
      h=var("height")
      u, utest = var_and_test("velocity")
      lubric=self.get_combined_equations().get_equation_of_type(LubricationEquations)
      assert isinstance(lubric,LubricationEquations)
      mu=lubric.mu
      sigma = lubric.sigma
      self.add_residual(weak(u+h**2/(3*mu)*grad(p)-h/(2*mu)*grad(sigma),utest))


class LubricationVelocityAtH(Equations):
   """
      Represents the velocity field at a given height, given by:

         u = 1 / mu * (grad(sigma) - h * grad(p)) * H + 1 / mu * grad(p) * H**2 / 2
      
      where u is the velocity field, h is the height of the liquid, p is the pressure, mu is the dynamic viscosity of the liquid, sigma is the surface tension of the liquid, and H is the height at which the velocity is calculated.
   
      Args:
         at_height (ExpressionOrNum): Height at which the velocity is calculated. Default is var("height").
         space (Optional[FiniteElementSpaceEnum]): Finite element space. Default is "C2".
         name (str): Name of the velocity field. Default is "velocity".
   """
      
   def __init__(self,at_height=var("height"),space:Optional[FiniteElementSpaceEnum]="C2",name="velocity"):
      super().__init__()
      self.at_height=at_height
      self.name=name
      self.space=space

   def define_fields(self):
      if self.space is not None:
         self.define_vector_field(self.name,self.space,scale=scale_factor("spatial")/scale_factor("temporal"),testscale=scale_factor("temporal")/scale_factor("spatial"))

   def define_residuals(self):
      lubric=self.get_combined_equations().get_equation_of_type(LubricationEquations)
      mu=lubric.mu
      h,p=var(["height","pressure"])
      sigma=lubric.sigma
      u1=1/mu*(grad(sigma)-h*grad(p))
      u2=1/mu*grad(p)
      udest=u1*self.at_height+u2/2*self.at_height**2
      if self.space is not None:
         u,utest=var_and_test(self.name)
         self.add_residual(weak(u-udest,utest))
      else:
         self.add_local_function(self.name,udest)
         

class LubricationBulkField(Equations):
   """
      Represents the evolution of a field in the bulk of the lubrication system, e.g. temperature, concentration, etc., given by:

         d(hc)/dt + div(-h**2 * c / (2*mu) * grad(sigma) + h**3 * c / (3*mu) * grad(p)) - D * h * div(grad(c)) = 0
      
      where h is the height of the liquid, c is the field, p is the pressure, mu is the dynamic viscosity of the liquid, sigma is the surface tension of the liquid, and D is the diffusion coefficient of the field.

      Args:
         name (str): Name of the field. Default is "c".
         space (FiniteElementSpaceEnum): Finite element space. Default is "C2".
         diffusion (ExpressionOrNum): Diffusion coefficient. Default is 1.
         sigma (ExpressionNumOrNone): Surface tension. Default is None.
         height_multiplied (bool): Whether the field is multiplied by the height. Default is False.
         scheme (TimeSteppingScheme): Time stepping scheme. Default is "BDF2".
   """
   def __init__(self,*,name:str="c",space:FiniteElementSpaceEnum="C2",diffusion:ExpressionOrNum=1,sigma:ExpressionNumOrNone=None,height_multiplied:bool=False,scheme:TimeSteppingScheme="BDF2"):
      super(LubricationBulkField, self).__init__()
      self.name=name
      self.space:FiniteElementSpaceEnum=space
      self.diffusion=diffusion
      self.sigma=sigma
      self.height_multiplied=height_multiplied
      self.scheme=scheme

   def define_fields(self):
      if self.height_multiplied:
         self.define_scalar_field("h_times_"+self.name,self.space,testscale=scale_factor("temporal")/(scale_factor("height")*scale_factor(self.name)),scale=scale_factor("height"))
         self.define_field_by_substitution(self.name,var("h_times_"+self.name)/var("height"),also_on_interface=True)
         self.add_local_function(self.name,var("h_times_"+self.name)/var("height"))
      else:
         self.define_scalar_field(self.name,self.space,testscale=scale_factor("temporal")/(scale_factor("height")*scale_factor(self.name)))

   def define_residuals(self):
      sigma=self.sigma
      lubric=self.get_combined_equations().get_equation_of_type(LubricationEquations)
      if sigma is None:         
         assert isinstance(lubric,LubricationEquations)
         sigma=lubric.sigma
      assert isinstance(lubric,LubricationEquations)
      mu=lubric.mu

      h=var("height")
      p=var("pressure")
      if self.height_multiplied:
         ch,c_test=var_and_test("h_times_"+self.name)
         c=ch/var("height")
      else:
         c,c_test=var_and_test(self.name)
         ch=c*h
      #Scale is c*h/t*<dx>
      # MaScale*
      #EQ=partial_t(c*h)*c_test*dx\
      #   -MaScale/2*h**2*c*dot(grad(sigma),grad(c_test))*dx\
      #   +EpsScale**2/3*h**3*c*dot(grad(p),grad(c_test))*dx\
      #   +self.diffusion*h*dot(grad(c),grad(c_test))*dx

      self.add_residual(weak(time_scheme(self.scheme, partial_t(ch)),c_test))
      self.add_residual(weak(time_scheme(self.scheme,-h*ch/(2*mu)*grad(sigma)),grad(c_test)))
      self.add_residual(weak(time_scheme(self.scheme,h**2*ch/(3*mu)*grad(p)),grad(c_test)))
      self.add_residual(weak(time_scheme(self.scheme,self.diffusion* h*grad(c)),grad(c_test)))
      #self.add_residual(EQ)



class LubricationEquationOnNavierStokes(Equations):
   """
      To be used in combination with Navier-Stokes equations, this class represents the lubrication equations on the Navier-Stokes free interface, given by:

      TODO

      Args:
         dynamic_viscosity (ExpressionOrNum): Dynamic viscosity of the liquid. Default is 1.0.
         sigma_top (ExpressionOrNum): Surface tension of the liquid at the top interface. Default is 1.0.
         disjoining_pressure (ExpressionOrNum): Disjoining pressure. Default is 0.
         normal (ExpressionNumOrNone): Normal vector. Default is None.
         add_traction_to_ns (bool): Whether to add the traction to the Navier-Stokes equations. Default is True.
         gravity (ExpressionOrNum): Gravity vector. Default is 0.
   """
   def __init__(self, dynamic_viscosity:ExpressionOrNum, sigma_top:ExpressionOrNum, disjoining_pressure:ExpressionOrNum=0,normal:ExpressionNumOrNone=None, add_traction_to_ns:bool=True,gravity:ExpressionOrNum=0):
      super(LubricationEquationOnNavierStokes, self).__init__()
      self.mu = subexpression(dynamic_viscosity)
      self.sigma = subexpression(sigma_top)
      self.add_ns_traction = add_traction_to_ns
      self.disjoining_pressure=disjoining_pressure
      self.gravity=gravity
      self.normal = normal

   def define_fields(self):
      self.define_scalar_field("hdrop", "C2")  # Height of the droplet itself (h_top - h_bottom)
      self.define_scalar_field("htop","C2")  # Absolute height of the upper interface (h_top), mainly for output and calcing the surface tension
      self.define_scalar_field("ptop", "C2")  # Pressure contribution from the top interface

   def define_residuals(self):
      if self.normal is None:
         normal = [0.0] * self.get_nodal_dimension()
         normal[-1] = 1.0
         normal = vector(*normal)
      else:
         normal = self.normal

      dx = self.get_dx(use_scaling=False)

      hd, hd_test = var_and_test("hdrop")
      htop, htop_test = var_and_test("htop")
      pt, pt_test = var_and_test("ptop")

      H = self.get_scaling("hdrop")
      T=self.get_scaling("temporal")
      P=self.get_scaling("ptop")

      # Bottom tangential velocity
      ubottom = var("velocity")
      ub_tang = ubottom - dot(normal, ubottom) * normal

      # NOTE The lateral velocity in as function of the height reads:
      #       u=u0+1/mu*grad(p)*(z**2/2-hd*z) + 1/mu*grad(sigma)*z + u0
      # for z from 0 to hd, i.e. relative height starting at hbottom and ending at htop
      # with u0=ub_tang

      # Arg of the divergence
      div_inner = 1 / (2 * self.mu) * hd ** 2 * grad(self.sigma) - 1 / (3 * self.mu) * hd ** 3 * grad(pt) + hd * ub_tang

      # Height equation
      self.add_residual(T / H * ((partial_t(hd, ALE=False)) * hd_test * dx - dot(div_inner, grad(hd_test)) * dx))

      # Calculating the top height
      ipos = var("mesh")
      y = dot(normal, ipos)
      self.add_residual(1 / H * ((htop - hd - y) * htop_test) * dx)

      # Laplace pressure from the top interface
      self.add_residual(1 / P * (pt * pt_test * dx - self.sigma * dot(grad(htop), grad(pt_test)) * dx))
      self.add_residual(1 / P * (-self.disjoining_pressure) * pt_test * dx )

      # Add normal and tangential traction
      if self.add_ns_traction:
         utest = testfunction("velocity")
         self.add_residual((1 / P) * (pt-self.disjoining_pressure+dot(self.gravity,normal)*hd) * dot(normal, utest) * dx)  # Laplace pressure from top interface
         # Tangential shear
         tang_shear = -hd * grad(pt) + grad(self.sigma)
         tang_test = utest - dot(normal, utest) * normal
         self.add_residual((1 / P) * dot(-tang_shear, tang_test) * dx)  # Shear connection


