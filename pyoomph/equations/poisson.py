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
 
 
from ..generic import Equations, InterfaceEquations
from ..meshes.bcs import AutomaticNeumannCondition
from ..expressions import * #Import grad et al



#Define a bulk element that solves the Poisson equation
class PoissonEquation(Equations):
    """
    Represents the Poisson equation in the form:
        
        -div(coeff*grad(u)) = f
            
    where u is the dependent variable, coeff is the coefficient, and f is the source term.
        
    Args:
        name(str): The name of the dependent variable. Default is "u".
        space(FiniteElementSpaceEnum): The finite element space. Default is "C2", i.e. second order continuous Lagrangian elements.
        source(ExpressionNumOrNone): The source term. Default is None.
        coefficient(ExpressionOrNum): The coefficient. Default is 1.
        DG_alpha(ExpressionOrNum): The penalty parameter for the interior facet terms in case of use of Discontinuous Galerkin discretisation. Default is 1.
    """

    def __init__(self,name:str="u",*,space:"FiniteElementSpaceEnum"="C2",source:ExpressionNumOrNone=None,coefficient:ExpressionOrNum=1,DG_alpha:ExpressionOrNum=1):
        super().__init__()
        self.name=name
        self.space:"FiniteElementSpaceEnum"=space
        self.source=source
        self.coefficient=coefficient
        # DG implementation requires facets
        self.requires_interior_facet_terms=is_DG_space(self.space,allow_DL_and_D0=True)
        self.DG_alpha=DG_alpha

    def get_information_string(self) -> str:
        return "-div(<coeff>*grad(<"+self.name+">))=<source> with <coeff>="+str(self.coefficient)+" and <source>="+str(self.source)

    #First step: Define the fields solved by this element
    def define_fields(self):
        self.define_scalar_field(self.name,self.space,testscale=1/scale_factor(self.name)) #u is second order continuous

    #Second step: Add the contribution to the residuals, i.e. the weak form equations
    def define_residuals(self):
        u,u_test=var_and_test(self.name) #u and u_test are now the shape function expansion of u and the corresponding test function (Galerkin)
        self.add_residual(weak(self.coefficient*grad(u),grad(u_test)))
        if self.source is not None:
            self.add_residual(-weak(self.source,u_test))
        if self.requires_interior_facet_terms!=is_DG_space(self.space,allow_DL_and_D0=True):
            raise RuntimeError("You apparently changed the space of this equation manually, not in the constructor")
        if self.requires_interior_facet_terms:
            order=get_order_of_space(self.space)
            stab=1 if order==0 else self.DG_alpha*(order+1)*order
            n=var("normal") # facet normal
            h=var("cartesian_element_length_h") # element size
            facet_res=-weak(jump(self.coefficient*u)*n,avg(grad(u_test)))
            facet_res+=-weak(avg(self.coefficient*grad(u)),jump(u_test)*n) 
            facet_res+= weak(stab/avg(h)*jump(self.coefficient*u)*n,jump(u_test)*n)
            self.add_interior_facet_residual(facet_res)

    def get_weak_dirichlet_terms_for_DG(self,fieldname:str,value:ExpressionOrNum)->ExpressionNumOrNone:
        if not self.requires_interior_facet_terms:
            return None # Continuous spaces are imposed strongly
        if fieldname==self.name:
            domain=".." # Seen from the Dirichlet interface, it must be the bulk domain due to the grads
            u,u_test=var_and_test(self.name,domain=domain)
            u_test/=scale_factor("spatial") # But we have to correct the spatial scale for the test function
            n=var("normal")
            h=var("cartesian_element_length_h",domain=domain) # element size
            order=get_order_of_space(self.space)
            stab=1 if order==0 else self.DG_alpha*(order+1)*order
            return -weak(self.coefficient*(u-value)*n,grad(u_test)) -weak(self.coefficient*grad(u),u_test*n)+ weak(self.coefficient*stab/h*(u-value),u_test)

        
        


class PoissonFlux(AutomaticNeumannCondition):

    required_parent_type = PoissonEquation
    neumann_sign = -1



class PoissonFarFieldMonopoleCondition(InterfaceEquations):
    """
    Represents a far-field condition for the Poisson equation in the form:

    .. math::
        u + R \dfrac{\partial u}{\partial r} = u_{\infty}

    where u is the dependent variable, far_value is the value at infinity, and R is the distance from the origin and r is the radial coordinate.
    Hence, works only correctly in axisymmetric or 3D.
    For 2D, we use:

    .. math::
        u + R \dfrac{\partial u}{\partial r} / \log(R / L) = u_{\infty}

    Due to the logarithmic solution behavior in 2D, the farfield length L must be provided here.

    This class requires the parent equations to be of type PoissonEquation, meaning that if PoissonEquation (or subclasses) are not defined in the parent domain, an error will be raised.
            
    Args:
        far_value(ExpressionOrNum): The value at infinity. Default is 0.
        name(str): The name of the dependent variable. Default is None, meaning that the name of the parent Poisson equation is used.
        coefficient(ExpressionOrNum): The coefficient. Default is None, meaning that the coefficient of the parent Poisson equation is used.
        origin(ExpressionOrNum): The origin of the far-field condition. Default is (0,0,0).
        farfield_length(ExpressionNumOrNone): The far-field length scale L used in the 2D case. Default is None, meaning that no length scale is used and the condition is applied as in 3D.
    """
    required_parent_type = PoissonEquation

    def __init__(self,far_value:ExpressionOrNum=0,name:Optional[str]=None,coefficient:ExpressionNumOrNone=None,origin:ExpressionOrNum=vector([0]),farfield_length:ExpressionNumOrNone=None):
        super(PoissonFarFieldMonopoleCondition, self).__init__()
        self.far_value=far_value
        self.name=name
        self.coefficient=coefficient
        self.origin=origin
        self.farfield_length=farfield_length

    def define_residuals(self):
        n=self.get_normal()
        d=var("coordinate")-self.origin
        name=self.name
        parent=self.get_parent_equations()
        if name is None:
            assert isinstance(parent,PoissonEquation)
            name=parent.name
        coefficient = self.coefficient
        if coefficient is None:
            assert isinstance(parent,PoissonEquation)
            coefficient = parent.coefficient
        c,c_test=var_and_test(name)
        R = square_root(dot(d, d))
        # 2D-axisymmetric (i.e. 3D) case
        if isinstance(self.get_coordinate_system(),AxisymmetricCoordinateSystem) or isinstance(self.get_coordinate_system(),AxisymmetryBreakingCoordinateSystem) or (isinstance(self.get_coordinate_system(),CartesianCoordinateSystem) and self.get_nodal_dimension() == 3):
            self.add_residual(weak(coefficient * (c - self.far_value) * dot(n,d)/dot(d,d),c_test))
        # 2D case
        else:
            if self.farfield_length is None:
                raise RuntimeError("For 2D far-field monopole conditions, a farfield_length must be provided")
            self.add_residual(weak(coefficient * (c - self.far_value) * dot(n,d)/(dot(d,d)*log(R/self.farfield_length)),c_test))


class DiffusionEquation(PoissonEquation):
    """
    Represents the diffusion equation in the form:
            
            partial_t(u) - div(coeff*grad(u)) = f
                
    where u is the dependent variable, coeff is the diffusivity, and f is the source term.

    This class is a subclass of PoissonEquation and inherits all its arguments.                    

    Args:
        name(str): The name of the dependent variable. Default is "u".
        space(FiniteElementSpaceEnum): The finite element space. Default is "C2", i.e. second order continuous Lagrangian elements.
        source(ExpressionNumOrNone): The source term. Default is None.
        diffusivity(ExpressionOrNum): The diffusivity. Default is 1.
        temporal_factor(ExpressionOrNum): The temporal factor. Default is 1.
    """

    def __init__(self, *, name:str="u", space:FiniteElementSpaceEnum="C2", source:ExpressionNumOrNone=None,diffusivity:ExpressionOrNum=1,temporal_factor:ExpressionOrNum=1):
        super(DiffusionEquation, self).__init__(name=name,space=space,source=source,coefficient=diffusivity)
        self.temporal_factor=temporal_factor

    def define_residuals(self):
        super(DiffusionEquation, self).define_residuals()
        u,u_test=var_and_test(self.name) #u and u_test are now the shape function expansion of u and the corresponding test function (Galerkin)
        dx=self.get_dx()									#and the integral dx size
        self.add_residual(self.temporal_factor*partial_t(u)*u_test*dx)	#Add the residual