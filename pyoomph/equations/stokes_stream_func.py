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
 

from ..generic import Equations,InterfaceEquations
from ..expressions import * #Import grad et al
from ..expressions.coordsys import AxisymmetricCoordinateSystem

#Only works in axisymmetric and 2d cartesian
class StreamFunctionFromVelocity(Equations):
    """
        This class defines the equations for the stream function of a velocity field in 2D Cartesian or axisymmetric coordinates.
        The stream function is defined as the scalar field phi such that u = (dphi/dy,-dphi/dx) in 2D Cartesian coordinates or
        u = (-1/r dphi/dz,dphi/dr) in axisymmetric coordinates.
        
        Args:
            name (str): Name of the stream function field. Default is "streamfunc".
            space (FiniteElementSpaceEnum): The finite element space for the stream function. Default is "C2".
            velocity (Expression): The velocity field for which the stream function is defined. Default is var("velocity").
            with_error_estimator (bool): If True, an error estimator is added to the equations. Default is True.
            axisymmetric (Union[Literal["auto"],bool]): If True, the equations are defined in axisymmetric coordinates. If "auto", the equations are defined in axisymmetric coordinates if the coordinate system is axisymmetric. Default is "auto".
            DG_alpha (ExpressionOrNum): The penalty parameter for the DG formulation of the stream function. If the stream function is defined in a DG space, this parameter is used to penalize jumps in the stream function across element boundaries. Default is 10.
    """
    def __init__(self,*,name:str="streamfunc",space:FiniteElementSpaceEnum="C2",velocity:Expression=var("velocity"),with_error_estimator:bool=True,axisymmetric:Union[Literal["auto"],bool]="auto",DG_alpha:ExpressionOrNum=10):
        super(StreamFunctionFromVelocity, self).__init__()
        self.space:FiniteElementSpaceEnum=space
        self.name=name
        self.DG_alpha=DG_alpha
        self.velocity=velocity
        self.with_error_estimator=with_error_estimator
        self.axisymmetric=axisymmetric        
        self.requires_interior_facet_terms=is_DG_space(self.space,allow_DL_and_D0=True)
        
        # Old formulation (axisymm only) stems from minimization of I=integral[ 2*pi*r*F*dr*dz ] with
        #   F=(dP/dz+r*u_r)^2+(dP/dr-r*u_z)^2
        # New formulation considers
        #   F=(1/r*dP/dz+u_r)^2+(1/r*dP/dr-u_z)^2
        # instead
        self.old_formulation=False 

    def get_axisymmetry_flag_and_dir(self,coordsys:OptionalCoordinateSystem=None):
        if coordsys is None:
            coordsys=self.get_coordinate_system()
        if self.axisymmetric=="auto":
            if isinstance(coordsys,AxisymmetricCoordinateSystem):
                return True,"y" if coordsys.use_x_as_symmetry_axis else "x"
            else:
                return False,"x"
        elif self.axisymmetric==True:
            if isinstance(coordsys,AxisymmetricCoordinateSystem):
                return True,"y" if coordsys.use_x_as_symmetry_axis else "x"
            else:
                return True,"x"
        else:
            return False,"x"

    def define_fields(self):
        self.define_scalar_field(self.name,self.space)

    def define_scaling(self):
        axisymm,_=self.get_axisymmetry_flag_and_dir()
        X,U=scale_factor("spatial"),scale_factor("velocity")
        scals={self.name:X**(2 if axisymm else 1)*U}
        self.set_scaling(**scals)
        tscals={self.name:1/U}
        #if axisymm and not self.old_formulation:
        tscals[self.name]*=X
        self.set_test_scaling(**tscals)

    def define_residuals(self):
        phi,phi_test=var_and_test(self.name)
        u=self.velocity
        axisymm,direct=self.get_axisymmetry_flag_and_dir()
        r=var("coordinate_"+direct)
        
        if not axisymm:
            self.add_weak(-grad(phi), grad(phi_test))
            self.add_weak(-(partial_x(u[1]) - partial_y(u[0])),phi_test)
        elif self.old_formulation:            
            self.add_weak(-grad(phi), grad(phi_test))
            self.add_weak(-(r * (partial_x(u[1]) - partial_y(u[0])) + 2 * u[1 if direct=="x" else 0]),phi_test)
        else:
            self.add_weak(-grad(phi)/r, grad(phi_test))            
            self.add_weak( -((partial_x(u[1]) - partial_y(u[0])) +  u[1 if direct=="x" else 0]/r),phi_test)
        
        if is_DG_space(self.space,allow_DL_and_D0=True):
            alpha=self.DG_alpha
            n=var("normal")
            h=var("cartesian_element_length_h")
            if not axisymm or self.old_formulation:
                facet_res=- weak(alpha/avg(h)*jump(phi)*n,jump(phi_test)*n)
                facet_res+=weak(jump(phi)*n,avg(grad(phi_test)))
                facet_res+=weak(avg(grad(phi)),jump(phi_test)*n) 
            else:
                facet_res=- weak(alpha/(r*avg(h))*jump(phi)*n,jump(phi_test)*n)
                facet_res+=weak(jump(phi)*n/r,avg(grad(phi_test)))
                facet_res+=weak(avg(grad(phi))/r,jump(phi_test)*n) 
                #raise RuntimeError("TODO New formulation")
            
            self.add_interior_facet_residual(facet_res)
            

    def define_error_estimators(self):
        if self.with_error_estimator:
            self.add_spatial_error_estimator(grad(nondim(self.name),nondim=True))


    def get_weak_dirichlet_terms_for_DG(self, fieldname: str, value: ExpressionOrNum) -> ExpressionNumOrNone:
        if fieldname!=self.name or not self.requires_interior_facet_terms:
            return None
        phi,phi_test=var_and_test(self.name,domain="..")
        n=var("normal")
        alpha=self.DG_alpha
        axisymm,direct=self.get_axisymmetry_flag_and_dir()
        h=var("cartesian_element_length_h",domain="..")
        extra_factor=1 if (not axisymm or self.old_formulation) else 1/(var("coordinate_"+direct)*scale_factor("spatial"))
        facet_res=- weak(extra_factor*alpha/h*(phi-value)*n,phi_test*n)
        facet_res+=weak(extra_factor*(phi-value)*n,grad(phi_test))
        facet_res+=weak(extra_factor*grad(phi),phi_test*n)             
        return facet_res
        
        

#Add on open interfaces
class StreamFunctionFromVelocityInterface(InterfaceEquations):
    """
        This class defines the Neuamnn boundary conditions for the calculation of the stream function from the velocity field in 2D Cartesian or axisymmetric coordinates.

        This class requires the parent equations to be of type StreamFunctionFromVelocity, meaning that if StreamFunctionFromVelocity (or subclasses) are not defined in the parent domain, an error will be raised.
    """
    required_parent_type = StreamFunctionFromVelocity
    def define_residuals(self):
        bulk=self.get_parent_equations()
        if not isinstance(bulk,StreamFunctionFromVelocity):
            raise RuntimeError(self.__class__.__name__+" is not attached to an interface on a bulk domain with equations of type "+"StokesStreamFunctionFromVelocity")
        _,phi_test=var_and_test(bulk.name,domain="..") # To work with D0
        phi_test/=scale_factor("spatial")
        u=bulk.velocity
        axisymm,direct=bulk.get_axisymmetry_flag_and_dir()
        r=var("coordinate_"+direct)
        n=self.get_normal()
        if axisymm and bulk.old_formulation:
            self.add_residual(weak(r * (n[0] * u[1] - n[1] * u[0]),phi_test))
        else:
            self.add_residual(weak(n[0] * u[1] - n[1] * u[0],phi_test))

