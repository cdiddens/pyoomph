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
 
 
"""
A module to map a rectangular domain [R1,R2] x [Phi1,Phi2] to the polar coordinates (r,phi).
By using e.g. ``set_coordinate_system(rectangular_to_polar)``, a ``RectangularQuadMesh(size=[1,2*pi])`` becomes a unit circle. 
"""


from ..expressions.coordsys import BaseCoordinateSystem
from ..expressions import pi,nondim,var,Expression,ExpressionOrNum,vector,diff,matrix,scale_factor,testfunction,test_scale_factor
from ..output.plotting import PlotTransform
from ..meshes.meshdatacache import MeshDataCacheOperatorBase

from ..typings import *
import numpy

class PlotTransformPolarToCartesian(PlotTransform):
    """
    Transform the plots from polar coordinates in rectangular mesh to cartesian coordinates.
    """
    def __init__(self):
        super(PlotTransformPolarToCartesian, self).__init__()

    def apply(self, coordinates:NPFloatArray,values:Optional[NPFloatArray])->Tuple[NPFloatArray,NPFloatArray]:
        cs=coordinates.copy()
        if values is not None and len(values.shape)>1:
            vecdim=values.shape[0]
        else:
            vecdim=None
        if vecdim is not None and len(values.shape)>2:
            tensdim=values.shape[1]
        else:
            tensdim=None
        
        # Convert R, theta coordinates to x,y coordinates
        R, theta = cs[0].copy(), cs[1].copy()  # type:ignore
        cs[0] = R * numpy.cos(theta)  # type:ignore
        cs[1] = R * numpy.sin(theta)  # type:ignore

        # Convert vector values into new coordinate system
        if vecdim is not None and vecdim>1:
            # Assuming values[0] is the x-component and values[1] is the y-component
            ur, utheta = values[0].copy(), values[1].copy()  # type:ignore
            values[0] = ur * numpy.cos(theta) - utheta * numpy.sin(theta)
            values[1] = ur * numpy.sin(theta) + utheta * numpy.cos(theta)
        return numpy.array(cs),numpy.array(values) #type:ignore

class RectangularToPolarMappingCoordinateSystem(BaseCoordinateSystem):
    def __init__(self):
        super().__init__()
        self.cartesian_error_estimation:bool = False

    def get_actual_dimension(self, reduced_dim:int)->int:
        return reduced_dim
        
    def get_id_name(self)->str:
        return "rectangular_to_polar"        

    def volumetric_scaling(self, spatial_scale:ExpressionOrNum, elem_dim:int)->ExpressionOrNum:
        return spatial_scale ** elem_dim 
    
    def integral_dx(self, nodal_dim:int, edim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:
        if edim >= 3:
            raise RuntimeError("Rectangular does not work for dimension " + str(edim)) # TODO: Could be extended in a third direction
        if lagrangian:
            if with_scale:
                return spatial_scale ** edim * nondim("lagrangian_x") * nondim("dX")
            else:
                return nondim("lagrangian_x") * nondim("dX")
        else:
            if with_scale:
                return spatial_scale ** edim * 2 * pi * nondim("coordinate_x") * nondim("dx")
            else:
                return nondim("coordinate_x") * nondim("dx")    
            
    def get_r_and_phi(self,with_scales,lagrangian):
        coords=self.get_coords(3, with_scales=False, lagrangian=lagrangian)
        r=coords[0]*(scale_factor("spatial") if with_scales else 1)
        phi=coords[1]
        return r, phi
            
    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:        
        if ndim!=2:
            raise RuntimeError("Rectangular to polar mapping only works for 2D")
        r,phi=self.get_r_and_phi(with_scales,lagrangian)
        res:List[ExpressionOrNum] = [diff(arg,r),1/r*diff(arg,phi)]            
        return vector(res)            
    
    
    def vector_gradient_dimension(self, basedim:int, lagrangian:bool=False)->int:
        # Just alwas 3
        return 3    
    
    def geometric_jacobian(self)->Expression:
        if self.cartesian_error_estimation:
            return Expression(1)
        else:
            return nondim("coordinate_x")

    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        if ndim!=2:
            raise RuntimeError("Rectangular to polar mapping only works for 2D")        
        r,phi = self.get_r_and_phi(with_scales,lagrangian)        
        Ar=arg[0]
        Aphi=arg[1]
        res:List[List[ExpressionOrNum]] = [[diff(Ar, r), 1/r*diff(Ar, phi)-Aphi/r, 0],
                   [diff(Aphi, r), diff(Aphi, phi)/r+Ar/r, 0], [0, 0, 0]]
        return matrix(res)    
    
    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations")->Tuple[List[Expression],List[Expression],List[str]]:
        if ndim!=2:
            raise RuntimeError("Rectangular to polar mapping only works for 2D")        
        zero=Expression(0)
        s = scale_factor(name)
        S = test_scale_factor(name)                  
        vx = element.define_scalar_field(name + "_x", space)
        vy = element.define_scalar_field(name + "_y", space)
        vx = var(name + "_x")
        vy = var(name + "_y")
        element.set_scaling(**{name + "_x": name, name + "_y": name})
        element.set_test_scaling(**{name + "_x": name, name + "_y": name})          
        return [vx / s, vy / s, zero], [testfunction(name + "_x") / S,testfunction(name + "_y") / S, zero], [name + "_x",name + "_y"]
        

    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        if ndim!=2:
            raise RuntimeError("Rectangular to polar mapping only works for 2D")        
        
        r,phi=self.get_r_and_phi(with_scales,lagrangian)
        res = diff(r*arg[0], r)/r + 1/r*diff(arg[1], phi)
        return res
    
    def define_tensor_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations", symmetric:bool)->Tuple[List[List[Expression]],List[List[Expression]],List[List[str]]]:
        raise RuntimeError("Rectangular to polar mapping does not support tensors yet")
    
    def tensor_divergence(self, T:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Rectangular to polar mapping does not support tensors yet")
    
    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int,with_scales:bool,)->Expression:        
        raise RuntimeError("Rectangular to polar mapping does not support tensors yet")


class MeshDataPolarToCartesian(MeshDataCacheOperatorBase):
    def apply(self, base):
        
        rs=base.nodal_values[:,base.nodal_field_inds["coordinate_x"]].copy()
        phis=base.nodal_values[:,base.nodal_field_inds["coordinate_y"]].copy()
        
        base.nodal_values[:,base.nodal_field_inds["coordinate_x"]]=rs*numpy.cos(phis)
        base.nodal_values[:,base.nodal_field_inds["coordinate_y"]]=rs*numpy.sin(phis)
        for _vfield,compos in base.vector_fields.items():
            ur=base.nodal_values[:,base.nodal_field_inds[compos[0]]].copy()
            uphi=base.nodal_values[:,base.nodal_field_inds[compos[1]]].copy()
            base.nodal_values[:,base.nodal_field_inds[compos[0]]]=ur*numpy.cos(phis) - uphi*numpy.sin(phis)
            base.nodal_values[:,base.nodal_field_inds[compos[1]]]=ur*numpy.sin(phis) + uphi*numpy.cos(phis)

rectangular_to_polar=RectangularToPolarMappingCoordinateSystem()
