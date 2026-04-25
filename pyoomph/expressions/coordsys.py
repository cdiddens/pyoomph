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
from .generic import *

from ..typings import *
from ..expressions import *


if TYPE_CHECKING:
    from ..generic.codegen import Equations,FiniteElementCodeGenerator,FiniteElementSpaceEnum


pi = 2 * _pyoomph.GiNaC_asin(1)


class BaseCoordinateSystem(_pyoomph.CustomCoordinateSystem):
    
    expansion_eps=_pyoomph.GiNaC_new_symbol("eps") # One epsilon to rule them all, one epsilon to find them, one epsilon to bring them all and in the darkness bind them
    # This is used to expand the normal mode in the eigenvalue problem, but if you have e.g. some Cartesian coordinates on in an azimuthal eigenproblem, you want to set epsilon to zero to calculate e.g. the grad(f) in Caretsian, which otherwise does not work
    has_normal_mode_expasion:bool=False # If False, it removes all epsilon terms in the coordinates
    
    def __init__(self):
        super(BaseCoordinateSystem, self).__init__()
        self.x_rel_scale=1
        self.y_rel_scale=1
        self.z_rel_scale=1

    def map_to_zero_epsilon(self,input):
        if isinstance(input,list):
            return [self.map_to_zero_epsilon(i) for i in input]
        else:
            return  _pyoomph.GiNaC_SymSubs(input,self.expansion_eps,Expression(0))
        
    def map_to_first_order_epsilon(self,input,with_epsilon:bool=True):
        if isinstance(input,list):
            return [self.map_to_first_order_epsilon(i,with_epsilon) for i in input]
        else:
            return _pyoomph.GiNaC_SymSubs(diff(_pyoomph.GiNaC_expand(input), self.expansion_eps), self.expansion_eps, Expression(0))*(self.expansion_eps if with_epsilon else 1)            


    def define_scalar_field(self, name:str, space:"FiniteElementSpaceEnum", element:"Equations", scale:Optional[ExpressionOrNum]=None, testscale:Optional[ExpressionOrNum]=None, discontinuous_refinement_exponent:Optional[ExpressionOrNum]=None):
        element._internal_define_scalar_field(name, space, scale=scale, testscale=testscale, discontinuous_refinement_exponent=discontinuous_refinement_exponent)

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations")->Tuple[List[Expression],List[Expression],List[str]]:
        raise RuntimeError("Implement the define_vector_field function for this coordinate system")

    def define_tensor_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations",symmetric:bool)->Tuple[List[List[Expression]],List[List[Expression]],List[List[str]]]:
        raise RuntimeError("Implement the define_tensor_field function for this coordinate system")


    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the vector_gradient for this coordinate system. Occured upon taking the grad of " + str(arg))

    # Dimension of the vector gradient matrix for dimension dim
    def vector_gradient_dimension(self, basedim:int, lagrangian:bool=False)->int:
        return basedim  # Normally identity

    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the vector_divergence for this coordinate system. Occured upon taking the div of " + str(arg))

    def tensor_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the tensor_divergence for this coordinate system. Occured upon taking the div of " + str(arg))


    def geometric_jacobian(self)->Expression:
        raise RuntimeError("Implement the geometric_jacobian for this coordinate system")

    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the scalar_gradient for this coordinate system. Occured upon taking the grad of " + str(arg))

    def surface_scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        return self.scalar_gradient(arg, ndim, edim, with_scales, lagrangian)

    def surface_vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        return self.vector_gradient(arg, ndim, edim, with_scales, lagrangian)

    def surface_vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        return self.vector_divergence(arg, ndim, edim, with_scales, lagrangian)

    def grad(self, arg:Expression, ndim:int, edim:int, flags:int) -> Expression:
        with_scales = (flags & 1 != 0)
        lagrangian = (flags & 8 != 0)
        is_vector=0
        if flags & 6 == 0:
            if _pyoomph.GiNaC_is_a_matrix(arg):
                is_vector = 2
            else:
                is_vector = 1
        if flags & 6 == 2 or is_vector == 1:
            if ndim != edim:
                return self.surface_scalar_gradient(arg, ndim, edim, with_scales, lagrangian)
            else:
                return self.scalar_gradient(arg, ndim, edim, with_scales, lagrangian)
        elif flags & 6 == 4 or is_vector == 2:
            if ndim == edim:
                return self.vector_gradient(arg.evalm(), ndim, edim, with_scales, lagrangian)
            elif ndim == edim + 1:
                return self.surface_vector_gradient(arg.evalm(), ndim, edim, with_scales, lagrangian)
            else:
                raise RuntimeError("Trying to perform a vector gradient on an element with dimension " + str(
                    edim) + " but with a nodal dimension of " + str(ndim))
        else:
            raise RuntimeError("Strange flags in grad: ", flags, flags & 6)



    def get_normal_vector_or_component(self,cg:"FiniteElementCodeGenerator",component:Optional[int]=None,only_base_mode:bool=False,only_perturbation_mode:bool=False,where:str="Residual"):
        dim = cg.get_nodal_dimension()
        if component is None:
            posscompos=[nondim("normal_"+d,domain=cg,only_base_mode=only_base_mode,only_perturbation_mode=only_perturbation_mode) for d in ["x","y","z"]]
            mycomps=posscompos[:dim]        
            return vector(*mycomps)
        else:
            return cg._get_normal_component(component)            
        
        

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int)->Expression:
        raise RuntimeError("Implement the directional tensor derivative for this coordinate system")


    def directional_derivative(self,arg:Expression,direct:Expression,ndim:int,edim:int,flags:int)->Expression:
        with_scales = (flags & 1 != 0)
        lagrangian = (flags & 8 != 0)
        vectorial=(flags & 4 !=0)
        tensorial=(flags &16 !=0)
        if tensorial:
            return self.directional_tensor_derivative(arg,direct,lagrangian,with_scales,ndim,edim,with_scales)
        elif vectorial:
            return matproduct(grad(arg,lagrangian=lagrangian,nondim=not with_scales),direct)
        else:
            return dot(grad(arg,lagrangian=lagrangian,nondim=not with_scales),direct)
    



    def div(self, arg:Expression, ndim:int, edim:int, flags:int)->Expression:
        with_scales = (flags % 2 == 1)
        lagrangian = (flags & 8 != 0)
        tensorial=(flags & 16 != 0)
        #		is_vector=(flags//2)	#TODO

        if tensorial:
            return self.tensor_divergence(arg.evalm(),ndim,edim,with_scales,lagrangian)
        elif edim == ndim:
            return self.vector_divergence(arg.evalm(), ndim, edim, with_scales, lagrangian)
        elif ndim == edim + 1:
            return self.surface_vector_divergence(arg.evalm(), ndim, edim, with_scales, lagrangian)
        else:
            raise RuntimeError("Cannot calculate this divergence yet: element dimension "+str(edim)+" and nodal dimension "+str(ndim))

    def volumetric_scaling(self, spatial_scale:ExpressionOrNum, elem_dim:int)->ExpressionOrNum:
        return spatial_scale ** elem_dim

    def integral_dx(self, nodal_dim:int, edim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:        
        if lagrangian:
            if with_scale:
                return spatial_scale ** (edim) * nondim("dX" )
            else:
                return nondim("dX")
        else:
            if with_scale:
                return spatial_scale ** (edim) * nondim("dx" )
            else:
                return nondim("dx")

    def get_coords(self, ndim:int, with_scales:bool, lagrangian:bool,mesh_coords:bool=False,only_base_mode:bool=False) -> List[Expression]:
        rel_scales = [self.x_rel_scale, self.y_rel_scale, self.z_rel_scale]
        if lagrangian:
            if with_scales:
                x, y, z = var(["lagrangian_x", "lagrangian_y", "lagrangian_z"]) 
                x,y,z=rel_scales[0]*x,rel_scales[1]*y,rel_scales[2]*z
            else:
                x, y, z = nondim(["lagrangian_x", "lagrangian_y", "lagrangian_z"])
        else:            
            if with_scales:
                if mesh_coords:
                    x, y, z = var(["mesh_x", "mesh_y", "mesh_z"])
                else:
                    x, y, z = var(["coordinate_x", "coordinate_y", "coordinate_z"],only_base_mode=only_base_mode) # In case combined with something else
                x,y,z=rel_scales[0]*x,rel_scales[1]*y,rel_scales[2]*z
            else:
                if mesh_coords:
                    x, y, z = nondim(["mesh_x", "mesh_y", "mesh_z"])
                else:
                    x, y, z = nondim(["coordinate_x", "coordinate_y", "coordinate_z"],only_base_mode=only_base_mode) # In case combined with something else
                
        all_coords = [x, y, z]
        if not self.has_normal_mode_expasion:
            all_coords = self.map_to_zero_epsilon(all_coords)
        if ndim == 1:
            return [x]
        else:
            return all_coords[:ndim]


    def get_id_name(self)->str:
        return "<unknown coord.sys>"

    def get_actual_dimension(self, reduced_dim:int)->int:
        return reduced_dim

    def jacobian_for_element_size(self)->Expression:
        return self.geometric_jacobian()


    def expand_coordinate_or_mesh_vector(self,cg:"FiniteElementCodeGenerator", name:str,dimensional:bool,no_jacobian:bool,no_hessian:bool):        
        dim=cg.get_nodal_dimension()
        if dimensional:
            vr=lambda n : var(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
        else:
            vr=lambda n : nondim(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
        if dim == 1:
            return vector([vr(name+"_x")])
        elif dim == 2:
            return vector([vr(name+"_x"), vr(name+"_y")])
        elif dim == 3:
            return vector([vr(name+"_x"), vr(name+"_y"), vr(name+"_z")])

#####################


class ODECoordinateSystem(BaseCoordinateSystem):
    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool) -> Expression:
        return Expression(0)

    def geometric_jacobian(self)->Expression:
        return Expression(1)

    def expand_coordinate_or_mesh_vector(self,cg:"FiniteElementCodeGenerator", name:str,dimensional:bool,no_jacobian:bool,no_hessian:bool):        
        return Expression(0)
        #raise RuntimeError("ODEs don't have coordinates")

# Cartesian
class CartesianCoordinateSystem(BaseCoordinateSystem):
    def __init__(self,x_rel_scale:ExpressionOrNum=1,y_rel_scale:ExpressionOrNum=1,z_rel_scale:ExpressionOrNum=1):
        super().__init__()
        self.x_rel_scale:ExpressionOrNum=x_rel_scale
        self.y_rel_scale:ExpressionOrNum=y_rel_scale
        self.z_rel_scale:ExpressionOrNum=z_rel_scale

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations")->Tuple[List[Expression],List[Expression],List[str]]:
        inds = ["x", "y", "z"]
        namelist:List[str] = []
        for i in range(ndim):
            namelist.append(name + "_" + inds[i])
        v:List[Expression] = []
        vtest:List[Expression] = []
        s = scale_factor(name)
        S = test_scale_factor(name)
        for i, f in enumerate(namelist):
            if i >= ndim: break
            element.set_scaling(**{f: name})
            vc = element.define_scalar_field(f, space)
            vc = var(f)
            v.append(vc / s)
            element.set_test_scaling(**{f: name})
            vtest.append(testfunction(f) / S)
        return v, vtest, namelist


    def define_tensor_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations", symmetric:bool)->Tuple[List[List[Expression]],List[List[Expression]],List[List[str]]]:
        inds = ["x", "y", "z"]
        namelist:List[List[str]] = []
        for i in range(ndim):
            nlst:List[str]=[]
            for j in range(ndim):
                if symmetric and j<i:
                    nlst.append(name + "_" + inds[j]+ inds[i])
                else:
                    nlst.append(name + "_" + inds[i]+ inds[j])
            namelist.append(nlst)
        v:List[List[Expression]] = []
        vtest:List[List[Expression]] = []
        s = scale_factor(name)
        S = test_scale_factor(name)
        for i, fl in enumerate(namelist):
            if i >= ndim: break
            vl:List[Expression]=[]
            vtl:List[Expression]=[]
            for j, f in enumerate(fl):
                if j >= ndim: break
                if symmetric and j<i:
                    vl.append(v[j][i])
                    vtl.append(vtest[j][i])
                else:
                    element.set_scaling(**{f: name})
                    vc = element.define_scalar_field(f, space)
                    vc = var(f)
                    vl.append(vc / s)
                    element.set_test_scaling(**{f: name})
                    vtl.append(testfunction(f) / S)
            v.append(vl)
            vtest.append(vtl)
        #print("NL",namelist)
        #exit()
        #if symmetric:
        #    for i, fl in enumerate(namelist):
        #        for j, f in enumerate(fl):
        #            if j<i:
        #                namelist[i].remove(f)
        return v, vtest, namelist


    def geometric_jacobian(self)->Expression:
        return rational_num(1, 1)

    def get_id_name(self)->str:
        return "Cartesian"

    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[ExpressionOrNum] = []
        for a in self.get_coords(ndim, with_scales, lagrangian):
            res.append(diff(arg, a))
        if len(res) == 0:
            return Expression(0)
        return vector(res)

    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:Expression = Expression(0)
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        for i in range(arg.nops()):
            res += diff(arg[i], coords[i])
        return res
    
    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:        
        res:List[List[ExpressionOrNum]] = []
        for b in range(arg.nops()):
            line:List[ExpressionOrNum] = []
            entry = arg[b]
            for a in self.get_coords(ndim, with_scales, lagrangian):
                line.append(diff(entry, a))
            res.append(line)
        return matrix(res)
    
    def tensor_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[Expression] = []
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        for i in range(3):
            div_line = 0 
            for j,coord in enumerate(coords):
                entry=arg[j,i]
                div_line+=diff(entry, coord)
            res.append(div_line)
        return vector(res)

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int,with_scales:bool)->Expression:
        res:List[List[ExpressionOrNum]] = [[0]*3 for _x in range(3)]
        coords=self.get_coords(ndim, with_scales, lagrangian)
        for i in range(3):
            for j in range(3):
                for k,coord in enumerate(coords):                       
                    res[i][j]+=diff(T[i,j],coord)*direct[k]
        return matrix(res)



# Axisymmetric

class AxisymmetricCoordinateSystem(BaseCoordinateSystem):
    def __init__(self,r_rel_scale:ExpressionOrNum=1,z_rel_scale:ExpressionOrNum=1,cartesian_error_estimation:bool=False,use_x_as_symmetry_axis:bool=False,):
        super().__init__()
        self.x_rel_scale=r_rel_scale
        self.y_rel_scale=z_rel_scale
        self.cartesian_error_estimation=cartesian_error_estimation
        self.use_x_as_symmetry_axis=use_x_as_symmetry_axis

    def get_actual_dimension(self, reduced_dim:int)->int:
        return reduced_dim + 1

    def get_id_name(self)->str:
        return "Axisymmetric"

    def volumetric_scaling(self, spatial_scale:ExpressionOrNum, elem_dim:int)->ExpressionOrNum:
        return spatial_scale ** (elem_dim + 1)

    def integral_dx(self, nodal_dim:int, edim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:
        if edim >= 3:
            raise RuntimeError("Axisymmetry does not work for dimension " + str(edim))
        edim_offs=edim+1
        if lagrangian:
            if with_scale:
                return spatial_scale ** edim_offs * 2 * pi * nondim("lagrangian_y" if self.use_x_as_symmetry_axis else "lagrangian_x") * nondim("dX")
            else:
                return 2 * pi * nondim("lagrangian_y" if self.use_x_as_symmetry_axis else "lagrangian_x") * nondim("dX")
        else:
            if with_scale:
                return spatial_scale ** edim_offs * 2 * pi * nondim("coordinate_y" if self.use_x_as_symmetry_axis else "coordinate_x") * nondim("dx")
            else:
                return 2 * pi * nondim("coordinate_y" if self.use_x_as_symmetry_axis else "coordinate_x") * nondim("dx")

    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[ExpressionOrNum] = []
        for i, a in enumerate(self.get_coords(3, with_scales, lagrangian)):
            res.append(diff(arg, a) if i < ndim else 0)
        return vector(res)

    def vector_gradient_dimension(self, basedim:int, lagrangian:bool=False)->int:
        # Just alwas 3
        return 3

    def geometric_jacobian(self)->Expression:
        if self.cartesian_error_estimation:
            return Expression(1)
        else:
            return 2 * pi * nondim("coordinate_y" if self.use_x_as_symmetry_axis else "coordinate_x")

    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        if ndim >= 3:
            raise RuntimeError("Vector gradient in axisymmetry does not work for dimension " + str(ndim))
        if ndim == 1:
            if self.use_x_as_symmetry_axis:
                raise RuntimeError("Cannot have use_x_as_symmetry_axis in an axisymmetric coordinate system in 1d")
            r, = self.get_coords(ndim, with_scales, lagrangian)
            res:List[List[ExpressionOrNum]] = [[diff(arg[0], r), 0, 0], [0, 0, 0], [0, 0, arg[0] / r]]
        else:
            if arg.nops() != 3:
                raise RuntimeError(
                    "Cannot take a 2d axisymmetric vector gradient from a vector with dim!=2:  " + str(arg))
            x, y = self.get_coords(ndim, with_scales, lagrangian)
            r= y if self.use_x_as_symmetry_axis else x
            ri = 1 if self.use_x_as_symmetry_axis else 0
            res:List[List[ExpressionOrNum]] = [[diff(arg[0], x), diff(arg[0], y), 0],
                   [diff(arg[1], x), diff(arg[1], y), 0], [0, 0, arg[ri] / r]]
        return matrix(res)
    

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations")->Tuple[List[Expression],List[Expression],List[str]]:
        zero=Expression(0)
        s = scale_factor(name)
        S = test_scale_factor(name)          
        if ndim == 2:
            vx = element.define_scalar_field(name + "_x", space)
            vy = element.define_scalar_field(name + "_y", space)
            vx = var(name + "_x")
            vy = var(name + "_y")
            element.set_scaling(**{name + "_x": name, name + "_y": name})
            element.set_test_scaling(**{name + "_x": name, name + "_y": name})          
            return [vx / s, vy / s, zero], [testfunction(name + "_x") / S,testfunction(name + "_y") / S, zero], [name + "_x",name + "_y"]
        elif ndim == 1:
            vx = element.define_scalar_field(name + "_x", space)
            vx = var(name + "_x")
            element.set_scaling(**{name + "_x": name})
            element.set_test_scaling(**{name + "_x": name})
            return [vx / s, zero], [testfunction(name + "_x") / S], [name + "_x"]
        else:
            raise RuntimeError("Axisymmetric vector fields do not work for dimension " + str(ndim))

    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res = Expression(0)
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        nops = arg.nops()
        if nops >= 1:
            res += diff(arg[0], coords[0]) + (arg[1] / coords[1] if self.use_x_as_symmetry_axis  else arg[0] / coords[0])
        if nops >= 2:
            res += diff(arg[1], coords[1])
        return res
    
    def define_tensor_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations", symmetric:bool)->Tuple[List[List[Expression]],List[List[Expression]],List[List[str]]]:
        s = scale_factor(name)
        S = test_scale_factor(name)
        if ndim==1:
            txx = element.define_scalar_field(name + "_xx", space)
            txx = var(name + "_xx")
            taa = element.define_scalar_field(name + "_aa", space)
            taa = var(name + "_aa")
            element.set_scaling(**{name + "_xx": name,name+"_aa":name})
            element.set_test_scaling(**{name + "_xx": name, name + "_aa": name})
            return [[txx / s, 0, 0], [0, 0, 0], [0, 0, taa/s]], [[testfunction(name + "_xx") / S, 0, 0], [0, 0, 0], [0, 0, testfunction(name + "_aa") / S]], [[name + "_xx", 0,0], [0,0, 0], [0, 0, name + "_aa"]] 
        elif ndim==2:
            txx = element.define_scalar_field(name + "_xx", space)
            txx = var(name + "_xx")
            tyy = element.define_scalar_field(name + "_yy", space)
            tyy = var(name + "_yy")
            txy = element.define_scalar_field(name + "_xy", space)
            txy = var(name + "_xy")
            element.set_scaling(**{name + "_xx": name, name + "_yy": name, name + "_xy": name})
            element.set_test_scaling(**{name + "_xx": name, name + "_yy": name, name + "_aa": name, name + "_xy": name})
            taa = element.define_scalar_field(name + "_aa", space)
            taa = var(name + "_aa")
            element.set_scaling(**{name + "_aa": name})
            element.set_test_scaling(**{name + "_aa": name})
        if not symmetric:
            tyx = element.define_scalar_field(name + "_yx", space)
            tyx = var(name + "_yx")
            element.set_scaling(**{name + "_yx": name})
            element.set_test_scaling(**{name + "_yx": name})
            return [[txx / s, txy / s, 0], [tyx / s, tyy / s, 0], [0, 0, taa / s]], [[testfunction(name + "_xx") / S, testfunction(name + "_xy") / S, 0], [testfunction(name + "_yx") / S, testfunction(name + "_yy") / S, 0], [0, 0, testfunction(name + "_aa") / S]], [[name + "_xx", name + "_xy", 0], [name + "_yx", name + "_yy", 0], [0, 0, name + "_aa"]] 
        else:
            return [[txx / s, txy / s, 0], [txy / s, tyy / s, 0], [0, 0, taa / s]], [[testfunction(name + "_xx") / S, testfunction(name + "_xy") / S, 0], [testfunction(name + "_xy") / S, testfunction(name + "_yy") / S, 0], [0, 0, testfunction(name + "_aa") / S]], [[name + "_xx", name + "_xy", 0], [name + "_xy", name + "_yy", 0], [0, 0, name + "_aa"]] 


    def tensor_divergence(self, T:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        coords = self.get_coords(T.nops(), with_scales, lagrangian)
        if self.use_x_as_symmetry_axis:
            if ndim==1:
                raise RuntimeError("Cannot have use_x_as_symmetry_axis in an axisymmetric coordinate system in 1d")
            else:
                div_x=diff(T[0,0],coords[0])+diff(T[1,0],coords[1])+T[1,0] / coords[1] 
                div_y=diff(T[0,1],coords[0])+diff(T[1,1],coords[1])+(T[1,1]-T[2,2]) / coords[1]
                div_theta=diff(T[0,2],coords[0])+diff(T[1,2],coords[1])+(T[1,2] - T[2,1]) / coords[1]
            return vector(div_x,div_y,div_theta)
        else:
            div_x=diff(T[0,0],coords[0])+(T[0,0] - T[2,2]) / coords[0]
            div_y=diff(T[0,1],coords[0])+T[0,1] / coords[0]
            div_theta=diff(T[0,2],coords[0])+(T[0,2] - T[2,0]) / coords[0]
            if ndim==2:
                div_x+=diff(T[1,0],coords[1])
                div_y+= diff(T[1,1],coords[1])
                div_theta+=diff(T[1,2],coords[1])
            return vector(div_x, div_y, div_theta)

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int,with_scales:bool,)->Expression:        
        if ndim==1:
            if self.use_x_as_symmetry_axis:
                raise RuntimeError("Cannot have use_x_as_symmetry_axis in an axisymmetric coordinate system in 1d")
            res:List[List[ExpressionOrNum]] = [[0]*3 for _x in range(3)]
            coords = self.get_coords(1, with_scales, lagrangian)
            res[0][0]=diff(T[0,0],coords[0])*direct[0]
            res[0][2]=1/coords[0]*(T[0,0]-T[2,2])*direct[2]
            res[2][0]=res[0][2]
            res[2][2]=diff(T[2,2],coords[0])*direct[0]

            return matrix(res)
        elif ndim==2:
            if self.use_x_as_symmetry_axis:
                #raise RuntimeError("TODO")
                res:List[List[ExpressionOrNum]] = [[0]*3 for _x in range(3)]
                coords = self.get_coords(T.nops(), with_scales, lagrangian)
                #diff_tensor_theta=[[-T[2,0]-T[0,2], T[2,1], T[0,0]-T[2,2]], [-T[1,2], 0, T[1,0]], [T[0,0]-T[2,2], T[0,1], T[0,2]+T[2,0]]]
                diff_tensor_theta=[[0, -T[0,2], T[1,0]], [-T[2,0], -T[2,1]-T[1,2], T[1,1]-T[2,2]], [T[1,0], T[1,1]-T[2,2], T[1,2]+T[2,1]]]
                for i in range(3):
                    for j in range(3):
                        res[i][j]=direct[0]*diff(T[i,j],coords[0])+direct[1]*diff(T[i,j],coords[1])+direct[2]/coords[1]*diff_tensor_theta[i][j]
                return matrix(res)
            else:
                res:List[List[ExpressionOrNum]] = [[0]*3 for _x in range(3)]
                coords = self.get_coords(T.nops(), with_scales, lagrangian)
                diff_tensor_theta=[[-T[2,0]-T[0,2], T[2,1], T[0,0]-T[2,2]], [-T[1,2], 0, T[1,0]], [T[0,0]-T[2,2], T[0,1], T[0,2]+T[2,0]]]
                for i in range(3):
                    for j in range(3):
                        res[i][j]=direct[0]*diff(T[i,j],coords[0])+direct[1]*diff(T[i,j],coords[1])+direct[2]/coords[0]*diff_tensor_theta[i][j]
                return matrix(res)
        else:
            raise RuntimeError("Not possible")


class CartesianCoordinateSystemWithAdditionalNormalMode(CartesianCoordinateSystem):
    has_normal_mode_expasion = True
    def __init__(self,normal_mode:Expression,map_real_part_of_mode0:bool=False):
        super().__init__()
        self.imaginary_i=_pyoomph.GiNaC_imaginary_i()
        self.normal_mode=normal_mode
        
        self.k_symbol = _pyoomph.GiNaC_new_symbol("k_normal_mode")
        self.xadd = _pyoomph.GiNaC_new_symbol("xadd_normal_mode")
        self.field_mode=_pyoomph.GiNaC_FakeExponentialMode(self.imaginary_i*self.k_symbol*self.xadd)
        # Mode of the test functions
        self.test_mode=_pyoomph.GiNaC_FakeExponentialMode(-self.imaginary_i*self.k_symbol*self.xadd)
        self.map_real_part_of_mode0=map_real_part_of_mode0 #Normally not necessary
        
        self.expand_with_modes_for_python_debugging=True # If you want to use expand_expression_for_debugging, set this
        
        self.with_normal_component_in_mesh_coordinates=False # Actually, quite important!


    def get_wavenumber_k(self,dimensional:bool=True):
        return self.k_symbol/(scale_factor("spatial") if dimensional else 1)
    
    def get_additional_coordinate(self,dimensional:bool=True):
        return self.xadd*(scale_factor("spatial") if dimensional else 1)
    
    def get_mode_expansion_of_var_or_test(self,code:_pyoomph.FiniteElementCode,fieldname:str,is_field:bool,is_dim:bool,expr:Expression,where:str,expansion_mode:int)->Expression:
        if where!="Residual" and (not self.expand_with_modes_for_python_debugging or where!="Python"):
            return expr # Don't do this for integral expressions, fluxes, initial and dirichlets
        
        base_flag=1
        pert_flag=1
        if expansion_mode!=0:
            if expansion_mode==-1:
                pert_flag=0
                expr=_pyoomph.GiNaC_eval_at_expansion_mode(expr,_pyoomph.Expression(0))
            elif expansion_mode==-2:
                base_flag=0
                expr=_pyoomph.GiNaC_eval_at_expansion_mode(expr,_pyoomph.Expression(0))
            
        
        ignore_fields={"time","lagrangian_x","lagrangian_y","lagrangian_z","local_coordinate_1","local_coordinate_2","local_coordinate_3"}
        if not code._coordinates_as_dofs:
            ignore_fields=ignore_fields.union({"coordinate_x","coordinate_y","coordinate_z","mesh_x","mesh_y","mesh_z"})
        if fieldname in ignore_fields:
            return expr # Do not modify these
        
        # Each field and test function are expanded with a mode perturbation
        if is_field:
            # Expand fields as Ubase+epsilon*Upert*exp(I*m*phi)            
            return base_flag*expr+pert_flag*self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(expr,_pyoomph.Expression(1))*self.field_mode
        else:
            # Test functions are not required to be expanded. They only enter linearly
            return expr*self.test_mode

    def map_residual_on_base_mode(self,residual:Expression)->Expression:
        zero=_pyoomph.Expression(0)
        # The base equations are obtained by setting epsilon and m to zero
        res=_pyoomph.GiNaC_SymSubs(_pyoomph.GiNaC_SymSubs(residual,self.k_symbol,zero),self.expansion_eps,zero)
        if self.map_real_part_of_mode0:
            res=_pyoomph.GiNaC_get_real_part(res)
        return res

    def _map_residal_on_additional_eigenproblem(self,residual:Expression,re_im_mapping:Callable[[Expression],Expression])->Expression:
        zero = _pyoomph.Expression(0)
        first_order_in_eps = _pyoomph.GiNaC_SymSubs(diff(_pyoomph.GiNaC_expand(residual), self.expansion_eps), self.expansion_eps, zero)
        replaced_m = _pyoomph.GiNaC_SymSubs(first_order_in_eps, self.k_symbol, self.normal_mode)        
        res=re_im_mapping(_pyoomph.GiNaC_split_subexpressions_in_real_and_imaginary_parts(replaced_m))
        return 0+res
        #return _pyoomph.GiNaC_collect_common_factors(res)

    def map_residual_on_normal_mode_eigenproblem_real(self,residual:Expression)->Expression:
        real_part=_pyoomph.GiNaC_get_real_part
        #print("MAPPING REAL",residual)
        #print(self._map_residal_on_additional_eigenproblem(residual,real_part))
        #print("EXPANDED",_pyoomph._currently_generated_element().expand_placeholders(self._map_residal_on_additional_eigenproblem(residual,real_part),True))
        #print("DONE MAPPING REAL")
        return self._map_residal_on_additional_eigenproblem(residual,real_part)

    def map_residual_on_normal_mode_eigenproblem_imag(self,residual:Expression)->Expression:
        imag_part=_pyoomph.GiNaC_get_imag_part
        return self._map_residal_on_additional_eigenproblem(residual,imag_part)

    def get_id_name(self)->str:
        raise RuntimeError("Is this required?")
        return "Cartesian"    

    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[ExpressionOrNum] = []
        dcoords=self.map_to_zero_epsilon(self.get_coords(3, with_scales, lagrangian))
        pcoords=self.map_to_first_order_epsilon(self.get_coords(3, with_scales, lagrangian,mesh_coords=True))
        xadd=self.get_additional_coordinate(with_scales)
        for i, a in enumerate(dcoords):
            if i < ndim:    
                res.append(diff(arg, a))
            elif i == ndim:                
                if not lagrangian:
                    res.append(diff(arg, xadd))
                else:
                    res.append(0) 
            else:
                res.append(0)
                
        if not lagrangian:
            mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")
            Xk=pcoords[0]                        
            k=self.get_wavenumber_k(dimensional=with_scales)
            I=self.imaginary_i
            x=dcoords[0]
            if ndim==2:
                Yk=pcoords[1]
                y=dcoords[1]
                res[0]+=mm*(-diff(Xk, x)*diff(arg, x) - diff(Yk, x)*diff(arg, y) )
                res[1]+=mm*(-diff(Xk, y)*diff(arg, x) - diff(Yk, y)*diff(arg, y)  )                
                res[2]+=mm*( I*k*(-Xk*diff(arg, x) - Yk*diff(arg, y)) )
            elif ndim==1:
                res[0]+=mm*(-diff(Xk, x)*diff(arg, x))
                res[1]+=mm*(-I*k*Xk*diff(arg, x)) # XXX Not according to Duarte
                pass
            elif ndim==0:
                pass
            else:
                raise RuntimeError("Not implemented")
   
        return vector(res)
    
    

    def vector_divergence(self, arg: _pyoomph.Expression, ndim: int, edim: int, with_scales: bool, lagrangian: bool) -> _pyoomph.Expression:
        I=self.imaginary_i
        k=self.get_wavenumber_k(dimensional=with_scales)
        dcoords=self.map_to_zero_epsilon(self.get_coords(3, with_scales, lagrangian))
        pcoords=self.map_to_first_order_epsilon(self.get_coords(3, with_scales, lagrangian,mesh_coords=True))
        Xk=pcoords[0]
        x=dcoords[0]
        xadd=self.get_additional_coordinate(with_scales)
        mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")*(1 if not lagrangian else 0)
        if ndim==0:
            return diff(arg[0], xadd) # TODO: Test this
        elif ndim==1:
            if edim==0:     
                # TODO Such things might be problematic when you e.g. calculate div(var("u",domain="..")). 
                # If ".." has edim=ndim, then this expression will likely be evaluated with edim=ndim-1, which is not what we want
                return diff(arg[1], xadd)*(1 if not lagrangian else 0) + mm*( I*Xk*k*diff(arg[0], xadd) )
            elif edim==1:                
                return diff(arg[0], x) + diff(arg[1], xadd)*(1 if not lagrangian else 0) + mm*( -I*Xk*k*diff(arg[1], x) - diff(Xk, x)*diff(arg[0], x) )
        elif ndim==2:
            Yk=pcoords[1]
            y=dcoords[1]
           
            if edim==0:
                return diff(arg[2], xadd)*(1 if not lagrangian else 0) + mm*( I*Xk*k*diff(arg[0], xadd) + I*Yk*k*diff(arg[1], xadd) )
            elif edim==1:
                res=diff(arg[0], x) + diff(arg[1], y) + diff(arg[2], xadd)*(1 if not lagrangian else 0) #+ mm * (    )
                mmterm=0
                Xk1,Xk2=pcoords[0],pcoords[1]
                sadd=xadd
                u1,u2,u3=arg[0],arg[1],arg[2]
                X01=self.map_to_zero_epsilon(var("coordinate_x"))
                X02=self.map_to_zero_epsilon(var("coordinate_y"))
                s1=var("local_coordinate_1")
                mmterm+=I*k*Xk1*diff(u1, sadd) + I*k*Xk2*diff(u2, sadd) - I*k*Xk1*diff(X01, s1)**2*diff(u1, sadd)/(diff(X01, s1)**2 + diff(X02, s1)**2) - I*k*Xk1*diff(X01, s1)*diff(X02, s1)*diff(u2, sadd)/(diff(X01, s1)**2 + diff(X02, s1)**2) - I*k*Xk1*diff(X01, s1)*diff(u3, s1)/(diff(X01, s1)**2 + diff(X02, s1)**2) - I*k*Xk2*diff(X01, s1)*diff(X02, s1)*diff(u1, sadd)/(diff(X01, s1)**2 + diff(X02, s1)**2) - I*k*Xk2*diff(X02, s1)**2*diff(u2, sadd)/(diff(X01, s1)**2 + diff(X02, s1)**2) - I*k*Xk2*diff(X02, s1)*diff(u3, s1)/(diff(X01, s1)**2 + diff(X02, s1)**2) + 2*diff(0, s1)*diff(X01, s1)**2*diff(u1, s1)/(diff(X01, s1)**4 + 2*diff(X01, s1)**2*diff(X02, s1)**2 + diff(X02, s1)**4) + 2*diff(0, s1)*diff(X01, s1)*diff(X02, s1)*diff(u1, s1)/(diff(X01, s1)**4 + 2*diff(X01, s1)**2*diff(X02, s1)**2 + diff(X02, s1)**4) + 2*diff(0, s1)*diff(X01, s1)*diff(X02, s1)*diff(u2, s1)/(diff(X01, s1)**4 + 2*diff(X01, s1)**2*diff(X02, s1)**2 + diff(X02, s1)**4) + 2*diff(0, s1)*diff(X02, s1)**2*diff(u2, s1)/(diff(X01, s1)**4 + 2*diff(X01, s1)**2*diff(X02, s1)**2 + diff(X02, s1)**4) - 2*diff(X01, s1)**2*diff(Xk1, s1)*diff(u1, s1)/(diff(X01, s1)**4 + 2*diff(X01, s1)**2*diff(X02, s1)**2 + diff(X02, s1)**4) - 2*diff(X01, s1)*diff(X02, s1)*diff(Xk1, s1)*diff(u2, s1)/(diff(X01, s1)**4 + 2*diff(X01, s1)**2*diff(X02, s1)**2 + diff(X02, s1)**4) - 2*diff(X01, s1)*diff(X02, s1)*diff(Xk2, s1)*diff(u1, s1)/(diff(X01, s1)**4 + 2*diff(X01, s1)**2*diff(X02, s1)**2 + diff(X02, s1)**4) - 2*diff(X02, s1)**2*diff(Xk2, s1)*diff(u2, s1)/(diff(X01, s1)**4 + 2*diff(X01, s1)**2*diff(X02, s1)**2 + diff(X02, s1)**4) - diff(0, s1)*diff(u1, s1)/(diff(X01, s1)**2 + diff(X02, s1)**2) - diff(0, s1)*diff(u2, s1)/(diff(X01, s1)**2 + diff(X02, s1)**2) + diff(Xk1, s1)*diff(u1, s1)/(diff(X01, s1)**2 + diff(X02, s1)**2) + diff(Xk2, s1)*diff(u2, s1)/(diff(X01, s1)**2 + diff(X02, s1)**2)
                res+=mm*mmterm
                return res
            elif edim==2:
                return diff(arg[0], x) + diff(arg[1], y) + diff(arg[2], xadd)*(1 if not lagrangian else 0) + mm * ( -I*k*Xk*diff(arg[2], x) - I*k*Yk*diff(arg[2], y) - diff(Xk, x)*diff(arg[0], x) - diff(Xk, y)*diff(arg[1], x) - diff(Yk, x)*diff(arg[0], y) - diff(Yk, y)*diff(arg[1], y) )
        raise RuntimeError("Any other combinations are not implemented yet!")
        
        
    def integral_dx(self, nodal_dim:int, edim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:        
        if lagrangian:
            if with_scale:
                return spatial_scale ** (edim) * nondim("dX" )
            else:
                return nondim("dX")
        else:
            from ..expressions import square_root
            mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")
            
            dcoords=self.map_to_zero_epsilon(self.get_coords(nodal_dim, with_scale, lagrangian))
            pcoords=self.map_to_first_order_epsilon(self.get_coords(nodal_dim, with_scale, lagrangian,mesh_coords=True))
            
            dx_eps=sum([diff(pc,dc) for pc,dc in zip(pcoords,dcoords)])*nondim("dx")            
            mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")
            if with_scale:
                return spatial_scale ** (edim) * (nondim("dx") + mm*dx_eps )
            else:
                return nondim("dx")+ mm*dx_eps
    
    
    def tensor_divergence(self, arg: _pyoomph.Expression, ndim: int, edim: int, with_scales: bool, lagrangian: bool) -> _pyoomph.Expression:
        raise RuntimeError("Not implemented")
    
    def vector_gradient(self, arg: _pyoomph.Expression, ndim: int, edim: int, with_scales: bool, lagrangian: bool) -> _pyoomph.Expression:
        res:List[List[ExpressionOrNum]] = []
        # TODO MOVING COORDINATES
        dcoords=self.map_to_zero_epsilon(self.get_coords(ndim, with_scales, lagrangian))
        pcoords=self.map_to_first_order_epsilon(self.get_coords(ndim, with_scales, lagrangian,mesh_coords=True))
        k=self.get_wavenumber_k(dimensional=with_scales)
        xadd=self.get_additional_coordinate(with_scales)
        for b in range(ndim):
            line:List[ExpressionOrNum] = []
            entry = arg[b]
            for a in range(ndim):
                line.append(diff(entry, dcoords[a]))                
            if not lagrangian:
                line.append(diff(entry, xadd))
            else:
                line.append(0)
            res.append(line)
        line:List[ExpressionOrNum] = []
        if not lagrangian:
            for a in range(ndim):
                line.append(diff(arg[ndim], dcoords[a]))
            line.append(diff(arg[ndim],xadd))
        else:
            for a in range(ndim+1):        
                line.append(0)            
               
        res.append(line)
        
        if not lagrangian: 
            if edim!=ndim:
                raise RuntimeError("Vector gradient on the CartesianCoordinateSystemWithAdditionalNormalMode is not implemented for edim!=ndim. If you need it, implement it!")
            I=self.imaginary_i            
            mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")
            if ndim==1:               
                Xk=pcoords[0]
                vX,vY=arg[0],arg[1]                
                x,y=dcoords[0],xadd
                
                #res[0][0]+=mm*( I*k*Xk*diff(vY, x) + I*k*vY*diff(Xk, x) - diff(Xk, x)*diff(vX, x) )
                #res[0][1]+=mm*( I*k*Xk*diff(vY, x) + I*k*vY*diff(Xk, x) - diff(Xk, x)*diff(vX, x) )
                #res[1][0]+=mm*( -diff(Xk, x)*diff(vY, x) )
                #res[1][1]+=mm*( -I*k*Xk*diff(vY, x) )
                res[0][0]+=mm*(-diff(Xk, x)*diff(vX, x))
                res[0][1]+=mm*(-I*k*Xk*diff(vX, x))
                res[1][0]+=mm*(-diff(Xk, x)*diff(vY, x))
                res[1][1]+=mm*(-I*k*Xk*diff(vY, x))
                

            else:
                Xk,Yk=pcoords[0],pcoords[1]
                u_x,u_y,u_z=arg[0],arg[1],arg[2]
                x,y,z=dcoords[0],dcoords[1],xadd
                
                res[0][0]+=mm*( -diff(Xk, x)*diff(u_x, x) - diff(Yk, x)*diff(u_x, y) )
                res[0][1]+=mm*( -diff(Xk, y)*diff(u_x, x) - diff(Yk, y)*diff(u_x, y) )
                res[0][2]+=mm*( -I*k*Xk*1*diff(u_x, x) - I*k*Yk*diff(u_x, y) )
                res[1][0]+=mm*( -diff(Xk, x)*diff(u_y, x) - diff(Yk, x)*diff(u_y, y) )
                res[1][1]+=mm*( -diff(Xk, y)*diff(u_y, x) - diff(Yk, y)*diff(u_y, y) )
                res[1][2]+=mm*( -I*k*Xk*1*diff(u_y, x) - I*k*Yk*diff(u_y, y) )
                res[2][0]+=mm*( -(diff(Xk, x)*diff(u_z, x) + diff(Yk, x)*diff(u_z, y)) )
                res[2][1]+=mm*( -(diff(Xk, y)*diff(u_z, x) + diff(Yk, y)*diff(u_z, y)) )
                res[2][2]+=mm*( -I*k*(Xk*diff(u_z, x) + Yk*diff(u_z, y)) )                                
        
        return matrix(res)

    def tensor_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the tensor_divergence for this coordinate system. Occured upon taking the div of " + str(arg))

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int)->Expression:
        raise RuntimeError("Implement the directional tensor derivative for this coordinate system")

    def expand_coordinate_or_mesh_vector(self,cg:"FiniteElementCodeGenerator", name:str,dimensional:bool,no_jacobian:bool,no_hessian:bool):        
        if not cg._coordinates_as_dofs:
            dim=cg.get_nodal_dimension()
            if dim==0:
                return Expression(0)
            return super().expand_coordinate_or_mesh_vector(cg,name,dimensional,no_jacobian,no_hessian)
        else:
            dim=cg.get_nodal_dimension()
            if dimensional:
                vr=lambda n : var(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
            else:
                vr=lambda n : nondim(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
            if self.with_normal_component_in_mesh_coordinates:
                raise RuntimeError("Not implemented")
                if dim == 1:
                    return vector([vr(name+"_normal")])
                elif dim == 1:
                    return vector([vr(name+"_x"),vr(name+"_normal")])
                elif dim == 2:
                    return vector([vr(name+"_x"), vr(name+"_y"),vr(name+"_normal")])
                elif dim == 3:
                    raise RuntimeError("Cannot use normal mode expansion coordinate system in 3d")
            else:
                if dim==0:
                    return Expression(0)
                if dim == 1:
                    return vector([vr(name+"_x"),0])
                elif dim == 2:
                    return vector([vr(name+"_x"), vr(name+"_y"),0])
                elif dim == 3:
                    raise RuntimeError("Cannot use normal mode expansion coordinate system in 3d")
        
    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations") -> Tuple[List[Expression], List[Expression], List[str]]:
        inds = ["x", "y", "z"]
        namelist:List[str] = []
        if ndim==3:
            raise RuntimeError("Cannot use a normal mode in 3D")
        for i in range(ndim):
            namelist.append(name + "_" + inds[i])
        namelist.append(name + "_normal")
        
        v:List[Expression] = []
        vtest:List[Expression] = []
        s = scale_factor(name)
        S = test_scale_factor(name)
        for i, f in enumerate(namelist):
            if i >= ndim+1: break
            element.set_scaling(**{f: name})
            vc = element.define_scalar_field(f, space)
            vc = var(f)
            v.append(vc / s)
            element.set_test_scaling(**{f: name})
            vtest.append(testfunction(f) / S)
        return v, vtest, namelist
    
    def get_normal_vector_or_component(self,cg:"FiniteElementCodeGenerator",component:Optional[int]=None,only_base_mode:bool=False,only_perturbation_mode:bool=False,where:str="Residual"):
        dim = cg.get_nodal_dimension()
        edim=cg.get_element_dimension()
        if not cg._coordinates_as_dofs or (where!="Residual" and (not self.expand_with_modes_for_python_debugging or where!="Python")):
            return super().get_normal_vector_or_component(cg,component,only_base_mode,only_perturbation_mode,where=where)
        
        if dim not in [1,2]:
            RuntimeError("Cannot use this coordinate system with a nodal dimension of "+str(dim))
        if component is None:
            comp=lambda s : nondim(s,only_base_mode=only_base_mode,only_perturbation_mode=only_perturbation_mode)
            if dim==0:
                return vector(comp("normal_x"),0,0)
            elif dim==1:
                return vector(comp("normal_x"),comp("normal_y"),0)
            else:
                return vector(comp("normal_x"),comp("normal_y"),comp("normal_z")) #TODO: Normal z here?
        else:
            # Normal expansion            
            base_factor=0 if only_perturbation_mode else 1
            pert_factor=0 if only_base_mode else 1
            
            dcoords=self.map_to_zero_epsilon(self.get_coords(dim, with_scales=True, lagrangian=False))
            pcoords=self.map_to_first_order_epsilon(self.get_coords(dim, with_scales=True, lagrangian=False,mesh_coords=True))
            
            if dim==2:
                if edim==1:
                    n0=[cg._get_normal_component(0),cg._get_normal_component(1),0]
                    Xk,Yk=pcoords[0],pcoords[1]
                    x,y=dcoords[0],dcoords[1]
                    I=self.imaginary_i
                    k=self.get_wavenumber_k(dimensional=True)
                    mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")
                    
                    #neps2=[-n0[1]*(d_by_dx(Yk)-d_by_dy(Xk))]
                    #neps2.append(n0[0]*(d_by_dx(Yk)-d_by_dy(Xk)))
                    #neps2.append(-I*k*(n0[0]*Xk+n0[1]*Yk))
                    if component==0:
                        return base_factor*n0[0]-pert_factor* mm*n0[1]*(diff(Yk,x)-diff(Xk,y))
                    elif component==1:
                        return base_factor*n0[1]+pert_factor*mm*n0[0]*(diff(Yk,x)-diff(Xk,y))
                    elif component==2:
                        return -pert_factor*mm*I*k*(n0[0]*Xk+n0[1]*Yk)
            
            if component<dim: 
                
                return base_factor*cg._get_normal_component(component) +pert_factor*self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(cg._get_normal_component(component),Expression(1))*self.field_mode
            elif component==dim:                

                #return base_factor*cg._get_normal_component(component)
                nx0=cg._get_normal_component(0)
                #rm=self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(var("mesh_x"),_pyoomph.Expression(1))*self.field_mode
                Xk=pcoords[0]
                n0_dot_xeigen=nx0*Xk
                if dim==2:
                    ny0=cg._get_normal_component(1)
                    Yk=pcoords[1]
                    #zm=self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(var("mesh_y"),_pyoomph.Expression(1))*self.field_mode
                    n0_dot_xeigen+=ny0*Yk
                if self.with_normal_component_in_mesh_coordinates:
                    raise RuntimeError("Not implemented")
                    xadd_shift=var("mesh_normal",only_perturbation_mode=True)                     
                else:
                    xadd_shift=0
                #return pert_factor*self.expansion_eps*(nr0*phim-self.imaginary_i*self.angular_mode/var("mesh_x",only_base_mode=True)*(n0_dot_xeigen))*self.field_mode
                #print(pert_factor*(nr0*phim-self.imaginary_i*self.angular_mode/var("mesh_x",only_base_mode=True)*(n0_dot_xeigen)))                
                #exit()
                #return pert_factor*(nr0*phim-self.imaginary_i*self.normal_mode/var("mesh_x",only_base_mode=True)*(n0_dot_xeigen))
                #return pert_factor*(-self.imaginary_i*self.k_symbol*(n0_dot_xeigen))
                return pert_factor*((-self.imaginary_i*self.k_symbol*n0_dot_xeigen))
                
            else:
                raise RuntimeError("Normal component "+str(component)+" not available")
        


class AxisymmetryBreakingCoordinateSystem(AxisymmetricCoordinateSystem):
    has_normal_mode_expasion=True
    def __init__(self,angular_mode:Expression,map_real_part_of_mode0:bool=False):        
        self.imaginary_i=_pyoomph.GiNaC_imaginary_i()
        super(AxisymmetryBreakingCoordinateSystem, self).__init__()
        self.angular_mode = angular_mode
        # Expansion smallness parameters epsilon: U=U_base+epsilon*U_angular
        #self.expansion_eps=_pyoomph.GiNaC_new_symbol("eps")
        # m_angular and angular variable phi.
        # We first take a generic symbol for m and replace it later with the parameter or 0 (base state)
        self.m_angular_symbol = _pyoomph.GiNaC_new_symbol("m_angular")
        self.phi = _pyoomph.GiNaC_new_symbol("phi")
        # Field mode exp(I*m*phi)
        self.field_mode=_pyoomph.GiNaC_FakeExponentialMode(self.imaginary_i*self.m_angular_symbol*self.phi)
        # Mode of the test functions
        self.test_mode=_pyoomph.GiNaC_FakeExponentialMode(-self.imaginary_i*self.m_angular_symbol*self.phi)
        self.map_real_part_of_mode0=map_real_part_of_mode0 #Normally not necessary
        self.with_phi_component_in_mesh_coordinates=False # TODO: Implement that some day
        self.expand_with_modes_for_python_debugging=False # If you want to use expand_expression_for_debugging, set this

        

    def get_mode_expansion_of_var_or_test(self,code:_pyoomph.FiniteElementCode,fieldname:str,is_field:bool,is_dim:bool,expr:Expression,where:str,expansion_mode:int)->Expression:
        if where!="Residual" and (not self.expand_with_modes_for_python_debugging or where!="Python"):
            return expr # Don't do this for integral expressions, fluxes, initial and dirichlets
        
        base_flag=1
        pert_flag=1
        if expansion_mode!=0:
            if expansion_mode==-1:
                pert_flag=0
                expr=_pyoomph.GiNaC_eval_at_expansion_mode(expr,_pyoomph.Expression(0))
            elif expansion_mode==-2:
                base_flag=0
                expr=_pyoomph.GiNaC_eval_at_expansion_mode(expr,_pyoomph.Expression(0))
            
        
        ignore_fields={"time","lagrangian_x","lagrangian_y","lagrangian_z","local_coordinate_1","local_coordinate_2","local_coordinate_3"}
        if not code._coordinates_as_dofs:
            ignore_fields=ignore_fields.union({"coordinate_x","coordinate_y","coordinate_z","mesh_x","mesh_y","mesh_z"})
        if fieldname in ignore_fields:
            return expr # Do not modify these
        
        # Each field and test function are expanded with a mode perturbation
        if is_field:
            # Expand fields as Ubase+epsilon*Upert*exp(I*m*phi)
            return base_flag*expr+pert_flag*self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(expr,_pyoomph.Expression(1))*self.field_mode
        else:
            # Test functions are not required to be expanded. They only enter linearly
            return expr*self.test_mode

    def map_residual_on_base_mode(self,residual:Expression)->Expression:
        zero=_pyoomph.Expression(0)
        # The base equations are obtained by setting epsilon and m to zero
        res=_pyoomph.GiNaC_SymSubs(_pyoomph.GiNaC_SymSubs(residual,self.m_angular_symbol,zero),self.expansion_eps,zero)
        if self.map_real_part_of_mode0:
            res=_pyoomph.GiNaC_get_real_part(res)
        return res

    def _map_residal_on_angular_eigenproblem(self,residual:Expression,re_im_mapping:Callable[[Expression],Expression])->Expression:
        zero = _pyoomph.Expression(0)
        # First order in epsilon is just derive by epsilon and set epsilon to zero afterwards
        first_order_in_eps = _pyoomph.GiNaC_SymSubs(diff(_pyoomph.GiNaC_expand(residual), self.expansion_eps), self.expansion_eps, zero)
        replaced_m = _pyoomph.GiNaC_SymSubs(first_order_in_eps, self.m_angular_symbol, self.angular_mode)        
        real_or_imag = re_im_mapping(_pyoomph.GiNaC_split_subexpressions_in_real_and_imaginary_parts(replaced_m))        
        return 0+real_or_imag
        #return _pyoomph.GiNaC_collect_common_factors(real_or_imag)

    def map_residual_on_angular_eigenproblem_real(self,residual:Expression)->Expression:
        real_part=_pyoomph.GiNaC_get_real_part
        return self._map_residal_on_angular_eigenproblem(residual,real_part)

    def map_residual_on_angular_eigenproblem_imag(self,residual:Expression)->Expression:
        imag_part=_pyoomph.GiNaC_get_imag_part
        return self._map_residal_on_angular_eigenproblem(residual,imag_part)

    def get_id_name(self)->str:
        raise RuntimeError("Is this required?")
        return "Axisymmetric"


    def integral_dx(self, nodal_dim:int, edim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:        
        
        if edim >= 3:
            raise RuntimeError("Axisymmetry does not work for dimension " + str(edim))
        edim_offs=edim+1
       
        if lagrangian:
            if with_scale:
                return spatial_scale ** edim_offs * 2 * pi * nondim("lagrangian_x") * nondim("dX")
            else:
                return 2 * pi * nondim("lagrangian_x") * nondim("dX") 
            
        
        mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")
        
        dcoords=self.map_to_zero_epsilon(self.get_coords(nodal_dim, with_scale, lagrangian))
        pcoords=self.map_to_first_order_epsilon(self.get_coords(nodal_dim, with_scale, lagrangian,mesh_coords=True))
        
        if nodal_dim==2:
            dx_eps=(dcoords[0]*diff(pcoords[0], dcoords[0]) + dcoords[0]*diff(pcoords[1], dcoords[1]) + pcoords[0])*nondim("dx")            
        elif nodal_dim==1:
            dx_eps=(dcoords[0]*diff(pcoords[0], dcoords[0])  + pcoords[0])*nondim("dx")            
        else:
            raise RuntimeError("Not implemented")
        mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")
        if with_scale:
            return 2*pi* spatial_scale ** (edim) * (dcoords[0]* nondim("dx") + mm*dx_eps )
        else:
            return 2*pi*(dcoords[0]*nondim("dx")+ mm*dx_eps)
            
    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[ExpressionOrNum] = []
        coords = self.get_coords(3, with_scales, lagrangian)
        dcoords=self.map_to_zero_epsilon(coords)
        pcoords=self.map_to_first_order_epsilon(coords)
        for i, a in enumerate(dcoords):
            if i < ndim:
                res.append(diff(arg, a))
            elif i == ndim and not lagrangian:
                res.append( diff(arg,self.phi) / dcoords[0])
            else:
                res.append(0)
        import _pyoomph
        if not lagrangian:
            mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")
            x=dcoords[0]
            Xp=pcoords[0]
            psi=arg
            phi=self.phi
            m=self.m_angular_symbol
            I=self.imaginary_i
            if ndim==1:
                res[0]+=mm*(-diff(Xp, x)*diff(psi, x))
                res[1]+=mm*(-I*m*Xp*diff(psi, x)/x - Xp*diff(psi, phi)/x**2)         
            elif ndim==2:
                y=dcoords[1]
                Yp=pcoords[1]
                res[0]+=mm*(-diff(Xp, x)*diff(psi, x) - diff(Yp, x)*diff(psi, y))
                res[1]+=mm*(-diff(Xp, y)*diff(psi, x) - diff(Yp, y)*diff(psi, y))
                res[2]+=mm*( -I*m*Xp*diff(psi, x)/x - I*m*Yp*diff(psi, y)/x - Xp*diff(psi, phi)/x**2    )
            else:
                raise RuntimeError("Not implemented")
        return vector(res)

    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool) -> Expression:
        if lagrangian:
            return super().vector_gradient(arg, ndim, edim, with_scales, lagrangian)

        if edim!=ndim:
            raise RuntimeError("TODO: Cannot take a vector gradient in axisymmetry-breaking coordinate system for edim!=ndim")

        zero=Expression(0)
        I=self.imaginary_i
        m=self.m_angular_symbol
        phi=self.phi
        mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")                                                        
        if ndim >= 3:
            raise RuntimeError("Vector gradient in axisymmetry does not work for dimension " + str(ndim))
        if ndim == 1:
            r, = self.get_coords(ndim, with_scales, lagrangian)
            X0=self.map_to_zero_epsilon(r)
            Xm=self.map_to_first_order_epsilon(r)
            if not lagrangian:
                res:List[List[ExpressionOrNum]] = [[diff(arg[0], X0), diff(arg[0],self.phi) / X0 - arg[1] / X0,zero],
                   [diff(arg[1], X0), diff(arg[1],self.phi) / X0 + arg[0] / X0,zero],[zero,zero,zero]]                
                res[0][0]+=mm*(-diff(Xm, X0)*diff(arg[0], X0))
                res[0][1]+=mm*(-I*m*Xm*diff(arg[0], X0)/X0 + Xm*arg[1]/X0**2 - Xm*diff(arg[0], phi)/X0**2)
                res[1][0]+=mm*(-diff(Xm, X0)*diff(arg[1], X0))
                res[1][1]+=mm*(-I*m*Xm*diff(arg[1], X0)/X0 - Xm*arg[0]/X0**2 - Xm*diff(arg[1], phi)/X0**2)
            else:
                res:List[List[ExpressionOrNum]] = [[diff(arg[0], r), 0, 0], [0, 0, 0], [0, 0, arg[0] / r]]
        else:
            if arg.nops() != 3:
                raise RuntimeError(
                    "Cannot take a 2d axisymmetric vector gradient from a vector with dim!=2:  " + str(arg))
            xc, yc = self.get_coords(ndim, with_scales, lagrangian)
            x=self.map_to_zero_epsilon(xc)
            y=self.map_to_zero_epsilon(yc)
            Xp=self.map_to_first_order_epsilon(xc)
            Yp=self.map_to_first_order_epsilon(yc)
            #pr=r-dr            
            #res:List[List[ExpressionOrNum]]=[[ diff(arg[0],dr),diff(arg[0],dz),diff(arg[0],self.phi)/dr-arg[2]/dr],
             #    [diff(arg[1],dr),diff(arg[1],dz),diff(arg[1],self.phi)/dr],
             #    [diff(arg[2],dr) ,diff(arg[2],dz),diff(arg[2],self.phi)/dr+arg[0]/dr]]
            res:List[List[ExpressionOrNum]]= [[diff(arg[0], x), diff(arg[0], y), -arg[2]/x + diff(arg[0], phi)/x], 
                                              [diff(arg[1], x), diff(arg[1], y), diff(arg[1], phi)/x], 
                                              [diff(arg[2], x), diff(arg[2], y), arg[0]/x + diff(arg[2], phi)/x]]
            res[0][0]+=mm*(-diff(Xp, x)*diff(arg[0], x) - diff(Yp, x)*diff(arg[0], y))
            res[0][1]+=mm*(-diff(Xp, y)*diff(arg[0], x) - diff(Yp, y)*diff(arg[0], y))
            res[0][2]+=mm*(-I*m*Xp*diff(arg[0], x)/x - I*m*Yp*diff(arg[0], y)/x + Xp*arg[2]/x**2 - Xp*diff(arg[0], phi)/x**2)
            res[1][0]+=mm*(-diff(Xp, x)*diff(arg[1], x) - diff(Yp, x)*diff(arg[1], y))
            res[1][1]+=mm*(-diff(Xp, y)*diff(arg[1], x) - diff(Yp, y)*diff(arg[1], y))
            res[1][2]+=mm*(-I*m*Xp*diff(arg[1], x)/x - I*m*Yp*diff(arg[1], y)/x - Xp*diff(arg[1], phi)/x**2)
            res[2][0]+=mm*(-diff(Xp, x)*diff(arg[2], x) - diff(Yp, x)*diff(arg[2], y))
            res[2][1]+=mm*(-diff(Xp, y)*diff(arg[2], x) - diff(Yp, y)*diff(arg[2], y))
            res[2][2]+=mm*(-I*m*Xp*diff(arg[2], x)/x - I*m*Yp*diff(arg[2], y)/x - Xp*arg[0]/x**2 - Xp*diff(arg[2], phi)/x**2)

        return matrix(res)



    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        if lagrangian:
            return super().vector_divergence(arg, ndim, edim, with_scales, lagrangian)
        res = 0
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        dcoords=self.map_to_zero_epsilon(coords)
        pcoords=self.map_to_first_order_epsilon(coords)
        mm=_pyoomph.GiNaC_EvalFlag("moving_mesh")
        m=self.m_angular_symbol
        I=self.imaginary_i     
        phi=self.phi       

        if ndim== 1:            
            if edim==1:
                res = diff(arg[0], dcoords[0]) + arg[0] / dcoords[0] + diff(arg[1],self.phi)/dcoords[0]
                res+=mm*(-I*m*pcoords[0]*diff(arg[1],dcoords[0])/dcoords[0] - diff(pcoords[0], dcoords[0])*diff(arg[0], dcoords[0]) - pcoords[0]*arg[0]/dcoords[0]**2 - pcoords[0]*diff(arg[1], phi)/dcoords[0]**2)
                return res
            else: # edim=0 case                     
                res=(arg[0] + diff(arg[1], phi))/dcoords[0]
                res+=mm*(pcoords[0]*(-I*m*arg[1] + I*m*diff(arg[0], phi) - arg[0] - diff(arg[1], phi))/dcoords[0]**2)
                return res

        elif ndim == 2:
            x=dcoords[0]
            y=dcoords[1]
            Xp=pcoords[0]
            Yp=pcoords[1]
            if edim==2:                
                res = diff(arg[0], dcoords[0]) + arg[0] / dcoords[0] + diff(arg[1], dcoords[1])+diff(arg[2],self.phi) / dcoords[0]
                res+=mm*(-I*m*Xp*diff(arg[2], x)/x - I*m*Yp*diff(arg[2], y)/x - diff(Xp, x)*diff(arg[0], x) - diff(Xp, y)*diff(arg[1], x) - diff(Yp, x)*diff(arg[0], y) - diff(Yp, y)*diff(arg[1], y) - Xp*arg[0]/x**2 - Xp*diff(arg[2], phi)/x**2)
                return res
            elif edim==1:
                s=var("local_coordinate_1")
                res = diff(arg[0], dcoords[0]) + arg[0] / dcoords[0] + diff(arg[1], dcoords[1])+diff(arg[2],self.phi) / dcoords[0]
                # TODO: We really must simplify this!
                res+=mm*( -I*m*Xp*arg[2]/x**2 + I*m*Xp*diff(arg[0], phi)/x**2 + I*m*Yp*diff(arg[1], phi)/x**2 + I*m*Xp*arg[2]*diff(x, s)**2/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) - I*m*Xp*diff(x, s)**2*diff(arg[0], phi)/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) - I*m*Xp*diff(x, s)*diff(y, s)*diff(arg[1], phi)/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) + I*m*Yp*arg[2]*diff(x, s)*diff(y, s)/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) - I*m*Yp*diff(x, s)*diff(y, s)*diff(arg[0], phi)/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) - I*m*Yp*diff(y, s)**2*diff(arg[1], phi)/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) - I*m*Xp*diff(x, s)*diff(arg[2], s)/(x*diff(x, s)**2 + x*diff(y, s)**2) - I*m*Yp*diff(y, s)*diff(arg[2], s)/(x*diff(x, s)**2 + x*diff(y, s)**2) + Xp*arg[0]/x**2 + Xp*diff(arg[2], phi)/x**2 - 2*x**2*diff(x, s)**2*diff(Xp, s)*diff(arg[0], s)/(x**2*diff(x, s)**4 + 2*x**2*diff(x, s)**2*diff(y, s)**2 + x**2*diff(y, s)**4) - 2*x**2*diff(x, s)*diff(Xp, s)*diff(y, s)*diff(arg[1], s)/(x**2*diff(x, s)**4 + 2*x**2*diff(x, s)**2*diff(y, s)**2 + x**2*diff(y, s)**4) - 2*x**2*diff(x, s)*diff(y, s)*diff(Yp, s)*diff(arg[0], s)/(x**2*diff(x, s)**4 + 2*x**2*diff(x, s)**2*diff(y, s)**2 + x**2*diff(y, s)**4) - 2*x**2*diff(y, s)**2*diff(Yp, s)*diff(arg[1], s)/(x**2*diff(x, s)**4 + 2*x**2*diff(x, s)**2*diff(y, s)**2 + x**2*diff(y, s)**4) - 2*x*Xp*diff(x, s)**3*diff(arg[0], s)/(x**2*diff(x, s)**4 + 2*x**2*diff(x, s)**2*diff(y, s)**2 + x**2*diff(y, s)**4) - 2*x*Xp*diff(x, s)**2*diff(y, s)*diff(arg[1], s)/(x**2*diff(x, s)**4 + 2*x**2*diff(x, s)**2*diff(y, s)**2 + x**2*diff(y, s)**4) - 2*x*Xp*diff(x, s)*diff(y, s)**2*diff(arg[0], s)/(x**2*diff(x, s)**4 + 2*x**2*diff(x, s)**2*diff(y, s)**2 + x**2*diff(y, s)**4) - 2*x*Xp*diff(y, s)**3*diff(arg[1], s)/(x**2*diff(x, s)**4 + 2*x**2*diff(x, s)**2*diff(y, s)**2 + x**2*diff(y, s)**4) + diff(Xp, s)*diff(arg[0], s)/(diff(x, s)**2 + diff(y, s)**2) + diff(Yp, s)*diff(arg[1], s)/(diff(x, s)**2 + diff(y, s)**2) - 2*x**3*arg[0]*diff(x, s)*diff(Xp, s)/(x**4*diff(x, s)**2 + x**4*diff(y, s)**2) - 2*x**3*arg[0]*diff(y, s)*diff(Yp, s)/(x**4*diff(x, s)**2 + x**4*diff(y, s)**2) - 2*x**3*diff(x, s)*diff(Xp, s)*diff(arg[2], phi)/(x**4*diff(x, s)**2 + x**4*diff(y, s)**2) - 2*x**3*diff(y, s)*diff(Yp, s)*diff(arg[2], phi)/(x**4*diff(x, s)**2 + x**4*diff(y, s)**2) - 2*x**2*Xp*arg[0]*diff(x, s)**2/(x**4*diff(x, s)**2 + x**4*diff(y, s)**2) - 2*x**2*Xp*arg[0]*diff(y, s)**2/(x**4*diff(x, s)**2 + x**4*diff(y, s)**2) - 2*x**2*Xp*diff(x, s)**2*diff(arg[2], phi)/(x**4*diff(x, s)**2 + x**4*diff(y, s)**2) - 2*x**2*Xp*diff(y, s)**2*diff(arg[2], phi)/(x**4*diff(x, s)**2 + x**4*diff(y, s)**2) + 2*x*arg[0]*diff(x, s)*diff(Xp, s)/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) + 2*x*arg[0]*diff(y, s)*diff(Yp, s)/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) + 2*x*diff(x, s)*diff(Xp, s)*diff(arg[2], phi)/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) + 2*x*diff(y, s)*diff(Yp, s)*diff(arg[2], phi)/(x**2*diff(x, s)**2 + x**2*diff(y, s)**2) + 2*Xp*diff(x, s)*diff(arg[0], s)/(x*diff(x, s)**2 + x*diff(y, s)**2) + 2*Xp*diff(y, s)*diff(arg[1], s)/(x*diff(x, s)**2 + x*diff(y, s)**2))
                return res
                #raise RuntimeError("divergence with ndim=2, edim=1 not implemented")
            else:
                raise RuntimeError("divergence with ndim=2, edim=0 not implemented")
            return res
            
        else:
            raise RuntimeError("Cannot use this coordinate system on a 3d mesh")
        
        


    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations") -> Tuple[List[Expression], List[Expression], List[str]]:
        s = scale_factor(name)
        S = test_scale_factor(name)        
        if ndim == 2:
            vx = element.define_scalar_field(name + "_x", space)
            vy = element.define_scalar_field(name + "_y", space)
            vt = element.define_scalar_field(name + "_phi", space)
            vx = var(name + "_x")
            vy = var(name + "_y")
            vt = var(name + "_phi")
            element.set_scaling(**{name + "_x": name, name + "_y": name, name + "_phi": name})
            element.set_test_scaling(**{name + "_x": name, name + "_y": name,name + "_phi": name})            
            return [vx / s, vy / s, vt/s], [testfunction(name + "_x") / S,testfunction(name + "_y") / S, testfunction(name + "_phi") / S], [name + "_x",name + "_y",name+"_phi"]
        elif ndim == 1:
            vx = element.define_scalar_field(name + "_x", space)
            vt = element.define_scalar_field(name + "_phi", space)
            vx = var(name + "_x")
            vt = var(name + "_phi")
            element.set_scaling(**{name + "_x": name,name+"_phi":name})
            element.set_test_scaling(**{name + "_x": name,name+"_phi":name})
            return [vx / s, vt/s], [testfunction(name + "_x") / S,testfunction(name + "_phi") / S], [name + "_x",name + "_phi"]
        else:
            raise RuntimeError("Axisymmetric vector fields do not work for dimension " + str(ndim))


    def expand_coordinate_or_mesh_vector(self,cg:"FiniteElementCodeGenerator", name:str,dimensional:bool,no_jacobian:bool,no_hessian:bool):        
        if not cg._coordinates_as_dofs:
            return super().expand_coordinate_or_mesh_vector(cg,name,dimensional,no_jacobian,no_hessian)
        else:
            dim=cg.get_nodal_dimension()
            if dimensional:
                vr=lambda n : var(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
            else:
                vr=lambda n : nondim(n,no_jacobian=no_jacobian,no_hessian=no_hessian) # TODO: Apply at domain=cg?
            if self.with_phi_component_in_mesh_coordinates:
                if dim == 1:
                    return vector([vr(name+"_x"),vr(name+"_phi")])
                elif dim == 2:
                    return vector([vr(name+"_x"), vr(name+"_y"),vr(name+"_phi")])
                elif dim == 3:
                    raise RuntimeError("Cannot use symmetry-breaking coordinate system in 3d")
            else:
                if dim == 1:
                    return vector([vr(name+"_x"),0])
                elif dim == 2:
                    return vector([vr(name+"_x"), vr(name+"_y"),0])
                elif dim == 3:
                    raise RuntimeError("Cannot use symmetry-breaking coordinate system in 3d")



    def get_normal_vector_or_component(self,cg:"FiniteElementCodeGenerator",component:Optional[int]=None,only_base_mode:bool=False,only_perturbation_mode:bool=False,where:str="Residual"):
        dim = cg.get_nodal_dimension()        
        edim=cg.get_element_dimension()
        if not cg._coordinates_as_dofs or (where!="Residual" and (not self.expand_with_modes_for_python_debugging or where!="Python")):
            return super().get_normal_vector_or_component(cg,component,only_base_mode,only_perturbation_mode,where=where)
        
        if dim not in [1,2]:
            RuntimeError("Cannot use this coordinate system with a nodal dimension of "+str(dim))
        if component is None:
            comp=lambda s : nondim(s,only_base_mode=only_base_mode,only_perturbation_mode=only_perturbation_mode)
            if dim==1:
                return vector(comp("normal_x"),comp("normal_y"),0)
            else:
                return vector(comp("normal_x"),comp("normal_y"),comp("normal_z")) #TODO: Normal z here?
        else:
            # Normal expansion            
            base_factor=0 if only_perturbation_mode else 1
            pert_factor=0 if only_base_mode else 1
            dcoords=self.map_to_zero_epsilon(self.get_coords(dim, with_scales=True, lagrangian=False))
            pcoords=self.map_to_first_order_epsilon(self.get_coords(dim, with_scales=True, lagrangian=False,mesh_coords=True))
            
            if edim==2:
                raise RuntimeError("Not implemented")
                
            if component<dim:                 
                return base_factor*cg._get_normal_component(component) +pert_factor*self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(cg._get_normal_component(component),Expression(1))*self.field_mode
            elif component==dim:                

                #return base_factor*cg._get_normal_component(component)
                nx0=cg._get_normal_component(0)
                Xk=pcoords[0]
                n0_dot_xeigen=nx0*Xk
                if dim==2:
                    ny0=cg._get_normal_component(1)
                    Yk=pcoords[1]
                    n0_dot_xeigen+=ny0*Yk
                if self.with_phi_component_in_mesh_coordinates:
                    raise RuntimeError("Not implemented")
                else:
                    pass
                return pert_factor*((-self.imaginary_i*self.m_angular_symbol/dcoords[0]*n0_dot_xeigen))
                
            else:
                raise RuntimeError("Normal component "+str(component)+" not available")
            ## Old code
            if False:
                # Normal expansion            
                base_factor=0 if only_perturbation_mode else 1
                pert_factor=0 if only_base_mode else 1
                
                if component<dim: # Radial normal
                    return base_factor*cg._get_normal_component(component) +pert_factor*self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(cg._get_normal_component_eigenexpansion(component),Expression(1))*self.field_mode
                elif component==dim:                
                    nr0=cg._get_normal_component(0)
                    #rm=self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(var("mesh_x"),_pyoomph.Expression(1))*self.field_mode
                    rm=var("mesh_x",only_perturbation_mode=True)
                    n0_dot_xeigen=nr0*rm
                    if dim==2:
                        nz0=cg._get_normal_component(1)
                        zm=var("mesh_y",only_perturbation_mode=True)
                        #zm=self.expansion_eps*_pyoomph.GiNaC_eval_at_expansion_mode(var("mesh_y"),_pyoomph.Expression(1))*self.field_mode
                        n0_dot_xeigen+=nz0*zm
                    if self.with_phi_component_in_mesh_coordinates:
                        phim=var("mesh_phi",only_perturbation_mode=True)
                    else:
                        phim=0
                    #return pert_factor*self.expansion_eps*(nr0*phim-self.imaginary_i*self.angular_mode/var("mesh_x",only_base_mode=True)*(n0_dot_xeigen))*self.field_mode
                    #print(pert_factor*(nr0*phim-self.imaginary_i*self.angular_mode/var("mesh_x",only_base_mode=True)*(n0_dot_xeigen)))                
                    #exit()
                    return pert_factor*(nr0*phim-self.imaginary_i*self.angular_mode/var("mesh_x",only_base_mode=True)*(n0_dot_xeigen))
                    
                else:
                    raise RuntimeError("Normal component "+str(component)+" not available")
            print(dim)
            exit()
            if component is None:
                posscompos=[nondim("normal_"+d,domain=cg) for d in ["x","y","z"]]
                mycomps=posscompos[:dim]        
                return vector(*mycomps)
            else:
                return cg._get_normal_component(component)            
            

    def tensor_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the tensor_divergence for this coordinate system. Occured upon taking the div of " + str(arg))

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int)->Expression:
        raise RuntimeError("Implement the directional tensor derivative for this coordinate system")



# Radial symmetric (1d only)

class RadialSymmetricCoordinateSystem(BaseCoordinateSystem):

    def __init__(self, Rcenter:ExpressionOrNum=0):
        super(RadialSymmetricCoordinateSystem, self).__init__()
        self.Rcenter = Rcenter

    def get_actual_dimension(self, reduced_dim:int)->int:
        return 3

    def get_id_name(self)->str:
        return "RadialSymmetric"

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", ndim:int, element:"Equations") -> Tuple[List[Expression], List[Expression], List[str]]:
        if ndim == 1:
            vx = element.define_scalar_field(name + "_x", space)
            vx = var(name + "_x")
            element.set_scaling(**{name + "_x": name})
            element.set_test_scaling(**{name + "_x": name})
            s = scale_factor(name)
            S = test_scale_factor(name)
            return [vx / s], [testfunction(name + "_x") / S], [name + "_x"]
        else:
            raise RuntimeError("Radialsymmetric vector fields do not work for dimension " + str(ndim))

    def volumetric_scaling(self, spatial_scale:ExpressionOrNum, elem_dim:int) -> ExpressionOrNum:
        return spatial_scale ** (elem_dim + 2)

    def integral_dx(self, nodal_dim:int, edim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:
        if edim >= 3:
            raise RuntimeError("Does not work for dimension " + str(edim))        
        if lagrangian:
            if with_scale:
                return spatial_scale ** (edim + 2) * 4 * pi * (nondim(
                    "lagrangian_x") - self.Rcenter) ** 2 * nondim("dX")
            else:
                return 4 * pi * (nondim(
                    "lagrangian_x") - self.Rcenter / scale_factor(
                    "spatial")) ** 2 * nondim("dX")
        else:
            if with_scale:
                return spatial_scale ** (edim + 2) * 4 * pi * (nondim(
                    "coordinate_x") - self.Rcenter / scale_factor(
                    "spatial")) ** 2 * nondim("dx")
            else:
                return 4 * pi * (nondim(
                    "coordinate_x") - self.Rcenter / scale_factor(
                    "spatial")) ** 2 * nondim("dx")

    def geometric_jacobian(self) -> Expression:
        return 4 * pi * (nondim("coordinate_x") - self.Rcenter / scale_factor(
            "spatial")) ** 2

    def vector_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        if ndim != 1:
            raise RuntimeError("Vector gradient in radial symmetry does not work for dimension " + str(ndim))
        else:
            r, = self.get_coords(ndim, with_scales, lagrangian)
            if with_scales:
                r = r - self.Rcenter
            else:
                r = r - self.Rcenter / scale_factor("spatial")
            res:List[List[ExpressionOrNum]] = [[diff(arg[0], r), 0, 0], [0, arg[0] / r, 0], [0, 0, arg[0] / r]]
        return matrix(res)

    def vector_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        if ndim != 1:
            raise RuntimeError("Vector divergence in radial symmetry does not work for dimension " + str(ndim))
        res = 0
        coords = self.get_coords(arg.nops(), with_scales, lagrangian)
        if with_scales:
            coords[0] -= self.Rcenter
        else:
            coords[0] -= self.Rcenter / scale_factor("spatial")
        res += diff(arg[0], coords[0]) + 2 * arg[0] / coords[0]
        return res

    def scalar_gradient(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        res:List[ExpressionOrNum] = []
        for i, a in enumerate(self.get_coords(3, with_scales, lagrangian)):
            res.append(diff(arg, a) if i < ndim else 0)
        return vector(res)


    def tensor_divergence(self, arg:Expression, ndim:int, edim:int, with_scales:bool, lagrangian:bool)->Expression:
        raise RuntimeError("Implement the tensor_divergence for this coordinate system. Occured upon taking the div of " + str(arg))

    def directional_tensor_derivative(self,T:Expression,direct:Expression,lagrangian:bool,dimensional:bool,ndim:int,edim:int)->Expression:
        raise RuntimeError("Implement the directional tensor derivative for this coordinate system")








class BaseDifferentialGeometryCoordinateSystem(BaseCoordinateSystem):
    """A coordinate system that uses differential geometry to define grad, div, etc.
    This is just a base class. Please inherit from this class and implement at least the method :py:meth:`~pyoomph.expressions.coordsys.get_real_position_vector_from_mesh_coordinates`.
    """
    def __init__(self):
        super().__init__()
        #: Select the maximum supported nodal dimension of the mesh
        self.max_nodal_dimension = 3
        #: Select suffixes of the vector components
        self.vector_component_suffixes = ["x", "y", "z"]
        #: Additional vector components (used for e.g. additional normal mode)
        self.additional_vector_component_suffixes = []
        #: Additional local coordinates (used for e.g. additional normal mode)
        self.additional_local_coordinates=[]        
        #: Use subexpressions for the transformations
        self.use_subexpressions = True	
        
        self._values_to_substitute = {} # Values to subsitute at the end of an expansion
        
        self._dx_integration_factor=1
        # Cache for the covariant basis vectors, map from (bool[lagrangian],bool[dimensional],ndim,edim) to a list of basis vectors        
        self._cached_basis_vectors = {}
        # Cache for the covariant metric tensors, map from (bool[lagrangian],bool[dimensional],ndim,edim) to a g_ab
        self._cached_covariant_metrics = {}
        # Cache for the contravariant metric tensors, map from (bool[lagrangian],bool[dimensional],ndim,edim) to a g^ab
        self._cached_contravariant_metrics = {}


	# The following functions must or should be overridden

    def get_real_position_vector_from_mesh_coordinates(self, mesh_xs: Expression, ndim: int, edim: int,dimensional:bool,lagrangian:bool) -> Expression:
        raise NotImplementedError("Please implement the position of the real position vector in the local coordinate system by overriding the method get_real_position_vector_from_mesh_coordinates of your class "+str(self.__class__))

    def geometric_jacobian(self) -> Expression:
        # This function is only used to weight error estimates in the mesh adaptation
        return Expression(1)



	# The following functions usually do not need any overriding
 
    def substitute_values_for_additional_local_coordinates(self,expr:Expression)->Expression:
        import _pyoomph
        for k,v in self._values_to_substitute.items():
            expr=_pyoomph.GiNaC_SymSubs(expr,k,v)
        return expr 

    def add_additional_parametric_variable(self, name: str,value_to_substitute:Optional[ExpressionOrNum]=0,dx_integration_factor:Optional[ExpressionOrNum]=None) -> Expression:
        """Add a new local coordinate to the coordinate system. This can be e.g. phi in an axisymmetric coordinate system

        """
        import _pyoomph
        res_symb=_pyoomph.GiNaC_new_symbol(name)
        self.additional_local_coordinates.append(res_symb)
        if value_to_substitute is not None:
            value_to_substitute=Expression(value_to_substitute)
            self._values_to_substitute[res_symb]=value_to_substitute
        if dx_integration_factor is not None:
            self._dx_integration_factor*=dx_integration_factor
        return res_symb

    def volumetric_scaling(self, spatial_scale:ExpressionOrNum, elem_dim:int)->ExpressionOrNum:
        eaug=self.get_augmented_edim(None,elem_dim)
        return spatial_scale ** (eaug)

    def integral_dx(self, nodal_dim:int, edim:int, with_scale:bool, spatial_scale:ExpressionOrNum, lagrangian:bool) -> Expression:            
        from ..expressions import determinant,square_root
        eaug=self.get_augmented_edim(nodal_dim,edim)			
        if eaug==0:
            J=1        
        else:
            g_ab=self.get_covariant_metric_tensor(nodal_dim,edim,lagrangian,False) # Cannot do it with scales here=> Can mess up the sqrt
            g=self.substitute_values_for_additional_local_coordinates(determinant(g_ab))
            J=square_root(g)
            if self.use_subexpressions:
                J=subexpression(J)                                
        if with_scale:
            vs=self.volumetric_scaling(spatial_scale,edim)
        else:
            vs=1
        return self._dx_integration_factor*vs*J * nondim("dx_unity") 
		
        
        
    def vector_gradient_dimension(self, basedim:int, lagrangian:bool=False)->int:
        # Just alwas 3
        return 3
    
    def get_local_coordinate(self, i:int, ndim: int, edim: int,lagrangian:bool,dimensional:bool) -> List[Expression]:
        if i<edim:            
            return nondim("local_coordinate_"+str(i+1))
        elif i<edim+len(self.additional_local_coordinates):
            return self.additional_local_coordinates[i-edim]
        else:
            raise RuntimeError("The local coordinate index is out of bounds.")
        
    def get_all_local_coordinates(self, ndim: int, edim: int,lagrangian:bool,dimensional:bool) -> List[Expression]:
        eaug=self.get_augmented_edim(ndim,edim)
        return [self.get_local_coordinate(i,ndim,edim,lagrangian,dimensional) for i in range(eaug)]
    
    def get_augmented_edim(self, ndim:Optional[int],edim: int) -> int:
        return edim+len(self.additional_local_coordinates)

    def get_covariant_basis_vectors(self, ndim: int, edim: int,lagrangian:bool,dimensional:bool) -> List[Expression]:
        if (lagrangian,dimensional,ndim, edim) in self._cached_basis_vectors:
            return self._cached_basis_vectors[(lagrangian,dimensional,ndim, edim)]
        s = var("local_coordinate")  # Local coordinate
        if dimensional:
            mesh_coord=var("lagrangian" if lagrangian else "coordinate")
        else:
            mesh_coord=nondim("lagrangian" if lagrangian else "coordinate")
        x=self.get_real_position_vector_from_mesh_coordinates(mesh_coord,ndim,edim,lagrangian,dimensional)
        t=[diff(x,s[i]) for i in range(edim)] # covariant basis vector        
        for sadd in self.additional_local_coordinates:
            t.append(diff(x,sadd))
        if len(t)!=self.get_augmented_edim(ndim,edim):
            raise RuntimeError("The number of basis vectors does not match the augmented dimension. Make sure that the length of additional_local_coordinates is agrees with the augmented dimension, which must be implemented via the get_augmented_edim method.")
        self._cached_basis_vectors[(lagrangian,dimensional,ndim, edim)] = t
        return t
    
    def get_covariant_metric_tensor(self, ndim: int, edim: int,lagrangian:bool,dimensional:bool) -> Expression:
        if (lagrangian,dimensional,ndim, edim) in self._cached_covariant_metrics:
            return self._cached_covariant_metrics[(lagrangian,dimensional,ndim, edim)]
        t=self.get_covariant_basis_vectors(ndim,edim,lagrangian,dimensional)
        eaug=self.get_augmented_edim(ndim,edim)
        g_covar=matrix([[dot(t[i],t[j]) for j in range(eaug)] for i in range(eaug)])
        self._cached_covariant_metrics[(lagrangian,dimensional,ndim, edim)] = g_covar
        return g_covar
        
    def get_contravariant_metric_tensor(self, ndim: int, edim: int,lagrangian:bool,dimensional:bool) -> Expression:
        
        from ..expressions import inverse_matrix # Don't know why I have to import it manually here
        
        if (lagrangian,dimensional,ndim, edim) in self._cached_contravariant_metrics:
            return self._cached_contravariant_metrics[(lagrangian,dimensional,ndim, edim)]
        g_covar=self.get_covariant_metric_tensor(ndim,edim,lagrangian,dimensional)
        g_contra=inverse_matrix(g_covar,n=self.get_augmented_edim(ndim,edim),use_subexpression_for_det=self.use_subexpressions)
        if self.use_subexpressions:
            g_contra=subexpression(g_contra)
        self._cached_contravariant_metrics[(lagrangian,dimensional,ndim, edim)] = g_contra
        return g_contra
        

    def define_vector_field(self, name: str, space: "FiniteElementSpaceEnum", ndim: int, element: "Equations") -> Tuple[List[Expression], List[Expression], List[str]]:
        namelist: List[str] = []
        if ndim > self.max_nodal_dimension:
            raise RuntimeError(
                "Cannot use a this coordinate system in "+str(ndim)+"D")
        for i in range(ndim):
            namelist.append(name + "_" + self.vector_component_suffixes[i])
        for v in self.additional_vector_component_suffixes:
            namelist.append(name + "_" + v)  # TODO: Axisymmetric?

        v: List[Expression] = []
        vtest: List[Expression] = []
        s = scale_factor(name)
        S = test_scale_factor(name)
        for i, f in enumerate(namelist):
            if i >= ndim+1:
                break
            element.set_scaling(**{f: name})
            vc = element.define_scalar_field(f, space)
            vc = var(f)
            v.append(vc / s)
            element.set_test_scaling(**{f: name})
            vtest.append(testfunction(f) / S)
        return v, vtest, namelist

    def scalar_gradient(self, arg: Expression, ndim: int, edim: int, with_scales: bool, lagrangian: bool) -> Expression:
        g_contra=self.get_contravariant_metric_tensor(ndim,edim,lagrangian,with_scales)
        t=self.get_covariant_basis_vectors(ndim,edim,lagrangian,with_scales)
        eaug=self.get_augmented_edim(ndim,edim)
        s=self.get_all_local_coordinates(ndim,edim,lagrangian,with_scales)
        # TODO: This is likely not right!
        return self.substitute_values_for_additional_local_coordinates(sum([g_contra[a,b]*t[a]*diff(arg,s[b]) for a in range(eaug) for b in range(eaug)]))
    
    def vector_gradient(self, arg: List[Expression], ndim: int, edim: int, with_scales: bool, lagrangian: bool) -> List[Expression]:
        g_contra=self.get_contravariant_metric_tensor(ndim,edim,lagrangian,with_scales)
        t=self.get_covariant_basis_vectors(ndim,edim,lagrangian,with_scales)
        eaug=self.get_augmented_edim(ndim,edim)
        s=self.get_all_local_coordinates(ndim,edim,lagrangian,with_scales)
        # TODO: This is likely not right!
        return self.substitute_values_for_additional_local_coordinates(sum([g_contra[a,b]*dyadic(t[a],diff(arg,s[b])) for a in range(eaug) for b in range(eaug)]))
    
    def vector_divergence(self, arg: Expression, ndim: int, edim: int, with_scales: bool, lagrangian: bool) -> Expression:
        g_contra=self.get_contravariant_metric_tensor(ndim,edim,lagrangian,with_scales)
        t=self.get_covariant_basis_vectors(ndim,edim,lagrangian,with_scales)
        eaug=self.get_augmented_edim(ndim,edim)
        s=self.get_all_local_coordinates(ndim,edim,lagrangian,with_scales)
        return self.substitute_values_for_additional_local_coordinates(sum([g_contra[a,b]*dot(t[a].evalm(),diff(arg,s[b]).evalm()) for a in range(eaug) for b in range(eaug)]))
    

