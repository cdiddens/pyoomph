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
 
 
import math
from ..typings import *

import _pyoomph
import numpy



# Expressions are any kind of mathematical expressions, i.e. combinations of numbers (with units), constants, functions, additions and multiplications and variables.



Expression=_pyoomph.Expression
ExpressionOrNum=Union[Expression,int,float]
ExpressionNumOrNone=Union[Expression,int,float,None]
#NameStrSequence = Union[Tuple[str], List[str]]
#ExprStrSequence = Union[Tuple[Expression], List[Expression]]
NameStrSequence = Union[Tuple[str], List[str]]
ExprStrSequence = Union[Tuple[Expression], List[Expression]]
GlobalParameter=_pyoomph.GiNaC_GlobalParam
SingleOrMultipleExpressions=Union[Expression,Tuple[Expression,...]]
OptionalCoordinateSystem=Union[None,_pyoomph.CustomCoordinateSystem]
TimeSteppingScheme=Literal["BDF1","BDF2","Newmark2","TPZ","MPT","Simpson","Boole","trapezoidal","Kepler","Milne","midpoint"]
OptionalTimeSteppingScheme=Union[None,TimeSteppingScheme]

FiniteElementSpaceEnum=Literal["C1","C1TB","C2","C2TB","D1","D1TB","D2","D2TB","DL","D0"]
def assert_valid_finite_element_space(inp:str)->FiniteElementSpaceEnum:
	spaces={"C1","C1TB","C2","C2TB","DL","D0","D1","D1TB","D2","D2TB"}
	if inp in spaces:
		return cast(FiniteElementSpaceEnum,inp)
	else:
		raise RuntimeError(inp+" is not a valid finite element space. Valid spaces are "+str(spaces))

def is_DG_space(space:FiniteElementSpaceEnum,allow_DL_and_D0:bool=False):
	"""
	Check if the given space is a discontinuous Galerkin space. By default, ``"DL"`` and ``"D0"`` are not considered DG spaces, so that it returns ``True`` for ``"D2TB"``,``"D2"``,``"D1TB"`` and ``"D1"``

	Args:
		space: The space to check.
		allow_DL_and_D0: Flag indicating whether the pure elemental spaces ``"DL"`` and ``"D0"`` also should return True. Defaults to False.

	Returns:
		Whether the space is a discontinuous Galerkin space.
	"""
	if allow_DL_and_D0:
		return space in {"D2TB","D2","D1TB","D1","DL","D0"}
	else:
		return space in {"D2TB","D2","D1TB","D1"}

def get_order_of_space(space:FiniteElementSpaceEnum)->int:
	"""
	Get the order of the given finite element space.
	"""
	if space in {"D2TB","D2","C2","C2TB"}:
		return 2
	elif space in {"D1TB","D1","C1","C1TB","DL"}:
		return 1
	elif space=="D0":
		return 0
	else:
		raise RuntimeError("Unknown space: "+str(space))

def find_dominant_element_space(*spaces:FiniteElementSpaceEnum):
	res=""
	for r in spaces:
		if res=="":
			if r=="":
				continue
			if r=="D0" or r=="DL":
				r="C1"
			elif r[0]=="D":
				r="C"+r[1:]
			res=r
			continue
		if r=="DL" or r=="D0":
			continue

		elif r[0]=="D":
			r="C"+r[1:]

		if (r =="C2" and res=="C1TB") or (r =="C1TB" and res=="C2"):
			res="C2TB" # Only space that can hold both
			continue
		space_in_order=["C1","C1TB","C2","C2TB"]
		if space_in_order.index(r)>space_in_order.index(res):
			res=r
	return res

if TYPE_CHECKING:
	from ..generic.codegen import FiniteElementCodeGenerator

def substitute_in_expression(expr:ExpressionOrNum,field_subst:Dict[str,ExpressionOrNum],nondim_subst:Dict[str,ExpressionOrNum]={},global_param_subst:Dict[str,ExpressionOrNum]={})->Expression:
	fs,nf,gp={},{},{}
	if not isinstance(expr,Expression):
		expr=Expression(expr)
	for n,v in field_subst.items():
		fs[n]=_pyoomph.Expression(v)
	for n,v in nondim_subst.items():
		nf[n]=_pyoomph.Expression(v)
	for n,v in global_param_subst.items():
		gp[n]=_pyoomph.Expression(v)
	return _pyoomph.GiNaC_subsfields(expr,fs,nf,gp)

@overload
def var(arg:str,*,tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None,no_jacobian:bool=False,no_hessian:bool=False,only_base_mode:bool=False,only_perturbation_mode:bool=False)->Expression: ...

@overload
def var(arg:NameStrSequence,*,tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None,no_jacobian:bool=False,no_hessian:bool=False,only_base_mode:bool=False,only_perturbation_mode:bool=False)->Tuple[Expression,...]: ...

def var(arg:Union[str,NameStrSequence],*,tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None,no_jacobian:bool=False,no_hessian:bool=False,only_base_mode:bool=False,only_perturbation_mode:bool=False)->SingleOrMultipleExpressions:
	r"""
	Binds a variable or a list of variables for usage in an expression by supplying the name(s)

	Args:
		arg: A single variable name or a sequence of variable names.
		tag: A list of tags to associate with the variable(s). Defaults to [].
		domain: The domain of the variable(s). Defaults to None, meaning the current domain. To get the bulk domain, use ``".."``, for the opposite side of the interface, use ``"|."``, for the opposite bulk domain use ``"|.."``.
		no_jacobian: Flag indicating whether derivatives in Jacobian should be ignored. Defaults to False.
		no_hessian: Flag indicating whether derivatives in the Hessian should be ignored. Defaults to False.
		only_base_mode: Flag indicating whether only the axisymmetric base mode of the azimuthal mode expansion should be considered. Defaults to False.
		only_perturbation_mode: Flag indicating whether only the azimuthal perturbation mode should be considered. Defaults to False.

	Returns:
		A single expression or a tuple of expressions representing the variable(s) to be used in expressions.
  
	Notes:
		Special variable names are:		
			* ``"time"`` : Current time 
			* ``"coordinate"`` : indepedent coordinate vector. Time derivatives of this variable are zero 
			* ``"coordinate_x"`` : indepedent x-coordinate. Time derivatives of this variable are zero 
			* ``"coordinate_y"`` : indepedent y-coordinate. Time derivatives of this variable are zero 
			* ``"coordinate_z"`` : indepedent z-coordinate. Time derivatives of this variable are zero 
			* ``"mesh"`` : mesh coordinate vector, similar to ``"coordinate"``, but the time derivative gives the mesh velocity 
			* ``"mesh_x"`` : mesh x-coordinate, similar to ``"coordinate_x"``, but the time derivative gives the mesh x-velocity 
			* ``"mesh_y"`` : mesh y-coordinate, similar to ``"coordinate_y"``, but the time derivative gives the mesh y-velocity 
			* ``"mesh_z"`` : mesh z-coordinate, similar to ``"coordinate_z"``, but the time derivative gives the mesh z-velocity 
			* ``"lagrangian"`` : Lagrangian coordinate vector. By default, initialized with the initial Eulerian ``"coordinate"``
			* ``"lagrangian_x"`` : Lagrangian x-coordinate 
			* ``"lagrangian_y"`` : Lagrangian y-coordinate 
			* ``"lagrangian_z"`` : Lagrangian z-coordinate  
			* ``"normal"`` : Normal vector. To be used at elements with co-dimension, i.e. interface elements 
			* ``"normal_x"`` : x-component of the normal 
			* ``"normal_y"`` : y-component of the normal 
			* ``"normal_z"`` : z-component of the normal 
			* ``"dx"`` : Can be used like in FEniCS to express ``weak(a,b)`` as ``a*b*var("dx")``. It does not respect the functional determinant of the coordinate system, though. 
			* ``"dX"`` : Same as ``"dx"``, but for Lagrangian integrals 
			* ``"element_size_Eulerian"`` : Eulerian integration of the volume/area/length of the current element. Uses the coordinate system of the element, i.e. considers e.g. :math:`2\\pi\:r` in axisymmetry 
			* ``"cartesian_element_size_Eulerian"`` : Same as above, but does not consider the coordinate system 
			* ``"element_size_Lagrangian"`` : Same as "element_size_Eulerian", but by Lagrangian integration 
			* ``"cartesian_element_size_Lagrangian"`` : Same as ``"cartesian_element_size_Eulerian"``, but by Lagrangian integration 
			* ``"element_length_h"`` : Typical length scale of the element, calculated by taking ``"element_size_Eulerian"`` to the power of one over the element dimension 
			* ``"cartesian_element_length_h"`` : Typical length scale of the element, but calculated in a Cartesian coordinate system   
	"""
	
	res:List[Expression]=[]
	tag=tag.copy()
	if isinstance(tag,str):
		tag+=[tag]
	if isinstance(domain,str):
		tag+=["domain:"+domain]
		domain=None
	if no_jacobian:
		# raise RuntimeError("No jacobian does not work yet")
		tag+=["flag:no_jacobian"]
	if no_hessian:
		# raise RuntimeError("No jacobian does not work yet")
		tag+=["flag:no_hessian"]
	if only_base_mode:
		if only_perturbation_mode:
			raise RuntimeError("Cannot combine only_base_mode and only_perturbation_mode")
		tag+=["flag:only_base_mode"]        
	elif only_perturbation_mode:        
		tag+=["flag:only_perturbation_mode"]                
	if isinstance(arg,str):
		return _pyoomph.GiNaC_field(arg,domain,tag)
	else:
		for a in arg:
			res.append(_pyoomph.GiNaC_field(a,domain,tag))
		return tuple(res)

@overload
def nondim(arg:str,tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None,no_jacobian:bool=False,no_hessian:bool=False,only_base_mode:bool=False,only_perturbation_mode:bool=False)->Expression: ...

@overload
def nondim(arg:NameStrSequence,tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None,no_jacobian:bool=False,no_hessian:bool=False,only_base_mode:bool=False,only_perturbation_mode:bool=False)->Tuple[Expression,...]: ...

def nondim(arg:Union[str,NameStrSequence],tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None,no_jacobian:bool=False,no_hessian:bool=False,only_base_mode:bool=False,only_perturbation_mode:bool=False)->SingleOrMultipleExpressions:
	"""
	This returns the nondimensional equivalent of a field, i.e. it is the same as var(...)/scale_factor(...).
	
 	See var for further details.
	"""
	
	res:List[Expression]=[]
	if isinstance(domain,str):
		tag=["domain:"+domain]
		domain=None
	if no_jacobian:
		tag+=["flag:no_jacobian"]
	if no_hessian:
		tag+=["flag:no_hessian"]        
	if only_base_mode:
		if only_perturbation_mode:
			raise RuntimeError("Cannot combine only_base_mode and only_perturbation_mode")
		tag+=["flag:only_base_mode"]        
	elif only_perturbation_mode:        
		tag+=["flag:only_perturbation_mode"]                        
	if isinstance(arg,str):
		return _pyoomph.GiNaC_nondimfield(arg,domain,tag)
	else:
		for a in arg:
			res.append(_pyoomph.GiNaC_nondimfield(a,domain,tag))
		return tuple(res)

import fractions

#Arg can be string (eg "1/2"), int or double. Best is always use rationals etc
def num(arg:Union[float,int,str], order10: Union[int,None] = None,*,rational:bool=False)->Expression:
	if rational: #Try to convert it to rational
		f=fractions.Fraction(arg)
		return _pyoomph.Expression(str(f.numerator)+"/"+str(f.denominator))
	res = _pyoomph.Expression(arg)
	if order10 is not None:
		res = res * _pyoomph.Expression(10) ** _pyoomph.GiNaC_rational_number(order10, 1)
	return res

@overload
def scale_factor(arg:str,tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None)->Expression: ...

@overload
def scale_factor(arg:NameStrSequence,tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None)->Tuple[Expression,...]: ...

def scale_factor(arg: Union[str, NameStrSequence], tag: List[str] = [], domain: Union[str, "FiniteElementCodeGenerator", None] = None) -> SingleOrMultipleExpressions:
	"""
	Returns the scale factor of an unknown used for nondimensionalization. Will be expanded during code generation.
	If you pass scale=... as keyword argument to Equations.define_scalar_field, .define_vector_field or .define_ode_variable, this determines the scale factor.
	If the scale factor is not set there, it will look at parent equations and ultimately at the problem level, where you can set it via Problem.set_scaling.
	All scales are by default 1.

	Parameters:
		arg (Union[str, NameStrSequence]): The field names for which to get the scale factors used for nondimensionalization
		tag (List[str], optional): Additional tags to associate with the scale factor. Defaults to an empty list.
		domain (Union[str, "FiniteElementCodeGenerator", None], optional): The domain for which to compute the scale factor. You might want to get e.g. the scale of a field on the opposite side of the interface via domain="``|.``".

	Returns:
		SingleOrMultipleExpressions: The scale factor(s), will be expanded only during code generation.
	"""
	res: List[Expression] = []
	if isinstance(domain, str):
		tag = ["domain:" + domain]
		domain = None
	if isinstance(arg, str):
		return _pyoomph.GiNaC_scale(arg, domain, tag)
	else:
		for a in arg:
			res.append(_pyoomph.GiNaC_scale(a, domain, tag))
		return tuple(res)


@overload
def test_scale_factor(arg:str,tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator", None]=None)->Expression: ...

@overload
def test_scale_factor(arg:NameStrSequence,tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None)->Tuple[Expression,...]: ...

def test_scale_factor(arg:Union[str,NameStrSequence],tag:List[str]=[],domain:Union[str,"FiniteElementCodeGenerator",None]=None)->SingleOrMultipleExpressions:
	"""
	Returns the scale factor of a test function or multiple test functions used for nondimensionalization. Will be expanded during code generation.
	If you pass testscale=... as keyword argument to Equations.define_scalar_field, .define_vector_field or .define_ode_variable, this determines the test scale factor.
	If the test scale factor is not set there, it will look at parent equations, which gives an additional factor 1/scale_factor("spatial") per level, since the integration measure (e.g. dx and dS) is always nondimensional.
	All test scales are by default 1.

	Parameters:
		arg (Union[str, NameStrSequence]): The field names for which to get the scale factors used for nondimensionalization
		tag (List[str], optional): Additional tags to associate with the scale factor. Defaults to an empty list.
		domain (Union[str, "FiniteElementCodeGenerator", None], optional): The domain for which to compute the scale factor. You might want to get e.g. the scale of a field on the opposite side of the interface via domain="``|.``".

	Returns:
		SingleOrMultipleExpressions: The test scale factor(s), will be expanded only during code generation.
	"""
	res:List[Expression]=[]
	if isinstance(domain,str):
		tag=["domain:"+domain]
		domain=None
	if isinstance(arg,str):
		return _pyoomph.GiNaC_testscale(arg,domain,tag)
	else:
		for a in arg:
			res.append(_pyoomph.GiNaC_testscale(a,domain,tag))	
		return tuple(res)

test_scale_factor.__test__=False


def is_zero(arg:ExpressionOrNum,parameters_to_float:bool=False)->bool:
	"""
	Check if the given argument (Expression or numerical value) is zero.

	Parameters:
	arg (ExpressionOrNum): The argument to be checked.
 	parameters_to_float: Flag indicating whether to convert a global parameter to its float values. Defaults to False, i.e. always False applied on a global parameter.

	Returns:
	bool: True if the argument is zero, False otherwise.

	Raises:
	ValueError: If the argument cannot be tested for zero.

	"""
	if isinstance(arg,(float,int,bool)):
		return arg==0
	elif isinstance(arg,Expression): # type: ignore
		return arg.is_zero()
	elif isinstance(arg,_pyoomph.GiNaC_GlobalParam):
	    if parameters_to_float:
	        return arg.value==0
	    else:
	        return False
	else:
		raise ValueError("Cannot test for zero: "+repr(arg))



#furtherargs are either one number -> dimension
#	the keyword nondim
# a coordinate system

def grad(arg:ExpressionOrNum,lagrangian:bool=False,nondim:bool=False,coordsys:OptionalCoordinateSystem=None,vector:Union[None,bool]=None)->Expression:
	"""
	Compute the gradient of the given argument. On surfaces, i.e. with a co-dimension, it is the surface gradient.

	Parameters:
	arg (ExpressionOrNum): The argument for which the gradient is computed.
	lagrangian (bool, optional): Flag indicating whether the computation is with respect to Lagrangian coordinates. Default is False.
	nondim (bool, optional): Flag indicating whether the computation is with respect to non-dimensional coordinates. Default is False.
	coordsys (OptionalCoordinateSystem, optional): The coordinate system in which the gradient is computed. Default is None, which is the coordinate system of the Equation, potential parent equations or given at problem level.
	vector (Union[None,bool], optional): Flag indicating whether the gradient is a vector. Default is None, meaning it will find out automatically.

	Returns:
	Expression: The computed gradient expression.
 
 	Notes:
		if you calculate grad(u) on a boundary, you will get the surface gradient, even if u is defined in the bulk.
		To get the bulk gradient at the boundary, use grad(var("u",domain="..")) instead.
	"""
	if isinstance(arg,str):
		arg=var(arg)
	if isinstance(arg,float) or isinstance(arg,int) or isinstance(arg,_pyoomph.GiNaC_GlobalParam):
		return _pyoomph.Expression(0)
	flag=(0 if nondim else 1)+(0 if vector is None else (2 if vector==False else 4) ) + (8 if lagrangian else 0) #Code the flag
	if coordsys is None:
		coordsysEx=_pyoomph.Expression(0)
	else:
		coordsysEx=0+_pyoomph.GiNaC_wrap_coordinate_system(coordsys)
	return _pyoomph.GiNaC_grad(arg,_pyoomph.Expression(-1),_pyoomph.Expression(-1),coordsysEx,_pyoomph.Expression(flag))



_weak_mode_conjugate_second_arg=False


def set_weak_conjugate_second_argument(conjugate:bool):
	global _weak_mode_conjugate_second_arg
	_weak_mode_conjugate_second_arg=conjugate


def weak(a:ExpressionOrNum,b:ExpressionOrNum,*,dimensional_dx:bool=False,lagrangian:bool=False,coordinate_system:OptionalCoordinateSystem=None)->Expression:
	"""
	Construct a term of a weak form, i.e. (a,b)=integral_Omega a*b dOmega where Omega is the domain.
	a is usually an expression depending on unknowns and b is usually a test function or spatial differentiations thereof.

	Args:
		a (ExpressionOrNum): An expression, usually a expression depending on unknowns.
		b (ExpressionOrNum): A testfunction or any linear function/operator applied on a testfunction.
		dimensional_dx (bool, optional): Flag indicating whether consider spatial integration units (e.g. m, m^2, m^3). Defaults to False.
		lagrangian (bool, optional): Flag indicating whether to integrate with respect to the Lagrangian coordinates and domain. Defaults to False.
		coordinate_system (OptionalCoordinateSystem, optional): The coordinate system to use. Defaults to None, meaning the coordinate system at equation level, parent equation level or problem level.

	Returns:
		Expression: A weak form that can be further used in expressions or added to the residuals of equations by the method add_residual.
	"""

	flags=0
	if coordinate_system is None:
		coordsys=_pyoomph.Expression(0)
	else:
		coordsys = 0 + _pyoomph.GiNaC_wrap_coordinate_system(coordinate_system)
	if dimensional_dx:
		flags+=2
	if lagrangian:
		flags+=1
	if not isinstance(a,_pyoomph.Expression):
		a=_pyoomph.Expression(a)
	if not isinstance(b,_pyoomph.Expression):
		b=_pyoomph.Expression(b)
	if _weak_mode_conjugate_second_arg:
		b=_pyoomph.GiNaC_get_real_part(b)-_pyoomph.GiNaC_imaginary_i()*_pyoomph.GiNaC_get_imag_part(b)
		#a = _pyoomph.GiNaC_get_real_part(a) - _pyoomph.GiNaC_imaginary_i() * _pyoomph.GiNaC_get_imag_part(a)
	return _pyoomph.GiNaC_weak(a,b,_pyoomph.Expression(flags),coordsys)

# Lagrangian weak
def Weak(a:ExpressionOrNum,b:ExpressionOrNum,*,dimensional_dx:bool=False,coordinate_system:OptionalCoordinateSystem=None)->Expression:
	"""
	Shortcut for weak(a,b,dimensional_dx=dimensional_dx,lagrangian=True,coordinate_system=coordinate_system)
	"""
	return weak(a,b,dimensional_dx=dimensional_dx,lagrangian=True,coordinate_system=coordinate_system)



def minimize_functional_derivative(F:ExpressionOrNum,only_with_respect_to:Optional[Union[Expression,set[Expression],List[Expression],Tuple[Expression,...]]]=None,*,coordinate_system:OptionalCoordinateSystem=None,lagrangian:bool=False,dimensional_dx:bool=False,dimensional_testfunctions:bool=True)->Expression:
    if only_with_respect_to is None:
        only_with_respect_to=[]
    elif isinstance(only_with_respect_to,(set,tuple,list)):
        only_with_respect_to=list(only_with_respect_to)
    else:
        only_with_respect_to=[only_with_respect_to]
        
    flags=0
    if coordinate_system is None:
        coordinate_system=_pyoomph.Expression(0)
    else:
        coordinate_system = 0 + _pyoomph.GiNaC_wrap_coordinate_system(coordinate_system)
    # Must agree with weak(a,b) flags
    if dimensional_dx:
        flags+=2
    if lagrangian:
        flags+=1
    # Additional flags
    if dimensional_testfunctions:
        flags+=64
    flags=Expression(flags)
    if not isinstance(F,_pyoomph.Expression):
        F=_pyoomph.Expression(F)
          
    return _pyoomph.GiNaC_minimize_functional_derivative(F,only_with_respect_to,flags,coordinate_system)


def timestepper_weight(order:int,index:int,scheme:TimeSteppingScheme="BDF1")->Expression:
	return _pyoomph.GiNaC_time_stepper_weight(order,index,scheme)



def contract(a:ExpressionOrNum,b:ExpressionOrNum)->Expression:
	"""
	Contract a and b. 
	If both are scalars, it is just a*b. 
	If both are vectors, it is the dot product.
	If both are rank-2-tensors/matrices, it is the Frobenius product.
	If one is a vector and the other a matrix, it is the matrix-vector product.
	If one is a scalar, it is just the multiplication by a scalar.

	Args:
		a (ExpressionOrNum): Argument a (scalar, vector or matrix/tensor)
		b (ExpressionOrNum): Argument b (scalar, vector or matrix/tensor)

	Returns:
		Expression: The symbolic contraction of a and b.
	"""
	if not isinstance(a,_pyoomph.Expression):
		a=_pyoomph.Expression(a)
	if not isinstance(b,_pyoomph.Expression):
		b=_pyoomph.Expression(b)
	return _pyoomph.GiNaC_contract(a,b)  # dot product between two vectors: v1*v2


def eval_flag(which:str)->Expression:
	return _pyoomph.GiNaC_EvalFlag(which)


def mesh_velocity(scheme:OptionalTimeSteppingScheme=None,nondim:bool=False)->Expression:
	"""
	Get the mesh velocity, i.e. the time derivative of the mesh coordinates without ALE correction.
 
	Returns:
		Expression: Just a shorthand for `partial_t(var("mesh"),ALE=False)`.
	"""
	return partial_t(nondim("mesh") if nondim else var("mesh"),ALE=False,scheme=scheme,nondim=nondim)


def partial_t(f:Union[ExpressionOrNum,str],order:int=1,ALE:Union[Literal["auto"],bool]="auto",scheme:OptionalTimeSteppingScheme=None,nondim:bool=False)->Expression:
	"""
	Compute the partial derivative of a function with respect to time, i.e. .. :math:`\\partial_t^n f`. 
	With `ALE=False`, this is evaluated at the nodal values directly, i.e. co-moving with a moving mesh. To correct for it by the mesh velocity :math:`\\dot{\\vec{X}}`, use the `ALE=True` or `ALE="auto"`. 
 	In the latter case, the correction will only be considered if equations for the moving mesh are present, i.e. if :py:meth:`~pyoomph.generic.codegen.Equations.activate_coordinates_as_dofs` has been called by any :py:class:`~pyoomph.generic.codegen.Equations` added to this or any parent domain.
  
	If you want to obtain the mesh velocity, please use it with `ALE=False`.
  
 	
	Args:
		f : The function to differentiate. It can be an expression or a string representing a variable.
		order: The order of the derivative. Defaults to 1, maximum is 2.
		ALE : Flag indicating whether to use Arbitrary Lagrangian-Eulerian (ALE) formulation. Defaults to "auto". "auto" will activate the ALE correction if you combine it with a moving mesh only.
		scheme : The time stepping scheme to use. Defaults to None, meaning the default time stepping scheme set at problem level.
		nondim : Flag indicating whether to use non-dimensional time. Defaults to False.

	Returns:
		The partial derivative of the function with respect to time.
	"""
	
	if isinstance(f,str):
		f=var(f)
	if order==0:
		return Expression(f)
	if scheme in {"TPZ","MPT","Simpson","Boole"}:
		scheme="BDF1"

	TS=Expression(1) if nondim else scale_factor("time")
	if scheme is not None:
		availschemes= {"BDF1","BDF2","Newmark2"}
		if scheme in availschemes:
			t=TS*_pyoomph.GiNaC_get_global_symbol("_dt_"+scheme)
		else:
			raise RuntimeError("Unknown time stepping scheme: "+str(scheme)+". Possible are "+str(availschemes))
	else:
		#t = var("time")
		t=TS * _pyoomph.GiNaC_get_global_symbol("t")

	if ALE!=False:
		if ALE==True:
			if order!=1:
				raise ValueError("Currently, I can only take the first order time derivative with ALE")
			#return diff(f,t)-contract(partial_t(var("mesh"),ALE=False),grad(f))
			return diff(f,t)-directional_derivative(f,mesh_velocity(scheme=scheme))
		elif ALE=="auto":
			if order==1:
				#return diff(f, t) - eval_flag("moving_mesh")*contract(partial_t(var("mesh"),ALE=False),grad(f))
				return diff(f, t) - eval_flag("moving_mesh")*directional_derivative(f,mesh_velocity(scheme=scheme))
			else:
				v = [t] * order
				return diff(f, *v)
		else:
			raise RuntimeError("Unknown ALE flag "+str(ALE)+". Use either True (always ALE), False (never ALE) or 'auto' (ALE only if coordinates are degrees of freedom)")
	else:
		v=[t]*order
		return diff(f,*v)

def material_derivative(f:Union[ExpressionOrNum,str],velocity:Union[ExpressionOrNum,str],ALE:Union[Literal["auto"],bool]="auto",dt_scheme:OptionalTimeSteppingScheme=None,nondim:bool=False,lagrangian:bool=False,dt_factor:ExpressionOrNum=1,advection_factor:ExpressionOrNum=1,coordsys:OptionalCoordinateSystem=None)->Expression:
	"""
	Compute the material derivative of a function with respect to time, i.e. :math:`\\partial_t f + \\nabla f \\cdot \\vec{u}`. Note that for tensorial quantities, one usually uses the :py:func:`upper_convected_derivative` instead.

	Args:
		f: Any scalar, vectorial or tensorial function to be advected.
		velocity: Expression for the velocity field.
		ALE: Use ALE correction. If set to ``"auto"``, it will only be used if the coordinates are degrees of freedom.
		dt_scheme: Used time stepping scheme. If set to None, the default time stepping scheme set at problem level will be used.
		nondim: Using non-dimensional time. Defaults to False.
		lagrangian: Using Lagrangian coordinates for the gradient.
		dt_factor: Factor to weight the :math:`\\partial_t f` term. 
		advection_factor: Factor to weight the advection term
		coordsys: Optional coordinate system to use. Defaults to None, meaning the coordinate system at equation level, parent equation level or problem level.

	Returns:
		The material derivative of the expression ``f`` advected by ``velocity``.
	"""
	if isinstance(f,str):
		f=var(f)
	if isinstance(velocity,str):
		velocity=var(velocity)
	if isinstance(f,float) or isinstance(f,int):
		return Expression(0)
	if not isinstance(velocity,Expression):
		velocity=Expression(velocity)
	adv_term=directional_derivative(f,velocity,nondim=nondim,lagrangian=lagrangian,coordsys=coordsys)
	return dt_factor * partial_t(f, ALE=ALE, scheme=dt_scheme, nondim=nondim) + advection_factor * adv_term

def convected_derivative(A:ExpressionOrNum,velocity:ExpressionOrNum,alpha:ExpressionOrNum=0,ALE:Union[Literal["auto"],bool]=False,dt_scheme:OptionalTimeSteppingScheme=None,nondim:bool=False,lagrangian:bool=False,dt_factor:ExpressionOrNum=1,advection_factor:ExpressionOrNum=1,coordsys:OptionalCoordinateSystem=None)->Expression:
	res=material_derivative(A,velocity,ALE,dt_scheme,nondim,lagrangian,dt_factor,advection_factor,coordsys)
	gradv=grad(velocity,lagrangian=lagrangian,nondim=nondim) # Due to different conventions, this might be transposed in other references!
	g_alpha=0.5*((1+alpha)*gradv-(1-alpha)*transpose(gradv))
	res-=advection_factor*(matproduct(A,transpose(g_alpha))+matproduct(g_alpha,A))
	return res

def upper_convected_derivative(A:ExpressionOrNum,velocity:ExpressionOrNum,ALE:Union[Literal["auto"],bool]="auto",dt_scheme:OptionalTimeSteppingScheme=None,nondim:bool=False,lagrangian:bool=False,dt_factor:ExpressionOrNum=1,advection_factor:ExpressionOrNum=1,coordsys:OptionalCoordinateSystem=None)->Expression:
	"""
 	Returns the upper-convected derivative of a tensor field :math:`\\mathbf{A}`, i.e. :math:`\\partial_t \\mathbf{A} + \\vec{u}\\cdot\\nabla\\mathbf{A}-(\\nabla \\vec{u})^\\mathrm{T}\\cdot \\mathbf{A} - \\mathbf{A}\\cdot\\nabla\\vec{u}`.

	Args:
		A: Rank-2 tensor field to be advected.
		velocity: Advection velocity
		ALE: Use ALE correction. If set to ``"auto"``, it will only be used if the coordinates are degrees of freedom.
		dt_scheme: Used time stepping scheme. If set to None, the default time stepping scheme set at problem level will be used.
		nondim: Using non-dimensional time. Defaults to False.
		lagrangian: Using Lagrangian coordinates for the gradient.
		dt_factor: Factor to weight the :math:`\\partial_t \\mathbf{A}` term.
		advection_factor: Factor to weight the advection term
		coordsys: Optional coordinate system to use. Defaults to None, meaning the coordinate system at equation level, parent equation level or problem level.

	Returns:
		The upper convected derivative of the tensor field :math:`\\mathbf{A}` advected by the velocity field :math:`\\vec{u}`.
	"""
	res=material_derivative(A,velocity,ALE,dt_scheme,nondim,lagrangian,dt_factor,advection_factor,coordsys)
	gradv=grad(velocity,lagrangian=lagrangian,nondim=nondim) # Due to different conventions, this might be transposed in other references! 
	res-=advection_factor*(matproduct(gradv,A)+matproduct(A,transpose(gradv)))
	return res

def directional_derivative(f:ExpressionOrNum,direction:ExpressionOrNum,nondim:bool=False,lagrangian:bool=False,coordsys:OptionalCoordinateSystem=None):
	if isinstance(f,float) or isinstance(f,int):
		return Expression(0)
	flag=(0 if nondim else 1) + (8 if lagrangian else 0) #Code the flag
	if coordsys is None:
		coordsysEx=_pyoomph.Expression(0)
	else:
		coordsysEx=0+_pyoomph.GiNaC_wrap_coordinate_system(coordsys)
	if not isinstance(direction,Expression):
		direction=Expression(direction)	
	return _pyoomph.GiNaC_directional_derivative(f,direction,_pyoomph.Expression(-1),_pyoomph.Expression(-1),coordsysEx,_pyoomph.Expression(flag))

@overload
def testfunction(arg:str,tag:List[str]=[],domain:Union[None,_pyoomph.FiniteElementCode,str]=None,dimensional:bool=True)->Expression: ...

@overload
def testfunction(arg:Expression,tag:List[str]=[],domain:Union[None,_pyoomph.FiniteElementCode,str]=None,dimensional:bool=True)->Expression: ...

@overload
def testfunction(arg:NameStrSequence,tag:List[str]=[],domain:Union[None,_pyoomph.FiniteElementCode,str]=None,dimensional:bool=True)->Tuple[Expression,...]: ...

@overload
def testfunction(arg:ExprStrSequence,tag:List[str]=[],domain:Union[None,_pyoomph.FiniteElementCode,str]=None,dimensional:bool=True)->Tuple[Expression,...]: ...

def testfunction(arg:Union[str,Expression,NameStrSequence,ExprStrSequence],tag:List[str]=[],domain:Union[None,_pyoomph.FiniteElementCode,str]=None,dimensional:bool=True)->SingleOrMultipleExpressions:
	"""
	Return the testfunction corresponding to the field(s) 

	Parameters:
		arg: Select the test function either by by name or by a var expression (or a list thereof).	
		tag: A list of tags associated with the argument. Default is an empty list, see also the tag argument of the var function.
		domain: The domain associated with the argument. Default is None, see also the domain argument of the var function.
		dimensional: A boolean flag indicating whether the we should consider the dimensional test_scale of the testfunction. Default is True.

	Returns:
		The corresponding testfunction(s) as an expression or a tuple of expressions.
	"""
	
	if isinstance(domain,str):
		tag=["domain:"+domain]
		domain=None
	if isinstance(arg,str):
		if dimensional:
			return _pyoomph.GiNaC_dimtestfunction(arg,domain,tag)
		else:
			return _pyoomph.GiNaC_testfunction(arg, domain, tag)
	elif isinstance(arg,Expression):
		if len(tag)==0 and domain is None:
			#Get the domain and tags from the parent
			if dimensional:
				return _pyoomph.GiNaC_dimtestfunction_from_var(arg)
			else:
				return _pyoomph.GiNaC_testfunction_from_var(arg,dimensional)
		else:
			if dimensional:
				return _pyoomph.GiNaC_dimtestfunction(str(arg.op(0)),domain,tag)
			else:
				return _pyoomph.GiNaC_testfunction(str(arg.op(0)), domain, tag)
	else:
		res:List[Expression]=[]
		for n in arg:
			res.append(testfunction(n,tag=tag,domain=domain,dimensional=dimensional))		
		return tuple(res)

testfunction.__test__=False


def diff(f:ExpressionOrNum,*arg:Expression)->Expression:
	"""
	Compute the derivative of a given expression with respect to one or more variables.

	Parameters:
		f (ExpressionOrNum): The expression to differentiate.
		*arg (Expression): The variables with respect to which the differentiation is performed.

	Returns:
		Expression: The resulting expression after differentiation.

	Raises:
		ValueError: If no variables are provided for differentiation.
	"""
	if len(arg)==0:
		raise ValueError("Need to derive with respect to something")
	res=_pyoomph.Expression(f)
	for a in arg:
		res=_pyoomph.GiNaC_Diff(res,a)
	return res


   
def symbolic_diff(expr:ExpressionOrNum,x:Union[Expression,str],hold_until_codegen:bool=True):
	"""
	Compute the symbolic differentiation of an expression with respect to a variable. It will be held, i.e. not applied, until code generation by default.

	Parameters:
		expr (ExpressionOrNum): The expression to differentiate.
		x (Union[Expression,str]): The variable with respect to which to differentiate.
		hold_until_codegen (bool, optional): Whether to hold the expression until code generation. Defaults to True.

	Returns:
		Expression: The result of the symbolic differentiation.

	Raises:
		RuntimeError: If the differentiation cannot be performed.
	"""
	dummy=_pyoomph.GiNaC_new_symbol("__symdiff_dx_"+str(symbolic_diff.counter))
	varis={}
	nondims={}
	params={}
	iplace_subs={}
	   
	if not isinstance(expr,Expression):
		return 0
	
	if isinstance(x,str):    
		varis[x]=dummy        
		nondims[x]=dummy/scale_factor(x)
		iplace_subs[var(x)]=dummy
		iplace_subs[nondim(x)]=dummy/scale_factor(x)
		revsubs=var(x)
	elif isinstance(x,_pyoomph.GiNaC_GlobalParam):
		xn=x.get_name()
		params[xn]=dummy
		iplace_subs[0+x]=dummy
		revsubs=0+x
	else:
		ti=x.get_type_information()
		if ti.get("class_name")=="function" and ti.get("function_name")=="field":
			xn=str(x.op(0))
			varis[xn]=dummy
			nondims[xn]=dummy/scale_factor(xn)
			iplace_subs[x]=dummy
			iplace_subs[nondim(xn)]=dummy/scale_factor(xn)
			revsubs=x
		elif ti.get("class_name")=="function" and ti.get("function_name")=="nondimfield":
			xn=str(x.op(0))
			varis[xn]=dummy*scale_factor(xn)
			nondims[xn]=dummy
			iplace_subs[var(x)]=dummy
			iplace_subs[x]=dummy/scale_factor(xn)
			revsubs=x   
		else:
			raise RuntimeError("Cannot symbolically derive with respect to "+str(x))         
	if hold_until_codegen:
		exprs=_pyoomph.GiNaC_subsfields(expr,varis,nondims,params)
	else:
		exprs=expr
		for fr,to in iplace_subs.items():            
			exprs=_pyoomph.GiNaC_subs(exprs,fr,to)
	  
	dbydummy=_pyoomph.GiNaC_diff(exprs,dummy)
	if hold_until_codegen:
		res=_pyoomph.GiNaC_SymSubs(dbydummy,dummy,revsubs)
	else:
		res=_pyoomph.GiNaC_subs(dbydummy,dummy,revsubs)
	return res
symbolic_diff.counter=0
        

def matrix(mlist:List[List[ExpressionOrNum]],fill_to_max_vector_dim:bool=True,fill_identity:bool=False)->Expression:
	"""
	Create a GiNaC matrix from a list of lists.

	Args:
		mlist (List[List[ExpressionOrNum]]): The input matrix represented as a list of lists.
		fill_to_max_vector_dim (bool, optional): Whether to fill the matrix to the maximum vector dimension 3. Defaults to True.
		fill_identity (bool, optional): Whether to fill the matrix diagonal by 1 when filling to max_vector_dim of 3. Defaults to False.

	Returns:
		Expression: The matrix expression created from the input.

	Raises:
		ValueError: If the number of matrix rows exceeds the maximum vector dimension.
		ValueError: If the input matrix is not a valid matrix.
		ValueError: If a matrix row exceeds the maximum vector dimension.
	"""
	nrow=len(mlist)
	ncol=len(mlist[0])
	res:List[Expression]=[]
	vd=_pyoomph.GiNaC_vector_dim()
	if nrow>vd:
		raise ValueError("too many matrix rows: " + str(mlist))
	for ri,row in enumerate(mlist):
		if ncol!=len(row):
			raise ValueError("Not a valid Matrix:"+str(mlist))
		if len(row)>vd:
			raise ValueError("too large matrix row: "+str(row))
		for entry in row:
			if isinstance(entry,_pyoomph.Expression):
				res.append(entry)
			else:
				res.append(_pyoomph.Expression(entry))
		if fill_to_max_vector_dim:
			for i in range(len(row),vd): # type: ignore
				res.append(_pyoomph.Expression(1 if fill_identity and ri==i else 0))
	if fill_to_max_vector_dim:
		for i in range(len(mlist),vd): # type: ignore
			for j in range(vd): # type: ignore
				res.append(_pyoomph.Expression(1 if fill_identity and i==j else 0))
		ncol=vd

	return _pyoomph.GiNaC_Matrix(ncol,res)

def identity_matrix(dim: int = -1) -> Expression:
	"""
	Returns the identity matrix of the specified dimension.

	Parameters:
		dim (int): The dimension of the identity matrix. If not provided, the dimension is 3.

	Returns:
		Expression: The identity matrix.

	"""
	rs = []
	if dim == -1:
		dim = _pyoomph.GiNaC_vector_dim()
	for i in range(dim):
		rs.append([Expression(1) if i == j else Expression(0) for j in range(dim)])
	return matrix(rs)


def dyadic(a: Expression, b: Expression) -> Expression:
    """
	Compute the dyadic product of two expressions.

	Args:
		a (Expression): The first expression.
		b (Expression): The second expression.

	Returns:
		Expression: The dyadic product of `a` and `b`.
    """
    return matrix([[a[i] * b[j] for j in range(3)] for i in range(3)])


def unit_vector(dir: Union[int, Literal["x", "y", "z"]]) -> Expression:
	"""
	Returns a unit vector in the specified direction.

	Parameters:
		dir (Union[int, Literal["x", "y", "z"]]): The direction of the unit vector.
			Can be either an integer (0, 1, or 2) or a string ("x", "y", or "z").

	Returns:
		Expression: A vector expression representing the unit vector in the specified direction.
	"""

	if isinstance(dir, str):
		if dir == "x":
			dir = 0
		elif dir == "y":
			dir = 1
		elif dir == "z":
			dir = 2

	comps = [1 if i == dir else 0 for i in range(3)]
	return vector(comps)


# Calculates the jump for discontinuous expressions. 
# By default, they are evaluated in the bulk of both elements
# This is helpful for jump(grad(f)), where you usually want to have the grad with repect to the bulk element, not the surface gradient along the facet
# If you however want to calculate e.g. an upwind scheme, you usually have something like jump(un*f), where un=(dot(u, n) + absolute(dot(u, n)))/2
# In that case, jump would evaluate the normal n in the bulk element and you must set at_facet=True to get the facet normals
def jump(f:ExpressionOrNum, at_facet:bool=False):
	"""
	Calculate the jump of the given expression for discontinuous Galerkin methods.

	Parameters:
		f (ExpressionOrNum): The expression to calculate the jump for.
		at_facet (bool, optional): If True, calculate the jump at the facet, i.e. between the values at the facet directly. If you wrap it on a spatial differential operator, you will get the jump of e.g. the surface gradient.
			If False, calculate the jump between the two bulk domains. Here, the bulk gradients will be taken when applied with e.g. a grad. Default is False.

	Returns:
		Expression: The jump across the interface, f_inside - f_outside.

	"""
	if at_facet:
		return evaluate_in_domain(f, '+|') - evaluate_in_domain(f, '|-')
	else:
		return evaluate_in_domain(f, '+') - evaluate_in_domain(f, '-')

	
# Calculates the avergage for discontinuous expressions. 
# By default, they are evaluated in the bulk of both elements
# This is helpful for avg(grad(f)), where you usually want to have the grad with repect to the bulk element, not the surface gradient along the facet
def avg(f:ExpressionOrNum,at_facet:bool=False):
	"""
	Calculate the average of the given expression for discontinuous Galerkin methods.

	Parameters:
		f (ExpressionOrNum): The expression to calculate the average for.
		at_facet (bool, optional): If True, calculate the average at the facet, i.e. between the values at the facet directly. If you wrap it on a spatial differential operator, you will get the average of e.g. the surface gradient.
			If False, calculate the average between the two bulk domains. Here, the bulk gradients will be taken when applied with e.g. a grad. Default is False.

	Returns:
		Expression: The average across the interface.  (f_inside + f_outside)/2.

	"""    
	if at_facet:
		return (evaluate_in_domain(f,'+|')+evaluate_in_domain(f,'|-'))/2
	else:
		return (evaluate_in_domain(f,'+')+evaluate_in_domain(f,'-'))/2

def vector(*args:Union[ExpressionOrNum,List[ExpressionOrNum]])->Expression:
	"""
	Create a vector expression from the given components.

	Args:
		*args: Variable number of arguments representing the vector components. Each component can be either an Expression, float, or int. 
			   If a single list is provided as the first argument, it is treated as the list of vector components.

	Returns:
		Expression: The vector expression created from the given components.

	Raises:
		RuntimeError: If the arguments are not provided correctly.
		ValueError: If the vector has too many components or no components at all.
	"""
	if len(args)==0:
		return _pyoomph.Expression(0)
	a0=args[0]
	if isinstance(a0,list):
		vlist=a0
		if len(args)!=1:
			raise RuntimeError("Either call vector(compo1,compo2,...) or vector([compo1,compo2,...])")
	else:
		vlist:List[ExpressionOrNum]=[]
		for a in args:
			a=0+a
			if not isinstance(a,(_pyoomph.Expression,float,int)):
				raise RuntimeError("Strange vector component "+str(a))
			vlist.append(a)	
	
	exlist:List[Expression]=[]
	for a in vlist:
		if isinstance(a,_pyoomph.Expression):
			exlist.append(a)
		else:
			exlist.append(_pyoomph.Expression(a))

	if len(exlist)==0:
		raise ValueError("Cannot create an empty vector")
	while len(exlist)<_pyoomph.GiNaC_vector_dim():
		exlist.append(_pyoomph.Expression(0))
	if len(exlist)>_pyoomph.GiNaC_vector_dim():
		raise ValueError("Vector has too many components: "+str(exlist))
	return _pyoomph.GiNaC_Vect(exlist)


def convert_to_expression(a:Union[ExpressionOrNum,NPAnyArray])->Expression:
	if isinstance(a,_pyoomph.Expression):
		return a
	if isinstance(a,numpy.ndarray): 
		return _pyoomph.Expression([convert_to_expression(c) for c in a])
	else:
		return _pyoomph.Expression(a)

def dot(a:Union[ExpressionOrNum,str],b:Union[ExpressionOrNum,str])->Expression:    
	"""
	Compute the dot product between two vectors.

	Parameters:
	a: The first vector. String will be wrapped in a var expression.
	b: The second vector. String will be wrapped in a var expression.

	Returns:
	Expression: The dot product of the two vectors.
	"""
	if isinstance(a,str):
		a=var(a)
	if isinstance(b,str):
		b=var(b)
	if not isinstance(a,_pyoomph.Expression):
		a=convert_to_expression(a)
	if not isinstance(b,_pyoomph.Expression):
		b=convert_to_expression(b)
	return _pyoomph.GiNaC_dot(a,b)


def double_dot(A:Expression,B:Expression):
	"""
	Compute the double dot product of two expressions A and B.
	
	Parameters:
		A (Expression): The first expression.
		B (Expression): The second expression.
	
	Returns:
		Expression: The result of the double dot product.
	"""
	return _pyoomph.GiNaC_double_dot(A,B)

def matproduct(a:ExpressionOrNum,b:ExpressionOrNum)->Expression:
	"""
	Compute the matrix product of two expressions.

	Parameters:
	a (ExpressionOrNum): The first expression.
	b (ExpressionOrNum): The second expression.

	Returns:
	Expression: The matrix product of a and b.

	Note:
	- If either a or b is not an Expression, it will be converted to one.
	- If either a or b is zero, the function will return zero.
	- The matrix product can be computed for matrices as well as vectors.
	"""
	
	if not isinstance(a,_pyoomph.Expression):
		a=_pyoomph.Expression(a)
	if not isinstance(b,_pyoomph.Expression):
		b=_pyoomph.Expression(b)
	if a.is_zero() or b.is_zero():
		return Expression(0)
	return _pyoomph.GiNaC_matproduct(a,b) #Matrix product M1*M2 (can of course also be vectors)


def transpose(A: Expression):
	"""
	Transposes the given matrix.

	Args:
		A (Expression): The matrix expression to be transposed.

	Returns:
		Expression: The transposed matrix.
	"""
	return _pyoomph.GiNaC_transpose(A)

unit_matrix=identity_matrix


def subexpression(what: ExpressionOrNum) -> Expression:
	"""
	Wraps the expression in a subexpression. This will be calculated and derived in beforehand during the code generation and can speed up the assembly.
	Does not work in symbolical Hessians at the moment.

	Parameters:
		what (ExpressionOrNum): What to wrap in a subexpression.

	Returns:
		Expression: A resulting expression where the expression is marked to be calculated in beforehand.

	Raises:
		None	
	"""
	if isinstance(what, _pyoomph.Expression):
		return _pyoomph.GiNaC_subexpression(what)
	else:
		return _pyoomph.GiNaC_subexpression(_pyoomph.Expression(what))

def delayed_lambda_expansion(func:Callable[[],Expression])->Expression:
	"""
	Wrapping a callable f, so that it will be evaluated only at the moment of code generation. Can be used to e.g. access potentially changing problem parameters the moment the code is generated.

	Args:
		func (Callable[[], Expression]): The function to be expanded when the code is generated.

	Returns:
		Expression: An expression which you can combine with other expressions, f will be evaluated at code generation time.

	Raises:
		RuntimeError: If the input `func` is not callable.
		ValueError: If the result of the expansion cannot be converted into an expression.
	"""
	def wrapped() -> Expression:
		res=func()
		if not isinstance(res,_pyoomph.Expression): # type:ignore
			try:
				res=_pyoomph.Expression(res)
			except:
				raise ValueError("delayed_lambda_expansion is evaluated to '"+str(res)+"', which cannot be converted into an expression")
		return res
	if not callable(func):
		raise RuntimeError("must pass a callable. Usually just 'lambda : return_expression'")
	return 0+_pyoomph.GiNaC_delayed_expansion(wrapped)


def evaluate_in_domain(expr:ExpressionOrNum,domain:Union[str,_pyoomph.FiniteElementCode,None]):
	"""
	Evaluates the given expression within the specified domain. Each var inside the expression will be evaluated in the given domain. Useful to e.g. evaluate the expression in the bulk (domain="``..``") when at an boundary element, or at the opposite side of the interface (domain="``|.``")

	Args:
		expr (ExpressionOrNum): The expression to be evaluated.
		domain (Union[str,_pyoomph.FiniteElementCode,None]): The domain within which the expression should be evaluated. 
			It can be a string representing the domain name or e.g. "``..``" for the bulk, "``|.``" for the opposite interface side or "``|..``" for the opposite bulk side

	Returns:
		The evaluated expression within the specified domain.
	"""
	tags=[]
	if isinstance(domain,str):
		tags=["domain:"+domain]
		domain=None
	if not isinstance(expr,_pyoomph.Expression):
		expr=_pyoomph.Expression(expr)
	return _pyoomph.GiNaC_eval_in_domain(expr,domain,tags)


def evaluate_in_past(expr:ExpressionOrNum,timestep_offset:Union[int,float]=1)->Expression:
	"""
	Evaluate the given expression in the past at a specified time offset.

	Args:
		expr: The expression to be evaluated.
		timestep_offset: The time offset in the past. Defaults to 1, i.e. the previously converged solution.

	Returns:
		The evaluated expression in the past.

	Raises:
		RuntimeError: If the time offset is negative.
		RuntimeError: If the time offset is a variable step.

	Note:
		If the time offset is an integer, the function evaluates the expression at that specific time step.
		If the time offset is a float, the function linearly interpolates between the results of the two nearest time steps.
		If the time offset is a variable step, the function raises a RuntimeError.
	"""
	if not isinstance(expr,_pyoomph.Expression):
		expr=_pyoomph.Expression(expr)
	if timestep_offset<0:
		raise RuntimeError("Cannot evaluate in future, i.e. time offset needs to be >=0")
	if round(timestep_offset)==timestep_offset:
		if timestep_offset==0:
			return expr
		else:			
			return 0+_pyoomph.GiNaC_eval_in_past(expr,_pyoomph.Expression(int(timestep_offset)),_pyoomph.Expression(0))
	elif isinstance(timestep_offset,float):
		tlow:int=math.floor(timestep_offset)
		thigh:int=math.ceil(timestep_offset)
		frac_high=timestep_offset-tlow
		frac_low=thigh-timestep_offset
		if tlow==0:
			low=expr
		else:
			low=_pyoomph.GiNaC_eval_in_past(expr,_pyoomph.Expression(tlow),_pyoomph.Expression(0))
		high=_pyoomph.GiNaC_eval_in_past(expr,_pyoomph.Expression(thigh),_pyoomph.Expression(0))
		return low*frac_low+high*frac_high
	else:
		raise RuntimeError("cannot yet evaluate at a variable step. But you can use e.g. (1-theta)*evalulate_at_past(expr,1)+theta*expr for some variable theta to blend between current and previous time step")


def evaluate_at_midpoint(expr:ExpressionOrNum, midpt:Union[float,int]=0.5)->Expression:
	"""
	Evaluates the given expression by replacing each var by a blending between the history values.

	Args:
		expr: The expression to be evaluated.
		midpt: The blending at which to evaluate the var statements. Defaults to 0.5, i.e. all variables are evaluated at the average between current and previous time step (midpoint rule)

	Returns:
		The evaluated expression.

	Raises:
		RuntimeError: If the expression is not an Expression.
		RuntimeError: If the midpoint is less than 0.
		RuntimeError: If the midpoint is a variable fraction (not yet supported).
	"""
	if not isinstance(expr,_pyoomph.Expression):
		expr=_pyoomph.Expression(expr)
	if midpt < 0:
		raise RuntimeError("Cannot evaluate in future, i.e. midpt to be >=0")
	if round(midpt) == midpt:
		if midpt == 0:
			return expr
		else:
			return _pyoomph.GiNaC_eval_in_past(expr, _pyoomph.Expression(int(midpt)),_pyoomph.Expression(0))
	elif isinstance(midpt, float):
		return _pyoomph.GiNaC_eval_in_past(expr, _pyoomph.Expression(midpt),_pyoomph.Expression(0))
	else:
		raise RuntimeError("cannot yet evaluate at a variable midpoint fraction")

def time_scheme(scheme:TimeSteppingScheme,expr:ExpressionOrNum,only_implicit_terms:bool=False)->Expression:
	"""
	Selects a time stepping scheme for the given expression by replacing all partial_t terms by the corresponding time stepping and expanding all other terms by appropriate evalulations in the past.

	Args:
		scheme: The time stepping scheme to apply ("BDF1","BDF2","Newmark2","TPZ","MPT","Simpson","Boole")
		expr: The expression to apply the time stepping scheme to.
		only_implicit_terms: Whether to only evaluate the implicit terms (history terms will be not affected). Defaults to False.

	Returns:
		The result of applying the time stepping scheme to the expression.

	Raises:
		ValueError: If an unknown time stepping scheme is provided.
	"""
	
	if not isinstance(expr,_pyoomph.Expression):
		expr=_pyoomph.Expression(expr)
	def ev(expr:Expression,where:float,taction:int)->Expression:
		if only_implicit_terms:
			if where==0:
				return _pyoomph.GiNaC_eval_in_past(expr, _pyoomph.Expression(where), _pyoomph.Expression(taction))
			elif where!=1 and where!=2:
				raise RuntimeError("Cannot do this right now")
			else:
				return Expression(0)
		else:
			return _pyoomph.GiNaC_eval_in_past(expr, _pyoomph.Expression(where),_pyoomph.Expression(taction))
	if scheme=="BDF1":
		return ev(expr, 0,1)
	elif scheme=="BDF2":
		return ev(expr, 0,2)
	elif scheme=="Newmark2":
		return ev(expr, 0,3)
	elif scheme=="TPZ" or scheme=="trapezoidal":
		return 0.5*(ev(expr, 0,1)+ev(expr, 1,1))
	elif scheme=="MPT" or scheme=="midpoint":
		return ev(expr, 0.5,1)
	elif scheme=="Simpson" or scheme=="Kepler":
		return (ev(expr, 0,1)+4*ev(expr, 0.5,1)+ev(expr, 1,1))/6
	elif scheme=="Boole" or scheme=="Milne" or scheme=="Bode":
		return (7*ev(expr,0,1)+32*ev(expr,0.25,1)+12*ev(expr,0.5,1)+32*ev(expr,0.75,1)+7*ev(expr,1,1))/90
	else:
		raise ValueError("Unknown time stepping scheme: "+str(scheme))


get_global_symbol=_pyoomph.GiNaC_get_global_symbol




def rational_num(numer_or_float_str: Union[int, str], denom: int = 1) -> Expression:
	"""
	Create a rational number expression at full accuracy. Opposed to floats, this will not introduce any rounding errors.

	Args:
		numer_or_float_str (Union[int, str]): The numerator of the rational number.
			If a string is provided, it will be interpreted as a float that should be converted to a rational.
		denom (int, optional): The denominator of the rational number. Defaults to 1.

	Returns:
		Expression: The rational number expression.

	Raises:
		RuntimeError: If `numer_or_float_str` is a string and `denom` is not equal to 1.

	"""
	if isinstance(numer_or_float_str, str):
		if denom != 1:
			raise RuntimeError("Either call it with two integers (numer,denom) or with a string, e.g. a float that should be converted to a rational")
		return num(numer_or_float_str, rational=True)
	else:
		return _pyoomph.GiNaC_rational_number(numer_or_float_str, denom)
