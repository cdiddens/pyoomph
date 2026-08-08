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
 
"""
This module provides the core functionality to formulate mathematical expressions in the pyoomph library. 
"""
 
import  pyoomph._pyoomph_core as _pyoomph

from .generic import *
from .coordsys import *
from .coordsys import BaseCoordinateSystem
from .cb import *

from ..typings import *
if TYPE_CHECKING:
	from ..generic.codegen import FiniteElementCodeGenerator

cartesian=CartesianCoordinateSystem()
axisymmetric=AxisymmetricCoordinateSystem()
axisymmetric_flipped=AxisymmetricCoordinateSystem(use_x_as_symmetry_axis=True)
radialsymmetric=RadialSymmetricCoordinateSystem()

pi=pi

debug_ex=_pyoomph.GiNaC_debug_ex

def __wrap_ginac_func(f:Callable[..., Expression])->Callable[..., Expression]:
	def _checkargs(*args:ExpressionOrNum) -> Expression:
		newargs=[a if isinstance(a,_pyoomph.Expression) else _pyoomph.Expression(a) for a in args]
		return f(*newargs)
	return _checkargs




def sin(x:ExpressionOrNum) -> Expression:
	"""
	Compute the sine of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The sine of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_sin(x)



def cos(x:ExpressionOrNum) -> Expression:
	"""
	Compute the cosine of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The cosine of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_cos(x)


def sinh(x:ExpressionOrNum) -> Expression:
	"""
	Compute the hyperbolic sine of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The hyperbolic sine of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_sinh(x)


def cosh(x:ExpressionOrNum) -> Expression:
	"""
	Compute the hyperbolic cosine of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The hyperbolic cosine of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_cosh(x)


def tan(x:ExpressionOrNum) -> Expression:
	"""
	Compute the tangent of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The tangent of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_tan(x)

def tanh(x:ExpressionOrNum) -> Expression:
	"""
	Compute the hyperbolic tangent of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The hyperbolic tangent of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_tanh(x)


def atan(x:ExpressionOrNum) -> Expression:
	"""
	Compute the inverse tangent of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The inverse tangent of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_atan(x)


def atan2(y:ExpressionOrNum,x:ExpressionOrNum) -> Expression:
	"""
	Compute atan2(y,x) of the input expression or number.

	Parameters:
		y (ExpressionOrNum): First argument, expression or number.
  		x (ExpressionOrNum): Second argument. expression or number.

	Returns:
		Expression: atan2(y,x), i.e. atan(y/x) with case distinction.

	"""
	y=y if isinstance(y,_pyoomph.Expression) else _pyoomph.Expression(y) 	
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	  	
	return _pyoomph.GiNaC_atan2(y,x)


def asin(x:ExpressionOrNum) -> Expression:
	"""
	Compute the inverse sine of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The inverse sine of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_asin(x)

def acos(x:ExpressionOrNum) -> Expression:
	"""
	Compute the inverse cosine of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The inverse cosine of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x)
	return _pyoomph.GiNaC_acos(x)

def asinh(x:ExpressionOrNum) -> Expression:
	"""
	Compute the inverse hyperbolic sine of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The inverse hyperbolic sine of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x)
	return _pyoomph.GiNaC_asinh(x)

def acosh(x:ExpressionOrNum) -> Expression:
	"""
	Compute the inverse hyperbolic cosine of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The inverse hyperbolic cosine of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x)
	return _pyoomph.GiNaC_acosh(x)

def atanh(x:ExpressionOrNum) -> Expression:
	"""
	Compute the inverse hyperbolic tangent of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The inverse hyperbolic tangent of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x)
	return _pyoomph.GiNaC_atanh(x)

def erf(x:ExpressionOrNum) -> Expression:
	"""
	Compute the error function of the input expression or number.

	The argument must be dimensionless and is assumed to be real-valued. Differentiates to
	2/sqrt(pi)*exp(-x**2), so it can be used in residuals without further ado.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The error function of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x)
	return _pyoomph.GiNaC_erf(x)

def erfc(x:ExpressionOrNum) -> Expression:
	"""
	Compute the complementary error function, i.e. 1-erf(x), of the input expression or number.

	Prefer it over writing 1-erf(x) by hand: for large arguments, the latter is entirely
	cancellation, whereas erfc retains its accuracy.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The complementary error function of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x)
	return _pyoomph.GiNaC_erfc(x)

def exp(x:ExpressionOrNum) -> Expression:
	"""
	Compute the exponential of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The exponential of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_exp(x)

def log(x:ExpressionOrNum) -> Expression:
	"""
	Compute the logarithm of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The logarithm of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_log(x)

def absolute(x:ExpressionOrNum) -> Expression:
	"""
	Compute the absolute of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The absolute of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_absolute(x)


def signum(x:ExpressionOrNum) -> Expression:
	"""
	Compute the signum of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The signum of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	
	return _pyoomph.GiNaC_signum(x)




def maximum(x:ExpressionOrNum,y:ExpressionOrNum) -> Expression:
	"""
	Compute the maximum of both input expressions or numbers.

	Parameters:
		x (ExpressionOrNum): First argument, expression or number.
  		y (ExpressionOrNum): Second argument. expression or number.

	Returns:
		Expression: max(x,y).

	"""	
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	  	
	y=y if isinstance(y,_pyoomph.Expression) else _pyoomph.Expression(y) 	
	return _pyoomph.GiNaC_maximum(x,y)


def minimum(x:ExpressionOrNum,y:ExpressionOrNum) -> Expression:
	"""
	Compute the minimum of both input expressions or numbers.

	Parameters:
		x (ExpressionOrNum): First argument, expression or number.
  		y (ExpressionOrNum): Second argument. expression or number.

	Returns:
		Expression: min(x,y).

	"""	
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x) 	  	
	y=y if isinstance(y,_pyoomph.Expression) else _pyoomph.Expression(y) 	
	return _pyoomph.GiNaC_minimum(x,y)

def imaginary_i():
	"""
	Return the imaginary unit i.

	Returns:
		Expression: The imaginary unit i.

	"""	
	return _pyoomph.GiNaC_imaginary_i()

def real_part(x:ExpressionOrNum) -> Expression:
	"""
	Compute the real part of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The real part of the input.

	"""
	x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(Expression(1)*x) 	
	return _pyoomph.GiNaC_get_real_part(x)

def imag_part(x:ExpressionOrNum)->Expression:
    """
	Compute the imaginary part of the input expression or number.

	Parameters:
		x (ExpressionOrNum): The input expression or number.

	Returns:
		Expression: The imaginary part of the input.

	"""	
    x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(Expression(1)*x) 	
    return _pyoomph.GiNaC_get_imag_part(x)
    



def square_root(what:ExpressionOrNum, order:int=2) -> Expression:
	"""
	Calculates the square root of the given expression or number.

	Parameters:
		what (ExpressionOrNum): The expression or number to calculate the square root of.
		order (int): The order of the square root. Default is 2.

	Returns:
		Expression: The square root of the given expression or number.
	"""
	what = what if isinstance(what, _pyoomph.Expression) else _pyoomph.Expression(what)
	return what ** rational_num(1, order)


#def piecewise(condition,true_result,false_result):
#	true_result=true_result if isinstance(true_result, _pyoomph.Expression) else _pyoomph.Expression(true_result)
#	false_result = false_result if isinstance(false_result, _pyoomph.Expression) else _pyoomph.Expression(false_result)
#	if isinstance(condition,bool):
#		if condition:
#			return true_result
#		else:
#			return false_result
#	else:
#		condition=condition if isinstance(condition,_pyoomph.Expression) else _pyoomph.Expression(condition)
#		return _pyoomph.GiNaC_piecewise(condition,true_result,false_result)

def heaviside(x:ExpressionOrNum):
    """
	Heaviside step function: 1 for `x`>0, 0 for `x`<0 and 1/2 for `x`=0.

	`x` may carry any unit (its scale and unit are divided out), the result is always dimensionless.
	Note that it differentiates to zero everywhere, i.e. the delta contribution at the jump is not
	added to the Jacobian.

	Parameters:
		x (ExpressionOrNum): The argument of the step function.

	Returns:
		Expression: The step function of the input.
	"""
    x=x if isinstance(x,_pyoomph.Expression) else _pyoomph.Expression(x)
    return _pyoomph.GiNaC_heaviside(x)

def piecewise_geq0(cond:ExpressionOrNum,iftrue:ExpressionOrNum,iffalse:ExpressionOrNum)->Expression:
	"""
	Returns a piecewise function that evaluates to `iftrue` when `cond` is greater than or equal to zero,
	and evaluates to `iffalse` otherwise.

	All three arguments may carry units: `cond` only decides the branch, so its unit is arbitrary and
	divided out, whereas `iftrue` and `iffalse` must agree in units, which then are the units of the
	result. A plain 0 is accepted as a branch irrespective of the unit of the other branch.

	Parameters:
		cond (ExpressionOrNum): The condition to check.
		iftrue (ExpressionOrNum): The value to return if `cond` is greater than or equal to zero.
		iffalse (ExpressionOrNum): The value to return if `cond` is less than zero.

	Returns:
		Expression: The resulting piecewise function.
	"""
	cond=cond if isinstance(cond,_pyoomph.Expression) else _pyoomph.Expression(cond)
	iftrue=iftrue if isinstance(iftrue,_pyoomph.Expression) else _pyoomph.Expression(iftrue)
	iffalse=iffalse if isinstance(iffalse,_pyoomph.Expression) else _pyoomph.Expression(iffalse)	
	#print("cond",cond,iftrue,iffalse)
	return _pyoomph.GiNaC_piecewise_geq0(cond,iftrue,iffalse)
	#return heaviside(cond)*(iftrue-iffalse)+iffalse

def trace(M:Expression)->Expression:
	"""
	Compute the trace of a matrix expression.

	Parameters:
	M (Expression): The matrix expression for which to compute the trace.

	Returns:
	Expression: The trace of the matrix expression.
	"""
	return _pyoomph.GiNaC_trace(M)

def determinant(M:Expression,n:int=0)->Expression:
	"""
	Compute the determinant of a matrix expression.

	Parameters:
		M (Expression): The matrix expression for which to compute the determinant.
		n (int): Range of the matrix to consider for the determinant. Default is 0 (extract nonzero block), <0 means full matrix, >0 upper left matrix of n x n.
  

	Returns:
		Expression: The determinant of the matrix expression.
	"""
	return _pyoomph.GiNaC_determinant(M,Expression(n))


def inverse_matrix(M:Expression,n:int=0,use_subexpression_for_det:bool=True,fill_to_vector_dim_3:bool=False,skip_empty_rows_and_cols:bool=False)->Expression:
	"""
	Compute the inverse of a matrix expression.

	Parameters:
		M (Expression): The matrix expression for which to compute the determinant.
		n (int): Range of the matrix to consider for the inverse. Default is 0 (extract nonzero block), <0 means full matrix, >0 upper left matrix of n x n.
		use_subexpression_for_det (bool): Flag indicating whether to use a subexpression for the determinant. Default is True.
  		skip_empty_rows_and_cols: Analyze the input and skip empty rows and columns. These empty rows and columns will be added to the result as zero rows and columns of the inverse. Default is False.
  

	Returns:
		Expression: The symbolical inverse of the matrix expression.
	"""
	flags=1 if use_subexpression_for_det else 0
	flags+=2 if fill_to_vector_dim_3 else 0
	flags+=4 if skip_empty_rows_and_cols else 0
	return _pyoomph.GiNaC_inverse_matrix(M,Expression(n),Expression(flags))


def var_and_test(n: str, tag: list[str] = [], domain: "None | str | FiniteElementCodeGenerator" = None) -> tuple[Expression, Expression]:
	"""
	Bind a variable of an unknown field the corresponding test function for a given name.

	Args:
		n (str): The name of the unkown.
		tag (List[str], optional): List of tags for the variable and test function. Defaults to [], see :py:func:`~pyoomph.expressions.generic.var`
		domain (Union[None, str, "FiniteElementCodeGenerator"], optional): The domain of the variable and test function. Defaults to None, see :py:func:`~pyoomph.expressions.generic.var`

	Returns:
		Tuple[Expression, Expression]: A tuple containing the field and test function as expressions.
	"""
	return var(n, tag=tag, domain=domain), testfunction(n, tag=tag, domain=domain)


def sym(a: Expression) -> Expression:
	"""
	Calculate the symmetric part of a given matrix.

	Parameters:
		a (Expression): The input matrix expression.

	Returns:
		Expression: The symmetric part of the matrix.
	"""
	return (a + transpose(a)) / 2


def partial_x(f:ExpressionOrNum, order:int=1) -> Expression:
	"""
	Compute the partial derivative of a given expression with respect to the x-coordinate.

	Parameters:
		f (ExpressionOrNum): The expression to differentiate.
		order (int): The order of differentiation (default is 1).

	Returns:
		Expression: The resulting expression after differentiation.
	"""
	if order == 0:
		if isinstance(f, Expression):
			return f
		else:
			return _pyoomph.Expression(f)
	x = var("coordinate_x")
	v = [x] * order
	return diff(f, *v)



def partial_y(f:ExpressionOrNum,order:int=1)->Expression:
	"""
	Compute the partial derivative of a given expression with respect to the y-coordinate.

	Parameters:
		f (ExpressionOrNum): The expression to differentiate.
		order (int): The order of differentiation (default is 1).

	Returns:
		Expression: The resulting expression after differentiation.
	"""
	if order==0:
		if isinstance(f,Expression):
			return f
		else:
			return _pyoomph.Expression(f)
	y=var("coordinate_y")
	v=[y]*order
	return diff(f,*v)


def partial_z(f:ExpressionOrNum,order:int=1)->Expression:
	"""
	Compute the partial derivative of a given expression with respect to the y-coordinate.

	Parameters:
		f (ExpressionOrNum): The expression to differentiate.
		order (int): The order of differentiation (default is 1).

	Returns:
		Expression: The resulting expression after differentiation.
	"""
	if order==0:
		if isinstance(f,Expression):
			return f
		else:
			return _pyoomph.Expression(f)
	y=var("coordinate_z")
	v=[y]*order
	return diff(f,*v)





def div(arg:ExpressionOrNum,lagrangian:bool=False,matrix:bool | None=None,nondim:bool=False,coordsys:"BaseCoordinateSystem | None"=None) -> Expression:
	"""
	Compute the divergence of the given argument. On surfaces, i.e. with a co-dimension, it is the surface divergence.	

	Parameters:
	arg (ExpressionOrNum): The argument for which the divergence is computed.
	lagrangian (bool, optional): Flag indicating whether the computation is with respect to Lagrangian coordinates. Defaults to False.
	matrix (bool, optional): Flag indicating whether the computation is for a matrix expression. Defaults to None, i.e. auto-select.
	nondim (bool, optional): Flag indicating whether the computation is with respect to non-dimensional coordinates. Defaults to False.
	coordsys (BaseCoordinateSystem, optional): The coordinate system in which the computation is performed. Defaults to None, i.e. the coordinate system of either the current or parent equations or the problem.

	Returns:
	Expression: The computed divergence expression.
 
	Notes:
		if you calculate div(u) on a boundary, you will get the surface divergence, even if u is defined in the bulk.
		To get the bulk divergence at the boundary, use div(var("u",domain="..")) instead.

		Index order: for a rank-2 tensor, ``div(T)[i]`` is :math:`\\partial_j T_{ij}`, contracting the second index. This
		makes ``div`` the adjoint of :py:func:`~pyoomph.expressions.generic.grad` (which stores
		:math:`\\partial u_i/\\partial x_j`), so ``div(grad(u))`` is the vector Laplacian, and it makes ``div(T)`` the
		integration-by-parts partner of ``weak(T,grad(v))`` together with the traction ``matproduct(T,n)``. For symmetric
		tensors, i.e. every usual stress tensor, the distinction does not matter. Note that a flux tensor has to be
		assembled accordingly: the momentum flux carrying :math:`\\vec{u}` along :math:`\\rho\\vec{q}` is
		``dyadic(u,rho*q)``, so that :math:`F_{ij}=\\rho u_i q_j`.
	"""
	
	if isinstance(arg,float) or isinstance(arg,int):
		return Expression(0)
	with_scaling=not nondim
	flag=(1 if with_scaling else 0)+(0 if matrix is None else (2 if matrix==False else 4) ) + (8 if lagrangian else 0) #Code the flag
	if coordsys is None:
		coordsysE=_pyoomph.Expression(0)
	else:
		coordsysE=0+_pyoomph.GiNaC_wrap_coordinate_system(coordsys)
	return _pyoomph.GiNaC_div(arg,_pyoomph.Expression(-1),_pyoomph.Expression(-1),coordsysE,_pyoomph.Expression(flag))


def time_derivative_of_integral(expr:ExpressionOrNum,scheme:Literal["BDF1","BDF2","Newmark2","BDF2_degr","Newmark2_degr","TPZ","MPT","Simpson","Boole","trapezoidal","Kepler","Milne","midpoint"]="BDF2_degr",apply_on_others:bool=True)->Expression:
    """
    Computes the time derivative of an integral expression using a given time stepping scheme.
    For moving meshes, this can be different, i.e. ``weak(partial_t(u),v) != d_by_dt_of_integral(weak(u,v))``.
    The latter is d/dt (Integral_Element(t) u(t)*v(t)*dx ), i.e. considers the change of the element size as well.
    
    Args:
        expr: The expression to differentiate.
        scheme: The time stepping scheme to apply ("BDF1","BDF2","Newmark2","BDF2_degr","Newmark2_degr"). Defaults to "BDF2_degr", "_degr" means that the time derivative is approximated with a lower order scheme in the first step, since initial conditions might not have history values.
        apply_on_others: Whether the history evaluations also take the normal, the Eulerian element sizes and the Eulerian spatial derivatives in grad/div from the mesh of the corresponding history step. Defaults to True: this is a derivative of the integral over the moving element, so each history term belongs to the configuration the element had then. Has no effect unless the mesh moves.
    """        
    if scheme in ["TPZ","MPT","Simpson","Boole","trapezoidal","Kepler","Milne","midpoint"]:
        scheme="BDF1" # These schemes are not supported for time derivatives of integrals, since they are not linear multistep methods. We just use BDF1 instead, which is the same as the trapezoidal rule for linear functions.    
    numterms={"BDF1":2,"BDF2":3,"Newmark2":3,"BDF2_degr":3,"Newmark2_degr":3}
    if scheme not in numterms.keys():
        raise RuntimeError("Time scheme "+str(scheme)+" not supported for d_by_dt_of_integrals. Supported schemes are "+str(list(numterms.keys())))
    res:ExpressionOrNum=0
    # Time derivatives are just approximated as d/dt(expr) = sum_i weight_i * expr(t=t_i), where t0=t, t1=t-dt_0, t2=t1-dt_1, etc. The weights are given by the timestepper_weight function.
    for i in range(numterms[scheme]):
        res+=timestepper_weight(1,i,scheme=cast("TimeSteppingScheme",scheme))*evaluate_in_past(expr,i,apply_on_integral_dx=True,apply_on_others=apply_on_others)
    # The finite difference above is a difference of integrals, not a partial_t of a field, so the code
    # generator would find no mass-matrix contribution in it whatsoever: the __partial_t_mass_matrix probe
    # only responds to derived first-order time derivatives, and history terms differentiate to zero.
    # The marker term supplies it. d/dt I(U) = sum_j dI/dU_j * dU_j/dt, hence d(residual)/d(dU/dt) = dI/dU,
    # and that is exactly what differentiating the marked expression by every unknown gives - including
    # the mesh positions (through dx) and, if the density is a field, its dofs. The marker term is
    # substituted by zero in the residual and in the Jacobian, so it changes neither. It sits inside the
    # same division by the temporal scale as the difference, so that the mass matrix stays consistent with
    # the residual it belongs to.
    res+=_pyoomph.GiNaC_mass_matrix_marker()*expr
    return res/scale_factor("temporal") # And we divide by the temporal scaling, since the time derivative is scaled with the temporal scaling
