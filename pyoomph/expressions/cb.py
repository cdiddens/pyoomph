from __future__ import annotations
#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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
 
 
from .. import _pyoomph_core as _pyoomph
from abc import abstractmethod
import math
import numpy
from ..typings import *
from .generic import ExpressionOrNum, Expression


class CustomMathExpression(_pyoomph.CustomMathExpression):
    """
    A custom math expression class.

    This class allows to provide custom expressions programmed in python. After creating, you can call the object with Expression arguments to obtain the symbolical function call.
    This can be then used as normal Expression. When the C code is generated, we will call back the eval method to obtain the numerical result. 
    
    By default, finite difference derivatives are calculated. If you can provide a symbolic derivative, you can override the derivative method.

    Attributes:
        fd_epsilon (float): The finite difference epsilon value for numerical differentiation.
    """

    def __init__(self):
        super().__init__()
        self._symbolic_derivative: dict[int, CustomMathExpression] = {}
        self.fd_epsilon = 1e-8
        # Guard reference used in outer_derivative() to keep the parent CustomMathExpression
        # alive on the Python side (see comment there for why this is required).
        self._diff_parent_guard: Optional["CustomMathExpression"] = None

    def get_id_name(self) -> str:
        """
        Get the name of the expression class.

        Returns:
            str: The name of the expression class.
        """
        return self.__class__.__name__

    def derivative(self, index: int) -> "CustomMathExpression":
        """
        Calculate the derivative of the expression with respect to a given index.
        Override it specifically, if you can provide a symbolical derivative of the eval function with respect to the argument at index i.

        Args:
            index (int): The index of the variable with respect to which the derivative is calculated.

        Returns:
            CustomMathExpression: The derivative of the expression.
        """
        return FiniteDifferenceDerivative(self, index, epsilon=self.fd_epsilon)

    def __call__(self, *a: _pyoomph.Expression | float | int) -> _pyoomph.Expression:
        """
        Evaluate the expression with the given arguments.

        Args:
            *a (Union[Expression, float, int]): The arguments to evaluate the expression.

        Returns:
            Expression: The evaluated expression.
        """
        b: list[_pyoomph.Expression] = []
        for c in a:
            if isinstance(c, _pyoomph.Expression):
                b.append(0+c)
            else:
                b.append(_pyoomph.Expression(c))
        return _pyoomph.GiNaC_python_cb_function(self, b)
    
    def get_argument_unit(self,index:int)->_pyoomph.Expression:
        """Get the expected unit of the argument at the given index.
        Before eval is called, the arguments are divided by this units to make them dimensionless.

        Args:
            index (int): index of the argument for which we want to get the unit

        Returns:
            Expression: Unit of the argument at the given index
        """
        return super().get_argument_unit(index)
    
    def get_result_unit(self) -> Expression:
        """Get the result unit. After eval is called, the result will be multiplied by this unit.

        Returns:
            Expression: Result unit
        """
        return super().get_result_unit()
        

    @abstractmethod
    def eval(self, arg_array: NPFloatArray) -> float:
        """
        Evaluate the expression with the given array of arguments.
        This function must be implemented in the derived class to specify the functionality of the CustomMathExpression.

        Args:
            arg_array (NPFloatArray): The array of arguments to evaluate the expression.

        Returns:
            float: The evaluated expression result.
        
        Raises:
            RuntimeError: If the eval function is not implemented.
        """
        raise RuntimeError("Implement the eval function of "+str(self))
        pass

    def outer_derivative(self, x: _pyoomph.Expression, index: int) -> _pyoomph.Expression:
        """
        Calculate the outer derivative of the expression with respect to a given index.

        Args:
            x (Expression): The expression with respect to which the derivative is calculated.
            index (int): The index of the variable with respect to which the derivative is calculated.

        Returns:
            Expression: The outer derivative of the expression.
        """
        if self.get_diff_index() >= 0:
            dp = self.get_diff_parent()
            assert isinstance(dp, CustomMathExpression)
            i = self.get_diff_index()
            if (i > index):
                if dp._symbolic_derivative.get(index, None) is None:
                    dp._symbolic_derivative[index] = dp.derivative(index)
                    dp._symbolic_derivative[index].set_as_derivative(dp, index)
                    # set_as_derivative() no longer keeps "dp" alive on the C++ side (that used an
                    # nb::keep_alive edge invisible to Python's cyclic GC, which combined with the
                    # ordinary self<->parent cycle these derivative helpers form made the whole thing
                    # permanently uncollectible). Every built-in derivative helper already stores this
                    # via a plain "parent" attribute, but a custom derivative() override might not -
                    # guard unconditionally here so correctness never depends on that convention.
                    dp._symbolic_derivative[index]._diff_parent_guard=dp
                return dp._symbolic_derivative[index].outer_derivative(x, i)

        if self._symbolic_derivative.get(index, None) is None:
            self._symbolic_derivative[index] = self.derivative(index)
            self._symbolic_derivative[index].set_as_derivative(self, index)
            self._symbolic_derivative[index]._diff_parent_guard=self
        xn = [x.op(i) for i in range(x.nops())]
        return self._symbolic_derivative[index](*xn)

    def real_part(self,invokation:_pyoomph.Expression, arglst:Sequence[_pyoomph.Expression]):
        # Just assume everything is real here, i.e. replicate myself
        return self(*arglst)

    def imag_part(self,invokation:_pyoomph.Expression, arglst:Sequence[_pyoomph.Expression]):
        # Just assume everything is real here, i.e. return 0
        return Expression(0)

###

class FiniteDifferenceDerivative(CustomMathExpression):
    def __init__(self, parent: CustomMathExpression, index: int, epsilon: float = 1e-8):
        super().__init__()
        self.index = index
        self.epsilon = epsilon
        self.parent = parent

    def get_id_name(self) -> str:
        return "FD["+str(self.index)+"]"+self.parent.get_id_name()

    def derivative(self, index: int) -> CustomMathExpression:
        if index == self.index:
            return FiniteDifferenceDerivative2ndII(self.parent, index, self.epsilon)
        else:
            return FiniteDifferenceDerivative2ndIJ(self.parent, self.index, index, self.epsilon, self.epsilon)

    def eval(self, arg_array: NPFloatArray) -> float:
        index = self.index
        old = arg_array[index]
        arg_array[index] -= self.epsilon
        xm = self.parent.eval(arg_array)
        arg_array[index] = old+self.epsilon
        xp = self.parent.eval(arg_array)
        arg_array[index] = old
        return (xp-xm)/(2*self.epsilon)

# Second deriv partial_ii


class FiniteDifferenceDerivative2ndII(CustomMathExpression):
    def __init__(self, parent: CustomMathExpression, index: int, epsilon: float = 1e-8):
        super().__init__()
        self.index = index
        self.epsilon = epsilon
        self.parent = parent

    def get_id_name(self) -> str:
        return "FD["+str(self.index)+","+str(self.index)+"]"+self.parent.get_id_name()

    def derivative(self, index: int) -> CustomMathExpression:
        raise RuntimeError(
            "3rd order finite differences of CustomMathExpression would be required, but it is not implemented")

    def eval(self, arg_array: NPFloatArray) -> float:
        index = self.index
        old = arg_array[index]
        x0 = self.parent.eval(arg_array)
        arg_array[index] -= self.epsilon
        xm = self.parent.eval(arg_array)
        arg_array[index] = old+self.epsilon
        xp = self.parent.eval(arg_array)
        arg_array[index] = old
        return (xp+xm-2*x0)/(self.epsilon*self.epsilon)

# Second deriv partial_ij with i!=j


class FiniteDifferenceDerivative2ndIJ(CustomMathExpression):
    def __init__(self, parent: CustomMathExpression, index1: int, index2: int, epsilon1: float, epsilon2: float = 1e-8):
        super().__init__()
        self.index1 = index1
        self.index2 = index2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.parent = parent

    def derivative(self, index: int):
        raise RuntimeError(
            "3rd order finite differences of CustomMathExpression would be required, but it is not implemented")

    def get_id_name(self):
        return "FD["+str(self.index1)+","+str(self.index2)+"]"+self.parent.get_id_name()

    def eval(self, arg_array: NPFloatArray) -> float:
        old1 = arg_array[self.index1]
        old2 = arg_array[self.index2]
        arg_array[self.index1] -= self.epsilon1
        arg_array[self.index2] -= self.epsilon2
        umm = self.parent.eval(arg_array)
        arg_array[self.index2] = old2+self.epsilon2
        ump = self.parent.eval(arg_array)
        arg_array[self.index1] = old1+self.epsilon1
        upp = self.parent.eval(arg_array)
        arg_array[self.index2] = old2-self.epsilon2
        upm = self.parent.eval(arg_array)
        arg_array[self.index1] = old1
        arg_array[self.index2] = old2
        return (upp-ump-upm+umm)/(4*self.epsilon1*self.epsilon2)


# =================================================================================================
# Exact derivatives of an eval() that was only ever written to compute values
# =================================================================================================

class HyperDual:
    """A number carrying its first and second derivatives with respect to the callback arguments.

    Forward-mode automatic differentiation, second order. Arithmetic on these propagates
    ``(value, gradient, hessian)`` by the chain rule, so pushing HyperDual arguments through an
    ordinary ``eval()`` implementation yields its EXACT Jacobian and second derivatives without any
    of them being written out by hand, and without the round-off of differencing.

    ``numpy.log``/``numpy.exp``/``numpy.sqrt`` on a Python object dispatch to the object's own
    ``log()``/``exp()``/``sqrt()``, which is why an eval() written with ``numpy.log(x)`` needs no
    change to be differentiated this way. What such an eval() must NOT do is cast its arguments to
    ``float`` or write into a float array - see fill_python_derivatives_by_AD.

    Cost is O(nargs^2) per elementary operation, i.e. one pass instead of the nargs*nargs callback
    evaluations a difference-of-differences needs, and exact rather than accurate to sqrt(epsilon).
    """

    __slots__ = ("value", "grad", "hess")

    def __init__(self, value: float, grad: NPFloatArray, hess: NPFloatArray | None):
        self.value = value
        self.grad = grad
        #: None means first order only. Second derivatives cost an (nargs x nargs) array and two
        #: outer products per elementary operation, so carrying them when only the Jacobian is
        #: wanted made an analytic Jacobian an order of magnitude slower than the finite differences
        #: it replaced - measured, not assumed.
        self.hess = hess

    @staticmethod
    def independent_variables(values: Sequence[float], second_order: bool = True) -> list["HyperDual"]:
        """The seed: argument i with dx_i/dx_j = delta_ij and zero second derivatives."""
        n = len(values)
        res: list[HyperDual] = []
        for i in range(n):
            g = numpy.zeros(n)
            g[i] = 1.0
            res.append(HyperDual(float(values[i]), g, numpy.zeros((n, n)) if second_order else None))
        return res

    @staticmethod
    def _split(o: Any) -> tuple[float, Any, Any]:
        if isinstance(o, HyperDual):
            return o.value, o.grad, o.hess
        return float(o), None, None

    def _combine(self, o: Any, dvalue: float, d_self: float, d_other: float,
                 d2_self: float, d2_other: float, d2_cross: float) -> "HyperDual":
        """One chain-rule step for a binary operation, given its partial derivatives."""
        ov, og, oh = self._split(o)
        if self.hess is None:
            grad = d_self * self.grad
            if og is not None:
                grad = grad + d_other * og
            return HyperDual(dvalue, grad, None)
        grad = d_self * self.grad
        hess = d_self * self.hess + d2_self * numpy.outer(self.grad, self.grad)
        if og is not None:
            grad = grad + d_other * og
            hess = (hess + d_other * oh + d2_other * numpy.outer(og, og)
                    + d2_cross * (numpy.outer(self.grad, og) + numpy.outer(og, self.grad)))
        return HyperDual(dvalue, grad, hess)

    def __add__(self, o: Any) -> "HyperDual":
        ov, _, _ = self._split(o)
        return self._combine(o, self.value + ov, 1.0, 1.0, 0.0, 0.0, 0.0)
    __radd__ = __add__

    def __neg__(self) -> "HyperDual":
        return HyperDual(-self.value, -self.grad, None if self.hess is None else -self.hess)

    def __sub__(self, o: Any) -> "HyperDual":
        ov, _, _ = self._split(o)
        return self._combine(o, self.value - ov, 1.0, -1.0, 0.0, 0.0, 0.0)

    def __rsub__(self, o: Any) -> "HyperDual":
        return (-self) + o

    def __mul__(self, o: Any) -> "HyperDual":
        ov, _, _ = self._split(o)
        return self._combine(o, self.value * ov, ov, self.value, 0.0, 0.0, 1.0)
    __rmul__ = __mul__

    def __truediv__(self, o: Any) -> "HyperDual":
        ov, og, _ = self._split(o)
        if og is None:
            return self._combine(o, self.value / ov, 1.0 / ov, 0.0, 0.0, 0.0, 0.0)
        return self._combine(o, self.value / ov, 1.0 / ov, -self.value / (ov * ov),
                             0.0, 2.0 * self.value / (ov ** 3), -1.0 / (ov * ov))

    def __rtruediv__(self, o: Any) -> "HyperDual":
        ov = float(o)
        return self._unary(ov / self.value, -ov / self.value ** 2, 2.0 * ov / self.value ** 3)

    def _unary(self, value: float, d1: float, d2: float) -> "HyperDual":
        if self.hess is None:
            return HyperDual(value, d1 * self.grad, None)
        return HyperDual(value, d1 * self.grad,
                         d1 * self.hess + d2 * numpy.outer(self.grad, self.grad))

    def __pow__(self, o: Any) -> "HyperDual":
        if isinstance(o, HyperDual):
            # a**b == exp(b ln a); no shipped callback needs it, and getting it silently wrong
            # would be worse than saying so.
            return (self.log() * o).exp()
        p = float(o)
        return self._unary(self.value ** p, p * self.value ** (p - 1.0),
                           p * (p - 1.0) * self.value ** (p - 2.0))

    def __rpow__(self, o: Any) -> "HyperDual":
        lb = math.log(float(o))
        v = float(o) ** self.value
        return self._unary(v, v * lb, v * lb * lb)

    def log(self) -> "HyperDual":
        return self._unary(math.log(self.value), 1.0 / self.value, -1.0 / (self.value * self.value))

    def exp(self) -> "HyperDual":
        v = math.exp(self.value)
        return self._unary(v, v, v)

    def sqrt(self) -> "HyperDual":
        v = math.sqrt(self.value)
        return self._unary(v, 0.5 / v, -0.25 / (v * self.value))

    def cosh(self) -> "HyperDual":
        return self._unary(math.cosh(self.value), math.sinh(self.value), math.cosh(self.value))

    def sinh(self) -> "HyperDual":
        return self._unary(math.sinh(self.value), math.cosh(self.value), math.sinh(self.value))

    def tanh(self) -> "HyperDual":
        v = math.tanh(self.value)
        return self._unary(v, 1.0 - v * v, -2.0 * v * (1.0 - v * v))

    # Comparisons act on the value, so an eval() that branches on its arguments still branches the
    # same way. The derivative of the branch taken is what AD gives, which is the right answer
    # everywhere except exactly on the switching surface - the same caveat as any piecewise model.
    def __lt__(self, o: Any) -> bool: return self.value < self._split(o)[0]
    def __le__(self, o: Any) -> bool: return self.value <= self._split(o)[0]
    def __gt__(self, o: Any) -> bool: return self.value > self._split(o)[0]
    def __ge__(self, o: Any) -> bool: return self.value >= self._split(o)[0]
    # Deliberately NO __float__. With one defined, math.sqrt(x) and friends coerce silently and
    # return a plain number, so an eval() written against the `math` module rather than numpy would
    # come back with the right value and NO derivatives at all - a silently wrong Jacobian, which is
    # exactly the failure this class exists to avoid. Without it those calls raise a TypeError
    # naming the operation, and the class is simply reported as not differentiable this way.

    def __abs__(self) -> "HyperDual":
        # Smooth away from zero, which is where a callback that branches on abs() is using it.
        return self if self.value >= 0.0 else -self
    def __repr__(self) -> str: return "HyperDual(" + repr(self.value) + ")"


class CustomMultiReturnExpression(_pyoomph.CustomMultiReturnExpression):
    def __init__(self) -> None:
        super().__init__()
        self.use_c_code: Literal["auto"] | bool = "auto"
        self.return_tuple_for_single_return:bool=False
        self.set_debug_python_vs_c_epsilon(-1.0) # No C vs Python debugging by default
        # Step of the finite-difference fallback for the SECOND derivatives, in Python and in the
        # generated C alike. Deliberately larger than a Jacobian FD step: this differences an already
        # differenced quantity, so the usual 1e-8 would leave nothing but round-off.
        self.second_derivative_fd_epsilon:float=1e-6
        #: How eval_second_derivatives() gets its numbers when it is not overridden.
        #: "fd" finite-differences the Jacobian; "ad" runs eval() itself on HyperDual numbers,
        #: which is exact but requires eval() to be written in plain arithmetic and numpy calls
        #: (no math.* and no float() on the arguments - see fill_python_derivatives_by_AD).
        self.second_derivative_mode:Literal["fd","ad"]="fd"
        pass

    def get_id_name(self) -> str:
        return self.__class__.__name__

    # Before calling eval, we can decompose our arguments. E.g. tensors split into scalars. The returning list may not have any phyiscal dimensions
    def process_args_to_scalar_list(self, *args: "ExpressionOrNum") -> list["ExpressionOrNum"]:
        return [*args]

    # Before returning, we can assemble things back to e.g. tensors or multiple returnals
    def process_result_list_to_results(self, result_list: list["Expression"]) -> tuple["ExpressionOrNum", ...]:
        return tuple(result_list)

    # We must know how many scalars are returned by eval, i.e. the length of the return_list buffer
    def get_num_returned_scalars(self,nargs:int) -> int:
        raise RuntimeError(
            "Please implement get_num_returned_scalars that returns the number of scalar quantities, i.e. the required length of the result_list array in the eval method")

    # The actual evaluation: Taking arg_list, processing it to result_list
    # If flag is set, we also must fill the derivative_matrix by hand!

    def eval(self, flag: int, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray) -> None:
        raise RuntimeError("This must be implemented")

    # The same, plus the second derivatives. Only called while an analytic Hessian is assembled, i.e.
    # after setup_for_stability_analysis(analytic_hessian=True), and never during a residual or
    # Jacobian assembly - so implementing it costs nothing anywhere else.
    # second_derivative_tensor[i, j, k] is d^2 result[i] / d arg[j] d arg[k]. It MUST come out
    # symmetric in j and k: the generated code exploits that and only ever reads the j<=k half.
    def eval_second_derivatives(self, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray, second_derivative_tensor: NPFloatArray) -> None:
        if self.second_derivative_mode == "ad":
            self.fill_python_derivatives_by_AD(arg_list, result_list, derivative_matrix, second_derivative_tensor)
            return
        self.eval(1, arg_list, result_list, derivative_matrix)
        self.fill_python_second_derivatives_by_FD(arg_list, result_list, derivative_matrix, second_derivative_tensor)

    # Sometimes, we know that some derivative is e.g. a constant or even zero. In that case, we can return it here. It will be substituted in the derived expression
    # If it is e.g. 0, this simplifies the Jacobian term and requires less computation
    def use_symbolic_derivative(self,arg_list: Sequence[Expression],i_res:int,j_arg:int)->ExpressionOrNum | None:
        return None # By default, always do the numerical ones

    def _get_symbolic_derivative(self,arg_list:Sequence[Expression],i_res:int,j_arg:int)->tuple[bool,Expression]:
        res=self.use_symbolic_derivative(arg_list,i_res,j_arg)
        zero=Expression(0)
        if res is None:
            return (False,zero)
        else:
            if not isinstance(res,Expression):
                res=Expression(res)
            return (True,res)

    # The same for a second derivative. Only ever asked with j_arg <= k_arg, since the tensor is
    # symmetric. Note that a first derivative given symbolically here needs nothing further: it is an
    # ordinary expression, which the code generator differentiates again by itself.
    def use_symbolic_second_derivative(self,arg_list: Sequence[Expression],i_res:int,j_arg:int,k_arg:int)->ExpressionOrNum | None:
        return None

    def _get_symbolic_second_derivative(self,arg_list:Sequence[Expression],i_res:int,j_arg:int,k_arg:int)->tuple[bool,Expression]:
        res=self.use_symbolic_second_derivative(arg_list,i_res,j_arg,k_arg)
        zero=Expression(0)
        if res is None:
            return (False,zero)
        else:
            if not isinstance(res,Expression):
                res=Expression(res)
            return (True,res)

    # No C code there if not overwritten
    # If there should be C code, override this function
    # Numerical arguments can be accessed via arg_list[...]
    # Results must be returned via result_list[...]
    # Derivatives must be returned (only "if (flag)"") via derivative_matrix[i*nargs+j]
    # where j is the argument index and i is the result index
    # If you are lazy, you can use finite difference by adding
    #   FILL_MULTI_RET_JACOBIAN_BY_FD(1.0e-8)
    # At the end of the C code

    def generate_c_code(self) -> str:
        return ""

    def _get_c_code(self) -> str:
        if self.use_c_code == "auto":
            return self.generate_c_code()
        elif self.use_c_code:
            res = self.generate_c_code()
            if res == "":
                raise RuntimeError(
                    "You set use_c_code=True, but you haven't specified the C code")
            return res
        else:
            return ""

    # C code filling the second derivatives, written into a function of its own that the code
    # generator emits alongside the one generate_c_code() lands in. Available there, besides the
    # usual arg_list/result_list/derivative_matrix/nargs/nret:
    #   second_derivative_tensor[(i*nargs + j)*nargs + k] = d^2 result[i] / d arg[j] d arg[k]
    # which has to be filled symmetrically in j and k.
    #
    # This body has to fill result_list and derivative_matrix as well, and they arrive UNSET: it is
    # a separate function, not a continuation of generate_c_code(), so nothing has run that body
    # yet. Unless the second derivatives are cheaper to get alongside the value, open with
    #   CURRENT_MULTIRET_FUNCTION(PYOOMPH_MULTIRET_FLAG_DERIVATIVES, arg_list, result_list, derivative_matrix, nargs, nret);
    # which calls this callback's own generate_c_code() function - the macro is defined around both
    # functions for exactly this. Forgetting it does not fail to compile and does not warn: the
    # Hessian is then built on an uninitialised value and Jacobian, and only a Hessian-vs-finite-
    # difference check (Problem.debug_analytic_hessian_by_fd) tells you. The default
    # FILL_MULTI_RET_HESSIAN_BY_FD body below starts with that same call.
    #
    # Only consulted for a callback that also has generate_c_code(): with no C implementation at
    # all there is no generated function to put this in, and everything - values, Jacobian and
    # second derivatives alike - goes back into Python through eval()/eval_second_derivatives().
    #
    # If this returns "", the generated function instead ends in
    #   FILL_MULTI_RET_HESSIAN_BY_FD(second_derivative_fd_epsilon)
    # which finite-differences the derivative matrix. That keeps every existing generate_c_code()
    # implementation working under a Hessian without changes, at nargs extra evaluations per call -
    # or nargs*nargs if the Jacobian is itself filled by FILL_MULTI_RET_JACOBIAN_BY_FD.
    def generate_c_code_second_derivatives(self) -> str:
        return ""

    def _get_c_code_second_derivatives(self) -> str:
        if self.use_c_code is False:
            return ""
        return self.generate_c_code_second_derivatives()

    def get_second_derivative_fd_epsilon(self) -> float:
        return self.second_derivative_fd_epsilon

    def __call__(self, *args: "ExpressionOrNum", **kwds: Any) -> Any:
        pargs = self.process_args_to_scalar_list(*args)
        all_numeric = True
        for pa in pargs:
            if isinstance(pa, _pyoomph.Expression):
                try:
                    _f = float(pa)
                except:
                    all_numeric = False
                    break
        num_ret = self.get_num_returned_scalars(len(pargs))
        if all_numeric:
            fargs = numpy.array([float(p) for p in pargs], dtype=numpy.float64)
            dummyderiv = numpy.zeros((0))
            ret = numpy.zeros((num_ret,))
            self.eval(0, fargs, ret, dummyderiv)
            res=self.process_result_list_to_results([r for r in ret])
            if isinstance(res,(list,tuple)) and len(res)==1 and not self.return_tuple_for_single_return:
                return res[0]
            else:
                return res
        
        else:
            eargs:list[Expression]=[]
            for pa in pargs:
                if not isinstance(pa,Expression):
                    eargs.append(Expression(pa))
                else:
                    eargs.append((pa))
            funcexpr = _pyoomph.GiNaC_python_multi_cb_function(self, eargs, num_ret)
            indexed = [_pyoomph.GiNaC_python_multi_cb_indexed_result(funcexpr, i) for i in range(num_ret)]
            res=self.process_result_list_to_results(indexed)
            if isinstance(res,(list,tuple)) and len(res)==1 and not self.return_tuple_for_single_return:
                return res[0]
            else:
                return res


    # Add this function after your derivative matrix calculation at the end of the eval function (if flag is set)
    def debug_python_derivatives_with_FD(self, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray, fd_epsilion=1.0e-8, error_threshold=1e-5,stop_on_error=False,additional_float_information:float | tuple[float, ...] | list[float] | None=None):
        derivative_matrix_p = derivative_matrix.copy()
        self.fill_python_derivatives_by_FD(arg_list,result_list,derivative_matrix_p,fd_epsilion)
        for iret in range(len(result_list)):
            for iarg in range(len(arg_list)):
                diff = derivative_matrix_p[iret,iarg]-derivative_matrix[iret, iarg]
                if abs(diff) > error_threshold:
                    msg="DIFFERENCE IN "+str(self)+": Result "+str(iret)+" derived by arg "+str(iarg) +" should be "+str(derivative_matrix_p[iret, iarg])+", but is "+str(derivative_matrix[iret, iarg])
                    msg+=" Args are: "+str(arg_list)+" Result is: "+str(result_list)
                    if additional_float_information:
                        msg+="Additional float information is "+str(additional_float_information)
                    if stop_on_error:
                        raise RuntimeError(msg)
                    else:
                        print(msg)

    # If you are too lazy for analytic derivatives, add a
    #   if flag:
    #       fill_python_derivatives_by_FD(arg_list,result_list,derivative_matrix)
    # at the end of your eval function. It will fill it with FD
    def fill_python_derivatives_by_FD(self, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray, fd_epsilion=1.0e-8):
        arg_list_p = arg_list.copy()
        result_list_p = result_list.copy()
        derivative_matrix_dummy = derivative_matrix.copy()
        for iarg in range(len(arg_list)):
            arg_list_p[:] = arg_list[:]
            arg_list_p[iarg] += fd_epsilion
            self.eval(0, arg_list_p, result_list_p, derivative_matrix_dummy)
            derivative_matrix[:, iarg] = (
                result_list_p[:]-result_list[:])/fd_epsilion

    # The second-derivative counterpart, and the default implementation of eval_second_derivatives.
    # It differences the DERIVATIVE MATRIX rather than the results, so a callback with an analytic
    # Jacobian gets second derivatives of the same quality; where the Jacobian is itself filled by
    # fill_python_derivatives_by_FD this becomes a difference of a difference - correct, but only to
    # about sqrt(epsilon), and at nargs*nargs evaluations of the callback.
    # The closing symmetrisation is not cosmetic: a forward difference of a Jacobian is symmetric
    # only to O(epsilon), and the generated code reads just one of the two halves of each pair.
    def fill_python_second_derivatives_by_FD(self, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray, second_derivative_tensor: NPFloatArray, fd_epsilion: float | None = None):
        if fd_epsilion is None:
            fd_epsilion = self.second_derivative_fd_epsilon
        nargs = len(arg_list)
        arg_list_p = numpy.array(arg_list, dtype=numpy.float64)
        result_list_p = numpy.array(result_list, dtype=numpy.float64)
        derivative_matrix_p = numpy.array(derivative_matrix, dtype=numpy.float64)
        for karg in range(nargs):
            arg_list_p[:] = arg_list[:]
            arg_list_p[karg] += fd_epsilion
            derivative_matrix_p.fill(0.0)
            self.eval(1, arg_list_p, result_list_p, derivative_matrix_p)
            second_derivative_tensor[:, :, karg] = (
                derivative_matrix_p[:, :]-derivative_matrix[:, :])/fd_epsilion
        for jarg in range(nargs):
            for karg in range(jarg+1, nargs):
                sym = 0.5*(second_derivative_tensor[:, jarg, karg]+second_derivative_tensor[:, karg, jarg])
                second_derivative_tensor[:, jarg, karg] = sym
                second_derivative_tensor[:, karg, jarg] = sym

    # Exact first and second derivatives, obtained by running this class's own eval() on HyperDual
    # numbers instead of floats (forward-mode AD, second order - see HyperDual above). Nothing has to
    # be differentiated by hand, and the result is exact rather than accurate to sqrt(epsilon).
    #
    # An eval() qualifies if it only does arithmetic and numpy.log/exp/sqrt/power on its arguments.
    # It does NOT qualify if it casts them with float(), or writes results into an array that was
    # allocated as float64 - hence the object-dtype scratch arrays below. A class whose eval() does
    # either can usually be made to qualify by moving the cast behind a flag; see the AIOMFAC and
    # UNIFAC callbacks, which do exactly that.
    #
    # Use it by pointing eval_second_derivatives at it:
    #     def eval_second_derivatives(self, args, res, deriv, second):
    #         self.fill_python_derivatives_by_AD(args, res, deriv, second)
    def fill_python_derivatives_by_AD(self, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray, second_derivative_tensor: NPFloatArray | None = None) -> None:
        nargs = len(arg_list)
        nres = len(result_list)
        duals = HyperDual.independent_variables([float(a) for a in arg_list],
                                                second_order=second_derivative_tensor is not None)
        arg_obj = numpy.empty(nargs, dtype=object)
        for i, d in enumerate(duals):
            arg_obj[i] = d
        res_obj = numpy.zeros(nres, dtype=object)
        deriv_dummy = numpy.zeros((nres, nargs), dtype=object)
        self.eval(0, arg_obj, res_obj, deriv_dummy)
        for i in range(nres):
            r = res_obj[i]
            if not isinstance(r, HyperDual):
                # A result that came out a plain number does not depend on the arguments at all
                # (a constant branch, say), so all of its derivatives are zero.
                result_list[i] = float(r)
                derivative_matrix[i, :] = 0.0
                if second_derivative_tensor is not None:
                    second_derivative_tensor[i, :, :] = 0.0
                continue
            result_list[i] = r.value
            derivative_matrix[i, :] = r.grad
            if second_derivative_tensor is not None:
                second_derivative_tensor[i, :, :] = r.hess

    # Add this at the end of an eval_second_derivatives implementation to check it against FD.
    def debug_python_second_derivatives_with_FD(self, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray, second_derivative_tensor: NPFloatArray, fd_epsilion: float | None = None, error_threshold: float = 1e-5, stop_on_error: bool = False):
        reference = numpy.zeros_like(numpy.array(second_derivative_tensor, dtype=numpy.float64))
        self.fill_python_second_derivatives_by_FD(arg_list, result_list, derivative_matrix, reference, fd_epsilion)
        for iret in range(reference.shape[0]):
            for jarg in range(len(arg_list)):
                for karg in range(jarg, len(arg_list)):
                    diff = reference[iret, jarg, karg]-second_derivative_tensor[iret, jarg, karg]
                    if abs(diff) > error_threshold:
                        msg = ("DIFFERENCE IN "+str(self)+": d2 Result "+str(iret)+" derived by args "
                               + str(jarg)+" and "+str(karg)+" should be "+str(reference[iret, jarg, karg])
                               + ", but is "+str(second_derivative_tensor[iret, jarg, karg]))
                        msg += " Args are: "+str(arg_list)+" Result is: "+str(result_list)
                        if stop_on_error:
                            raise RuntimeError(msg)
                        else:
                            print(msg)


    # Helper to generate the derivative code. It won't work out of the box, i.e. you might have to temporarily replace e.g. numpy.sqrt by sympy.sqrt etc in the eval function
    def generate_derivative_code_by_sympy(self,arg_names:list[str | None],fill_zero_before:bool=True):
        arg_names=arg_names.copy()
        for i,a in enumerate(arg_names):
            if a is None or a=="":
                arg_names[i]="arg_list["+str(i)+"]"
        import sympy
        arg_symbs=sympy.symbols(arg_names)
        nargs=len(arg_symbs)
        nres=self.get_num_returned_scalars(nargs)
        res=numpy.array([sympy.zeros(1)[0] for _i in range(nres)])
        Jdummy=numpy.zeros((nres,nargs),dtype=object)
        self.eval(0,arg_symbs,res,Jdummy) #type:ignore
        for i,r in enumerate(res):
            if isinstance(r,(float,int)):
                res[i]=sympy.Number(r)        
        diffmat=[[r.diff(a) for a in arg_symbs] for r in res]
        listed=[]
        for r in diffmat:
            for e in r:
                listed.append(e)
        sub_exprs, simplified_exprs = cast("Tuple[List[Any], List[Any]]", sympy.cse(tuple(listed)))
        for s in sub_exprs:
            print(s[0],"=",s[1])
        print("#Jacobian entries:")
        if fill_zero_before:
            print("derivative_matrix.fill(0.0)")
        diffmat=[]
        for i in range(len(res)):
            for j in range(len(arg_symbs)):
                if fill_zero_before:
                    if simplified_exprs[i*len(arg_symbs)+j].is_number:
                        if float(simplified_exprs[i*len(arg_symbs)+j])==0.0:
                            continue
                print("derivative_matrix["+str(i)+","+str(j)+"] = "+str(simplified_exprs[i*len(arg_symbs)+j]))
        exit()


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
