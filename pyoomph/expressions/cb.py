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
from abc import abstractmethod
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
        self._symbolic_derivative: Dict[int, CustomMathExpression] = {}
        self.fd_epsilon = 1e-8

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

    def __call__(self, *a: Union[_pyoomph.Expression, float, int]) -> _pyoomph.Expression:
        """
        Evaluate the expression with the given arguments.

        Args:
            *a (Union[Expression, float, int]): The arguments to evaluate the expression.

        Returns:
            Expression: The evaluated expression.
        """
        b: List[_pyoomph.Expression] = []
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
                return dp._symbolic_derivative[index].outer_derivative(x, i)

        if self._symbolic_derivative.get(index, None) is None:
            self._symbolic_derivative[index] = self.derivative(index)
            self._symbolic_derivative[index].set_as_derivative(self, index)
        xn = [x.op(i) for i in range(x.nops())]
        return self._symbolic_derivative[index](*xn)

    def real_part(self,invokation:_pyoomph.Expression, arglst:List[_pyoomph.Expression]):          
        # Just assume everything is real here, i.e. replicate myself
        return self(*arglst) 
    
    def imag_part(self,invokation:_pyoomph.Expression, arglst:List[_pyoomph.Expression]):
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


class CustomMultiReturnExpression(_pyoomph.CustomMultiReturnExpression):
    def __init__(self) -> None:
        super().__init__()
        self.use_c_code: Union[Literal["auto"], bool] = "auto"
        self.return_tuple_for_single_return:bool=False
        self.set_debug_python_vs_c_epsilon(-1.0) # No C vs Python debugging by default
        pass

    def get_id_name(self) -> str:
        return self.__class__.__name__

    # Before calling eval, we can decompose our arguments. E.g. tensors split into scalars. The returning list may not have any phyiscal dimensions
    def process_args_to_scalar_list(self, *args: "ExpressionOrNum") -> List["ExpressionOrNum"]:
        return [*args]

    # Before returning, we can assemble things back to e.g. tensors or multiple returnals
    def process_result_list_to_results(self, result_list: List["Expression"]) -> Tuple["ExpressionOrNum", ...]:
        return tuple(result_list)

    # We must know how many scalars are returned by eval, i.e. the length of the return_list buffer
    def get_num_returned_scalars(self,nargs:int) -> int:
        raise RuntimeError(
            "Please implement get_num_returned_scalars that returns the number of scalar quantities, i.e. the required length of the result_list array in the eval method")

    # The actual evaluation: Taking arg_list, processing it to result_list
    # If flag is set, we also must fill the derivative_matrix by hand!

    def eval(self, flag: int, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray) -> None:
        raise RuntimeError("This must be implemented")

    # Sometimes, we know that some derivative is e.g. a constant or even zero. In that case, we can return it here. It will be substituted in the derived expression
    # If it is e.g. 0, this simplifies the Jacobian term and requires less computation
    def use_symbolic_derivative(self,arg_list: List[Expression],i_res:int,j_arg:int)->Optional[ExpressionOrNum]:
        return None # By default, always do the numerical ones

    def _get_symbolic_derivative(self,arg_list:List[Expression],i_res:int,j_arg:int)->Tuple[bool,Expression]:
        res=self.use_symbolic_derivative(arg_list,i_res,j_arg)
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
            eargs:List[Expression]=[]
            for pa in pargs:
                if not isinstance(pa,Expression):
                    eargs.append(Expression(pa))
                else:
                    eargs.append((pa))
            funcexpr = _pyoomph.GiNaC_python_multi_cb_function(self, eargs, num_ret)
            res = [_pyoomph.GiNaC_python_multi_cb_indexed_result(funcexpr, i) for i in range(num_ret)]
            res=self.process_result_list_to_results(res)
            if isinstance(res,(list,tuple)) and len(res)==1 and not self.return_tuple_for_single_return:
                return res[0]
            else:
                return res


    # Add this function after your derivative matrix calculation at the end of the eval function (if flag is set)
    def debug_python_derivatives_with_FD(self, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray, fd_epsilion=1.0e-8, error_threshold=1e-5,stop_on_error=False,additional_float_information:Optional[Union[float,Tuple[float,...],List[float]]]=None):
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


    # Helper to generate the derivative code. It won't work out of the box, i.e. you might have to temporarily replace e.g. numpy.sqrt by sympy.sqrt etc in the eval function
    def generate_derivative_code_by_sympy(self,arg_names:List[Optional[str]],fill_zero_before:bool=True):
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
        self.eval(0,arg_symbs,res,Jdummy)
        for i,r in enumerate(res):
            if isinstance(r,(float,int)):
                res[i]=sympy.Number(r)        
        diffmat=[[r.diff(a) for a in arg_symbs] for r in res]
        listed=[]
        for r in diffmat:
	        for e in r:
		        listed.append(e)
        sub_exprs, simplified_exprs = sympy.cse(tuple(listed))
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


