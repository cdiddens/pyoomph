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
 
 
import sympy #type:ignore
from ..expressions import (cos,sin,log,exp,pi,var,nondim,ExpressionOrNum,Expression) 
import _pyoomph
import math



from ..typings import *
_SympyType=Any

__func_conversion_table:List[Tuple[str,_SympyType,Callable[...,Expression]]]=[
    ("cos",sympy.cos,cos),
    ("sin",sympy.sin,sin),
    ("exp",sympy.exp,exp),
    ("log",sympy.log,log),
    ]

__func_conversion_sympy2pyoomph={entry[1]:entry[2] for entry in __func_conversion_table}
__func_conversion_pyoomph2sympy={entry[0]:entry[1] for entry in __func_conversion_table}

def sympy_to_pyoomph(expr:_SympyType,use_nondim:bool=False,var_map:Dict[str,ExpressionOrNum]={})->ExpressionOrNum:
    if isinstance(expr,sympy.Mul):
        return math.prod(map(lambda x : sympy_to_pyoomph(x,use_nondim,var_map), expr.args)) #type:ignore
    elif isinstance(expr,sympy.Add):
        return sum(map(lambda x : sympy_to_pyoomph(x,use_nondim,var_map), expr.args)) #type:ignore
    elif isinstance(expr,sympy.Pow):
        if len(expr.args)!=2: #type:ignore
            raise RuntimeError("Power does not work yet with other than 2 args")
        return sympy_to_pyoomph(expr.args[0],use_nondim=use_nondim,var_map=var_map)**sympy_to_pyoomph(expr.args[1],use_nondim,var_map=var_map) #type:ignore
    if isinstance(expr,sympy.Function):
        if not expr.func in __func_conversion_sympy2pyoomph.keys():
            raise RuntimeError("Sympy function "+str(expr)+" cannot be converted yet to pyoomph")
        else:
            return __func_conversion_sympy2pyoomph[expr.func](*map(sympy_to_pyoomph, expr.args,use_nondim,var_map)) #type:ignore
    if isinstance(expr,sympy.Number):
        if isinstance(expr,sympy.Integer):
            return _pyoomph.Expression(int(expr))
        elif isinstance(expr,sympy.Rational):
            return _pyoomph.Expression(int(expr.p))/_pyoomph.Expression(int(expr.q)) #type:ignore
        elif isinstance(expr,sympy.Float):
            return _pyoomph.Expression(float(expr))
        else:
            raise RuntimeError("Sympy numeric "+str(expr)+" cannot be converted yet to pyoomph")
    elif isinstance(expr,sympy.NumberSymbol):
        if expr==sympy.pi: #type:ignore
            return pi
        if expr == sympy.E: #type:ignore
            return exp(1)
        else:
            raise RuntimeError("Sympy number symbol "+str(expr)+" cannot be converted yet to pyoomph")
    elif isinstance(expr,sympy.Symbol):
        vn=str(expr)
        #print(vn,var_map)
        if vn in var_map.keys():
            vn=var_map[vn]
        if not isinstance(vn,str):
            return vn
        else:
            if use_nondim:
                return nondim(str(expr))
            else:
                return var(str(expr))
    else:
        print(repr(expr),expr.__class__)
        raise RuntimeError("Sympy expression "+str(expr)+" cannot be converted yet to pyoomph")



def pyoomph_to_sympy(expr:ExpressionOrNum,handle_undefined=None):
    if not isinstance(expr,Expression):
        return expr # Float or int

    def gen_args():
        return [pyoomph_to_sympy(expr.op(i),handle_undefined=handle_undefined) for i in range(expr.nops())]

    typeinfo=expr.get_type_information()      
    if typeinfo["class_name"] =="function":
        fnname=typeinfo["function_name"]
        if fnname in __func_conversion_pyoomph2sympy.keys():
            return __func_conversion_pyoomph2sympy[fnname](*gen_args())
        elif fnname=="field":
            ## Field conversion
            return sympy.Symbol(str(expr.op(0)),real=True)
        elif fnname=="nondimfield":
            ## Field conversion
            return sympy.Symbol(str(expr.op(0)),real=True)
        elif handle_undefined is not None:
            return handle_undefined(expr)
        else:
            print(typeinfo)
            raise RuntimeError("Cannot convert yet "+str(expr))

    elif expr.nops()==0:
        if typeinfo["class_name"]=="numeric":
            if typeinfo["is_integer"]=="true":
                return sympy.Number(int(expr))
            elif typeinfo["is_rational"]=="true":
                return sympy.Rational(int(expr.numer()),int(expr.denom()))
            elif typeinfo["is_real"]=="true":
                return sympy.Number(float(expr))
            else:
                raise RuntimeError("Cannot convert yet "+str(expr))
        elif typeinfo["class_name"]=="constant":
            cnst=str(expr)
            if cnst=="Pi":
                return sympy.pi
            elif handle_undefined is not None:
                return handle_undefined(expr)
            else:
                print(typeinfo)
                raise RuntimeError("Cannot convert yet "+str(expr))
        elif handle_undefined is not None:
            return handle_undefined(expr)
        else:
                print(typeinfo)
                raise RuntimeError("Cannot convert yet "+str(expr))
    elif expr.nops()==1:
        print("One OP: "+str(expr))
        print(typeinfo)
        raise RuntimeError("Cannot convert yet "+str(expr))
    else:
        
        if typeinfo["class_name"]=="add":
            return sympy.Add(*gen_args())
        if typeinfo["class_name"]=="mul":
            return sympy.Mul(*gen_args())            
        if typeinfo["class_name"]=="power":
            return sympy.Pow(*gen_args())            
        print("NMore OP: "+str(expr)+"  "+str(typeinfo))
        print(typeinfo)
        raise RuntimeError("Cannot convert yet "+str(expr))
    