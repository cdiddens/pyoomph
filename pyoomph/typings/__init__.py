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
 
from typing import Union,Any,Sequence,Iterable,Callable,Iterator,Optional,TYPE_CHECKING,Type,Set,Literal,List,Dict,overload,Tuple,cast,TypeVar,Generator,OrderedDict,SupportsFloat

import numpy
import numpy.typing

import sys
#if sys.version_info.major>3 or sys.version_info.minor>=10:
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    TypeAlias=Any

NPFloatArray=numpy.typing.NDArray[numpy.float64]
NPIntArray=numpy.typing.NDArray[numpy.int32]
NPComplexArray=numpy.typing.NDArray[numpy.complex128]
NPAnyArray=numpy.typing.NDArray[Any]
NPUInt64Array= numpy.typing.NDArray[numpy.uint64]
NPInt32Array=numpy.typing.NDArray[numpy.uint32]

_AnyPyoomphType=TypeVar("_AnyPyoomphType",bound=Any)
def assert_type(obj:Any,typ:_AnyPyoomphType)->_AnyPyoomphType:
    if not isinstance(obj,typ):
        raise RuntimeError("Expected type "+str(typ)+", but got "+str(type(obj)))
    else:
        return cast(Type[typ],obj) # type: ignore
    
__all__ = ["Union","Any","Sequence","Iterable","Callable","Iterator","Optional","TYPE_CHECKING","NPFloatArray","NPIntArray","NPComplexArray","NPUInt64Array","NPInt32Array","Type","Set","Literal","List","Dict","overload","Tuple","cast","NPAnyArray","TypeVar","Generator","OrderedDict","SupportsFloat","TypeAlias","assert_type"]


