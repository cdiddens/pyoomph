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
 
import sys
from typing import Union,Any,Optional,TYPE_CHECKING,Type,Set,Literal,List,Dict,overload,Tuple,cast,TypeVar,SupportsFloat,TypeAlias,TypedDict

if sys.version_info>=(3,11):
    from typing import Self
else:
    # typing.Self was only added in 3.11, but pyoomph still claims 3.10, and typing_extensions is
    # not a dependency - importing Self unconditionally made `import pyoomph` an ImportError there.
    # Every use of it (codegen.py) is a string annotation, so an alias is enough at runtime.
    Self=Any
from collections import OrderedDict
from collections.abc import Sequence, Iterable, Callable, Iterator, Generator, Mapping

import typing as typing_module
import collections as collections_module
import collections.abc as collections_abc_module

import numpy
import numpy.typing

NPFloatArray=numpy.typing.NDArray[numpy.float64]
NPIntArray=numpy.typing.NDArray[numpy.int32]
#: An integer array of unspecified width, for arrays whose dtype is not pinned to one (e.g. the
#: int64 keys and int32 block lengths that the state files and the mesh data merge mix).
NPAnyIntArray=numpy.typing.NDArray[numpy.integer[Any]]
NPComplexArray=numpy.typing.NDArray[numpy.complex128]
NPAnyArray=numpy.typing.NDArray[Any]
NPUInt64Array= numpy.typing.NDArray[numpy.uint64]
NPInt32Array=numpy.typing.NDArray[numpy.uint32]
NPBoolArray=numpy.typing.NDArray[numpy.bool_]

_AnyPyoomphType=TypeVar("_AnyPyoomphType",bound=Any)
def assert_type(obj:Any,typ:_AnyPyoomphType)->_AnyPyoomphType:
    if not isinstance(obj,typ):
        raise RuntimeError("Expected type "+str(typ)+", but got "+str(type(obj)))
    else:
        return cast(type[typ],obj) # type: ignore
    
__all__ = ["Union","Any","Sequence","Mapping","Iterable","Callable","Iterator","Optional","TYPE_CHECKING","NPFloatArray","NPIntArray","NPAnyIntArray","NPComplexArray","NPUInt64Array","NPInt32Array","Type","Set","Literal","List","Dict","overload","Tuple","cast","NPAnyArray","NPBoolArray","TypeVar","Self","Generator","OrderedDict","SupportsFloat","TypeAlias","assert_type","TypedDict"]


# The names above that are just re-exports of the standard library. Modules all over pyoomph do
# "from ..typings import *" for their annotations, and since a module without __all__ exports every
# public name it happens to hold, these travelled along every wildcard chain and ended up in the user's
# namespace as well: "from pyoomph import *" defined Callable, Iterator, List, cast, ... The names are
# only meant for annotating pyoomph's own code.
# "cast" is the exception: the tutorials use it in user code (cast("MyProblem", self.get_problem()) to
# narrow a type for the IDE) and have always gotten it from "from pyoomph import *", so it stays exported.
_STDLIB_TYPING_REEXPORTS = frozenset(
    n for n in __all__
    if n != "cast"
    and any(getattr(m, n, None) is globals()[n] for m in (typing_module, collections_module, collections_abc_module))
)


def _set_public_api(namespace: "Dict[str,Any]") -> None:
    """Set __all__ of the calling module to everything public in it except the typing helpers above.

    Call as ``_set_public_api(globals())`` at the very end of a module (after all definitions and
    imports). Only "from module import *" is affected - the names stay reachable as attributes and via
    explicit imports, so pyoomph's own "from ..typings import *" keeps working everywhere. Imported
    modules (numpy, os, ...) are deliberately left in: tutorial scripts use numpy after a bare
    "from pyoomph import *".
    """
    namespace["__all__"] = sorted(
        n for n in namespace if not n.startswith("_") and n not in _STDLIB_TYPING_REEXPORTS
    )


