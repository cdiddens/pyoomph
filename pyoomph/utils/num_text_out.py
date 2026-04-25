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
 
from .. import Expression
import _pyoomph 
import numpy
from ..typings import *


class NumericalTextOutputFile:
    def __init__(self, filename: str, open_mode: str = "w",header:Optional[List[str]]=None):
        f = open(filename, open_mode)
        if f is None:
            raise RuntimeError("Could not open file "+str(filename))
        self.file = f
        if header:
            self.header(*header)

    def add_row(self, *args: Union[float, Any]):
        if self.file is None:
            raise RuntimeError("File was closed before")
        def params_to_float(p):
            if isinstance(p,(Expression,_pyoomph.GiNaC_GlobalParam)):
                return float(p)
            else:
                return p
        if len(args)==1 and isinstance(args[0],(list,tuple)):            
            strargs = map(str, map(params_to_float,args[0]))
        else:
            strargs = map(str, map(params_to_float,[*args]))
        line = "\t".join(strargs)+"\n"
        self.file.write(line)
        self.file.flush()

    def header(self, *args: Union[float, str, Any]):
        if self.file is None:
            raise RuntimeError("File was closed before")
        line = "#"+("\t".join(map(str, [*args]))) + "\n"
        self.file.write(line)
        self.file.flush()

    def close(self):
        if self.file is None:
            raise RuntimeError("File was already closed before")
        self.file.close()
        self.file = None

    def flush(self) -> None:
        if self.file is None:
            raise RuntimeError("File was already closed before")
        self.file.flush()


class LoadedTextDataFile:
    def __init__(self, filename: str) -> None:
        try:
            f = open(filename, "r")
        except:
            raise RuntimeError("Cannot open the file '"+str(filename)+"'")
        header = f.readline().strip()
        f.close()
        if len(header) == 0 or header[0] != "#":
            raise RuntimeError("Found no header in the file "+str(filename))
        header = header.lstrip("#")
        self.descs = header.split("\t")
        self.data: NPFloatArray = numpy.loadtxt(
            filename, ndmin=2)  # type:ignore

    def get_column(self, index_or_name_start: Union[Sequence[Union[str, int]], str, int], exact_name: bool = False) -> NPFloatArray:
        if isinstance(index_or_name_start, (list, tuple)):
            rs: List[NPFloatArray] = []
            for i in index_or_name_start:
                rs.append(self.get_column(i, exact_name=exact_name))
            return numpy.vstack(rs).transpose()  # type:ignore

        if isinstance(index_or_name_start, str):
            # Find a unique column
            index = None
            for i, d in enumerate(self.descs):
                if (exact_name and d == index_or_name_start) or (not exact_name and d.startswith(index_or_name_start)):
                    if index is None:
                        index = i
                    else:
                        raise RuntimeError(
                            "At least two columns where found by the identifier '"+index_or_name_start+"'")
            if index is None:
                raise RuntimeError(
                    "Could not find a column beginning with the identifier '"+index_or_name_start+"'")
        else:
            index = index_or_name_start

        return self.data[:, index]  # type:ignore
