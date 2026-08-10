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
 
 
import struct
from typing import IO
from ..typings import *
import numpy
import numpy.lib.format
#import zipfile
import zlib
import io

class DumpFile:
    # fname may also be an already-open binary stream (anything with read/write/seek/tell, e.g. an
    # io.BytesIO). Nothing in here ever touched the file system beyond the open() below, so an
    # in-memory state costs one branch: that is what Problem._snapshot_state uses to keep a state
    # around for rollback without writing to the output directory.
    def __init__(self,fname:str | IO[bytes],save:bool,compression_level:int | None=None):
        self.save=save
        # A stream is NOT closed by close() - we do not own it, and the caller still has to read the
        # bytes back out of it.
        self._owns_file=isinstance(fname,str)
        self.file=open(fname,"wb" if save else "rb") if isinstance(fname,str) else fname
        self.fname=fname if isinstance(fname,str) else "<stream>"
        # Version of the format being written or read. Set by Problem._define_state_header, so the
        # sections below can tell an old file from a current one.
        self.version:str=""
        # How the mesh data in this file is spread over files: "global" means this one file holds the
        # whole problem, which is all that is written today. A per-rank sharded variant would say so
        # here, so that a reader can tell the two apart instead of desynchronizing on the first array
        # (see dev_docs/distributed_state_files.md).
        self.sharding:str="global"
        #self.file=zipfile.ZipFile(fname,"w" if save else "r",allowZip64=True)
        self._float_size=struct.calcsize("<d")
        self.compression_level=compression_level
        

    def close(self):
        if self._owns_file:
            self.file.close()

    def version_at_least(self,*parts:int) -> bool:
        """Whether the version of this file is at least the given one, compared componentwise.

        Not a string comparison: "0.10.0" < "0.2.0" lexicographically, which would silently pick the
        wrong branch the first time a component reaches double digits."""
        try:
            mine=tuple(int(p) for p in self.version.split("."))
        except ValueError:
            return False # something that does not look like a version at all is treated as ancient
        want=tuple(parts)
        length=max(len(mine),len(want))
        return mine+(0,)*(length-len(mine)) >= want+(0,)*(length-len(want))

    def write_footer(self,footer:str):
        if not self.save:
            raise RuntimeError("Can only do this while saving")
        self.write_string_data(footer)

    def check_footer(self,footer:str) -> bool:
        if self.save:
            raise RuntimeError("Can only do this while loading")
        conv=footer.encode("ascii")
        offs=len(conv)+8
        # Everything is checked before anything is read: on a file that is not a state file at all, the
        # length prefix is whatever those eight bytes happen to mean, and reading that many bytes used
        # to end in a MemoryError instead of "this is not a state file".
        self.file.seek(0, io.SEEK_END)
        if self.file.tell()<offs:
            self.file.seek(0, io.SEEK_SET)
            return False
        self.file.seek(-offs,io.SEEK_END)
        length=self.read_int_data()
        if length!=len(conv):
            self.file.seek(0, io.SEEK_SET)
            return False
        test=self.file.read(length)
        self.file.seek(0, io.SEEK_SET)
        return test==conv

    def assert_equal(self,val:Any, expected:Any)->Any:
        if val!=expected:
            raise RuntimeError("Expected "+str(expected)+" in state file, but read "+str(val))
        #assert val == expected
        return val

    def assert_leq(self,val:Any, expected:Any)->Any:
        assert val <= expected
        return val

    def read_int_data(self,size:int=8,byteorder:Literal['little','big']='little', signed:bool=True) -> int:
        b=self.file.read(size)
        return int.from_bytes(b,byteorder=byteorder,signed=signed)

    def write_int_data(self,d:int,size:int=8,byteorder:Literal['little','big']='little', signed:bool=True) -> None:
        self.file.write(d.to_bytes(size,byteorder=byteorder,signed=signed))

    def read_string_data(self,encoding:str="ascii") -> str:
        l=self.read_int_data()
        b=self.file.read(l)
        return b.decode(encoding)

    def write_string_data(self,s:str,encoding:str="ascii") -> None:
        self.write_int_data(len(s))
        self.file.write(s.encode(encoding))

    def read_float_data(self)->float:
        b=self.file.read(self._float_size)
        return float(struct.unpack("<d", b)[0])

    def write_float_data(self,f:float) -> None:
        self.file.write(struct.pack("<d",f))


    def write_numpy_data(self,v:NPAnyArray) -> None:
        if self.compression_level is None:
            numpy.lib.format.write_array(self.file,numpy.array([v])) #type:ignore
        else:
            np_bytes = io.BytesIO()
            numpy.save(np_bytes, v, allow_pickle=True) #type:ignore
            compress=zlib.compress(np_bytes.getvalue(),level=self.compression_level)            
            self.write_int_data(len(compress))
            self.file.write(compress)
        

    def read_numpy_data(self)->NPAnyArray:
        if self.compression_level is None:
            return numpy.lib.format.read_array(self.file)[0] #type:ignore
        else:
            l=self.read_int_data()
            compr=self.file.read(l)
            by=zlib.decompress(compr)
            np_bytes = io.BytesIO(by)
            return numpy.load(np_bytes, allow_pickle=True)
        #return None #TODO

    def string_data(self,getter:Callable[[],str],setter:Callable[[str],str]) -> str:
        if self.save:
            s=getter()
            self.write_string_data(s)
            return s
        else:
            s=self.read_string_data()
            s=setter(s)
            return s

    def float_data(self,getter:Callable[[],float],setter:Callable[[float], float] | Callable[[float], None]) -> float:
        if self.save:
            s=getter()
            self.write_float_data(s)
            return s
        else:
            s=self.read_float_data()
            sres=setter(s)
            if sres is not None:
                return sres
            else:
                return s

    def int_data(self,getter:Callable[[],int],setter:Callable[[int], int] | Callable[[int], None]) -> int:
        if self.save:
            s=getter()
            self.write_int_data(s)
            return s
        else:
            s=self.read_int_data()
            sres=setter(s)
            if sres is not None:
                return sres
            else:
                return s

    def numpy_data(self,getter:Callable[[],NPAnyArray],setter:Callable[[NPAnyArray],NPAnyArray])->NPAnyArray:
        if self.save:
            s=getter()
            self.write_numpy_data(s)
            return s
        else:
            s=self.read_numpy_data()
            s=setter(s)
            return s


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
