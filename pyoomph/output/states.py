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
 
 
import struct
from ..typings import *
import numpy
import numpy.lib.format
#import zipfile
import zlib
import io

class DumpFile:
    def __init__(self,fname:str,save:bool,compression_level:Optional[int]=None):
        self.save=save
        self.file=open(fname,"wb" if save else "rb")
        self.fname=fname
        #self.file=zipfile.ZipFile(fname,"w" if save else "r",allowZip64=True)
        self._float_size=struct.calcsize("<d")
        self.compression_level=compression_level
        

    def close(self):
        self.file.close()

    def write_footer(self,footer:str):
        if not self.save:
            raise RuntimeError("Can only do this while saving")
        self.write_string_data(footer)

    def check_footer(self,footer:str) -> bool:
        if self.save:
            raise RuntimeError("Can only do this while loading")
        conv=footer.encode("ascii")
        offs=len(conv)+8
        self.file.seek(-offs,io.SEEK_END)
        test=self.read_string_data()
        self.file.seek(0, io.SEEK_SET)
        return test==footer

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

    def float_data(self,getter:Callable[[],float],setter:Union[Callable[[float],float],Callable[[float],None]]) -> float:
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

    def int_data(self,getter:Callable[[],int],setter:Union[Callable[[int],int],Callable[[int],None]]) -> int:
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