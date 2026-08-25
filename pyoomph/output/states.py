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
 
 
import filecmp
import os
import shutil
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


# ---------------------------------------------------------------------------------------------
# Moving a state file elsewhere
#
# A state file names the .msh file its mesh template was built from, RELATIVE to the directory the
# dump itself sits in (MeshTemplate._define_state_file), and a load resolves it against the directory
# of the file being loaded. So a dump that is merely copied somewhere else points at nothing as soon
# as the two directories differ in depth - which is exactly what "put a copy of this point next to
# its output" does. The functions below read those names out again and rewrite them, so the .msh can
# travel with the dump.
#
# They walk the first few entries of the format by hand rather than through Problem, because there is
# no problem to hand: the point is to fix a file up without loading it. That makes them the one place
# outside Problem._define_state_file that knows the layout, so they only go as far as they must - the
# header, five fixed-width entries and then the templates - and refuse anything they do not
# recognise. Keep them in step with Problem._define_state_file / _define_state_header when the entries
# before the mesh templates change.

def _state_read_int(f:IO[bytes])->int:
    b=f.read(8)
    if len(b)<8:
        raise RuntimeError("truncated state file")
    return int.from_bytes(b,byteorder="little",signed=True)


def _state_read_string(f:IO[bytes])->str:
    return f.read(_state_read_int(f)).decode("ascii")


def _scan_state_mesh_files(f:IO[bytes])->"List[Tuple[int,str]]":
    """The mesh files a state file names, as (offset of the entry, stored path).

    The offset is that of the entry's length prefix, i.e. what has to be replaced to store a
    different path. Templates without a mesh file of their own store "" and are listed as such, so
    that the returned list is one entry per mesh template, in file order.
    """
    f.seek(0)
    header=_state_read_string(f)
    if header!="pyoomph_dump":
        raise RuntimeError("not a pyoomph state file")
    version=_state_read_string(f)
    try:
        parts=tuple(int(v) for v in version.split("."))
    except ValueError:
        raise RuntimeError("state file with an unreadable version '"+version+"'")
    if parts+(0,)*(3-len(parts))>=(0,1,1):
        _sharding=_state_read_string(f)
    f.read(struct.calcsize("<d"))   # current time
    _state_read_int(f)              # output step
    _state_read_int(f)              # continue section step
    _state_read_int(f)              # numpy compression level
    ntempl=_state_read_int(f)
    if ntempl<0 or ntempl>100000:
        raise RuntimeError("state file with an implausible number of mesh templates")
    res:"List[Tuple[int,str]]"=[]
    for _i in range(ntempl):
        offset=f.tell()
        res.append((offset,_state_read_string(f)))
        has_remesher=_state_read_int(f)
        if has_remesher:
            _state_read_int(f)      # the remesher's counter
    return res


def get_state_file_mesh_files(fname:str)->"List[str]":
    """The mesh files a state file refers to, as absolute paths. Templates without one are skipped."""
    with open(fname,"rb") as f:
        entries=_scan_state_mesh_files(f)
    base=os.path.dirname(os.path.abspath(fname))
    return [os.path.normpath(os.path.join(base,rel)) for _o,rel in entries if rel!=""]


def copy_state_file(src:str,dst:str,copy_mesh_files:bool=True,extra_mesh_extensions:"Sequence[str]"=(".geo",".geo_unrolled")) -> "List[str]":
    """Copy a state file to ``dst``, taking the mesh files it refers to along.

    Each referenced .msh is copied next to ``dst`` and the path stored in the copy is rewritten to
    point at it, so the destination directory stands on its own and can be moved or shipped. Files
    of the same trunk listed in ``extra_mesh_extensions`` (the .geo Gmsh unrolled the geometry into)
    come along as documentation; nothing reads them back.

    The mesh keeps its own name unless that would overwrite a DIFFERENT mesh already sitting there -
    two states from a remeshed run refer to two files both called e.g. GmshTemplate.msh, and exports
    put them in one directory. An identical file is shared instead of numbered, which is the common
    case when several dumps of one run are copied side by side.

    Returns the mesh files written. With ``copy_mesh_files=False`` this is a plain copy, which is
    only correct while ``dst`` sits in the same directory as ``src``.
    """
    if not copy_mesh_files:
        shutil.copy2(src,dst)
        return []
    with open(src,"rb") as f:
        data=f.read()
        f.seek(0)
        entries=_scan_state_mesh_files(f)
    srcdir=os.path.dirname(os.path.abspath(src))
    dstdir=os.path.dirname(os.path.abspath(dst))
    written:"List[str]"=[]
    out=bytearray()
    read_upto=0
    copied_as:"Dict[str,str]"={}
    for offset,rel in entries:
        if rel=="":
            continue
        mshsrc=os.path.normpath(os.path.join(srcdir,rel))
        if not os.path.exists(mshsrc):
            # Nothing to relocate, and no way to check what the old path meant. Left exactly as it
            # was rather than rewritten to a file that is not there either.
            continue
        name=copied_as.get(mshsrc)
        if name is None:
            trunk,ext=os.path.splitext(os.path.basename(mshsrc))
            i=0
            while True:
                name=trunk+ext if i==0 else trunk+"_"+str(i)+ext
                target=os.path.join(dstdir,name)
                taken=name in copied_as.values() or (os.path.exists(target)
                                                     and not filecmp.cmp(mshsrc,target,shallow=False))
                if not taken:
                    break
                i+=1
            copied_as[mshsrc]=name
            os.makedirs(dstdir,exist_ok=True)
            shutil.copy2(mshsrc,os.path.join(dstdir,name))
            written.append(os.path.join(dstdir,name))
            for ext in extra_mesh_extensions:
                side=os.path.splitext(mshsrc)[0]+ext
                if os.path.exists(side):
                    sidedst=os.path.join(dstdir,os.path.splitext(name)[0]+ext)
                    shutil.copy2(side,sidedst)
                    written.append(sidedst)
        out+=data[read_upto:offset]
        enc=name.encode("ascii")
        out+=len(enc).to_bytes(8,byteorder="little",signed=True)+enc
        read_upto=offset+8+len(rel.encode("ascii"))
    out+=data[read_upto:]
    with open(dst,"wb") as f:
        f.write(bytes(out))
    shutil.copystat(src,dst)
    return written


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
