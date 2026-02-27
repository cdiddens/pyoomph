#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
import os
import subprocess
import sys
import shlex


import setuptools 
import distutils
import distutils.ccompiler
import distutils.log
import distutils.errors


from ..typings import *

_TypeVarCompiler=TypeVar("_TypeVarCompiler",bound=Type["BaseCCompiler"])


class BaseCCompiler(_pyoomph.SharedLibCCompiler):
    compiler_id:str

    def __init__(self):
        super(BaseCCompiler, self).__init__()

    @staticmethod
    def check_avail()->bool: 
        return True

    @staticmethod
    def call_cmd( cmd:List[str], shell:bool=False, env:Optional[Dict[str,str]]=None,quiet:bool=False)->str:
        if quiet==False:
            print(shlex.join(cmd))
        if env is None:
            pipes = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)
        else:
            pipes = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell, env=env)
        std_out, _ = pipes.communicate()
        if pipes.returncode != 0:
            errmsg = " ".join(cmd) + "\n\n" + std_out.decode('ascii', errors="ignore")
            print(errmsg)
            raise Exception("Compilation failed with error code:", pipes.returncode, "\n\n" + errmsg)
        return std_out.decode('ascii', errors="ignore")

    def get_code_filename(self)->str:
        return self.get_code_trunk() + ".c"

    def get_lib_filename(self)->str:
        return self.get_code_trunk() + self.get_shared_lib_extension()
        
    def expand_full_library_name(self, arg0: str) -> str:
        return os.path.join(os.getcwd(),arg0)

    _registered_compilers={"_internal_":_pyoomph.CCompiler}

    @classmethod
    def register_compiler(cls,*,override:bool=False):
        def decorator(subclass:_TypeVarCompiler)->_TypeVarCompiler:
            name=subclass.compiler_id
            if name in cls._registered_compilers.keys():
                    if not override:
                        raise RuntimeError("You tried to register the compiler "+name+", but there is already one defined. Please add override=True to the arguments of @BaseCCompiler.register_compiler(override=True)")
            cls._registered_compilers[name] = subclass
            return subclass
        return decorator

    @classmethod
    def available_compilers(cls) -> Dict[str, int]:
        compiler_dict:Dict[str,int]={}
        for n,compclass in cls._registered_compilers.items():
            quality=0
            if n!="_internal_":
                assert issubclass(compclass,BaseCCompiler)
                if not compclass.check_avail():
                    continue
                quality=compclass.compiler_quality
            compiler_dict[n]=quality
        return compiler_dict

    @classmethod
    def factory_compiler(cls,name:str) -> _pyoomph.CCompiler:
        if name in cls._registered_compilers.keys():
            return cls._registered_compilers[name]()
        else:
            raise RuntimeError(
                "Unknown CCompiler solver: '" + name + "'. Following are available on your system " + str(list(cls.available_compilers().keys())))


@BaseCCompiler.register_compiler()
class TCCBoxCompiler(BaseCCompiler):
    compiler_id="tccbox"
    compiler_quality = 4
    @staticmethod
    def check_avail() -> bool:    
        try:
            import tccbox
        except:
            return False
        return True
    
    def compile(self, suppress_compilation: bool, suppress_code_writing: bool, quiet: bool, extra_flags: List[str]) -> bool:
        if suppress_compilation:
            return True
        if not quiet:
            print("Compiling " + self.get_code_filename() + " with jit-include dir " + self.get_jit_include_dir())
        soname = self.get_lib_filename()        
        if not suppress_code_writing:
            fullcmd=[sys.executable, "-m", "tccbox", "-I", self.get_jit_include_dir()]
            fullcmd+=["-shared"]            
            if os.name!="nt":
                fullcmd+=["-nostdinc", "-nostdlib"]
            else:
                fullcmd+=["-rdynamic"]
            fullcmd+=["-DPYOOMPH_TCC_TO_MEMORY","-Dsize_t=unsigned long long"]+extra_flags+[self.get_code_filename(), "-o",soname]
            self.call_cmd( fullcmd,quiet=quiet)
        return True


@BaseCCompiler.register_compiler()
class SystemCCompiler(BaseCCompiler):
    compiler_id = "system"
    compiler_quality=5
    def __init__(self,compile_args:Optional[List[str]]=None):
        super(SystemCCompiler, self).__init__()
        self.comp=distutils.ccompiler.new_compiler(verbose=1)
        self.comp.add_include_dir(self.get_jit_include_dir())
        self.compile_args=compile_args
        self._optimize_full_speed:bool=False
        distutils.log.set_verbosity(2)

    def optimize_for_max_speed(self):
        self._optimize_full_speed=True

    def has_function(self, funcname:str, includes:Optional[List[str]]=None, include_dirs:Optional[List[str]]=None,
                     libraries:Optional[List[str]]=None, library_dirs:Optional[List[str]]=None) -> bool:
        import tempfile
        if includes is None:
            includes = []
        if include_dirs is None:
            include_dirs = []
        if libraries is None:
            libraries = []
        if library_dirs is None:
            library_dirs = []
        fd, fname = tempfile.mkstemp(".c", funcname, text=True)
        _, fname_exe = tempfile.mkstemp(".out", funcname, text=False)
        f = os.fdopen(fd, "w")
        try:
            for incl in includes:
                f.write("""#include "%s"\n""" % incl)
            f.write("""\
int main (int argc, char **argv) {
    %s();
    return 0;
}
""" % funcname)
        finally:
            f.close()
        try:
            objects = self.comp.compile([fname],output_dir=os.path.dirname(fname), include_dirs=include_dirs)
        except distutils.errors.CompileError:
            return False

        try:
            self.comp.link_executable(objects, fname_exe,
                                 libraries=libraries,
                                 library_dirs=library_dirs)
        except (distutils.errors.LinkError, TypeError):
            return False
        return True


    @staticmethod
    def check_avail()->bool:
        inst=SystemCCompiler()
        distutils.log.set_verbosity(0)
        res:bool=inst.has_function("abort",includes=["stdlib.h"]) 
        distutils.log.set_verbosity(2)
        return res

    def compile(self, suppress_compilation:bool, suppress_code_writing:bool,quiet:bool,extra_flags:List[str]) -> bool:
        if suppress_compilation:
            return True
        distutils.log.set_verbosity(2 if not quiet else 0)
        preargs=[]
        link_extra_postargs=[]
        if self.compile_args is None:
            if self.comp.compiler_type=="unix": #type:ignore
                if os.environ.get('PYOOMPH_DEBUG') == "1":
                    preargs = ["-O0","-g3","-fPIC"]
                else:
                    preargs=["-O3","-fPIC"]
                    if sys.platform!="darwin":
                        preargs+=["-march=native"]                   

            elif self.comp.compiler_type=="msvc": #type:ignore
                preargs = ["/O2"] #Optionally set things like /arch:AVX2 /O2 /Ob2 #Test /fp:fast
            else:
                raise RuntimeError("Compiler type "+str(self._comp.compiler_type)) #type:ignore
        else:
            preargs=self.compile_args

        if self._optimize_full_speed:
            if self.comp.compiler_type=="msvc": #type:ignore
                preargs+=["/fp:fast"]
            else:
                preargs+=["-ffast-math"]

        preargs+=extra_flags
        if self.comp.compiler_type == "msvc": #type:ignore
            link_extra_postargs = ["/DLL"]            

        src=os.path.relpath(self.get_code_filename())
        #print("Compiling "+src+" to shared library "+self.get_lib_filename()+" with compiler "+str(self.comp.compiler_type)+" i.e. "+self.comp.compiler_so[0]) #type:ignore
        obj=self.comp.compile([src],extra_preargs=preargs,extra_postargs=preargs,debug=os.environ.get('PYOOMPH_DEBUG') == "1")
        self.comp.link(self.comp.SHARED_LIBRARY, obj,self.get_lib_filename(),extra_postargs=link_extra_postargs) #type:ignore
        return True



_global_compilers:Dict[str,_pyoomph.CCompiler]={}

def get_ccompiler(comp:Optional[str]=None)->_pyoomph.CCompiler:   #If None, we set the best one
#    print("GET CCOMPILER ",comp)
    if comp is None:
        avail=BaseCCompiler.available_compilers()
        best=max(avail, key=lambda key: avail[key])
        print("Using automatically chosen compiler "+best)
        return get_ccompiler(best)

    if isinstance(comp,str): #type:ignore
        if comp in _global_compilers.keys():
            return _global_compilers[comp]
        compinst=BaseCCompiler.factory_compiler(comp)
        _global_compilers[comp]=compinst
        return  compinst

    else:
        raise RuntimeError("Should not end here")




@BaseCCompiler.register_compiler()
class CCacheCCompiler(SystemCCompiler):
    compiler_id = "ccache"
    compiler_quality=6   
    def __init__(self, compile_args = None):
        super().__init__(compile_args)            
        self.comp.compiler_so=["ccache"]+self.comp.compiler_so
        self.comp.linker_so=["ccache"]+self.comp.linker_so
        # TODO: Check whether the c expression mode is "deterministic" and warn if not?