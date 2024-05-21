from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

from pybind11.setup_helpers import ParallelCompile


import sys
import setuptools
import glob

import os.path
import pathlib



config_file=os.environ.get("PYOOMPH_CONFIG_FILE",'pyoomph_config.env')
if os.path.isfile(config_file):
    envfile=open(config_file)
    for line in envfile:
        line=line.strip()
        if line.startswith('#') or line=="":
            continue                        
        key, value = line.split('=', 1)
        key=key.strip()
        value=value.strip()
        if key.startswith("export "):
           key=key[len("export "):]
           key=key.strip()
        os.environ[key] = value
    envfile.close()
        
march_native=os.environ.get("PYOOMPH_MARCH_NATIVE")!="false"
paranoid=os.environ.get("PYOOMPH_PARANOID")=="true"
debug_symbols=os.environ.get("PYOOMPH_DEBUG_INFOS")!="false"
with_tcc=os.environ.get("PYOOMPH_NO_TCC")=="false"

cross_compile_for_win=os.environ.get("PYOOMPH_CROSS_COMPILE_MINGW")=="true"
fast_multi_version_build=os.environ.get("PYOOMPH_FAST_MULTI_VERSION_BUILD")=="true"

print("CONFIGURATION READ FROM "+config_file)
print("==========================================")
for k,v in os.environ.items():
   if k.startswith("PYOOMPH"):
     print(k,v)
print("==========================================")
     


     
__version__ = '0.1.2'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        
        return os.environ.get("PYOOMPH_PYBIND11_INCLUDE",pybind11.get_include())



odir="./src/thirdparty/oomph-lib"
oidir=odir+"/include/"
oldir=odir+"/lib/"
tccdir="./src/thirdparty/tinycc"



 
c_source_files=([] if fast_multi_version_build else glob.glob("src/*.cpp"))+glob.glob("src/pybind/*.cpp")#+glob.glob("src/thirdparty/oomph-lib/include/*.cc")
#c_source_files=list(set(c_source_files)-set(glob.glob("src/thirdparty/oomph-lib/include/*.template.cc")))

def get_all_include_dirs():
   res=[get_pybind_include(),oidir,"./src/","./src/thirdparty/"]
   if "PYOOMPH_GINAC_INCLUDE_DIR" in os.environ:
     res.append(os.environ.get("PYOOMPH_GINAC_INCLUDE_DIR"))
   if "PYOOMPH_CLN_INCLUDE_DIR" in os.environ:
     res.append(os.environ.get("PYOOMPH_CLN_INCLUDE_DIR"))     
   return res
   
def get_all_lib_dirs():
   res=[oldir]
   if with_tcc:
     res.append(tccdir)
   if "PYOOMPH_GINAC_LIB_DIR" in os.environ:
     res.append(os.environ.get("PYOOMPH_GINAC_LIB_DIR"))
   if "PYOOMPH_GINAC_LIB_DIR" in os.environ:
     res.append(os.environ.get("PYOOMPH_GINAC_LIB_DIR"))     
   if "PYOOMPH_CROSS_COMPILE_PYTHON_LIB_DIR" in os.environ:
     res.append(os.environ.get("PYOOMPH_CROSS_COMPILE_PYTHON_LIB_DIR"))           
   if fast_multi_version_build:
     res.append("./src/lib")
   return res
   

win_cross_python_lib=[os.environ.get("PYOOMPH_CROSS_COMPILE_PYTHON_LIB_FILE",None)]
if win_cross_python_lib[0] is None:
	win_cross_python_lib=[]

ext_modules = [
    Extension(
        '_pyoomph',
				sorted(c_source_files),
        include_dirs=get_all_include_dirs(),
				library_dirs=get_all_lib_dirs(),
				libraries=(["pyoomph_main"] if fast_multi_version_build else [])+['generic',"ginac","cln"]+([] if cross_compile_for_win else ["dl"] )+(["tcc"] if with_tcc else [])+win_cross_python_lib, #,
        language='c++'
    ),
]

if os.environ.get("PYOOMPH_SKIP_EXTENSION")=="true":
    ext_modules=[]

#ext_modules[0].linker[0] = ext_modules[0].compiler_cxx[0]

# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
#    flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    flags = ['-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++'] #, '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts
    elif sys.platform =="linux":
        l_opts['unix'] += ["-s"]
                
    if debug_symbols:
       c_opts['unix'] += ["-g3"]
       l_opts['unix'] +=  ["-g3"]
    else:
       c_opts['unix'] += ["-g0"]
       l_opts['unix'] +=  ["-g0"]    

    def build_extensions(self):

      # Remove the -Wstrict-prototypes option, it's not valid for C++.  Fixed
        # in Py3.7 as bpo-5755.
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except (ValueError, AttributeError):
            pass

        if "PYOOMPH_PYTHON_HEADER_PATH" in os.environ.keys():
           self.compiler.include_dirs=[os.environ["PYOOMPH_PYTHON_HEADER_PATH"]]
        
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
#        print("COPTS",opts)
#        print("LOPTS",link_opts)
#        print("EXT",self.extensions)
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                if sys.platform!="darwin": # Clashes with GiNaC on Mac
                    opts.append('-fvisibility=hidden')
                
        if march_native:
            if sys.platform=="linux":
                opts.append("-march=native")
            opts.append("-O3") 
        else:
         opts.append("-O2")       	

#        link_opts+=["-Wl,--dynamic-list=src/exported.txt"]

#        print("COOOOOMPH CXX",self.compiler.compiler_cxx)
#        print("COOOOOMPH CC",self.compiler.compiler)
#        print("COOOOOMPH SO",self.compiler.compiler_so)
#        print("COOOOOMPH",dir(self.compiler))
        #self.compiler.linker[0] = self.compiler.compiler_cxx[0]


        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))] + [('OOMPH_HAS_MPI',None)]
            ext.define_macros = ext.define_macros + ([('PARANOID', None)] if paranoid else []) + ([] if with_tcc else [('PYOOMPH_NO_TCC',None)])
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)




excluded = []
#if is_wheel:
#    excluded.append('extlibs.future')

def exclude_package(pkg):
    for exclude in excluded:
        if pkg.startswith(exclude):
            return True
    return False

def create_package_list(base_package):
    return ([base_package] +
            [base_package + '.' + pkg
             for pkg
             in find_packages(base_package)
             if not exclude_package(pkg)])



ParallelCompile("NPY_NUM_BUILD_JOBS",default=4).install()

setup(
    name='pyoomph',
    version=__version__,
		packages=create_package_list('pyoomph')+["_pyoomph-stubs"],
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    install_requires=["meshio", "pygmsh","numpy","scipy","matplotlib","mkl","more_itertools"]+([] if with_tcc else ["tccbox"]),
    package_data={'pyoomph.jitbridge': ['*.h']+(["libtcc1.a"] if with_tcc else []),  "_pyoomph-stubs" : ["py.typed","__init__.pyi"]},
    include_package_data=True,
    
)
