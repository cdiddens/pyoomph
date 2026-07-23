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
 
from glob import glob
import sys
import os
import json
import argparse
import tempfile


class _BaseMainEntry:
   def show(self):
      pass



class _CheckEntry(_BaseMainEntry):
   pass


entries = {"check": _CheckEntry()}

parser = argparse.ArgumentParser()
parser.add_argument(
    "command", help="Use one of the following commands: check, cbrange, cache")
parser.add_argument("check_type", nargs='?',
                    help="What to check: solver, eigen or compiler. If command=='cbrange', it is the mode (currently only 'merge'). If command=='cache', it is the action ('usage' or 'clear')", default="")
parser.add_argument("check_name", nargs='?',
                    help="Which solver/eigensolver/compiler to check. If command=='cbrange', it is the output dir", default="")
parser.add_argument("cbrange_in", nargs='*',
                    help="Input directories to merge the colorbar ranges")
arglist = parser.parse_args()

mac_machine = os.uname().machine if sys.platform == "darwin" else ""
is_mac_arm64 = (sys.platform == "darwin" and mac_machine in {"arm64", "aarch64"})
is_mac_x86_64 = (sys.platform == "darwin" and mac_machine == "x86_64")


def test_solver(solver):
   from . generic.problem import Problem
   from . equations.harmonic_oscillator import HarmonicOscillator
   from . equations.generic import InitialCondition
   from . expressions import var
   
   with tempfile.TemporaryDirectory(prefix="pyoomph_test_"+solver+"_") as tempdir:
      # Use Problem as a context manager so its compiled DLL is unloaded (release()) before
      # the TemporaryDirectory above tries to delete it - on Windows, deleting a directory
      # that still contains a loaded DLL fails with WinError 32 ("used by another process").
      with Problem() as p:
         p.quiet()

         p.set_linear_solver(solver)
         p.set_output_directory(tempdir)

         p+=HarmonicOscillator(omega=1,name="y")@"globals"
         p+=InitialCondition(y=1-var("time"))@"globals"

         p.run(endtime=1,timestep=0.1)
         ydest=-0.25942176379851877854
         if abs(float(p.get_ode("globals").get_value("y"))-ydest)>1e-7:
            raise RuntimeError("Solver did not compute the correct result, got {}, expected {}".format(float(p.get_ode("globals").get_value("y")),ydest))
      
def test_compiler(compiler):
   from . generic.problem import Problem
   from . equations.harmonic_oscillator import HarmonicOscillator
   from . equations.generic import InitialCondition
   from . expressions import var
   
   with tempfile.TemporaryDirectory(prefix="pyoomph_test_"+compiler+"_") as tempdir:
      # See test_solver() above: release the compiled DLL before the TemporaryDirectory
      # cleanup runs, or Windows refuses to delete the directory (WinError 32).
      with Problem() as p:
         p.quiet()

         p.set_c_compiler(compiler)
         p.set_output_directory(tempdir)

         p+=HarmonicOscillator(omega=1,name="y")@"globals"
         p+=InitialCondition(y=1-var("time"))@"globals"

         p.initialise()

def check_jit_cache(compiler):
   # Verifies the JIT code cache (pyoomph/generic/jit_cache.py) actually gets used, not just
   # that it fails to crash: run the same tiny problem twice against a fresh, empty cache
   # directory. The first pass can't possibly find anything cached yet ("ignoring the cache"
   # in the sense of not benefiting from whatever might already be in the real default cache
   # location), but writes its result into it normally. The second pass reuses that same
   # (now populated) directory, and should be served entirely from what the first pass wrote -
   # checked via the cache's own hit counter, since test_compiler() runs quiet() and the
   # ordinary "JIT cache hit" progress message is suppressed.
   from .generic.jit_cache import get_jit_cache, is_enabled as jit_cache_is_enabled
   import shutil

   if not jit_cache_is_enabled():
      print("","","JIT code cache is disabled (see pyoomph.generic.jit_cache), skipping cache verification")
      return

   old_cache_dir_env = os.environ.get("PYOOMPH_JIT_CACHE_DIR")
   tempdir = tempfile.mkdtemp(prefix="pyoomph_jitcache_check_")
   os.environ["PYOOMPH_JIT_CACHE_DIR"] = tempdir
   try:
      test_compiler(compiler)  # Pass 1: populates the fresh cache directory
      cache = get_jit_cache()
      if cache is None:
         print("","","JIT code cache is disabled (see pyoomph.generic.jit_cache), skipping cache verification")
         return
      hits_before = cache.hits
      test_compiler(compiler)  # Pass 2: should be served entirely from pass 1's writes
      if cache.hits > hits_before:
         print("","","JIT code cache seems to work ("+str(cache.hits-hits_before)+" hit(s) on the second pass)")
      else:
         print("","","JIT code cache does NOT seem to be used (0 hits on the second pass, "+str(cache.misses)+" entries were written on the first pass)")
   finally:
      if old_cache_dir_env is None:
         os.environ.pop("PYOOMPH_JIT_CACHE_DIR", None)
      else:
         os.environ["PYOOMPH_JIT_CACHE_DIR"] = old_cache_dir_env
      shutil.rmtree(tempdir, ignore_errors=True)

def test_eigen(eigensolver):
   from . generic.problem import Problem
   from . equations.harmonic_oscillator import HarmonicOscillator
   
   with tempfile.TemporaryDirectory(prefix="pyoomph_test_"+eigensolver+"_") as tempdir:
      # See test_solver() above: release the compiled DLL before the TemporaryDirectory
      # cleanup runs, or Windows refuses to delete the directory (WinError 32).
      with Problem() as p:
         p.quiet()

         p.set_eigensolver(eigensolver)
         p.set_output_directory(tempdir)

         p+=HarmonicOscillator(omega=1,damping=0.1,name="y",first_derivative_name="yprime")@"globals"

         p.initialise()
         p.solve()
         p.solve_eigenproblem(1)
         dest_eigenvalue=-0.1+0.99498743710662j
         calced_eigenvalue=p.get_last_eigenvalues()[0]
         if abs(calced_eigenvalue-dest_eigenvalue)>1e-7 and abs(calced_eigenvalue-dest_eigenvalue.conjugate())>1e-7:
            raise RuntimeError("Eigensolver did not compute the correct result, got {}, expected {}".format(calced_eigenvalue,dest_eigenvalue))
   

if arglist.command == "cbrange":
   process_type = arglist.check_type
   valid_process_types = {"merge"}
   if process_type not in valid_process_types:
      raise RuntimeError(
         "Please add an argument what to do with the cb_ranges: "+str(valid_process_types))
   outf = arglist.check_name
   if outf == "":
      raise RuntimeError("Please specify an output directory")
   rest = arglist.cbrange_in
   if len(rest) < 2:
      raise RuntimeError(
         "Require at least two input plot directories to merge the cb_ranges")
   merged:dict[int,dict[str,list[float]]] = {}
   for d in rest:
      gl = glob(os.path.join(d, "cb_ranges_*.txt"))
      gldict={}
      for entry in gl:
         num=int(entry.split("_")[-1].rstrip(".txt"))
         data:dict[str,list[float]]=json.load(open(entry,"r"))
         if num in merged.keys():
            for d,v in data.items():
               if d in merged[num].keys():
                  merged[num][d][0]=min(merged[num][d][0],data[d][0])
                  merged[num][d][1]=max(merged[num][d][1],data[d][1])
         else:
            merged[num]=data
   os.makedirs(outf,exist_ok=True)
   for d,data in merged.items():
      write_f=open(os.path.join(outf,"cb_ranges_{:05d}.txt".format(d)),"w")
      json.dump(data,write_f)

elif arglist.command=="check":

   checkopts={"solver","eigen","compiler"}
   if arglist.check_type=="all":
      check_types=checkopts
      arglist.check_name="all"
   elif arglist.check_type not in checkopts:
      raise RuntimeError("Please specify 'check all', 'check solver', 'check eigen' or 'check compiler'")
   else:
      check_types=[arglist.check_type]
   for check_type in check_types:
      if check_type=="solver":

         from .generic import Problem
         from .solvers.generic import GenericLinearSystemSolver
         p=Problem()
               
         sublist={"pardiso","superlu","accelerate","petsc","petsc_mumps"}
         #if arglist.check_name not in sublist:
         #   raise RuntimeError("Can only check the following: "+str(sublist))
         if arglist.check_name=="all":
            checklist=list(sublist)
         else:
            checklist=[arglist.check_name]
         
         for check in checklist:
            print("Checking "+check_type+" / "+check)
            if check=="pardiso" and is_mac_arm64:               
               print("","skipping on macOS arm64")
               continue
            if check=="accelerate" and not (is_mac_x86_64 or is_mac_arm64):               
               print("","skipping on non-macOS")
               continue
                        
            try:
               GenericLinearSystemSolver.factory_solver(check,p)
               print("","loading seems to work")
               try:
                  test_solver(check)
                  print("","","running seems to work")
               except Exception as e:
                  print("","","running does not work: "+str(e.with_traceback(None)))
                  if check=="pardiso":
                     print("Hint: Try downgrading MKL Pardiso via")
                     print("","pip install mkl==2021.4.0")
                  
            except Exception as e:
               print("","does not work: "+str(e.with_traceback(None)))

      elif check_type=="eigen":

         from .generic import Problem
         from .solvers.generic import GenericEigenSolver
         p=Problem()
               
         sublist={"pardiso","scipy","accelerate","slepc","slepc_mumps"}
         #if arglist.check_name not in sublist:
         #   raise RuntimeError("Can only check the following: "+str(sublist))
         if arglist.check_name=="all":
            checklist=list(sublist)
         else:
            checklist=[arglist.check_name]
         
         for check in checklist:
            print("Checking "+check_type+" / "+check)
            if check=="pardiso" and is_mac_arm64:               
               print("","skipping on macOS arm64")
               continue
            if check=="accelerate" and not (is_mac_x86_64 or is_mac_arm64):               
               print("","skipping on non-macOS")
               continue            
            try:
               GenericEigenSolver.factory_solver(check,p)
               print("","loading seems to work")
               try:
                  test_eigen(check)
                  print("","","running seems to work")
               except Exception as e:
                  print("","","running does not work: "+str(e.with_traceback(None)))
                  
            except Exception as e:
               print("","does not work: "+str(e.with_traceback(None)))

      elif check_type=="compiler":
         from .generic.ccompiler import BaseCCompiler
         compilers={"system"}
         if sys.platform == "win32":
            install_doc_url="https://pyoomph.readthedocs.io/en/latest/tutorial/installation/pypa.html#windows"
         elif sys.platform == "darwin":
            install_doc_url="https://pyoomph.readthedocs.io/en/latest/tutorial/installation/pypa.html#mac"
         else:
            install_doc_url="https://pyoomph.readthedocs.io/en/latest/tutorial/installation/pypa.html#linux"
         if arglist.check_name=="all":
            checklist=list(compilers)
         else:
            checklist=[arglist.check_name]
         for to_check in checklist:
            print("Checking "+check_type+" / "+to_check)
            cc=None
            try:
               cc=BaseCCompiler.factory_compiler(to_check)
               # factory_compiler is declared to return the C++ base class _pyoomph.CCompiler,
               # but at runtime it always instantiates one of the registered BaseCCompiler
               # subclasses (see BaseCCompiler._registered_compilers), which is where
               # check_avail()/toolchain_located() are actually defined.
               assert isinstance(cc,BaseCCompiler)
               if cc.check_avail():
                  print("","loading seems to work")
                  try:
                     test_compiler(to_check)
                     print("","","running seems to work")
                     check_jit_cache(to_check)
                  except Exception as e:
                     print("","","C compilation seems to work: "+str(e.with_traceback(None)))
               else:
                  raise RuntimeError("Sanity check not working...")
            except Exception as e:
               # check_avail() above compiles+links a real temp file, which
               # conflates "no compiler" with "can't write to the temp dir"
               # (e.g. a locked-down Windows machine). If a lighter, no-file
               # -write probe is available for this compiler (currently just
               # MSVC, via toolchain_located()), use it to give a clearer
               # diagnostic than the raw exception below.
               located=None
               if cc is not None:
                  assert isinstance(cc,BaseCCompiler)
                  try:
                     located=cc.toolchain_located()
                  except Exception:
                     pass
               if located is False:
                  print("","does not work: compiler toolchain not found (e.g. no Visual Studio/Build Tools installation located)")
                  print("","For instructions on how to install a compiler, see "+install_doc_url)
               elif located is True:
                  print("","toolchain found, but the sanity check failed - this can happen e.g. if the "
                            "temporary directory used for the test could not be written to: "+str(e.with_traceback(None)))
               else:
                  print("","does not work: "+str(e.with_traceback(None)))
                  print("","For instructions on how to install a compiler, see "+install_doc_url)
      else:
         raise RuntimeError("TODO: ")

elif arglist.command=="cache":
   from .generic.jit_cache import get_cache_dir, is_enabled as jit_cache_is_enabled, JITCache, clear_cache

   def _format_bytes(n):
      f = float(n)
      for unit in ("B","KB","MB","GB","TB"):
         if f < 1024 or unit=="TB":
            return "{:.1f} {}".format(f,unit) if unit!="B" else "{:.0f} {}".format(f,unit)
         f/=1024
      return "{:.1f} TB".format(f) # unreachable, keeps type checkers happy

   action = arglist.check_type
   if action not in {"usage","clear"}:
      raise RuntimeError("Please specify 'cache usage' or 'cache clear'")

   cache_dir = get_cache_dir()

   if action=="usage":
      print("Cache directory: "+cache_dir)
      if jit_cache_is_enabled():
         print("JIT code cache is currently ACTIVE")
      else:
         print("JIT code cache is currently DISABLED (see pyoomph.generic.jit_cache for why - "
               "e.g. GiNaC was not built with the deterministic-hash patches, or PYOOMPH_JIT_CACHE=0)")
      if not os.path.isdir(cache_dir):
         print("","Cache directory does not exist yet (nothing has been cached so far)")
      else:
         stats=JITCache(cache_dir).get_usage_stats()
         print("","Compiled code objects: "+str(stats["objects_count"])+" entries, "
               +_format_bytes(stats["objects_bytes"])+" / "+_format_bytes(stats["objects_max_bytes"])
               +" (PYOOMPH_JIT_CACHE_MAX_MB)")
         print("","Tier-2 fingerprint bookkeeping: "+str(stats["fingerprints_count"])+" entries "
               +"/ "+str(stats["fingerprints_max_count"])+" (PYOOMPH_JIT_CACHE_MAX_FINGERPRINTS)")

   elif action=="clear":
      if clear_cache(cache_dir):
         print("Cleared JIT code cache directory: "+cache_dir)
      else:
         print("Cache directory did not exist, nothing to clear: "+cache_dir)

else:
   raise RuntimeError("Please use one of the following commands: check, cbrange, cache")


