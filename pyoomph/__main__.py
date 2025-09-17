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
 
from glob import glob
import sys
import os
import json
import argparse
import tempfile

import subprocess
from typing import Dict, Tuple, Set, List

class _BaseMainEntry:
   def show(self):
      pass



class _CheckEntry(_BaseMainEntry):
   pass


entries = {"check": _CheckEntry()}

parser = argparse.ArgumentParser()
parser.add_argument(
    "command", help="Use one of the following commands: check, cbrange")
parser.add_argument("check_type", nargs='?',
                    help="What to check: solver, eigen or compiler. If command=='cbrange', it is the mode (currently only 'merge')", default="")
parser.add_argument("check_name", nargs='?',
                    help="Which solver/eigensolver/compiler to check. If command=='cbrange', it is the output dir", default="")
parser.add_argument("cbrange_in", nargs='*',
                    help="Input directories to merge the colorbar ranges")
arglist = parser.parse_args()


def test_solver(solver):
   from . generic.problem import Problem
   from . equations.harmonic_oscillator import HarmonicOscillator
   from . equations.generic import InitialCondition
   from . expressions import var
   
   with tempfile.TemporaryDirectory(prefix="pyoomph_test_"+solver+"_") as tempdir:
      p=Problem()
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
      p=Problem()
      p.quiet()
      
      p.set_c_compiler(compiler)
      p.set_output_directory(tempdir)      
      
      p+=HarmonicOscillator(omega=1,name="y")@"globals"
      p+=InitialCondition(y=1-var("time"))@"globals"
      
      p.initialise()

def test_eigen(eigensolver):
   from . generic.problem import Problem
   from . equations.harmonic_oscillator import HarmonicOscillator
   from . equations.generic import InitialCondition
   from . expressions import var
   
   with tempfile.TemporaryDirectory(prefix="pyoomph_test_"+eigensolver+"_") as tempdir:
      p=Problem()
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
   merged:Dict[int,Dict[str,Tuple[float,float]]] = {}
   for d in rest:
      gl = glob(os.path.join(d, "cb_ranges_*.txt"))
      gldict={}
      for entry in gl:
         num=int(entry.split("_")[-1].rstrip(".txt"))
         data:Dict[str,Tuple[float,float]]=json.load(open(entry,"r"))
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
               
         sublist={"pardiso","superlu"}
         #if arglist.check_name not in sublist:
         #   raise RuntimeError("Can only check the following: "+str(sublist))
         if arglist.check_name=="all":
            checklist=list(sublist)
         else:
            checklist=[arglist.check_name]
         
         for check in checklist:
            print("Checking "+check_type+" / "+check)         
            try:
               GenericLinearSystemSolver.factory_solver(check,p)
               print("","loading seems to work")
               try:
                  test_solver(check)
                  print("","running seems to work")
               except Exception as e:
                  print("","running does not work: "+str(e.with_traceback(None)))
                  if check=="pardiso":
                     print("Hint: Try downgrading MKL Pardiso via")
                     print("","pip install mkl==2021.4.0")
                  
            except Exception as e:
               print("","does not work: "+str(e.with_traceback(None)))

      elif check_type=="eigen":

         from .generic import Problem
         from .solvers.generic import GenericEigenSolver
         p=Problem()
               
         sublist={"pardiso","scipy"}
         #if arglist.check_name not in sublist:
         #   raise RuntimeError("Can only check the following: "+str(sublist))
         if arglist.check_name=="all":
            checklist=list(sublist)
         else:
            checklist=[arglist.check_name]
         
         for check in checklist:
            print("Checking "+check_type+" / "+check)         
            try:
               GenericEigenSolver.factory_solver(check,p)
               print("","loading seems to work")
               try:
                  test_eigen(check)
                  print("","running seems to work")
               except Exception as e:
                  print("","running does not work: "+str(e.with_traceback(None)))
                  
            except Exception as e:
               print("","does not work: "+str(e.with_traceback(None)))

      elif check_type=="compiler":
         from .generic.ccompiler import BaseCCompiler
         compilers={"system"}
         if arglist.check_name=="all":
            checklist=list(compilers)
         else:
            checklist=[arglist.check_name]
         for to_check in checklist:
            print("Checking "+check_type+" / "+to_check)     
            try:
               cc=BaseCCompiler.factory_compiler(to_check)
               if cc.check_avail():
                  print("","loading seems to work")
                  try:
                     test_compiler(to_check)
                     print("","running seems to work")
                  except Exception as e:
                     print("","C compilation seems to work: "+str(e.with_traceback(None)))
               else:
                  raise RuntimeError("Sanity check not working...")
            except Exception as e:
               print("","does not work: "+str(e.with_traceback(None)))
      else:
         raise RuntimeError("TODO: ")
else:
   raise RuntimeError("Please use one of the following commands: check")


