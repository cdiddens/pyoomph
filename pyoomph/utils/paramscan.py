#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2024  Christian Diddens & Duarte Rocha
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
 
import functools
import io
#from concurrent.futures import *
import concurrent.futures 
import subprocess
import os
from pathlib import Path
import sys

import __main__

import operator

from ..typings import *


#Sorts an array [0,1,2,3,4] like this [0,4,2,1,3]
#Useful to for parameter scans: First do the extremes and the center, then gradually refine the spaces in between
def alternate_sorting(inp:List[float])->List[float]:
    if len(inp) < 3:
        return inp
    centrali = len(inp) // 2
    res = [inp[0], inp[-1], inp[centrali]]
    sublist:List[List[float]] = []
    if centrali > 1:
        sublist.append(inp[1:centrali])
    if centrali + 1 < len(inp):
        sublist.append(inp[centrali + 1:-1])
    while len(sublist) > 0:
        old = sublist
        sublist = []
        for s in old:
            if len(s) > 0:
                centrali = len(s) // 2
                res.append(s[centrali])
                if centrali > 0:
                    sublist.append(s[0:centrali])
                if centrali < len(s):
                    sublist.append(s[centrali + 1:])
    return res


class SimulationNamespace:
    """
    A class representing a namespace for simulation parameters for the parallel parameter scan.
    If your problem has e.g. an nested property like problem.droplet.contact_angle, you must create a SimulationNamespace object for the SingleParallelParameterSimulation object, i.e. 
    
        sim.droplet=SimulationNamespace()
        sim.droplet.contact_angle=...
        
    Otherwise, nested properties cannot be set.
    """

    def _INTERNAL_add_to_arglist(self,argnames:List[str],trunk:str)->None:
        for x in dir(self):
            if x.startswith("_INTERNAL_") or x.startswith("__"):
                continue
            if isinstance(getattr(self,x),SimulationNamespace):
                getattr(self, x)._INTERNAL_add_to_arglist(argnames,trunk+"."+x)
            else:
                argnames.append(trunk+"."+x)


class SingleParallelParameterSimulation:
    """
    Storage class for the used parameters of a single simulation in a parallel parameter scan.
    Instances are generated by 
        
    .. code-block:: python
    
        sim=ParallelParameterScan.new_sim(...)
        
    Once generated, the parameters can be set as attributes of the instance sim, e.g.

    .. code-block:: python
            
        sim.paramA=...
        
    For nested parameters, use the SimulationNamespace class, e.g.

    .. code-block:: python
        
        sim.droplet=SimulationNamespace()
        sim.droplet.contact_angle=...
    """
    def __init__(self,subdir:Optional[str],additional_args:List[str]):
        self._INTERNAL_subdir=subdir
        self._INTERNAL_additional_args=additional_args
        self._INTERNAL_pararunner:Optional["ParallelParameterScan"]=None
        self._INTERNAL_script:Optional[str]=None

    def _INTERNAL_assemble_args(self) -> Optional[List[str]]:
        argnames:List[str]=[]
        for x in dir(self):
            if x[0]=="_":
                continue
            if isinstance(getattr(self,x),SimulationNamespace):
                getattr(self, x)._INTERNAL_add_to_arglist(argnames,x)
            else:
                argnames.append(x)



        #print(argnames)
        if self._INTERNAL_subdir is None:
            if len(argnames)==0:
                raise RuntimeError("Cannot run a ParallelParameterScan new_sim without sub-directory if there are no parameters passed")
            #self._INTERNAL_subdir= "__".join([an + "_" + str(getattr(self, an)) for an in argnames])
            self._INTERNAL_subdir = "__".join([an + "_" + str(operator.attrgetter(an)(self)) for an in argnames])

        if self._INTERNAL_pararunner is None:
            raise RuntimeError("The simulation was not correctly created with ParallelParameterScan.new_sim(...)")
        assert self._INTERNAL_script is not None
        args=["-u",self._INTERNAL_script,"--outdir", os.path.join(self._INTERNAL_pararunner._output_dir, self._INTERNAL_subdir)] #type: ignore
        if len(argnames)>0:
            args.append("-P")
            for an in argnames:
                root=self
                splt=an.split(".")
                for i in range(len(splt)-1):
                    root=getattr(root,splt[i])
                args.append(an + "=" + str(getattr(root, splt[-1])))
        args += self._INTERNAL_additional_args
        return args

def _para_sim_call(args:List[str],logfilename:str,env:Dict[str,str]) -> bool:
    print("STARTING ",args)
    #p = subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,shell=0)
    if logfilename is not None:
        logfile=io.open(logfilename,"w")
    else:
        logfile=subprocess.DEVNULL
    p = subprocess.Popen(args, stdout=logfile, stderr=logfile, shell=False,env=env)
    res=p.wait() == 0
    print("DONE ",res, args)
    return res



class ParallelParameterScan:
    """
    A class for performing parallel parameter scans.

    Args:
        script_to_call (str): The script to call for each parameter simulation.
        output_dir (Optional[str]): The output directory for the parameter simulations. If not provided, a default directory will be created based on the script's name.
        max_procs (Optional[int]): The maximum number of processes to use for parallel execution. Defaults to the number of CPUs in the system.
        single_threaded_childs (bool): Whether to run each child process in single-threaded mode. Defaults to True.
        interpreter (str): The path to the Python interpreter to use. Defaults to the system's default interpreter.
    
    """
    def __init__(self,script_to_call:str,output_dir:Optional[str]=None,max_procs:Optional[int]=os.cpu_count(),single_threaded_childs:bool=True,interpreter:str=sys.executable):
        self._script=script_to_call
        self._interpreter=interpreter
        if output_dir is None:
            parascript=__main__.__file__
            bn=os.path.splitext(parascript)[0]
            output_dir=os.path.join(os.path.dirname(bn), os.path.basename(bn))
        self._output_dir=output_dir
        self._max_procs=max_procs   #0 means Nprocs of system
        self._single_threaded_childs=single_threaded_childs
        self._sims:List[SingleParallelParameterSimulation]=[]
        self._donefile=None

    def mark_as_done(self,sim:SingleParallelParameterSimulation,args:List[str])->None:
        if self._donefile is None:
            self._donefile=open(os.path.join(self._output_dir,"DONE_SIMS.txt"),"a")
        if self.already_done(args):
            return
        self._donefile.write("\t".join(args)+"\n")
        self._donefile.flush()

    #only_by_script_and_outdir is important, since passed expressions might be differently ordered!
    def already_done(self,args:List[str],only_by_script_and_outdir:bool=True)->bool:
        try:
            _donefile = open(os.path.join(self._output_dir, "DONE_SIMS.txt"), "r")
            if only_by_script_and_outdir:
                args=args[0:min(5,len(args))]
            tofind = "\t".join(args)+"\t"
            if not only_by_script_and_outdir:
                tofind+= "\n"
            for l in _donefile.readlines():
                if only_by_script_and_outdir:
                    if l.startswith(tofind):
                        return True
                else:
                    if l == tofind:
                        return True
        except:
            return False

        return False


    def new_sim(self, subdir: Optional[str] = None, additional_args: List[str] = []) -> SingleParallelParameterSimulation:
        """
        Create a new simulation. You can set the parameters of the simulation by setting attributes of the returned instance.

        Args:
            subdir (str, optional): The subdirectory where the simulation will be saved. Defaults to None, will be determined automatically based on the parameters.
            additional_args (List[str], optional): Additional arguments to be passed to the simulation. Defaults to an empty list.

        Returns:
            SingleParallelParameterSimulation: The newly created simulation instance.
        """
        res = SingleParallelParameterSimulation(subdir, additional_args.copy())
        res._INTERNAL_script = self._script  # type: ignore
        res._INTERNAL_pararunner = self  # type: ignore
        self._sims.append(res)
        return res
    

    def clear(self):
        """
        Remove all simulations from the list.
        """
        self._sims=[]

    def done_callback(self,sim): #type: ignore
        if sim.result(): #type: ignore
            self.mark_as_done(sim._sim,sim._sim._args) #type: ignore

    def run_all(self, skip_done: bool = False):
        """
        Run all added simulations in parallel, by default single-CPU each, always using a maximum of max_procs processes.

        Args:
            skip_done (bool, optional): Whether to skip simulations that are already completed. Defaults to False.
        """

        Path(self._output_dir).mkdir(parents=True, exist_ok=True)
        my_env = os.environ.copy()
        if self._single_threaded_childs:
            my_env["OMP_NUM_THREADS"] = "1"
            my_env["MKL_NUM_THREADS"] = "1"
            my_env["OPENBLAS_NUM_THREADS"] = "1"
            my_env["MKL_DOMAIN_NUM_THREADS"] = "MKL_BLAS=1, MKL_DOMAIN_PARDISO=1"
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_procs) as executor:
            for s in self._sims:
                args = s._INTERNAL_assemble_args()  # type: ignore
                args: List[str] = [self._interpreter] + args
                s._args = args  # type: ignore
                if skip_done:
                    if self.already_done(args):
                        print("SKIPPING COMPLETED ", args)
                        continue

                logfile = os.path.join(self._output_dir, "log_" + s._INTERNAL_subdir + ".txt")  # type: ignore
                future = executor.submit(functools.partial(_para_sim_call, args, logfile, my_env))
                future._sim = s  # type: ignore
                future.add_done_callback(lambda sim: self.done_callback(sim))  # type: ignore



#para=ParallelParameterScan("myscript.py")
#sim=para.new_sim("SimA")
#sim.paramA="bla"
#sim.paramB=100
#para.run_all()

