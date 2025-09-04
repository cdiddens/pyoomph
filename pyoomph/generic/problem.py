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
 
import glob

import scipy.sparse

from ..expressions.generic import is_zero #type:ignore

#import pyoomph.generic
from .mpi import *
import _pyoomph
import math


import __main__

import os
import gc
from pathlib import Path

import argparse
import numpy
from ..meshes.mesh import  AnyMesh,AnySpatialMesh, MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d, ODEStorageMesh, InterfaceMesh,MeshFromTemplate,MeshFromTemplateBase,MeshTemplate
from .codegen import EquationTree,BaseEquations, FiniteElementCodeGenerator,CombinedEquations,DummyEquations, InterfaceEquations #ODEEquations
from ..solvers.generic import DefaultMatrixType, EigenSolverWhich, GenericLinearSystemSolver,GenericEigenSolver
#from ..solvers.scipy import SuperLUSerial,ScipyEigenSolver
from ..expressions.units import *
from ..expressions import get_global_symbol,cartesian,axisymmetric,axisymmetric_flipped,radialsymmetric,BaseCoordinateSystem,nondim,testfunction,evaluate_in_past,weak,OptionalCoordinateSystem
from ..solvers.load_solver_from_cmd_line import *
from ..solvers.generic import get_default_linear_solver,get_default_eigen_solver
from ..meshes.interpolator import _DefaultInterpolatorClass,ODEInterpolator 
from ..output.states import DumpFile
from ..expressions import ExpressionOrNum,ExpressionNumOrNone
from ..meshes.meshdatacache import MeshDataCacheStorage, MeshDataCacheOperatorBase, MeshDataEigenModes,MeshDataCacheEntry

from .ccompiler import BaseCCompiler,SystemCCompiler

import types 

from ..typings import *

if TYPE_CHECKING:
    from ..output.plotting import MatplotlibPlotter
    from ..meshes.remesher import RemesherBase
    from ..meshes.interpolator import BaseMeshToMeshInterpolator
    from .assembly import CustomAssemblyBase
    from ..utils.num_text_out import NumericalTextOutputFile

Z2ErrorEstimator=_pyoomph.Z2ErrorEstimator

import subprocess

import signal
def breakpoint():
    os.kill(os.getpid(), signal.SIGTRAP)

#To use "with problem.custom_adapt:" statement
class _CustomAdaptWithHelper:
    def __init__(self, problem: "Problem",skip_init_call:bool=False):
        self._problem=problem
        self._skip_init_call=skip_init_call
    def __enter__(self):
        if not self._skip_init_call and  not self._problem.is_initialised():
            self._problem.initialise()
        self._problem.actions_before_adapt()
    def __exit__(self,exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[types.TracebackType]):
        self._problem.actions_after_adapt()
        self._problem.before_assigning_equation_numbers(self._problem._dof_selector) #type: ignore
        num=self._problem.assign_eqn_numbers(True)
        if not self._problem.is_quiet():
            print("Number of equations: "+str(num))


class _AzimuthalStabilityInfo:
    def __init__(self):
        super(_AzimuthalStabilityInfo, self).__init__()
        self.real_contribution_name="real_contrib_azimuthal_stability"
        self.imag_contribution_name="imag_contrib_azimuthal_stability"
        self.azimuthal_param_m_name = "azimuthal_m"
        
class _CartesianNormalModeStabilityInfo:
    def __init__(self):
        super(_CartesianNormalModeStabilityInfo, self).__init__()
        self.real_contribution_name="real_contrib_normal_mode_stability"
        self.imag_contribution_name="imag_contrib_normal_mode_stability"
        self.normal_mode_param_k_name = "normal_mode_k"        


class GenericProblemHooks:
    """
    A class that can be attached to a problem to call additional functions after e.g. newton solves, etc.
    """
    def __init__(self):
        self._problem:Optional["Problem"]=None
        
    def get_problem(self)->"Problem":
        if self._problem is None:
            raise RuntimeError("Problem not set")
        return self._problem

    def actions_after_remeshing(self):
        pass 
    
    def actions_after_change_in_global_parameter(self,param:str):
        pass
    
    def actions_before_remeshing(self,active_remeshers:List["RemesherBase"]):
        pass

    def actions_after_newton_solve(self):
        pass
    
    def actions_before_newton_solve(self):
        pass
    
    def actions_after_newton_step(self):
        pass
    
    def before_assigning_equation_numbers(self,dof_selector:Optional["_DofSelector"],before_equation_system:bool):
        pass
    
    def actions_after_parameter_increase(self,param:str):
        pass
    
    def actions_after_initialise(self):
        pass
    
    def actions_on_output(self,outstep):
        pass
    


        
class PeriodicOrbit:
    """ 
    A class representing a periodic orbit.
    """
    def __init__(self,problem:"Problem",mode,lyap_coeff,param,omega,pvalue,pdvalue,al,order,GL_order,T_constraint):
         self.problem=problem
         self.mode=mode
         self.order,self.GL_order=order,GL_order
         self.T_constraint=T_constraint
         self.emerging_info={"lyap_coeff":lyap_coeff,"param":param,"omega":omega,"pvalue":pvalue,"dpvalue":pdvalue,"al":al}
         
    def __enter__(self):
        return self
    
    def __exit__(self,exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[types.TracebackType]):
        #  Setup the history dofs for a transient continuation
        N=self.get_num_time_steps()
        T=self.get_T(dimensional=False)
        dt=T/N        
        self._get_handler().backup_dofs()
        history=[]
        for s in [0,-1/N,-2/N]:
            self._get_handler().set_dofs_to_interpolated_values(s) # TODO: We might need the time history for e.g. local expressions, integrals, etc. involving partial_t            
            history.append(self.problem.get_current_dofs()[0][:self._get_handler().get_base_ndof()])
        self._get_handler().restore_dofs()
        self.problem.deactivate_bifurcation_tracking()
        for i,h in enumerate(history):
            #print(i,h)                        
            self.problem.set_history_dofs(i,h)
        
        self.problem.initialise_dt(dt)
        
        for i,h in enumerate(history):
            self.problem.time_pt().set_dt(i,dt)            
            self.problem.set_history_dofs(i,h)
        self.problem.time_stepper_pt().set_weights()
        self.problem.shift_time_values()
        self.problem.shift_time_values()
        self.problem.shift_time_values()
        self.problem.time_stepper_pt().undo_make_steady()
        self.problem._taken_already_an_unsteady_step=True
        self.problem._last_step_was_stationary=False
        self.problem.actions_before_transient_solve()
        
    
    def _get_handler(self)->_pyoomph.PeriodicOrbitHandler:
        handler=self.problem.assembly_handler_pt()
        if not isinstance(handler,_pyoomph.PeriodicOrbitHandler):
            raise ValueError("Periodic orbit handler not activated (anymore)")
        return handler
    
    def get_T(self,dimensional=True):
        """
        Returns the period time of the orbit
        """
        return self._get_handler().get_T()*(self.problem.get_scaling("temporal") if dimensional else 1)
    
    def get_init_ds(self):
        """
        Returns a reasonable initial step size for arclength continuation        
        """
        if abs(self.emerging_info["dpvalue"]-self.emerging_info["pvalue"])<=5e-10:
            return 5e-10*(1 if self.emerging_info["dpvalue"]-self.emerging_info["pvalue"]>0 else -1)            
        return self.emerging_info["dpvalue"]-self.emerging_info["pvalue"]
    
    def get_num_time_steps(self):
        """
        Returns the number of time steps of the discretized orbit
        """
        return self._get_handler().get_num_time_steps()
    
    def update_phase_constraint(self):
        """
        Updates the phase constraint history (u0) for the orbit
        """
        self._get_handler().update_phase_constraint_information()
    
    def output_orbit(self,subdir:str,Tstart:Optional[float]=None,Tend:Optional[float]=None,N:Optional[int]=None,set_current_time:bool=True,endpoint:bool=True):
        olddir=self.problem.get_output_directory()
        write_states=self.problem.write_states
        outstep=self.problem._output_step
        self.problem.write_states=False
        self.problem._change_output_directory(self.problem.get_output_directory(subdir))
        for sample in self.iterate_over_samples(Tstart=Tstart,Tend=Tend,N=N,set_current_time=set_current_time,endpoint=endpoint):
            self.problem.output(quiet=True)
        self.problem._change_output_directory(olddir)
        self.problem.write_states=write_states
        self.problem._output_step=outstep
    
    def iterate_over_samples(self,Tstart:Optional[float]=None,Tend:Optional[float]=None,N:Optional[int]=None,set_current_time:bool=True,endpoint:bool=True):
        tbackup=self.problem.get_current_time(dimensional=False,as_float=True)
        TS=self.problem.get_scaling("temporal")
        T=self.get_T(dimensional=False)
        if N is None:
            N=self.get_num_time_steps()
        if Tstart is None:
            Tstart=0.0
        else:
            Tstart=float(Tstart/TS)
        if Tend is None:
            Tend=Tstart+T
        else:
            Tend=float(Tend/TS)
        
        ssamples=numpy.linspace(Tstart,Tend,N,endpoint=endpoint)/T
        print("Backing up dofs")
        self._get_handler().backup_dofs()
        for s in ssamples:
            self._get_handler().set_dofs_to_interpolated_values(s) # TODO: We might need the time history for e.g. local expressions, integrals, etc. involving partial_t            
            self.problem.invalidate_cached_mesh_data()
            Tcurr=s*T
            if set_current_time:
                self.problem.set_current_time(Tcurr,dimensional=False,as_float=True)
            yield Tcurr*TS
        print("Restoring dofs")
        self._get_handler().restore_dofs()
        self.problem.set_current_time(tbackup,dimensional=False,as_float=True)
        
    def get_floquet_multipliers(self,n:Optional[int]=None,valid_threshold:Optional[float]=10000,shift:Optional[Union[float]]=None,ignore_periodic_unity:Union[bool,float]=False,quiet:bool=True):
        return self.problem.get_floquet_multipliers(n=n,valid_threshold=valid_threshold,shift=shift,ignore_periodic_unity=ignore_periodic_unity,quiet=quiet)
    
    def starts_supercritically(self):
        """
        When started at a Hopf bifurcation, this function tells you whether the first Lyaupnov coefficient is negative, corresponding to a supercritical Hopf bifurcation with initially stable orbits
        """
        
        return self.emerging_info["lyap_coeff"]<0
    
    def evalulate_observable_time_integral(self,*observables:str):
        if len(observables)==0:
            raise ValueError("No observables given")
        accus={n:0 for n in observables}
        obs_info:Dict[str,Tuple[AnySpatialMesh,str]]={}
        for o in observables:
            splt=o.split("/")
            if len(splt)<=1:
                raise ValueError("Observables must be given like 'domain/observable', i.e. first the mesh path, then the observable")
            meshpath,observable="/".join(splt[:-1]),splt[-1]
            mesh=self.problem.get_mesh(meshpath)
            obs_info[o]=(mesh,observable)
        
        self._get_handler().backup_dofs()
        for (s,w) in self._get_handler().get_s_integration_samples():            
            self._get_handler().set_dofs_to_interpolated_values(s) # TODO: We might need the time history for e.g. local expressions, integrals, etc. involving partial_t
            self.problem.invalidate_cached_mesh_data()
            for o in observables:
                val=obs_info[o][0].evaluate_observable(obs_info[o][1])                
                accus[o]+=val*w            
        self._get_handler().restore_dofs()
        T=self.get_T()
        if len(observables)>1:
            return tuple(accus[o]*T for o in observables)
        else:
            return accus[observables[0]]*T
        
    
    def change_sampling(self,*,mode:Literal["collocation","central","bspline","BDF2"]=None,NT:Optional[int]=None, order:Optional[int]=None,GL_order:Optional[int]=None,T_constraint:Optional[Literal["plane","phase"]]=None,do_solve:bool=True):
        if mode is None:
            mode=self.mode
        if order is None:
            order=self.order
        if GL_order is None:
            GL_order=self.GL_order
        if T_constraint is None:
            T_constraint=self.T_constraint            
        if NT is None:
            NT=self.get_num_time_steps()
        history_dofs=[]
        Nbase=self._get_handler().get_base_ndof()
        for T in self.iterate_over_samples(N=NT):
            history_dofs.append(self.problem.get_current_dofs()[0][:Nbase])
        T=self.get_T()
        self.problem.deactivate_bifurcation_tracking()
        self.problem.set_current_dofs(history_dofs.pop())
        self.problem.activate_periodic_orbit_handler(T,history_dofs,mode=mode,T_constraint=T_constraint,order=order,GL_order=GL_order)
        self.mode=mode
        self.order=order
        self.GL_order=GL_order
        self.T_constraint=T_constraint
        if do_solve:
            self.problem.solve()
        
            

#Problem with some automatic behaviour
class Problem(_pyoomph.Problem):
    """A class representing a problem in the pyoomph library.

    This class provides methods and attributes for defining and solving a problem.
    Usually, in the :py:meth:`__init__` method, you define any parameters with default settings.
    The problem itself is defined by the :py:meth:`define_problem` method, where you define the equations and the mesh(es).
    After creation of an instance, you can solve the problem by calling the :py:meth:`run` method (transient solves) or the :py:meth:`solve` method (stationary solve).
    Outputs (potentially with plots) can be generated by calling the :py:meth:`output` method.

    Attributes:
        
        additional_equations (Union[Literal[0], EquationTree]): Additional equations for the problem.
        always_take_one_newton_step (bool): Flag indicating whether to always take one Newton step.
        continuation_data_in_states (bool): Flag indicating whether to store continuation data in the states.
        default_1d_file_extension (Union[Literal["txt", "mat"], List[Literal["txt", "mat"]]]): Default file extension for 1D files.
        default_ccode_expression_mode (str): Default C code expression mode.
        default_spatial_integration_order (Union[int, None]): Default spatial integration order.
        default_timestepping_scheme (Literal["BDF2", "BDF1", "Newmark2"]): Default timestepping scheme.
        eigen_data_in_states (Union[int, bool]): Flag indicating whether to store eigen data in the states.
        eigenvector_position_scale (float): Scaling factor for eigenvector positions.
        extra_compiler_flags (List[str]): Extra compiler flags for the problem.
        ignore_command_line (bool): Flag indicating whether to ignore command line arguments.
        latex_printer (Optional[LaTeXPrinter]): LaTeX printer for the problem.
        max_residuals (float): Maximum residuals for the problem.
        plot_in_dedicated_process (bool): Flag indicating whether to plot in a dedicated process.
        remove_macro_elements_after_initial_adaption (Union[bool, Literal["auto"]]): Flag indicating whether to remove macro elements after initial adaption.
        scaling (Dict[str, Union[str, ExpressionOrNum]]): Dictionary of scaling factors.
        states_compression_level (Union[int, None]): Compression level for the states.
        timestepper (MultiTimeStepper): Timestepper for the problem.
        write_states (bool): Flag indicating whether to write states.

    """
    def __init__(self):
        """
        Initialise the problem object. After calling ``super().__init__()``, you should set the default parameters for the problem. Afterwards, the user can change them before solving the problem. 
        """
        super(Problem, self).__init__()
        self._initialised:bool=False
        self._during_initialization:bool=False

        import pyoomph
        self.set_c_compiler(pyoomph.get_default_c_compiler())

        if hasattr(__main__,"__file__"):
            scriptfile=os.path.splitext(__main__.__file__)[0]
        else:
            scriptfile="_pyoomph_output_.py"
        #self._outdir=os.path.join(os.path.dirname(scriptfile),os.path.basename(scriptfile))
        self._outdir:str = os.path.basename(scriptfile)

        self._bulk_element_code_counter:int=0

        self._first_step:bool=True
        self._suppress_code_writing:bool=False
        self._suppress_compilation:bool=False
        self._debug_largest_residual:int=0
        self.ignore_command_line:bool=False

        self._ccode_dir:str="_ccode"
        self._dof_selector:Union[_DofSelector,None]=None # The desired selected dofs
        self._dof_selector_used:Union[_DofSelector,None,Literal["INVALID"]]=None


        self._use_first_order_timestepper:bool=False
        self._domains_to_remesh:Set[MeshTemplate]=set()

        self.max_residuals=1e10
        self.max_newton_iterations=10
        self.newton_solver_tolerance=1e-8
        self._call_output_after_adapt:bool=False

        #: Spatial adaption steps for the initial condition. If set to ``None``, we refine initially up to :py:attr:`max_refinement_level`.
        self.initial_adaption_steps:Union[None,int]=None #Adapting in the first step
        self.remove_macro_elements_after_initial_adaption:Union[bool,Literal["auto"]]="auto" # "auto" means: Only if the coordinates are free

        #: Minimum error of all meshes for spatial adaptivity. If the error is below this threshold, we may unrefine locally.
        self.min_permitted_error:float=0.0001	#Some defaults for the meshes
        #: Maximum error of all meshes for spatial adaptivity. If the error is above this threshold, we must refine locally.
        self.max_permitted_error:float=0.001
        #: Maximum number of refinements of all meshes. 
        self.max_refinement_level:int=8
        #: Minimum refinement level of all meshes.       
        self.min_refinement_level:int=0
        #: Add a .gitignore with content "*" to output folders
        self.gitignore_output:bool=True
        #: Name of the logfile (or None for no logfile), relative to the output directory
        self.logfile_name:Optional[str]="_pyoomph_logfile.txt"
        #: When set to True, we warn about unused global parameters for arclength continuation or bifurcation tracking. If set to "error", we raise an error.
        self.warn_about_unused_global_parameters:Union[bool,Literal["error"]]="error"
        #:  There are different methods implemented in oomph-lib to fill the sparse matrices (Jacobian, mass matrix, etc.). Depending on the problem, one or the other method may be faster or more memory efficient. The default method is "vectors_of_pairs", which is the most general one.                
        self.sparse_assembly_method:Literal["vectors_of_pairs","two_vectors","lists","maps","two_arrays"]="vectors_of_pairs"
        self.only_write_logfile_on_proc0:bool=True
        #: Checks whether the elements in the meshes are nicely oriented (facing) so that refinement works as it should. Can be only done once initially or at each refinement step
        self.check_mesh_integrity:Union[bool,Literal["initially"]]="initially"

        self._meshtemplate_list:List[MeshTemplate]=[]
        self._meshdict={}
        self._residual_mapping_functions:List[Callable[[str,Expression],Union[Expression,Dict[str,Expression]]]]=[]

        self._named_vars:Dict[str,ExpressionOrNum]={}

        self._coordinate_system=cartesian

        self.scaling:Dict[str,Union[str,ExpressionOrNum]]={} #Add scales here, i.e. spatial=1*centi*meter, temporal=...
        self.scaling["time"]="temporal"
        self.scaling["coordinate"]="spatial" #Link the default fields to the main scales
        self.scaling["coordinate_x"]="spatial"
        self.scaling["coordinate_y"]="spatial"
        self.scaling["coordinate_z"]="spatial"

        self.scaling["mesh"]="spatial" #Link the default fields to the main scales
        self.scaling["mesh_x"]="spatial"
        self.scaling["mesh_y"]="spatial"
        self.scaling["mesh_z"]="spatial"

        self.scaling["lagrangian"] = "spatial"  # Link the default fields to the main scales
        self.scaling["lagrangian_x"] = "spatial"
        self.scaling["lagrangian_y"] = "spatial"
        self.scaling["lagrangian_z"] = "spatial"


        self._lasolver=get_default_linear_solver()

        self._num_threads:Optional[int]=None # Default
        self._eigensolver=get_default_eigen_solver()

        self._runmode="delete"
        self._continue_initialized=False
        self._where_expression="True"

        self._dump_header = "pyoomph_dump"
        self._dump_version = "0.0.1"
        self._last_bc_setting="init"

        self._output_step:int=0
        self._continue_section_step:int=0
        self._continue_section_step_loaded:int=0
        self._nondim_time_after_last_run_statement=0 # Required for continue

        self._interfacemeshes:List[InterfaceMesh]=[]
        self._last_eigenvalues:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
        self._last_eigenvectors:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
        self._last_eigenvalues_m:Optional[NPIntArray]=None
        self._last_eigenvalues_k:Optional[NPFloatArray]=None
        self._azimuthal_mode_param_m=None
        self._normal_mode_param_k=None
        self._azimuthal_stability=_AzimuthalStabilityInfo()
        self._cartesian_normal_mode_stability=_CartesianNormalModeStabilityInfo()
        self._bifurcation_tracking_parameter_name:Optional[str]=None
        self._improved_pitchfork_tracking_coordinate_system:"OptionalCoordinateSystem"=None
        self._improved_pitchfork_tracking_position_coordinate_system:"OptionalCoordinateSystem"=None
        self._shared_shapes_for_multi_assemble=False
        self._setup_azimuthal_stability_code=False
        self._setup_additional_cartesian_stability_code=False
        self._solve_in_arclength_conti=None
        self._adapt_eigenindex:Optional[int]=None # Which eigenvector to use during adaptation
        self._adapted_eigeninfo:Optional[List[Any]]=None # Store the eigenfunction, eigenvalue and m and k after adaptation
        self._last_arclength_parameter=None
        self._taken_already_an_unsteady_step=False
        self._last_step_was_stationary=None
        self._already_set_ic = False
        self._resetting_first_step=False
        self._in_transient_newton_solve=False
        
        self._hooks:List[GenericProblemHooks]=[]
        
        #: Flag indicating whether to call remeshing when necessary. Can be set to ``False`` to disable remeshing, e.g. when tracking bifurcations, it is better to check it manually invoking the remeshing method :py:meth:`remesh_handler_during_continuation` after each solve.
        self.do_call_remeshing_when_necessary:bool=True

        self.default_timestepping_scheme:Literal["BDF2","BDF1","Newmark2"]="BDF2"

        self.default_spatial_integration_order:Union[int,None] = None

        self._equation_system:EquationTree
        self._interinter_connections:Set[str]=set() # Interface/interface intersections, i.e. codimension 2+ intersections

        self.timestepper = _pyoomph.MultiTimeStepper(True)
        self.add_time_stepper_pt(self.timestepper)

        #: Set this to a (list of ) plotter(s) to automatically plot on :py:meth:`output` calls. If set to ``None``, no plotting will be done.
        self.plotter:Optional[Union[List["MatplotlibPlotter"],"MatplotlibPlotter"]]=None
        self.plot_in_dedicated_process:bool=False
        self._plotting_process:Optional[subprocess.Popen]=None
        self.latex_printer:Optional[_pyoomph.LaTeXPrinter]=None

        self.write_states:bool=True
        self.states_compression_level:Union[int,None]=6
        self.eigen_data_in_states:Union[int,bool]=False # Either True (all calced eigenvalues/vectors or a number to limit the number of stored eigendata)
        self.continuation_data_in_states:bool=False
        self.additional_equations:Union[Literal[0],"EquationTree"]=0

        self.default_1d_file_extension:Union[Literal["txt","mat"],List[Literal["txt","mat"]]]="txt"

        self.always_take_one_newton_step=True

        self._mesh_data_cache=MeshDataCacheStorage()
        self.eigenvector_position_scale:float=1 # if eigenmode="real" or "imag", we shift the positions multiplied with this factor (for "abs" or "angle") is is not done
        self._abort_current_run=False

        self._custom_assembler:Optional["CustomAssemblyBase"]=None

        self.default_ccode_expression_mode:str="" # Try to factor all expressions with "factor"
        self.extra_compiler_flags:List[str]=[]


        #: Must be set to the participant name when using preCICE. Default is an empty string, if you do not use preCICE.
        self.precice_participant:str=""
        #: Must be set to the config file when using preCICE
        self.precice_config_file:str=""
        self._precice_interface=None #type:ignore

    # Use weak(u,psi) instead of vectorial U*Psi for the symmetry-breaking constraint
    def improve_pitchfork_tracking_on_unstructured_meshes(self,coord_sys:"OptionalCoordinateSystem"=None,pos_coord_sys:"OptionalCoordinateSystem"=None):
        self._improved_pitchfork_tracking_on_unstructured_meshes=True
        self._improved_pitchfork_tracking_coordinate_system=coord_sys
        self._improved_pitchfork_tracking_position_coordinate_system=pos_coord_sys
        #self.enable_store_local_dof_pt_in_elements()

    def abort_current_run(self):
        """If called within a run(...) statement, e.g. from some action_{after/before}_* methods, the run will abort
        """
        self._abort_current_run=True

    def can_continue_section(self, id:Optional[str]=None) -> bool:
        if id is not None:
            raise RuntimeError("TODO: id for continue sections")
        if not self._initialised:
            self.initialise()

#		if self.write_states:
#			raise RuntimeError("Section grouping by can_continue_section works only with write_states=False")

        if self._runmode != "continue":

            statedir:str = os.path.join(self.get_output_directory(), "_states")
            Path(statedir).mkdir(parents=True, exist_ok=True)
            statefname:str = os.path.join(statedir, "state_{:06d}.dump".format(self._continue_section_step))
            self.save_state(statefname)
            self._continue_section_step += 1
            return False


        if self._continue_section_step_loaded>self._continue_section_step:
            self._continue_section_step += 1
            print("SKIPPING CONTINUE SECTION")
            return True
        else:
            return False
        
    # Shortcut to add a (GlobalLagrangeMultiplier(name=equation_contribution)+Scaling(name=scaling)+TestScaling(name=testscaling))@domain and return var(name,domain=domain),testfunction(name,domain=domain)
    def add_global_dof(self,name:str,equation_contribution:ExpressionOrNum=0,*,scaling:ExpressionNumOrNone=None,testscaling:ExpressionNumOrNone=None,domain:str="globals",only_for_stationary_solve:bool=False,initial_condition:ExpressionNumOrNone=None,set_zero_on_normal_mode_eigensolve:bool=True):
        """
        Add a global degree of freedom, e.g. a global Lagrange multiplier to the problem.

        Args:
            name (str): The name of the degree of freedom.
            equation_contribution (ExpressionOrNum): The global contribution of the degree of freedom to its equation. Defaults to 0.
            scaling (ExpressionNumOrNone, optional): The scaling factor for the degree of freedom. Defaults to None.
            testscaling (ExpressionNumOrNone, optional): The scaling factor for the test function. Defaults to None.
            domain (str, optional): The domain to which the degree of freedom belongs. Defaults to "globals".
            only_for_stationary_solve (bool, optional): Whether the degree of freedom is only used for stationary solves, if set, it will be 0 and pinned during transient solves. Defaults to False.
            initial_condition (ExpressionNumOrNone, optional): The initial condition for the degree of freedom. Defaults to None.
            set_zero_on_normal_mode_eigensolve: Deactivate this dof for normal mode eigensolves. Defaults to True.

        Returns:
            tuple: A tuple containing the variable and test function associated with the degree of freedom.
        """            
        from ..generic.codegen import GlobalLagrangeMultiplier,var,testfunction
        from ..equations.generic import Scaling,TestScaling,InitialCondition
        neweqs=GlobalLagrangeMultiplier(**{name:equation_contribution},only_for_stationary_solve=only_for_stationary_solve,set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve)
        if scaling is not None:
            neweqs+=Scaling(**{name:scaling})
        if testscaling is not None:
            neweqs+=TestScaling(**{name:testscaling})
        if initial_condition is not None:
            neweqs+=InitialCondition(**{name:initial_condition})
        self+=neweqs@domain
        return var(name,domain=domain),testfunction(name,domain=domain)


    def get_cached_mesh_data(self,msh:Union[str,AnySpatialMesh],nondimensional:bool=False,tesselate_tri:bool=True,eigenvector:Optional[Union[int,Sequence[int]]]=None,eigenmode:MeshDataEigenModes="abs",history_index:int=0,with_halos:bool=False,operator:Optional[MeshDataCacheOperatorBase]=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True) -> "MeshDataCacheEntry":
        """Return the current data (i.e. values) of a mesh. These are cached in case they are required multiple times, e.g. for plotting and output. 
        The cache is invalidated whenever we solve the problem or set some initial condition.  

        Args:
            msh (Union[str,AnySpatialMesh]): Mesh object or mesh name
            nondimensional (bool, optional): Getting nondimensional values instead of dimensional ones. Defaults to False.
            tesselate_tri (bool, optional): Split quad elements into tris. Helpful e.g. for plotting via matplotlib, which requires triangular meshes. Defaults to True.
            eigenvector (Optional[Union[int,Sequence[int]]], optional): If not None, we can obtain the values of the eigenfunction with the given index. Defaults to None.
            eigenmode (MeshDataEigenModes, optional): Since eigenfunctions are in general complex, we must select the desired projection to real numbers here in case of eigenvector!=None. Defaults to "abs".
            history_index (int, optional): Set to 1 or 2 to access the previous time steps. Defaults to 0.
            with_halos (bool, optional): Include halos to the output. Defaults to False.
            operator (Optional[MeshDataCacheOperatorBase], optional): Apply an operator on the cache, e.g. to add eigenvectors or extrude it in 3d. Defaults to None.

        Returns:
            MeshDataCacheEntry: The combined information of the mesh cache
        """
        
        if isinstance(msh,str):
            msh=self.get_mesh(msh)
        
        return self._mesh_data_cache.get_data(msh,nondimensional=nondimensional,tesselate_tri=tesselate_tri,eigenvector=eigenvector,eigenmode=eigenmode,history_index=history_index,with_halos=with_halos,operator=operator,discontinuous=discontinuous,add_eigen_to_mesh_positions=add_eigen_to_mesh_positions)

    def invalidate_cached_mesh_data(self,only_eigens:bool=False):
        """Mesh data is cached for potentially multiple usage (e.g. plotting and output to file). Whenever we change anything (e.g. changing values), we must hence invalidate the cache.

        Args:
            only_eigens (bool, optional): Only flush the cache of eigenfunctions, not of the current state. Defaults to False.
        """
        self._mesh_data_cache.clear(only_eigens)

    def set_tolerance_for_singular_jacobian(self,tol:float):
        _pyoomph.set_tolerance_for_singular_jacobian(tol)  
        
    def get_current_normal_mode_k(self,dimensional:bool=True):  
        if self._normal_mode_param_k is None:
            raise RuntimeError("No normal mode parameter k set. Please use setup_for_stability_analysis(additional_cartesian_mode=True) first.")
        if dimensional:
            return self._normal_mode_param_k.value/self.get_scaling("spatial")
        else:
            return self._normal_mode_param_k.value

    def add_equations(self,eqs:EquationTree)->None:
        """Add equations to the system. Should be called within the define_problem() method.

        Args:
            eqs (EquationTree): The equations (restricted to a domain) to add to the problem. 
        """
        if not isinstance(eqs,EquationTree): #type: ignore
            err=ValueError("Cannot add "+str(eqs)+' to the system. Equations need to be restricted via <equation> @ "<name in equation tree>"')
            eqs.add_exception_info(err)
        if not self._during_initialization:
            raise RuntimeError("You cannot use add_equations outside define_problem (or in functions called from there). Use additional_equations+=... instead, if you want to add something before initialization.")
        if not hasattr(self,"_equation_system") or self._equation_system is None:
            self._equation_system=eqs
        else:
            self._equation_system+=eqs

    def get_equations(self,path:str,error_if_not_found:bool=True)->Optional[BaseEquations]:
        """Return the equations added at the specified path.

        Args:
            path (str): Path to the domain
            error_if_not_found (bool, optional): Raise an error if the domain path is not valid. Defaults to True.

        Raises:
            RuntimeError: In case you specify a path where not equations are defined and error_if_not_found==True, you will get this error

        Returns:
            Optional[BaseEquations]: The equations at the desired domain path. If error_if_not_found==False, it is None if there are no equations at the desired path specified.
        """
        if self._equation_system is None:
            return None
        eqtree=self._equation_system.get_by_path(path)
        if eqtree is None:
            if error_if_not_found:
                raise RuntimeError("Cannot get equations at "+path)
            else:
                return None
        return eqtree._equations


    @overload
    def assemble_jacobian(self,with_residual:Literal[True]=...,which_one:str=...)->Tuple[List[float],DefaultMatrixType]: ...

    @overload
    def assemble_jacobian(self,with_residual:Literal[False],which_one:str)->DefaultMatrixType: ...
    
    def assemble_jacobian(self,with_residual:bool=True,which_one:str="")->Union[DefaultMatrixType,Tuple[List[float],DefaultMatrixType]]:
        res, n, _, _, J_values_arr, J_colindex_arr, J_row_start_arr=self._assemble_residual_jacobian(which_one)
        J = scipy.sparse.csr_matrix((J_values_arr, J_colindex_arr, J_row_start_arr), shape=(n, n)) #type:ignore
        if with_residual:
            return res,J #type:ignore
        else:
            return J

    def remove_equations(self, path:str, of_type:Optional[Type[BaseEquations]]=None, only_if:Callable[[BaseEquations],bool]=lambda eqn: True,fail_if_not_exist:bool=False):
        if hasattr(self,"_equation_system"):
            eqtree = self._equation_system.get_by_path(path)
        else:
            eqtree=self.additional_equations.get_by_path(path)
        if eqtree is None:
            if fail_if_not_exist:
                raise RuntimeError("No equations found at the path "+str(path))
            else:
                return
        eqs = eqtree._equations 
        if isinstance(eqs, CombinedEquations):
            if (of_type is None) or (isinstance(of_type, CombinedEquations)):
                if only_if(eqs):
                    eqtree._equations = DummyEquations() 
                    eqtree._equations._problem=self
            else:
                if of_type is None:
                    if only_if(eqs):
                        eqtree._equations = DummyEquations() 
                        eqtree._equations._problem=self
                else:
                    eqs._subelements = [e for e in eqs._subelements if not (isinstance(e, of_type) and only_if(e))] 
                    if len(eqs._subelements) == 0: 
                        eqtree._equations = DummyEquations() 
                        eqtree._equations._problem=self
        else:
            if (of_type is not None):
                if not isinstance(eqs, of_type):
                    return
            if eqs is not None and only_if(eqs):
                eqtree._equations = DummyEquations() 
                eqtree._equations._problem=self

    def get_default_timestepping_scheme(self,order:int) -> Literal['Newmark2', 'BDF2', 'BDF1']:
        if order==2:
            return "Newmark2"
        else:
            return self.default_timestepping_scheme

    def get_default_spatial_integration_order(self) -> int:
        if self.default_spatial_integration_order is None:
            return 0
        else:
            return self.default_spatial_integration_order

    #This must be used via "with problem.custom_adapt(): ..."
    def custom_adapt(self,skip_init_call:bool=False) -> _CustomAdaptWithHelper:
        return _CustomAdaptWithHelper(self,skip_init_call)
    
    def _change_output_directory(self,newdir:str):
        Path(newdir).mkdir(parents=True,exist_ok=True)
        self._equation_system._change_output_directory(newdir)
        if isinstance(self.plotter,(list,tuple)):
            for p in self.plotter:
                p._change_output_directory(newdir)
        elif self.plotter is not None:
            self.plotter._change_output_directory(newdir)
        

    def get_output_directory(self,relative_path:Optional[str]=None)->str:
        """Return the output directory of the problem. Set it with set_output_directory(). Otherwise, it will default to the name of the invoked script minus the extension .py.
        Optionally, you can add a relative path to assemble e.g. a file name within the output directory.

        Args:
            relative_path (Optional[str], optional): If set, we join this relative additional path to the output directory. Defaults to None.

        Returns:
            str: The output directory of the problem (potentially joined with the additionally passed relative_path)
        """
        if relative_path is not None:
            return os.path.join(self.get_output_directory(),relative_path)
        else:
            return self._outdir

    def set_output_directory(self,d:str)->None:
        """Change the output directory of the problem. Note: It should not be changed after the problem is initialised.

        Args:
            d: Output directory
        """
        self._outdir=d

    def has_named_var(self, name:str)->bool:
        return name in self._named_vars.keys()

    def get_named_var(self, name:str, default:Optional[ExpressionOrNum]=None)->ExpressionNumOrNone:
        return self._named_vars.get(name, default)

    def define_named_var(self, **kwargs:ExpressionOrNum):
        """Named vars are global expressions that are bound to a name. 
        
        You can e.g. use define_named_var(temperature=20*celsius) to define a temperature variable at problem level. When an equation tried to expand var("temperature") and no field "temperature" is defined on the current domain or its parents, the variable will be expanded by the global variable.  
        """
        for name,expr in kwargs.items():
            if not isinstance(expr,_pyoomph.Expression):
                expr=_pyoomph.Expression(expr)
            self._named_vars[name]=expr

    def __enter__(self):
        return self

    def release(self):
        def release_spatial_mesh(m:AnySpatialMesh):
            cg=m.get_code_gen()
            cg._code=None 
            cg._problem=None
            for im in m._interfacemeshes.values(): 
                release_spatial_mesh(im)
     
            m._interfacemeshes.clear() 
            m._eqtree._equations=None 
            m._eqtree=None #type:ignore

        for m in self._meshdict.values():
            if not isinstance(m,ODEStorageMesh):
                release_spatial_mesh(m)
            else:
                m._eqtree=None 
                m._element=None 

        self._lasolver:Optional[Union[str,GenericLinearSystemSolver]] = None
        self._eigensolver:Optional[Union[str,GenericEigenSolver]] = None
        self._meshtemplate_list = []
        self._meshdict:Dict[str,"AnyMesh"] = {}
        self.invalidate_cached_mesh_data()
        self.invalidate_eigendata()        
        self.flush_sub_meshes()
        self._unload_all_dlls()
        gc.collect()
        gc.collect()
        gc.collect()


    def __exit__(self, type, value, traceback): #type:ignore
        if isinstance(type,Exception):
            raise type
        else:
            self.release()


    @overload
    def get_mesh(self, name:str,return_None_if_not_found:Literal[False]=...)->AnySpatialMesh: ...

    @overload
    def get_mesh(self, name:str,return_None_if_not_found:Literal[True])->Optional[AnySpatialMesh]: ...

    def get_mesh(self, name:str,return_None_if_not_found:bool=False)->Optional[AnySpatialMesh]:
        """Get the mesh at the desired domain path. Invokes initialization if the problem is not initialised!

        Args:
            name (str): Domain path of the mesh
            return_None_if_not_found (bool, optional): If True, None will be returned if the given domain path is invalid. If False, an error will be raised in that case. Defaults to False.

        Raises:
            RuntimeError: Raised if there is no mesh at the given domain path in case of return_None_if_not_found==False (default). Same happens if an ODE domain is tried to be accessed like this. Use get_ode() for this.

        Returns:
            Optional[AnySpatialMesh]: The mesh at the domain path. None can only be returned if return_None_if_not_found==True and the domain path is invalid.
        """
        if not self._initialised:
            self.initialise()
        splt=name.split("/")
        if len(splt)==1:
            if return_None_if_not_found:
                res=self._meshdict.get(name,None)
                if isinstance(res,ODEStorageMesh):
                    return None
                else:
                    return res
            elif name in self._meshdict.keys():
                res=self._meshdict[name]
                if isinstance(res,ODEStorageMesh):
                    raise RuntimeError("There is an ODE, not a spatial Mesh at "+name+". So please use get_ode instead of get_mesh here")
                else:
                    return res
            else:
                raise RuntimeError("Cannot get mesh "+str(name)+", since it is not defined")
        else:
            msh=self._meshdict.get(splt[0],None)
            if msh is None:
                if return_None_if_not_found:
                    return None
                else:
                    raise RuntimeError("Cannot get mesh "+name+" since parent mesh "+splt[0]+" was not found")            
            if isinstance(msh,ODEStorageMesh):
                if return_None_if_not_found:
                    return None
                else:
                    raise RuntimeError("There is an ODE, not a spatial Mesh at "+name+". So please use get_ode instead of get_mesh here")
            if return_None_if_not_found:
                return msh.get_mesh("/".join(splt[1:]),return_None_if_not_found=True)
            else:
                return msh.get_mesh("/".join(splt[1:]),return_None_if_not_found=False)

    def get_ode(self,name:str)->ODEStorageMesh:
        """Return the ODE object at the given domain path. Invokes initialization if the problem is not initialised!

        Args:
            name (str): Domain path of the ODE

        Raises:
            RuntimeError: If the given domain path is invalid or a spatial mesh is defined at this domain, this error will occur.

        Returns:
            ODEStorageMesh: The ODE object at the given domain path
        """
        if not self._initialised:
            self.initialise()
        res=self._meshdict.get(name, None)
        if res is None:
            raise RuntimeError("No ODE domain with name "+str(name)+" in the system")
        if not isinstance(res,ODEStorageMesh):
            raise RuntimeError("You tried to get an ODE with name "+str(name)+", but apparently, this is not an ODE!")
        return res

    def get_all_values_at_current_time(self,with_pos:bool)->Tuple[NPFloatArray,List[bool],NPFloatArray]:
        dofs,positional_dof=self.get_current_dofs()
        pinned=self.get_current_pinned_values(with_pos)
        return numpy.array(dofs),positional_dof,numpy.array(pinned) #type:ignore

    def set_all_values_at_current_time(self,dofs:Union[NPFloatArray, List[float]],pinned:Union[NPFloatArray, List[float]],with_pos:bool):
        self.set_current_dofs(dofs) #type:ignore
        self.set_current_pinned_values(pinned,with_pos) #type:ignore


    def setup_pinned_values_of_eigenfunction(self,pv:NPFloatArray,n:int,mode:"MeshDataEigenModes")->NPFloatArray:	#Can be customised
        return 0.0*pv #type:ignore	 # Default: All pinned values are zero

    @overload
    def set_eigenfunction_as_dofs(self,n:int,*,mode:"MeshDataEigenModes"="abs",additive_mesh_positions:bool=True,perturb_amplitude:Literal[None]=...)->Tuple[NPFloatArray,NPFloatArray]: ...
    
    @overload
    def set_eigenfunction_as_dofs(self,n:int,*,mode:"MeshDataEigenModes"="abs",additive_mesh_positions:bool=True,perturb_amplitude:float)->Tuple[NPFloatArray,NPFloatArray,float]: ...

    def set_eigenfunction_as_dofs(self,n:int,*,mode:"MeshDataEigenModes"="abs",additive_mesh_positions:bool=True,eigenvector_position_scale:Optional[float]=None,perturb_amplitude:Optional[float]=None)->Union[Tuple[NPFloatArray,NPFloatArray,float],Tuple[NPFloatArray,NPFloatArray]]:
        if n>=len(self._last_eigenvectors):
            raise RuntimeError("Cannot set eigenfunction "+str(n)+" as dofs, since we have calculated only "+str(len(self._last_eigenvectors))+" eigenfunctions")
        with_pos=not additive_mesh_positions
        actual_dofs,positional_dofs,pinned_values=self.get_all_values_at_current_time(with_pos)
        if eigenvector_position_scale is None:
            eigenvector_position_scale=self.eigenvector_position_scale
        newpinned=self.setup_pinned_values_of_eigenfunction(numpy.array(pinned_values),n,mode) #type:ignore
        #print(newpinned)
        pert=self._last_eigenvectors[n]
        if len(pert)<len(actual_dofs):
            pert=numpy.pad(pert,(0,len(actual_dofs)-len(pert))) #type:ignore
        if mode=="abs":
            newdofs:NPFloatArray=numpy.absolute(pert)
        elif mode=="real":
            newdofs=numpy.real(pert) #type:ignore
        elif mode=="imag":
            newdofs=numpy.imag(pert) #type:ignore
        elif mode=="angle":
            newdofs=numpy.angle(pert) #type:ignore
        else:
            raise ValueError("Unknown eigenvector -> dof mode : "+str(mode))

        pos_indicator = numpy.array(positional_dofs, dtype="float64") #type:ignore
        if (mode=="real" or mode=="imag") and additive_mesh_positions:
            newdofs=newdofs*(1-pos_indicator)+pos_indicator*(eigenvector_position_scale*newdofs+actual_dofs) #type:ignore # Shift only in real or imag mode
        elif additive_mesh_positions:
            newdofs=newdofs*(1-pos_indicator)+pos_indicator*actual_dofs # Cannot shift in a good way here, take the old ones        
        aampl=1.0
        if perturb_amplitude is not None:
            if mode!="real" and mode!="imag":
                raise RuntimeError("Perturb mode only works in real or imag")
            aampl:float=numpy.amax(newdofs)-numpy.amin(newdofs) #type:ignore
            if aampl<1e-20:
                newdofs=actual_dofs
            else:
                newdofs=perturb_amplitude*newdofs/aampl+actual_dofs
            newpinned=pinned_values.copy()
            
        self.set_all_values_at_current_time(newdofs,newpinned,with_pos)
        if perturb_amplitude is not None:
            return actual_dofs, pinned_values,aampl #type:ignore
        else:
            return actual_dofs,pinned_values #type:ignore


    def get_coordinate_system(self) -> BaseCoordinateSystem:
        """
        Get the coordinate system set at problem level.

        Returns:
            BaseCoordinateSystem: The coordinate system at problem level.
        """
        return self._coordinate_system

    def set_coordinate_system(self,csys:Union[Literal["axisymmetric","axisymmetric_flipped","cartesian","radialsymmetric"],BaseCoordinateSystem]):                
        """Set the default coordinate system at problem level. 
        You can specify coordinate systems also at equation level, but if you don't do, the coordinate system will default to this one.

        Args:
            csys (Union[Literal["axisymmetric","axisymmetric_flipped","cartesian","radialsymmetric"],BaseCoordinateSystem]): The coordinate system to set as default.

        Raises:
            RuntimeError: Raised in case we do not set a valid coordinate system 
        """
        if csys is None:
            raise RuntimeError("Cannot set the problem coordinate system to None")
        if isinstance(csys,str):
            if csys=="axisymmetric":
                csys=axisymmetric
            elif csys=="axisymmetric_flipped":
                csys=axisymmetric_flipped
            elif csys=="cartesian":
                csys=cartesian
            elif csys=="radialsymmetric":
                csys=radialsymmetric
            else:
                raise RuntimeError("Unknown coordinate system: "+csys)

        self._coordinate_system=csys


    @overload
    def get_scaling(self,s:str,none_if_not_set:Literal[False]=...)->ExpressionOrNum: ...
    @overload
    def get_scaling(self,s:str,none_if_not_set:Literal[True])->ExpressionNumOrNone: ...

    def get_scaling(self,s:str,none_if_not_set:bool=False)->ExpressionNumOrNone:
        """
        Get the scaling factor for the problem variables for nondimensionalization.

        Args:
            s: Name of the scale to get.
            none_if_not_set: Returns None if this scaling is not set. Otherwise, the default scale 1 is returned. Defaults to ``False``.

        Returns:
            Scaling set by :py:meth:`~Problem.set_scaling` or None if ``none_if_not_set==True`` and the scale is not set.
        """
        scale=s
        while isinstance(scale,str):
            scale=self.scaling.get(scale,None if none_if_not_set else 1)
            if scale is None:
                return None
        if isinstance(scale,int) or isinstance(scale,float):
            scale=_pyoomph.Expression(scale)
        return scale

    def set_scaling(self,**kwargs:Union[ExpressionOrNum,str])->None:
        """
        Set the scaling factors for the problem variables for nondimensionalization.
        You can provide also scaling at equation level, but if not set there, it will ultimately default to the problem level scaling.
        Particular scales are ``"temporal"`` for the time and ``"spatial"`` for the spatial coordinates.

        Parameters:
            **kwargs: Keyword arguments specifying the scaling factors.
                The keys are the variable names, and the values are either numerical scaling factors
                or string expressions. In the latter case, we can set one scaling to another one, e.g.
                
                    ``set_scaling(u=1*meter/second,v="u")`` 
                
                would set the scaling of "v" to the one of "u"
        """            
        for k,v in kwargs.items():
            if type(v)==str:
                continue
            elif isinstance(v,_pyoomph.Expression):
                def merge_units(expr):
                    import _pyoomph
                    numfactor,unit,rest,success=_pyoomph.GiNaC_collect_units(expr)
                    if not success:
                        return expr
                    # Merge the unit once more
                    numfactor2,unit2,rest2,success2=_pyoomph.GiNaC_collect_units(unit)    
                    return numfactor*rest*numfactor2*rest2*unit2
                self.scaling[k]=merge_units(v).evalf()
            else:
                self.scaling[k]=v


        for k,v in kwargs.items():
            if type(v)!=str:
                continue
            self.scaling[k]=v


    def set_eigensolver(self,solv:Union[str,GenericEigenSolver]):
        """
        Set the eigensolver backend. "scipy", "pardiso", "slepc" are available (the latter two only if the packages MKL and/or petsc4py/slepc4py are installed)

        Returns:
            The eigenproblem solver instance after setting
        """
        if isinstance(solv,str):
            solv=GenericEigenSolver.factory_solver(solv,self)
        self._eigensolver=solv        
        print("EIGEN SOLVER WAS SET TO: "+self._eigensolver.idname)
        return self._eigensolver

    def set_linear_solver(self,solv:Union[str,GenericLinearSystemSolver]):
        
        """
        Set the linear solver backend. "scipy", "umfpack", "pardiso", "petsc" are available (the latter two only if the packages MKL and/or petsc4py are installed)

        Returns:
            The linear solver instance after setting
        """
        
        if isinstance(solv,str):
            solv=GenericLinearSystemSolver.factory_solver(solv,self)
        if self._num_threads is not None:
            solv.set_num_threads(self._num_threads)
        self._lasolver=solv        
        print("LINEAR SOLVER WAS SET TO: "+self._lasolver.idname)
        return self._lasolver

    def set_num_threads(self,nthread:Optional[int]):
        self._num_threads=nthread
        if self._lasolver is not None:
            if isinstance(self._lasolver,str):
                self.set_linear_solver(self._lasolver)
            else:
                self._lasolver.set_num_threads(self._num_threads)


    def get_eigen_solver(self)->GenericEigenSolver:
        """Get the eigenproblem solver instance.

        Returns:
            GenericEigenSolver: The currently used eigensolver
        """
        
        if self._eigensolver is None:
            self._eigensolver=get_default_eigen_solver()
        if isinstance(self._eigensolver,str):
            self._eigensolver=GenericEigenSolver.factory_solver(self._eigensolver,self)
        assert isinstance(self._eigensolver,GenericEigenSolver)
        return self._eigensolver


    def get_la_solver(self)->"GenericLinearSystemSolver":
        
        """Get the linear solver instance.

        Returns:
            GenericLinearSystemSolver: The currently used linear solver
        """
        
        if self._lasolver is None:
            self._lasolver=get_default_linear_solver()
        if isinstance(self._lasolver,str):
            self._lasolver=GenericLinearSystemSolver.factory_solver(self._lasolver,self)
        assert isinstance(self._lasolver,GenericLinearSystemSolver)
        return self._lasolver

    def _activate_solver_callback(self):
        _pyoomph.get_Solver_callback().set_problem(self) #type:ignore


    def is_initialised(self)->bool:
        """Returns whether the problem has been initialised or not.

        Returns:
            bool: True if already initialised, False otherwise
        """
        return self._initialised

    def output_at_increased_time(self,dt:Optional[ExpressionOrNum]=None)->None:
        """
        Increases the current time by the specified time step (dt, default scale_factor("temporal")) and calls the output method.
        Useful for Paraview PVD output of multiple stationary solutions, which otherwise overlays multiple outputs at the same time step.

        Args:
            dt (Optional[ExpressionOrNum]): The time step to increase the current time by. If not provided,
                the scaling factor for temporal is used.

        Returns:
            None
        """               
        if dt is None:
            dt=self.get_scaling("temporal")
        self.set_current_time(self.get_current_time()+dt)
        self.output()

    def perform_plot(self):
        if self._plotting_process is not None:
            raise RuntimeError("Should not end up here")
        if isinstance(self.plotter, list):
            for p in self.plotter:
                p._output_step = self._output_step
                if p.active:
                    if p._problem is None:
                        p._problem=self
                        p._named_problems[""]=self
                    p.plot()
        elif self.plotter is not None:
            self.plotter._output_step = self._output_step  
            if self.plotter.active:
                if self.plotter._problem is None:
                    self.plotter._problem=self
                    self.plotter._named_problems[""]=self                    
                self.plotter.plot()
                
                
    def create_eigendynamics_animation(self,outdir:str,plotter:"MatplotlibPlotter",eigenvector:int=0,init_amplitude:Optional[float]=None,max_amplitude:Optional[float]=None,numperiods:float=1,numouts:int=25,phi0:float=0):
        """
        Creates an animation of the eigenfunction dynamics. The eigenfunction is animated by varying the time and the amplitude of the eigenfunction, which is added to the degrees of freedom at each time.
        All images are saved in the specified output directory (relative to the output directory of the problem). The plotter is used to create the images. 
        Azimuthal instabilities will automatically mirror the eigenfunction to the left in the appropriate way.

        Args:
            outdir: Output directory for the animation images relative to the output directory of the problem.
            plotter: Plotter class to use for the animation.
            eigenvector: Optional index of the eigenfunction to animate. Defaults to 0.
            init_amplitude: Initial amplitude of the eigenperturbation. If this and ``max_amplitude`` is not provided, the amplitude is set to 1. Defaults to None.
            max_amplitude: Maximum amplitude of the eigenperturbation. If this is provided, the amplitude is set to this value at the beginning and decreases over time (eigenvalue has negative real part) or will reach this amplitude at the end of the considered time (eigenvalue has positive real part). Defaults to None.
            numperiods: Number of periods to animate. For purely real eigenvalues, the characteristic time is given by the real part. Defaults to 1.
            numouts: Number of output steps. Defaults to 25.
            phi0: Initial phase. Defaults to 0.
        
        """
        if len(self.get_last_eigenvalues())<eigenvector+1:
            raise RuntimeError("Eigenvalue/vector at index "+str(eigenvector)+" not calculated")
        eigenvalue=self.get_last_eigenvalues()[eigenvector]
        eigenfunction=self.get_last_eigenvectors()[eigenvector]
        olddofs,_=self.get_current_dofs()                
        
        phi0=float(phi0)
        if numouts<2:
            raise RuntimeError("Number of outputs must be at least 2")
        
        if plotter._problem is None:
            plotter._problem=self
            plotter._named_problems[""] = self
            
        # TODO: Backup here
        old_odir=plotter._output_dir
        old_outstep=plotter._output_step        
        plotter._output_dir=outdir
        plotter._output_step=0
        additional_factor_right=1
        additional_factor_left=1
        plotter._eigenanimation_m=0
        plotter._eigenanimation_lambda=eigenvalue
        if abs(numpy.imag(eigenvalue))>1e-7:
            inv_tperiod=abs(numpy.imag(eigenvalue))/(2*numpy.pi)
        else:
            inv_tperiod=abs(numpy.real(eigenvalue))/(2*numpy.pi)
            
        if init_amplitude is not None:
            if max_amplitude is not None:
                raise RuntimeError("Please specify either init_amplitude or max_amplitude, not both")
            amplitude=init_amplitude
        elif max_amplitude is not None:
            if numpy.real(eigenvalue)>0:
                amplitude=max_amplitude/numpy.exp(numpy.real(eigenvalue)/inv_tperiod*numperiods)
            else:
                amplitude=max_amplitude            
        else:
            amplitude=1        
        if self.get_last_eigenmodes_m() is not None and self.get_last_eigenmodes_m()[eigenvector]!=0:
            additional_factor_right=numpy.exp(1j*self.get_last_eigenmodes_m()[eigenvector]*phi0)
            additional_factor_left=numpy.exp(1j*self.get_last_eigenmodes_m()[eigenvector]*(phi0+numpy.pi))
            plotter._eigenanimation_m=self.get_last_eigenmodes_m()[eigenvector]
            
        plotter._eigenvector_for_animation=eigenfunction
        from pathlib import Path
        Path(os.path.join(self.get_output_directory(),outdir)).mkdir(parents=True, exist_ok=True)
        for i in range(numouts):
            t=numperiods/inv_tperiod*i/(numouts-1)
            print("Doing Eigenanimation:",i/(numouts-1)*100,r"% done")            
            #self.invalidate_cached_mesh_data()
            plotter._eigenfactor_right=additional_factor_right*amplitude*numpy.exp(eigenvalue*t)
            plotter._eigenfactor_left=additional_factor_left*amplitude*numpy.exp(eigenvalue*t)
            #self.set_current_dofs(olddofs+numpy.real(amplitude*eigenfunction*numpy.exp(eigenvalue*t)*additional_factor_right))            
            #self.invalidate_cached_mesh_data()
            plotter.plot()
            plotter._output_step+=1     
        plotter._output_dir=old_odir
        plotter._output_step=old_outstep           
        plotter._eigenfactor_right=None
        plotter._eigenfactor_left=None
        plotter._eigenvector_for_animation=None
        plotter._eigenanimation_m=None
        plotter._eigenanimation_lambda=None

    def _update_output_scales(self):
        for _n,m in self._meshdict.items():
            m._setup_output_scales()
            if not isinstance(m,ODEStorageMesh):
                def recu_interf(m):
                    for _in,im in m._interfacemeshes.items():
                        im._setup_output_scales()
                        recu_interf(im)
                recu_interf(m)


    def output(self, stage: str = "", quiet: Optional[bool] = None) -> None:
        """
        Invoke an output of the current solution at the current time by calling all Output objects.

        Args:
            stage (str): The stage of the output, at the moment, only "" is meaninfull.
            quiet (bool, optional): Flag to control the verbosity of the output.

        Returns:
            None
        """
        if not self.is_initialised():
            self.initialise()
        if quiet is None:
            quiet = self.is_quiet()
        if not quiet:
            paramstr = ""
            paramnames = [pn for pn in self.get_global_parameter_names() if not pn.startswith("_")]
            if len(paramnames) > 0:
                paramstr = ". Parameters: " + ", ".join([n + "=" + str(self.get_global_parameter(n).value) for n in paramnames])
            if not self.is_distributed():
                if get_mpi_rank() == 0:
                    print("OUTPUT at t=" + str(self.get_current_time()) + paramstr)
            else:
                print("OUTPUT of proc " + str(get_mpi_rank()) + " at t=" + str(self.get_current_time()) + paramstr)
                
        for hook in self._hooks:
            hook.actions_on_output(self._output_step)
        self._equation_system._do_output(self._output_step, stage)

        if self.write_states:
            statedir = os.path.join(self.get_output_directory(), "_states")
            Path(statedir).mkdir(parents=True, exist_ok=True)
            statefname = os.path.join(statedir, "state_{:06d}.dump".format(self._output_step))
            self.save_state(statefname)
            
        if self.plotter is not None:
            if self._plotting_process is None:
                self.perform_plot()


        if self._plotting_process is not None:
            if self._plotting_process.poll() is not None:
                raise RuntimeError("Plotting process failed. Have a look at " + self.get_output_directory("_dedicated_plotter_log.txt"))
            print("State file written, invoking plotting process")
            self._plotting_process.stdin.write((statefname + "\n").encode("utf-8"))
            self._plotting_process.stdin.flush()


            self._output_step += 1  # Write with the updated outstep here ??
        else:
            self._output_step += 1

        


    def init_output(self,redefined:bool=False):
        cinfo=None
        if redefined:
            cinfo={"redefined":True}
        if self._runmode=="continue":
            cinfo={"outstep":self._output_step,"dimtime":self.get_current_time(),"nondimtime":self.get_current_time(dimensional=False,as_float=True),"floattime":self.get_current_time(dimensional=True,as_float=False)}
        self._equation_system._init_output(continue_info=cinfo,rank=get_mpi_rank()) 


    def define_problem(self):
        """
        Define the problem by creating the mesh(es) and other necessary components.

        This method should be overridden by subclasses to define the specific problem.
        """
        pass
        #raise NotImplementedError("Please override the function define_problem to create the mesh(es) and so on")


    def flush_mesh_templates(self):
        self._meshtemplate_list=[]

    _TypeVarMeshTemplate=TypeVar("_TypeVarMeshTemplate",bound=MeshTemplate)
    def add_mesh(self,mesh:_TypeVarMeshTemplate)->_TypeVarMeshTemplate:
        """
        Adds a mesh to the problem. Based on the domain and boundary names of the mesh, equations can be added by using the same domain and boundary names.

        Args:
            mesh: Any mesh instance to be added (1d, 2d, 3d, etc.)

        Returns:
            Returns itself for chaining
        """
        self._meshtemplate_list.append(mesh)
        return mesh

    # Will be deprecated soon
    def add_mesh_template(self,mesh:_TypeVarMeshTemplate) -> _TypeVarMeshTemplate:
        """
        Same as self+=mesh or self.add_mesh(mesh). Will be deprecated soon.
        """
        return self.add_mesh(mesh)


   


    def relink_external_data(self):
        for ism in range(self.nsub_mesh()):
            submesh=self.mesh_pt(ism)
            if isinstance(submesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,ODEStorageMesh,InterfaceMesh)):
                assert submesh._codegen is not None 
                submesh._codegen._perform_external_ode_linkage() 
                #if not isinstance(submesh,ODEStorageMesh):
                #    assert isinstance(submesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh))
                submesh.ensure_external_data()


    def _adapt_with_interfacial_errors(self) -> Tuple[int, int]:
        
        #Resetting the element error override
        def reset(mesh:AnySpatialMesh):
            mesh._reset_elemental_error_max_override() 
            for _n,imesh in mesh._interfacemeshes.items():
                reset(imesh)
        for name,mesh in self._meshdict.items():
            if isinstance(mesh,ODEStorageMesh): continue
            reset(mesh)
            errs = mesh.get_elemental_errors()
            for i,b in enumerate(mesh.elements()):
                b._elemental_error_max_override=errs[i]
                
        if self._adapt_eigenindex is not None and self._last_eigenvectors is not None and len(self._last_eigenvectors)>self._adapt_eigenindex:
            _pyoomph.set_use_eigen_Z2_error_estimators(True)
            evect=self._last_eigenvectors[self._adapt_eigenindex]
            has_imag=numpy.amax(numpy.absolute(numpy.imag(evect)))>0.00001*numpy.amax(numpy.absolute(numpy.real(evect)))
            #print("EIG AS DOF")
            backup,backup_pinned=self.set_eigenfunction_as_dofs(self._adapt_eigenindex,mode="real")
            for name,mesh in self._meshdict.items():
                if isinstance(mesh,ODEStorageMesh): continue            
                #print("EVEM ",name)
                errs = mesh.get_elemental_errors()
                for i,b in enumerate(mesh.elements()):
                    b._elemental_error_max_override=max(errs[i],b._elemental_error_max_override)
            #print("DONE")
            if has_imag:
                #print("HAS IMAG",numpy.amax(numpy.absolute(numpy.imag(evect))),numpy.amax(numpy.absolute(numpy.real(evect))))
                self.set_eigenfunction_as_dofs(self._adapt_eigenindex,mode="imag")
                for name,mesh in self._meshdict.items():
                    if isinstance(mesh,ODEStorageMesh): continue            
                    errs = mesh.get_elemental_errors()
                    for i,b in enumerate(mesh.elements()):
                        b._elemental_error_max_override=max(errs[i],b._elemental_error_max_override)
            #print("DONE IMAG")
            #print(backup)
            #print(backup_pinned)
            self.set_all_values_at_current_time(backup,backup_pinned,False)
            _pyoomph.set_use_eigen_Z2_error_estimators(False)
            #print("RESET")
            

        if True:
            #Now, we first have to go through all meshes at the deepest level in the tree
            def get_errs(mesh:AnySpatialMesh,depth:int):
                if not mesh.refinement_possible():
                    return
                if depth==0:
                    #print("GET ERRS ON MESH",mesh,mesh.get_name())
                    assert mesh._codegen is not None
                    mesh._codegen.calculate_error_overrides() 
                elif depth>0:
                    for _,imesh in mesh._interfacemeshes.items(): 
                        get_errs(imesh,depth-1)
                        
            def override(mesh:AnySpatialMesh,depth:int):
                if not mesh.refinement_possible():
                    return
                if depth==0:
                    #print("OVERRIDE ON MESH",mesh,mesh.get_name())
                    if isinstance(mesh, InterfaceMesh):
                        mesh._override_bulk_errors_where_necessary() 
                elif depth>0:
                    for _n,imesh in mesh._interfacemeshes.items():
                        override(imesh,depth-1)

            for depth in reversed(range(3)):
                for name,mesh in self._meshdict.items():
                    if isinstance(mesh,ODEStorageMesh): continue
                    get_errs(mesh,depth)
            for depth in reversed(range(3)):
                for name,mesh in self._meshdict.items():
                    if isinstance(mesh,ODEStorageMesh): continue
                    override(mesh,depth)


            errs:Dict[str,List[float]]={}
            for name,mesh in self._meshdict.items():
                if isinstance(mesh,ODEStorageMesh): continue
                assert not isinstance(mesh,InterfaceMesh)
                errs[name]=[e._elemental_error_max_override for e in mesh.elements()]
                #errs[name]=mesh._merge_my_error_with_elemental_max_override()   # This is done in advance now
                

            # Ensure same refinement at connected interfaces

            for name,mesh in self._meshdict.items():
                if isinstance(mesh,ODEStorageMesh): continue
                for inam,imesh in mesh._interfacemeshes.items():                      
                    if imesh._opposite_interface_mesh is not None:  
                        if inam=="_internal_facets_":
                            raise RuntimeError("TODO: Adaption with internal facets")
                        # make sure that we override the mesh errors correctly

                        obm=imesh._opposite_interface_mesh._parent  
                        assert obm is not None and not isinstance(obm,InterfaceMesh)
                        my_maxerr=mesh.max_permitted_error
                        my_minerr = mesh.min_permitted_error
                        opp_maxerr = obm.max_permitted_error
                        opp_minerr = obm.min_permitted_error
                        for ie in imesh.elements():
                            be=ie.get_bulk_element()
                            obe=ie.get_opposite_bulk_element()
                            if obe._elemental_error_max_override<opp_minerr and be._elemental_error_max_override>=my_minerr:
                                obe._elemental_error_max_override=0.5*(opp_minerr+opp_maxerr)
                            elif obe._elemental_error_max_override>=opp_minerr and be._elemental_error_max_override<my_minerr:
                                be._elemental_error_max_override=0.5*(my_minerr+my_maxerr)
                        for i,e in enumerate(mesh.elements()):
                            errs[name][i]=max(errs[name][i],e._elemental_error_max_override)
                        for i,e in enumerate(obm.elements()):
                            errs[obm._name][i]=max(errs[obm._name][i],e._elemental_error_max_override) 


        messed_around_in_history=False
        has_arclength_data=False
        # TODO: Ensure same refinement at connected interfaces
        if self._last_arclength_parameter is not None:
            dof_deriv=self.get_arclength_dof_derivative_vector()
            if len(dof_deriv)>0:
                has_arclength_data=True
                _actual_dofs,_positional_dofs,pinned_values=self.get_all_values_at_current_time(True)            
                dof_current=self.get_arclength_dof_current_vector()
                self.set_current_pinned_values(0*pinned_values,True,5)
                self.set_current_pinned_values(0*pinned_values,True,6)
                self.set_history_dofs(5,dof_deriv)
                self.set_history_dofs(6,dof_current)
                messed_around_in_history=True
            
        if self._adapt_eigenindex is not None:
            _actual_dofs,_positional_dofs,pinned_values=self.get_all_values_at_current_time(True)
            self.set_current_pinned_values(0*pinned_values,True,3)
            self.set_current_pinned_values(0*pinned_values,True,4)
            self.set_history_dofs(3,numpy.real(self._last_eigenvectors[self._adapt_eigenindex]))
            self.set_history_dofs(4,numpy.imag(self._last_eigenvectors[self._adapt_eigenindex]))
            messed_around_in_history=True        

        nref=0
        nuref=0
        with self.custom_adapt(True):
            for name,errors in errs.items():
                mesh=self.get_mesh(name)
                if mesh.refinement_possible():
                    mesh.adapt_by_elemental_errors(errors)
                    if not self.is_quiet():
                        print("IN MESH "+name+" ref=",mesh.nrefined(),"unref=",mesh.nunrefined())
                    nref += mesh.nrefined()
                    nuref += mesh.nunrefined()
                    
        if has_arclength_data:
            dof_deriv=self.get_history_dofs(5)
            dof_current=self.get_history_dofs(6)
            self._update_dof_vectors_for_continuation(dof_deriv,dof_current)
            
        if self._adapt_eigenindex is not None:
            eigfunc=self.get_history_dofs(3)+1j*self.get_history_dofs(4)
            self._last_eigenvectors=[eigfunc]
            self._last_eigenvalues=[self._last_eigenvalues[self._adapt_eigenindex]]
            lastm,lastk=None,None
            if self._last_eigenvalues_m is not None:
                self._last_eigenvalues_m=[self._last_eigenvalues_m[self._adapt_eigenindex]]
                lastm=self._last_eigenvalues_m[0]
            if self._last_eigenvalues_k is not None:
                self._last_eigenvalues_k=[self._last_eigenvalues_k[self._adapt_eigenindex]]
                lastk=self._last_eigenvalues_k[0]
            self._adapted_eigeninfo=[eigfunc,self._last_eigenvalues[0],lastm,lastk]
            
        if messed_around_in_history:
            self.assign_initial_values_impulsive() # We messed around. So me must reassign the initial values
            
        return nref,nuref

    def _adapt(self) -> Tuple[int, int]:
        nref,nunref=self._adapt_with_interfacial_errors()
        return nref,nunref

    def compile_meshes(self):
        for _,mesh in self._meshdict.items():
            if isinstance(mesh,ODEStorageMesh):
                mesh._compile_bulk_equations() 
        for _,mesh in self._meshdict.items():
            if not isinstance(mesh,ODEStorageMesh):
                assert not isinstance(mesh,InterfaceMesh)
                mesh._compile_bulk_equations() 
        #Now all bulks are compiled
        #We now must add the interior facets contributions, if set
        for _,mesh in self._meshdict.items():
            if not isinstance(mesh,ODEStorageMesh):
                assert not isinstance(mesh,InterfaceMesh)
                has_interior_contribs=False
                eqs=mesh._eqtree.get_equations()
                for _,int_contrib in eqs._interior_facet_residuals.items():
                    if not is_zero(int_contrib):
                        has_interior_contribs=True
                        break
                if has_interior_contribs:
                    # Check if we already have an _interior_facets_ domain there
                    if "_internal_facets_" not in mesh._eqtree.get_children().keys():
                        raise RuntimeError("TODO: Auto create _interior_facets_ domain. For the time being, you have to add it by hand (just use eqs+=Equations()@'_internal_facets_').\nOr, even better, set self.requires_interior_facet_terms=True in the __init__ of the Equations class you add interior facet contributions.")
                        facetdom=EquationTree(InterfaceEquations(),parent=mesh._eqtree)                                                
                        mesh._eqtree._children["_internal_facets_"]=facetdom                        
                        facetdom._finalize_equations(mesh.get_problem())                        
                        facetmesh=InterfaceMesh(mesh.get_problem(),mesh,"_internal_facets_",facetdom)                                                
                        mesh._interfacemeshes["_internal_facets_"]=facetmesh
                        facetdom._mesh=mesh._interfacemeshes["_internal_facets_"]
                        facetdom.get_code_gen()._set_bulk_element(mesh._eqtree.get_code_gen())
                        cg_b=mesh._eqtree.get_code_gen()
                        nodal_dim=cg_b.get_nodal_dimension()                        
                        while nodal_dim==0 and cg_b.get_parent_domain() is not None:
                            cg_b=cg_b.get_parent_domain()                    
                        nodal_dim=cg_b.get_nodal_dimension()
                        facetdom.get_code_gen()._set_nodal_dimension(nodal_dim)                        
                        facetdom.get_code_gen()._do_define_fields(mesh._eqtree.get_code_gen().get_element_dimension()-1)
                        facetdom._create_dummy_domains_for_DG(mesh.get_problem())
                        
                        
                    internal_eqs=mesh._eqtree.get_child("_internal_facets_").get_equations()
                    for destination,int_contrib in eqs._interior_facet_residuals.items():
                        if destination in internal_eqs._additional_residuals.keys():
                            internal_eqs._additional_residuals[destination]+=int_contrib
                        else:
                            internal_eqs._additional_residuals[destination]=int_contrib


        for tree_depth in range(2):
            for _,mesh in self._meshdict.items():
                mesh._problem=self
                mesh._eqtree._equations.get_combined_equations()._problem=self
                if isinstance(mesh,ODEStorageMesh): continue
                mesh._pre_compile_interface_equations(tree_depth) 

            for _,mesh in self._meshdict.items():
                if isinstance(mesh,ODEStorageMesh): continue
                mesh._compile_interface_equations(tree_depth) 

            for _,mesh in self._meshdict.items():
                if isinstance(mesh,ODEStorageMesh): continue
                mesh._generate_interface_elements(tree_depth) 

        for _,mesh in self._meshdict.items():
            if isinstance(mesh, ODEStorageMesh): continue
            assert not isinstance(mesh,InterfaceMesh)
            mesh._link_periodic_corner_nodes()  

    

    def before_compile_equations(self, eqs: BaseEquations):
        eqs.get_current_code_generator().use_shared_shape_buffer_during_multi_assemble=self._shared_shapes_for_multi_assemble
        if self._improved_pitchfork_tracking_on_unstructured_meshes:
            for fn,_space in eqs._fields_defined_on_my_domain.items():
                u=nondim(fn,tag=["flag:only_base_mode"]) 
                utest=testfunction(fn,dimensional=False)
                # This will give a nice mass matrix! The Jacobian will be J_lk=psi^l*psi^k*dx                
                eqs.add_residual(weak(u,utest,coordinate_system=self._improved_pitchfork_tracking_coordinate_system),destination="_simple_mass_matrix_of_defined_fields")
            if eqs.get_current_code_generator()._coordinates_as_dofs and (eqs.get_parent_domain() is None) : # Only accumulate on the moving bulk domain
                u=nondim("mesh",tag=["flag:only_base_mode"]) 
                utest=testfunction("mesh",dimensional=False)
                cs=self._improved_pitchfork_tracking_coordinate_system
                if self._improved_pitchfork_tracking_position_coordinate_system:
                    cs=self._improved_pitchfork_tracking_position_coordinate_system
                eqs.add_residual(weak(u,utest,coordinate_system=cs),destination="_simple_mass_matrix_of_defined_fields")
            # Residuals not writtten to C, wont be used
            eqs.get_current_code_generator().set_ignore_residual_assembly("_simple_mass_matrix_of_defined_fields")
            
        # We cannot write the residuals for the normal modes, since e.g. some eigenexpansions are there in linear, which cannot be calculated in the C code
        # We must suppress the generation of the residual code and only add Jacobian code, where all these terms will vanish
        if self._azimuthal_mode_param_m is not None:
            eqs.get_current_code_generator().set_ignore_residual_assembly(self._azimuthal_stability.real_contribution_name)
            eqs.get_current_code_generator().set_ignore_residual_assembly(self._azimuthal_stability.imag_contribution_name)
            eqs.get_current_code_generator().set_derive_jacobian_by_expansion_mode(self._azimuthal_stability.real_contribution_name,1)
            eqs.get_current_code_generator().set_derive_jacobian_by_expansion_mode(self._azimuthal_stability.imag_contribution_name,1)
            eqs.get_current_code_generator().set_ignore_dpsi_coord_diffs_in_jacobian(self._azimuthal_stability.real_contribution_name)
            eqs.get_current_code_generator().set_ignore_dpsi_coord_diffs_in_jacobian(self._azimuthal_stability.imag_contribution_name)
            eqs.get_current_code_generator().set_derive_hessian_by_expansion_mode(self._azimuthal_stability.real_contribution_name,0)
            eqs.get_current_code_generator().set_derive_hessian_by_expansion_mode(self._azimuthal_stability.imag_contribution_name,0)
        if self._normal_mode_param_k is not None:
            eqs.get_current_code_generator().set_ignore_residual_assembly(self._cartesian_normal_mode_stability.real_contribution_name)
            eqs.get_current_code_generator().set_ignore_residual_assembly(self._cartesian_normal_mode_stability.imag_contribution_name)
            eqs.get_current_code_generator().set_derive_jacobian_by_expansion_mode(self._cartesian_normal_mode_stability.real_contribution_name,1)
            eqs.get_current_code_generator().set_derive_jacobian_by_expansion_mode(self._cartesian_normal_mode_stability.imag_contribution_name,1)
            eqs.get_current_code_generator().set_ignore_dpsi_coord_diffs_in_jacobian(self._cartesian_normal_mode_stability.real_contribution_name)
            eqs.get_current_code_generator().set_ignore_dpsi_coord_diffs_in_jacobian(self._cartesian_normal_mode_stability.imag_contribution_name)
            eqs.get_current_code_generator().set_derive_hessian_by_expansion_mode(self._cartesian_normal_mode_stability.real_contribution_name,0)
            eqs.get_current_code_generator().set_derive_hessian_by_expansion_mode(self._cartesian_normal_mode_stability.imag_contribution_name,0)
            #eqs.get_current_code_generator().set_remove_underived_modes(self._cartesian_normal_mode_stability.real_contribution_name,set([1]))
            #eqs.get_current_code_generator().set_remove_underived_modes(self._cartesian_normal_mode_stability.imag_contribution_name,set([1]))

    def set_custom_assembler(self,assm:Optional["CustomAssemblyBase"]) -> None:
        if self._custom_assembler:
            self._custom_assembler.finalize()
            
        self._custom_assembler=assm
        if self._custom_assembler:        
            self.use_custom_residual_jacobian=True
            self._custom_assembler._set_problem(self)
            self._custom_assembler.initialize()
        else:
            self.use_custom_residual_jacobian=False

    def get_custom_assembler(self) -> Optional["CustomAssemblyBase"]:
        return self._custom_assembler
    

    def get_custom_residuals_jacobian(self, info:_pyoomph.CustomResJacInfo) -> None:
        if self._custom_assembler is None:
            raise RuntimeError("If you set use_custom_residual_jacobian=True, you must specify a custom assembler or override get_custom_residuals_jacobian yourself")
        if info.require_jacobian():
            if info.get_parameter_name()!="":
                raise RuntimeError("Cannot derive custom Jacobian with respect to a parameter yet")
            res,J=self._custom_assembler.get_residuals_and_jacobian(True)
            assert res.dtype==numpy.float64, "Expected float residuals, but got "+str(res.dtype) #type:ignore
            info.set_custom_residuals(res)
            assert J.indptr.dtype==numpy.int32 and J.indices.dtype==numpy.int32 and J.data.dtype==numpy.float64 #type:ignore
            info.set_custom_jacobian(J.data,J.indices,J.indptr) #type:ignore
        else:
            paramname=info.get_parameter_name()
            if paramname=="":
                paramname=None
            res=self._custom_assembler.get_residuals_and_jacobian(False,paramname)
            assert res.dtype==numpy.float64 #type:ignore
            info.set_custom_residuals(res)

    @overload
    def set_c_compiler(self,compiler_or_name:Literal["tcc"])->_pyoomph.CCompiler: ...

    @overload
    def set_c_compiler(self,compiler_or_name:Literal["system"])->"SystemCCompiler": ...

    def set_c_compiler(self,compiler_or_name:Union[str,BaseCCompiler])->Union[_pyoomph.CCompiler,BaseCCompiler]:
        """
        Selects the C compiler for the problem. 
        "tcc" is fast in compilation, but slower in execution. Good for setting up a problem class.
        "system" is slower in compilation, but faster in execution. Good for running the final problem.
        set_c_compiler("system").optimize_for_max_speed() makes it even faster by using compiler flags for maximum speed.

        Args:
            compiler_or_name (Union[str, BaseCCompiler]): The C compiler to use ("tcc" or "system" at the moment).

        Returns:
            Union[_pyoomph.CCompiler, BaseCCompiler]: The C compiler that was set.
        """      
                
        from .ccompiler import get_ccompiler
        if isinstance(compiler_or_name,str):
            if compiler_or_name=="tcc":
                if _pyoomph.has_tcc():
                    compiler_or_name="_internal_"
                else:
                    compiler_or_name="tccbox"
            elif compiler_or_name=="distutils":
                compiler_or_name="system"
            cc=get_ccompiler(compiler_or_name)
        else:
            cc=compiler_or_name
        self._set_ccompiler(cc)
        return self.get_ccompiler()


    def __iadd__(self,other:Union[MeshTemplate,EquationTree,GenericProblemHooks,"MatplotlibPlotter"]):
        from pyoomph.output.plotting import BasePlotter
        if self._initialised and not isinstance(other,(BasePlotter,GenericProblemHooks)):
            raise RuntimeError("Cannot add anything to a problem once it is initialized!")
        if isinstance(other,MeshTemplate):
            self.add_mesh(other)
        elif isinstance(other,EquationTree):
            self.additional_equations+=other
        elif isinstance(other,BasePlotter):
            if self.plotter is None:
                self.plotter=other
            elif isinstance(self.plotter,list):
                self.plotter.append(other)
            else:
                self.plotter=[self.plotter,other]
        elif isinstance(other,GenericProblemHooks):
            if other._problem is None:
                other._problem=self
            elif other._problem is not self:
                raise RuntimeError("Cannot add a problem hook to a different problem")
            self._hooks.append(other)
        else:
            addinfo=""
            if isinstance(other,BaseEquations):
                addinfo="  -- You must restrict equations to a domain using e.g. @'domain'"

            raise RuntimeError("cannot add this to a Problem: " +str(other)+addinfo)
        return self

    def cmdline_desc(self) -> str:
        return "Generic Pyoomph Problem"

    def setup_cmd_line(self):              
        self.cmdlineparser = argparse.ArgumentParser(description=self.cmdline_desc())        
        self.cmdlineparser.add_argument('--petsc',help="use PETSc solver",action='store_true')
        self.cmdlineparser.add_argument('--pastix',help="use PaSTiX solver",action='store_true')
        self.cmdlineparser.add_argument('--superlu',help="use serial SuperLu solver",action='store_true')
        self.cmdlineparser.add_argument('--umfpack', help="use UMFPACK solver", action='store_true')
        self.cmdlineparser.add_argument('--pardiso', help="use Pardiso solver", action='store_true')
        self.cmdlineparser.add_argument('--mumps', help="use MUMPS solver", action='store_true')
        self.cmdlineparser.add_argument('--slepc',help="use SLEPc as eigensolver. Specify your own backend for the matrix inversion during eigensolve here",action="store_true")
        self.cmdlineparser.add_argument('--slepc_mumps',help="use SLEPc as eigensolver with MUMPS as backend",action="store_true")        
        self.cmdlineparser.add_argument('--petsc_mumps',help="use PETSc as linear solver with MUMPS as backend",action="store_true")                
        self.cmdlineparser.add_argument('--arpack',action="store_true")
        self.cmdlineparser.add_argument('--tcc', help="use internal TCC compiler", action='store_true')
        self.cmdlineparser.add_argument('--distutils', help="use system C compiler detected by distutils", action='store_true')
        self.cmdlineparser.add_argument('--fast-math', help="activate fast math compiler flags (only with distutils, not with tcc)", action='store_true')        
        self.cmdlineparser.add_argument('--distribute',help="Distribute mesh in parallel",action='store_true')
        self.cmdlineparser.add_argument('--outdir', help="output directory",type=str)
        self.cmdlineparser.add_argument('--suppress_code_writing',help="do not write FEM codes. Useful for debugging",action='store_true')
        self.cmdlineparser.add_argument('--suppress_compilation',help="do not compile FEM codes. Useful for debugging",action='store_true')
        self.cmdlineparser.add_argument('-P','--parameter', help="Override some problem parameters",nargs='+', type=str)
        self.cmdlineparser.add_argument("--runmode",help="Selects the runmode ([d]elete and run, [o]verride and run, [c]ontinue, [p]lot again",type=str)
        self.cmdlineparser.add_argument("--recompile_on_continue",help="When using --runmode c, compilation and code writing is usually suppressed. You can recompile the code anyhow with this flag",action="store_true")
        self.cmdlineparser.add_argument("--verbose",help="Gives a lot of output",action='store_true')
        self.cmdlineparser.add_argument("--where",help="Python bool expression involving variables time or step. Only used in runmodes c and p",type=str,default="True")
        self.cmdlineparser.add_argument("--largest_residuals",help="Debug the largest residuals",type=int,default=self._debug_largest_residual)
        self.cmdlineparser.add_argument("--generate_precice_cfg",help="Generate some parts of a preCICE configuration file from the coupling equations",action="store_true")

    def parse_cmd_line(self):
        from ..materials.generic import MaterialProperties
        if self.ignore_command_line:
            self.cmdlineargs, self.further_cmdlineargs = self.cmdlineparser.parse_known_args(args="")
        else:
            self.cmdlineargs, self.further_cmdlineargs = self.cmdlineparser.parse_known_args()
        if self.cmdlineargs.superlu:
            if self.cmdlineargs.petsc or self.cmdlineargs.pastix or self.cmdlineargs.umfpack or self.cmdlineargs.pardiso or self.cmdlineargs.mumps or self.cmdlineargs.petsc_mumps:
                raise ValueError("Cannot set two solvers simultaneously")
            self.set_linear_solver("superlu")
        elif self.cmdlineargs.petsc:
            if self.cmdlineargs.superlu or self.cmdlineargs.pastix or self.cmdlineargs.umfpack or self.cmdlineargs.pardiso or self.cmdlineargs.mumps or self.cmdlineargs.petsc_mumps:
                raise ValueError("Cannot set two solvers simultaneously")
                import pyoomph.solvers.petsc
            self.set_linear_solver("petsc")
        elif self.cmdlineargs.umfpack:
            if self.cmdlineargs.petsc or self.cmdlineargs.pastix or self.cmdlineargs.superlu or self.cmdlineargs.pardiso or self.cmdlineargs.mumps or self.cmdlineargs.petsc_mumps:
                raise ValueError("Cannot set two solvers simultaneously")
            self.set_linear_solver("umfpack")
        elif self.cmdlineargs.pardiso:
            if self.cmdlineargs.petsc  or self.cmdlineargs.pastix or self.cmdlineargs.superlu or self.cmdlineargs.umfpack or self.cmdlineargs.mumps or self.cmdlineargs.petsc_mumps:
                raise ValueError("Cannot set two solvers simultaneously")
            self.set_linear_solver("pardiso")
        elif self.cmdlineargs.mumps:
            if self.cmdlineargs.petsc or self.cmdlineargs.pastix or self.cmdlineargs.superlu or self.cmdlineargs.umfpack or self.cmdlineargs.pardiso or self.cmdlineargs.petsc_mumps:
                raise ValueError("Cannot set two solvers simultaneously")
            self.set_linear_solver("mumps")
        elif self.cmdlineargs.pastix:
            if self.cmdlineargs.mumps or self.cmdlineargs.petsc or self.cmdlineargs.superlu or self.cmdlineargs.umfpack or self.cmdlineargs.pardiso or self.cmdlineargs.petsc_mumps:
                raise ValueError("Cannot set two solvers simultaneously")
            self.set_linear_solver("pastix")
        elif self.cmdlineargs.petsc_mumps:
            if self.cmdlineargs.mumps or self.cmdlineargs.petsc or self.cmdlineargs.superlu or self.cmdlineargs.umfpack or self.cmdlineargs.pardiso or self.cmdlineargs.pastix:
                raise ValueError("Cannot set two solvers simultaneously")
            self.set_linear_solver("petsc").use_mumps()

        if self.cmdlineargs.tcc:
            if self.cmdlineargs.distutils:
                raise RuntimeError("Cannot set --tcc and --distutils together")
            self.set_c_compiler("tcc")
            if self.cmdlineargs.fast_math:
                raise RuntimeError("Cannot use --fast-math with --tcc")
        elif self.cmdlineargs.distutils:
            ccomp=self.set_c_compiler("system")
            if self.cmdlineargs.fast_math:
                ccomp.optimize_for_max_speed()
        elif self.cmdlineargs.fast_math:
            self.set_c_compiler("system").optimize_for_max_speed()

        if self.cmdlineargs.arpack:
            if self.cmdlineargs.slepc or self.cmdlineargs.slepc_mumps:
                raise RuntimeError("Cannot be used together: --arpack and --slepc/--slepc_mumps")
            self.set_eigensolver("scipy") # Not using the pardiso arpack then
        elif self.cmdlineargs.slepc or self.cmdlineargs.slepc_mumps:
            import pyoomph.solvers.petsc
            self.set_eigensolver("slepc")
            if self.cmdlineargs.slepc_mumps:
                eigsolv=self.get_eigen_solver()
                assert isinstance(eigsolv,pyoomph.solvers.petsc.SlepcEigenSolver)
                eigsolv.use_mumps()
                


        if self.cmdlineargs.outdir:
            self._outdir=self.cmdlineargs.outdir

        if self.cmdlineargs.suppress_code_writing:
            self._suppress_code_writing=True
            
        if self.cmdlineargs.suppress_compilation:
            self._suppress_compilation=True

        if self.cmdlineargs.verbose:
            _pyoomph.set_verbosity_flag(1)

        possible_runmodes=["continue","delete","overwrite","replot"]
        if self.cmdlineargs.runmode=="c" or self.cmdlineargs.runmode=="continue":
            self._runmode="continue"
        elif self.cmdlineargs.runmode=="d" or self.cmdlineargs.runmode=="delete":
            self._runmode="delete"
        elif self.cmdlineargs.runmode=="o" or self.cmdlineargs.runmode=="overwrite":
            self._runmode="overwrite"
        elif self.cmdlineargs.runmode=="p" or self.cmdlineargs.runmode=="replot":
            self._runmode="replot"
        elif self.cmdlineargs.runmode is not None:
            raise RuntimeError("Unknown runmode "+self.cmdlineargs.runmode+". Possible are "+", ".join(possible_runmodes))

        if not self._runmode in possible_runmodes:
            raise RuntimeError(
                "Unknown runmode " + self._runmode + ". Possible are " + ", ".join(possible_runmodes))

        if self._runmode=="continue" or self._runmode=="replot":
            self._suppress_code_writing=(not self.cmdlineargs.recompile_on_continue) or self._runmode=="replot"
            self._suppress_compilation=(not self.cmdlineargs.recompile_on_continue) or self._runmode=="replot"

        self._where_expression=self.cmdlineargs.where

        self._debug_largest_residual=self.cmdlineargs.largest_residuals
        if self._debug_largest_residual>0:
            self.enable_store_local_dof_pt_in_elements()

        if self.cmdlineargs.parameter is not None:
            for cmdset in self.cmdlineargs.parameter:

                splt=cmdset.split("=")
                varname=splt[0]
                mode="="
                if varname.endswith("*"):
                    mode="*"
                elif varname.endswith("/"):
                    mode="/"
                elif varname.endswith("+"):
                    mode="+"
                elif varname.endswith("-"):
                    mode="-"
                if mode!="=":
                    varname=varname[0:-1]
                val="=".join(splt[1:])
                splt_varname=varname.split(".")
                obj:Any=self
                current=None
                for i,v in enumerate(splt_varname[:-1]):
                    if isinstance(obj,dict) and not isinstance(obj,Problem):
                        if v in obj.keys():
                            obj=obj.get(v) #type:ignore
                        else:
                            found_in_dict_by_name=False
                            for dict_entry in obj.keys(): #type:ignore
                                print("CHECKING",dict_entry)  #type:ignore
                                if hasattr(dict_entry,"name") and getattr(dict_entry,"name")==v: #type:ignore
                                    if found_in_dict_by_name:
                                        raise RuntimeError("Found two dict key entries with property name == '"+str(v)+"' in "+self.__class__.__name__ + "." + ".".join(splt_varname[:i + 1]))
                                    found_in_dict_by_name=True
                                    obj=dict_entry

                            if not found_in_dict_by_name:
                                raise RuntimeError("Cannot set parameter " + varname + " due to undefined property " + self.__class__.__name__ + "." + ".".join(splt_varname[:i + 1]))
                    else:
                        try:
                            obj=getattr(obj,v)
                        except:
                            raise RuntimeError("Cannot set parameter "+varname+" due to undefined property "+self.__class__.__name__+"."+".".join(splt_varname[:i+1]))

                if isinstance(obj,dict):
                    current = obj[splt_varname[-1]] #type:ignore
                elif hasattr(obj,splt_varname[-1]):
                    current=getattr(obj,splt_varname[-1])
                elif isinstance(obj,Problem) and splt_varname[-1] in obj.get_global_parameter_names():
                    current=obj.get_global_parameter(splt_varname[-1])
                else:
                    raise RuntimeError("Cannot set undefined property/parameter "+".".join(splt_varname)+". Currently at "+str(obj)+" and trying to access "+str(splt_varname[-1])+"\nFollowing properties are known:"+str(dir(obj)))
                
                if not self.is_quiet():
                    print("SETTING PARAMETER", varname,"FROM",current, "TO",val) #type:ignore
                #TODO: Complete this
                if isinstance(current,int):
                    if isinstance(current,bool):
                        if val=="True":
                            newvalue=True
                        elif val=="False":
                            newvalue=False
                        else:
                            raise RuntimeError("Cannot set the bool property "+varname+" to "+str(val))
                    else:
                        try:
                            newvalue=int(val)
                        except ValueError:
                            try:
                                newvalue=float(val)
                            except ValueError:
                                raise RuntimeError("Cannot set the integer property "+varname+" to "+str(val))
                elif isinstance(current,float):
                    try:
                        newvalue = float(val)
                    except ValueError:
                        raise RuntimeError("Cannot set the float property " + varname + " to " + str(val))
                elif isinstance(current,_pyoomph.Expression):
                    Pi=pi #type:ignore
                    try:
                        newvalue=eval(val)
                    except Exception as e:
                        raise RuntimeError("Cannot set the property " + varname + " to " + str(val)+"\n"+str(e))
                elif isinstance(current,str):
                    newvalue=val
                elif isinstance(current,_pyoomph.GiNaC_GlobalParam):
                    try:
                        newvalue=float(val)
                    except:
                        raise ValueError("Cannot set a global parameter value to "+str(val))
                    if mode=="=":
                        current.value=newvalue
                    elif mode=="*":
                        current.value *= newvalue
                    elif mode=="/":
                        current.value /= newvalue
                    elif mode=="+":
                        current.value += newvalue
                    elif mode=="-":
                        current.value -= newvalue
                    continue
                elif isinstance(current,MaterialProperties):
                    from ..materials.generic import Mixture,get_pure_material,get_pure_liquid,get_pure_gas,get_surfactant,get_interface_properties,get_pure_solid #type:ignore
                    try:
                        newvalue = eval(val)
                    except Exception as e:
                        raise RuntimeError("Cannot set the material " + varname + " to " + str(val) + "\n" + str(e))
                    if mode!="=":
                        raise RuntimeError("Cannot set material properties with e.g. +=, *=, -=, /=")
                elif isinstance(current,(list,tuple)):
                    try:
                        newvalue=eval(val)
                    except Exception as e:
                        raise RuntimeError("Cannot set the list " + varname + " to " + str(val) + "\n" + str(e))
                else:
                    raise RuntimeError("Implement setting parameter of type "+str(type(current))+" value="+str(current)) #type:ignore
                if isinstance(obj,dict):
                    if mode=="=":
                        obj[splt_varname[-1]]= newvalue
                    elif mode=="*":
                        obj[splt_varname[-1]]*= newvalue
                    elif mode=="/":
                        obj[splt_varname[-1]]/= newvalue
                    elif mode=="-":
                        obj[splt_varname[-1]]-= newvalue
                    elif mode=="+":
                        obj[splt_varname[-1]]+= newvalue
                else:
                    if mode=="=":
                        setattr(obj, splt_varname[-1], newvalue)
                    else:
                        old=getattr(obj,splt_varname[-1])
                        if mode=="*":
                            setattr(obj, splt_varname[-1], old*newvalue)
                        elif mode=="/":
                            setattr(obj, splt_varname[-1], old/ newvalue)
                        elif mode=="+":
                            setattr(obj, splt_varname[-1], old + newvalue)
                        elif mode=="-":
                            setattr(obj, splt_varname[-1], old - newvalue)
                if not self.is_quiet():
                    print("PARAMETER ", varname, "SET TO",newvalue)

    def before_assigning_equation_numbers(self,dof_selector:Optional["_DofSelector"]):
        for hook in self._hooks:
            hook.before_assigning_equation_numbers(dof_selector,True)
        self._equation_system._before_assigning_equations(dof_selector) 
        for hook in self._hooks:
            hook.before_assigning_equation_numbers(dof_selector,False)


    def actions_before_remeshing(self,active_remeshers:List["RemesherBase"]):
        for hook in self._hooks:
            hook.actions_before_remeshing(active_remeshers)



    def actions_after_change_in_global_parameter(self,param:str):
        for hook in self._hooks:
            hook.actions_after_change_in_global_parameter(param)

    def actions_after_parameter_increase(self,param:str):
        for hook in self._hooks:
            hook.actions_after_parameter_increase(param)

    def actions_after_remeshing(self):
        self._equation_system._after_remeshing() 
        self.reapply_boundary_conditions()
        self.invalidate_cached_mesh_data()
        if self._custom_assembler:
            self._custom_assembler.actions_after_remeshing()
        for hook in self._hooks:
            hook.actions_after_remeshing()


    def actions_before_newton_solve(self):
        self._domains_to_remesh.clear()
        for ism in range(self.nsub_mesh()):
            submesh=self.mesh_pt(ism)
            if isinstance(submesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh,ODEStorageMesh)):
                #print("DIRCHLET UPDATE ",submesh,submesh.get_full_name())
                submesh.setup_Dirichlet_conditions(True)
        self._equation_system._before_newton_solve() 
        for hook in self._hooks:
            hook.actions_before_newton_solve()
        if self._debug_largest_residual>0:
            self.debug_largest_residual(self._debug_largest_residual)

    def last_newton_step_failed(self):
        last_res=self.get_last_residual_convergence()
        if len(last_res)==0 or last_res[-1]>self.newton_solver_tolerance:
            return True
        return False

    def actions_after_newton_solve(self):
        if self.last_newton_step_failed():
            return # Don't do this if it has not converged
        self._equation_system._after_newton_solve() 
        for ism in range(self.nsub_mesh()):
            submesh=self.mesh_pt(ism)
            if isinstance(submesh,MeshFromTemplateBase):
                submesh._solves_since_remesh+=1 
        if len(self._domains_to_remesh)>0:
            if (self._solve_in_arclength_conti is None) and self.do_call_remeshing_when_necessary:
                self.force_remesh(self._domains_to_remesh)
        self.invalidate_cached_mesh_data()
        if self._custom_assembler:
            self._custom_assembler.actions_after_successful_newton_solve()
        for hook in self._hooks:
            hook.actions_after_newton_solve()

    def remeshing_necessary(self):        
        """
        Checks whether any RemeshWhen object indicates that remeshing should be done.

        Returns:
            bool: True if remeshing would be required, False otherwise.
        """
        if len(self._domains_to_remesh)>0:
            return True
        return False

    def remesh_if_necessary(self) -> bool:
        """
        Invokes remeshing if one RemeshWhen object indicates that.

        Returns:
            bool: True if remeshing was performed, False otherwise.
        """
        res=False
        if len(self._domains_to_remesh)>0:
            self.force_remesh(self._domains_to_remesh)
            res=True
        return res
    

    # Can be used for go_to_param or 
    def remesh_handler_during_continuation(self, force: bool = False, resolve: bool = True, resolve_before_eigen: bool = False, reactivate_biftrack_neigen: int = 4, reactivate_biftrack_shift:float=0,resolve_max_newton_steps : Optional[int]=None,num_adapt:Optional[int]=None,resolve_globally_convergent_newton:bool=False):
        """
        Handle remeshing during continuation. We might have to calculate e.g. a new eigenvector when doing bifurcation tracking.
        In that case, set Problem.do_remeshing_when_necessary to False to prevent any automatic remeshing.

        Args:
            force (bool, optional): Force remeshing even if not necessary. Defaults to False.
            resolve (bool, optional): Resolve the problem after remeshing. Defaults to True.
            resolve_before_eigen (bool, optional): Resolve the problem before solving the eigenproblem. Defaults to False.
            reactivate_biftrack_neigen (int, optional): Number of eigenvalues to reactivate bifurcation tracking. Defaults to 4.
            reactivate_biftrack_shift (float, optional): Shift for the eigenvalues to reactivate bifurcation tracking. Defaults to 0.
            resolve_max_newton_steps (int, optional): Maximum number of Newton steps to resolve the problem. 
            resolve_globally_convergent_newton: Use a globally convergent Newton solver. Defaults to False.
            

        Returns:
            bool: True if remeshing was performed, False otherwise.
        """        
        #print("ENTER",len(self.get_arclength_dof_derivative_vector()))
        if not force and not self.remeshing_necessary():
            return False
        biftrack = self.get_bifurcation_tracking_mode()
        biftrack_param = self._bifurcation_tracking_parameter_name
        if biftrack == "azimuthal":
            m=self._azimuthal_mode_param_m.value
            k=None
        elif biftrack == "cartesian_normal_mode":
            k=self.get_current_normal_mode_k(dimensional=True)
            m=None
        else:
            m=None
            k=None
        #print("BIFTRACK",biftrack)
        if biftrack != "":
            # TODO: Keep the continuation data here!
            self.reset_arc_length_parameters()
            self.deactivate_bifurcation_tracking()
            self.reset_arc_length_parameters()
            

            
            
        self.force_remesh(num_adapt=num_adapt)
        
        # Reobtain the arclength vectors
        if self._last_arclength_parameter is not None:
            dof_deriv=self.get_history_dofs(5)
            dof_current=self.get_history_dofs(6)
            self._update_dof_vectors_for_continuation(dof_deriv,dof_current)
        
        if biftrack != "":
            if resolve_before_eigen:
                self.actions_before_stationary_solve(force_reassign_eqs=True)
                self.solve(max_newton_iterations=resolve_max_newton_steps)
            print("RESOLVING EIGENPROBLEM AT ",k,m)
            self.solve_eigenproblem(reactivate_biftrack_neigen,azimuthal_m=m,normal_mode_k=k,shift=reactivate_biftrack_shift)
            self.activate_bifurcation_tracking(biftrack_param, biftrack)
            if resolve:
                self.solve(max_newton_iterations=resolve_max_newton_steps,globally_convergent_newton=resolve_globally_convergent_newton)
        elif resolve:
            self.solve(max_newton_iterations=resolve_max_newton_steps,globally_convergent_newton=resolve_globally_convergent_newton)

    def _link_geometry_and_equations(self):
        #Go through the templates and create them
        domset:Set[str]=set()
        for m in self._meshtemplate_list:
            m._do_define_geometry(self) 
            mydoms=set(m.available_domains())
            inters=domset.intersection(mydoms)
            if len(inters)>0:
                raise RuntimeError("Following domains are added multiple times: "+str(inters))
            domset.update(mydoms)


        if not hasattr(self,"_equation_system") or self._equation_system is None:
            raise RuntimeError("Please add at least one equation to the problem via add_equations()")

        self._equation_system._fill_dummy_equations(self)
        self._interinter_connections.clear()
        for m in self._meshtemplate_list:
            m._ensure_opposite_eq_tree_nodes(self._equation_system) 
            inters=m._find_interface_intersections()
            for m in inters:
                dom=m.split("/")[0]
                if dom in self._equation_system._children and self._equation_system._children[dom]._equations is not None:
                    self._interinter_connections.add(m)
        if len(self._interinter_connections)>0:
            self._equation_system._fill_interinter_connections(self._interinter_connections)
        
        


        self._equation_system._finalize_equations(self) 
        
        for m in self._meshtemplate_list:
            m._connect_opposite_interfaces(self._equation_system) 
        self._equation_system._set_parent_to_equations(self) 

        #TODO: ODEs added to the root
        for meshname,eqtree in self._equation_system.get_children().items(): 
            #Find the mesh that generates the mesh we want to have
            if eqtree._equations is None: 
                raise RuntimeError("Empty bulk equations")
            mesh=None
            for m in self._meshtemplate_list:
                if m.has_domain(meshname):
                    mesh=MeshFromTemplate(self,m,meshname,eqtree)
                    self._meshdict[meshname]=mesh
                    assert eqtree._equations is not None
                    eqtree._equations._mesh=mesh  #type:ignore
            if eqtree._equations._is_ode(): 
                if mesh is not None:
                    if not isinstance(mesh,ODEStorageMesh):
                        raise RuntimeError("Cannot add an ODE to a spatial mesh yet")
                mesh=ODEStorageMesh(self,eqtree,meshname)
                eqtree.get_code_gen()._mesh=mesh 
                eqtree.get_code_gen().set_latex_printer(self.latex_printer)
                self._meshdict[meshname]=mesh
            else:
                if mesh is None:
                    #print(str(self._equation_system))
                    avdoms=set()
                    for m in self._meshtemplate_list:
                        avdoms.update(set(m._domains.keys()))
                    raise RuntimeError("No mesh template with a domain named '"+meshname+'" was added, but there are equations defined on this domain. Available domains are '+str(avdoms))

        self._equation_system._create_dummy_domains_for_DG(self) 
        self._equation_system._finalize_equations(self,second_loop=True) 
        if not self.is_quiet():
            print("SOLVING THE FOLLOWING SYSTEM:\n"+str(self._equation_system))
        if self._outdir is not None:
            destpath = os.path.join(self._outdir, self._ccode_dir)
            Path(destpath).mkdir(parents=True, exist_ok=True)
            infofile=open(os.path.join(self.get_output_directory(),self._ccode_dir,"_equation_tree.txt"),"w")
            infofile.write(str(self._equation_system))
            infofile.close()

    def before_defining_problem(self,redefine:bool=False,old_meshes:Optional[Dict[str,AnyMesh]]=None,old_mesh_templates:Optional[List[MeshTemplate]]=None):
        pass

    def redefine_problem(self, code_dir:str,interpolator:Type["BaseMeshToMeshInterpolator"]=_DefaultInterpolatorClass,num_adapt:Optional[int]=None):
        """
        Redefines the problem by recompiling equations. 
        This can in principle be used if problem parameters have changed, but it is not recommended to change the problem structure.
        If possible, it is advised to use the global parameter system to change any parameters.

        Args:
            code_dir: Subdirectory in the output directory where the C++ code of the redefined problem will be written.
            interpolator: Mesh interpolator to map the fields of the old meshes to the new ones.
            num_adapt: Number of adaption steps after redefining the problem. If None, the number of adaption steps is determined by the max_refinement_level attribute.

        Raises:
            RuntimeError: If the problem contains no equations after the redefinition
        """
        self._ccode_dir = code_dir

        if not self.is_initialised():
            self.initialise()
            return

        self._equation_system = None #type:ignore
        old_meshtemplate_list = self._meshtemplate_list
        old_mesh_dict = self._meshdict
        self._meshtemplate_list = []
        self._meshdict = {}
        self.before_defining_problem(redefine=True, old_meshes=old_mesh_dict, old_mesh_templates=old_meshtemplate_list)
        self.define_problem()
        if self.additional_equations != 0:
            self.add_equations(self.additional_equations)

        self._link_geometry_and_equations()

        if len(self._meshdict) == 0:
            raise RuntimeError("No mesh or ODE added to the problem, do it in the define_problem() method")

        self.compile_meshes()

        self.rebuild_global_mesh_from_list(rebuild=True)

        self.relink_external_data()

        self.setup_pinning()
        self.before_assigning_equation_numbers(self._dof_selector)
        self.reapply_boundary_conditions()

        if self.cmdlineargs.distribute:
            self.distribute()

        self.init_output(redefined=True)
        self.rebuild_global_mesh_from_list(rebuild=True)
        self.reapply_boundary_conditions()


        num_adapt = self.max_refinement_level if num_adapt is None else num_adapt

        interpolators:Dict[str,"BaseMeshToMeshInterpolator"]={}

        def perform_interpolation():
            for _, interp in interpolators.items():
                interp.interpolate()

        for name, newmesh in self._meshdict.items():
            if name in old_mesh_dict.keys():
                omesh=old_mesh_dict[name]
                if isinstance(newmesh,MeshFromTemplateBase) and isinstance(omesh,MeshFromTemplateBase):
                    interpolators[name]=interpolator(omesh,newmesh)
                elif isinstance(newmesh,ODEStorageMesh) and isinstance(omesh,ODEStorageMesh):
                    interpolators[name]=ODEInterpolator(omesh,newmesh)
                omesh.get_eqtree()._before_mesh_to_mesh_interpolation(interpolators[name])

        if num_adapt > 0:
            no_need_to_reassign = False
            for s in range(num_adapt):
                perform_interpolation()
                if not self.is_quiet():
                    print("Remeshing adaption:", s, "of", num_adapt)
                nref, nunref = self._adapt()
                if nref == 0 and nunref == 0:
                    no_need_to_reassign = True
                    break
            if num_adapt > 0 and not (no_need_to_reassign):
                self.map_nodes_on_macro_elements()
                perform_interpolation()
        else:
            self.map_nodes_on_macro_elements()
            perform_interpolation()

        # TODO: Unload unused DLLs

    def before_parsing_cmd_line(self):
        pass
    
    
    def _write_log_header(self):
        from .. _version import __version__
        import datetime
        _pyoomph._write_to_log_file("Pyoomph version: "+str(__version__)+os.linesep)
        info=_pyoomph._get_core_information()
        _pyoomph._write_to_log_file("Core version: "+str(info)+os.linesep)
        _pyoomph._write_to_log_file("Python interpreter: "+sys.executable+os.linesep)
        _pyoomph._write_to_log_file("Python version: "+sys.version+os.linesep)        
        _pyoomph._write_to_log_file("Python path: "+str(sys.path)+os.linesep)
        _pyoomph._write_to_log_file("Platform: "+str(sys.platform)+os.linesep)
        #modules={modul.__name__:getattr(modul,"__version__","UNKNOWN") for _,modul in sys.modules if isinstance(modul, types.ModuleType)}
        modules= {m.__name__:m.__version__ for m in sorted(sys.modules.values(),key=lambda a : getattr(a,"__name__","")) if hasattr(m,"__name__") and hasattr(m,"__version__") and len(m.__name__.split("."))==1}
        _pyoomph._write_to_log_file("Loaded module versions: "+str(modules)+os.linesep)                
        _pyoomph._write_to_log_file("Log file started: "+str(datetime.datetime.now())+os.linesep)
        _pyoomph._write_to_log_file("####################"+os.linesep)
        _pyoomph._write_to_log_file("Args: "+str(sys.argv)+os.linesep)
        _pyoomph._write_to_log_file("####################"+os.linesep)
        _pyoomph._write_to_log_file(os.linesep)
        

    def initialise(self):
        """
        Initializes the problem by performing the necessary setup and initialization steps.
        If not done before, this method is automatically called by several methods, e.g. :py:meth:`solve`, :py:meth:`run` or :py:meth:`output`. 
        After initialization, you cannot change the problem anymore, except for global parameter values.

        Raises:
            RuntimeError: If the problem is already initialised or if a function that calls initialize is called during initialization.
        """
                    
        if self.is_initialised():
            raise RuntimeError("Is already initialised")
        if self._during_initialization:
            raise RuntimeError("During initialization, you have called a function that calls initialize...")
        self._during_initialization=True

        self.setup_cmd_line()
        self.before_parsing_cmd_line()
        self.parse_cmd_line()
        if not self.is_quiet():
            print("OUTPUT WILL BE WRITTEN TO",self._outdir)
        if self._outdir is not None:
            Path(self._outdir).mkdir(parents=True, exist_ok=True)
            keyfile=os.path.join(self._outdir,"_pyoomph_run_.txt")
        else:
            keyfile=None
            
        if self.logfile_name is not None:
            if not self.only_write_logfile_on_proc0 and get_mpi_rank()>1:
                raise RuntimeError("Cannot write log file on all processors yet")
            self._open_log_file(os.path.join(self._outdir,self.logfile_name),True)
            from . logging import pyoomph_activate_logging_to_file
            pyoomph_activate_logging_to_file()
            self._write_log_header()
            
            

        if self._runmode=="continue":
            # Find the highest dump
            dumpdir = os.path.join(self.get_output_directory(), "_states")
            dumps = sorted(glob.glob(os.path.join(dumpdir, "*.dump")))
            if len(dumps)==0 or keyfile is None or not os.path.isfile(keyfile):
                print("Cannot continue, starting over")
                self._runmode="overwrite"
        elif self._runmode=="delete":
            if keyfile is not None and os.path.isfile(keyfile) and get_mpi_rank()<=0:
                if not self.is_quiet():
                    print("Removing contents of output dir")

                    def rem_subdir(subdir:str,filter:Union[str,List[str],Tuple[str]],remglob:Optional[Iterable[str]]=None):
                        top=os.path.join(self._outdir,subdir)
                        if not os.path.exists(top) or not os.path.isdir(top):
                            return
                        if not isinstance(filter,(list,tuple)):
                            filter=[filter]
                        lst:List[str]=[]
                        for g in filter:
                            glb=glob.glob(os.path.join(top,g))
                            if remglob:
                                for rg in remglob:
                                    glb=list(set(glb)-set(glob.glob(os.path.join(top,rg))))
                                #print("GLOB AFTER REMBLOG",glb,rg)
                            lst+=glb
                        for f in lst:
                            if os.path.isfile(f):
                                os.remove(f)
                                #print("REM",f)
                        #if not os.listdir(top):
                        #	os.rmdir(top)
                    if not self._suppress_code_writing and not self._suppress_compilation:
                        rem_subdir("_ccode",["*.c","*.dll","*.o",".dylib","*.so"])
                    rem_subdir("_states", ["*.dump"])
                    rem_subdir("_plots", ["*.*"])

                    #rem_subdir(".", ["*.txt","*.pvd","*.mat"])
                    subdirs=[f.parts[-1] for f in Path(self._outdir).iterdir() if f.is_dir()]
                    remglob:Optional[List[str]]=None
                    if self._suppress_code_writing:
                        remglob=["*.c"]
                    if self._suppress_compilation:
                        if remglob is None:
                            remglob=[]
                        remglob+=["*.c","*.dll","*.o",".dylib","*.so"]
                    for s in subdirs:
                        rem_subdir(s,["*"],remglob)

        mpi_barrier()

        if get_mpi_rank()<=0 and keyfile is not None:
            f=open(keyfile,"w+")
            #TODO: Add information
            f.close()
            if self.gitignore_output:
                gitignore=open(os.path.join(self._outdir,".gitignore"),"w")
                gitignore.write("*\n")
                gitignore.close()

        mpi_barrier()

        self.before_defining_problem()
        self.define_problem()
        if self._setup_azimuthal_stability_code:
            if  self._setup_additional_cartesian_stability_code:
                raise RuntimeError("Cannot set up both azimuthal and additional cartesian coordinate stability simultaneously yet")
            self.define_problem_for_axial_symmetry_breaking_investigation()
        elif self._setup_additional_cartesian_stability_code:
            self.define_problem_for_additional_cartesian_stability_investigation()
        if self.additional_equations != 0:
            self.add_equations(self.additional_equations)


        self._link_geometry_and_equations()

        if len(self._meshdict)==0:
            raise RuntimeError("No mesh or ODE added to the problem, do it in the define_problem() method")

        self.compile_meshes()
        #print("MESH COMPILE DONE")

        infofile = open(os.path.join(self.get_output_directory(), "_numerical_factors.txt"), "w")
        infofile.write(self._equation_system.numerical_factors_to_string())
        infofile.close()

        if self.latex_printer is not None:
#            raise RuntimeError("LATEX PRINTER")
            self.latex_printer.write_to_file(os.path.join(self.get_output_directory(), "system_info.tex"))


        self.rebuild_global_mesh_from_list(rebuild=False)

        self.relink_external_data()


        self.setup_pinning()
        self.before_assigning_equation_numbers(self._dof_selector)
        self.reapply_boundary_conditions()


        if self.cmdlineargs.distribute:
            self.actions_before_distribute()
            self.distribute()
            self.actions_after_distribute()
            
            
        if self.check_mesh_integrity:
            for _,m in self._meshdict.items():
                assert m._codegen is not None                                
                if isinstance(m,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d)):
                    m.check_integrity()

        if self._runmode!="continue" and self._runmode!="replot":
            if self.initial_adaption_steps is None:
                self.initial_adaption_steps=self.max_refinement_level
            if  (self.initial_adaption_steps>0):
                no_need_to_reassign=False
                for s in range(self.initial_adaption_steps):
                    self.map_nodes_on_macro_elements()
                    self.set_initial_condition()
                    self._initialised = True
                    self._during_initialization=False
                    if not self.is_quiet():
                        print("Initial adaption:",s,"of",self.initial_adaption_steps)
                    nref,nunref=self._adapt()
                    if nref==0 and nunref==0:
                        no_need_to_reassign=True
                        break
                if self.initial_adaption_steps>0 and not (no_need_to_reassign):
                    self.map_nodes_on_macro_elements()
                    self.set_initial_condition()
            else:
                self.map_nodes_on_macro_elements()
                self.set_initial_condition()
        else:
            self._initialised=True
            self._during_initialization=False

        if self.remove_macro_elements_after_initial_adaption:
            self.remove_macro_elements(self.remove_macro_elements_after_initial_adaption)

        if self._runmode=="replot":
            self._perform_replot()
            self.release()
            exit()

        if self._runmode=="continue":
            # Find the highest dump
            dumpdir=os.path.join(self.get_output_directory(),"_states")
            dumps=sorted(glob.glob(os.path.join(dumpdir,"*.dump")))
            while len(dumps)>0:
                dump_to_load=dumps.pop()
                if not self.is_quiet():
                    print("Loading state "+dump_to_load)
                try:
                    self._initialised = True
                    self._during_initialization=False
                    self.load_state(dump_to_load)
                    #self.save_state("_states/_continued_at_.dmp",relative_to_output=True)
                    break
                except Exception as e:
                    print("Cannot load state"+dump_to_load,e)
            else:
                raise RuntimeError("Cannot load any state file to continue")
            self._continue_initialized=True
            self._output_step+=1

        self._initialised = True
        self._during_initialization=False
        self.init_output()
        self.rebuild_global_mesh_from_list(rebuild=True)
        self.reapply_boundary_conditions()

        if self._custom_assembler:
            self._custom_assembler.initialize()

        if not self.is_quiet():
            print("PROBLEM IS NOW INITIALIZED")
            print("Following solvers will be used:")
            print("    Sparse Matrix Inversion: "+self.get_la_solver().idname)
            print("   Generalized Eigen Solver: "+self.get_eigen_solver().idname)       
            compiler_name="internal TCC compiler" 
            ccomp=self.get_ccompiler()
            if isinstance(ccomp,BaseCCompiler):
                compiler_name=ccomp.compiler_id
            print("  Equation code compiled by: "+ compiler_name)
            print("==========================")

        if self.plot_in_dedicated_process:
            if not self.write_states:
                raise RuntimeError("Cannot use 'plot_in_dedicated_process' without 'write_states'")
            try:
                mycmd=sys.orig_argv.copy()
            except:
                raise RuntimeError("Problem.plot_in_dedicated_process=True only works for Python>=3.10")
            mycmd+=["--runmode","p","--where","__pipe__"]
            plotlog=open(self.get_output_directory("_dedicated_plotter_log.txt"),"w")
            #self._plotting_process=subprocess.Popen(mycmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if not self.is_quiet():
                print("Creating dedicated plot process: "+str(mycmd))
            self._plotting_process=subprocess.Popen(mycmd,stdin=subprocess.PIPE,stdout=plotlog,stderr=plotlog)
            #print(self._plotting_process.)

        for hook in self._hooks:
            hook.actions_after_initialise()
            
            
        if self.cmdlineargs.generate_precice_cfg:
            print("Generating preCICE configuration file")
            from ..solvers.precice_adapter import get_pyoomph_precice_adapter
            get_pyoomph_precice_adapter().generate_precice_config_file(self)
            exit()


    def _perform_replot(self):
        if self._where_expression=="__pipe__":
            print("LISTENING FOR PLOTTING STATES..., __exit__ to close")
            for cmd in sys.stdin:
                cmd=cmd.rstrip()
                if cmd=="__exit__":
                    break
                else:
                    self.load_state(cmd)
                    self.timestepper.set_weights()
                    self.perform_plot()
        else:
            dumpdir = os.path.join(self.get_output_directory(), "_states")
            dumps = sorted(glob.glob(os.path.join(dumpdir, "*.dump")))
            for d in dumps:
                time,step=self._get_time_of_state_file(d)
                where_res=eval(self._where_expression,{},{"step":step,"time":time})
                #print(d,where_res)
                if where_res:
                    self.load_state(d)
                    self.timestepper.set_weights()
                    self.perform_plot()


    def rebuild_global_mesh_from_list(self,rebuild:bool=True):
        def recu_add_imeshes(sm:Union[MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh]):
            for _k, im in sm._interfacemeshes.items():   # Interface meshes
                assert im._codegen is not None
                im._codegen._mesh = im 
                im.ensure_external_data() 
                self.add_sub_mesh(im) 
                self._interfacemeshes.append(im) 
            for _k, im in sm._interfacemeshes.items():   # Interface meshes
                recu_add_imeshes(im) 


        if rebuild:
            self.flush_sub_meshes()
            self._interfacemeshes=[]
        for _,m in self._meshdict.items():
            assert m._codegen is not None
            m._codegen._mesh=m 
            self.add_sub_mesh(m)
            if isinstance(m,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d)):
                recu_add_imeshes(m)
        if rebuild:
            if not self.is_quiet():
                print("REBUILDING GLOBAL MESH FROM LIST")
            self.rebuild_global_mesh()
        else:
            if not self.is_quiet():
                print("BUILDING GLOBAL MESH FROM LIST")
            self.build_global_mesh()
        for m in self._meshtemplate_list:
            m._connect_opposite_elements(self._equation_system) 



    def setup_pinning(self):
        self.ensure_dummy_values_to_be_dummy()
        for ism in range(self.nsub_mesh()):
            submesh=self.mesh_pt(ism)
            if isinstance(submesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh,ODEStorageMesh)):
                #print("DIRCHLET SET ", submesh, submesh.get_full_name())
                submesh.setup_Dirichlet_conditions(False)
                assert submesh._codegen is not None 
                submesh._codegen.on_apply_boundary_conditions(submesh) 




    def set_initial_condition(self, ic_name: str = "", all_unset_dofs_to_zero: bool = False):
        """
        Set the initial condition for the problem.

        Args:
            ic_name (str, optional): Name of the initial condition. Multiple initial conditions can be defined in the problem definition by using the optional argument InitialConditions. Defaults to "".
            all_unset_dofs_to_zero (bool, optional): Flag indicating whether to set all unset degrees of freedom, i.e. without any InitialCondition with this ic_name, to zero. Defaults to False.
        """
        if all_unset_dofs_to_zero:
            self.set_current_dofs([0.0] * self.ndof())
        if not self.is_quiet():
            print("SETTING IC", ic_name)
        if self._runmode != "continue":
            for _, m in self._meshdict.items():
                m.setup_initial_conditions_with_interfaces(self._resetting_first_step, ic_name)
                if isinstance(m, ODEStorageMesh):
                    continue
                # for n, sm in m._interfacemeshes.items():
                #     sm.setup_initial_conditions()
                #     TODO Recursive
                #     print(sm)
                # print("ICMESH", m)
        self.setup_pinning()
        self.reapply_boundary_conditions()
        self.invalidate_cached_mesh_data()
        if self._custom_assembler:
            self._custom_assembler.actions_after_setting_initial_condition()



    def get_time_symbol(self,with_scaling:bool=True) -> Expression:
        return get_global_symbol("t")*(self.get_scaling("temporal") if with_scaling else 1)


    def actions_before_adapt(self):
        for m in self._interfacemeshes:
            m.clear_before_adapt()
            #print("CLEARED INTERFACE MESH",m.nelement())
        if len(self._interfacemeshes):
            if not self.is_quiet():
                print("REBUILDING GLOBAL MESH")
            self.rebuild_global_mesh()
        self.invalidate_cached_mesh_data()


    def actions_before_distribute(self):
        self.actions_before_adapt()
        for _, m in self._meshdict.items():
            if isinstance(m, ODEStorageMesh):
                continue
            m.ensure_halos_for_periodic_boundaries()
            
    def actions_after_distribute(self):
        self.actions_after_adapt()

    def map_nodes_on_macro_elements(self):
        self.invalidate_cached_mesh_data()
        for _,m in self._meshdict.items():
            if isinstance(m,ODEStorageMesh):
                continue
            for ei in range(m.nelement()):
                elem=m.element_pt(ei)
                macro=elem.get_macro_element()
                if macro:
                    elem.map_nodes_on_macro_element()
        self._equation_system._after_mapping_on_macro_elements() 


    def remove_macro_elements(self,mode:Union[bool,Literal["auto"]]="auto"):        
        for _,m in self._meshdict.items():
            if isinstance(m,ODEStorageMesh):
                continue
            if mode=="auto" and m._codegen is not None and not m._codegen._coordinates_as_dofs: 
                continue

            for e in m.elements():
                e.set_macro_element(None,False)
                while  e.get_father_element() is not None:
                    e=e.get_father_element()
                    e.set_macro_element(None, False)

    def describe_equation(self,dofindex:int) -> str:
        res = "unknown(" + str(dofindex) + ")"
        for mesh_name, mesh in self._meshdict.items():
            if isinstance(mesh,ODEStorageMesh):
                continue
            for node in mesh.nodes():
                for vi in range(node.nvalue()):
                    eq = node.eqn_number(vi)
                    if eq == dofindex:
                        res="nodal value index "+str(vi)+ ". Node is located at " + ", ".join(map(str, [node.x(xi) for xi in range(node.ndim())]))+" in mesh "+mesh_name
                        break
                pd=node.variable_position_pt()
                for vi in range(pd.nvalue()):
                    eq=pd.eqn_number(vi)
                    if eq == dofindex:
                        res="nodal position index "+str(vi)+ ". Node is located at " + ", ".join(map(str, [node.x(xi) for xi in range(node.ndim())]))+" in mesh "+mesh_name
                        break

        for ism in range(self.nsub_mesh()):
            sm=self.mesh_pt(ism)
            assert isinstance(sm,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh,ODEStorageMesh))
            for e in sm.elements():
                for di in range(e.ndof()):
                    eq=e.eqn_number(di)
                    if eq==dofindex:
                        dn=e.get_dof_names()
                        res=res+"\n"+"Found in element of submesh "+sm.get_name()+" elem internal:"+str(e.ninternal_data())+" external:"+str(e.nexternal_data())+" dofname: "+dn[di]
                        break
        return res


    def reapply_boundary_conditions(self):
        self.setup_pinning()
        self.before_assigning_equation_numbers(self._dof_selector)
        self._dof_selector_used=self._dof_selector
        neq=self.assign_eqn_numbers(True)
        if not self.is_quiet():
            print("Number of equations: " + str(neq))
        if self._custom_assembler:
            self._custom_assembler.actions_after_equation_numbering()

    def get_dof_description(self):
        """
        Returns two arrays containing the description of the degrees of freedom.
        The first is a list of dof-type indices, where the i-th entry is the type of the i-th degree of freedom.
        To resolve what each dof-type index means, the second array contains the type names of the degrees of freedom.
        
        For a simple Poisson equation for a field ``u`` on a line domain name ``"domain"``, the first returned array will contain numbers between 0 and 2 (dof-type indices), and the second array will contain the type names ``["domain/u", "domain/left/u", "domain/right/u"]``.
        Note how the boundaries get their own dof-type indices.
                        
        Returns:
            A pair of arrays containing the dof-type indices and the type names to classify the degrees of freedom.
        """
        doflist:NPIntArray=numpy.array([],dtype=numpy.int32) #type:ignore
        dofnames:List[str] = []

        def process(m:AnyMesh):
            nonlocal doflist,dofnames
            types, names = m.describe_global_dofs()
            types=numpy.array(types) #type:ignore
            #if numpy.all(types<0):
            #	print("MESH "+m.get_full_name()+" does not identify any dofs...")
            #print(m.get_full_name(),types,names)
            if len(doflist)==0:
                doflist = numpy.array(types,dtype=numpy.int32) #type:ignore
            offset = len(dofnames)
            trunk = m.get_full_name()
            for n in names:
                dofnames.append(trunk + "/" + n)
            for k in range(len(doflist)):
                if k>=len(types):
                    raise RuntimeError("Strange. Should not happen: "+m.get_full_name()+" NAMES: "+str(names)+" TYPES: "+str(types),"DOFLIST: "+str(doflist))
                if types[k] >= 0:
                    doflist[k] = offset + types[k]
            if not isinstance(m,ODEStorageMesh):
                for im in m._interfacemeshes.values(): 
                    process(im)

        for _, bm in self._meshdict.items():
            process(bm)

        if numpy.any(doflist<0): #type:ignore
            if self.get_bifurcation_tracking_mode()=="azimuthal":
                print("DOING AZIMUTHAL PATCHING",self.ndof())
                num_unassigned=len(numpy.argwhere(doflist<0))
                num_assigned=len(doflist)-num_unassigned
                
                if num_unassigned==2*num_assigned+2:
                    has_imag=True
                    N_base=(len(doflist)-2)//3
                elif num_unassigned==num_assigned+1:
                    has_imag=False
                    N_base=(len(doflist)-1)//2
                else:
                    raise RuntimeError("Strange here",num_assigned,num_unassigned)
                
                dof_base=len(dofnames)                
                dofnames+=[d+"__(ReEigen)" for d in dofnames]
                if has_imag:
                    dofnames+=[d+"__(ImEigen)" for d in dofnames[:dof_base]]
                dofnames+=["Bifurcation_Parameter_or_LambdaRe"]
                if has_imag:
                    dofnames+=["LambdaIm"]
                    doflist[-1]=len(dofnames)-1
                    doflist[-2]=len(dofnames)-2
                else:
                    doflist[-1]=len(dofnames)-1
                doflist[N_base:2*N_base]=doflist[:N_base]+dof_base
                if has_imag:
                    doflist[2*N_base:3*N_base]=doflist[:N_base]+2*dof_base
            # TODO: Other handlers
            else:                
                print("UNASSIGNED DOF IN DOFLIST")
                print("NUM:",len(numpy.argwhere(doflist<0)),"of",len(doflist)) #type:ignore

        return doflist, dofnames


    def search_dof_in_mesh(self,mesh:AnyMesh,dofindex:int):
        location = None
        typ = None
        if not isinstance(mesh,ODEStorageMesh):
            for n in mesh.nodes():
                found_in_node = False
                for iv in range(n.nvalue()):
                    if n.eqn_number(iv) == dofindex:
                        found_in_node = True
                        break
                if not found_in_node:
                    for ip in range(n.ndim()):
                        if n.variable_position_pt().eqn_number(ip) == dofindex:
                            found_in_node = True
                            break
                if found_in_node:
                    location = [n.x(i) for i in range(n.ndim())]
                    if n.is_on_boundary():
                        typ = "boundary node"
                        bn = mesh.get_boundary_names()
                        onbounds:List[str] = []
                        for i in range(len(bn)):
                            if n.is_on_boundary(i):
                                onbounds.append(bn[i])
                        if len(onbounds) > 0:
                            typ += " (" + ", ".join(onbounds) + ")"
                        else:
                            typ += " (NO BOUND NAMES FOUND)"
                    else:
                        typ = "bulk node"
                    break
        if location is None:
            for e in mesh.elements():
                for nid in range(e.ninternal_data()):
                    id = e.internal_data_pt(nid)
                    for vid in range(id.nvalue()):
                        if id.eqn_number(vid) == dofindex:
                            location = e.get_Eulerian_midpoint()
                            typ = "element"
                            break
        if location is None:
            for e in mesh.elements():
                for nid in range(e.nexternal_data()):
                    id = e.external_data_pt(nid)
                    for vid in range(id.nvalue()):
                        if id.eqn_number(vid) == dofindex:
                            location = e.get_Eulerian_midpoint()
                            typ = "data stored in other element"
                            break

        return location,typ

    def debug_largest_residual(self, nres:int=4):
        if not self.is_initialised():
            self.initialise()
        descr, names = self.get_dof_description()
        #print(names)
        #print(descr)
        res_vect:NPFloatArray = numpy.array(self.get_residuals()) #type:ignore
        highdofsI:NPIntArray = numpy.argsort(numpy.absolute(res_vect)) #type:ignore
        highdofs:List[int] = list(reversed(highdofsI[-1 - nres+1:])) #type:ignore
        print("========MAX. RESIDUALS========")
        for idof, dofindex in enumerate(highdofs):
            print("Highest residual", idof + 1, " with a value of", res_vect[dofindex], "Eqn number:", dofindex)
            if descr[dofindex] >= 0:
                dofstr:str = names[descr[dofindex]]
                print("   belongs to " + dofstr)
                # Find the dof
                splt = dofstr.split("/")
                meshname = "/".join(splt[0:-1])
                #dofname = splt[-1]
                mesh = self.get_mesh(meshname, return_None_if_not_found=True)
                if mesh is not None:
                    if self.get_bifurcation_tracking_mode()=="azimuthal":
                        has_imag=False
                        for ddd in names:
                            if ddd.endswith("__(ImEigen)"):
                                has_imag=True
                                break
                        if has_imag:
                            ndof_base=(self.ndof()-2)//3
                        else:
                            ndof_base=(self.ndof()-1)//2
                        if splt[-1].endswith("__(ReEigen)"):
                            dofindex-=ndof_base
                        elif splt[-1].endswith("__(ImEigen)"):
                            dofindex-=2*ndof_base
                    location,typ=self.search_dof_in_mesh(mesh,dofindex)
                    if location is None or typ is None:
                        print("   ... cannot find any node or element containing this dof...")
                    else:
                        print("   found at " + str(location) + ". Type: " + typ)
                else:
                    print("   cannot find the mesh " + meshname)
            else:
                print("   cannot find a description...")
                for ism in range(self.nsub_mesh()):
                    mesh=self.mesh_pt(ism)
                    assert isinstance(mesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh,ODEStorageMesh))
                    location, typ = self.search_dof_in_mesh(mesh, dofindex)
                    if location is not None and typ is not None:
                        print("     but found in mesh "+mesh.get_full_name()+" at " + str(location) + ". Type: " + typ)
                    else:
                        print(" 		not found in mesh "+mesh.get_full_name())
                    #print("DESCRIBE",self.describe_equation(dofindex))
                    for e in mesh.elements():
                        for d in range(e.ndof()):
                            if e.eqn_number(d)==dofindex:
                                print("				FOUND IN ELEMENT at "+str(e.get_Eulerian_midpoint())+" nnode "+str(e.nnode())+" ninternal "+str(e.ninternal_data())+" nexternal "+str(e.nexternal_data()))
                                for ni in range(e.nnode()):
                                    en=e.node_pt(ni)
                                    for nv in range(en.nvalue()):
                                        if en.eqn_number(nv)==dofindex:
                                            print("             FOUND AT ELEMENTAL NODE AT INDEX "+str(ni)+", "+str(nv)+ " located at "+str([en.x(pi) for pi in range(en.ndim()) ]))
                                        else:
                                            print("                   nonmatching nodal "+str(ni)+", "+str(nv)+"  "+str(en.eqn_number(nv)))
                                for ne in range(e.ninternal_data()):
                                    id=e.internal_data_pt(ne)
                                    for nv in range(id.nvalue()):
                                        if id.eqn_number(nv)==dofindex:
                                            print("             FOUND AT INTERNAL DATA AT INDEX "+str(ne)+", "+str(nv))
                                for ne in range(e.nexternal_data()):
                                    id=e.external_data_pt(ne)
                                    for nv in range(id.nvalue()):
                                        if id.eqn_number(nv)==dofindex:
                                            print("             FOUND AT EXTERNAL DATA AT INDEX "+str(ne)+", "+str(nv))
        print("=====END OF MAX. RESIDUALS======")
        print()

    def actions_before_newton_convergence_check(self)->None:
        accept_step=self._equation_system._before_newton_convergence_check() 
        if not accept_step:
            print("Invalidating step by filling the dof vector with random exteme values")
            dofs,_=self.get_current_dofs()
            dofs=numpy.array(dofs)
            dofs[:]=1e40*dofs[:]+numpy.random.rand(len(dofs))*1e40
            self.set_current_dofs(dofs)            

    def actions_after_newton_step(self):
        #if self._solve_in_arclength_conti is not None:
        #    self.actions_after_change_in_global_parameter(self._solve_in_arclength_conti)
        if self._debug_largest_residual>0:
            self.debug_largest_residual(self._debug_largest_residual)
        if self.get_bifurcation_tracking_mode()!="" and not self.is_quiet():
            paramnames=[pn for pn in self.get_global_parameter_names() if not pn.startswith("_")]                        
            if len(paramnames)>0:
                paramstr="Currently at parameters: "+", ".join([n+"="+str(self.get_global_parameter(n).value) for n in paramnames])                
                if self._bifurcation_tracking_parameter_name=="<LAMBDA_TRACKING>":
                    paramstr+=". Lambda tracking at: "+str(complex(self._get_lambda_tracking_real(),self._get_bifurcation_omega()))                                
                print(paramstr)
        for h in self._hooks:
            h.actions_after_newton_step()

    def actions_after_adapt(self):
        for m in self._interfacemeshes:
            m.rebuild_after_adapt()
            m.ensure_external_data()
            #print("REBUILD INTERFACE MESH",m,m.get_name(), m.nelement(),m.element_pt(0))
        if not self.is_quiet():
            print("REBUILDING GLOBAL MESH")
        self.rebuild_global_mesh()
        for m in self._meshtemplate_list:
            m._connect_opposite_elements(self._equation_system)
        self.setup_pinning()
        self.reapply_boundary_conditions()
        
        if self.check_mesh_integrity is True:
            for _,m in self._meshdict.items():
                assert m._codegen is not None                                
                if isinstance(m,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d)):
                    m.check_integrity()
                    
        if self._call_output_after_adapt:
            self.output()
        if self._custom_assembler:
            self._custom_assembler.actions_after_adapt()


    def compile_bulk_element_code(self,elementtype:FiniteElementCodeGenerator,bulkmesh:AnyMesh,subname:str) -> _pyoomph.DynamicBulkElementInstance:
        if self._outdir is not None:
            destpath=os.path.join(self._outdir,self._ccode_dir)
            Path(destpath).mkdir(parents=True, exist_ok=True)
            trunk=os.path.join(destpath,subname)
        else:
            trunk=""
        suppress_compilation=False
        suppress_writing=False
        if self._suppress_compilation or get_mpi_rank()>0:
            ccomp=self.get_ccompiler()
            if not ccomp.compiling_to_memory():
                suppress_compilation=True
        if self._suppress_code_writing or get_mpi_rank()>0:
            suppress_writing=True
        mpi_barrier()
        
        res=self.generate_and_compile_bulk_element_code(elementtype,trunk,suppress_writing,suppress_compilation,bulkmesh,self.is_quiet(),self.extra_compiler_flags)
        #print("REt")
        mpi_barrier()
        #print("REt MPI")
        self._bulk_element_code_counter=self._bulk_element_code_counter+1
        return res


    def set_current_time(self, val: ExpressionOrNum, dimensional: bool = True, as_float: bool = False):
        """
        Set the current time of the problem.

        Args:
            val (ExpressionOrNum): The value of the time to set.
            dimensional (bool, optional): Flag indicating whether the time is dimensional or not. Defaults to True.
            as_float (bool, optional): Flag indicating whether to convert the time to a float. Defaults to False.

        Raises:
            ValueError: If the nondimensional time is not a number.
            ValueError: If the dimensional time cannot be nondimensionalized.

        Returns:
            None
        """
        tp = self.time_pt()
        if not dimensional:
            if not (isinstance(val, int) or isinstance(val, float)):
                raise ValueError("Nondimensional time needs to be a number, not " + str(val))
            tp.set_time(val)
        else:
            ts = self.get_scaling("temporal")
            t = val / ts
            if as_float:
                if isinstance(t, _pyoomph.Expression):
                    tin = t
                else:
                    tin = _pyoomph.Expression(t)
                factor, _, _, _ = _pyoomph.GiNaC_collect_units(tin)
                t = float(factor)
            else:
                try:
                    t = float(t)
                except:
                    raise ValueError("Cannot nondimensionalise time " + str(val) + " with time scale " + str(ts))
            tp.set_time(t)
    

    def define_global_parameter(self, **params: float) -> Union[_pyoomph.GiNaC_GlobalParam, Tuple[_pyoomph.GiNaC_GlobalParam, ...]]:
        """
        Define one or more global parameters for the problem.

        Args:
            **params: A dictionary of parameter names and their corresponding initial values.

        Returns:
            Union[_pyoomph.GiNaC_GlobalParam, Tuple[_pyoomph.GiNaC_GlobalParam, ...]]: 
            The defined global parameter(s), which can be used in expressions.

        """
        res = []
        for p, v in params.items():
            res.append(self.get_global_parameter(p))
            res[-1].value = v
        if len(res) == 1:
            return res[0]
        else:
            return tuple([*res])
        
    def setup_for_stability_analysis(self,analytic_hessian:bool=True,use_hessian_symmetry:bool=True,improve_pitchfork_on_unstructured_mesh:bool=False,improve_pitchfork_coordsys:"OptionalCoordinateSystem"=None,improve_pitchfork_position_coordsys:"OptionalCoordinateSystem"=None,shared_shapes_for_multi_assemble:Optional[bool]=None,azimuthal_stability:Optional[bool]=None,additional_cartesian_mode:Optional[bool]=None):
        """
        Sets up the problem for stability analysis, e.g. for improved pitchfork tracking on unsymmetric meshes, azimuthal stability, etc.
        Arguments which are None are not changed.

        Args:
            analytic_hessian (bool, optional): Flag indicating whether to use an analytically derived symbolical Hessian. Defaults to True.
            use_hessian_symmetry (bool, optional): Flag indicating whether to use symmetry in the Hessian. Defaults to True.
            improve_pitchfork_on_unstructured_mesh (bool, optional): Flag indicating whether to improve pitchfork tracking on unsymmetric meshes. Defaults to False.
            improve_pitchfork_coordsys (OptionalCoordinateSystem, optional): Coordinate system for improving pitchfork tracking unsymmetric meshes. Defaults to None.
            improve_pitchfork_position_coordsys (OptionalCoordinateSystem, optional): Coordinate system for improving pitchfork position space. Defaults to None.
            shared_shapes_for_multi_assemble (Optional[bool], optional): Flag indicating whether to use shared shapes for multi-assemble. Defaults to None.
            azimuthal_stability (Optional[bool], optional): Flag indicating whether to set up azimuthal stability code. Defaults to None.
        """           
        if self.is_initialised():
            raise RuntimeError("Cannot call setup_for_stability_analysis after problem is initialised") 
        if analytic_hessian:
            # May not use symmetric Hessian for azimuthal stability
            self.set_analytic_hessian_products(True,use_hessian_symmetry and (not azimuthal_stability and not additional_cartesian_mode))
        else:
            self.set_analytic_hessian_products(False)
        #if azimuthal_stability:
        #    self.set_analytic_hessian_products(False) # We may not use it here!
        if improve_pitchfork_on_unstructured_mesh:
            self.improve_pitchfork_tracking_on_unstructured_meshes(coord_sys=improve_pitchfork_coordsys,pos_coord_sys=improve_pitchfork_position_coordsys)
        if shared_shapes_for_multi_assemble is not None:
            self._shared_shapes_for_multi_assemble=shared_shapes_for_multi_assemble
        if azimuthal_stability:
            self._setup_azimuthal_stability_code=azimuthal_stability
        if additional_cartesian_mode:
            self._setup_additional_cartesian_stability_code=additional_cartesian_mode
        
    def is_normal_mode_stability_set_up(self)->Union[Literal["azimuthal","cartesian"],Literal[False]]:
        """
        Returns True when :py:meth:`~pyoomph.generic.problem.Problem.setup_for_stability_analysis` has been called with ``azimuthal_stability=True`` or ``additional_cartesian_mode=True``.
        Can be used to e.g. set additional BCs for velocity_phi or similar.
        """
        if self._setup_azimuthal_stability_code:
            return "azimuthal"
        elif self._setup_additional_cartesian_stability_code:
            return "cartesian"
        else:
            return False



    @overload
    def get_current_time(self,dimensional:bool=...,as_float:Literal[False]=...)->Expression: ...

    @overload
    def get_current_time(self,dimensional:bool=...,as_float:Literal[True]=...)->float: ...

    def get_current_time(self, dimensional: bool = True, as_float: bool = False) -> ExpressionOrNum:
        """
        Get the current time of the problem.

        Args:
            dimensional (bool, optional): Flag indicating whether to return the dimensional time. Defaults to True.
            as_float (bool, optional): Flag indicating whether to return the time as a float. Defaults to False.

        Returns:
            ExpressionOrNum: The current time of the problem.

        Raises:
            ValueError: If the problem is not initialized.

        """
        if not self.is_initialised():
            self.initialise()
        t = self.time_pt().time()
        if not dimensional:
            return t
        ts = self.get_scaling("temporal")
        t = t * ts
        if not as_float:
            return t
        try:
            t = float(t)
        except:
            t = float(t / second)
        return t

    def process_eigenvectors(self,eigenvectors:NPFloatArray):
        """Here, you can optionally scale or negate the eigenvectors, e.g. normalize them, by overriding this method

        Args:
            eigenvectors (NPFloatArray): 2d array of eigenvectors

        Returns:
            NPFloatArray: Processed array of eigenvectors
        """
        # 
        return eigenvectors

    def solve_eigenproblem(self, n:int, shift:Union[float,complex,None]=0, quiet:bool=False, azimuthal_m:Optional[Union[int,List[int]]]=None,normal_mode_k:Optional[Union[ExpressionOrNum,List[ExpressionOrNum]]]=None,normal_mode_L:Optional[Union[ExpressionOrNum,List[ExpressionOrNum]]]=None,report_accuracy:bool=False,sort:bool=True,which:"EigenSolverWhich"="LM",OPpart:Optional[Literal["r","i"]]=None,v0:Optional[Union[NPFloatArray,NPComplexArray]]=None,filter:Optional[Callable[[complex],bool]]=None,target:Optional[complex]=None)->Tuple[NPComplexArray,NPComplexArray]:
        """
        Solves the associated generalized eigenproblem for the given number of eigenvalues and eigenvectors.

        Args:
            n (int): The number of eigenvalues and eigenvectors to compute.
            shift (Union[float, complex, None], optional): The shift applied for shift-inverted approaches to solve the eigenproblem. Defaults to 0.
            quiet (bool, optional): If True, suppresses the output. Defaults to False.
            azimuthal_m (Optional[Union[int, List[int]]], optional): The azimuthal mode number(s) for axial symmetry breaking. Defaults to None, i.e. the axisymmetric mode.
            normal_mode_k: The wave number(s) for an additional direction in Cartesian coordinates. Defaults to None, i.e. the base mode.            
            normal_mode_L: The periodic length for an additional direction in Cartesian coordinates. Defaults to None, i.e. the base mode.            
            report_accuracy (bool, optional): If True, reports the accuracy of the computed eigenvalues. Defaults to False.
            sort (bool, optional): If True, sorts the eigenvalues in ascending order. Defaults to True.
            which ("EigenSolverWhich", optional): The type of eigenvalues to compute. Defaults to "LM".
            OPpart (Optional[Literal["r", "i"]], optional): The part of the operator to use. Defaults to None.
            v0 (Optional[Union[NPFloatArray, NPComplexArray]], optional): The initial guess for the eigenvectors. Defaults to None.
            filter (Optional[Callable[[complex], bool]], optional): A function to filter the computed eigenvalues. Only the eigenvalues for which the filter returns True will be kept. Defaults to None.
            target (Optional[complex], optional): The target eigenvalue. Defaults to None.

        Returns:
            Tuple[NPComplexArray, NPComplexArray]: A tuple containing the computed eigenvalues and eigenvectors.
        """
        self._solve_eigenproblem_helper(n,shift,quiet,azimuthal_m,normal_mode_k,normal_mode_L,report_accuracy,sort,which,OPpart,v0,filter,target)
        self._last_eigenvectors=self.process_eigenvectors(self._last_eigenvectors)
        return self._last_eigenvalues,self._last_eigenvectors
        
    def _solve_eigenproblem_helper(self, n:int, shift:Union[float,complex,None]=0, quiet:bool=False, azimuthal_m:Optional[Union[int,List[int]]]=None,normal_mode_k:Optional[Union[ExpressionOrNum,List[ExpressionOrNum]]]=None,normal_mode_L:Optional[Union[ExpressionOrNum,List[ExpressionOrNum]]]=None,report_accuracy:bool=False,sort:bool=True,which:"EigenSolverWhich"="LM",OPpart:Optional[Literal["r","i"]]=None,v0:Optional[Union[NPFloatArray,NPComplexArray]]=None,filter:Optional[Callable[[complex],bool]]=None,target:Optional[complex]=None)->Tuple[NPComplexArray,NPComplexArray]:
        """
        Real eigensolving: Called from solve_eigenproblem()
        """
        if not self.is_initialised():
            self.initialise()
        _dtorder=self._get_max_dt_order()
        if _dtorder!=1:
            if _dtorder==0:
                raise RuntimeError("Cannot calculate eigenvalues/vectors without any time derivatives. This would give an empty mass matrix")
            else:
                raise RuntimeError("Cannot calculate eigenvalues/vectors when you have an time derivative order of "+str(_dtorder)+". Consider using auxiliary unknowns and equations to reduce the order of all time derivatives to 1.")
        if normal_mode_L is not None:
            if normal_mode_k is not None:
                raise ValueError("Cannot specify both normal_mode_L and normal_mode_k")
            if isinstance(normal_mode_L,(list,tuple)):
                normal_mode_k=[2*pi/L for L in normal_mode_L]
            else:
                normal_mode_k=2*pi/normal_mode_L
            normal_mode_L=None
            
        if azimuthal_m is not None:
            if normal_mode_k is not None:
                raise ValueError("Cannot specify both azimuthal_m and normal_mode_k")
            if normal_mode_L is not None:
                raise ValueError("Cannot specify both azimuthal_m and normal_mode_L")
            return self._solve_normal_mode_eigenproblem(n, azimuthal_m=azimuthal_m, shift=shift, quiet=quiet,filter=filter,report_accuracy=report_accuracy,v0=v0,target=target,sort=sort)
        elif normal_mode_k is not None:
            if isinstance(normal_mode_k,(list,tuple)):
                normal_mode_k=[float(k*self.get_scaling("spatial")) for k in normal_mode_k]
            else:
                normal_mode_k=float(normal_mode_k*self.get_scaling("spatial"))
            return self._solve_normal_mode_eigenproblem(n, cartesian_k=normal_mode_k, shift=shift, quiet=quiet,filter=filter,report_accuracy=report_accuracy,v0=v0,target=target,sort=sort)
        if self._dof_selector_used is not self._dof_selector:
            self.reapply_boundary_conditions()
            self.reapply_boundary_conditions() # Must be done twice to correctly setup the equation remapping
        if self.get_bifurcation_tracking_mode()!="":
            raise RuntimeError("Cannot calculate eigenvalues/vectors when bifurcation tracking is active. You can access the critical eigenvector(s) by get_last_eigenvectors()")
        ntstep=self.ntime_stepper()
        was_steady=[False]*ntstep
        for i in range(ntstep):
            ts=self.time_stepper_pt(i)
            was_steady[i]=ts.is_steady()
            ts.make_steady()
        self.actions_before_eigen_solve()
        self.invalidate_cached_mesh_data(only_eigens=True)
        self.setup_forced_zero_dof_list_for_eigenproblems()
        self._last_eigenvalues,self._last_eigenvectors,J,M=self.get_eigen_solver().solve(n,shift=shift,sort=sort,which=which,OPpart=OPpart,v0=v0,target=target)
        self._last_eigenvalues, self._last_eigenvectors=self._last_eigenvalues.copy(),self._last_eigenvectors.copy()
        if filter is not None:
            filtered_indices=numpy.array([filter(ev) for ev in self._last_eigenvalues]).nonzero()
            self._last_eigenvalues=self._last_eigenvalues[filtered_indices]
            self._last_eigenvectors=self._last_eigenvectors[filtered_indices]        

        #self._last_eigenvectors=numpy.transpose(self._last_eigenvectors)
        if (not self.is_quiet()) and (not quiet) :
            if report_accuracy:                
                for i,l in enumerate(self._last_eigenvalues):
                    v=self._last_eigenvectors[i,:]
                    lhs =l*(M@v) #type:ignore
                    rhs=J@v #type:ignore
                    diff=lhs-rhs #type:ignore
                    abs_err=numpy.max(numpy.absolute(diff))
                    rel_err=abs_err/numpy.linalg.norm(v)
                    print("Eigenvalue",i,":",l,"Error (abs/rel):",abs_err,rel_err) #type:ignore
                pass
            else:
                for i,l in enumerate(self._last_eigenvalues):
                    print("Eigenvalue",i,":",l)
        for i in range(ntstep):
            if not was_steady[i]:
                self.time_stepper_pt(i).undo_make_steady()
        return self._last_eigenvalues,self._last_eigenvectors




    def refine_eigenfunction(self, numadapt:int=1,eigenindex:int=0,resolve_base_state:bool=True,resolve_neigen:int=1,use_startvector:bool=False):
        """
        After calculating an eigenproblem, you can adapt the mesh according ot the eigenfunction of a specific eigenvalue.
        This can be useful to refine the mesh in regions where the eigenfunction has a high gradient. It requires SpatialErrorEsitmators to be added to the problem and an adaptive mesh.
        
        Args:
            numadapt: The number of adaptations to perform. Defaults to 1.
            eigenindex: The index of the eigenvalue to refine the mesh for. Defaults to 0.
            resolve_base_state: If True, the base state is resolved after each adaptation. Defaults to True.
            resolve_neigen: The number of eigenvalues to resolve after each adaptation. Defaults to 1.
            
        Returns:
            Tuple[float, NPFloatArray]: The eigenvalue and eigenvector of the adapted eigenproblem.
            
        """
        if eigenindex<0:
            raise ValueError("Eigenindex must be non-negative")
        elif eigenindex>=len(self.get_last_eigenvalues()):
            raise ValueError("Eigenindex must be smaller than the number of calculated eigenvalues")
        self._adapt_eigenindex=eigenindex
        
        for i in range(numadapt):
            with self.custom_adapt(True):
                nref,nunref=self.adapt()
                if nref==0 and nunref==0:                    
                    return self.get_last_eigenvalues()[0],self.get_last_eigenvectors()[0]
            if resolve_base_state:
                self.solve()
            self._adapt_eigenindex=0
            #print("V0",self._adapted_eigeninfo[0])
            #print("V0AMPL",numpy.linalg.norm(self._adapted_eigeninfo[0]))
            if use_startvector:
                startvector=self._adapted_eigeninfo[0].copy()
            else:
                startvector=None
            if self.get_eigen_solver().supports_target():                
                self.solve_eigenproblem(resolve_neigen,v0=startvector,target=self._adapted_eigeninfo[1],shift=self._adapted_eigeninfo[1],azimuthal_m=self._adapted_eigeninfo[2],normal_mode_k=self._adapted_eigeninfo[3])
            else:
                self.solve_eigenproblem(resolve_neigen,v0=startvector,azimuthal_m=self._adapted_eigeninfo[2],normal_mode_k=self._adapted_eigeninfo[3])
        self._adapted_eigeninfo=None
        self._adapt_eigenindex=None
        return self.get_last_eigenvalues()[0],self.get_last_eigenvectors()[0]
            


    def _override_for_this_solve(self,*,max_newton_iterations:Optional[int]=None,newton_relaxation_factor:Optional[float]=None,newton_solver_tolerance:Optional[float]=None,globally_convergent_newton:Optional[bool]=None):
        old:Dict[str,Union[int,bool,float]]={}
        if max_newton_iterations is not None:
            old["max_newton_iterations"]=self.max_newton_iterations
            self.max_newton_iterations=max_newton_iterations
        if newton_relaxation_factor is not None:
            old["newton_relaxation_factor"]=self.newton_relaxation_factor
            self.newton_relaxation_factor=newton_relaxation_factor
        if newton_solver_tolerance is not None:
            old["newton_solver_tolerance"]=self.newton_solver_tolerance
            self.newton_solver_tolerance=newton_solver_tolerance
        if globally_convergent_newton is not None:
            old["globally_convergent_newton"]=False
            self._set_globally_convergent_newton_method(globally_convergent_newton)
        return old
    
    def is_global_parameter_used(self,param:Union[str,_pyoomph.GiNaC_GlobalParam])->bool:
        if isinstance(param,str):
            if param not in self.get_global_parameter_names():
                return False
            else:
                param=self.get_global_parameter(param)
                
        def check_interfaces(m:AnySpatialMesh):
            for _,im in m._interfacemeshes.items():
                if im.get_code_gen().get_code().has_parameter_contribution(param.get_name()):
                    return True
                elif check_interfaces(im):
                    return True                
            return False
        
        for _,m in self._meshdict.items():
            if m.get_code_gen().get_code().has_parameter_contribution(param.get_name()):
                return True
            if not isinstance(m,ODEStorageMesh):
                if check_interfaces(m):
                    return True
                
        return False

    def set_arc_length_parameter(self,desired_proportion_of_arc_length:Optional[float]=None,scale_arc_length:Optional[bool]=None,use_FD:Optional[bool]=None,use_continuation_timestepper:Optional[bool]=None,Desired_newton_iterations_ds:Optional[int]=None):
        if desired_proportion_of_arc_length is not None:
            self._set_arclength_parameter("Desired_proportion_of_arc_length",desired_proportion_of_arc_length)
        if scale_arc_length is not None:
            self._set_arclength_parameter("Scale_arc_length",1 if scale_arc_length else 0)
        if use_FD is not None:
            self._set_arclength_parameter("Use_finite_differences_for_continuation_derivatives",1 if use_FD else 0)
        if use_continuation_timestepper is not None:
            self._set_arclength_parameter("Use_continuation_timestepper",1 if use_continuation_timestepper else 0)
        if Desired_newton_iterations_ds is not None:
            self._set_arclength_parameter("Desired_newton_iterations_ds",Desired_newton_iterations_ds)

    def arclength_continuation(self, parameter: Union[str, _pyoomph.GiNaC_GlobalParam], step: float, *,
                              spatial_adapt: int = 0, max_ds: Optional[float] = None,
                              max_newton_iterations: Optional[int] = None,
                              newton_relaxation_factor: Optional[float] = None,
                              newton_solver_tolerance: Optional[float] = None,
                              min_ds: Optional[float] = None, dof_direction: Optional[List[float]] = None,
                              globally_convergent_newton: Optional[bool] = False) -> float:
        """
        Perform arclength continuation on the basis of a given parameter.

        Args:
            parameter (Union[str, _pyoomph.GiNaC_GlobalParam]): The parameter to perform arclength continuation on.
            step (float): The step for the continuation.
            spatial_adapt (int, optional): The level of spatial adaptation. Defaults to 0.
            max_ds (float, optional): The maximum step size. Defaults to None.
            max_newton_iterations (int, optional): The maximum number of Newton iterations. Defaults to None.
            newton_relaxation_factor (float, optional): The relaxation factor for the Newton solver. Defaults to None.
            newton_solver_tolerance (float, optional): The tolerance for the Newton solver. Defaults to None.
            min_ds (float, optional): The minimum step size. Defaults to None.
            dof_direction (List[float], optional): The direction of degrees of freedom. Defaults to None.
            globally_convergent_newton (bool, optional): Whether to use globally convergent Newton solver. Defaults to False.

        Returns:
            float: The new step size for the continuation.
        """
        self._activate_solver_callback()
        self.invalidate_cached_mesh_data()
        if not self.is_initialised():
            self.initialise()
            self._activate_solver_callback()

        step = float(step)
        if max_ds is not None:
            max_ds = float(max_ds)

        if min_ds is not None:
            old_min_ds = self.minimum_arclength_ds
            self.minimum_arclength_ds = min_ds

        if isinstance(parameter, _pyoomph.GiNaC_GlobalParam):
            parameter = parameter.get_name()
            
        if parameter not in self.get_global_parameter_names():
            raise RuntimeError("Cannot perform arclength continuation in parameter '" + parameter + "' since it is not part of the problem")
        
        if self.warn_about_unused_global_parameters and not self.is_global_parameter_used(parameter):
            if self.warn_about_unused_global_parameters=="error":
                raise RuntimeError("Arclength continuation in the global parameter '" + parameter + "', which is used in the problem. This may lead to unexpected behaviour. Set <Problem>.warn_about_unused_global_parameters to False to suppress this error.")
            else:
                print("WARNING: Arclength continuation in the global parameter '" + parameter + "', which is used in the problem. This may lead to unexpected behaviour. Set <Problem>.warn_about_unused_global_parameters to False to suppress this warning.")
                
        if self._bifurcation_tracking_parameter_name is not None:
            if parameter == self._bifurcation_tracking_parameter_name:
                raise RuntimeError("Cannot perform arclength continuation in the global parameter '" + parameter + "' since it is simultaneously used for bifurcation tracking. Continue in a different parameter or call <Problem>.deactivate_bifurcation_tracking() before")

        if self._last_arclength_parameter is not None:
            if self._last_arclength_parameter != parameter:
                self.reset_arc_length_parameters()
        self._last_arclength_parameter = parameter

        if not self.is_quiet():
            print("Continuation in parameter " + parameter + "=" + str(self.get_global_parameter(parameter).value) +
                  " with step " + str(step))
        oldsettings = self._override_for_this_solve(max_newton_iterations=max_newton_iterations,
                                                    newton_relaxation_factor=newton_relaxation_factor,
                                                    newton_solver_tolerance=newton_solver_tolerance,
                                                    globally_convergent_newton=globally_convergent_newton)
        if max_ds is not None:
            if abs(step) > abs(max_ds):
                step = abs(max_ds) * (1 if step > 0 else -1)
        if dof_direction is not None:
            self._set_dof_direction_arclength(dof_direction)  # does not really work

        self.invalidate_eigendata()

        self._solve_in_arclength_conti = parameter
        self.actions_before_stationary_solve()
        newds = self._arc_length_step(parameter, step, spatial_adapt)
        self._last_step_was_stationary = True
        self._solve_in_arclength_conti = None

        if self.get_bifurcation_tracking_mode() != "":
            if  self._bifurcation_tracking_parameter_name== "<LAMBDA_TRACKING>":
                self._last_eigenvalues = numpy.array([self._get_lambda_tracking_real() + self._get_bifurcation_omega() * 1j], dtype=numpy.complex128)  # type:ignore
            else:
                self._last_eigenvalues = numpy.array([0 + self._get_bifurcation_omega() * 1j], dtype=numpy.complex128)  # type:ignore
            self._last_eigenvectors = numpy.array([self._get_bifurcation_eigenvector()], dtype=numpy.complex128)  # type:ignore            
            if self.get_bifurcation_tracking_mode() == "azimuthal":
                assert self._azimuthal_mode_param_m is not None
                self._last_eigenvalues_m = numpy.array([int(self._azimuthal_mode_param_m.value)], dtype=numpy.int32)  # type:ignore
            elif self.get_bifurcation_tracking_mode()=="cartesian_normal_mode":
                    self._last_eigenvalues_k=numpy.array([self._normal_mode_param_k.value]) #type:ignore
            else:
                self._last_eigenvalues_m = None
                self._last_eigenvalues_k = None
                
            self._last_eigenvectors = self.process_eigenvectors(self._last_eigenvectors)

        if not self.is_quiet():
            print("GETTING NEW DS ", newds, "PLANNED", step)
        self._override_for_this_solve(**oldsettings)
        if max_ds is not None:
            if abs(newds) > abs(max_ds):
                newds = abs(max_ds) * (1 if newds > 0 else -1)
        if not self.is_quiet():
            print("RETURNING NEW DS ", newds, "PLANNED", step)

        if min_ds is not None:
            self.minimum_arclength_ds = old_min_ds  # type:ignore
        return newds

    def go_to_param(self, *, reset_pars:bool=True, startstep:Optional[float]=None, call_after_step:Optional[Callable[[float],None]]=None,final_adaptive_solve:Union[bool,int]=False,max_newton_iterations:Optional[int]=None, epsilon:float=1e-6, max_step:Optional[float]=None,**kwargs:float)->None:
        """
        Perform arclength continuation in a parameter until we reach the desired value.

        Args:
            reset_pars (bool, optional): Whether to reset arc length parameters. Defaults to True.
            startstep (float, optional): The initial step size for the parameter continuation. Defaults to None.
            call_after_step (Callable[[float],None], optional): A function to call after each step. If it returns "stop", we stop any further continuation. Defaults to None.
            final_adaptive_solve (Union[bool,int], optional): Whether to perform a final adaptive solve. Defaults to False.
            max_newton_iterations (int, optional): The maximum number of Newton iterations. Defaults to None.
            epsilon: The tolerance for considering as converged to the parameter
            max_step: The maximum step size for the continuation. Defaults to None.
            **kwargs (float): The parameter name and desired value.

        Raises:
            RuntimeError: If more than one parameter is provided.
            RuntimeError: If the specified parameter is not part of the problem.

        Returns:
            None
        """
                        
        if len(kwargs) != 1:
            raise RuntimeError("Please only give one parameter as keyword argument (you might have misspelled an optional keyword argument)!")
        pname:str=""
        desired_val:float=0.0
        for a, b in kwargs.items():
            pname = a
            desired_val = float(b)
        if pname not in self.get_global_parameter_names():
            raise RuntimeError("Cannot go to parameter "+str(pname)+"="+str(desired_val)+", since the parameter '"+str(pname)+"' is not part of the problem. Available parameters are: "+str(self.get_global_parameter_names()))
        if not self.is_initialised():
            self.initialise()

        if self._dof_selector_used is not self._dof_selector:
            self.reset_arc_length_parameters()
            self.reapply_boundary_conditions()
            self.reapply_boundary_conditions() # Must be done twice to correctly setup the eqn_remapping


        ds = desired_val - self.get_global_parameter(pname).value
        if max_step is not None:
            if abs(ds) > abs(max_step):
                ds = abs(max_step) * (1 if ds > 0 else -1)
        if startstep is not None:
            dsold=ds
            ds = float(startstep)
            if dsold*ds<0:
                ds=-ds
            if abs(dsold)<abs(ds):
                ds=abs(dsold)*(-1 if ds<0 else 1)
        while abs(desired_val - self.get_global_parameter(pname).value) > epsilon:
            ds = self.arclength_continuation(pname, ds, max_ds=desired_val - self.get_global_parameter(pname).value,max_newton_iterations=max_newton_iterations)
            #print("AFTER DS WE HAVE NEW DS",ds,"param deriv",self.get_arc_length_parameter_derivative())
            if reset_pars:
                self.reset_arc_length_parameters()
                if ds * (desired_val - self.get_global_parameter(pname).value) < 0:
                    ds *= -1  # Always move towards the parameter
            if call_after_step is not None:
                if call_after_step(ds)=="stop":
                    return
            if max_step is not None:
                if abs(ds) > abs(max_step):
                    ds = abs(max_step) * (1 if ds > 0 else -1)
        self.get_global_parameter(pname).value = desired_val
        if self.max_refinement_level > 0 and final_adaptive_solve:
            if isinstance(final_adaptive_solve,bool) and final_adaptive_solve:
                self.solve(spatial_adapt=self.max_refinement_level,max_newton_iterations=max_newton_iterations)
            else:
                self.solve(spatial_adapt=final_adaptive_solve,max_newton_iterations=max_newton_iterations)


    def invalidate_eigendata(self):
        self._last_eigenvectors:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
        self._last_eigenvalues:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
        self._last_eigenvalues_m=None
        self._last_eigenvalues_k=None

    # Warning: This must be used with "for parameter, eigenvalue in find_bifurcation_via_eigenvalues(...):"
    def find_bifurcation_via_eigenvalues(self, parameter:Union[str,_pyoomph.GiNaC_GlobalParam], initstep:float, shift:Union[None,float,complex]=0, neigen:int=6, spatial_adapt:int=0, epsilon:float=1e-8, reset_arclength:bool=False, max_ds:Optional[Union[float,Callable[[float],float]]]=None, stay_stable_file:Optional[str]=None, before_eigensolving:Optional[Callable[[float],None]]=None, do_solve:bool=True, azimuthal_m:Optional[Union[int,List[int]]]=None, normal_mode_k:ExpressionNumOrNone=None, eigenindex:int=0):
        """
        Approximates a bifurcation point by bisecting on the basis of the eigenvalues.
        Must be called as a generator, e.g.

        .. code-block:: python        
        
            for parameter, eigenvalue in find_bifurcation_via_eigenvalues(...):
                print("Currently at ",parameter,eigenvalue)            

        Parameters:
				parameter (Union[str,_pyoomph.GiNaC_GlobalParam]): The parameter to vary to find a bifurcation. It can be either a string representing the name of a global parameter or a global parameter directly.
				initstep (float): The initial step size for the bisection.
				shift (Union[None,float,complex]): The shift value for the eigenvalue problem. It can be a float, complex number, or None.
				neigen (int): The number of eigenvalues to compute.
				spatial_adapt (int): The spatial adaptation level.
				epsilon (float): The tolerance for determining the real part of an eigenvalue to be close to zero.
				reset_arclength (bool): Whether to reset the arc length parameters after each step.
				max_ds (Optional[Union[float,Callable[[float],float]]]): The maximum step size for the continuation. It can be a float, a callable function that takes the current parameter value and returns a float, or None.
				stay_stable_file (str, optional): The file path to save the state when the solution is stable. If it is unstable, the state is reloaded.
				before_eigensolving (Optional[Callable[[float],None]]): A callable function to be called before solving the eigenvalue problem.
				do_solve (bool): Whether to solve the problem before continuation. If the solution does not depend on the parameter, it can be set to False.
				azimuthal_m (Optional[Union[int,List[int]]]): The azimuthal mode number if you want to find azimuthal perturbations.
                normal_mode_k: The wave number(s) for an additional direction in Cartesian coordinates. Defaults to None, i.e. the base mode.
				eigenindex (int): The index of the eigenvalue to track. Defaults to 0, i.e. the one with the largest real part.

        Yields:
				param(float): The current parameter value.
				eigenvalue (complex): The eigenvalue corresponding to the specified eigenindex.

        Raises:
				RuntimeError: If the eigenindex is greater than or equal to neigen.
				RuntimeError: If the initial solution is already unstable.
        """
       
        
        
        max_ds_func=max_ds
        if isinstance(parameter, str):
            parameter = self.get_global_parameter(parameter)
        param_is_normal_mode_k=False
        if self._normal_mode_param_k is not None and is_zero(parameter- self._normal_mode_param_k,parameters_to_float=False):
            param_is_normal_mode_k=True
        if do_solve:
            self.solve(spatial_adapt=spatial_adapt)
        else:
            if not self.is_initialised():
                self.initialise()
        if eigenindex>=neigen:
            raise RuntimeError("eigenindex must be less than neigen")
        # Get the initial eigenvalues
        if azimuthal_m is not None and  normal_mode_k is not None:                    
            raise ValueError("Cannot specify both azimuthal_m and normal_mode_k")
        if normal_mode_k is not None and not is_zero(normal_mode_k):        
            if param_is_normal_mode_k:
                oldk=self._normal_mode_param_k.value
            evals0, _ = self._solve_normal_mode_eigenproblem(neigen, cartesian_k=normal_mode_k, shift=shift)            
            if param_is_normal_mode_k:
                self._normal_mode_param_k.value=oldk
        else:
            if azimuthal_m is None or azimuthal_m==0 and normal_mode_k is None:
                evals0, _ = self.solve_eigenproblem(neigen, shift=shift)
            else:
                evals0, _ = self._solve_normal_mode_eigenproblem(neigen, azimuthal_m, shift=shift)
        self.invalidate_cached_mesh_data()
        param0 = parameter.value
        sign0 = evals0[eigenindex].real
        if evals0[eigenindex].real > epsilon:
            raise RuntimeError("Starting already with an unstable solution")
        elif evals0[eigenindex].real >= -epsilon:
            yield param0, evals0[eigenindex]
            return
        if stay_stable_file is not None:
            self.save_state(stay_stable_file,relative_to_output=True)
        ds = initstep
        firstSignChange = False
        if reset_arclength:
            self.reset_arc_length_parameters()
        while True:
            ds0 = ds
            if do_solve:
                if callable(max_ds_func):
                    max_ds=max_ds_func(parameter.value)
                    max_ds=abs(max_ds)
                    print("MAX DS SET TO",max_ds)
                ds = self.arclength_continuation(parameter, ds,max_ds=max_ds)
                self.invalidate_cached_mesh_data()
            else:
                parameter.value=parameter.value+ds
                if max_ds is not None:
                    if callable(max_ds_func):
                        max_ds=max_ds_func(parameter.value)
                        max_ds=abs(max_ds)
                        print("MAX DS SET TO",max_ds)
                    ds=min(abs(1.5*ds),max_ds)*(1 if ds>0 else -1)
            if reset_arclength:
                self.reset_arc_length_parameters()
            if before_eigensolving is not None:
                before_eigensolving(param0)
            if normal_mode_k is not None and not is_zero(normal_mode_k):
                if param_is_normal_mode_k:
                    oldk=self._normal_mode_param_k.value
                evals1, _ = self.solve_eigenproblem(neigen, normal_mode_k=normal_mode_k, shift=shift)
                if param_is_normal_mode_k:
                    self._normal_mode_param_k.value=oldk
            else:
                if azimuthal_m is None:
                    evals1, _ = self.solve_eigenproblem(neigen, shift=shift)
                else:
                    evals1,_=self._solve_normal_mode_eigenproblem(neigen, azimuthal_m, shift=shift)
            self.invalidate_cached_mesh_data()
            param1 = parameter.value
            if abs(evals1[eigenindex].real) < epsilon:
                yield param1, evals1[eigenindex]
                return
            sign = evals1[eigenindex].real
            if sign * sign0 < 0:
                if (stay_stable_file is not None) and evals1[eigenindex].real > epsilon:
                    self.load_state(stay_stable_file, relative_to_output=True)
                    self.invalidate_cached_mesh_data()
                    self.reset_arc_length_parameters()
                    # find the intersection with zero by linear approximation
                    # eigenval=(evals1[0].real-evals0[0].real)/(param1-param0)*(p-param0)+evals0[0].real
                    ds=-evals0[eigenindex].real*(param1-param0)/(evals1[eigenindex].real-evals0[eigenindex].real)
                    continue

                firstSignChange = True
                dsmagn = max(abs(ds), abs(ds0))
                ds = -0.5 * dsmagn * (-1 if ds < 0 else 1)
            else:
                if firstSignChange:
                    dsmagn = max(abs(ds), abs(ds0))
                    ds = dsmagn * (-1 if ds < 0 else 1)
                    ds = ds0 - 0.5 * ds
            yield param1, evals1[eigenindex]
            if (stay_stable_file is not None) and evals1[eigenindex].real<epsilon:
                self.save_state(stay_stable_file, relative_to_output=True)
            sign0 = sign
            evals0=evals1
            param0=param1

    def set_max_refinement_level(self,level:int,do_adapt:bool=True):
        if level<0:
            raise RuntimeError("Must be >=0")
        
        def set_level_for_mesh(mesh:AnySpatialMesh,level):            
            assert isinstance(mesh,MeshFromTemplate2d)            
            mesh._templatemesh.get_template()._max_refinement_level=level            
            maxref=0
            for e in mesh.elements():
                maxref=max(maxref,e.refinement_level())
            res=maxref-mesh.max_refinement_level               
            mesh.max_refinement_level=level
            return res
        must_unref=0
        for _n,m in self._meshdict.items():
            if not isinstance(m,ODEStorageMesh):
                must_unref=max(must_unref, set_level_for_mesh(m,level))
        self.max_refinement_level=level
        if must_unref>0 and do_adapt:
            with self.custom_adapt():
                for i in range(must_unref):
                    self.adapt()


    def perturb_dofs(self,dofpert:NPFloatArray):
        """
        Perturbs all degrees of freedom by a given perturbation array (must have the length of :py:meth:`ndof`)

        Args:
            dofpert: Perturbation array to be added to the degrees of freedom (nondimensional)
        """
        dofs,_=self.get_current_dofs()
        self.invalidate_cached_mesh_data()
        self.invalidate_eigendata()
        self.set_current_dofs(numpy.array(dofs)+dofpert) #type:ignore

    def deactivate_bifurcation_tracking(self):
        
        """
        Deactivate bifurcation tracking. Afterwards, the problem can be solved as usual.
        """
        
        last_tracking=self.get_bifurcation_tracking_mode()
        self._start_bifurcation_tracking("","",False,[],[],0.0,{})
        self._bifurcation_tracking_parameter_name=None        
        if last_tracking=="azimuthal":
            self.actions_before_stationary_solve()
            self.reapply_boundary_conditions()
            self.reapply_boundary_conditions()
            self._last_bc_setting="normal"
            self._azimuthal_mode_param_m.value=0

    # Assuming that Re(lambda)=0 is also stable, which is not exactly true
    def is_stable_solution(self)->bool:
        """
        Shortcut to check whether we have a stable solution. This is only possible after calling solve_eigenproblem(...).
        """
        if len(self._last_eigenvalues)==0:
            raise RuntimeError("Can only find out whether a solution is stable after calling solve_eigenproblem(...)")
        if self.get_bifurcation_tracking_mode()!="":
            raise RuntimeError("Cannot find out whether a solution is stable when bifurcation tracking is active")
        return numpy.real(self._last_eigenvalues[0])<=0 #type:ignore

    def guess_nearest_bifurcation_type(self,eigenvector:int=0)->Literal["hopf","fold","pitchfork","azimuthal"]:
        """
        Guesses the nearest bifurcation type based on the last computed eigenvalues. This is only possible after calling solve_eigenproblem(...).
        It cannot guess e.g. pitchfork or transcritical bifurcations, only "hopf" or "fold" - or "azimuthal" if the last eigenvalues correspond to azimuthal modes m!=0.
        Returns:
            str: Guessed bifurcation type
        """
        if len(self._last_eigenvalues)==0:
            raise RuntimeError("Can only guess the closest bifurcation type after calling solve_eigenproblem(...)")
        if self.get_bifurcation_tracking_mode()!="":
            raise RuntimeError("Cannot guess the closest bifurcation type when bifurcation tracking is active")
        if self._last_eigenvalues_m is None or len(self._last_eigenvalues_m)==0 or self._last_eigenvalues_m[eigenvector]==0:
            if self._last_eigenvalues_k is None or len(self._last_eigenvalues_k)==0 or abs(self._last_eigenvalues_k[eigenvector])<1e-7:
                if abs(numpy.imag(self._last_eigenvalues[eigenvector]))<1e-7:
                    return "fold"
                else:
                    return "hopf"
            else:
                return "cartesian_normal_mode"
        else:
            return "azimuthal"


    
    def dof_strings_to_global_equations(self,string_dof_set:Union[str,Set[str],List[str]]):
        """Takes strings like ``"domain/velocity_x"`` and returns a set of global equations

        Args:
            string_dof_set: Degrees of freedom you want to resolve to equation numbers

        Returns:
            Set[int]: Global equation set
        """
        from ..solvers.generic import EigenMatrixSetDofsToZero
        if isinstance(string_dof_set,str):
            string_dof_set=set([string_dof_set])
        elif isinstance(string_dof_set,list):
            string_dof_set=set(string_dof_set)
        resolver=EigenMatrixSetDofsToZero(self,*string_dof_set)
        zeromap:Set[int]=set()
        for d in resolver.doflist:
            eqs=resolver.resolve_equations_by_name(d)
            #print("DOF",d,"EQS",eqs)
            zeromap=zeromap.union(eqs)
        return zeromap
            

    def activate_eigenbranch_tracking(self,branch_type:Optional[Literal["real","complex","normal_mode"]]=None,eigenvector:Optional[int]=None,eigenvalue:Optional[complex]=None):
        """Activates eigenbranch tracking for the specified eigenbranch type. Subsequent calls of solve(...) and arclength_continuation(...) will then track the eigenbranch.
        This is similar to bifurcation tracking, but it does not adjust a parameter to find a bifurcation, i.e. where Re(lambda)=0. Instead, it starts with a eigenvalue/eigenvector pair. Once activated, you can follow the eigenbranch by calling arclength_continuation(...).        
        At each step, the eigenvalue/eigenvector pair will be updated and is available via get_last_eigenvalues()[0] and get_last_eigenvectors()[0].
        
        Args:
            branch_type (Optional[Literal["real", "complex", "normal_mode"]]): The type of eigenbranch to track. Defaults to None, i.e. auto-detect.
            eigenvector (Optional[int]): The previously calculated eigenvector index to use for tracking. Defaults to None, i.e. the eigenvector at index zero.
        """
        self.activate_bifurcation_tracking(None,bifurcation_type=branch_type,eigenvector=eigenvector,eigenvalue_for_branch_tracking=eigenvalue)

    def activate_bifurcation_tracking(self,parameter:Optional[Union[str,_pyoomph.GiNaC_GlobalParam]],bifurcation_type:Optional[Literal["hopf","fold","pitchfork","azimuthal","cartesian_normal_mode"]]=None,blocksolve:bool=False,eigenvector:Optional[Union[NPFloatArray,NPComplexArray,int]]=None,omega:Optional[float]=None,azimuthal_mode:Optional[int]=None,cartesian_wavenumber_k:Optional[ExpressionOrNum]=None,eigenvalue_for_branch_tracking:Optional[complex]=None):
        """
        Activates bifurcation tracking for the specified parameter and bifurcation type. Subsequent calls of solve(...) and arclength_continuation(...) will then track the bifurcation.

        Args:
            parameter: The parameter to change in order to find the bifurcation. If None, we track the current eigenbranch, i.e. Re(lambda) will be found and is not necessarily 0.
            bifurcation_type (Optional[Literal["hopf", "fold", "pitchfork", "azimuthal"]]): The type of bifurcation to track. Defaults to None, i.e. auto-detect.
            blocksolve (bool): Flag indicating whether to use block solve. Defaults to False. Should be kept False.
            eigenvector (Optional[Union[NPFloatArray, NPComplexArray, int]]): The eigenvector to use for tracking. Defaults to None, which means the eigenvector corresponding to the eigenvalue with largest real part. Can be either an index or a custom vector.
            omega (Optional[float]): The omega value for Hopf bifurcation tracking. Defaults to None, then it will be Im(lambda).
            azimuthal_mode (Optional[int]): The azimuthal mode for azimuthal bifurcation tracking. Defaults to None.
        """        
        

        if parameter is None:
            # We track the current eigenbranch, i.e. Re(lambda) will be found and is not necessarily 0
            parameter="<LAMBDA_TRACKING>"
            eigenvector_v=None
            if eigenvector is None:
                eigenvector=0
            if isinstance(eigenvector,int):
                if eigenvector>=len(self.get_last_eigenvectors()):
                    raise RuntimeError("Eigenvector "+str(eigenvector)+" not calculated")
                self._set_lambda_tracking_real(numpy.real(self.get_last_eigenvalues()[eigenvector]))    
                eigenvector_v=self.get_last_eigenvectors()[eigenvector]
            else:
                #raise RuntimeError("Can only track eigenbranches, not custom vectors. Please set eigenvector to and integer (for the index of the calculate eigenvector or None, meaning index 0) ")
                eigenvector_v=eigenvector
                if eigenvalue_for_branch_tracking is None:
                    raise RuntimeError("Please set eigenvalue_for_branch_tracking if you track a custom eigenvector")
                self._set_lambda_tracking_real(numpy.real(eigenvalue_for_branch_tracking))
                omega=numpy.imag(eigenvalue_for_branch_tracking)
                #if bifurcation_type is None:
                #    if numpy.abs(numpy.imag(eigenvalue_for_branch_tracking))>1e-6:
                #        bifurcation_type="hopf"
            if bifurcation_type is None:
                bifurcation_type=self.guess_nearest_bifurcation_type(eigenvector)
            
            if bifurcation_type=="fold" or bifurcation_type=="real":
                if omega is not None and omega!=0:
                    raise RuntimeError("Cannot track eigenbranch for a real branch with a non-zero omega")
                if azimuthal_mode is not None and azimuthal_mode!=0:
                    raise RuntimeError("Cannot track eigenbranch for a real branch with a non-zero azimuthal mode")
                if cartesian_wavenumber_k is not None and not is_zero(cartesian_wavenumber_k):
                    raise RuntimeError("Cannot track eigenbranch for a real branch with a non-zero cartesian wavenumber")
                bifurcation_type="fold" # Use the modified fold tracker for this
                print("Activating eigenbranch tracking for a real branch with starting eigenvalue",self._get_lambda_tracking_real())
            elif bifurcation_type=="hopf" or bifurcation_type=="complex":                
                if azimuthal_mode is not None and azimuthal_mode!=0:
                    raise RuntimeError("Cannot track eigenbranch for a complex branch with a non-zero azimuthal mode. Use normal_mode instead")
                if cartesian_wavenumber_k is not None and not is_zero(cartesian_wavenumber_k):
                    raise RuntimeError("Cannot track eigenbranch for a complex branch with a non-zero additional cartesian wavenumber. Use normal_mode instead")
                bifurcation_type="hopf"
                if omega is None:
                    omega=numpy.imag(self.get_last_eigenvalues()[eigenvector])
                print("Activating eigenbranch tracking for a complex branch with with starting eigenvalue",str(complex(self._get_lambda_tracking_real(),omega)))
            elif bifurcation_type=="azimuthal" or bifurcation_type=="cartesian_normal_mode" or bifurcation_type=="normal_mode":
                if azimuthal_mode is None and cartesian_wavenumber_k is None:
                    if self.get_last_eigenmodes_k() is not None and len(self.get_last_eigenmodes_k())>eigenvector and not is_zero(self.get_last_eigenmodes_k()[eigenvector]):                        
                        bifurcation_type="cartesian_normal_mode"                        
                        cartesian_wavenumber_k=self.get_last_eigenmodes_k()[eigenvector]
                    elif self.get_last_eigenmodes_m() is not None and len(self.get_last_eigenmodes_m()>eigenvector):
                        bifurcation_type="azimuthal"
                        azimuthal_mode=self.get_last_eigenmodes_m()[eigenvector]
                elif azimuthal_mode is not None and cartesian_wavenumber_k is not None:
                    raise RuntimeError("Cannot track eigenbranch for both azimuthal and cartesian normal mode")
                elif azimuthal_mode is not None:
                    bifurcation_type="azimuthal"
                else:
                    bifurcation_type="cartesian_normal_mode"
                if omega is None:
                    omega=numpy.imag(self.get_last_eigenvalues()[eigenvector])
                if azimuthal_mode is not None:
                    print("Activating eigenbranch tracking for a azimuthal branch with m="+str(azimuthal_mode)+" with with starting eigenvalue",str(complex(self._get_lambda_tracking_real(),omega)))
                else:
                    print("Activating eigenbranch tracking for an normal Cartesian mode branch with k="+str(cartesian_wavenumber_k)+" with with starting eigenvalue",str(complex(self._get_lambda_tracking_real(),omega)))
            else:                
                raise RuntimeError("Cannot track eigenbranch for bifurcation type "+bifurcation_type)
            if eigenvector_v is not None:
                eigenvector=eigenvector_v
        else:
            if eigenvalue_for_branch_tracking is not None:
                raise RuntimeError("Cannot use eigenvalue_for_branch_tracking except for eigenbranch continuation")
            if isinstance(eigenvector,int):
                if eigenvector>=len(self.get_last_eigenvectors()):
                    raise RuntimeError("Eigenvector "+str(eigenvector)+" not calculated")
                if bifurcation_type is None:
                    bifurcation_type=self.guess_nearest_bifurcation_type(eigenvector)
                    print("Assuming nearest bifurcation is of type: "+bifurcation_type)
                if omega is None and bifurcation_type in {"hopf","azimuthal","cartesian_normal_mode"}:
                    omega=numpy.imag(self.get_last_eigenvalues()[eigenvector])
                if bifurcation_type=="azimuthal" and azimuthal_mode is None:
                    azimuthal_mode=self.get_last_eigenmodes_m()[eigenvector]
                elif bifurcation_type=="cartesian_normal_mode" and cartesian_wavenumber_k is None:
                    cartesian_wavenumber_k=self.get_last_eigenmodes_k()[eigenvector]
                eigenvector=self.get_last_eigenvectors()[eigenvector]
                
            if bifurcation_type is None:
                bifurcation_type=self.guess_nearest_bifurcation_type()
                print("Assuming nearest bifurcation is of type: "+bifurcation_type)

            if self._dof_selector_used is not self._dof_selector:
                self.reapply_boundary_conditions()
                self.reapply_boundary_conditions()
            if isinstance(parameter,_pyoomph.GiNaC_GlobalParam):
                parameter=parameter.get_name()
            
            if not parameter in self.get_global_parameter_names():
                raise RuntimeError("Cannot perform bifurcation tracking in parameter '"+parameter+"' since it is not part of the problem")
            
            if self.warn_about_unused_global_parameters and not self.is_global_parameter_used(parameter):
                if self.warn_about_unused_global_parameters=="error":
                    raise RuntimeError("Bifurcation tracking in the global parameter '" + parameter + "', which is used in the problem. This may lead to unexpected behaviour. Set <Problem>.warn_about_unused_global_parameters to False to suppress this error.")
                else:
                    print("WARNING: Bifurcation tracking in the global parameter '" + parameter + "', which is used in the problem. This may lead to unexpected behaviour. Set <Problem>.warn_about_unused_global_parameters to False to suppress this warning.")
            if not self.is_quiet():
                print("Bifurcation tracking activated for "+parameter)
        self._bifurcation_tracking_parameter_name=parameter
        if bifurcation_type=="fold":
#            must_reapply_bcs=self._equation_system._before_eigen_solve(self.get_eigen_solver(), 0)
#            if must_reapply_bcs:
#                self.reapply_boundary_conditions() # Equation numbering might have been changed. Update it here!
#                self._last_bc_setting="eigen"
            if azimuthal_mode is not None or cartesian_wavenumber_k is not None:
                raise RuntimeError("Cannot use azimuthal_mode or cartesian_wavenumber_k for fold solving")
            if eigenvector is None:
                eigenvector = next(iter(self.get_last_eigenvectors()), None)
            if eigenvector is None or len(eigenvector)==0:
                self._start_bifurcation_tracking(parameter,bifurcation_type,blocksolve,[],[],0.0,{})
            else:
                self._start_bifurcation_tracking(parameter,bifurcation_type,blocksolve,numpy.real(eigenvector),[],0.0,{}) #type:ignore
        elif bifurcation_type=="hopf":
            if azimuthal_mode is not None or cartesian_wavenumber_k is not None:
                raise RuntimeError("Cannot use azimuthal_mode or cartesian_wavenumber_k for Hopf solving")
            if eigenvector is None:
                eigenvector=next(iter(self.get_last_eigenvectors()),None)
            if omega is None:
                omega=next(iter(self.get_last_eigenvalues()),None)
                if omega is not None:
                    omega=numpy.imag(omega) #type:ignore
            if eigenvector is None:
                raise RuntimeError("Please pass the kwarg eigenvector to the bifurcation tracking for Hopf bifurcations")
            elif omega is None:
                raise RuntimeError("Please pass a guess to omega for a Hopf bifurcation")
            elif float(omega)==0.0:
                raise RuntimeError("Hopf bifurcation cannot have zero complex part of the eigenvalue")
            else:
                #if not self.is_quiet():
                    #print("OMEGA",omega,numpy.real(eigenvector),numpy.imag(eigenvector))
                    #print("PARAMDERIV",self.get_parameter_derivative(parameter))
                #print("STARTING WITH OMEGA=",omega)
                #eigenvector=prerotate_eigenvector(eigenvector)
                #eigenvector = prerotate_eigenvector(eigenvector)
                #print(eigenvector)

                self._start_bifurcation_tracking(parameter,bifurcation_type,blocksolve,numpy.real(eigenvector),numpy.imag(eigenvector),omega,{}) #type:ignore
        elif bifurcation_type=="pitchfork":
            if azimuthal_mode is not None or cartesian_wavenumber_k is not None:
                raise RuntimeError("Cannot use azimuthal_mode or cartesian_wavenumber_k for pitchfork solving")
            if eigenvector is None:
                eigenvector=next(iter(self.get_last_eigenvectors()),None)
            if eigenvector is None:
                raise RuntimeError("Pitchfork tracking requires at least a symmetry vector passed via the eigenvector kwarg")
            self._start_bifurcation_tracking(parameter,bifurcation_type,blocksolve,numpy.real(eigenvector),[],0.0,{}) #type:ignore
        elif bifurcation_type=="azimuthal":
            if self._azimuthal_mode_param_m is None:
                raise RuntimeError("Cannot use azimuthal bifurcation tracking if not called setup_for_stability_analysis(azimuthal_stability=True) before")
            if azimuthal_mode is None:
                # Try to get the most unstable mode
                if self._last_eigenvalues_m is None or len(self._last_eigenvalues_m)==0:
                    raise RuntimeError("Must specify azimuthal_mode or solve an azimuthal eigenproblem before")
                azimuthal_mode=self._last_eigenvalues_m[0]
                assert azimuthal_mode is not None
            self._azimuthal_mode_param_m.value=azimuthal_mode
            
        
            if eigenvector is None:
                if self._last_eigenvalues_m is None or len(self._last_eigenvalues_m) == 0:
                    raise RuntimeError("Cannot find a good eigenvector guess since you have not calculated any one for mode "+str(azimuthal_mode))
                # Try to find an eigenvector corresponding to this mode
                eigenindices = numpy.where(numpy.array(self._last_eigenvalues_m)==azimuthal_mode)[0] #type:ignore
                if len(eigenindices)==0:
                    raise RuntimeError("Cannot find a good eigenvector guess since you have not calculated any one for mode " + str(azimuthal_mode))
                eigenvector = self.get_last_eigenvectors()[eigenindices[0]]
                if omega is None:
                    omega=numpy.imag(self.get_last_eigenvalues()[eigenindices[0]]) #type:ignore
            else:
                if omega is None:
                    omega = next(iter(self.get_last_eigenvalues()), None)
                if omega is not None:
                    pass
                else:
                    omega = 0

            # First, we get all equations which must be zero for the base state and on the eigenvector
            must_reapply_bcs=self._equation_system._before_eigen_solve(self.get_eigen_solver(), azimuthal_mode)
            if must_reapply_bcs:
                self.reapply_boundary_conditions() # Equation numbering might have been changed. Update it here!
                self.reapply_boundary_conditions()
                self._last_bc_setting="eigen"
            

            

            #print("BASE DOFS")
            base_zero_dofs=self._equation_system._get_forced_zero_dofs_for_eigenproblem(self.get_eigen_solver(),0,None)             
            base_zero_dofs=self.dof_strings_to_global_equations(base_zero_dofs)
            
            #print("EIGEN DOFS")
            eigen_zero_dofs=self._equation_system._get_forced_zero_dofs_for_eigenproblem(self.get_eigen_solver(),azimuthal_mode,None) 
            eigen_zero_dofs=self.dof_strings_to_global_equations(eigen_zero_dofs)


            contribs={"azimuthal_real_eigen":self._azimuthal_stability.real_contribution_name,"azimuthal_imag_eigen":self._azimuthal_stability.imag_contribution_name}

  
            self._start_bifurcation_tracking(parameter, bifurcation_type, blocksolve, numpy.real(eigenvector),numpy.imag(eigenvector), omega,contribs) #type:ignore
            self.assembly_handler_pt().set_global_equations_forced_zero(base_zero_dofs,eigen_zero_dofs) #type:ignore
            
        elif bifurcation_type=="cartesian_normal_mode":            
            if self._normal_mode_param_k is None:
                raise RuntimeError("Cannot use Cartesian normal mode bifurcation tracking if not called setup_for_stability_analysis(additional_cartesian_mode=True) before")
            if cartesian_wavenumber_k is None:
                # Try to get the most unstable mode
                if self._last_eigenvalues_k is None or len(self._last_eigenvalues_k)==0:
                    raise RuntimeError("Must specify cartesian_wavenumber_k or solve an normal mode eigenproblem before")
                cartesian_wavenumber_k=self._last_eigenvalues_k[0]
                assert cartesian_wavenumber_k is not None
            self._normal_mode_param_k.value=cartesian_wavenumber_k
            if eigenvector is None:
                if self._last_eigenvalues_k is None or len(self._last_eigenvalues_k) == 0:
                    raise RuntimeError("Cannot find a good eigenvector guess since you have not calculated any one for wave number "+str(cartesian_wavenumber_k))
                # Try to find an eigenvector corresponding to this mode
                eigenindices = numpy.where(numpy.array(self._last_eigenvalues_k)==cartesian_wavenumber_k)[0] #type:ignore
                if len(eigenindices)==0:
                    raise RuntimeError("Cannot find a good eigenvector guess since you have not calculated any one for wave number " + str(cartesian_wavenumber_k))
                eigenvector = self.get_last_eigenvectors()[eigenindices[0]]
                if omega is None:
                    omega=numpy.imag(self.get_last_eigenvalues()[eigenindices[0]]) #type:ignore
            else:
                if omega is None:
                    omega = next(iter(self.get_last_eigenvalues()), None)                
                if omega is not None:
                    pass
                    #omega = numpy.imag(omega) #type:ignore
                else:
                    omega = 0

            # First, we get all equations which must be zero for the base state and on the eigenvector
            must_reapply_bcs=self._equation_system._before_eigen_solve(self.get_eigen_solver(), normal_k=cartesian_wavenumber_k)
            if must_reapply_bcs:
                self.reapply_boundary_conditions() # Equation numbering might have been changed. Update it here!
                self.reapply_boundary_conditions()
                self._last_bc_setting="eigen"
            base_zero_dofs=self._equation_system._get_forced_zero_dofs_for_eigenproblem(self.get_eigen_solver(),None,None) 
            eigen_zero_dofs=self._equation_system._get_forced_zero_dofs_for_eigenproblem(self.get_eigen_solver(),None,cartesian_wavenumber_k) 

            

            base_zero_dofs=self.dof_strings_to_global_equations(base_zero_dofs)
            eigen_zero_dofs=self.dof_strings_to_global_equations(eigen_zero_dofs)

            #print("BASE DOFS",base_zero_dofs)
            #print("EIGEN DOFS",eigen_zero_dofs)
            #print("OMEGA {:g}".format(omega))
            contribs={"azimuthal_real_eigen":self._cartesian_normal_mode_stability.real_contribution_name,"azimuthal_imag_eigen":self._cartesian_normal_mode_stability.imag_contribution_name}
            has_imag=self._set_solved_residual(self._cartesian_normal_mode_stability.imag_contribution_name,raise_error=False)
            if not has_imag:
                contribs["azimuthal_imag_eigen"]="<NONE>"
            self._set_solved_residual("")
            #print("GOING FOR IT ",parameter, bifurcation_type, blocksolve,  -omega,contribs)
            #print("KVALUE",self._normal_mode_param_k.value,"HAS IMAG",has_imag)
            
            self._start_bifurcation_tracking(parameter, bifurcation_type, blocksolve, numpy.real(eigenvector),numpy.imag(eigenvector), omega,contribs) #type:ignore
            self.assembly_handler_pt().set_global_equations_forced_zero(base_zero_dofs,eigen_zero_dofs) #type:ignore            
            
        else:
            raise ValueError("Unknown bifurcation type:"+str(bifurcation_type))


    def activate_periodic_orbit_handler(self,T:ExpressionOrNum,history_dofs=[],mode:Literal["collocation","floquet","bspline","central","BDF2"]="collocation",  order:int=2,GL_order:int=-1,T_constraint:Literal["plane","phase"]="phase")->PeriodicOrbit:
        """
        Activates periodic orbit tracking based on history dofs. Use :py:meth:`set_current_dofs` to set the first time point of the orbit guess. The other time points must be shipped with the history_dofs argument.
        
        Args:
            T: The guessed period of the orbit
            history_dofs: The history dofs to use for the orbit tracking. Must be non-empty.
            mode: The mode of the time discretization.
            order: The order of the time discretization.
            GL_order: The Gauss-Legendre order for some time discretization modes. Defaults to -1, meaning a suitable integration order is chosen automatically based on the interpolation order.
            T_constraint: The constraint for the period. Defaults to "phase".
            
        Returns:
            PeriodicOrbit: The resulting periodic orbit. Note that it still must be solved, i.e. it is only the provided guess at this stage.
        """
        self.deactivate_bifurcation_tracking()        
        self.time_stepper_pt().make_steady()
        if len(history_dofs)==0:
            raise ValueError("No history dofs provided")
        knots=[]
        if T_constraint=="plane":
            T_constraint=0
        elif T_constraint=="phase":
            T_constraint=1
        else:
            raise ValueError("Invalid T_constraint: "+str(T_constraint))
        T_nd=float(T/self.get_scaling("temporal"))
        if mode=="floquet":
            self._start_orbit_tracking(history_dofs,T_nd,0,-1,knots,T_constraint)
        elif mode=="bspline":
            if order<1:
                raise ValueError("Invalid bspline order: "+str(order))
            self._start_orbit_tracking(history_dofs,T_nd,order,GL_order,knots,T_constraint)
        elif mode=="central":
            self._start_orbit_tracking(history_dofs,T_nd,-1,-1,knots,T_constraint)
        elif mode=="BDF2":
            self._start_orbit_tracking(history_dofs,T_nd,-2,-1,knots,T_constraint)
        elif mode=="collocation":
            if order<1:
                raise ValueError("Invalid collocation order: "+str(order))
            self._start_orbit_tracking(history_dofs,T_nd,-2-order,GL_order,knots,T_constraint)
        else:
            raise ValueError("Invalid mode: "+str(mode))
        res=PeriodicOrbit(self,mode,0,None,0,None,None,0,order,GL_order,T_constraint)
        return res

    def switch_to_hopf_orbit(self,eps:float=0.01,dparam:Optional[float]=None,NT:int=30,mode:Literal["collocation","floquet","central","BDF2","bspline"]="collocation",order:int=3,GL_order:int=-1,T_constraint:Literal["phase","plane"]="phase",amplitude_factor:float=1,FD_delta:float=1e-5,FD_param_delta=1e-3,do_solve:bool=True,solve_kwargs:Dict[str,Any]={},check_collapse_to_stationary:bool=True,orbit_amplitude:Optional[float]=None,patch_number_of_nodes:bool=True)->PeriodicOrbit:
        """After solving for a Hopf bifurcation by bifurcation tracking, this method will calculate the first Lyapunov exponent and initializes a good guess for the tracking of the periodic orbits originating at this Hopf bifurcation.
        
        It is best to call it like:
        
            with problem.switch_to_hopf_orbit(...) as orbit:
                ...
                
        to deactivate orbit tracking after the with-statement.

        Args:
            eps: A small number to construct the initial guess of the orbit and shift the parameter accordingly. Defaults to 0.01.
            dparam: Optional parameter shift. If given and orbit_amplitude is also given, eps is ignored. Defaults to None.
            NT: Number of discrete time steps to consider for the orbit. Defaults to 30.
            mode: Selects the time discretization and interpolation mode. Defaults to "collocation".
            order: Selects the order of the time discretization method. Defaults to 3.
            GL_order: Selects the Gauss-Legendre integration order for some time discretization modes. Defaults to -1, which is auto-select depending on the order.
            T_constraint: Either use the "plane" or the "phase" constraint as equation for T. Defaults to "phase".
            amplitude_factor: Additional multiplicative factor for the amplitude of the orbit guess. Defaults to 1.
            FD_delta: Finite difference step for the third order calculations used in the determination of the first Lyapunov coefficient. Defaults to 1e-5.
            FD_param_delta: Finite difference step to determine the change of the real part of the eigenvalue with respect to the parameter. Defaults to 1e-3.
            do_solve: Solve the orbit guess. Defaults to True.
            solve_kwargs: Additional keywords arguments to pass to the solve method for the initial solve. Defaults to {}.
            check_collapse_to_stationary: Since an orbit can collapse to the stationary Hopf branch, we can check for it to make sure we are actually on an orbit. Defaults to True.
            orbit_amplitude: Amplitude for the orbit. If set together with dparam, eps is ignored. Defaults to None.
            patch_number_of_nodes: Depending on the order, we might have to slightly modify NT to have the right number of time nodes. Defaults to True.

        

        Returns:
            PeriodicOrbit: The periodic orbit object
        """
        
        from pyoomph.generic.bifurcation_tools import get_hopf_lyapunov_coefficient    
        
        if self._bifurcation_tracking_parameter_name is None or self.get_bifurcation_tracking_mode()!="hopf" or len(self.get_last_eigenvalues())!=1:
            raise ValueError("Hopf bifurcation tracking not activated or solved. Please call activate_bifurcation_tracking first, then solve. Then call this routine.")        
        # Store the information from the Hopf tracker
        omega=self.get_last_eigenvalues()[0].imag                            
        #q=self.get_last_eigenvectors()[0]
        
        
        q=self.assembly_handler_pt().get_nicely_rotated_eigenfunction()
        if omega<0:
            omega=-omega
            q=numpy.conj(q)
        
        param=self._bifurcation_tracking_parameter_name
        parameter=self.get_global_parameter(param)
        pvalue=self.get_global_parameter(param).value
        # Deactivate the bifurcation tracking
        self.deactivate_bifurcation_tracking()
        self.timestepper.make_steady()
        #self.solve()
        # Get the Lyapunov coefficient
        if dparam is not None and orbit_amplitude is not None:
            parameter.value+=dparam
            sign=1 if dparam>0 else 0
            al=orbit_amplitude
            qR,qI=numpy.real(q),numpy.imag(q)
            lyap_coeff=0
        else:
            lyap_coeff,sign,al,qR,qI=get_hopf_lyapunov_coefficient(self,param,omega=omega,q=q,FD_delta=FD_delta,FD_param_delta=FD_param_delta)
            print("AL",al,"QR MAGNITUDE",numpy.linalg.norm(qR+1j*qI))
            if dparam:
                eps=numpy.sqrt(abs(dparam))        
            parameter.value+=-eps**2*sign
        u0=self.get_current_dofs()[0]
        
        if patch_number_of_nodes and mode=="collocation":            
            if order<=0:
                raise RuntimeError("Invalid order for collocation")
            if NT%order!=0:
                NT=(((NT)//order)+1)*order
            
            
        
        T=2*numpy.pi/omega*self.get_scaling("temporal")
        upert=lambda t: u0+2*eps*al*amplitude_factor*numpy.real(numpy.exp(1j*omega*t)*(qR+1j*qI))
        print("Amplitude perturbation factor:",2*eps*al*amplitude_factor)
        print("Parameter step",-eps**2*sign)
        history_dofs=[]
        for t in numpy.linspace(0,2*numpy.pi/omega,NT,endpoint=False):
            history_dofs.append(upert(t))        
        self.set_current_dofs(history_dofs[0])
        self.activate_periodic_orbit_handler(T,history_dofs[1:],mode,order=order,GL_order=GL_order,T_constraint=T_constraint)
        history_dofs.append(history_dofs[0])
        res=PeriodicOrbit(self,mode,lyap_coeff,param,omega,pvalue,parameter.value,al,order,GL_order,T_constraint)
        if check_collapse_to_stationary:
            avg_dists0=0
            ncnt=0            
            for T in res.iterate_over_samples():
                dofs=self.get_current_dofs()[0][:self.assembly_handler_pt().get_base_ndof()]
                avg_dists0+=(numpy.dot(numpy.array(history_dofs[ncnt])-numpy.array(u0),numpy.array(dofs)-numpy.array(u0)))
                ncnt+=1
            avg_dists0/=ncnt
                
            
        if do_solve:
            self.solve(**solve_kwargs)
            if check_collapse_to_stationary:
                avg_dists=0
                i=0
                for T in res.iterate_over_samples():
                    dofs=self.get_current_dofs()[0][:self.assembly_handler_pt().get_base_ndof()]
                    add=numpy.dot(numpy.array(history_dofs[i])-numpy.array(u0),numpy.array(dofs)-numpy.array(u0))
                    #print("adding",add,numpy.amax(numpy.absolute(numpy.array(history_dofs[i])-numpy.array(u0))))
                    avg_dists+=add
                    i+=1
                avg_dists/=ncnt
                print("Average 'radius'^2 of the starting guess orbit:",avg_dists0)
                print("Average 'radius'^2 of the solved guess orbit:",avg_dists)
                if avg_dists<1e-10*avg_dists0:
                    raise RuntimeError("The solved orbit is likely collapsed")
                
                start=None
                nontrivial=False
                i=0
                skip=False
                for T in res.iterate_over_samples():
                    if skip:
                        continue
                    if i==0:
                        start=self.get_current_dofs()[0][:self.assembly_handler_pt().get_base_ndof()]
                    else:
                        dist=numpy.linalg.norm(numpy.array(start)-numpy.array(self.get_current_dofs()[0][:self.assembly_handler_pt().get_base_ndof()]))
                        if dist>1e-5*avg_dists0:
                            nontrivial=True
                            skip=True
                    i+=1
                if not nontrivial:
                    raise RuntimeError("The solved orbit is likely collapsed")
                    #print("DOT",numpy.sqrt(numpy.dot(numpy.array(history_dofs[i])-numpy.array(u0),numpy.array(dofs)-numpy.array(u0))))
        return res
        
    def get_floquet_multipliers(self,n:Optional[int]=None,valid_threshold:Optional[float]=10000,shift:Optional[Union[float]]=None,ignore_periodic_unity:Union[bool,float]=False,quiet:bool=True)->NPComplexArray:
        """
        TODO; Add documentation
        """
        # Main ideas from here: https://arxiv.org/html/2407.18230v1#S2.E6
        import _pyoomph
        import scipy
        if not isinstance(self.assembly_handler_pt(),_pyoomph.PeriodicOrbitHandler):
            raise RuntimeError("Periodic orbit handler not active. Call activate_periodic_orbit_handler first, then solve the orbit, then call this function")
        if not self.assembly_handler_pt().is_floquet_mode():
            raise RuntimeError("Floquet mode not active. Call activate_periodic_orbit_handler with mode='floquet' first, then solve the orbit, then call this function")
        nbase=self.assembly_handler_pt().get_base_ndof()
        if n is None:
            n=nbase
        if n<=0:
            raise ValueError("Invalid number of Floquet multipliers requested: "+str(n))
        
        Jfull=self.assemble_jacobian(with_residual=False)        
        nMat=Jfull.shape[0]-1
        Jfull=Jfull[:nMat,:nMat] # Remove the T equation        
        Mdiag=numpy.zeros(nMat)
        Mdiag[nMat-nbase:]=1.0 
        Mfull=scipy.sparse.csr_matrix(scipy.sparse.diags_array(Mdiag).tocsr()) # Make the mass matrix         
        eigs,eigv,_,_=self.get_eigen_solver().solve(neval=n,custom_J_and_M=(Jfull,Mfull),shift=shift,quiet=quiet) # Solve the eigenproblem
        valid_eigs=numpy.array([e for e in eigs if numpy.isfinite(e) and not numpy.isnan(e)])
        if valid_threshold is not None:
            valid_inds=numpy.argwhere(numpy.abs(valid_eigs)<valid_threshold).flatten()            
            eigv=eigv[valid_inds,:]
            valid_eigs=valid_eigs[valid_inds]        
        gamms=1/(1-valid_eigs)
        
        if ignore_periodic_unity is True:
            ignore_periodic_unity=1e-5
        if ignore_periodic_unity is not False:
            unity_eigval=numpy.argwhere(numpy.abs(gamms-1)<ignore_periodic_unity).flatten()            
            if unity_eigval.size>0:
                if unity_eigval.size>1:
                    print("WARNING: Found multiple unity Floquet multipliers. Usually, only one is present (except at distinct bifurcations of the orbit) ")
                gamms=numpy.delete(gamms,unity_eigval)
                eigv=numpy.delete(eigv,unity_eigval,axis=0)  # TODO: Check if this is correct
        # Sort by magnitude
        sortinds=numpy.argsort(numpy.abs(gamms))
        gamms=gamms[sortinds]
        eigv=eigv[sortinds,:]
        self._last_eigenvalues=gamms
        self._last_eigenvectors=numpy.c_[eigv,numpy.zeros(eigv.shape[0])]
        self._last_eigenvalues_m=None
        self._last_eigenvalues_k=None
        return gamms

    def get_last_eigenvalues(self,dimensional:bool=False)->NPComplexArray:
        """Returns the last computed eigenvalues.

        Returns:
            NPComplexArray: Eigenvalues as array.
        """                
        if dimensional:
            if self._last_eigenvalues is None:
                return None
            else:
                imaginary_i=_pyoomph.GiNaC_imaginary_i()
                return [numpy.real(x)/self.get_scaling("temporal")+imaginary_i*numpy.imag(x)/self.get_scaling("temporal") for x in self._last_eigenvalues]                
        return self._last_eigenvalues

    def get_last_eigenvectors(self)->NPComplexArray:
        """Return the last computed eigenvector.

        Returns:
            NPComplexArray: Eigenvectors as 2d array.
        """
        return self._last_eigenvectors

    def get_last_eigenmodes_m(self) -> Optional[NPIntArray]:
        """Get the azimuthal mode numbers for the last computed eigenvalues.

        Returns:
            Optional[NPIntArray]: Array containing the azimuthal mode numbers corresponding to the eigenvalues.
        """
        return self._last_eigenvalues_m
    
    def get_last_eigenmodes_k(self)->Optional[NPFloatArray]:
        """Get the cartesian normal mode numbers for the last computed eigenvalues.

        Returns:
            Optional[NPFloatArray]: Array containing the cartesian normal mode numbers corresponding to the eigenvalues.
        """
        return self._last_eigenvalues_k


    def rotate_eigenvectors(self,eigenvectors,dofs_to_real:Union[str,List[str],Set[str]],normalize_dofs:bool=False,normalize_amplitude:Union[float,complex]=1,normalize_max:bool=True):
        """
        Should be called within the method :py:meth:`process_eigenvectors` to rotate the eigenvectors to e.g. a common phase. 
        This is optional, but avoids phase jumps in the eigenvectors when following an eigenbranch.

        Args:
            eigenvectors: Eigenvectors to rotate, usually the ones passed in the automatically method :py:meth:`process_eigenvectors`.
            dofs_to_real: Which degrees of freedom to consider to find the phase. Can be a single string, a list of strings or a set of strings.
            normalize_dofs: Normalizes the eigenvector with respect to the selected dofs as well. Defaults to False.
            normalize_amplitude: If normalization is active, we can scale the overall magnitude of the eigenvector by this value. Defaults to 1.
            normalize_max: If True, we normalize by the maximum magnitude of the listed dofs, otherwise by the average magnitude. Defaults to True.

        Returns:
            The processed eigenvectors, return it as result of the method :py:meth:`process_eigenvectors`.
        """
        neweigen=[]
        dofs=self.dof_strings_to_global_equations(dofs_to_real)
        dofs=numpy.array(list(dofs),dtype=numpy.int64)
        for ev in eigenvectors:
            avg_angle=numpy.angle(numpy.average(ev[dofs]))
            #print("AVERAGE ANGLE",avg_angle)
            if normalize_dofs:
                if normalize_max:
                    magnitude=numpy.amax(numpy.absolute(ev[dofs]))
                else:
                    magnitude=numpy.average(numpy.absolute(ev[dofs]))
            else:
                magnitude=1
            #print("AMPLITUDE",normalize_amplitude/magnitude)
            neweigen.append(ev*numpy.exp(-1j*avg_angle)/magnitude*normalize_amplitude)            
        return numpy.array(neweigen)

    
    def define_problem_for_axial_symmetry_breaking_investigation(self):
        from ..expressions.coordsys import AxisymmetryBreakingCoordinateSystem
        self._azimuthal_mode_param_m = self.get_global_parameter(self._azimuthal_stability.azimuthal_param_m_name)
        coordsys = AxisymmetryBreakingCoordinateSystem(self._azimuthal_mode_param_m.get_symbol())
        oldcoordsys=self.get_coordinate_system()
        if oldcoordsys is not None:
            if isinstance(oldcoordsys,AxisymmetryBreakingCoordinateSystem):
                coordsys.cartesian_error_estimation=oldcoordsys.cartesian_error_estimation
        self.set_coordinate_system(coordsys)

        if len(self._residual_mapping_functions) != 0:
            raise RuntimeError("TODO: combine it with more residual mapping functions")
        self._residual_mapping_functions = [
            lambda dest, expr: {dest: coordsys.map_residual_on_base_mode(expr),
                                self._azimuthal_stability.real_contribution_name+dest: coordsys.map_residual_on_angular_eigenproblem_real(expr),
                                self._azimuthal_stability.imag_contribution_name+dest: coordsys.map_residual_on_angular_eigenproblem_imag(expr)}]



    def define_problem_for_additional_cartesian_stability_investigation(self):
        from ..expressions.coordsys import CartesianCoordinateSystemWithAdditionalNormalMode
        self._normal_mode_param_k = self.get_global_parameter(self._cartesian_normal_mode_stability.normal_mode_param_k_name)
        coordsys = CartesianCoordinateSystemWithAdditionalNormalMode(self._normal_mode_param_k.get_symbol())
        self.set_coordinate_system(coordsys)

        if len(self._residual_mapping_functions) != 0:
            raise RuntimeError("TODO: combine it with more residual mapping functions")
        self._residual_mapping_functions = [
            lambda dest, expr: {dest: coordsys.map_residual_on_base_mode(expr),
                                self._cartesian_normal_mode_stability.real_contribution_name+dest: coordsys.map_residual_on_normal_mode_eigenproblem_real(expr),
                                self._cartesian_normal_mode_stability.imag_contribution_name+dest: coordsys.map_residual_on_normal_mode_eigenproblem_imag(expr)}]


    def setup_forced_zero_dof_list_for_eigenproblems(self):
        m,normal_k=None,None
        if self._azimuthal_mode_param_m is not None:
            if self._normal_mode_param_k is not None:
                raise RuntimeError("Cannot use both azimuthal and cartesian normal mode at the same time")
            mv=self._azimuthal_mode_param_m.value
            m=round(mv)
            if abs(m-mv)>1e-6:
                raise RuntimeError("Angular mode m is not an integer! "+str(mv))
        elif self._normal_mode_param_k is not None:
            normal_k=self._normal_mode_param_k.value
        else:
            m=None
        to_zero_dofs=self._equation_system._get_forced_zero_dofs_for_eigenproblem(self.get_eigen_solver(),m,normal_k) 
        if len(to_zero_dofs) and _pyoomph.get_verbosity_flag()!=0:
            print("For the eigenvalues "+("" if m is None else "[azimuthal_m="+str(int(m))+"]")+" we set following fields to zero: "+str(to_zero_dofs))        
        from ..solvers.generic import EigenMatrixSetDofsToZero
        esolve = self.get_eigen_solver()
        esolve.clear_matrix_manipulators()  # Flush the matrix manipulators
        if len(to_zero_dofs)>0:
            # And add a Matrix manipulator that sets the constrained degrees of freedom to zero
            esolve.add_matrix_manipulator(EigenMatrixSetDofsToZero(self, *to_zero_dofs))
        return to_zero_dofs


    def _solve_normal_mode_eigenproblem(self, n:int, azimuthal_m:Optional[Union[List[int],Tuple[int],int]]=None, cartesian_k:Optional[Union[List[float],Tuple[float],float]]=None, shift:Optional[Union[float,complex]]=0,quiet:bool=False,filter:Optional[Callable[[complex],bool]]=None,report_accuracy:bool=False,target:Optional[complex]=None,v0:Optional[Union[NPFloatArray,NPComplexArray]]=None,sort:bool=True)->Tuple[NPComplexArray,NPComplexArray]:
        
        if azimuthal_m and (self._azimuthal_mode_param_m is None):
            raise RuntimeError("Must use setup_for_stability_analysis(azimuthal_stability=True) before initialialising the problem")
        if cartesian_k and (self._normal_mode_param_k is None):
            raise RuntimeError("Must use setup_for_stability_analysis(additional_cartesian_mode=True) before initialialising the problem")
        
        if cartesian_k is not None and azimuthal_m is not None:
            raise RuntimeError("TODO: Both simultaneously")
        elif cartesian_k is not None:
            param=self._normal_mode_param_k
            vlist=cartesian_k
        elif azimuthal_m is not None:
            param=self._azimuthal_mode_param_m
            vlist=azimuthal_m
            
        
        if isinstance(vlist,(list,tuple)):
            if report_accuracy:
                raise RuntimeError("report_accuracy=True for normal mode eigenproblems only works if you select a single mode, not a list like "+str(vlist))
            alleigenvals:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
            alleigenvects:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
            minfoL:List[int]=[]
            for ms in vlist:
                param.value = ms
                self.actions_before_eigen_solve()
                self._solve_eigenproblem_helper(n, shift,quiet=True,filter=filter,report_accuracy=report_accuracy,target=target,v0=v0,sort=sort)
                if len(alleigenvals)==0:
                    alleigenvals=self.get_last_eigenvalues().copy()
                else:
                    alleigenvals:NPComplexArray=numpy.hstack([alleigenvals,self.get_last_eigenvalues().copy()]) #type:ignore
                minfoL+=[ms]*len(self.get_last_eigenvalues())
                if len(alleigenvects)==0:
                    alleigenvects:NPComplexArray= numpy.array(self.get_last_eigenvectors()).copy() #type:ignore
                else:
                    alleigenvects:NPComplexArray=numpy.vstack([alleigenvects,numpy.array(self.get_last_eigenvectors()).copy()]) #type:ignore

            if sort:
                if target:
                    srt=numpy.argsort(numpy.abs(alleigenvals-target)) #type:ignore
                else:
                    srt = numpy.argsort(-alleigenvals) #type:ignore
                alleigenvals:NPComplexArray=alleigenvals[srt] #type:ignore
                alleigenvects:NPComplexArray = alleigenvects[srt,:] #type:ignore
                minfo:NPIntArray=numpy.array(minfoL)[srt] #type:ignore
            else:
                minfo:NPIntArray=numpy.array(minfoL)

            self._last_eigenvalues, self._last_eigenvectors = alleigenvals,alleigenvects
            if self.azimuthal_m is not None:
                self._last_eigenvalues_m=minfo
            else:
                self._last_eigenvalues_k=minfo
            
            if (not self.is_quiet()) and (not quiet):
                for i, l in enumerate(self._last_eigenvalues):
                    m=minfo[i]
                    print("Eigenvalue [m="+str(m)+"]", i, ":", l)
            param.value = 0
        else:
            param.value = vlist
            self.actions_before_eigen_solve()
            self._solve_eigenproblem_helper(n, shift,filter=filter,report_accuracy=report_accuracy,target=target,v0=v0,sort=sort)
            param.value = 0
            if azimuthal_m is not None:
                self._last_eigenvalues_m=numpy.array([vlist]*len(self.get_last_eigenvalues()),dtype=numpy.int32) #type:ignore
            else:
                self._last_eigenvalues_k=numpy.array([vlist]*len(self.get_last_eigenvalues()),dtype=numpy.float64) #type:ignore
        return self._last_eigenvalues, self._last_eigenvectors

    # will be called when a stationary solve is tried after a transient solve or when solving for the first time
    def actions_before_stationary_solve(self,force_reassign_eqs:bool=False):
        must_reassign_eqs=self._equation_system._before_stationary_or_transient_solve(stationary=True) 
        if must_reassign_eqs or force_reassign_eqs:
            self.reapply_boundary_conditions()
            self.relink_external_data()
            self.reapply_boundary_conditions()
            self._last_bc_setting="stationary"


    # will be called when a transient solve is tried after a stationary solve or when solving for the first time
    def actions_before_transient_solve(self,force_reassign_eqs:bool=False):
        must_reassign_eqs = self._equation_system._before_stationary_or_transient_solve(stationary=False) 
        if must_reassign_eqs or force_reassign_eqs:
            self.reapply_boundary_conditions()
            self.relink_external_data()
            self.reapply_boundary_conditions()
            self._last_bc_setting="transient"

    # will be called when an eigenproblem is about to be solved
    def actions_before_eigen_solve(self,force_reassign_eqs:bool=False): 
        eigen_m,eigen_k=None,None
        if self._azimuthal_mode_param_m is not None:
            if self._normal_mode_param_k is not None:
                raise RuntimeError("Cannot use both azimuthal and additional cartesian modes simultaneously")
            mv=self._azimuthal_mode_param_m.value
            eigen_m=round(mv)
            if abs(eigen_m-mv)>1e-6:
                raise RuntimeError("Angular mode m is not an integer! "+str(mv))
        if self._normal_mode_param_k is not None:
            kv=self._normal_mode_param_k.value
            eigen_k=kv
            
        must_reassign_eqs = self._equation_system._before_eigen_solve(self.get_eigen_solver(),eigen_m,eigen_k) 
        #print("MUST REASSIGN IS",must_reassign_eqs,eigen_m,eigen_k)
        #exit()
        if must_reassign_eqs or force_reassign_eqs:

            self.reapply_boundary_conditions()
            self.relink_external_data()
            self.reapply_boundary_conditions()
            self._last_bc_setting="eigen"

        if eigen_m is not None and int(eigen_m)!=0:
            self.get_eigen_solver().setup_matrix_contributions(self._azimuthal_stability.real_contribution_name,self._azimuthal_stability.imag_contribution_name)
        elif eigen_k is not None and eigen_k!=0:
            self.get_eigen_solver().setup_matrix_contributions(self._cartesian_normal_mode_stability.real_contribution_name,self._cartesian_normal_mode_stability.imag_contribution_name)
        else:
            self.get_eigen_solver().setup_matrix_contributions("",None)

    def solve(self,*,spatial_adapt:int=0,timestep:Union[ExpressionNumOrNone,List[ExpressionNumOrNone]]=None,shift_values:bool=True,temporal_error:Optional[float]=None,max_newton_iterations:Optional[int]=None,newton_relaxation_factor:Optional[float]=None,suppress_resolve_after_adapt:bool=False,newton_solver_tolerance:Optional[float]=None,do_not_set_IC:bool=False,globally_convergent_newton:bool=False)->ExpressionOrNum: #,continuation=None)
        """
        Solves the problem stationary, unless a timestep is given. In that case, the time step is taken.

        Parameters:
        - spatial_adapt (int): The level of spatial adaptation. Default is 0.
        - timestep (Union[ExpressionNumOrNone, List[ExpressionNumOrNone]]): The time step(s) for the transient solve. Can be a single value or a list of values. Default is None, meaning stationary solve without advancing in time.
        - shift_values (bool): Whether to shift the values during the solve, i.e. shifting the history value buffer. Default is True.
        - temporal_error (Optional[float]): The temporal error for adaptive time stepping. Default is None.
        - max_newton_iterations (Optional[int]): Override the maximum number of Newton iterations. Default is None.
        - newton_relaxation_factor (Optional[float]): Override the relaxation factor for the Newton solver. Default is None.
        - suppress_resolve_after_adapt (bool): Whether to suppress resolving after adaptation. Default is False.
        - newton_solver_tolerance (Optional[float]): Override the tolerance for the Newton solver. Default is None.
        - do_not_set_IC (bool): Whether to not set the initial condition in the first call. Default is False.
        - globally_convergent_newton (bool): Whether to use globally convergent Newton solver. Default is False.

        Returns:
        - ExpressionOrNum: The current time after solving.
        """  
                

        if isinstance(timestep,(list,tuple)):
            lastres=0
            for t in timestep: #type:ignore
                assert isinstance(t,(float,int,Expression)) or t is None
                self._in_transient_newton_solve=True
                lastres=self.solve(timestep=t,spatial_adapt=spatial_adapt,shift_values=shift_values,temporal_error=temporal_error,max_newton_iterations=max_newton_iterations,newton_relaxation_factor=newton_relaxation_factor,suppress_resolve_after_adapt=suppress_resolve_after_adapt,newton_solver_tolerance=newton_solver_tolerance,globally_convergent_newton=globally_convergent_newton)
                self._in_transient_newton_solve=False
            return lastres

        timestep_normalized=False
        self._activate_solver_callback()
        self.invalidate_cached_mesh_data()
        if isinstance(spatial_adapt,bool) and spatial_adapt==True:
            spatial_adapt=self.max_refinement_level

        TSCALE=self.scaling.get("temporal",1)
        assert not isinstance(TSCALE,str)

        if not self.is_initialised():
            self.initialise()
            TSCALE=self.scaling.get("temporal",1)
            self._activate_solver_callback()
            if (timestep is not None):
                timestep=timestep/TSCALE
                try:
                    timestep=float(timestep)
                except RuntimeError as _:
                    raise RuntimeError("Time step needs to match the dimension of the temporal scale "+str(self.scaling.get("temporal",1)))
                timestep_normalized=True
                if self._runmode!="continue":
                    self.initialise_dt(timestep)
                    if not do_not_set_IC:
                        self.set_initial_condition()
                    self.timestepper.set_num_unsteady_steps_done(0)
                    self._taken_already_an_unsteady_step=True
        elif self._taken_already_an_unsteady_step==False and (timestep is not None):
            timestep = timestep / TSCALE
            try:
                timestep = float(timestep)
            except RuntimeError as _:
                raise RuntimeError("Time step needs to match the dimension of the temporal scale " + str(self.scaling.get("temporal", 1)))
            timestep_normalized = True
            self.initialise_dt(timestep)
            if not do_not_set_IC:
                self.set_initial_condition() #This will calc the weights etc and history values correctly
            self.timestepper.set_num_unsteady_steps_done(0)
            self._taken_already_an_unsteady_step = True

        oldsettings=self._override_for_this_solve(max_newton_iterations=max_newton_iterations,newton_relaxation_factor=newton_relaxation_factor,newton_solver_tolerance=newton_solver_tolerance,globally_convergent_newton=globally_convergent_newton)
#		if continuation:
#			if (not isinstance(continuation,list)) or len(continuation)!=2:
#				raise ValueError("kwarg continuation needs to be a list [global parameter, step]")
#			res=self.arclength_continuation(continuation[0],continuation[1],spatial_adapt=spatial_adapt)
#			self._override_for_this_solve(**oldsettings)
#			return res

        paramstr = ""
        paramnames=[pn for pn in self.get_global_parameter_names() if not pn.startswith("_")]        
        if len(paramnames) > 0:
            paramstr = ". Parameters: " + ", ".join(
                [n + "=" + str(self.get_global_parameter(n).value) for n in paramnames])

        if self._dof_selector_used is not self._dof_selector:
            self.reapply_boundary_conditions()
            self.reapply_boundary_conditions() # Must be done twice to correctly setup the eqn_remappings

        #Get rid of the eigen info... It will change!
        self.invalidate_eigendata()
        
        if timestep is None:
            self.actions_before_stationary_solve()
            if not self.is_quiet():
                print("STATIONARY SOLVE"+paramstr)
            self.steady_newton_solve(spatial_adapt)
            self._last_step_was_stationary = True
            if self.get_bifurcation_tracking_mode()!="":
                if self._bifurcation_tracking_parameter_name=="<LAMBDA_TRACKING>":
                    self._last_eigenvalues=numpy.array([self._get_lambda_tracking_real() +self._get_bifurcation_omega()*1j],dtype=numpy.complex128) #type:ignore    
                else:
                    self._last_eigenvalues=numpy.array([0+self._get_bifurcation_omega()*1j],dtype=numpy.complex128) #type:ignore
                self._last_eigenvectors=numpy.array([self._get_bifurcation_eigenvector()],dtype=numpy.complex128) #type:ignore
                if self.get_bifurcation_tracking_mode()=="azimuthal":
                    self._last_eigenvalues_m=numpy.array([int(self._azimuthal_mode_param_m.value)],dtype=numpy.int32) #type:ignore
                elif self.get_bifurcation_tracking_mode()=="cartesian_normal_mode":
                    self._last_eigenvalues_k=numpy.array([self._normal_mode_param_k.value]) #type:ignore
                self._last_eigenvectors=self.process_eigenvectors(self._last_eigenvectors)
            else:
                self._last_eigenvalues_m=None
                self._last_eigenvalues_k=None
            self._override_for_this_solve(**oldsettings)
            return 0
        else:
            if (timestep is not None) and (not timestep_normalized):
                timestep=timestep/TSCALE
                try:
                    timestep=float(timestep)
                except RuntimeError as _:
                    raise RuntimeError("Time step needs to match the dimension of the temporal scale "+str(self.scaling.get("temporal",1)))
                timestep_normalized=True
            if not self.is_quiet():
                print("TRANSIENT SOLVE with nondim dt",timestep,"at current time "+str(self.get_current_time(as_float=False))+paramstr)

            self.actions_before_transient_solve()
            self._last_step_was_stationary=False
            assert isinstance(timestep,(float,int))
            if spatial_adapt==0:                
                if temporal_error is None:
                    desired_dt=timestep                    
                    self.unsteady_newton_solve(timestep,shift_values)
                else:
                    desired_dt=self.adaptive_unsteady_newton_solve(timestep,temporal_error,shift_values)
                self._first_step=False
                self._override_for_this_solve(**oldsettings)
                self.timestepper.increment_num_unsteady_steps_done()
                return desired_dt*TSCALE
            else:
                if self._first_step:
                    self._resetting_first_step=True
                else:
                    self._resetting_first_step=False
                if temporal_error is None:
                    desired_dt=timestep
                    self.unsteady_newton_solve(timestep,spatial_adapt,self._first_step,shift_values)
                else:
                    desired_dt=self.doubly_adaptive_unsteady_newton_solve(timestep,temporal_error,spatial_adapt,int(suppress_resolve_after_adapt),self._first_step,shift_values)
                self._first_step=False
                self._override_for_this_solve(**oldsettings)
                self.timestepper.increment_num_unsteady_steps_done()
                return desired_dt*TSCALE


    def run(self, endtime:ExpressionOrNum, timestep:ExpressionNumOrNone=None,*, outstep:Union[ExpressionNumOrNone,bool]=None, numouts:Optional[int]=None, out_initially:Union[bool,None]=None,
            temporal_error:Union[None,float]=None, outstep_relative_to_zero:bool=True,spatial_adapt:int=0,startstep:ExpressionNumOrNone=None,maxstep:ExpressionNumOrNone=None,newton_solver_tolerance:Union[None,float]=None,do_not_set_IC:bool=False,globally_convergent_newton:bool=False,max_newton_iterations:Union[None,int]=None,starttime:ExpressionNumOrNone=None,suppress_resolve_after_adapt=False,max_newton_to_increase_time_step:Optional[int]=None)->ExpressionOrNum:
        """
        Run the problem for a specified duration, potential with output calls and temporal and/or spatial adaptivity.
        All time quantities must be given in dimensional units, e.g. ``second``, if you use e.g. :py:meth:`~Problem.set_scaling` with e.g. ``temporal=1*second`` for a dimensional problem.

        Args:
            endtime: The end time of the simulation.
            timestep: The time step size. If not specified, it will be determined automatically, e.g. by the outstep.
            outstep: The time interval between outputs. If set to True, outputs will be generated at each time step. If set to False, no outputs will be generated. If not specified, it will be set to the value of `timestep`.
            numouts: The number of outputs to generate. If specified, it will override the value of `outstep`.
            out_initially: Whether to generate an output at the initial time. If not specified, it will be set to `True` if `outstep` is not `False`, otherwise it will be set to `False`.
            temporal_error: The temporal error tolerance. If specified, it will be used to control the time step size.
            outstep_relative_to_zero: Whether the `outstep` is relative to the initial time or the current time. If set to `True`, the `outstep` will be relative to the initial time. If set to `False`, the `outstep` will be relative to the current time.
            spatial_adapt: The level of spatial adaptation. If specified, it will be used to control the spatial refinement level.
            startstep: The time step size at the start of the simulation (for temporal adaptivity). If specified, it will override the value of `timestep`.
            maxstep: The maximum time step size. If specified, it will be used to limit the time step size during temporal adaptivity.
            newton_solver_tolerance: The tolerance for the Newton solver. If specified, it will be used to control the convergence of the solver during this run call.
            do_not_set_IC: Whether to set the initial condition. If set to `True`, the initial condition will not be set.
            globally_convergent_newton: Whether to use a globally convergent Newton solver. If set to `True`, a globally convergent Newton solver will be used.
            max_newton_iterations: The maximum number of iterations for the Newton solver. If specified, it will override to limit the number of iterations for this run call.
            starttime: The start time of the simulation. If specified, it will override the current time.
            suppress_resolve_after_adapt: Whether to suppress the resolve after adaptation. If set to `True`, the resolve after adaptation will be suppressed.
            max_newton_to_increase_time_step: The maximum number of Newton iterations to increase the time step size. If specified, the adaptive time step will only be increased if the number of Newton iterations is less than this value.

        Returns:
            The final time of the simulation after the run call.

        Raises:
            ValueError: If `endtime` is not specified.
            ValueError: If `outstep` and `numouts` are specified simultaneously.
            RuntimeError: If a suitable time step cannot be determined.

        """
        if endtime is None:
            raise ValueError("Must specify an endtime")
        
        if self._bifurcation_tracking_parameter_name is not None:
            raise RuntimeError("Cannot use run with bifurcation tracking enabled. Use solve instead to find the bifurcation or call deactivate_bifurcation_tracking() before")
        if isinstance(self.assembly_handler_pt(),_pyoomph.PeriodicOrbitHandler):
            raise RuntimeError("Cannot use run with periodic orbit tracking enabled. Use solve instead to find the periodic orbit or call deactivate_bifurcation_tracking() before")

        if spatial_adapt>self.max_refinement_level:
            spatial_adapt=self.max_refinement_level
        elif isinstance(spatial_adapt,bool) and spatial_adapt==True:
            spatial_adapt=self.max_refinement_level

        if temporal_error is not None and temporal_error <= 0:
            temporal_error = None

        if numouts is not None and numouts <= 0:
            numouts = None
            outstep = False

        if (outstep is not None):
            if numouts is not None:
                raise ValueError("Cannot use outstep and numouts simultaneously")

        if isinstance(numouts,bool) and numouts == True:
            outstep=True
            numouts=None

        if starttime is not None:
            self.set_current_time(starttime)
        if (not self.is_initialised()) or self._taken_already_an_unsteady_step==False:
            #We need to calculate the initial time step already now to initialize appropriately!
            _tstart=self.get_current_time() #This might call initialise!
            if self._runmode!="continue":
                if numouts is not None:
                    if isinstance(numouts,bool) and numouts==True:
                        raise RuntimeError("TODO: Init with a suitable time step")
                    else:
                        _ts=float((endtime-_tstart)/(numouts*self.get_scaling("temporal")))
                        self.initialise_dt(_ts)
                        if not do_not_set_IC:
                            self.set_initial_condition()
                elif startstep is not None:
                    _ts=float(startstep/self.get_scaling("temporal"))
                    self.initialise_dt(_ts)
                    if not do_not_set_IC:
                        self.set_initial_condition()
                elif timestep is not None:
                    _ts = float(timestep / self.get_scaling("temporal"))
                    self.initialise_dt(_ts)
                    if not do_not_set_IC:
                        self.set_initial_condition()
                elif isinstance(outstep,float) or isinstance(outstep,_pyoomph.Expression) or (isinstance(outstep,int) and not (isinstance(outstep,bool))):
                    _ts = float(outstep / self.get_scaling("temporal"))
                    self.initialise_dt(_ts)
                    if not do_not_set_IC:
                        self.set_initial_condition()
                elif maxstep is not None:
                    _ts = float(maxstep / self.get_scaling("temporal"))
                    self.initialise_dt(_ts)
                    if not do_not_set_IC:
                        self.set_initial_condition()
                else:
                    raise RuntimeError("TODO: Init with a suitable time step. Pass e.g. startstep as keyword arg")
                if out_initially is None:
                        out_initially = outstep != False
            if out_initially is None:
                if not self.is_initialised():
                    out_initially = outstep != False
                else:
                    out_initially = False
            if out_initially and self._runmode!="continue":
                self.output()

        starttime = self.get_current_time()
        _tfactor,_tunit=assert_dimensional_value(starttime-endtime)
        if _tfactor>=0.0:
            print("Skipping run call since starttime "+str(starttime)+" is larger than endtime "+str(endtime))
            self._nondim_time_after_last_run_statement=float(endtime/self.get_scaling("temporal"))
            return 0
        elif self._runmode=="continue":
            # Calculate the remaining numouts
            if numouts is not None:
                ct=float(self.get_current_time()/self.get_scaling("temporal"))
                et=float(endtime/self.get_scaling("temporal"))
                progress=(ct-self._nondim_time_after_last_run_statement)/(et-self._nondim_time_after_last_run_statement)
                numouts=int(numouts*(1-progress))
            timestep = self.timestepper.time_pt().dt(0) * self.get_scaling("temporal")

        #TODO Further checking for the end time

        if timestep is None:
            if not self.is_initialised():
                self.initialise()
                timestep = self.get_scaling("temporal")
            else:
                timestep = self.timestepper.time_pt().dt(0) * self.get_scaling("temporal")
            _tdiff,_tunit=assert_dimensional_value(starttime+timestep-endtime)
            if _tdiff>0:
                timestep=endtime-starttime
        if startstep is not None:
            timestep=startstep


        if outstep is None:
            outstep = timestep
        TS = self.get_scaling("temporal")
        ndouttimes:Optional[NPFloatArray] = None 
        if not isinstance(outstep, bool):
            currentdt = min(float(timestep / TS), float(outstep / TS)) * TS
            if maxstep is not None:
                currentdt=min(float(currentdt/TS),float(maxstep/TS))*TS
            if outstep_relative_to_zero:
                if numouts:
                    dtout = (endtime - starttime) / numouts
                    soffs = math.ceil(float(starttime / dtout)) * dtout
                    endout = soffs + numouts * dtout
                    ndouttimes = numpy.linspace(float(soffs / TS), float(endout / TS), num=numouts + 1) #type:ignore
                else:
                    dtout=outstep
                    numouts=int(float((endtime - starttime)/dtout))
                    soffs = math.ceil(float(starttime / dtout)) * dtout
                    endout = soffs + numouts * dtout
                    ndouttimes = numpy.linspace(float(soffs / TS), float(endout / TS), num=numouts + 1) #type:ignore

            else:
                if numouts:
                    ndouttimes = numpy.linspace(float((starttime) / TS), float(endtime / TS), num=numouts + 1,endpoint=True) #type:ignore
                else:
                    raise RuntimeError("TODO")
            outcntvalue=float((ndouttimes[-1]-starttime/ TS) )/(numouts)+ ndouttimes[-1]
            ndouttimes=numpy.hstack([ndouttimes,[outcntvalue]]) #type:ignore
        else:
            currentdt = timestep

        nextdt_was_clamped_for_output:ExpressionNumOrNone=None # When clamping a time step to hit the next output dt, enlarge it afterwards
        while self.get_current_time(as_float=True, dimensional=False) < float(endtime / TS):
            if self._abort_current_run:
                self._abort_current_run=False
                return currentdt
            tnd = self.get_current_time(as_float=True, dimensional=False)
            possibly_larger_dt:ExpressionNumOrNone=None
            if ndouttimes is not None:
                # Check if the current timestep would exceed the next output
                currind:int = numpy.nonzero(ndouttimes <= tnd)[0] #type:ignore
                currind = -1 if len(currind) == 0 else currind[-1] #type:ignore                
                nextndout = ndouttimes[currind + 1] #type:ignore
                if tnd + float(currentdt / TS) * 1.01 > nextndout:
                    possibly_larger_dt=currentdt
                    currentdt = (nextndout - tnd) * TS

            self._in_transient_newton_solve=True
            nextdt = self.solve(timestep=currentdt, temporal_error=temporal_error,spatial_adapt=spatial_adapt,newton_solver_tolerance=newton_solver_tolerance,do_not_set_IC=do_not_set_IC,globally_convergent_newton=globally_convergent_newton,max_newton_iterations=max_newton_iterations,suppress_resolve_after_adapt=suppress_resolve_after_adapt)
            self._in_transient_newton_solve=False
            if max_newton_to_increase_time_step is not None and float(nextdt/TS)>float(currentdt/TS*1.00001):
                last_res=self.get_last_residual_convergence()
                if len(last_res)>max_newton_to_increase_time_step:
                    print("Do not increase time step, since we used too many iterations")
                    nextdt=currentdt
            
        
            if possibly_larger_dt is not None:
                if float(possibly_larger_dt/nextdt)>1.0:
                    pass
                    #print("Taking larger dt",nextdt,possibly_larger_dt)
                    #nextdt=possibly_larger_dt
            # problem._ofile.write("\t".join(map(str, [tnd, currentdt,nextdt]))+"\n")
            # problem._ofile.flush()
            if isinstance(outstep,bool) and outstep == True:
                self.output()
                if nextdt_was_clamped_for_output is not None:
                        nextdt=max(1.0,float(nextdt_was_clamped_for_output/nextdt))*nextdt 
                        nextdt_was_clamped_for_output=None
            elif outstep != False:
                tndnew = self.get_current_time(as_float=True, dimensional=False)
                nextindA = numpy.nonzero(ndouttimes <= tndnew)[0] #type:ignore
                nextind:int = -1 if len(nextindA) == 0 else nextindA[-1] #type:ignore
                if nextind > currind: #type:ignore
                    self.output()
                    if nextdt_was_clamped_for_output is not None:
                        nextdt=max(1.0,float(nextdt_was_clamped_for_output/nextdt))*nextdt 
                        nextdt_was_clamped_for_output=None
                else:
                    #  Finally check whether the next dt would be very close to the next output. If so, better do two smaller steps
                    # TODO: This needs to be checked further
                    tnext = tndnew + float(nextdt / TS) * 1.15
                    futureindA = numpy.nonzero(ndouttimes <= tnext)[0] #type:ignore
                    futureind:int = -1 if len(futureindA) == 0 else futureindA[-1] #type:ignore
                    if futureind > nextind and ndouttimes is not None:
                        #print("clamping nextdt",nextdt,(ndouttimes[futureind] - tndnew)  * TS)
                        nextdt_was_clamped_for_output=nextdt
                        nextdt = (ndouttimes[futureind] - tndnew)  * TS #type:ignore

            currentdt = nextdt
            if maxstep is not None:
                currentdt=TS*min(float(currentdt/TS),float(maxstep/TS))
            if self.get_current_time(as_float=True, dimensional=False)+float(currentdt/TS) > float(endtime / TS):
                currentdt=1.00001*float(endtime/TS-self.get_current_time(as_float=True, dimensional=False))*TS

        self._nondim_time_after_last_run_statement=float(self.get_current_time()/TS)
        return currentdt
    

    def deflated_solve_by_eigenperturbation(self, eigenindex:int=0, keep_deflation_active:bool=False, perturbation_factor:float=1,deflation_alpha:float=0.1,deflation_power:int=2,*, max_newton_iterations:Optional[int]=None, newton_relaxation_factor:Optional[float]=None, newton_solver_tolerance:Optional[float]=None, globally_convergent_newton:bool=False):        
        """Tries to find another stationary solution by deflation. The procedure is implemented according to 'Deflation techniques for finding distinct solutions of nonlinear partial differential equations' by
Patrick E. Farrell, Ásgeir Birkisson & Simon W. Funke, https://arxiv.org/pdf/1410.5620.pdf .

        Args:
            deflation_alpha (float, optional): Shift of the deflation operator. Defaults to 0.1.
            deflation_p (int, optional): Order of the deflation. Defaults to 2.
            perturbation_amplitude (float, optional): Perturbation amplitude to move away from the previous solution. Defaults to 1.
            max_newton_iterations (Optional[int], optional): Optional override of the number of Newton iterations to try. Defaults to None.
            newton_relaxation_factor (Optional[float], optional): Optional override of the Newton relaxation factor. Defaults to None.        
            
        """
        if eigenindex < 0:
            raise ValueError("Eigenindex must be non-negative.")
        if self.get_last_eigenvectors() is None or len(self.get_last_eigenvectors())<=eigenindex:            
            raise ValueError("No eigenvector at index "+str(eigenindex)+" available to perturb. Please solve the eigenproblem first.")

        from pyoomph.generic.bifurcation_tools import DeflationAssemblyHandler        
        old=self.get_custom_assembler()
        if not isinstance(old, DeflationAssemblyHandler):
            defl=DeflationAssemblyHandler(alpha=deflation_alpha, p=deflation_power)
            self.set_custom_assembler(defl)            
            defl.add_known_solution(self.get_current_dofs()[0])  
        else:
            defl=old
        self.perturb_dofs(self.get_last_eigenvectors()[0]*perturbation_factor)
        self.solve(max_newton_iterations=max_newton_iterations,newton_relaxation_factor=newton_relaxation_factor,newton_solver_tolerance=newton_solver_tolerance,globally_convergent_newton=globally_convergent_newton)
        
        if not keep_deflation_active:
            self.set_custom_assembler(old)
        else:
            defl.add_known_solution(self.get_current_dofs()[0])
            

    def iterate_over_multiple_solutions_by_deflation(self,deflation_alpha:float=0.1,deflation_p:int=2,perturbation_amplitude:float=0.5,max_newton_iterations:Optional[int]=None,newton_relaxation_factor:Optional[float]=None,use_eigenperturbation:bool=False,skip_initial_solution:bool=False,num_random_tries:int=1,keep_deflation_operator_active:bool=False)-> Generator[NPFloatArray,None,None]:
        """Tries to find multiple stationary solutions by deflation. The procedure is implemented according to 'Deflation techniques for finding distinct solutions of nonlinear partial differential equations' by
Patrick E. Farrell, Ásgeir Birkisson & Simon W. Funke, https://arxiv.org/pdf/1410.5620.pdf .

        Args:
            deflation_alpha (float, optional): Shift of the deflation operator. Defaults to 0.1.
            deflation_p (int, optional): Order of the deflation. Defaults to 2.
            perturbation_amplitude (float, optional): Perturbation amplitude to move away from the previous solution. Defaults to 0.5.
            max_newton_iterations (Optional[int], optional): Optional override of the number of Newton iterations to try. Defaults to None.
            newton_relaxation_factor (Optional[float], optional): Optional override of the Newton relaxation factor. Defaults to None.

        Yields:
            The found solutions as lists of degrees of freedom
            
        """        
            
        from pyoomph.generic.bifurcation_tools import DeflationAssemblyHandler
        deflation=DeflationAssemblyHandler(alpha=deflation_alpha,p=deflation_p)
        if not self.is_initialised():
            self.initialise()
        self.set_custom_assembler(deflation)
        
        self.solve(max_newton_iterations=max_newton_iterations,newton_relaxation_factor=newton_relaxation_factor)
        numtries=1
        U=self.get_current_dofs()[0]
        found_sols=[U]
        eigen_perts=[]
        if use_eigenperturbation:
            self.solve_eigenproblem(1)
            eigv=numpy.real(self.get_last_eigenvectors()[0])
            eigv=eigv/numpy.amax(abs(eigv))
            eigen_perts.append(eigv*perturbation_amplitude)
        if not skip_initial_solution:
            yield U
        deflation.add_known_solution(U)
        while True:
            new_sols=[]
            for i,Ustart in enumerate(found_sols):    
                
                if use_eigenperturbation:
                    self.set_current_dofs(Ustart+eigen_perts[i])
                    try:
                        numtries+=1
                        self.solve(max_newton_iterations=max_newton_iterations,newton_relaxation_factor=newton_relaxation_factor)
                        Unew=self.get_current_dofs()[0]
                        self.solve_eigenproblem(1)
                        eigv=numpy.real(self.get_last_eigenvectors()[0])
                        eigv=eigv/numpy.amax(abs(eigv))
                        eigen_perts.append(eigv*perturbation_amplitude)
                        new_sols.append(Unew)
                        deflation.add_known_solution(Unew)
                        
                        yield Unew
                    except:
                        print("Eigenperturbation of solution "+str(i)+" failed to converge. Trying random perturbation")
                for j in range(num_random_tries):
                    self.set_current_dofs(Ustart+(numpy.random.rand(self.ndof())-0.5)*(perturbation_amplitude))
                    try:
                        numtries+=1
                        self.solve(max_newton_iterations=max_newton_iterations,newton_relaxation_factor=newton_relaxation_factor)
                        Unew=self.get_current_dofs()[0]
                        new_sols.append(Unew)
                        if use_eigenperturbation:
                            self.solve_eigenproblem(1)
                            eigv=numpy.real(self.get_last_eigenvectors()[0])
                            eigv=eigv/numpy.amax(abs(eigv))
                            eigen_perts.append(eigv*perturbation_amplitude)
                        deflation.add_known_solution(Unew)
                        yield Unew
                    except:
                        print("Random perturbation "+str(j+1)+"/"+str(num_random_tries)+" of solution "+str(i)+" failed to converge")
            if len(new_sols)==0:
                print("No new solutions found. Stopping deflation. Found in total "+str(len(found_sols))+" in "+str(numtries)+" attempts.")
                if not keep_deflation_operator_active:
                    self.set_custom_assembler(None)
                self.set_current_dofs(U)                
                return
            else:
                found_sols+=new_sols
                

    def deflated_continuation(self,deflation_alpha:float=0.1,deflation_p:int=2,perturbation_amplitude:float=0.5,max_newton_iterations:Optional[int]=None,newton_relaxation_factor:Optional[float]=None,use_eigenperturbation:bool=False,skip_initial_solution:bool=False,num_random_tries:int=1,max_branches:Optional[int]=None,branch_continue_iterations:int=10,**param_range):
        """Scan over a parameter range and try to find multiple solutions for each parameter step by deflation
        This is an implemetation according to: The computation of disconnected bifurcation diagrams by Patrick E. Farrell, Casper H. L. Beentjes, Ásgeir Birkisson
         https://arxiv.org/pdf/1603.00809.pdf
        
        Args:
            deflation_alpha : Shift of the deflation operator. Defaults to 0.1.
            deflation_p: Order of the deflation. Defaults to 2.
            perturbation_amplitude: Perturbation amplitude to move away from the previous solution. Defaults to 0.5.
            max_newton_iterations: Optional override of the number of Newton iterations during deflated search for additional solutions. Defaults to None.
            newton_relaxation_factor: Optional override of the Newton relaxation factor during deflated search for additional solutions. Defaults to None.
            use_eigenperturbation: Whether to use eigen perturbation for the next solution during deflation. Defaults to False.            
            num_random_tries: Number of random tries for finding solutions during deflation. Defaults to 1.
            max_branches: Maximum number of branches to find. Defaults to None.
            branch_continue_iterations: Number of iterations for continuing branches. Defaults to 10.
            
        Yields:
            A tuple of branch index (from 0 to ...), the current parameter value and the current degrees of freedom (dofs) for the solution.
        """ 
        from pyoomph.generic.bifurcation_tools import DeflationAssemblyHandler
        param=None
        rang=None
        for k,v in param_range.items():
            if param is None:
                param=k
                rang=[pv for pv in v]
            else:
                raise RuntimeError("Please specify only one parameter range")
        if param is None:
            raise RuntimeError("Please specify a parameter range like e.g. parameter_name=linspace(0,1,10)")
        if param not in self.get_global_parameter_names():
            raise RuntimeError("Please specify a parameter that is defined in the problem")
        param_obj=self.get_global_parameter(param)
        active_branches={} # Branch index -> current dofs
        
        # Find the first solutions
        self.go_to_param(**{param:rang.pop(0)})
        self.solve()
        branch_index=0
        for dofs in self.iterate_over_multiple_solutions_by_deflation(max_newton_iterations=max_newton_iterations,perturbation_amplitude=perturbation_amplitude,deflation_alpha=deflation_alpha,deflation_p=deflation_p,newton_relaxation_factor=newton_relaxation_factor,use_eigenperturbation=use_eigenperturbation,skip_initial_solution=skip_initial_solution,num_random_tries=num_random_tries,keep_deflation_operator_active=True):
            active_branches[branch_index]=dofs
            yield branch_index,param_obj.value,dofs
            branch_index+=1            
        deflator=cast(DeflationAssemblyHandler,self._custom_assembler)
        if len(active_branches)==0:
            print("No solution found to start with")
            self.set_custom_assembler(None)
            return
        
        for pv in rang:
            deflator.clear_known_solutions()
            param_obj.value=pv
            branches_to_remove=[]
            branches_to_add={}
            old_branches=active_branches.copy()
            for bi,dofs in active_branches.items():
                self.set_current_dofs(dofs)
                param_obj.value=pv
                try:
                    self.solve(max_newton_iterations=branch_continue_iterations)
                    newdofs=self.get_current_dofs()[0]
                    deflator.add_known_solution(newdofs)
                    active_branches[bi]=newdofs
                    yield bi,param_obj.value,newdofs
                except:
                    branches_to_remove.append(bi)
            
            # It could have happened that we accidentially switched branches due to the order of the deflation selection
            # Reorder them by distance in the dofs
            new_branches_to_remove=[]
            for bind_to_rem in branches_to_remove:
                switch_index=None
                mindist=numpy.linalg.norm(active_branches[bind_to_rem]-old_branches[bind_to_rem])
                for other_branch,otherdofs in active_branches.items():
                    if other_branch in branches_to_remove:
                        continue
                    cdist=numpy.linalg.norm(otherdofs-old_branches[bind_to_rem])
                    if cdist<mindist:
                        cdist=mindist
                        switch_index=other_branch
                if switch_index is not None:
                    print("Switching branch {} with {}".format(bind_to_rem,switch_index))
                    new_branches_to_remove.append(switch_index)
                    active_branches[bind_to_rem]=active_branches[switch_index]
                else:
                    new_branches_to_remove.append(bind_to_rem)                        
                
            branches_to_remove=new_branches_to_remove
                    
            for bi,dofs in active_branches.items():
                success=True
                if max_branches is not None and len(active_branches)+len(branches_to_add)-len(branches_to_remove)>max_branches:
                    break
                remaining_perturbation_tries=num_random_tries
                while success:
                    
                    print("Checking for a new solution",branch_index,branches_to_remove)
                    self.set_current_dofs(dofs)
                    self.perturb_dofs((numpy.random.rand(self.ndof())-0.5)*(perturbation_amplitude))
                    param_obj.value=pv
                    try:                    
                        self.solve(max_newton_iterations=max_newton_iterations,newton_relaxation_factor=newton_relaxation_factor)
                        print("Found new solution after ",len(self.get_last_residual_convergence()),"steps",self.get_last_residual_convergence())
                        newdofs=self.get_current_dofs()[0]
                        deflator.add_known_solution(newdofs)
                        branches_to_add[branch_index]=newdofs                        
                        yield branch_index,param_obj.value,newdofs
                        branch_index+=1
                    except:
                        remaining_perturbation_tries-=1
                        if remaining_perturbation_tries<=0:
                            success=False
                        
            for bi in branches_to_remove:
                del active_branches[bi]
            for bi,newdofs in branches_to_add.items():
                active_branches[bi]=newdofs
        
        self.set_custom_assembler(None)
        return

    def force_remesh(self, only_domains:Optional[Set[MeshTemplate]]=None, num_adapt:Optional[int]=None,interpolator:Type["BaseMeshToMeshInterpolator"]=_DefaultInterpolatorClass):
        remeshers:List["RemesherBase"] = []
        if only_domains is not None:
            for t in only_domains:
                if t.remesher is not None:
                    remeshers.append(t.remesher)
        else:
            for t in self._meshtemplate_list:
                if t.remesher is not None:
                    remeshers.append(t.remesher)

        if len(remeshers)==0:
            return
        self.invalidate_cached_mesh_data()
        print("REMESHING")
        
        has_continuation_data=False
        if self._last_arclength_parameter is not None:  
            dof_deriv=self.get_arclength_dof_derivative_vector()
            if len(dof_deriv)>0:
                dof_current=self.get_arclength_dof_current_vector()
                # Store the arclength in the history
                _actual_dofs,_positional_dofs,pinned_values=self.get_all_values_at_current_time(True)            
                self.set_current_pinned_values(0*pinned_values,True,5)
                self.set_current_pinned_values(0*pinned_values,True,6)
                self.set_history_dofs(5,dof_deriv)
                self.set_history_dofs(6,dof_current)
                has_continuation_data=True
                print("STORING CONTINATION DATA BEFORE REMESHING")
                
        
                
        self.actions_before_remeshing(remeshers)
        for r in remeshers:
            r.remesh()

        

        new_meshes:Dict[str,Union[MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d]] = {}
        old_meshes:Dict[str,Union[MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d]] = {}

        # Now remove all interfaces and so on from the previous meshes
        for name, mesh in self._meshdict.items():
            if isinstance(mesh, ODEStorageMesh): continue
            for r in remeshers:
                if name in r._old_meshes.keys():                      
                    # Clean up
                    # for iname,imesh in mesh._interfacemeshes.items():
                    #    imesh.clear_before_adapt()
                    mesh = MeshFromTemplate(self, r.get_new_template(), name, r._old_meshes[name]._eqtree,previous_mesh=r._old_meshes[name]) 
                    new_meshes[name] = mesh
                    old_meshes[name] = r._old_meshes[name] 

        # Replace
        for name, newmesh in new_meshes.items():
            oldmesh = old_meshes[name]
            self._meshdict[name] = newmesh
            assert oldmesh._codegen is not None
            oldmesh._codegen._mesh = newmesh 
            assert oldmesh._codegen._code is not None            
            oldmesh._codegen._code._exchange_mesh(newmesh) 
            newmesh._construct_after_remesh() 

            for tree_depth in range(2):
                newmesh._generate_interface_elements(tree_depth)

            newmesh._tracers=oldmesh._tracers 
            for _,tracercoll in newmesh._tracers.items(): 
                tracercoll._set_mesh(newmesh)


        # Rebuild
        self.rebuild_global_mesh_from_list(rebuild=True)
        for m in self._interfacemeshes:
            m.rebuild_after_adapt()
            m.ensure_external_data()
        # print("REBUILD INTERFACE MESH",m.nelement())
        if len(self._interfacemeshes):
            if not self.is_quiet():
                print("REBUILDING GLOBAL MESH")
            self.rebuild_global_mesh()
        for m in self._meshtemplate_list:
            m._connect_opposite_elements(self._equation_system) 

        self.rebuild_global_mesh_from_list(rebuild=True)
        self.reapply_boundary_conditions()

        interpolators:Dict[str,"BaseMeshToMeshInterpolator"]={}
        # Apply the interpolation on each mesh: First on the boundaries and then down to the bulk mesh
        def perform_interpolation():
            for _, interp in interpolators.items(): 
                interp.interpolate() 


        if has_continuation_data:
            print("RESTORING CONTINUATION DATA")
            dof_deriv=self.get_history_dofs(5)
            dof_current=self.get_history_dofs(6)
            self._update_dof_vectors_for_continuation(dof_deriv,dof_current)

        num_adapt = self.max_refinement_level if num_adapt is None else num_adapt


        for name, newmesh in new_meshes.items():
            
            oldmesh = old_meshes[name]
            #oldmesh.prepare_interpolation() # This one will change the Lagrangian coordinates!
            interpolators[name]=interpolator(oldmesh,newmesh)
            oldmesh.get_eqtree()._before_mesh_to_mesh_interpolation(interpolators[name])

        if num_adapt > 0:
            no_need_to_reassign = False
            for s in range(num_adapt):
                self.map_nodes_on_macro_elements()
                perform_interpolation()
                if not self.is_quiet():
                    print("Remeshing adaption:", s, "of", num_adapt)
                nref, nunref = self._adapt()
                if nref == 0 and nunref == 0:
                    no_need_to_reassign = True
                    break
            if num_adapt > 0 and not (no_need_to_reassign):
                self.map_nodes_on_macro_elements()
                perform_interpolation()
        else:
            self.map_nodes_on_macro_elements()
            perform_interpolation()

        self.remove_macro_elements()

        self.actions_after_remeshing()
        for r in remeshers:
            r.actions_after_remeshing()
        self.invalidate_cached_mesh_data()
        
        



    def _get_time_of_state_file(self,fname:str):
        state=DumpFile(fname,False)
        state.string_data(lambda: self._dump_header, lambda s: state.assert_equal(s, self._dump_header))
        _version_str = state.string_data(lambda: self._dump_version, lambda s: state.assert_leq(s, self._dump_version))
        # Current time
        t=state.float_data(lambda: self.get_current_time(dimensional=True, as_float=True),lambda t: t)
        s = state.int_data(lambda: self._output_step, lambda s: s)
        state.close()
        return t,s

    # This function defines the state file, i.e. storing or reading all relevant information of the current status of the simulations
    def define_state_file(self, state:DumpFile,ignore_loading_eigendata:bool=False,ignore_continuation_data=False,additional_info={}):
        # Please do not modify the first part, it is required in that order for peeking the current time  in _get_time_of_state_file()
        # Header
        state.string_data(lambda: self._dump_header, lambda s: state.assert_equal(s, self._dump_header))
        _version_str = state.string_data(lambda: self._dump_version, lambda s: state.assert_leq(s, self._dump_version))

        # Current time
        state.float_data(lambda: self.get_current_time(dimensional=True, as_float=True),lambda t: self.set_current_time(t, dimensional=True, as_float=True))
        self._output_step = state.int_data(lambda: self._output_step, lambda s: s)

        # Continue section step
        self._continue_section_step_loaded=state.int_data(lambda: self._continue_section_step,lambda v:v)

        # From here on, you can in principle modify. Of course, old state files are incompatible once you add/remove anything here

        # Numpy array compression level
        compression=-100 if state.compression_level is None else state.compression_level
        state.compression_level=state.int_data(lambda : compression, lambda v:v)
        if state.compression_level==-100:
            state.compression_level=None
            
        # Mesh templates
        state.int_data(lambda : len(self._meshtemplate_list),lambda n : state.assert_equal(n,len(self._meshtemplate_list)))

        new_meshes:Dict[str,Union[MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d]] = {}
        old_meshes:Dict[str,Union[MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d]]= {}
        for _i,templ in enumerate(self._meshtemplate_list):
            old=templ.get_template()
            new=templ.define_state_file(state,additional_info=additional_info)
            if not state.save:
#                print("OLD VS NEW",old,new)
                if old!=new:
#                    print("OLD VS NEW2", old, new)
                    for n,om in self._meshdict.items():
                        if old.has_domain(n):
                            assert isinstance(om,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d))
                            old_meshes[n]=om
                            new_meshes[n]=MeshFromTemplate(self,new,n,om._eqtree,om)                              

        if not state.save:
            for name, newmesh in new_meshes.items():
                oldmesh = old_meshes[name]
                self._meshdict[name] = newmesh
                assert oldmesh._codegen is not None
                oldmesh._codegen._mesh = newmesh 
                assert oldmesh._codegen._code is not None
                oldmesh._codegen._code._exchange_mesh(newmesh) 
#                print("REPLACING MESH ",name,"from",oldmesh,"to",newmesh)
                newmesh._construct_after_remesh() 
                for tree_depth in range(2):
                    newmesh._generate_interface_elements(tree_depth) 
            # Rebuild
            if len(new_meshes)>=0:
                self.rebuild_global_mesh_from_list(rebuild=True)
                for m in self._interfacemeshes:
                    m.rebuild_after_adapt()
                    m.ensure_external_data()
                # print("REBUILD INTERFACE MESH",m.nelement())
                if len(self._interfacemeshes):
                    if not self.is_quiet():
                        print("REBUILDING GLOBAL MESH")
                    self.rebuild_global_mesh()
                for m in self._meshtemplate_list:
                    m._connect_opposite_elements(self._equation_system) 

                self.rebuild_global_mesh_from_list(rebuild=True)
                self.reapply_boundary_conditions()



        # Time stepper dts
        time = self.timestepper.time_pt()
        ndt = state.int_data(lambda: time.ndt(), lambda ndt: state.assert_equal(ndt, time.ndt()))
        for dt in range(ndt):
            _dtval=state.float_data(lambda: time.dt(dt), lambda v: time.set_dt(dt, v))

        if not state.save:
            self.timestepper.set_weights()

        state.int_data(lambda: self.timestepper.get_num_unsteady_steps_done(),lambda t: self.timestepper.set_num_unsteady_steps_done(t))

        # Mesh list
        nummeshes = len(self._meshdict)
        nummeshes = state.int_data(lambda: nummeshes, lambda n: state.assert_equal(n, nummeshes)) #type:ignore
        mesh_name_list = list(sorted(list(self._meshdict.keys())))
        for nmesh in range(nummeshes):
            meshname = state.string_data(lambda: mesh_name_list[nmesh], lambda s: s)
            assert meshname in self._meshdict.keys()
            mesh = self._meshdict[meshname]
            assert not isinstance(mesh,InterfaceMesh)            
            mesh.define_state_file(state,additional_info={})
        # Global params
        gpars = list(sorted(self.get_global_parameter_names()))
        numgpars = len(gpars)
        numgpars = state.int_data(lambda: numgpars, lambda n: state.assert_equal(n, numgpars)) #type:ignore
        for ngpar in range(numgpars):
            gparname = state.string_data(lambda: gpars[ngpar], lambda s: state.assert_equal(s, gpars[ngpar]))
            gp = self.get_global_parameter(gparname)
            gp.value = state.float_data(lambda: gp.value, lambda v: v)




        # Eigendata if desired
        write_eigen=1 if (self.eigen_data_in_states is not False) else 0
        has_eigendata=state.int_data(lambda : write_eigen,lambda n : n)
        if has_eigendata:
            self._last_bc_setting=state.string_data(lambda : self._last_bc_setting,lambda s:s)            
            if state.save:
                if self.eigen_data_in_states is True:
                    numeigen=len(self._last_eigenvalues)
                elif isinstance(self.eigen_data_in_states,int): #type:ignore
                    numeigen=min(self.eigen_data_in_states,len(self._last_eigenvalues))
            else:
                numeigen=0
            numeigen=state.int_data(lambda : numeigen,lambda n : n)
            has_azimuthal=1 if (self._last_eigenvalues_m is not None) else 0
            if numeigen>0:
                # Eigenvectors
                if not state.save:
                    evals=state.numpy_data(lambda  : self._last_eigenvalues,lambda e:e)
                    evects=state.numpy_data(lambda  : self._last_eigenvectors,lambda e:e)                
                    if not ignore_loading_eigendata:
                        self._last_eigenvalues=evals.copy()
                        self._last_eigenvectors=evects.copy()
                    has_azimuthal=state.int_data(lambda : has_azimuthal, lambda e :e)
                    if has_azimuthal:
                        ms=state.numpy_data(lambda  : self._last_eigenvalues_m,lambda e:e) #type:ignore
                        if not ignore_loading_eigendata:
                            self._last_eigenvalues_m=ms.copy()
                else:
                    state.numpy_data(lambda  : self._last_eigenvalues[:numeigen],lambda e:e)
                    state.numpy_data(lambda  : self._last_eigenvectors[:numeigen,:],lambda e:e)
                    if state.int_data(lambda : has_azimuthal, lambda e :e):
                        state.numpy_data(lambda  : self._last_eigenvalues_m[:numeigen],lambda e:e) #type:ignore
        #return
        write_conti=1 if (self.continuation_data_in_states is not False) else 0
        if state.save:
            dofderiv=self.get_arclength_dof_derivative_vector()
            if len(dofderiv)==0:
                write_conti=0
        else:
            dofderiv=[]
        has_contidata=state.int_data(lambda : write_conti,lambda n : n)
        if has_contidata:
            dofderiv=state.numpy_data(lambda  : dofderiv,lambda e:e)
            paramderiv=state.float_data(lambda  : self.get_arc_length_parameter_derivative(),lambda e:e)
            thetasqr=state.float_data(lambda  : self.get_arc_length_theta_sqr(),lambda e:e)
            if not state.save and not ignore_continuation_data:            
                self._set_dof_direction_arclength(dofderiv)
                self._set_arc_length_parameter_derivative(paramderiv)
                self._set_arc_length_theta_sqr(thetasqr)
        else:
            if not state.save:
                self.reset_arc_length_parameters()
            

            
        # Save the last BC settings. E.g. eigensolvers may have different values pinned. This is important for the eigen-vector data to match
        #self._last_bc_setting=state.string_data(lambda : self._last_bc_setting,lambda s:s)
                

        


    def save_state(self, fname:str,relative_to_output:bool=False)->None:
        if self.is_distributed():
            raise RuntimeError("Distributed save state does not work. Consider to set write_states=False in the Problem class for the time being")
        elif get_mpi_rank()>0:
            return

        if not self.is_quiet():
            print("Saving state ", fname)
        if relative_to_output:
            fname=os.path.join(self.get_output_directory(),fname)
        dump = DumpFile(fname, True,compression_level=self.states_compression_level)
        self.define_state_file(dump)
        dump.write_footer("EOF_pyoomph")
        dump.close()


    def load_state(self, fname:str,ignore_outstep:bool=False,relative_to_output:bool=False,ignore_eigendata:bool=False,ignore_continuation_data:bool=False,additional_info:Dict[Any,Any]={}):
        if not self.is_initialised():
            self.initialise()
        if self.is_distributed():
            raise RuntimeError("Distributed load state")
        if relative_to_output:
            fname=os.path.join(self.get_output_directory(),fname)

        _pyoomph.set_interpolate_new_interface_dofs(False) # We may not interpolate the additional dofs on newly constructed interface nodes
        self.invalidate_cached_mesh_data()
        dump = DumpFile(fname, False)
        good=dump.check_footer("EOF_pyoomph")
        if not good:
            raise RuntimeError("Unsupported state file: "+fname)
        for m in self._interfacemeshes:
            m.clear_before_adapt()
        oldoutstep=self._output_step
        self.define_state_file(dump,ignore_loading_eigendata=ignore_eigendata,ignore_continuation_data=ignore_continuation_data,additional_info=additional_info)
        self.invalidate_cached_mesh_data()
        if ignore_outstep:
            self._output_step=oldoutstep
        print("State file "+fname+" loaded")
        for m in self._interfacemeshes:
            m.clear_before_adapt()
        self.invalidate_cached_mesh_data()
        self.rebuild_global_mesh_from_list(rebuild=True)
        self.actions_after_adapt()
        self.setup_pinning()
        self.reapply_boundary_conditions()
        self.invalidate_cached_mesh_data()
        dump.close()

        self.actions_after_remeshing() # Must call this to inform e.g. the outputters, that the mesh has changed!
        
        self.invalidate_cached_mesh_data()                
        if self._last_bc_setting=="eigen":          
            if self._azimuthal_mode_param_m is not None and len(self.get_last_eigenmodes_m()):
                self._azimuthal_mode_param_m.value=self.get_last_eigenmodes_m()[0]                
            self.actions_before_eigen_solve()
            if self._azimuthal_mode_param_m is not None:
                self._azimuthal_mode_param_m.value=0
        elif self._last_bc_setting=="transient":
            self.actions_before_transient_solve()
        elif self._last_bc_setting=="stationary":
            self.actions_before_stationary_solve()
        _pyoomph.set_interpolate_new_interface_dofs(True) # Activate the interpolation again, good for spatial adaptivity
        return True


    def continue_from_outdir(self,old_out_dir:str,statenumber:int=-1,ignore_outstep:bool=True):
        """Loads a previous state from another output directory. Make sure the scripts are in all specifications of equations, meshes, parameters, settings etc.

        Args:
            old_out_dir: Old output directory
            statenumber: Which state file to load (default: -1, i.e. the last one)
            ignore_outstep: Do not load the outstep (default: True)
        """
        import glob
        toglob=os.path.join(old_out_dir,"_states","state_"+("{:06d}.dump".format(statenumber) if statenumber>=0 else "*.dump"))
        globs=glob.glob(toglob)
        if len(globs)==0:
            raise RuntimeError(f"No state files found for {toglob}")         
        contifile=sorted(globs)[statenumber if statenumber<0 else 0]
        print("Continuing from",contifile)        
        self.load_state(contifile,ignore_outstep=ignore_outstep)
        
        
    def select_dofs(self) -> "_DofSelector":
        return _DofSelector(self)




    def is_precice_initialised(self):
        return self._precice_interface is not None

        
    def precice_initialise(self):
        """Initializes the preCICE adapter for the problem.
        You must set precice_participant and precice_config_file in the Problem class.
        """
        if self._precice_interface is not None:
            raise ValueError("Precice interface already initialised")
        if not self.is_initialised():
            self.initialise()
        if self.precice_participant is None and self.precice_participant!="":
            raise ValueError("precice_participant not set")
        if self.precice_config_file is None and self.precice_config_file!="":
            raise ValueError("precice_config_file not set")
        from ..solvers.precice_adapter import get_pyoomph_precice_adapter
        get_pyoomph_precice_adapter().initialize_problem(self)
     
        
        
    def precice_run(self,maxstep:Optional[float]=None,temporal_error:Optional[float]=None,output_initially:bool=True,fast_dof_backup:bool=False):
        """
        Runs a simulation with the precice adapter. To that end, you must set precice_participant and precice_config_file in the Problem class.
        There is less control compared to the normal py:meth:`pyoomph.generic.problem.Problem.run` (i.e. without preCICE), but a lot of settings can be adjusted in the preCICE configuration file.

        Args:
            maxstep: Maximum nondimensional time step. Defaults to None.
            temporal_error: Use temporal adaptivity with this given error factor. Defaults to None.
            output_initially: Outputs before the simulation starts. Defaults to True.
            fast_dof_backup: If True, only the DoFs  will be backed up, nothing else. Defaults to False.
        """
        if not self.is_precice_initialised():
            self.precice_initialise()
        from ..solvers.precice_adapter import get_pyoomph_precice_adapter
        get_pyoomph_precice_adapter().coupled_run(self,maxstep=maxstep,temporal_error=temporal_error,output_initially=output_initially,fast_dof_backup=fast_dof_backup)


    def create_text_file_output(self,filename:str,header:Optional[List[str]]=None,relative_to_output_dir:bool=True)->"NumericalTextOutputFile":
        """Creates a :py:class:`~pyoomph.utils.num_text_out.NumericalTextOutputFile`. By default, in the output directory.

        Args:
            filename: File name
            header: Header of the file Defaults to None.
            relative_to_output_dir: If True, the file is created in the output directory. Defaults to True.        
        """
        from ..utils.num_text_out import NumericalTextOutputFile
        if relative_to_output_dir:
            filename=self.get_output_directory(filename)
        return NumericalTextOutputFile(filename,header=header)
        

############## DOF SELECTOR ###################
class _DofSelector:
    def __init__(self,problem:"Problem"):
        self._problem=problem
        self._all_unselected:Optional[bool]=None
        self._tree:Dict[str,Any]={}

    def __enter__(self):
        if not self._problem.is_initialised():
            self._problem.initialise()
        self._previous_dof_selector = self._problem._dof_selector
        self._problem._dof_selector = self 
        for ism in range(self._problem.nsub_mesh()):
            submesh = self._problem.mesh_pt(ism)
            if isinstance(submesh, (MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh,ODEStorageMesh)):
                n=submesh.get_full_name()
                splt=n.split("/")
                node=self._tree
                for k in splt:
                    if not k in node.keys():
                        node[k]={}
                        node[k]["__parent__"]=node
                    node=node[k]
                fi=submesh.get_field_information()
                for fentry,space in fi.items():
                        node[fentry]=[space,None]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): #type:ignore
        # Set to the previous one
        self._problem._dof_selector=self._previous_dof_selector 


    def _traverse(self,n:Dict[str,Any],select:bool,onlydof:Optional[str]=None):
        for k,v in n.items():
            if k=="__parent__":
                continue
            if isinstance(v,list):
                if onlydof is not None:
                    if onlydof!=k:
                        continue
                n[k][1]=select # Unselect all
            elif isinstance(v,dict):
                self._traverse(n[k],select,onlydof=onlydof)

    def unselect_all(self):
        self._problem._dof_selector_used= "INVALID" 
        self._problem.invalidate_eigendata()        
        self._all_unselected=True
        self._traverse(self._tree,False)

    def select_all(self):
        self._problem._dof_selector_used= "INVALID" 
        self._problem.invalidate_eigendata()
        self._all_unselected=False
        self._traverse(self._tree,True)

    def _select_or_unselect(self,k:str,select:bool):
        splt = k.split("/")
        node = self._tree
        prev_node=None
        for k in splt:
            if k not in node.keys():
                raise RuntimeError("Cannot select or unselect " + k + " since it does not index a field or a mesh. " "Available fields on this mesh: "+str([nam for nam in node.keys() if nam!="__parent__"]))
            prev_node=node
            node = node[k]
        if isinstance(node, list):
            node[1] = select
            if prev_node is not None:
                self._traverse(prev_node, select,onlydof=k)
        elif isinstance(node, dict):
            self._traverse(node, select)

    # Selects meshes (e.g. "droplet") or degrees (e.g. "droplet/velocity_x"), both including interface meshes
    def select(self,*args:str):
        if self._all_unselected is None:
            self.unselect_all()
        self._problem._dof_selector_used = "INVALID" 
        self._problem.invalidate_eigendata()
        for k in args:
            self._select_or_unselect(k,True)

    def unselect(self,*args:str):
        if self._all_unselected is None:
            self.select_all()
        self._problem._dof_selector_used = "INVALID" 
        self._problem.invalidate_eigendata()
        for k in args:
            self._select_or_unselect(k,False)


    def _apply_on_domain(self,mesh:Optional[AnyMesh])->None:
        #print("APPLY ON DOMAIN",mesh)
        if mesh is None:
            return
        fn=mesh.get_full_name()
        splt = fn.split("/")
        #print(splt)
        node = self._tree
        for k in splt:
            node=node[k]
#        print(fn,"###########")
        selected:Set[str]=set()
        unselected:Set[str]=set()
        boundinds:Set[int]=set()
        for k,d in node.items():
            if isinstance(d,list):
                if d[1]:
                    selected.add(k)
        for bn in mesh.get_boundary_names():
            if node.get(bn,None) is not None:
                ind=mesh.get_boundary_index(bn)
                boundinds.add(ind)
        #print(selected,unselected,boundinds)
        mesh._pin_all_my_dofs(unselected,selected,boundinds)
