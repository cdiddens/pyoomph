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
 
import glob
import sys
import warnings

import scipy.sparse

from ..expressions.generic import is_zero #type:ignore

#import pyoomph.generic
from .mpi import *
from .. import _pyoomph_core as _pyoomph
import math


import __main__

import os
import gc
from pathlib import Path

import argparse
import numpy
from ..meshes.mesh import  AnyMesh,AnySpatialMesh,BulkTemplateMesh, MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d, ODEStorageMesh, InterfaceMesh,MeshFromTemplate,MeshFromTemplateBase,MeshTemplate
from .codegen import EquationTree,BaseEquations, FiniteElementCodeGenerator,CombinedEquations,DummyEquations, InterfaceEquations #ODEEquations
from ..solvers.generic import DefaultMatrixType, EigenSolverWhich, GenericLinearSystemSolver,GenericEigenSolver
from ..expressions.units import *
from ..expressions import get_global_symbol,cartesian,axisymmetric,axisymmetric_flipped,radialsymmetric,BaseCoordinateSystem,nondim,testfunction,weak,OptionalCoordinateSystem
from ..solvers.generic import get_default_linear_solver,get_default_eigen_solver
from ..meshes.interpolator import _DefaultInterpolatorClass,ODEInterpolator 
from ..output.states import DumpFile
from ..expressions import ExpressionOrNum,ExpressionNumOrNone
from ..meshes.meshdatacache import MeshDataCacheStorage, MeshDataCacheOperatorBase, MeshDataEigenModes,MeshDataCacheEntry

from .ccompiler import BaseCCompiler,SystemCCompiler
from .jit_cache import get_jit_cache,tier2_shadow_enabled

import types
import io
import contextlib
from typing import IO

from .adaptive_recovery import AdaptiveResolveRecovery, SpatialAdaptResolveError
from ..typings import *

if TYPE_CHECKING:
    from ..output.plotting import MatplotlibPlotter
    from ..meshes.remesher import RemesherBase
    from ..meshes.interpolator import BaseMeshToMeshInterpolator
    from .assembly import CustomAssemblyBase
    from ..utils.num_text_out import NumericalTextOutputFile
    from ..output.latex import LaTeXPrinter
    import precice

Z2ErrorEstimator=_pyoomph.Z2ErrorEstimator

class _NewtonSolveOverrideSettings(TypedDict,total=False):
    max_newton_iterations:int
    newton_relaxation_factor:float
    newton_solver_tolerance:float
    globally_convergent_newton:bool

import subprocess

import signal
def breakpoint():
    os.kill(os.getpid(), signal.SIGTRAP)

#To use "with problem.custom_adapt:" statement
_interface_conformity_check_mode_cache:list[int]=[]

def _interface_conformity_check_mode() -> int:
    """0 off, 1 report, 2 throw -- read once from PYOOMPH_CHECK_HALO_CONSISTENCY.

    Shares the halo consistency check's variable on purpose: both are "do the pieces that must agree
    still agree?" checks, both are off by default, and asking a user to know which of two variables
    applies to their symptom would be asking them to already know the answer.
    """
    if not _interface_conformity_check_mode_cache:
        val=os.environ.get("PYOOMPH_CHECK_HALO_CONSISTENCY","").strip().lower()
        if val in ("2","throw","raise"):
            _interface_conformity_check_mode_cache.append(2)
        elif val in ("1","warn","report"):
            _interface_conformity_check_mode_cache.append(1)
        else:
            _interface_conformity_check_mode_cache.append(0)
    return _interface_conformity_check_mode_cache[0]


class _CustomAdaptWithHelper:
    def __init__(self, problem: "Problem",skip_init_call:bool=False):
        self._problem=problem
        self._skip_init_call=skip_init_call
    def __enter__(self):
        if not self._skip_init_call and  not self._problem.is_initialised():
            self._problem.initialise()
        self._problem.actions_before_adapt()
    def __exit__(self,exc_type: type[BaseException] | None, exc: BaseException | None, traceback: types.TracebackType | None):
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
        self._problem:Problem | None = None
        
    def get_problem(self)->"Problem":
        if self._problem is None:
            raise RuntimeError("Problem not set")
        return self._problem

    def actions_after_remeshing(self):
        pass 
    
    def actions_after_change_in_global_parameter(self,param:str):
        pass
    
    def actions_before_remeshing(self,active_remeshers:list["RemesherBase"]):
        pass

    def actions_after_newton_solve(self):
        pass

    def actions_after_transient_solve(self):
        """Called once per accepted timestep, not once per Newton solve. See
        :py:meth:`Problem.actions_after_transient_solve`."""
        pass

    def actions_before_newton_solve(self):
        pass

    def actions_after_newton_step(self):
        pass
    
    def before_assigning_equation_numbers(self,dof_selector:_DofSelector | None,before_equation_system:bool):
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
    def __init__(self,problem:"Problem",mode:Literal["collocation","central","bspline","BDF2","floquet"],lyap_coeff,param,omega,pvalue,pdvalue,al,order:int,GL_order:int,T_constraint:Literal["plane", "phase"]):
         self.problem=problem
         self.mode:Literal["collocation","central","bspline","BDF2","floquet"]=mode
         self.order,self.GL_order=order,GL_order
         self.T_constraint:Literal["plane", "phase"]=T_constraint
         self.emerging_info={"lyap_coeff":lyap_coeff,"param":param,"omega":omega,"pvalue":pvalue,"dpvalue":pdvalue,"al":al}
         
    def __enter__(self):
        return self
    
    def __exit__(self,exc_type: type[BaseException] | None, exc: BaseException | None, traceback: types.TracebackType | None):
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
    
    @overload
    def get_T(self, dimensional: Literal[True]=True) -> ExpressionOrNum: ...
    @overload
    def get_T(self, dimensional: Literal[False]) -> float: ...
    
    def get_T(self, dimensional: bool = True) -> Union[ExpressionOrNum, float]:
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
    
    def output_orbit(self,subdir:str,Tstart:float | None=None,Tend:float | None=None,N:int | None=None,set_current_time:bool=True,endpoint:bool=True):
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
    
    def iterate_over_samples(self,Tstart:float | None=None,Tend:float | None=None,N:int | None=None,set_current_time:bool=True,endpoint:bool=True):
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
        
    def get_floquet_multipliers(self,n:int | None=None,valid_threshold:float | None=10000,shift:float | None=None,ignore_periodic_unity:bool | float=False,quiet:bool=True):
        return self.problem.get_floquet_multipliers(n=n,valid_threshold=valid_threshold,shift=shift,ignore_periodic_unity=ignore_periodic_unity,quiet=quiet)
    
    def starts_supercritically(self):
        """
        When started at a Hopf bifurcation, this function tells you whether the first Lyaupnov coefficient is negative, corresponding to a supercritical Hopf bifurcation with initially stable orbits
        """
        
        return self.emerging_info["lyap_coeff"]<0
    
    def evaluate_observable_time_integral(self,*observables:str):
        if len(observables)==0:
            raise ValueError("No observables given")
        accus:dict[str,Expression]={n:Expression(0) for n in observables}
        obs_info:dict[str,tuple[AnySpatialMesh,str]]={}
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
        
    
    def change_sampling(self,*,mode:Literal["collocation","central","bspline","BDF2","floquet"] | None =None,NT:int | None=None, order:int | None=None,GL_order:int | None=None,T_constraint:Literal["plane", "phase"] | None=None,do_solve:bool=True):
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
        
            

def _teardown_spatial_mesh(m:"AnySpatialMesh") -> None:
    # Under nanobind (unlike pybind11), a mesh kept alive on the C++ side by
    # nb::keep_alive<>() (set_mesh_pt/add_sub_mesh in problem.cpp) is invisible to Python's
    # cyclic garbage collector: as long as the owning Problem's C++ instance is alive, the mesh
    # Python object cannot be freed, no matter what gc.collect() does - and, since keep_alive
    # can never be revoked once granted, this holds even after a mesh is superseded (e.g. by
    # remeshing, see Problem.force_remesh()) and dropped from Problem._meshdict: it remains
    # pinned alive for the rest of the Problem's lifetime unless explicitly torn down here.
    # If the mesh in turn still holds a Python-visible "_parent"/etc. back-reference, it and
    # whatever it points to keep each other alive forever (one edge invisible, one visible, and
    # gc cannot break a cycle through an invisible edge). Explicitly nulling every such
    # back-reference breaks the visible side, and forcing immediate destruction of the
    # underlying C++ object (via _destroy_now(), see below) handles the invisible one directly
    # instead of waiting for the owning Problem to also become collectible. (Mesh<->Problem
    # itself no longer needs breaking here: MeshFromTemplate1d/2d/3d/InterfaceMesh don't store a
    # Python-level "_problem" attribute at all any more - get_problem() resolves it live via a
    # non-owning C++-side lookup, see mesh.py.)
    cg=m.get_code_gen()
    cg._code=None
    cg._set_problem(None) #type:ignore
    cg._mesh=None #type:ignore
    # Discontinuous-Galerkin domains additionally attach a handful of auxiliary
    # FiniteElementCodeGenerator objects to cg (for the internal-facet/DG coupling terms),
    # each with its own "_problem"/"_mesh" back-references that are otherwise never cleared.
    for dummy_attr in ("_dummy_codegen_for_internal_facets","_dummy_codegen_for_internal_facets_bulk",
                       "_dummy_codegen_for_internal_facets_bulk_bulk","_dummy_codegen_for_internal_facets_bulk_opp"):
        dummy_cg=getattr(cg,dummy_attr,None)
        if dummy_cg is not None:
            dummy_cg._set_problem(None) #type:ignore
            dummy_cg._code=None
            dummy_cg._mesh=None
            setattr(cg,dummy_attr,None)
    for im in m._interfacemeshes.values():
        _teardown_spatial_mesh(im)

    m._interfacemeshes.clear()
    # Close any output file handles (e.g. ODEFileOutput/IntegralObservableOutput) held open
    # by this mesh's equations before dropping the reference to them -- see the log-file
    # comment in Problem.release() for why this must happen proactively rather than being
    # left to eventual garbage collection.
    if m._eqtree is not None and m._eqtree._equations is not None:
        m._eqtree._equations._release_output_files()
    m._eqtree._equations=None
    m._eqtree=None #type:ignore
    # Break the remaining back-references specific to the mesh's own class:
    # MeshFromTemplate1d/2d/3d hold a reference to their originating MeshTemplate (which
    # itself has its own "_problem" back-reference, resolved via a non-owning C++-side
    # lookup just like the mesh/codegen classes - see MeshTemplate.get_problem()), while
    # InterfaceMesh holds references to its parent (bulk) mesh and, for two-sided
    # interfaces, to the opposite InterfaceMesh.
    if hasattr(m,"_templatemesh"):
        m._templatemesh._set_problem(None) #type:ignore
        m._templatemesh=None #type:ignore
    if hasattr(m,"_parent"):
        m._parent=None #type:ignore
    if hasattr(m,"_opposite_interface_mesh"):
        m._opposite_interface_mesh=None #type:ignore
    # Force the underlying C++ mesh (and, via its normal destructor, all of its elements/nodes)
    # to be destructed right now, synchronously - rather than whenever this Python wrapper
    # object eventually gets garbage collected, which could be much later (e.g. a user script
    # may still hold its own reference to this mesh, entirely legitimately) or never (if some
    # as-yet-undiscovered reference cycle remains). Callers must ensure this mesh is no longer
    # needed for anything (e.g. field interpolation into a replacement mesh) before calling
    # this, and, if unloading equation code DLLs afterwards, must do so only after this call:
    # an element destructed afterwards would dereference its DynamicBulkElementCode's function
    # table in an already-unloaded shared library and crash.
    m._destroy_now()


def _destroy_superseded_mesh(m:"AnySpatialMesh") -> None:
    # Counterpart to _teardown_spatial_mesh(), used for a mesh that has been superseded by
    # remeshing (see Problem.force_remesh()) rather than one whose owning Problem is being
    # released entirely. Unlike release(), a superseded mesh's _eqtree/_codegen must NOT be
    # touched here: for both the top-level (bulk) mesh and each of its interface meshes, that
    # same eqtree/codegen tree is explicitly shared with (part of) its replacement mesh (the
    # bulk mesh's via _exchange_mesh(), each interface mesh's because MeshFromTemplateBase's own
    # interface-mesh construction reuses the *same* child eqtree from self._eqtree.get_children()
    # for the new interface mesh at that position) - clearing them crashed/broke the next
    # mesh-construction or remesh call. So only _parent/_opposite_interface_mesh/_templatemesh
    # are cleared here (there is no Python-level "_problem" attribute left to clear at all -
    # get_problem() resolves it live via a non-owning C++-side lookup, see mesh.py), plus forcing
    # immediate destruction of the underlying C++ object (freeing the bulk of the memory - nodes,
    # elements, matrices - even though the lightweight Python wrapper itself remains pinned
    # alive by nb::keep_alive until the owning Problem is; see _teardown_spatial_mesh() for that
    # mechanism).
    # Dropping _templatemesh is what keeps the whole Problem collectible at all: unlike the
    # replacement mesh (which release() eventually tears down via _teardown_spatial_mesh), a
    # superseded mesh is never seen by release() again, yet stays pinned alive by the
    # unrevokable nb::keep_alive from the Problem. Its MeshTemplate used to reach the Problem
    # again through the template's remesher, which stored it in a plain attribute, closing a
    # Problem -> (invisible keep_alive) -> superseded mesh -> _templatemesh -> MeshTemplate ->
    # remesher -> Problem cycle that gc cannot break, so every remeshing script leaked its entire
    # Problem - meshes, nodes, elements, equations and all. That last edge is gone (RemesherBase
    # .problem is a live, non-owning lookup now - it had to be, since the *replacement* mesh keeps
    # its template and so hit the same cycle), but dropping _templatemesh here is still what stops
    # a superseded mesh's template graph from being kept alive for nothing. Only this mesh's
    # reference to the template is dropped; the template itself lives on (it is what the
    # replacement mesh was built from), and it must NOT be _set_problem(None)'d the way
    # _teardown_spatial_mesh() does, since it is still in active use.
    for im in m._interfacemeshes.values():
        _destroy_superseded_mesh(im)
    m._interfacemeshes.clear()
    if hasattr(m,"_parent"):
        m._parent=None #type:ignore
    if hasattr(m,"_opposite_interface_mesh"):
        m._opposite_interface_mesh=None #type:ignore
    if getattr(m,"_templatemesh",None) is not None:
        m._templatemesh=None #type:ignore
    m._destroy_now()


_TypeVarMeshTemplate=TypeVar("_TypeVarMeshTemplate",bound=MeshTemplate)

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
        self._released:bool=False

        from .. import get_default_c_compiler
        self.set_c_compiler(get_default_c_compiler())

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
        self._no_cache:bool=False
        self._debug_largest_residual:int=0
        self.ignore_command_line:bool=False

        self._ccode_dir:str="_ccode"
        self._dof_selector:_DofSelector | None=None # The desired selected dofs
        self._dof_selector_used:_DofSelector | None | Literal["INVALID"]=None


        self._use_first_order_timestepper:bool=False
        self._domains_to_remesh:set[MeshTemplate]=set()

        # Static condensation: the Python-side, mesh-independent form of the rules. See
        # _sync_static_condensation_rules() for why the rules are kept here at all.
        self._static_condensation_sources:dict[Any,tuple[str | None,tuple[tuple[str,tuple[int,...],str],...]]]={}
        self._static_condensation_applied:tuple[Any,...] | None=None
        self._static_condensation_applied_meshes:list["AnyMesh"]=[]
        self._static_condensation_source_counter:int=0

        self.max_residuals=1e10
        self.max_newton_iterations=10
        self.newton_solver_tolerance=1e-8

        #: Which mesh-to-mesh interpolator to use whenever the meshes are rebuilt - remeshing,
        #: :py:meth:`force_remesh`, :py:meth:`redefine_problem`, and the remesh handler used during
        #: continuation. Defaults to
        #: :py:class:`~pyoomph.meshes.interpolator.InternalInterpolator`, which transfers by
        #: interpolation: it evaluates the old solution at each new node, which is fast and
        #: pointwise accurate.
        #:
        #: Set it to :py:class:`~pyoomph.meshes.interpolator.ProjectionInternalInterpolator` to
        #: transfer by L2 projection instead. That is the better choice when the transferred
        #: quantity has to keep its integral - the projection conserves it, interpolation does not -
        #: at the cost of pointwise accuracy and one linear solve per history level. See
        #: dev_docs/mesh_point_locator.md.
        #:
        #: An explicit ``interpolator=`` argument still wins where one is passed.
        self.mesh_interpolator:type["BaseMeshToMeshInterpolator"]=_DefaultInterpolatorClass

        #: Write an output immediately before and immediately after every remeshing, so that the
        #: state on the OLD mesh and the state on the NEW mesh can be compared directly.
        #:
        #: Diagnosing a transfer otherwise means guessing from warnings which nodes went wrong. With
        #: this on, and a ``TextFileOutput`` on the interfaces of interest, both files carry the exact
        #: interface coordinates and their values, and the question becomes arithmetic on two files.
        self._debug_remeshing:bool=False

        #: Let :py:meth:`force_remesh` run on a distributed (``--distribute``) problem even in a
        #: configuration it refuses. Remeshing distributed works for the combination
        #: dev_docs/distributed_remeshing.md has built (recreation from merged boundary data,
        #: transfer pooled across the ranks); the parts that are not there yet are refused by name,
        #: because what they do instead is produce a plausible wrong answer rather than fail. This
        #: bypasses those refusals - to develop the remaining stages, or to accept the result.
        self.experimental_distributed_remeshing:bool=False

        self._call_output_after_adapt:bool=False

        #: Spatial adaption steps for the initial condition. If set to ``None``, we refine initially up to :py:attr:`max_refinement_level`.
        self.initial_adaption_steps:None | int=None #Adapting in the first step
        #: Cumulative number of elements that enforce_interface_conformity() had to refine after the
        #: fact to keep the two sides of a coupled interface matching. Diagnostic: it should stay at
        #: whatever the initial mesh setup needed, since the reconciliation during adapt is meant to get
        #: there first. Reset it yourself to measure a particular stretch of a run.
        self._interface_conformity_repairs:int=0
        #: Cumulative number of elements refined by the vertex-connected balance closure of
        #: enforce_interface_conformity(): elements that touch a coupled interface at a single vertex
        #: and carry no facet on it, which the facet-based conformity machinery cannot see and which
        #: would otherwise be left arbitrarily coarser than the interface next to them. Counted apart
        #: from _interface_conformity_repairs because, unlike a repair, this loses no information -
        #: these elements have never been refined, so their sons interpolate a current father.
        self._interface_vertex_balance_refinements:int=0
        self.remove_macro_elements_after_initial_adaption:bool | Literal["auto"]="auto" # "auto" means: Only if the coordinates are free
        #: In distributed runs, we call load balance after each non-uniform adaptions
        self.call_load_balance_in_initial_adaption=False

        #: Minimum error of all meshes for spatial adaptivity. If the error is below this threshold, we may unrefine locally.
        self.min_permitted_error:float=0.0001	#Some defaults for the meshes
        #: Maximum error of all meshes for spatial adaptivity. If the error is above this threshold, we must refine locally.
        self.max_permitted_error:float=0.001
        #: If set to an int, adaptation aims at this problem size instead of at a fixed error tolerance:
        #: the elements with the largest estimated errors are refined (or the smallest unrefined) until
        #: :py:meth:`ndof` is approximately ``desired_ndof``. While it is set, min_permitted_error and
        #: max_permitted_error are *outputs* of the controller rather than inputs - they are recomputed
        #: before every adaptation and restored when this is set back to None.
        self.desired_ndof:int | None=None
        #: Relative dead band of the desired_ndof controller. Inside it nothing is refined or unrefined,
        #: which is what makes the adaptation loops terminate instead of oscillating about the target.
        self.desired_ndof_tolerance:float=0.1
        #: Fraction of the remaining gap the desired_ndof controller closes per adaptation step. Below 1
        #: because 2:1 balancing refines further elements of its own accord, so a step that aimed exactly
        #: at the target would systematically overshoot.
        self.desired_ndof_damping:float=0.7
        #: Largest factor by which the desired_ndof controller may grow ndof in a single adaptation step.
        self.desired_ndof_max_growth:float=4.0
        #: Maximum number of refinements of all meshes. After initialization, use set_max_refinement_level instead of this property.
        self.max_refinement_level:int=8
        #: What to do when the Newton solve *after* a spatial adaptation fails. By default ``None``,
        #: i.e. such a failure ends the run, since the pre-adapt mesh is gone by then and nothing can
        #: be recovered. Set an :py:class:`~pyoomph.generic.adaptive_recovery.AdaptiveResolveRecovery`
        #: to snapshot the state before each adaptation and fall back to it instead.
        self.adaptive_resolve_recovery:"AdaptiveResolveRecovery | None"=None
        self._state_snapshot_counter:int=0
        self._state_snapshot_job_id:"int|None"=None # see _state_snapshot_name
        self._adapt_recovery_transient:bool=False # whether the solve being recovered is a timestep
        #: Minimum refinement level of all meshes.       
        self.min_refinement_level:int=0
        #: Add a .gitignore with content "*" to output folders
        self.gitignore_output:bool=True
        #: Name of the logfile (or None for no logfile), relative to the output directory
        self.logfile_name:str | None="_pyoomph_logfile.txt"
        #: When set to True, we warn about unused global parameters for arclength continuation or bifurcation tracking. If set to "error", we raise an error.
        self.warn_about_unused_global_parameters:bool | Literal["error"]="error"
        #:  There are different methods implemented in oomph-lib to fill the sparse matrices (Jacobian, mass matrix, etc.). Depending on the problem, one or the other method may be faster or more memory efficient. The default method is "vectors_of_pairs", which is the most general one.                
        self.sparse_assembly_method:Literal["vectors_of_pairs","two_vectors","lists","maps","two_arrays"]="vectors_of_pairs"
        self.only_write_logfile_on_proc0:bool=True
        #: Start time of the simulation, stamped into the log file header and used to compute the elapsed time in the footer
        self._log_start_time=None
        #: Guards against writing the elapsed-time log footer twice (release() vs. the atexit fallback)
        self._log_footer_written=False
        #: Checks whether the elements in the meshes are nicely oriented (facing) so that refinement works as it should. Can be only done once initially or at each refinement step
        self.check_mesh_integrity:bool | Literal["initially"]="initially"

        self._meshtemplate_list:list[MeshTemplate]=[]
        self._meshdict={}
        self._residual_mapping_functions:list[Callable[[str,Expression],Expression | dict[str, Expression]]]=[]

        self._named_vars:dict[str,ExpressionOrNum]={}

        self._coordinate_system=cartesian

        self.scaling:dict[str,str | ExpressionOrNum]={} #Add scales here, i.e. spatial=1*centi*meter, temporal=...
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

        self._num_threads:int | None=None # Default
        # Left unset on purpose: get_eigen_solver() below falls back to get_default_eigen_solver(),
        # and asking for the default here would trigger its autodetection (which imports SLEPc) for
        # every problem, including the many that never solve an eigenproblem.
        self._eigensolver=None

        self._runmode="delete"
        self._continue_initialized=False
        #: When resuming with ``--runmode continue``, carry on with the time step stored in the state
        #: file instead of the ``startstep`` (or any other initial dt) requested by the run statement
        #: we resume into. Only that one statement is affected; later run statements in the script
        #: were never entered, so their own initial dt still applies. Set to ``False`` to get the old
        #: behaviour, where a continued run always restarted from the scripted initial step.
        self.use_state_dt_when_continue:bool=True
        # Marks the state load of a --runmode continue as not yet consumed by a run statement. Only
        # the run statement we resume into may take its dt from the state: the later ones in the
        # script were never entered, so their startstep is still a genuine request.
        self._continue_dt_pending=False
        # Nondimensional dt the temporal error estimator asked for after the last transient step, i.e.
        # the step the run loop would take next. Stored in the state file (0.1.2 and later), because
        # time_pt().dt(0) only records the step that was just TAKEN: resuming from that alone repeats
        # one step and loses one adaptation, so a continued run drifted away from the uninterrupted
        # one it is supposed to reproduce. None until the first transient solve of the session.
        self._suggested_next_dt:float | None=None
        self._where_expression="True"

        self._dump_header = "pyoomph_dump"
        # 0.1.0 stores the mesh structurally (see pyoomph/meshes/meshstate.py) instead of by rank-local
        # element/node numbering, which is what makes states of distributed problems possible at all
        # and lets serial and distributed runs read each other's files. 0.1.1 adds the sharding field
        # to the header. 0.1.2 adds the adaptive time stepper's suggested next dt. 0.1.3 adds the
        # tracers' rolling position history, which the trail plots are drawn from. Older files still load.
        self._dump_version = "0.1.4"
        self._last_bc_setting="init"

        self._output_step:int=0
        self._continue_section_step:int=0
        self._continue_section_step_loaded:int=0
        self._nondim_time_after_last_run_statement=0 # Required for continue

        self._interfacemeshes:list[InterfaceMesh]=[]
        self._last_eigenvalues:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
        self._last_eigenvectors:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
        self._last_eigenvalues_m:NPIntArray | None=None
        self._last_eigenvalues_k:NPFloatArray | None=None
        self._azimuthal_mode_param_m=None
        self._normal_mode_param_k=None
        self._azimuthal_stability=_AzimuthalStabilityInfo()
        self._bifurcation_eigenvector_scaling:Literal["unit","auto"]="unit" # last choice passed to activate_bifurcation_tracking, so an adaption can restore it
        self._bifurcation_reactivation_after_adaptation=None
        self._cartesian_normal_mode_stability=_CartesianNormalModeStabilityInfo()
        self._bifurcation_tracking_parameter_name:str | None=None
        self._improved_pitchfork_tracking_coordinate_system:"OptionalCoordinateSystem"=None
        self._improved_pitchfork_tracking_position_coordinate_system:"OptionalCoordinateSystem"=None
        self._shared_shapes_for_multi_assemble=False
        self._setup_azimuthal_stability_code=False
        self._setup_additional_cartesian_stability_code=False
        self._solve_in_arclength_conti=None
        self._adapt_eigenindex:int | None=None # Which eigenvector to use during adaptation
        self._adapted_eigeninfo:list[Any] | None=None # Store the eigenfunction, eigenvalue and m and k after adaptation
        self._last_arclength_parameter=None
        self._taken_already_an_unsteady_step=False
        self._last_step_was_stationary=None
        self._already_set_ic = False
        self._resetting_first_step=False
        self._in_transient_newton_solve=False
        
        self._hooks:list[GenericProblemHooks]=[]
        
        #: Flag indicating whether to call remeshing when necessary. Can be set to ``False`` to disable remeshing, e.g. when tracking bifurcations, it is better to check it manually invoking the remeshing method :py:meth:`remesh_handler_during_continuation` after each solve.
        self.do_call_remeshing_when_necessary:bool=True

        self.default_timestepping_scheme:Literal["BDF2","BDF1","Newmark2"]="BDF2"

        self.default_spatial_integration_order:int | None = None

        self._equation_system:EquationTree
        self._interinter_connections:set[str]=set() # Interface/interface intersections, i.e. codimension 2+ intersections

        self.timestepper = _pyoomph.MultiTimeStepper(True)
        self.add_time_stepper_pt(self.timestepper)

        #: Set this to a (list of ) plotter(s) to automatically plot on :py:meth:`output` calls. If set to ``None``, no plotting will be done.
        self.plotter:list[MatplotlibPlotter] | MatplotlibPlotter | None=None
        self.plot_in_dedicated_process:bool=False
        self._plotting_process:subprocess.Popen | None=None
        self.latex_printer:"LaTeXPrinter | None"=None

        self.write_states:bool=True
        self.states_compression_level:int | None=6
        self.eigen_data_in_states:int | bool=False # Either True (all calced eigenvalues/vectors or a number to limit the number of stored eigendata)
        self.continuation_data_in_states:bool=False
        self.additional_equations:Literal[0] | EquationTree=0

        self.default_1d_file_extension:Literal["txt", "mat"] | list[Literal["txt", "mat"]]="txt"

        self.always_take_one_newton_step=True

        self._mesh_data_cache=MeshDataCacheStorage()
        self.eigenvector_position_scale:float=1 # if eigenmode="real" or "imag", we shift the positions multiplied with this factor (for "abs" or "angle") is is not done
        self._abort_current_run=False

        self._custom_assembler:CustomAssemblyBase | None=None

        self.default_ccode_expression_mode:str="" # Try to factor all expressions with "factor"
        #: Debugging the Jacobian by finite differences with a given epsilon (None or <=0 means no debugging). 
        self.debug_jacobian_by_fd_epsilon:float | None=-1 
        self.extra_compiler_flags:list[str]=[]
        
        #: After analyzing the Jacobian, a field with an empty Jacobian row will be pinned automatically
        self.automatically_remove_dofs_without_equations:bool=True
        #: When you have e.g. a field without any equations, it will stop the simulation and give a warning about the Jacobian structure. Setting this to False, it will just go through
        self.stop_on_jacobian_structure_warning=True

        #: Must be set to the participant name when using preCICE. Default is an empty string, if you do not use preCICE.
        self.precice_participant:str=""
        #: Must be set to the config file when using preCICE
        self.precice_config_file:str=""
        self._precice_interface:"precice.Participant | None"=None
        
        #: Set e.g. to {"domain/velocity_*":"u","domain/pressure":"p"} to automatically setup field split IS for PETSc with names "u" and "p". If None, the default split is set like the field indices in the Jacobian information file, i.e. using "0", "1", etc. as prefixes
        self.petsc_fieldsplit=None
        
        #: When set to True, we apply Dirichlet boundary conditions by removing the corresponding dofs from the system. This yields a smaller matrix, but iterative solvers using strided block matrices will run into troubles. If False, all DirichletBCs are kept in the dof vector and the matrix is augmented accordingly.
        self.apply_Dirichlet_BCs_by_dof_removing=True
        
        #: When set to True, we assign the initial conditions via projection, not on a nodal basis
        self.project_initial_conditions=False

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

    def can_continue_section(self, id:str | None=None) -> bool:
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
        from ..equations.generic import GlobalLagrangeMultiplier
        from ..generic.codegen import var,testfunction
        from ..equations.generic import Scaling,TestScaling,InitialCondition
        neweqs:BaseEquations=GlobalLagrangeMultiplier(**{name:equation_contribution},only_for_stationary_solve=only_for_stationary_solve,set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve)
        if scaling is not None:
            neweqs+=Scaling(**{name:scaling})
        if testscaling is not None:
            neweqs+=TestScaling(**{name:testscaling})
        if initial_condition is not None:
            neweqs+=InitialCondition(degraded_start="auto",IC_name="", **{name:initial_condition})
        self+=neweqs@domain
        return var(name,domain=domain),testfunction(name,domain=domain)


    @overload
    def get_cached_mesh_data(self,msh:str | AnySpatialMesh,nondimensional:bool=...,tesselate_tri:bool=...,eigenvector:int | Sequence[int] | None=...,eigenmode:MeshDataEigenModes=...,history_index:int=...,with_halos:bool=...,operator:MeshDataCacheOperatorBase | None=...,discontinuous:bool=...,add_eigen_to_mesh_positions:bool=...,global_mesh:Literal[False]=...) -> "MeshDataCacheEntry": ...

    @overload
    def get_cached_mesh_data(self,msh:str | AnySpatialMesh,nondimensional:bool=...,tesselate_tri:bool=...,eigenvector:int | Sequence[int] | None=...,eigenmode:MeshDataEigenModes=...,history_index:int=...,with_halos:bool=...,operator:MeshDataCacheOperatorBase | None=...,discontinuous:bool=...,add_eigen_to_mesh_positions:bool=...,global_mesh:bool=...) -> "MeshDataCacheEntry | None": ...

    def get_cached_mesh_data(self,msh:str | AnySpatialMesh,nondimensional:bool=False,tesselate_tri:bool=True,eigenvector:int | Sequence[int] | None=None,eigenmode:MeshDataEigenModes="abs",history_index:int=0,with_halos:bool=False,operator:MeshDataCacheOperatorBase | None=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True,global_mesh:bool=False) -> "MeshDataCacheEntry | None":
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
            global_mesh (bool, optional): Get the whole mesh instead of this process' partition. Only makes a difference on a mesh distributed with --distribute, where it is a collective call that returns the merged data on rank 0 and None on all other ranks. Every rank must reach it - either by calling this with the same arguments, or by being inside a :py:func:`~pyoomph.meshes.meshdatamerge.run_with_global_mesh_data` block while rank 0 asks. Defaults to False.

        Returns:
            MeshDataCacheEntry: The combined information of the mesh cache. None on the non-root ranks of a distributed mesh when global_mesh is set.
        """

        if isinstance(msh,str):
            msh=self.get_mesh(msh)

        return self._mesh_data_cache.get_data(msh,nondimensional=nondimensional,tesselate_tri=tesselate_tri,eigenvector=eigenvector,eigenmode=eigenmode,history_index=history_index,with_halos=with_halos,operator=operator,discontinuous=discontinuous,add_eigen_to_mesh_positions=add_eigen_to_mesh_positions,global_mesh=global_mesh)

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

    def get_equations(self,path:str,error_if_not_found:bool=True)->BaseEquations | None:
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
    def assemble_jacobian(self,with_residual:Literal[True]=...,which_one:str=...,global_csr:bool=...)->tuple[NPFloatArray,DefaultMatrixType]: ...

    @overload
    def assemble_jacobian(self,with_residual:Literal[False],which_one:str=...,global_csr:bool=...)->DefaultMatrixType: ...

    def assemble_jacobian(self,with_residual:bool=True,which_one:str="",global_csr:bool=True)->DefaultMatrixType | tuple[NPFloatArray, DefaultMatrixType]:
        """Assemble the Jacobian (and optionally the residual vector) as a scipy sparse matrix.

        Args:
            with_residual: Also return the residual vector, as (residuals, Jacobian).
            which_one: Name of the residual/Jacobian combination to assemble. Defaults to the one
                currently active.
            global_csr: Return the whole (n x n) Jacobian on every process. Under mpirun oomph
                row-partitions the matrix even without --distribute, so this costs an allgather and
                replicates the matrix on each rank. Pass False to get this rank's row block instead,
                shaped (nrow_local, n) with global column indices - cheaper, but the caller then has
                to know its own first_row. Irrelevant without MPI, where the two agree.

        Returns:
            The Jacobian, or (residuals, Jacobian) if with_residual. The residuals are always the
            full global vector, irrespective of global_csr.
        """
        res, n, _nzz, J_nrow_local, J_values_arr, J_colindex_arr, J_row_start_arr=self._assemble_residual_jacobian(which_one)
        # _assemble_residual_jacobian hands back the Jacobian in LOCAL CSR, because the PETSc callers
        # want exactly that. Under mpirun the row_start array is then shorter than n+1 and scipy
        # rejects it outright, so the row blocks have to be glued back together for the (n x n) form
        # that every caller of this method, and the gathered residual returned alongside, works in.
        if global_csr:
            J_values_arr,J_colindex_arr,J_row_start_arr=self._gather_distributed_csr_rows(n,J_nrow_local,J_values_arr,J_colindex_arr,J_row_start_arr)
            shape=(n,n)
        else:
            shape=(J_nrow_local,n)
        J = scipy.sparse.csr_matrix((J_values_arr, J_colindex_arr, J_row_start_arr), shape=shape) #type:ignore
        if with_residual:
            return res,J #type:ignore
        else:
            return J

    def _gather_distributed_csr_rows(self,n:int,nrow_local:int,values:NPFloatArray,colindex:NPIntArray,row_start:NPIntArray)->tuple[NPFloatArray,NPIntArray,NPIntArray]:
        """Assemble a globally indexed CSR triple from the local row block each rank holds.

        Returns the input unchanged when there is nothing to gather (serial, or a matrix that oomph
        left replicated). The column indices of a distributed CRDoubleMatrix are already global; only
        the rows have to be concatenated, in rank order, since oomph hands out ascending first_row.
        """
        from .mpi import get_mpi_nproc,get_mpi_world_comm
        nproc=get_mpi_nproc()
        if nproc<=1:
            return values,colindex,row_start
        comm=get_mpi_world_comm()
        assert comm is not None
        nrows_per_rank=comm.allgather(int(nrow_local)) #type:ignore
        if sum(nrows_per_rank)!=n:
            # Replicated: every rank already holds all n rows (sum could only match by accident here,
            # since a replicated block is n rows long on each of the nproc>1 ranks).
            return values,colindex,row_start
        all_values=comm.allgather(numpy.asarray(values)) #type:ignore
        all_colindex=comm.allgather(numpy.asarray(colindex)) #type:ignore
        all_row_start=comm.allgather(numpy.asarray(row_start)) #type:ignore
        glob_row_start=numpy.zeros(n+1,dtype=numpy.asarray(row_start).dtype)
        nnz_offset=0
        row_offset=0
        for rs,vals in zip(all_row_start,all_values): #type:ignore
            nloc=len(rs)-1 #type:ignore
            glob_row_start[row_offset:row_offset+nloc]=rs[:-1]+nnz_offset #type:ignore
            row_offset+=nloc
            nnz_offset+=len(vals) #type:ignore
        glob_row_start[n]=nnz_offset
        return numpy.concatenate(all_values),numpy.concatenate(all_colindex),glob_row_start #type:ignore

    def debug_analytic_hessian_by_fd(self, epsilon:float=1e-5, num_vectors:int=2, seed:int=1,
                                     verbose:bool=False,
                                     only_domains:list[str] | None=None) -> float:
        """Check the analytically generated Hessian against finite differences, element by element.

        For each element this forms random vectors Y and C and compares the analytic Hessian-vector
        product ``d2R_i/(du_j du_k) Y_i C_k`` against a finite difference of the *analytic* Jacobian
        along C. It therefore validates the Hessian only relative to the Jacobian; combine it with
        :py:attr:`debug_jacobian_by_fd_epsilon`, which validates the Jacobian against finite
        differences of the residual, to pin down the whole chain.

        Requires the analytic Hessian to have been generated, i.e.
        ``setup_for_stability_analysis(analytic_hessian=True)`` before initialisation.

        Args:
            epsilon: entries whose discrepancy exceeds this are printed. Also the threshold this
                method raises on, unless it is <= 0.
            num_vectors: how many C directions to test per element.
            seed: seed of the (deterministic) random vectors.
            verbose: print a line per element rather than only the offenders.
            only_domains: restrict to these mesh names; all meshes by default.

        Returns:
            The largest relative discrepancy found over all elements. Each entry is scaled by
            ``max(1, |analytic|, |finite difference|)``, since the products range over many orders of
            magnitude and an absolute threshold would either drown in noise or miss the small entries.

        Raises:
            RuntimeError: if that discrepancy exceeds ``epsilon`` and ``epsilon>0``.
        """
        import random as _random
        rng = _random.Random(seed)
        # The finite-difference half perturbs the element's dofs through oomph-lib's Dof_pt, which is
        # only populated when this is switched on before the equation numbers are assigned.
        self._enable_store_local_dof_pt_in_elements()
        self.assign_eqn_numbers()
        worst = 0.0
        worst_where = ""
        # _meshdict holds only the bulk meshes; the interface meshes hang off them and carry their own
        # elements, which is where e.g. a curvature or surface-tension residual lives. Walking only
        # _meshdict silently skipped every one of them.
        def _all_meshes(m, prefix):
            yield prefix, m
            for iname, imesh in getattr(m, "_interfacemeshes", {}).items():
                yield from _all_meshes(imesh, prefix + "/" + iname)

        for topname, topmesh in list(self._meshdict.items()):
          for name, mesh in _all_meshes(topmesh, topname):
            if only_domains is not None and name not in only_domains:
                continue
            for ie in range(mesh.nelement()):
                el = mesh.element_pt(ie)
                nd = el.ndof()
                if nd == 0:
                    continue
                Y = [rng.uniform(-1.0, 1.0) for _ in range(nd)]
                C = [[rng.uniform(-1.0, 1.0) for _ in range(nd)] for _ in range(num_vectors)]
                # A pure printout threshold here; the raising decision is made below on the
                # (possibly relative) value, so pass a threshold that only prints real offenders.
                delta = el._debug_hessian(Y, C, epsilon if not verbose else -1.0)
                if delta > worst:
                    worst = delta
                    worst_where = name + " element " + str(ie)
                if verbose:
                    print("HESSIAN FD CHECK", name, "element", ie, "ndof", nd, "delta", delta)
        if epsilon > 0 and worst > epsilon:
            raise RuntimeError("Analytical Hessian disagrees with finite differences of the Jacobian: "
                               "worst discrepancy " + str(worst) + " at " + worst_where +
                               " (threshold " + str(epsilon) + "). The offending entries were printed above.")
        return worst

    def remove_equations(self, path:str, of_type:type[BaseEquations] | None=None, only_if:Callable[[BaseEquations],bool]=lambda eqn: True,fail_if_not_exist:bool=False):
        if hasattr(self,"_equation_system"):
            eqtree = self._equation_system.get_by_path(path)
        elif self.additional_equations is not None and self.additional_equations!=0:
            eqtree=self.additional_equations.get_by_path(path)
        else:
            eqtree=None
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
        

    def get_output_directory(self,relative_path:str | None=None)->str:
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

    def get_named_var(self, name:str, default:ExpressionOrNum | None=None)->ExpressionNumOrNone:
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
        if self._released:
            return
        self._released=True
        for m in self._meshdict.values():
            if not isinstance(m,ODEStorageMesh):
                _teardown_spatial_mesh(m)
            else:
                cg=m.get_code_gen()
                cg._code=None
                cg._set_problem(None) #type:ignore
                if m._eqtree is not None and m._eqtree._equations is not None:
                    m._eqtree._equations._release_output_files()
                m._eqtree=None
                m._element=None
                m._set_problem(None,None) #type:ignore
                m._destroy_now()

        # GenericLinearSystemSolver/GenericEigenSolver instances hold a "problem" back-reference
        # (set in their own __init__), stored as a weakref precisely so it cannot keep this
        # Problem alive - clear it explicitly anyway so a stray later use of the solver fails
        # fast instead of silently resolving a dead weakref.
        if not isinstance(self._lasolver,(str,type(None))):
            self._lasolver.problem=None #type:ignore
        if not isinstance(self._eigensolver,(str,type(None))):
            self._eigensolver.problem=None #type:ignore
        self._lasolver:str | GenericLinearSystemSolver | None = None
        self._eigensolver:str | GenericEigenSolver | None = None
        self._meshtemplate_list = []
        self._meshdict:dict[str,"AnyMesh"] = {}
        self._equation_system = None #type:ignore
        # The static condensation registry keeps the meshes its rules resolved to referenced (so their
        # id()s cannot be recycled while a signature holds them); drop them with everything else.
        self._static_condensation_sources={}
        self._static_condensation_applied=None
        self._static_condensation_applied_meshes=[]
        # The process-wide solver callback singleton (pyoomph._pyoomph.get_Solver_callback(),
        # see pyoomph/__init__.py's solver_cb) remembers whichever Problem last called solve()
        # via set_Solver_callback()/set_problem(), as a weakref precisely so it cannot keep this
        # Problem alive - but clear it explicitly here too, so release() has an immediate effect
        # rather than waiting for the weakref to next resolve to None on its own.
        solver_cb = _pyoomph.get_Solver_callback()
        if solver_cb is not None:
            ref=getattr(solver_cb, "_current_problem_ref", None)
            if ref is not None and ref() is self:
                solver_cb._current_problem_ref = None # type:ignore
        # set_custom_assembler() (used e.g. for deflation/bifurcation tracking, see
        # bifurcation_tools.py) creates a plain Python-level mutual reference: this Problem's
        # own _custom_assembler points to the assembler, and the assembler's own "problem"
        # attribute points back here. Break it explicitly rather than relying on cyclic gc
        # picking it up eventually.
        if self._custom_assembler is not None:
            self._custom_assembler.problem=None #type:ignore
            self._custom_assembler=None
        # define_problem_for_additional_cartesian_stability_investigation()/
        # define_problem_for_axial_symmetry_breaking_investigation() (normal-mode/azimuthal
        # stability setup) install a lambda here that closes over "self" - a plain Python
        # self-referential cycle (self -> _residual_mapping_functions -> lambda -> self).
        # Break it explicitly instead of relying on cyclic gc, same rationale as above.
        self._residual_mapping_functions = []
        self.invalidate_cached_mesh_data()
        self.invalidate_eigendata()
        self.flush_sub_meshes()
        # Stamp the end/elapsed time into the log file before closing it.
        self._write_log_footer()
        # Close the log file (if any) now, rather than waiting for the C++ Problem
        # object's destructor: on Windows, a still-open log file handle prevents the
        # containing directory from being deleted (WinError 32), which bites e.g.
        # test_solver()/test_compiler() in __main__.py, whose TemporaryDirectory
        # cleanup runs before this Python wrapper object is garbage-collected.
        self._open_log_file("",False)
        # Run gc.collect() BEFORE unloading the compiled-equation-code DLLs, not after: any
        # mesh/element C++ object destructed after the DLLs are unloaded would dereference a
        # dangling codeinst/function-table pointer into already-dlclose()'d memory and crash.
        # This ordering only starts to matter once the cycle-breaking above actually lets
        # gc.collect() free such objects here instead of leaving that to interpreter exit.
        gc.collect()
        gc.collect()
        gc.collect()
        self._unload_all_dlls()


    def __exit__(self, type, value, traceback): #type:ignore
        if isinstance(type,Exception):
            raise type
        else:
            self.release()

    def __del__(self):
        # Fallback for scripts that never use "with SomeProblem() as problem:" (or call
        # .release() explicitly) at all: without this, nothing ever breaks the reference
        # cycles/keep_alive-pinning described in release()'s own comments, and this Problem's
        # meshes/elements would remain leaked for the rest of the process. release() is
        # idempotent (guarded by self._released) so this is harmless if release() already ran.
        #
        # Skip entirely once the interpreter itself is shutting down (sys.is_finalizing()):
        # verified empirically (via a debug build of this method) that relying on __del__ to
        # release a Problem that late is fundamentally unsafe, not just risky - Python does not
        # reliably call __del__ at all for objects still referenced only by a module-level
        # global once that module's dict is cleared during interpreter shutdown, and on the
        # rare occasion it is called, even elementary operations inside it (a plain `import`)
        # can fail with "TypeError: 'NoneType' object is not callable" because interpreter
        # teardown has already cleared parts of the import machinery out from under us. There
        # is no reliable amount of internal defensiveness that fixes this from inside __del__
        # itself - the only real fix is for a script to call .release() (or use "with") before
        # the interpreter starts shutting down, while everything __del__/release() depends on
        # is still guaranteed to work. Skipping here just leaves this Problem's memory to the
        # OS, exactly as it already was before this Problem became collectible at all - not a
        # regression, since such scripts never freed this memory during the run anyway.
        # Swallow all exceptions: __del__ can run at arbitrary/uncontrolled times, and an
        # exception escaping __del__ is merely printed as "Exception ignored in..." by Python,
        # never propagated - so there is nothing to gain from letting one through, and every
        # reason to avoid it (e.g. self._meshdict may not exist yet if __init__ failed early).
        if sys.is_finalizing():
            return
        try:
            self.release()
        except Exception:
            pass


    @overload
    def get_mesh(self, name:str,return_None_if_not_found:Literal[False]=...)->AnySpatialMesh: ...

    @overload
    def get_mesh(self, name:str,return_None_if_not_found:Literal[True])->AnySpatialMesh | None: ...

    def get_mesh(self, name:str,return_None_if_not_found:bool=False)->AnySpatialMesh | None:
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
        return self._lookup_mesh(name,return_None_if_not_found)

    def _lookup_mesh(self, name:str,return_None_if_not_found:bool=False)->AnySpatialMesh | None:
        """:py:meth:`get_mesh` without its initialisation guard, for callers that run *during*
        initialisation (where a nested initialise() raises) and know the meshes already exist."""
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

    # --- Static condensation: the rule registry -----------------------------------------------------
    #
    # The usual way to select condensable dofs is the StaticCondensation equation class (see
    # pyoomph/equations/generic.py), which states the selection where it belongs, namely in the
    # equation tree of the domain it applies to. It, and the two methods below, all end up here.
    #
    # A C++ rule holds a resolved pyoomph::Mesh*, which remeshing invalidates outright (the superseded
    # mesh is destroyed, see _destroy_superseded_mesh), so the durable form of a rule on this side is
    # the DOMAIN PATH plus the field spec. The registry holds those, keyed by whoever declared them,
    # and _sync_static_condensation_rules() pushes the whole set down whenever the resolved meshes or
    # the specs have changed - and does nothing whatsoever when they have not. Doing nothing is the
    # point: every add/clear on the C++ side bumps static_condensation_rules_revision, which is part of
    # the Jacobian structure id, so re-registering an unchanged rule set (which the equation class does
    # on every reapply_boundary_conditions()) would otherwise force a plan rebuild and a fresh symbolic
    # factorisation in the solver on every single solve.

    def _declare_static_condensation_rules(self,key:Any,domain:str | None,rules:Sequence[tuple[str,Sequence[int],str]])->None:
        """Register (or re-register) the condensation rules of one declaring object.

        ``key`` identifies the declarer, e.g. ``(id(equation), domainpath)``; re-declaring under the
        same key replaces its rules. ``domain`` is the domain path the rules apply to, or None for a
        problem-wide element-private rule. Re-declaring the same thing does not bump the rules revision:
        the synchronisation below compares before it acts. It is still called every time rather than
        skipped when the entry is unchanged, so that a re-registration also repairs a rule whose mesh
        has since been replaced - which is what the equation class's hook relies on after remeshing.
        """
        normalised=tuple((f,tuple(v),p) for f,v,p in rules)
        self._static_condensation_sources[key]=(domain,normalised)
        self._sync_static_condensation_rules()

    def _sync_static_condensation_rules(self)->None:
        """Push the registry into the C++ rule list, if and only if it now resolves to something else.

        Called after every registration and after remeshing (where the meshes a rule names are replaced
        by new objects, so the rules have to be restated even though nobody edited them)."""
        if not self._static_condensation_sources and self._static_condensation_applied is None:
            return  # nothing was ever declared: do not touch the C++ side, i.e. do not bump the revision
        resolved:list[tuple["AnyMesh | None",tuple[tuple[str,tuple[int,...],str],...]]]=[]
        for domain,rules in self._static_condensation_sources.values():
            # _lookup_mesh, not get_mesh: a StaticCondensation equation registers from within
            # setup_pinning(), i.e. during initialisation, where get_mesh()'s initialise() would raise.
            resolved.append((None if domain is None else self._lookup_mesh(domain),rules))
        # id() is enough to notice a mesh replacement because the resolved meshes are kept referenced
        # below, so their ids cannot be reused while they are part of the signature.
        signature=tuple((id(mesh),rules) for mesh,rules in resolved)
        if signature==self._static_condensation_applied:
            return
        super()._clear_static_condensation_rules()
        for mesh,rules in resolved:
            for field,values,part in rules:
                self._add_static_condensation_rule(mesh,field,list(values),part)
        self._static_condensation_applied=signature
        self._static_condensation_applied_meshes=[mesh for mesh,_ in resolved if mesh is not None]

    def _clear_static_condensation_rules(self)->None:
        """Drop every static condensation rule, and the registry they are restated from.

        Overrides the C++ binding of the same name so that a cleared rule does not come back at the
        next synchronisation. A :py:class:`~pyoomph.equations.generic.StaticCondensation` equation in
        the tree does re-register itself when the boundary conditions are next applied, since the
        equation tree is the declaration - use ``use_static_condensation = False`` to switch the
        feature off instead."""
        self._static_condensation_sources.clear()
        self._static_condensation_applied=None
        self._static_condensation_applied_meshes=[]
        super()._clear_static_condensation_rules()

    def condense_dofs(self,field:str,*,values:list[int] | None=None,part:str="all")->None:
        """Select degrees of freedom that static condensation may eliminate from the linear system.

        The equation-tree way of saying this is :py:class:`~pyoomph.equations.generic.StaticCondensation`,
        which is the recommended interface - ``eqs += StaticCondensation(velocity="bubble", pressure=[1,2])``
        states the selection on the domain it belongs to and switches the feature on by itself. This method is
        the problem-level plumbing underneath it, useful when the selection is decided outside the equations.

        **Experimental.** Static condensation eliminates element-local unknowns from the Jacobian
        before it reaches the linear solver and reconstructs them after the Newton update. It is exact - the solution
        is the same, iteration by iteration - but only Newton solves benefit: residual evaluations, eigenvalue and
        Hessian assemblies, and arclength continuation always see the full system, while Jacobian reuse and the
        globally convergent (line search) Newton method are refused with an error rather than ignored. Distributed
        (``--distribute``) runs are supported; a selection whose coupled block would be split across ranks is
        refused, collectively, and a replicated MPI run (``mpirun`` without ``--distribute``) is refused too.

        This only declares a rule. Unlike :py:class:`~pyoomph.equations.generic.StaticCondensation`, it does not
        switch condensation on: set ``use_static_condensation=True`` for that. Rules are stated in terms of a domain
        and a field, so they survive mesh adaptation and remeshing, and several rules can be added, their selections
        being unioned. Pinned values are never selected.

        The classical Crouzeix-Raviart elimination needs two rules, and needs both: the bubble velocities and the
        gradient modes of the DL pressure are only invertible together, and the constant pressure mode must stay a
        global unknown (taking it as well is refused, with an explanation)::

            problem.condense_dofs("domain/velocity", part="bubble")
            problem.condense_dofs("domain/pressure", values=[1,2])   # [1,2,3] in 3d
            problem.use_static_condensation = True

        Args:
            field (str): Full path of the field, i.e. ``"domain/fieldname"`` (or ``"domain/subdomain/fieldname"``). A vector field
                may be given by its base name, e.g. ``"domain/velocity"``, which selects all of its components.
            values (Optional[List[int]], optional): Restrict the selection to these value indices of the elemental data,
                e.g. ``[1,2]`` for the gradient modes of a DL pressure in 2d. Defaults to all values.
            part (str, optional): ``"all"``/``"internal"`` for an elemental (DL/D0/DG) field, ``"bubble"`` for the
                cell-interior bubble nodes of a nodal C1TB/C2TB field, or ``"DL_gradients"`` for every value of a
                ``"DL"`` field except the constant, i.e. ``values=[1,2]`` in 2d and ``[1,2,3]`` in 3d, taken from
                the dimension of the domain rather than from the script. Defaults to ``"all"``.
        """
        splt=field.split("/")
        if len(splt)<2:
            raise RuntimeError("condense_dofs expects a full field path 'domain/fieldname', but got '"+field+"'")
        domain="/".join(splt[:-1])
        mesh=self.get_mesh(domain)  # resolved here as well, so a wrong domain path is an error at the call site
        if part=="DL_gradients":
            # Not a part the C++ side knows: it is a name for "values 1..dim", which needs the mesh.
            if values is not None:
                raise RuntimeError("condense_dofs("+field+", part='DL_gradients') already says which values are meant, so it cannot be combined with values="+str(values)+".")
            from ..meshes.mesh import assert_spatial_mesh
            values=list(range(1,assert_spatial_mesh(mesh).get_dimension()+1))
            part="all"
        self._static_condensation_source_counter+=1
        self._declare_static_condensation_rules(("condense_dofs",self._static_condensation_source_counter),
                                                domain,[(splt[-1],() if values is None else tuple(values),part)])

    def condense_element_private_dofs(self,domain:str | None=None)->None:
        """Select every elemental (internal) degree of freedom that no other element reads for static condensation.

        The equation-tree way of saying this is ``eqs += StaticCondensation()``, i.e. a
        :py:class:`~pyoomph.equations.generic.StaticCondensation` without arguments, which is restricted to the
        domain it is added to and switches the feature on by itself. This method is the problem-level plumbing.

        **Experimental** - see :py:meth:`condense_dofs` for what that means. This is the convenient
        rule for auxiliary fields projected onto a discontinuous space (DL/D0/DG), e.g. a dissipation or a stress
        measure computed for output: they are unknowns of the system but couple to nothing outside their element, so
        condensing them removes them from the matrix the solver factorises without changing anything else.

        Internal data adopted as external data elsewhere - by an interface element on a free surface, or by an
        interior-facet DG coupling - is excluded, since such a dof is not element-local at all. That test is always
        made against the whole problem, even when ``domain`` restricts which dofs are considered. (A dof named
        explicitly by :py:meth:`condense_dofs` is *not* excluded this way; the elimination handles those correctly
        too, this rule simply has no way of knowing whether they were meant.) Like :py:meth:`condense_dofs`, this
        only declares a rule; ``use_static_condensation`` is a separate switch.

        Args:
            domain (Optional[str], optional): Restrict the rule to this domain. Defaults to every bulk domain.
        """
        if not self._initialised:
            self.initialise()
        if domain is not None:
            self.get_mesh(domain)
        self._static_condensation_source_counter+=1
        self._declare_static_condensation_rules(("condense_element_private_dofs",self._static_condensation_source_counter),
                                                domain,[("",(),"element_private")])

    def get_all_values_at_current_time(self,with_pos:bool)->tuple[NPFloatArray,NPBoolArray,NPFloatArray]:
        dofs,positional_dof=self.get_current_dofs()
        pinned=self.get_current_pinned_values(with_pos)
        return numpy.array(dofs),positional_dof,numpy.array(pinned) #type:ignore

    def set_all_values_at_current_time(self,dofs:NPFloatArray | list[float],pinned:NPFloatArray | list[float],with_pos:bool):
        self.set_current_dofs(dofs) #type:ignore
        self.set_current_pinned_values(pinned,with_pos) #type:ignore


    def setup_pinned_values_of_eigenfunction(self,pv:NPFloatArray,n:int,mode:"MeshDataEigenModes")->NPFloatArray:	#Can be customised
        return 0.0*pv #type:ignore	 # Default: All pinned values are zero

    @overload
    def set_eigenfunction_as_dofs(self,n:int,*,mode:"MeshDataEigenModes"="abs",additive_mesh_positions:bool=True,perturb_amplitude:Literal[None]=...)->tuple[NPFloatArray,NPFloatArray]: ...
    
    @overload
    def set_eigenfunction_as_dofs(self,n:int,*,mode:"MeshDataEigenModes"="abs",additive_mesh_positions:bool=True,perturb_amplitude:float)->tuple[NPFloatArray,NPFloatArray,float]: ...

    def set_eigenfunction_as_dofs(self,n:int,*,mode:"MeshDataEigenModes"="abs",additive_mesh_positions:bool=True,eigenvector_position_scale:float | None=None,perturb_amplitude:float | None=None)->tuple[NPFloatArray, NPFloatArray, float] | tuple[NPFloatArray, NPFloatArray]:
        if n>=len(self._last_eigenvectors):
            raise RuntimeError("Cannot set eigenfunction "+str(n)+" as dofs, since we have calculated only "+str(len(self._last_eigenvectors))+" eigenfunctions")
        # A base-length eigenvector would otherwise be silently zero-padded to the augmented length by
        # the numpy.pad below, i.e. written over the base dofs AND over the tracker's own unknowns.
        self._require_no_bifurcation_tracking("Pushing an eigenfunction into the dofs (set_eigenfunction_as_dofs)")
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

    def set_coordinate_system(self,csys:Literal["axisymmetric", "axisymmetric_flipped", "cartesian", "radialsymmetric"] | BaseCoordinateSystem):                
        """Set the default coordinate system at problem level. 
        You can specify coordinate systems also at equation level, but if you don't do, the coordinate system will default to this one.

        Args:
            csys (Union[Literal["axisymmetric","axisymmetric_flipped","cartesian","radialsymmetric"],BaseCoordinateSystem]): The coordinate system to set as default.

        Raises:
            RuntimeError: Raised in case we do not set a valid coordinate system 
        """
        if csys is None:
            raise RuntimeError("Cannot set the problem coordinate system to None")
        csysd:BaseCoordinateSystem
        if isinstance(csys,str):
            if csys=="axisymmetric":
                csysd=axisymmetric
            elif csys=="axisymmetric_flipped":
                csysd=axisymmetric_flipped
            elif csys=="cartesian":
                csysd=cartesian
            elif csys=="radialsymmetric":
                csysd=radialsymmetric
            else:
                raise RuntimeError("Unknown coordinate system: "+csys)
        elif not isinstance(csys,BaseCoordinateSystem):
            raise RuntimeError("Unknown coordinate system: "+str(csys))
        else:
            csysd=csys
        self._coordinate_system=csysd


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
        scale:"str | ExpressionOrNum | None"=s
        while isinstance(scale,str):
            scale=self.scaling.get(scale,None if none_if_not_set else 1)
            if scale is None:
                return None
        if isinstance(scale,int) or isinstance(scale,float):
            scale=_pyoomph.Expression(scale)
        return scale

    def set_scaling(self,**kwargs:ExpressionOrNum | str)->None:
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
                    from .. import _pyoomph_core as _pyoomph
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


    def set_eigensolver(self,solv:str | GenericEigenSolver):
        """
        Set the eigensolver backend. "scipy", "pardiso", "slepc" are available (the latter two only if the packages MKL and/or petsc4py/slepc4py are installed)

        Returns:
            The eigenproblem solver instance after setting
        """
        if isinstance(solv,str):
            solv=GenericEigenSolver.factory_solver(solv,self)
        self._eigensolver=solv        
        if not self.is_quiet():
            print("EIGEN SOLVER WAS SET TO: "+self._eigensolver.idname)
        return self._eigensolver

    def set_linear_solver(self,solv:str | GenericLinearSystemSolver):
        
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
        if not self.is_quiet():
            print("LINEAR SOLVER WAS SET TO: "+self._lasolver.idname)
        return self._lasolver

    def set_num_threads(self,nthread:int | None):
        """
        Set how many threads the linear solver may use. Most direct solvers are internally threaded, so
        this is what decides whether a serial run keeps one core or several busy.

        The count is remembered on the problem, not just handed to the current solver: it is applied
        again to any backend selected later with :py:meth:`set_linear_solver`. Passing ``None`` leaves
        the backend at its own default, which is usually whatever ``OMP_NUM_THREADS`` says.

        Under ``mpirun`` this matters mainly for the solvers that are not MPI-parallel: they gather the
        system onto rank 0 and solve it there, and the other ranks sleep so that rank 0 may use their
        cores - which it can only do if the launcher did not pin each process to one (``mpirun
        --bind-to none``).

        Args:
            nthread (Optional[int]): Number of threads, or ``None`` for the backend's own default.
        """
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


    def _require_non_distributed(self,what:str)->None:
        """Stop with a clear message if ``what`` is being attempted on a distributed problem.

        These callers assemble the eigenproblem matrices themselves and read the result as a square
        global matrix, which it stops being once the rows are partitioned across ranks -- scipy then
        raises about an indptr length, several frames away from the feature the user asked for. The
        augmented-system ones (bifurcation tracking, periodic orbits) additionally sit on top of
        sparse_assemble_row_or_column_compressed_base_problem, which throws "This likely does not work
        in distributed parallel" from C++. Failing here names the feature instead.
        """
        if self.is_distributed():
            raise RuntimeError(what+" is not supported on a distributed (--distribute) problem yet. Run it without --distribute (plain eigenvalue solving via SLEPc does work distributed).")

    def _require_no_bifurcation_tracking(self,what:str)->None:
        """Stop with a clear message if ``what`` is being attempted while a tracker is installed.

        For the callers that reach the dof vector by global equation number: while tracking, the dofs
        are the augmented ones (base dofs, then the eigenvector blocks and the scalars), so a
        base-length vector written into them lands partly in the eigenvector block. Solving the
        *eigenproblem* itself is fine and no longer refused -- see solve_eigenproblem().
        """
        if self.get_bifurcation_tracking_mode()!="":
            raise RuntimeError(what+" is not possible while bifurcation tracking is active, since the dof vector is then the augmented one (base dofs plus the eigenvector and the scalar unknowns). Call deactivate_bifurcation_tracking() first.")

    def _dirichlet_activation_snapshot(self)->list[tuple["AnyMesh",list[bool]]]:
        """Per-mesh snapshot of which Dirichlet conditions are currently active.

        Needed because ``_before_eigen_solve`` both *reports* that the equations must be renumbered
        and *has already flipped the flags* by the time it returns (AxisymmetryBC deactivates the
        strong axis conditions at m != 0). While a tracker is installed a renumbering is not allowed,
        so the flags have to be put back before refusing -- otherwise the problem is left describing
        boundary conditions that its equation numbering does not have.
        """
        res:list[tuple["AnyMesh",list[bool]]]=[]
        for mesh in self._iterate_all_meshes():
            res.append((mesh,list(mesh._get_dirichlet_active_flags()))) #type:ignore
        return res

    def _iterate_all_meshes(self)->Iterable["AnyMesh"]:
        """All bulk meshes and, recursively, the interface meshes hanging off them.

        _meshdict holds only the bulk meshes, but a Dirichlet condition (and hence a mode-dependent
        activation flag) can live on an interface just as well.
        """
        def _walk(m:"AnyMesh")->Iterable["AnyMesh"]:
            yield m
            for imesh in getattr(m,"_interfacemeshes",{}).values():
                yield from _walk(imesh)
        for topmesh in list(self._meshdict.values()):
            yield from _walk(topmesh)

    def _restore_dirichlet_activation(self,snapshot:list[tuple["AnyMesh",list[bool]]])->None:
        for mesh,flags in snapshot:
            mesh._set_dirichlet_active_flags(flags) #type:ignore

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

    def output_at_increased_time(self,dt:ExpressionOrNum | None=None)->None:
        """
        Increases the current time by the specified time step (dt, default scale_factor("temporal")) and calls the output method.
        Useful for Paraview PVD output of multiple stationary solutions, which otherwise overlays multiple outputs at the same time step.

        Args:
            dt (Optional[ExpressionOrNum]): The time step to increase the current time by. If not provided,
                the scaling factor for temporal is used.

        Returns:
            None

        .. deprecated::
            Use :py:meth:`output` with ``increase_time_for_PVD=True`` instead (or
            ``increase_time_for_PVD=dt`` for an explicit time step).
        """
        warnings.warn(
            "Problem.output_at_increased_time() is deprecated. Use "
            "Problem.output(increase_time_for_PVD=True) instead, or pass an explicit time step, "
            "i.e. Problem.output(increase_time_for_PVD=dt).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.output(increase_time_for_PVD=True if dt is None else dt)

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
                
                
    def create_eigendynamics_animation(self,outdir:str,plotter:"MatplotlibPlotter",eigenvector:int=0,init_amplitude:float | None=None,max_amplitude:float | None=None,numperiods:float=1,numouts:int=25,phi0:float=0):
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
        eigenmodes_m=self.get_last_eigenmodes_m()
        if eigenmodes_m is not None and eigenmodes_m[eigenvector]!=0:
            additional_factor_right=numpy.exp(1j*eigenmodes_m[eigenvector]*phi0)
            additional_factor_left=numpy.exp(1j*eigenmodes_m[eigenvector]*(phi0+numpy.pi))
            plotter._eigenanimation_m=eigenmodes_m[eigenvector]
            
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


    def output(self, stage: str = "", quiet: bool | None = None, increase_time_for_PVD: bool | ExpressionOrNum = False) -> None:
        """
        Invoke an output of the current solution at the current time by calling all Output objects.

        Args:
            stage (str): The stage of the output, at the moment, only "" is meaninfull.
            quiet (bool, optional): Flag to control the verbosity of the output.
            increase_time_for_PVD (bool | ExpressionOrNum): Advance the current time before writing.
                A Paraview PVD collection indexes its entries by time, so a sequence of stationary
                solutions - a parameter scan, a bifurcation diagram - all written at the same time
                would overlay each other there. ``True`` steps the time by ``scale_factor("temporal")``;
                pass a time step instead of ``True`` to control the increment.

        Returns:
            None
        """
        if not self.is_initialised():
            self.initialise()
        if increase_time_for_PVD is not False and increase_time_for_PVD is not None:
            dt = self.get_scaling("temporal") if increase_time_for_PVD is True else increase_time_for_PVD
            self.set_current_time(self.get_current_time() + dt)
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

        statefname: str | None = None
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
            if not self.write_states:
                raise RuntimeError("Plotting process is active, but write_states is False. Please set write_states to True to use the plotting process")
            print("State file written, invoking plotting process")
            assert self._plotting_process.stdin is not None
            assert statefname is not None
            self._plotting_process.stdin.write((statefname + "\n").encode("utf-8"))
            self._plotting_process.stdin.flush()


            self._output_step += 1  # Write with the updated outstep here ??
        else:
            self._output_step += 1




    def output_every_step_outputs(self, stage: str = "") -> None:
        """
        Invoke only those Output objects that were created with ``output_every_step=True`` (e.g.
        :py:class:`~pyoomph.output.generic.ODEFileOutput` and
        :py:class:`~pyoomph.output.generic.IntegralObservableOutput`). Called by
        :py:meth:`run` after every successful transient step that is not already an output step, so
        that these line-per-call files resolve the whole trajectory instead of just the output times.

        Deliberately none of the rest of :py:meth:`output`: no state dump, no plot, no "OUTPUT at t=..."
        line, and in particular no increment of the output step counter - the numbered outputs (mesh
        files, states, plots) must stay in step with the requested output times.

        Args:
            stage (str): The stage of the output, at the moment, only "" is meaningful.

        Returns:
            None
        """
        if not self.is_initialised():
            return
        self._equation_system._do_output(self._output_step, stage, only_every_step=True)


    def init_output(self,redefined:bool=False):
        cinfo:dict[str,Any] | None=None
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


    

    def _adapt_with_interfacial_errors(self) -> tuple[int, int]:
        biftrack_active,biftrack_eigen=self._get_bifurcation_tracking_info()
        biftrack_mode = self.get_bifurcation_tracking_mode()
        biftrack_param = self._bifurcation_tracking_parameter_name
        self._bifurcation_reactivation_after_adaptation=None # We will reactivate the bifurcation tracking after the adaptation, but not during the adaptation, to avoid that we adapt with changing eigenvalues during the adaptation. We will reactivate it at the end of the function if it was active at the beginning.
        if biftrack_active:
            print("ADAPT WITH INTERFACIAL ERRORS. BIF TRACKING PARAM: ",self._bifurcation_tracking_parameter_name,self.get_bifurcation_tracking_mode())
            m=None
            k=None
            self._last_eigenvalues=numpy.array([biftrack_eigen])
            self._last_eigenvectors=numpy.array([self._get_bifurcation_eigenvector()])            
            if biftrack_mode=="azimuthal" or biftrack_mode=="cartesian_normal_mode":
                print("Azimuthal:",self._azimuthal_mode_param_m,self._azimuthal_mode_param_m.value) #type:ignore
                print("Cartesian normal mode:",self._cartesian_normal_mode_param_k,self._cartesian_normal_mode_param_k.value) # type:ignore
                raise RuntimeError("Check on the bifurcation tracking, whether the values of m and k are still correct")
            self._adapt_eigenindex=0 # We will adapt with the first eigenfunction, which is the one that is critical at the bifurcation point. We could also make this user-definable in the future
            self.deactivate_bifurcation_tracking()            
            # Carry the eigenvector-scaling choice across the adaption too, so that reactivating
            # does not silently drop back to the default constraint scaling.
            self._bifurcation_reactivation_after_adaptation={"mode":biftrack_mode,"param":biftrack_param,"azimuthal_m":m,"cartesian_k":k,"eigenvector_scaling":self._bifurcation_eigenvector_scaling}
        #Resetting the element error override
        if self._custom_assembler is not None:
            raise RuntimeError("Adaption with custom assembler not supported yet")
        def reset(mesh:AnySpatialMesh):
            mesh._reset_elemental_error_max_override() 
            for _n,imesh in mesh._interfacemeshes.items():
                reset(imesh)
        for name,mesh in self._meshdict.items():
            if isinstance(mesh,ODEStorageMesh): continue
            reset(mesh)
            mesh_errs = mesh.get_elemental_errors()
            for i,b in enumerate(mesh.elements()):
                b._elemental_error_max_override=mesh_errs[i]
                
        if self._adapt_eigenindex is not None and self._last_eigenvectors is not None and len(self._last_eigenvectors)>self._adapt_eigenindex:
            _pyoomph.set_use_eigen_Z2_error_estimators(True)
            evect=self._last_eigenvectors[self._adapt_eigenindex]
            has_imag=numpy.amax(numpy.absolute(numpy.imag(evect)))>0.00001*numpy.amax(numpy.absolute(numpy.real(evect)))
            #print("EIG AS DOF")
            backup,backup_pinned=self.set_eigenfunction_as_dofs(self._adapt_eigenindex,mode="real")
            for name,mesh in self._meshdict.items():
                if isinstance(mesh,ODEStorageMesh): continue            
                #print("EVEM ",name)
                mesh_errs = mesh.get_elemental_errors()
                for i,b in enumerate(mesh.elements()):
                    b._elemental_error_max_override=max(mesh_errs[i],b._elemental_error_max_override)
            #print("DONE")
            if has_imag:
                #print("HAS IMAG",numpy.amax(numpy.absolute(numpy.imag(evect))),numpy.amax(numpy.absolute(numpy.real(evect))))
                self.set_eigenfunction_as_dofs(self._adapt_eigenindex,mode="imag")
                for name,mesh in self._meshdict.items():
                    if isinstance(mesh,ODEStorageMesh): continue            
                    mesh_errs = mesh.get_elemental_errors()
                    for i,b in enumerate(mesh.elements()):
                        b._elemental_error_max_override=max(mesh_errs[i],b._elemental_error_max_override)
            #print("DONE IMAG")
            #print(backup)
            #print(backup_pinned)
            self.set_all_values_at_current_time(backup,backup_pinned,False)
            # Restoring the dof vector puts the MASTERS back, but a hanging node has no dof: its position
            # and values live in its own raw storage, which the error estimation above has meanwhile
            # overwritten with the eigenfunction state (evaluating an element calls interpolate_hang_values).
            # Nothing refreshes that storage before the refinement that follows, and a stale hanging
            # POSITION is not just cosmetic: the triangle/tetrahedron node-sharing in
            # RefineableTElement<2>/<3>::build looks a candidate son node up in a snapshot of the existing
            # node positions, so a hanging node sitting where the eigenfunction put it is not recognised
            # and the son builds a SECOND node on top of it. That tears the mesh at those nodes and leaves
            # it with more nodes than the stored refinement pattern reproduces, so the saved state can no
            # longer be loaded. Push the restored state back into the hanging nodes. The interface meshes
            # are included because their elements own the hangs of the dofs an interface ADDS to a node,
            # which the bulk pass does not touch (see InterfaceElementBase::interpolate_hang_values_at_interface).
            def repair_hanging(mesh:AnySpatialMesh):
                mesh._interpolate_hanging_values()
                for _n,imesh in mesh._interfacemeshes.items():
                    repair_hanging(imesh)
            for _name,mesh in self._meshdict.items():
                if isinstance(mesh,ODEStorageMesh): continue
                repair_hanging(mesh)
            _pyoomph.set_use_eigen_Z2_error_estimators(False)
            #print("RESET")

        # The desired_ndof controller runs HERE, on the raw estimator errors and before any override
        # stage, and the position is not a matter of taste. The overrides below encode their verdict
        # as a magic error value relative to the thresholds -- must_refine = 100*max_permitted_error,
        # may_not_unrefine = 0.5*(max+min) -- so a controller that moved the thresholds afterwards
        # would leave those sentinels sitting in the wrong place. "May not unrefine" in particular
        # only stays above min_permitted_error while the thresholds keep roughly their original
        # ratio, so lowering max after the fact silently unrefines every element an interface or a
        # RefineAccordingToElement callback asked to protect. Deciding the thresholds first means the
        # sentinels are computed from the new values and the invariant holds by construction.
        # See dev_docs/spatial_error_estimators.md sections 4 and 5.
        self._apply_desired_ndof_controller()

        if True:
            #Now, we first have to go through all meshes at the deepest level in the tree
            def get_errs(mesh:AnySpatialMesh,depth:int):
                if not mesh.refinement_possible():
                    return
                if depth==0:
                    #print("GET ERRS ON MESH",mesh,mesh.get_name())
                    # The declarative criteria (RefineToLevel, RefineMaxElementSize) are evaluated in C++
                    # over every element this process holds, halo copies included, so they need no
                    # synchronisation afterwards. Only criteria that genuinely need Python -- a
                    # user-supplied callback -- are left to calculate_error_overrides below.
                    mesh._apply_refinement_directives()
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


            errs:dict[str,list[float]]={}
            for name,mesh in self._meshdict.items():
                if isinstance(mesh,ODEStorageMesh): continue
                assert not isinstance(mesh,InterfaceMesh)
                errs[name]=[e._elemental_error_max_override for e in mesh.elements()]
                #errs[name]=mesh._merge_my_error_with_elemental_max_override()   # This is done in advance now
                

            # Interfaces with an opposite side used to get a one-shot error nudge here -- if my bulk
            # element was above min_permitted_error and the opposite one was not, the opposite one's
            # error was bumped, and vice versa. That is gone. It could only ever approximate the goal,
            # for the reason set out in dev_docs/interface_refinement_coupling.md section 6: oomph merges
            # a father only if ALL of its sons agree, so an unrefinement vetoed by a son that does not
            # touch the interface is invisible to any comparison made at the interface, whatever it does
            # with the errors. It also never iterated (so an A-B-C chain stayed open), and under MPI
            # get_opposite_bulk_element() may return an element this rank does not own, or nothing at all.
            #
            # The two sides are now reconciled on their FLAGS instead, after both meshes have decided and
            # before either acts (see the adapt call below), which is exact; and whatever still slips
            # through is repaired by Problem.enforce_interface_conformity() afterwards.
            #
            # The interior-facet skeleton itself adapts fine (it is torn down and regenerated from the
            # refined bulk mesh, with its DL/D0 fields carried across by the snapshot/restore in
            # InterfaceMesh::rebuild_after_adapt). What is rejected here is only the exotic combination
            # of a skeleton that has ALSO been connected to a second mesh with ConnectMeshAtInterface:
            # the flag reconciliation below assumes the two sides are ordinary named boundaries with a
            # 1:1 element correspondence, which a facet soup does not have.
            for name,mesh in self._meshdict.items():
                if isinstance(mesh,ODEStorageMesh): continue
                for inam,imesh in mesh._interfacemeshes.items():
                    if imesh._opposite_interface_mesh is not None and inam=="_internal_facets_":
                        raise RuntimeError("The interior-facet skeleton '"+str(name)+"/_internal_facets_' has been connected to another interface mesh; adapting that combination is not supported.")


        messed_around_in_history=False
        has_arclength_data=False
        if self._last_arclength_parameter is not None:
            dof_deriv=self.get_arclength_dof_derivative_vector()
            if len(dof_deriv)>0:
                has_arclength_data=True
                _actual_dofs,_positional_dofs,pinned_values=self.get_all_values_at_current_time(True)            
                dof_current=self.get_arclength_dof_current_vector()
                self.set_current_pinned_values(0*pinned_values,True,5)
                self.set_current_pinned_values(0*pinned_values,True,6)
                if len(dof_deriv)>len(_actual_dofs):
                    # Strip the bifurcation tracker part... There is nothing you can do here
                    dof_deriv=dof_deriv[:len(_actual_dofs)]
                    dof_current=dof_current[:len(_actual_dofs)]                    
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
        # Adapt in two stages with a gap in between: decide for every mesh, then act. The gap is where
        # the two sides of a coupled interface are made to agree about what they decided, which is the
        # one point where they can be brought into agreement EXACTLY -- reconciling the errors instead
        # (which is what the block further up does, and all pyoomph could do before) cannot: oomph
        # merges a father only if all of its sons agree, so an unrefinement vetoed by a son that does
        # not touch the interface is invisible to any comparison made at the interface. See
        # dev_docs/interface_refinement_coupling.md sections 6 and 7.
        #
        # The gap is also where an adaptation that is not going to change anything is abandoned, see
        # below. Both are the reason the uncoupled case goes through the same two stages rather than
        # through adapt_by_elemental_errors(): that call is exactly select+execute+finalise, so there
        # is nothing to lose, and having one path means the abandoning is not tied to being coupled.
        #
        # Order matters and is deterministic on every process: errs follows _meshdict's insertion
        # order, and the refinement calls underneath are collective on a distributed mesh.
        adaptable=[(n,e) for n,e in errs.items() if self.get_mesh(n).refinement_possible()]
        coupled=self._collect_coupled_interfaces()
        # Negative-testing hatch, same purpose as PYOOMPH_DISABLE_INTERFACE_CONFORMITY: with this
        # set, the two sides act on their own decisions and the post-adapt repair has to clean up
        # afterwards -- which is exactly the lossy behaviour the reconciliation exists to avoid, and
        # the only way to demonstrate that it does.
        if os.environ.get("PYOOMPH_DISABLE_ADAPT_RECONCILIATION","") not in ("","0","off"):
            coupled=[]
        for name,errors in adaptable:
            self.get_mesh(name)._adapt_select(errors)
        if coupled:
            _pyoomph._harmonise_adapt_selection(coupled,40) #type:ignore

        # Deciding to do nothing is not a rare case but the normal end state: any mesh sitting at
        # max_refinement_level with errors still above the refinement tolerance decides it on every
        # solve, and oomph only leaves its own adaption loop once an adapt() has reported 0/0, so as
        # long as spatial_adapt>0 the last adaptation of every step is a no-op by construction. Acting
        # on it anyway is not free: actions_before_adapt() tears down every interface mesh,
        # actions_after_adapt() rebuilds them, and assign_eqn_numbers() invalidates the Jacobian
        # sparsity pattern unconditionally (Problem::invalidate_jacobian_structure), so the frozen
        # sparsity is rebuilt for a numbering that did not change. That is what is skipped here.
        #
        # One thing survives the skip: the node ORDER. An executed adaptation puts the nodes into the
        # order the elements walk them, and oomph-lib does the same in the branch of
        # execute_selected_adaptation() that decides not to bother, on purpose - "required to allow
        # dump/restart on refined meshes". Skipping that too is what the first version of this did, and
        # since the no-op adaptation is universal, it was what made every run agree on the order. Runs
        # then disagreed depending on the route to the mesh - load_state(), a real refinement and a
        # distribution rebuild reorder, a plain run did not - and whatever compared two of them compared
        # permuted states: restarts stopped being bit-identical (every value still exact, only its
        # position moved), distributed runs stopped matching their serial reference, and a script
        # reading mesh data back in coordinate order got all vertices first and the midside nodes
        # afterwards. Nightly 20260816 failed 10 tests and linear_response_drum.py on it.
        #
        # So the order is established here instead, and cheaply: the reordering is idempotent, so it
        # moves something only the first time it is reached (in practice the initial adaptation) and the
        # renumbering happens only then. Every later no-op adaptation finds the order already canonical
        # and keeps both the interface meshes and the sparsity pattern.
        #
        # The exception is a COUPLED interface on a DISTRIBUTED problem, which still goes the long way
        # round. There the node order is not the only thing an executed adaptation leaves behind: with
        # the interface geometry itself an unknown (ConnectMeshAtInterface), a rank that skips the
        # teardown keeps four dofs a serial run does not have - ale-tri_left in
        # test_mpi_interface_coupling.py, 2942 against 2938, reproducibly. Reordering does not repair
        # that, so whatever the post-adapt repair does there has to be understood before the skip can
        # cover it; the plain distributed case is not affected and does skip.
        #
        # Summed over the processes before either decision is taken: assign_eqn_numbers() is collective,
        # so ranks disagreeing about a branch would deadlock rather than diverge.
        npending=0
        for name,_ in adaptable:
            nr,nu=self.get_mesh(name)._adapt_pending_counts()
            npending+=nr+nu
        if get_mpi_nproc()>1:
            npending=int(get_mpi_sum(npending))

        if npending==0 and not (coupled and self.is_distributed()):
            reordered=0
            for name,_ in adaptable:
                mesh=self.get_mesh(name)
                mesh._adapt_abandon()
                if mesh._reorder_nodes_if_needed():
                    reordered+=1
            if get_mpi_nproc()>1:
                reordered=int(get_mpi_sum(reordered))
            if reordered:
                # First time here: the nodes moved, so the numbering has to follow them.
                with self.custom_adapt(True):
                    pass
            if not self.is_quiet():
                print("Nothing to refine or unrefine: leaving the meshes alone"
                      +(" (nodes reordered once)" if reordered else ""))
        else:
            with self.custom_adapt(True):
                for name,_ in adaptable:
                    self.get_mesh(name)._adapt_execute()
                for name,_ in adaptable:
                    self.get_mesh(name)._adapt_finalise()
                for name,_ in adaptable:
                    mesh=self.get_mesh(name)
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
            self._last_eigenvectors=numpy.array([eigfunc]) #type:ignore
            self._last_eigenvalues=numpy.array([self._last_eigenvalues[self._adapt_eigenindex]]) #type:ignore
            lastm,lastk=None,None
            if self._last_eigenvalues_m is not None:
                self._last_eigenvalues_m=numpy.array([self._last_eigenvalues_m[self._adapt_eigenindex]]) #type:ignore
                lastm=self._last_eigenvalues_m[0]
            if self._last_eigenvalues_k is not None:
                self._last_eigenvalues_k=numpy.array([self._last_eigenvalues_k[self._adapt_eigenindex]]) #type:ignore
                lastk=self._last_eigenvalues_k[0]
            self._adapted_eigeninfo=[eigfunc,self._last_eigenvalues[0],lastm,lastk]
            
        if messed_around_in_history:
            self.assign_initial_values_impulsive() # We messed around. So me must reassign the initial values
        self._adapt_eigenindex=None
        return nref,nuref

    def _desired_ndof_meshes(self) -> list[tuple[str,"AnySpatialMesh"]]:
        """The meshes the desired_ndof controller may act on: the bulk meshes that can actually be
        refined. Interface meshes are excluded because they are never adapted in their own right -
        they follow the bulk mesh, via the error overrides."""
        res:list[tuple[str,AnySpatialMesh]]=[]
        for name,mesh in self._meshdict.items():
            if isinstance(mesh,ODEStorageMesh): continue
            assert not isinstance(mesh,InterfaceMesh)
            if mesh.refinement_possible():
                res.append((name,mesh))
        return res

    def _global_error_order_statistic(self,errors:"NPFloatArray",count:int,largest:bool) -> float:
        """The value of the `count`-th largest (or smallest) error over ALL processes.

        Serial takes a partial sort. Under MPI the errors live on different ranks, so instead of
        gathering them the threshold itself is bisected, counting on each rank and MPI-summing the
        counts: a fixed number of cheap collective reductions, independent of how the mesh happens to
        be distributed, and giving every rank the same answer by construction."""
        import numpy
        if count<=0:
            return numpy.inf if largest else -numpy.inf
        if get_mpi_nproc()<=1:
            if count>=len(errors):
                return float(numpy.min(errors)) if largest else float(numpy.max(errors))
            srt=numpy.partition(errors,-count if largest else count-1)
            return float(srt[-count] if largest else srt[count-1])
        lo=float(get_mpi_min(numpy.min(errors) if len(errors) else numpy.inf))
        hi=float(get_mpi_max(numpy.max(errors) if len(errors) else -numpy.inf))
        if not (hi>lo):
            return hi
        # 60 bisections take the bracket to ~1e-18 of its width, far below any tie the caller could
        # act on differently. Every rank runs the same loop on the same summed counts, so they all
        # come out with the same threshold - which they must, since it becomes a mesh-wide tolerance.
        for _ in range(60):
            mid=0.5*(lo+hi)
            if largest:
                # count(err > mid) is decreasing in mid: keep the half where at least `count` remain.
                n=int(get_mpi_sum(int(numpy.count_nonzero(errors>mid))))
                lo,hi=(mid,hi) if n>=count else (lo,mid)
            else:
                # count(err < mid) is increasing in mid: keep the half where at most `count` are below.
                n=int(get_mpi_sum(int(numpy.count_nonzero(errors<mid))))
                lo,hi=(lo,mid) if n>=count else (mid,hi)
        return 0.5*(lo+hi)

    def _restore_thresholds_before_desired_ndof(self):
        """Put back the min/max_permitted_error the user chose, once desired_ndof is unset again."""
        saved=getattr(self,"_desired_ndof_saved_thresholds",None)
        if not saved:
            return
        for name,(mn,mx) in saved.items():
            mesh=self.get_mesh(name,return_None_if_not_found=True)
            if mesh is not None:
                mesh.min_permitted_error=mn
                mesh.max_permitted_error=mx
        self._desired_ndof_saved_thresholds=None

    def _apply_desired_ndof_controller(self):
        """Turn a target problem size into refine/unrefine thresholds for this adaptation step.

        The controller is *ordinal*: it picks an order statistic of the error distribution and puts
        the threshold there. It never uses the magnitude of an error, so on a single mesh it does not
        care how the estimator is normalised. The normalisation only matters when the ranking is
        pooled across several meshes, which is what makes the errors of one mesh comparable with
        another's - see SpatialErrorEstimator's normalize_relative and weight, and
        dev_docs/spatial_error_estimators.md section 7.
        """
        import numpy
        if self.desired_ndof is None:
            self._restore_thresholds_before_desired_ndof()
            return
        meshes=self._desired_ndof_meshes()
        if not meshes:
            return
        if getattr(self,"_desired_ndof_saved_thresholds",None) is None:
            self._desired_ndof_saved_thresholds={n:(m.min_permitted_error,m.max_permitted_error) for n,m in meshes}

        # Only elements that can still move count towards the model. An element already at
        # max_refinement_level cannot be refined however large its error, and one at
        # min_refinement_level cannot be merged away; counting either would make the controller aim
        # at a change it has no way to produce.
        errs_refinable:list[float]=[]
        errs_unrefinable:list[float]=[]
        nelem_local=0
        dim=2
        for _name,mesh in meshes:
            dim=max(dim,mesh.get_dimension())
            maxlev=mesh.max_refinement_level
            minlev=mesh.min_refinement_level
            for e in mesh.elements():
                # Halo copies are the same element as one owned elsewhere. Counting them would
                # inflate both the element count and every MPI-summed count in the bisection.
                if e.is_halo(): continue
                nelem_local+=1
                lev=e.refinement_level()
                err=e._elemental_error_max_override
                if lev<maxlev: errs_refinable.append(err)
                if lev>minlev: errs_unrefinable.append(err)
        nelem=int(get_mpi_sum(nelem_local))
        ndof=self.ndof()
        if nelem==0 or ndof==0:
            return

        target=float(self.desired_ndof)
        rel_gap=(target-ndof)/target
        sons=float(2**dim)

        # A step that asked for a change and got none means the request was not something the mesh
        # can act on -- overwhelmingly the unrefinement veto, since oomph merges a father only if all
        # of its sons agree and the smallest-error elements do not come in complete families. Asking
        # for the same amount again would stall forever, so escalate; a step that did move resets it.
        # Bounded, because past some point the answer really is "this mesh cannot go any coarser".
        #
        # Only steps that actually asked for something count. The dead band deliberately leaves ndof
        # where it is, and letting that escalate would mean a run that idles at the target arrives at
        # the next genuine adaptation with a 32x request behind it.
        last=getattr(self,"_desired_ndof_last_ndof",None)
        boost=getattr(self,"_desired_ndof_boost",1.0)
        if last is not None:
            boost=min(boost*2.0,32.0) if last==ndof else 1.0
        self._desired_ndof_boost=boost

        def set_thresholds(mx:float,mn:float):
            for _n,mesh in meshes:
                # oomph-lib throws outright if refine_tol <= unrefine_tol, and the two sentinel values
                # the override stages use are placed relative to this pair, so they have to stay a
                # sane, strictly ordered bracket even when the controller wants "nothing at all".
                mesh.max_permitted_error=mx
                mesh.min_permitted_error=min(mn,0.5*mx)

        if abs(rel_gap)<=self.desired_ndof_tolerance:
            # Inside the dead band: flag nothing, so this adaptation reports nref==nunref==0, which is
            # the signal every calling loop already breaks on. Equidistributing at constant ndof
            # (refining the worst elements while merging an equal dof-cost of the best) would be the
            # natural extension and is what a moving feature in a transient run wants; it is not done
            # here because it has no such termination signal.
            # A threshold above every error and an unrefine threshold below every error: nothing is
            # selected in either direction. Both are still a strictly ordered pair, which oomph-lib
            # insists on.
            set_thresholds(1e300,-1.0)
            # Nothing was requested, so there is nothing for the next call to conclude from ndof
            # having stayed put.
            self._desired_ndof_last_ndof=None
            self._desired_ndof_boost=1.0
            if not self.is_quiet():
                print("DESIRED NDOF: "+str(ndof)+" is within "+str(self.desired_ndof_tolerance*100)+"% of "+str(self.desired_ndof)+", not adapting")
            return

        # Nothing this controller can act on in the direction it wants to go: every element is
        # already at max_refinement_level (growing), or the mesh is at its coarsest and no element
        # has a father to be merged into (shrinking, which is what a target below the initial mesh
        # size asks for). Say so and leave the thresholds where they are - a target the mesh cannot
        # reach is a statement about the mesh, not an error.
        navail_grow=int(get_mpi_sum(len(errs_refinable)))
        navail_shrink=int(get_mpi_sum(len(errs_unrefinable)))
        if (rel_gap>0 and navail_grow==0) or (rel_gap<0 and navail_shrink==0):
            set_thresholds(1e300,-1.0)
            self._desired_ndof_last_ndof=None
            self._desired_ndof_boost=1.0
            if not self.is_quiet():
                print("DESIRED NDOF: "+str(self.desired_ndof)+" not reachable from ndof="+str(ndof)+
                      (": every element is already at max_refinement_level" if rel_gap>0
                       else ": the mesh is already at its coarsest"))
            return

        if rel_gap>0:
            # Grow. Refining one element replaces it by 2**dim, i.e. adds (2**dim - 1) elements, and
            # ndof follows the element count closely enough for a controller that re-measures every
            # step. Damped, because 2:1 balancing refines further elements this model knows nothing of.
            wanted=nelem*min(target/ndof,self.desired_ndof_max_growth)-nelem
            k=int(self.desired_ndof_damping*wanted/(sons-1.0))
            k=max(1,min(k,navail_grow))
            self._desired_ndof_last_ndof=ndof
            thresh=self._global_error_order_statistic(numpy.array(errs_refinable),k,largest=True)
            # Strictly below the k-th largest, since oomph refines on error > refine_tol.
            set_thresholds(thresh*(1.0-1e-9) if thresh>0 else 0.0,-1.0)
            if not self.is_quiet():
                print("DESIRED NDOF: "+str(ndof)+" -> "+str(self.desired_ndof)+", refining the "+str(k)+" worst of "+str(nelem)+" elements (max_permitted_error="+str(thresh)+")")
        else:
            # Shrink. Merging one father removes (2**dim - 1) elements but needs ALL 2**dim of its
            # sons flagged, and a single dissenting son vetoes the whole father, so the number of
            # elements below the threshold is only an upper bound on what actually merges. The
            # controller therefore undershoots in this direction and takes more steps than it does
            # growing; that is expected, not a bug to tune away.
            wanted=nelem-nelem*target/ndof
            # Not damped, unlike the growth direction: the 2:1-balancing overshoot that damping exists
            # to absorb is a refinement effect, and unrefinement already undershoots on its own.
            navail=navail_shrink
            m=int(boost*wanted*sons/(sons-1.0))
            if m>=navail and navail>0 and boost>1.0:
                # Everything that could merge is already flagged and it still is not moving. Say so
                # once rather than looping in silence: this is the mesh telling us the target is not
                # reachable by unrefinement alone (min_refinement_level, or sons that never agree).
                if not self.is_quiet():
                    print("DESIRED NDOF: cannot shrink below ndof="+str(ndof)+" (target "+str(self.desired_ndof)+"): every mergeable element is already flagged")
            m=max(1,min(m,navail))
            self._desired_ndof_last_ndof=ndof
            thresh=self._global_error_order_statistic(numpy.array(errs_unrefinable),m,largest=False)
            # Strictly above the m-th smallest, since oomph unrefines on error < unrefine_tol. The
            # refine threshold goes far above every error so that nothing refines while we are over
            # budget -- but the must_refine sentinel is 100x it, so mandatory refinements still fire.
            unref=thresh*(1.0+1e-9)
            set_thresholds(max(abs(unref),1e-300)*1e6,unref)
            if not self.is_quiet():
                print("DESIRED NDOF: "+str(ndof)+" -> "+str(self.desired_ndof)+", unrefining the "+str(m)+" best of "+str(nelem)+" elements (min_permitted_error="+str(thresh)+")")

    def _adapt(self) -> tuple[int, int]:
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
                    # The skeleton child is created in EquationTree._fill_dummy_equations, which runs long
                    # before this point, so it can only be missing if requires_interior_facet_terms was
                    # switched on afterwards. Auto-creating it here is too late: the code generators of
                    # this mesh (and its opposite-facet dummies) are already built and compiled, which is
                    # what the removed attempt below this raise never accounted for.
                    if "_internal_facets_" not in mesh._eqtree.get_children().keys():
                        raise RuntimeError("Interior facet residuals were added on domain '"+str(mesh.get_name())+"', but it has no '_internal_facets_' subdomain. Set self.requires_interior_facet_terms=True in the __init__ of the Equations class that adds them (before the problem is set up), or add the subdomain by hand with eqs+=Equations()@'_internal_facets_'.")
                    internal_eqs=mesh._eqtree.get_child("_internal_facets_").get_equations()
                    for destination,int_contrib in eqs._interior_facet_residuals.items():
                        if destination in internal_eqs._additional_residuals.keys():
                            internal_eqs._additional_residuals[destination]+=int_contrib
                        else:
                            internal_eqs._additional_residuals[destination]=int_contrib


        # Number the base elements before the first interface mesh is built. The interior-facet
        # skeleton picks one of the two elements sharing a facet as the "near" side, and does so by
        # comparing these numbers (Mesh::compare_structural_order), so an unnumbered mesh silently
        # falls back to local element order - which is the same order here, but leaves the rule
        # untested serially and only engaged after the distribution renumbered nothing.
        # Idempotent: the roots are fixed from here on (refinement subdivides them, it does not add
        # any), so the assignment before distribute() below reproduces exactly these numbers.
        for _,mesh in self._meshdict.items():
            if not isinstance(mesh,ODEStorageMesh):
                mesh.assign_global_base_element_indices()

        for tree_depth in range(3):
            for _,mesh in self._meshdict.items():
                # No mesh class stores a Python-level "_problem" attribute any more -
                # get_problem() resolves it live via the C++ side (see mesh.py). ODEStorageMesh
                # instances can be reused across a redefine_problem() cycle with a new owning
                # Problem, so still need re-stamping here - preserving the existing compiled
                # code instance (if any), since _set_problem() would otherwise reset it to None.
                if isinstance(mesh,ODEStorageMesh):
                    mesh._set_problem(self,mesh.get_code_gen()._code)
                assert mesh._eqtree is not None
                assert mesh._eqtree._equations is not None
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

    def set_custom_assembler(self,assm:"CustomAssemblyBase | None") -> None:
        if self._custom_assembler:
            self._custom_assembler.finalize()
            
        self._custom_assembler=assm
        if self._custom_assembler:        
            self.use_custom_residual_jacobian=True
            self._custom_assembler._set_problem(self)
            self._custom_assembler.initialize()
        else:
            self.use_custom_residual_jacobian=False

    def get_custom_assembler(self) -> "CustomAssemblyBase | None":
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
            paramname:str | None=info.get_parameter_name()
            if paramname=="":
                paramname=None
            res=self._custom_assembler.get_residuals_and_jacobian(False,paramname)
            assert res.dtype==numpy.float64 #type:ignore
            info.set_custom_residuals(res)

    @overload
    def set_c_compiler(self,compiler_or_name:Literal["tcc"])->_pyoomph.CCompiler: ...

    @overload
    def set_c_compiler(self,compiler_or_name:Literal["system"])->"SystemCCompiler": ...

    def set_c_compiler(self,compiler_or_name:str | BaseCCompiler)->_pyoomph.CCompiler | BaseCCompiler:
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
                compiler_or_name="tccbox"
            elif compiler_or_name=="distutils":
                compiler_or_name="system"
            cc=get_ccompiler(compiler_or_name)
        else:
            cc=compiler_or_name
        self._set_ccompiler(cc)
        return self.get_ccompiler()


    def __iadd__(self,other:"MeshTemplate | EquationTree | GenericProblemHooks | MatplotlibPlotter"):        
        if self._initialised:
            from ..output.plotting import BasePlotter
            if not isinstance(other,(BasePlotter,GenericProblemHooks)):
                raise RuntimeError("Cannot add anything to a problem once it is initialized!")
        if isinstance(other,MeshTemplate):
            self.add_mesh(other)
        elif isinstance(other,EquationTree):
            if other._equations is not None:
                raise RuntimeError("You try to add an EquationTree to the Problem with Equations defined on the root level. This is not allowed. Please restrict all Equations to a domain using e.g. @'domain'")
            
            self.additional_equations+=other        
        elif isinstance(other,GenericProblemHooks):
            if other._problem is None:
                other._problem=self
            elif other._problem is not self:
                raise RuntimeError("Cannot add a problem hook to a different problem")
            self._hooks.append(other)
            if self._initialised:
                other.actions_after_initialise()
        else:
            from ..output.plotting import BasePlotter
            if isinstance(other,BasePlotter):
                if self.plotter is None:
                    self.plotter=other
                elif isinstance(self.plotter,list):
                    self.plotter.append(other)
                else:
                    self.plotter=[self.plotter,other]
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
        # Mutually exclusive: argparse itself rejects any combination of two of these (e.g.
        # --superlu --pardiso) with a clear usage error, so the linear solver flags never need to
        # be cross-checked by hand in parse_cmd_line() below.
        linear_solver_group = self.cmdlineparser.add_mutually_exclusive_group()
        linear_solver_group.add_argument('--petsc',help="use PETSc solver",action='store_true')
        linear_solver_group.add_argument('--superlu',help="use serial SuperLu solver",action='store_true')
        linear_solver_group.add_argument('--umfpack', help="use UMFPACK solver", action='store_true')
        linear_solver_group.add_argument('--pardiso', help="use Pardiso solver", action='store_true')
        linear_solver_group.add_argument('--petsc_mumps',help="use PETSc as linear solver with MUMPS as backend",action="store_true")
        linear_solver_group.add_argument('--accelerate',help="use Apple Accelerate sparse solver (macOS only)",action='store_true')
        # Mutually exclusive for the same reason as linear_solver_group above.
        eigen_solver_group = self.cmdlineparser.add_mutually_exclusive_group()
        eigen_solver_group.add_argument('--slepc',help="use SLEPc as eigensolver. Specify your own backend for the matrix inversion during eigensolve here",action="store_true")
        eigen_solver_group.add_argument('--slepc_mumps',help="use SLEPc as eigensolver with MUMPS as backend",action="store_true")
        eigen_solver_group.add_argument('--arpack',action="store_true")
        # Mutually exclusive for the same reason as linear_solver_group above.
        ccompiler_group = self.cmdlineparser.add_mutually_exclusive_group()
        ccompiler_group.add_argument('--tcc', help="use internal TCC compiler", action='store_true')
        ccompiler_group.add_argument('--distutils', help="use system C compiler detected by distutils", action='store_true')
        # The hint is not a style preference: measured on a transcendental-heavy weak form, wrapping the
        # expensive terms in subexpression() beats -ffast-math outright (codegen then emits each exp/pow
        # once, and caches its derivatives, instead of leaving redundant libm calls for the compiler to
        # eliminate), and once it is used -ffast-math adds under a percent. On polynomial weak forms the
        # flag is within noise either way, since -O3 -march=native is already the default.
        self.cmdlineparser.add_argument('--fast-math', help="activate fast math compiler flags (only with distutils, not with tcc). Rarely pays off: consider wrapping expensive terms in subexpression() instead, which is faster and does not change the arithmetic", action='store_true')
        self.cmdlineparser.add_argument('--distribute',help="Distribute mesh in parallel",action='store_true')
        # Registered here mainly so that it shows up in --help and is consumed rather than handed on
        # to PETSc as an unrecognised option; it is read from sys.argv much earlier than this, in
        # pyoomph.generic.logging, because the MPI banner is printed at import time.
        self.cmdlineparser.add_argument('--mpi-output',help="Console output under mpirun: 'condensed' (default, rank 0 only), 'all' (every rank, tagged) or 'off' (unfiltered)",type=str,choices=["condensed","all","off"])
        self.cmdlineparser.add_argument('--outdir', help="output directory",type=str)
        self.cmdlineparser.add_argument('--suppress_code_writing',help="do not write FEM codes. Useful for debugging",action='store_true')
        self.cmdlineparser.add_argument('--suppress_compilation',help="do not compile FEM codes. Useful for debugging",action='store_true')
        self.cmdlineparser.add_argument('--no-cache',help="Do not use the JIT code cache (pyoomph.generic.jit_cache) - always regenerate/recompile FEM codes from scratch",action='store_true')
        # -P is pyoomph's only SHORT flag, and short flags are indistinguishable from PETSc options:
        # petsc4py is handed the command line, and PETSc reports every dash-prefixed token nothing read
        # as a possible spelling mistake. pyoomph/solvers/petsc.py therefore keeps -P out of what PETSc
        # is shown, by name -- a second short flag added here has to be added there as well.
        self.cmdlineparser.add_argument('-P','--parameter', help="Override some problem parameters",nargs='+', type=str)
        self.cmdlineparser.add_argument("--runmode",help="Selects the runmode ([d]elete and run, [o]verride and run, [c]ontinue, [p]lot again",type=str)
        self.cmdlineparser.add_argument("--recompile_on_continue",help="When using --runmode c, compilation and code writing is usually suppressed. You can recompile the code anyhow with this flag",action="store_true")
        self.cmdlineparser.add_argument("--verbose",help="Gives a lot of output",action='store_true')
        self.cmdlineparser.add_argument("--where",help="Python bool expression involving variables time or step. Only used in runmodes c and p",type=str,default="True")
        self.cmdlineparser.add_argument("--largest_residuals",help="Debug the largest residuals",type=int,default=self._debug_largest_residual)
        self.cmdlineparser.add_argument("--generate_precice_cfg",help="Generate some parts of a preCICE configuration file from the coupling equations",action="store_true")
        self.cmdlineparser.add_argument("--quick-test",help="Stops after the first successful Newton method. Useful for quick testing",action="store_true")

    def parse_cmd_line(self):
        from ..materials.generic import MaterialProperties
        if self.ignore_command_line:
            self.cmdlineargs, self.further_cmdlineargs = self.cmdlineparser.parse_known_args(args="")
        else:
            self.cmdlineargs, self.further_cmdlineargs = self.cmdlineparser.parse_known_args()
        if self.cmdlineargs.superlu:
            self.set_linear_solver("superlu")
        elif self.cmdlineargs.petsc:
            self.set_linear_solver("petsc")
        elif self.cmdlineargs.umfpack:
            self.set_linear_solver("umfpack")
        elif self.cmdlineargs.pardiso:
            self.set_linear_solver("pardiso")
        elif self.cmdlineargs.petsc_mumps:
            self.set_linear_solver("petsc_mumps")
        elif self.cmdlineargs.accelerate:
            self.set_linear_solver("accelerate")

        if self.cmdlineargs.tcc:
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
            self.set_eigensolver("scipy") # Not using the pardiso arpack then
        elif self.cmdlineargs.slepc_mumps:
            self.set_eigensolver("slepc_mumps")
        elif self.cmdlineargs.slepc:
            self.set_eigensolver("slepc")



        if self.cmdlineargs.outdir:
            self._outdir=self.cmdlineargs.outdir

        if self.cmdlineargs.mpi_output:
            # Re-applied through argparse so that an invalid value is caught here with a proper usage
            # error, and so that ignore_command_line=True gets the default back.
            from .logging import setup_mpi_console
            setup_mpi_console(get_mpi_rank(),get_mpi_nproc(),self.cmdlineargs.mpi_output)

        if self.cmdlineargs.suppress_code_writing:
            self._suppress_code_writing=True
            
        if self.cmdlineargs.suppress_compilation:
            self._suppress_compilation=True

        if self.cmdlineargs.no_cache:
            self._no_cache=True
            from .jit_cache import set_enabled
            set_enabled(False)

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
                    import difflib
                    closest_matches=difflib.get_close_matches(splt_varname[-1],dir(obj))
                    if len(closest_matches)>0:
                        raise RuntimeError("Cannot set undefined property/parameter "+".".join(splt_varname)+". Currently at "+str(obj)+" and trying to access "+str(splt_varname[-1])+"\nFollowing properties are known:"+str(dir(obj))+"\nClosest matches are: "+str(closest_matches))
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

    def before_assigning_equation_numbers(self,dof_selector:"_DofSelector | None"):
        for hook in self._hooks:
            hook.before_assigning_equation_numbers(dof_selector,True)
        self._equation_system._before_assigning_equations(dof_selector)         
        for hook in self._hooks:
            hook.before_assigning_equation_numbers(dof_selector,False)
        self.get_la_solver()._before_assigning_equation_numbers()
        # Only an eigensolver that has already been built: get_eigen_solver() would CONSTRUCT the
        # default one here, and that default is slepc_mumps wherever PETSc has MUMPS, so every run --
        # including one that solves with pardiso and never touches an eigenproblem -- imported
        # petsc4py/slepc4py, initialised PETSc and filled its options database, to call a hook that no
        # eigensolver overrides. One built later has nothing cached from before this reassignment.
        if isinstance(self._eigensolver,GenericEigenSolver):
            self._eigensolver._before_assigning_equation_numbers()


    def actions_before_remeshing(self,active_remeshers:list["RemesherBase"]):
        for hook in self._hooks:
            hook.actions_before_remeshing(active_remeshers)



    def actions_after_change_in_global_parameter(self,parameter_name:str):
        for hook in self._hooks:
            hook.actions_after_change_in_global_parameter(parameter_name)

    def actions_after_parameter_increase(self,parameter_name:str):
        for hook in self._hooks:
            hook.actions_after_parameter_increase(parameter_name)

    def actions_after_remeshing(self):
        self._equation_system._after_remeshing()
        self.reapply_boundary_conditions()
        # Remeshing destroys the superseded meshes, and a static condensation rule holds the mesh it
        # names, so the rules have to be restated against the new ones. reapply_boundary_conditions()
        # above has already done that for whatever a StaticCondensation equation declares; this covers
        # rules declared at problem level (condense_dofs), which nothing else would restate. A no-op
        # when the meshes are unchanged or nothing is declared at all.
        self._sync_static_condensation_rules()
        self.invalidate_cached_mesh_data()
        if self._custom_assembler:
            self._custom_assembler.actions_after_remeshing()
        for hook in self._hooks:
            hook.actions_after_remeshing()


    def _sync_diagonal_requirement_from_solver(self):
        """Ask the active linear solver whether it needs an explicit diagonal, and tell the assembly.

        Whether a stored diagonal is required is a property of the factorisation, not of the problem --
        PETSc's own LU rejects a matrix without one, MUMPS does not care -- so the answer has to come
        from the solver, and it has to be re-asked before each solve because PETSc options (and the
        solver itself) can change at any time. Ignored while the user has set
        ``force_jacobian_diagonal_entries`` explicitly; see that property.
        """
        if not self._force_jacobian_diagonal_entries_is_auto:
            return
        try:
            solver = self.get_la_solver()
        except Exception:
            return  # No solver yet (or it cannot be constructed); the assembly default stands
        try:
            required = bool(solver.requires_explicit_diagonal())
        except Exception as e:
            print("WARNING: " + type(solver).__name__ + ".requires_explicit_diagonal() raised (" + str(e)
                  + "); assuming the solver does not need an explicit Jacobian diagonal.")
            required = False
        self._set_solver_requires_explicit_diagonal(required)

    def actions_before_newton_solve(self):
        self._sync_diagonal_requirement_from_solver()
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

    def actions_after_transient_solve(self):
        """Dispatched once per **accepted** timestep, at the end of :py:meth:`solve` with a
        ``timestep``, after any spatial adaptation and after temporal adaptivity has settled on a
        step it accepts.

        This is where anything that must advance exactly once per step in time belongs.
        :py:meth:`actions_after_newton_solve` cannot serve that purpose: it also fires for
        stationary solves, for each arclength continuation step, for every discarded temporal
        retry, and once per adaptation level.
        """
        self._equation_system._after_transient_solve()
        for hook in self._hooks:
            hook.actions_after_transient_solve()

    def actions_after_newton_solve(self):
        if self.last_newton_step_failed():
            return # Don't do this if it has not converged
        self._equation_system._after_newton_solve()
        for ism in range(self.nsub_mesh()):
            submesh=self.mesh_pt(ism)
            if isinstance(submesh,MeshFromTemplateBase):
                submesh._solves_since_remesh+=1
        self._agree_on_domains_to_remesh()
        if len(self._domains_to_remesh)>0:
            if (self._solve_in_arclength_conti is None) and self.do_call_remeshing_when_necessary:
                self.force_remesh(self._domains_to_remesh)
        self.invalidate_cached_mesh_data()
        if self._custom_assembler:
            self._custom_assembler.actions_after_successful_newton_solve()
        for hook in self._hooks:
            hook.actions_after_newton_solve()
        if self._bifurcation_reactivation_after_adaptation is not None:
            self._reactivate_bifurcation_tracking_after_adaption()
        self._bifurcation_reactivation_after_adaptation=None
        
        if self.cmdlineargs.quick_test:
            print("QUICK TEST: STOPPING AFTER FIRST SUCCESSFUL NEWTON SOLVE, BUT DOING OUTPUT FIRST")
            self.output()
            print("QUICK TEST: STOPPING AFTER FIRST SUCCESSFUL NEWTON SOLVE")
            self.release()
            sys.exit(0)

    def _agree_on_domains_to_remesh(self):
        """Make the pending remeshing requests unanimous over the MPI processes.

        RemeshWhen (and the pinch-off/coalescence handler) judge the elements this rank happens to
        hold, so on a distributed mesh only the ranks owning the distorted part of the mesh ask for
        a remesh. force_remesh() is collective throughout - it rebuilds the mesh (gmsh, with its own
        collectives), interpolates and reassigns the equation numbers - so a request has to be
        answered by all ranks or by none, otherwise the ranks that abstain wait for the others
        forever. Any rank asking carries all of them, which is the safe side: a mesh nobody needed
        to rebuild is merely rebuilt.
        """
        if get_mpi_nproc()<=1:
            return
        # _meshtemplate_list is built by define_problem(), i.e. in the same order on every rank - but
        # it is not the whole story once a mesh has been remeshed. A Remesher2d hands the problem a
        # NEW template (GmshRemesher2d, see its get_new_template()), and RemeshWhen asks by the
        # template its mesh currently carries, so from the first such remesh onwards the request
        # names something this list does not contain. It then fell through to the "keep it" branch
        # below, which preserves the very asymmetry this method exists to remove: one rank remeshed,
        # the other went on, and the next collective paired a boundary merge with a file output.
        # _meshdict is built in the same order everywhere too, so adding the templates in use keeps
        # the list rank-independent.
        templates=list(self._meshtemplate_list)
        for _mn,_msh in self._meshdict.items():
            t=getattr(_msh,"_templatemesh",None)
            if t is not None and t not in templates:
                templates.append(t)
        if not any(t.remesher is not None for t in templates):
            return  # nothing can ever ask, so do not pay for a collective after every solve
        agreed=get_mpi_any_list([t in self._domains_to_remesh for t in templates])
        wanted={t for t,w in zip(templates,agreed) if w}
        # Anything still unknown here cannot be matched up across the ranks; keep it rather than
        # silently dropping the request, and say so, since it is the asymmetry described above.
        leftover=self._domains_to_remesh-set(templates)
        if leftover:
            print("WARNING: a remesh was requested for "+str(len(leftover))+" mesh template(s) this problem does "
                  "not know about, so the request cannot be made unanimous across the MPI ranks. If the ranks "
                  "disagree, the run will desynchronise.")
        self._domains_to_remesh=wanted | leftover

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
        # Collective, like the remeshing it may trigger: call it on all ranks, not from a
        # rank-dependent branch.
        self._agree_on_domains_to_remesh()
        if len(self._domains_to_remesh)>0:
            self.force_remesh(self._domains_to_remesh)
            res=True
        return res
    

    # Can be used for go_to_param or 
    def remesh_handler_during_continuation(self, force: bool = False, resolve: bool = True, resolve_before_eigen: bool = False, reactivate_biftrack_neigen: int = 4, reactivate_biftrack_shift:float=0,resolve_max_newton_steps : int | None=None,num_adapt:int | None=None,resolve_globally_convergent_newton:bool=False):
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
            assert self._azimuthal_mode_param_m is not None
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
            self.solve_eigenproblem(reactivate_biftrack_neigen,azimuthal_m=(int(m) if m is not None else None),normal_mode_k=k,shift=reactivate_biftrack_shift)
            self.activate_bifurcation_tracking(biftrack_param, cast(Literal["hopf", "fold", "pitchfork", "azimuthal", "cartesian_normal_mode"],biftrack))
            if resolve:
                self.solve(max_newton_iterations=resolve_max_newton_steps,globally_convergent_newton=resolve_globally_convergent_newton)
        elif resolve:
            self.solve(max_newton_iterations=resolve_max_newton_steps,globally_convergent_newton=resolve_globally_convergent_newton)

    def _domain_name_pattern_candidates(self,node:"EquationTree",depth:int)->tuple[set[str],str]:
        """The names a glob child of `node` may expand to, together with a phrase naming them for the
        error message. Only called for nodes that actually have a glob child, so the element-walking
        _find_interface_intersections() below is not paid for by problems that use no patterns."""
        if depth==0:
            res:set[str]=set()
            for m in self._meshtemplate_list:
                res.update(m.available_domains())
            return res,"Available domains"
        path=node.get_full_path().lstrip("/").split("/")
        templ=None
        for m in self._meshtemplate_list:
            if m.has_domain(path[0]):
                templ=m
                break
        if templ is None:
            raise RuntimeError("Cannot expand a domain name pattern at '"+node.get_full_path()+"': there is no mesh template with a domain named '"+path[0]+"'. ODE domains and domains without a mesh template have no boundaries to match against.")
        if depth==1:
            # Exactly the boundaries this domain touches with a whole facet, i.e. the same subset the
            # generated mesh will end up with.
            return set(templ.get_domain(path[0]).get_adjacent_boundary_names()),"Boundaries of '"+path[0]+"'"
        if depth==2:
            # _find_interface_intersections() inserts every permutation, so "domain/left/top" is present
            # iff "domain/top/left" is; filtering on the prefix therefore gives the codim-2 children.
            prefix=path[0]+"/"+path[1]+"/"
            return {p[len(prefix):] for p in templ._find_interface_intersections() if p.startswith(prefix) and "/" not in p[len(prefix):]},"Boundary intersections of '"+"/".join(path[:2])+"'"
        raise RuntimeError("Domain name patterns are only supported for bulk domains, their boundaries and the intersections of those boundaries. The pattern at '"+node.get_full_path()+"' is deeper than that, please list the names explicitly instead.")

    def _link_geometry_and_equations(self):
        #Go through the templates and create them
        domset:set[str]=set()
        for m in self._meshtemplate_list:
            m._do_define_geometry(self) 
            mydoms=set(m.available_domains())
            inters=domset.intersection(mydoms)
            if len(inters)>0:
                raise RuntimeError("Following domains are added multiple times: "+str(inters))
            domset.update(mydoms)


        if not hasattr(self,"_equation_system") or self._equation_system is None:
            raise RuntimeError("Please add at least one equation to the problem via add_equations()")

        # Resolve any glob domain name (e.g. DirichletBC(u=0)@"*") now: the geometry above is the first
        # point where real domain and boundary names exist, and everything below - the dummy equations,
        # the _internal_facets_ child, the opposite-interface nodes, the code generators - must only ever
        # see literal names.
        self._equation_system._expand_domain_name_patterns(self._domain_name_pattern_candidates)

        self._equation_system._fill_dummy_equations(self)
        self._interinter_connections.clear()
        for m in self._meshtemplate_list:
            m._ensure_opposite_eq_tree_nodes(self._equation_system)
            inters=m._find_interface_intersections()
            for im in inters:
                dom=im.split("/")[0]
                if dom in self._equation_system._children and self._equation_system._children[dom]._equations is not None:
                    self._interinter_connections.add(im)
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

    def before_defining_problem(self,redefine:bool=False,old_meshes:dict[str, AnyMesh] | None=None,old_mesh_templates:list[MeshTemplate] | None=None):
        pass

    def redefine_problem(self, code_dir:str,interpolator:type["BaseMeshToMeshInterpolator"] | None=None,num_adapt:int | None=None):
        """
        Redefines the problem by recompiling equations. 
        This can in principle be used if problem parameters have changed, but it is not recommended to change the problem structure.
        If possible, it is advised to use the global parameter system to change any parameters.

        Args:
            code_dir: Subdirectory in the output directory where the C++ code of the redefined problem will be written.
            interpolator: Mesh interpolator to map the fields of the old meshes to the new ones. If None, :py:attr:`mesh_interpolator` is used.
            num_adapt: Number of adaption steps after redefining the problem. If None, the number of adaption steps is determined by the max_refinement_level attribute.

        Raises:
            RuntimeError: If the problem contains no equations after the redefinition
        """
        if interpolator is None:
            interpolator=self.mesh_interpolator
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
        
        must_uniform_refine=False
        for mesh in self._meshdict.values():
            if isinstance(mesh,MeshFromTemplateBase) and len(mesh._initial_interface_refinement)>0:
                must_uniform_refine=True  
                break
        if must_uniform_refine:
            raise NotImplementedError("Not implemented yet: Uniform refinement of all meshes before redefining the problem...")

        self.relink_external_data()
        self._assemble_defined_field_list()

        self.setup_pinning()
        self.before_assigning_equation_numbers(self._dof_selector)
        self.reapply_boundary_conditions()

        if self.cmdlineargs.distribute:
            raise NotImplementedError("Not implemented yet: Redefining the problem with distribution...")
            self.distribute()

        self.init_output(redefined=True)
        self.rebuild_global_mesh_from_list(rebuild=True)
        self.reapply_boundary_conditions()


        num_adapt = self.max_refinement_level if num_adapt is None else num_adapt

        interpolators:dict[str,"BaseMeshToMeshInterpolator"]={}

        def perform_interpolation():
            for _, interp in interpolators.items():
                interp.interpolate()

        for name, newmesh in self._meshdict.items():
            if name in old_mesh_dict.keys():
                omesh=old_mesh_dict[name]
                if isinstance(newmesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d)) and isinstance(omesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d)):
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
        # Check the dotted-name length before hasattr(m,"__version__"): some
        # submodules (e.g. numpy.core, kept only as a deprecated compat shim
        # since numpy 2.0) raise a DeprecationWarning on any attribute access,
        # so they must be filtered out by name first rather than probed.
        modules= {m.__name__:m.__version__ for m in sorted(sys.modules.values(),key=lambda a : getattr(a,"__name__","")) if hasattr(m,"__name__") and len(m.__name__.split("."))==1 and hasattr(m,"__version__")}
        _pyoomph._write_to_log_file("Loaded module versions: "+str(modules)+os.linesep)
        self._log_start_time=datetime.datetime.now()
        _pyoomph._write_to_log_file("Log file started: "+str(self._log_start_time)+os.linesep)
        _pyoomph._write_to_log_file("####################"+os.linesep)
        _pyoomph._write_to_log_file("Args: "+str(sys.argv)+os.linesep)
        _pyoomph._write_to_log_file("####################"+os.linesep)
        _pyoomph._write_to_log_file(os.linesep)


    def _write_log_footer(self):
        # Idempotent: the footer must be written exactly once, whether we get here via
        # release() (with-block/explicit call) or via the atexit fallback registered in
        # initialise() for plain scripts that never call release() at all. Guard on
        # _log_start_time so nothing is written when no header was ever emitted.
        if self._log_start_time is None or self._log_footer_written:
            return
        self._log_footer_written=True
        import datetime
        end_time=datetime.datetime.now()
        _pyoomph._write_to_log_file(os.linesep)
        _pyoomph._write_to_log_file("####################"+os.linesep)
        _pyoomph._write_to_log_file("Log file ended: "+str(end_time)+os.linesep)
        _pyoomph._write_to_log_file("Elapsed time: "+str(end_time-self._log_start_time)+os.linesep)
        _pyoomph._write_to_log_file("####################"+os.linesep)


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
            if not self.only_write_logfile_on_proc0:
                raise RuntimeError("Cannot write log file on all processors yet")
            # ...which is exactly why every rank must not open it: they all opened the same file
            # and wrote into it independently, so an MPI log came out as interleaved half-lines.
            # The wrappers still go on all ranks - writing to the log is a no-op where none is open,
            # and they are what keeps the console output line-atomic.
            if get_mpi_rank()<=0:
                self._open_log_file(os.path.join(self._outdir,self.logfile_name),True)
            from . logging import pyoomph_activate_logging_to_file
            pyoomph_activate_logging_to_file()
            if get_mpi_rank()<=0:
                self._write_log_header()
            # Plain scripts (no "with" block, no explicit release()) never reach release()
            # before the interpreter shuts down - __del__ bails out at sys.is_finalizing() -
            # so the elapsed-time footer would be lost. Register an atexit fallback that writes
            # it while the C++ log-file handle is still open. Use a weakref so this registration
            # does not, by itself, keep the whole Problem alive until interpreter exit.
            import atexit, weakref
            _self_ref=weakref.ref(self)
            def _log_footer_atexit():
                _p=_self_ref()
                if _p is not None:
                    _p._write_log_footer()
            atexit.register(_log_footer_atexit)


        if self._runmode=="continue":
            # Find the highest dump
            dumpdir = os.path.join(self.get_output_directory(), "_states")
            dumps = sorted(glob.glob(os.path.join(dumpdir, "*.dump")))
            if len(dumps)==0 or keyfile is None or not os.path.isfile(keyfile):
                print("Cannot continue, starting over")
                self._runmode="overwrite"
        
        elif self._runmode=="delete":            
            mpi_barrier()
            if keyfile is not None and os.path.isfile(keyfile) and get_mpi_rank()<=0:
                if not self.is_quiet():
                    print("Removing contents of output dir",get_mpi_rank())

                    def rem_subdir(subdir:str,filter:str | list[str] | tuple[str],remglob:Iterable[str] | None=None):
                        top=os.path.join(self._outdir,subdir)
                        if not os.path.exists(top) or not os.path.isdir(top):
                            return
                        if not isinstance(filter,(list,tuple)):
                            filter=[filter]
                        lst:list[str]=[]
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
                    remglob:list[str] | None=None
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

        if get_mpi_rank()==0:
            infofile = open(os.path.join(self.get_output_directory(), "_numerical_factors.txt"), "w")
            infofile.write(self._equation_system.numerical_factors_to_string())
            infofile.close()

        if self.latex_printer is not None:
#            raise RuntimeError("LATEX PRINTER")
            self.latex_printer.write_to_file(os.path.join(self.get_output_directory(), "system_info.tex"))


        self.rebuild_global_mesh_from_list(rebuild=False)

        # Coupled domains must enter distribute() at a COMMON uniform level; whatever they do not share
        # is applied again after distribution (see _defer_uneven_initial_refinement).
        deferred_initial_refinement=self._defer_uneven_initial_refinement()

        must_uniform_refine=False
        for mesh in self._meshdict.values():
            if isinstance(mesh,MeshFromTemplateBase) and mesh._initial_uniform_refinement_level>0:
                must_uniform_refine=True  
                break
        if must_uniform_refine:
            self.actions_before_adapt()
            for mesh in self._meshdict.values():
                if isinstance(mesh,MeshFromTemplateBase) and mesh._initial_uniform_refinement_level>0:
                    for _ in range(mesh._initial_uniform_refinement_level):
                        mesh.refine_uniformly()
            self.relink_external_data()
            self.actions_after_adapt()

        self.relink_external_data()
        self._assemble_defined_field_list()
        
        jinfo_string, info_good=self._get_jacobian_information_string()
        if not info_good and self.stop_on_jacobian_structure_warning:
            raise RuntimeError("\n\nJacobian structure information indicates potential problems.\nSet stop_on_jacobian_structure_warning=False to ignore this warning.\n\n"+jinfo_string+"\n\n"+"This could be a result of missing equations, or misspelling a method in your Equation class, as e.g. 'define_residual' or 'define_equations' instead of 'define_residuals'.\nSet stop_on_jacobian_structure_warning=False in the Problem class if you are sure what you are doing")
        infofile=open(os.path.join(self.get_output_directory(),self._ccode_dir,"_jacobian_structure.txt"),"w")        
        infofile.write(str(jinfo_string))
        infofile.close()

        self._set_solved_residual("",False,True)
        # reapply_boundary_conditions() opens with exactly setup_pinning() +
        # before_assigning_equation_numbers(), so doing them here as well was one whole extra pinning
        # sweep over every element - and it was the *stale* one: reapply first flushes the additional
        # dof constraints, so the pinning state this used to compute was thrown away and rebuilt a few
        # lines later anyway.
        self.reapply_boundary_conditions()


        # Number the base elements while the meshes are still whole. State files address elements and
        # nodes relative to these numbers, which is what makes them independent of the partition - and
        # a serial run assigns exactly the same numbers here, so both write the same file.
        for _mn,_msh in self._meshdict.items():
            if not isinstance(_msh,ODEStorageMesh):
                _msh.assign_global_base_element_indices()

        if self.cmdlineargs.distribute:
            print("DISTRIBUTING THE PROBLEM")
            self.actions_before_distribute()
            self.distribute()
            self.actions_after_distribute()
            print("DISTRIBUTING DONE")

        # Distribution (if any) is done, so partial refinement is allowed again: give the coupled
        # domains the levels they do not share, and repair the conformity that breaks.
        self._apply_deferred_initial_refinement(deferred_initial_refinement)

        if self.check_mesh_integrity:
            for _,m in self._meshdict.items():
                assert m._codegen is not None                                
                if isinstance(m,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d)):
                    m.check_integrity()

        if self._runmode!="continue" and self._runmode!="replot":
            if self.initial_adaption_steps is None:
                self.initial_adaption_steps=self.max_refinement_level
            if  (self.initial_adaption_steps>0 and not self.project_initial_conditions):
                no_need_to_reassign=False
                for s in range(self.initial_adaption_steps):
                    self.map_nodes_on_macro_elements()
                    self.set_initial_condition()
                    self._initialised = True
                    self._during_initialization=False
                    if not self.is_quiet():
                        print("Initial adaption:",s,"of",self.initial_adaption_steps)
                    nref,nunref=self._adapt()
                    if get_mpi_nproc()>1:
                        # Make sure nref and nunref are all considered
                        nref_sum = get_mpi_sum(nref)
                        nunref_sum = get_mpi_sum(nunref)
                        if nref_sum == 0 and nunref_sum == 0:
                            no_need_to_reassign = True
                            break
                        pass
                    else:
                        if nref==0 and nunref==0:
                            no_need_to_reassign=True
                            break
                    if self.is_distributed() and self.call_load_balance_in_initial_adaption:                        
                        self.load_balance()
                if self.initial_adaption_steps>0 and not (no_need_to_reassign):
                    self.map_nodes_on_macro_elements()
                    self.set_initial_condition()
            else:
                self.map_nodes_on_macro_elements()                                
                if not self.project_initial_conditions:
                    self.set_initial_condition()
            if self.project_initial_conditions:
                self._initialised = True
                self._during_initialization=False
                self.set_initial_condition(numadapt=self.initial_adaption_steps)
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
            self._continue_dt_pending=True
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
        def recu_add_imeshes(sm:MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d | InterfaceMesh):
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

        # First odes
        for _, m in self._meshdict.items():
            if isinstance(m,ODEStorageMesh):
                assert m._codegen is not None
                m._codegen._mesh=m 
                self.add_sub_mesh(m)
        # Now bulks
        for _, m in self._meshdict.items():            
            if isinstance(m,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d)):
                assert m._codegen is not None
                m._codegen._mesh=m 
                self.add_sub_mesh(m)

        # And finally interfaces
        for _, m in self._meshdict.items():
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
        for mt in self._meshtemplate_list:
            mt._connect_opposite_elements(self._equation_system) 



    def setup_pinning(self):
        self.ensure_dummy_values_to_be_dummy()
        for ism in range(self.nsub_mesh()):
            submesh=self.mesh_pt(ism)
            if isinstance(submesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh,ODEStorageMesh)):
                #print("DIRCHLET SET ", submesh, submesh.get_full_name())
                submesh.setup_Dirichlet_conditions(False)
                if self.automatically_remove_dofs_without_equations:
                    submesh._pin_noncontributing_dofs()
                assert submesh._codegen is not None 
                submesh._codegen.on_apply_boundary_conditions(submesh) 
        if not self.apply_Dirichlet_BCs_by_dof_removing:
            self._unpin_Dirichlet_dofs_for_matrix_manipulation()




    def set_initial_condition(self, ic_name: str = "", all_unset_dofs_to_zero: bool = False,numadapt:int | None=0):
        """
        Set the initial condition for the problem.

        Args:
            ic_name (str, optional): Name of the initial condition. Multiple initial conditions can be defined in the problem definition by using the optional argument InitialConditions. Defaults to "".
            all_unset_dofs_to_zero (bool, optional): Flag indicating whether to set all unset degrees of freedom, i.e. without any InitialCondition with this ic_name, to zero. Defaults to False.
        """
        if numadapt is None:
            numadapt=0
        if all_unset_dofs_to_zero:
            self.set_current_dofs([0.0] * self.ndof())
        if not self.is_quiet():
            print("SETTING IC", ic_name)

        if self.project_initial_conditions:

            self._set_solved_residual("_IC_"+ic_name, True, True)
            self.reapply_boundary_conditions()
            print("Projecting initial condition",ic_name,numadapt)
            self.solve(spatial_adapt=numadapt)
            self._set_solved_residual("", False, True)
            self.reapply_boundary_conditions()
            
        if self._runmode != "continue":
            if not self._resetting_first_step:
                # Node construction leaves arbitrary content in the deeper position-history rows
                # (second history row, Newmark2 velocity/acceleration and the adaptive predictor
                # slots). The IC application below repairs only part of that - and nothing when dt
                # is still unset - so a freely floating solid with inertia (second-order partial_t
                # on the mesh coordinates) started with spurious momentum and translated at
                # constant velocity. Start impulsively from the current state; explicit initial
                # conditions below overwrite this where they are defined.
                self.assign_initial_values_impulsive()
            for _, m in self._meshdict.items():
                m.setup_initial_conditions_with_interfaces(self._resetting_first_step, ic_name)
                if isinstance(m, ODEStorageMesh):
                    continue
                # for n, sm in m._interfacemeshes.items():
                #     sm.setup_initial_conditions()
                #     TODO Recursive
                #     print(sm)
                # print("ICMESH", m)
        # The initial condition may well have overwritten a Dirichlet value, so the boundary
        # conditions have to be reapplied - but reapply_boundary_conditions() starts with
        # setup_pinning() itself, so calling it here first only doubled the sweep (see initialise()).
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


    def _global_mesh_nelement(self)->int:
        """The element count Problem::distribute() will partition, i.e. that of the global mesh.

        The global mesh itself has no Python handle (``mesh_pt()`` returns None for it), so count the
        submeshes that were added to it in :py:meth:`rebuild_global_mesh_from_list`: the bulk and ODE
        meshes plus every interface mesh.
        """
        n=0
        for m in self._meshdict.values():
            n+=m.nelement()
        for im in self._interfacemeshes:
            n+=im.nelement()
        return n

    def distribute(self):
        """Distribute the problem's meshes over the MPI processes.

        Only checks the prerequisites here - the distribution itself is done by oomph-lib. The check
        must happen on all ranks (see :py:func:`~pyoomph.generic.mpi.ensure_pymetis_available`).

        Leaves the problem undistributed (with a note) when there are fewer elements than processes,
        which is always the case for a pure ODE problem.
        """
        # On a single process oomph-lib ignores the request entirely and never calls METIS, so do
        # not make PyMetis a requirement for running the same script without mpirun.
        if get_mpi_nproc()>1:
            # oomph's Problem::distribute() throws when there are fewer elements than processes -
            # a pure ODE problem has a single element and hence always does. That is a request that
            # cannot be honoured, not an error: running the same problem replicated on every rank
            # gives the right answer (it is what mpirun without --distribute does), so do that
            # instead of aborting the job. Collective by construction: the meshes are still whole
            # here, so every rank counts the same elements and takes the same branch.
            #
            # The count is taken AFTER the initial uniform refinement in initialise() but before the
            # error-driven initial adaption, which is the only meaningful moment: the adaption leaves
            # the mesh non-uniformly refined, which Problem::distribute() refuses outright, so a mesh
            # that is too coarse now cannot be distributed later by growing.
            nelem=self._global_mesh_nelement()
            if nelem<get_mpi_nproc():
                if get_mpi_rank()==0:
                    msg=("NOTE: not distributing the problem: it has "+str(nelem)+" element(s) for "+
                         str(get_mpi_nproc())+" processes, and oomph-lib can only partition a mesh with "
                         "at least one element per process. The problem is solved replicated on every "
                         "process instead - which is correct, but of course pointless in parallel.")
                    if any(isinstance(m,MeshFromTemplateBase) for m in self._meshdict.values()):
                        msg+=(" Adaption cannot repair this, since only a uniformly refined mesh can be "
                              "distributed at all: to actually run in parallel, start from a finer mesh "
                              "or add RefineToLevel(n), which refines before the distribution.")
                    print(msg)
                return
            ensure_pymetis_available()
            # Unknowns on the interior-facet skeleton ARE supported here, in every discontinuous space
            # including the nodal ones (D1/D2/D1TB/D2TB): a facet shared by two ranks is owned by the
            # one that assembles it, and setup_interior_facet_halo_scheme() makes the other holder's
            # copy a halo, so it is numbered once and its numbers and values are copied across. That
            # scheme marks whole oomph::Data objects, so it does not care how many values each holds.
            #
            # A nodal facet space used to be refused right here, because distributing reaches the
            # skeleton through the adaptation path (actions_after_distribute -> actions_after_adapt),
            # which rebuilds every skeleton element from scratch and could only carry DL/D0 across.
            # The snapshot/refit carries every discontinuous space now (see
            # dev_docs/internal_facet_fields.md), so there is nothing left to refuse.
        super().distribute()


    def actions_before_distribute(self):
        for im in self._interfacemeshes:
            if im._opposite_interface_mesh is not None:
                opp=im._opposite_interface_mesh
                for e in opp.elements():
                    e.get_bulk_element().set_must_be_kept_as_halo(True)
        
        # Halo all ODEs
        for m in self._meshdict.values():            
            if isinstance(m,ODEStorageMesh):                                
                m.get_element().set_must_be_kept_as_halo(True)                
            
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
            # One call instead of three nanobind crossings per element (see Mesh::map_nodes_on_macro_elements)
            m.map_nodes_on_macro_elements()
        self._equation_system._after_mapping_on_macro_elements()


    def remove_macro_elements(self,mode:bool | Literal["auto"]="auto"):        
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


    def solve_auxiliary_residual(self, residual_name: str, **solve_kwargs) -> None:
        """
        Solve an auxiliary residual on its own, leaving the main solution untouched.

        Equations may be added to a named residual instead of the default one, by passing
        ``destination`` to :py:meth:`~pyoomph.generic.codegen.BaseEquations.add_residual`. Anything
        that is only ever postprocessed - a projection of some expression onto a field for output,
        say - belongs there rather than in the residual the Newton solver works on: on the main
        residual those unknowns have no Jacobian row at all, so they are pinned automatically and
        cost nothing, and this method solves for them when they are actually wanted. While it runs,
        the roles are reversed and the unknowns of the main residual are the pinned ones.

        Switching the residual is not sufficient by itself. :py:meth:`_set_solved_residual` only
        *marks* which fields have an empty Jacobian row; the pinning takes effect when the equation
        numbers are reassigned, which is what :py:meth:`reapply_boundary_conditions` does. Both are
        therefore done here, on the way in and on the way back out - the latter even if the solve
        raises, since leaving the problem on the auxiliary residual would silently corrupt every
        subsequent solve.

        Args:
            residual_name: name of the residual to solve, i.e. the ``destination`` the equations
                were added with.
            **solve_kwargs: passed on to :py:meth:`solve`.
        """
        previous = self._get_solved_residual()
        self._set_solved_residual(residual_name, True, True)
        self.reapply_boundary_conditions()
        try:
            self.solve(**solve_kwargs)
        finally:
            self._set_solved_residual(previous, False, True)
            self.reapply_boundary_conditions()

    def reapply_boundary_conditions(self):
        # Additional dof constraints (ConstrainFieldsToC1Space / ConstrainPositionsToC1Space) may live
        # on interface fields, whose elements sit in the (nested) interface meshes rather than the
        # top-level bulk meshes. So clear/apply must recurse into _interfacemeshes as well - otherwise
        # an interface element's setup_additional_dof_constraints (which pins the constrained interface
        # dof) is never called and the constraint silently does nothing.
        def _clear(m):
            if isinstance(m,ODEStorageMesh): return
            m.clear_additional_dof_constraints()
            for im in m._interfacemeshes.values():
                _clear(im)
        def _apply(m):
            if isinstance(m,ODEStorageMesh): return
            m.apply_additional_dof_constraints()
            for im in m._interfacemeshes.values():
                _apply(im)
        for m in self._meshdict.values():
            _clear(m)
        self.setup_pinning()
        self.before_assigning_equation_numbers(self._dof_selector)
        for m in self._meshdict.values():
            _apply(m)
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
        dofnames:list[str] = []

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


    def analyse_jacobian_singularity(self,k:int=2,ntop:int=6,quiet:bool=False,**kwargs:Any):
        """
        Find out which degrees of freedom and which equations make the Jacobian singular, which is
        the usual reason why Newton's method stalls after a boundary condition has been applied
        twice, e.g. an :py:class:`~pyoomph.equations.generic.EnforcedDirichlet` on a contact line where a
        kinematic boundary condition is already acting.

        It prints the dofs left undetermined, the equations that conflict, and everything sitting on
        the node that carries the singular mode. The problem must be initialised, but it does not
        have to be solved first: an over-constraint is present in the Jacobian of the initial
        condition already.

        The cost is one sparse LU plus a few triangular solves, so a good deal less than a nullspace
        computation. See :py:func:`~pyoomph.utils.jacobian_analysis.analyse_jacobian_singularity` for
        the remaining arguments.

        Args:
            k: Number of singular modes to report.
            ntop: Maximum number of dofs listed per singular vector.
            quiet: Only return the result instead of printing a report.

        Returns:
            The modes and the verdict, see :py:class:`~pyoomph.utils.jacobian_analysis.JacobianSingularityInfo`.
        """
        from ..utils.jacobian_analysis import analyse_jacobian_singularity
        return analyse_jacobian_singularity(self,k=k,ntop=ntop,quiet=quiet,**kwargs)


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
                        onbounds:list[str] = []
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
        highdofs:list[int] = list(reversed(highdofsI[-1 - nres+1:])) #type:ignore
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
                    submesh=self.mesh_pt(ism)
                    assert isinstance(submesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh,ODEStorageMesh))
                    location, typ = self.search_dof_in_mesh(submesh, dofindex)
                    if location is not None and typ is not None:
                        print("     but found in mesh "+submesh.get_full_name()+" at " + str(location) + ". Type: " + typ)
                    else:
                        print(" 		not found in mesh "+submesh.get_full_name())
                    #print("DESCRIBE",self.describe_equation(dofindex))
                    for e in submesh.elements():
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
            # Ask for the solve to be abandoned. This used to be done by multiplying the whole dof
            # vector by 1e40 and adding noise, so that the next residual evaluation would exceed
            # max_residuals and make oomph-lib throw. That destroyed the state -- the rejected
            # configuration, which is usually exactly what one wants to look at, was gone -- and it
            # went through get_current_dofs()/set_current_dofs(), which redistribute and are therefore
            # collective, while this decision is typically only reached on the ranks that hold the
            # offending part of the mesh. A rejection seen by some ranks and not others deadlocked.
            self._request_newton_abort("a step was rejected by before_newton_convergence_check")

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

    def _collect_coupled_interfaces(self) -> list[tuple[AnySpatialMesh,str,AnySpatialMesh,str,list[float]]]:
        """Every declared opposite-interface connection, expressed on the BULK meshes.

        The connections are declared as equation-tree paths ("domainA/interface"), but the machinery
        that keeps the two sides equally refined must not depend on interface meshes: those are torn
        down by clear_before_adapt() and only rebuilt afterwards. A bulk mesh plus a boundary name is
        permanent, and is exactly what the C++ side reads the boundary facets from.
        """
        conns:list[tuple[AnySpatialMesh,str,AnySpatialMesh,str,list[float]]]=[]
        seen:set[tuple[str,str,str,str]]=set()
        for templ in self._meshtemplate_list:
            for c in templ._opposite_interface_connections:
                a=c._sideA.split("/")
                b=c._sideB.split("/")
                # Only a bulk domain and one of its boundaries. A deeper path is an interface OF an
                # interface, whose conformity follows from that of the bulk facets it sits on.
                if len(a)!=2 or len(b)!=2:
                    continue
                # ORIENT THE PAIR CANONICALLY, by name. The order in which a connection is discovered is
                # NOT the same on every process -- it comes out of the C++ auto-detection, which walks
                # pointer-keyed containers, and heap addresses differ between processes. Two ranks then
                # disagree about which domain is "side A", and since the facet sets are unioned across
                # ranks per side, the union ends up mixing facets of BOTH domains into one side. The
                # result is a globally consistent-looking but entirely fictitious interface. Sorting by
                # (domain name, boundary name) is rank-independent by construction.
                #
                # This was found the hard way: two coupled triangle meshes in the same process, where the
                # allocation pattern happened to differ between ranks.
                if (b[0],b[1])<(a[0],a[1]):
                    a,b=b,a
                key=(a[0],a[1],b[0],b[1])
                if key in seen:
                    continue
                seen.add(key)
                ma=self._meshdict.get(a[0])
                mb=self._meshdict.get(b[0])
                if ma is None or mb is None:
                    continue
                if isinstance(ma,ODEStorageMesh) or isinstance(mb,ODEStorageMesh):
                    continue
                assert not isinstance(ma,InterfaceMesh) and not isinstance(mb,InterfaceMesh)
                if not ma.refinement_possible() and not mb.refinement_possible():
                    continue
                if a[1] not in ma.get_boundary_names() or b[1] not in mb.get_boundary_names():
                    continue # a side may legitimately be absent (dummy equations only)
                conns.append((ma,a[1],mb,b[1],[0.0,0.0,0.0]))
        # Same reasoning for the ORDER of the connections themselves: the C++ side refines meshes in the
        # order they appear here, and refine_selected_elements is collective on a distributed mesh, so
        # ranks that visited them in different orders would deadlock.
        conns.sort(key=lambda c:(c[0].get_name(),c[1],c[2].get_name(),c[3]))
        return conns

    def _defer_uneven_initial_refinement(self) -> list[tuple["BulkTemplateMesh",int]]:
        """Hold back the part of the initial uniform refinement that coupled domains do not share.

        Domains that start at different uniform levels are non-conforming from the outset. Repairing
        that the usual way -- refining the coarser one where the interface needs it -- is not available
        here: oomph's Problem::distribute() refuses to run on a mesh that is "no longer uniformly
        refined", since it has to preserve the tree forest, and a partial refinement is by definition
        non-uniform. Doing the repair before distribution would therefore break distribution instead.

        Levelling everyone UP to the maximum would keep the meshes uniform, but it over-refines: a domain
        asked for level 1 would silently get level 2, and RefineToLevel(1) then marks those level-2
        elements "may not unrefine", so the excess never goes away again.

        So: uniform-refine every coupled domain only as far as they agree, distribute, and apply the
        remainder afterwards -- where partial refinement is allowed and enforce_interface_conformity()
        can do its job. This lowers the stored levels to the common minimum and returns what it took, for
        the caller to re-apply after the distribute step.

        Coupling is transitive (A-B and B-C means all three must agree), so the minimum propagates to a
        fixed point over the coupling graph rather than in a single pass.
        """
        conns=self._collect_coupled_interfaces()
        if not conns:
            return []
        original:dict[BulkTemplateMesh,int]={}
        for ma,_bna,mb,_bnb,_off in conns:
            for m in (ma,mb):
                if isinstance(m,MeshFromTemplateBase) and m not in original:
                    original[m]=m._initial_uniform_refinement_level
        for _ in range(len(conns)+1):
            changed=False
            for ma,_bna,mb,_bnb,_off in conns:
                if not isinstance(ma,MeshFromTemplateBase) or not isinstance(mb,MeshFromTemplateBase):
                    continue
                lvl=min(ma._initial_uniform_refinement_level,mb._initial_uniform_refinement_level)
                for m in (ma,mb):
                    if m._initial_uniform_refinement_level>lvl:
                        m._initial_uniform_refinement_level=lvl
                        changed=True
            if not changed:
                break
        return [(m,original[m]-m._initial_uniform_refinement_level) for m in original
                if original[m]>m._initial_uniform_refinement_level]

    def _apply_deferred_initial_refinement(self,deferred:list[tuple["BulkTemplateMesh",int]]):
        """Apply what _defer_uneven_initial_refinement held back, once distribution is done."""
        if not deferred:
            return
        self.actions_before_adapt()
        for mesh,extra in deferred:
            for _ in range(extra):
                mesh.refine_uniformly()
        self.relink_external_data()
        self.actions_after_adapt() # repairs the conformity this just broke

    def enforce_interface_conformity(self) -> int:
        """Make both sides of every coupled interface carry identical boundary facets.

        oomph-lib adapts meshes one at a time, so two domains sharing an interface can refine
        different facets of it -- and then the opposite-element matcher (which pairs interface
        elements by exact vertex-position sets) has nothing to pair up and throws. This refines the
        coarser side wherever they disagree, interleaved with each mesh's own 2:1 balancing, until
        they agree. Refinement only, so it terminates.

        The same fixed point also closes the VERTEX-connected balance: an element touching a coupled
        interface at a single vertex carries no facet, so none of the above can see it, and it would
        otherwise be left arbitrarily coarser than the interface beside it. Those refinements are
        accumulated into :py:attr:`_interface_vertex_balance_refinements` instead, since they are a
        grading closure inside one mesh rather than a conformity repair.

        Collective under MPI: every process must call it. Returns the number of elements refined to
        repair facet conformity.
        """
        # Escape hatch for negative testing: a test that still passes with the enforcement switched
        # off is not measuring the enforcement. Not meant for production use.
        if os.environ.get("PYOOMPH_DISABLE_INTERFACE_CONFORMITY","")not in ("","0","off"):
            return 0
        conns=self._collect_coupled_interfaces()
        if not conns:
            return 0
        n,nvert=_pyoomph._enforce_interface_conformity(conns,40) #type:ignore
        self._interface_vertex_balance_refinements+=nvert
        # How much repairing this had to do. Zero is the good case and the interesting one: it means the
        # two sides agreed BEFORE they acted (the flag reconciliation in _adapt_with_interfacial_errors
        # got there first), rather than one of them being refined back afterwards. A repair is correct
        # but lossy -- re-refining an element that has just been merged away re-interpolates its sons
        # from the merged father, and the fine-scale information is gone. See
        # test_adapt_selection_is_reconciled_before_acting.
        self._interface_conformity_repairs+=n
        return n

    def check_interface_conformity(self,throw_on_mismatch:bool=False,when:str="") -> int:
        """Count boundary facets of a coupled interface that have no counterpart on the opposite side.

        Zero means the two sides are conforming and the opposite-element matcher will succeed. A
        non-zero result names the offending facets by position and says, for each, which side is the
        coarser one. Collective under MPI: every process must call it.
        """
        conns=self._collect_coupled_interfaces()
        if not conns:
            return 0
        return _pyoomph._check_interface_conformity(conns,when,2 if throw_on_mismatch else 1) #type:ignore

    def actions_after_adapt(self):
        # Announce that every cached element pointer is now stale. Refinement replaces a leaf element
        # by its sons and unrefinement DELETES them, so anything holding an element pointer across an
        # adaptation - tracer particles, point locators - is pointing into freed memory.
        #
        # This has to be done here rather than in pyoomph::Problem::actions_after_adapt, which also
        # does it: this override shadows the C++ one completely (it does not call super), so the C++
        # body never runs for a Python-defined Problem, which is every Problem.
        for _m in self._meshdict.values():
            _m.bump_topology_generation()
        for _m in self._interfacemeshes:
            _m.bump_topology_generation()

        # BEFORE the interface meshes are rebuilt below, and before _connect_opposite_elements pairs
        # them up: the two sides of a coupled interface may have been refined differently (the meshes
        # are adapted individually), and this is the one point where that can still be repaired.
        self.enforce_interface_conformity()
        # Same opt-in diagnostic knob as the halo consistency check, deliberately: both answer the same
        # question ("do the pieces that must agree still agree?") and a user should not have to know
        # which one to switch on. Checked AFTER the repair, so what it reports is a mismatch that
        # SURVIVED it -- which the repair cannot fix and the matcher below is about to die on.
        _confmode=_interface_conformity_check_mode()
        if _confmode:
            self.check_interface_conformity(throw_on_mismatch=_confmode>1,when="after enforcement")
        for m in self._interfacemeshes:
            #print("REBUILDING INTERFACE MESH",m,m.get_name(), m.nelement())
            m.rebuild_after_adapt()
            m.ensure_external_data()
            
        if not self.is_quiet():
            print("REBUILDING GLOBAL MESH")
        self.rebuild_global_mesh()
        for m in self._meshtemplate_list:
            m._connect_opposite_elements(self._equation_system)
        # After the whole loop above, not inside it: this is COLLECTIVE, and rebuild_after_adapt() can
        # throw per-rank (a non-conforming 3d skeleton, say), which would leave the ranks that did not
        # throw waiting in it for ever. Before setup_pinning(), because it decides which facet unknowns
        # exist at all - the ones on a halo facet must be numbered by their owner instead.
        self.setup_interior_facet_halo_scheme()
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
    
    def _reactivate_bifurcation_tracking_after_adaption(self):
        if self._bifurcation_reactivation_after_adaptation is not None:
            info=self._bifurcation_reactivation_after_adaptation
            self.deactivate_bifurcation_tracking()            
            print("Reactivating bifurcation tracking after adaption with info",info,"eigenvalue",self._last_eigenvalues)
            self.activate_bifurcation_tracking(info["param"],info["mode"],azimuthal_mode=info["azimuthal_m"],cartesian_wavenumber_k=info["cartesian_k"],eigenvector_scaling=info.get("eigenvector_scaling","unit"))
            self._bifurcation_reactivation_after_adaptation=None
            #self.reset_arc_length_parameters() # There is not much you can do here
            
            


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

        # Tier-2 JIT cache shadow mode: capture the cheap pre-codegen fingerprint
        # BEFORE write_code() runs (triggered inside generate_and_compile_bulk_element_code
        # below), so it can be compared afterwards against what write_code() actually
        # produced. See pyoomph/generic/jit_cache.py; this never skips codegen.
        # Ignored entirely whenever code writing or compilation is suppressed (debugging
        # runs, --runmode continue/replot without --recompile_on_continue, MPI rank>0):
        # there either is no .c file to fingerprint/compare against, or the whole point of
        # that mode is to leave everything alone rather than touch generated artifacts.
        fingerprint_text:str | None=None
        if not suppress_writing and not suppress_compilation and tier2_shadow_enabled():
            try:
                # generate_and_compile_bulk_element_code() below sets these two as a side effect,
                # immediately before calling write_code() - mirror that assignment here first, so
                # the fingerprint reflects the same up-to-date values write_code() will actually
                # see, not whatever they held from this element's previous compile (if any). Found
                # missing while investigating a Tier-2 shadow-mode mismatch on branch jit_cache
                # between two bifurcation-tracking scripts that differed only in whether Hessian-
                # vector-product code got emitted.
                elementtype.generate_hessian=self.are_hessian_products_calculated_analytically()
                elementtype.assemble_hessian_by_symmetry=self.get_symmetric_hessian_assembly()
                fingerprint_text=elementtype.get_precodegen_fingerprint_text()
            except Exception:
                fingerprint_text=None

        mpi_barrier()
        res=self.generate_and_compile_bulk_element_code(elementtype,trunk,suppress_writing,suppress_compilation,bulkmesh,self.is_quiet(),self.extra_compiler_flags)
        #print("REt")

        if fingerprint_text is not None:
            try:
                with open(trunk+".c","r") as f:
                    actual_code_text=f.read()
                cache=get_jit_cache()
                if cache is not None:
                    cache.check_fingerprint_shadow(fingerprint_text,actual_code_text,subname)
            except OSError:
                pass

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
    

    def define_global_parameter(self, **params: float) -> _pyoomph.GiNaC_GlobalParam:
        r"""
        Define a single global parameter for the problem.

        Args:
            **params: Exactly one keyword argument giving the parameter name and its initial value.

        Returns:
            _pyoomph.GiNaC_GlobalParam: The defined global parameter, which can be used in expressions.

        .. deprecated::
            Passing more than one keyword argument at once (to define several parameters in a
            single call, returning a tuple) still works, but is deprecated and emits a
            DeprecationWarning - call this once per parameter instead. Python's type system
            cannot express a return type that depends on the runtime length of \*\*params, so the
            declared return type here only reflects the single-parameter form; the multi-parameter
            form's actual tuple return is not statically visible.
        """
        if len(params) > 1:
            warnings.warn(
                "define_global_parameter(...) called with more than one keyword argument at once - "
                "this legacy form (returning a tuple of parameters) is deprecated. Call "
                "define_global_parameter once per parameter instead, e.g. "
                "p1=self.define_global_parameter(p1=...) and p2=self.define_global_parameter(p2=...).",
                DeprecationWarning,
                stacklevel=2,
            )
        res = []
        for p, v in params.items():
            res.append(self.get_global_parameter(p))
            res[-1].value = v
        if len(res) == 1:
            return res[0]
        else:
            return tuple([*res]) #type:ignore
        
    def setup_for_stability_analysis(self,analytic_hessian:bool=True,use_hessian_symmetry:bool=True,improve_pitchfork_on_unstructured_mesh:bool=False,improve_pitchfork_coordsys:"OptionalCoordinateSystem"=None,improve_pitchfork_position_coordsys:"OptionalCoordinateSystem"=None,shared_shapes_for_multi_assemble:bool | None=None,azimuthal_stability:bool | None=None,additional_cartesian_mode:bool | None=None,expand_element_size:bool=False):
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
            expand_element_size (bool, optional): Whether the mode expansion also perturbs the element
                size, i.e. whether a stabilization parameter tau built from it follows the mesh.
                Defaults to False, the "frozen tau" reading: the element size keeps its base-state
                value in the perturbation equations. That is the meaningful choice - the element size
                is a Cartesian mesh metric, and at m!=0 the perturbed configuration is not even a
                revolved 2d element, so its azimuthal extent is not represented at all - and it is
                also the one whose augmented Jacobian is exact. Only matters on a moving mesh, and
                only for equations that use the element size at all (the residual-based
                stabilizations do); on the rising bubble it moves the tracked onset by 0.36%.
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
        _pyoomph.set_expand_element_size_in_expansion_modes(expand_element_size)
        
    def is_normal_mode_stability_set_up(self)->Literal["azimuthal", "cartesian"] | Literal[False]:
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
    def get_current_time(self,dimensional:bool,as_float:Literal[True])->float: ...

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
        t:ExpressionOrNum = self.time_pt().time()
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

    def process_eigenvectors(self,eigenvectors:NPComplexArray)->NPComplexArray:
        """Here, you can optionally scale or negate the eigenvectors, e.g. normalize them, by overriding this method

        Args:
            eigenvectors (NPComplexArray): 2d array of eigenvectors

        Returns:
            NPComplexArray: Processed array of eigenvectors
        """
        #
        return eigenvectors

    def solve_eigenproblem(self, n:int, shift:float | complex | None=0, quiet:bool=False, azimuthal_m:int | list[int] | None=None,normal_mode_k:ExpressionOrNum | list[ExpressionOrNum] | None=None,normal_mode_L:ExpressionOrNum | list[ExpressionOrNum] | None=None,report_accuracy:bool=False,sort:bool=True,which:"EigenSolverWhich"="LM",OPpart:Literal["r", "i"] | None=None,v0:NPFloatArray | NPComplexArray | None=None,filter:Callable[[complex], bool] | None=None,target:complex | None=None,ncv:int | None=None)->tuple[NPComplexArray,NPComplexArray]:
        """
        Solves the associated generalized eigenproblem for the given number of eigenvalues and eigenvectors.

        With ``J`` the Jacobian and ``M`` the mass matrix of the residual ``R(U, dU/dt) = 0``, i.e.
        ``J = dR/dU`` and ``M = dR/d(dU/dt)``, the eigenvalues returned are the solutions of

            lambda * M * v + J * v = 0

        so a perturbation of the stationary solution evolves like ``v*exp(lambda*t)``: an eigenvalue
        with a positive real part means the solution is unstable (cf. :py:meth:`is_stable_solution`),
        and the imaginary part is the angular frequency of the oscillation. The same convention holds
        for the normal-mode variants selected by ``azimuthal_m`` or ``normal_mode_k``, and for the
        eigenvalue reported while a bifurcation or eigenbranch tracker is active.

        **While a bifurcation tracker is active** this solves the eigenproblem of the BASE state, i.e.
        of the first ``ndof`` entries of the augmented dof vector, which is how a secondary (codim-2)
        bifurcation is detected along a bifurcation locus. Nothing is renumbered and the tracked state
        is left untouched. Three things differ from an ordinary call:

        * ``shift`` must be non-zero. The tracker has converged the base state onto the bifurcation,
          so ``lambda = 0`` (fold, pitchfork) or ``+-i*omega`` (Hopf, azimuthal) is an exact
          eigenvalue -- exactly where the shift-invert transform would be asked to factorise. A zero
          (or ``None``) shift raises rather than handing the solver a singular operator.
        * without ``azimuthal_m``/``normal_mode_k`` the mode is whatever the corresponding global
          parameter currently holds, i.e. **the tracked mode** during azimuthal or normal-mode
          tracking. Pass ``azimuthal_m=0`` for the axisymmetric base-state spectrum.
        * a mode that would require the equations to be renumbered is refused. In practice that means
          ``azimuthal_m != 0`` while tracking a fold, Hopf or pitchfork, where the strong axis
          conditions are still active; while tracking an azimuthal or Cartesian normal-mode
          bifurcation they are already released and every mode is available.

        The results replace :py:meth:`get_last_eigenvalues`/:py:meth:`get_last_eigenvectors`, so they
        no longer hold the tracked critical eigenpair -- ask the handler through
        ``_get_bifurcation_eigenvector()`` if you need that afterwards.

        Args:
            n (int): The number of eigenvalues and eigenvectors to compute.
            shift (Union[float, complex, None], optional): The shift applied for shift-inverted approaches to solve the eigenproblem. Defaults to 0. Must be non-zero while bifurcation tracking is active, see above.
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
            ncv (Optional[int], optional): The number of Krylov (Arnoldi/Lanczos) basis vectors used by the underlying eigensolver. Defaults to None, i.e. the eigensolver's own default. Set this to a larger value than the default if the eigensolver fails to converge or returns inaccurate eigenvalues, e.g. for clustered eigenvalues.

        Returns:
            Tuple[NPComplexArray, NPComplexArray]: A tuple containing the computed eigenvalues and eigenvectors.
        """
        self._solve_eigenproblem_helper(n,shift,quiet,azimuthal_m,normal_mode_k,normal_mode_L,report_accuracy,sort,which,OPpart,v0,filter,target,ncv)
        self._last_eigenvectors=self.process_eigenvectors(self._last_eigenvectors)
        return self._last_eigenvalues,self._last_eigenvectors

    def _check_eigensolve_during_augmentation(self,shift:float | complex | None)->bool:
        """Decide whether an eigensolve may run with the current augmented system, and say so.

        Returns True when a bifurcation tracker is installed (so the caller knows not to renumber),
        False when the dof vector is the plain one. Raises for the augmentations that cannot serve a
        base-state eigenproblem, and for a shift that would factorise a matrix the tracker has just
        made singular.

        What makes tracking workable at all: oomph's get_eigenproblem_matrices() installs its own
        EigenProblemHandler, whose ndof/eqn_number delegate to the element, so the assembly is already
        the BASE one and the base state still sits in the node data. Only the row layout had to be put
        back, which Problem::BaseDofDistributionScope does in C++.
        """
        # Periodic orbits: the augmented dofs are nT copies of the base ones held in the handler, and
        # unlike the trackers there is no base distribution kept alive to fall back to. Note this used
        # to fall through the old guard entirely (get_bifurcation_tracking_mode() is "" for orbits) and
        # would assemble on the nT*Ndof+1 distribution.
        if isinstance(self.assembly_handler_pt(),_pyoomph.PeriodicOrbitHandler):
            raise RuntimeError("Cannot solve an eigenproblem while periodic orbit tracking is active. Use the Floquet multipliers of the orbit instead, or deactivate the orbit handler first.")
        # The Python-side custom augmentation (Problem.add_augmented_dofs / bifurcation_tools.py):
        # its dofs are appended to Dof_pt directly and the C++ scope knows nothing about them.
        if self._get_n_unaugmented_dofs()!=0: #type:ignore
            raise RuntimeError("Cannot solve an eigenproblem while a custom augmented system (add_augmented_dofs / a CustomBifurcationTracker) is active. Remove the augmentation first.")

        if self.get_bifurcation_tracking_mode()=="":
            return False

        # A zero shift is the default of solve_eigenproblem(), and it is exactly the one value that
        # cannot work here: the tracker has converged the base state onto a bifurcation, so lambda=0
        # (fold/pitchfork) or lambda=+-i*omega (Hopf/azimuthal) is an EXACT eigenvalue and the
        # shift-invert transform factorises J-sigma*M right at the singularity. SLEPc reports it as a
        # MUMPS zero pivot several frames away from the cause. shift=None is refused for the same
        # reason: SLEPc then targets 0 and still factorises there.
        if not shift:
            omega=self._get_bifurcation_omega()
            at="0" if not omega else "0 and +-"+str(abs(omega))+"j"
            raise RuntimeError("An eigensolve while bifurcation tracking is active needs a NON-ZERO shift: the tracked bifurcation puts an eigenvalue exactly at "+at+", which is where a zero shift asks the shift-invert transform to factorise. Pass e.g. shift=0.1 (or a value near the part of the spectrum you are looking for).")
        return True

    def _solve_eigenproblem_helper(self, n:int, shift:float | complex | None=0, quiet:bool=False, azimuthal_m:int | list[int] | None=None,normal_mode_k:ExpressionOrNum | list[ExpressionOrNum] | None=None,normal_mode_L:ExpressionOrNum | list[ExpressionOrNum] | None=None,report_accuracy:bool=False,sort:bool=True,which:"EigenSolverWhich"="LM",OPpart:Literal["r", "i"] | None=None,v0:NPFloatArray | NPComplexArray | None=None,filter:Callable[[complex], bool] | None=None,target:complex | None=None,ncv:int | None=None)->tuple[NPComplexArray,NPComplexArray]:
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
            return self._solve_normal_mode_eigenproblem(n, azimuthal_m=azimuthal_m, shift=shift, quiet=quiet,filter=filter,report_accuracy=report_accuracy,v0=v0,target=target,sort=sort,ncv=ncv)
        elif normal_mode_k is not None:
            normal_mode_k_nd:list[float] | float
            if isinstance(normal_mode_k,(list,tuple)):
                normal_mode_k_nd=[float(k*self.get_scaling("spatial")) for k in normal_mode_k]
            else:
                normal_mode_k_nd=float(normal_mode_k*self.get_scaling("spatial"))
            return self._solve_normal_mode_eigenproblem(n, cartesian_k=normal_mode_k_nd, shift=shift, quiet=quiet,filter=filter,report_accuracy=report_accuracy,v0=v0,target=target,sort=sort,ncv=ncv)
        tracking=self._check_eigensolve_during_augmentation(shift)
        if not tracking and self._dof_selector_used is not self._dof_selector:
            # Renumbers, so it is skipped while tracking -- where _check_eigensolve_during_augmentation
            # has already refused anything that would need a renumbering.
            self.reapply_boundary_conditions()
            self.reapply_boundary_conditions() # Must be done twice to correctly setup the equation remapping
        ntstep=self.ntime_stepper()
        was_steady=[False]*ntstep
        for i in range(ntstep):
            ts=self.time_stepper_pt(i)
            was_steady[i]=ts.is_steady()
            ts.make_steady()
        self.actions_before_eigen_solve(must_not_renumber=tracking)
        self.invalidate_cached_mesh_data(only_eigens=True)
        self.setup_forced_zero_dof_list_for_eigenproblems()
        eigen_solver=self.get_eigen_solver()
        # On a distributed problem each rank assembles only its own row block, and a backend that
        # cannot see that would quietly solve the block as if it were the whole eigenproblem and
        # return plausible, wrong eigenvalues.
        #
        # No eigensolver pyoomph ships answers no any more: SLEPc is genuinely distributed, and the
        # scipy/ARPACK family (including the Pardiso and Accelerate shift-invert variants derived from
        # it) gathers onto rank 0 instead of refusing. The check stays for a backend defined outside
        # pyoomph, which is the only thing that can still trip it.
        if self.is_distributed() and not eigen_solver.distributed_possible():
            raise RuntimeError("The eigensolver '"+str(getattr(eigen_solver,"idname",type(eigen_solver).__name__))+"' cannot solve an eigenproblem on a distributed (--distribute) problem. Use SLEPc instead, e.g. problem.set_eigensolver('slepc_mumps'), or run without --distribute.")
        if ncv is not None:
            eigen_solver.ncv=ncv
        self._last_eigenvalues,self._last_eigenvectors,J,M=eigen_solver.solve(n,shift=shift,sort=sort,which=which,OPpart=OPpart,v0=v0,target=target)
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
                    # J and M are (nrow_local x n) on a distributed problem while v is replicated at full
                    # length, so these products give this rank's rows of the residual and the max over
                    # them is a partial one. norm(v) needs no reduction -- v is already global.
                    lhs =l*(M@v) #type:ignore
                    rhs=J@v #type:ignore
                    diff=lhs-rhs #type:ignore
                    abs_err=numpy.max(numpy.absolute(diff)) if diff.shape[0]>0 else 0.0 #type:ignore
                    if get_mpi_nproc()>1:
                        abs_err=get_mpi_max(abs_err)
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
        # Unlike plain eigenvalue solving, this one is NOT available distributed: adapt() carries the
        # eigenfunction across the adaption in history levels 3 and 4, and the history dof accessors
        # refuse on a distributed problem (an equation number there is global while the vector holds
        # only this rank's rows -- see Problem::get_dofs(t,...) in src/problem.cpp). Raising here names
        # the feature instead of failing several frames into adapt().
        self._require_non_distributed("Mesh adaptation to an eigenfunction (refine_eigenfunction)")
        # adapt() renumbers, which pulls the augmented dof vector out from under the tracker.
        self._require_no_bifurcation_tracking("Mesh adaptation to an eigenfunction (refine_eigenfunction)")
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
            assert self._adapted_eigeninfo is not None, "adapt() should have populated _adapted_eigeninfo since _adapt_eigenindex was set"
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
            


    def _override_for_this_solve(self,*,max_newton_iterations:int | None=None,newton_relaxation_factor:float | None=None,newton_solver_tolerance:float | None=None,globally_convergent_newton:bool | None=None)->"_NewtonSolveOverrideSettings":
        old:_NewtonSolveOverrideSettings={}
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
    
    def is_global_parameter_used(self,param:str | _pyoomph.GiNaC_GlobalParam)->bool:
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

    def set_arc_length_parameter(self,desired_proportion_of_arc_length:float | None=None,scale_arc_length:bool | None=None,use_FD:bool | None=None,use_continuation_timestepper:bool | None=None,Desired_newton_iterations_ds:int | None=None):
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

    def arclength_continuation(self, parameter: str | _pyoomph.GiNaC_GlobalParam, step: float, *,
                              spatial_adapt: int = 0, max_ds: float | None = None,
                              max_newton_iterations: int | None = None,
                              newton_relaxation_factor: float | None = None,
                              newton_solver_tolerance: float | None = None,
                              min_ds: float | None = None, dof_direction: list[float] | None = None,
                              globally_convergent_newton: bool | None = False) -> float:
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
        if spatial_adapt>0 and self.get_bifurcation_tracking_mode()!="":
            raise RuntimeError("Cannot perform spatial adaptation during arclength continuation when bifurcation tracking is active. You can do the arclength step with spatial_adapt=0 followed by a solve(spatial_adapt="+str(spatial_adapt)+") to achieve a similar effect.")
        if self.is_distributed() and self.get_bifurcation_tracking_mode()!="":
            # Arclength continuation needs the dofs at history time levels (Problem::get_dofs(t,...)),
            # which are still refused when distributed -- otherwise this dies several frames deep in
            # C++ with a message about history dofs rather than about the thing being attempted.
            # Locating a bifurcation with solve() does work distributed; continuing it does not yet.
            raise RuntimeError("Arclength continuation while bifurcation tracking is active is not supported on a distributed (--distribute) problem yet (it needs history dofs, which are not distributed). A plain solve() to locate the bifurcation does work distributed.")
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
                raise RuntimeError("Arclength continuation in the global parameter '" + parameter + "', which is used in the problem. This may lead to unexpected behaviour. Have you defined it with define_global_parameter? Or have you overridden it by e.g. '" + parameter + "=<value>' instead of '" + parameter + ".value=<value>'? Have you defined it via define_global_parameter? Or have you overridden it by e.g. '" + parameter + "=<value>' instead of '" + parameter + ".value=<value>'?  Set <Problem>.warn_about_unused_global_parameters to False to suppress this error.")
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

    def go_to_param(self, _param:"dict[str,float] | None"=None, *, reset_pars:bool=True, startstep:float | None=None, call_after_step:Callable[[float], None] | None=None,final_adaptive_solve:bool | int=False,max_newton_iterations:int | None=None, epsilon:float=1e-6, max_step:float | None=None,**kwargs:float)->None:
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
        kwargs=dict(kwargs)
        if _param is not None:
            kwargs.update(_param)
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
        if self._bifurcation_reactivation_after_adaptation is not None:
            self._reactivate_bifurcation_tracking_after_adaption()
            self._bifurcation_reactivation_after_adaptation=None
        self._last_eigenvectors:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
        self._last_eigenvalues:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
        self._last_eigenvalues_m=None
        self._last_eigenvalues_k=None

    # Warning: This must be used with "for parameter, eigenvalue in find_bifurcation_via_eigenvalues(...):"
    def find_bifurcation_via_eigenvalues(self, parameter:str | _pyoomph.GiNaC_GlobalParam, initstep:float, shift:None | float | complex=0, neigen:int=6, spatial_adapt:int=0, epsilon:float=1e-8, reset_arclength:bool=False, max_ds:float | Callable[[float], float] | None=None, stay_stable_file:str | None=None, before_eigensolving:Callable[[float], None] | None=None, do_solve:bool=True, azimuthal_m:int | list[int] | None=None, normal_mode_k:ExpressionNumOrNone=None, eigenindex:int=0):
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
            oldk:float | None=None
            if param_is_normal_mode_k:
                assert self._normal_mode_param_k is not None
                oldk=self._normal_mode_param_k.value
            evals0, _ = self._solve_normal_mode_eigenproblem(neigen, cartesian_k=normal_mode_k, shift=shift) #type:ignore
            if param_is_normal_mode_k:
                assert self._normal_mode_param_k is not None
                assert oldk is not None
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
                assert not callable(max_ds)
                ds = self.arclength_continuation(parameter, ds,max_ds=max_ds)
                self.invalidate_cached_mesh_data()
            else:
                parameter.value=parameter.value+ds
                if callable(max_ds_func):
                    max_ds=max_ds_func(parameter.value)
                    max_ds=abs(max_ds)
                    print("MAX DS SET TO",max_ds)
                assert not callable(max_ds)
                # Enlarge the step while marching towards the first sign change. This used to be
                # skipped entirely unless max_ds was given, so a caller without max_ds marched at
                # the initial step all the way to the bifurcation -- 89 steps of 200 for the m=3
                # mode of the Rayleigh-Benard tutorial. Once the sign has changed we are bisecting
                # the bracket and must not grow anymore, otherwise the halving below is undone.
                if not firstSignChange:
                    dsnew=abs(1.5*ds)
                    if max_ds is not None:
                        dsnew=min(dsnew,max_ds)
                    ds=dsnew*(1 if ds>0 else -1)
            if reset_arclength:
                self.reset_arc_length_parameters()
            if before_eigensolving is not None:
                before_eigensolving(param0)
            if normal_mode_k is not None and not is_zero(normal_mode_k):
                oldk=None
                if param_is_normal_mode_k:
                    assert self._normal_mode_param_k is not None
                    oldk=self._normal_mode_param_k.value
                evals1, _ = self.solve_eigenproblem(neigen, normal_mode_k=normal_mode_k, shift=shift)
                if param_is_normal_mode_k:
                    assert self._normal_mode_param_k is not None
                    assert oldk is not None
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
                # Bisect the bracket [param0,param1], which is |ds0| wide. ds itself must not be
                # used here: it has already been enlarged for the next march step, so stepping
                # back by half of it would overshoot backwards past param0.
                dsmagn = abs(ds0)
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
        """After initialisation, the property max_refinement_level is not considered anymore. You can set the maximum refinement level of the meshes with this function"""
        if level<0:
            raise RuntimeError("Must be >=0")
        
        def set_level_for_mesh(mesh:AnySpatialMesh,level):
            assert not isinstance(mesh,InterfaceMesh)
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
        
        
    def perturb_by_eigenfunction(self,*,dt:ExpressionNumOrNone | None=None, eigenmode:int=0,time_steps_per_growth:float=20,desired_initial_residuals:float | None=1e-1)->ExpressionNumOrNone:
        """
        Perturb the current solution by an eigenfunction corresponding to the specified eigenmode index.
        The if the  time step dt is not set, it is chosen so that there are time_steps_per_growth time steps per growth time of the eigenmode.
        Eigenindex selects which eigenmode to use (default 0 for the most unstable mode).
        The perturbation amplitude is chosen so that the initial residuals after perturbation are approximately desired_initial_residuals (if not None).
        Returns the time step dt to be used for time stepping after the perturbation.
        
        Args:
            dt: Time step to use after perturbation. If None, it is computed based on the eigenvalue using time_steps_per_growth.
            eigenmode: Index of the eigenmode to use for the perturbation.
            time_steps_per_growth: Number of time steps per growth time of the eigenmode (used if dt is None).
            desired_initial_residuals: Desired initial residuals after perturbation. If None, no scaling is done.
        """
        eigenvals=self.get_last_eigenvalues()
        if eigenvals is None or eigenmode>=len(eigenvals):
            raise ValueError("No eigenvalues computed or eigenmode index out of range")
        
        TS=self.get_scaling("temporal")
        if dt is None:
            dtfixed=1.0/max(abs(eigenvals[eigenmode].real),abs(eigenvals[eigenmode].imag))*TS/time_steps_per_growth
        else:
            dtfixed=dt
        self.initialise_dt(float(dtfixed/TS))
        dofs,_=self.get_current_dofs()
        dofs=numpy.array(dofs)
        evect=self.get_last_eigenvectors()[eigenmode]
        
        self._taken_already_an_unsteady_step=True
        self.timestepper.set_num_unsteady_steps_done(2)
        self.time_stepper_pt().undo_make_steady()
        self.initialise_dt(float(dtfixed/TS))
        self.time_stepper_pt().set_weights()
        
        
        
        # TODO: look for complex conjugate pairs and handle appropriately
        def get_dofs(scale,toffset):
            return dofs+numpy.real(0.5*scale*(evect*numpy.exp(eigenvals[eigenmode]*float(toffset/TS))+numpy.conjugate(evect*numpy.exp(eigenvals[eigenmode]*float(toffset/TS)))))
        
        # TODO: Better way to  history dofs here:
        self.set_history_dofs(3,0*dofs) # Newmark velo
        self.set_history_dofs(4,0*dofs) # Newmark accel
        self.set_history_dofs(5,0*dofs) # BDF2 velocity
        self.set_history_dofs(6,0*dofs) # Predictor
        def set_ic(scale):
            self.set_current_dofs(get_dofs(scale,0.0))
            self.set_history_dofs(1,get_dofs(scale,-dtfixed))
            self.set_history_dofs(2,get_dofs(scale,-2*dtfixed))
            self.set_history_dofs(3,get_dofs(scale,-3*dtfixed))
            maxres=numpy.amax(numpy.absolute(self.get_residuals()))
            #print("Max residual after perturbation with scale {}: {}".format(scale,maxres))
            return maxres
    
        if desired_initial_residuals is not None:
            scale0=1.0
            res0=set_ic(scale0)
            scale1=scale0 # In case res0 already exactly equals desired_initial_residuals, neither while loop below runs
            if res0>desired_initial_residuals:
                while res0>desired_initial_residuals:
                    scale1=scale0/2
                    res1=set_ic(scale1)
                    if res1<desired_initial_residuals:
                        break
                    else:
                        res0=res1
                        scale0=scale1
            else:
                while res0<desired_initial_residuals:  
                    scale1=scale0*2
                    res1=set_ic(scale1)
                    if res1>desired_initial_residuals:
                        break
                    else:
                        scale0=scale1
                        res0=res1
            #print("Final scale0: {}, residual0: {}".format(scale0,res0))
            #print("Final scale1: {}, residual1: {}".format(scale1,res1))
            if scale0>scale1:
                scale0,scale1=scale1,scale0
            # Now do a bisection between scale0 and scale1
            scale:float=scale0
            for _ in range(20):
                scale=(scale0+scale1)/2
                res=set_ic(scale)
                #print("Bisection scale: {}, residual: {}".format(scale,res),desired_initial_residuals)
                if res>desired_initial_residuals:                    
                    scale1=scale
                else:
                    scale0=scale
        
            res=set_ic(scale)
            #print("Final scale: {}, residual: {}".format(scale,res))
        self.invalidate_cached_mesh_data()
        self.invalidate_eigendata()
        return dtfixed

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
            assert self._azimuthal_mode_param_m is not None
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

    def guess_nearest_bifurcation_type(self,eigenvector:int=0)->Literal["hopf","fold","pitchfork","azimuthal","cartesian_normal_mode"]:
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


    
    def dof_strings_to_global_equations(self,string_dof_set:str | Iterable[str | int]):
        """Takes strings like ``"domain/velocity_x"`` and returns a set of global equations.
        Entries that are already an int (a global equation number, as returned e.g. by some
        InterfaceEquations._get_forced_zero_dofs_for_eigenproblem implementations) are passed through as-is.

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
        string_names={d for d in string_dof_set if isinstance(d,str)}
        zeromap:set[int]={d for d in string_dof_set if isinstance(d,int)}
        resolver=EigenMatrixSetDofsToZero(self,*string_names)
        for d in resolver.doflist:
            assert isinstance(d,str) # only strings were passed to the resolver above
            eqs=resolver.resolve_equations_by_name(d)
            #print("DOF",d,"EQS",eqs)
            zeromap=zeromap.union(eqs)
        if self.is_distributed():
            # resolve_equations_by_name() only walks the LOCAL elements, so on a distributed problem
            # each rank sees only the part of the named mesh it holds -- and a rank holding none of it
            # gets nothing back at all. This function promises GLOBAL equation numbers, and its callers
            # use them to index globally replicated vectors (rotate_eigenvectors on an eigenvector, the
            # bifurcation zero-dof lists), so every rank has to end up with the same set.
            #
            # Without the union, np=4 on the rising-bubble tutorial died in rotate_eigenvectors with
            # "zero-size array to reduction operation maximum" on the rank owning none of
            # domain/interface. The quieter half was worse: at np=2 both ranks held part of the
            # interface, so nothing raised and each rank rotated the eigenvector by a phase computed
            # from its own subset -- a silently different eigenvector per rank.
            #
            # Collective, so it must be reached by every rank. It is: the callers are eigen-solve and
            # bifurcation setup paths, which are collective already.
            from .mpi import get_mpi_nproc,get_mpi_world_comm
            if get_mpi_nproc()>1:
                comm=get_mpi_world_comm()
                assert comm is not None
                for part in comm.allgather(sorted(zeromap)):
                    zeromap.update(part)
        return zeromap
            

    def activate_eigenbranch_tracking(self,branch_type:Literal["real", "complex", "normal_mode"] | None=None,eigenvector:NPFloatArray | NPComplexArray | int | None=None,eigenvalue:complex | None=None):
        """Activates eigenbranch tracking for the specified eigenbranch type. Subsequent calls of solve(...) and arclength_continuation(...) will then track the eigenbranch.
        This is similar to bifurcation tracking, but it does not adjust a parameter to find a bifurcation, i.e. where Re(lambda)=0. Instead, it starts with a eigenvalue/eigenvector pair. Once activated, you can follow the eigenbranch by calling arclength_continuation(...).        
        At each step, the eigenvalue/eigenvector pair will be updated and is available via get_last_eigenvalues()[0] and get_last_eigenvectors()[0].
        
        Args:
            branch_type (Optional[Literal["real", "complex", "normal_mode"]]): The type of eigenbranch to track. Defaults to None, i.e. auto-detect.
            eigenvector (Optional[int]): The previously calculated eigenvector index to use for tracking. Defaults to None, i.e. the eigenvector at index zero.
        """
        self.activate_bifurcation_tracking(None,bifurcation_type=branch_type,eigenvector=eigenvector,eigenvalue_for_branch_tracking=eigenvalue)

    def activate_bifurcation_tracking(self,parameter:str | _pyoomph.GiNaC_GlobalParam | None,bifurcation_type:Literal["hopf", "fold", "pitchfork", "azimuthal", "cartesian_normal_mode", "real", "complex", "normal_mode"] | None=None,blocksolve:bool=False,eigenvector:NPFloatArray | NPComplexArray | int | None=None,omega:float | None=None,azimuthal_mode:int | None=None,cartesian_wavenumber_k:ExpressionOrNum | None=None,eigenvalue_for_branch_tracking:complex | None=None,eigenvector_scaling:Literal["unit","auto"]="unit"):
        """
        Activates bifurcation tracking for the specified parameter and bifurcation type. Subsequent calls of solve(...) and arclength_continuation(...) will then track the bifurcation.

        Args:
            parameter: The parameter to change in order to find the bifurcation. If None, we track the current eigenbranch, i.e. Re(lambda) will be found and is not necessarily 0.
            bifurcation_type (Optional[Literal["hopf", "fold", "pitchfork", "azimuthal"]]): The type of bifurcation to track. Defaults to None, i.e. auto-detect.
            blocksolve (bool): Flag indicating whether to use block solve. Defaults to False. Should be kept False.
            eigenvector (Optional[Union[NPFloatArray, NPComplexArray, int]]): The eigenvector to use for tracking. Defaults to None, which means the eigenvector corresponding to the eigenvalue with largest real part. Can be either an index or a custom vector.
            omega (Optional[float]): The omega value for Hopf bifurcation tracking. Defaults to None, then it will be Im(lambda).
            azimuthal_mode (Optional[int]): The azimuthal mode for azimuthal bifurcation tracking. Defaults to None.
            eigenvector_scaling: How the eigenvector normalization constraint is scaled. "unit" (default) keeps the
                historical behaviour: the eigenvector guess is normalized to unit length and the constraint reads
                c.y = 1, so on a problem with N degrees of freedom the eigenvector unknowns are of order 1/sqrt(N).
                "auto" instead normalizes the guess by its largest entry and moves the constraint's right-hand side
                to match, keeping the eigenvector unknowns and the constraint row of order one however large the
                problem is. This is recommended for very large systems. It does not move the bifurcation itself,
                only the (always arbitrary) amplitude of the reported eigenfunction.
        """

        # Distributed (--distribute) support is decided per bifurcation type AFTER the type has been
        # resolved (see the check further down); no blanket refusal here anymore.
        self._bifurcation_eigenvector_scaling=eigenvector_scaling
        self.reset_arc_length_parameters()

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
                if bifurcation_type is None:
                    bifurcation_type=self.guess_nearest_bifurcation_type(eigenvector)
            else:
                #raise RuntimeError("Can only track eigenbranches, not custom vectors. Please set eigenvector to and integer (for the index of the calculate eigenvector or None, meaning index 0) ")
                eigenvector_v=eigenvector
                if eigenvalue_for_branch_tracking is None:
                    raise RuntimeError("Please set eigenvalue_for_branch_tracking if you track a custom eigenvector")
                self._set_lambda_tracking_real(numpy.real(eigenvalue_for_branch_tracking))
                omega=numpy.imag(eigenvalue_for_branch_tracking)
                if bifurcation_type is None:
                    bifurcation_type="hopf" if numpy.abs(numpy.imag(eigenvalue_for_branch_tracking))>1e-6 else "fold"
            
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
                    # omega is only unset here if eigenvector is an int index (the custom-vector
                    # branch above always sets omega from eigenvalue_for_branch_tracking)
                    assert isinstance(eigenvector,int)
                    omega=numpy.imag(self.get_last_eigenvalues()[eigenvector])
                assert omega is not None
                print("Activating eigenbranch tracking for a complex branch with with starting eigenvalue",str(complex(self._get_lambda_tracking_real(),omega)))
            elif bifurcation_type=="azimuthal" or bifurcation_type=="cartesian_normal_mode" or bifurcation_type=="normal_mode":
                if azimuthal_mode is None and cartesian_wavenumber_k is None and isinstance(eigenvector,int):
                    last_modes_k=self.get_last_eigenmodes_k()
                    last_modes_m=self.get_last_eigenmodes_m()
                    if last_modes_k is not None and len(last_modes_k)>eigenvector and not is_zero(last_modes_k[eigenvector]):
                        bifurcation_type="cartesian_normal_mode"
                        cartesian_wavenumber_k=last_modes_k[eigenvector]
                    elif last_modes_m is not None and len(last_modes_m)>eigenvector:
                        bifurcation_type="azimuthal"
                        azimuthal_mode=int(last_modes_m[eigenvector])
                elif azimuthal_mode is not None and cartesian_wavenumber_k is not None:
                    raise RuntimeError("Cannot track eigenbranch for both azimuthal and cartesian normal mode")
                elif azimuthal_mode is not None:
                    bifurcation_type="azimuthal"
                else:
                    bifurcation_type="cartesian_normal_mode"
                if omega is None:
                    assert isinstance(eigenvector,int)
                    omega=numpy.imag(self.get_last_eigenvalues()[eigenvector])
                assert omega is not None
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
                    last_modes_m=self.get_last_eigenmodes_m()
                    assert last_modes_m is not None, "No azimuthal modes available. Did you use setup_for_stability_analysis(azimuthal_stability=True)?"
                    azimuthal_mode=int(last_modes_m[eigenvector])
                elif bifurcation_type=="cartesian_normal_mode" and cartesian_wavenumber_k is None:
                    last_modes_k=self.get_last_eigenmodes_k()
                    assert last_modes_k is not None, "No cartesian normal modes available. Did you use setup_for_stability_analysis(additional_cartesian_mode=True)?"
                    cartesian_wavenumber_k=last_modes_k[eigenvector]
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
        if self.is_distributed():
            # The handlers ported to distributed operation in src/bifurcation.cpp. The remaining
            # refusals sit on genuinely serial machinery (blocksolve rebuilds replicated dof
            # vectors; orbit tracking overwrites arbitrary global dofs during assembly).
            _distributed_supported={"fold","hopf","pitchfork","azimuthal","cartesian_normal_mode"}
            if bifurcation_type not in _distributed_supported:
                raise RuntimeError("Bifurcation tracking of type '"+str(bifurcation_type)+"' is not supported on a distributed (--distribute) problem yet. Run it without --distribute.")
            if blocksolve:
                raise RuntimeError("blocksolve=True is not supported for bifurcation tracking on a distributed (--distribute) problem")
        self._bifurcation_tracking_parameter_name=parameter
        if bifurcation_type=="fold":
#            must_reapply_bcs=self._equation_system._before_eigen_solve(self.get_eigen_solver(), 0)
#            if must_reapply_bcs:
#                self.reapply_boundary_conditions() # Equation numbering might have been changed. Update it here!
#                self._last_bc_setting="eigen"
            if azimuthal_mode is not None or cartesian_wavenumber_k is not None:
                raise RuntimeError("Cannot use azimuthal_mode or cartesian_wavenumber_k for fold solving")
            if isinstance(eigenvector,int):
                eigenvector=self.get_last_eigenvectors()[eigenvector]
            elif eigenvector is None:
                eigenvector = next(iter(self.get_last_eigenvectors()), None)
            assert not isinstance(eigenvector,int)
            if eigenvector is None or len(eigenvector)==0:
                if self.is_distributed():
                    # The no-guess fold constructor derives its guess from a serial linear solve on
                    # a replicated vector; the C++ side throws the same way, but this traceback is clearer.
                    raise RuntimeError("Fold tracking on a distributed (--distribute) problem requires an explicit eigenvector guess -- solve an eigenproblem first or pass eigenvector=...")
                self._start_bifurcation_tracking(parameter,bifurcation_type,blocksolve,[],[],0.0,{},eigenvector_scaling)
            else:
                self._start_bifurcation_tracking(parameter,bifurcation_type,blocksolve,numpy.real(eigenvector),[],0.0,{},eigenvector_scaling) #type:ignore
        elif bifurcation_type=="hopf":
            if azimuthal_mode is not None or cartesian_wavenumber_k is not None:
                raise RuntimeError("Cannot use azimuthal_mode or cartesian_wavenumber_k for Hopf solving")
            if isinstance(eigenvector,int):
                eigenvector=self.get_last_eigenvectors()[eigenvector]
            elif eigenvector is None:
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

                self._start_bifurcation_tracking(parameter,bifurcation_type,blocksolve,numpy.real(eigenvector),numpy.imag(eigenvector),omega,{},eigenvector_scaling) #type:ignore
        elif bifurcation_type=="pitchfork":
            if azimuthal_mode is not None or cartesian_wavenumber_k is not None:
                raise RuntimeError("Cannot use azimuthal_mode or cartesian_wavenumber_k for pitchfork solving")
            if isinstance(eigenvector,int):
                eigenvector=self.get_last_eigenvectors()[eigenvector]
            elif eigenvector is None:
                eigenvector=next(iter(self.get_last_eigenvectors()),None)
            if eigenvector is None:
                raise RuntimeError("Pitchfork tracking requires at least a symmetry vector passed via the eigenvector kwarg")
            self._start_bifurcation_tracking(parameter,bifurcation_type,blocksolve,numpy.real(eigenvector),[],0.0,{},eigenvector_scaling) #type:ignore
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

  
            self._start_bifurcation_tracking(parameter, bifurcation_type, blocksolve, numpy.real(eigenvector),numpy.imag(eigenvector), omega,contribs,eigenvector_scaling) #type:ignore
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
            self._normal_mode_param_k.value=cartesian_wavenumber_k #type:ignore
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
            must_reapply_bcs=self._equation_system._before_eigen_solve(self.get_eigen_solver(), normal_k=cartesian_wavenumber_k) #type:ignore
            if must_reapply_bcs:
                self.reapply_boundary_conditions() # Equation numbering might have been changed. Update it here!
                self.reapply_boundary_conditions()
                self._last_bc_setting="eigen"
            base_zero_dofs=self._equation_system._get_forced_zero_dofs_for_eigenproblem(self.get_eigen_solver(),None,None)
            eigen_zero_dofs=self._equation_system._get_forced_zero_dofs_for_eigenproblem(self.get_eigen_solver(),None,cartesian_wavenumber_k) #type:ignore 

            

            base_zero_dofs=self.dof_strings_to_global_equations(base_zero_dofs)
            eigen_zero_dofs=self.dof_strings_to_global_equations(eigen_zero_dofs)

            #print("BASE DOFS",base_zero_dofs)
            #print("EIGEN DOFS",eigen_zero_dofs)
            #print("OMEGA {:g}".format(omega))
            contribs={"azimuthal_real_eigen":self._cartesian_normal_mode_stability.real_contribution_name,"azimuthal_imag_eigen":self._cartesian_normal_mode_stability.imag_contribution_name}
            has_imag=self._set_solved_residual(self._cartesian_normal_mode_stability.imag_contribution_name,False,False)
            if not has_imag:
                contribs["azimuthal_imag_eigen"]="<NONE>"
            self._set_solved_residual("",False,True)
            #print("GOING FOR IT ",parameter, bifurcation_type, blocksolve,  -omega,contribs)
            #print("KVALUE",self._normal_mode_param_k.value,"HAS IMAG",has_imag)
            
            self._start_bifurcation_tracking(parameter, bifurcation_type, blocksolve, numpy.real(eigenvector),numpy.imag(eigenvector), omega,contribs,eigenvector_scaling) #type:ignore
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
        self._require_non_distributed("Periodic orbit tracking")
        self.deactivate_bifurcation_tracking()
        self.time_stepper_pt().make_steady()
        if len(history_dofs)==0:
            raise ValueError("No history dofs provided")
        knots:list[float]=[]
        # T_constraint_mode is the integer code _start_orbit_tracking expects; T_constraint itself
        # (the "plane"/"phase" string) must stay unmodified below - it is also passed straight
        # through to PeriodicOrbit(...), whose own T_constraint attribute is declared (and used
        # elsewhere, e.g. PeriodicOrbit.change_sampling()) as that same Literal["plane","phase"]
        # string, not this integer code.
        if T_constraint=="plane":
            T_constraint_mode=0
        elif T_constraint=="phase":
            T_constraint_mode=1
        else:
            raise ValueError("Invalid T_constraint: "+str(T_constraint))
        T_nd=float(T/self.get_scaling("temporal"))
        if mode=="floquet":
            self._start_orbit_tracking(history_dofs,T_nd,0,-1,knots,T_constraint_mode)
        elif mode=="bspline":
            if order<1:
                raise ValueError("Invalid bspline order: "+str(order))
            self._start_orbit_tracking(history_dofs,T_nd,order,GL_order,knots,T_constraint_mode)
        elif mode=="central":
            self._start_orbit_tracking(history_dofs,T_nd,-1,-1,knots,T_constraint_mode)
        elif mode=="BDF2":
            self._start_orbit_tracking(history_dofs,T_nd,-2,-1,knots,T_constraint_mode)
        elif mode=="collocation":
            if order<1:
                raise ValueError("Invalid collocation order: "+str(order))
            self._start_orbit_tracking(history_dofs,T_nd,-2-order,GL_order,knots,T_constraint_mode)
        else:
            raise ValueError("Invalid mode: "+str(mode))
        res=PeriodicOrbit(self,mode,0,None,0,None,None,0,order,GL_order,T_constraint)
        return res

    def switch_to_hopf_orbit(self,eps:float=0.01,dparam:float | None=None,NT:int=30,mode:Literal["collocation","floquet","central","BDF2","bspline"]="collocation",order:int=3,GL_order:int=-1,T_constraint:Literal["phase","plane"]="phase",amplitude_factor:float=1,FD_delta:float=1e-5,FD_param_delta=1e-3,do_solve:bool=True,solve_kwargs:dict[str,Any]={},check_collapse_to_stationary:bool=True,orbit_amplitude:float | None=None,patch_number_of_nodes:bool=True)->PeriodicOrbit:
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
        
        from .bifurcation_tools import get_hopf_lyapunov_coefficient    
        
        if self._bifurcation_tracking_parameter_name is None or self.get_bifurcation_tracking_mode()!="hopf" or len(self.get_last_eigenvalues())!=1:
            raise ValueError("Hopf bifurcation tracking not activated or solved. Please call activate_bifurcation_tracking first, then solve. Then call this routine.")        
        # Store the information from the Hopf tracker
        omega=self.get_last_eigenvalues()[0].imag                            
        #q=self.get_last_eigenvectors()[0]
        
        
        q:NPComplexArray=numpy.array(cast(_pyoomph.HopfHandler,self.assembly_handler_pt()).get_nicely_rotated_eigenfunction()) #type:ignore
        if omega<0:
            omega=-omega
            q=numpy.conj(q) #type:ignore
        
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
        orbit_base_ndof=cast(_pyoomph.PeriodicOrbitHandler,self.assembly_handler_pt()).get_base_ndof()
        ncnt:int | None=None
        avg_dists0:float | None=None
        if check_collapse_to_stationary:
            avg_dists0=0
            ncnt=0
            for T in res.iterate_over_samples():
                dofs=self.get_current_dofs()[0][:orbit_base_ndof]
                avg_dists0+=(numpy.dot(numpy.array(history_dofs[ncnt])-numpy.array(u0),numpy.array(dofs)-numpy.array(u0)))
                ncnt+=1
            assert avg_dists0 is not None and ncnt is not None
            avg_dists0/=ncnt


        if do_solve:
            self.solve(**solve_kwargs)
            if check_collapse_to_stationary:
                assert ncnt is not None and avg_dists0 is not None
                avg_dists=0.0
                i=0
                for T in res.iterate_over_samples():
                    dofs=self.get_current_dofs()[0][:orbit_base_ndof]
                    add=numpy.dot(numpy.array(history_dofs[i])-numpy.array(u0),numpy.array(dofs)-numpy.array(u0))
                    #print("adding",add,numpy.amax(numpy.absolute(numpy.array(history_dofs[i])-numpy.array(u0))))
                    avg_dists+=add
                    i+=1
                avg_dists=avg_dists/ncnt
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
                        start=self.get_current_dofs()[0][:orbit_base_ndof]
                    else:
                        dist=numpy.linalg.norm(numpy.array(start)-numpy.array(self.get_current_dofs()[0][:orbit_base_ndof]))
                        if dist>1e-5*avg_dists0:
                            nontrivial=True
                            skip=True
                    i+=1
                if not nontrivial:
                    raise RuntimeError("The solved orbit is likely collapsed")
                    #print("DOT",numpy.sqrt(numpy.dot(numpy.array(history_dofs[i])-numpy.array(u0),numpy.array(dofs)-numpy.array(u0))))
        return res
        
    def get_floquet_multipliers(self,n:int | None=None,valid_threshold:float | None=10000,shift:float | None=None,ignore_periodic_unity:bool | float=False,quiet:bool=True)->NPComplexArray:
        """
        TODO; Add documentation
        """
        # Main ideas from here: https://arxiv.org/html/2407.18230v1#S2.E6
        from .. import _pyoomph_core as _pyoomph
        import scipy
        orbit_handler=self.assembly_handler_pt()
        if not isinstance(orbit_handler,_pyoomph.PeriodicOrbitHandler):
            raise RuntimeError("Periodic orbit handler not active. Call activate_periodic_orbit_handler first, then solve the orbit, then call this function")
        if not orbit_handler.is_floquet_mode():
            raise RuntimeError("Floquet mode not active. Call activate_periodic_orbit_handler with mode='floquet' first, then solve the orbit, then call this function")
        nbase=orbit_handler.get_base_ndof()
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
        gamms:NPComplexArray=1/(1-valid_eigs) #type:ignore
        
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
            imaginary_i=_pyoomph.GiNaC_imaginary_i()
            return numpy.array([numpy.real(x)/self.get_scaling("temporal")+imaginary_i*numpy.imag(x)/self.get_scaling("temporal") for x in self._last_eigenvalues]) #type:ignore
        return self._last_eigenvalues

    def get_last_eigenvectors(self)->NPComplexArray:
        """Return the last computed eigenvector.

        Returns:
            NPComplexArray: Eigenvectors as 2d array.
        """
        return self._last_eigenvectors

    def get_last_eigenmodes_m(self) -> NPIntArray | None:
        """Get the azimuthal mode numbers for the last computed eigenvalues.

        Returns:
            Optional[NPIntArray]: Array containing the azimuthal mode numbers corresponding to the eigenvalues.
        """
        return self._last_eigenvalues_m
    
    def get_last_eigenmodes_k(self)->NPFloatArray | None:
        """Get the cartesian normal mode numbers for the last computed eigenvalues.

        Returns:
            Optional[NPFloatArray]: Array containing the cartesian normal mode numbers corresponding to the eigenvalues.
        """
        return self._last_eigenvalues_k


    def rotate_eigenvectors(self,eigenvectors,dofs_to_real:str | list[str] | set[str],normalize_dofs:bool=False,normalize_amplitude:float | complex=1,normalize_max:bool=True):
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
        if len(dofs)==0:
            # Rather than let numpy report "zero-size array to reduction operation maximum" from the
            # averaging below, several frames away from the name that resolved to nothing.
            raise RuntimeError("rotate_eigenvectors: "+str(dofs_to_real)+" does not resolve to any degree of freedom, so there is nothing to fix the eigenvector's phase by. Check the name(s).")
        # Sorted, so that the averaging below adds the same values in the same order everywhere. The
        # set comes back identical on every rank (see dof_strings_to_global_equations), but a set's
        # iteration order is not part of that guarantee, and this feeds a floating-point sum.
        dofs=numpy.array(sorted(dofs),dtype=numpy.int64)
        from .mpi import get_mpi_nproc,get_mpi_sum,get_mpi_max
        # Both numbers below scale the WHOLE eigenvector, so ranks that disagree about them end up
        # holding different eigenvectors -- silently, since the eigenvalue is unaffected.
        #
        # Which of the two branches applies depends on what the eigensolver handed us. The PETSc one
        # currently replicates each eigenvector to full global length on every rank
        # (_vector_to_global_array), so we normally take the first branch: every rank already has
        # every selected dof, reduces over the identical sorted set, and needs no communication --
        # which also keeps the result bit-identical to the serial run. Under --distribute a solver may
        # instead hand back only the locally owned row block, and then the reduction has to span the
        # ranks.
        # The BASE layout: an eigenvector has base length even while a bifurcation tracker makes the
        # problem's own dof distribution the larger augmented one.
        n_global,nrow_local,first_row,_row_distributed=self._get_base_dof_distribution_info()
        for ev in eigenvectors:
            if get_mpi_nproc()>1 and len(ev)!=n_global:
                if len(ev)!=nrow_local:
                    raise RuntimeError("rotate_eigenvectors: the eigenvector has length "+str(len(ev))+
                                       ", which is neither the global dof count "+str(n_global)+" nor "
                                       "this rank's local row count "+str(nrow_local)+".")
                # Only the owned block is here, so each rank contributes the selected dofs that fall in
                # its row range -- restricting to owned rows is what stops shared/halo rows from being
                # counted more than once. Each quantity then has to be combined with the operator it is
                # actually made of: a maximum reduces with MAX, an average with a summed numerator AND a
                # summed count (averaging the per-rank averages would silently weight the ranks equally
                # however unevenly the dofs are spread over them). The mean whose angle we take is
                # complex, so both of its components have to travel.
                sel=dofs[(dofs>=first_row)&(dofs<first_row+nrow_local)]-first_row
                vals=ev[sel]
                count=get_mpi_sum(len(sel))
                total=complex(get_mpi_sum(float(numpy.sum(vals.real))),get_mpi_sum(float(numpy.sum(vals.imag))))
                avg_angle=numpy.angle(total/count)
                if normalize_dofs:
                    absvals=numpy.absolute(vals)
                    if normalize_max:
                        # 0.0 is the identity for a maximum of magnitudes, so a rank owning none of the
                        # selected dofs contributes nothing rather than tripping amax on an empty array.
                        magnitude=get_mpi_max(float(numpy.amax(absvals)) if len(sel) else 0.0)
                    else:
                        magnitude=get_mpi_sum(float(numpy.sum(absvals)))/count
                else:
                    magnitude=1
            else:
                avg_angle=numpy.angle(numpy.average(ev[dofs]))
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


    def _solve_normal_mode_eigenproblem(self, n:int, azimuthal_m:list[int] | tuple[int] | int | None=None, cartesian_k:list[float] | tuple[float] | float | None=None, shift:float | complex | None=0,quiet:bool=False,filter:Callable[[complex], bool] | None=None,report_accuracy:bool=False,target:complex | None=None,v0:NPFloatArray | NPComplexArray | None=None,sort:bool=True,ncv:int | None=None)->tuple[NPComplexArray,NPComplexArray]:
        
        if azimuthal_m is not None and (self._azimuthal_mode_param_m is None):
            raise RuntimeError("Must use setup_for_stability_analysis(azimuthal_stability=True) before initialialising the problem")
        if cartesian_k is not None and (self._normal_mode_param_k is None):
            raise RuntimeError("Must use setup_for_stability_analysis(additional_cartesian_mode=True) before initialialising the problem")

        if cartesian_k is not None and azimuthal_m is not None:
            raise RuntimeError("TODO: Both simultaneously")
        vlist:"list[int] | tuple[int] | int | list[float] | tuple[float] | float"
        if cartesian_k is not None:
            param=self._normal_mode_param_k
            vlist=cartesian_k
        elif azimuthal_m is not None:
            param=self._azimuthal_mode_param_m
            vlist=azimuthal_m
        else:
            raise RuntimeError("Must specify either azimuthal_m or cartesian_k")
        assert param is not None

        tracking=self._check_eigensolve_during_augmentation(shift)
        # The mode parameter used to be reset to 0 when done. That is wrong while tracking: the
        # azimuthal/normal-mode tracker reads the very same global parameter when it assembles its
        # eigen rows, so hard-coding 0 here would silently retune the tracked bifurcation to m=0.
        # Restore what was set on entry instead (which is 0 in the untracked case, as before).
        mode_value_on_entry=param.value
        try:
            return self._solve_normal_mode_eigenproblem_impl(n,param,vlist,azimuthal_m,shift,quiet,filter,report_accuracy,target,v0,sort,ncv,tracking)
        finally:
            param.value=mode_value_on_entry

    def _solve_normal_mode_eigenproblem_impl(self,n:int,param:"_pyoomph.GiNaC_GlobalParam",vlist:"list[int] | tuple[int] | int | list[float] | tuple[float] | float",azimuthal_m:list[int] | tuple[int] | int | None,shift:float | complex | None,quiet:bool,filter:Callable[[complex], bool] | None,report_accuracy:bool,target:complex | None,v0:NPFloatArray | NPComplexArray | None,sort:bool,ncv:int | None,tracking:bool)->tuple[NPComplexArray,NPComplexArray]:
        if isinstance(vlist,(list,tuple)):
            if report_accuracy:
                raise RuntimeError("report_accuracy=True for normal mode eigenproblems only works if you select a single mode, not a list like "+str(vlist))
            alleigenvals:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
            alleigenvects:NPComplexArray=numpy.array([],dtype=numpy.complex128) #type:ignore
            minfoL:list[int | float]=[]
            for ms in vlist:
                param.value = ms
                self.actions_before_eigen_solve(must_not_renumber=tracking)
                self._solve_eigenproblem_helper(n, shift,quiet=True,filter=filter,report_accuracy=report_accuracy,target=target,v0=v0,sort=sort,ncv=ncv)
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
                minfo:NPAnyArray=numpy.array(minfoL)[srt] #type:ignore
            else:
                minfo=numpy.array(minfoL)

            self._last_eigenvalues, self._last_eigenvectors = alleigenvals,alleigenvects
            if azimuthal_m is not None:
                self._last_eigenvalues_m=minfo #type:ignore
            else:
                self._last_eigenvalues_k=minfo #type:ignore

            if (not self.is_quiet()) and (not quiet):
                for i, l in enumerate(self._last_eigenvalues):
                    m=minfo[i]
                    print("Eigenvalue [m="+str(m)+"]", i, ":", l)
        else:
            param.value = vlist
            self.actions_before_eigen_solve(must_not_renumber=tracking)
            self._solve_eigenproblem_helper(n, shift,filter=filter,report_accuracy=report_accuracy,target=target,v0=v0,sort=sort,ncv=ncv)
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
    def actions_before_eigen_solve(self,force_reassign_eqs:bool=False,must_not_renumber:bool=False):
        """Prepare the mode-dependent state of an eigensolve.

        ``must_not_renumber`` is set while a bifurcation tracker is installed: the tracker cached the
        base equation count and pushed dof pointers built against the current numbering, so a
        reapply_boundary_conditions() here would leave it describing a problem that no longer exists.
        The mode is then refused rather than accommodated -- see the snapshot dance below.
        """
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

        # _before_eigen_solve does not merely REPORT that a renumbering is needed: AxisymmetryBC has
        # already deactivated the strong axis conditions by the time it answers. When we are not
        # allowed to renumber we therefore have to put the flags back before refusing. This is not
        # tidiness: without it the problem is left with axis conditions released while the equation
        # numbering still has them pinned, and the NEXT eigensolve aborts (SIGABRT) rather than
        # returning anything wrong. Removing the restore below fails tests/test_eigen_during_tracking.py
        # ::test_refusals with exactly that crash.
        snapshot=self._dirichlet_activation_snapshot() if must_not_renumber else None
        must_reassign_eqs = self._equation_system._before_eigen_solve(self.get_eigen_solver(),eigen_m,eigen_k)
        #print("MUST REASSIGN IS",must_reassign_eqs,eigen_m,eigen_k)
        #exit()
        if must_not_renumber and (must_reassign_eqs or force_reassign_eqs):
            assert snapshot is not None
            self._restore_dirichlet_activation(snapshot)
            mode="m="+str(eigen_m) if eigen_m is not None else ("k="+str(eigen_k) if eigen_k is not None else "the base mode")
            raise RuntimeError("The eigenproblem for "+mode+" needs the equations to be renumbered (it changes which boundary conditions are strongly enforced), which is not possible while bifurcation tracking is active -- the tracker's augmented dof vector is built against the current numbering. "
                               "A non-axisymmetric (m!=0 or k!=0) eigenproblem is available while tracking an azimuthal or cartesian_normal_mode bifurcation, where the axis conditions are already released; while tracking a fold, Hopf or pitchfork they are not. "
                               "Currently tracking: '"+self.get_bifurcation_tracking_mode()+"'. Deactivate the tracking to solve this eigenproblem.")
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

    def solve(self,*,spatial_adapt:int=0,timestep:ExpressionNumOrNone | list[ExpressionNumOrNone] | tuple[ExpressionNumOrNone,...]=None,shift_values:bool=True,temporal_error:float | None=None,max_newton_iterations:int | None=None,newton_relaxation_factor:float | None=None,suppress_resolve_after_adapt:bool=False,newton_solver_tolerance:float | None=None,do_not_set_IC:bool=False,globally_convergent_newton:bool=False)->ExpressionOrNum: #,continuation=None)
        """
        Solves the problem stationary, unless a timestep is given. In that case, the time step is taken.

        Parameters:
            spatial_adapt (int): The level of spatial adaptation. Default is 0.
            timestep (Union[ExpressionNumOrNone, List[ExpressionNumOrNone], Tuple[ExpressionNumOrNone,...]]): The time step(s) for the transient solve. Can be a single value or a list/tuple of values, which are then taken one after the other. Default is None, meaning stationary solve without advancing in time.
            shift_values (bool): Whether to shift the values during the solve, i.e. shifting the history value buffer. Default is True.
            temporal_error (Optional[float]): The temporal error for adaptive time stepping. Default is None.
            max_newton_iterations (Optional[int]): Override the maximum number of Newton iterations. Default is None.
            newton_relaxation_factor (Optional[float]): Override the relaxation factor for the Newton solver. Default is None.
            suppress_resolve_after_adapt (bool): Whether to suppress resolving after adaptation. Default is False.
            newton_solver_tolerance (Optional[float]): Override the tolerance for the Newton solver. Default is None.
            do_not_set_IC (bool): Whether to not set the initial condition in the first call. Default is False.
            globally_convergent_newton (bool): Whether to use globally convergent Newton solver. Default is False.

        Returns:
            ExpressionOrNum: The current time after solving.
        """
                
        self._bifurcation_reactivation_after_adaptation=None
        if isinstance(timestep,(list,tuple)):
            lastres:ExpressionOrNum=0
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
            assert not isinstance(TSCALE,str)
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
            self._solve_with_adapt_recovery(spatial_adapt,False,True,
                                            lambda adapt_level,_shift: self.steady_newton_solve(adapt_level))
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
                self._suggested_next_dt=desired_dt
                self.actions_after_transient_solve()
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
                    self._solve_with_adapt_recovery(spatial_adapt,True,shift_values,
                        lambda adapt_level,shift: self.unsteady_newton_solve(timestep,adapt_level,self._first_step,shift))
                else:
                    desired_dt=self._solve_with_adapt_recovery(spatial_adapt,True,shift_values,
                        lambda adapt_level,shift: self.doubly_adaptive_unsteady_newton_solve(timestep,temporal_error,adapt_level,int(suppress_resolve_after_adapt),self._first_step,shift))
                self._first_step=False
                self._suggested_next_dt=desired_dt
                self.actions_after_transient_solve()
                self._override_for_this_solve(**oldsettings)
                self.timestepper.increment_num_unsteady_steps_done()
                return desired_dt*TSCALE


    def run(self, endtime:ExpressionOrNum, timestep:ExpressionNumOrNone=None,*, outstep:ExpressionNumOrNone | bool=None, numouts:int | None=None, out_initially:bool | None=None,
            temporal_error:None | float=None, outstep_relative_to_zero:bool=True,spatial_adapt:int=0,startstep:ExpressionNumOrNone=None,maxstep:ExpressionNumOrNone=None,newton_solver_tolerance:None | float=None,do_not_set_IC:bool | Literal["auto"]="auto",globally_convergent_newton:bool=False,max_newton_iterations:None | int=None,starttime:ExpressionNumOrNone=None,suppress_resolve_after_adapt=False,max_newton_to_increase_time_step:int | None=None)->ExpressionOrNum:
        """
        Run the problem for a specified duration, potential with output calls and temporal and/or spatial adaptivity.
        All time quantities must be given in dimensional units, e.g. ``second``, if you use e.g. :py:meth:`~Problem.set_scaling` with e.g. ``temporal=1*second`` for a dimensional problem.

        Args:
            endtime: The end time of the simulation.
            timestep: The time step size. If not specified, it will be determined automatically, e.g. by the outstep.
            outstep: The time interval between outputs. If set to True, outputs will be generated at each time step. If set to False, no outputs will be generated. If not specified, it defaults to the value of `timestep`, except with temporal adaptivity (`temporal_error`), where it defaults to `True`. Note that time steps are clamped to hit the output times exactly, so a finite `outstep` is also an upper bound for the adaptive time step.
            numouts: The number of outputs to generate. If specified, it will override the value of `outstep`.
            out_initially: Whether to generate an output at the initial time. If not specified, it will be set to `True` if `outstep` is not `False`, otherwise it will be set to `False`.
            temporal_error: The temporal error tolerance. If specified, it will be used to control the time step size.
            outstep_relative_to_zero: Whether the `outstep` is relative to the initial time or the current time. If set to `True`, the `outstep` will be relative to the initial time. If set to `False`, the `outstep` will be relative to the current time.
            spatial_adapt: The level of spatial adaptation. If specified, it will be used to control the spatial refinement level.
            startstep: The time step size at the start of the simulation (for temporal adaptivity). If specified, it will override the value of `timestep`. It is ignored when this run statement is resumed by ``--runmode continue``, unless :py:attr:`~Problem.use_state_dt_when_continue` is set to ``False``.
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
        if do_not_set_IC=="auto":
            do_not_set_IC=self.is_initialised()
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
        keep_state_dt=False
        _tfactor,_tunit=assert_dimensional_value(starttime-endtime)
        if _tfactor>=0.0:
            # Deliberately returns without clearing _continue_dt_pending: a run statement the state
            # file has already passed is not the one we are resuming into.
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
            self._first_step=False # TODO This would be better stored in the state file so that a solve from state_000000 will still have it true
            # This is the run statement being resumed: the dt the run had worked its way up to is the
            # one to carry on with. Without this, startstep below would throw it away and a continued
            # adaptive run would restart from the tiny initial step after every interruption.
            keep_state_dt=self.use_state_dt_when_continue and self._continue_dt_pending
            self._continue_dt_pending=False
            if keep_state_dt and self._suggested_next_dt is not None:
                # Not dt(0): that is the step already taken, and resuming with it would take one extra
                # step at the old size before the estimator catches up again, so the continued run
                # would no longer step where the uninterrupted one did. The suggestion is the step the
                # interrupted run was about to take, so picking it up here reproduces it exactly.
                timestep = self._suggested_next_dt * self.get_scaling("temporal")

        #TODO Further checking for the end time
        single_step_desired=False
        if timestep is None:
            if not self.is_initialised():
                self.initialise()
                timestep = self.get_scaling("temporal")
            else:
                timestep = self.timestepper.time_pt().dt(0) * self.get_scaling("temporal")
            _tdiff,_tunit=assert_dimensional_value(starttime+timestep-endtime)
            if _tdiff>0:
                timestep=endtime-starttime
                single_step_desired=True
        if startstep is not None and not keep_state_dt:
            timestep=startstep


        if outstep is None:
            if temporal_error is not None and numouts is None:
                # Every step is clamped to land exactly on the next output time, so dt can never
                # exceed outstep. Defaulting to outstep=timestep would therefore turn the *initial*
                # step (usually a deliberately tiny startstep) into a permanent cap and make
                # temporal_error look completely ineffective. Output after each accepted step instead.
                outstep = True
            else:
                outstep = timestep
        TS = self.get_scaling("temporal")
        ndouttimes:NPFloatArray | None = None 
        currentdt:ExpressionOrNum
        if not isinstance(outstep, bool):
            currentdt = min(float(timestep / TS), float(outstep / TS)) * TS
            if outstep_relative_to_zero:
                dtout:ExpressionOrNum
                if numouts:
                    dtout = (endtime - starttime) / numouts
                    soffs = math.ceil(float(starttime / dtout)) * dtout
                    endout = soffs + numouts * dtout
                    ndouttimes = numpy.linspace(float(soffs / TS), float(endout / TS), num=numouts + 1) #type:ignore
                else:
                    dtout=outstep
                    numouts=int(float((endtime - starttime)/dtout))
                    # Absolute multiples of dtout, not a linspace anchored at the current time. That is
                    # what outstep_relative_to_zero means, and it makes the grid independent of where a
                    # run was resumed: since the time steps are clamped onto this grid, a linspace from
                    # the resume time put the output instants an ulp beside the uninterrupted run's and
                    # the continued run stopped being a bit-for-bit reproduction of it.
                    k0 = math.ceil(float(starttime / dtout))
                    ndouttimes = (numpy.arange(numouts + 1) + k0) * float(dtout / TS) #type:ignore

            else:
                if numouts:
                    ndouttimes = numpy.linspace(float((starttime) / TS), float(endtime / TS), num=numouts + 1,endpoint=True) #type:ignore
                else:
                    raise RuntimeError("TODO")
            outcntvalue=float((ndouttimes[-1]-starttime/ TS) )/(numouts)+ ndouttimes[-1]
            ndouttimes=numpy.hstack([ndouttimes,[outcntvalue]]) #type:ignore
        else:
            currentdt = timestep

        # Applies to both branches: maxstep is the maximum step, and with outstep=True the clamp used
        # to be missing here entirely. That went unnoticed while the first step of a run was always a
        # small startstep, but a run resumed from its state file starts from the dt it had reached,
        # which is routinely at maxstep and whose successor the estimator wants four times larger.
        if maxstep is not None:
            currentdt=min(float(currentdt/TS),float(maxstep/TS))*TS
        if keep_state_dt:
            # The end-of-iteration clamps below shorten the last step so it lands on endtime. The
            # interrupted run applied them to this very dt after the step whose state we loaded, so
            # repeat them here rather than letting the resumed run overshoot where the original did not.
            remaining=float(endtime/TS)-self.get_current_time(as_float=True, dimensional=False)
            if remaining<float(currentdt/TS):
                currentdt=1.00001*remaining*TS

        nextdt_was_clamped_for_output:ExpressionNumOrNone=None # When clamping a time step to hit the next output dt, enlarge it afterwards
        first_step=True
        while self.get_current_time(as_float=True, dimensional=False) < float(endtime / TS):
            if self._abort_current_run:
                self._abort_current_run=False
                return currentdt
            tnd = self.get_current_time(as_float=True, dimensional=False)
            if ndouttimes is not None:
                # Check if the current timestep would exceed the next output
                currind:int = numpy.nonzero(ndouttimes <= tnd)[0] #type:ignore
                currind = -1 if len(currind) == 0 else currind[-1] #type:ignore
                nextndout = ndouttimes[currind + 1] #type:ignore
                if tnd + float(currentdt / TS) * 1.01 > nextndout:
                    currentdt = (nextndout - tnd) * TS

            self._in_transient_newton_solve=True
            nextdt = self.solve(timestep=currentdt, temporal_error=temporal_error,spatial_adapt=spatial_adapt,newton_solver_tolerance=newton_solver_tolerance,do_not_set_IC=do_not_set_IC,globally_convergent_newton=globally_convergent_newton,max_newton_iterations=max_newton_iterations,suppress_resolve_after_adapt=suppress_resolve_after_adapt)
            if first_step and float(nextdt/currentdt)>=1.0-1e-14:
                if single_step_desired:
                    test=self.get_current_time(as_float=True, dimensional=False) - float(endtime / TS)
                    if test>-1e-14:
                        self.set_current_time(float(endtime/TS),dimensional=False) # Will stop the run loop for sure
            else:
                single_step_desired=False
                
            first_step=False
            self._in_transient_newton_solve=False
            if max_newton_to_increase_time_step is not None and float(nextdt/TS)>float(currentdt/TS*1.00001):
                last_res=self.get_last_residual_convergence()
                if len(last_res)>max_newton_to_increase_time_step:
                    print("Do not increase time step, since we used too many iterations")
                    nextdt=currentdt

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
                    # Not an output time, so the outputs marked output_every_step still get their line
                    # here. Guarded by the else branch rather than done unconditionally: on an output
                    # time the full output() above has already written them, and they would be doubled.
                    self.output_every_step_outputs()
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
            remaining=float(endtime/TS)-self.get_current_time(as_float=True, dimensional=False)
            # A rounding sliver must not become a time step. Whenever a step is REJECTED -- by
            # before_newton_convergence_check, by an inverted element, by a solver failure -- the
            # accepted dt is not the requested one, so the accumulated time misses endtime by an ulp
            # or two. The clamp below would then ask for a dt of ~1e-16, which the Newton solver
            # cannot converge (1/dt swamps the Jacobian); oomph-lib halves it, again, until it falls
            # under Problem::Minimum_dt and kills the run with "Tried to reduce dt to 5.55e-17 which
            # is less than the minimum dt" -- at the very end of a simulation that was otherwise
            # finished. A gap eight orders of magnitude below the step we were about to take is
            # rounding, not physics, so treat endtime as reached. No legitimate final step is lost:
            # for one to be skipped, dt would have to have been planned 1e8 times larger than the
            # time actually left.
            if remaining<1e-8*float(currentdt/TS):
                break
            if remaining<float(currentdt/TS):
                currentdt=1.00001*remaining*TS

        self._nondim_time_after_last_run_statement=float(self.get_current_time()/TS)
        return currentdt
    

    def deflated_solve_by_eigenperturbation(self, eigenindex:int=0, keep_deflation_active:bool=False, perturbation_factor:float=1,deflation_alpha:float=0.1,deflation_power:int=2,*, max_newton_iterations:int | None=None, newton_relaxation_factor:float | None=None, newton_solver_tolerance:float | None=None, globally_convergent_newton:bool=False):        
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

        from .bifurcation_tools import DeflationAssemblyHandler        
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
            

    def iterate_over_multiple_solutions_by_deflation(self,deflation_alpha:float=0.1,deflation_p:int=2,perturbation_amplitude:float=0.5,max_newton_iterations:int | None=None,newton_relaxation_factor:float | None=None,use_eigenperturbation:bool=False,skip_initial_solution:bool=False,num_random_tries:int=1,keep_deflation_operator_active:bool=False)-> Generator[NPFloatArray,None,None]:
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
            
        from .bifurcation_tools import DeflationAssemblyHandler
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
                

    def deflated_continuation(self,deflation_alpha:float=0.1,deflation_p:int=2,perturbation_amplitude:float=0.5,max_newton_iterations:int | None=None,newton_relaxation_factor:float | None=None,use_eigenperturbation:bool=False,skip_initial_solution:bool=False,num_random_tries:int=1,max_branches:int | None=None,branch_continue_iterations:int=10,**param_range):
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
        from .bifurcation_tools import DeflationAssemblyHandler
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
        assert rang is not None # rang is always set together with param in the loop above
        if param not in self.get_global_parameter_names():
            raise RuntimeError("Please specify a parameter that is defined in the problem")
        param_obj=self.get_global_parameter(param)
        active_branches={} # Branch index -> current dofs
        
        # Find the first solutions
        self.go_to_param({param:rang.pop(0)})
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
            branches_to_remove:list[int]=[]
            branches_to_add:dict[int,NPFloatArray]={}
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

    def _check_distributed_remeshing_scope(self,remeshers:list["RemesherBase"],num_adapt:int | None)->None:
        """Refuse what the distributed remeshing path cannot do, *before* anything is rebuilt.

        Both limitations below are properties of the request, so they are known here - and they have
        to be raised here, because force_remesh() has no way back once it has started replacing
        meshes: the problem is then left half rebuilt, with the new meshes installed and the
        superseded ones not yet torn down, which does not even survive interpreter shutdown.

        Unanimous by construction: every rank has the same meshes, the same remeshers and the same
        argument.
        """
        if not self.is_distributed() or get_mpi_nproc()<=1:
            return
        rebuilt={name for name,m in self._meshdict.items()
                 if not isinstance(m,ODEStorageMesh) and any(r.template.has_domain(name) for r in remeshers)}
        untouched=sorted(name for name,m in self._meshdict.items()
                         if not isinstance(m,ODEStorageMesh) and name not in rebuilt)
        if untouched:
            # oomph's Mesh::distribute() partitions a whole mesh; it is not a re-partitioning of one
            # that is already split. A domain left alone stays partitioned from before, and there is
            # no way to fit the rebuilt ones into that partition. See _redistribute_after_remeshing.
            raise RuntimeError("Remeshing only some domains of a distributed (--distribute) problem is not "
                               "supported: the rebuilt meshes are replicated on every rank and have to be "
                               "partitioned again, but "+", ".join(untouched)+" would still be partitioned "
                               "from before, and oomph-lib cannot distribute a mesh twice. Remesh all domains "
                               "at once, or run without --distribute. See dev_docs/distributed_remeshing.md.")
        if num_adapt is not None and num_adapt>0:
            # Never silently dropped: an explicit num_adapt is a request about the resulting mesh.
            raise RuntimeError("force_remesh(num_adapt="+str(num_adapt)+") is not supported on a distributed "
                               "(--distribute) problem: adapting the new mesh leaves it non-uniformly refined, "
                               "and oomph-lib only distributes uniformly refined meshes, so the remeshed problem "
                               "could not be partitioned again. Pass num_adapt=0, or run without --distribute. "
                               "See dev_docs/distributed_remeshing.md.")

    def _remesh_adaption_steps(self,num_adapt:int | None)->int:
        """How many adaption rounds the remesh should run. Zero on a distributed problem.

        Adapting would leave the new mesh non-uniformly refined, which Problem::distribute() refuses,
        so the re-distribution afterwards would be the thing that fails. An explicit num_adapt>0 has
        already been refused in _check_distributed_remeshing_scope(); this is the default, derived
        from max_refinement_level, which the user did not ask for by name."""
        if num_adapt is None and self.is_distributed() and get_mpi_nproc()>1:
            # Deliberately not behind is_quiet(): it changes the mesh that comes out.
            print("NOTE: not adapting the remeshed mesh, since a distributed problem can only be "
                  "partitioned again while its meshes are uniformly refined "
                  "(dev_docs/distributed_remeshing.md, stage 2)")
            return 0
        return self.max_refinement_level if num_adapt is None else num_adapt

    def _refuse_distributed_remeshing(self,remeshers:list["RemesherBase"],interpolator:type["BaseMeshToMeshInterpolator"])->None:
        """Stop a remesh of a distributed problem whose ingredients do not all support one.

        Remeshing under ``--distribute`` works for the combination the campaign in
        dev_docs/distributed_remeshing.md has built: a remesher that rebuilds the geometry from
        globally merged boundary data, and an interpolator that pools across the ranks what each of
        them could place. Anything else does not merely fail, it succeeds wrongly - it rebuilds the
        mesh from one rank's partition, or fills a third of the new mesh with a nearest-node blend
        over local nodes - so it is refused by name rather than left to produce a plausible answer.

        :py:attr:`experimental_distributed_remeshing` bypasses this, which is how the remaining
        stages get developed.

        Collective by construction rather than by communication: is_distributed(), the set of
        remeshers, the interpolator and the mesh structure are the same on every rank, so either all
        of them raise or none does.
        """
        if not self.is_distributed() or self.experimental_distributed_remeshing:
            return
        reasons:list[str]=[]
        for r in remeshers:
            if r.distributed_limitation is not None:
                reason=type(r).__name__+": "+r.distributed_limitation
                if reason not in reasons:
                    reasons.append(reason)
        interpolator_limitation=interpolator.distributed_limitation
        if interpolator_limitation is not None:
            reasons.append(interpolator.__name__+": "+interpolator_limitation)
        if not reasons:
            return
        raise RuntimeError("Remeshing is not supported on a distributed (--distribute) problem in this "
                           "configuration:\n"+"\n".join("  "+r for r in reasons)+"\n"+
                           "Run without --distribute, or set Problem.experimental_distributed_remeshing=True to "
                           "run it anyway and get whatever it produces. See dev_docs/distributed_remeshing.md.")

    def _reregister_refinement_directives(self,new_meshes:dict[str,"MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d"])->None:
        """Re-state the declarative refinement criteria on meshes that have just been replaced.

        :py:class:`~pyoomph.equations.generic.RefineToLevel` and
        :py:class:`~pyoomph.equations.additional.RefineMaxElementSize` register their criterion on the
        mesh object, whereas a replacement (remeshing, or loading a state file whose mesh template
        differs) reuses the compiled code and only swaps the mesh - so ``after_compilation`` never runs
        again and every such criterion, bulk and interface alike, was simply gone from the first remesh
        on: the adaption that follows found nothing to refine and the mesh came back at its base level.

        Only the domains that were actually replaced: their meshes are new and carry no directives yet,
        while an untouched domain still holds its own and would collect a duplicate per remesh. Call
        only after ``rebuild_global_mesh_from_list()``, which is what points every code generator -
        the interface ones included - at its new mesh; ``register_refinement_directives`` reads
        ``codegen._mesh``."""
        for _name,newmesh in new_meshes.items():
            newmesh.get_eqtree()._register_refinement_directives()

    def _redistribute_after_remeshing(self)->None:
        """Partition the rebuilt meshes again, since remeshing replaces them whole and replicated.

        A remesh regenerates the mesh from its geometry, which every rank does in full - the same
        state the problem is in at startup, just before its first :py:meth:`distribute`. Left like
        that, every rank holds the entire mesh while oomph-lib still has ``Problem_has_been_distributed``
        set, so the equation numbering counts every locally owned node once per rank and ``ndof``
        comes out ``nproc`` times too large. So do what startup does, at the point where the meshes
        are whole and the transfer of the old solution is finished. See
        dev_docs/distributed_remeshing.md, stage 2.

        Both preconditions oomph-lib puts on distribute() - a whole mesh, uniformly refined - were
        established before any mesh was touched, by _check_distributed_remeshing_scope() and
        _remesh_adaption_steps(). They are asserted rather than checked here: there is no way back
        from this point, so a refusal would leave the problem half rebuilt instead of helping.
        """
        if not self.is_distributed() or get_mpi_nproc()<=1:
            return
        meshes=[m for m in self._meshdict.values() if not isinstance(m,ODEStorageMesh)]
        assert not any(m.is_mesh_distributed() for m in meshes), \
            "a mesh survived the remesh still partitioned, which _check_distributed_remeshing_scope should have refused"
        assert all(self._is_uniformly_refined(m) for m in meshes), \
            "the remeshed mesh came out non-uniformly refined, which _remesh_adaption_steps should have prevented"

        # The base element numbers are what state files address elements and nodes by, and they can
        # only be assigned while the mesh is still whole (Mesh::assign_global_base_element_indices).
        # Once the meshes below are partitioned, the lazy assignment in BaseMesh._define_state_file
        # deliberately does not step in any more, so this is the only chance.
        for m in meshes:
            m.assign_global_base_element_indices()

        if not self.is_quiet():
            print("REDISTRIBUTING THE REMESHED PROBLEM")
        self.actions_before_distribute()
        self.distribute()
        self.actions_after_distribute()

    def _is_uniformly_refined(self,mesh:"AnySpatialMesh")->bool:
        """Whether every element of ``mesh`` sits at the same refinement level.

        Asked of the local mesh, which is the whole mesh here - _redistribute_after_remeshing() only
        calls it on meshes it has just established are replicated."""
        if not mesh.refinement_possible():
            return True
        levels={e.refinement_level() for e in mesh.elements()}
        return len(levels)<=1

    def force_remesh(self, only_domains:set[MeshTemplate] | None=None, num_adapt:int | None=None,interpolator:type["BaseMeshToMeshInterpolator"] | None=None):
        if interpolator is None:
            interpolator=self.mesh_interpolator
        if self._debug_remeshing:
            # The state on the OLD mesh, written before anything is rebuilt.
            if not self.is_quiet():
                print("Writing an output BEFORE remeshing (_debug_remeshing)")
            self.output()
        remeshers:list["RemesherBase"] = []
        if only_domains is not None:
            for t in only_domains:
                if t.remesher is not None:
                    remeshers.append(t.remesher)
        else:
            # Without a given selection, we remesh everything that can actually give a different mesh. The latter
            # matters since every MeshedMeshTemplate carries a remesher by default: a define_geometry that does not
            # react on remeshing at all would just be rebuilt identically here (see _remeshing_can_change_the_mesh).
            for t in self._meshtemplate_list:
                if t.remesher is not None and t._remeshing_can_change_the_mesh():
                    remeshers.append(t.remesher)

        if len(remeshers)==0:
            return
        # Both deliberately after the "is there anything to remesh at all" test, so that a
        # remesh_if_necessary() which finds nothing to do still returns quietly when distributed,
        # and both before the first mesh is touched - see _check_distributed_remeshing_scope.
        self._refuse_distributed_remeshing(remeshers,interpolator)
        self._check_distributed_remeshing_scope(remeshers,num_adapt)
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

        

        new_meshes:dict[str,MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d] = {}
        old_meshes:dict[str,MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d] = {}

        # Now remove all interfaces and so on from the previous meshes
        for name, mesh in self._meshdict.items():
            if isinstance(mesh, ODEStorageMesh): continue
            for r in remeshers:
                if name in r._old_meshes.keys():                      
                    # Clean up
                    # for iname,imesh in mesh._interfacemeshes.items():
                    #    imesh.clear_before_adapt()
                    print("Creating new mesh for ",name,r,r.get_new_template())
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

            for tree_depth in range(3):
                newmesh._generate_interface_elements(tree_depth)

            newmesh._tracers=oldmesh._tracers
            for _,tracercoll in newmesh._tracers.items():
                tracercoll._set_mesh(newmesh)
            # oldmesh's underlying C++ mesh is still needed further below (read by
            # InternalInterpolator across possibly several adaptive interpolation rounds) - do
            # NOT force-destroy it here; it is torn down at the very end of this method instead,
            # once nothing needs it any more.

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
        for mt in self._meshtemplate_list:
            mt._connect_opposite_elements(self._equation_system) 

        self.rebuild_global_mesh_from_list(rebuild=True)
        self.reapply_boundary_conditions()
        # Before the adaption loop below: that is what the criteria are for.
        self._reregister_refinement_directives(new_meshes)

        interpolators:dict[str,"BaseMeshToMeshInterpolator"]={}
        # Apply the interpolation on each mesh: First on the boundaries and then down to the bulk mesh
        def perform_interpolation():
            for _, interp in interpolators.items(): 
                interp.interpolate() 
            if self._debug_remeshing:
                # And the state on the NEW mesh, so the two outputs bracket the transfer exactly.
                if not self.is_quiet():
                    print("Writing an output AFTER remeshing (_debug_remeshing)")
                self.output()


        if has_continuation_data:
            print("RESTORING CONTINUATION DATA")
            dof_deriv=self.get_history_dofs(5)
            dof_current=self.get_history_dofs(6)
            self._update_dof_vectors_for_continuation(dof_deriv,dof_current)

        num_adapt = self._remesh_adaption_steps(num_adapt)


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

        # Before actions_after_remeshing(), so that user code sees the mesh in its final state - which
        # on a distributed problem means partitioned, exactly as it is in every other callback.
        self._redistribute_after_remeshing()

        self.actions_after_remeshing()
        for r in remeshers:
            r.actions_after_remeshing()
        self.invalidate_cached_mesh_data()

        # Now, and only now, is every old (superseded) mesh truly no longer needed - the
        # adaptive interpolation loop above may read from it (via InternalInterpolator) across
        # multiple rounds, so destroying it any earlier (e.g. right after its _codegen/_eqtree
        # were transferred to the new mesh, above) crashes. Without this, oldmesh's underlying
        # C++ object (and its elements/nodes) would remain permanently pinned alive by
        # nb::keep_alive on the C++ side for the rest of this Problem's lifetime, once per
        # remesh event, since dropping it from self._meshdict earlier does not revoke that.
        # See _destroy_superseded_mesh() for why oldmesh's _eqtree/_codegen (and, recursively,
        # the same for its interface meshes) must NOT be touched here, unlike Problem.release()'s
        # teardown - and why its _templatemesh, in contrast, must be.
        for name, oldmesh in old_meshes.items():
            _destroy_superseded_mesh(oldmesh)
        
        



    def _define_state_header(self,state:DumpFile)->str:
        """Read or write the header of a state file: what it is, which format version, and how it is sharded.

        Shared by define_state_file and _get_time_of_state_file, which peeks at the first entries
        without reading the rest - so the order of these entries is part of the format and the two must
        not drift apart. Returns the version."""
        state.string_data(lambda: self._dump_header, lambda s: state.assert_equal(s, self._dump_header))
        state.version = state.string_data(lambda: self._dump_version, lambda s: state.assert_leq(s, self._dump_version))
        if state.save or state.version_at_least(0,1,1):
            sharding=state.string_data(lambda: state.sharding, lambda s: s)
            if not state.save:
                if sharding!="global":
                    # Deliberately explicit: a sharded state is a set of files, and reading one of them
                    # as if it were the whole problem would not fail on the spot, it would restore a
                    # fraction of the mesh (see dev_docs/distributed_state_files.md §7)
                    raise RuntimeError("The state file '"+state.fname+"' says its mesh data is sharded ('"+sharding+"'), and reading sharded state files is not supported yet. Only 'global' files, which hold the whole problem, can be read")
                state.sharding=sharding
        return state.version

    def _get_time_of_state_file(self,fname:str):
        state=DumpFile(fname,False)
        self._define_state_header(state)
        # Current time
        t=state.float_data(lambda: self.get_current_time(dimensional=True, as_float=True),lambda t: t)
        s = state.int_data(lambda: self._output_step, lambda s: s)
        state.close()
        return t,s

    # This function defines the state file, i.e. storing or reading all relevant information of the current status of the simulations
    def define_state_file(self, state:DumpFile,ignore_loading_eigendata:bool=False,ignore_continuation_data=False,additional_info={}):
        # The header comes first and in that order, because _get_time_of_state_file peeks at the entries
        # up to the output step without reading the rest. Both go through _define_state_header.
        self._define_state_header(state)

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

        new_meshes:dict[str,MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d] = {}
        old_meshes:dict[str,MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d]= {}
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
                for tree_depth in range(3):
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
                for mt in self._meshtemplate_list:
                    mt._connect_opposite_elements(self._equation_system) 

                self.rebuild_global_mesh_from_list(rebuild=True)
                self.reapply_boundary_conditions()
                # Same reason as in force_remesh(): these meshes were built without recompiling, so
                # nothing has stated the refinement criteria on them yet.
                self._reregister_refinement_directives(new_meshes)

                # These meshes are brand new: they were built from the state file's own template,
                # long after initialise() stripped the macro elements off the meshes it had built
                # (remove_macro_elements_after_initial_adaption). So they arrive carrying macro
                # elements again, and on a moving mesh those are actively wrong - the macro element
                # freezes the geometry of the template it was built from, while the nodal positions
                # are dofs that have since moved. Adaptive refinement positions every new node
                # through father->get_x(), which goes through the macro map whenever one is
                # attached, so refining after such a load snapped new nodes back onto the geometry
                # the template was generated with. force_remesh() drops them for exactly this
                # reason; do the same here. mode="auto" keeps them on meshes whose coordinates are
                # not dofs, where the macro map is still the right (and only) source of the curved
                # geometry between the nodes.
                self.remove_macro_elements()

            # Loading a state whose mesh template differs from the current one replaces the
            # meshes wholesale, exactly like remeshing does (see force_remesh()), so the
            # superseded ones must be torn down here for the same reasons - and they are of no
            # further use: unlike after a remesh, nothing is interpolated from them (all dof
            # values come from the state file), and previous_mesh above is only read inside
            # MeshFromTemplate.__init__(). Without this they stayed pinned alive by the
            # unrevokable nb::keep_alive from the Problem for the rest of the session, which not
            # only leaked their nodes/elements (plus the whole Problem, via the
            # _templatemesh -> remesher -> Problem cycle described in _destroy_superseded_mesh())
            # but crashed at interpreter shutdown: their elements were then destructed after
            # release() had already dlclose()'d the equation code they point into.
            for _name, oldmesh in old_meshes.items():
                _destroy_superseded_mesh(oldmesh)

        # Time stepper dts
        time = self.timestepper.time_pt()
        ndt = state.int_data(lambda: time.ndt(), lambda ndt: state.assert_equal(ndt, time.ndt()))
        for dt in range(ndt):
            _dtval=state.float_data(lambda: time.dt(dt), lambda v: time.set_dt(dt, v))

        if not state.save:
            self.timestepper.set_weights()

        state.int_data(lambda: self.timestepper.get_num_unsteady_steps_done(),lambda t: self.timestepper.set_num_unsteady_steps_done(t))

        # The step the adaptive time stepper asked for next, which is not recoverable from the dts
        # above (those are the steps already taken). Written as a NaN when there is none yet, so the
        # entry keeps a fixed size. Old files have no such entry at all, hence the version guard.
        # (state.version is the version being written when saving, so this one condition covers both
        # directions - unlike the "state.save or ..." form used above, lowering _dump_version to write
        # an old-format file on purpose stays consistent between writer and reader.)
        if state.version_at_least(0,1,2):
            _next_dt=state.float_data(lambda: float("nan") if self._suggested_next_dt is None else self._suggested_next_dt, lambda v: v)
            if not state.save:
                self._suggested_next_dt=None if math.isnan(_next_dt) else _next_dt
        elif not state.save:
            # An older file has no suggestion in it, and whatever this session is holding belongs to a
            # different point in time. Clear it, so run() falls back to the dt that was last taken.
            self._suggested_next_dt=None

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
        # Interface and skeleton element data, i.e. everything a facet field owns: DL/D0 and the nodal
        # DG spaces alike live in the interface element's own internal Data, and no other block of the
        # file holds them. Without this the file could only reproduce the bulk state and would refit
        # whatever the LOADING process happened to have on its facets.
        #
        # On loading this only READS: the interface elements do not exist in their final form yet -
        # they are rebuilt from the loaded bulk mesh afterwards, in actions_after_adapt() - so the
        # values are parked and applied by _apply_interface_states() once they do.
        if state.version_at_least(0,1,4):
            from ..meshes.meshstate import save_interface_state, read_interface_state
            imeshes = sorted(self._interfacemeshes, key=lambda m: m.get_full_name())
            n_if = len(imeshes)
            n_if = state.int_data(lambda: n_if, lambda n: n) #type:ignore
            if state.save:
                for _m in imeshes:
                    state.string_data(lambda _m=_m: _m.get_full_name(), lambda s: s)
                    save_interface_state(_m, state)
            else:
                self._pending_interface_states = {}
                for _ in range(int(n_if)):
                    _name = state.string_data(lambda: "", lambda s: s)
                    self._pending_interface_states[_name] = read_interface_state(state)
        # Global params
        gpars = list(sorted(self.get_global_parameter_names()))
        numgpars = len(gpars)
        numgpars = state.int_data(lambda: numgpars, lambda n: state.assert_equal(n, numgpars)) #type:ignore
        for ngpar in range(numgpars):
            gparname = state.string_data(lambda: gpars[ngpar], lambda s: state.assert_equal(s, gpars[ngpar]))
            gp = self.get_global_parameter(gparname)
            gp.value = state.float_data(lambda: gp.value, lambda v: v)




        # Eigendata if desired
        if state.save and self.is_distributed() and (self.eigen_data_in_states is not False or self.continuation_data_in_states is not False):
            # Both are indexed by dof number, which distribute() permutes AND splits over the ranks, so
            # they cannot be written in a partition-independent way the way the mesh can. The way out is
            # to key them by the node/element storage they belong to, which is designed but not built;
            # until then this refuses rather than writing something that reads back scrambled.
            raise RuntimeError("Eigen and continuation data cannot be stored in the state file of a distributed problem yet. Set eigen_data_in_states=False and continuation_data_in_states=False, see dev_docs/distributed_state_files.md")
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
            dofderiv=numpy.zeros((0,))
        has_contidata=state.int_data(lambda : write_conti,lambda n : n)
        if has_contidata:
            dofderiv=state.numpy_data(lambda  : numpy.array(dofderiv),lambda e:e)
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
                

        


    def save_state(self, fname:str | IO[bytes],relative_to_output:bool=False,quiet:bool=False)->None:
        """Write the full state of the problem to a file, so that it can be restored with load_state.

        ``fname`` may also be an open binary stream (e.g. an ``io.BytesIO``), which is what
        :py:meth:`_snapshot_state` uses to keep a state in memory instead of on disk.

        On a distributed problem this is **collective**: every rank contributes its part of the mesh,
        the data is merged into one partition-independent file (see dev_docs/distributed_state_files.md)
        and rank 0 writes it. The resulting file is the same one a serial run would write and can be
        read back on any number of processes.

        Writing to a FILE is collective under MPI in every case, distributed or not: no rank returns
        before the file is complete on disk, so a load_state of it right afterwards is safe. Writing
        to a stream is rank-local, since the stream is."""
        distributed=self.is_distributed()
        to_file=isinstance(fname,str)
        # Every rank holds the whole problem when it is not distributed, so one of them writing the
        # FILE is enough. The others used to return right here - and then raced ahead into a
        # load_state of that very file while rank 0 was still writing it: FileNotFoundError on one
        # rank, and the rest hanging in the next PETSc collective waiting for a rank that is gone
        # (Advanced_Linear_Dynamics/eigenbranch_continuation.py under mpirun -n 2). They wait at the
        # collective at the end of this method instead, which also hands them rank 0's failure.
        # A stream is per-rank private, so there is no redundant writer to skip - and a snapshot that
        # only rank 0 held could not be restored, since every rank has to read the whole state back.
        redundant_writer=(not distributed) and to_file and get_mpi_rank()>0

        error=None
        if not redundant_writer:
            if not self.is_quiet() and not quiet and get_mpi_rank()==0:
                print("Saving state ", fname)
            if relative_to_output:
                assert isinstance(fname,str), "relative_to_output makes no sense when writing to a stream"
                fname=os.path.join(self.get_output_directory(),fname)
            # On a DISTRIBUTED problem all ranks walk through define_state_file - that is where they
            # hand their part of the mesh over - but only rank 0 writes anything, the others send
            # their bytes to /dev/null, because the result is one merged, partition-independent
            # stream.
            #
            # When the problem is NOT distributed there is no merge: every rank holds the whole
            # problem and writes a complete stream of its own. Sending the others to /dev/null here
            # would be wrong for a stream - the redundant writers of a FILE are already out, so the
            # only ranks still here are ones that need their own copy. (They used to be dropped
            # unconditionally, which is why this line could assume rank 0.)
            writes_here = (get_mpi_rank()==0) or (not distributed)
            dump = DumpFile(fname if writes_here else os.devnull, True,compression_level=self.states_compression_level)
            try:
                self.define_state_file(dump)
                dump.write_footer("EOF_pyoomph")
                dump.close()
            except BaseException as e:
                error=e
        # Both of these are collective, and that is the point twice over: a rank that failed would
        # otherwise leave the others waiting in the next collective - the run would hang instead of
        # reporting the failure - and for a FILE they also pin every rank here until it is written.
        if distributed:
            # Every rank ran define_state_file, so any of them can be the one that failed.
            mpi_share_any_failure(error,context="writing the state file")
        elif to_file:
            # Only rank 0 wrote; the others skipped the section entirely and learn about it here.
            mpi_share_root_failure(error,context="writing the state file")
        elif error is not None:
            raise error # a per-rank private stream: nothing to agree on


    def load_state(self, fname:str | IO[bytes],ignore_outstep:bool=False,relative_to_output:bool=False,ignore_eigendata:bool=False,ignore_continuation_data:bool=False,additional_info:dict[Any,Any]={},quiet:bool=False):
        """Restore a state written by save_state. Every rank reads the whole file for itself.

        **Collective** under MPI, in two ways that both matter. The load itself already was - it
        rebuilds the global mesh and renumbers the equations - but the outcome was not: a rank that
        could not read the file unwound alone while the others went on into the next collective and
        waited there for good. So the outcome is agreed on here, and since that agreement is
        collective it is also the barrier that keeps a rank from overwriting or deleting the file
        while another one is still reading it."""
        error=None
        try:
            self._load_state(fname,ignore_outstep,relative_to_output,ignore_eigendata,ignore_continuation_data,additional_info,quiet)
        except BaseException as e:
            error=e
        mpi_share_any_failure(error,context="loading the state file "+(fname if isinstance(fname,str) else "<stream>"))
        return True

    def _apply_interface_states(self):
        """Push the interface/skeleton element data read from a state file onto the rebuilt meshes.

        Separate from reading because the elements are rebuilt in between; see define_state_file. An
        interface named in the file but absent now, or an element whose key is not in the file, is left
        alone: that is a mesh which has changed since the file was written, and the rebuild's own
        transfer is a better answer there than nothing."""
        pending = getattr(self, "_pending_interface_states", None)
        if not pending:
            return
        from ..meshes.meshstate import apply_interface_state
        try:
            for m in self._interfacemeshes:
                rec = pending.get(m.get_full_name())
                if rec is not None:
                    apply_interface_state(m, *rec)
        finally:
            self._pending_interface_states = None

    def _load_state(self, fname:str | IO[bytes],ignore_outstep:bool=False,relative_to_output:bool=False,ignore_eigendata:bool=False,ignore_continuation_data:bool=False,additional_info:dict[Any,Any]={},quiet:bool=False):
        if not self.is_initialised():
            self.initialise()
        # No guard for distributed problems: every rank reads the whole file and picks out the part of
        # the mesh it holds, halo copies included, which needs no communication at all.
        if relative_to_output:
            assert isinstance(fname,str), "relative_to_output makes no sense when reading from a stream"
            fname=os.path.join(self.get_output_directory(),fname)

        _pyoomph.set_interpolate_new_interface_dofs(False) # We may not interpolate the additional dofs on newly constructed interface nodes
        self.invalidate_cached_mesh_data()
        dump = DumpFile(fname, False)
        good=dump.check_footer("EOF_pyoomph")
        if not good:
            raise RuntimeError("Unsupported state file: "+dump.fname)
        for m in self._interfacemeshes:
            m.clear_before_adapt()
        oldoutstep=self._output_step
        self.define_state_file(dump,ignore_loading_eigendata=ignore_eigendata,ignore_continuation_data=ignore_continuation_data,additional_info=additional_info)
        self.invalidate_cached_mesh_data()
        if ignore_outstep:
            self._output_step=oldoutstep
        # A problem that was mid-transient when the state was written must know that it is mid-transient
        # now. Otherwise the next solve(timestep=...) takes the branch for the very first unsteady step:
        # it re-initialises dt, re-applies the initial condition and resets the step counter, so the
        # step is taken with the degraded first-order start instead of continuing the scheme. The state
        # is restored exactly either way - the difference only shows in the NEXT step, as an O(dt^2)
        # deviation from an uninterrupted run, and not at all on a problem that has reached a steady
        # state, which is what made it easy to miss.
        # Derived from the step counter rather than stored separately, so old state files behave too.
        if self.timestepper.get_num_unsteady_steps_done()>0:
            self._taken_already_an_unsteady_step=True
            self._first_step=False
            self._last_step_was_stationary=False
        if not quiet:
            print("State file "+dump.fname+" loaded")
        for m in self._interfacemeshes:
            m.clear_before_adapt()
        self.invalidate_cached_mesh_data()
        self.rebuild_global_mesh_from_list(rebuild=True)
        self.actions_after_adapt()
        # Now the interface elements exist again, so the file's facet values can go in. AFTER
        # actions_after_adapt on purpose: rebuild_after_adapt refits whatever this process was holding
        # onto the loaded geometry, and the file's values must overwrite that approximation, not race it.
        self._apply_interface_states()
        self.setup_pinning()
        self.reapply_boundary_conditions()
        self.invalidate_cached_mesh_data()
        dump.close()

        self.actions_after_remeshing() # Must call this to inform e.g. the outputters, that the mesh has changed!
        
        self.invalidate_cached_mesh_data()                
        if self._last_bc_setting=="eigen":
            last_eigenmodes_m=self.get_last_eigenmodes_m()
            if self._azimuthal_mode_param_m is not None and last_eigenmodes_m is not None and len(last_eigenmodes_m):
                self._azimuthal_mode_param_m.value=int(last_eigenmodes_m[0])
            self.actions_before_eigen_solve()
            if self._azimuthal_mode_param_m is not None:
                self._azimuthal_mode_param_m.value=0
        elif self._last_bc_setting=="transient":
            self.actions_before_transient_solve()
        elif self._last_bc_setting=="stationary":
            self.actions_before_stationary_solve()
        _pyoomph.set_interpolate_new_interface_dofs(True) # Activate the interpolation again, good for spatial adaptivity
        return True

    ############################################################################################
    # State snapshots for the failed-resolve-after-adaptation recovery.
    # See pyoomph/generic/adaptive_recovery.py and dev_docs/adaptive_resolve_recovery.md.
    ############################################################################################

    def _snapshot_state(self,to_memory:bool=True)->bytes | str:
        """Take a full state snapshot, without touching the user's output directory.

        Returns whatever :py:meth:`_restore_state` needs to put it back: the raw bytes when
        ``to_memory``, otherwise the name of a temporary file. Everything a restore needs is in
        there - the refinement pattern, the dofs, the history values, the pinned values, the current
        time and all dts - because this is the very same stream ``save_state`` writes."""
        if not to_memory:
            fname=self._state_snapshot_name()
            os.makedirs(os.path.dirname(fname),exist_ok=True)
            self.save_state(fname,quiet=True)
            return fname
        buf=io.BytesIO()
        self.save_state(buf,quiet=True)
        if self.is_distributed():
            # save_state merges the mesh into ONE partition-independent stream that only rank 0 ends
            # up holding, while load_state needs every rank to read the whole thing. So the buffer
            # has to travel; that broadcast is the only extra cost a distributed snapshot has.
            comm=get_mpi_world_comm()
            assert comm is not None
            return cast(bytes,comm.bcast(buf.getvalue() if get_mpi_rank()==0 else None,root=0))
        # Not distributed: every rank holds the whole problem and wrote its own complete buffer
        # (save_state skips the redundant-writer guard for streams), so there is nothing to share.
        return buf.getvalue()

    def _state_snapshot_name(self)->str:
        """The file name of the next on-disk snapshot - the SAME one on every rank.

        The pid is in the name so that two runs sharing an output directory cannot overwrite each
        other's snapshots. Under MPI it may not be the *local* pid, though: save_state writes one
        file, rank 0's, and _restore_state has every rank read that file back. A per-rank name left
        every other rank pointing at a file nobody had written, so the restore died with a
        FileNotFoundError on those ranks - and before load_state agreed on its outcome, it hung the
        rest of the job with it. Rank 0's pid is what identifies the job, and it is unique in the
        same way the local one was.
        """
        if self._state_snapshot_job_id is None:
            job_id=os.getpid()
            if get_mpi_nproc()>1:
                comm=get_mpi_world_comm()
                assert comm is not None
                job_id=cast(int,comm.bcast(job_id,root=0))
            self._state_snapshot_job_id=job_id
        fname=os.path.join(self.get_output_directory(),"_states",
                           "_snapshot_{:d}_{:d}.dump".format(self._state_snapshot_job_id,self._state_snapshot_counter))
        self._state_snapshot_counter+=1
        return fname

    def _restore_state(self,snapshot:bytes | str)->None:
        """Put back a state taken by :py:meth:`_snapshot_state`."""
        self.load_state(snapshot if isinstance(snapshot,str) else io.BytesIO(snapshot),quiet=True)

    def _discard_state_snapshot(self,snapshot:bytes | str)->None:
        # Only rank 0 wrote the file, so only rank 0 removes it - all ranks racing to unlink the one
        # shared name means the loser raises on a file the winner has just taken away. No barrier is
        # needed before it: load_state ends in a collective, so no rank can still be reading.
        if isinstance(snapshot,str) and get_mpi_rank()==0:
            try:
                os.remove(snapshot)
            except FileNotFoundError:
                pass # discarded twice, e.g. when _first_pre and _pre are the same snapshot

    @contextlib.contextmanager
    def _suppress_unrefinement(self):
        """Let the adaptation refine but never unrefine, for the duration of the block.

        Unrefinement is what usually breaks the re-solve: it removes resolution exactly where the
        solution is stiff, and the interpolation onto the coarser mesh then lands outside the Newton
        basin. Refinement alone is nearly always benign for the conditioning.

        Errors are non-negative and oomph-lib unrefines on ``error < min_permitted_error``, so a
        negative threshold takes unrefinement out entirely."""
        saved:list[tuple[Any,float]]=[]
        saved_desired_ndof=self.desired_ndof
        # The desired_ndof controller recomputes BOTH thresholds on every adaptation, so it would
        # simply overwrite what we set here. Unset it and let it hand the user's thresholds back
        # right away, before we override them.
        self.desired_ndof=None
        self._restore_thresholds_before_desired_ndof()
        for _name,mesh in self._meshdict.items():
            if isinstance(mesh,ODEStorageMesh) or not mesh.refinement_possible():
                continue
            saved.append((mesh,mesh.min_permitted_error))
            mesh.min_permitted_error=-1.0
        try:
            yield
        finally:
            for mesh,val in saved:
                mesh.min_permitted_error=val
            self.desired_ndof=saved_desired_ndof

    @contextlib.contextmanager
    def _temporary_newton_settings(self,**overrides:Any):
        """Apply Newton solver overrides for the duration of the block and put the old ones back."""
        old=self._override_for_this_solve(**overrides)
        try:
            yield
        finally:
            self._override_for_this_solve(**old)

    def _adapt_recovery_unsolved_result(self)->Any:
        """What solve() should return when a recovery accepted a state instead of solving.

        For a stationary solve that is 0, exactly as the normal path returns; for a transient one it
        is the timestep that was actually taken, which is the one the restored state carries."""
        if self._adapt_recovery_transient:
            return self.timestepper.time_pt().dt(0)
        return 0

    def _solve_with_adapt_recovery(self,spatial_adapt:int,transient:bool,shift_values:bool,do_solve:Callable[[int,bool],Any])->Any:
        """Run the oomph-lib solve call, letting :py:attr:`adaptive_resolve_recovery` handle a failed
        re-solve after an adaptation. Without a policy this is just ``do_solve(...)``."""
        policy=self.adaptive_resolve_recovery
        if policy is None or not policy.active or spatial_adapt<=0:
            return do_solve(spatial_adapt,shift_values)
        return policy.run(self,do_solve,spatial_adapt,shift_values,transient)

    def _adaptive_solve_checkpoint(self,isolve:int,just_adapted:bool)->None:
        """Called from oomph-lib immediately before and after every adapt() in an adaptive solve."""
        policy=self.adaptive_resolve_recovery
        if policy is not None:
            policy.checkpoint(self,isolve,just_adapted)

    def _recover_from_failed_adaptive_resolve(self,linear_solver_error:bool,iterations:int)->bool:
        """Called from oomph-lib instead of abandoning the run. See AdaptiveResolveRecovery."""
        policy=self.adaptive_resolve_recovery
        if policy is None:
            return False
        try:
            return policy.handle_failure(self,linear_solver_error,iterations)
        except Exception as e:
            # A handler that throws would replace a diagnosable Newton failure with a confusing one
            # raised from inside a C++ catch block, so report it and fall back to the old, fatal
            # behaviour rather than letting it escape.
            print("The failed-adaptation recovery handler itself failed ("+type(e).__name__+": "+str(e)+
                  "); reporting the original Newton failure instead.")
            return False


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
        """
        Returns a :py:class:`~pyoomph.utils.dof_selector._DofSelector` object that allows to select degrees of freedom (DoFs) for further operations. For example, you can select all DoFs on a certain domain or with a certain equation type. The selected DoFs can then be used to e.g. get their values, set their values, or apply boundary conditions.
        It should be wrapped in a `with` statement to ensure proper cleanup after use. For example:

        .. code-block:: python

            with problem.select_dofs() as dofs:
                dofs.select("domain/velocity_x", "domain/velocity_y", "domain/pressure")
                dofs.unselect("domain/temperature")
                problem.solve()  # Only solve for the velocity/pressure fields, not for the temperature field
        """
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
     
        
        
    def precice_run(self,maxstep:float | None=None,temporal_error:float | None=None,output_initially:bool=True,fast_dof_backup:bool=False):
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


    def create_text_file_output(self,filename:str,header:list[str] | None=None,relative_to_output_dir:bool=True)->"NumericalTextOutputFile":
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


    # Called from load_balance
    def _build_mesh(self):
        print("Building mesh in Python","On enter, we have",self.nsub_mesh(),"submeshes, and the mesh dict keys are: "+str(self._meshdict.keys()))
        for meshname,eqtree in self._equation_system.get_children().items(): 
            #Find the mesh that generates the mesh we want to have
            if eqtree._equations is None: 
                raise RuntimeError("Empty bulk equations")
            mesh=None
            for m in self._meshtemplate_list:
                if m.has_domain(meshname):
                    previous_mesh=self._meshdict.get(meshname,None)
                    assert previous_mesh is None or isinstance(previous_mesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d))
                    mesh=MeshFromTemplate(self,m,meshname,eqtree,previous_mesh=previous_mesh)

                    self._meshdict[meshname]=mesh
                    assert eqtree._equations is not None
                    eqtree._equations._mesh=mesh  #type:ignore
                    eqtree.get_code_gen()._mesh=mesh
                    mesh._finalise_creation()
                    print("Mesh '"+meshname+"' generated from template '"+str(m)+"'","NELEMENTS: ",mesh.nelement())
            if eqtree._equations._is_ode(): 
                if mesh is not None:
                    if not isinstance(mesh,ODEStorageMesh):
                        raise RuntimeError("Cannot add an ODE to a spatial mesh yet")
                mesh=ODEStorageMesh(self,eqtree,meshname)
                eqtree.get_code_gen()._mesh=mesh 
                eqtree.get_code_gen().set_latex_printer(self.latex_printer)
                self._meshdict[meshname]=mesh
                raise RuntimeError("ODE meshes are not fully implemented yet. Please check whether this works")
            else:
                if mesh is None:
                    #print(str(self._equation_system))
                    avdoms=set()
                    for m in self._meshtemplate_list:
                        avdoms.update(set(m._domains.keys()))
                    raise RuntimeError("No mesh template with a domain named '"+meshname+'" was added, but there are equations defined on this domain. Available domains are '+str(avdoms))
        print("Finished building mesh in Python")
        print("Mesh dict keys: "+str(self._meshdict.keys()))
        self._interfacemeshes=[]
        print(self._interfacemeshes)
        self.rebuild_global_mesh_from_list(rebuild=False)
        #self.rebuild_global_mesh()
        print("NSUBMESH",self.nsub_mesh())
        import gc
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()
        self.rebuild_global_mesh()
        
        self.invalidate_cached_mesh_data()
        #eqs=self._equation_system.get_by_path("domain")
        
        #out=cast(CombinedEquations,eqs)
        #out.
        #print(eqs)
        #exit()
        
    def load_balance(self):
        if not self.is_distributed():
            return
        
        super().load_balance()

        self.rebuild_global_mesh_from_list(rebuild=True)
        # Repartitioning moves facets between ranks, so which of them is a halo has changed. This path
        # does not go through actions_after_adapt(), so the skeletons' halo scheme is rebuilt here.
        self.setup_interior_facet_halo_scheme()
        self.actions_after_remeshing()
        self.reapply_boundary_conditions()
        print("After load balance, we have",self.nsub_mesh(),"submeshes, and the mesh dict keys are: "+str(self._meshdict.keys()))
        print("ON PROC",get_mpi_rank(),self.ndof(),self.nsub_mesh(),self.mesh_pt(0).nelement()) #type:ignore
        
        
        
                

############## DOF SELECTOR ###################
class _DofSelector:
    """
    Should be only created via :py:meth:`~pyoomph.generic.problem.Problem.select_dofs`.
    """
    
    def __init__(self,problem:"Problem"):
        self._problem=problem
        self._all_unselected:bool | None=None
        self._tree:dict[str,Any]={}

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


    def _traverse(self,n:dict[str,Any],select:bool,onlydof:str | None=None):
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
        """
        Unselects all degree of freedom from solving within the `with` block
        """
        self._problem._dof_selector_used= "INVALID" 
        self._problem.invalidate_eigendata()        
        self._all_unselected=True
        self._traverse(self._tree,False)

    def select_all(self):
        """
        Selects all degree of freedom for solving within the `with` block
        """
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
        """
        Selects the dofs passed as arguments, e.g. ``select("droplet/velocity_x","droplet/velocity_y")`` or similar.
        If nothing has been selected/unselected before, everything else will be unselected.
        """
        if self._all_unselected is None:
            self.unselect_all()
        self._problem._dof_selector_used = "INVALID" 
        self._problem.invalidate_eigendata()
        for k in args:
            self._select_or_unselect(k,True)

    def unselect(self,*args:str):
        """
        Unselects the dofs passed as arguments, e.g. ``unselect("droplet/velocity_x","droplet/velocity_y")`` or similar.
        If nothing has been selected/unselected before, everything else will be selected.
        """
        if self._all_unselected is None:
            self.select_all()
        self._problem._dof_selector_used = "INVALID" 
        self._problem.invalidate_eigendata()
        for k in args:
            self._select_or_unselect(k,False)


    def _apply_on_domain(self,mesh:AnyMesh | None)->None:
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
        selected:set[str]=set()
        unselected:set[str]=set()
        boundinds:set[int]=set()
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


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
