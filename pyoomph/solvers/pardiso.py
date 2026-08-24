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
 
import ctypes.util


from ctypes import CDLL

import os
from pathlib import Path

import sys

from more_itertools import first

from ..typings import *
import numpy

from importlib import metadata

if TYPE_CHECKING:
    from ..generic.problem import Problem

def _mkl_rt_from_package()->CDLL | None:
    # Most deterministic source when the pip 'mkl' wheel is installed: it records
    # the (versioned) runtime library in its RECORD as a path relative to
    # site-packages, e.g. "../../libmkl_rt.so.2" -> <env>/lib/libmkl_rt.so.2.
    # Note there is never a bare "libmkl_rt.so" - the wheel only ships the
    # versioned name, so we must not assume a single, unversioned match.
    try:
        files=metadata.files('mkl') or []
    except metadata.PackageNotFoundError:
        return None
    cands:list[str]=[]
    for p in files:
        base=os.path.basename(str(p))
        if base.startswith(("libmkl_rt","mkl_rt")):
            loc=os.path.realpath(p.locate())  # p.locate() is un-normalized (has ../..)
            if os.path.exists(loc):
                cands.append(loc)
    # Prefer an unversioned name if one ever appears, else try each candidate.
    def _rank(path:str)->tuple[int,str]:
        b=os.path.basename(path)
        return (0 if b in ("libmkl_rt.so","libmkl_rt.dylib","mkl_rt.dll") else 1, b)
    for loc in sorted(set(cands),key=_rank):
        try:
            return CDLL(loc)
        except OSError:
            continue
    return None


def _try_to_find_lib(nam:str | list[str])->CDLL | None:
    if isinstance(nam,list):
        for l in nam:
            res=_try_to_find_lib(l)
            if res is not None:
                return res
        return None

    res = None
    try:
        resl=CDLL(nam)
        return resl
    except:
        expname=ctypes.util.find_library(nam)
        if expname is None:
            return None
        try:
            resl=CDLL(expname)
            if resl is None:
                return None
            return resl
        except:
            res = None

    return None


MKLlib:"CDLL | None"
if sys.platform == "linux":
    if "PYOOMPH_PARDISO_LIB" in os.environ.keys():
        MKLlib=CDLL(os.environ["PYOOMPH_PARDISO_LIB"])
    else:
        MKLlib=_mkl_rt_from_package() or _try_to_find_lib(["libmkl_rt.so",os.path.join(Path.home(), ".local/lib/libmkl_rt.so"),"mkl_rt",os.path.join(Path.home(), ".local/lib/libmkl_rt.so.2"),os.path.join(Path.home(), ".local/lib/libmkl_rt.so.3"),os.path.join(Path.home(), ".local/lib/libmkl_rt.so.4")])
elif sys.platform == "win32":
    if "PYOOMPH_PARDISO_LIB" in os.environ.keys():
        MKLlib=CDLL(os.environ["PYOOMPH_PARDISO_LIB"])
    else:
        MKLlib = _mkl_rt_from_package() or _try_to_find_lib(["mkl_rt.dll", "mkl_rt.1.dll","mkl_rt.2.dll","mkl_rt.3.dll","mkl_rt.4.dll", "mkl_rt"])
elif sys.platform=="darwin":
    if "PYOOMPH_PARDISO_LIB" in os.environ.keys():
        MKLlib=CDLL(os.environ["PYOOMPH_PARDISO_LIB"])
    else:
        MKLlib = _mkl_rt_from_package() or _try_to_find_lib(["libmkl_rt.dylib", "libmkl_rt.1.dylib", "libmkl_rt.2.dylib","libmkl_rt.3.dylib","libmkl_rt.4.dylib", "mkl_rt"])
else:
    raise RuntimeError("Unknown platform: "+sys.platform)

if MKLlib is None:
    raise RuntimeError("Pardiso not found")

from builtins import object

# from pyMKL import pardisoinit, pardiso, mkl_get_version
from ctypes import POINTER, byref, c_longlong, c_int, Structure, c_char_p
import numpy as np
import scipy.sparse as sp #type:ignore
from numpy import ctypeslib

from .generic import GenericLinearSystemSolver, GenericEigenSolver, SolverError


class PardisoError(SolverError):
    """MKL Pardiso reported one of its documented `error` codes, i.e. it did not solve the system.

    A SolverError rather than a plain RuntimeError, so that a caller able to retry with a smaller step
    does so instead of the run ending here -- see the note in _check_pardiso_error.
    """

####

pardisoinit = MKLlib.pardisoinit

pardisoinit.argtypes = [POINTER(c_longlong),
                        POINTER(c_int),
                        POINTER(c_int)]
pardisoinit.restype = None

feastinit = MKLlib.feastinit
# Sparse interfaces
# Real general
# {p}d{i}feast_gcsr{ev,gv}{x}
# where gv: generalized
# i inexact iterative
# p parallel

# feastcall=MKLlib.feast_gcsr
# print(dir(MKLlib))
# print()
# exit()


pardiso = MKLlib.pardiso

pardiso.argtypes = [POINTER(c_longlong),  # pt
                    POINTER(c_int),  # maxfct
                    POINTER(c_int),  # mnum
                    POINTER(c_int),  # mtype
                    POINTER(c_int),  # phase
                    POINTER(c_int),  # n
                    POINTER(None),  #type:ignore # a
                    POINTER(c_int),  # ia
                    POINTER(c_int),  # ja
                    POINTER(c_int),  # perm
                    POINTER(c_int),  # nrhs
                    POINTER(c_int),  # iparm
                    POINTER(c_int),  # msglvl
                    POINTER(None),  #type:ignore # b
                    POINTER(None),  #type:ignore # x
                    POINTER(c_int)]  # error)
pardiso.restype = None


class pyMKLVersion(Structure):
    _fields_ = [('MajorVersion', c_int),
                ('MinorVersion', c_int),
                ('UpdateVersion', c_int),
                ('ProductStatus', c_char_p),
                ('Build', c_char_p),
                ('Processor', c_char_p),
                ('Platform', c_char_p)]


_mkl_get_version = MKLlib.mkl_get_version
_mkl_get_version.argtypes = [POINTER(pyMKLVersion)]
_mkl_get_version.restype = None


def mkl_get_version():
    MKLVersion = pyMKLVersion()
    _mkl_get_version(MKLVersion)
    version = {'MajorVersion': MKLVersion.MajorVersion,
               'MinorVersion': MKLVersion.MinorVersion,
               'UpdateVersion': MKLVersion.UpdateVersion,
               'ProductStatus': MKLVersion.ProductStatus,
               'Build': MKLVersion.Build,
               'Platform': MKLVersion.Platform}

    versionString = 'Intel(R) Math Kernel Library Version {MajorVersion}.{MinorVersion}.{UpdateVersion} {ProductStatus} Build {Build} for {Platform} applications'.format(
        **version)

    return versionString


_mkl_get_max_threads = MKLlib.mkl_get_max_threads
_mkl_get_max_threads.argtypes = None #type:ignore
_mkl_get_max_threads.restype = c_int


def mkl_get_max_threads():
    max_threads = _mkl_get_max_threads()
    return max_threads


_mkl_set_num_threads = MKLlib.mkl_set_num_threads
_mkl_set_num_threads.argtypes = [POINTER(c_int)]
_mkl_set_num_threads.restype = None


def mkl_set_num_threads(num_threads:int):
    _mkl_set_num_threads(c_int(num_threads))


# MKL factorises with STATIC pivoting, and on hard systems that fails in two ways -- only one of which
# it reports. Both were measured on the heated-cylinder tutorial (docs/source/tutorial/pde/adapt) at a
# desired_ndof above ~170k, where the joint error criterion deliberately leaves an advection-dominated
# tracer under-resolved:
#
#   * it perturbs pivots, iterative refinement cannot repair them, and phase 33 returns error -4; or
#   * it reports a perfectly clean factorisation (IPARM(14) == 0), therefore does no refinement at
#     all, and returns a solution whose backward error is O(1) -- 7.6 was measured. Newton is then
#     fed nonsense and stops at its iteration limit, which from the outside looks like a physics or
#     continuation problem rather than a linear-solver one. This is the more dangerous of the two.
#
# The matrices are not singular, whatever MKL's message for -4 says: UMFPACK, which pivots
# dynamically, solves every one of them. These two settings cure both failure modes. IPARM(13)=2 is
# MKL's two-level weighted matching, its own first recommendation for -4; IPARM(10)=8 perturbs a
# doubtful pivot earlier instead of using it. Neither changed any solution that already worked -- the
# tutorial's dofs came back bit-identical, and the recovered runs reproduce UMFPACK's answer exactly
# (ndof 240172) about six times faster.
_ESCALATED_IPARM = {12: 2, 9: 8}  # 0-based indices, i.e. IPARM(13) and IPARM(10)

# How far to refine a symmetric-mtype solve that MKL left unrefined (see _needs_explicit_refinement).
# TWO steps, because that is exactly what MKL grants a general mtype in the same situation -- this is
# parity with the path the symmetric factorisation replaced, not a stronger repair, so it must not cost
# more either. It is enough where it was measured: on the four-domain cross point of
# tests/test_adaptive_interface_coupling.py it takes the backward error from 4.6e-05 to 5.0e-14 (a
# third step reaches 1.2e-15, and the Newton residual is 7e-16 from two steps on). Anything left over
# is caught by the backward-error check in PardisoSolver, and a run that genuinely needs more can ask
# for repair_bad_solves, which refines up to 20 steps and escalates the pivoting. MKL spends the whole
# cap once one is given -- it reports IPARM(7) == cap for every value tried -- so the cap IS the cost,
# and 20 here would be 20 triangular solves on every perturbed factorisation.
_SYMMETRIC_AUTO_REFINEMENT_STEPS = 2

# Backward error above which a solve is disbelieved and the escalation above is tried. Healthy solves
# of that same tutorial peak at 1e-6 while the broken ones sit at 1e0, so any threshold in between
# separates them. 1e-4 keeps two orders of margin on the side that must not fire, because a false
# positive costs a full refactorisation.
_BACKWARD_ERROR_LIMIT = 1e-4

# What to suggest when a solve comes back wrong: a solver that exchanges a bad pivot instead of
# perturbing it, which is the property MKL does not have and no amount of iparm tuning gives it.
#
# UMFPACK is the one measured above. MUMPS belongs on the list for the same reason: it does threshold
# partial pivoting inside each frontal matrix and DELAYS a pivot it cannot eliminate to the parent
# front. That is not a footnote from its manual -- it is visible in this codebase, in PETSCSolver's
# handling of INFOG(1) = -9 (see solvers/petsc.py): MUMPS' workspace cannot be predicted from the
# analysis phase precisely because pivoting decides the real fill-in, and there the off-diagonal pivot
# count INFOG(12) was watched going 288 -> 2638 as a continuation walked away from a stale ordering.
# A statically pivoting solver would have had nothing to report there.
_DYNAMIC_PIVOTING_ADVICE = "a dynamically pivoting solver (umfpack, or MUMPS via --petsc_mumps)"


class pardisoSolver(object):
    
    def __init__(self, matA:Any, mtype:int=11, verbose:bool=False,iparm_override:dict[int,int]={},repair_bad_solves:bool=False):
            #mode  11 : real, nonsymmetric
            #mode  13 : complex,  nonsymmetric

        # Whether solve_checked() verifies its own answer (see there). Off means the residual is never
        # formed and the pivoting escalation never fires, i.e. whatever MKL returns is handed on.
        self.repair_bad_solves = repair_bad_solves

        self.mtype = mtype
        if mtype in [1, 3]:
            msg = "mtype = 1,3 not implemented yet."
            raise NotImplementedError(msg)
        elif mtype in [2, -2, 4, -4, 6, 11, 13]:
            pass
        else:
            msg = "Invalid mtype: mtype={}".format(mtype)
            raise ValueError(msg)

        self.n = matA.shape[0]

        self.dtype:"type[np.complexfloating[Any,Any]] | type[np.floating[Any]]"
        if mtype in [4, -4, 6, 13]:
            # Complex matrix
            self.dtype = np.complex128
        elif mtype in [2, -2, 11]:
            # Real matrix
            self.dtype = np.float64
        self.ctypes_dtype = ctypeslib.ndpointer(self.dtype)

        if mtype in [2, -2, 4, -4, 6]:
            matA = sp.triu(matA, format='csr') #type:ignore
        elif mtype in [11, 13]:
            matA = matA.tocsr()

        if not matA.has_sorted_indices:
            matA.sort_indices()

        self.a = matA.data
        self.ia = matA.indptr
        self.ja = matA.indices

        self._MKL_a = self.a.ctypes.data_as(self.ctypes_dtype)
        self._MKL_ia = self.ia.ctypes.data_as(POINTER(c_int))
        self._MKL_ja = self.ja.ctypes.data_as(POINTER(c_int))

        self.maxfct = 1
        self.mnum = 1
        self.perm = 0

        if verbose:
            self.msglvl = 1
        else:
            self.msglvl = 0

        self.pt = np.zeros(64, np.int64) #type:ignore
        self._MKL_pt = self.pt.ctypes.data_as(POINTER(c_longlong))

        self.iparm = np.zeros(64, dtype=np.int32) #type:ignore
        self._MKL_iparm = self.iparm.ctypes.data_as(POINTER(c_int))

        # Init call
        pardisoinit(self._MKL_pt, byref(c_int(self.mtype)), self._MKL_iparm)

        verstring = mkl_get_version()

        if '11.3.3' in verstring:
            self.iparm[1] = 0
        else:
            self.iparm[1] = 3  
        self.iparm[23] = 1  
        self.iparm[34] = 1
          
        for k,v in iparm_override.items():
            self.iparm[k-1]=v

        # Two separate things, which used to be one flag and must not be:
        #
        #   _escalation_spent -- the escalation is one-shot. There is only one stronger setting to
        #       fall back to, so a second attempt would repeat the first and turn a failure into an
        #       infinite loop.
        #   _escalated_iparm  -- the escalated settings are in force RIGHT NOW. This is what the
        #       carry-over in PardisoSolver.solve reads, so it has to go back to False when an
        #       escalation is withdrawn again (see _deescalate_pivoting).
        #
        # Both start True when the caller already asked for any part of it (via iparm_override,
        # including that carry-over), which skips the pointless retry and leaves a deliberate choice
        # of these two knobs alone.
        #
        # Read off iparm_override, NOT off the resulting iparm array: pardisoinit gives mtype -2 an
        # IPARM(10) of 8 by itself (its default pivot perturbation for symmetric indefinite), which is
        # the very value _ESCALATED_IPARM asks for, so testing the array declared every single
        # symmetric factorisation "already escalated". That was wrong twice over -- it spent the
        # one-shot escalation before it could ever be used, and PardisoSolver then folded
        # _ESCALATED_IPARM into iparm_override for the REST OF THE RUN. The carried-over IPARM(13)=2
        # (two-level weighted matching) went on to fail the reordering with error -6 on the next
        # general-mtype factorisation of a matrix with unstored diagonal entries, i.e. on any
        # Lagrange-multiplier system whose verdict flips (a bifurcation tracker being activated).
        self._escalated_iparm = any(iparm_override.get(k + 1) == v for k, v in _ESCALATED_IPARM.items())
        self._escalation_spent = self._escalated_iparm
        self._pre_escalation_iparm:dict[int,int] | None = None

        # Backward error of the solution solve_checked() last returned, None when it could not be
        # formed (symmetric mtypes) or no solve has happened yet. Read by PardisoSolver to decide
        # whether the factorisation is still worth reusing, which is the one thing that happens even
        # when repair_bad_solves is off.
        self.last_backward_error:float | None=None

        self.last_mem_used_in_kb:int | None=None

    def update_matrix_values(self, matA:Any,mtype:int=11):
        if self.n != matA.shape[0]:
            return False
        if self.mtype != mtype:
            return False
        
        if len(matA.data) != len(self.a):
            return False
        if len(matA.indptr) != len(self.ia):
            return False
        if len(matA.indices) != len(self.ja):  
            return False
        
        if not matA.has_sorted_indices:
            matA.sort_indices()

        self.a[:] = matA.data[:]
        self.ia[:]=matA.indptr[:]
        self.ja[:]=matA.indices[:]
        
        self._MKL_a = self.a.ctypes.data_as(self.ctypes_dtype)
        self._MKL_ia = self.ia.ctypes.data_as(POINTER(c_int))
        self._MKL_ja = self.ja.ctypes.data_as(POINTER(c_int))
        return True

    def clear(self):        
        self.run_pardiso(phase=-1)

    def __del__(self, _is_finalizing:Any=sys.is_finalizing):
        # Skip once the interpreter itself is shutting down: this instance may only be
        # reachable this late because the owning Problem never called .release()/used
        # "with" (see project_nanobind_migration memory) - by that point module globals
        # this depends on (e.g. numpy as np, imported above) may already be cleared to
        # None, which would otherwise surface as a harmless but scary "Exception ignored
        # in __del__" (AttributeError: 'NoneType' object has no attribute 'zeros').
        # Freeing MKL's internal memory here is unnecessary anyway - the OS reclaims it
        # on process exit regardless.
        # _is_finalizing is bound as a default arg (evaluated once, at function-definition
        # time) rather than looked up as "sys.is_finalizing" at call time: CPython clears
        # nearly all module-level globals to None during interpreter shutdown, including
        # this module's own "sys" name - a bare "sys.is_finalizing()" call here can itself
        # raise "'NoneType' object has no attribute 'is_finalizing'" depending on shutdown
        # GC ordering (confirmed empirically). A default-arg value lives in the function
        # object itself, not the module dict, so it survives regardless.
        if _is_finalizing():
            return
        self.clear()

    def factor(self):
        #print("PARDISO FACTOR")
        out = self.run_pardiso(phase=12) #type:ignore

    def factor_numeric_only(self):
        # Phase 22 = numerical factorisation only, reusing the fill-reducing ordering and the symbolic
        # factorisation computed by an earlier phase 11 (or 12). Only valid while ia/ja are unchanged;
        # the caller must have established that (see PardisoSolver._reuse_symbolic_factorisation).
        out = self.run_pardiso(phase=22) #type:ignore

    def refactor(self):
        out = self.run_pardiso(phase=23) #type:ignore

    def solve(self, rhs:NPFloatArray | NPComplexArray)->NPFloatArray | NPComplexArray:
        #print("PARDISO SOLVE")
        x = self.run_pardiso(phase=33, rhs=rhs)
        return x

    # Backward error ||A x - rhs||_inf / ||rhs||_inf of a computed solution, or None when it cannot be
    # formed cheaply. Only mtype 11/13 store the whole matrix; the symmetric mtypes keep the upper
    # triangle alone (see __init__), for which this would be wrong rather than merely unavailable.
    def _backward_error(self, x:Any, rhs:Any)->float | None:
        if self.mtype not in (11, 13):
            return None
        scale = float(np.amax(np.absolute(rhs))) if rhs.size else 0.0
        if not (scale > 0.0):
            return None
        A = sp.csr_matrix((self.a, self.ja, self.ia), shape=(self.n, self.n))
        return float(np.amax(np.absolute(A * x - rhs))) / scale

    # Solve, giving the solver enough iterative refinement when its own factorisation says it needs it.
    #
    # MKL Pardiso factorises with STATIC pivoting: a pivot that comes out too small is perturbed instead
    # of exchanged, and repairing the damage is left to iterative refinement. Left to itself
    # (iparm[7] == 0) it refines only when it perturbed a pivot, and then for AT MOST TWO STEPS. That is
    # not always enough: on a Taylor-Hood Stokes saddle-point system on a moving mesh (the 3d hex+pyramid
    # ALE box of tests/test_adaptive_3d_campaign.py) two steps stop at a backward error of ~7e-8, so the
    # Newton step built on that solution stalls at a residual of 1e-8 instead of reaching machine zero.
    # Eight steps get there. The matrix is not to blame -- its condition number is 1.8e4, and SuperLU and
    # UMFPACK both solve it to 1e-15.
    #
    # The trigger is Pardiso's own report of how many pivots it perturbed (IPARM(14), an output of the
    # factorisation that survives until the next one). It is exactly the condition MKL itself uses to
    # decide whether to refine at all, so this only overrides HOW FAR it refines, and it costs nothing on
    # a factorisation that perturbed nothing -- which is every other layout of that same family. Raising
    # iparm[7] unconditionally would not be free: given an explicit cap MKL abandons the
    # "only if perturbed" rule and refines until its own criterion is met, which on the healthy layouts
    # means 4 extra triangular solves per solve where none were needed at all.
    # IPARM(14) == 0 means MKL perturbed no pivot. That is NOT the same as the answer being right --
    # see _ESCALATED_IPARM, where the worst solves measured came out of factorisations that reported
    # exactly this -- so the result is checked against the residual either way, and the refinement
    # below is only the cheap repair that is worth attempting first.
    def _solve_refined(self, rhs:Any, max_steps:int)->Any:
        if self.iparm[13] <= 0:
            return self.solve(rhs)
        old = self.iparm[7]
        self.iparm[7] = max_steps
        try:
            return self.solve(rhs)
        finally:
            self.iparm[7] = old

    # Whether MKL will refine this solve on its own, or whether we have to ask for it.
    #
    # The "at most two steps when a pivot was perturbed" rule above is what keeps the general mtypes
    # usable on a saddle-point system, and it is NOT applied to the symmetric ones. Measured on the
    # Lagrange-multiplier interfaces of tests/test_adaptive_interface_coupling.py: mtype -2 reports 25
    # perturbed pivots out of 138 dofs (the multiplier rows, whose diagonal is zero) and then performs
    # ZERO refinement steps -- IPARM(7) comes back 0 -- leaving a backward error of up to 9e-5 and a
    # Newton step that stalls at 1e-10 instead of machine zero. The same meshes under mtype 11 perturb
    # too, but come back refined in 2 steps at 1e-16. So on the symmetric mtypes the refinement is not
    # an extra repair to be opted into, it is the one MKL applies by itself everywhere else.
    def _needs_explicit_refinement(self)->bool:
        return self.mtype not in (11, 13) and self.iparm[13] > 0

    # Rebuild the factorisation with MKL's stronger pivoting (_ESCALATED_IPARM). IPARM(13) is read by
    # the REORDERING phase, so this cannot reuse the existing analysis -- it has to go back through
    # phase 12, and the handle must be released first or MKL leaks its internal workspace. Returns
    # False when the escalation has already been spent, so callers stop instead of looping.
    def _escalate_pivoting(self)->bool:
        if self._escalation_spent:
            return False
        self._escalation_spent = True
        self._pre_escalation_iparm = {k: int(self.iparm[k]) for k in _ESCALATED_IPARM}
        self.run_pardiso(phase=-1)
        for k, v in _ESCALATED_IPARM.items():
            self.iparm[k] = v
        try:
            self.factor()
        except PardisoError:
            # Leave the handle the way it was found, i.e. with the settings that at least got as far as
            # producing an answer last time rather than the ones that just failed to factorise at all.
            # This is for a caller holding a pardisoSolver directly: under PardisoSolver the raise now
            # goes on to discard this object entirely (_invalidate_factorisation), and the retry builds
            # a fresh one out of iparm_override.
            for k, v in self._pre_escalation_iparm.items():
                self.iparm[k] = v
            raise
        self._escalated_iparm = True
        if self.msglvl:
            print("PARDISO: escalated to IPARM(13)=2, IPARM(10)=8 and refactorised")
        return True

    # Put back the settings _escalate_pivoting replaced, for the case where the escalation did not
    # help. Withdrawing it matters far more than the wasted refactorisation costs: the escalated
    # settings otherwise stay in force for every later solve on this handle AND get folded into the
    # problem's iparm_override, so one marginal solve permanently changes how the rest of the run is
    # factorised. Measured on two tutorials -- two_layer_flow_single_domain died with an infinite
    # Newton residual on the next step, and droplet_spread_marangoni_and_gravity stalled every
    # subsequent arclength step at a residual of 1.1e-8 instead of reaching 1e-14, until the arc
    # length fell below its minimum. Neither had anything wrong with it that the escalation fixed.
    # The one-shot guard is deliberately NOT cleared: the escalation has been tried and failed.
    def _deescalate_pivoting(self)->None:
        if not self._escalated_iparm or self._pre_escalation_iparm is None:
            return
        self.run_pardiso(phase=-1)
        for k, v in self._pre_escalation_iparm.items():
            self.iparm[k] = v
        self.factor()
        self._escalated_iparm = False
        if self.msglvl:
            print("PARDISO: the escalation did not help, went back to IPARM(13)=%d, IPARM(10)=%d"
                  % (self.iparm[12], self.iparm[9]))

    def solve_checked(self, rhs:Any, max_steps:int=20)->Any:
        self.last_backward_error = None
        if not self.repair_bad_solves:
            # Repairs opted out of (PardisoSolver.repair_bad_solves): no extra refinement, no escalation
            # to stronger pivoting, no de-escalation, and no raise -- whatever MKL returns is the answer.
            # All three are repairs, which is what the flag is named for; the DIAGNOSIS below is not one
            # and stays. It cannot be replaced by anything downstream either: MKL reports a solution of
            # order 1e13 on a singular matrix as error 0, so nothing else in the stack can tell that this
            # solve failed. Recording it lets PardisoSolver throw the factorisation away rather than build
            # the next step on it (see _invalidate_factorisation) -- opting out of repairs is not opting
            # in to reusing a factorisation that has demonstrably stopped working.
            #
            # Dropping the EXTRA refinement is deliberate rather than incidental. It is a repair by
            # MKL's own machinery, and one that raises: on a singular matrix MKL reports the damage
            # refinement cannot repair as error -4 out of phase 33. The reason it was added is still
            # live, though -- without it the ALE box of tests/test_adaptive_3d_campaign.py stalls at a
            # Newton residual of 1e-8, so this flag is not free on hard systems.
            #
            # What is NOT dropped is the refinement MKL would have done by itself: on the symmetric
            # mtypes it does none (see _needs_explicit_refinement), so asking for it here is what makes
            # a proven-symmetric factorisation behave like the general one it replaced, rather than an
            # opt-in repair on top. It costs nothing when no pivot was perturbed, which is the healthy
            # case. The -4 exposure comes with it, and is the same one every mtype 11 solve has always
            # had.
            #
            # A factorisation that fails outright (phases 12/22, e.g. out of memory) still raises from
            # run_pardiso; there is no solution vector to hand on in that case. See _check_pardiso_error.
            x = (self._solve_refined(rhs, _SYMMETRIC_AUTO_REFINEMENT_STEPS)
                 if self._needs_explicit_refinement() else self.solve(rhs))
            self.last_backward_error = self._backward_error(x, rhs)
            if self.last_backward_error is not None and self.last_backward_error > _BACKWARD_ERROR_LIMIT:
                # Said out loud even though the repairs were opted out of: the alternative is a run that
                # neither converges nor explains itself, which is exactly the failure this whole path
                # was written for. It costs one line, and only on a solve that is genuinely wrong.
                print("PARDISO WARNING: backward error %.3e exceeds the %.0e limit. repair_bad_solves is "
                      "off, so the solution is used as it stands and only the factorisation is "
                      "discarded. Set repair_bad_solves=True to have stronger pivoting tried, or switch "
                      "to %s."
                      % (self.last_backward_error, _BACKWARD_ERROR_LIMIT, _DYNAMIC_PIVOTING_ADVICE))
            return x
        try:
            x = self._solve_refined(rhs, max_steps)
        except PardisoError as first:
            # A hard failure leaves nothing usable, so there is no fallback answer to keep: either
            # the escalation produces one or an exception stands. It is the FIRST one that stands:
            # on a genuinely singular matrix the escalated reordering (two-level weighted matching)
            # gives up with -6 "reordering failed", which describes the retry rather than the matrix,
            # while the original -4 names the zero pivot that is the actual problem.
            try:
                if self._escalate_pivoting():
                    return self._solve_refined(rhs, max_steps)
            except PardisoError:
                pass
            raise first

        err = self._backward_error(x, rhs)
        self.last_backward_error = err  # kept in step with whichever x is returned below
        if self.msglvl:
            print("PARDISO: %d perturbed pivot(s), refined the solve in %d step(s), backward error %s"
                  % (self.iparm[13], self.iparm[6], err))
        if err is None or err <= _BACKWARD_ERROR_LIMIT:
            return x
        try:
            escalated = self._escalate_pivoting()
        except PardisoError as e:
            # The escalated REORDERING could not even factorise: on a singular matrix the two-level
            # weighted matching gives up with -6. There is no answer to fall back on -- x is known
            # wrong, which is why we are here -- so this is one of the hard failures. Report what was
            # measured, because -6 on its own describes the retry rather than the matrix.
            raise PardisoError("MKL Pardiso returned a solution with a backward error of %.3e, and "
                               "refactorising with stronger pivoting then failed outright (%s). The "
                               "matrix is singular or nearly so at this state." % (err, e)) from e
        if not escalated:
            return x

        x2 = self._solve_refined(rhs, max_steps)
        err2 = self._backward_error(x2, rhs)
        if err2 is None or err2 > err:
            # No improvement, so the escalation was a false alarm on this matrix -- undo it before it
            # becomes the setting for the whole run, and keep the answer that was less wrong.
            self._deescalate_pivoting()
            if err2 is not None:
                # Factual rather than alarming: this fires on marginal solves too (4.4e-4 against the
                # 1e-4 limit on droplet_spread_marangoni_and_gravity), which Newton then absorbs
                # without trouble. The number is what tells the two apart.
                print("PARDISO WARNING: backward error %.3e exceeds the %.0e limit, and stronger "
                      "pivoting made it %.3e, so that was withdrawn and the original solution kept. "
                      "If the run does not converge, %s is the alternative."
                      % (err, _BACKWARD_ERROR_LIMIT, err2, _DYNAMIC_PIVOTING_ADVICE))
            return x
        if err2 > _BACKWARD_ERROR_LIMIT:
            # Better but still bad. Warn rather than raise: before this check existed such a solve was
            # returned silently, and refusing it outright would turn a badly-converging run into a
            # crashing one on problems that never reported anything wrong.
            print("PARDISO WARNING: backward error %.3e after escalation (was %.3e); the solution is "
                  "not trustworthy. Consider %s here."
                  % (err2, err, _DYNAMIC_PIVOTING_ADVICE))
        self.last_backward_error = err2
        return x2

    # MKL's documented meanings for the `error` output. Every value is a hard failure -- Pardiso has no
    # "converged badly but usable" answer to report, so anything nonzero means the contents of x are
    # not a solution.
    _ERROR_MEANINGS = {
        -1: "input inconsistent",
        -2: "not enough memory",
        -3: "reordering problem",
        -4: "zero pivot, numerical factorisation or iterative refinement problem "
            "(the matrix is singular or nearly so at this state)",
        -5: "unclassified (internal) error",
        -6: "reordering failed",
        -7: "diagonal matrix is singular",
        -8: "32-bit integer overflow",
        -9: "not enough memory for out-of-core",
        -10: "error opening out-of-core files",
        -11: "read/write error with out-of-core files",
        -12: "pardiso_64 called from the 32-bit library",
        -13: "interrupted by the mkl_progress function",
        -14: "internal error, e.g. no license file found",
        -15: "internal error, e.g. from the weighted matching (try iparm[12]=0)",
    }

    _PHASE_NAMES = {-1: "release of internal memory", 11: "reordering and symbolic factorisation",
                    12: "reordering, symbolic and numerical factorisation",
                    22: "numerical factorisation", 23: "numerical factorisation and solve",
                    33: "solve and iterative refinement"}

    def _check_pardiso_error(self,phase:int,code:int)->None:
        """Turn MKL's `error` output into an exception, rather than into a silently wrong solution.

        This check used to exist but could never fire: the argument was passed as
        ``byref(c_int(ERR))``, which builds a throwaway ctypes object from the Python int, so MKL wrote
        its code into a temporary that was discarded on the very next line and the ``if ERR != 0``
        below it compared the untouched Python int against zero. Pardiso has therefore never been able
        to report any of its own errors here, for as long as the call has existed.

        Raising does not end a run that could have recovered. This is a PardisoError, i.e. a
        SolverError, and the solver shim (src/nanobind/solver.cpp) reports those to oomph-lib as a
        failed Newton solve, so an adaptive time step or an arclength step rejects and retries with a
        smaller one -- which matters here because solve_checked() makes a singular matrix raise rather
        than return a huge solution. Which code carries that depends on the MKL: 2025.0 perturbs the
        zero pivot, refines, returns error 0 and a solution of order 1e13, and it is the backward
        error that catches it; the -4 in phase 33 that this was written for is one MKL's way of saying
        the same thing, not the only one. Note this deliberately does not extend to the MPI
        rejection in PardisoSolver.__init__: no smaller step makes Pardiso MPI-parallel.

        The release phase is the one exception, and it is deliberate: clear() is called from __del__,
        where raising only produces an "Exception ignored in __del__" and there is nothing left to
        salvage anyway -- the OS reclaims MKL's memory at process exit regardless.
        """
        if code == 0:
            return
        what = self._ERROR_MEANINGS.get(code, "undocumented error code")
        where = self._PHASE_NAMES.get(phase, "phase " + str(phase))
        msg = ("MKL Pardiso failed during the " + where + ": error " + str(code) + " (" + what + ")")
        if phase == -1:
            print("WARNING: " + msg)
            return
        raise PardisoError(msg + ". The solution vector is meaningless, so the solve is aborted here "
                                 "rather than handing it on as an answer.")

    def run_pardiso(self, phase:int, rhs:NPFloatArray | NPComplexArray | None=None)->NPFloatArray | NPComplexArray:

        if rhs is None:
            nrhs = 0
            x = np.zeros(1) #type:ignore
            rhs = np.zeros(1) #type:ignore
        else:
            if rhs.ndim == 1:
                nrhs = 1
            elif rhs.ndim == 2:
                nrhs = rhs.shape[1]
            else:
                msg = "Can only solve for 1 or 2 RHS"
                raise NotImplementedError(msg)
            rhs = rhs.astype(self.dtype).flatten(order='f') #type:ignore
            x = np.zeros(nrhs * self.n, dtype=self.dtype) #type:ignore

        MKL_rhs = rhs.ctypes.data_as(self.ctypes_dtype) #type:ignore
        MKL_x = x.ctypes.data_as(self.ctypes_dtype)
        # A named ctypes object, not byref(c_int(0)): the argument is an OUTPUT, so the object MKL
        # writes into has to outlive the call for its value to be readable afterwards.
        error = c_int(0)

        pardiso(self._MKL_pt,  # pt
                byref(c_int(self.maxfct)),  # maxfct
                byref(c_int(self.mnum)),  # mnum
                byref(c_int(self.mtype)),  # mtype
                byref(c_int(phase)),  # phase
                byref(c_int(self.n)),  # n
                self._MKL_a,  # a
                self._MKL_ia,  # ia
                self._MKL_ja,  # ja
                byref(c_int(self.perm)),  # perm
                byref(c_int(nrhs)),  # nrhs
                self._MKL_iparm,  # iparm
                byref(c_int(self.msglvl)),  # msglvl
                MKL_rhs,  # b
                MKL_x,  # x
                byref(error))  # error

        self._check_pardiso_error(phase, error.value)

        if self._MKL_iparm[14]!=0 or self._MKL_iparm[15]!=0 or self._MKL_iparm[16]!=0:
            self.last_mem_used_in_kb=max(self._MKL_iparm[14],self._MKL_iparm[15]+self._MKL_iparm[16])

        if nrhs > 1:
            x = x.reshape((self.n, nrhs), order='f') #type:ignore
        return x #type:ignore


from scipy.sparse import  csr_matrix #type:ignore


@GenericLinearSystemSolver.register_solver()
class PardisoSolver(GenericLinearSystemSolver):
    idname = "pardiso"

    # MKL Pardiso is not MPI-parallel, so under mpirun the base class gathers the system onto rank 0
    # and calls solve_serial() below there. That reuses everything on this class -- symbolic
    # factorisation reuse, solve_checked's backward-error test, the repairs -- which a separate
    # distributed implementation could not, and none of it issues an MPI collective, as the flag
    # requires. MKL's cluster_sparse_solver would be genuinely parallel, but is only reachable through
    # PETSc as mkl_cpardiso.
    #
    # This used to raise instead, on the grounds that a gathered solve is written back onto only half
    # of a replicated dof vector. That is not what happens: Problem::newton_solve redistributes dx onto
    # Dof_distribution_pt, which is non-distributed without --distribute, and DoubleVector::redistribute
    # then MPI_Allgathervs it to full length on every rank. Measured on a 529-dof Newton solve at 1, 2
    # and 3 ranks, both regimes: every rank's dof vector is bitwise identical and matches the serial
    # one. The claim predates the two CRDoubleMatrix::redistribute fixes of 8 Aug 2026.
    gathers_to_root_under_mpi=True

    def __init__(self, problem:"Problem",verbose:bool=False):
        super().__init__(problem)
        self._current_pardiso:"pardisoSolver | None" = None
        self.try_to_reuse_solver=False
        self.verbose=verbose
        self.iparm_override:dict[int,int]={}
        # Whether a solve that comes back wrong is REPAIRED: extra iterative refinement when MKL reports
        # perturbed pivots, then stronger pivoting and a refactorisation, then a raise if it is still
        # hopeless (all of it in pardisoSolver.solve_checked).
        #
        # Off by default, because every repair is speculative and two of them cost a full
        # refactorisation apiece -- a solve just over the limit pays for one to discover it was a false
        # alarm, and the escalated settings then have to be withdrawn again. On the tutorials that led
        # to this code, the repairs were the exception and the false alarms were not.
        #
        # It does NOT switch off the residual check itself, which is one sparse matrix-vector product
        # and the only way to know a solve failed at all: MKL returns a solution of order 1e13 on a
        # singular matrix as error 0. Either way, a solve over the limit discards the factorisation
        # (_invalidate_factorisation), so the next Newton or timestep retry starts from a fresh one
        # instead of a phase-22 refresh of factors that have stopped working.
        #
        # Turn it on for a run that will not converge, or whose PARDISO WARNINGs say the backward error
        # is over the limit: that is the case the repairs were measured on. The cost of leaving it off
        # is that a singular Jacobian reaches Newton as that 1e13 solution rather than as a retryable
        # SolverError, i.e. as an ordinary divergence -- which max_residuals still rejects, so an
        # adaptive step or an arclength step retries either way, just without a diagnosis. A
        # factorisation that fails outright raises regardless of this flag.
        self.repair_bad_solves=False
        # Skip Pardiso's reordering/symbolic phase (11) whenever the Jacobian sparsity pattern is
        # unchanged since the last factorisation, i.e. run phase 22 instead of phase 12. This requires
        # problem.keep_structural_zeros to be on -- otherwise the pattern follows the dof values and
        # jacobian_structure_id is 0, which disables this automatically. Distinct from
        # try_to_reuse_solver, which gambles on reusing the *numerical* factorisation as well; that is
        # only sometimes valid and is checked a posteriori, whereas symbolic reuse is always exact.
        self.reuse_symbolic_factorisation=True
        self._structure_id:int=0 # Pattern the current symbolic factorisation was computed for; 0 = none
        self._lastA:Any=None
        self._pattern_verified:bool=False # Whether the values now in Pardiso sit on the SAME pattern it was factorised for
        # Which path each op_flag==1 took. Counters rather than timings, so a benchmark cannot mistake a
        # silent fallback for a win (see dev_docs/structural_assembly.md, Phase 2).
        self.n_full_factorisations:int=0    # phase 12: reordering + symbolic + numeric
        self.n_numeric_factorisations:int=0 # phase 22: numeric only, symbolic reused
        self.n_numeric_reuses:int=0         # factors kept entirely, used as a CGS preconditioner
        self.n_symmetric_factorisations:int=0 # op_flag==1 calls that engaged mtype -2 (proven symmetry)
        self._active_mtype:int=11           # mtype the current/last factorisation was decided for

    def set_num_threads(self,nthreads:int | None):
        if nthreads is None or nthreads==0:
            mkl_set_num_threads(mkl_get_max_threads())
        else:
            mkl_set_num_threads(nthreads)


    def get_last_used_mem_size_in_kb(self):
        if self._current_pardiso is None:
            return 0
        elif self._current_pardiso.last_mem_used_in_kb is None:
            return 0
        else:
            return self._current_pardiso.last_mem_used_in_kb

            

    def get_jacobian_matrix(self,n:int,values:NPFloatArray, rowind:NPIntArray, colptr:NPIntArray)->Any:
        # The .copy() is load-bearing, not redundant: csr_matrix((values,rowind,colptr)) only *wraps*
        # the incoming arrays, which are zero-copy numpy views onto oomph-lib's CRDoubleMatrix buffers
        # (see src/nanobind/solver.cpp). pardisoSolver keeps ctypes pointers into a/ia/ja and MKL
        # Pardiso dereferences them again at the solve phase, after oomph may already have reassembled
        # (freed/reallocated) that Jacobian -> use-after-free (the "Valgrind can report problems"). The
        # copy gives Pardiso storage it owns for the whole factor->solve cycle. This is the one
        # unavoidable matrix copy every direct backend needs; see also the PETSc solver notes.
        return csr_matrix((values, rowind, colptr), shape=(n, n)).copy() #type:ignore

    def get_b(self,n:int,b:NPFloatArray):
        return b

    def _diagonal_fully_stored(self,A:Any)->bool:
        # MKL's symmetric mtypes require every diagonal position to be PRESENT in the pattern (an
        # explicit zero suffices). Structural check: same pattern with all-ones values, so stored
        # zeros count too - A.diagonal() itself cannot tell a stored zero from an absent entry.
        ones=csr_matrix((numpy.ones_like(A.data),A.indices,A.indptr),shape=A.shape)
        return int(ones.diagonal().sum())==A.shape[0]

    def requires_explicit_diagonal(self)->bool:
        # Only when the symmetric path (mtype -2) would actually engage: MKL needs the full diagonal
        # stored then. Keying on the verdict, not just the flag, keeps the matrix - and hence the
        # pivoting - of nonsymmetric problems bit-identical to a build without this feature.
        if not self.exploit_proven_symmetry:
            return False
        return self.problem._get_proven_matrix_symmetry(self.problem._get_solved_residual())[0]

    def _invalidate_factorisation(self)->None:
        """Throw the current factorisation away, so the next op_flag==1 builds a completely fresh one.

        Called on the two ways a factorisation stops being worth keeping: something raised, or a solve
        came back with a backward error over the limit. Both reuse tiers key off state that neither of
        those touches -- the symbolic one off _structure_id (phase 22 on an MKL handle whose last phase
        errored), the numeric one off _current_pardiso still being there (update_matrix_values keeps the
        broken factors and op_flag==2 then uses them as a preconditioner). So without this the retry
        walks straight back into the handle that just failed, and oomph-lib's retry is a SMALLER STEP,
        i.e. a different Jacobian that deserves its own reordering rather than a refresh of the old one.

        This matters most with repair_bad_solves off. With it on, a bad solve has usually been through
        _escalate_pivoting, which releases and rebuilds the handle itself; with it off nothing does.
        """
        self._structure_id = 0
        self._pattern_verified = False
        if self._current_pardiso is not None:
            try:
                self._current_pardiso.clear()
            except Exception:
                # Best-effort: releasing an MKL handle whose last phase errored may error again, and
                # dropping the reference is what actually matters here. __del__ retries the release.
                pass
            self._current_pardiso = None

    def _reuse_symbolic_factorisation(self,A:Any,mtype:int=11)->bool:
        """Feed a re-assembled Jacobian into the existing factorisation object without redoing the
        symbolic phase. Returns False if that is not possible, in which case the caller must build a
        fresh solver.

        The pattern is *verified* rather than merely trusted. problem.jacobian_structure_id promises it
        is unchanged, but that promise does not hold everywhere -- an augmented system's pattern is
        value-filtered, see _report_structure_id_mismatch -- and believing a stale id would mean
        Pardiso applies an old elimination tree to a new pattern, a silently wrong answer rather than a
        crash. Comparing the index arrays costs O(nnz) integer compares (~1 ms for 1.8M nonzeros)
        against the ~180 ms the symbolic phase takes, so the insurance is essentially free.
        """
        if not self.reuse_symbolic_factorisation:
            return False
        ps = self._current_pardiso
        if ps is None or self._structure_id == 0:
            return False
        if ps.mtype != mtype:
            # Intentional flip between symmetric and general (a tracker was toggled), not a pattern
            # bug - a fresh symbolic factorisation is required, quietly.
            return False
        if mtype not in (11, 13):
            # pardisoSolver holds only the upper triangle for symmetric mtypes: compare and copy
            # against the same half, or an identical pattern reads as a mismatch. Sort first - triu
            # of an unsorted CSR would otherwise be compared against the sorted ps.ja.
            if not A.has_sorted_indices:
                A.sort_indices()
            A = sp.triu(A, format='csr')
        if ps.n != A.shape[0] or len(ps.a) != len(A.data):
            return False
        # oomph-lib does not emit the column indices of a row in ascending order, and pardisoSolver
        # sorts them in its constructor. Sort here too, or ps.ja (sorted) and A.indices (as assembled)
        # would compare unequal even for an identical pattern -- and, worse, copying A.data into the
        # sorted ps.a would scramble the values. Sorting permutes indices and data together.
        if not A.has_sorted_indices:
            A.sort_indices()
        if not numpy.array_equal(ps.ia, A.indptr) or not numpy.array_equal(ps.ja, A.indices):
            self._report_structure_id_mismatch("the symbolic factorisation")
            return False
        ps.a[:] = A.data[:]
        return True

    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        try:
            return self._solve_serial(op_flag,n,nnz,nrhs,values,rowind,colptr,b,ldb,transpose)
        except Exception:
            self._invalidate_factorisation()
            raise

    def _solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        #print("CALL WITH OP FLAG ",op_flag,ldb,transpose)
        #print("PARDISO ", op_flag)
        if op_flag == 1:
#            print("INFO",len(values),len(rowind),len(colptr))
            A = self.get_jacobian_matrix(n,values, rowind, colptr)  # That is not optimal, of course
            # mtype -2 = real symmetric indefinite (Bunch-Kaufman), never 2 (SPD is not provable
            # symbolically). MKL additionally requires every diagonal entry to be STORED for the
            # symmetric mtypes; requires_explicit_diagonal() asks the assembly for that, but guard
            # here anyway - not every path to this solve runs the pre-Newton sync.
            mode = 11
            if self._use_symmetric_factorisation_now():
                if self._diagonal_fully_stored(A):
                    mode = -2
                else:
                    self.last_symmetry_decision = False
                    self.last_symmetry_decision_reason = "diagonal not fully stored (MKL symmetric mtypes require it)"
            self._active_mtype = mode
            if mode == -2:
                self.n_symmetric_factorisations += 1
            structure_id = self.problem.jacobian_structure_id
            if (not self.try_to_reuse_solver) and structure_id != 0 and structure_id == self._structure_id \
                    and self._reuse_symbolic_factorisation(A,mode):
                # Same pattern, new values: only the numerical factorisation has to be redone.
                self._lastA = A
                self._current_pardiso.factor_numeric_only() #type:ignore
                self.n_numeric_factorisations+=1
                if self.verbose:
                    print("PARDISO reused symbolic factorisation (structure id "+str(structure_id)+")")
                return 0
            self._structure_id = structure_id
            if self.try_to_reuse_solver:
                self._lastA=A
                if self._current_pardiso is None:
                    self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override,repair_bad_solves=self.repair_bad_solves)
                    if self.verbose: print("CREATED NEW PARDISO AND FACTOR")
                    self._current_pardiso.factor()
                    self.n_full_factorisations+=1
                    self._pattern_verified=True
                elif self._reuse_symbolic_factorisation(A,mode):
                    # Values copied in, factors deliberately left alone: this branch is the numeric
                    # reuse, and op_flag==2 will use those factors as a CGS preconditioner. The call
                    # above also VERIFIED the pattern, which is what lets the fallback there be a
                    # phase-22 refactorisation instead of a full rebuild.
                    self._pattern_verified=True
                    self.n_numeric_reuses+=1
                elif self._current_pardiso.update_matrix_values(A if mode in (11,13) else sp.triu(A,format='csr'),mtype=mode):
                    # Same sizes but the pattern was not verified (or symbolic reuse is off), so the
                    # cheap fallback in op_flag==2 is not available. pardisoSolver holds only the upper
                    # triangle for symmetric mtypes, so hand it the same half; its own mtype check
                    # routes an mtype flip (tracker toggled) to the rebuild branch below.
                    self._pattern_verified=False
                    self.n_numeric_reuses+=1
                else:
                    self._current_pardiso.clear()  # TODO: Only if matrix is entirely changed                
                    self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override,repair_bad_solves=self.repair_bad_solves)
                    if self.verbose: print("CREATED NEW PARDISO AND FACTOR")
                    self._current_pardiso.factor()                    
                    self.n_full_factorisations+=1
                    self._pattern_verified=True
            else:
                if self._current_pardiso:
                    self._current_pardiso.clear()  # TODO: Only if matrix is entirely changed
                # Kept even though this branch never reuses anything: it is what a resolve refactorises
                # from when the factorisation was discarded under it, see op_flag==2 below.
                self._lastA = A
                self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override,repair_bad_solves=self.repair_bad_solves)
                self._current_pardiso.factor()
                self.n_full_factorisations+=1
                if self.verbose:
                    print("PARDISO FACTOR IPARM",self._current_pardiso.iparm)                
        elif op_flag == 2:
            self.setup_solver()
            if self._current_pardiso is None:
                # A resolve is "same matrix, new right-hand side", so it normally has the factors from
                # the op_flag==1 that preceded it. It can arrive without them: _invalidate_factorisation()
                # drops a factorisation whose solve came back with too large a backward error, and with
                # repair_bad_solves off that is the ONLY thing that happens to it - nothing refactorises
                # on the way here. Arclength continuation resolves for its extra right-hand sides, so it
                # walked straight into this and asserted; three tutorials died that way
                # (hopf_switch, droplet_spread_marangoni_and_gravity, rising_bubble).
                #
                # _lastA is the matrix that factorisation was built from and is not cleared by the
                # invalidation, so the honest answer is to build the factors again rather than to
                # refuse. _structure_id and _pattern_verified were reset with it, so no reuse tier
                # can smuggle the discarded state back in.
                if self._lastA is None:
                    raise RuntimeError("Pardiso was asked to resolve (op_flag=2) before any matrix was "
                                       "factorised, so there is nothing to solve with.")
                if self.verbose:
                    print("PARDISO: refactorising for a resolve, the previous factorisation was discarded")
                # _lastA is the FULL matrix either way; pardisoSolver extracts the triangle itself for
                # symmetric mtypes, so rebuilding with the mtype of the discarded factorisation is safe.
                self._current_pardiso = pardisoSolver(self._lastA, mtype=getattr(self,'_active_mtype',11), verbose=self.verbose,
                                                      iparm_override=self.iparm_override,
                                                      repair_bad_solves=self.repair_bad_solves)
                self._current_pardiso.factor()
                self.n_full_factorisations+=1
            if self.try_to_reuse_solver:
                maxiters=30
                self._current_pardiso.iparm[7]=maxiters
                #self._current_pardiso.iparm[8]=1
                
                self._current_pardiso.iparm[3] = 63
                bv=self.get_b(n,b)
                # The reuse branch verifies its own solve and may refactorise, so it cannot hand a
                # plain re-solve callable to an augmented handler. Deflation only rescales the
                # increment and is applied once, after the accuracy check below.
                if self._custom_solve_routine_active():
                    raise NotImplementedError("An augmented assembly handler's custom solve routine is not supported while Pardiso reuses its factorisation. Set try_to_reuse_solver=False.")
                sol=self._current_pardiso.solve(bv)
                #self._current_pardiso.iparm[3] = 0
                err=numpy.amax(numpy.absolute(self._lastA*sol-bv))
                if self._current_pardiso.iparm[6]==maxiters or err>1e-10:
                    if self.verbose:
                        print("MUST RECOMPUTE FACTORIZATION","ITER",self._current_pardiso.iparm[6],"ERR",err)
                    # The two reuse tiers compose here. Reusing the numerical factors as a
                    # preconditioner has just failed, but that says nothing about the SPARSITY, so when
                    # the pattern is known unchanged the fallback only has to redo the numbers (phase
                    # 22) rather than the reordering and symbolic factorisation as well (phase 12).
                    # Before, a stalled numeric reuse always paid for a full rebuild -- which on the 3D
                    # benchmark is ~180 ms of reordering thrown away every time it happens.
                    if self._pattern_verified and self.reuse_symbolic_factorisation:
                        self._current_pardiso.iparm[3] = 0  # plain solve, not CGS, for the retry
                        self._current_pardiso.factor_numeric_only()
                        self.n_numeric_factorisations+=1
                    else:
                        if self._current_pardiso:
                            self._current_pardiso.clear()
                        mode=getattr(self,'_active_mtype',11)
                        self._current_pardiso = pardisoSolver(self._lastA, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override,repair_bad_solves=self.repair_bad_solves)
                        self._current_pardiso.factor()
                        self.n_full_factorisations+=1
                    sol=self._current_pardiso.solve(bv)
                elif self.verbose:
                    print("REUSE PARDISO AND REFACTOR DONE, ERROR",err,"IN ",self._current_pardiso.iparm[6],"ITERATIONS")
                b[:]=self._postprocess_newton_step(sol)
            else:
                # solve_checked, not solve: this branch has no accuracy check of its own (unlike the
                # try_to_reuse_solver branch above, which verifies its reused factors and refactorises),
                # so an under-refined Pardiso solve would be returned silently. See solve_checked.
                pd = self._current_pardiso
                # The RAW solve and the Newton step are kept apart, because the backward error below
                # is only meaningful for the raw one: the deflation rescale multiplies it by a scalar,
                # so ||A*sol - b|| would be a large number every time and would condemn a perfectly
                # good factorisation on every Newton step. (Measured: 21 full factorisations instead
                # of 1 + 21 phase-22 refreshes on a 14161-dof deflated solve.) An augmented handler's
                # custom solve routine returns something that is not a solution of J x = b at all, so
                # there is no raw vector to check in that case.
                if self._custom_solve_routine_active():
                    _raw = None
                    sol = self._solve_newton_step(lambda rhs : pd.solve_checked(rhs), self.get_b(n,b)) #type:ignore
                else:
                    _raw = pd.solve_checked(self.get_b(n,b)) #type:ignore
                    sol = self._postprocess_newton_step(_raw)
                # For the symmetric mtypes, pardisoSolver holds only the upper triangle and cannot
                # form the residual itself (last_backward_error stays None), which would silently
                # disable the over-limit invalidation below - and the in-solver pivot escalation of
                # solve_checked never fires either. The full matrix is still here, so form the same
                # backward error at the same cost (one SpMV); discard-and-refactorise is the safety
                # net that replaces the escalation. b still holds the right-hand side at this point.
                if _raw is not None and self._current_pardiso.last_backward_error is None and self._current_pardiso.mtype not in (11,13) and self._lastA is not None:
                    _scale=float(numpy.amax(numpy.absolute(b))) if b.size else 0.0
                    if _scale>0.0:
                        self._current_pardiso.last_backward_error=float(numpy.amax(numpy.absolute(self._lastA*_raw-b)))/_scale
                # Once a factorisation has had to escalate its pivoting AND KEPT IT, the next one on
                # this problem almost certainly will too -- under spatial adaptivity the mesh only
                # gets harder from here. (An escalation that did not help withdraws itself and clears
                # the flag again, precisely so that it does not get carried over: see
                # _deescalate_pivoting.) Folding it in means the following pardisoSolver starts there
                # instead of rediscovering it through another failed solve and refactorisation.
                if self._current_pardiso._escalated_iparm:
                    for k, v in _ESCALATED_IPARM.items():
                        self.iparm_override[k + 1] = v  # iparm_override is 1-based, _ESCALATED_IPARM is not
            if self.verbose:
                print("PARDISO SOLVE IPARM",self._current_pardiso.iparm)
            b[:] = sol[:]
            # A solve that came back wrong condemns the factorisation it came out of, whether or not
            # anything was done about it. This is the only consequence a bad solve has when
            # repair_bad_solves is off -- MKL reports these as error 0, so nothing raises and the retry
            # would otherwise walk straight back into the same factors via phase 22. Last, because
            # _invalidate_factorisation drops _current_pardiso, which the lines above still read.
            bwerr = self._current_pardiso.last_backward_error
            if bwerr is not None and bwerr > _BACKWARD_ERROR_LIMIT:
                self._invalidate_factorisation()
        else:
            raise RuntimeError("Cannot handle Pardiso mode " + str(op_flag) + " yet")
            return 666

        return 0  # TODO: Return sign of Jacobian


from .scipy import ScipyEigenSolver,DefaultMatrixType

class PardisoInvOp(object):
    def __init__(self, A:DefaultMatrixType, M:DefaultMatrixType | None=None,sigma:float | complex | None=None,mode:int=11):
        if sigma is None:
            self.mat=A
        else:
            self.mat=A-sigma*M #type:ignore
        if mode==-2:
            # Same MKL requirement as in PardisoSolver: every diagonal position of the shifted matrix
            # must be stored. J-sigma*M usually has it (M carries the time-derivative diagonal), but
            # fall back to the general mtype rather than fail when it does not.
            _mat=self.mat.tocsr() #type:ignore
            _ones=sp.csr_matrix((numpy.ones_like(_mat.data),_mat.indices,_mat.indptr),shape=_mat.shape)
            if int(_ones.diagonal().sum())!=_mat.shape[0]:
                mode=11
        self._current_pardiso=pardisoSolver(self.mat, mtype=mode, verbose=False) #type:ignore
        self._current_pardiso.factor()


    def __call__(self, b): #type:ignore
        x = self._current_pardiso.solve(b) #type:ignore
        return x

    matvec  = __call__  #type:ignore # ? 

    @property
    def shape(s): #type:ignore
        return s.mat.shape #type:ignore

    @property
    def dtype(s): #type:ignore
        return s.mat.dtype #type:ignore


@GenericEigenSolver.register_solver()
class PardisoArpackEigenSolver(ScipyEigenSolver):
    idname = "pardiso"

    def get_OPInv(self,M:DefaultMatrixType,J:DefaultMatrixType,shift:float | complex):
        if shift is None:
            OPinv = None
        else:
            mode=11
            if M.dtype==numpy.dtype("complex128") or J.dtype==numpy.dtype("complex128"):
                mode=13
            elif self.last_symmetry_decision and numpy.imag(shift)==0:
                # Set by ScipyEigenSolver.solve before asking for the operator: J-shift*M is then real
                # symmetric (indefinite), factorised via Bunch-Kaufman. The shift guard is defensive -
                # the symmetric decision is only ever True for a purely real solve path anyway.
                mode=-2
            OPinv = PardisoInvOp(J, M, sigma=shift,mode=mode)
        return OPinv


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
