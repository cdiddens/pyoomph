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

from ..generic.mpi import mpi_barrier,get_mpi_nproc,get_mpi_rank,get_mpi_world_comm
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
        if base.startswith(("libmkl_rt.so","libmkl_rt.dylib","mkl_rt")):
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

# Backward error above which a solve is disbelieved and the escalation above is tried. Healthy solves
# of that same tutorial peak at 1e-6 while the broken ones sit at 1e0, so any threshold in between
# separates them. 1e-4 keeps two orders of margin on the side that must not fire, because a false
# positive costs a full refactorisation.
_BACKWARD_ERROR_LIMIT = 1e-4


class pardisoSolver(object):
    
    def __init__(self, matA:Any, mtype:int=11, verbose:bool=False,iparm_override:dict[int,int]={}):
            #mode  11 : real, nonsymmetric
            #mode  13 : complex,  nonsymmetric

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
        self._escalated_iparm = any(self.iparm[k] == v for k, v in _ESCALATED_IPARM.items())
        self._escalation_spent = self._escalated_iparm
        self._pre_escalation_iparm:dict[int,int] | None = None

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
            # Leave the handle the way it was found. The caller turns this into a retryable
            # SolverError, so oomph-lib comes back with a smaller step and factorises this same
            # object again -- with settings that at least got as far as producing an answer last
            # time, rather than the ones that just failed to factorise at all.
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
                      "If the run does not converge, a dynamically pivoting solver (umfpack) is the "
                      "alternative." % (err, _BACKWARD_ERROR_LIMIT, err2))
            return x
        if err2 > _BACKWARD_ERROR_LIMIT:
            # Better but still bad. Warn rather than raise: before this check existed such a solve was
            # returned silently, and refusing it outright would turn a badly-converging run into a
            # crashing one on problems that never reported anything wrong.
            print("PARDISO WARNING: backward error %.3e after escalation (was %.3e); the solution is "
                  "not trustworthy. A dynamically pivoting solver (umfpack) may be needed here."
                  % (err2, err))
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

    def __init__(self, problem:"Problem",verbose:bool=False):
        super().__init__(problem)
        # MKL Pardiso is not MPI-parallel. Neither MPI mode works correctly:
        #  - with --distribute it is unsupported outright (the distributed mesh cannot be handled);
        #  - without --distribute the mesh is replicated (is_distributed()==False) while the linear
        #    algebra is still row-partitioned across ranks, so the gather-to-root solve returns the
        #    correct solution but only each rank's half is written back into its (full, replicated)
        #    dof vector. The un-owned dofs are never synchronised, so the subsequent residual
        #    assembly reads stale/uninitialised values -> intermittently wrong or NaN residuals.
        # Fail fast with a clear message instead of silently returning garbage.
        if get_mpi_nproc() > 1:
            raise RuntimeError(
                "The Pardiso linear solver cannot be used under MPI (running with "
                + str(get_mpi_nproc()) + " processes): MKL Pardiso is not MPI-parallel, and pyoomph's "
                "gather-to-root fallback does not correctly propagate the solution back onto a "
                "replicated (non-distributed) mesh. Use Pardiso only in serial (a single process), or "
                "for MPI runs switch to a distributed-capable solver, e.g. --petsc_mumps (or --petsc "
                "with an iterative preconditioner) together with --distribute.")
        self._current_pardiso = None
        self.try_to_reuse_solver=False
        self.verbose=verbose
        self.iparm_override:dict[int,int]={}
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

    def _reuse_symbolic_factorisation(self,A:Any)->bool:
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
        #print("CALL WITH OP FLAG ",op_flag,ldb,transpose)
        #print("PARDISO ", op_flag)
        if op_flag == 1:
#            print("INFO",len(values),len(rowind),len(colptr))
            A = self.get_jacobian_matrix(n,values, rowind, colptr)  # That is not optimal, of course
            mode = 11
            structure_id = self.problem.jacobian_structure_id
            if (not self.try_to_reuse_solver) and structure_id != 0 and structure_id == self._structure_id \
                    and self._reuse_symbolic_factorisation(A):
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
                    self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override)
                    if self.verbose: print("CREATED NEW PARDISO AND FACTOR")
                    self._current_pardiso.factor()
                    self.n_full_factorisations+=1
                    self._pattern_verified=True
                elif self._reuse_symbolic_factorisation(A):
                    # Values copied in, factors deliberately left alone: this branch is the numeric
                    # reuse, and op_flag==2 will use those factors as a CGS preconditioner. The call
                    # above also VERIFIED the pattern, which is what lets the fallback there be a
                    # phase-22 refactorisation instead of a full rebuild.
                    self._pattern_verified=True
                    self.n_numeric_reuses+=1
                elif self._current_pardiso.update_matrix_values(A):
                    # Same sizes but the pattern was not verified (or symbolic reuse is off), so the
                    # cheap fallback in op_flag==2 is not available.
                    self._pattern_verified=False
                    self.n_numeric_reuses+=1
                else:
                    self._current_pardiso.clear()  # TODO: Only if matrix is entirely changed                
                    self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override)
                    if self.verbose: print("CREATED NEW PARDISO AND FACTOR")
                    self._current_pardiso.factor()                    
                    self.n_full_factorisations+=1
                    self._pattern_verified=True
            else:
                if self._current_pardiso:
                    self._current_pardiso.clear()  # TODO: Only if matrix is entirely changed                
                self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override)
                self._current_pardiso.factor()
                self.n_full_factorisations+=1
                if self.verbose:
                    print("PARDISO FACTOR IPARM",self._current_pardiso.iparm)                
        elif op_flag == 2:
            self.setup_solver()
            assert self._current_pardiso is not None
            if self.try_to_reuse_solver:
                maxiters=30
                self._current_pardiso.iparm[7]=maxiters
                #self._current_pardiso.iparm[8]=1
                
                self._current_pardiso.iparm[3] = 63
                bv=self.get_b(n,b)
                if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
                    raise NotImplementedError("Custom solve not implemented for this case")
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
                        mode=11
                        self._current_pardiso = pardisoSolver(self._lastA, mtype=mode, verbose=self.verbose,iparm_override=self.iparm_override)
                        self._current_pardiso.factor()
                        self.n_full_factorisations+=1
                    sol=self._current_pardiso.solve(bv)
                elif self.verbose:
                    print("REUSE PARDISO AND REFACTOR DONE, ERROR",err,"IN ",self._current_pardiso.iparm[6],"ITERATIONS")
                b[:]=sol[:]
            else:
                # solve_checked, not solve: this branch has no accuracy check of its own (unlike the
                # try_to_reuse_solver branch above, which verifies its reused factors and refactorises),
                # so an under-refined Pardiso solve would be returned silently. See solve_checked.
                if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
                    pd = self._current_pardiso
                    sol=self.problem._custom_assembler.custom_solve_routine(lambda rhs : pd.solve_checked(rhs), b) #type:ignore
                else:
                    sol = self._current_pardiso.solve_checked(self.get_b(n,b))
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
        else:
            raise RuntimeError("Cannot handle Pardiso mode " + str(op_flag) + " yet")
            return 666

        return 0  # TODO: Return sign of Jacobian

    def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:        
        # NOTE: This does not solve the system via MPI Pardiso. Instead it solves it on the root process and scatters the solution. This is not optimal, but MKL Pardiso is not MPI parallel. MKL cluster_sparse_solver is, but this must be accessed via PETSc using mkl_cpardiso
        from mpi4py import MPI
        rank=get_mpi_rank()
        nproc=get_mpi_nproc()
        if op_flag==1:
            global_col_index = col_index
            rows = numpy.empty(len(col_index), dtype=np.int64)
            for i in range(nrow_local):
                rows[row_start[i]:row_start[i + 1]] = first_row + i
            # NOTE: `data` is the oomph-lib solver-state handle, which this (non-native-MPI)
            # Pardiso wrapper does not use - it is passed as None. The local non-zero count is
            # already supplied as the `nnz_local` argument (and only `values`/`col_index`/`row_start`
            # are needed below), so do NOT recompute it from `data` (that raised
            # "object of type 'NoneType' has no len()").
            cols = global_col_index
            data_values = values
            comm=get_mpi_world_comm()
            assert comm is not None
            #all_nnz = comm.gather(nnz_local, root=0)
            all_rows = comm.gather(rows, root=0)
            all_cols = comm.gather(cols, root=0)
            all_data = comm.gather(data_values, root=0)

            if rank==0:
                assert all_rows is not None and all_cols is not None and all_data is not None
                global_rows = np.concatenate(all_rows)
                global_cols = np.concatenate(all_cols)
                global_data = np.concatenate(all_data)
                #assert isinstance(A,csr_matrix)
                A = csr_matrix((global_data, (global_rows,global_cols)),shape=(n, n))
                A.eliminate_zeros()
                A.sort_indices()
                if self._current_pardiso:
                    self._current_pardiso.clear()  # TODO: Only if matrix is entirely changed
                mode = 11
                self._current_pardiso = pardisoSolver(A, mtype=mode, verbose=False)
                self._current_pardiso.factor()

                
            mpi_barrier()            
        elif op_flag==2:
            comm=get_mpi_world_comm()
            assert comm is not None
            counts = comm.gather(len(b), root=0)
            if rank == 0:
                counts = np.array(counts, dtype=np.int32)    
                displs = np.zeros(len(counts), dtype=np.int32)
                displs[1:] = np.cumsum(counts[:-1])
#                x_global = sol.copy()
                b_global = np.empty(n, dtype=b.dtype)
            else:
                displs = None
                b_global = None
            
            comm.Gatherv(sendbuf=b,recvbuf=[b_global, counts, displs, MPI.DOUBLE],root=0)

            sol:NPFloatArray | NPComplexArray | None = None
            if rank==0:
                self.setup_solver()
                assert self._current_pardiso is not None
                assert b_global is not None
                pd = self._current_pardiso
                if self.try_to_reuse_solver:
                    raise NotImplementedError("try_to_reuse_solver not implemented yet when running with MPI")
                # solve_checked for the same reason as the serial branch: nothing else verifies this one.
                if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
                    sol=self.problem._custom_assembler.custom_solve_routine(lambda rhs : pd.solve_checked(rhs), b) #type:ignore
                else:
                    sol = self._current_pardiso.solve_checked(self.get_b(n,b_global))

            if rank == 0:
                assert sol is not None
                counts = np.array(counts, dtype=np.int32)
                displs = np.zeros(len(counts), dtype=np.int32)
                displs[1:] = np.cumsum(counts[:-1])
                x_global = sol.copy()
            else:
                displs = None
                x_global = None
            #print("GATHERV SOLUTION",displs,counts)
            x_local = np.empty(len(b), dtype=np.float64)
            
            comm.Scatterv([x_global, counts, displs, MPI.DOUBLE],x_local,root=0)
            b[:] = x_local[:]
            mpi_barrier()
        else:
            raise RuntimeError("Not implemented")

from .scipy import ScipyEigenSolver,DefaultMatrixType

class PardisoInvOp(object):
    def __init__(self, A:DefaultMatrixType, M:DefaultMatrixType | None=None,sigma:float | complex | None=None,mode:int=11):
        if sigma is None:
            self.mat=A
        else:
            self.mat=A-sigma*M #type:ignore
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
            OPinv = PardisoInvOp(J, M, sigma=shift,mode=mode)
        return OPinv


