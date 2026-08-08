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
 
from .generic import GenericLinearSystemSolver, GenericEigenSolver, EigenSolverWhich,DefaultMatrixType, SolverError
import atexit
import hashlib
from collections import OrderedDict
import petsc4py #type:ignore
import sys

# The command line is handed on so that PETSc/SLEPc options can be given on it -- but only the
# single-dashed part of it. PETSc records EVERY dash-prefixed token it is given, whether or not it
# knows the name, and lists the ones nothing read at PetscFinalize as possible spelling mistakes; with
# the raw argv that meant a plain `python3 script.py --petsc_mumps --outdir out` ended in a warning
# about pyoomph's own flags. Dropping the double-dashed ones costs nothing, because PETSc files
# `--ksp_type` under a name its own lookups never match (hasName("ksp_type") is False for it), so a
# double-dashed PETSc option has never taken effect anyway.
#
# Every other single-dashed token is passed through untouched -- that is how PETSc options are written
# -- except pyoomph's one short flag, -P (i.e. --parameter), which is indistinguishable from a PETSc
# option and would be reported on every run that uses it. Only the flag itself has to go: PETSc ignores
# tokens that do not start with a dash, so the values left behind by a dropped flag are already inert.
_petsc_argv=[arg for arg in sys.argv if not (arg.startswith("--") or arg=="-P")]

petsc4py.init(_petsc_argv) #type:ignore

import slepc4py #type:ignore

slepc4py.init(_petsc_argv) #type:ignore

from petsc4py import PETSc #type:ignore
from slepc4py import SLEPc #type:ignore
from ..generic.mpi import *
from ..typings import *
import numpy

if TYPE_CHECKING:
    from ..generic.problem import Problem


class PETScSolverError(SolverError):
    """PETSc could not solve the system -- a failed factorisation, an iterative solve that gave up.

    A SolverError rather than a plain RuntimeError, so that an adaptive time step or an arclength step
    rejects and retries with a smaller one instead of the run ending here. Deliberately NOT used for
    the configuration errors in this module (no MUMPS in this PETSc build, an unmatched field-split
    name): no smaller step fixes those, and retrying would bury the message.
    """



# MUMPS INFOG(1) codes that all say the same thing: an internal work array was sized from the fill-in
# the analysis phase predicted, numerical pivoting then needed more room than that, and MUMPS stopped
# instead of reallocating. Its own manual's answer is to raise ICNTL(14) -- the percentage of slack
# added to the prediction -- and rerun.
_MUMPS_ICNTL14_ERRORS = (-8, -9, -11, -12, -14, -15, -17, -20)

# INFOG(1) = -19 is deliberately NOT in that list although it reads like a memory error. It means the
# factorisation exceeded ICNTL(23), the hard cap on working memory in MB, and raising ICNTL(14) against
# a cap only asks for more of something already forbidden. The lever there is ICNTL(23) itself, which
# is the user's to set (pyoomph never does), so this is reported rather than escalated.
_MUMPS_ICNTL23_ERROR = -19


def _mumps_infog_from_pc(pc:Any,which:int)->int | None:
    """MUMPS' INFOG(which) from a PC's factor matrix, or None if MUMPS did not do the factorisation.

    Two gates before anything MUMPS-specific is touched, because neither PETSc nor petsc4py need have
    MUMPS at all:

    * the installation must have been built with it -- ``PETSc.Sys.hasExternalPackage``, the same check
      ``use_mumps()`` makes before configuring it;
    * and this PC must actually be running it. getFactorSolverType() answers that without raising: it
      is None for a PC that has no factor matrix in the first place (jacobi, gamg, hypre, none), where
      getFactorMatrix() raises PETSc error 56, and it names the package for the ones that do, so a
      SuperLU or UMFPACK factorisation is turned away rather than being asked for an INFOG it does not
      have.

    Everything MUMPS-specific in this file hangs off a non-None answer from here, so on a PETSc without
    MUMPS none of it is ever reached.
    """
    if pc is None:
        return None
    if not PETSc.Sys.hasExternalPackage("mumps"): #type:ignore
        return None
    try:
        if str(pc.getFactorSolverType()).lower() != "mumps": #type:ignore
            return None
        return int(pc.getFactorMatrix().getMumpsInfog(which)) #type:ignore
    except Exception:
        return None   # No factor matrix yet, or a petsc4py too old for the MUMPS accessors


def _increase_mumps_icntl14(option:str,quiet:bool)->bool:
    """Double the ICNTL(14) held under `option`, and report whether it actually moved.

    Set for the rest of the run rather than only for the retry: a matrix that needed the extra room
    once will need it again at the next step, and re-failing each time to rediscover that is waste.
    The option name differs per caller -- ``mat_mumps_icntl_14`` for the linear solver, and the
    ``st_``-prefixed one for the KSP that lives inside SLEPc's spectral transform.
    """
    opts = PETSc.Options() #type:ignore
    current = 20   # MUMPS' own default, which PETSc does not override
    if opts.hasName(option): #type:ignore
        try:
            current = int(opts.getInt(option)) #type:ignore
        except Exception:
            pass
    new_value = min(max(2*current, 40), 1000)
    if new_value == current:
        return False
    _SetDefaultPetscOption(option, new_value, force=True)
    if not quiet:
        print("MUMPS ran out of working space; retrying with -" + option + " " + str(new_value))
    return True


@GenericLinearSystemSolver.register_solver()
class PETSCSolver(GenericLinearSystemSolver):
    idname = "petsc"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self._do_not_set_any_args:bool=False
        self.petsc_mat=None
        self.petsc_rhs=None
        self.ksp=None
        self.x=None

        self._dofs_to_field_info=None

        # Keep the Mat (and the KSP built on it) across solves whenever the Jacobian sparsity pattern is
        # unchanged, updating only the values. This is what lets PETSc detect SAME_NONZERO_PATTERN and
        # reuse the symbolic factorisation / preconditioner setup instead of redoing it every Newton
        # step. Requires problem.keep_structural_zeros; without it jacobian_structure_id is 0 and this
        # switches itself off, so the behaviour is unchanged for existing scripts.
        self.reuse_matrix_structure=True
        self._structure_id:int=0   # Pattern the current Mat/KSP were built for; 0 = none
        self._structure_nnz:int=-1
        self._structure_nrow_local:int=-1
        self._structure_digest:bytes | None=None   # Fingerprint of the pattern the Mat was built for

        # Whether a factorisation that has failed twice (see _ksp_solve_checked) raises, rather than
        # letting the untouched solution vector travel on as if it were an answer. Only ever consulted
        # for a genuine factorisation failure -- an iterative KSP that merely stops on its iteration
        # limit is never affected by this, whichever way it is set.
        self.raise_on_failed_solve=True

    #		opts=PETSc.Options().getAll()
    #		if "add_zero_diagonal" in opts.keys():
    #			problem.set_diagonal_zero_entries(True)

    # Factorisation preconditioners whose PETSc-native implementation walks the diagonal explicitly and
    # errors out if an entry is missing (MatLUFactorSymbolic_SeqAIJ and friends). Iterative and
    # multigrid PCs are not listed: they do not factorise the matrix themselves.
    _FACTORISING_PC_TYPES = ("lu", "ilu", "cholesky", "icc")
    # Third-party factorisation packages, which build their own internal structure from the CSR and do
    # not require the caller to have supplied a diagonal. MUMPS in particular does not.
    _EXTERNAL_FACTOR_PACKAGES = ("mumps", "superlu", "superlu_dist", "pastix", "cholmod", "umfpack",
                                 "klu", "mkl_pardiso", "mkl_cpardiso", "strumpack", "sparseelemental")

    def requires_explicit_diagonal(self)->bool:
        """True only when PETSc's OWN factorisation is in play.

        Deciding this from the options database rather than hard-coding it per solver class is what
        makes ``petsc_mumps`` and a hand-configured ``-pc_type lu -pc_factor_mat_solver_type mumps``
        give the same (correct) answer. Every ``*pc_type`` option is scanned, not just the top-level
        one, so a factorisation sitting under a fieldsplit is still seen.

        Erring towards False is the safe direction: an unnecessary yes costs stored zeros on every
        diagonal and perturbs the factoriser's pivoting, whereas a wrong no surfaces as PETSc's own
        explicit "Matrix is missing diagonal entry" error, which the user can answer with
        ``problem.force_jacobian_diagonal_entries = True``.
        """
        opts = PETSc.Options().getAll() #type:ignore
        factor_package = None
        for key, val in opts.items(): #type:ignore
            if key.endswith("pc_factor_mat_solver_type"):
                factor_package = str(val).lower()
                break
        if factor_package is not None and factor_package in self._EXTERNAL_FACTOR_PACKAGES:
            return False  # An external package builds its own structure; MUMPS is the common case here
        for key, val in opts.items(): #type:ignore
            if key.endswith("pc_type") and str(val).lower() in self._FACTORISING_PC_TYPES:
                return True
        # setup_solver() falls back to pc_type "lu" when nothing has been configured, and with no
        # factor package that is PETSc's own LU, which does need the diagonal.
        if not self._do_not_set_any_args and factor_package is None:
            return True
        return False

    def _force_zero_diagonal(self,mat:Any)->None:
        # Deliberately does nothing to the matrix. The diagonal, when this solver needs one, is supplied
        # by the ASSEMBLY (problem.force_jacobian_diagonal_entries, driven by requires_explicit_diagonal
        # above) rather than patched in afterwards.
        #
        # The historical approach was mat.shift(0.0), which does not work: MatShift returns early on a
        # zero shift, so it never inserted anything -- verified on every combination of matrix options,
        # including MAT_NEW_NONZERO_ALLOCATION_ERR=False. Kept as a named hook so the intent is
        # documented at the two call sites rather than being a silent omission.
        return

    def _remember_structure(self,indptr:Any,indices:Any)->None:
        """Keep a fingerprint of the pattern the current Mat was preallocated for.

        A hash rather than the arrays themselves: PETSc already stores the indices internally, so
        keeping a second copy would add (nnz + nrow) integers to a Mat that is often the largest
        object in the run, for insurance that is only ever read once per solve. blake2b over the two
        index arrays runs at memory bandwidth -- microseconds against the milliseconds a numeric
        factorisation takes -- and a 128-bit digest makes an undetected change not a practical concern.
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(numpy.ascontiguousarray(indptr).view(numpy.uint8))
        h.update(numpy.ascontiguousarray(indices).view(numpy.uint8))
        self._structure_digest = h.digest()

    def _structure_matches(self,indptr:Any,indices:Any)->bool:
        """Whether the pattern about to be written is the one the Mat was preallocated for.

        Verified rather than trusted, for the same reason the Pardiso path verifies it: a stale
        jacobian_structure_id would otherwise be a silently wrong matrix, not a crash. It is worse
        here than for a factorisation, because setValuesCSR only writes the entries it is given --
        anything the previous pattern had and this one does not keeps its OLD value -- and pyoomph
        disables NEW_NONZERO_ALLOCATION_ERR, so PETSc will not object to the ones it has never seen
        either. Neither the nnz nor the size check catches a pattern that changed shape without
        changing its nonzero count, which is exactly what an augmented (bifurcation-tracking) system
        can do: there the elemental block is larger than the field description, so no symbolic mask
        applies and the pattern falls back to being value-filtered.
        """
        if self._structure_digest is None:
            return False
        h = hashlib.blake2b(digest_size=16)
        h.update(numpy.ascontiguousarray(indptr).view(numpy.uint8))
        h.update(numpy.ascontiguousarray(indices).view(numpy.uint8))
        if h.digest() == self._structure_digest:
            return True
        self._report_structure_id_mismatch("the PETSc matrix")
        return False

    def _can_reuse_structure(self,n:int,nnz:int,indptr:Any=None,indices:Any=None)->bool:
        structure_id = self.problem.jacobian_structure_id
        if not (self.reuse_matrix_structure and structure_id != 0
                and structure_id == self._structure_id
                and self.petsc_mat is not None
                and self._structure_nnz == nnz
                and self.petsc_mat.getSize()[0] == n): #type:ignore
            return False
        if indptr is None:
            return True
        return self._structure_matches(indptr,indices)

    def _can_reuse_structure_distributed(self,nrow_local:int,nnz_local:int,indptr:Any=None,indices:Any=None)->bool:
        structure_id = self.problem.jacobian_structure_id
        if not (self.reuse_matrix_structure and structure_id != 0
                and structure_id == self._structure_id
                and self.petsc_mat is not None
                and self._structure_nnz == nnz_local
                and self._structure_nrow_local == nrow_local):
            return False
        if indptr is None:
            return True
        return self._structure_matches(indptr,indices)

    def _agree_on_reuse_structure_distributed(self,nrow_local:int,nnz_local:int,indptr:Any=None,indices:Any=None)->bool:
        """Collective form of _can_reuse_structure_distributed: reuse only if EVERY rank can.

        The two branches this picks between -- update the values in place, or destroy the Mat/KSP and
        rebuild -- are both collective, so a rank that answers differently from the others does not
        merely lose the reuse, it hangs the job in mismatched PETSc collectives (one rank in
        MatAssemblyBegin while another is inside KSPDestroy). The per-rank inputs are meant to agree,
        but they are derived from the LOCAL sparsity pattern and so cannot be trusted to: a genuine
        pattern change that only shows up on some ranks is a possible outcome, not a contradiction.
        One allreduce of a bool per solve is nothing next to a factorisation, and disagreeing now
        costs a rebuild instead of a deadlock.
        """
        local=self._can_reuse_structure_distributed(nrow_local,nnz_local,indptr,indices)
        if get_mpi_nproc()>1 and get_mpi_any(not local):
            return False
        return local

    def _before_assigning_equation_numbers(self):
        if self._dofs_to_field_info is not None:
            if len(self._dofs_to_field_info)>2:
                for IS in self._dofs_to_field_info[2].values():                    
                    IS.destroy() #type:ignore
        self._dofs_to_field_info=None # Reset the mapping, it will be re-created when needed. This is needed to properly handle changes in the dofs due to e.g. field splits or changes in the meshes
        return super()._before_assigning_equation_numbers()
    
    def use_mumps(self,mumps_param14:int | None=None):
        if not PETSc.Sys.hasExternalPackage("mumps"): #type:ignore
            raise RuntimeError("Your PETSc installation was not compiled with MUMPS support (--download-mumps=yes). Please recompile PETSc with MUMPS or use a different linear solver.")
        _SetDefaultPetscOption("mat_mumps_icntl_6",5)
        # ICNTL(24)=1 -- detect and handle null pivots.
        #
        # Needed because pyoomph deliberately stores STRUCTURAL zeros (problem.keep_structural_zeros),
        # so that the sparsity pattern is a function of the equation numbering alone and can be reused
        # across Newton steps. On a saddle-point system that turns a genuinely absent diagonal into a
        # diagonal entry that is present and exactly zero -- interface Lagrange multipliers and
        # pressure dofs, whose self-coupling is zero by construction. MUMPS chooses its elimination
        # order from the STRUCTURE, so those stored zeros invite it to plan an elimination that then
        # hits a zero pivot at factorisation time; with detection off (the default) it divides by it
        # and returns a silently wrong solution, and Newton diverges from a nearly converged state.
        # Measured on the two-domain ALE case: INFOG(28) reports 8 null pivots encountered.
        #
        # Costs nothing on a matrix that has no null pivots, and only ever replaces a garbage answer
        # with a usable one. Note the flip side: on a genuinely singular system MUMPS now returns a
        # pseudo-solution rather than producing nonsense, so a singular problem shows up as a Newton
        # that fails to converge rather than one that diverges spectacularly.
        _SetDefaultPetscOption("mat_mumps_icntl_24",1)
        _SetDefaultPetscOption("ksp_type","preonly")
        _SetDefaultPetscOption("pc_type","lu")
        _SetDefaultPetscOption("pc_factor_mat_solver_type","mumps")
        if mumps_param14 is not None:
            _SetDefaultPetscOption("mat_mumps_icntl_14",mumps_param14)
        return self

    def set_options(self,**kwargs:Any):
        for a,b in kwargs.items():
            PETSc.Options().setValue(a,b) #type:ignore
            
    def set_default_petsc_option(self,name:str,val:Any=None,force:bool=False)->None:
        _SetDefaultPetscOption(name,val, force) #type:ignore

    def get_PETSc(self)->Any:
        """
        Returns access to PETSc
        If defining derived classes that need access to PETSc, get PETSc from here, do not import petsc4py again
        """
        return PETSc
    
    def _field_split_required(self)->bool:
        # The field-index sets (IS) built by setup_field_split() are only ever consumed to configure a
        # PETSc "fieldsplit" preconditioner (via pc.setFieldSplitIS in setup_solver, or by a derived
        # solver calling get_field_split_IS). Building them is not free: it triggers an O(ndof) sweep
        # over every submesh in the problem (see Problem::get_dof_to_global_field_index_mapping) on every
        # solve after the equations have been (re)assigned. So only do it when a fieldsplit PC is actually
        # in play, which happens in two ways:
        #   (a) the user requested a named split via problem.petsc_fieldsplit, or
        #   (b) a fieldsplit preconditioner was selected through PETSc options, e.g. -pc_type fieldsplit
        #       (also matches a prefixed option such as -fieldsplit_..._pc_type fieldsplit). This covers
        #       options set programmatically via PETSc.Options()[...] too, not just the command line.
        # NOTE: this runs before setup_solver creates the PC, so a fieldsplit PC configured purely by hand
        # on the PC object (pc.setType("fieldsplit") in an overridden setup_solver, with neither signal
        # above set) is invisible here. Such a subclass should either set one of the two signals above, or
        # fetch its indices via get_field_split_IS(), which builds the mapping lazily on demand.
        if self.problem.petsc_fieldsplit is not None:
            return True
        for key, val in PETSc.Options().getAll().items(): #type:ignore
            if key.endswith("pc_type") and val == "fieldsplit":
                return True
        return False

    def setup_field_split(self):
        if not self.problem.is_quiet():
            print("Setting up field split for PETSc solver")
        def process_indices(indices,name):
            if get_mpi_nproc()<2:
                return indices
            else:
                ownership_range=self.petsc_mat.getOwnershipRange() #type:ignore                
                #print("OWNERSHIP RANGE",name,get_mpi_rank(),ownership_range,"TOTAL INDICES",len(indices)) #type:ignore
                #print("On rank",name,get_mpi_rank(), "ALL INDICES FOR FIELD SPLIT: ", indices) #type:ignore
                #if ownership_range[0]>0 or ownership_range[1]<self.petsc_mat.getSize()[0]:
                my_indices=indices[(indices < ownership_range[1]) & (indices >= ownership_range[0])]                    
                #print("PROCESSED INDICES FOR FIELD SPLIT ON RANK",name, get_mpi_rank(),": ","LEN",len(my_indices),my_indices) #type:ignore                
                return numpy.sort(my_indices)
                
        names=self.problem._get_global_field_names()
        mapping=numpy.array(self.problem._get_dof_to_global_field_index_mapping())            
        #print("Global field names:", names)
        #print("DOF to field mapping:", get_mpi_rank(),mapping)
        unique_fields=numpy.unique(mapping)
        unique_fields=unique_fields[unique_fields>=0] # Filter out any dofs that are not assigned to a field (e.g. due to field splits, where some dofs might be assigned to a new field index of -1 or similar)
        # self.problem.petsc_fieldsplit is only ever assigned None in Problem.__init__ (its declared
        # type there is therefore just "None"), but user code may set it to a dict at runtime (see the
        # docstring on that attribute). Capture it into a locally, correctly-typed variable here instead
        # of widening the type in problem.py (out of scope for this file).
        petsc_fieldsplit:dict[str,Any] | None = self.problem.petsc_fieldsplit
        if petsc_fieldsplit is None:
            if not self.problem.is_quiet():
                print("Using default PETSc DOF to field mapping:")
                for uf in unique_fields:
                    print("  Field "+str(uf)+": "+names[uf])
            field_is={}
            for f in unique_fields:
                indices = numpy.where(mapping == f)[0].astype(numpy.int32)
                iset = PETSc.IS().createGeneral(process_indices(indices,names[f]),comm=PETSc.COMM_WORLD) #type:ignore
                field_is[f] = iset

        else:
            if not self.problem.is_quiet():
                print("Using user-defined PETSc DOF to field mapping:",petsc_fieldsplit)
            field_is={}
            is_collections={}
            handled_fields=set()
            for k,v in petsc_fieldsplit.items():
                if not v in is_collections.keys():
                    is_collections[v]=[]
                if "*" in k:
                    import fnmatch
                    matches = fnmatch.filter(names, k)
                    if len(matches)==0:
                        raise RuntimeError("Cannot find any field matching "+k+" specified in petsc_fieldsplit")
                    for m in matches:
                        if m in handled_fields:
                            raise RuntimeError("Field "+str(m)+" is already assigned to a field split. Cannot assign it again with "+k)
                        is_collections[v].append(m)
                        handled_fields.add(m)
                else:                                        
                    if k in handled_fields:
                        raise RuntimeError("Field "+str(k)+" is already assigned to a field split. Cannot assign it again with "+k)
                    if k not in names:
                        raise RuntimeError("Cannot find the field "+k+" specified in petsc_fieldsplit")
                    is_collections[v].append(k)
                    handled_fields.add(k)
                    
            if len(handled_fields)<len(unique_fields):
                raise RuntimeError("Not all fields are assigned to a field split.\nUnassigned fields: "+str(set(names)-handled_fields)+"\nHandled fields are: "+str(handled_fields))
            
            for v, fields in is_collections.items():
                v=str(v)
                mergedindices=set(names.index(f) for f in fields)
                #indices = numpy.where(mapping in mergedindices)[0].astype(numpy.int32)                        
                indices= numpy.where(numpy.isin(mapping, list(mergedindices)))[0].astype(numpy.int32)     
                #print("ON",get_mpi_rank(), "INDICES FOR FIELD "+str(v)+": "+str(indices),"LEN",len(indices))
                #print("CHECKING ON RANK",v,get_mpi_rank(),mapping[indices])                
                #print("PROCESSES ON RANK",v,get_mpi_rank(),mapping[process_indices(indices,v)])                
                iset = PETSc.IS().createGeneral(process_indices(indices,v),comm=PETSc.COMM_WORLD) #type:ignore
                field_is[v] = iset
                if not self.problem.is_quiet():
                    print("  Field "+str(v)+": "+str(fields))
                #print("    mapping", mapping[indices])
                #print("    IS size",iset.getSize(),len(indices))
                #print()
        self._dofs_to_field_info=[names,mapping,field_is]
        
    def get_field_split_IS(self,splitname:str)->PETSc.IS: #type:ignore
        # Lazily build the field split on demand. _field_split_required() (checked before solve) only
        # sees fieldsplit requested via problem.petsc_fieldsplit or the PETSc options database -- it runs
        # before setup_solver and so cannot detect a fieldsplit PC configured by hand on the PC object.
        # A subclass whose overridden setup_solver calls this method to fetch the IS is exactly such a
        # manual path, so build the mapping here rather than relying on the detector having fired first.
        if self._dofs_to_field_info is None:
            self.setup_field_split()
        assert self._dofs_to_field_info is not None
        if splitname not in self._dofs_to_field_info[2].keys():
            raise RuntimeError("Requested field split "+splitname+" not found. Available splits: "+str(self._dofs_to_field_info[2].keys()))
        return self._dofs_to_field_info[2][splitname]

    def _setup_solver_if_needed(self):
        # setup_solver() builds a brand new KSP (and PC) on every call, which throws away any
        # factorisation or preconditioner PETSc has already computed. When the matrix structure is being
        # reused, the Mat object survives across solves, so the KSP built on it stays valid too and
        # PETSc can reuse the symbolic factorisation -- keep it. Without structure reuse this falls back
        # to the previous behaviour (a fresh KSP per solve) exactly, so that options set between solves
        # and subclasses overriding setup_solver() keep behaving as before.
        if self.ksp is not None and self.reuse_matrix_structure and self._structure_id != 0 \
                and self._structure_id == self.problem.jacobian_structure_id:
            return
        self.setup_solver()

    # The only two failures worth acting on. Both mean the vector PETSc handed back is not a solution
    # of anything, and both can be cured by factorising again from scratch.
    #
    # Everything else a KSP can report negative is deliberately left alone, DIVERGED_MAX_IT above all:
    # an iterative solver stopping on its iteration limit has produced a real, merely inaccurate,
    # iterate, and Newton is often perfectly happy with it. Rebuilding the KSP there would throw away a
    # preconditioner (a hypre/GAMG setup is not cheap) and re-run the same solve to reach the same
    # place, and aborting on it would break any hand-configured iterative solver that has always been
    # allowed to return early. Those cases get a warning and their iterate, exactly as before.
    _RECOVERABLE_KSP_FAILURES = (PETSc.KSP.ConvergedReason.DIVERGED_PCSETUP_FAILED, #type:ignore
                                 PETSc.KSP.ConvergedReason.DIVERGED_NANORINF)       #type:ignore

    def _mumps_infog(self,which:int)->int | None:
        """MUMPS' INFOG(which) from the current factorisation, or None if MUMPS did not do it."""
        return _mumps_infog_from_pc(self.ksp.getPC() if self.ksp is not None else None, which) #type:ignore

    def _ksp_solve_checked(self,bv:Any)->None:
        """Solve, and turn a failed *factorisation* into a retry, and then into an error.

        PETSc reports a failed factorisation by setting a negative KSPConvergedReason and returning
        immediately, leaving the solution vector exactly as it found it. Nothing downstream can tell
        that apart from a real solution, so it used to reach the Newton solver as an ``inf`` residual
        with nothing to say where it came from.

        Only the two reasons in _RECOVERABLE_KSP_FAILURES are acted on; every other way a KSP can end
        negative keeps its old behaviour and merely gains a warning.

        The retry is not cosmetic; it is what makes the failure recoverable at all. Keeping the KSP
        alive across solves (see _setup_solver_if_needed) means PETSc sees SAME_NONZERO_PATTERN and
        reuses MUMPS' *analysis* -- and with ICNTL(6)=5 that analysis is value-dependent: it fixes a
        maximum-transversal permutation and a scaling from the matrix it first saw. As a continuation
        walks the values away from that matrix the ordering stops being a good one, off-diagonal
        pivoting grows (INFOG(12) went 288 -> 2638 on the case this was written for) and the fill-in
        finally exceeds the workspace that stale analysis predicted: INFOG(1) = -9. Discarding the KSP
        forces a fresh analysis for the values actually on hand, which is what makes the retry work.

        Without that, the failure is permanent, not transient: the failed PC stays cached, so every
        later solve returns in microseconds with the same error. An arclength continuation then halves
        its step against a solver that can no longer solve anything, all the way down to its minimum
        step, and reports a spurious "arc-length step has fallen below minimum tolerance".

        Rebuilding rather than switching the reuse off is deliberate: on that same case the reused
        symbolic factorisation was worth 2.7x on the in-process KSP solve time (2.4 s against 6.5 s
        over the run), so the fast path is worth keeping and paying for a rebuild when it goes stale.

        Safe under MPI: KSPSolve reduces the PC failure across the communicator, so getConvergedReason
        agrees on every rank and all of them take the same (collective) rebuild branch.
        """
        assert self.ksp is not None
        self.ksp.solve(bv, self.x) #type:ignore
        reason:int = self.ksp.getConvergedReason() #type:ignore
        if reason >= 0:
            return

        def described(r:int)->str:
            infog1 = self._mumps_infog(1)
            return ("KSPConvergedReason " + str(r)
                    + (" (MUMPS INFOG(1)=" + str(infog1) + ")" if infog1 is not None else ""))

        if reason not in self._RECOVERABLE_KSP_FAILURES:
            # An unconverged iterate, not a failed factorisation. Say so and hand it on unchanged --
            # this is the behaviour every configuration had before the check existed.
            print("WARNING: the PETSc linear solve did not converge (" + described(reason)
                  + "). The solution it returned is being used as it is.")
            return

        if not self.problem.is_quiet():
            print("PETSc linear solve failed (" + described(reason)
                  + "); discarding the cached factorisation and retrying")
        infog1 = self._mumps_infog(1)
        if infog1 is not None and infog1 in _MUMPS_ICNTL14_ERRORS:
            _increase_mumps_icntl14("mat_mumps_icntl_14", self.problem.is_quiet())
        elif infog1 == _MUMPS_ICNTL23_ERROR:
            print("MUMPS hit the working-memory cap set by -mat_mumps_icntl_23. Raise or remove that "
                  "cap; ICNTL(14) cannot buy room a cap forbids.")

        self.ksp.destroy() #type:ignore
        self.ksp = None
        self.setup_solver()
        assert self.ksp is not None
        self.ksp.solve(bv, self.x) #type:ignore
        reason = self.ksp.getConvergedReason() #type:ignore
        if reason >= 0:
            return

        msg = ("The PETSc linear solver failed: " + described(reason)
               + ", also on a retry with a freshly built factorisation. The solution vector is "
                 "meaningless, so the solve is aborted here rather than handing it to the Newton "
                 "solver as an inf/nan residual. Set "
                 "problem.get_la_solver().raise_on_failed_solve=False to downgrade this to a warning.")
        if self.raise_on_failed_solve:
            raise PETScSolverError(msg)
        print("WARNING: " + msg)

    def setup_solver(self):
        #print("Setting up solver")
        opts = PETSc.Options().getAll() #type:ignore
        #if "add_zero_diagonal" in opts.keys(): #type:ignore
            #			print(dir(self.petsc_mat))
        #    self.petsc_mat.setOption(19, 0) #type:ignore
        #    self.petsc_mat.shift(0) #type:ignore

        self.ksp = PETSc.KSP().create() #type:ignore
        self.ksp.setOperators(self.petsc_mat) #type:ignore
        if not self._do_not_set_any_args:
            self.ksp.setType('preonly') #type:ignore
        pc = self.ksp.getPC() #type:ignore
        if not self._do_not_set_any_args:
            pc.setType('lu') #type:ignore
        opts = PETSc.Options().getAll() #type:ignore
        if "pc_factor_mat_solver_type" in opts.keys(): #type:ignore
            if hasattr(pc, "setFactorSolverPackage"): #type:ignore
                if not self._do_not_set_any_args: #type:ignore
                    pc.setFactorSolverPackage(opts["pc_factor_mat_solver_type"]) #type:ignore

        pc.setFromOptions() #type:ignore
        if self._dofs_to_field_info is not None:
            field_is=self._dofs_to_field_info[2]
            splt=[(str(a),b) for a,b in field_is.items()]
            pc.setFieldSplitIS(*splt) #type:ignore
        self.ksp.setFromOptions() #type:ignore
        #print('Solving with:', self.ksp.getType())  # ,dir(pc)

    def solve_serial(self,op_flag:int,n:int,nnz:int,nrhs:int,values:NPFloatArray,rowind:NPIntArray,colptr:NPIntArray,b:NPFloatArray,ldb:int,transpose:int)->int:
        if op_flag == 1:
            if self._can_reuse_structure(n,nnz,colptr,rowind):
                # Same nonzero pattern, new values. Overwriting them in place (rather than destroying
                # and rebuilding the Mat) keeps the KSP's factorisation/preconditioner reusable.
                self.petsc_mat.setValuesCSR(colptr.astype(PETSc.IntType, copy=False), rowind.astype(PETSc.IntType, copy=False), values.astype(PETSc.ScalarType, copy=False)) #type:ignore
                self.petsc_mat.assemble() #type:ignore
                return 0
            if self.petsc_mat is not None:
                self.petsc_mat.destroy()
                self.petsc_mat=None
            if self.ksp is not None:
                self.ksp.destroy() #type:ignore
                self.ksp=None
            if self.x is not None:
                self.x.destroy() #type:ignore
                self.x=None
            # NOTE: _dofs_to_field_info (the field-split index sets) is deliberately NOT reset here.
            # The IS only depend on the global DOF numbering, which changes solely when the equations
            # are reassigned -- and that path already invalidates them in _before_assigning_equation_numbers.
            # Recreating the matrix (this op_flag==1) with an unchanged numbering leaves the IS valid, so
            # keeping them cached avoids rebuilding the (O(ndof)) field map on every single solve.

            # The CSR arrays arrive as zero-copy numpy views onto oomph-lib's CRDoubleMatrix buffers
            # (see src/nanobind/solver.cpp) and are normally already in PETSc's own index/scalar
            # dtypes. astype(..., copy=False) is then a no-op that returns those same views; it only
            # allocates a converted copy when the dtypes genuinely differ (e.g. a 64-bit-index or
            # complex PETSc build). Targeting PETSc.IntType/ScalarType keeps it correct on any build,
            # and avoids the redundant ~nnz*12 byte transient copy the old hard-coded astype made.
            # (createAIJ then copies the CSR into PETSc's own AIJ storage. A zero-copy
            # MatCreateSeqAIJWithArrays was tried but silently breaks hypre/BoomerAMG -- CG diverges
            # with KSP reason DIVERGED_ITS -- so we keep the safe, universally-correct copying path.)
            self.petsc_mat = PETSc.Mat().createAIJ(size=(n, n), csr=(colptr.astype(PETSc.IntType, copy=False), rowind.astype(PETSc.IntType, copy=False), values.astype(PETSc.ScalarType, copy=False)),comm=get_mpi_world_comm()) #type:ignore

            self.petsc_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False) #type:ignore
            # Force diagonal:
            #diag = self.petsc_mat.getDiagonal()
            #self.petsc_mat.setDiagonal(diag, addv=PETSc.InsertMode.INSERT_VALUES)
            self._force_zero_diagonal(self.petsc_mat)

            self.petsc_mat.assemble()
            self._structure_id = self.problem.jacobian_structure_id
            self._structure_nnz = nnz
            self._remember_structure(colptr,rowind)

            self.x = PETSc.Vec().createSeq(n) #type:ignore
        elif op_flag == 2:
            #print("Solving linear system with PETSc", op_flag, n, nnz, nrhs, transpose, "SPLIT INFO",self._dofs_to_field_info)
            # _dofs_to_field_info is reset to None whenever the equations are reassigned
            # (_before_assigning_equation_numbers), so this only recomputes the field split after a
            # reassignment, and only when a fieldsplit preconditioner actually needs the indices.
            if self._dofs_to_field_info is None and self._field_split_required():
                self.setup_field_split()
            bv = PETSc.Vec().createWithArray(b) #type:ignore
            self.petsc_rhs=bv
            self._setup_solver_if_needed()

            if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
                raise RuntimeError("Cannot use custom solve routine with PETSc yet. Also, iterative solving might require different handling here")
            else:
                import time
                start_time = time.time()
                self._ksp_solve_checked(bv)
                end_time = time.time()
                if not self.problem.is_quiet():
                    print("PETSc KSP solve time:", end_time - start_time, "seconds")
                xv = self.x.getArray() #type:ignore
            # On a complex PETSc build (the one the eigensolvers need) the solution vector is complex
            # even though this system is real, so the imaginary part is pure roundoff -- drop it
            # explicitly instead of letting numpy discard it with a ComplexWarning.
            b[:] = xv.real if xv.dtype.kind == "c" and b.dtype.kind != "c" else xv #type:ignore

            #print('Converged in', self.ksp.getIterationNumber(), 'iterations.') #type:ignore


            self.petsc_rhs=None
            bv.destroy() #type:ignore

        else:
            raise RuntimeError("Cannot handle Petsc mode " + str(op_flag) + " yet")
        return 0  # TODO: Return sign of Jacobian

    def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:
        #print("solve distributed with flag ",op_flag)
        if op_flag == 1:
            # Same reuse logic as solve_serial, but taken collectively: rebuilding the Mat is itself
            # collective, so a split decision deadlocks rather than just losing the reuse. The inputs
            # (jacobian_structure_id, local row count, local nnz, pattern digest) are all meant to
            # agree across ranks, but the last three are LOCAL quantities and are not trusted to.
            if self._agree_on_reuse_structure_distributed(nrow_local,nnz_local,row_start,col_index):
                self.petsc_mat.setValuesCSR(row_start.astype(PETSc.IntType, copy=False), col_index.astype(PETSc.IntType, copy=False), values.astype(PETSc.ScalarType, copy=False)) #type:ignore
                self.petsc_mat.assemble() #type:ignore
                return
            if self.petsc_mat is not None:
                self.petsc_mat.destroy()
                self.petsc_mat=None
            if self.ksp is not None:
                self.ksp.destroy() #type:ignore
                self.ksp=None
            if self.x is not None:
                self.x.destroy() #type:ignore
                self.x=None
            # See solve_serial: the field-split IS depend only on the DOF numbering, so they are
            # invalidated in _before_assigning_equation_numbers (on reassignment), not on every matrix rebuild.
            #print("PETSCINF",nrow_local,n)
            #print("Creating petsc mat ")
            # astype(..., copy=False) rather than the raw arrays: on a PETSc built with 64-bit indices
            # (or complex scalars) the int32/float64 arrays oomph hands over are the wrong dtype, and
            # this is the only createAIJ in the file that was still passing them through unconverted --
            # so the distributed path disagreed with both the serial one and its own update call below.
            # On a matching build every conversion is a no-op returning the same view.
            self.petsc_mat = PETSc.Mat().createAIJ(size=((nrow_local, n), (nrow_local, n),),
                                                   csr=(row_start.astype(PETSc.IntType, copy=False),
                                                        col_index.astype(PETSc.IntType, copy=False),
                                                        values.astype(PETSc.ScalarType, copy=False))) #type:ignore

            self.petsc_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False) #type:ignore
            # Force diagonal:
            #diag = self.petsc_mat.getDiagonal()
            #self.petsc_mat.setDiagonal(diag, addv=PETSc.InsertMode.INSERT_VALUES)
            self._force_zero_diagonal(self.petsc_mat)

            self.petsc_mat.assemble()
            self._structure_id = self.problem.jacobian_structure_id
            self._structure_nnz = nnz_local
            self._structure_nrow_local = nrow_local
            self._remember_structure(row_start,col_index)

            #print("OWNERSHIP RANGE",self.petsc_mat.getOwnershipRange()) #type:ignore
        #			print("PROCESSOR Ns",get_mpi_rank(),nrow_local,n)
        #			print("PROCESSOR RS",get_mpi_rank(),row_start)
        #			print("PROCESSOR CI",get_mpi_rank(),col_index)
        #			print("FIRST ROW",get_mpi_rank(),first_row)
        # self.petsc_mat
            
        elif op_flag == 2:

            # See solve_serial: only (re)build the field split after an equation reassignment, and only
            # when a fieldsplit preconditioner actually needs the field indices.
            if self._dofs_to_field_info is None and self._field_split_required():
                self.setup_field_split()
            bv = PETSc.Vec().createWithArray(b) #type:ignore
            self.petsc_rhs=bv
            self.x = self.petsc_rhs.duplicate()

            self._setup_solver_if_needed()

            if self.problem._custom_assembler is not None and self.problem._custom_assembler.has_custom_solve_routine():
                raise RuntimeError("Cannot use custom solve routine with PETSc yet. Also, iterative solving might require different handling here")
            else:
                import time
                start_time = time.time()
                self._ksp_solve_checked(bv)
                end_time = time.time()
                if not self.problem.is_quiet():
                    print("PETSc KSP solve time:", end_time - start_time, "seconds")
                xv = self.x.getArray() #type:ignore
            # On a complex PETSc build (the one the eigensolvers need) the solution vector is complex
            # even though this system is real, so the imaginary part is pure roundoff -- drop it
            # explicitly instead of letting numpy discard it with a ComplexWarning.
            b[:] = xv.real if xv.dtype.kind == "c" and b.dtype.kind != "c" else xv #type:ignore

            #print('Converged in', self.ksp.getIterationNumber(), 'iterations.') #type:ignore


            self.petsc_rhs=None
            bv.destroy() #type:ignore
        else:
            raise RuntimeError("Cannot handle Petsc mode " + str(op_flag) + " yet")


    def assemble_matrix(self,which_one:str):
        """Assemble a second matrix -- typically for building a preconditioner -- from the named
        residual/Jacobian combination.

        Note this is a DIFFERENT sparsity pattern from the main Jacobian's, because a different
        residual has different field couplings and different pinning. The problem's pattern cache holds
        several patterns at once precisely so that alternating between the two does not rebuild either
        (see dev_docs/structural_assembly.md 7d A5); if a workflow alternates between more distinct
        patterns than the cache holds, raise ``problem._frozen_sparsity_cache_capacity``.
        """
        res, n, _nzz, nrow_local, values, col_index, row_start=self.problem._assemble_residual_jacobian(which_one)
        res=PETSc.Mat().createAIJ(size=((nrow_local, n), (n, n),),csr=(row_start.astype(PETSc.IntType, copy=False), col_index.astype(PETSc.IntType, copy=False), values.astype(PETSc.ScalarType, copy=False)), comm=PETSc.COMM_WORLD) #type:ignore
        res.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False) #type:ignore
        # No shift(0.0) here: it is a no-op in PETSc (verified on every combination of matrix options,
        # see the dev doc), so it never inserted the diagonal entries it appeared to be there for. Use
        # the same policy as the main solve path instead.
        self._force_zero_diagonal(res)
        res.assemble()
        return res


@GenericLinearSystemSolver.register_solver()
class PETSCMUMPSSolver(PETSCSolver):
    # Pre-configured with MUMPS, unlike PETSCSolver: set_linear_solver("petsc").use_mumps() needs a live Problem to
    # chain onto, which set_default_linear_solver(...) cannot provide since it is only given a plain idname string.
    idname = "petsc_mumps"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self.use_mumps()


# Every option pyoomph itself puts into PETSc's global database, as opposed to the ones the user typed
# on the command line. See _account_for_own_petsc_options.
_own_petsc_options:set[str]=set()


def _SetDefaultPetscOption(key:str, val:Any,force:bool=False):
    if force or (not PETSc.Options().hasName(key)): #type:ignore
        if isinstance(val, complex):
            print("GOT COMPLEX",val)
            val=str(val.real)+("+" if val.imag>=0 else "")+str(val.imag)+"i"
            print("CASTED TO",val)
        PETSc.Options().setValue(key, val) #type:ignore
        _own_petsc_options.add(key)


def _account_for_own_petsc_options()->None:
    """Take responsibility for pyoomph's own options before PetscFinalize audits the database.

    PetscFinalize warns about every option that was set but never read ("There are N unused database
    options ... could be spelling mistake"). That check is a typo detector for what the USER typed, and
    it is worth keeping. What it should not do is report pyoomph's own defaults, which are set
    speculatively and quite often go unread: use_mumps() configures SLEPc's spectral transform when the
    eigensolver is CONSTRUCTED, and slepc_mumps is the autodetected default wherever PETSc has MUMPS,
    so a run that solves with pardiso or superlu and never touches an eigenproblem ended with five
    st_-prefixed options the user never asked for and cannot act on.

    Reading each of them here marks it used, which is the truthful thing to say -- pyoomph put them
    there and pyoomph is done with them -- and leaves everything else, including the whole database
    under -options_view, exactly as it was. (If a future PETSc stops marking on lookup, the warning
    simply comes back; opts.delValue(key) would be the heavier fallback.)

    An explicit -options_left asks for the unabridged report, so it is left alone.

    Runs before petsc4py's own atexit handler: that one was registered when petsc4py.PETSc was
    imported, i.e. before this module could register anything, and atexit unwinds last-in-first-out.
    """
    if not _own_petsc_options:
        return
    try:
        opts=PETSc.Options() #type:ignore
        if opts.hasName("options_left"): #type:ignore
            return
        for key in _own_petsc_options:
            opts.hasName(key)   # the lookup itself is what marks the option as read #type:ignore
    except Exception:
        pass   # PETSc already finalized: nothing left to account for, and nothing worth reporting.


atexit.register(_account_for_own_petsc_options)


@GenericEigenSolver.register_solver()
class SlepcEigenSolver(GenericEigenSolver):
    idname = "slepc"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self.spectral_transformation:str | None="sinvert"
        self.store_basis:bool=False
        self._last_basis:NPComplexArray | NPFloatArray | None=None
        self._last_eps_attempt:Any=None   # See _eps_solve_with_workspace_retry
        # (nrow_local, first_row, parallel) of the last solve; see _eigen_parallel_layout.
        self.last_parallel_layout:tuple[int,int,bool]=(0,0,False)

    def supports_target(self):
        return True

    def _eigen_parallel_layout(self,n:int)->tuple[int,int,bool]:
        """``(nrow_local, first_row, parallel)`` for the PETSc matrices of an n-row eigenproblem.

        Under ``mpirun -n>1`` the eigenproblem is ALWAYS solved in parallel, on COMM_WORLD, whether or
        not the mesh was distributed. The two cases differ only in where the row split comes from:

        - ``--distribute``: from oomph's dof distribution, because that is how the matrices were
          assembled and each rank genuinely holds only its own rows.
        - plain ``mpirun``: oomph assembles in parallel and then redistributes the result back to a
          globally replicated form, so every rank holds the whole matrix. A contiguous split of ``n``
          is imposed here and each rank contributes only its slice. Nothing is recomputed; the slicing
          is a view onto a matrix the rank already has.

        The replicated case is in one respect the safer of the two: each row is contributed by exactly
        one rank, from matrices that are identical across ranks, so the blocks cannot disagree.

        What this does NOT save is matrix memory -- every rank still stores the whole J and M. The win
        is the shift-and-invert factorisation, whose factors are typically far larger than the matrix
        and which is where both the time and the real memory go.
        """
        nproc=get_mpi_nproc()
        if nproc<=1:
            self.last_parallel_layout=(n,0,False)
            return n,0,False
        _,nrow_local,first_row,distributed=self.get_eigen_row_layout()
        if not distributed:
            # Replicated: impose a contiguous split here, so that the matrix, the initial space and the
            # row slicing all agree on it.
            rank=get_mpi_rank()
            base,rem=divmod(n,nproc)
            nrow_local=base+(1 if rank<rem else 0)
            first_row=rank*base+min(rank,rem)
        # Recorded so a caller can verify the eigensolve was genuinely split rather than infer it from
        # an answer that comes out the same either way (tests/mpi_eigen_worker.py).
        self.last_parallel_layout=(nrow_local,first_row,True)
        return nrow_local,first_row,True

    def _require_parallel_capable(self)->None:
        """Stop, with the reason, if this PETSc/SLEPc cannot solve an eigenproblem in parallel.

        Deliberately an error and not a fall back to solving redundantly on every rank: `mpirun -n 8`
        is a request for eight processes to share the work, and quietly doing the same eigenproblem
        eight times would look like it succeeded while being slower than serial.
        """
        nproc=get_mpi_nproc()
        if nproc<=1:
            return
        petsc_nproc=PETSc.COMM_WORLD.getSize() #type:ignore
        if petsc_nproc!=nproc:
            raise RuntimeError(
                "This petsc4py/slepc4py is not MPI-aware: pyoomph is running on "+str(nproc)+
                " processes but PETSc's COMM_WORLD has "+str(petsc_nproc)+". Either it was built with "
                "--with-mpi=0, or it is linked against a different MPI than mpi4py. Rebuild PETSc/SLEPc "
                "against the same MPI, or run the eigenproblem on a single process.")
        # Only relevant if the spectral transform will actually factorise. A user who has configured an
        # iterative st_ksp needs no direct solver at all, and is not second-guessed here.
        opts=PETSc.Options() #type:ignore
        st_pc=opts.getString("st_pc_type","lu") if opts.hasName("st_pc_type") else "lu" #type:ignore
        if st_pc!="lu" and st_pc!="cholesky":
            return
        if not (PETSc.Sys.hasExternalPackage("mumps") or PETSc.Sys.hasExternalPackage("superlu_dist")): #type:ignore
            raise RuntimeError(
                "Solving an eigenproblem on "+str(nproc)+" processes needs a PARALLEL direct solver for "
                "the shift-and-invert transform, and this PETSc has neither MUMPS nor SuperLU_DIST "
                "(PETSc's own LU is sequential only). Rebuild PETSc with --download-mumps=yes, choose an "
                "iterative spectral transform yourself via the -st_ksp_type/-st_pc_type options, or run "
                "on a single process.")

    def _local_row_block(self,mat:"DefaultMatrixType",first_row:int,nrow_local:int)->"DefaultMatrixType":
        """This rank's contiguous slice of rows of a GLOBALLY assembled matrix.

        Column indices stay global, which is what MatMPIAIJSetPreallocationCSR wants; only the row
        pointers are rebased, which scipy's slicing does.
        """
        sub=mat[first_row:first_row+nrow_local,:]
        return sub if isinstance(sub,DefaultMatrixType) else sub.tocsr() #type:ignore

    def _create_petsc_matrix(self,mat:"DefaultMatrixType",n:int,nrow_local:int,first_row:int,parallel:bool,rows_are_local:bool)->Any:
        """Build a PETSc Mat from the eigenproblem's J or M.

        ``rows_are_local`` says whether ``mat`` already holds only this rank's rows (the ``--distribute``
        case) or the whole global matrix that has to be sliced first (everything else).

        astype(..., copy=False) for the same reason as everywhere else in this file: the CSR arrays may
        be zero-copy views onto oomph-lib's CRDoubleMatrix buffers, and the conversion is a no-op unless
        this PETSc has 64-bit indices or complex scalars.
        """
        if not parallel:
            size=((n,n),(n,n))
        else:
            if not rows_are_local:
                mat=self._local_row_block(mat,first_row,nrow_local)
            size=((nrow_local,n),(nrow_local,n))
        return PETSc.Mat().createAIJ(size=size, #type:ignore
                                     csr=(mat.indptr.astype(PETSc.IntType, copy=False), #type:ignore
                                          mat.indices.astype(PETSc.IntType, copy=False), #type:ignore
                                          mat.data.astype(PETSc.ScalarType, copy=False)), #type:ignore
                                     comm=PETSc.COMM_WORLD) #type:ignore

    def _create_petsc_vector(self,arr:NPComplexArray | NPFloatArray,n:int,nrow_local:int,first_row:int,parallel:bool)->Any:
        """A PETSc Vec on the eigenproblem's comm from a GLOBALLY indexed numpy array."""
        if parallel:
            local=numpy.ascontiguousarray(arr[first_row:first_row+nrow_local])
            return PETSc.Vec().createWithArray(local,size=(nrow_local,n),comm=PETSc.COMM_WORLD) #type:ignore
        return PETSc.Vec().createWithArray(numpy.ascontiguousarray(arr),comm=PETSc.COMM_WORLD) #type:ignore

    def _vector_to_global_array(self,v:Any,parallel:bool)->NPComplexArray | NPFloatArray:
        """A distributed Vec as a full-length array on EVERY rank.

        Replicating rather than keeping the eigenvectors distributed is deliberate. Everything
        downstream of the eigensolver -- set_eigenfunction_as_dofs(), the mesh data cache and the VTK
        output -- indexes eigenvectors by GLOBAL equation number and reaches the dofs through
        get_current_dofs()/set_current_dofs(), which already gather and scatter. With a replicated
        eigenvector none of that has to change, and the cost is neval vectors, not a matrix.
        """
        if not parallel:
            return 0+v.getArray() #type:ignore
        scatter,full=PETSc.Scatter.toAll(v) #type:ignore
        scatter.scatter(v,full,PETSc.InsertMode.INSERT,PETSc.ScatterMode.FORWARD) #type:ignore
        res=0+full.getArray() #type:ignore
        full.destroy() #type:ignore
        scatter.destroy() #type:ignore
        return res #type:ignore

    def get_last_basis(self)->NPComplexArray | NPFloatArray | None:
        return self._last_basis

    def further_setup(self,E): #type:ignore
        pass
    
    def set_default_option(self,name:str,val:Any=None,force:bool=False)->None:
        _SetDefaultPetscOption(name,val, force)
    
    def use_mumps(self,mumps_param14:int | None=None):
        if not PETSc.Sys.hasExternalPackage("mumps"): #type:ignore
            raise RuntimeError("Your PETSc installation was not compiled with MUMPS support (--download-mumps=yes). Please recompile PETSc with MUMPS or use a different eigensolver.")
        _SetDefaultPetscOption("st_ksp_type","preonly")
        _SetDefaultPetscOption("st_pc_type","lu")
        _SetDefaultPetscOption("st_pc_factor_mat_solver_type","mumps")
        _SetDefaultPetscOption("st_mat_mumps_icntl_6",5)
        _SetDefaultPetscOption("st_mat_mumps_icntl_24",1)  # null pivots; see PETSCSolver.use_mumps
        if mumps_param14 is not None:
            _SetDefaultPetscOption("st_mat_mumps_icntl_14",mumps_param14)
        return self

    def _eps_solve_with_workspace_retry(self,build_and_solve:Callable[[],Any])->Any:
        """Run an EPS solve, retrying once with more MUMPS workspace if that is what it ran out of.

        Deliberately much smaller than PETSCSolver._ksp_solve_checked, because only one of the two
        problems that method exists for is present here.

        *Not* the stale-analysis problem. Every call to solve() builds J, M and the EPS from scratch
        and destroys all three at the end, so the ST's KSP, its PC and the MUMPS instance inside it are
        new each time and MUMPS always re-runs its analysis on the matrix actually being factorised.
        The value-dependent ordering that goes stale when a KSP is kept alive (dev_docs 7h) cannot
        arise; there is nothing here to discard and rebuild.

        *Not* the silent-garbage problem either. SLEPc's STSetDefaultKSP calls
        KSPSetErrorIfNotConverged(st->ksp, PETSC_TRUE), so a failed solve inside the spectral transform
        raises out of EPSSolve instead of returning an untouched vector, and PETSc's MUMPS wrapper
        already names the code: "MUMPS error in numerical factorization: INFOG(1)=-9". No reason to
        add a convergence check the library performs itself.

        What *is* worth having is the escalation. A shift-and-invert factorises J - sigma*M, which is
        commonly denser than J alone, so it runs out of MUMPS workspace more readily than the linear
        solver does -- and pyoomph never sets st_mat_mumps_icntl_14 unless use_mumps() was given an
        explicit mumps_param14, which asks the user to have foreseen the whole thing. Doubling it and
        running once more is MUMPS' own prescription, and it turns a dead run into a slower one.

        Anything that is not a MUMPS workspace error is re-raised untouched, so a genuinely singular
        shifted matrix, a bad target or a non-MUMPS factorisation fails exactly as it did before.
        """
        try:
            return build_and_solve()
        except Exception as first_error:
            E = getattr(self, "_last_eps_attempt", None)
            infog1 = _mumps_infog_from_pc(E.getST().getKSP().getPC(), 1) if E is not None else None #type:ignore
            if infog1 == _MUMPS_ICNTL23_ERROR:
                print("MUMPS hit the working-memory cap set by -st_mat_mumps_icntl_23. Raise or remove "
                      "that cap; ICNTL(14) cannot buy room a cap forbids.")
                raise
            if infog1 is None or infog1 not in _MUMPS_ICNTL14_ERRORS:
                raise
            if not _increase_mumps_icntl14("st_mat_mumps_icntl_14", False):
                raise
            if E is not None:
                E.destroy() #type:ignore
                self._last_eps_attempt = None
            return build_and_solve()

    def solve(self, neval:int, shift:float | None | complex=None,sort:bool=True,which:EigenSolverWhich="LM",OPpart:Literal["r", "i"] | None=None,v0:NPComplexArray | NPFloatArray | None=None,target:complex | None=None,custom_J_and_M:tuple["DefaultMatrixType","DefaultMatrixType"] | None=None,with_left_eigenvectors:bool=False,quiet:bool=True)->tuple[NPComplexArray,NPComplexArray,"DefaultMatrixType","DefaultMatrixType"]:
        if which!="LM":
            raise RuntimeError("Implement which="+str(which))
        if OPpart is not None:
            raise RuntimeError("Implement OPpart="+str(OPpart))
#        if v0 is not None:
#            raise RuntimeError("Implement v0="+str(v0))
    
        if with_left_eigenvectors:
            raise RuntimeError("Implement with_left_eigenvectors")    
        if custom_J_and_M is not None:
            Jin=custom_J_and_M[0]
            Min=custom_J_and_M[1]
            n=Jin.shape[0]
            # A caller-supplied pair is always a plain GLOBAL matrix, so it gets sliced like the
            # replicated case if we are running in parallel.
            self._require_parallel_capable()
            nrow_local,first_row,parallel=self._eigen_parallel_layout(n)
            rows_are_local=False
            if not isinstance(Jin,DefaultMatrixType):
                Jin=Jin.tocsr()
                assert isinstance(Jin,DefaultMatrixType)
            if not isinstance(Min,DefaultMatrixType):
                Min=Min.tocsr()
                assert isinstance(Min,DefaultMatrixType)

            M=self._create_petsc_matrix(Min,n,nrow_local,first_row,parallel,rows_are_local)
            J=self._create_petsc_matrix(Jin,n,nrow_local,first_row,parallel,rows_are_local)

        else:
            Jin,Min,n,complex_mat=self.get_J_M_n_and_type()
            self._require_parallel_capable()
            distributed=self.get_eigen_row_layout()[3]
            nrow_local,first_row,parallel=self._eigen_parallel_layout(n)
            # Only a distributed assembly hands back rows that are already this rank's own; without
            # --distribute oomph replicates the assembled matrices, so they still have to be sliced.
            rows_are_local=distributed
            upscale_to_complex=complex_mat and (PETSc.ScalarType in {numpy.float64,numpy.float128,numpy.float32}) #type:ignore
            if upscale_to_complex:
                raise RuntimeError("Your PETSc/SLEPc installation cannot handle a complex eigenvalue problem. Please compile another PETSc/SLEPc version with complex number and adjust the PYTHONPATH accordingly so that the complex petsc4py / slepc4py is used.")
            M=self._create_petsc_matrix(Min,n,nrow_local,first_row,parallel,rows_are_local)
            J=self._create_petsc_matrix(Jin,n,nrow_local,first_row,parallel,rows_are_local)

            # On a distributed problem get_J_M_n_and_type() leaves the manipulators alone (it has no
            # ownership range to apply them with) and they are applied here instead, on the PETSc
            # matrices. Otherwise -- serial, or replicated under plain mpirun -- they have already been
            # folded into Jin/Min while those were still whole.
            if distributed:
                for manip in self.matrix_manipulators:
                    J,M=manip.apply_on_distributed_J_and_M(self,J,M)

        # TODO: Working example
        ##--petsc -st_pc_type lu -st_pc_factor_mat_solver_type umfpack
        # print(dir(PETSc.Options.hasName))
        # exit()
        
        _SetDefaultPetscOption("eps_type", "krylovschur") # krylovschur
        target_set=target is not None
        if target is None:
            if shift is not None:
                target=shift

        
        if self.spectral_transformation:
            _SetDefaultPetscOption("st_ksp_type", "preonly")
            _SetDefaultPetscOption("st_type", self.spectral_transformation)
                            
        if neval==0:
            neval=1
        #E.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
        #ncv=max(2 * neval + 1, 5 + neval)
        ncv=self.ncv if self.ncv is not None else max(2 * neval + 1, 5 + neval)
        mdp=ncv #TODO: Can be smaller for higher

        # Built in a closure so the whole thing can simply be run twice. The retry has to rebuild the
        # EPS rather than re-solve the existing one: a PETSc option is consumed when the object it
        # configures is set up, so a raised ICNTL(14) is only seen by a KSP/PC that has not been set up
        # yet, and the spectral transform's was.
        def build_and_solve()->Any:
            E = SLEPc.EPS()  #type:ignore
            E.create(comm=PETSc.COMM_WORLD) #type:ignore
            # Kept on self so that _eps_solve_with_workspace_retry can still reach the ST's PC to read
            # MUMPS' INFOG after a failure -- by then the exception has unwound past this local.
            self._last_eps_attempt = E
            if target is not None:
                E.setTarget(target)
                E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE) #type:ignore
            else:
                E.setTarget(0)
                E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL) #type:ignore

                #trgt=PETSc.toScalar(target)
                #print(trgt)
                #E.setTarget(trgt)
            E.setOperators(J, M) #type:ignore
            E.setProblemType(SLEPc.EPS.ProblemType.GNHEP) #type:ignore

            E.setDimensions(neval,ncv,mdp) #type:ignore

            if v0 is not None:
                # v0 arrives globally indexed (it is typically a previous eigenvector, which pyoomph
                # keeps replicated); _create_petsc_vector slices out this rank's rows when parallel.
                if len(v0.shape)==1:
                    _v0=self._create_petsc_vector(v0,n,nrow_local,first_row,parallel)
                    E.setInitialSpace(_v0)
                    _v0.destroy()
                else:
                    ispace=[]
                    for i in range(min(v0.shape[0],ncv)):
                        ispace.append(self._create_petsc_vector(v0[i,:],n,nrow_local,first_row,parallel))
                    E.setInitialSpace(ispace)
                    for _v0 in ispace:
                        _v0.destroy()

            #print(dir(E))
            #exit()
            # E.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
            E.setFromOptions() #type:ignore

            if self.spectral_transformation and shift:
                E.getST().setShift(shift)
            self.further_setup(E) #type:ignore
            E.solve() #type:ignore
            return E

        E = self._eps_solve_with_workspace_retry(build_and_solve)

        if quiet:
            Print = lambda *pargs,**kwargs: None
        else:
            # The EPS is always on COMM_WORLD now, and PETSc.Sys.Print prints once per communicator, so
            # this already reports exactly once per run rather than once per rank.
            Print = PETSc.Sys.Print #type:ignore
        Print()
        Print("******************************")
        Print("*** SLEPc Solution Results ***")
        Print("******************************")
        Print()
        its = E.getIterationNumber() #type:ignore
        Print("Number of iterations of the method: %d" % its) #type:ignore
        eps_type = E.getType() #type:ignore
        Print("Solution method: %s" % eps_type) #type:ignore
        nev, ncv, mpd = E.getDimensions()  #type:ignore
        Print("Number of requested eigenvalues: %d" % nev)
        tol, maxit = E.getTolerances() #type:ignore
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit)) #type:ignore
        nconv = E.getConverged() #type:ignore
        Print("Number of converged eigenpairs %d" % nconv) #type:ignore

        #Print(M) #type:ignore

        evals = []
        evects = []
        if nconv > 0:
            # Create the results vectors

            vr, wr = J.getVecs() #type:ignore
            vi, wi = J.getVecs() #type:ignore
            #
            Print()
            Print(" k ||Ax-kx||/||kx|| ")
            Print("----------------- ------------------")
            #lastev = None
            for i in range(nconv): #type:ignore
                k = E.getEigenpair(i, vr, vi) #type:ignore
                #k=E.getEigenvalue(i) #type:ignore
                #E.getEigenvector(i, vr, vi) #type:ignore
                error = E.computeError(i) #type:ignore
                evals.append(k) #type:ignore
                # Gathered to full global length here, once per eigenpair, so that everything after this
                # point -- the sorting, the caller, the whole output stack -- sees the same globally
                # indexed eigenvectors it sees in a serial run. See _vector_to_global_array().
                _vr = self._vector_to_global_array(vr,parallel) #type:ignore
                #_vi=0+vi.getArray() #type:ignore
                # TODO: Something seems to be wrong in complex SLEPc. At least here, with complex shift, it can be messed up
                #Print("IN K %9f%+9f j"%(k.real,k.imag)+" error: %12g" % error) #type:ignore
                #print("EQ ",k*(Min*_vr)-Jin*_vr)
                if k.imag != 0.0: #type:ignore
                    #Print("LASTEV "+("None" if lastev is None else "NOTNONE"))

                    if False:
                        if lastev is not None:
                            Print("DIFF  "+str(numpy.abs(lastev - numpy.conjugate(k))))
                            if numpy.abs(lastev - numpy.conjugate(k)) == 0.0:
                                Print("ADDING VI  "+str(vi.getArray()))
                                evects.append(0+vi.getArray())
                            #lastev = None
                        else:
                            evects.append(0+_vr)
                            #lastev = k
                        
                    else:
                        evects.append(0+_vr+self._vector_to_global_array(vi,parallel)*1j) #type:ignore
                        Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
                else:
                    #lastev = None
                    evects.append(0+_vr) #type:ignore
                    Print(" %12f %12g" % (k.real, error)) #type:ignore

        evals = numpy.array(evals) #type:ignore
        if sort:
            if sort==True:
                if target_set:
                    # target_set is only True if the "target" parameter was non-None on entry, and the
                    # "if target is None: target=shift" block above never touches target in that case,
                    # so target is still guaranteed non-None here.
                    assert target is not None
                    srt = numpy.argsort(numpy.abs(evals-complex(target)))[0:min(neval, len(evals))]
                else:
                    srt = numpy.argsort(-evals)[0:min(neval, len(evals))] #type:ignore
            else:
                srt = numpy.argsort(numpy.array([sort(x) for x in evals]))[0:min(neval, len(evals))] #type:ignore
            #print("SORTING",evals,srt)
            evals = evals[srt] #type:ignore
            evects = numpy.array(evects)[srt] #type:ignore
        else:
            evects = numpy.array(evects) #type:ignore
            
        if self.store_basis:
            last_basis_list:list[Any]=[]
            basis=E.getBV()
            nbasis=basis.getSizes()[1]
            for i in range(nbasis):
                bv=basis.createVec()
                basis.copyVec(i,bv)
                # Gathered like the eigenvectors: get_last_basis()'s consumers index it globally.
                last_basis_list.append(self._vector_to_global_array(bv,parallel))
                bv.destroy()
            self._last_basis=numpy.array(last_basis_list)
        else:
            self._last_basis=None
        
        M.destroy() #type:ignore
        J.destroy() #type:ignore    
        E.destroy() #type:ignore
        
        return numpy.array(evals), numpy.array(evects),Jin,Min #type:ignore

    def get_PETSc(self)->Any:
        """
        Returns access to PETSc
        If defining derived classes that need access to PETSc, get PETSc from here, do not import petsc4py again
        """
        return PETSc

    def get_SLEPc(self)->Any:
        """
        Returns access to SLEPc
        If defining derived classes that need access to SLEPc, get SLEPc from here, do not import slepc4py again
        """        
        return SLEPc

@GenericEigenSolver.register_solver()
class SlepcMUMPSEigenSolver(SlepcEigenSolver):
    # See PETSCMUMPSSolver above for why this pre-configured variant exists.
    idname = "slepc_mumps"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self.use_mumps()


class FieldSplitPETSCSolver(PETSCSolver):
    def __init__(self,problem:"Problem"):
        super(FieldSplitPETSCSolver, self).__init__(problem)
        self._fieldsplit_map:NPIntArray | None=None
        self._fieldsplit:list[tuple[str, Any]] | None=None
        self.default_field_split:int | None=None
        self._fieldsplit_names:dict[int,str]={}
        self.preconditioner_matrix_name=None
        self._nullspaces=[]
        
    def add_constant_nullspace(self,*dofnames):
        self._nullspaces.append(("constant",dofnames))
        
        
    def set_fieldsplit_names(self,**kargs:int)->None:
        for k,v in kargs.items():
            self._fieldsplit_names[v]=k

    def define_options(self):
        pass

    def define_field_split(self):
        #Call split_fields here (with e.g. domain/velocity_x=0, ...)
        pass

    def split_fields(self,**kwargs:int):
        meshblocks:OrderedDict[str,dict[str,int]]=OrderedDict()
        wholemesh:OrderedDict[str,int]=OrderedDict()
        allkeys:OrderedDict[str,bool]=OrderedDict()
        for n,b in kwargs.items():
            tmesh=self.problem.get_mesh(n,return_None_if_not_found=True)
            if tmesh is None:
                #Simple field only
                sp=n.split("/")
                sn="/".join(sp[0:-1])
                tmesh = self.problem.get_mesh(sn, return_None_if_not_found=True)
                if tmesh is None:
                    if len(sp)==2 and self.problem._meshdict.get(sp[0], None) is not None:
                        ode=self.problem.get_ode(sp[0])
                        #ode_elem = ode._get_ODE("ODE")                        
                        inds=ode.get_code_gen().get_code().get_elemental_field_indices()
                        if sp[-1] in inds.keys():
                            if not sn in meshblocks.keys():
                                meshblocks[sn]={}
                            meshblocks[sn][sp[-1]]=b
                            allkeys[sn]=True
                            continue
                    raise RuntimeError("Cannot perform a field split for the unknown field "+n)
                if not (sn in meshblocks.keys()):
                    meshblocks[sn]={}
                meshblocks[sn][sp[-1]]=b
                allkeys[sn]=True
            else: #Whole mesh
                if n in wholemesh.keys():
                    raise RuntimeError("Duplicated argument "+n)
                wholemesh[n]=b
                allkeys[n]=True

        
        for k in allkeys.keys():
            mesh=self.problem.get_mesh(k,return_None_if_not_found=True)
            if mesh is None:
                mesh=self.problem.get_ode(k)
               
            typesI, names = mesh.describe_global_dofs()
            name_look_up={v:i for i,v in enumerate(names)}
            types:NPIntArray = numpy.array(typesI,dtype=numpy.int32) #type:ignore
            if self._fieldsplit_map is None:
                self._fieldsplit_map=0*types-1
            if k in wholemesh.keys():
                dest=wholemesh[k]
                where=numpy.where(types>=0)[0] #type:ignore
                self._fieldsplit_map[where]=dest
            if k in meshblocks.keys():
                for vn,dest in meshblocks[k].items():
                    if not (vn in name_look_up.keys()):
                        raise RuntimeError("Cannot find the field "+vn+" on mesh "+k+" to split")
                    where = numpy.where(types == name_look_up[vn])[0] #type:ignore
                    self._fieldsplit_map[where] = dest


    def _perform_field_split(self):
        if len(self._nullspaces)>0:
            nsvects=[]
            for ns in self._nullspaces:
                if ns[0]=="constant":
                    alldofinds=None
                    for n in ns[1]:
                        splt=n.split("/")
                        mesh=self.problem.get_mesh("/".join(splt[:-1]),return_None_if_not_found=True)
                        if mesh is None:
                            mesh=self.problem.get_ode(n)
                        typesI, names = mesh.describe_global_dofs()
                        
                        name_look_up={v:i for i,v in enumerate(names)}
                        types:NPIntArray = numpy.array(typesI,dtype=numpy.int32)
                        if alldofinds is None:
                            alldofinds=types*0
                        where = numpy.where(types == name_look_up[splt[-1]])[0]
                        alldofinds[where]=1
                    nsvects.append(alldofinds)                    
                else:
                    raise RuntimeError("Unknown nullspace type "+ns[0])
                
            petscvects=[PETSc.Vec().createWithArray(v) for v in nsvects] #type:ignore
            ns=self.get_PETSc().NullSpace().create(constant=False,vectors=petscvects)
            self.petsc_mat.setNullSpace(ns) #type:ignore
            
        if self._fieldsplit_map is None:
            return
        where = numpy.where(self._fieldsplit_map == -1)[0] #type:ignore
        if len(where)>0:
            if self.default_field_split is None:
                raise RuntimeError("Found a defined field split. Use either default_field_split or specify all fields in the define_field_split")
            else:
                self._fieldsplit_map[where] = self.default_field_split

        numfields:int=numpy.amax(self._fieldsplit_map)+1 #type:ignore
        fields = []
        # By the time _perform_field_split runs (called from setup_solver, after the matrix has
        # been assembled in solve_serial/solve_distributed), self.petsc_mat is always a real PETSc
        # Mat, never None. Capture it locally so pyright can see a single non-None type throughout.
        petsc_mat=self.petsc_mat
        assert petsc_mat is not None
        ownerrange=petsc_mat.getOwnershipRange() #type:ignore
        globsize=petsc_mat.getSize()[0] #type:ignore
        
        for i in range(numfields):
            IS = PETSc.IS() #type:ignore
            subdofs = numpy.where(self._fieldsplit_map == i)[0] #type:ignore
            if ownerrange[0]>0 or ownerrange[1]<globsize:
                subdofs=subdofs[(subdofs < ownerrange[1]) & (subdofs >= ownerrange[0])]                               
            
            subdofs:NPIntArray  = numpy.array(subdofs, dtype="int32") #type:ignore
            name = self._fieldsplit_names.get(i,str(i))
            IS.createGeneral(subdofs) #type:ignore
            
            fields.append((name, IS)) #type:ignore
        pc = self.ksp.getPC() #type:ignore
        pc.setFieldSplitIS(*fields) #type:ignore
        self._fieldsplit=fields

    def assemble_preconditioner(self,name:str,restrict_on_field_split:int | None=None)->Any:               
        _res, n, _M_nzz, nrow_local, M_values_arr, M_colindex_arr, M_row_start_arr=self.problem._assemble_residual_jacobian(name)
        # Same dtype conversion as everywhere else in this file; see solve_distributed().
        P=PETSc.Mat().createAIJ(size=((nrow_local, n), (nrow_local, n),),
                                csr=(M_row_start_arr.astype(PETSc.IntType, copy=False),
                                     M_colindex_arr.astype(PETSc.IntType, copy=False),
                                     M_values_arr.astype(PETSc.ScalarType, copy=False))) #type:ignore # TODO: Must be destroyed!
        if restrict_on_field_split is not None:
            assert self._fieldsplit is not None
            ps = self._fieldsplit[restrict_on_field_split][1] #type:ignore
            P = P.createSubMatrix(ps, ps) #type:ignore
        return P #type:ignore

    def define_preconditioner(self):
        if self.preconditioner_matrix_name is not None:
            P=self.assemble_preconditioner(self.preconditioner_matrix_name)
            # setup_solver() (the only caller of define_preconditioner) always creates self.ksp
            # beforehand via the base class's setup_solver(), so it is never None here.
            ksp=self.ksp
            assert ksp is not None
            ksp.setOperators(self.petsc_mat,P) #type:ignore

    def setup_solver(self):
        
        self.define_options()
        super().setup_solver()
        self._fieldsplit_map=None
        self._nullspaces=[]        
        self.define_field_split()
        self._perform_field_split()
        self.define_preconditioner()

    def get_PC(self)->Any:
        pc = self.ksp.getPC() #type:ignore
        return pc #type:ignore





