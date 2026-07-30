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
 
from .generic import GenericLinearSystemSolver, GenericEigenSolver, EigenSolverWhich,DefaultMatrixType
from collections import OrderedDict
import petsc4py #type:ignore
import sys

petsc4py.init(sys.argv) #type:ignore

import slepc4py #type:ignore

slepc4py.init(sys.argv) #type:ignore

from petsc4py import PETSc #type:ignore
from slepc4py import SLEPc #type:ignore
from ..generic.mpi import *
from ..typings import *
import numpy

if TYPE_CHECKING:
    from ..generic.problem import Problem


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

    def _can_reuse_structure(self,n:int,nnz:int)->bool:
        structure_id = self.problem.jacobian_structure_id
        return (self.reuse_matrix_structure and structure_id != 0
                and structure_id == self._structure_id
                and self.petsc_mat is not None
                and self._structure_nnz == nnz
                and self.petsc_mat.getSize()[0] == n) #type:ignore

    def _can_reuse_structure_distributed(self,nrow_local:int,nnz_local:int)->bool:
        structure_id = self.problem.jacobian_structure_id
        return (self.reuse_matrix_structure and structure_id != 0
                and structure_id == self._structure_id
                and self.petsc_mat is not None
                and self._structure_nnz == nnz_local
                and self._structure_nrow_local == nrow_local)

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
            if self._can_reuse_structure(n,nnz):
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
                self.ksp.solve(bv, self.x) #type:ignore
                end_time = time.time()
                if not self.problem.is_quiet():
                    print("PETSc KSP solve time:", end_time - start_time, "seconds")
                xv = self.x.getArray() #type:ignore
            b[:] = xv[:] #type:ignore

            #print('Converged in', self.ksp.getIterationNumber(), 'iterations.') #type:ignore

            
            self.petsc_rhs=None
            bv.destroy() #type:ignore
            
        else:
            raise RuntimeError("Cannot handle Petsc mode " + str(op_flag) + " yet")
        return 0  # TODO: Return sign of Jacobian

    def solve_distributed(self, op_flag: int, allow_permutations: int, n: int, nnz_local: int, nrow_local: int, first_row: int, values: NPFloatArray, col_index: NPIntArray, row_start: NPIntArray, b: NPFloatArray, nprow: int, npcol: int, doc: int, data: NPUInt64Array, info: NPIntArray)->None:
        #print("solve distributed with flag ",op_flag)
        if op_flag == 1:
            # Same reuse logic as solve_serial. jacobian_structure_id is bumped inside the collective
            # assign_eqn_numbers(), so every rank agrees on it and either all reuse or all rebuild --
            # which matters, because rebuilding the Mat is itself collective and a split decision would
            # deadlock. _can_reuse_structure compares the LOCAL row count and local nnz, both of which
            # are per-rank quantities derived from the same global pattern.
            if self._can_reuse_structure_distributed(nrow_local,nnz_local):
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
                self.ksp.solve(bv, self.x) #type:ignore
                end_time = time.time()
                if not self.problem.is_quiet():
                    print("PETSc KSP solve time:", end_time - start_time, "seconds")
                xv = self.x.getArray() #type:ignore
            b[:] = xv[:] #type:ignore

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


def _SetDefaultPetscOption(key:str, val:Any,force:bool=False):
    if force or (not PETSc.Options().hasName(key)): #type:ignore
        if isinstance(val, complex):
            print("GOT COMPLEX",val)
            val=str(val.real)+("+" if val.imag>=0 else "")+str(val.imag)+"i"
            print("CASTED TO",val)
        PETSc.Options().setValue(key, val) #type:ignore


@GenericEigenSolver.register_solver()
class SlepcEigenSolver(GenericEigenSolver):
    idname = "slepc"

    def __init__(self, problem:"Problem"):
        super().__init__(problem)
        self.spectral_transformation:str | None="sinvert"
        self.store_basis:bool=False
        self._last_basis:NPComplexArray | NPFloatArray | None=None
        
    def supports_target(self):
        return True
        
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
            if not isinstance(Jin,DefaultMatrixType):
                Jin=Jin.tocsr()
                assert isinstance(Jin,DefaultMatrixType)
            if not isinstance(Min,DefaultMatrixType):
                Min=Min.tocsr()
                assert isinstance(Min,DefaultMatrixType)
                
            # Min/Jin's CSR arrays may be zero-copy views onto oomph-lib's CRDoubleMatrix buffers
            # (see get_J_M_n_and_type() / src/nanobind/problem.cpp). astype(..., copy=False) is a
            # no-op when the dtypes already match PETSc's own (the common case) and only allocates a
            # converted copy on a 64-bit-index or complex PETSc build; mirrors the fix applied to
            # PETSCSolver.solve_serial()/assemble_matrix() above.
            M=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Min.indptr.astype(PETSc.IntType, copy=False), Min.indices.astype(PETSc.IntType, copy=False), Min.data.astype(PETSc.ScalarType, copy=False))) #type:ignore
            J=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Jin.indptr.astype(PETSc.IntType, copy=False), Jin.indices.astype(PETSc.IntType, copy=False), Jin.data.astype(PETSc.ScalarType, copy=False))) #type:ignore

        else:
            Jin,Min,n,complex_mat=self.get_J_M_n_and_type()
            upscale_to_complex=complex_mat and (PETSc.ScalarType in {numpy.float64,numpy.float128,numpy.float32}) #type:ignore
            if upscale_to_complex:
                raise RuntimeError("Your PETSc/SLEPc installation cannot handle a complex eigenvalue problem. Please compile another PETSc/SLEPc version with complex number and adjust the PYTHONPATH accordingly so that the complex petsc4py / slepc4py is used.")
            M=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Min.indptr.astype(PETSc.IntType, copy=False), Min.indices.astype(PETSc.IntType, copy=False), Min.data.astype(PETSc.ScalarType, copy=False))) #type:ignore
            J=PETSc.Mat().createAIJ(size=((n, n), (n, n),), csr=(Jin.indptr.astype(PETSc.IntType, copy=False), Jin.indices.astype(PETSc.IntType, copy=False), Jin.data.astype(PETSc.ScalarType, copy=False))) #type:ignore
            
        #if self.imag_contribution is not None:
        #    raise RuntimeError("Cannot have imaginary matrix contributions yet here")
#        for manip in self.matrix_manipulators:
#            raise RuntimeError("Cannot have MatrixManipulators yet here: "+str(manip))
            #J, M = manip.apply_on_J_and_M(self, J, M)

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
                            
        E = SLEPc.EPS()  #type:ignore
        E.create() #type:ignore
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
        
        if neval==0:
            neval=1
        #E.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
        #ncv=max(2 * neval + 1, 5 + neval)
        ncv=self.ncv if self.ncv is not None else max(2 * neval + 1, 5 + neval)
        mdp=ncv #TODO: Can be smaller for higher
        
        E.setDimensions(neval,ncv,mdp) #type:ignore
        
        if v0 is not None:
            if len(v0.shape)==1:
                _v0=PETSc.Vec().createWithArray(v0) #type:ignore
                E.setInitialSpace(_v0)
                _v0.destroy()
            else:
                ispace=[]
                for i in range(min(v0.shape[0],ncv)):
                    ispace.append(PETSc.Vec().createWithArray(v0[i,:])) #type:ignore
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

        if quiet:
            Print = lambda *pargs,**kwargs: None
        else:
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
                _vr = 0+vr.getArray() #type:ignore
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
                        evects.append(0+_vr+vi.getArray()*1j) #type:ignore
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
                last_basis_list.append(bv.getArray())
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





