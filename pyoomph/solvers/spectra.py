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

"""The Spectra eigensolver backend (idname "spectra").

Spectra (https://spectralib.org) is a header-only Arnoldi eigensolver on top of Eigen, downloaded and
compiled into the extension module by cmake/ThirdPartySpectra.cmake. It exists here for the platforms
that have no PETSc/SLEPc at all - Windows above all, where the wheels are built without MPI and hence
without PETSc - and it is the only backend there that can TARGET an eigenvalue: the ARPACK-based
"scipy"/"pardiso" backends raise on any target, which leaves Hopf tracking and the normal-form
calculations falling back to worse routes (see supports_complex_target() in .generic).

Compared with SLEPc this covers everything except the contour-integral region search
(SlepcEigenSolver.set_eigenvalue_region), and it is serial: a distributed problem is gathered onto
rank 0 by GenericEigenSolver._solve_gathered_on_root, exactly as the scipy backend does.

Spectra itself only offers the STANDARD eigenproblem for a general matrix. Its generalized solvers
(SymGEigsSolver, SymGEigsShiftSolver) require A symmetric and B positive definite, which pyoomph mass
matrices violate: they are positive semi-definite and SINGULAR, since pressure and pinned rows carry
no time derivative. So the shift-and-invert transform is applied here instead, exactly as SLEPc's ST
does internally:

    J v = lambda M v    <=>    (J - sigma*M)^-1 M v = nu v ,   lambda = sigma + 1/nu

which works for an arbitrary M - non-symmetric, indefinite or singular. The infinite eigenvalues a
singular M produces map to nu = 0, which the "largest magnitude" selection never converges to.
"""

from .generic import GenericEigenSolver,DefaultMatrixType,EigenSolverWhich,SolverError
from .. import _pyoomph_core as _pyoomph

import numpy
import scipy.linalg #type:ignore
import scipy.sparse #type:ignore
import scipy.sparse.linalg #type:ignore

from ..typings import *
if TYPE_CHECKING:
    from ..generic.problem import Problem

# Raised as an ImportError, and deliberately at import time: GenericEigenSolver.factory_solver()
# imports this module to find the solver, so failing here means the class is never registered and the
# user gets the standard "unknown eigensolver, the following are available" message instead of one
# that appears to offer a backend which cannot run.
if not getattr(_pyoomph,"has_spectra",False):
    raise ImportError("this pyoomph build was compiled without Spectra support "
                      "(configure with -DPYOOMPH_HAS_SPECTRA=ON)")


class SpectraEigenSolverError(SolverError):
    """Spectra could not solve the eigenproblem, or the shifted matrix could not be factorised."""


class _SpluInvOp:
    """The scipy/SuperLU counterpart of PardisoInvOp, used when MKL is not available.

    That is the normal situation on Windows, where the mkl package is an optional dependency, so this
    is a first-class path rather than a fallback nobody exercises. SuperLU factorises complex matrices
    natively, which is what the complex shifts here need.
    """
    def __init__(self,A:DefaultMatrixType):
        self.mat=A
        self._lu=scipy.sparse.linalg.splu(A.tocsc()) #type:ignore

    def __call__(self,b:Any)->Any:
        return self._lu.solve(b) #type:ignore

    matvec=__call__

    @property
    def shape(self)->Any:
        return self.mat.shape

    @property
    def dtype(self)->Any:
        return self.mat.dtype


@GenericEigenSolver.register_solver()
class SpectraEigenSolver(GenericEigenSolver):
    idname="spectra"

    def __init__(self,problem:"Problem"):
        super().__init__(problem)
        self.shift:float | complex=0.0
        self.tol=1e-10
        self.maxit=1000
        self.ncv:int | None=None
        self.max_retries=3
        #: Factorise the shifted matrix with MKL Pardiso when it is available. Set to False to force
        #: the SuperLU path, which is what a machine without MKL uses anyway.
        self.use_pardiso=True
        #: How far off a requested target the shift is placed when the target itself is used as the
        #: shift, relative to |target|. A target is typically an exact eigenvalue, which would make
        #: J - target*M singular.
        self.target_shift_offset=1e-7
        #: Ritz values of the transformed problem below this fraction of the largest one are the
        #: infinite eigenvalues of a singular mass matrix and are dropped.
        self.infinite_eigenvalue_cutoff=1e-12

    def supports_target(self)->bool:
        return True

    def supports_complex_target(self)->bool:
        # Unlike PETSc/SLEPc this needs no separately built complex library: the complex arithmetic is
        # a second template instantiation of the same header-only solver, always compiled in.
        return True

    # ------------------------------------------------------------------ shift, target and sorting

    #: which -> Spectra sort rule. These act on the TRANSFORMED spectrum nu = 1/(lambda-sigma), which
    #: is also what scipy.sparse.linalg.eigs does whenever sigma is given, so the meaning of `which`
    #: is unchanged with respect to the scipy/pardiso backends.
    _SORT_RULES:dict[str,str]={"LM":"LargestMagn","SM":"SmallestMagn","LR":"LargestReal",
                               "SR":"SmallestReal","SI":"SmallestImag"}

    def _choose_sigma(self,shift:float | complex | None,target:complex | None)->complex:
        """Pick the spectral-transform shift from the caller's shift and target.

        The two are not interchangeable and the callers genuinely disagree about which one locates
        the wanted mode:

        - get_hopf_eigenvector passes target=1j*omega0 together with shift=(1j+omega_epsilon)*omega0,
          i.e. a shift deliberately nudged off a target that IS an eigenvalue. There the shift is the
          informed choice and must win.
        - NormalFormCalculator.get_left_eigenvector passes target=lamb with a nominal shift=1e-7.
          Shift-inverting at 1e-7 converges to the modes nearest zero and its own guard then rejects
          the result as belonging to a different mode, so there the target must win.

        Whether the shift says anything about the target is what separates the two.
        """
        if target is None:
            return complex(shift if shift is not None else 0.0)
        t=complex(target)
        scale=max(abs(t),1.0)
        if shift is not None and abs(complex(shift)-t)<=0.1*scale:
            return complex(shift)
        return t+self.target_shift_offset*scale

    def _order(self,evals:Any,evects:Any,neval:int,sort:Any,target:complex | None,
               solver_order_is_targeted:bool)->tuple[Any,Any]:
        """Order (and truncate) the result the way SlepcEigenSolver.solve does.

        `sort` may be True, False, or a callable used as a sort key - the last is what
        Problem.solve_eigenproblem passes through and SLEPc honours, so it is honoured here too.

        sort=False does NOT mean "arbitrary order": SLEPc returns its own target-ordered list there,
        and callers rely on it. get_hopf_lyapunov_coefficient asks for the adjoint at target=-i*omega
        with sort=False and then reads evals[0], so handing back an unordered list gives it the +i*omega
        conjugate and it aborts with "Could not find the correct eigenvector". Spectra's own Arnoldi
        order is already nearest-sigma-first, but the dense fallback's is whatever LAPACK produced,
        hence solver_order_is_targeted.
        """
        if not sort:
            if target is None or solver_order_is_targeted:
                return evals,evects
            srt=numpy.argsort(numpy.abs(evals-complex(target)))
            return evals[srt],evects[srt]
        if sort is True:
            if target is not None:
                srt=numpy.argsort(numpy.abs(evals-complex(target)))
            else:
                srt=numpy.argsort(-evals)
        else:
            srt=numpy.argsort(numpy.array([sort(x) for x in evals]))
        srt=srt[0:min(neval,len(evals))]
        return evals[srt],evects[srt]

    # ------------------------------------------------------------------ the operator

    def _make_operator(self,J:DefaultMatrixType,M:DefaultMatrixType,sigma:complex,use_sym:bool,
                       quiet:bool,force_complex:bool)->tuple[Any,bool]:
        """Factorise J - sigma*M and return (operator, is_complex)."""
        # sigma is a python complex throughout, and multiplying a real matrix by one promotes it to
        # complex128 even when the imaginary part is zero - which would put every real problem on the
        # complex path at twice the cost. Demote it when it really is real.
        shift_scalar:float | complex=sigma.real if sigma.imag==0.0 else sigma
        A=(J-shift_scalar*M).tocsr() #type:ignore
        if force_complex and A.dtype.kind!="c":
            # A complex start vector cannot be handed to a real factorisation.
            A=A.astype(numpy.complex128)
        complex_op=(A.dtype.kind=="c")
        if self.use_pardiso:
            try:
                from .pardiso import PardisoInvOp
                # The mtype comes from the dtype of the SHIFTED matrix, not from J and M. A real
                # pencil with a complex shift - the ordinary Hopf case - gives a complex A while J and
                # M are both real, and picking the real mtype there fails inside ctypes rather than
                # returning something wrong. (PardisoArpackEigenSolver.get_OPInv still decides from
                # J/M and has that hole; it cannot be reached from there because the scipy backend
                # refuses a target in the first place.)
                mode=13 if complex_op else (-2 if use_sym else 11)
                # sigma=None: the matrix is already shifted, so PardisoInvOp only factorises it. It
                # still applies its own mtype -2 guard on the diagonal.
                return PardisoInvOp(A,None,None,mode=mode),complex_op
            except Exception as e:
                # Not just an ImportError: pyoomph.solvers.pardiso imports perfectly well without MKL
                # and only fails when the shared library is actually loaded, which happens here.
                if not quiet:
                    print("Spectra: MKL Pardiso is not usable ("+str(e)+"), factorising with SuperLU instead")
        return _SpluInvOp(A),complex_op

    # ------------------------------------------------------------------ solve

    def solve(self,neval:int,shift:float | complex | None=None,sort:bool=True,which:EigenSolverWhich="LM",OPpart:Literal["r", "i"] | None=None,v0:NPComplexArray | NPFloatArray | None=None,target:complex | None=None,custom_J_and_M:tuple[DefaultMatrixType,DefaultMatrixType] | None=None,with_left_eigenvectors:bool=False,quiet:bool=True)->tuple[NPComplexArray,NPComplexArray,DefaultMatrixType,DefaultMatrixType]:
        # Serial solver: a distributed problem is gathered onto rank 0, which then re-enters here with
        # the whole square system. custom_J_and_M is always global and so never needs gathering.
        if custom_J_and_M is None and self.get_eigen_row_layout()[3]:
            return self._solve_gathered_on_root(neval,shift=shift,sort=sort,which=which,OPpart=OPpart,
                                                v0=v0,target=target,
                                                with_left_eigenvectors=with_left_eigenvectors,quiet=quiet)
        if with_left_eigenvectors:
            raise RuntimeError("Implement with_left_eigenvectors")
        if OPpart is not None:
            # OPpart selects the real or imaginary part of ARPACK's real-arithmetic complex-shift mode,
            # which exists only because ARPACK avoids complex arithmetic. Here a complex shift is
            # simply a complex factorisation, so there is nothing for it to select.
            raise RuntimeError("OPpart is an ARPACK-specific option and has no meaning for the Spectra "
                               "eigensolver, which shift-inverts in complex arithmetic directly")
        if which not in self._SORT_RULES:
            raise RuntimeError("Unknown eigenvalue selection '"+str(which)+"', expected one of "
                               +str(sorted(self._SORT_RULES.keys())))
        if shift is None:
            shift=self.shift

        self.problem._set_solved_residual(self.real_contribution,True,False)

        if custom_J_and_M is not None:
            J,M=custom_J_and_M
            n=J.shape[0]
        else:
            J,M,n,_is_complex=self.get_J_M_n_and_type()

        if neval<=0:
            neval=n

        sigma=self._choose_sigma(shift,target)

        # Symmetry bookkeeping, in the same order and with the same reason strings as the scipy
        # backend, so that last_symmetry_decision means the same thing whichever backend is the
        # default on a given machine (tests/test_symmetric_solver_switch.py asserts on it).
        #
        # Note that this backend does not actually NEED M to be positive semi-definite: it never uses
        # M as an inner product, always running the general Arnoldi iteration. The verdict only picks
        # Pardiso's symmetric-indefinite factorisation (mtype -2) for J - sigma*M. The screen is kept
        # so the recorded decision is backend-independent.
        use_sym=(self._use_symmetric_eigensolver_now() and numpy.imag(sigma)==0
                 and which=="LM"
                 and J.dtype.kind!="c" and M.dtype.kind!="c"
                 and (v0 is None or not numpy.iscomplexobj(v0)))
        # collective only when every rank is here; on the gathered path rank 0 is alone.
        if use_sym and not self._mass_matrix_can_be_positive_semidefinite(M,collective=custom_J_and_M is None):
            use_sym=False
            self.last_symmetry_decision_reason="mass matrix is symmetric but not positive semi-definite"
        self.last_symmetry_decision=use_sym

        # Spectra's Arnoldi needs 1 <= nev <= n-2 and nev+2 <= ncv <= n, so tiny systems - the 2-dof
        # pendulum ODE in the symmetry tests, or any call asking for every eigenvalue - have to go
        # dense. Same branch, and same caveat about generalized eigh, as the scipy backend.
        if n<4 or neval>=n-2:
            return self._solve_dense(J,M,n,neval,sort,target,use_sym)

        return self._solve_arnoldi(J,M,n,neval,sigma,which,sort,target,v0,use_sym,quiet)

    def _solve_dense(self,J:DefaultMatrixType,M:DefaultMatrixType,n:int,neval:int,sort:Any,
                     target:complex | None,use_sym:bool)->tuple[NPComplexArray,NPComplexArray,DefaultMatrixType,DefaultMatrixType]:
        # scipy.linalg.eig and not eigh even for a proven-symmetric pencil: the generalized eigh
        # requires a strictly positive definite b, which a singular M violates.
        if use_sym:
            self.last_symmetry_decision=False
            self.last_symmetry_decision_reason="dense path (generalized eigh needs strictly PD M)"
        evals,evects=scipy.linalg.eig(J.toarray(),b=M.toarray(),left=False) #type:ignore
        evals=numpy.asarray(evals,dtype=numpy.complex128)
        evects=numpy.ascontiguousarray(numpy.transpose(evects),dtype=numpy.complex128) #type:ignore
        # A singular M makes scipy.linalg.eig report infinite eigenvalues. The Arnoldi path never sees
        # them (they sit at nu=0, which the selection does not converge to), but here they are part of
        # the result and have to go before anything is ordered.
        finite=numpy.isfinite(evals)
        evals,evects=evals[finite],evects[finite]
        # LAPACK's order says nothing about the target, so a targeted request must be reordered even
        # when sort is False - see _order().
        evals,evects=self._order(evals,evects,neval,sort,target,solver_order_is_targeted=False)
        return cast(NPComplexArray,evals),cast(NPComplexArray,evects),J,M

    def _solve_arnoldi(self,J:DefaultMatrixType,M:DefaultMatrixType,n:int,neval:int,sigma:complex,
                       which:EigenSolverWhich,sort:Any,target:complex | None,
                       v0:NPComplexArray | NPFloatArray | None,use_sym:bool,
                       quiet:bool)->tuple[NPComplexArray,NPComplexArray,DefaultMatrixType,DefaultMatrixType]:
        # A start vector forces the arithmetic even when the matrices and the shift are real.
        force_complex=(v0 is not None and numpy.iscomplexobj(v0))

        op=None
        for attempt in range(self.max_retries+1):
            try:
                op,complex_op=self._make_operator(J,M,sigma,use_sym,quiet,force_complex)
                break
            except Exception as e:
                # A singular J - sigma*M means the shift landed exactly on an eigenvalue, which is
                # likely rather than unlucky: a target usually IS one, and _choose_sigma's offset can
                # still be too small for a badly scaled problem.
                if attempt>=self.max_retries:
                    raise SpectraEigenSolverError("could not factorise J - sigma*M at sigma="+str(sigma)
                                                  +" ("+str(e)+")") from e
                sigma=sigma+10*self.target_shift_offset*max(abs(sigma),1.0)
                if not quiet:
                    print("Spectra: the shifted matrix could not be factorised, retrying at sigma="+str(sigma))
        assert op is not None

        dt=numpy.complex128 if complex_op else numpy.float64
        # Promote M once rather than on every matrix-vector product.
        Mc=M.astype(numpy.complex128) if (complex_op and M.dtype.kind!="c") else M

        def matvec(x:Any)->Any:
            return numpy.ascontiguousarray(op.matvec(Mc@x),dtype=dt) #type:ignore

        start=None
        if v0 is not None:
            v0=numpy.asarray(v0)
            if v0.ndim!=1 or v0.shape[0]!=n:
                # SLEPc accepts a whole initial basis; Spectra takes a single start vector.
                if not quiet:
                    print("Spectra: ignoring the given v0, which is not a single vector of length "+str(n))
            elif not numpy.any(v0):
                # Spectra rejects a zero residual vector outright.
                if not quiet:
                    print("Spectra: ignoring the given v0, which is entirely zero")
            else:
                start=numpy.ascontiguousarray(v0,dtype=dt)

        rule="LargestMagn" if target is not None else self._SORT_RULES[which]
        ncv=self.ncv if self.ncv is not None else max(2*neval+1,neval+5)
        ncv=min(max(ncv,neval+2),n)
        maxit=self.maxit
        fn=_pyoomph.spectra_eigensolve_complex if complex_op else _pyoomph.spectra_eigensolve_real

        nu=None
        for attempt in range(self.max_retries+1):
            nu,Vt,nconv,niter,info,_nmatvec=fn(matvec,n,neval,ncv,maxit,self.tol,start,rule)
            if info=="successful" or nconv>=neval:
                break
            if info=="numerical_issue":
                # A broken factorisation or a breakdown in the iteration; a larger subspace does not
                # help with either.
                raise SpectraEigenSolverError("Spectra reported a numerical issue at sigma="+str(sigma)
                                              +" (nconv="+str(nconv)+" of "+str(neval)+")")
            if attempt>=self.max_retries:
                if nconv==0:
                    raise SpectraEigenSolverError("Spectra converged no eigenvalue at all at sigma="
                                                  +str(sigma)+" after "+str(niter)+" iterations "
                                                  +"(ncv="+str(ncv)+", maxit="+str(maxit)+", tol="+str(self.tol)+")")
                # A partial result is returned rather than raised, matching SLEPc's nconv < nev, which
                # the callers already handle.
                break
            ncv=min(max(2*ncv,neval+2),n)
            maxit*=2
            if not quiet:
                print("Spectra did not converge ("+str(nconv)+" of "+str(neval)+"), retrying with ncv="
                      +str(ncv)+", maxit="+str(maxit)+" (attempt "+str(attempt+1)+"/"+str(self.max_retries)+")")

        assert nu is not None
        nu=numpy.asarray(nu,dtype=numpy.complex128)
        Vt=numpy.asarray(Vt,dtype=numpy.complex128)

        # The infinite eigenvalues of a singular M sit at nu = 0. LargestMagn never converges to them,
        # so this is a guard rather than the routine case - which is exactly why there is no `infcrop`
        # here as in the scipy backend, where ARPACK hands the infinities back as part of the result.
        if nu.size:
            cutoff=max(self.infinite_eigenvalue_cutoff*float(numpy.max(numpy.abs(nu))),
                       numpy.finfo(numpy.float64).tiny)
            keep=numpy.abs(nu)>cutoff
            nu,Vt=nu[keep],Vt[keep]

        evals=sigma+1.0/nu
        # The eigenvectors need no back-transform at all:
        #   (J-sigma*M)^-1 M v = nu v   <=>   M v = nu (J - sigma*M) v   <=>   J v = (sigma + 1/nu) M v
        evects=Vt
        finite=numpy.isfinite(evals)
        evals,evects=evals[finite],evects[finite]

        # Spectra's own order is already nearest-sigma-first (LargestMagn on nu), which is what a
        # targeted request wants, so an unsorted result may be handed back as it is.
        evals,evects=self._order(evals,evects,neval,sort,target,solver_order_is_targeted=True)

        evects=numpy.ascontiguousarray(evects,dtype=numpy.complex128)
        return cast(NPComplexArray,evals),cast(NPComplexArray,evects),J,M


from ..typings import _set_public_api
_set_public_api(globals())
