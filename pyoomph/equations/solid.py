from __future__ import annotations
#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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
from .. import *
from ..expressions import *
from ..typings import *
from .. import _pyoomph_core as _pyoomph
from .ALE import BaseMovingMeshEquations

class BaseSolidConstitutiveLaw:
    """
        Base class for solid constitutive laws. The method get_sigma must be implemented in derived classes.

        Args:
            use_subexpressions: Use subexpressions for the tensor entries. Defaults to True.
            use_inverse_routine: Use a specific routine for the inversion of the metric tensor. It cannot be used when requiring Hessians, e.g. for bifurcation tracking. Defaults to True.
            active_deformation: Optional active deformation gradient F_a (multiplicative Kroener-Lee/growth split). The rest metric becomes F_a^T*g*F_a, i.e. strains are measured relative to an intrinsically deformed state - muscle-like actuation, anisotropic growth, swelling. May depend on the Lagrangian coordinates, time or other fields. Only F_a^T*F_a enters, so rotational parts of F_a are irrelevant. Since it only modifies the constitutive law, it exerts no net force or torque on a free body. See also active_fiber_deformation. Defaults to None.
    """
    def __init__(self,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        self.use_subexpressions=use_subexpressions
        self.use_inverse_routine=use_inverse_routine
        self.active_deformation=active_deformation
        
    def _subexpression(self,expr):
        if self.use_subexpressions:
            return subexpression(expr)
        else:
            return expr
        
    def is_incompressible(self)->bool:
        """
        If this returns True, the constitutive law is incompressible. This means that the determinant of the deformation gradient is equal to the determinant of the undeformed configuration. The solid equations will then introduce a pressure variable that is used to enforce this condition.
        """
        return False
    
    def get_Gij(self,dim):     
        """
        Returns the covariant metric tensor of the deformed configuration. 
        """           
        x=var("mesh")
        return self._subexpression(matproduct(transpose(grad(x,lagrangian=True)),grad(x,lagrangian=True)))
    
    def get_G_up_ij(self,dim):
        # G^ij=(G^(-1))_ij
        Gij=self.get_Gij(dim)        
        if self.use_inverse_routine and dim>1: # InvertMatrix requires n=2 or n=3, the 1x1 case is trivial anyways
            from ..expressions.tensor_funcs import InvertMatrix
            #inv=InvertSymmetricMatrix(n=dim)        
            inv=InvertMatrix(n=dim,matrix_type="symmetric")            
            return self._subexpression(inv(Gij))
        else:
            return self._subexpression(inverse_matrix(Gij,fill_to_vector_dim_3=True,skip_empty_rows_and_cols=True))
        
    
    def get_gij(self,dim:int,isotropic_growth_factor:ExpressionOrNum=1):
        """
        Returns the covariant metric tensor of the undeformed configuration. Without growth and active deformation, this is the identity matrix in Cartesian coordinates. An active deformation gradient F_a turns it into the pulled-back rest metric F_a^T*g*F_a.
        """
        X=var("lagrangian")
        grf=pow(isotropic_growth_factor,rational_num(2,dim))
        g=grf*matproduct(transpose(grad(X,lagrangian=True)),grad(X,lagrangian=True))
        if self.active_deformation is not None:
            # Entries of F_a beyond the element dimension cannot enter: the corresponding rows
            # and columns of g are zero, so a 2d problem may use e.g. active_fiber_deformation
            # (which fills the full 3x3) without contaminating the metric.
            Fa=self.active_deformation
            g=matproduct(transpose(Fa),matproduct(g,Fa))
        return self._subexpression(g)
        
    
    def get_g_up_ij(self,dim:int,isotropic_growth_factor:ExpressionOrNum=1):
        """
        Returns the contravariant metric tensor g^ij of the undeformed configuration, i.e. the inverse of :py:meth:`get_gij` on the actual dimensions.
        """
        gij=self.get_gij(dim,isotropic_growth_factor)
        if self.active_deformation is None:
            # Here, g_ij=grf*P with P=grad(X)^T*grad(X), where grad(X) (including the metric entries of the coordinate system, e.g. R/R in the
            # hoop direction) is the projector onto the actual dimensions. Hence, P*P=P and g^ij=P/grf=g_ij/grf**2. This holds in any
            # coordinate system, without needing to know which rows/columns of the 3x3 tensors are actually used.
            grf=pow(isotropic_growth_factor,rational_num(2,dim))
            return self._subexpression(gij/grf**2)
        if self.use_inverse_routine and dim>1:
            from ..expressions.tensor_funcs import InvertMatrix
            inv=InvertMatrix(n=dim,matrix_type="symmetric")
            return self._subexpression(inv(gij))
        else:
            return self._subexpression(inverse_matrix(gij,fill_to_vector_dim_3=True,skip_empty_rows_and_cols=True))

    def get_gammaij(self,dim:int,isotropic_growth_factor:ExpressionOrNum=1):
        """
        Returns the Green strain tensor 
        """
        return self._subexpression((self.get_Gij(dim)-self.get_gij(dim,isotropic_growth_factor))/2)
    
    def get_sigma(self,dim:int,isotropic_growth_factor:ExpressionNumOrNone=1,pressure_var:Expression | None=None)->Expression:
        """
        Returns the second Piola-Kirchhoff stress tensor. The method must be implemented in derived classes.

        Args:
            dim: Element dimension.
            isotropic_growth_factor: Expression or numerical value representing the isotropic growth factor. Defaults to 1.
            pressure_var (Optional[Expression], optional): Pressure variable for incompressible cases. Defaults to None.

        """
        raise RuntimeError("get_sigma must be implemented in the derived class")
    
def active_fiber_deformation(direction:Expression,stretch:ExpressionOrNum,isochoric_dim:int | None=None)->Expression:
    """Active deformation gradient for a single fiber family: F_a = stretch*(d⊗d) + t*(1-d⊗d) with the normalized fiber direction d, i.e. an intrinsic stretch along the (possibly spatially varying) direction. With isochoric_dim set to the problem dimension, the transverse factor t=stretch^(-1/(dim-1)) compensates the volume (muscle-like, det F_a=1 within the element dimension); otherwise t=1 and det F_a=stretch (growth along the fiber).

    Args:
        direction: Fiber direction (any nonzero vector expression, normalized internally). May vary in space, e.g. follow a midline.
        stretch: Intrinsic stretch factor along the fiber, e.g. 1+kappa*Y for a bending actuation.
        isochoric_dim: If set, compensate transversally so the activation is volume-preserving in this dimension. Defaults to None.
    """
    d=direction/square_root(dot(direction,direction))
    P=dyadic(d,d)
    if isochoric_dim is None:
        transverse:ExpressionOrNum=1
    else:
        if isochoric_dim<2:
            raise ValueError("isochoric_dim must be at least 2, since a one-dimensional fiber stretch cannot be volume-compensated")
        transverse=pow(stretch,rational_num(-1,isochoric_dim-1))
    return transverse*(identity_matrix()-P)+stretch*P


class IncompressibleSolidConstitutiveLaw(BaseSolidConstitutiveLaw):
    """
    Base class for incompressible solid constitutive laws. The method get_sigma must be implemented in derived classes.     
    """
    def is_incompressible(self)->bool:
        """
        Returns True, indicating that this is an incompressible solid constitutive law.
        """
        return True
    
class GeneralizedHookeanSolidConstitutiveLaw(BaseSolidConstitutiveLaw):
    """
        Generalized Hookean solid constitutive law. 
        
        Args:
            E: Young's modulus
            nu: Poisson's ratio
            use_subexpressions (bool, optional): Use subexpressions for the tensor entries. Defaults to True.
            use_inverse_routine (bool, optional): Use a specific routine for the inversion of the metric tensor. It cannot be used when requiring Hessians, e.g. for bifurcation tracking. Defaults to True.
            active_deformation (Optional[Expression], optional): Active deformation gradient F_a, see BaseSolidConstitutiveLaw. Defaults to None.
    """
    def __init__(self, E:ExpressionOrNum=1, nu:ExpressionOrNum=0.4,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):

        super().__init__(use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        self.E=E
        self.nu=nu
        
        
        

    def get_sigma(self,dim:int,isotropic_growth_factor:ExpressionNumOrNone=1,pressure_var:Expression | None=None):
        se=lambda x : self._subexpression(x)
        Gup=self.get_G_up_ij(dim)
        igf=isotropic_growth_factor if isotropic_growth_factor is not None else 1
        gammakl=self.get_gammaij(dim,igf)
        if dim is None:
            raise RuntimeError("dim must be specified")

        E=lambda i,j,k,l : se(self.E/(1+self.nu))*(se(self.nu/(1-2*self.nu))*Gup[i,j]*Gup[k,l]+(Gup[i,k]*Gup[j,l]+Gup[i,l]*Gup[j,k])/2)
        sigma_up_ij=lambda i,j : sum(E(i,j,k,l)*gammakl[k,l] for k in range(3) for l in range(3))
        sigma=matrix([[sigma_up_ij(i,j) for j in range(3)] for i in range(3)])
        return se(sigma)
        
class StVenantKirchhoffSolidConstitutiveLaw(BaseSolidConstitutiveLaw):
    """
        St. Venant-Kirchhoff solid: the second Piola-Kirchhoff stress is linear in the Green-Lagrange strain,

        .. math:: S^{ij} = \\lambda\\,g^{ij}g^{kl}\\gamma_{kl} + \\mu\\left(g^{ik}g^{jl}+g^{il}g^{jk}\\right)\\gamma_{kl}

        i.e. :math:`S=\\lambda\\operatorname{tr}(\\gamma)I+2\\mu\\gamma` in Cartesian reference coordinates, with the Lame parameters
        :math:`\\lambda=E\\nu/((1+\\nu)(1-2\\nu))` and :math:`\\mu=E/(2(1+\\nu))`. Contrary to :py:class:`GeneralizedHookeanSolidConstitutiveLaw`,
        the indices are raised with the metric :math:`g^{ij}` of the *undeformed* (reference) configuration, not with the deformed one.
        It is hyperelastic, with the strain energy :math:`W=\\lambda/2\\operatorname{tr}(\\gamma)^2+\\mu\\,\\gamma:\\gamma`, and is e.g. the material
        model of the Turek-Hron FSI benchmark. It is only suitable for moderate strains: under strong compression, the stress response softens
        and eventually vanishes.

        Args:
            E: Young's modulus
            nu: Poisson's ratio, must be below 1/2
            use_subexpressions (bool, optional): Use subexpressions for the tensor entries. Defaults to True.
            use_inverse_routine (bool, optional): Use a specific routine for the inversion of the reference metric tensor (only needed with an active deformation). It cannot be used when requiring Hessians, e.g. for bifurcation tracking. Defaults to True.
            active_deformation (Optional[Expression], optional): Active deformation gradient F_a, see BaseSolidConstitutiveLaw. Defaults to None.
    """
    def __init__(self, E:ExpressionOrNum=1, nu:ExpressionOrNum=0.4,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        super().__init__(use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        if is_zero(nu-rational_num(1,2)):
            raise ValueError("StVenantKirchhoffSolidConstitutiveLaw requires nu<1/2. It has no incompressible limit.")
        self.E=E
        self.nu=nu

    def get_sigma(self,dim:int,isotropic_growth_factor:ExpressionNumOrNone=1,pressure_var:Expression | None=None):
        if dim is None:
            raise RuntimeError("dim must be specified")
        se=lambda x : self._subexpression(x)
        igf=isotropic_growth_factor if isotropic_growth_factor is not None else 1
        gup=self.get_g_up_ij(dim,igf)
        gamma=self.get_gammaij(dim,igf)
        lam=self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        mu=self.E/(2*(1+self.nu))
        gup_gamma=matproduct(gup,gamma)
        # S = lambda*tr(g^-1 gamma)*g^-1 + 2*mu*g^-1 gamma g^-1 (gamma is symmetric)
        return se(lam*trace(gup_gamma)*gup+2*mu*matproduct(gup_gamma,gup))


class HyperelasticSolidConstitutiveLaw(BaseSolidConstitutiveLaw):
    """
        Generic isotropic hyperelastic solid, defined by a strain energy density :math:`W(I_1,I_2,J)` per undeformed volume. Pass the strain energy
        as a callable ``strain_energy(I1,I2,J)`` or override :py:meth:`get_strain_energy` in a subclass. The second Piola-Kirchhoff stress is obtained
        by symbolic differentiation,

        .. math:: S=2\\frac{\\partial W}{\\partial C}=2W_{,I_1}g^{-1}+2W_{,I_2}\\left(I_1g^{-1}-g^{-1}Gg^{-1}\\right)+W_{,J}JG^{-1}

        with the deformed and undeformed metric tensors :math:`G` and :math:`g` (i.e. :math:`C=g^{-1}G`), :math:`I_1=\\operatorname{tr}C`,
        :math:`I_2=\\frac{1}{2}\\left(I_1^2-\\operatorname{tr}C^2\\right)` and :math:`J=\\sqrt{\\det G/\\det g}`. The invariants are always the ones
        of the full three-dimensional deformation: in a plane-strain (2d Cartesian) or 1d setting, the missing directions have a unit stretch.
        Hence, a strain energy written for 3d (e.g. with :math:`I_1-3`) is correct in all dimensions and coordinate systems.

        For ``incompressible=True``, the strain energy is evaluated at :math:`J=1` (it may only depend on :math:`I_1` and :math:`I_2`), a pressure field
        is added to enforce :math:`J=1` and the stress reads :math:`S=2W_{,I_1}g^{-1}+2W_{,I_2}(I_1g^{-1}-g^{-1}Gg^{-1})-pG^{-1}`.
        In a dimensional problem, the scale of this pressure field must be set, e.g. ``problem.set_scaling(pressure=E)``.

        Args:
            strain_energy: Callable ``(I1,I2,J)->W``, returning the strain energy density (i.e. with the units of a stress). Can be None, if :py:meth:`get_strain_energy` is overridden.
            incompressible: Enforce incompressibility by a pressure field. Defaults to False.
            use_subexpressions (bool, optional): Use subexpressions for the tensor entries. Defaults to True.
            use_inverse_routine (bool, optional): Use a specific routine for the inversion of the metric tensors. It cannot be used when requiring Hessians, e.g. for bifurcation tracking. Defaults to True.
            active_deformation (Optional[Expression], optional): Active deformation gradient F_a, see BaseSolidConstitutiveLaw. Defaults to None.
    """
    _symbol_counter=0

    def __init__(self,strain_energy:Callable[[Expression,Expression,ExpressionOrNum],ExpressionOrNum] | None=None,*,incompressible:bool=False,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        super().__init__(use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        self.strain_energy=strain_energy
        self.incompressible=incompressible

    def is_incompressible(self)->bool:
        return self.incompressible

    def get_strain_energy(self,I1:Expression,I2:Expression,J:ExpressionOrNum)->ExpressionOrNum:
        """
        Returns the strain energy density as function of the invariants. ``J`` is ``1`` for an incompressible law.
        """
        if self.strain_energy is None:
            raise RuntimeError("Either pass strain_energy to HyperelasticSolidConstitutiveLaw or override get_strain_energy")
        return self.strain_energy(I1,I2,J)

    def get_invariants(self,dim:int,isotropic_growth_factor:ExpressionOrNum=1)->Tuple[Expression,Expression,Expression]:
        """
        Returns the invariants I1, I2 and J of the full three-dimensional deformation, i.e. unit stretches are assumed in directions which are not resolved.
        """
        gup=self.get_g_up_ij(dim,isotropic_growth_factor)
        C=matproduct(gup,self.get_Gij(dim))
        I1d=self._subexpression(trace(C))
        I2d=self._subexpression((I1d**2-trace(matproduct(C,C)))/2)
        nmiss=3-dim # directions with unit stretch
        I1=I1d+nmiss
        I2=I2d+nmiss*I1d+rational_num(nmiss*(nmiss-1),2)
        detG=determinant(self.get_Gij(dim))
        detg=determinant(self.get_gij(dim,isotropic_growth_factor))
        J=self._subexpression(square_root(detG/detg))
        return I1,I2,J

    @classmethod
    def _new_symbol(cls,name:str)->Expression:
        cls._symbol_counter+=1
        return _pyoomph.GiNaC_new_symbol("__hyperelastic_"+name+"_"+str(cls._symbol_counter))

    def get_sigma(self,dim:int,isotropic_growth_factor:ExpressionNumOrNone=1,pressure_var:Expression | None=None):
        if dim is None:
            raise RuntimeError("dim must be specified")
        se=lambda x : self._subexpression(x)
        igf=isotropic_growth_factor if isotropic_growth_factor is not None else 1
        I1,I2,J=self.get_invariants(dim,igf)
        sI1,sI2,sJ=self._new_symbol("I1"),self._new_symbol("I2"),self._new_symbol("J")
        W=Expression(self.get_strain_energy(sI1,sI2,1 if self.incompressible else sJ))

        def dW(sym:Expression)->Expression:
            # derivative with respect to an invariant, then substitute the actual invariants
            res=diff(W,sym)
            for s,v in ((sI1,I1),(sI2,I2),(sJ,J)):
                res=_pyoomph.GiNaC_subs(res,s,v)
            return res

        gup=self.get_g_up_ij(dim,igf)
        Gup=self.get_G_up_ij(dim)
        S:ExpressionOrNum=0
        W1,W2=dW(sI1),dW(sI2)
        if not is_zero(W1):
            S+=2*se(W1)*gup
        if not is_zero(W2):
            gGg=matproduct(gup,matproduct(self.get_Gij(dim),gup))
            S+=2*se(W2)*(I1*gup-gGg)
        if self.incompressible:
            if pressure_var is None:
                raise RuntimeError("pressure_var must be specified for an incompressible law")
            S+=-pressure_var*Gup
        else:
            WJ=dW(sJ)
            if not is_zero(WJ):
                S+=se(WJ)*J*Gup
        return se(S)


class NeoHookeanSolidConstitutiveLaw(HyperelasticSolidConstitutiveLaw):
    """
        Compressible neo-Hookean solid with the strain energy (Ciarlet / Simo-Pister form)

        .. math:: W=\\frac{\\mu}{2}\\left(I_1-3\\right)-\\mu\\ln J+\\frac{\\lambda}{2}\\left(\\ln J\\right)^2

        with the Lame parameters :math:`\\lambda=E\\nu/((1+\\nu)(1-2\\nu))` and :math:`\\mu=E/(2(1+\\nu))`, i.e. it agrees with linear elasticity
        for small strains. Contrary to :py:class:`StVenantKirchhoffSolidConstitutiveLaw`, it is suitable for large strains: the stress diverges
        for :math:`J\\to 0`. See :py:class:`HyperelasticSolidConstitutiveLaw` for the general framework.

        Args:
            E: Young's modulus
            nu: Poisson's ratio, must be below 1/2. Use :py:class:`IncompressibleNeoHookeanSolidConstitutiveLaw` for the incompressible limit.
            Further args: see :py:class:`HyperelasticSolidConstitutiveLaw`.
    """
    def __init__(self,E:ExpressionOrNum=1,nu:ExpressionOrNum=0.4,*,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        super().__init__(None,incompressible=False,use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        if is_zero(nu-rational_num(1,2)):
            raise ValueError("NeoHookeanSolidConstitutiveLaw requires nu<1/2. Use IncompressibleNeoHookeanSolidConstitutiveLaw instead.")
        self.E=E
        self.nu=nu

    def get_strain_energy(self,I1,I2,J):
        lam=self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        mu=self.E/(2*(1+self.nu))
        return mu/2*(I1-3)-mu*log(J)+lam/2*log(J)**2


class IncompressibleNeoHookeanSolidConstitutiveLaw(HyperelasticSolidConstitutiveLaw):
    """
        Incompressible neo-Hookean solid, :math:`W=\\frac{\\mu}{2}(I_1-3)` with :math:`J=1` enforced by a pressure field.
        Either the Young's modulus ``E`` (then :math:`\\mu=E/3`) or the shear modulus ``mu`` must be given.

        Args:
            E: Young's modulus.
            mu: Shear modulus, alternatively to E.
            Further args: see :py:class:`HyperelasticSolidConstitutiveLaw`.
    """
    def __init__(self,E:ExpressionNumOrNone=None,*,mu:ExpressionNumOrNone=None,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        super().__init__(None,incompressible=True,use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        if (E is None)==(mu is None):
            raise ValueError("Specify exactly one of E and mu")
        self.mu=mu if mu is not None else E/3

    def get_strain_energy(self,I1,I2,J):
        return self.mu/2*(I1-3)


class _IsochoricHyperelasticLaw(HyperelasticSolidConstitutiveLaw):
    """
        Common base of rubber-like laws with a strain energy :math:`W_\\text{iso}(\\bar I_1,\\bar I_2)`. Without bulk modulus, the law is
        incompressible and :math:`\\bar I_k=I_k` (since :math:`J=1`). With a bulk modulus :math:`\\kappa`, it is nearly incompressible, using the
        isochoric invariants :math:`\\bar I_1=J^{-2/3}I_1`, :math:`\\bar I_2=J^{-4/3}I_2` and the volumetric energy :math:`\\frac{\\kappa}{2}(J-1)^2`.
    """
    def __init__(self,kappa:ExpressionNumOrNone,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        super().__init__(None,incompressible=kappa is None,use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        self.kappa=kappa

    def get_isochoric_strain_energy(self,I1bar:Expression,I2bar:Expression)->ExpressionOrNum:
        raise RuntimeError("Implement get_isochoric_strain_energy")

    def get_strain_energy(self,I1,I2,J):
        if self.kappa is None:
            return self.get_isochoric_strain_energy(I1,I2)
        return self.get_isochoric_strain_energy(I1*pow(J,rational_num(-2,3)),I2*pow(J,rational_num(-4,3)))+self.kappa/2*(J-1)**2


class MooneyRivlinSolidConstitutiveLaw(_IsochoricHyperelasticLaw):
    """
        Mooney-Rivlin solid, :math:`W=C_1(\\bar I_1-3)+C_2(\\bar I_2-3)` (shear modulus :math:`\\mu=2(C_1+C_2)`), incompressible if ``kappa`` is None,
        otherwise nearly incompressible with the volumetric energy :math:`\\frac{\\kappa}{2}(J-1)^2`. :math:`C_2=0` gives the neo-Hookean solid.

        Args:
            C1, C2: Material parameters (units of a stress).
            kappa: Bulk modulus. None (default) for an incompressible solid.
            Further args: see :py:class:`HyperelasticSolidConstitutiveLaw`.
    """
    def __init__(self,C1:ExpressionOrNum,C2:ExpressionOrNum=0,kappa:ExpressionNumOrNone=None,*,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        super().__init__(kappa,use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        self.C1,self.C2=C1,C2

    def get_isochoric_strain_energy(self,I1bar,I2bar):
        return self.C1*(I1bar-3)+self.C2*(I2bar-3)


class YeohSolidConstitutiveLaw(_IsochoricHyperelasticLaw):
    """
        Yeoh solid, :math:`W=C_1(\\bar I_1-3)+C_2(\\bar I_1-3)^2+C_3(\\bar I_1-3)^3` (shear modulus :math:`\\mu=2C_1`), e.g. for filled rubbers.
        Incompressible if ``kappa`` is None, otherwise nearly incompressible with the volumetric energy :math:`\\frac{\\kappa}{2}(J-1)^2`.

        Args:
            C1, C2, C3: Material parameters (units of a stress).
            kappa: Bulk modulus. None (default) for an incompressible solid.
            Further args: see :py:class:`HyperelasticSolidConstitutiveLaw`.
    """
    def __init__(self,C1:ExpressionOrNum,C2:ExpressionOrNum=0,C3:ExpressionOrNum=0,kappa:ExpressionNumOrNone=None,*,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        super().__init__(kappa,use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        self.C1,self.C2,self.C3=C1,C2,C3

    def get_isochoric_strain_energy(self,I1bar,I2bar):
        return self.C1*(I1bar-3)+self.C2*(I1bar-3)**2+self.C3*(I1bar-3)**3


class GentSolidConstitutiveLaw(_IsochoricHyperelasticLaw):
    """
        Gent solid, :math:`W=-\\frac{\\mu J_m}{2}\\ln\\left(1-\\frac{\\bar I_1-3}{J_m}\\right)`, which stiffens strongly when :math:`\\bar I_1-3`
        approaches the locking parameter :math:`J_m` (limited chain extensibility). For :math:`J_m\\to\\infty`, it is the neo-Hookean solid.
        Incompressible if ``kappa`` is None, otherwise nearly incompressible with the volumetric energy :math:`\\frac{\\kappa}{2}(J-1)^2`.

        Args:
            mu: Shear modulus.
            Jm: Locking parameter (dimensionless).
            kappa: Bulk modulus. None (default) for an incompressible solid.
            Further args: see :py:class:`HyperelasticSolidConstitutiveLaw`.
    """
    def __init__(self,mu:ExpressionOrNum,Jm:ExpressionOrNum,kappa:ExpressionNumOrNone=None,*,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        super().__init__(kappa,use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        self.mu,self.Jm=mu,Jm

    def get_isochoric_strain_energy(self,I1bar,I2bar):
        return -self.mu*self.Jm/2*log(1-(I1bar-3)/self.Jm)


class KelvinVoigtSolidConstitutiveLaw(BaseSolidConstitutiveLaw):
    """
        Adds a Kelvin-Voigt viscous stress to an arbitrary (elastic) constitutive law:

        .. math:: S=S_\\text{elastic}+2\\eta\\,g^{-1}\\dot{E}g^{-1}+\\zeta\\operatorname{tr}\\left(g^{-1}\\dot{E}\\right)g^{-1}

        with the Green-Lagrange strain rate :math:`\\dot{E}=\\frac{1}{2}\\partial_t G` (Lagrangian time derivative, independent of growth or active
        deformation rates). It is frame-indifferent and gives a physical damping of the solid dynamics. In stationary solves, it does not contribute.

        Args:
            elastic_law: The underlying constitutive law. Its incompressibility, growth and active deformation settings are taken over.
            viscosity: Shear-like viscosity :math:`\\eta`.
            bulk_viscosity: Additional volumetric viscosity :math:`\\zeta`. Defaults to 0.
    """
    def __init__(self,elastic_law:BaseSolidConstitutiveLaw,viscosity:ExpressionOrNum,bulk_viscosity:ExpressionOrNum=0):
        super().__init__(use_subexpressions=elastic_law.use_subexpressions,use_inverse_routine=elastic_law.use_inverse_routine,active_deformation=elastic_law.active_deformation)
        self.elastic_law=elastic_law
        self.viscosity=viscosity
        self.bulk_viscosity=bulk_viscosity

    def is_incompressible(self)->bool:
        return self.elastic_law.is_incompressible()

    def get_sigma(self,dim:int,isotropic_growth_factor:ExpressionNumOrNone=1,pressure_var:Expression | None=None):
        igf=isotropic_growth_factor if isotropic_growth_factor is not None else 1
        S=self.elastic_law.get_sigma(dim,isotropic_growth_factor,pressure_var)
        gup=self.get_g_up_ij(dim,igf)
        Edot=partial_t(self.get_Gij(dim),ALE=False)/2
        gEdot=matproduct(gup,Edot)
        Sv=2*self.viscosity*matproduct(gEdot,gup)
        if not is_zero(self.bulk_viscosity):
            Sv+=self.bulk_viscosity*trace(gEdot)*gup
        return self._subexpression(S+Sv)


class IncompressibleHookeanSolidConstitutiveLaw(IncompressibleSolidConstitutiveLaw):
    """
        Incompressible Hookean solid constitutive law.
        Args:
            E: Young's modulus
            use_subexpressions (bool, optional): Use subexpressions for the tensor entries. Defaults to True.
            use_inverse_routine (bool, optional): Use a specific routine for the inversion of the metric tensor. It cannot be used when requiring Hessians, e.g. for bifurcation tracking. Defaults to True.
            active_deformation (Optional[Expression], optional): Active deformation gradient F_a, see BaseSolidConstitutiveLaw. Note that the incompressibility constraint det G = det g then enforces the actual volume to follow det F_a - use a volume-preserving F_a (e.g. active_fiber_deformation with isochoric_dim) for pure actuation. Defaults to None.
    """
    def __init__(self,E:ExpressionOrNum=1,use_subexpressions=True,use_inverse_routine:bool=True,active_deformation:Expression | None=None):
        super().__init__(use_subexpressions=use_subexpressions,use_inverse_routine=use_inverse_routine,active_deformation=active_deformation)
        self.E=E

    def get_sigma(self,dim:int,isotropic_growth_factor:ExpressionNumOrNone=1,pressure_var:Expression | None=None):
        se=lambda x : self._subexpression(x)
        Gup=self.get_G_up_ij(dim)
        igf=isotropic_growth_factor if isotropic_growth_factor is not None else 1
        gammakl=self.get_gammaij(dim,igf)
        if dim is None:
            raise RuntimeError("dim must be specified")
        bar_sigma_up_ij=lambda i,j : se(sum((Gup[i,k]*Gup[j,l]+Gup[i,l]*Gup[j,k])*gammakl[k,l] for k in range(3) for l in range(3)))
        if pressure_var is None:
            raise RuntimeError("pressure_var must be specified")
        return -pressure_var*Gup+self.E/3*matrix([[bar_sigma_up_ij(i,j) for j in range(3)] for i in range(3)])
    

class BaseDeformableSolidEquations(BaseMovingMeshEquations):
    """
    Base class of the deformable solid equations, providing the inertia and bulk force terms and the
    scaling of the mesh (displacement) test function.

    .. note::
        **Scaling to set on problem level.** By default the mesh test function is scaled with
        ``scale_factor("temporal")**2/(scale_factor("mass_density")*scale_factor("spatial"))``, so
        ``mass_density`` -- no field of the system -- must be set by
        ``problem.set_scaling(mass_density=...)``. Passing ``modulus_for_scaling`` replaces it by an
        elastic modulus, and ``scale_for_FSI`` replaces it by ``scale_factor("pressure")``, which
        then likewise has to be set on problem level unless the solid solves for a pressure field of
        its own (incompressible constitutive law).
    """
    def __init__(self, mass_density:ExpressionOrNum=0,bulkforce:ExpressionOrNum=0,coordinate_space = None,first_order_time_derivative=False,scale_for_FSI:bool=False,modulus_for_scaling:ExpressionNumOrNone=None,isotropic_growth_factor:ExpressionOrNum=1,pressure_space:FiniteElementSpaceEnum="DL",with_error_estimator=False,overdamped:bool=False):
        super().__init__(None)
        # Kept here although the moving-mesh base class no longer takes one: a solid usually defines
        # no other continuous field, so there is nothing for the space to be inferred from, and
        # first_order_time_derivative needs a space to put "dt_mesh" on. Applied below with the same
        # "raise to at least this" rule as ElementSpace, not by assignment.
        self.coordinate_space=coordinate_space
        self.scale_for_FSI=scale_for_FSI
        self.mass_density=mass_density
        self.modulus_for_scaling=modulus_for_scaling
        self.bulkforce=bulkforce
        self.first_order_time_derivative=first_order_time_derivative
        self.isotropic_growth_factor=isotropic_growth_factor
        self.pressure_space:FiniteElementSpaceEnum=pressure_space
        self.pressure_name="pressure"
        self.with_error_estimator=with_error_estimator
        self.overdamped=overdamped

    def is_incompressible(self)->bool:
        return False


    def define_fields(self):
        if self.coordinate_space is not None:
            if self.coordinate_space not in {"C2TB","C2","C1TB","C1"}:
                raise ValueError("Can only set the coordinate space to either C2TB, C2, C1TB or C1")
            cg=self.get_current_code_generator()
            cg._coordinate_space=find_dominant_element_space(cast("FiniteElementSpaceEnum",cg._coordinate_space),self.coordinate_space)
        super().define_fields()
        if self.is_incompressible():
            self.define_scalar_field(self.pressure_name,self.pressure_space)
        if self.first_order_time_derivative:
            coordinate_space=self.coordinate_space
            if coordinate_space is None:
                raise RuntimeError("coordinate_space must be specified for first order time derivative")
            if self.overdamped:
                raise RuntimeError("overdamped=True is not compatible with first_order_time_derivative=True, since overdamped already reduces the time derivative order to one.")
            self.define_vector_field("dt_mesh",assert_valid_finite_element_space(coordinate_space),scale=scale_factor("spatial")/scale_factor("temporal"),testscale=scale_factor("temporal")/scale_factor("spatial"))
                    
    def define_scaling(self):        
        self.set_scaling(mesh= scale_factor("spatial"))
        if self.scale_for_FSI:
            self.set_test_scaling(mesh=scale_factor("spatial")/scale_factor("pressure"))
        else:
            if self.modulus_for_scaling is None:
                self.set_test_scaling(mesh=1/(scale_factor("mass_density")/scale_factor("temporal")**2*scale_factor("spatial")))
            else:
                self.set_test_scaling(mesh=1/self.modulus_for_scaling*scale_factor("spatial"))    
                

    def _before_mesh_to_mesh_interpolation(self, eqtree, interpolator):
        pass


    def define_residuals(self):        
        x,xtest=var_and_test("mesh")
        if self.first_order_time_derivative:
            self.add_weak(var("dt_mesh")-mesh_velocity(),testfunction("dt_mesh"),lagrangian=True)            
            accel=partial_t(var("dt_mesh"),ALE=False)            
        else:
            if self.overdamped:
                accel=partial_t(x,ALE=False)
            else:
                accel=partial_t(x,2,ALE=False)                
                   
        self.add_weak(self.mass_density*accel-self.bulkforce,self.isotropic_growth_factor* xtest,lagrangian=True )

class LinearElasticitySolidEquations(BaseDeformableSolidEquations):
    """
    Linear elasticity, i.e. the Cauchy stress ``E/(1+nu)*(nu/(1-2*nu)*trace(eps)+eps)`` of the
    linearized strain ``eps=sym(grad(u,lagrangian=True))``. For ``nu=None`` or ``nu=1/2``, the
    incompressible limit is solved with a pressure Lagrange multiplier instead.

    See :py:class:`BaseDeformableSolidEquations` for the scalings that must be set on problem level.

    Args:
        E: Young's modulus.
        nu: Poisson's ratio. ``None`` or ``1/2`` selects the incompressible formulation.
        mass_density: Mass density, relevant for the inertia term. Defaults to 0.
        bulkforce: Bulk force density (in the undeformed frame!). Defaults to 0.
        coordinate_space: Space to use for the mesh. Defaults to None.
        first_order_time_derivative: If set, a velocity is explicitly introduced, reducing the
            maximum time derivative order to unity (good for eigenanalysis). Defaults to False.
        scale_for_FSI: Scale the mesh test function as the velocity test function ([X]/[P]), which
            balances the tractions of a coupled fluid correctly. Defaults to False.
        modulus_for_scaling: Reference modulus to nondimensionalize with, instead of the mass
            density. Defaults to None.
        isotropic_growth_factor: Factor of growing with respect to the undeformed configuration.
        pressure_space: Space of the pressure field in the incompressible case. Defaults to "DL".
        with_error_estimator: If set, error estimators based on the strain are introduced.
    """
    def __init__(self, E:ExpressionOrNum,nu:ExpressionNumOrNone, mass_density = 0, bulkforce = 0, coordinate_space=None, first_order_time_derivative=False, scale_for_FSI = False, modulus_for_scaling = None,isotropic_growth_factor:ExpressionOrNum=1,pressure_space:FiniteElementSpaceEnum="DL",with_error_estimator=False):
        super().__init__(mass_density, bulkforce, coordinate_space, first_order_time_derivative, scale_for_FSI, modulus_for_scaling,isotropic_growth_factor=isotropic_growth_factor,pressure_space=pressure_space,with_error_estimator=with_error_estimator)
        self.E,self.nu=E,nu
        
    def is_incompressible(self)->bool:
        return self.nu is None or is_zero(self.nu-0.5) or is_zero(self.nu-rational_num(1,2))
        
    def define_residuals(self):
        if not is_zero(self.isotropic_growth_factor-1):
            raise RuntimeError("LinearElasticitySolidEquations does not support isotropic growth factor != 1. Use DeformableSolidEquations instead.")
        x,xtest=var_and_test("mesh")
        X=var("lagrangian")
        u=x-X
        
        eps=sym(grad(u,lagrangian=True))
        
        if self.is_incompressible():
            # self.nu may be None here (used as a shorthand for "incompressible, nu unspecified"), in which
            # case the incompressible limit nu->1/2 is used for the (non-singular) shear part of the stress.
            nu_eff=self.nu if self.nu is not None else rational_num(1,2)
            tau=self.E/(1+nu_eff)*(eps)-var(self.pressure_name)*identity_matrix()  # Pressure is introduced as a Lagrange multiplier
            self.add_weak(div(u,lagrangian=True),testfunction(self.pressure_name),lagrangian=True)
        else:
            # is_incompressible() being False here guarantees self.nu is not None (see is_incompressible above)
            assert self.nu is not None
            tau=self.E/(1+self.nu)*(self.nu/(1-2*self.nu)*trace(eps)*identity_matrix()+eps)
        
        super().define_residuals()  # Call the base class method to define the residuals for the inertia and bulk force terms
        self.add_weak(tau,self.isotropic_growth_factor*grad(xtest,lagrangian=True),lagrangian=True)

    def define_error_estimators(self):        
        if self.with_error_estimator:
            dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
            strain=sym(grad(var("mesh")-var("lagrangian"),lagrangian=True))
            for i in range(dim):
                self.add_spatial_error_estimator(strain[i,i])
            for i in range(dim):
                for j in range(i+1,dim):
                    self.add_spatial_error_estimator(strain[i,j])    


class DeformableSolidEquations(BaseDeformableSolidEquations):
    
    """Nonlinear solid elasticity equations for deformable solids. Requires a constitutive law, which gives the particular material properties.

        Args:
            constitutive_law: Particular solid constitutive law, which must be derived from BaseSolidConstitutiveLaw.
            mass_density: Mass density (relevant for the inertia term in temporal dynamics). Defaults to 0.
            bulkforce: Bulk force density (in the undeformed frame!). Defaults to 0.
            coordinate_space: Space to use for the mesh. Defaults to None.
            first_order_time_derivative: If set, a velocity is explicitly introduced, reducing the maximum time derivative order to unity (good for eigenanalysis). Defaults to False.
            pressure_space: If the constitutive law is incompressible, a pressure field is required. This controls the space of the pressure. Defaults to "DL".
            with_error_estimator: If set, error estimators based on the strain are introduced. Defaults to False.
            isotropic_growth_factor: Factor of growing with respect to the undeformed configuration. Defaults to 1.
            modulus_for_scaling: By default, nondimensionalization is made with respect to the scales ``mass_density``, ``spatial`` and ``temporal``. Here, you can set a reference Young's modulus to nondimensionalize with respect to this instead. Defaults to None.
            scale_for_FSI: If set, the scaling of the test function agrees with the scaling of the velocity test function ([X]/[P]). This is important to balance the tractions correctly

    .. note::
        **Scaling to set on problem level.** Unless ``modulus_for_scaling`` or ``scale_for_FSI`` is
        given, the mesh test function is scaled with the non-field scale ``mass_density``, i.e.
        ``problem.set_scaling(mass_density=...)`` is required. See
        :py:class:`BaseDeformableSolidEquations`.
    """
    # TODO: Bulk force density in Lagrangian or Eulerian coordinates? Make two or a flag?
    def __init__(self, constitutive_law:BaseSolidConstitutiveLaw, mass_density:ExpressionOrNum=0,bulkforce:ExpressionOrNum=0,coordinate_space = None,first_order_time_derivative=False,pressure_space:FiniteElementSpaceEnum="DL",with_error_estimator=False,isotropic_growth_factor:ExpressionOrNum=1,modulus_for_scaling:ExpressionNumOrNone=None,scale_for_FSI:bool=False,overdamped:bool=False):
        super().__init__(mass_density=mass_density,bulkforce=bulkforce,coordinate_space=coordinate_space, first_order_time_derivative=first_order_time_derivative,scale_for_FSI=scale_for_FSI,modulus_for_scaling=modulus_for_scaling,isotropic_growth_factor=isotropic_growth_factor,pressure_space=pressure_space,with_error_estimator=with_error_estimator,overdamped=overdamped)
        self.constitutive_law=constitutive_law

    def is_incompressible(self)->bool:
        return self.constitutive_law.is_incompressible()
        
    def define_fields(self):
        super().define_fields()                
        
        # Allow to access the deformed mass density
        if self.is_incompressible():
            rho_deformed=self.mass_density
        else:
            dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
            detG=self.constitutive_law._subexpression(determinant(self.constitutive_law.get_Gij(dim)))
            detg=self.constitutive_law._subexpression(determinant(self.constitutive_law.get_gij(dim,self.isotropic_growth_factor)))
            rho_deformed=self.constitutive_law._subexpression(self.mass_density*square_root(detg/detG))
        self.define_field_by_substitution("deformed_mass_density",rho_deformed,also_on_interface=True)


        
    def define_residuals(self):
        x,xtest=var_and_test("mesh")
        super().define_residuals()  # Call the base class method to define the residuals for the inertia and bulk force terms
        
        dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
        pvar=var(self.pressure_name) if self.constitutive_law.is_incompressible() else None       
        #print("TEST",dim,self.expand_expression_for_debugging(self.constitutive_law.get_Gij(dim)))
        #exit() 
        sigma=self.constitutive_law.get_sigma(dim,self.isotropic_growth_factor,pvar)       
                                
        self.add_weak((matproduct(grad(x,lagrangian=True),sigma)),self.isotropic_growth_factor*grad(xtest,lagrangian=True),lagrangian=True)
                
        if self.constitutive_law.is_incompressible():
            detG=self.constitutive_law._subexpression(determinant(self.constitutive_law.get_Gij(dim)))
            detg=self.constitutive_law._subexpression(determinant(self.constitutive_law.get_gij(dim,self.isotropic_growth_factor)))
            self.add_weak(detG - detg,testfunction(self.pressure_name),lagrangian=True)

    
    def define_error_estimators(self):        
        if self.with_error_estimator:
            dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
            strain=self.constitutive_law.get_gammaij(dim,self.isotropic_growth_factor)
            for i in range(dim):
                self.add_spatial_error_estimator(strain[i,i])
            for i in range(dim):
                for j in range(i+1,dim):
                    self.add_spatial_error_estimator(strain[i,j])



class SolidTraction(InterfaceEquations):
    """Imposes a traction vector on the solid interface. 

    Args:
        T: traction to apply
    """
    def __init__(self,T:ExpressionOrNum):
        super().__init__()
        self.T=T
        
    def define_residuals(self):
        self.add_weak(-self.T,testfunction("mesh"),lagrangian=False)         

class SolidNormalTraction(SolidTraction):
    """Imposes a normal traction on the solid interface. This is used to apply pressure loads on the solid surface.

    Args:
        P: Pressure to apply
    """
    def __init__(self,P:ExpressionOrNum):
        super().__init__(-P*var("normal"))
        
        


class FSIConnection(InterfaceEquations):
    """
    Can be added to the fluid side of a fluid-structure interaction interface to couple the mesh deformation and the velocity.    
    
    Args:
        velocity_offset: An offset to the velocity. You can e.g. add/substract a normal velocity to allow for fluid penetration into the solid. 
    """
    def __init__(self,*,velocity_offset:ExpressionOrNum=0):
        super().__init__()
        self.velocity_offset=velocity_offset
        
    def define_fields(self):
        pdom=self.get_parent_domain()
        coordspace=cast(FiniteElementSpaceEnum, pdom._coordinate_space)
        assert coordspace is not None and coordspace!="", "FSIConnection requires the parent domain to have a coordinate space"        
        if coordspace=="C1TB":
            coordspace="C1"
        elif coordspace=="C2TB":
            coordspace="C2"
        self.define_vector_field("_mesh_connection",coordspace,testscale=1/scale_factor("spatial"),scale=scale_factor("spatial")**2)
        self.define_vector_field("_velo_connection",coordspace,testscale=scale_factor("temporal")/scale_factor("spatial"),scale=scale_factor("pressure"))
        
    def define_residuals(self):
        from .navier_stokes import StokesEquations
        floweqs=self.get_parent_domain().get_equations().get_equation_of_type(StokesEquations,always_as_list=True)
        solideqs=self.get_opposite_parent_domain().get_equations().get_equation_of_type(BaseDeformableSolidEquations,always_as_list=True)
        if len(floweqs) != 1 or len(solideqs) != 1:
            raise RuntimeError("FSIConnection can only be used with a single fluid on the inside domain and a single solid equation on the outside domain")
        if not cast(BaseDeformableSolidEquations,solideqs[0]).scale_for_FSI:
            raise RuntimeError("FSIConnection can only be used with a solid equation that has scale_for_FSI=True. Otherwise, the stresses would be balanced wrongly.")
        lm,lmtest=var_and_test("_mesh_connection")
        lu,lutest=var_and_test("_velo_connection")
        x,xtest=var_and_test("mesh")
        u,utest=var_and_test("velocity")
        xsol,xsoltest=var_and_test("mesh",domain="|.")        
        usol=mesh_velocity()
        self.add_weak(x-xsol,lmtest)
        self.add_weak(lm,xtest)
        self.add_weak(u-usol+self.velocity_offset,lutest)
        self.add_weak(lu,utest)
        self.add_weak(-lu,xsoltest)
        
    def before_assigning_equations_postorder(self, mesh:"AnyMesh"):
        comps=[ x[-1] for x in cast(Equations,self.get_combined_equations())._vectorfields["_mesh_connection"] ]
        assert isinstance(mesh,InterfaceMesh)  # FSIConnection is an InterfaceEquations, so the mesh is always an InterfaceMesh here
        for direct in comps:
            self.pin_redundant_lagrange_multipliers(mesh,"_mesh_connection_"+direct,["mesh_"+direct])
            self.pin_redundant_lagrange_multipliers(mesh,"_velo_connection_"+direct,["velocity_"+direct])


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
