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
 
 
from ..generic import Equations, InterfaceEquations
from ..expressions import * #Import grad et al
from ..expressions.phys_consts import *
from .navier_stokes import StokesEquations
from .electrostatics import ElectricPotentialEquations
from ..meshes.mesh import AnyMesh, InterfaceMesh
from ..typings import *

if TYPE_CHECKING:
    from ..generic import Problem


def maxwell_stress_tensor(permittivity:ExpressionOrNum,electric_field:Expression,*,
                          electrostriction:ExpressionNumOrNone=None)->Expression:
    r"""
    The Maxwell stress tensor

    .. math:: \vec{\vec{\sigma}}_\mathrm{M} = \varepsilon\,\vec{E}\otimes\vec{E}
              - \tfrac{1}{2}\left(\varepsilon-\rho\frac{\partial\varepsilon}{\partial\rho}\right)
                |\vec{E}|^2\,\vec{\vec{I}}

    the divergence of which is the electric body force.

    For a **constant** permittivity and no electrostriction,
    :math:`\nabla\cdot\vec{\vec{\sigma}}_\mathrm{M}=\rho_\mathrm{e}\vec{E}` holds *exactly*: since
    :math:`\vec{E}=-\nabla\phi` is a gradient, :math:`(\vec{E}\cdot\nabla)\vec{E}` equals
    :math:`\tfrac{1}{2}\nabla|\vec{E}|^2` identically and the two non-Gauss terms cancel. That is why
    :py:class:`MaxwellStressEquations` and :py:class:`ElectricBodyForceEquations` describe the same
    PDE -- but not the same *weak form*, see the warning in the latter.

    Args:
        permittivity: The absolute permittivity.
        electric_field: The electric field vector.
        electrostriction: :math:`\rho\,\partial\varepsilon/\partial\rho`. ``None`` (the default)
            gives the incompressible/constant-permittivity form.
    """
    eps=permittivity
    trace_coeff=eps if electrostriction is None else eps-electrostriction
    E=electric_field
    return eps*dyadic(E,E)-identity_matrix()*(trace_coeff*dot(E,E)/2)


def electric_body_force(permittivity:ExpressionOrNum,electric_field:Expression,
                        charge_density:ExpressionOrNum,*,electrostriction:ExpressionNumOrNone=None,
                        coulomb:bool=True,dielectrophoresis:bool=True)->Expression:
    r"""
    The electric body force, i.e. :math:`\nabla\cdot\vec{\vec{\sigma}}_\mathrm{M}` written out:

    .. math:: \vec{f}_\mathrm{e} = \rho_\mathrm{e}\vec{E}
              - \tfrac{1}{2}|\vec{E}|^2\nabla\varepsilon
              + \tfrac{1}{2}\nabla\!\left(\rho\frac{\partial\varepsilon}{\partial\rho}|\vec{E}|^2\right)

    The three terms are the Coulomb force on the free charge, the dielectrophoretic force on a
    permittivity gradient, and electrostriction. Pass this as ``bulkforce=`` to the (Navier-)Stokes
    equations, or use :py:class:`ElectricBodyForceEquations`.

    Args:
        permittivity: The absolute permittivity, possibly field dependent.
        electric_field: The electric field vector.
        charge_density: The free charge density.
        electrostriction: :math:`\rho\,\partial\varepsilon/\partial\rho`, or ``None`` to omit.
        coulomb: Include the Coulomb term. Defaults to True.
        dielectrophoresis: Include the permittivity-gradient term. Defaults to True.
    """
    E=electric_field
    res:ExpressionOrNum=0
    if coulomb:
        res=res+charge_density*E
    if dielectrophoresis:
        res=res-dot(E,E)*grad(permittivity)/2
    if electrostriction is not None:
        res=res+grad(electrostriction*dot(E,E))/2
    return convert_to_expression(res)


class _ElectricFlowCoupling(Equations):
    """Shared plumbing: find the co-located potential equations and the flow they act on."""

    def __init__(self,*,permittivity:ExpressionNumOrNone=None,potential_name:"str | None"=None,
                 velocity_name:"str | None"=None,electrostriction:ExpressionNumOrNone=None,
                 time_scheme:"TimeSteppingScheme | None"=None):
        super().__init__()
        self.permittivity=permittivity
        self.potential_name=potential_name
        self.velocity_name=velocity_name
        self.electrostriction=electrostriction
        self.time_scheme:"TimeSteppingScheme | None"=time_scheme

    def get_electrostatics(self,domain:"str | FiniteElementCodeGenerator | None"=None)->ElectricPotentialEquations:
        eqs=self.get_combined_equations().get_equation_of_type(ElectricPotentialEquations)
        if isinstance(eqs,list):
            if len(eqs)!=1:
                raise RuntimeError(type(self).__name__+" found "+str(len(eqs))+" ElectricPotentialEquations "+
                                   "on this domain; it cannot tell which one drives the flow.")
            eqs=eqs[0]
        if not isinstance(eqs,ElectricPotentialEquations):
            raise RuntimeError(type(self).__name__+" needs an ElectricPotentialEquations (or a subclass, "+
                               "e.g. PoissonBoltzmannEquations) on the same domain.")
        return eqs

    def get_flow(self)->StokesEquations:
        eqs=self.get_combined_equations().get_equation_of_type(StokesEquations)
        if isinstance(eqs,list):
            if len(eqs)!=1:
                raise RuntimeError(type(self).__name__+" found "+str(len(eqs))+" StokesEquations on this domain")
            eqs=eqs[0]
        if not isinstance(eqs,StokesEquations):
            raise RuntimeError(type(self).__name__+" needs (Navier-)Stokes equations on the same domain")
        return eqs

    def _potential_name(self)->str:
        return self.potential_name if self.potential_name is not None else self.get_electrostatics().name

    def _velocity_name(self)->str:
        return self.velocity_name if self.velocity_name is not None else self.get_flow().velocity_name

    def _permittivity(self,domain:"str | FiniteElementCodeGenerator | None"=None)->ExpressionOrNum:
        if self.permittivity is not None:
            return self.permittivity if domain is None else \
                   evaluate_in_domain(convert_to_expression(self.permittivity),domain)
        return self.get_electrostatics().get_permittivity(domain)

    def _electric_field(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        return -grad(var(self._potential_name(),domain=domain))

    def get_maxwell_stress(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        r"""
        The Maxwell stress tensor of this domain, as an expression.

        **Pass** ``domain=".."`` **when calling this from an interface**: on an interface ``grad``
        is the surface gradient, so without it the normal part of :math:`\vec{E}` is silently lost
        and the resulting traction is wrong rather than merely inaccurate.

        The total stress of the fluid is this plus the viscous and pressure contributions of the
        (Navier-)Stokes equations, so a drag integral over a wall reads

        .. code-block:: python

            sigma = -p*identity_matrix() + 2*mu*sym(grad(u)) + eqs.get_maxwell_stress(domain="..")
            traction = matproduct(sigma, var("normal"))
        """
        return maxwell_stress_tensor(self._permittivity(domain),self._electric_field(domain),
                                     electrostriction=self.electrostriction)

    def get_electric_traction(self,normal:Expression,
                              bulk_domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        r"""
        :math:`\vec{n}\cdot\vec{\vec{\sigma}}_\mathrm{M}`, i.e. what this domain's Maxwell stress
        deposits on one of its boundaries.

        Structurally the analogue of
        :py:meth:`~pyoomph.equations.navier_stokes.StokesEquations.get_stabilization_traction`, and
        for the same reason -- a bulk term written against ``grad(v)`` leaves a surface integral
        behind. The crucial difference is that the stabilization footprint is *spurious* and is
        subtracted by every traction boundary condition, whereas this one is **physical** and must
        be kept. What a free surface still has to add is the traction of the *other* phase, which is
        what :py:class:`MaxwellStressInterface` is for.
        """
        return matproduct(self.get_maxwell_stress(bulk_domain),normal)


class MaxwellStressEquations(_ElectricFlowCoupling):
    r"""
    Adds the Maxwell stress to the momentum equation of a co-located (Navier-)Stokes,

    .. math:: \text{residual} \mathrel{+}= (\vec{\vec{\sigma}}_\mathrm{M},\nabla\vec{v})\,,

    which is the same residual row and the same sign convention the viscous and pressure stresses
    use, so this is exactly "the Maxwell stress is in the stress tensor". It is written as a
    separate ``Equations`` object -- the idiom
    :py:class:`~pyoomph.equations.viscoelastic.ViscoelasticEquations` uses for the polymer stress --
    so that nothing about the flow equations has to change::

        eqs = NavierStokesEquations(...) + ElectricPotentialEquations(...) + MaxwellStressEquations()

    .. note::
        **You usually do not need this class.**
        :py:class:`~pyoomph.equations.electrostatics.ElectricPotentialEquations` already applies the
        Maxwell stress to a co-located (Navier-)Stokes by itself
        (``add_maxwell_stress_to_momentum=True``, the default), so the snippet above is redundant --
        and, because it would count the stress twice, it is refused unless that flag is turned off.

        Reach for this class when you need something the automatic path does not offer: a different
        permittivity or potential from the one the field equations use, an explicit ``time_scheme``,
        or the stress of a potential solved on another domain.

    The equivalent one-liner is ``NavierStokesEquations(..., extra_stress=maxwell_stress_tensor(...))``.

    Args:
        permittivity: Overrides the permittivity of the co-located potential equations.
        potential_name / velocity_name: Field names, taken from the co-located equations by default.
        electrostriction: :math:`\rho\,\partial\varepsilon/\partial\rho`, see
            :py:func:`maxwell_stress_tensor`.
        time_scheme: Time stepping scheme applied to the stress, matching the momentum equation's.
        output_stress: Add ``add_local_function`` entries for the stress components.
    """

    def __init__(self,*,permittivity:ExpressionNumOrNone=None,potential_name:"str | None"=None,
                 velocity_name:"str | None"=None,electrostriction:ExpressionNumOrNone=None,
                 time_scheme:"TimeSteppingScheme | None"=None,output_stress:bool=False):
        super().__init__(permittivity=permittivity,potential_name=potential_name,
                         velocity_name=velocity_name,electrostriction=electrostriction,
                         time_scheme=time_scheme)
        self.output_stress=output_stress

    def get_information_string(self)->str:
        return "Maxwell stress of <"+str(self.potential_name or "phi")+"> in the momentum equation"

    def define_scaling(self):
        flow=self.get_flow()
        self.add_named_numerical_factor(maxwell_stress_in_momentum_eq=
            scale_factor("permittivity")*scale_factor("electric_field")**2
            *test_scale_factor(flow.velocity_name)/scale_factor("spatial"))

    def define_residuals(self):
        flow=self.get_flow()
        if flow.extra_stress is not None:
            raise RuntimeError("The (Navier-)Stokes equations on this domain already carry an "+
                               "extra_stress. Adding MaxwellStressEquations on top of it would put the "+
                               "Maxwell stress into the momentum equation twice. Use one or the other.")
        # The co-located field equations apply the stress themselves by default; that clash is
        # detected there, where the flag that resolves it lives.
        self.get_electrostatics()
        stress=self.get_maxwell_stress()
        if self.time_scheme is not None:
            stress=time_scheme(self.time_scheme,stress)
        self.add_residual(weak(stress,grad(testfunction(self._velocity_name()))))

    def define_additional_functions(self):
        if not self.output_stress:
            return
        s=self.get_maxwell_stress()
        comps="xyz"
        for i in range(self.get_nodal_dimension()):
            for j in range(self.get_nodal_dimension()):
                self.add_local_function("maxwell_stress_"+comps[i]+comps[j],s[i,j])


class ElectricBodyForceEquations(_ElectricFlowCoupling):
    r"""
    Adds the electric body force to the momentum equation of a co-located (Navier-)Stokes,

    .. math:: \text{residual} \mathrel{-}= (\vec{f}_\mathrm{e},\vec{v})\,,\qquad
              \vec{f}_\mathrm{e}=\rho_\mathrm{e}\vec{E}-\tfrac{1}{2}|\vec{E}|^2\nabla\varepsilon+\ldots

    .. warning::
        This is the same **PDE** as :py:class:`MaxwellStressEquations` for a constant permittivity,
        but **not the same weak form**: the two differ by the surface integral
        :math:`\langle\vec{n}\cdot\vec{\vec{\sigma}}_\mathrm{M},\vec{v}\rangle` on *every* boundary
        of the domain. Two consequences, both of which have bitten people:

        * The **pressure differs** by :math:`\varepsilon|\vec{E}|^2/2`. The stress form absorbs it,
          the body-force form does not. Comparing pressures between the two, or reading a pressure
          off without knowing which form is in use, gives a wrong answer.
        * On a **free surface** the body-force form supplies no electric traction at all, so the
          normal *and* tangential stress balance is missing the entire Maxwell contribution. That is
          silent -- the simulation converges and the interface shape is simply wrong. Either use
          :py:class:`MaxwellStressEquations`, or add :py:class:`MaxwellStressInterface` with
          ``mode="jump"`` on every free surface.

        For a closed domain with no free surface, the two are interchangeable.

    Requires ``add_maxwell_stress_to_momentum=False`` on the co-located
    :py:class:`~pyoomph.equations.electrostatics.ElectricPotentialEquations`, whose default is to
    apply the stress route; having both is refused rather than silently doubling the force.

    Args:
        charge_density: Overrides the charge density of the co-located potential equations.
        coulomb / dielectrophoresis: Switch the individual terms off, e.g. to isolate one effect.
        Other arguments as in :py:class:`MaxwellStressEquations`.
    """

    def __init__(self,*,permittivity:ExpressionNumOrNone=None,charge_density:ExpressionNumOrNone=None,
                 potential_name:"str | None"=None,velocity_name:"str | None"=None,
                 electrostriction:ExpressionNumOrNone=None,coulomb:bool=True,
                 dielectrophoresis:bool=True,time_scheme:"TimeSteppingScheme | None"=None):
        super().__init__(permittivity=permittivity,potential_name=potential_name,
                         velocity_name=velocity_name,electrostriction=electrostriction,
                         time_scheme=time_scheme)
        self.charge_density=charge_density
        self.coulomb=coulomb
        self.dielectrophoresis=dielectrophoresis

    def get_information_string(self)->str:
        return "electric body force of <"+str(self.potential_name or "phi")+"> in the momentum equation"

    def get_electric_body_force(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        """The body force as an expression. Pass ``domain=".."`` when calling from an interface."""
        rho=self.charge_density
        if rho is None:
            rho=self.get_electrostatics().get_charge_density(domain)
            if rho is None:
                rho=0
        elif domain is not None:
            rho=evaluate_in_domain(convert_to_expression(rho),domain)
        return electric_body_force(self._permittivity(domain),self._electric_field(domain),rho,
                                   electrostriction=self.electrostriction,coulomb=self.coulomb,
                                   dielectrophoresis=self.dielectrophoresis)

    def define_residuals(self):
        f=self.get_electric_body_force()
        if self.time_scheme is not None:
            f=time_scheme(self.time_scheme,f)
        self.add_residual(-weak(f,testfunction(self._velocity_name())))


class MaxwellStressInterface(InterfaceEquations):
    r"""
    Applies the electric traction on a free surface or a two-fluid interface,

    .. math:: \vec{n}\cdot\left(\vec{\vec{\sigma}}^\mathrm{hydro}_\mathrm{in}
              -\vec{\vec{\sigma}}^\mathrm{hydro}_\mathrm{out}\right)
              = -\gamma\kappa\vec{n}+\nabla_\mathrm{s}\gamma
              - \vec{n}\cdot\left[\vec{\vec{\sigma}}_\mathrm{M}\right]\,,

    i.e. the electric part of the jump condition. Both the normal component (which deforms the
    interface) and the tangential one (which drives the Taylor circulation of a leaky-dielectric
    drop) are included -- this is why it cannot be folded into
    :py:class:`~pyoomph.equations.navier_stokes.NavierStokesFreeSurface`, whose
    ``additional_normal_traction`` is a scalar. Both write into the same velocity test row, so they
    simply compose::

        ifeqs = NavierStokesFreeSurface(surface_tension=gamma) + MaxwellStressInterface()

    ``mode`` says which side's Maxwell stress this equation has to supply, which depends on how each
    bulk is already coupled. Note that with the **default**
    ``add_maxwell_stress_to_momentum=True`` on the field equations, a bulk that has flow already
    carries its own traction, so ``"opposite_only"`` is the common case for a free surface against a
    passive phase, and a two-fluid interface with flow on both sides needs nothing here at all:

    ========================  ================================================================
    ``mode``                  use when
    ========================  ================================================================
    ``"jump"``                neither bulk carries the Maxwell stress in its momentum row
                              (e.g. both use :py:class:`ElectricBodyForceEquations`), or the
                              opposite side has no flow at all
    ``"opposite_only"``       the parent bulk already uses
                              :py:class:`MaxwellStressEquations`, so its own traction is
                              already in the natural boundary condition
    ``"parent_only"``         the mirror image of the above
    ========================  ================================================================

    With :py:class:`MaxwellStressEquations` in **both** bulks nothing is needed here at all: each
    natural boundary condition already carries its own :math:`\vec{n}\cdot\vec{\vec{\sigma}}_\mathrm{M}`
    and the jump balances by itself. Adding this class on top would double-count.

    Args:
        mode: See the table above. Defaults to ``"jump"``.
        potential_name: Name of the potential field. Taken from the parent equations by default.
        permittivity / opposite_permittivity: Override the two permittivities.
        outside_potential_name: Name of the potential on the opposite side, if it differs.
        electrostriction: See :py:func:`maxwell_stress_tensor`.
        subtract_stabilization_traction: Subtract the footprint a *stabilized* bulk flow leaves on
            this boundary, which is spurious and must not be balanced against the surface tension.
            Zero for unstabilized equations. Defaults to True.
    """
    required_parent_type = StokesEquations

    def __init__(self,*,mode:Literal["jump","parent_only","opposite_only"]="jump",
                 potential_name:"str | None"=None,permittivity:ExpressionNumOrNone=None,
                 opposite_permittivity:ExpressionNumOrNone=None,
                 outside_potential_name:"str | None"=None,
                 electrostriction:ExpressionNumOrNone=None,
                 subtract_stabilization_traction:bool=True):
        super().__init__()
        if mode not in {"jump","parent_only","opposite_only"}:
            raise ValueError("mode must be 'jump', 'parent_only' or 'opposite_only', not "+repr(mode))
        self.mode=mode
        self.potential_name=potential_name
        self.permittivity=permittivity
        self.opposite_permittivity=opposite_permittivity
        self.outside_potential_name=outside_potential_name
        self.electrostriction=electrostriction
        self.subtract_stabilization_traction=subtract_stabilization_traction

    def _side_stress(self,inside:bool)->Expression:
        domain=".." if inside else "|.."
        if inside:
            eqs=self.get_parent_domain().get_equations().get_equation_of_type(ElectricPotentialEquations)
            eps,name=self.permittivity,self.potential_name
        else:
            oppdom=self.get_opposite_parent_domain(raise_error_if_none=False)
            if oppdom is None:
                raise RuntimeError("MaxwellStressInterface with mode='"+self.mode+"' needs a meshed "+
                                   "domain on the other side of the interface.")
            eqs=oppdom.get_equations().get_equation_of_type(ElectricPotentialEquations)
            eps,name=self.opposite_permittivity,self.outside_potential_name or self.potential_name
        if isinstance(eqs,list):
            eqs=eqs[0] if len(eqs)==1 else None
        if eps is None:
            if not isinstance(eqs,ElectricPotentialEquations):
                raise RuntimeError("MaxwellStressInterface cannot find the permittivity on the "+
                                   ("inside" if inside else "opposite")+" side. Pass it explicitly.")
            eps=eqs.get_permittivity(domain)
        else:
            eps=evaluate_in_domain(convert_to_expression(eps),domain)
        if name is None:
            if not isinstance(eqs,ElectricPotentialEquations):
                raise RuntimeError("MaxwellStressInterface cannot find the potential name on the "+
                                   ("inside" if inside else "opposite")+" side. Pass it explicitly.")
            name=eqs.name
        # domain=".."/"|..": on an interface grad() is the SURFACE gradient, which would drop exactly
        # the normal field the traction is mostly made of.
        E=-grad(var(name,domain=domain))
        return maxwell_stress_tensor(eps,E,electrostriction=self.electrostriction)

    def define_residuals(self):
        flow=self.get_parent_equations(StokesEquations)
        assert isinstance(flow,StokesEquations)
        _,u_test=var_and_test(flow.velocity_name)
        n=self.get_normal()
        stress:Expression=Expression(0)
        if self.mode in ("jump","parent_only"):
            stress=stress+self._side_stress(True)
        if self.mode in ("jump","opposite_only"):
            stress=stress-self._side_stress(False)
        # An imposed traction t enters as -weak(t, u_test); the electric part of the jump condition
        # moves -n.[sigma_M] onto the right-hand side, hence the plus here.
        self.add_residual(weak(matproduct(stress,n),u_test))
        if self.subtract_stabilization_traction:
            tstab=flow.get_stabilization_traction(n,self.get_parent_domain())
            if not is_zero(tstab):
                self.add_residual(-weak(tstab,u_test))


def helmholtz_smoluchowski_velocity(permittivity:ExpressionOrNum,zeta_potential:ExpressionOrNum,
                                    dynamic_viscosity:ExpressionOrNum,*,potential_name:str="phi",
                                    tangential_field:"Expression | None"=None)->Expression:
    r"""
    The Helmholtz-Smoluchowski slip velocity :math:`\vec{u}_\mathrm{s}
    = -\varepsilon\zeta\vec{E}_\mathrm{t}/\mu`.

    The tangential field defaults to ``-grad(var(potential_name))`` **evaluated on the interface**,
    where ``grad`` is the surface gradient -- which is exactly :math:`\vec{E}_\mathrm{t}`. This is
    one of the few places in this module where *not* reaching into the parent domain is the correct
    thing to do, so do not "fix" it by passing ``domain=".."``.

    Args:
        permittivity: The solvent permittivity.
        zeta_potential: The zeta potential of the wall.
        dynamic_viscosity: The solvent viscosity.
        potential_name: Name of the potential field.
        tangential_field: Override the tangential field expression.
    """
    Et=tangential_field if tangential_field is not None else -grad(var(potential_name))
    return -permittivity*zeta_potential*Et/dynamic_viscosity


class ElectroosmoticSlip(InterfaceEquations):
    r"""
    Electroosmotic slip for an **unresolved** (thin) double layer,

    .. math:: \vec{u}_\mathrm{t} = -\frac{\varepsilon\zeta}{\mu}\vec{E}_\mathrm{t}
                                   + \vec{u}_\mathrm{wall,t}\,,

    imposed with a Lagrange multiplier, following
    :py:class:`~pyoomph.equations.navier_stokes.NavierStokesPrescribedNormalVelocity` but
    tangentially. This is the *alternative* to resolving the Debye layer with Poisson-Nernst-Planck:
    the layer is collapsed into a boundary condition and the bulk is electroneutral.

    Its validity is bounded by the Dukhin number :math:`\mathrm{Du}=K_\mathrm{s}/(\sigma_\mathrm{c}a)`
    -- once surface conduction matters the slip formula stops being the whole story.

    Args:
        zeta_potential: The zeta potential of the wall.
        permittivity / dynamic_viscosity: Taken from the parent equations if not given.
        potential_name: Name of the potential field. Defaults to ``"phi"``.
        wall_velocity: Velocity of the wall itself, added to the slip. Defaults to 0.
        impose_no_penetration: Also enforce :math:`\vec{u}\cdot\vec{n}=0`. Defaults to True.
        lagr_mult_name: Name of the Lagrange multiplier field.
    """
    required_parent_type = StokesEquations

    def __init__(self,*,zeta_potential:ExpressionNumOrNone=None,permittivity:ExpressionNumOrNone=None,
                 dynamic_viscosity:ExpressionNumOrNone=None,potential_name:str="phi",
                 wall_velocity:ExpressionOrNum=0,impose_no_penetration:bool=True,
                 lagr_mult_name:str="_lagr_eo_slip"):
        super().__init__()
        self.zeta_potential=zeta_potential
        self.permittivity=permittivity
        self.dynamic_viscosity=dynamic_viscosity
        self.potential_name=potential_name
        self.wall_velocity=wall_velocity
        self.impose_no_penetration=impose_no_penetration
        self.lagr_mult_name=lagr_mult_name

    def _slip_velocity(self)->Expression:
        flow=self.get_parent_equations(StokesEquations)
        assert isinstance(flow,StokesEquations)
        mu=self.dynamic_viscosity if self.dynamic_viscosity is not None else flow.dynamic_viscosity
        eps=self.permittivity
        if eps is None:
            pot=self.get_parent_domain().get_equations().get_equation_of_type(ElectricPotentialEquations)
            if isinstance(pot,list):
                pot=pot[0] if len(pot)==1 else None
            if not isinstance(pot,ElectricPotentialEquations):
                raise RuntimeError("ElectroosmoticSlip cannot find the permittivity; pass it explicitly.")
            eps=pot.get_permittivity("..")
        if self.zeta_potential is None:
            raise ValueError("ElectroosmoticSlip needs a zeta_potential")
        u_s=helmholtz_smoluchowski_velocity(eps,self.zeta_potential,mu,potential_name=self.potential_name)
        return u_s+self.wall_velocity if not is_zero(convert_to_expression(self.wall_velocity)) else u_s

    def define_fields(self):
        flow=self.get_parent_equations(StokesEquations)
        assert isinstance(flow,StokesEquations)
        space=flow.get_velocity_space_from_mode(for_interface=True)
        self.define_vector_field(self.lagr_mult_name,space,
                                 scale=1/test_scale_factor(flow.velocity_name),
                                 testscale=1/scale_factor(flow.velocity_name))

    def define_residuals(self):
        flow=self.get_parent_equations(StokesEquations)
        assert isinstance(flow,StokesEquations)
        n=self.get_normal()
        u,u_test=var_and_test(flow.velocity_name)
        l,l_test=var_and_test(self.lagr_mult_name)
        tang=lambda v:v-dot(v,n)*n
        u_s=self._slip_velocity()
        self.add_residual(weak(tang(u-u_s),tang(l_test)))
        self.add_residual(weak(tang(l),u_test))
        if self.impose_no_penetration:
            self.add_residual(weak(dot(u,n),dot(l_test,n)))
            self.add_residual(weak(dot(l,n)*n,u_test))
        else:
            # Nothing constrains the normal multiplier, so pin it rather than leave the Jacobian
            # singular; the normal direction is then do-nothing (natural).
            self.add_residual(weak(dot(l,n),dot(l_test,n)))
        tstab=flow.get_stabilization_traction(n,self.get_parent_domain())
        if not is_zero(tstab):
            self.add_residual(-weak(tang(tstab),tang(u_test)))

    def before_assigning_equations_postorder(self,mesh:"AnyMesh"):
        assert isinstance(mesh,InterfaceMesh)
        flow=self.get_parent_equations(StokesEquations)
        assert isinstance(flow,StokesEquations)
        self.pin_redundant_lagrange_multipliers(mesh,self.lagr_mult_name,flow.velocity_name)
        super().before_assigning_equations_postorder(mesh)


def lippmann_surface_tension(surface_tension_at_pzc:ExpressionOrNum,*,capacitance:ExpressionOrNum,
                             potential:ExpressionOrNum,potential_of_zero_charge:ExpressionOrNum=0)->Expression:
    r"""
    The integrated Lippmann equation :math:`\mathrm{d}\gamma/\mathrm{d}V=-q_\mathrm{s}` with
    :math:`q_\mathrm{s}=C_\mathrm{dl}(V-V_\mathrm{pzc})`, i.e.

    .. math:: \gamma = \gamma_\mathrm{pzc} - \tfrac{1}{2}C_\mathrm{dl}(V-V_\mathrm{pzc})^2\,.

    Electrocapillarity: charging an interface always *lowers* its tension, which is what makes
    electrowetting work. Feed the result straight to
    :py:class:`~pyoomph.equations.navier_stokes.NavierStokesFreeSurface`, which accepts an arbitrary
    field-dependent expression and produces the Marangoni term from it automatically.

    .. note::
        This is the **thin-double-layer** description. With a resolved double layer (
        :py:func:`~pyoomph.equations.electrostatics.PoissonNernstPlanck`) the ion osmotic pressure
        and the Maxwell stress already produce the electrocapillary effect, and adding this on top
        counts it twice.
    """
    return surface_tension_at_pzc-capacitance*(potential-potential_of_zero_charge)**2/2


def surface_charge_surface_tension(surface_tension_at_pzc:ExpressionOrNum,*,
                                   capacitance:ExpressionOrNum,
                                   surface_charge_density:ExpressionOrNum)->Expression:
    r"""
    The Lippmann relation written in the charge variable,
    :math:`\gamma=\gamma_\mathrm{pzc}-q_\mathrm{s}^2/(2C_\mathrm{dl})`.

    This is the form to use together with a dynamically solved surface charge, since it depends on
    the charge rather than on an absolute potential.
    """
    return surface_tension_at_pzc-surface_charge_density**2/(2*capacitance)


def debye_huckel_surface_tension(surface_tension_at_zero_charge:ExpressionOrNum,*,
                                 permittivity:ExpressionOrNum,debye_length:ExpressionOrNum,
                                 zeta_potential:ExpressionOrNum)->Expression:
    r"""
    The diffuse-layer free energy in the Debye-Hueckel limit,
    :math:`\gamma=\gamma_0-\varepsilon\zeta^2/(2\lambda_\mathrm{D})`, i.e.
    :py:func:`lippmann_surface_tension` with the diffuse-layer capacitance
    :math:`C_\mathrm{dl}=\varepsilon/\lambda_\mathrm{D}`.
    """
    return surface_tension_at_zero_charge-permittivity*zeta_potential**2/(2*debye_length)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
