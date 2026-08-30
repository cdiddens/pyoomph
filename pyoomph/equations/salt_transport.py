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
 
"""
Electroneutral transport of dissolved salts, i.e. what an electrolyte does when nobody solves for
the electric potential.

This is the model to use for salt in an evaporating drop, for salt-induced Marangoni flow, and for
anything else where the double layer is thin, no external field is applied and the bulk is
electroneutral to any accuracy that matters. It is deliberately *not* electrohydrodynamics: there is
no potential, no Poisson equation and no Maxwell stress here.

**One field per salt, not per ion.** Without a potential the two ions of a salt cannot be given
separate degrees of freedom: they would diffuse apart at their own rates, and nothing in the system
would pull them back. Solving the salt instead makes electroneutrality structural -- the ion
concentrations follow by stoichiometry, exactly, on every mesh -- and the rate at which the pair
moves is the ambipolar diffusivity
:py:func:`~pyoomph.materials.generic.ambipolar_diffusivity`, which is what a Nernst-Planck solution
reduces to outside the double layer anyway.

**The ion concentrations still exist**, as substituted fields with the same names
:py:class:`~pyoomph.equations.electrostatics.NernstPlanckEquations` solves for (``c_Na_p`` for Na+,
see :py:func:`~pyoomph.equations.electrostatics.ion_fieldname_stem`). That is the whole point of the
naming: a surface tension law, a boundary condition, an observable or an output written against
``c_Na_p`` does not know or care which of the two models produced it, so moving a problem from here
to full Poisson-Nernst-Planck changes the equations and nothing else.
"""

from __future__ import annotations

from ..generic import Equations
from ..expressions import *
from ..expressions.phys_consts import *
from .stabilization import ScalarTransportEquations, ScalarTransportStabilization
from .electrostatics import ion_fieldname_stem
from ..typings import *

# Imported at runtime, not under TYPE_CHECKING: sphinx evaluates the annotations below and cannot
# resolve them otherwise - and AnyMaterialProperties is a string alias, so every class it names has to
# be here too. materials.generic reaches back into this package only from inside functions,
# so there is no import cycle.
from ..materials.generic import AnyMaterialProperties, DissolvedSalt, MaterialProperties, \
    BaseLiquidProperties, BaseGasProperties, BaseSolidProperties, PureSolidProperties, \
    PureLiquidProperties, PureGasProperties, MixtureLiquidProperties, MixtureGasProperties


class SaltTransportEquations(ScalarTransportEquations):
    r"""
    Advection-diffusion of dissolved salts, one field :math:`c_s` per salt,

    .. math:: \partial_t c_s + \nabla\cdot\left(c_s\vec{u}-D_s\nabla c_s\right)=0

    with :math:`D_s` the ambipolar diffusivity of the salt in this solvent. The concentration of each
    ion follows as :math:`c_i=\sum_s\nu_{i,s}c_s` and is supplied as a substituted field, so the
    solution is electroneutral by construction rather than by convergence.

    The natural boundary condition of the weak form is zero diffusive flux, which is what an
    impermeable, *stationary* wall wants. An evaporating free surface is not that: the liquid flows
    through it while the salt does not, so the salt piles up, and that condition is supplied by
    :py:class:`~pyoomph.equations.multi_component.MultiComponentNavierStokesInterface` alongside the
    one it already supplies for the volatile components. Without it, salt would leave with the vapour.

    Args:
        salts: The salts, as :py:class:`~pyoomph.materials.generic.DissolvedSalt` objects or a
            liquid material whose dissolved salts are read. Usually left to ``fluid_props``.
        fluid_props: The liquid. Supplies the salts, their concentrations and, through its own
            viscosity, their diffusivities.
        space: Finite element space of the concentrations. Defaults to ``"C2"``.
        wind: The advecting velocity. Defaults to ``var("velocity")``.
        temperature: Temperature at which the diffusivities are taken. ``None`` (the default) leaves
            it symbolic, i.e. ``var("temperature")``, which an isothermal problem answers with
            ``define_named_var(temperature=...)``.
        field_prefix: Prefix of the concentration field names, for both the salts and the ions.
            Defaults to ``"c_"``, i.e. the same names Nernst-Planck uses.
        diffusivities: Per-salt overrides of the ambipolar diffusivity, as ``{salt_name: D}``.
        dt_factor: Multiplicative factor on the time derivative.
        advection_by_parts: Integrate the advective term by parts.
        GCL: Write the transient term as the derivative of the whole integral and advect with the
            velocity *relative to the mesh*, i.e. the conservative ALE form, exactly as
            :py:class:`~pyoomph.equations.multi_component.CompositionAdvectionDiffusionEquations`
            does. On a mesh that follows an evaporating interface this conserves the dissolved
            amount to machine precision instead of to the order of the time stepping. Implies
            ``advection_by_parts``. Set for you by ``CompositionFlowEquations(GCL=True)``.
        concentration_scale: Named scale shared by all salts. Defaults to ``"ion_concentration"``,
            the same one the electrostatics module uses.
        set_bulk_initial_conditions: Initialise each salt at the concentration it was dissolved at.
        stabilization: Optional residual-based stabilization, see
            :py:class:`~pyoomph.equations.stabilization.ScalarTransportStabilization`.

    .. note::
        **Scaling to set on problem level.** The salt fields are scaled with the *named* scale
        ``concentration_scale`` (``"ion_concentration"`` by default), which is not a field name and
        therefore has to be set by ``problem.set_scaling(ion_concentration=...)``, or by
        :py:func:`~pyoomph.equations.electrostatics.set_electrostatic_scaling`.
    """

    def __init__(self,salts:"Sequence[DissolvedSalt] | AnyMaterialProperties | None"=None,*,
                 fluid_props:"AnyMaterialProperties | None"=None,
                 space:"FiniteElementSpaceEnum"="C2",wind:ExpressionOrNum=var("velocity"),
                 temperature:ExpressionNumOrNone=None,field_prefix:str="c_",
                 diffusivities:"dict[str,ExpressionOrNum] | None"=None,
                 dt_factor:ExpressionOrNum=1,scheme:"TimeSteppingScheme"="BDF2",
                 advection_by_parts:bool=False,GCL:bool=False,
                 concentration_scale:str="ion_concentration",
                 velocity_name_for_scaling:str="velocity",set_bulk_initial_conditions:bool=True,
                 consider_scaling:bool=True,
                 stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None):
        super().__init__()
        if salts is None:
            salts=fluid_props
        if salts is None:
            raise ValueError("SaltTransportEquations needs either salts or a fluid_props whose "+
                             "dissolved salts it can read")
        if not isinstance(salts,(list,tuple)):
            if fluid_props is None:
                fluid_props=cast("AnyMaterialProperties",salts)
            getter=getattr(salts,"get_salts",None)
            if getter is None:
                raise ValueError("The material '"+str(getattr(salts,"name","<unnamed>"))+
                                 "' cannot carry dissolved salts (only liquids can).")
            salts=list(getter().values())
        if not salts:
            raise ValueError("No salts are dissolved. Use add_salt() on the material first, or do "+
                             "not add SaltTransportEquations at all.")
        if fluid_props is None:
            raise ValueError("SaltTransportEquations needs a fluid_props to take the salt "+
                             "diffusivities from, or explicit diffusivities=...")
        # sorted: this fixes the order the fields are defined in, hence the dof numbering and the
        # output column order, so it must not depend on how the salts were dissolved.
        self.salts:"list[DissolvedSalt]"=sorted(salts,key=lambda s:s.name)
        self.fluid_props=fluid_props
        self.space:"FiniteElementSpaceEnum"=space
        self.wind=wind
        self.temperature=temperature
        self.field_prefix=field_prefix
        self.diffusivities:dict[str,ExpressionOrNum]=dict(diffusivities) if diffusivities else {}
        for n in self.diffusivities:
            if n not in {s.name for s in self.salts}:
                raise ValueError("diffusivities has an entry for '"+n+"', which is not one of the "+
                                 "dissolved salts "+str(sorted(s.name for s in self.salts)))
        self.dt_factor=dt_factor
        self.scheme:"TimeSteppingScheme"=scheme
        if GCL and not advection_by_parts:
            # The conservative form *is* the by-parts form: d/dt of the integral needs the advection
            # written against grad(v), or the two do not describe the same equation.
            advection_by_parts=True
        self.advection_by_parts=advection_by_parts
        self.GCL=GCL
        self.concentration_scale=concentration_scale
        self.velocity_name_for_scaling=velocity_name_for_scaling
        self.set_bulk_initial_conditions=set_bulk_initial_conditions
        self.consider_scaling=consider_scaling
        self.spatial_error_estimators=True
        self._init_stabilization(stabilization,velocity_scale=velocity_name_for_scaling)

    # ---- bookkeeping ------------------------------------------------------------------------------

    @property
    def salt_names(self)->list[str]:
        return [s.name for s in self.salts]

    def fieldname_of(self,salt_name:str)->str:
        """The name of the concentration field of one salt, e.g. ``c_NaCl``."""
        return self.field_prefix+ion_fieldname_stem(salt_name)

    def ion_fieldname_of(self,ion_name:str)->str:
        """The name of the (substituted) concentration field of one ion, e.g. ``c_Na_p`` -- the same
        name :py:class:`~pyoomph.equations.electrostatics.NernstPlanckEquations` would solve for."""
        return self.field_prefix+ion_fieldname_stem(ion_name)

    def _salt_of(self,fieldname:str)->"DissolvedSalt":
        for s in self.salts:
            if self.fieldname_of(s.name)==fieldname:
                return s
        raise KeyError("No salt corresponds to the field '"+fieldname+"'")

    def get_ion_names(self)->list[str]:
        """Every ion the dissolved salts contribute, in a deterministic order."""
        names:set[str]=set()
        for s in self.salts:
            names.add(s.cation.name)
            names.add(s.anion.name)
        return sorted(names)

    def get_diffusivity(self,salt_name:str)->ExpressionOrNum:
        """The ambipolar diffusivity of one salt in this solvent."""
        if salt_name in self.diffusivities:
            return self.diffusivities[salt_name]
        for s in self.salts:
            if s.name==salt_name:
                return s.get_diffusivity(cast("BaseLiquidProperties",self.fluid_props),self.temperature)
        raise KeyError("No salt named '"+salt_name+"'")

    def get_ion_concentration(self,ion_name:str,domain:"str | None"=None)->Expression:
        r""":math:`c_i=\sum_s\nu_{i,s}c_s`, the concentration one ion is present at."""
        return convert_to_expression(sum(s.stoichiometry_of(ion_name)*var(self.fieldname_of(s.name),domain=domain)
                                         for s in self.salts))

    def get_charge_density(self,domain:"str | None"=None)->Expression:
        r""":math:`\rho_\mathrm{e}=F\sum_i z_ic_i`, which is identically zero here -- that is what
        "electroneutral" means, and it is exact rather than approximate because the ion
        concentrations are stoichiometric multiples of the salt fields."""
        return Expression(0)

    def get_ionic_strength(self,domain:"str | None"=None)->Expression:
        r""":math:`I=\frac{1}{2}\sum_i z_i^2c_i`, the local ionic strength."""
        res:ExpressionOrNum=0
        for n in self.get_ion_names():
            z=self._charge_number_of(n)
            res=res+z**2*self.get_ion_concentration(n,domain)
        return convert_to_expression(res/2)

    def _charge_number_of(self,ion_name:str)->int:
        for s in self.salts:
            if s.cation.name==ion_name:
                return s.cation.charge_number
            if s.anion.name==ion_name:
                return s.anion.charge_number
        raise KeyError("No ion named '"+ion_name+"'")

    def get_information_string(self)->str:
        return "Electroneutral transport of "+", ".join(self.salt_names)

    # ---- the equations ----------------------------------------------------------------------------

    def define_fields(self):
        # Both models call an ion's concentration c_<ion>. Here it is a substitution, there a real
        # field, and a substitution that is silently shadowed by a field of the same name is the
        # kind of thing that produces a plausible wrong answer rather than an error.
        from .electrostatics import NernstPlanckEquations
        np_eqs=self.get_combined_equations().get_equation_of_type(NernstPlanckEquations)
        if isinstance(np_eqs,NernstPlanckEquations):
            clash=sorted(set(self.get_ion_names())&set(np_eqs.ion_names))
            if clash:
                raise RuntimeError("Both SaltTransportEquations and NernstPlanckEquations are on "+
                                   "this domain and both describe "+str(clash)+". Use one: the "+
                                   "electroneutral model here, or Nernst-Planck plus a potential "+
                                   "with CompositionFlowEquations(..., salts=False).")
        for s in self.salts:
            fn=self.fieldname_of(s.name)
            if self.consider_scaling:
                ts=scale_factor("spatial")/scale_factor(self.velocity_name_for_scaling)/scale_factor(fn) \
                    if self._advective() else \
                    scale_factor("spatial")**2/(self.get_diffusivity(s.name)*scale_factor(fn))
            else:
                ts=1
            self.define_scalar_field(fn,self.space,scale=scale_factor(self.concentration_scale),
                                     testscale=ts)
        # The ions, by stoichiometry, under the names Nernst-Planck would solve for. also_on_interface
        # because a surface tension law or a boundary condition written in ion concentrations is
        # evaluated on the interface, not in the bulk.
        for n in self.get_ion_names():
            fn=self.ion_fieldname_of(n)
            # The substitution is written nondimensionally, so the field needs its scale registered
            # separately -- otherwise var("c_Na_p") is a bare number here and a dimensional
            # concentration under Nernst-Planck, and an expression written for both would be right
            # in only one of them.
            self.set_scaling(**cast("dict[str,Any]",{fn:scale_factor(self.concentration_scale)}))
            self.define_field_by_substitution(fn,self.get_ion_concentration(n)/scale_factor(self.concentration_scale),
                                              also_on_interface=True)

    def define_residuals(self):
        ts:Callable[[Any],Any]=lambda x:time_scheme(self.scheme,x)
        for s in self.salts:
            fn=self.fieldname_of(s.name)
            c,c_test=var_and_test(fn)
            D=self.get_diffusivity(s.name)
            if self.advection_by_parts:
                if self.GCL:
                    # The conservative ALE form, term for term as the composition equations write
                    # it: the derivative of the whole integral, and advection with the velocity
                    # relative to the mesh. Its natural boundary condition is zero flux *through a
                    # moving boundary*, which is exactly what a non-volatile solute at an
                    # evaporating surface needs -- so that interface needs no term at all, and the
                    # dissolved amount is then conserved to machine precision.
                    self.add_residual(self.dt_factor*time_derivative_of_integral(
                        weak(c,c_test),scheme=self.scheme))
                    w=mesh_velocity(scheme=self.scheme)
                    self.add_residual(-weak(ts((self.wind-w)*c),grad(c_test)))
                else:
                    self.add_residual(weak(ts(self.dt_factor*partial_t(c)),c_test))
                    self.add_residual(-weak(ts(c*self.wind),grad(c_test)))
            else:
                self.add_residual(weak(ts(self.dt_factor*partial_t(c,ALE="auto")),c_test))
                if self._advective():
                    self.add_residual(weak(ts(dot(self.wind,grad(c))),c_test))
            self.add_residual(weak(ts(D*grad(c)),grad(c_test)))
            if self.set_bulk_initial_conditions and s.concentration is not None:
                self.set_initial_condition(fn,s.concentration,degraded_start="auto")
        self.add_stabilization_residuals(ts)

    def define_additional_functions(self):
        self.add_local_function("ionic_strength",self.get_ionic_strength())

    def define_error_estimators(self):
        if self.spatial_error_estimators:
            for s in self.salts:
                # nondim on both the field and the gradient, see NernstPlanckEquations: an error
                # estimator expression must be fully dimensionless.
                self.add_spatial_error_estimator(grad(nondim(self.fieldname_of(s.name)),nondim=True))

    def _advective(self)->bool:
        return not is_zero(convert_to_expression(self.wind))

    # ---- stabilization hooks ----------------------------------------------------------------------

    def stabilized_fieldnames(self)->list[str]:
        return [self.fieldname_of(s.name) for s in self.salts]

    def stabilization_wind(self)->ExpressionOrNum:
        return self.wind

    def stabilization_diffusivity(self,fieldname:str)->ExpressionOrNum:
        return self.get_diffusivity(self._salt_of(fieldname).name)

    def stabilization_residual_scale(self,fieldname:str)->ExpressionOrNum:
        return 1

    def strong_residual(self,fieldname:str)->Expression:
        s=self._salt_of(fieldname)
        c=var(fieldname)
        R=self.dt_factor*partial_t(c,ALE="auto")
        if self._advective():
            # The advective term as it is actually assembled, not as the non-conservative form
            # writes it: with advection_by_parts (which GCL implies) the Galerkin part is
            # -weak(c*wind,grad(v)), whose strong counterpart is div(wind*c).
            conservative=self.stab_cfg.conservative_residual
            if conservative=="auto":
                conservative=bool(self.advection_by_parts)
            R=R+(div(self.wind*c) if conservative else dot(self.wind,grad(c)))
        if self.stab_cfg.include_diffusion_in_residual:
            R=R-div(self.get_diffusivity(s.name)*grad(c))
        return R


class FrozenSaltConcentrations(Equations):
    """
    Defines the salt and ion concentrations of a material as *constants*, at the values they were
    dissolved at.

    The fallback for a liquid that carries salts when nothing solves for them: a surface tension law
    written against ``c_Na_p`` then still evaluates -- to the uniform bulk value, so the absolute
    surface tension is right and there is simply no gradient to drive a Marangoni flow. Without it,
    switching the salt transport off would leave the interface referencing a field nobody defines.

    Defines nothing at all when an electrolyte model is present on the same domain, which is checked
    at ``define_fields`` time, i.e. once the whole equation tree exists. That is what lets
    ``CompositionFlowEquations(..., salts=False)`` be the route to Poisson-Nernst-Planck and the
    route to a frozen salt at the same time.
    """
    def __init__(self,fluid_props:"AnyMaterialProperties",*,field_prefix:str="c_",
                 concentration_scale:str="ion_concentration"):
        super().__init__()
        self.fluid_props=fluid_props
        self.field_prefix=field_prefix
        self.concentration_scale=concentration_scale

    def define_fields(self):
        from .electrostatics import NernstPlanckEquations
        eqs=self.get_combined_equations()
        if eqs.get_equation_of_type(SaltTransportEquations) is not None \
                or eqs.get_equation_of_type(NernstPlanckEquations) is not None:
            return
        salts=cast("BaseLiquidProperties",self.fluid_props).get_salts()
        values:dict[str,ExpressionOrNum]={}
        for s in salts.values():
            values[self.field_prefix+ion_fieldname_stem(s.name)]=s.concentration
            for ion,nu in ((s.cation,s.cation_stoichiometry),(s.anion,s.anion_stoichiometry)):
                fn=self.field_prefix+ion_fieldname_stem(ion.name)
                values[fn]=values.get(fn,0)+nu*s.concentration
        for fn,value in values.items():
            self.set_scaling(**cast("dict[str,Any]",{fn:scale_factor(self.concentration_scale)}))
            self.define_field_by_substitution(fn,value/scale_factor(self.concentration_scale),
                                              also_on_interface=True)

    def get_information_string(self)->str:
        return "Frozen concentrations of "+", ".join(sorted(
            cast("BaseLiquidProperties",self.fluid_props).get_salts()))


class SaltConcentrationsFromMassFractions(Equations):
    """
    Supplies the salt and ion concentration fields of a material whose salts are *composition
    fields*, deriving them from the mass fractions.

    The counterpart of :py:class:`FrozenSaltConcentrations`, for the other treatment. Whichever way
    the salt is carried -- a concentration field of its own, a mass fraction, or the ion fields of
    Poisson-Nernst-Planck -- ``c_<ion>`` means the same thing afterwards, so a surface tension law,
    an activity coefficient or an observable written against it does not know or care which mode is
    running. That is what makes the modes interchangeable.
    """
    def __init__(self,fluid_props:"AnyMaterialProperties",*,field_prefix:str="c_",
                 concentration_scale:str="ion_concentration"):
        super().__init__()
        self.fluid_props=fluid_props
        self.field_prefix=field_prefix
        self.concentration_scale=concentration_scale

    def define_fields(self):
        liquid=cast("BaseLiquidProperties",self.fluid_props)
        rho=liquid.mass_density
        values:dict[str,ExpressionOrNum]={}
        for name,salt in liquid.get_salts().items():
            # Moles of salt per unit volume: the mass fraction times the density, over the molar mass.
            c_salt=rho*var("massfrac_"+name)/salt.molar_mass
            values[self.field_prefix+ion_fieldname_stem(name)]=c_salt
            for ion,nu in ((salt.cation,salt.cation_stoichiometry),
                           (salt.anion,salt.anion_stoichiometry)):
                fn=self.field_prefix+ion_fieldname_stem(ion.name)
                values[fn]=values.get(fn,0)+nu*c_salt
        for fn,value in values.items():
            self.set_scaling(**cast("dict[str,Any]",{fn:scale_factor(self.concentration_scale)}))
            self.define_field_by_substitution(fn,value/scale_factor(self.concentration_scale),
                                              also_on_interface=True)

    def get_information_string(self)->str:
        return "Concentrations from the mass fractions of "+", ".join(sorted(
            cast("BaseLiquidProperties",self.fluid_props).get_salts()))


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
