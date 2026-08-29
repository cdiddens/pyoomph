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
 
 
from .._deprecation import deprecated_kwargs as _deprecated_kwargs, deprecated_attribute_alias as _deprecated_attribute_alias
from ..generic import Equations, InterfaceEquations
from ..generic.codegen import BaseEquations
from ..expressions import * #Import grad et al
from ..expressions.phys_consts import * # epsilon_0, faraday_constant, gas_constant, ... (and the units)
from .generic import ProjectExpression, DirichletBC, get_interface_field_connection_space
from .generic import GlobalLagrangeMultiplier, WeakContribution, Scaling, TestScaling
from .generic import interface_transport_velocities
from ..generic.codegen import sorted_field_kwargs
from .stabilization import ScalarTransportEquations, ScalarTransportStabilization
from .poisson import farfield_monopole_residual
from ..meshes.mesh import AnyMesh, InterfaceMesh
from ..typings import *

if TYPE_CHECKING:
    from ..generic import Problem
    from ..generic.codegen import FiniteElementCodeGenerator
    from ..materials.generic import AnyMaterialProperties, BaseInterfaceProperties
    # AnyMaterialProperties is a string-valued TypeAlias (see materials/generic.py) whose members
    # (MaterialProperties, PureLiquidProperties, ...) are only resolvable via a wildcard import -
    # needed so tools that resolve forward references in type annotations (e.g. sphinx_autodoc_typehints)
    # can look them up in this module's namespace too, same as materials/generic.py itself does.
    from ..materials.generic import *
    from .navier_stokes import StokesEquations


def ion_fieldname_stem(ion_name:str)->str:
    """
    The field-name stem of an ion, i.e. its name made into a valid variable name.

    Ions are naturally written ``"Na+"``, ``"Cl-"``, ``"SO4 2-"``, but a pyoomph field name may only
    contain letters, digits and underscores. The charge signs become ``_p`` and ``_m`` and anything
    else invalid becomes ``_``, so ``"Na+"`` is solved for as ``c_Na_p``. That is the name a
    :py:class:`~pyoomph.equations.generic.DirichletBC` on it has to use, and
    :py:meth:`NernstPlanckEquations.fieldname_of` is the way to ask for it rather than to guess.
    """
    res=ion_name.replace("+","_p").replace("-","_m")
    res="".join(c if (c.isalnum() and c.isascii()) or c=="_" else "_" for c in res)
    while "__" in res:
        res=res.replace("__","_")
    res=res.strip("_")
    if not res or res[0].isdigit():
        raise ValueError("Cannot build a field name from the ion name "+repr(ion_name))
    return res


def ions_from_material(fluid_props:"AnyMaterialProperties",*,
                       temperature:ExpressionNumOrNone=var("temperature"))->list["IonSpec"]:
    """
    The :py:class:`IonSpec` list of whatever is dissolved in a liquid, in a deterministic order.

    Lets the electrolyte equations be driven from a material rather than from literals::

        import pyoomph.materials.ions   # registers the standard ions
        water = get_pure_liquid("water")
        water.add_salt("Na+", "Cl-", 1*milli*mol/liter)
        eqs = PoissonNernstPlanck(ions_from_material(water), fluid_props=water)

    which is also what passing ``fluid_props=`` directly does.
    """
    getter=getattr(fluid_props,"get_ions",None)
    if getter is None:
        raise ValueError("The material '"+str(getattr(fluid_props,"name","<unnamed>"))+
                         "' cannot carry dissolved ions (only liquids can).")
    ions=getter()
    if not ions:
        raise ValueError("No ions are dissolved in '"+str(getattr(fluid_props,"name","<unnamed>"))+
                         "'. Use its add_ion()/add_salt() first, or pass the ions explicitly.")
    # get_ion_diffusivity, not the ion's own get_diffusivity: the tabulated value is for water at
    # 25 degC, and only the solvent knows the viscosity that carries it to this temperature and this
    # liquid. A material that carries ions but predates that method still works.
    diff=getattr(fluid_props,"get_ion_diffusivity",None)
    # Only a liquid reaches this line: the get_ions probe above rejected everything else.
    bulk_of=getattr(fluid_props,"get_bulk_concentration")
    # None is how the material spells "leave the temperature symbolic", which is not the same thing
    # as being handed var("temperature") to substitute -- substituting a field into a viscosity
    # correlation is what fails there.
    Targ=None if temperature is None or is_zero(convert_to_expression(temperature-var("temperature"))) else temperature
    return [IonSpec(n,i.charge_number,
                    diff(n,Targ) if diff is not None else i.get_diffusivity(temperature if temperature is not None else var("temperature")),
                    bulk_concentration=bulk_of(n),
                    molar_mass=getattr(i,"molar_mass",None))
            for n,i in ions.items()]


def _resolve_scale(what:"ExpressionOrNum | str")->ExpressionOrNum:
    """A scale argument that may either be a named scale or an explicit expression.

    Same convention as :py:class:`~pyoomph.equations.generic.ProjectExpression`: a string is looked
    up with :py:func:`~pyoomph.expressions.generic.scale_factor`, anything else is used as it is.
    """
    return scale_factor(what) if isinstance(what,str) else what


class ElectricPotentialEquations(Equations):
    r"""
    .. _ElectricPotentialEquations:

    Gauss's law in the electrostatic potential formulation,

    .. math:: -\nabla\cdot(\varepsilon\nabla\phi) = \rho_\mathrm{e}\,,\qquad
              \vec{E}=-\nabla\phi\,,\qquad \vec{D}=\varepsilon\vec{E}\,,

    with the weak form

    .. math:: (\varepsilon\nabla\phi,\nabla v) - (\rho_\mathrm{e},v)
              + \langle \vec{n}\cdot\vec{D}, v\rangle = 0\,.

    The electric field is **not** an unknown -- pyoomph has no :math:`H(\mathrm{div})` or
    :math:`H(\mathrm{curl})` spaces, so a mixed formulation is not available. ``E`` is instead
    provided everywhere (bulk *and* interfaces) as ``var("electric_field")`` by symbolic
    substitution. Add :py:class:`ElectricFieldProjection` if a genuine, :math:`L^2`-projected finite
    element field is needed, e.g. for a smooth output or a spatial error estimator.

    .. note::
        **The flow coupling is on by default.** Adding these equations next to a (Navier-)Stokes on
        the same domain adds the Maxwell stress
        :math:`\varepsilon(\vec{E}\otimes\vec{E}-\tfrac{1}{2}|\vec{E}|^2\vec{\vec{I}})` to the
        momentum equation, the same way
        :py:class:`~pyoomph.equations.viscoelastic.ViscoelasticEquations` adds the polymer stress.
        Nothing happens if there is no flow on the domain, so a pure electrostatics problem is
        unaffected.

        Pass ``add_maxwell_stress_to_momentum=False`` to switch it off -- which is required if you
        instead use :py:class:`~pyoomph.equations.electrohydrodynamics.ElectricBodyForceEquations`
        or an explicit
        :py:class:`~pyoomph.equations.electrohydrodynamics.MaxwellStressEquations`, since those
        would then be counted on top of it. That is detected and refused rather than silently
        doubling the force.

    This class is deliberately **not** a subclass of
    :py:class:`~pyoomph.equations.poisson.PoissonEquation`, although the strong form coincides: the
    latter hardcodes ``testscale=1/scale_factor(name)``, whereas the whole cross-domain coupling of
    this module rests on a test scale built from a *shared* permittivity scale (see
    :py:attr:`permittivity_scale`).

    Args:
        permittivity: The absolute permittivity :math:`\varepsilon`. Mutually exclusive with
            ``relative_permittivity`` and ``fluid_props``.
        relative_permittivity: The relative permittivity :math:`\varepsilon_\mathrm{r}`, i.e.
            :math:`\varepsilon=\varepsilon_\mathrm{r}\varepsilon_0`.
        charge_density: The volumetric free charge density :math:`\rho_\mathrm{e}`. ``None`` (the
            default) means no charge at all and generates no term.
        conductivity: The Ohmic conductivity :math:`\sigma_\mathrm{c}`. It does **not** enter this
            equation, which stays Gauss's law; it only makes :py:meth:`get_conduction_current`
            available, which is what :py:class:`SurfaceChargeConservation` needs in the
            Gauss-driven leaky-dielectric formulation (see there). Use
            :py:class:`OhmicConductionEquations` if the conductivity should govern the bulk instead.
        name: Name of the potential field. Defaults to ``"phi"``.
        space: Finite element space of the potential. Defaults to ``"C2"``.
        fluid_props: Material properties to take ``relative_permittivity`` (and, for
            :py:class:`OhmicConductionEquations`, ``electric_conductivity``) from.
        temperature: Temperature at which to evaluate the material properties. ``None`` (the
            default) leaves them as functions of ``var("temperature")``, which then has to be a
            field on the domain; an isothermal problem should pass the temperature here. The
            permittivity of water varies by a third over its liquid range, so this is not a detail.
        permittivity_scale: The permittivity entering the *test function* scale. Either an
            expression or the name of a registered scale. It must be the **same** in all coupled
            domains -- that is what makes the potential rows of e.g. a gas (:math:`\varepsilon_0`)
            and an electrolyte (:math:`78\varepsilon_0`) commensurate, so that the Lagrange
            multiplier of :py:class:`ElectricPotentialConnection` transmits the correct flux.
            Defaults to :py:data:`~pyoomph.expressions.phys_consts.epsilon_0`.
        define_electric_field: Provide ``var("electric_field")`` by substitution. Defaults to True.
        electric_field_name: Name under which the substituted field is provided. Defaults to
            ``"electric_field"``.
        add_maxwell_stress_to_momentum: If (Navier-)Stokes equations are present on the same domain,
            add the Maxwell stress to their momentum equation, i.e. make the problem
            electrohydrodynamic without any further plumbing. Defaults to **True**, and does nothing
            at all when there is no flow. See the note below.
        electrostriction: :math:`\rho\,\partial\varepsilon/\partial\rho`, entering the Maxwell
            stress. ``None`` (the default) gives the incompressible/constant-permittivity form.
        output_fields: Add ``add_local_function`` entries for the field components and the charge
            density, so that they appear in the output. Defaults to True.
        consider_scaling: Apply the test scaling described above. Defaults to True.

    .. note::
        **Scalings to set on problem level.** Besides the potential field, the nondimensionalization
        uses the non-field scales ``charge_density`` for the charge term and ``electric_field`` for
        the substituted field of that name. Together with the shared ``permittivity``, all of them
        are registered consistently by :py:func:`set_electrostatic_scaling`, which is the
        recommended way to set them.
    """

    def __init__(self,*,permittivity:ExpressionNumOrNone=None,relative_permittivity:ExpressionNumOrNone=None,
                 charge_density:ExpressionNumOrNone=None,conductivity:ExpressionNumOrNone=None,
                 name:str="phi",space:"FiniteElementSpaceEnum"="C2",
                 fluid_props:"AnyMaterialProperties | None"=None,
                 temperature:ExpressionNumOrNone=None,
                 permittivity_scale:"ExpressionOrNum | str"=epsilon_0,
                 define_electric_field:bool=True,electric_field_name:str="electric_field",
                 add_maxwell_stress_to_momentum:bool=True,electrostriction:ExpressionNumOrNone=None,
                 output_fields:bool=True,consider_scaling:bool=True):
        super().__init__()
        self.name=name
        self.space:"FiniteElementSpaceEnum"=space
        self.fluid_props=fluid_props
        self.temperature=temperature
        self.permittivity=self._resolve_permittivity(permittivity,relative_permittivity,fluid_props,
                                                     temperature)
        self.charge_density=charge_density
        if conductivity is None and fluid_props is not None:
            conductivity=getattr(fluid_props,"electric_conductivity",None)
        self.conductivity=conductivity
        self.permittivity_scale:"ExpressionOrNum | str"=permittivity_scale
        self.define_electric_field=define_electric_field
        self.electric_field_name=electric_field_name
        self.add_maxwell_stress_to_momentum=add_maxwell_stress_to_momentum
        self.electrostriction=electrostriction
        self.output_fields=output_fields
        self.consider_scaling=consider_scaling

    def _T(self)->ExpressionOrNum:
        """The temperature to evaluate material properties at: the given one, or the field."""
        return self.temperature if self.temperature is not None else var("temperature")

    def _resolve_permittivity(self,permittivity:ExpressionNumOrNone,relative_permittivity:ExpressionNumOrNone,
                              fluid_props:"AnyMaterialProperties | None",
                              temperature:ExpressionNumOrNone=None)->ExpressionOrNum:
        given=[x for x in (permittivity,relative_permittivity) if x is not None]
        if len(given)>1:
            raise ValueError("Please pass either permittivity or relative_permittivity, not both")
        if permittivity is not None:
            return permittivity
        if relative_permittivity is not None:
            return relative_permittivity*epsilon_0
        if fluid_props is not None:
            getter=getattr(fluid_props,"get_absolute_permittivity",None)
            if getter is not None:
                try:
                    # With a temperature, the material's expression is evaluated at it. Without one
                    # it stays a function of var("temperature"), which then has to exist as a field
                    # on the domain -- an isothermal problem should just pass the temperature.
                    return getter(temperature)
                except RuntimeError:
                    pass  # falls through to the readable message below
            eps_r=getattr(fluid_props,"relative_permittivity",None)
            if eps_r is None:
                raise ValueError("The material '"+str(getattr(fluid_props,"name","<unnamed>"))+
                                 "' does not define a relative_permittivity. Set it on the material or "+
                                 "pass permittivity/relative_permittivity explicitly.")
            return eps_r*epsilon_0
        raise ValueError("Please pass one of permittivity, relative_permittivity or fluid_props to "+
                         type(self).__name__)

    def get_information_string(self)->str:
        return "-div(<eps>*grad(<"+self.name+">))=<rho_e> with <eps>="+str(self.permittivity)+ \
               " and <rho_e>="+str(self.charge_density)

    # ---- the quantities other equations ask for -------------------------------------------------
    # Every one of these takes a domain, and passing it is not cosmetic: on an interface grad() is
    # the SURFACE gradient, so anything built from grad(phi) without domain=".." silently loses all
    # normal derivatives. Same trap that StokesEquations.get_stabilization_traction documents.

    def get_permittivity(self,domain:"str | FiniteElementCodeGenerator | None"=None)->ExpressionOrNum:
        """The absolute permittivity, optionally evaluated in another domain."""
        if domain is None:
            return self.permittivity
        return evaluate_in_domain(convert_to_expression(self.permittivity),domain)

    def get_potential(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        """The potential field. Pass ``domain=".."`` to reach the bulk from an interface."""
        return var(self.name,domain=domain)

    def get_electric_field(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        r""":math:`\vec{E}=-\nabla\phi`. **Pass** ``domain=".."`` **when calling this from an
        interface**, otherwise ``grad`` is the surface gradient and only the tangential field comes
        out (which is occasionally what is wanted, e.g. for electroosmotic slip -- but then say so
        explicitly)."""
        return -grad(var(self.name,domain=domain))

    def get_displacement_field(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        r""":math:`\vec{D}=\varepsilon\vec{E}`."""
        return self.get_permittivity(domain)*self.get_electric_field(domain)

    def get_charge_density(self,domain:"str | FiniteElementCodeGenerator | None"=None)->ExpressionNumOrNone:
        r"""The volumetric free charge density :math:`\rho_\mathrm{e}`, or ``None`` if there is none."""
        if self.charge_density is None:
            return None
        if domain is None:
            return self.charge_density
        return evaluate_in_domain(convert_to_expression(self.charge_density),domain)

    def get_conductivity(self,domain:"str | FiniteElementCodeGenerator | None"=None)->ExpressionNumOrNone:
        """The Ohmic conductivity, or ``None`` for a perfect dielectric."""
        if self.conductivity is None:
            return None
        if domain is None:
            return self.conductivity
        return evaluate_in_domain(convert_to_expression(self.conductivity),domain)

    def get_maxwell_stress(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        r"""
        The Maxwell stress tensor
        :math:`\varepsilon(\vec{E}\otimes\vec{E})-\tfrac{1}{2}(\varepsilon-\rho\partial_\rho\varepsilon)|\vec{E}|^2\vec{\vec{I}}`.

        **Pass** ``domain=".."`` **when calling this from an interface**, or ``grad`` is the surface
        gradient and the normal part of :math:`\vec{E}` -- which is most of the traction -- is
        silently lost.

        The total stress of the fluid is this plus the viscous and pressure parts, so a drag
        integral over a wall reads

        .. code-block:: python

            sigma = -p*identity_matrix() + 2*mu*sym(grad(u)) + eqs.get_maxwell_stress(domain="..")
            traction = matproduct(sigma, var("normal"))
        """
        from .electrohydrodynamics import maxwell_stress_tensor
        return maxwell_stress_tensor(self.get_permittivity(domain),self.get_electric_field(domain),
                                     electrostriction=self.electrostriction)

    def get_conduction_current(self,domain:"str | FiniteElementCodeGenerator | None"=None)->ExpressionNumOrNone:
        r"""The Ohmic conduction current :math:`\vec{J}=\sigma_\mathrm{c}\vec{E}`, or ``None`` if no
        conductivity was given. Pass ``domain=".."``/``"|.."`` when calling this from an interface."""
        sig=self.get_conductivity(domain)
        if sig is None:
            return None
        return sig*self.get_electric_field(domain)

    # ---- the equation itself ---------------------------------------------------------------------

    def _testscale(self)->ExpressionOrNum:
        if not self.consider_scaling:
            return 1
        # eps*Phi/X^2 is the magnitude of each term of the strong equation, and the weak form adds a
        # nondimensional dx, so this is what makes the assembled residual O(1). Built from the SHARED
        # permittivity scale, never from self.permittivity -- see the class docstring.
        return scale_factor("spatial")**2/(_resolve_scale(self.permittivity_scale)*scale_factor(self.name))

    def define_fields(self):
        self.define_scalar_field(self.name,self.space,testscale=self._testscale())
        if self.define_electric_field:
            # Substituted fields are expanded as get_scaling(name)*<substitution>, exactly like a real
            # field is scale*nondim, so the substitution has to be the NONDIMENSIONAL value. Dividing
            # by the very scale_factor the expansion multiplies back in makes the two cancel
            # identically, so var("electric_field") is the correct dimensional field whether or not
            # anybody ever registered an "electric_field" scale (an unset scale is simply 1).
            # Passing domain=<bulk> is what keeps grad() the full gradient when this is read from an
            # interface, where a bare grad() would be the surface gradient.
            mydom=self.get_my_domain()
            self.define_field_by_substitution(self.electric_field_name,
                                              -grad(var(self.name,domain=mydom))/scale_factor(self.electric_field_name),
                                              also_on_interface=True)

    def define_scaling(self):
        if not self.consider_scaling:
            return
        # Reported into _numerical_factors.txt. For a 1:1 electrolyte at Phi=RT/F this is exactly
        # 2*(X/lambda_D)**2, i.e. the squared Debye ratio -- the number that says whether the mesh
        # can possibly resolve the double layer.
        rho=self.get_charge_density()
        if rho is not None:
            self._add_named_numerical_factor(charge_density_in_poisson_eq=
                                            scale_factor("charge_density")*test_scale_factor(self.name))

    def _find_flow(self)->"StokesEquations | None":
        """The co-located (Navier-)Stokes equations, or None if this domain has no flow."""
        from .navier_stokes import StokesEquations
        eqs=self.get_combined_equations().get_equation_of_type(StokesEquations)
        if isinstance(eqs,list):
            eqs=eqs[0] if len(eqs)==1 else None
        return eqs if isinstance(eqs,StokesEquations) else None

    def _apply_maxwell_stress_to_momentum(self):
        """Write the Maxwell stress into a co-located momentum row, the ViscoelasticEquations way.

        The (Navier-)Stokes equations assemble their momentum residual as weak(stress, grad(v)) with
        the stress containing -p*identity, so the Maxwell stress is simply added with the same sign.
        """
        if not self.add_maxwell_stress_to_momentum:
            return
        flow=self._find_flow()
        if flow is None:
            return  # no flow on this domain: a pure electrostatics problem, nothing to couple to
        from .electrohydrodynamics import MaxwellStressEquations, ElectricBodyForceEquations
        for cls in (MaxwellStressEquations,ElectricBodyForceEquations):
            other=self.get_combined_equations().get_equation_of_type(cls)
            if isinstance(other,list):
                other=other[0] if other else None
            if other is not None:
                raise RuntimeError(
                    "This domain has both "+type(self).__name__+" with "
                    "add_maxwell_stress_to_momentum=True (the default) and a "+cls.__name__+", "
                    "so the electric force would enter the momentum equation twice. Pass "
                    "add_maxwell_stress_to_momentum=False to "+type(self).__name__+" if you want "
                    "to keep the explicit "+cls.__name__+".")
        if flow.extra_stress is not None:
            raise RuntimeError(
                "The (Navier-)Stokes equations on this domain already carry an extra_stress, and "
                +type(self).__name__+" would add the Maxwell stress on top of it. Pass "
                "add_maxwell_stress_to_momentum=False if that extra_stress is already the Maxwell one.")
        self.add_residual(weak(self.get_maxwell_stress(),grad(testfunction(flow.velocity_name))))

    def define_residuals(self):
        phi,phi_test=var_and_test(self.name)
        eps=self.get_permittivity()
        self.add_residual(weak(eps*grad(phi),grad(phi_test)))
        rho=self.get_charge_density()
        if rho is not None and not is_zero(rho):
            self.add_residual(-weak(rho,phi_test))
        self._apply_maxwell_stress_to_momentum()

    def define_additional_functions(self):
        if not self.output_fields:
            return
        # E is not an unknown, so without this it would not appear in the output at all. These are
        # local expressions (nodal evaluations), i.e. free -- unlike ElectricFieldProjection, which
        # adds real degrees of freedom and an L2 solve, and is only worth it when a *continuous* E
        # is genuinely needed (a smooth plot, an error estimator, a post-processing integral).
        #
        # Skipped when such a projection is in use: its vector field outputs under exactly these
        # component names, so adding them here as well would duplicate the output columns.
        mydom=self.get_my_domain()
        stem=self.electric_field_name
        comps=[stem+"_"+c for c in ("x","y","z")][:self.get_nodal_dimension()]
        if all(mydom.get_space_of_field(c)=="" for c in comps):
            E=self.get_electric_field()
            # Registered as one vector, not component by component: only then does the field land in
            # _vectorfields and get written as a single vector array (rather than loose scalars) to
            # the vtu. Truncated to the nodal dimension, so that an out-of-plane component -- an
            # azimuthal one, say -- is not silently mislabelled as "_z".
            self.add_local_function(stem,vector([E[i] for i in range(len(comps))]))
        rho=self.get_charge_density()
        if rho is not None and not is_zero(rho):
            self.add_local_function("charge_density",rho)


class IonSpec:
    """
    One ionic species: what the electrolyte models need to know about it.

    Not a material -- see :py:class:`~pyoomph.materials.generic.IonProperties` for that. This is the
    lightweight description that :py:class:`PoissonBoltzmannEquations` and the Nernst-Planck
    equations accept directly, so that a quick test does not have to declare a material first.

    Args:
        name: Name of the species, used to build the concentration field name.
        valence: The charge number :math:`z_i`, e.g. ``+1`` for Na+ and ``-2`` for SO4(2-).
        diffusivity: The diffusivity :math:`D_i`.
        mobility: The molar mobility :math:`m_i`, i.e. :math:`\\vec{v}_i=-m_i z_i F\\nabla\\phi`.
            Defaults to the Einstein relation :math:`m_i=D_i/(RT)`, which is what you want unless
            you have measured otherwise.
        bulk_concentration: The reservoir concentration :math:`c_i^\\infty`, used by the
            Poisson-Boltzmann models and as the default initial condition.
        molar_mass: The molar mass, only needed if the ions also carry mass in a flow model.
    """
    def __init__(self,name:str,valence:int,diffusivity:ExpressionNumOrNone=None,*,
                 mobility:ExpressionNumOrNone=None,bulk_concentration:ExpressionNumOrNone=None,
                 molar_mass:ExpressionNumOrNone=None):
        self.name=name
        self.valence=valence
        self.diffusivity=diffusivity
        self.mobility=mobility
        self.bulk_concentration=bulk_concentration
        self.molar_mass=molar_mass

    def get_mobility(self,temperature:ExpressionOrNum)->ExpressionOrNum:
        """The molar mobility, from the Einstein relation if it was not given explicitly."""
        if self.mobility is not None:
            return self.mobility
        if self.diffusivity is None:
            raise ValueError("Ion '"+self.name+"' has neither a mobility nor a diffusivity")
        return self.diffusivity/(gas_constant*temperature)

    def __repr__(self)->str:
        return "IonSpec("+self.name+", z="+str(self.valence)+")"


def symmetric_electrolyte(bulk_concentration:ExpressionOrNum,valence:int=1,*,
                          cation_name:str="cation",anion_name:str="anion",
                          cation_diffusivity:ExpressionNumOrNone=None,
                          anion_diffusivity:ExpressionNumOrNone=None)->list[IonSpec]:
    r"""
    A symmetric :math:`z\!:\!z` electrolyte, i.e. the pair of ions that a "1 mM KCl" means.

    Args:
        bulk_concentration: The reservoir concentration of *each* ion.
        valence: The charge number :math:`z` of the cation; the anion gets :math:`-z`.
        cation_name / anion_name: Field name stems.
        cation_diffusivity / anion_diffusivity: Diffusivities, only needed for Nernst-Planck.
    """
    return [IonSpec(cation_name,+valence,cation_diffusivity,bulk_concentration=bulk_concentration),
            IonSpec(anion_name,-valence,anion_diffusivity,bulk_concentration=bulk_concentration)]


class PoissonBoltzmannEquations(ElectricPotentialEquations):
    r"""
    The Poisson-Boltzmann equation, i.e. Gauss's law closed with the *equilibrium* ion distribution
    :math:`c_i=c_i^\infty\exp(-z_iF\phi/(RT))` instead of a transport equation:

    .. math:: -\nabla\cdot(\varepsilon\nabla\phi)
              = \sum_i z_i F c_i^\infty \exp\!\left(-\frac{z_i F\phi}{RT}\right)\,.

    For a symmetric :math:`z\!:\!z` electrolyte the right-hand side collapses to
    :math:`-2zFc^\infty\sinh(zF\phi/(RT))`, which is what this class emits in that case -- one
    expression node instead of two exponentials, and exactly antisymmetric.

    With ``linearized=True`` the exponential is expanded to first order, which (using the bulk
    electroneutrality :math:`\sum_i z_i c_i^\infty=0`) gives the Debye-Hueckel equation
    :math:`\nabla^2\phi=\phi/\lambda_\mathrm{D}^2`. See :py:class:`DebyeHuckelEquations`, which is
    the more convenient entry point for that and does not need an ion table at all.

    This model assumes the ions are in equilibrium with a reservoir, i.e. **no flow and no imposed
    current**. If either matters, use the Nernst-Planck equations; at equilibrium they reproduce
    this solution, which is the sharpest available cross-check on both.

    Args:
        ions: The ionic species, as :py:class:`IonSpec` objects or a liquid material with ions
            dissolved in it. Alternatively give ``bulk_concentration`` (and ``valence``) for the
            symmetric shortcut, or nothing at all if ``fluid_props`` already carries ions.
        bulk_concentration: Reservoir concentration for the symmetric :math:`z\!:\!z` shortcut.
        valence: Charge number for the symmetric shortcut. Defaults to 1.
        temperature: The temperature. Defaults to the field ``var("temperature")``.
        linearized: Linearize the Boltzmann factor, giving Debye-Hueckel. Defaults to False.
        reference_potential: The potential the reservoir sits at. Defaults to 0.
        exponent_limit: If given, the Boltzmann exponent is clamped to this magnitude. Newton
            *transiently* visits potentials the solution never reaches, and ``exp`` overflows above
            about 700; clamping is a line-search-like safeguard that changes the equation off the
            solution but not on it. Defaults to ``None``, i.e. no clamping -- prefer continuation in
            the applied potential, or starting from the linearized solution.
    """

    def __init__(self,*,ions:"Sequence[IonSpec] | AnyMaterialProperties | None"=None,bulk_concentration:ExpressionNumOrNone=None,
                 valence:int=1,temperature:ExpressionNumOrNone=None,linearized:bool=False,
                 reference_potential:ExpressionOrNum=0,exponent_limit:float | None=None,**kwargs:Any):
        super().__init__(temperature=temperature,**kwargs)
        if ions is None:
            fp=kwargs.get("fluid_props")
            if bulk_concentration is not None:
                ions=symmetric_electrolyte(bulk_concentration,valence)
            elif fp is not None and getattr(fp,"get_ions",lambda:{})():
                ions=ions_from_material(fp,temperature=temperature)
            else:
                raise ValueError("Please pass either ions=[IonSpec(...), ...], a bulk_concentration "+
                                 "for a symmetric z:z electrolyte, or a fluid_props with ions "+
                                 "dissolved in it")
        elif not isinstance(ions,(list,tuple)):
            ions=ions_from_material(cast("AnyMaterialProperties",ions),temperature=temperature)
        self.ions=list(ions)
        for ion in self.ions:
            if ion.bulk_concentration is None:
                raise ValueError("Ion '"+ion.name+"' has no bulk_concentration, which the "+
                                 "Poisson-Boltzmann closure requires")
        self.linearized=linearized
        self.reference_potential=reference_potential
        self.exponent_limit=exponent_limit

    def get_information_string(self)->str:
        kind="linearized " if self.linearized else ""
        return kind+"Poisson-Boltzmann with <eps>="+str(self.permittivity)+" and ions "+ \
               ", ".join(str(i) for i in self.ions)

    def thermal_voltage(self)->ExpressionOrNum:
        r"""The thermal voltage :math:`RT/F`, i.e. the potential scale of this model."""
        return gas_constant*self._T()/faraday_constant

    def _bulk(self,ion:"IonSpec")->ExpressionOrNum:
        """The bulk concentration of an ion, which __init__ has already refused to do without."""
        assert ion.bulk_concentration is not None
        return ion.bulk_concentration

    def ionic_strength(self)->ExpressionOrNum:
        r"""The reservoir molar ionic strength :math:`I=\frac{1}{2}\sum_i z_i^2 c_i^\infty`."""
        return sum(ion.valence**2*self._bulk(ion) for ion in self.ions)/2

    def debye_length(self)->ExpressionOrNum:
        r"""The screening length :math:`\lambda_\mathrm{D}=\sqrt{\varepsilon RT/(2F^2I)}`."""
        return debye_length(self.permittivity,self.ionic_strength(),self._T())

    def _is_symmetric(self)->bool:
        if len(self.ions)!=2:
            return False
        a,b=self.ions
        return a.valence==-b.valence and a.valence>0 and \
               is_zero(convert_to_expression(self._bulk(a)-self._bulk(b)))

    def get_charge_density(self,domain:"str | FiniteElementCodeGenerator | None"=None)->ExpressionNumOrNone:
        phi=var(self.name,domain=domain)-self.reference_potential
        VT=self.thermal_voltage()
        if self.linearized:
            # Expanding exp(-z F phi/RT) to first order and using the bulk electroneutrality
            # sum_i z_i c_i^inf = 0, which kills the constant term, leaves -eps*phi/lambda_D**2.
            rho:ExpressionOrNum=-sum(faraday_constant*ion.valence**2*self._bulk(ion)
                                     for ion in self.ions)/VT*phi
        else:
            arg=-phi/VT
            if self.exponent_limit is not None:
                lim=self.exponent_limit
                arg=maximum(minimum(arg,lim),-lim)
            if self._is_symmetric():
                z,cinf=self.ions[0].valence,self._bulk(self.ions[0])
                rho=-2*z*faraday_constant*cinf*sinh(z*(-arg))
            else:
                rho=sum(faraday_constant*ion.valence*self._bulk(ion)*exp(ion.valence*arg)
                        for ion in self.ions)
        extra=super().get_charge_density(domain)
        return rho if extra is None else rho+extra


class DebyeHuckelEquations(PoissonBoltzmannEquations):
    r"""
    The linearized Poisson-Boltzmann (Debye-Hueckel) equation,

    .. math:: \nabla^2\phi = \frac{\phi-\phi_\infty}{\lambda_\mathrm{D}^2}\,,

    valid while the potential stays below the thermal voltage, :math:`|zF\zeta/(RT)|\lesssim 1`.

    Unlike :py:class:`PoissonBoltzmannEquations` this can be used **without declaring any ions at
    all** -- just give the Debye length, which is usually the number that is actually known. It is
    also the natural starting guess for a nonlinear Poisson-Boltzmann or Nernst-Planck solve.

    Args:
        debye_length: The screening length :math:`\lambda_\mathrm{D}` directly. Mutually exclusive
            with ``ions``/``bulk_concentration``.
        reference_potential: The reservoir potential :math:`\phi_\infty`. Defaults to 0.
    """
    def __init__(self,*,debye_length:ExpressionNumOrNone=None,**kwargs:Any):
        self._explicit_debye_length=debye_length
        if debye_length is not None:
            if kwargs.get("ions") is not None or kwargs.get("bulk_concentration") is not None:
                raise ValueError("Pass either debye_length or ions/bulk_concentration, not both")
            # A placeholder ion table: it is never used, since get_charge_density and debye_length
            # are both overridden below, but it keeps the base class constructor happy.
            kwargs["bulk_concentration"]=0*mol/meter**3
        kwargs["linearized"]=True
        super().__init__(**kwargs)

    def get_information_string(self)->str:
        return "Debye-Hueckel with <eps>="+str(self.permittivity)+" and lambda_D="+str(self.debye_length())

    def debye_length(self)->ExpressionOrNum:
        if self._explicit_debye_length is not None:
            return self._explicit_debye_length
        return super().debye_length()

    def get_charge_density(self,domain:"str | FiniteElementCodeGenerator | None"=None)->ExpressionNumOrNone:
        if self._explicit_debye_length is None:
            return super().get_charge_density(domain)
        phi=var(self.name,domain=domain)-self.reference_potential
        rho:ExpressionOrNum=-self.permittivity*phi/self._explicit_debye_length**2
        extra=ElectricPotentialEquations.get_charge_density(self,domain)
        return rho if extra is None else rho+extra


class ElectrodeBC(DirichletBC):
    r"""
    A perfect conductor held at a prescribed potential, :math:`\phi=V`.

    A named alias of :py:class:`~pyoomph.equations.generic.DirichletBC` -- it exists because
    ``ElectrodeBC(5*volt)`` says what it means, whereas ``DirichletBC(phi=5*volt)`` requires the
    reader to know the field name.

    Args:
        voltage: The imposed potential.
        potential_name: Name of the potential field. Defaults to ``"phi"``.
    """
    def __init__(self,voltage:ExpressionOrNum,*,potential_name:str="phi"):
        # dict[str,Any]: the keyword is a variable field name, not one of the named parameters
        super().__init__(**cast("dict[str,Any]",{potential_name:voltage}))


class SurfaceChargeBC(InterfaceEquations):
    r"""
    A free surface charge :math:`\sigma_\mathrm{s}` on a boundary whose exterior is field-free
    (a perfect insulator, or a grounded conductor), i.e.

    .. math:: \vec{n}\cdot\vec{D} = -\sigma_\mathrm{s}

    with :math:`\vec{n}` the **outward** normal of the domain.

    Sign, since it is easy to get wrong and everything else in this module keys off it: the bulk
    residual is :math:`+(\varepsilon\nabla\phi,\nabla v)`, whose integration by parts leaves
    :math:`+\langle\vec{n}\cdot\vec{D},v\rangle`; a pillbox with no field on the far side gives
    :math:`\vec{n}\cdot\vec{D}=-\sigma_\mathrm{s}`; so the term to add is
    :math:`-\langle\sigma_\mathrm{s},v\rangle`, **the same sign as the volumetric**
    :math:`-(\rho_\mathrm{e},v)`. In 1D with a wall at :math:`x=0` this gives
    :math:`\partial_x\phi|_0=-\sigma_\mathrm{s}/\varepsilon`, so a positively charged wall has a
    positive zeta potential, and matching against the Debye-Hueckel profile
    :math:`\phi=\zeta e^{-x/\lambda_\mathrm{D}}` reproduces
    :math:`\sigma_\mathrm{s}=\varepsilon\zeta/\lambda_\mathrm{D}`.

    Args:
        surface_charge_density: The imposed free surface charge density.
        interface_props: Interface properties to take ``surface_charge_density`` from instead.
    """
    required_parent_type = ElectricPotentialEquations

    def __init__(self,surface_charge_density:ExpressionNumOrNone=None,*,
                 interface_props:"BaseInterfaceProperties | None"=None):
        super().__init__()
        if surface_charge_density is None:
            if interface_props is None:
                raise ValueError("SurfaceChargeBC needs either a surface_charge_density or an "+
                                 "interface_props carrying one")
            surface_charge_density=interface_props.surface_charge_density
        self.surface_charge_density=surface_charge_density
        self.interface_props=interface_props

    def define_residuals(self):
        parent=self.get_parent_equations(ElectricPotentialEquations)
        assert isinstance(parent,ElectricPotentialEquations)
        _,phi_test=var_and_test(parent.name)
        self.add_residual(-weak(self.surface_charge_density,phi_test))


class ElectricFarFieldCondition(InterfaceEquations):
    r"""
    A far-field condition for an unbounded electrostatic problem,

    .. math:: \phi + R\,\frac{\partial\phi}{\partial r} = \phi_\infty\,,

    i.e. the monopole decay of the potential at an artificial outer boundary. Only valid when the
    charge density vanishes sufficiently fast towards the far field. In 2D the solution decays
    logarithmically and a ``farfield_length`` must be given; see
    :py:class:`~pyoomph.equations.poisson.PoissonFarFieldMonopoleCondition`, with which this class
    shares its implementation.

    Args:
        far_value: The potential at infinity. Defaults to 0.
        origin: The origin the monopole is measured from. Defaults to the coordinate origin.
        farfield_length: The far-field length scale :math:`L` required in 1D and 2D.
    """
    required_parent_type = ElectricPotentialEquations

    def __init__(self,far_value:ExpressionOrNum=0,*,origin:ExpressionOrNum=vector([0]),
                 farfield_length:ExpressionNumOrNone=None):
        super().__init__()
        self.far_value=far_value
        self.origin=origin
        self.farfield_length=farfield_length

    def define_residuals(self):
        parent=self.get_parent_equations(ElectricPotentialEquations)
        assert isinstance(parent,ElectricPotentialEquations)
        real_dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
        self.add_residual(farfield_monopole_residual(parent.get_permittivity(),parent.name,self.far_value,
                                                     self.origin,self.farfield_length,real_dim,
                                                     self.get_normal()))


class ElectricFieldProjection(ProjectExpression):
    r"""
    An :math:`L^2` projection of :math:`\vec{E}=-\nabla\phi` onto a real vector finite element
    field, for a smooth output, a :py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` or
    any post-processing that needs a continuous field rather than the elementwise gradient.

    Since it defines a *field* of that name, the co-located
    :py:class:`ElectricPotentialEquations` must not also provide the same name by substitution --
    construct it with ``define_electric_field=False``, or rename one of the two.

    Args:
        name: Name of the projected field. Defaults to ``"electric_field"``.
        space: Space to project onto. Defaults to ``"C1"``, since ``E`` is one order below ``phi``.
        potential_name: Name of the potential field. Defaults to ``"phi"``.
        scale: Scale of the projected field, either a named scale or an expression.
    """
    def __init__(self,*,name:str="electric_field",space:"FiniteElementSpaceEnum"="C1",
                 potential_name:str="phi",scale:"ExpressionOrNum | str"="electric_field"):
        super().__init__(scale=scale,space=space,field_type="vector",
                         **{name:-grad(var(potential_name))})
        self.name=name
        self.potential_name=potential_name

    def define_fields(self):
        pot=self.get_combined_equations().get_equation_of_type(ElectricPotentialEquations)
        if isinstance(pot,ElectricPotentialEquations) and pot.define_electric_field and pot.electric_field_name==self.name:
            raise RuntimeError("ElectricFieldProjection defines the field '"+self.name+"', but the "+
                               "co-located "+type(pot).__name__+" already provides it by substitution. "+
                               "Pass define_electric_field=False there, or rename one of the two.")
        super().define_fields()


class OhmicConductionEquations(ElectricPotentialEquations):
    r"""
    The Taylor-Melcher leaky-dielectric bulk: a weakly conducting fluid in which charge relaxes so
    fast that the interior is electroneutral and **all** free charge sits on the interfaces,

    .. math:: -\nabla\cdot(\sigma_\mathrm{c}\nabla\phi)=0\,,

    i.e. conservation of the Ohmic current :math:`\vec{J}=\sigma_\mathrm{c}\vec{E}` rather than
    Gauss's law. The permittivity is still needed -- and still required as an argument -- because the
    Maxwell stress, which is what actually drives the flow, depends on it and not on the conductivity.

    Because this operator conserves the **current**, the natural boundary condition it leaves on an
    interface is a jump in the current, not in the displacement field. So
    :py:class:`ElectricPotentialConnection` here means *current* continuity, and passing a
    ``surface_charge_density`` to it would impose a spurious current source. For a dynamic surface
    charge, prefer the Gauss-driven pairing described in :py:class:`SurfaceChargeConservation`; use
    this class when the steady current distribution is the thing being solved for.

    Other arguments are as in :py:class:`ElectricPotentialEquations`.

    Args:
        conductivity: The Ohmic conductivity :math:`\sigma_\mathrm{c}`.
        permittivity / relative_permittivity / fluid_props: The permittivity, needed for the stress.
        conductivity_scale: The conductivity entering the test function scale. Like
            ``permittivity_scale`` it must be the **same** in all coupled domains. Defaults to
            ``"electric_conductivity"``, i.e. a named problem-level scale.
        charge_relaxation: Not implemented, see below.

    .. note::
        ``charge_relaxation=True`` raises. The bulk charge relaxation time
        :math:`\tau_\mathrm{e}=\varepsilon/\sigma_\mathrm{c}` is nanoseconds to microseconds against
        millisecond-to-second flow times, so a transient fully coupled system is stiff by six to nine
        orders of magnitude. The physically meaningful transient is the *interfacial* one, which
        :py:class:`SurfaceChargeConservation` does keep.
    """

    def __init__(self,*,conductivity:ExpressionNumOrNone=None,
                 conductivity_scale:"ExpressionOrNum | str"="electric_conductivity",
                 charge_relaxation:bool=False,**kwargs:Any):
        super().__init__(**kwargs)
        if conductivity is not None:
            self.conductivity=conductivity
        if self.conductivity is None:
            raise ValueError("OhmicConductionEquations needs a conductivity")
        self.conductivity_scale:"ExpressionOrNum | str"=conductivity_scale
        if charge_relaxation:
            raise NotImplementedError(
                "Bulk charge relaxation is not implemented: eps/sigma_c is ns-to-us against ms-to-s "
                "flow times, so the coupled transient is stiff by 6-9 orders of magnitude and would "
                "not converge in any useful number of steps. The interfacial charge transient, which "
                "relaxes on the (much slower) interfacial RC time, is in SurfaceChargeConservation.")

    def get_information_string(self)->str:
        return "-div(<sigma_c>*grad(<"+self.name+">))=0 with <sigma_c>="+str(self.conductivity)

    def get_charge_density(self,domain:"str | FiniteElementCodeGenerator | None"=None)->ExpressionNumOrNone:
        r"""
        :math:`\rho_\mathrm{e}=-\nabla\cdot(\varepsilon\nabla\phi)`, which the current-conservation
        equation turns into
        :math:`\varepsilon\nabla\sigma_\mathrm{c}\!\cdot\!\nabla\phi/\sigma_\mathrm{c}
        -\nabla\varepsilon\cdot\nabla\phi`. It is **zero** wherever both material properties are
        uniform, which is the whole point of the leaky-dielectric model: the charge is on the
        interfaces, not in the bulk.
        """
        phi=var(self.name,domain=domain)
        eps,sig=self.get_permittivity(domain),cast(ExpressionOrNum,self.get_conductivity(domain))
        rho=eps*dot(grad(sig),grad(phi))/sig-dot(grad(eps),grad(phi))
        extra=ElectricPotentialEquations.get_charge_density(self,domain)
        return rho if extra is None else rho+extra

    def _testscale(self)->ExpressionOrNum:
        if not self.consider_scaling:
            return 1
        return scale_factor("spatial")**2/(_resolve_scale(self.conductivity_scale)*scale_factor(self.name))

    def define_scaling(self):
        if not self.consider_scaling:
            return
        # The charge relaxation time over the flow time: the number that says whether treating the
        # bulk as quasi-static is defensible at all.
        self._add_named_numerical_factor(electric_reynolds_number=
            _resolve_scale(self.permittivity_scale)/_resolve_scale(self.conductivity_scale)
            /scale_factor("temporal"))

    def define_residuals(self):
        phi,phi_test=var_and_test(self.name)
        self.add_residual(weak(cast(ExpressionOrNum,self.get_conductivity())*grad(phi),grad(phi_test)))
        self._apply_maxwell_stress_to_momentum()


class SurfaceChargeConservation(InterfaceEquations):
    r"""
    The dynamic free surface charge of the Taylor-Melcher leaky-dielectric model,

    .. math:: \partial_t q + \nabla_\mathrm{s}\cdot(q\vec{u}_\mathrm{s})
              - \nabla_\mathrm{s}\cdot(K_\mathrm{s}\nabla_\mathrm{s}q)
              + \nabla_\mathrm{s}\cdot(\kappa_\mathrm{s}\vec{E}_\mathrm{t})
              = \vec{n}\cdot(\vec{J}_\mathrm{in}-\vec{J}_\mathrm{out}) + \dot{q}_\mathrm{ads}\,,

    i.e. the surface charge accumulates whatever Ohmic current the two bulks deliver, is carried
    along the interface by the flow and by surface conduction, and gains whatever ad- or desorbs.
    :math:`\vec{n}` is the outward normal of the parent domain, so it points from "in" to "out" and
    the current term is the net current arriving at the interface.

    **On a moving mesh this is assembled conservatively** (``form="conservative"``, the default), as
    the derivative of the whole surface integral minus the flux relative to the mesh::

        time_derivative_of_integral(weak(q, q_test)) - weak(q*(u - w), grad(q_test))

    Testing that with :math:`v=1` telescopes exactly -- the second term dies, and the first is a
    finite difference of the total charge -- so **the total charge on a closed interface is conserved
    to the Newton tolerance rather than to the order of the time stepping**. That is what an
    evaporating drop needs: evaporation removes solvent and not charge, so as the area shrinks
    :math:`q` must rise and :math:`\int q\,\mathrm{d}A` must not move. ``form="legacy"`` restores the
    older non-conservative assembly, which drifts at :math:`O(\Delta t^p)` under a normal slip (=
    mass transfer) *and* under a purely tangential mesh slip, where the interface does not even
    change shape. The measurements are in ``dev_docs/surfactant_transport.md``, whose surfactant
    equation has exactly the same structure; they are not repeated here.

    Two things the conservative form deliberately does *not* do, both measured there: it does not
    project :math:`\vec{u}-\vec{w}` onto the tangent plane (``grad`` on an interface *is* the surface
    gradient, so ``grad(q_test)`` is already orthogonal to the element normal and a normal component
    cannot contribute), and it does not smooth the normal (under mass transfer that makes the error
    ten times worse). It is also why the conservative form takes :math:`\vec{u}` and :math:`\vec{w}`
    separately rather than one combined :math:`\vec{u}_\mathrm{s}`, and why passing the old combined
    velocity to ``advection_velocity`` is nevertheless exact: the two differ only by a normal
    component, which the surface gradient annihilates.

    **A consequence at the ends of an open interface.** The conservative form integrates the
    advection by parts, so its natural condition at a contact line, corner or edge is now *zero total
    flux* -- nothing leaves. That is what makes the conservation exact; use
    :py:class:`SurfaceChargeEndFlux` where a nonzero end flux is wanted. The legacy form has no such
    term and does let charge advect out at an end.

    **Which bulk equation to pair this with -- getting it wrong is silent.** The leaky-dielectric
    model can be closed in two equivalent ways, and they need *different* bulk operators, because a
    bulk operator's natural boundary condition is a jump in whatever flux it conserves:

    *Gauss-driven (the usual choice for EHD drops, and the one to reach for first).* The bulk solves
    Gauss's law, so its natural boundary condition is the jump in the **displacement** field, which
    is exactly the surface charge. The charge is the dynamic unknown, driven by the Ohmic current
    jump::

        bulk   = ElectricPotentialEquations(relative_permittivity=..., conductivity=...)
        ifeqs  = SurfaceChargeConservation(name="qs")
        ifeqs += ElectricPotentialConnection(surface_charge_density="qs")

    *Current-driven (steady DC conduction problems).* The bulk solves current conservation, so its
    natural boundary condition is the jump in the **current**, and the surface charge is then a
    derived algebraic quantity rather than the thing that closes the potential::

        bulk   = OhmicConductionEquations(conductivity=..., relative_permittivity=...)
        ifeqs  = ElectricPotentialConnection()          # NOTE: no surface_charge_density here

    In the current-driven form, passing ``surface_charge_density=`` to the connection would impose a
    spurious *current* source rather than a charge jump, since that is the flux the Ohmic operator
    conserves. Do not mix the two.

    Unlike the bulk relaxation time, **this** transient is physically meaningful: it is the
    interfacial RC time, which is comparable to the flow time and is what sets the transient of a
    deforming drop.

    Note that ``grad`` on an interface *is* the surface gradient, which is exactly what the surface
    diffusion and surface conduction terms want. Only the bulk currents need ``domain=".."`` and
    ``"|.."``, and those are supplied here.

    Args:
        name: Name of the surface charge field. Defaults to ``"surface_charge_density"``.
        charge_scale: The scale of the charge field, as a named scale or an expression. Defaults to
            the named scale ``"surface_charge"``, which
            :py:func:`set_electrostatic_scaling` registers. It deliberately does *not* share the name
            of the field: a field whose scale is looked up under its own name is self-referential and
            fails at code generation with "Cannot expand the expression any further", which is what
            the default pair used to do to anyone who did not also rename the field.
        space: Its finite element space. Defaults to ``"C2"``.
        surface_diffusivity: :math:`K_\mathrm{s}`, diffusion of the adsorbed charge. Defaults to 0.
        surface_conductivity: :math:`\kappa_\mathrm{s}`, the excess surface conductance that the
            Dukhin number measures. ``None`` takes ``interface_props.surface_conductance``, else 0.
        form: ``"conservative"`` (default) or ``"legacy"``, see above.
        scheme: Time stepping scheme of the conservative transient term. The default ``"BDF2_degr"``
            degrades to first order in the very first step, where an initial condition has no
            history. Plain ``"BDF2"`` here makes the *whole run* first order, which is easy to miss.
        dt_factor: Multiplicative factor on the time derivative. 0 is the steady charge balance.
        fluid_velocity: The velocity of the liquid, which carries the charge tangentially. ``None``
            takes ``var("velocity")`` if a co-located (Navier-)Stokes exists, else 0.
        interface_velocity: The velocity of the interface itself, which the charge follows normally.
            ``None`` takes the mesh velocity if the mesh moves, else 0.
        advection_velocity: **Deprecated**, kept because it is the published name. ``"auto"``
            resolves the two above; ``0`` means no fluid velocity (the interface still carries its
            charge along, which is what conservation means); an expression is used as
            ``fluid_velocity``. In ``form="legacy"`` it keeps its literal old meaning, the *total*
            advection velocity.
        adsorption: Ad-/desorption onto the interface, positive towards it. Either an expression, a
            net **charge** flux in C/(m^2 s), or a ``{ion_name: molar_rate}`` mapping in mol/(m^2 s),
            which contributes :math:`\sum_i z_i F R_i` to the charge and takes exactly :math:`R_i`
            out of a co-located :py:class:`NernstPlanckEquations` bulk. ``None`` takes
            ``interface_props.surface_charge_adsorption_rate`` / ``.ion_adsorption_rate``.
        bulk_coupling: Whether the ``{ion: rate}`` form also removes the ions from the bulk.
            ``"auto"`` couples every species the parent Nernst-Planck transports.
        bulk_currents: The Ohmic current jump driving the charge. ``"auto"`` builds it from the
            bulk conductivities as before; pass 0 for a problem that has none, e.g. a prescribed
            interface motion or a charge fed purely by adsorption.
        interface_props: Interface properties to take ``surface_conductivity`` and ``adsorption``
            from. Note that ``interface_props.surface_charge_density`` is what
            :py:class:`SurfaceChargeBC` imposes as a *fixed* charge: using both classes on one
            interface counts the charge twice.
        initial_charge: Initial condition for the charge. Defaults to 0.
        quasi_static: Sugar for ``dt_factor=0``. Defaults to False.
    """
    required_parent_type = ElectricPotentialEquations

    def __init__(self,*,name:str="surface_charge_density",space:"FiniteElementSpaceEnum"="C2",
                 surface_diffusivity:ExpressionOrNum=0,
                 surface_conductivity:"ExpressionNumOrNone"=None,
                 form:Literal["conservative","legacy"]="conservative",
                 scheme:"IntegralTimeSteppingScheme"="BDF2_degr",
                 dt_factor:ExpressionOrNum=1,
                 fluid_velocity:"ExpressionNumOrNone"=None,
                 interface_velocity:"ExpressionNumOrNone"=None,
                 advection_velocity:"ExpressionOrNum | Literal['auto']"="auto",
                 adsorption:"ExpressionOrNum | dict[str,ExpressionOrNum] | None"=None,
                 bulk_coupling:"Literal['auto'] | bool"="auto",
                 bulk_currents:"Literal['auto'] | ExpressionOrNum"="auto",
                 interface_props:"BaseInterfaceProperties | None"=None,
                 initial_charge:ExpressionOrNum=0,quasi_static:bool=False,
                 charge_scale:"ExpressionOrNum | str"="surface_charge"):
        super().__init__()
        if form not in ("conservative","legacy"):
            raise ValueError("SurfaceChargeConservation form must be 'conservative' or 'legacy', "+
                             "not '"+str(form)+"'")
        if advection_velocity!="auto" and fluid_velocity is not None:
            raise ValueError("Pass either fluid_velocity or the deprecated advection_velocity to "+
                             "SurfaceChargeConservation, not both.")
        self.name=name
        self.space:"FiniteElementSpaceEnum"=space
        self.surface_diffusivity=surface_diffusivity
        self.form:Literal["conservative","legacy"]=form
        self.scheme:"IntegralTimeSteppingScheme"=scheme
        self.dt_factor:ExpressionOrNum=0 if quasi_static else dt_factor
        self.fluid_velocity=fluid_velocity
        self.interface_velocity=interface_velocity
        self.advection_velocity=advection_velocity
        self.bulk_coupling:"Literal['auto'] | bool"=bulk_coupling
        self.bulk_currents:"Literal['auto'] | ExpressionOrNum"=bulk_currents
        self.interface_props=interface_props
        self.initial_charge=initial_charge
        self.quasi_static=quasi_static
        self.charge_scale:"ExpressionOrNum | str"=charge_scale
        props=interface_props
        if surface_conductivity is None:
            surface_conductivity=getattr(props,"surface_conductance",None) if props is not None else None
            if surface_conductivity is None:
                surface_conductivity=0
        self.surface_conductivity:ExpressionOrNum=surface_conductivity
        if adsorption is None and props is not None:
            # A per-ion table and a lumped charge rate are both allowed on a material; a material
            # that sets neither leaves this None and nothing is assembled.
            ions=getattr(props,"ion_adsorption_rate",None)
            if ions:
                adsorption=dict(ions)
            else:
                adsorption=getattr(props,"surface_charge_adsorption_rate",None)
        self.adsorption:"ExpressionOrNum | dict[str,ExpressionOrNum] | None"=adsorption

    def _bulk_currents(self)->"Expression | None":
        if self.bulk_currents!="auto":
            e=convert_to_expression(self.bulk_currents)
            return None if is_zero(e) else e
        inside=self.get_parent_equations(ElectricPotentialEquations)
        assert isinstance(inside,ElectricPotentialEquations)
        J_in=inside.get_conduction_current("..")
        if J_in is None:
            raise RuntimeError("SurfaceChargeConservation needs a conducting bulk, i.e. "+
                               "OhmicConductionEquations (or another class providing "+
                               "get_conduction_current), on the parent domain. Pass "+
                               "bulk_currents=0 if the charge is not driven by an Ohmic current at "+
                               "all, e.g. when it is fed by adsorption or the motion is prescribed.")
        n=self.get_normal()
        res=dot(n,J_in)
        outside=self.get_opposite_parent_equations(ElectricPotentialEquations)
        if isinstance(outside,ElectricPotentialEquations):
            J_out=outside.get_conduction_current("|..")
            if J_out is not None:
                res=res-dot(n,J_out)
        return res

    def _velocities(self)->tuple[Expression,Expression]:
        """The fluid velocity and the interface velocity of the conservative form."""
        u=self.fluid_velocity
        if u is None and self.advection_velocity!="auto":
            u=cast(ExpressionOrNum,self.advection_velocity)
        return interface_transport_velocities(self,u,self.interface_velocity)

    def _legacy_advection(self)->"Expression | None":
        """The single combined advection velocity of the pre-conservative assembly.

        ``advection_velocity=<expression>`` is still taken raw, exactly as it was, so that a script
        written against the old class reproduces bit for bit. The newer ``fluid_velocity`` /
        ``interface_velocity`` pair is combined into the same ``u_s`` the automatic path builds --
        those names promise a fluid velocity and an interface velocity in *both* forms, and handing
        the raw fluid velocity to a legacy assembly would advect the charge with the liquid across an
        evaporating interface instead of with the interface.
        """
        if self.advection_velocity!="auto":
            given=convert_to_expression(cast(ExpressionOrNum,self.advection_velocity))
            return None if is_zero(given) else given
        if self.fluid_velocity is None:
            from .navier_stokes import StokesEquations
            flow=self.get_parent_domain().get_equations().get_equation_of_type(StokesEquations)
            if isinstance(flow,list):
                flow=flow[0] if len(flow)==1 else None
            if not isinstance(flow,StokesEquations):
                u=None
            else:
                u=var(flow.velocity_name)
            if u is None:
                return None
        else:
            u=convert_to_expression(self.fluid_velocity)
        n=self.get_normal()
        # Tangential fluid velocity plus the normal motion of the interface itself. On a static mesh
        # the second term is zero and this is just the tangential slip.
        res=u-dot(u,n)*n
        if self.interface_velocity is not None:
            res=res+dot(convert_to_expression(self.interface_velocity),n)*n
        elif self.get_current_code_generator()._coordinates_as_dofs:
            res=res+dot(mesh_velocity(),n)*n
        return res

    def _adsorption_terms(self)->"tuple[Expression | None, list[tuple[str,ExpressionOrNum]]]":
        """The charge gained per area and time, and the molar rates to take out of the bulk."""
        if self.adsorption is None:
            return None,[]
        if not isinstance(self.adsorption,dict):
            e=convert_to_expression(self.adsorption)
            return (None if is_zero(e) else subexpression(e)),[]
        parent=self.get_parent_equations(NernstPlanckEquations)
        if isinstance(parent,list):
            parent=parent[0] if len(parent)==1 else None
        valences:dict[str,int]={}
        if isinstance(parent,NernstPlanckEquations):
            valences={i.name:i.valence for i in parent.ions}
        charge:ExpressionOrNum=0
        bulk:list[tuple[str,ExpressionOrNum]]=[]
        # sorted: this fixes the order the terms are added in, hence the generated code, so it must
        # not depend on how the dict was written.
        for ion_name,rate in sorted(self.adsorption.items()):
            if ion_name not in valences:
                if valences:
                    raise ValueError("SurfaceChargeConservation got an adsorption rate for '"+
                                     ion_name+"', which the parent domain does not transport. It "+
                                     "has: "+str(sorted(valences)))
                raise ValueError("SurfaceChargeConservation got a per-ion adsorption rate for '"+
                                 ion_name+"', but the parent domain carries no NernstPlanckEquations "+
                                 "to take the valence from. Pass a net charge flux in C/(m^2 s) "+
                                 "instead, or add the ion transport.")
            rate=subexpression(convert_to_expression(rate))
            charge=charge+valences[ion_name]*faraday_constant*rate
            couple=self.bulk_coupling if self.bulk_coupling!="auto" else True
            if couple:
                bulk.append((cast(NernstPlanckEquations,parent).fieldname_of(ion_name),rate))
        e=convert_to_expression(charge)
        return (None if is_zero(e) else e),bulk

    def define_fields(self):
        # A field defined ON the interface gets no inherited test scale, so this has to make the
        # residual dimensionless by itself: d_t(q)*testscale must be 1, hence T/Q. (A *bulk* field's
        # test function used on an interface is different -- it inherits the parent scale with an
        # extra 1/spatial per level, which is what already makes e.g. weak(sigma, div(u_test))
        # dimensionless in NavierStokesFreeSurface.) time_derivative_of_integral divides by the
        # temporal scale itself, exactly as partial_t does, so the conservative form needs no other
        # scaling than the non-conservative one did.
        self.define_scalar_field(self.name,self.space,scale=_resolve_scale(self.charge_scale),
                                 testscale=scale_factor("temporal")/_resolve_scale(self.charge_scale))

    def define_residuals(self):
        q,q_test=var_and_test(self.name)
        if self.form=="conservative":
            if not is_zero(convert_to_expression(self.dt_factor)):
                # dt_factor multiplies the derivative of the whole integral from OUTSIDE: inside, the
                # history terms would carry it at their own time levels.
                self.add_residual(self.dt_factor*time_derivative_of_integral(weak(q,q_test),
                                                                             scheme=self.scheme))
            u,w=self._velocities()
            # The GCL transient strongly supplies dt_factor*(d_t q|_E + div_s(q w)), while the
            # equation wants dt_factor*d_t q|_E + div_s(q u). So the flux term has to carry
            # u - dt_factor*w, not u - w. They coincide at dt_factor=1; at dt_factor=0 this collapses
            # to the correct steady transport div_s(q u), where a bare (u-w) would have solved
            # div_s(q(u-w))=0 instead. Conservation is unaffected either way, since the whole term
            # vanishes for the constant test function.
            adv=u-self.dt_factor*w
            if not is_zero(adv):
                self.add_residual(-weak(q*adv,grad(q_test)))
        else:
            if not is_zero(convert_to_expression(self.dt_factor)):
                self.add_residual(weak(self.dt_factor*partial_t(q,ALE="auto"),q_test))
            u_s=self._legacy_advection()
            if u_s is not None:
                self.add_residual(weak(div(q*u_s),q_test))
        if not is_zero(convert_to_expression(self.surface_diffusivity)):
            self.add_residual(weak(self.surface_diffusivity*grad(q),grad(q_test)))
        if not is_zero(convert_to_expression(self.surface_conductivity)):
            parent=self.get_parent_equations(ElectricPotentialEquations)
            assert isinstance(parent,ElectricPotentialEquations)
            # Surface current i_s = -kappa_s*grad_s(phi); div_s(i_s) integrated by parts is
            # -weak(i_s, grad(v)), hence the plus.
            self.add_residual(weak(self.surface_conductivity*grad(var(parent.name)),grad(q_test)))
        currents=self._bulk_currents()
        if currents is not None:
            self.add_residual(-weak(currents,q_test))
        charge_rate,bulk_rates=self._adsorption_terms()
        if charge_rate is not None:
            self.add_residual(-weak(charge_rate,q_test))
        for fieldname,rate in bulk_rates:
            # An ion adsorbing leaves the bulk, i.e. it is an OUTFLUX n.J = rate, and the bulk
            # assembles int(d_t c v) - int(J.grad v), so the boundary term to add is +weak(rate,v).
            # Same convention (and the same helper) as IonFluxBC, whose docstring used to claim the
            # opposite sign.
            self.add_residual(weak(rate,testfunction(fieldname)))
        self.set_initial_condition(self.name,self.initial_charge,degraded_start="auto")


class SurfaceChargeEndFlux(InterfaceEquations):
    r"""
    An imposed surface-charge flux at an end point of an interface -- a contact line, a corner, or an
    edge where the interface stops.

    The conservative form of :py:class:`SurfaceChargeConservation` integrates the advection by parts,
    so its natural end condition is *zero total flux*: nothing leaves the interface, which is what
    makes the total charge exactly conserved. Add this class where that is not what is wanted, e.g.
    where charge drains into a grounded substrate at the contact line.

    A positive flux means charge *leaving* the interface through this end point.

    **Units: a flux per unit length of the end point**, i.e. A/m, in every coordinate system --
    including a two-dimensional Cartesian one, where the end point is a point and its integration
    measure is the dimensionless 1. That is not an accident of this class: a test scale in pyoomph
    gains a factor 1/spatial per domain level, so the charge's test function, which belongs to the
    interface, already carries the length the point measure does not. The two cancel, and the result
    is independent of ``Problem.set_scaling(spatial=...)``.

    Note that in an axisymmetric problem the measure of a point domain carries :math:`2\pi r`, so a
    flux imposed at an end point sitting on the symmetry axis contributes nothing at all -- correctly
    so, since the ring it lives on has zero circumference. Pass ``coordsys=cartesian`` if a
    plain point value is wanted there instead.

    Args:
        flux: The outward flux per unit end-point length.
        coordsys: Override the coordinate system of the point integral. The former name
            ``coordinate_system`` is deprecated, but still accepted.
    """
    required_parent_type = SurfaceChargeConservation

    coordinate_system = _deprecated_attribute_alias("coordinate_system","coordsys")

    @_deprecated_kwargs(coordinate_system="coordsys")
    def __init__(self,flux:ExpressionOrNum,*,coordsys:"OptionalCoordinateSystem"=None):
        super().__init__()
        self.flux=flux
        self.coordsys=coordsys

    def define_residuals(self):
        parent=self.get_parent_equations(SurfaceChargeConservation)
        assert isinstance(parent,SurfaceChargeConservation)
        self.add_residual(weak(self.flux,testfunction(parent.name),
                               coordsys=self.coordsys))


class NernstPlanckEquations(ScalarTransportEquations):
    r"""
    Transport of :math:`N` ionic species by advection, diffusion and electromigration,

    .. math:: \partial_t c_i + \nabla\cdot\vec{J}_i = R_i\,,\qquad
              \vec{J}_i = c_i\vec{u} - D_i\nabla c_i - z_i m_i F c_i\nabla\phi\,,

    with the molar mobility :math:`m_i=D_i/(RT)` by the Einstein relation unless given explicitly.

    This class does **not** solve for the potential. Add an
    :py:class:`ElectricPotentialEquations` on the same domain with
    ``charge_density=<these equations>.get_charge_density()``, or use the
    :py:func:`PoissonNernstPlanck` factory, which does exactly that.

    **The natural boundary condition depends on how the advection was assembled**, and it is *not*
    the zero total flux this docstring used to claim. Only the terms written against
    ``grad(c_test)`` leave a boundary term behind, so with the default ``advection_by_parts=False``
    the natural condition is zero *diffusive plus migration* flux and the advective part is
    unconstrained, while ``advection_by_parts=True`` makes it the zero *total* flux. Either way an
    insulating wall needs no interface equation at all; :py:class:`IonFluxBC` is only needed to
    prescribe a nonzero flux, or to subtract the footprint the stabilization leaves behind.

    .. note::
        Stabilization is per-species: both the wind
        (:py:meth:`stabilization_wind_for_field`) and the reaction rate
        (:py:meth:`stabilization_reaction_rate`) that migration contributes are supplied, so
        :math:`\tau` is sized from the drift each ion actually experiences rather than from the
        fluid velocity. A consequence worth knowing: a Nernst-Planck equation is *advective* even
        with ``wind=0``, since the migration drift is not zero, so a stabilized quiescent
        electrolyte does reference ``scale_factor("velocity")`` -- set it to the migration drift
        scale, or pass a ``ScalarTransportStabilization`` with a different ``velocity_scale``.

    Args:
        ions: The species, as :py:class:`IonSpec` objects, a ``{name: valence}`` mapping, or a
            liquid material with ions dissolved in it (see
            :py:meth:`~pyoomph.materials.generic.BaseLiquidProperties.add_salt`). ``None`` reads
            them from ``fluid_props``.
        potential_name: Name of the potential field this migrates in. Defaults to ``"phi"``.
        space: Finite element space of the concentrations. Defaults to ``"C2"``.
        wind: The advecting velocity. Defaults to ``var("velocity")``; pass 0 for a quiescent
            electrolyte, which also removes the velocity scale from the problem.
        temperature: Temperature entering the Einstein relation. Defaults to ``var("temperature")``.
        field_prefix: Prefix of the concentration field names. Defaults to ``"c_"``.
        fluid_props: A liquid material whose dissolved ions are read when ``ions`` is not given.
        reactions: Volumetric source terms, as a ``{ion_name: rate}`` mapping.
        dt_factor: Multiplicative factor on the time derivative. Defaults to 1.
        time_scheme: Time stepping scheme. Defaults to None, i.e. the problem's default.
        advection_by_parts: Integrate the advective term by parts. ``"auto"`` follows ``GCL``, which
            requires it.
        GCL: Write the transient term as the derivative of the whole integral and advect with the
            velocity *relative to the mesh*, i.e. the conservative ALE form, exactly as
            :py:class:`~pyoomph.equations.salt_transport.SaltTransportEquations` and
            :py:class:`~pyoomph.equations.multi_component.CompositionAdvectionDiffusionEquations` do.
            ``"auto"`` (the default) switches it on whenever the mesh moves, and leaves a static mesh
            untouched. Implies ``advection_by_parts``. **It changes what a boundary with no interface
            equation means**, which is the one way to be silently wrong here: what a non-volatile ion
            needs at an evaporating interface depends on how the bulk was assembled. With the
            advection not by parts the natural condition is zero *diffusive* flux and the interface
            needs ``-c*j_total/rho``; by parts without ``GCL`` it is zero flux in the lab frame and
            the interface needs ``+c*u_mesh.n``; under ``GCL`` it is zero flux *through the moving
            boundary* and the interface needs **nothing at all**. So an :py:class:`IonFluxBC` written
            for one of the other two becomes a double count here. ``dev_docs/salt_transport.md``
            section 3 has the measurements.
        gcl_scheme: Time stepping scheme of the ``GCL`` transient. A different set of schemes from
            ``time_scheme``: only these five understand a derivative of an integral, and conversely
            ``time_scheme`` rejects the ``_degr`` names. The default ``"BDF2_degr"`` degrades to
            first order in the very first step, where an initial condition has no history; plain
            ``"BDF2"`` here makes the *whole run* first order, which is easy to miss.
        concentration_scale: Named scale shared by all species. Defaults to ``"ion_concentration"``.
        set_bulk_initial_conditions: Initialise each species at its ``bulk_concentration``.
        stabilization: Optional residual-based stabilization, see the note above.

    .. note::
        **Scalings to set on problem level.** The concentration fields are scaled with the *named*
        scale ``concentration_scale`` (``"ion_concentration"`` by default), which is deliberately not
        a field name, and the substituted charge density uses ``charge_density``. Both are registered
        by :py:func:`set_electrostatic_scaling`.
    """

    def __init__(self,ions:"Sequence[IonSpec] | dict[str,int] | AnyMaterialProperties | None"=None,*,potential_name:str="phi",
                 space:"FiniteElementSpaceEnum"="C2",wind:ExpressionOrNum=var("velocity"),
                 temperature:ExpressionOrNum=var("temperature"),field_prefix:str="c_",
                 fluid_props:"AnyMaterialProperties | None"=None,
                 reactions:"dict[str,ExpressionOrNum] | None"=None,dt_factor:ExpressionOrNum=1,
                 time_scheme:"TimeSteppingScheme | None"=None,
                 advection_by_parts:"bool | Literal['auto']"="auto",
                 GCL:"bool | Literal['auto']"="auto",
                 gcl_scheme:"IntegralTimeSteppingScheme"="BDF2_degr",
                 concentration_scale:str="ion_concentration",velocity_name_for_scaling:str="velocity",
                 set_bulk_initial_conditions:bool=True,consider_scaling:bool=True,
                 stabilization:"str | Iterable[str] | ScalarTransportStabilization | None"=None):
        super().__init__()
        if ions is None:
            if fluid_props is None:
                raise ValueError("NernstPlanckEquations needs either ions or a fluid_props whose "+
                                 "dissolved ions it can read")
            ions=fluid_props
        if not isinstance(ions,(list,tuple,dict)):
            # A material: read whatever is dissolved in it.
            ions=ions_from_material(cast("AnyMaterialProperties",ions),temperature=temperature)
        if isinstance(ions,dict):
            ions=[IonSpec(n,z) for n,z in sorted(ions.items())]
        # sorted: this list fixes the order the concentration fields are defined in, hence the dof
        # numbering and the output column order, so it must not depend on dict insertion order.
        self.ions:list[IonSpec]=sorted(ions,key=lambda i:i.name)
        self.potential_name=potential_name
        self.space:"FiniteElementSpaceEnum"=space
        self.wind=wind
        self.temperature=temperature
        self.field_prefix=field_prefix
        self.reactions:dict[str,ExpressionOrNum]=dict(reactions) if reactions else {}
        for n in self.reactions:
            if n not in {i.name for i in self.ions}:
                raise ValueError("reactions has an entry for '"+n+"', which is not one of the ions")
        self.dt_factor=dt_factor
        self.time_scheme:"TimeSteppingScheme | None"=time_scheme
        if GCL is False and advection_by_parts=="auto":
            advection_by_parts=False
        if GCL is True and advection_by_parts is False:
            raise ValueError("NernstPlanckEquations cannot combine GCL=True with "+
                             "advection_by_parts=False: the conservative ALE form *is* the by-parts "+
                             "form, since d/dt of the whole integral needs the advection written "+
                             "against grad(c_test) or the two do not describe the same equation.")
        self.advection_by_parts:"bool | Literal['auto']"=advection_by_parts
        self.GCL:"bool | Literal['auto']"=GCL
        self.gcl_scheme:"IntegralTimeSteppingScheme"=gcl_scheme
        self.concentration_scale=concentration_scale
        self.velocity_name_for_scaling=velocity_name_for_scaling
        self.set_bulk_initial_conditions=set_bulk_initial_conditions
        self.consider_scaling=consider_scaling
        self.fluid_props=fluid_props
        self.spatial_error_estimators=True
        self._init_stabilization(stabilization,velocity_scale=velocity_name_for_scaling)

    # ---- bookkeeping ---------------------------------------------------------------------------

    @property
    def ion_names(self)->list[str]:
        return [i.name for i in self.ions]

    def fieldname_of(self,ion_name:str)->str:
        """The name of the concentration field of one ion, see :py:func:`ion_fieldname_stem`."""
        return self.field_prefix+ion_fieldname_stem(ion_name)

    def _ion_of(self,fieldname:str)->IonSpec:
        for i in self.ions:
            if self.fieldname_of(i.name)==fieldname:
                return i
        raise KeyError("No ion corresponds to the field '"+fieldname+"'")

    def get_valence(self,ion_name:str)->int:
        return self._ion_of(self.fieldname_of(ion_name)).valence

    def get_diffusivity(self,ion_name:str)->ExpressionOrNum:
        D=self._ion_of(self.fieldname_of(ion_name)).diffusivity
        if D is None:
            raise ValueError("Ion '"+ion_name+"' has no diffusivity set")
        return D

    def get_migration_mobility(self,ion_name:str)->ExpressionOrNum:
        r"""The coefficient :math:`z_i m_i F` in :math:`\vec{J}_i^\mathrm{mig}=-z_i m_i F c_i\nabla\phi`."""
        ion=self._ion_of(self.fieldname_of(ion_name))
        return ion.valence*ion.get_mobility(self.temperature)*faraday_constant

    def get_information_string(self)->str:
        return "Nernst-Planck for "+", ".join(str(i) for i in self.ions)+" migrating in <"+ \
               self.potential_name+">"

    # ---- what the electrostatics asks for --------------------------------------------------------

    def get_charge_density(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        r""":math:`\rho_\mathrm{e}=F\sum_i z_i c_i`, i.e. what to hand to
        :py:class:`ElectricPotentialEquations`."""
        return faraday_constant*sum(i.valence*var(self.fieldname_of(i.name),domain=domain)
                                    for i in self.ions)

    def get_ionic_strength(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        r""":math:`I=\frac{1}{2}\sum_i z_i^2 c_i`, the *local* ionic strength."""
        return convert_to_expression(sum(i.valence**2*var(self.fieldname_of(i.name),domain=domain) for i in self.ions)/2)

    def get_debye_length(self,permittivity:ExpressionOrNum,
                         domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        return debye_length(permittivity,self.get_ionic_strength(domain),self.temperature)

    def get_current_density(self,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        r""":math:`\vec{i}=F\sum_i z_i\vec{J}_i`, the total ionic current density."""
        return faraday_constant*sum(i.valence*self.get_flux(i.name,domain=domain) for i in self.ions)

    def get_flux(self,ion_name:str,domain:"str | FiniteElementCodeGenerator | None"=None)->Expression:
        r""":math:`\vec{J}_i=c_i\vec{u}-D_i\nabla c_i-z_i m_i F c_i\nabla\phi`. Pass ``domain=".."``
        when calling this from an interface, or ``grad`` is the surface gradient."""
        c=var(self.fieldname_of(ion_name),domain=domain)
        phi=var(self.potential_name,domain=domain)
        res=-self.get_diffusivity(ion_name)*grad(c)-self.get_migration_mobility(ion_name)*c*grad(phi)
        if not is_zero(convert_to_expression(self.wind)):
            wind=self.wind if domain is None else evaluate_in_domain(convert_to_expression(self.wind),domain)
            res=res+c*wind
        return res

    # ---- the equations -------------------------------------------------------------------------

    def define_fields(self):
        for i in self.ions:
            fn=self.fieldname_of(i.name)
            if self.consider_scaling:
                ts=scale_factor("spatial")/scale_factor(self.velocity_name_for_scaling)/scale_factor(fn) \
                    if self._advective() else scale_factor("spatial")**2/ \
                       (convert_to_expression(i.diffusivity if i.diffusivity is not None else 1)*scale_factor(fn))
            else:
                ts=1
            self.define_scalar_field(fn,self.space,scale=scale_factor(self.concentration_scale),testscale=ts)
        self.define_field_by_substitution("charge_density",
                                          self.get_charge_density()/scale_factor("charge_density"),
                                          also_on_interface=True)

    def _advective(self)->bool:
        """Whether an advective term is present, for the *test scaling*.

        Deliberately blind to the mesh velocity, even under GCL where a quiescent electrolyte on a
        moving mesh does assemble a flux term. Flipping this on a moving mesh would swap the
        diffusive test scale spatial^2/(D*c) for spatial/velocity/c in every such problem and would
        reference scale_factor("velocity"), which the class docstring promises wind=0 removes from
        the problem. Whether the flux *term* is emitted is a separate question, see _mesh_moves.
        """
        return not is_zero(convert_to_expression(self.wind))

    def _mesh_moves(self)->bool:
        """Whether the parent domain has its coordinates as unknowns.

        Only valid from define_residuals: activate_coordinates_as_dofs runs from another equation's
        define_fields and the order within a domain is not guaranteed.
        """
        return bool(self.get_current_code_generator()._coordinates_as_dofs)

    def _use_gcl(self)->bool:
        """Resolve ``GCL="auto"``: on iff the mesh moves.

        On a static mesh the conservative branch is the same equation, but integrating the advection
        by parts changes the natural boundary condition from "zero diffusive plus migration flux" to
        "zero total flux", which differs wherever there is through-flow. "auto" therefore turns the
        conservative form on exactly where it is the point - a moving mesh, where the alternative
        does not conserve the dissolved amount at all - and leaves a static problem untouched.
        """
        if self.GCL!="auto":
            return bool(self.GCL)
        return self._mesh_moves()

    def define_residuals(self):
        if self.time_scheme is None:
            ts:Callable[[Expression],Expression]=lambda x:x
        else:
            ts=lambda x:time_scheme(cast("TimeSteppingScheme",self.time_scheme),x)
        gphi=grad(var(self.potential_name))
        gcl=self._use_gcl()
        by_parts=gcl if self.advection_by_parts=="auto" else bool(self.advection_by_parts)
        adv:ExpressionOrNum=0
        if gcl:
            w=mesh_velocity() if self._mesh_moves() else Expression(0)
            adv=convert_to_expression(self.wind)-self.dt_factor*w
        for i in self.ions:
            fn=self.fieldname_of(i.name)
            c,c_test=var_and_test(fn)
            if gcl:
                # The conservative ALE form, term for term as SaltTransportEquations and the
                # composition equations write it: the derivative of the whole integral, so that the
                # change of the element volume is taken by the same finite difference that advances
                # the field, and advection with the velocity relative to the mesh. dt_factor
                # multiplies from OUTSIDE - inside, the history terms would carry it at their own
                # time levels.
                self.add_residual(self.dt_factor*time_derivative_of_integral(weak(c,c_test),
                                                                             scheme=self.gcl_scheme))
                # dt_factor multiplies w as well: the GCL transient strongly supplies
                # dt_factor*(d_t c|_E + div(c w)), while the equation wants dt_factor*d_t c|_E +
                # div(c u), so the flux has to carry u - dt_factor*w. They coincide at dt_factor=1.
                if not is_zero(adv):
                    self.add_residual(-weak(ts(c*adv),grad(c_test)))
            else:
                self.add_residual(weak(ts(self.dt_factor*partial_t(c,ALE="auto")),c_test))
                if self._advective():
                    if by_parts:
                        self.add_residual(-weak(ts(c*self.wind),grad(c_test)))
                    else:
                        self.add_residual(weak(ts(dot(self.wind,grad(c))),c_test))
            self.add_residual(weak(ts(self.get_diffusivity(i.name)*grad(c)),grad(c_test)))
            # Migration. J_mig = -z m F c grad(phi); the by-parts weak form of div(J) is
            # -int J.grad(v), so the sign here is PLUS. Written with E instead of grad(phi) this
            # would read -weak(z*m*F*c*E, grad(v)) -- the same thing, since E = -grad(phi).
            self.add_residual(weak(ts(self.get_migration_mobility(i.name)*c*gphi),grad(c_test)))
            if i.name in self.reactions:
                self.add_residual(-weak(ts(convert_to_expression(self.reactions[i.name])),c_test))
            if self.set_bulk_initial_conditions and i.bulk_concentration is not None:
                self.set_initial_condition(fn,i.bulk_concentration,degraded_start="auto")
        self.add_stabilization_residuals(ts if self.time_scheme is not None else None)

    def define_error_estimators(self):
        if self.spatial_error_estimators:
            for i in self.ions:
                # nondim on BOTH the field and the gradient: an error estimator expression must be
                # fully dimensionless, and grad() alone still carries the 1/spatial scale, which
                # leaks the unit symbol straight into the generated C.
                self.add_spatial_error_estimator(grad(nondim(self.fieldname_of(i.name)),nondim=True))

    def define_additional_functions(self):
        self.add_local_function("charge_density",self.get_charge_density())
        self.add_local_function("ionic_strength",self.get_ionic_strength())

    # ---- stabilization hooks (see the note in the class docstring) --------------------------------

    def stabilized_fieldnames(self)->list[str]:
        return [self.fieldname_of(i.name) for i in self.ions]

    def stabilization_wind(self)->ExpressionOrNum:
        # The part every species shares. The migration drift is per-species and lives in
        # stabilization_wind_for_field below.
        return self.wind

    def stabilization_wind_for_field(self,fieldname:str)->ExpressionOrNum:
        r"""The species' own wind :math:`\vec{u}-z_i m_i F\nabla\phi`.

        Both the streamline direction of the SUPG weight and the cell Peclet number in
        :math:`\tau` differ per species -- a cation and an anion drift in *opposite* directions in
        the same field -- so sizing either from the fluid velocity alone is wrong wherever migration
        dominates, which is precisely inside a Debye layer. Note that this is nonzero even in a
        quiescent electrolyte, so a stabilized Nernst-Planck is advective while a stabilized
        advection-diffusion with the same ``wind=0`` is not.
        """
        ion=self._ion_of(fieldname)
        return self.wind-self.get_migration_mobility(ion.name)*grad(var(self.potential_name))

    def stabilization_reaction_rate(self,fieldname:str)->ExpressionOrNum:
        r"""The rate :math:`-z_i m_i F\nabla^2\phi`, i.e. :math:`z_i m_i F\rho_\mathrm{e}/\varepsilon`.

        Expanding the conservative migration term gives
        :math:`-\nabla\cdot(z_i m_i F c_i\nabla\phi)
        = -z_i m_i F\nabla\phi\cdot\nabla c_i - z_i m_i F c_i\nabla^2\phi`: an advective part,
        which is in :py:meth:`stabilization_wind_for_field`, **plus a term linear in** :math:`c_i`
        **itself**, which is a reaction rate. In a thin Debye layer that rate is of order
        :math:`D/\lambda_\mathrm{D}^2` and dominates every other rate in :math:`\tau`, so leaving
        it out leaves :math:`\tau` far too large exactly where the stabilization is being asked to
        work.

        Written as the Laplacian of the potential rather than as
        :math:`\rho_\mathrm{e}/\varepsilon` so that it mirrors, term for term, what
        :py:meth:`strong_residual` assembles -- which is the whole point of a residual-based
        stabilization. It needs second derivatives of the potential, as the migration term of the
        strong residual already does.
        """
        ion=self._ion_of(fieldname)
        return -self.get_migration_mobility(ion.name)*div(grad(var(self.potential_name)))

    def stabilization_diffusivity(self,fieldname:str)->ExpressionOrNum:
        return self.get_diffusivity(self._ion_of(fieldname).name)

    def stabilization_residual_scale(self,fieldname:str)->ExpressionOrNum:
        return 1

    def strong_residual(self,fieldname:str)->Expression:
        ion=self._ion_of(fieldname)
        c=var(fieldname)
        R=self.dt_factor*partial_t(c,ALE="auto")
        if self._advective():
            # Mirror the advective term the Galerkin part actually assembles. This used to be
            # dot(wind,grad(c)) unconditionally, i.e. the stabilization of an advection_by_parts
            # problem was built on a different equation than the one being solved.
            conservative=self.stab_cfg.conservative_residual
            if conservative=="auto":
                # The GCL branch produces the same expression as the plain by-parts one, and that is
                # right: its transient is strongly d_t c|_E + div(c w) and its flux term adds
                # div(c(u-w)), which sum to d_t c|_E + div(c u).
                conservative=self._use_gcl() or (self.advection_by_parts is True)
            R=R+(div(self.wind*c) if conservative else dot(self.wind,grad(c)))
        R=R-div(self.get_migration_mobility(ion.name)*c*grad(var(self.potential_name)))
        if ion.name in self.reactions:
            R=R-convert_to_expression(self.reactions[ion.name])
        if self.stab_cfg.include_diffusion_in_residual:
            R=R-div(self.get_diffusivity(ion.name)*grad(c))
        return R

    # ---- constraints ------------------------------------------------------------------------------

    def with_fixed_amounts(self,problem:"Problem",*,ode_domain_name:str="globals",
                           lagrange_prefix:str="_lagr_ion_amount_",
                           **amounts:ExpressionOrNum)->"Equations":
        r"""
        Constrain the total amount :math:`\int c_i\,\mathrm{d}V` of one or more species.

        With blocking walls everywhere the amounts are only *conserved*, not *fixed*: the steady
        problem then has one nullspace direction per species and the Jacobian is exactly singular.
        This removes them with one global Lagrange multiplier per constrained species.

        Args:
            problem: The problem, needed to add the ODE domain holding the multipliers.
            ode_domain_name: Domain for the global Lagrange multipliers.
            lagrange_prefix: Prefix of the multiplier names.
            **amounts: ``ion_name=total_amount`` pairs.
        """
        eqs:"Equations"=self
        ode_add:"BaseEquations | None"=None
        for n,total in sorted(amounts.items()):
            if n not in self.ion_names:
                raise ValueError("'"+n+"' is not one of the ions")
            fn,lname=self.fieldname_of(n),lagrange_prefix+n
            eqs=eqs+WeakContribution(var(fn),testfunction(lname,domain=ode_domain_name),dimensional_dx=True)
            eqs=eqs+WeakContribution(var(lname,domain=ode_domain_name),testfunction(fn),dimensional_dx=True)
            add:"BaseEquations"=GlobalLagrangeMultiplier(**cast("dict[str,Any]",{lname:total}))
            add=add+TestScaling(**{lname:1/scale_factor(fn)})+Scaling(**{lname:1/test_scale_factor(fn)})
            ode_add=add if ode_add is None else ode_add+add
        if ode_add is not None:
            problem.add_equations(ode_add@ode_domain_name)
        return eqs


class IonFluxBC(InterfaceEquations):
    r"""
    A prescribed molar flux :math:`\vec{n}\cdot\vec{J}_i` of one or more ionic species.

    The *natural* condition of :py:class:`NernstPlanckEquations` is already a blocking wall, so this
    class is only needed to impose a nonzero flux -- or, with ``subtract_stabilization_flux``, to
    make the imposed flux the physical one when the bulk is stabilized. It is an
    :py:class:`~pyoomph.generic.codegen.InterfaceEquations` rather than a plain ``Equations``
    precisely so that it can reach the bulk for that correction.

    **Sign: a positive value is an OUTFLUX**, i.e. species leaving the domain along the outward
    normal. This docstring used to claim the opposite, and nothing caught it because no test imposes
    a nonzero flux. The bulk assembles :math:`\int\partial_t c\,v-\int\vec{J}\cdot\nabla v`,
    so the boundary term it omits is :math:`-\oint(\vec{n}\cdot\vec{J})v` and adding
    ``weak(g,c_test)`` here imposes :math:`\vec{n}\cdot\vec{J}=g`.
    :py:class:`~pyoomph.equations.generic.NeumannBC` uses the same convention once its value is read
    as a flux rather than as a gradient.

    **Which flux** :math:`g` **is depends on the parent's** ``advection_by_parts``: with the default
    ``False`` only the terms written against ``grad(c_test)`` leave a boundary term, so :math:`g` is
    the diffusive plus migration flux; with it on, the advective part joins in and :math:`g` is the
    total :math:`\vec{n}\cdot\vec{J}`.

    Args:
        subtract_stabilization_flux: Subtract the footprint the bulk stabilization leaves on this
            boundary. Has no effect unless the bulk is stabilized with ``natural_bc_correction``.
        **fluxes: ``ion_name=flux`` pairs. Species not listed keep the blocking condition.
    """
    required_parent_type = NernstPlanckEquations

    def __init__(self,*,subtract_stabilization_flux:bool=True,**fluxes:ExpressionOrNum):
        super().__init__()
        self.fluxes=sorted_field_kwargs(fluxes)
        self.subtract_stabilization_flux=subtract_stabilization_flux

    def define_residuals(self):
        parent=self.get_parent_equations(NernstPlanckEquations)
        assert isinstance(parent,NernstPlanckEquations)
        for n in self.fluxes:
            if n not in parent.ion_names:
                raise ValueError("IonFluxBC got a flux for '"+n+"', which is not one of the ions")
        n_vec=self.get_normal()
        for ion in parent.ions:
            fn=parent.fieldname_of(ion.name)
            _,c_test=var_and_test(fn)
            if ion.name in self.fluxes:
                self.add_residual(weak(self.fluxes[ion.name],c_test))
            if self.subtract_stabilization_flux:
                corr=parent.get_stabilization_flux(fn,n_vec,self.get_parent_domain())
                if not is_zero(corr):
                    self.add_residual(-weak(corr,c_test))


def PoissonNernstPlanck(ions:"Sequence[IonSpec] | dict[str,int] | AnyMaterialProperties | None"=None,*,permittivity:ExpressionNumOrNone=None,
                        relative_permittivity:ExpressionNumOrNone=None,
                        fluid_props:"AnyMaterialProperties | None"=None,
                        potential_name:str="phi",potential_space:"FiniteElementSpaceEnum"="C2",
                        permittivity_scale:"ExpressionOrNum | str"=epsilon_0,
                        **kwargs:Any)->"Equations":
    r"""
    The fully resolved Poisson-Nernst-Planck system: :py:class:`NernstPlanckEquations` for the ion
    transport plus an :py:class:`ElectricPotentialEquations` whose charge density is
    :math:`F\sum_i z_i c_i`.

    This is the model that resolves the Debye layer dynamically. Be aware of what that costs:
    :math:`\lambda_\mathrm{D}` is 1-100 nm, so against a micrometre-or-larger geometry the mesh has
    to be strongly graded, and at equilibrium the answer is by construction the one
    :py:class:`PoissonBoltzmannEquations` gives much more cheaply. Resolve the layer when the ions
    are *not* in equilibrium -- an imposed current, a fast flow, an applied AC field.

    Args:
        ions: The species, see :py:class:`NernstPlanckEquations`.
        permittivity / relative_permittivity / fluid_props: The solvent permittivity.
        potential_name: Name of the potential field.
        potential_space: Space of the potential, which need not equal that of the concentrations.
        permittivity_scale: See :py:class:`ElectricPotentialEquations`.
        **kwargs: Forwarded to :py:class:`NernstPlanckEquations`.
    """
    np_eqs=NernstPlanckEquations(ions,potential_name=potential_name,fluid_props=fluid_props,**kwargs)
    es_eqs=ElectricPotentialEquations(name=potential_name,space=potential_space,
                                      permittivity=permittivity,
                                      relative_permittivity=relative_permittivity,
                                      fluid_props=fluid_props,
                                      temperature=kwargs.get("temperature"),
                                      permittivity_scale=permittivity_scale,
                                      charge_density=np_eqs.get_charge_density())
    return np_eqs+es_eqs


class ElectricPotentialConnection(InterfaceEquations):
    r"""
    Couples the electrostatic potential of two adjacent bulk domains,

    .. math:: \phi_\mathrm{in} = \phi_\mathrm{out}\,,\qquad
              \vec{n}\cdot(\vec{D}_\mathrm{out}-\vec{D}_\mathrm{in}) = \sigma_\mathrm{s}

    with :math:`\vec{n}` the **outward** normal of the domain this equation is attached to. The
    first line says there is no dipole layer, the second is Gauss's law for the free surface charge
    :math:`\sigma_\mathrm{s}`.

    Continuity is enforced by a Lagrange multiplier :math:`\lambda`, which is added with
    :math:`+\lambda` to the inside test row and :math:`-\lambda` to the outside one. Because the two
    outward normals are anti-parallel, that alone already *is* the flux continuity
    :math:`\vec{n}\cdot\vec{D}_\mathrm{in}=\vec{n}\cdot\vec{D}_\mathrm{out}` -- **with the correct
    permittivity on each side, neither of which appears in this equation**. The multiplier is
    therefore :math:`\lambda=\vec{n}\cdot\vec{D}_\mathrm{out}`, and putting the surface charge on the
    inside row with a minus sign then gives exactly the same sign convention as
    :py:class:`SurfaceChargeBC` -- so removing the opposite domain degenerates this class
    continuously into that one.

    A consequence worth stating: a *simplified* model on one side (say
    :py:class:`DebyeHuckelEquations` in a liquid) and a plain
    :py:class:`ElectricPotentialEquations` on the other (a gas) couple through this class without
    either side knowing what the other solves, since ``required_opposite_parent_type`` only asks for
    the base class.

    .. warning::
        ``surface_charge_density`` is a jump in the **displacement** field, i.e. it only means a
        charge when the bulk equation is Gauss's law. On an
        :py:class:`OhmicConductionEquations` bulk, whose natural boundary condition is a jump in the
        *current*, this argument imposes a spurious current source instead. See
        :py:class:`SurfaceChargeConservation` for the two consistent pairings.

    Args:
        surface_charge_density: The free surface charge. Either an expression, or the *name* of an
            interface field, e.g. the one solved by :py:class:`SurfaceChargeConservation`. Defaults to 0.
        lagr_mult_name: Name of the Lagrange multiplier field.
        use_highest_space: If the two sides discretise the potential differently, use the higher of
            the two spaces for the multiplier instead of the lower. Defaults to False.
        check_consistent_scaling: Check that both sides agree on the scale and test scale of the
            potential, and raise a readable error if they do not. **Leave this on.** Differing scales
            silently break the flux continuity, because the multiplier is weighted by the test scale
            of each side; this is the trap documented in the multi-domain conduction tutorial.
    """
    required_parent_type = ElectricPotentialEquations
    required_opposite_parent_type = ElectricPotentialEquations

    def __init__(self,*,surface_charge_density:"ExpressionOrNum | str"=0,
                 lagr_mult_name:str="_lagr_phi_conn",use_highest_space:bool=False,
                 check_consistent_scaling:bool=True):
        super().__init__()
        self.surface_charge_density=surface_charge_density
        self.lagr_mult_name=lagr_mult_name
        self.use_highest_space=use_highest_space
        self.check_consistent_scaling=check_consistent_scaling

    def _inside(self)->ElectricPotentialEquations:
        res=self.get_parent_equations(ElectricPotentialEquations)
        assert isinstance(res,ElectricPotentialEquations)
        return res

    def _outside(self)->ElectricPotentialEquations:
        res=self.get_opposite_parent_equations(ElectricPotentialEquations)
        if not isinstance(res,ElectricPotentialEquations):
            raise RuntimeError("ElectricPotentialConnection requires an "+
                               ElectricPotentialEquations.__name__+" on the opposite side of the "+
                               "interface. Use SurfaceChargeBC if the exterior is field-free.")
        return res

    def _surface_charge(self)->ExpressionOrNum:
        if isinstance(self.surface_charge_density,str):
            return var(self.surface_charge_density)
        return self.surface_charge_density

    def define_fields(self):
        inside,outside=self._inside(),self._outside()
        pdom=self.get_parent_domain()
        assert pdom is not None
        inside_space=pdom.get_space_of_field(inside.name)
        oppdom=self.get_opposite_side_of_interface().get_parent_domain()
        assert oppdom is not None
        outside_space=oppdom.get_space_of_field(outside.name)
        if inside_space=="" or outside_space=="":
            raise RuntimeError("Cannot find the potential field on both sides of the interface")
        space=get_interface_field_connection_space(assert_valid_finite_element_space(inside_space),
                                                   assert_valid_finite_element_space(outside_space),
                                                   use_highest_space=self.use_highest_space,
                                                   parent_space=str(self.get_parent_domain()._coordinate_space),
                                                   parent_dim=int(self.get_parent_domain().dimension))
        self.define_scalar_field(self.lagr_mult_name,assert_valid_finite_element_space(space),
                                 scale=1/test_scale_factor(inside.name))

    def _check_scaling(self):
        inside,outside=self._inside(),self._outside()
        opp=self.get_opposite_side_of_interface()
        for what,getter in (("scaling",scale_factor),("test function scaling",test_scale_factor)):
            diff=getter(inside.name)-getter(outside.name,domain=opp)
            diff=self.expand_expression_for_debugging(diff,raise_error=False,unit_error=False)
            if not diff.is_zero():
                raise self._add_exception_info(RuntimeError(
                    "The two sides of this ElectricPotentialConnection disagree on the "+what+" of the "+
                    "potential, which silently breaks the continuity of the normal displacement field.\n"+
                    "  inside  ("+inside.name+"): "+str(self.expand_expression_for_debugging(getter(inside.name)))+"\n"+
                    "  outside ("+outside.name+"): "+str(self.expand_expression_for_debugging(getter(outside.name,domain=opp)))+"\n"+
                    "Set ONE problem-level scale for the potential (see set_electrostatic_scaling) and "+
                    "give both domains the same permittivity_scale, or pass check_consistent_scaling=False."))

    def define_residuals(self):
        inside,outside=self._inside(),self._outside()
        if self.check_consistent_scaling:
            self._check_scaling()
        opp=self.get_opposite_side_of_interface()
        dx=self.get_dx(use_scaling=False)
        l,l_test=var_and_test(self.lagr_mult_name)
        pin,pin_test=var_and_test(inside.name)
        pout,pout_test=var_and_test(outside.name,domain=opp)
        scal=self.get_scaling(inside.name)
        self.add_residual((pin-pout)/scal*l_test*dx)
        self.add_residual(l*pin_test*dx)
        self.add_residual(-l*pout_test*dx)
        sigma_s=self._surface_charge()
        if not is_zero(sigma_s):
            self.add_residual(-weak(sigma_s,pin_test))

    def before_assigning_equations_postorder(self,mesh:"AnyMesh"):
        # Without this a corner where the interface meets an electrode -- both sides strongly pinned
        # -- leaves the multiplier unconstrained and the Jacobian exactly singular.
        assert isinstance(mesh,InterfaceMesh)
        self.pin_redundant_lagrange_multipliers(mesh,self.lagr_mult_name,self._inside().name,
                                                self._outside().name)
        super().before_assigning_equations_postorder(mesh)


class ThinDielectricLayer(InterfaceEquations):
    r"""
    An unresolved thin dielectric layer of thickness :math:`d` and permittivity
    :math:`\varepsilon_\mathrm{l}` sitting on an interface, i.e. a pure capacitance

    .. math:: \vec{n}\cdot\vec{D}_\mathrm{in} = C\,(\phi_\mathrm{in}-\phi_\mathrm{out})\,,
              \qquad C=\varepsilon_\mathrm{l}/d\,,

    which is what a coating, an oxide layer or a Stern layer does. The layer is not meshed; the
    potential is expanded linearly across it and integrated analytically, exactly as
    :py:class:`~pyoomph.equations.multi_component.ThinLayerThermalConductionEquation` does for a
    thermal contact resistance, from which the implementation is transcribed. Only the
    :math:`\partial_z\Psi\,\partial_z\Psi` block survives here, since the layer stores no charge in
    its interior and has no time derivative.

    Works both between two meshed domains and against a prescribed exterior potential (an
    electrode behind the coating), which is what ``outside_potential`` is for.

    Args:
        thickness: The layer thickness :math:`d`. Ignored if ``capacitance`` is given.
        permittivity / relative_permittivity: The layer permittivity.
        capacitance: The areal capacitance :math:`C=\varepsilon_\mathrm{l}/d` directly.
        outside_potential: If given, the exterior is not meshed and is held at this potential.
        surface_charge_density: Charge trapped *inside* the layer, split evenly onto the two faces.
    """
    required_parent_type = ElectricPotentialEquations

    def __init__(self,*,thickness:ExpressionNumOrNone=None,permittivity:ExpressionNumOrNone=None,
                 relative_permittivity:ExpressionNumOrNone=None,capacitance:ExpressionNumOrNone=None,
                 outside_potential:ExpressionNumOrNone=None,surface_charge_density:ExpressionOrNum=0):
        super().__init__()
        if capacitance is None:
            if thickness is None:
                raise ValueError("Please pass either capacitance or thickness")
            eps=permittivity if permittivity is not None else \
                (relative_permittivity*epsilon_0 if relative_permittivity is not None else None)
            if eps is None:
                raise ValueError("Please pass the layer permittivity or relative_permittivity")
            capacitance=eps/thickness
        self.capacitance:ExpressionOrNum=capacitance
        self.outside_potential=outside_potential
        self.surface_charge_density=surface_charge_density

    def define_residuals(self):
        inside=self.get_parent_equations(ElectricPotentialEquations)
        assert isinstance(inside,ElectricPotentialEquations)
        phi_in,phi_in_test=var_and_test(inside.name)
        phi_out_test:"Expression | None"
        if self.outside_potential is None:
            outside=self.get_opposite_parent_equations(ElectricPotentialEquations)
            if not isinstance(outside,ElectricPotentialEquations):
                raise RuntimeError("ThinDielectricLayer needs either an opposite domain solving the "+
                                   "potential, or an outside_potential.")
            phi_out,phi_out_test=var_and_test(outside.name,domain=self.get_opposite_side_of_interface())
        else:
            phi_out,phi_out_test=convert_to_expression(self.outside_potential),None
        C=self.capacitance
        # n.D_in = C*(phi_in - phi_out): D points from the high to the low potential along n, the same
        # orientation the thermal template has for the heat flux.
        self.add_residual(weak(C*(phi_in-phi_out),phi_in_test))
        if phi_out_test is not None:
            self.add_residual(weak(C*(phi_out-phi_in),phi_out_test))
        if not is_zero(self.surface_charge_density):
            half=self.surface_charge_density/2 if phi_out_test is not None else self.surface_charge_density
            self.add_residual(-weak(half,phi_in_test))
            if phi_out_test is not None:
                self.add_residual(-weak(half,phi_out_test))


class SternLayer(ThinDielectricLayer):
    r"""
    The compact (Stern/Helmholtz) layer of an electric double layer: the few-Angstrom shell of
    solvent and adsorbed ions between a charged wall and the outer Helmholtz plane, across which the
    ions cannot move and the potential therefore drops linearly.

    Electrochemically it is a capacitance in series with the diffuse layer, which is exactly what
    :py:class:`ThinDielectricLayer` models -- this subclass only supplies the electrochemical
    naming and defaults.

    Args:
        stern_capacitance: The compact-layer capacitance, typically 0.1-0.2 F/m^2 for aqueous
            electrolytes. Alternatively give ``thickness`` and a (much reduced, ~6-30) layer
            ``relative_permittivity``, since the solvent is dielectrically saturated there.
        wall_potential: The potential of the metal behind the layer, if it is not meshed.
        interface_props: Interface properties to take ``stern_layer_capacitance`` from instead.
    """
    def __init__(self,*,stern_capacitance:ExpressionNumOrNone=None,thickness:ExpressionNumOrNone=None,
                 relative_permittivity:ExpressionNumOrNone=None,permittivity:ExpressionNumOrNone=None,
                 wall_potential:ExpressionNumOrNone=None,surface_charge_density:ExpressionOrNum=0,
                 interface_props:"BaseInterfaceProperties | None"=None):
        if stern_capacitance is None and thickness is None and interface_props is not None:
            stern_capacitance=interface_props.stern_layer_capacitance
            if stern_capacitance is None:
                raise ValueError("The given interface properties do not define a "+
                                 "stern_layer_capacitance")
        super().__init__(capacitance=stern_capacitance,thickness=thickness,
                         relative_permittivity=relative_permittivity,permittivity=permittivity,
                         outside_potential=wall_potential,surface_charge_density=surface_charge_density)


def set_electrostatic_scaling(problem:"Problem",*,potential:"ExpressionNumOrNone | Literal['thermal']"="thermal",
                              temperature:ExpressionOrNum=293*kelvin,potential_name:str="phi",
                              charge_density:ExpressionNumOrNone=None,
                              ion_concentration:ExpressionNumOrNone=None,
                              electric_field:ExpressionNumOrNone=None,
                              surface_charge_density:ExpressionNumOrNone=None,
                              permittivity:ExpressionOrNum=epsilon_0,
                              length:ExpressionNumOrNone=None)->None:
    r"""
    Sets the problem-level electric scales consistently.

    The potential must have **one** scale shared by all coupled domains: a per-domain scale silently
    breaks the flux continuity across an interface, because the Lagrange multiplier transmitting the
    flux is weighted by the test scale of each side. This helper exists so that the natural way to
    set the scales is also the correct one.

    Args:
        problem: The problem to set the scales on.
        potential: The potential scale. ``"thermal"`` (the default) uses the thermal voltage
            :math:`RT/F`, which is the natural scale of every electrokinetic problem.
        temperature: Temperature used for the thermal voltage.
        potential_name: Name of the potential field. Defaults to ``"phi"``.
        charge_density: Charge density scale. Defaults to :math:`F c` if an ion concentration is
            given, otherwise :math:`\varepsilon\Phi/L^2`.
        ion_concentration: Concentration scale shared by all ionic species.
        electric_field: Field scale. Defaults to :math:`\Phi/L`.
        surface_charge_density: Surface charge scale, registered under both ``"surface_charge"``
            (which :py:class:`SurfaceChargeConservation` uses) and ``"surface_charge_density"``.
            Defaults to :math:`\varepsilon\Phi/L`.
        permittivity: Permittivity used to derive the above. Defaults to :py:data:`epsilon_0`.
        length: Length scale used to derive the above. Defaults to the problem's spatial scale.
    """
    Phi=thermal_voltage(temperature) if potential=="thermal" else potential
    if Phi is None:
        raise ValueError("Please pass a potential scale, or 'thermal' for the thermal voltage")
    scals:dict[str,ExpressionOrNum]={potential_name:Phi}
    X=length if length is not None else problem.get_scaling("spatial")
    if ion_concentration is not None:
        scals["ion_concentration"]=ion_concentration
    if charge_density is not None:
        scals["charge_density"]=charge_density
    elif ion_concentration is not None:
        scals["charge_density"]=faraday_constant*ion_concentration
    else:
        scals["charge_density"]=permittivity*Phi/X**2
    scals["electric_field"]=electric_field if electric_field is not None else Phi/X
    # Registered under both names: "surface_charge" is what SurfaceChargeConservation looks up (it
    # cannot be "surface_charge_density", which is that class's default *field* name), while
    # "surface_charge_density" stays for anything written against the older name.
    sigma_scale=surface_charge_density if surface_charge_density is not None else permittivity*Phi/X
    scals["surface_charge"]=sigma_scale
    scals["surface_charge_density"]=sigma_scale
    scals["permittivity"]=permittivity
    problem.set_scaling(**scals)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
