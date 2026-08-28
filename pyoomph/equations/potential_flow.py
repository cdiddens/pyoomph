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
 
 
from .. import *
from ..expressions import *
from ..meshes.mesh import AnyMesh, InterfaceMesh, assert_spatial_mesh
from ..expressions.units import degree
from ..typings import *
import warnings

class PotentialFlow(Equations):
    r"""
    Potential flow, i.e. an irrotational and incompressible velocity :math:`\vec{u}=\nabla\phi`
    obtained from :math:`\nabla^2\phi=0`. The velocity and (with a mass density) the pressure are
    provided either as local expressions or projected onto a field.

    .. note::
        **Scaling to set on problem level.** The potential is scaled with
        ``scale_factor("velocity")*scale_factor("spatial")`` by default, i.e. even though the
        velocity is usually not a solved field here, ``problem.set_scaling(velocity=...)`` is
        required (or pass an explicit ``scale``).

    Args:
        potential_name: Name of the velocity potential field. Defaults to "phi".
        space: Finite element space of the potential. Defaults to "C2".
        scale: Scale of the potential. Defaults to ``scale_factor("velocity")*scale_factor("spatial")``.
        velo_projection: False: no velocity at all, True: a local expression, a space: projected onto
            a field of that space. Defaults to True.
        velocity_name: Name of the velocity. Defaults to "velocity".
        mass_density: Mass density, required to obtain the pressure from the Bernoulli equation.
        dynamic_viscosity: If set, the velocity at free interfaces is calculated, which is required
            for the viscous normal traction there.
        pressure_projection: As ``velo_projection``, but for the pressure. Only done if a mass
            density is given. Defaults to True.
        pressure_name: Name of the pressure. Defaults to "pressure".
        bulk_force_potential: Potential :math:`V` of a bulk force :math:`\vec{f}=-\nabla V`,
            entering the Bernoulli equation. Defaults to 0.
    """
    def __init__(self,potential_name:str="phi",space:FiniteElementSpaceEnum="C2",scale=scale_factor("velocity")*scale_factor("spatial"),velo_projection:bool | FiniteElementSpaceEnum=True,velocity_name:str="velocity",mass_density:ExpressionNumOrNone=None,dynamic_viscosity:ExpressionNumOrNone=None,pressure_projection:bool | FiniteElementSpaceEnum=True,pressure_name="pressure",bulk_force_potential:ExpressionOrNum=0):
        super().__init__()
        self.potential_name=potential_name
        self.velocity_name=velocity_name
        self.pressure_name=pressure_name
        self.space:FiniteElementSpaceEnum=space
        self.scale=scale
        self.rho=mass_density

        # If viscosity is set, we calculate the velocity at free interfaces, even if we do not project the velocity 
        # we need it, since we cannot calculate the gradient of u, which is second order derivative of phi
        self.dynamic_viscosity=dynamic_viscosity
        self.velo_at_free_interface_space:FiniteElementSpaceEnum="D2" # D1 should be sufficient. But must be DG
        self.velo_at_free_interface_name="dgvelo"
        
        self.velo_projection:bool | FiniteElementSpaceEnum=velo_projection # False: No output/calculate, True: LocalExpression, FiniteElementSpace: Projection
        self.pressure_projection:bool | FiniteElementSpaceEnum=pressure_projection # same as above, but only calculated if mass_density is set (can't be done without rho)
        # Only used to get the pressure. We set it to a potential V that fulfills that the bulk force is f=-grad(V), requires of course that rot(f)=0
        self.bulk_force_potential=bulk_force_potential


        

    def define_fields(self):
        self.define_scalar_field(self.potential_name,self.space,scale=self.scale,testscale=scale_factor("spatial")**2/self.scale)
        if not isinstance(self.velo_projection,bool):
            self.define_vector_field(self.velocity_name,self.velo_projection,testscale=1/scale_factor(self.velocity_name))
        elif self.dynamic_viscosity is not None:
            self.define_vector_field(self.velo_at_free_interface_name,self.velo_at_free_interface_space,scale=scale_factor("velocity"),testscale=1/scale_factor("velocity"))

        if not isinstance(self.pressure_projection,bool) and self.rho is not None:
            self.define_scalar_field(self.pressure_name,self.pressure_projection,testscale=1/scale_factor(self.pressure_name))

    def define_residuals(self):
        phi,phitest=var_and_test(self.potential_name)
        self.add_weak(grad(phi),grad(phitest))
        if not isinstance(self.velo_projection,bool):
            u,utest=var_and_test(self.velocity_name)
            self.add_weak(u-grad(phi),utest,coordsys=cartesian)
        elif self.velo_projection:            
            self.add_local_function(self.velocity_name,grad(phi))
        if self.rho is not None and self.pressure_projection:
            # Unsteady Bernoulli: p=-rho*(dphi/dt+|grad(phi)|^2/2). The 1/2 was missing here,
            # while get_dynamic_boundary_condition of the free interfaces had it correctly.
            pdef=-self.rho*(partial_t(phi)+dot(grad(phi),grad(phi))/2)+self.bulk_force_potential
            if not isinstance(self.pressure_projection,bool):
                p,ptest=var_and_test(self.pressure_name)
                self.add_weak(p-pdef,ptest)
            elif self.pressure_projection:            
                self.add_local_function(self.pressure_name,pdef)

        if self.dynamic_viscosity is not None and isinstance(self.velo_projection,bool):
            # We need the gradient of phi for u as dof anyways to add viscosities at free surfaces
            ui,uitest=var_and_test(self.velo_at_free_interface_name)
            self.add_weak(ui-grad(phi),uitest,coordsys=cartesian)

    def get_interface_velocity_for_viscous_at_interfaces(self,domain=None):
        if self.dynamic_viscosity is not None:
            if isinstance(self.velo_projection,bool):
                return var(self.velo_at_free_interface_name,domain=domain)
            else: 
                return var(self.velocity_name,domain=domain)
        


    def before_assigning_equations_postorder(self, mesh: AnyMesh):
        if self.dynamic_viscosity is None or not isinstance(self.velo_projection,bool):
            return super().before_assigning_equations_postorder(mesh)
        mesh=assert_spatial_mesh(mesh)
        # Pin all degrees in the bulk that are not part of an element connected to one of the desired interfaces
        dirs=["x","y","z"]
        vcomps=[self.velo_at_free_interface_name+"_"+dirs[i] for i in range(self.get_nodal_dimension())]
        # Pin all
        for e in mesh.elements():
            for vcompo in vcomps:
                for ni,ind in e.get_field_data_list(vcompo,False):
                    ni.pin(ind)
                    ni.set_value(ind,0)
        # Unpin the elements that are attached to at least one interface elements
        for iname in mesh._interfacemeshes.keys():
            imesh=mesh.get_mesh(iname)
            free_inters=imesh.get_eqtree().get_equations().get_equation_of_type(_PotentialFlowFreeInterfaceBase,always_as_list=True)
            if len(free_inters)>0:
                for iel in imesh.elements():
                    e=iel.get_bulk_element()
                    for vcompo in vcomps:
                        for ni,ind in e.get_field_data_list(vcompo,False):
                            ni.unpin(ind)
                    
        return super().before_assigning_equations_postorder(mesh)


class _PotentialFlowInterfaceEquations(InterfaceEquations):
    required_parent_type=PotentialFlow
    def get_potential_flow(self)->PotentialFlow:
        potflow=self.get_parent_equations(of_type=PotentialFlow)
        assert isinstance(potflow,PotentialFlow)
        return potflow
    
    def get_phi_and_test(self,bulk:bool=False):
        potflow=self.get_potential_flow()
        phi,phi_test=var_and_test(potflow.potential_name,domain=".." if bulk else None)
        return phi,phi_test

class PotentialFlowNormalVelocity(_PotentialFlowInterfaceEquations):
    def __init__(self,unorm:ExpressionOrNum=0):
        super().__init__()
        self.unorm=unorm

    def define_residuals(self):        
        _phi,phi_test=self.get_phi_and_test()
        self.add_weak(-self.unorm,phi_test)

class PotentialFlowFarField(_PotentialFlowInterfaceEquations):
    r"""
    Truncates an unbounded potential flow at an artificial outer boundary by the monopole decay

    .. math:: \vec{n}\cdot\nabla\phi = -\left(\phi-\phi_\infty\right)
              \frac{\vec{n}\cdot\vec{d}}{\left|\vec{d}\right|^{2}}\,,
              \qquad \vec{d}=\vec{x}-\vec{x}_0\,,

    which on a sphere of radius :math:`R` is :math:`\phi+R\,\partial_r\phi=\phi_\infty`, i.e. exact
    for a :math:`\phi\sim 1/r` source. The truncation radius then barely matters: for a pulsating
    bubble the period is reproduced at :math:`R_\text{out}/R_0=5` as well as at 20, whereas simply
    pinning :math:`\phi=0` there is in error by 88 % and 24 % respectively.

    Args:
        phi: The potential at infinity :math:`\phi_\infty`. Defaults to 0.
        origin: The origin :math:`\vec{x}_0` the decay is measured from.
        speed_of_sound: If given, add acoustic radiation damping, see below. ``None`` (the default)
            is the strictly incompressible flow, which does not radiate at all.

    **Radiation damping.** The bulk solves :math:`\nabla^2\phi=0`, so the liquid is incompressible
    and a pulsating body loses no energy to sound. Real bubble dynamics does: for a 50 um bubble in
    water driven near its Minnaert resonance the radiation damping is a few times the viscous one,
    so leaving it out overestimates the amplitude badly. Passing ``speed_of_sound`` adds the
    first-order compressible correction, i.e. the outgoing-wave term of the Sommerfeld condition,

    .. math:: \vec{n}\cdot\nabla\phi = -\left(\phi-\phi_\infty\right)
              \frac{\vec{n}\cdot\vec{d}}{\left|\vec{d}\right|^{2}}
              - \frac{1}{c}\,\partial_t\phi\,.

    That single term is not a fudge factor: testing the energy flux
    :math:`\oint p\,\vec{u}\cdot\vec{n}\,\mathrm{d}S` with :math:`\phi=-Q/(4\pi r)` shows it removes
    exactly :math:`\rho\dot{Q}^{2}/(4\pi c)`, the acoustic power of a compact monopole of source
    strength :math:`Q=\dot{V}`. It is therefore independent of where the boundary is put, and it
    damps only the monopole -- a volume-preserving shape oscillation has a far field that decays
    faster and is left alone, which is what the physics requires.

    The correction is first order in :math:`1/c`, the same order as the Keller-Miksis equation, and
    it does not shift the frequency at that order. It does **not** make the bulk compressible: there
    is still no wave in the domain, only the correct energy loss through its boundary.

    .. note::
        **Keep the outer boundary acoustically compact**, i.e. :math:`\omega R_\text{out}/c\lesssim0.1`.
        The interior is solved as a Laplace problem, so it responds instantaneously while the
        outgoing-wave term at :math:`R_\text{out}` carries a lag :math:`R_\text{out}/c`. That lag
        suppresses the damping by :math:`1/\left(1+(\omega R_\text{out}/c)^{2}\right)`. Measured on a
        free bubble against the exact linear radiation damping
        :math:`\beta=\omega_0^{2}R_0/(2c)`:

        =====================  ==========================
        :math:`\omega R_\text{out}/c`  measured / exact
        =====================  ==========================
        0.04                   1.002
        0.06                   1.002
        0.10                   0.995 - 0.998
        0.20                   0.973
        0.41                   0.876
        0.82                   0.614
        =====================  ==========================

        This costs nothing in practice: the monopole condition itself is already exact at
        :math:`R_\text{out}=3R_0`, so the boundary can simply be brought in. For a 50 um bubble in
        water driven at 55 kHz, anything up to about :math:`R_\text{out}=8R_0` stays within 1 %.
    """
    def __init__(self,phi:ExpressionOrNum=0,origin:ExpressionOrNum=vector(0),speed_of_sound:ExpressionNumOrNone=None):
        super().__init__()
        self.far_phi_value=phi
        self.origin=origin
        self.speed_of_sound=speed_of_sound

    def define_residuals(self):        
        phi,phi_test=self.get_phi_and_test()
        n=var("normal")
        d=var("coordinate")-self.origin                
        self.add_residual(weak( (phi - self.far_phi_value) * dot(n,d)/dot(d,d),phi_test))
        if self.speed_of_sound is not None:
            # The bulk residual is +(grad(phi),grad(psi)), whose integration by parts leaves
            # +<n.grad(phi),psi>, so adding +<dt(phi)/c,psi> imposes n.grad(phi) = ... - dt(phi)/c.
            self.add_residual(weak(partial_t(phi)/self.speed_of_sound,phi_test))
        
        

class _PotentialFlowFreeInterfaceBase(_PotentialFlowInterfaceEquations):
    def __init__(self,*,additional_pressure:ExpressionOrNum=0,surface_tension:ExpressionNumOrNone=None,curvature_sign:int=-1,total_mass_transfer_rate:ExpressionOrNum | None=None):
        super().__init__()
        self.additional_pressure=additional_pressure
        self.sigma=surface_tension
        self.curvature_sign=curvature_sign
        self.total_mass_transfer_rate=total_mass_transfer_rate

    def get_potential_flow_on_opposite_side(self)->PotentialFlow | None:
        if self.get_opposite_parent_domain(raise_error_if_none=False) is not None:
            oppot_eqs=self.get_opposite_parent_equations(PotentialFlow)
            if oppot_eqs is None:
                return None
            oppot=oppot_eqs.get_equation_of_type(PotentialFlow,always_as_list=True)
            if len(oppot)>0:
                if len(oppot)>1:
                    raise RuntimeError("Multiple PotentialFlow equations found in the opposite domain")
                result=oppot[0]
                assert isinstance(result,PotentialFlow)
                return result
            else:
                return None
        else:
            return None

    def define_fields(self):
        potflow=self.get_potential_flow()
        if self.sigma is not None:
            self.define_vector_field("_proj_normal",potflow.space)
            curvspace="C2"
            self.define_scalar_field("_curvature",curvspace,scale=1/scale_factor("spatial"),testscale=scale_factor("spatial"))        
        if self.get_potential_flow_on_opposite_side():
            self.define_scalar_field("_lagr_un_connect",potflow.space,testscale=1/scale_factor("velocity"),scale=1/scale_factor("temporal"))

    def define_residuals(self):
        n=var("normal")
        if self.sigma is not None:            
            pn,pn_test=var_and_test("_proj_normal")
            curv,curv_test=var_and_test("_curvature")
            self.add_weak(pn-n,pn_test)
            self.add_weak(curv-self.curvature_sign*div(pn),curv_test)
        opposite=self.get_potential_flow_on_opposite_side()
        if opposite:
            l,ltest=var_and_test("_lagr_un_connect")
            phiI,phiItest=self.get_phi_and_test(bulk=True)
            phiO,phiOtest=var_and_test(opposite.potential_name,domain="|..")
            #self.add_weak(l*scale_factor("spatial"),ltest)
            self.add_weak(dot(n,grad(phiI)-grad(phiO)),ltest)
            if self.total_mass_transfer_rate:
                pf=self.get_potential_flow()
                if pf.rho is None or opposite.rho is None:
                    raise RuntimeError("Requires mass_density to be set in both PotentialFlow domains for mass transfer")
                self.add_weak(-self.total_mass_transfer_rate*(1/pf.rho-1/opposite.rho),ltest)
            self.add_weak(l,phiItest)
            self.add_weak(-l,phiOtest)

    def get_laplace_pressure(self):
        if self.sigma is None:
            return 0
        else:
            return var("_curvature")*self.sigma
        
    def get_kinematic_boundary_condition(self,vectorial:bool=False):
        n=var("normal")
        x=var("mesh")
        potflow=self.get_potential_flow()
        if potflow.rho is None:
            raise RuntimeError("Requires mass_density to be set in the PotentialFlow")
        u=grad(var(potflow.potential_name,domain=".."))
        j=self.total_mass_transfer_rate if self.total_mass_transfer_rate is not None else 0
        if vectorial:
            return mesh_velocity()-u+j/potflow.rho*n # (xdot-u)=0
        else:
            return dot(n,mesh_velocity()-u)+j/potflow.rho # n*(xdot-u)=0

    def get_kinematic_boundary_condition_flux(self)->Expression:
        """The normal flux the kinematic condition prescribes, i.e. ``n.grad(phi)`` = this.

        Same statement as :py:meth:`get_kinematic_boundary_condition`, only solved for the flux, so
        that it can be handed to the potential equation as its natural boundary condition instead of
        being enforced with a Lagrange multiplier of its own."""
        potflow=self.get_potential_flow()
        rho=potflow.rho
        j=self.total_mass_transfer_rate if self.total_mass_transfer_rate is not None else 0
        res=dot(var("normal"),mesh_velocity())
        if j:
            if rho is None:
                raise RuntimeError("Requires mass_density to be set in the PotentialFlow for mass transfer")
            res=res+j/rho
        return convert_to_expression(res)

    def get_dynamic_boundary_condition(self):
        potflow=self.get_potential_flow()
        if potflow.rho is None:
            raise RuntimeError("Requires mass_density to be set in the PotentialFlow")
        phi=var(potflow.potential_name)
        phiB=var(potflow.potential_name,domain="..")
        # The unsteady Bernoulli equation at a node that moves with xdot is exactly
        #     dphi/dt|_nodal - xdot.grad(phi) + 1/2|grad(phi)|^2 + p/rho = 0 ,
        # written here with the BULK gradient, since the interface-restricted grad() would only be
        # the surface one and would drop the normal part. This used to be specialised to a fully
        # Lagrangian interface (xdot = grad(phi)), where the first two terms collapse to
        # -1/2|grad(phi)|^2; that is only correct if the mesh follows the flow TANGENTIALLY too,
        # which no mesh smoother does. The general form costs one dot product and is right for any
        # tangential mesh motion, which is what removes the need for a separate vectorial-kinematic
        # variant of this class.
        inertia=partial_t(phi,ALE=False)-dot(mesh_velocity(),grad(phiB))+1/2*dot(grad(phiB),grad(phiB))
        pL=self.get_laplace_pressure()
        # TODO: Viscosity
        traction=-(-pL+self.additional_pressure)
        if potflow.dynamic_viscosity is not None:
            n=var("normal")
            u=potflow.get_interface_velocity_for_viscous_at_interfaces(domain="..")
            assert u is not None
            traction-=2*potflow.dynamic_viscosity*dot(n,matproduct(sym(grad(u)),n))
        opposite=self.get_potential_flow_on_opposite_side()
        if opposite:
            if opposite.rho is None:
                raise RuntimeError("Requires mass_density to be set in the opposite side's PotentialFlow")
            phiO=var(opposite.potential_name,domain="|..")
            inertia-=(partial_t(phiO,ALE=False)-dot(mesh_velocity(),grad(phiO))
                      +1/2*dot(grad(phiO),grad(phiO)))*opposite.rho/potflow.rho
            if opposite.dynamic_viscosity is not None:
                n=var("normal")
                u=opposite.get_interface_velocity_for_viscous_at_interfaces(domain="|..")
                assert u is not None
                traction+=2*opposite.dynamic_viscosity*dot(n,matproduct(sym(grad(u)),n))*(opposite.rho/potflow.rho)
        return inertia-traction/potflow.rho


class PotentialFlowFreeInterface(_PotentialFlowFreeInterfaceBase):
    r"""
    Free surface of a potential flow: the kinematic and the dynamic boundary condition.

    The kinematic condition is the natural (Neumann) condition of the potential equation and needs
    no Lagrange multiplier of its own,

    .. math:: \vec{n}\cdot\nabla\phi = \vec{n}\cdot\dot{\vec{x}} + \frac{j}{\rho}\,,

    with :math:`j` the mass transfer rate. The dynamic condition is the unsteady Bernoulli equation
    at the (moving) interface node,

    .. math:: \partial_t\phi\big|_\text{nodal} - \dot{\vec{x}}\cdot\nabla\phi
              + \tfrac{1}{2}\left|\nabla\phi\right|^{2} + \frac{p}{\rho} = 0\,,\qquad
              p = p_\text{add} + \sigma\nabla_S\!\cdot\!\vec{n} - 2\mu\,\vec{n}\cdot\mathbf{D}\vec{n}\,,

    imposed by a Lagrange multiplier that moves the mesh along its normal. Written this way it is
    valid for *any* tangential mesh motion; it does not assume the interface nodes follow the flow.

    .. warning::
        **Where the interface ends -- a symmetry plane, an axis, a substrate -- add a**
        :py:class:`PotentialFlowInterfaceEnd`. Nothing else tells the end node what angle the
        interface makes there, and the one-sided curvature it then invents makes the whole problem
        wrong and unstable: on a 2d capillary wave the physical modes disappear from the spectrum
        entirely and a growing mode appears whose rate *increases* under refinement (Re = 0.005,
        0.013, 0.049 at 8, 16 and 32 elements); on an axisymmetric drop the growth rate is 9.8. With
        the end condition the same problems reproduce the exact capillary-wave dispersion to 0.005 %
        and Lamb's drop frequencies to 1e-6, and are neutrally stable. An interface with no ends at
        all (periodic, or a closed drop whose meridian ends on the axis, where the point measure
        :math:`2\pi r` vanishes) needs nothing and was always correct.

    Args:
        additional_pressure: The external pressure :math:`p_\text{add}` acting on the interface.
        surface_tension: The surface tension. ``None`` switches the Laplace pressure off entirely.
        curvature_sign: Sign convention of the reconstructed ``_curvature`` field, which is
            :math:`-\nabla_S\!\cdot\!\vec{n}` by default.
        total_mass_transfer_rate: Mass leaving the domain through the interface, per area and time.
    """
    def __init__(self, *, additional_pressure: ExpressionOrNum = 0, surface_tension: ExpressionNumOrNone = None, curvature_sign: int = -1,total_mass_transfer_rate:ExpressionOrNum | None=None):
        super().__init__(additional_pressure=additional_pressure, surface_tension=surface_tension, curvature_sign=curvature_sign,total_mass_transfer_rate=total_mass_transfer_rate)

    def define_fields(self):
        potflow=self.get_potential_flow()
        self.define_scalar_field("_lagr_dynbc",potflow.space,scale=1/test_scale_factor("mesh"), testscale=scale_factor("temporal")/scale_factor(potflow.potential_name))
        return super().define_fields()

    def define_residuals(self):
        potflow=self.get_potential_flow()
        _x,xtest=var_and_test("mesh")
        _phi,phitest=var_and_test(potflow.potential_name)
        n=var("normal")
        ldyn,ldyn_test=var_and_test("_lagr_dynbc")

        dyn_bc=self.get_dynamic_boundary_condition()
        self.add_weak(dyn_bc,ldyn_test)
        self.add_weak(ldyn*n,xtest)

        # The kinematic condition as the natural BC of the potential equation. It used to be written
        # out as -dot(mesh_velocity(),n) here, which silently dropped the j/rho of an evaporating or
        # condensing interface although the constructor accepts it.
        self.add_weak(-self.get_kinematic_boundary_condition_flux(),phitest)
        return super().define_residuals()

    def before_assigning_equations_postorder(self, mesh: AnyMesh):
        assert isinstance(mesh,InterfaceMesh)
        self.pin_redundant_lagrange_multipliers(mesh,"_lagr_dynbc","mesh")


class PotentialFlowInterfaceEnd(InterfaceEquations):
    r"""
    The contact angle at an end of a potential-flow free surface -- where it meets a wall, a
    symmetry plane or a substrate. Attach it to that end::

        eqs += PotentialFlowFreeInterface(surface_tension=sigma) @ "interface"
        eqs += PotentialFlowInterfaceEnd(interface_normal=vector(0,1)) @ "interface/wall"

    It prescribes the interface normal at the end node, which is the information the curvature
    reconstruction is otherwise missing there. Without it the end node invents a one-sided curvature
    and the whole problem is wrong and unstable -- see the warning in
    :py:class:`PotentialFlowFreeInterface`.

    Note that the normal has to be *prescribed*: it cannot be derived from the interface's own
    normal at that node, which is exactly the quantity being constrained. Deriving it (for instance
    through the ``bac-cab`` rule that :py:class:`~pyoomph.equations.navier_stokes.NavierStokesContactAngle`
    can use for its wall tangent) gives an identity, i.e. no condition at all, and leaves the
    original failure in place.

    Args:
        interface_normal: The interface normal at the end, pointing out of the liquid. The direct
            and unambiguous way to state the condition.
        wall_normal: Alternatively, the normal of the wall, pointing out of the domain, together
            with ``wall_tangent`` and ``contact_angle``. The end normal is then
            :math:`\cos\theta\,\vec{n}_\text{w}-\sin\theta\,\vec{t}_\text{w}`, which for the default
            :math:`\theta=90^\circ` is simply :math:`-\vec{t}_\text{w}`.
        wall_tangent: Direction along the wall pointing *into* the domain. Required with
            ``wall_normal``; it is what fixes which way the interface turns and there is no way to
            infer it here.
        contact_angle: The angle between the interface and the wall, measured inside the liquid.
            Defaults to 90 degrees, i.e. the symmetry-plane / free-slip case.
    """
    def __init__(self,*,interface_normal:"Expression | None"=None,
                 wall_normal:"Expression | None"=None,wall_tangent:"Expression | None"=None,
                 contact_angle:ExpressionOrNum=90*degree):
        super().__init__()
        if interface_normal is None:
            if wall_normal is None or wall_tangent is None:
                raise ValueError("PotentialFlowInterfaceEnd needs either interface_normal, or "
                                 "wall_normal together with wall_tangent (the direction along the "
                                 "wall pointing into the domain). The latter cannot be guessed: "
                                 "deriving it from the interface normal makes the condition an "
                                 "identity and imposes nothing.")
            nw=wall_normal/square_root(dot(wall_normal,wall_normal))
            tw=wall_tangent/square_root(dot(wall_tangent,wall_tangent))
            interface_normal=cos(contact_angle)*nw-sin(contact_angle)*tw
        self.interface_normal=interface_normal

    def define_residuals(self):
        n=self.interface_normal
        n=n/square_root(dot(n,n))
        for i,d in enumerate(["x","y","z"][0:self.get_nodal_dimension()]):
            self.set_Dirichlet_condition("_proj_normal_"+d,n[i])


def _deprecated_free_interface(name:str,what:str):
    """The numbered free-surface variants, kept so that existing scripts keep running.

    There used to be five public classes here, but only two distinct schemes: variants 1 and the old
    ``PotentialFlowFreeInterface(new_version=False)`` were bit-for-bit identical, and variants 2, 3
    and ``new_version=True`` likewise (3 differed only through its vectorial kinematic condition,
    which a purely radial test cannot see). Both schemes reproduce Rayleigh-Plesset, the one kept
    about twelve times more accurately at equal cost and with one Lagrange multiplier fewer. The
    vectorial kinematic condition of variant 3 existed to justify the fully-Lagrangian form of the
    dynamic boundary condition; that form has been generalised, so it is no longer needed.
    """
    class _Deprecated(PotentialFlowFreeInterface):
        def __init__(self,**kwargs:Any):
            warnings.warn(name+" is deprecated: it "+what+" Use PotentialFlowFreeInterface "
                          "instead, and add a PotentialFlowInterfaceEnd wherever the interface ends.",
                          DeprecationWarning,stacklevel=2)
            super().__init__(**kwargs)
    _Deprecated.__name__=name
    _Deprecated.__qualname__=name
    _Deprecated.__doc__="Deprecated alias of :py:class:`PotentialFlowFreeInterface`."
    return _Deprecated


PotentialFlowFreeInterface1=_deprecated_free_interface(
    "PotentialFlowFreeInterface1","is the scheme PotentialFlowFreeInterface now implements, except "
    "that it silently dropped total_mass_transfer_rate from its kinematic boundary condition.")
PotentialFlowFreeInterface2=_deprecated_free_interface(
    "PotentialFlowFreeInterface2","imposes the kinematic condition with a Lagrange multiplier "
    "rather than as the natural boundary condition, which costs an extra field and was measured to "
    "be an order of magnitude less accurate on Rayleigh-Plesset.")
PotentialFlowFreeInterface3=_deprecated_free_interface(
    "PotentialFlowFreeInterface3","forces a fully Lagrangian interface, which tangles the mesh. It "
    "only existed because the dynamic boundary condition assumed one; it no longer does.")


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
