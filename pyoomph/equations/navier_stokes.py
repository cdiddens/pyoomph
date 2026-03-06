#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 


from .. import WeakContribution, GlobalLagrangeMultiplier
from ..generic import Equations, InterfaceEquations
from .generic import get_interface_field_connection_space, TestScaling, Scaling
from ..meshes.bcs import BoundaryCondition,DirichletBC
from ..meshes.mesh import AnyMesh, InterfaceMesh
from ..expressions import *  # Import grad et al
from ..expressions.units import degree


if TYPE_CHECKING:
    from _pyoomph import Node
    from ..solvers.generic import GenericEigenSolver
    from ..generic.codegen import EquationTree
    from ..materials.generic import AnyFluidProperties,PureLiquidProperties
    from ..generic.problem import Problem



###################################

# PRESSURE FIXATIONS ##
# if you have a Navier-Stokes domain with pure Dirichlet-BCs,
# you need exactly one degree of pressure to be pinned to remove the null-space
# These will take care of it by either pinning one nodal pressure dof (TH) or one elemental pressure dof (CR)
# The corresponding BC can be created via StokesElement.create_pressure_fixation()

class PressureFixationTaylorHood(BoundaryCondition):
    def __init__(self, pname,value:Optional[float]):
        super().__init__()
        self.value:Optional[float] = value
        self.node:Optional["Node"] = None
        self.pname=pname
        

    def setup(self):
        assert self.mesh is not None
        self.pindex = self.mesh.element_pt(0).get_code_instance().get_nodal_field_index(self.pname)
        if self.pindex < 0:
            allfields = self.mesh.element_pt(0).get_code_instance().get_nodal_field_indices()
            for k,v in allfields.items():
                print(k,v,self.mesh.element_pt(0).get_code_instance().get_nodal_field_index(k))
            raise RuntimeError("Missing nodal data for 'pressure'. Found only: "+str(allfields))
        if self.node is None:
            self.node = self.mesh.element_pt(0).node_pt(0)  # Is definitely a C1 node
            ps=[self.node.x(i) for i in range(self.node.ndim())]
            print("Got Node at " + str(ps))

    def apply(self):
        assert self.node is not None
        self.node.pin(self.pindex)
        if self.value is not None:
            self.node.set_value(self.pindex, self.value)
        print("PINNING some pressure with value",self.value)

    def _before_eigen_solve(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver",angular_m:Optional[int]) -> bool:
        if angular_m is not None:
            raise RuntimeError("Do not use pressure_fixation with angular eigensolving. Use [Navier]StokesEquation(...).with_pressure_integral_constraint(problem) instead...")
        return False



class PressureFixationScottVogelius(BoundaryCondition):
    def __init__(self, pname,value:Optional[float]):
        super().__init__()
        self.value = value
        self.pname= pname
    
    def apply(self):
        assert self.mesh is not None
        for e in self.mesh.elements():
            fl=e.get_field_data_list(self.pname,False)
            for d,ind in fl:
                d.unpin(ind)
        fl=self.mesh.element_pt(0).get_field_data_list(self.pname,False)[0]
        fl[0].pin(fl[1])
        if self.value is not None:
            fl[0].set_value(fl[1], self.value)

    def _before_eigen_solve(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver",angular_m:Optional[float]) -> bool:
        if angular_m is not None:
            raise RuntimeError("Do not use pressure_fixation with angular eigensolving. Use [Navier]StokesEquation(...).with_pressure_integral_constraint(problem) instead...")
        return False

class PressureFixationCrouzeixRaviart(BoundaryCondition):
    def __init__(self, pname, value:Optional[float]):
        super().__init__()
        self.value = value
        self.pname=pname

    def setup(self):
        assert self.mesh is not None
        self.pindex = self.mesh.element_pt(0).get_code_instance().get_discontinuous_field_index(self.pname)
        if self.pindex < 0:
            raise RuntimeError("Missing internal data for 'pressure'")

    def apply(self):
        assert self.mesh is not None
        for ei in range(self.mesh.nelement()):
            self.mesh.element_pt(ei).internal_data_pt(self.pindex).unpin(0)
        self.mesh.element_pt(0).internal_data_pt(self.pindex).pin(0)
        if self.value is not None:
            self.mesh.element_pt(0).internal_data_pt(self.pindex).set_value(0, self.value)

    def _before_eigen_solve(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver",angular_m:Optional[float]) -> bool:
        if angular_m is not None:
            raise RuntimeError("Do not use pressure_fixation with angular eigensolving. Use [Navier]StokesEquation(...).with_pressure_integral_constraint(problem) instead...")
        return False


###################################

class PFEMOptions:
    def __init__(self,*,active:bool=True,first_order_system:bool=False,direct_position_update:bool=False) -> None:
        self.active=active
        self.first_order_system=first_order_system
        self.direct_position_update=direct_position_update

###################################

class StokesEquations(Equations):
    """
    .. _StokesEquations:


    Represents the Stokes equations, defined by the second-order partial differential equations (PDEs):

    .. math:: \\partial_t \\rho + \\nabla \\cdot (\\rho \\vec{u}) = 0
    .. math:: \\nabla \\cdot [-\\nabla p \\vec{\\vec{I}} + \\mu (\\nabla \\vec{u} + (\\nabla \\vec{u})^\\text{T})] + f = 0 \\,

    where :math:`p` is the pressure, :math:`\\vec{\\vec{I}}` is the identity matrix, :math:`\\vec{u}` the velocity, :math:`\\mu` the dynamic viscosity, :math:`\\rho` is the mass density and :math:`f` the bulk force.
    
    In the weak form, the equations are written as:

    .. math:: (\\partial_t \\rho, q) + (\\rho \\vec{u}, \\nabla q) = 0
    .. math:: - (-p \\vec{\\vec{I}} + \\mu [\\nabla \\vec{u} + (\\nabla \\vec{u})^\\text{T}], \\nabla \\vec{v} + (\\nabla \\vec{v})^\\text{T}) + (f,\\vec{v}) + \\langle \\vec{n} \\cdot [-p I + \\mu (\\nabla \\vec{u} + (\\nabla \\vec{u})^\\text{T})], \\vec{v} \\rangle = 0 \\,

    where :math:`\\vec{v}` and :math:`q` are test functions of the velocity and pressure, respectively.

    Args:
        dynamic_viscosity (ExpressionOrNum, optional): Dynamic viscosity of the fluid (if `fluid_props==None`). Defaults to 1.0.
        mode (Literal["TH","CR"], optional): Use Taylor-Hood ("TH") or Crouzeix-Raviart ("CR") elements. Defaults to "TH".
        bulkforce (ExpressionNumOrNone, optional): Bulk force term. Defaults to None.
        fluid_props (Optional[AnyFluidProperties], optional): Fluid properties. Defaults to None, meaning mass density and dynamic viscosity will be set by the "mass_density" and "dynamic_viscosity" arguments.
        gravity (ExpressionNumOrNone, optional): Gravity term in vector form. Defaults to None.
        boussinesq (bool, optional): Use Boussinesq approximation, i.e. use constant density in continuity equation. Defaults to False.
        mass_density (ExpressionNumOrNone, optional): Mass density of the fluid (if 'fluid_props==None'). Only used for gravity and the continuity equation. Defaults to None.
        pressure_sign_flip (bool, optional): Reverse the pressure sign to get a symmetric matrix. Defaults to False.
        momentum_scheme (TimeSteppingScheme, optional): Time-stepping scheme for the momentum equation. Defaults to "BDF2".
        continuity_scheme (TimeSteppingScheme, optional): Time-stepping scheme for the continuity equation. Defaults to "BDF2".
        wrong_strain (bool, optional): Use mu*grad(u) instead of 2*mu*sym(grad(u)) for strain. Defaults to False.
        pressure_factor (ExpressionOrNum, optional): Multiplicative factor for the pressure in the momentum equation. Defaults to 1.
        PFEM (Union[PFEMOptions,bool], optional): Options for Particle Finite Element Method (PFEM). Defaults to False.
        stress_tensor (ExpressionNumOrNone, optional): Custom stress tensor. Defaults to None.
        velocity_name (str, optional): Name of the velocity field. Defaults to "velocity".
        pressure_name (str, optional): Name of the pressure field. Defaults to "pressure".        
        DG_alpha (ExpressionNumOrNone, optional): If using Discontinuous Galerkin discretisation, set penalty coefficient alpha for jump terms of the stress tensor. Defaults to None.
        symmetric_test_function (Union[Literal['auto'],bool], optional): Use symmetric test functions for the momentum equation. Defaults to 'auto'.
        pressure_test_scaling_factor (float, optional): Multiplicative scaling factor for the pressure test function. Defaults to 1.0.
        hele_shaw_thickness (ExpressionOrNum, optional): Adds a Hele-Shaw drag term to the bulk force i.e -12*mu*u/delta**2, with the given thickness as parameter. Defaults to None.
    """
    def __init__(self, *, dynamic_viscosity:ExpressionOrNum=1.0, mode:Literal["TH","CR","SV","C1","D2D1","D1D0","D2TBD1","mini","C2DL"]="TH", bulkforce:ExpressionNumOrNone=None, fluid_props:Optional["AnyFluidProperties"]=None, gravity:ExpressionNumOrNone=None, boussinesq:bool=False, mass_density:ExpressionNumOrNone=None,
                 pressure_sign_flip:bool=False,momentum_scheme:TimeSteppingScheme="BDF2",continuity_scheme:TimeSteppingScheme="BDF2",wrong_strain:bool=False,pressure_factor:ExpressionOrNum=1, PFEM:Union[PFEMOptions,bool]=False, stress_tensor:ExpressionNumOrNone=None,velocity_name="velocity",pressure_name="pressure",DG_alpha:ExpressionNumOrNone=None,symmetric_test_function:Union[Literal['auto'],bool]='auto',pressure_test_scaling_factor:float=1, hele_shaw_thickness:ExpressionOrNum=None):
        super().__init__()
        self.gravity = gravity  # Some gravity direction, i.e. g*<unit vector of direction>
        if mode not in {"CR","TH","C1","C2","SV","D2TBD1","D2D1","D1D0","mini","C2DL"}:
            raise ValueError(
                "(Navier-)Stokes equations argument 'mode' needs to be 'CR' for Crouzeix-Raviart element or 'TH' for Taylor-Hood equations. Experimentally also 'C1' and 'C2' are possible. If the mesh is constructed correctly, 'SV' for Scott-Vogelius also works. 'mini' elements are only possible on triangles. Also, discontinuous variants 'D2D1' and 'D1D0' are currently in development, i.e. experimental")
        self.mode:Literal["TH","CR","C1","C2","SV","D2D1","D1D0","mini","C2DL"] = mode
        self.requires_interior_facet_terms=self.mode in {"D2D1","D1D0","D2TBD1"}
        self.DG_alpha=DG_alpha
        self.pressure_test_scaling_factor=pressure_test_scaling_factor
        if self.mode in {"D2D1","D1D0"}:
            if self.DG_alpha is None:
                raise RuntimeError(f"Must set DG_alpha if mode=='{self.mode}'")
            if symmetric_test_function=="auto":
                symmetric_test_function=True
        else:
            if symmetric_test_function=="auto":
                symmetric_test_function=False
        self.symmetric_test_function:bool=symmetric_test_function
        
        self.boussinesq = boussinesq  # If set, we only solve div(u)=0, else we solve div(u)=-1/rho*(partial_t(rho)+u*grad(rho))
        if fluid_props is not None:
            self.fluid_props = fluid_props
            self.dynamic_viscosity = fluid_props.dynamic_viscosity
            self.mass_density = fluid_props.mass_density
        else:
            self.fluid_props = None
            self.dynamic_viscosity = dynamic_viscosity
            self.mass_density = mass_density

        if hele_shaw_thickness is not None:
            hsdamp = -12 * self.dynamic_viscosity * var("velocity") / hele_shaw_thickness ** 2
            bulkforce = hsdamp if bulkforce is None else bulkforce+hsdamp
        self.bulkforce = bulkforce  # Some arbitrary bulk-force vector
        
        self.pressure_sign_flip=pressure_sign_flip
        self.momentum_scheme:TimeSteppingScheme=momentum_scheme        
        self.continuity_scheme:TimeSteppingScheme=continuity_scheme
        if PFEM:
            if PFEM==True:
                self.PFEM_options=PFEMOptions()
            else:
                self.PFEM_options=PFEM
            if not self.PFEM_options.first_order_system:
                self.momentum_scheme="Newmark2"
                self.continuity_scheme="Newmark2"
        else:
            self.PFEM_options=False
        self.wrong_strain=wrong_strain
        self.pressure_factor=pressure_factor
        self.stress_tensor=stress_tensor
        self.velocity_name=velocity_name
        self.pressure_name=pressure_name

    def get_velocity_space_from_mode(self,for_interface=False):
        velospace={"C1":"C1","CR":"C2TB","TH":"C2","SV":"C2","D2D1":"D2","D1D0":"D1","D2TBD1":"D2TB","mini":"C1TB","C2DL":"C2","C2":"C2"}
        res=velospace[self.mode]
        if for_interface:
            if res=="C2TB":
                res="C2"
            elif res=="C1TB":
                res="C1"
            elif res[0]=="D":
                raise RuntimeError("Discont here")
        return res
    
    def get_pressure_space_from_mode(self):
        pspace={"C1":"C1","CR":"DL","TH":"C1","SV":"D1","D2D1":"D1","D1D0":"D0","D2TBD1":"D1","mini":"C1","C2DL":"DL","C2":"C2"}
        return pspace[self.mode]

    def define_fields(self):
        vspace=self.get_velocity_space_from_mode()
        pspace=self.get_pressure_space_from_mode()

        if self.PFEM_options and self.PFEM_options.active:
            self.activate_coordinates_as_dofs(coordinate_space=vspace)            
            if self.PFEM_options.first_order_system:
                self.define_vector_field(self.velocity_name, vspace)
        else:
            self.define_vector_field(self.velocity_name, vspace)
        self.define_scalar_field(self.pressure_name, pspace)

    def define_scaling(self):
        U = self.get_scaling(self.velocity_name)
        X = self.get_scaling("spatial")
        P = self.get_scaling(self.pressure_name)
        self.set_test_scaling(**{self.velocity_name:X / P})
        if self.PFEM_options and self.PFEM_options.active:
            if self.PFEM_options.first_order_system:
                self.set_test_scaling(mesh_x=scale_factor("temporal")/scale_factor(self.velocity_name))
                self.set_test_scaling(mesh_y=scale_factor("temporal")/scale_factor(self.velocity_name))
                self.set_test_scaling(mesh_z=scale_factor("temporal")/scale_factor(self.velocity_name))
            else:
                self.set_test_scaling(mesh_x=self.velocity_name)
                self.set_test_scaling(mesh_y=self.velocity_name)
                self.set_test_scaling(mesh_z=self.velocity_name)
        self.set_test_scaling(pressure=X / U*self.pressure_test_scaling_factor)
        self.add_named_numerical_factor(p_in_momentum_eq=scale_factor(self.pressure_name)*test_scale_factor(self.velocity_name)/scale_factor("spatial")*self.pressure_test_scaling_factor)
        self.add_named_numerical_factor(div_u__in_conti_eq=scale_factor(self.velocity_name) * test_scale_factor(self.pressure_name) / scale_factor("spatial"))

    def define_stress_tensor(self):
        u = var(self.velocity_name)
        p = var(self.pressure_name)

        visc = self.dynamic_viscosity
        if visc is None:
            raise RuntimeError("viscosity not set")
        if self.wrong_strain:
            strain=grad(u)/2
        else:
            strain = sym(grad(u))  # 1/2*(grad(u)+grad(u)^t)
        
        # Newtonian fluid
        if self.stress_tensor==None:
            stress_tensor = 2 * visc * strain - identity_matrix() * self.pressure_factor*p*(-1 if self.pressure_sign_flip else 1)
        else:
            stress_tensor = self.stress_tensor
        return stress_tensor

    def define_residuals(self):
        if self.PFEM_options and self.PFEM_options.active:
            if not self.PFEM_options.first_order_system:
                x, u_test = var_and_test("mesh")
                u=mesh_velocity(scheme=self.momentum_scheme)
                vectcomps:List[str]=[]
                for i,direct in enumerate((["x","y","z"])[:self.get_nodal_dimension()]):
                    vectcomps.append("velocity_"+direct)
                    self.define_field_by_substitution("velocity_"+direct,mesh_velocity(scheme=self.momentum_scheme)[i],also_on_interface=True)
                    self.define_testfunction_by_substitution("velocity_"+direct,testfunction("mesh_"+direct),also_on_interface=True)
                self.define_testfunction_by_substitution(self.velocity_name,vector(*[testfunction(f)/test_scale_factor(self.velocity_name) for f in vectcomps]),also_on_interface=True)
                self.define_testfunction_by_substitution("mesh",vector(*[testfunction(f)/test_scale_factor(self.velocity_name) for f in vectcomps]),also_on_interface=True)
                self.define_field_by_substitution(self.velocity_name,vector(*[var(f)/scale_factor(self.velocity_name) for f in vectcomps]),also_on_interface=True)
                self.add_local_function(self.velocity_name,var(self.velocity_name))
                self._get_combined_element()._vectorfields[self.velocity_name]=vectcomps
            else:
                u, u_test = var_and_test(self.velocity_name)
                if self.PFEM_options.direct_position_update:
                    self.add_residual(weak(var("mesh")-evaluate_in_past(var("mesh"))-var(self.velocity_name)*timestepper_weight(1,0,"BDF2"),testfunction("mesh")))
                else:
                    self.add_residual(weak(partial_t(var("mesh"))-var(self.velocity_name),testfunction("mesh")))
        else:
            u, u_test = var_and_test(self.velocity_name)
        p, p_test = var_and_test(self.pressure_name)

        # Get stress tensor
        stress_tensor = self.define_stress_tensor()

        # Residuals
        if self.symmetric_test_function:
            Dv=sym(grad(u_test))
        else:
            Dv=grad(u_test)
        self.add_residual(weak(time_scheme(self.momentum_scheme,stress_tensor), Dv))  # total stress
        self.add_residual(weak(time_scheme(self.continuity_scheme,div(u)), p_test))  # Incompressibility
        if not self.boussinesq and self.mass_density is not None:
            rho = self.mass_density
            self.add_residual(weak(time_scheme(self.continuity_scheme,partial_t(rho, ALE=False) / rho + dot(u, grad(rho)) / rho), p_test))  # Incompressibility

        if self.bulkforce is not None:
            self.add_residual(-weak(time_scheme(self.momentum_scheme,self.bulkforce), u_test))  # bulk force
        if self.gravity is not None:
            if self.mass_density is None:
                raise RuntimeError("Must set mass_density if using gravity")
            self.add_residual(-weak(time_scheme(self.momentum_scheme,self.gravity * self.mass_density), u_test))  # gravity force force
        
        if self.requires_interior_facet_terms:
            n,h=var(["normal","element_length_h"])
            # optionally, at_facet=True can be added to all jumps/averages that don't have grad in their argument
            Du=grad(u)/2 if self.wrong_strain else sym(grad(u))
            Dv=grad(u_test) if not self.symmetric_test_function else sym(grad(u_test))
            facet_terms=weak(-matproduct(2*self.dynamic_viscosity*avg(Du),n),jump(u_test)) 
            facet_terms+=weak(-jump(u),matproduct(avg(Dv),n))
            facet_terms+=weak(self.DG_alpha*2**(2 if self.mode=="D2D1" else 1)/h*jump(u),jump(u_test))
            facet_terms+=weak(dot(jump(u),n),avg(p_test))  # okay
            facet_terms+=weak(avg(p),dot(jump(u_test),n))

            self.add_interior_facet_residual(facet_terms)

    # In case of complete Dirichlet velocity conditions, we need to fix a single dof of pressure
    # A single node is selected in case of Taylor hood, otherwise a single element is selected
    def create_pressure_fixation(self, *, value:Optional[float]=None)->Union[PressureFixationTaylorHood,PressureFixationCrouzeixRaviart,PressureFixationScottVogelius]:
        if self.mode in {"TH","C1","mini"}:
            return PressureFixationTaylorHood(self.pressure_name,value)
        elif self.mode == "CR":
            return PressureFixationCrouzeixRaviart(self.pressure_name,value)
        elif self.mode == "SV":
            return PressureFixationScottVogelius(self.pressure_name,value)
        else:
            raise RuntimeError("Cannot add a pressure fixation for this mode")

    # To be used a StokesEquation(...).with_pressure_fixation()
    def with_pressure_fixation(self,*,nondim_p_value:Optional[float]=None) -> Equations:
        """
        Instead of adding ``StokesEquation``, add ``StokesEquation.with_pressure_fixation(...)`` to remove the pressure nullspace in case of pure Dirichlet boundary conditions for the normal flow.
        With this method, a single pressure dof is pinned to remove the nullspace.

        Args:
            problem: The problem where the equations are added.
            integral_value: The integral value of the pressure over the domain.
            ode_domain_name: Domain name for the Lagrange multiplier enforcing the pressure integral. Defaults to "globals".
            lagrange_name: Name of the global Lagrange multiplier. Defaults to "lagr_intconstr_pressure".
            set_zero_on_normal_mode_eigensolve: Deactivate when solving angular eigenvalue problems. Defaults to True.

        Returns:
            The (Navier-)Stokes equations with the pressure integral constraint.
        """
        fix=self.create_pressure_fixation(value=nondim_p_value)
        return self+fix


    def with_pressure_integral_constraint(self, problem:"Problem", integral_value:ExpressionOrNum=0, *, ode_domain_name:str="globals",lagrange_name:str="lagr_intconstr_pressure", set_zero_on_normal_mode_eigensolve:bool=True) -> Equations:
        """
        Instead of adding ``StokesEquation``, add ``StokesEquation.with_pressure_integral_constraint(...)`` to remove the pressure nullspace in case of pure Dirichlet boundary conditions for the normal flow.
        With this method, the integral of the pressure is constrained to a given value via a global Lagrange multiplier.

        Args:
            problem: The problem where the equations are added.
            integral_value: The integral value of the pressure over the domain.
            ode_domain_name: Domain name for the Lagrange multiplier enforcing the pressure integral. Defaults to "globals".
            lagrange_name: Name of the global Lagrange multiplier. Defaults to "lagr_intconstr_pressure".
            set_zero_on_normal_mode_eigensolve: Deactivate when solving angular eigenvalue problems. Defaults to True.

        Returns:
            The (Navier-)Stokes equations with the pressure integral constraint.
        """
        eq_additions = self

        eq_additions += WeakContribution(var(self.pressure_name), testfunction(lagrange_name, domain=ode_domain_name),dimensional_dx=False)
        eq_additions += WeakContribution(var(lagrange_name, domain=ode_domain_name), testfunction(self.pressure_name),dimensional_dx=False)
        ode_additions = GlobalLagrangeMultiplier(**{lagrange_name:integral_value},set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve)
        ode_additions +=TestScaling(**{lagrange_name:1/scale_factor(self.pressure_name)})
        ode_additions += Scaling(**{lagrange_name: 1 / test_scale_factor(self.pressure_name)})
        problem.add_equations(ode_additions @ ode_domain_name)
        return eq_additions


##################################

class NavierStokesEquations(StokesEquations):   
    """Represents the Navier-Stokes-Equations, defined by the second-order partial differential equations (PDEs):

    .. math:: \\partial_t \\rho + \\nabla \\cdot (\\rho \\vec{u}) = 0 \\,
    .. math:: \\rho (\\partial_t \\vec{u} + \\vec{u} \\cdot \\nabla \\vec{u} ) = \\nabla \\cdot [-\\nabla p \\vec{\\vec{I}} + \\mu (\\nabla \\vec{u} + (\\nabla \\vec{u})^\\text{T})] + f \\,
        
    where :math:`p` is the pressure, :math:`\\vec{\\vec{I}}` is the identity matrix, :math:`\\vec{u}` the velocity, :math:`\\mu` the dynamic viscosity, :math:`\\rho` is the mass density and :math:`f` the bulk force.

    In the weak form, the equations are written as:

    .. math:: (\\partial_t \\rho, q) + (\\rho \\vec{u}, \\nabla q) = 0
    .. math:: \\rho (\\partial_t \\vec{u}, \\vec{v}) + \\rho (\\vec{u} \\cdot \\nabla \\vec{u}, \\vec{v}) + (-p \\vec{\\vec{I}} + \\mu [\\nabla \\vec{u} + (\\nabla \\vec{u})^\\text{T}], \\nabla \\vec{v} + (\\nabla \\vec{v})^\\text{T}) - (f,\\vec{v}) 
    .. math:: - \\langle \\vec{n} \\cdot [-p I + \\mu (\\nabla \\vec{u} + (\\nabla \\vec{u})^\\text{T})], \\vec{v} \\rangle = 0 \\,

    where :math:`\\vec{v}` and :math:`q` are test functions of the velocity and pressure, respectively.

    This class is a subclass of :ref:`StokesEquations <StokesEquations>` and inherits all its arguments.                    
        
    Args:
        dynamic_viscosity (ExpressionOrNum, optional): Dynamic viscosity of the fluid (if `fluid_props==None`). Defaults to 1.0.
        mode (Literal["TH","CR"], optional): Use Taylor-Hood ("TH") or Crouzeix-Raviart ("CR") elements. Defaults to "TH".
        bulkforce (ExpressionNumOrNone, optional): Bulk force term. Defaults to None.
        fluid_props (Optional[AnyFluidProperties], optional): Fluid properties. Defaults to None, meaning mass density and dynamic viscosity will be set by the "mass_density" and "dynamic_viscosity" arguments.
        gravity (ExpressionNumOrNone, optional): Gravity term in vector form. Defaults to None.
        boussinesq (bool, optional): Use Boussinesq approximation, i.e. use constant density in continuity equation. Defaults to False.
        mass_density (ExpressionNumOrNone, optional): Mass density of the fluid (if 'fluid_props==None'). Only used for gravity and the continuity equation. Defaults to None.
        pressure_sign_flip (bool, optional): Reverse the pressure sign to get a symmetric matrix. Defaults to False.
        momentum_scheme (TimeSteppingScheme, optional): Time-stepping scheme for the momentum equation. Defaults to "BDF2".
        continuity_scheme (TimeSteppingScheme, optional): Time-stepping scheme for the continuity equation. Defaults to "BDF2".
        wrong_strain (bool, optional): Use mu*grad(u) instead of 2*mu*sym(grad(u)) for strain. Defaults to False.
        pressure_factor (ExpressionOrNum, optional): Multiplicative factor for the pressure in the momentum equation. Defaults to 1.
        PFEM (Union[PFEMOptions,bool], optional): Options for Particle Finite Element Method (PFEM). Defaults to False.
        stress_tensor (ExpressionNumOrNone, optional): Custom stress tensor. Defaults to None.
        velocity_name (str, optional): Name of the velocity field. Defaults to "velocity".
        pressure_name (str, optional): Name of the pressure field. Defaults to "pressure".        
        DG_alpha (ExpressionNumOrNone, optional): If using Discontinuous Galerkin discretisation, set coefficient alpha for stress tensor. Defaults to None.
        symmetric_test_function (Union[Literal['auto'],bool], optional): Use symmetric test functions for the momentum equation. Defaults to 'auto'.
        dt_factor (ExpressionOrNum, optional): Multiplicative factor to scale or deactivate the time derivative. Defaults to 1.
        nonlinear_factor (ExpressionNumOrNone, optional): Multiplicative factor to scale or deactivate the nonlinearity, i.e. dot(u,grad(u))). Defaults to None.
        wrap_params_in_subexpressions (bool, optional): Wrap parameters in subexpressions using GiNaC. Defaults to True.
    """
                 
        
    def __init__(self, *, dynamic_viscosity:ExpressionOrNum=1.0, mode:Literal["TH","CR","SV"]="TH", mass_density:ExpressionOrNum=1.0, bulkforce:ExpressionNumOrNone=None, fluid_props:Optional["AnyFluidProperties"]=None,
                 dt_factor:ExpressionOrNum=1, nonlinear_factor:ExpressionNumOrNone=None, gravity:ExpressionNumOrNone=None, boussinesq:bool=False,momentum_scheme:TimeSteppingScheme="BDF2",continuity_scheme:TimeSteppingScheme="BDF2",wrong_strain:bool=False,pressure_factor:ExpressionOrNum=1,wrap_params_in_subexpressions:bool=True,PFEM:Union[PFEMOptions,bool]=False, stress_tensor:ExpressionNumOrNone=None,velocity_name="velocity",pressure_name="pressure",symmetric_test_function:Union[Literal['auto'],bool]='auto',pressure_test_scaling_factor:float=1, hele_shaw_thickness:ExpressionOrNum=None):
        super().__init__(dynamic_viscosity=dynamic_viscosity, mode=mode, bulkforce=bulkforce, fluid_props=fluid_props,
                         gravity=gravity, boussinesq=boussinesq,momentum_scheme=momentum_scheme,continuity_scheme=continuity_scheme,wrong_strain=wrong_strain,pressure_factor=pressure_factor,PFEM=PFEM, stress_tensor=stress_tensor,velocity_name=velocity_name,pressure_name=pressure_name,symmetric_test_function=symmetric_test_function,pressure_test_scaling_factor=pressure_test_scaling_factor, hele_shaw_thickness=hele_shaw_thickness)
        if self.fluid_props is not None:
            self.mass_density = self.fluid_props.mass_density
        else:
            self.mass_density = mass_density
        if nonlinear_factor is None: # nonlinear_factor overrides Hele-Shaw
            if hele_shaw_thickness is None:
                nonlinear_factor = 1
            else:
                nonlinear_factor = 6 / 5
        self.dt_factor = dt_factor  # Factors to scale or deactivate the time derivative
        self.nonlinear_factor = nonlinear_factor  # and the nonlinearity
        self.wrap_params_in_subexpressions=wrap_params_in_subexpressions

    def define_scaling(self):
        super(NavierStokesEquations, self).define_scaling()
        self.add_named_numerical_factor(inertia_in_momentum_eq=self.dt_factor*scale_factor("mass_density") * scale_factor(self.velocity_name)*test_scale_factor(self.velocity_name) / scale_factor("temporal"))

    def define_residuals(self):
        super().define_residuals()  # add the Stokes part
        if self.PFEM_options and self.PFEM_options.active and self.PFEM_options.first_order_system:
            x, u_test = var_and_test("mesh")
            u=mesh_velocity(scheme=self.momentum_scheme)
        else:
            u, u_test = var_and_test(self.velocity_name)
        if self.wrap_params_in_subexpressions:
            rho = subexpression(self.mass_density)
        else:
            rho=self.mass_density
        if self.PFEM_options and self.PFEM_options.active and self.PFEM_options.first_order_system:
            if self.dt_factor!=1 or self.nonlinear_factor!=1:
                raise RuntimeError("Can only use entirely Lagrangian mode if dt_factor and nonlinear_factor==1")
            self.add_residual(weak(time_scheme(self.momentum_scheme,rho*partial_t(x,2,ALE=False)), u_test))
            #self.add_residual(weak(time_scheme(self.momentum_scheme,rho*partial_t(u,1,ALE=False)), u_test))
#        elif self.PFEM_options and self.PFEM_options.active and self.PFEM_options.direct_position_update:
 #           raise RuntimeError("TODO")
        else:
            self.add_residual(weak(time_scheme(self.momentum_scheme,rho*material_derivative(u,u,dt_factor=self.dt_factor,advection_factor=self.nonlinear_factor)), u_test))




class NavierStokesNormalTraction(InterfaceEquations):
    required_parent_type = StokesEquations

    def __init__(self, normal_traction:ExpressionOrNum):
        super(NavierStokesNormalTraction, self).__init__()
        self.normal_traction = normal_traction

    def define_residuals(self):
        peqs=self.get_parent_equations(StokesEquations)
        assert isinstance(peqs,StokesEquations)
        _, utest = var_and_test(peqs.velocity_name)
        n = self.get_normal()
        self.add_residual(weak(self.normal_traction, dot(n, utest)))


class NavierStokesFreeSurface(InterfaceEquations):
    """
    Represents the free surface kinematic and dynamic boundary conditions for the Navier-Stokes equations, defined by the second-order partial differential equations (PDEs):

    .. math:: \\vec{n} \\cdot [-p \\vec{\\vec{I}} + \\mu (\\nabla \\vec{u} + (\\nabla \\vec{u})^\\text{T})] = -\\sigma \\kappa \\vec{n} + \\nabla_s \\sigma
        
    where :math:`p` is the pressure, :math:`\\vec{\\vec{I}}` is the identity matrix, :math:`\\vec{u}` the velocity, :math:`\\mu` the dynamic viscosity, :math:`\\rho` is the mass density, :math:`\\sigma` the surface tension, :math:`\\kappa` the curvature of the interface and :math:`\\vec{n}` the normal vector of the interface.
    The operator :math:`\\nabla_s` denotes the surface gradient, defined at the interface.

    In the weak form, the equations are written as:

    .. math:: \\langle \\vec{n} \\cdot [-p \\vec{\\vec{I}} + \\mu (\\nabla \\vec{u} + (\\nabla \\vec{u})^\\text{T})], \\vec{v} \\rangle = - \\langle \\sigma, \\nabla_s \\vec{v} \\rangle + \\Big[ \\sigma \\vec{N}, \\vec{v} \\Big] \\,

    where :math:`\\vec{v}` is a test function of the velocity and :math:`\\vec{N}` is the outwards normal vector of the interface's bounds.
    
    This class requires the parent equations to be of type :ref:`StokesEquations <StokesEquations>`.
                 
    Args:
        surface_tension (ExpressionOrNum, optional): Surface tension of the fluid. Defaults to 1.
        kinbc_name (str, optional): Name of the kinematic boundary condition field. Defaults to "_kin_bc".
        static_interface (Union[Literal["auto"],bool], optional): If True, the mesh of the interface is static. Defaults to "auto".
        additional_normal_traction (ExpressionOrNum, optional): Additional normal traction. Defaults to 0.
        mass_transfer_rate (ExpressionOrNum, optional): Mass transfer rate in case there is mass transfer across the interface. Defaults to 0.        
        impose_marangoni_directly (bool, optional): If False, the weak form of grad_s(sigma) is integrated by parts. Defaults to False.
        kinematic_bc_coordinate_sys (Optional[BaseCoordinateSystem], optional): Coordinate system for the kinematic boundary condition. Defaults to None.
        remove_redundant_kinematic_bcs (bool, optional): If True, redundant kinematic boundary conditions are removed. Defaults to True.
    """


    required_parent_type = StokesEquations

    def __init__(self, *, surface_tension:ExpressionOrNum=1, kinbc_name:str="_kin_bc", static_interface:Union[Literal["auto"],bool]="auto", additional_normal_traction:ExpressionOrNum=0,
                 mass_transfer_rate:ExpressionOrNum=0,impose_marangoni_directly:bool=False,kinematic_bc_coordinate_sys:Optional[BaseCoordinateSystem]=None,remove_redundant_kinematic_bcs=True):
        super(NavierStokesFreeSurface, self).__init__()
        self.kinbc_name = kinbc_name
        self.static_interface = static_interface
        self.surface_tension = surface_tension
        self.mass_transfer_rate = mass_transfer_rate
        self.additional_normal_traction = additional_normal_traction
        self.impose_marangoni_directly=impose_marangoni_directly
        self.kinematic_bc_coordinate_sys=kinematic_bc_coordinate_sys
        self.remove_redundant_kinematic_bcs=remove_redundant_kinematic_bcs


    def define_fields(self):
        flow_eqs=self.get_parent_equations(StokesEquations)
        if flow_eqs == []:
            raise RuntimeError("No (Navier-)Stokes defined on domain "+self.get_parent_domain().get_domain_name())
        assert isinstance(flow_eqs,StokesEquations)
        if not flow_eqs.PFEM_options or not flow_eqs.PFEM_options.active or flow_eqs.PFEM_options.first_order_system:                    
            vspace=flow_eqs.get_velocity_space_from_mode(for_interface=True)
            static=self.static_interface
            if static=="auto":
                static=not self.get_current_code_generator().get_parent_domain()._coordinates_as_dofs

            if not static in {"auto",False,True}:
                raise RuntimeError("property static_interface must be either 'auto', True or False")
            if not static:
                vspace=self.get_current_code_generator().get_parent_domain()._coordinate_space
                #raise RuntimeError("Find out the position space for the kinbc")
            if vspace=="C2TB":
                vspace="C2"
            elif vspace=="C1TB":
                vspace="C1"
            self.define_scalar_field(self.kinbc_name, vspace)

    def define_scaling(self):
        flow_eqs=self.get_parent_equations(StokesEquations)
        assert isinstance(flow_eqs,StokesEquations)
        if not flow_eqs.PFEM_options or not flow_eqs.PFEM_options.active:                    
            static=self.static_interface
            if static=="auto":
                static=not self.get_current_code_generator().get_parent_domain()._coordinates_as_dofs

            if not static in {"auto",False,True}:
                raise RuntimeError("property static_interface must be either 'auto', True or False")

            if static:
                self.set_scaling(**{self.kinbc_name:1/test_scale_factor(flow_eqs.velocity_name)})
                self.set_test_scaling(**{self.kinbc_name: 1 / scale_factor(flow_eqs.velocity_name)})
            else:
                self.set_scaling(**{self.kinbc_name: 1 /test_scale_factor("mesh")})
                self.set_test_scaling(**{self.kinbc_name: 1 / scale_factor(flow_eqs.velocity_name)})


    def define_residuals(self):
        flow_eqs=self.get_parent_equations(StokesEquations)
        assert isinstance(flow_eqs,StokesEquations)
        #n = self.get_normal()
        n=var("normal")
        if not flow_eqs.PFEM_options or not flow_eqs.PFEM_options.active:                    
            u, u_test = var_and_test(flow_eqs.velocity_name)
            R, R_test = var_and_test("mesh")
            l, l_test = var_and_test(self.kinbc_name)            
            static=self.static_interface
            if static=="auto":
                static=not self.get_current_code_generator()._coordinates_as_dofs

            if not static in {"auto",False,True}:
                raise RuntimeError("property static_interface must be either 'auto', True or False")

            kbc_sign=1
            if static:
                    
                kin_bc = -dot(u, n)
                if self.mass_transfer_rate is not None and self.mass_transfer_rate!=0:
                    bulkeqs = self.get_parent_domain().get_equations()
                    nsbulk = bulkeqs.get_equation_of_type(StokesEquations)
                    assert isinstance(nsbulk,StokesEquations)
                    assert nsbulk.mass_density is not None
                    kin_bc+= self.mass_transfer_rate / nsbulk.mass_density
                self.add_residual(kbc_sign*weak(kin_bc , l_test,coordinate_system=self.kinematic_bc_coordinate_sys))
                self.add_residual(kbc_sign*weak(l , dot(n, u_test),coordinate_system=self.kinematic_bc_coordinate_sys))
            else:
                

                kin_bc = dot(mesh_velocity() - u, n)
                if self.mass_transfer_rate is not None and self.mass_transfer_rate!=0:
                    bulkeqs = self.get_parent_domain().get_equations()
                    nsbulk = bulkeqs.get_equation_of_type(StokesEquations)
                    assert isinstance(nsbulk,StokesEquations)
                    assert nsbulk.mass_density is not None
                    kin_bc+= self.mass_transfer_rate / nsbulk.mass_density
                self.add_residual(kbc_sign*weak( kin_bc , l_test,coordinate_system=self.kinematic_bc_coordinate_sys))

                dt_weight = 1
                self.add_residual(-dt_weight*weak(l, dot(n,  R_test),coordinate_system=self.kinematic_bc_coordinate_sys))
        else:
            x, u_test = var_and_test("mesh")

        if self.impose_marangoni_directly:
            if not static:
                raise RuntimeError("impose_marangoni_directly only works on static interfaces")
            else:
                self.add_residual(-weak(grad(self.surface_tension), u_test))
        else:
            self.add_residual(weak(self.surface_tension, div(u_test)) )
        self.add_residual(weak(self.additional_normal_traction ,dot(n, u_test)) )

    def before_assigning_equations_postorder(self, mesh:"AnyMesh"):
        flow_eqs=self.get_parent_equations(StokesEquations)
        assert isinstance(flow_eqs,StokesEquations)
        if (not flow_eqs.PFEM_options or not flow_eqs.PFEM_options.active) and (self.remove_redundant_kinematic_bcs):                    
            static=self.static_interface
            if static=="auto":
                static=not self.get_current_code_generator()._coordinates_as_dofs

            if not static in {"auto",False,True}:
                raise RuntimeError("property static_interface must be either 'auto', True or False")
            assert isinstance(mesh,InterfaceMesh)
            if static:
                self.pin_redundant_lagrange_multipliers(mesh,self.kinbc_name,flow_eqs.velocity_name)
            else:
                self.pin_redundant_lagrange_multipliers(mesh, self.kinbc_name, "mesh")


# TODO: Merge this with the free surface and activate it if there is an opposite side
class ConnectVelocityAtInterface(InterfaceEquations):
    """
    Connects the velocity field at the interface between two domain. 
    This is done by introducing Lagrange multipliers for each velocity component.
    The Lagrange multipliers are then used to enforce the continuity of the velocity field across the interface.
        
    This class requires the parent equations to be of type StokesEquations, meaning that if StokesEquations (or subclasses) are not defined in the parent domain, an error will be raised.
        
    Args:
        lagr_mult_prefix (str, optional): Prefix for name of the Lagrange multipliers. Defaults to "_lagr_velo_conn".
        mass_transfer_rate (ExpressionOrNum, optional): Mass transfer rate in case there is mass transfer across the interface. Defaults to 0.
        use_highest_space (bool, optional): If True, the highest space is used for the Lagrange multipliers. Defaults to False.
    """

    required_parent_type = StokesEquations

    def __init__(self, lagr_mult_prefix:str="_lagr_velo_conn", mass_transfer_rate:ExpressionOrNum=0,use_highest_space:bool=False,normal_velocity_jump:ExpressionOrNum=0):
        super(ConnectVelocityAtInterface, self).__init__()
        self.lagr_mult_prefix = lagr_mult_prefix
        self.mass_transfer_rate = mass_transfer_rate
        self.use_highest_space=use_highest_space
        self.normal_velocity_jump=normal_velocity_jump

    def get_required_fields(self) -> List[str]:
        flow_eqs=self.get_parent_equations(StokesEquations)
        assert isinstance(flow_eqs,StokesEquations)
        fields = [flow_eqs.velocity_name+ "_x", flow_eqs.velocity_name+"_y", flow_eqs.velocity_name+"_z"]
        if isinstance(self.get_coordinate_system(),AxisymmetryBreakingCoordinateSystem):
            return fields[0:self.get_nodal_dimension()]+[flow_eqs.velocity_name+"_phi"]
        elif isinstance(self.get_coordinate_system(),CartesianCoordinateSystemWithAdditionalNormalMode):
            return fields[0:self.get_nodal_dimension()]+[flow_eqs.velocity_name+"_normal"]
        else:
            return fields[0:self.get_nodal_dimension()]

    def define_fields(self):
        fields=self.get_required_fields()
        for f in fields:
            if self.get_opposite_side_of_interface(raise_error_if_none=False) is None:
                raise RuntimeError("Cannot connect any fields at the interface if no opposite side is present")
            inside_space = cast(Union[FiniteElementSpaceEnum,Literal[""]], self.get_parent_domain().get_space_of_field(f))
            if inside_space == "":
                raise RuntimeError(
                    "Cannot connect field " + f + " at the interface, since it cannot find in the inner domain")            
            opp_parent=self.get_opposite_side_of_interface().get_parent_domain()
            assert opp_parent is not None
            outside_space = cast(Union[FiniteElementSpaceEnum,Literal[""]], opp_parent.get_space_of_field(f))
            if outside_space == "":
                raise RuntimeError(
                    "Cannot connect field " + f + " at the interface, since it cannot find in the outer domain")
            space=get_interface_field_connection_space(inside_space,outside_space,self.use_highest_space)
            assert space!=""
            self.define_scalar_field(self.lagr_mult_prefix + f, space)
            
        # However, we must remove the m=1 connection for the radial component
        aziinfo=self.get_azimuthal_r0_info()
        csys=self.get_coordinate_system()
        for f in fields:
            for i in [0,1,2]:
                if f in aziinfo[i]:
                    aziinfo[i].add(self.lagr_mult_prefix + f)
                else:
                    if self.lagr_mult_prefix + f in aziinfo[i]:
                        aziinfo[i].remove(self.lagr_mult_prefix + f)

    def define_scaling(self):
        fields=self.get_required_fields()
        flow_eqs=self.get_parent_equations(StokesEquations)
        assert isinstance(flow_eqs,StokesEquations)
        for f in fields:
            self.set_test_scaling(**{self.lagr_mult_prefix + f:1/scale_factor(flow_eqs.velocity_name)})
            self.set_scaling(**{self.lagr_mult_prefix + f: 1 / test_scale_factor(flow_eqs.velocity_name)})


    def define_residuals(self):
        fields = self.get_required_fields()
        n = self.get_normal()
        ins_ns=self.get_parent_domain().get_equations().get_equation_of_type(StokesEquations)
        opp_par=self.get_opposite_side_of_interface().get_parent_domain()
        assert opp_par is not None
        out_ns=opp_par.get_equations().get_equation_of_type(StokesEquations)
        assert isinstance(ins_ns,StokesEquations)
        assert isinstance(out_ns,StokesEquations)
        rho_inside = ins_ns.mass_density
        rho_outside = out_ns.mass_density
        if rho_outside is not None:
            rho_outside = evaluate_in_domain(rho_outside, self.get_opposite_side_of_interface())
        else:
            rho_outside=0

        if (self.mass_transfer_rate is not None) and self.mass_transfer_rate != 0:
            assert rho_inside is not None
            masstrans = subexpression(self.mass_transfer_rate / rho_inside - self.mass_transfer_rate / rho_outside)
        else:
            masstrans = 0

        for i, f in enumerate(fields):
            l, l_test = var_and_test(self.lagr_mult_prefix + f)
            inside, inside_test = var_and_test(f)
            outside, outside_test = var_and_test(f, domain=self.get_opposite_side_of_interface())
            self.add_residual(weak(inside - outside - masstrans * n[i]+self.normal_velocity_jump*n[i],  l_test ))
            self.add_residual(weak(l, inside_test) )
            self.add_residual(-weak(l , outside_test))

    def before_assigning_equations_postorder(self, mesh:"AnyMesh"):
        assert isinstance(mesh,InterfaceMesh)
        assert mesh._opposite_interface_mesh is not None 
        if mesh.nelement() == 0 or mesh._opposite_interface_mesh.nelement() == 0: 
            return
        fields=self.get_required_fields()
        for i,_ in enumerate(fields):
            self.pin_redundant_lagrange_multipliers(mesh,self.lagr_mult_prefix + fields[i],[fields[i]],opposite_interface=[fields[i]])


class NoSlipBC(DirichletBC):
    """
    No-slip boundary condition for the Navier-Stokes equations.
    This is enforced by setting the degrees of freedom of the velocity field to zero at the specified boundary.
    It is a subclass of DirichletBC and inherits all its arguments.
                
    Args:
        velocity_name (str, optional): Name of the velocity field. Defaults to "velocity".
    """
            
    def __init__(self):
        super().__init__()
        self.veloname="velocity"

    def define_residuals(self):
        self._dcs={}
        dim = self.get_nodal_dimension()
        dirs = ["x", "y", "z"]

        ns=self.get_parent_domain().get_equations().get_equation_of_type(StokesEquations)
        lagr=False
        if isinstance(ns,StokesEquations) and ns.PFEM_options and ns.PFEM_options.active:
            for i in range(dim):
                self._dcs["mesh_"+dirs[i]]=True
            lagr=True
            if ns.PFEM_options.first_order_system:
                for i in range(dim):
                    self._dcs[self.veloname+"_"+dirs[i]]=0
        else:
            for i in range(dim):
                self._dcs[self.veloname+"_"+dirs[i]]=0
        cs=self.get_coordinate_system()
        if cs is not None:
            if isinstance(cs,AxisymmetryBreakingCoordinateSystem):
                if lagr:
                    raise RuntimeError("TODO")
                self._dcs[self.veloname+"_phi"]=0
            elif isinstance(cs,CartesianCoordinateSystemWithAdditionalNormalMode):
                if lagr:
                    raise RuntimeError("TODO")
                self._dcs[self.veloname+"_normal"]=0                
        super(NoSlipBC, self).define_residuals()


class NavierStokesSlipLength(InterfaceEquations):
    """
    Represents the Navier-Stokes slip length boundary condition for the Navier-Stokes equations, by setting the tangential velocity to a prescribed slip length.
    The normal no-penetration still needs to be ensured, e.g. by pinning or Lagrange multipliers on curved surfaces.
    Furthermore, it assumes zero velocity of the solid.
                
    This class requires the parent equations to be of type StokesEquations, meaning that if StokesEquations (or subclasses) are not defined in the parent domain, an error will be raised.
            
    Args:
        sliplength (ExpressionOrNum): Slip length of the fluid.
        surface_tension (ExpressionNumOrNone, optional): Surface tension of the fluid. Defaults to None.
        tangential_wall_velocity (ExpressionOrNum, optional): Tangential velocity of the wall. Defaults to 0.
    """

    required_parent_type = StokesEquations

    def __init__(self, sliplength:ExpressionOrNum,surface_tension:ExpressionNumOrNone=None,tangential_wall_velocity:ExpressionOrNum=0):
        super(NavierStokesSlipLength, self).__init__()
        self.sliplength = sliplength
        self.surface_tension=surface_tension
        self.tangential_wall_velocity=tangential_wall_velocity

    def define_residuals(self):
        flow_eqs=self.get_parent_equations(StokesEquations)
        assert isinstance(flow_eqs,StokesEquations)
        n = self.get_normal()
        u, utest = var_and_test(flow_eqs.velocity_name)
        utang = u - dot(u, n) * n
        utang_wall=self.tangential_wall_velocity-dot(self.tangential_wall_velocity, n) * n
        utest_tang = utest - dot(utest, n) * n
        peqs=self.get_parent_equations()
        assert isinstance(peqs,StokesEquations)
        mu=peqs.dynamic_viscosity
        factor = mu / (self.sliplength)
        self.add_residual(factor * weak(utang-utang_wall, utest_tang))
        if self.surface_tension is not None:
            impose_surface_tension_directly=False
            if impose_surface_tension_directly:
                self.add_residual(-weak(grad(self.surface_tension), utest))
            else:
                self.add_residual(weak(self.surface_tension, div(utest)) )


class NavierStokesPrescribedNormalVelocity(InterfaceEquations):
    """
    Prescribes the normal velocity at the interface between two domains.
    This is done by introducing a Lagrange multiplier for the normal velocity component and enforcing it in the weak sense.
    The normal no-penetration still needs to be ensured, e.g. by pinning or Lagrange multipliers on curved surfaces.
    Furthermore, it assumes zero velocity of the solid
        
    This class requires the parent equations to be of type StokesEquations, meaning that if StokesEquations (or subclasses) are not defined in the parent domain, an error will be raised.
        
    Args:
        normal_velocity (ExpressionOrNum): Normal velocity to be imposed at the interface. Defaults to 0.
    """


    required_parent_type = StokesEquations

    def __init__(self, normal_velocity:ExpressionOrNum=0):
        super(NavierStokesPrescribedNormalVelocity, self).__init__()
        self.normal_velocity = normal_velocity
        self.lagrange_multiplier_name="_lagr_normal_velo"

    def define_fields(self):
        flow_eqs=self.get_parent_equations(StokesEquations)
        assert isinstance(flow_eqs,StokesEquations)
        space=flow_eqs.get_velocity_space_from_mode(for_interface=True)
        self.define_scalar_field(self.lagrange_multiplier_name,space,scale=1/test_scale_factor(flow_eqs.velocity_name),testscale=1/scale_factor(flow_eqs.velocity_name))

    def define_residuals(self):
        flow_eqs=self.get_parent_equations(StokesEquations)
        assert isinstance(flow_eqs,StokesEquations)
        n = self.get_normal()
        u, utest = var_and_test(flow_eqs.velocity_name)
        l,ltest=var_and_test(self.lagrange_multiplier_name)
        self.add_residual(weak(dot(u,n)-self.normal_velocity,ltest))
        self.add_residual(weak(l,dot(utest,n)))

    def before_assigning_equations_postorder(self, mesh:"AnyMesh"):
        assert isinstance(mesh,InterfaceMesh)
        flow_eqs=self.get_parent_equations(StokesEquations)
        assert isinstance(flow_eqs,StokesEquations)
        self.pin_redundant_lagrange_multipliers(mesh, self.lagrange_multiplier_name, flow_eqs.velocity_name)




class NavierStokesAzimuthalComponent(Equations):
    """
        Adds an azimuthal component to the velocity defined with Navier-Stokes equations in axisymmetric coordinates.
        This is independent of the azimuthal angle phi, but will contribute to all equations

        Args:
            prescribed_value (ExpressionNumOrNone, optional): Prescribed value for the azimuthal component. Defaults to None.
    """
        
    def __init__(self,prescribed_value:ExpressionNumOrNone=None):
        super(NavierStokesAzimuthalComponent, self).__init__()
        self.prescribed_value=prescribed_value

    def define_fields(self):
        ns=self.get_combined_equations().get_equation_of_type(StokesEquations)
        if not isinstance(ns,StokesEquations):
            raise RuntimeError("Must be combined with either a StokesEquation, NavierStokesEquation or subclasses of these")
        else:            
            self.define_scalar_field(ns.velocity_name+"_phi","C2TB" if ns.mode == "CR" else "C2",scale=ns.velocity_name,testscale=test_scale_factor(ns.velocity_name))

    def define_residuals(self):
        ns = self.get_combined_equations().get_equation_of_type(StokesEquations)
        if not isinstance(ns,StokesEquations):
            raise RuntimeError("Must be combined with either a StokesEquation, NavierStokesEquation or subclasses of these")
        if not isinstance(self.get_coordinate_system(),AxisymmetricCoordinateSystem):
            raise RuntimeError("Can only use NavierStokesAzimuthalComponent when the coordinate system is axisymmetric")
        ut,uttest=var_and_test(ns.velocity_name+"_phi")
        ur,urtest=var_and_test(ns.velocity_name+"_x")
        uz,_ = var_and_test(ns.velocity_name+"_y")

        dutdr=partial_x(ut)
        dutdz = partial_y(ut)
        r=var("coordinate_x")

        # Flipped sign compared to oomph-lib: In pyoomph du/dt has a positive sign!
        sign=-1

        if ns.mass_density is not None:
            rho = subexpression(ns.mass_density)
        else:
            rho=0
        mu = subexpression(ns.dynamic_viscosity)
        momentum_scheme = ns.momentum_scheme
        if isinstance(ns,NavierStokesEquations):
            rho_adv=rho*ns.nonlinear_factor
            rho_dt = rho * ns.dt_factor
        else:
            rho_adv=0
            rho_dt=0


        if ns.bulkforce is None or is_zero(ns.bulkforce):
            body_force=vector(0)
        else:
            body_force=ns.bulkforce
            if not isinstance(body_force,Expression):
                body_force=Expression(body_force)

        if ns.gravity is None or is_zero(ns.gravity):
            gravity=vector(0)
        else:
            gravity=ns.gravity
            if not isinstance(gravity,Expression):
                gravity=Expression(gravity)

        self.add_residual(weak(time_scheme(momentum_scheme,sign*rho_adv*ut**2/r),urtest))
        #TODO: OOMPH-LIB HAS THIS TERM, WHICH I DON'T SEE:
        #weak(2.0*scaled_re_inv_ro*ut,urtest) # Should be a Coriolis term, but why ut is linear here? Direction should not matter!
        #Apparently only used if we have a rotating coordinate system!

        polar=self.get_nodal_dimension()==1


        if self.prescribed_value is not None:
            self.add_residual(weak(ut-self.prescribed_value,uttest/test_scale_factor(ns.velocity_name+"_phi")/scale_factor(ns.velocity_name+"_phi")))
        else:
            # Momentum equation for the azimuthal component
            # Body force
            # residuals[local_eqn] += r*body_force[2]*testf[l]*W;
            self.add_residual(weak(time_scheme(momentum_scheme,sign * body_force[2]), uttest))

            # Gravity
            # residuals[local_eqn] += r*scaled_re_inv_fr*testf[l]*G[2]*W;
            self.add_residual(weak(time_scheme(momentum_scheme,sign * rho * gravity[2]), uttest))

            # Viscosity terms
            # residuals[local_eqn] -= visc_ratio*(r*interpolated_dudx(2,0) - Gamma[0]*interpolated_u[2])*dtestfdx(l,0)*W;
            self.add_residual(-sign * weak(time_scheme(momentum_scheme,mu * (dutdr - ut / r)), partial_x(uttest)))
            # residuals[local_eqn] -= visc_ratio*r*interpolated_dudx(2,1)*dtestfdx(l,1)*W;
            if not polar:
                self.add_residual(-sign * weak(time_scheme(momentum_scheme,mu * dutdz), partial_y(uttest)))
            # residuals[local_eqn] -= visc_ratio*((interpolated_u[2]/r) - Gamma[0]*interpolated_dudx(2,0))*testf[l]*W;
            self.add_residual(-sign * weak(time_scheme(momentum_scheme,mu * (ut / r - dutdr)), uttest / r))

            # Inertia terms
            # residuals[local_eqn] -= scaled_re_st*r*dudt[2]*testf[l]*W;
            self.add_residual(-sign * weak(time_scheme(momentum_scheme,rho_dt * partial_t(ut)), uttest))
            # residuals[local_eqn] -= scaled_re*(r*interpolated_u[0]*interpolated_dudx(2,0)+ interpolated_u[0]*interpolated_u[2]+ r*interpolated_u[1]*interpolated_dudx(2,1))*testf[l]*W;
            self.add_residual(-sign * weak(time_scheme(momentum_scheme,rho_adv * (ur * dutdr + ur * ut / r + uz * dutdz*(0 if polar else 1))), uttest))




class NavierStokesContactAngle(InterfaceEquations):
        
    """
        Enforces a constant contact angle for a droplet. 

        This class requires the parent equations to be of type StokesEquations, meaning that if StokesEquations (or subclasses) are not defined in the parent domain, an error will be raised.

        Args:    
            contact_angle (ExpressionOrNum): Contact angle of the droplet. Defaults to 90 * degree.
            wall_normal (Expression): Normal vector of the wall (should be normalized). Defaults to vector([0, 1]).
            wall_tangent: Tangential vector of the wall (should be normalized). If None, it is calculated using the bac-cab rule to point inward to the bulk domain, along the substrate (i.e. orthogonal to the wall normal).            with_respect_to_tangent (bool): If True, the contact angle is defined with respect to the tangent vector. Defaults to None.
    """
    
    #required_parent_type = NavierStokesFreeSurface

    def __init__(self, contact_angle:ExpressionOrNum=90 * degree, *, wall_normal:Expression=vector([0, 1]), wall_tangent:Expression=None,
                 with_respect_to_tangent:bool=True):
        super(NavierStokesContactAngle, self).__init__()
        self.wall_normal = 0+wall_normal
        
        if wall_tangent is None:
            # bac-cab rule assuming the wall_normal is normalized. Pointing inward to the bulk domain, along the substrate (i.e. orthogonal to the wall normal)
            self.wall_tangent=self.wall_normal*dot(self.wall_normal,var("normal",domain=".."))-var("normal",domain="..")
            self.wall_tangent=subexpression(self.wall_tangent/square_root(dot(self.wall_tangent,self.wall_tangent)))
        else:
            self.wall_tangent = 0+wall_tangent
        self.contact_angle = 0+contact_angle
        self.with_respect_to_tangent = with_respect_to_tangent

    def define_residuals(self):
        if self.with_respect_to_tangent:
            m = sin(self.contact_angle) * self.wall_normal + cos(self.contact_angle) * self.wall_tangent
        else:
            m = cos(self.contact_angle) * self.wall_normal + sin(self.contact_angle) * self.wall_tangent
        _, utest = var_and_test("velocity")
        nseq = self.get_parent_equations(of_type=NavierStokesFreeSurface)
        if isinstance(nseq,list):
            if len(nseq)==0:
                nseq=None
            elif len(nseq)>1:
                raise RuntimeError("Multiple NavierStokesFreeSurface equations found")
            else:
                nseq=nseq[0]                
        if nseq is None:
            from ..equations.multi_component import MultiComponentNavierStokesInterface
            nseq=self.get_parent_equations(of_type=MultiComponentNavierStokesInterface)
            if isinstance(nseq,list):
                if len(nseq)==0:
                    nseq=None
                elif len(nseq)>1:
                    raise RuntimeError("Multiple NavierStokesFreeSurface equations found")
                else:
                    nseq=nseq[0]
        if nseq is None:
            raise RuntimeError("Must be applied on NavierStokesFreeSurface or a MultiComponentNavierStokesInterface")
        if isinstance(nseq,NavierStokesFreeSurface):
            sigma = 0+nseq.surface_tension
        else:
            # TODO: Use surface tension projection if present
            sigma = 0+nseq.interface_props.surface_tension
        
        #self.add_local_function("mx",m[0])
        #self.add_local_function("my",m[1])
        self.add_residual( weak(sigma , dot(m, utest)))

class StokesFlowRadialFarField(InterfaceEquations):
    """
    Enforces a far-field condition for a radialsymmetric inward or outward flow.

    Args:
        infinity_pressure (ExpressionOrNum): Pressure at infinity. Defaults to 0.
    """
    def __init__(self,infinity_pressure:ExpressionOrNum=0):
        super().__init__()
        self.pinfty=infinity_pressure
    required_parent_type=StokesEquations
    def define_residuals(self):
        stokes_eqs=self.get_parent_equations()
        if not isinstance(stokes_eqs,StokesEquations):
            raise RuntimeError("Must be applied on StokesEquations")
        u,utest=var_and_test(stokes_eqs.velocity_name)
        uB,uBtest=var_and_test(stokes_eqs.velocity_name,domain="..")
        
        strain=2*stokes_eqs.dynamic_viscosity*sym(grad(uB))
        n=var("normal")
        normstrain=matproduct(strain,n)
        p,ptest=var_and_test(stokes_eqs.pressure_name)
        self.add_weak(-normstrain+self.pinfty*n,utest)
