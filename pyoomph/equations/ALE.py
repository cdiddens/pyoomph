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
 
from ..meshes.mesh import InterfaceMesh, AnyMesh
from .. import GlobalLagrangeMultiplier, WeakContribution, IntegralConstraint
from ..generic import Equations,InterfaceEquations,ODEEquations
from .generic import get_interface_field_connection_space
from ..expressions import *  # Import grad et al

from ..typings import *
if TYPE_CHECKING:
    from ..generic.problem import Problem
    from ..generic.codegen import EquationTree

class BaseMovingMeshEquations(Equations):
    """
        Defines the base class for moving mesh equations. This class should be inherited by all moving mesh equations.
            
        Args:
            coordinate_space(Optional[str]): The coordinate space. Default is None.
            constrain_bulk_to_C1(bool): If True, the bulk position space is constrained to C1, except on boundaries. Currently not working. Default is False.
            coordsys(Optional[BaseCoordinateSystem]): The coordinate system. Default is None.
    """

    def __init__(self,coordinate_space:Optional[str]=None,constrain_bulk_to_C1:bool=False,coordsys:Optional[BaseCoordinateSystem]=None):
        super().__init__()
        self.coordsys=coordsys
        self.constrain_bulk_to_C1 = constrain_bulk_to_C1
        if self.constrain_bulk_to_C1:
            raise RuntimeError("constrain_bulk_to_C1 is not working at the moment")
        self.coordinate_space=coordinate_space
        
        self.min_coordinate_space:Optional[str]=None
        if self.coordinate_space is not None:
            self.min_coordinate_space = self.coordinate_space

    def define_fields(self):
        self.activate_coordinates_as_dofs(coordinate_space=self.min_coordinate_space)
        if self.constrain_bulk_to_C1:
            self.get_current_code_generator().bulk_position_space_to_C1=True

    def define_scaling(self):
        self.set_scaling(mesh= scale_factor("spatial"))
        self.set_test_scaling(mesh=1/scale_factor("spatial"))

    def after_mapping_on_macro_elements(self):
        self.get_mesh().set_lagrangian_nodal_coordinates()
        self.get_mesh().invalidate_lagrangian_kdtree()

    def with_average_position_constraint(self, problem:"Problem", *, act_on:str="mesh",ode_domain_name:str="globals",lagrange_prefix:str="lagr_intconstr_mesh_", set_zero_on_normal_mode_eigensolve:bool=True, **avg_pos:ExpressionOrNum)->Equations:

        lagrs:Dict[str,ExpressionOrNum]={}
        for c,v in avg_pos.items():
            if c not in {"x","y","z"}:
                raise RuntimeError("can only set average positions of x,y,z, but not "+str(c))
            lagrs[c]=v

        ode_additions = GlobalLagrangeMultiplier(**{lagrange_prefix+c:0 for c,_ in lagrs.items()},set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve)
        #ode_additions +=TestScaling(**{lagrange_name:1/scale_factor("pressure")})
        #ode_additions += Scaling(**{lagrange_name: 1 / test_scale_factor("pressure")})

        eq_additions = self
        for c,v in lagrs.items():
            l=var(lagrange_prefix+c,domain=ode_domain_name)
            eq_additions += WeakContribution(l-v, testfunction(act_on+"_"+c))
            eq_additions += WeakContribution(var("mesh_"+c), testfunction(l))
        problem.add_equations(ode_additions @ ode_domain_name)
        return eq_additions

    def get_squared_spatial_factor(self)->ExpressionOrNum:
        raise RuntimeError("Specify")


class PseudoElasticMesh(BaseMovingMeshEquations):
    """
        Represents a deformable solid mesh defined by the a kinematic boundary condition:

        lambda = 2 * E / 2 / (1 + nu) * (E * nu / (1 + nu) / (1 - 2 * nu)) / (nu / (1 + nu) / (1 - 2 * nu) + 2 * E / 2 / (1 + nu))
        sigma = lambda * tr(sym(grad(x - X))) * I + 2 * mu * sym(grad((x - X)))
        div(sigma) = 0

        where x is the unknown Eulerian coordinate, X is the is the Lagrangian coordinate, E is the Young's modulus, nu is the Poisson's ratio, lambda is the Lamé parameter, sigma is the stress tensor, tr is the trace operator, sym(grad()) is the symmetric gradient operator, div is the divergence operator, f is the bulk force, and I is the identity matrix.

        This class is a subclass of BaseMovingMeshEquations and inherits all its arguments.                    

        Args:
            E (ExpressionOrNum): The Young's modulus. Default is 1*scale_factor("spatial")**2.
            nu (ExpressionOrNum): The Poisson's ratio. Default is rational_num(3,10).
            spatial_error_factor (Optional[float]): The spatial error factor. Default is None.
            coordinate_space (Optional[str]): The coordinate space. Default is None.
            constrain_bulk_to_C1 (bool): If True, the bulk position space is constrained to C1. Default is False.
            coordsys (Optional[BaseCoordinateSystem]): The coordinate system. Default is cartesian.
    """
    def __init__(self, E:ExpressionOrNum=1*scale_factor("spatial")**2, nu:ExpressionOrNum=rational_num(3,10), spatial_error_factor:Optional[float]=None,coordinate_space:Optional[str]=None,constrain_bulk_to_C1:bool=False,coordsys:Optional[BaseCoordinateSystem]=cartesian):
        super(PseudoElasticMesh, self).__init__(coordinate_space=coordinate_space,constrain_bulk_to_C1=constrain_bulk_to_C1,coordsys=coordsys)
        self.E = E
        self.nu = nu
        self.ALE_factor = 1
        self.spatial_error_factor = spatial_error_factor
        self.min_coordinate_space=coordinate_space

    def get_squared_spatial_factor(self)->ExpressionOrNum:
        return self.E

    def define_residuals(self):
        E = self.E
        nu = self.nu

        mu = E / 2 / (1 + nu)
        lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        lmbda = 2 * mu * lmbda / (lmbda + 2 * mu)
        if self.coordsys:
            eps:Callable[[Expression],Expression] = lambda v: sym(grad(v,  coordsys=self.coordsys, lagrangian=True))
        else:
            eps:Callable[[Expression],Expression] = lambda v: sym(grad(v,  lagrangian=True))
        # vdim = self.get_coordinate_system().vector_gradient_dimension(self.get_element_dimension(), lagrangian=True)
        sigma:Callable[[Expression],Expression] = lambda v: lmbda * trace(eps(v)) * identity_matrix() + 2 * mu * eps(v)

        x,x_test = var_and_test("mesh")
        X = var("lagrangian")
        displ = x - X
        check=sym(grad(x_test,  coordsys=self.coordsys, lagrangian=True))        
        self.add_residual(self.ALE_factor * Weak(sigma(displ), eps(x_test),coordinate_system=self.coordsys) )


class LaplaceSmoothedMesh(BaseMovingMeshEquations):
    """
        Represents a Laplace smoothed mesh. The Laplace smoothed mesh is defined by the kinematic boundary condition:

            laplace(x - X) = 0
        
        where x is the unknown Eulerian coordinate, X is the is the Lagrangian coordinate and lapalce represents the Laplacian operator.
    
        This class is a subclass of BaseMovingMeshEquations and inherits all its arguments. 

        Args:
            factor (ExpressionOrNum): The factor. Default is scale_factor("spatial")**2.
            constrain_bulk_to_C1 (bool): If True, the bulk position space is constrained to C1. Default is False.
    """
    def __init__(self,factor:ExpressionOrNum=scale_factor("spatial")**2,constrain_bulk_to_C1:bool=False,coordinate_space:Optional[str]=None,coordsys:Optional[BaseCoordinateSystem]=cartesian,symmetrize:bool=False):
        super(LaplaceSmoothedMesh, self).__init__(coordinate_space=coordinate_space,constrain_bulk_to_C1=constrain_bulk_to_C1,coordsys=coordsys)
        self.factor=factor
        self.symmetrize=symmetrize

    def get_squared_spatial_factor(self)->ExpressionOrNum:
        return self.factor

    def define_residuals(self):
        x,x_test = var_and_test("mesh")
        X = var("lagrangian")
        displ = x - X
        coordsys=self.coordsys
        if self.symmetrize:
            tens=sym(grad(displ,coordsys=coordsys,lagrangian=True))
        else:
            tens=grad(displ,coordsys=coordsys,lagrangian=True)
        self.add_residual(self.factor*Weak(tens, grad(x_test,coordsys=coordsys, lagrangian=True),coordinate_system=coordsys) )


class SingleDirectionLaplaceSmoothedMesh(LaplaceSmoothedMesh):
    def __init__(self, direction:Union[int,Literal["x","y","z"]], factor: ExpressionOrNum = scale_factor("spatial") ** 2, constrain_bulk_to_C1: bool = False, coordinate_space: Optional[str] = None, coordsys: OptionalCoordinateSystem = cartesian):
        super().__init__(factor, constrain_bulk_to_C1, coordinate_space, coordsys, symmetrize=False)
        self.direction=direction
        if isinstance(direction,str):
            self.direction={"x":0,"y":1,"z":2}[direction]
    
    def define_residuals(self):
        dirn=["x","y","z"][self.direction]
        x,x_test = var_and_test("mesh_"+dirn)
        X = var("lagrangian_"+dirn)
        displ = x - X
        coordsys=self.coordsys        
        tens=grad(displ,coordsys=coordsys,lagrangian=True)[self.direction]
        self.add_residual(self.factor*Weak(tens, grad(x_test,coordsys=coordsys, lagrangian=True)[self.direction],coordinate_system=coordsys) )
        ndim=self.get_mesh().get_code_gen().get_nodal_dimension()
        for i in range(ndim):
            if i!=self.direction:
                self.set_Dirichlet_condition("mesh_"+["x","y","z"][i],True)



class HyperelasticSmoothedMesh(BaseMovingMeshEquations):
    """Hyperelastic mesh smoothing. The mesh is smoothed by minimizing the energy functional:
            
            W = integral( mu/2*(I1-d)+kappa/2*(J-1)**2 dOmega )
    
        where I1 is the first invariant of the right Cauchy-Green deformation tensor, J is the determinant of the deformation gradient, mu is the shear modulus, and kappa is the bulk modulus. d is the dimension of the mesh.

    Args:
        mu (float): The shear modulus. Default is 1.
        kappa (float): The bulk modulus. Default is 1.
        coordinate_space (Optional[str]): The coordinate space. Default is None.
        constrain_bulk_to_C1 (bool): If True, the bulk position space is constrained to C1. Default is False.
        coordsys (Optional[BaseCoordinateSystem]): The coordinate system. Default is cartesian.
    """
    def __init__(self,mu:float=1,kappa:float=1, coordinate_space: Optional[str] = None, constrain_bulk_to_C1: bool = False, coordsys: Optional[BaseCoordinateSystem] = cartesian,use_subexpressions:bool=False):
        super().__init__(coordinate_space, constrain_bulk_to_C1, coordsys)
        self.use_subexpressions=use_subexpressions
        self.mu=mu
        self.kappa=kappa
        
        
    def define_residuals(self):
        x=var("mesh")
        dxdX=grad(x,lagrangian=True,coordsys=self.coordsys)
        J=determinant(dxdX)
        if self.use_subexpressions:
            J=subexpression(J)
        I1=trace( matproduct(transpose(dxdX),dxdX) )*J**rational_num(-2,3)        
        I1min=I1-self.get_nodal_dimension() # or 3?
        if self.use_subexpressions:
            I1min=subexpression(I1min)
        F=self.mu/2*I1min+self.kappa/2*(J-1)**2
        self.add_functional_minimization(F,dimensional_testfunctions=False,coordinate_system=self.coordsys,lagrangian=True)
        
class YeohSmoothedMesh(BaseMovingMeshEquations):
    """Yeoh mesh smoothing. The mesh is smoothed by minimizing the energy functional:
                
                W = integral( 1/2 * (C1*I1min+C2*I1min**2+C3*I1min**3+kappa*(J-1)**2 ) * dOmega )
        
            where I1min=I1-d is the first invariant of the right Cauchy-Green deformation tensor minus the dimension d of the mesh, J is the determinant of the deformation gradient, C1, C2, and C3 are the Yeoh constants, and kappa is the bulk modulus.

    Args:
        kappa (float): The bulk modulus. Default is 1.
        C1 (float): The Yeoh constant C1. Default is 1.
        C2 (float): The Yeoh constant C2. Default is 10.
        C3 (float): The Yeoh constant C3. Default is 0.
        coordinate_space (Optional[str]): The coordinate space. Default is None.
        constrain_bulk_to_C1 (bool): If True, the bulk position space is constrained to C1. Default is False.
        coordsys (Optional[BaseCoordinateSystem]): The coordinate system. Default is cartesian
        
    """
    def __init__(self,kappa:float=1, C1:float=1,C2:float=10,C3:float=0, coordinate_space: Optional[str] = None, constrain_bulk_to_C1: bool = False, coordsys: Optional[BaseCoordinateSystem] = cartesian,use_subexpressions:bool=False):
        super().__init__(coordinate_space, constrain_bulk_to_C1, coordsys)
        self.use_subexpressions=use_subexpressions
        self.C1=C1
        self.C2=C2
        self.C3=C3
        self.kappa=kappa
        
        
    def define_residuals(self):
        x=var("mesh")
        dxdX=grad(x,lagrangian=True,coordsys=self.coordsys)
        J=determinant(dxdX)
        if self.use_subexpressions:
            J=subexpression(J)
        I1=trace( matproduct(transpose(dxdX),dxdX) )*J**rational_num(-2,3)
        I1min=I1-self.get_nodal_dimension()
        F=(self.C1*I1min+self.C2*I1min**2+self.C3*I1min**3+self.kappa*(J-1)**2)/2                                
        self.add_functional_minimization(scale_factor("spatial")*F,dxdX,dimensional_testfunctions=False,coordinate_system=self.coordsys,lagrangian=True)
        



class PinMeshCoordinates(Equations):
    def __init__(self,*directions:Union[int,Literal["x","y","z"]]):
        super(PinMeshCoordinates, self).__init__()
        if len(directions)>0:
            self.directions=set()        
            for d in directions:
                if isinstance(d,str):
                    self.directions.add({"x":0,"y":1,"z":2}[d])
                else:
                    self.directions.add(d)
        else:
            self.directions=None
    
    def define_residuals(self):
        if self.directions is None:
            for d in range(self.get_mesh().get_code_gen().get_nodal_dimension()):
                self.set_Dirichlet_condition("mesh_"+["x","y","z"][d],True)
        else:
            for d in self.directions:
                self.set_Dirichlet_condition("mesh_"+["x","y","z"][d],True)
        

class SetLagrangianToEulerianAfterSolve(Equations):
    """
        Sets the Lagrangian nodal coordinates to the Eulerian nodal coordinates after the Newton solve.    
    """
    def __init__(self):
        super(SetLagrangianToEulerianAfterSolve, self).__init__()
        self.active=True
    def after_newton_solve(self):
        super(SetLagrangianToEulerianAfterSolve, self).after_newton_solve()
        if self.active:
            self.get_mesh().set_lagrangian_nodal_coordinates()
            self.get_mesh().invalidate_lagrangian_kdtree()


class ConnectMeshAtInterface(InterfaceEquations):
    """
        Connects the mesh at the interface by enforcing the equality of the nodal coordinates at the interface.

        Args:
            lagr_mult_prefix(str): The prefix for the Lagrange multipliers. Default is "_lagr_conn_".
            use_highest_space(bool): If True, the highest space used in other elements is used for the Lagrange Multipliers. Default is False.
    """
    def __init__(self,lagr_mult_prefix:str="_lagr_conn_",use_highest_space:bool=False):
        super(ConnectMeshAtInterface, self).__init__()
        self.lagr_mult_prefix=lagr_mult_prefix
        self.use_highest_space=use_highest_space

    def get_required_fields(self) -> List[str]:
        dim = self.get_nodal_dimension()
        fields = ["mesh_x", "mesh_y", "mesh_z"]
        return fields[0:dim]

    def define_fields(self):
        for f in self.get_required_fields():
            if self.get_opposite_side_of_interface(raise_error_if_none=False) is None:
                raise self.add_exception_info(RuntimeError("Cannot connect any fields at the interface if no opposite side is present"))
            inside_space=self.get_parent_domain()._coordinate_space            
            if inside_space=="":
                raise RuntimeError("Cannot connect field "+f+" at the interface, since it cannot find in the inner domain. You might have to set coordinate_space explicitly in the moving mesh class")
            outdom=self.get_opposite_side_of_interface().get_parent_domain()
            assert outdom is not None
            outside_space=outdom._coordinate_space #type:ignore
            inside_space=cast(FiniteElementSpaceEnum,inside_space)
            outside_space=cast(FiniteElementSpaceEnum,outside_space)
            space = get_interface_field_connection_space(inside_space, outside_space,self.use_highest_space)
            if space=="":
                raise RuntimeError("Cannot connect field "+f+" at the interface, since it cannot find in the inner domain. You might have to set coordinate_space explicitly in the moving mesh class")
            self.define_scalar_field(self.lagr_mult_prefix+f,space)

    def define_scaling(self):
        super(ConnectMeshAtInterface, self).define_scaling()
        for f in self.get_required_fields():
            self.set_scaling(**{self.lagr_mult_prefix+f:1/test_scale_factor(f)})
            self.set_test_scaling(**{self.lagr_mult_prefix + f: 1 / scale_factor(f)})

    def define_residuals(self):
        for f in self.get_required_fields():
            l, l_test=var_and_test(self.lagr_mult_prefix+f)
            inside, inside_test=var_and_test(f)
            outside, outside_test=var_and_test(f,domain=self.get_opposite_side_of_interface())
            self.add_residual(weak(inside-outside,l_test))
            self.add_residual(weak(l,inside_test))
            self.add_residual(-weak(l,outside_test))

    def before_assigning_equations_postorder(self, mesh:"AnyMesh"):
        fields=self.get_required_fields()
        assert isinstance(mesh,InterfaceMesh)
        for _, f in enumerate(fields):
            self.pin_redundant_lagrange_multipliers(mesh, self.lagr_mult_prefix + f, [f],opposite_interface=[f])


    def after_newton_solve(self):
        fields = self.get_required_fields()
        dim = len(fields)
        mesh=self.get_mesh()
        #mesh=self.get_current_code_generator()._mesh
        assert isinstance(mesh,InterfaceMesh)
        for ninside, noutside in mesh.nodes_on_both_sides():
            if noutside:
                for i in range(dim):
                    noutside.set_x(i,ninside.x(i)) # coincide perfectly. Otherwise problems at remeshing



class StabilizeElementSizeAtMovingInterface(InterfaceEquations):
    """
        Ensures that the size of the interface elements remains the same. 

        This class requires the parent equations to be of type BaseMovingMeshEquations, meaning that if BaseMovingMeshEquations (or subclasses) are not defined in the parent domain, an error will be raised.
    
        Args:
            factor(float): Multiplicative stabilization factor. Default is 1.
    """
    required_parent_type=BaseMovingMeshEquations

    def __init__(self,factor:float):
        super().__init__()
        self.factor=factor

    def define_fields(self):
        self.define_scalar_field("_elemscale","D0")

    def define_residuals(self):
        es,estest=var_and_test("_elemscale")
        _x,xtest=var_and_test("mesh")
        parent=self.get_parent_equations(BaseMovingMeshEquations)
        assert isinstance(parent,BaseMovingMeshEquations)
        self.add_residual(weak(es,estest,coordinate_system=parent.coordsys)) # es=size_lagr/size_euler
        self.add_residual(-weak(1,estest,lagrangian=True,coordinate_system=parent.coordsys))
        self.set_initial_condition("_elemscale",1)
        spatial_square=parent.get_squared_spatial_factor()
        self.add_residual(weak(-self.factor*spatial_square*(es-1),scale_factor("spatial")*div(xtest,lagrangian=False,coordsys=parent.coordsys),coordinate_system=parent.coordsys))

    def after_remeshing(self, eqtree: "EquationTree"):
        mesh=eqtree.get_mesh()
        assert isinstance(mesh,InterfaceMesh)
        index=mesh.get_code_gen().get_code().get_discontinuous_field_index("_elemscale")
        for e in mesh.elements():
            e.internal_data_pt(index).set_value(index,1.0)
        



class VolumeEnforceStorage(ODEEquations):
    """
        Stores the volume that should be enforced. 
         
        Args:
            volume(ExpressionOrNum): The volume that should be enforced.
            scale(Union[Literal["auto"],ExpressionOrNum]): The scale factor. Default is "auto".
    """
    def __init__(self,volume:ExpressionOrNum,scale:Union[Literal["auto"],ExpressionOrNum]="auto"):
        super(VolumeEnforceStorage, self).__init__()
        self.volume=volume
        self.scale=scale

    def define_fields(self):
        if self.scale=="auto":
            scaleE=self.volume
        else:
            scaleE=self.scale
            assert isinstance(scaleE,(Expression,float,int))
        self.define_ode_variable("volume_enforcing",scale=scaleE,testscale=1/scaleE)

    def define_residuals(self):
        _,vltest=var_and_test("volume_enforcing")
        self.add_residual(weak(-self.volume,vltest))


class VolumeEnforcingBoundary(Equations):
    """
        Add these to the boundaries of a volume that should be enforced and pass the var("volume_enforcing",domain="<ode domain>") as arg.
        The volume is enforced by the weak form of the volume constraint:

            V=(1,vltest)=(div(x)/NORM,vltest)=-1/NORM*<x*n,vltest>

        Args:
            storage_var(Expression): The storage variable that contains the volume that should be enforced.
    """
    def __init__(self,storage_var:Expression):
        super(VolumeEnforcingBoundary, self).__init__()
        self.storage_var=storage_var

    def define_residuals(self):
        x,_=var_and_test("mesh")
        xtest_n=testfunction("mesh",dimensional=False)
        n=var("normal")
        coordsys=self.get_coordinate_system()
        dVfactor=coordsys.volumetric_scaling(scale_factor("spatial"),self.get_nodal_dimension())
        norm=1/coordsys.get_actual_dimension(self.get_nodal_dimension())

        self.add_residual(norm*weak(dot(x,n),testfunction(self.storage_var),dimensional_dx=True))
        self.add_residual(norm/dVfactor*weak(self.storage_var,dot(xtest_n,n),dimensional_dx=False))


class EnforceVolumeByPressure(IntegralConstraint):
    """Enforces a volume of an ALE mesh with a (Navier-)Stokes equation by adjusting the pressure until the volume is correct. Usually, you need a free surface for this as well which can deform to grow or shrink until the desired volume is reached by adjusting the pressure.

    Args:
        volume: The desired volume to be enforced. This can be a constant or any expression, e.g. a global parameter or a function of time.
    """
    def __init__(self,volume:ExpressionOrNum,*,ode_storage_domain: Optional[str] = None, only_for_stationary_solve: bool = False, set_zero_on_normal_mode_eigensolve: bool = True, scaling_factor:Union[str,ExpressionNumOrNone]=None):
        if scaling_factor is None:
            scaling_factor=1
        super().__init__(dimensional_dx=True,ode_storage_domain=ode_storage_domain, only_for_stationary_solve = only_for_stationary_solve, set_zero_on_normal_mode_eigensolve= set_zero_on_normal_mode_eigensolve, scaling_factor=scaling_factor,pressure=volume)        
        
    def get_constraint(self,field:str,u:Expression):
        return 1
    
    def define_residuals(self):
        static=not self.get_current_code_generator()._coordinates_as_dofs
        if static:
            raise RuntimeError("EnforceVolumeByPressure only works with moving meshes")
        return super().define_residuals()
        
        
        


class EnforcedInterfacialLaplaceSmoothing(InterfaceEquations):
    """
    This class can be attached to interfaces of a moving mesh. It ensures that the nodes along this boundary will be placed equidistantly along the interface line with respect to the initial configuration.
    This can be helpful if you e.g. add a mesh deformation at a single point, e.g. a contact line, which will deform the bulk elements quite dramatically. If the interfaces are associated with this class, the interfaces will move nicely along, keeping the bulk elements in better shape.
    In order to use it, add it to the interface and use the `with_corners` method to specify the corners of the interface. 
    So if e.g. a line "substrate" of a domain "droplet" starts at a corner with the boundary "axis" and ends at a corner at the boundary "liquid_gas", just add
    
        EnforcedInterfacialLaplaceSmoothing().with_corners("axis","liquid_gas")@"substrate"
        
    to the droplet domain. The same should be done with 
    
        EnforcedInterfacialLaplaceSmoothing().with_corners("axis","substrate")@"liquid_gas"
        
    So that a motion of the contact line will be smooth.
    
    """
    required_parent_type=BaseMovingMeshEquations
    def __init__(self,coordinate_system=cartesian):
        super().__init__()
        self.coordsys=coordinate_system
        self.verbose=True
        
    def define_fields(self):        
        # Get the coordinate space
        space=self._get_combined_element()._assert_codegen()._coordinate_space
        # each interface will need a unique name, so that we have individual fields for each interface
        # This won't be necessary in the general case, since usually, you have corners between two boundaries
        fn=self._get_combined_element()._assert_codegen().get_full_name()
        iname="__".join(fn.split("/")[1:])        
        self.define_scalar_field("_s_fixed_"+iname,space) # Fixed arclength of the reference configuration
        self.define_scalar_field("_s_solved_"+iname,space,testscale=scale_factor("spatial")**2) # solved arclength between start and end points
        self.define_scalar_field("_tang_shift_"+iname,space,scale=1/test_scale_factor("mesh")) # Lagrange multiplier for the tangential shift of the interface nodes, which is used to enforce the tangential movement of the interface nodes along the line.
        
                    

        
    def define_residuals(self):
        if self.get_nodal_dimension()!=2: 
            raise RuntimeError("EnforcedInterfacialLaplaceSmoothing is only implemented for 2d meshes")
        # Bind everything
        fn=self._get_combined_element()._assert_codegen().get_full_name()
        iname="__".join(fn.split("/")[1:])
        s,stest=var_and_test("_s_solved_"+iname)
        
        s0=var("_s_fixed_"+iname)
        l,ltest=var_and_test("_tang_shift_"+iname)
        
        # If you want to use it with normal mode expansion eigenproblems, we don't have to expand it
        s=var("_s_solved_"+iname,only_base_mode=True) # This is the arclength between the start and end points of the interface
        s0=var("_s_fixed_"+iname,only_base_mode=True)
        l=var("_tang_shift_"+iname,only_base_mode=True)
        n=var("normal")
        
        self.add_weak(grad(s,coordsys=self.coordsys),grad(stest,coordsys=self.coordsys),coordinate_system=self.coordsys)
        
        t=vector(-n[1],n[0])  # Tangential vector
        self.add_weak(s-s0,ltest,coordinate_system=self.coordsys) # Ensure that the arclength is equal to the initial arclength 
        # by shifting the nodes tangentially along the line
        self.add_weak(l*t,testfunction("mesh"),coordinate_system=self.coordsys)
        # Fix the reference configuration
        self.set_Dirichlet_condition("_s_fixed_"+iname,True)
        
    def before_assigning_equations_postorder(self, mesh):
        # Just make sure to initialize the arclengths of the interface nodes
        fn=self._get_combined_element()._assert_codegen().get_full_name()
        iname="__".join(fn.split("/")[1:])
        data=mesh.get_problem().get_cached_mesh_data(mesh)
        segs,_=data.get_interface_line_segments()
        coords=data.get_coordinates()                
        nodes=mesh.fill_node_index_to_node_map()
        fixed_index=mesh.has_interface_dof_id("_s_fixed_"+iname)
        dyn_index=mesh.has_interface_dof_id("_s_solved_"+iname)        
        for seg in segs:
            al=0.0
            lastx=coords[0,seg[0]]
            lasty=coords[1,seg[0]]
            for s in seg:
                x,y=coords[0,s],coords[1,s]
                delta=numpy.sqrt((x-lastx)**2+(y-lasty)**2)
                al+=delta
                nodes[s].set_value(nodes[s].additional_value_index(fixed_index),al)
                nodes[s].set_value(nodes[s].additional_value_index(dyn_index),al)
                lastx,lasty=x,y


    def with_corners(self, *corners):
        """Easy wrapper to add corners to the interface equations. These will pin the values of the arclength and deactivate the tangential shift at this nodes"""
        res=Equations()+self
        for c in corners:
            res+=EnforcedInterfacialLaplaceSmoothingCorner()@c
        return res
    
           
    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree, eigensolver, angular_mode, normal_k):
        if angular_mode is None:
            return set()
        
        angular_mode=int(angular_mode)
        fn=self._get_combined_element()._assert_codegen().get_full_name()
        iname="__".join(fn.split("/")[1:])        
        
        if angular_mode==0:
            return set()
        else:            
            info={fn+"/_s_fixed_"+iname,fn+"/_s_solved_"+iname,fn+"/_tang_shift_"+iname}
            print("EnforcedInterfacialLaplaceSmoothing (mode m="+str(angular_mode)+"): Imposed zero tangential shift correction",self.get_current_code_generator().get_full_name(),"for",info)
            return info
        
            

class EnforcedInterfacialLaplaceSmoothingCorner(InterfaceEquations):
    """Helper class to pin the arclength and deactivate the tangential shift at a corner of an interface. This is used in EnforcedInterfacialLaplaceSmoothing.with_corners"""
    required_parent_type=EnforcedInterfacialLaplaceSmoothing
    def define_residuals(self):
        fn=self._get_combined_element()._assert_codegen().get_full_name()
        iname="__".join(fn.split("/")[1:-1])
        self.set_Dirichlet_condition("_s_solved_"+iname,True) # fix the arclength of the corner
        self.set_Dirichlet_condition("_tang_shift_"+iname,0) # deactivate the tangential shift of the corner
