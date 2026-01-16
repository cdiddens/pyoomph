from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import BaseMovingMeshEquations

class BaseSolidConstitutiveLaw:
    """
        Base class for solid constitutive laws. The method get_sigma must be implemented in derived classes.

        Args:
            use_subexpressions (bool, optional): Use subexpressions for the tensor entries. Defaults to True.
    """
    def __init__(self,use_subexpressions=True):        
        self.use_subexpressions=use_subexpressions
        
    def _subexpression(self,expr):
        if self.use_subexpressions:
            return subexpression(expr)
        else:
            return expr
        
    def is_incompressible(self):
        """
        If this returns True, the constitutive law is incompressible. This means that the determinant of the deformation gradient is equal to the determinant of the undeformed configuration. The solid equations will then introduce a pressure variable that is used to enforce this condition.
        """
        return False
    
    def get_Gij(self):     
        """
        Returns the covariant metric tensor of the deformed configuration. 
        """           
        x=var("mesh")
        return self._subexpression(matproduct(transpose(grad(x,lagrangian=True)),grad(x,lagrangian=True)))
    
    def get_G_up_ij(self):
        # G^ij=(G^(-1))_ij
        Gij=self.get_Gij()        
        return self._subexpression(inverse_matrix(Gij,fill_to_vector_dim_3=True,skip_empty_rows_and_cols=True))
        
    
    def get_gij(self,dim:int,isotropic_growth_factor:ExpressionOrNum=1):
        """
        Returns the covariant metric tensor of the undeformed configuration. This is the identity matrix in Cartesian coordinates.
        """
        X=var("lagrangian")
        grf=pow(isotropic_growth_factor,rational_num(2,dim))
        return self._subexpression(grf*matproduct(transpose(grad(X,lagrangian=True)),grad(X,lagrangian=True)))
        
    
    def get_gammaij(self,dim:int,isotropic_growth_factor:ExpressionOrNum=1):
        """
        Returns the Green strain tensor 
        """
        return self._subexpression((self.get_Gij()-self.get_gij(dim,isotropic_growth_factor))/2)
    
    def get_sigma(self,dim:int,isotropic_growth_factor:ExpressionNumOrNone=1,pressure_var:Optional[Expression]=None):
        """
        Returns the second Piola-Kirchhoff stress tensor. The method must be implemented in derived classes.

        Args:
            dim: Element dimension. 
            isotropic_growth_factor: Expression or numerical value representing the isotropic growth factor. Defaults to 1.
            pressure_var (Optional[Expression], optional): Pressure variable for incompressible cases. Defaults to None.
        
        """
        raise RuntimeError("get_sigma must be implemented in the derived class")
    
class IncompressibleSolidConstitutiveLaw(BaseSolidConstitutiveLaw):
    """
    Base class for incompressible solid constitutive laws. The method get_sigma must be implemented in derived classes.     
    """
    def is_incompressible(self):
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
    """
    def __init__(self, E:ExpressionOrNum=1, nu:ExpressionOrNum=0.4,use_subexpressions=True):
        
        super().__init__()
        self.E=E
        self.nu=nu
        self.use_subexpressions=use_subexpressions
        

    def get_sigma(self,dim:int,isotropic_growth_factor:ExpressionOrNum=1,pressure_var:Optional[Expression]=None):
        se=lambda x : self._subexpression(x)
        Gup=self.get_G_up_ij()
        gammakl=self.get_gammaij(dim,isotropic_growth_factor)
        if dim is None:
            raise RuntimeError("dim must be specified")
        
        E=lambda i,j,k,l : se(self.E/(1+self.nu))*(se(self.nu/(1-2*self.nu))*Gup[i,j]*Gup[k,l]+(Gup[i,k]*Gup[j,l]+Gup[i,l]*Gup[j,k])/2)
        sigma_up_ij=lambda i,j : sum(E(i,j,k,l)*gammakl[k,l] for k in range(3) for l in range(3))
        sigma=matrix([[sigma_up_ij(i,j) for j in range(3)] for i in range(3)])
        return se(sigma)
        
class IncompressibleHookeanSolidConstitutiveLaw(IncompressibleSolidConstitutiveLaw):
    def __init__(self,E:ExpressionOrNum=1,use_subexpressions=True):
        super().__init__(use_subexpressions=use_subexpressions)
        self.E=E

    def get_sigma(self,dim:int,isotropic_growth_factor:ExpressionNumOrNone=1,pressure_var:Optional[Expression]=None):
        se=lambda x : self._subexpression(x)
        Gup=self.get_G_up_ij()
        gammakl=self.get_gammaij(dim,isotropic_growth_factor)
        if dim is None:
            raise RuntimeError("dim must be specified")
        bar_sigma_up_ij=lambda i,j : se(sum((Gup[i,k]*Gup[j,l]+Gup[i,l]*Gup[j,k])*gammakl[k,l] for k in range(3) for l in range(3)))
        if pressure_var is None:
            raise RuntimeError("pressure_var must be specified")
        return -pressure_var*Gup+self.E/3*matrix([[bar_sigma_up_ij(i,j) for j in range(3)] for i in range(3)])
    

class BaseDeformableSolidEquations(BaseMovingMeshEquations):
    def __init__(self, mass_density:ExpressionOrNum=0,bulkforce:ExpressionOrNum=0,coordinate_space = None,first_order_time_derivative=False,scale_for_FSI:bool=False,modulus_for_scaling:ExpressionOrNum=None,isotropic_growth_factor:ExpressionOrNum=1,pressure_space:FiniteElementSpaceEnum="DL",with_error_estimator=False,overdamped:bool=False):        
        super().__init__(coordinate_space, False, None)
        self.scale_for_FSI=scale_for_FSI
        self.mass_density=mass_density
        self.modulus_for_scaling=modulus_for_scaling
        self.bulkforce=bulkforce
        self.first_order_time_derivative=first_order_time_derivative
        self.isotropic_growth_factor=isotropic_growth_factor
        self.pressure_space=pressure_space
        self.pressure_name="pressure" 
        self.with_error_estimator=with_error_estimator
        self.overdamped=overdamped
        
    def is_incompressible(self):
        return False
        
        
    def define_fields(self):
        super().define_fields()                
        if self.is_incompressible():
            self.define_scalar_field(self.pressure_name,self.pressure_space)
        if self.first_order_time_derivative:
            if self.coordinate_space is None:
                raise RuntimeError("coordinate_space must be specified for first order time derivative")
            if self.overdamped:
                raise RuntimeError("overdamped=True is not compatible with first_order_time_derivative=True, since overdamped already reduces the time derivative order to one.")
            self.define_vector_field("dt_mesh",self.coordinate_space,scale=scale_factor("spatial")/scale_factor("temporal"),testscale=scale_factor("temporal")/scale_factor("spatial"))
                    
    def define_scaling(self):        
        self.set_scaling(mesh= scale_factor("spatial"))
        if self.scale_for_FSI:
            self.set_test_scaling(mesh=scale_factor("spatial")/scale_factor("pressure"))
        else:
            if self.modulus_for_scaling is None:
                self.set_test_scaling(mesh=1/(scale_factor("mass_density")/scale_factor("temporal")**2*scale_factor("spatial")))
            else:
                self.set_test_scaling(mesh=1/self.modulus_for_scaling*scale_factor("spatial"))    
                

    def before_mesh_to_mesh_interpolation(self, eqtree, interpolator):
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
    def __init__(self, E:ExpressionOrNum,nu:ExpressionNumOrNone, mass_density = 0, bulkforce = 0, coordinate_space=None, first_order_time_derivative=False, scale_for_FSI = False, modulus_for_scaling = None,isotropic_growth_factor:ExpressionOrNum=1,pressure_space:FiniteElementSpaceEnum="DL",with_error_estimator=False):
        super().__init__(mass_density, bulkforce, coordinate_space, first_order_time_derivative, scale_for_FSI, modulus_for_scaling,isotropic_growth_factor=isotropic_growth_factor,pressure_space=pressure_space,with_error_estimator=with_error_estimator)
        self.E,self.nu=E,nu
        
    def is_incompressible(self):
        return self.nu is None or is_zero(self.nu-0.5) or is_zero(self.nu-rational_num(1,2))
        
    def define_residuals(self):
        if not is_zero(self.isotropic_growth_factor-1):
            raise RuntimeError("LinearElasticitySolidEquations does not support isotropic growth factor != 1. Use DeformableSolidEquations instead.")
        x,xtest=var_and_test("mesh")
        X=var("lagrangian")
        u=x-X
        
        eps=sym(grad(u,lagrangian=True))
        
        if self.is_incompressible():
            tau=self.E/(1+self.nu)*(eps)-var(self.pressure_name)*identity_matrix()  # Pressure is introduced as a Lagrange multiplier
            self.add_weak(div(u,lagrangian=True),testfunction(self.pressure_name),lagrangian=True)
        else:
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
    """
    # TODO: Bulk force density in Lagrangian or Eulerian coordinates? Make two or a flag?
    def __init__(self, constitutive_law:BaseSolidConstitutiveLaw, mass_density:ExpressionOrNum=0,bulkforce:ExpressionOrNum=0,coordinate_space = None,first_order_time_derivative=False,pressure_space:FiniteElementSpaceEnum="DL",with_error_estimator=False,isotropic_growth_factor:ExpressionOrNum=1,modulus_for_scaling:ExpressionOrNum=None,scale_for_FSI:bool=False,overdamped:bool=False):        
        super().__init__(mass_density=mass_density,bulkforce=bulkforce,coordinate_space=coordinate_space, first_order_time_derivative=first_order_time_derivative,scale_for_FSI=scale_for_FSI,modulus_for_scaling=modulus_for_scaling,isotropic_growth_factor=isotropic_growth_factor,pressure_space=pressure_space,with_error_estimator=with_error_estimator,overdamped=overdamped)
        self.constitutive_law=constitutive_law        
                
    def is_incompressible(self):
        return self.constitutive_law.is_incompressible()
        
    def define_fields(self):
        super().define_fields()                
        
        # Allow to access the deformed mass density
        if self.is_incompressible():
            rho_deformed=self.mass_density
        else:
            dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
            detG=self.constitutive_law._subexpression(determinant(self.constitutive_law.get_Gij()))
            detg=self.constitutive_law._subexpression(determinant(self.constitutive_law.get_gij(dim,self.isotropic_growth_factor)))
            rho_deformed=self.constitutive_law._subexpression(self.mass_density*square_root(detg/detG))
        self.define_field_by_substitution("deformed_mass_density",rho_deformed,also_on_interface=True)


        
    def define_residuals(self):
        x,xtest=var_and_test("mesh")
        super().define_residuals()  # Call the base class method to define the residuals for the inertia and bulk force terms
        
        dim=self.get_coordinate_system().get_actual_dimension(self.get_nodal_dimension())
        pvar=var(self.pressure_name) if self.constitutive_law.is_incompressible() else None        
        sigma=self.constitutive_law.get_sigma(dim,self.isotropic_growth_factor,pvar)       
                                
        self.add_weak((matproduct(grad(x,lagrangian=True),sigma)),self.isotropic_growth_factor*grad(xtest,lagrangian=True),lagrangian=True)
                
        if self.constitutive_law.is_incompressible():
            detG=self.constitutive_law._subexpression(determinant(self.constitutive_law.get_Gij()))
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
        self.define_vector_field("_mesh_connection","C2",testscale=1/scale_factor("spatial"),scale=scale_factor("spatial")**2)
        self.define_vector_field("_velo_connection","C2",testscale=scale_factor("temporal")/scale_factor("spatial"),scale=scale_factor("pressure"))
        
    def define_residuals(self):
        from pyoomph.equations.navier_stokes import StokesEquations
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
        
    def before_assigning_equations_postorder(self, mesh):
        comps=[ x[-1] for x in self.get_combined_equations()._vectorfields["_mesh_connection"] ]
        for direct in comps:
            self.pin_redundant_lagrange_multipliers(mesh,"_mesh_connection_"+direct,["mesh_"+direct])
            self.pin_redundant_lagrange_multipliers(mesh,"_velo_connection_"+direct,["velocity_"+direct])