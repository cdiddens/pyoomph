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
 
 

from .. import var_and_test,var
from ..generic.codegen import  InterfaceEquations,Equations,BaseEquations,ODEEquations,FiniteElementCodeGenerator
from ..expressions.generic import ExpressionOrNum,ExpressionNumOrNone,FiniteElementSpaceEnum, grad,nondim, scale_factor,test_scale_factor,Expression,assert_valid_finite_element_space, testfunction,find_dominant_element_space

#Connects one or multiple fields at both sides of the interfaces via Lagrange multipliers
#i.e. it ensures the same Neumann flux on both sides, whereas the magnitude of this flux is given by the Lagrange multiplier
#which is automatically chosen that way that the condition <inner>=<outer> is satisfied.
from ..meshes.mesh import MeshFromTemplateBase,Element

from ..typings import *
if TYPE_CHECKING:
    from ..meshes.remesher import RemesherBase,RemesherPointEntry
    from ..expressions.coordsys import BaseCoordinateSystem
    from ..meshes import AnyMesh
    from ..generic.problem import Problem,EquationTree
    


# TODO Check this
def get_interface_field_connection_space(inside_space:Union[FiniteElementSpaceEnum,Literal[""]],outside_space:Union[FiniteElementSpaceEnum,Literal[""]],use_highest_space:bool=False)->Union[FiniteElementSpaceEnum,Literal[""]]:
    if outside_space == "":
        return inside_space
    elif inside_space == "":
        return outside_space    
    if outside_space[0]!=inside_space[0]:
        raise RuntimeError("TODO: Think about what space is lower/higher ") #TODO: Is e.g. D2 lower or higher than C2TB? hard to tell
    space_order=["D2TB","C2TB","D2","C2","D1TB","C1TB","D1","C1"]
    for sp in space_order:
        if inside_space==sp:
            if outside_space==sp or use_highest_space:
                return sp
            else:
                return outside_space
        elif outside_space==sp:
            return inside_space

    raise RuntimeError("Should not happen: Cannot get field connection space for "+inside_space+" and "+outside_space)

class ConnectFieldsAtInterface(InterfaceEquations):
    """
    Enforces continuity of fields at the interface. The fields are connected via Lagrange multipliers. The Lagrange multipliers are automatically chosen such that the condition <inner>=<outer> is satisfied. 

    Args:
        fields: Either a single field name or a list of field names when the fields have the same name on both sides. Alternatively, a dict mapping each inner to each outer name if the fields have different names.
        lagr_mult_prefix: Prefix for the Lagrange multipliers. Defaults to "_lagr_conn_".
        use_highest_space: Flag indicating whether to use the highest space for the Lagrange multipliers. If the fields have different spatial discretizations on both sides, we have to decide which space to use for the Lagrange multipliers. If this flag is set to True, the highest space will be used. Defaults to False.
    """
    def __init__(self,fields:Union[str,Dict[str,str],List[str]],*,lagr_mult_prefix:str="_lagr_conn_",use_highest_space:bool=False):
        super(ConnectFieldsAtInterface, self).__init__()
        self.lagr_mult_prefix=lagr_mult_prefix
        self.use_highest_space=use_highest_space
        if not isinstance(fields,dict):
            if isinstance(fields,list):
                self.fields={x:x for x in fields}
            elif isinstance(fields,str): #type:ignore
                self.fields={fields:fields}
            else:
                raise ValueError("Unsupported argument for fields: "+str(self.fields))
        else:
            self.fields=fields.copy()


    def define_fields(self):
        for finner,fouter in self.fields.items():
            if self.get_opposite_side_of_interface(raise_error_if_none=False) is None:
                raise RuntimeError("Cannot connect any fields at the interface if no opposite side is present")
            inside_space=self.get_parent_domain().get_space_of_field(finner)
            if inside_space=="":
                raise RuntimeError("Cannot connect field "+finner+" at the interface, since it cannot find in the inner domain")
            opppdom=self.get_opposite_side_of_interface().get_parent_domain()
            assert opppdom is not None
            outside_space=opppdom.get_space_of_field(fouter)
            if outside_space=="":
                raise RuntimeError("Cannot connect field "+fouter+" at the interface, since it cannot find in the outer domain")
            inside_space=assert_valid_finite_element_space(inside_space)
            outside_space=assert_valid_finite_element_space(outside_space)            
            space=get_interface_field_connection_space(inside_space,outside_space,use_highest_space=self.use_highest_space) 
            space=assert_valid_finite_element_space(space)
            self.define_scalar_field(self.lagr_mult_prefix+finner+"_"+fouter,space,scale=1/test_scale_factor(finner)) 



    def define_residuals(self):
        dx = self.get_dx(use_scaling=False)
        for finner,fouter in self.fields.items():
            l, l_test=var_and_test(self.lagr_mult_prefix+finner+"_"+fouter)
            inside, inside_test=var_and_test(finner)
            outside, outside_test=var_and_test(fouter,domain=self.get_opposite_side_of_interface())
            scal=self.get_scaling(finner)
            self.add_residual((inside-outside)/scal*l_test*dx) #TODO: Possibly nodal connection?
            self.add_residual(l*inside_test*dx)
            self.add_residual(-l*outside_test*dx)

    def before_assigning_equations_postorder(self, mesh: "AnyMesh") -> None:
        for finner,fouter in self.fields.items():
            lname=self.lagr_mult_prefix+finner+"_"+fouter
            self.pin_redundant_lagrange_multipliers(mesh,lname,finner,fouter)

        super().before_assigning_equations_postorder(mesh)
        
    def with_removed_overconstraining(self,*corners:str):
        return self+sum([ConnectFieldsAtInterfaceRemoveOverconstraining(self.fields)@corner for corner in corners])


class ConnectFieldsAtInterfaceRemoveOverconstraining(InterfaceEquations):
    required_parent_type = ConnectFieldsAtInterface
    def __init__(self,fields:Union[str,Dict[str,str],List[str]]):
        super(ConnectFieldsAtInterfaceRemoveOverconstraining, self).__init__()
        self.lagr_mult_prefix = "_lagr_conn_"
        if not isinstance(fields,dict):
            if isinstance(fields,list):
                self.fields={x:x for x in fields}
            elif isinstance(fields,str): #type:ignore
                self.fields={fields:fields}
            else:
                raise ValueError("Unsupported argument for fields: "+str(self.fields))
        else:
            self.fields=fields.copy()

    def define_residuals(self):
#        parent=self.get_parent_equations()
        for finner, fouter in self.fields.items():
            self.set_Dirichlet_condition(self.lagr_mult_prefix+finner+"_"+fouter,0)

class SpatialErrorEstimator(Equations):

    """
    Spatial error estimators are used to estimate where a mesh should be refined. You can either pass variable name(s) and numerical factor(s), e.g.

            SpatialErrorEstimator(u=1,v=10)
    
    In that case, the jumps of the gradients grad(u) and 10*grad(v) will be used as error estimators. 
    Alternatively, you can also provide custom expressions as estimators, e.g. for discontinuous fields, it might be better to just add

            SpatialErrorEstimator(5*var("u"))

    so that the jump in "u" is used, after weighting by the factor 5, as error estimator.
    Error estimators expressions must be nondimensional.
    
    for_which controls whether these error estimators are used for the base solution, potential eigenfunctions or both.
    """

    def __init__(self,*fluxes:Union[str,Expression],for_which:Literal["both","base","eigen"]="both",**kwargs:ExpressionOrNum):
        super(SpatialErrorEstimator, self).__init__()
        self.fluxes:Dict[Union[str,Expression],ExpressionOrNum]={x:1.0 for x in fluxes}
        for lhs,rhs in kwargs.items():
            self.fluxes[lhs]=rhs
        if for_which=="both":
            self.for_base_solution=True
            self.for_eigenfunction=True
        elif for_which=="base":
            self.for_base_solution=True
            self.for_eigenfunction=False
        elif for_which=="eigen":
            self.for_base_solution=False
            self.for_eigenfunction=True
        else:
            raise ValueError("Unsupported value for for_which: "+str(for_which))

    def define_error_estimators(self):
        for flux,factor in self.fluxes.items():
            if isinstance(flux,str):
                if flux=="normal":
                    jflux=nondim("normal") #Normal is not derived
                elif flux=="mesh":
                    jflux=grad(nondim("mesh"),nondim=True,lagrangian=True)
                else:
                    jflux=grad(nondim(flux),nondim=True)
            else:
                jflux=flux
            self.add_spatial_error_estimator(factor*jflux,for_base=self.for_base_solution,for_eigen=self.for_eigenfunction)

    def get_information_string(self) -> str:
        return ", ".join(map(str,self.fluxes))


class SpatialIntegrationOrder(Equations):
    """
    Sets the order of the Gauss-Lengendre quadrature for spatial integration. 
    The default is depends on the element space, can be adjusted problem-wide by setting the attribute :py:attr:`~pyoomph.generic.problem.Problem.spatial_integration_order`, or locally by adding this equation to the equations.
    
    Note that not all orders are supported for all element spaces. Pyoomph will select the closest supported order.

    Args:
        order: The desired order of the Gauss-Legendre quadrature (2,3,4,5 are supported by most, but not all finite element spaces).
    """
    def __init__(self,order:int):
        super(SpatialIntegrationOrder, self).__init__()
        self.order=order

    def define_additional_functions(self):
        self.get_current_code_generator()._set_integration_order(self.order)


class RefineToLevel(Equations):
    """
    Refine elements to a certain level. If the level is set to "max", the elements will be refined to the maximum level set by e.g. :py:attr:`~pyoomph.generic.problem.Problem.max_refinement_level`.
    """
    def __init__(self, level:Union[Literal["max"],int]="max"):
        super(RefineToLevel, self).__init__()
        self.level = level

    def calculate_error_overrides(self):
        mesh=self.get_current_code_generator()._mesh 
        assert mesh is not None
        must_refine = 100 * mesh.max_permitted_error
        may_not_unrefine = 0.5 * (mesh.max_permitted_error+mesh.min_permitted_error)
        for e in mesh.elements():
            #e._elemental_error_max_override
            if self.level!="max":
                assert isinstance(self.level,int)
                blk=e
                while blk.get_bulk_element() is not None:
                    blk=blk.get_bulk_element()
                lvl=blk.refinement_level()
                if lvl>=self.level: 
                    e._elemental_error_max_override = max(e._elemental_error_max_override,may_not_unrefine)
                    continue
            e._elemental_error_max_override=must_refine


RefineToMaxLevel=RefineToLevel

# An "equation" that will refine elements whenever they are larger than a non-dimensional element size threshold
class RefineMaxElementSize(Equations):
    def __init__(self,max_nondim_cartesian_size:float):
        super(RefineMaxElementSize, self).__init__()
        self.max_nondim_size=max_nondim_cartesian_size

    def calculate_error_overrides(self):
        mesh = self.get_current_code_generator()._mesh  # get the mesh the equation is defined on
        assert mesh is not None
        must_refine = 100 * mesh.max_permitted_error # To force a refinement, we just set a error sufficiently large
        may_not_unrefine = 0.5 * (mesh.max_permitted_error + mesh.min_permitted_error) # We also prevent unrefinement if the element is still quite large
        unrefine_factor=2**(mesh.get_element_dimension()) # Factor an element growth or shrinks when it is (un)refined: 2**(element_dimension)
        for e in mesh.elements():
            size=e.get_current_cartesian_nondim_size() # get the current size of the element
            if size>self.max_nondim_size: # if it is too large
                e._elemental_error_max_override = must_refine # Setting large error -> invoking refinemenet
            elif size*unrefine_factor>self.max_nondim_size: # If an unrefinement would cause a refinement in the next step
                e._elemental_error_max_override = max(e._elemental_error_max_override, may_not_unrefine) # prevent unrefinement



# Refine an element to a level specified by a callback with the element
class RefineAccordingToElement(Equations):
    def __init__(self, level_func:Callable[[Element],int],prevent_unrefinement:bool=True):
        super(RefineAccordingToElement, self).__init__()
        self.level_func = level_func
        self.prevent_unrefinement=prevent_unrefinement
        

    def calculate_error_overrides(self):
        mesh=self.get_current_code_generator()._mesh 
        assert mesh is not None
        must_refine = 100 * mesh.max_permitted_error
        may_not_unrefine = 0.5 * (mesh.max_permitted_error+mesh.min_permitted_error)
        
        for e in mesh.elements():            
            blk=e
            while blk.get_bulk_element() is not None:
                blk=blk.get_bulk_element()
            currlevel=blk.refinement_level()
            desired_level=self.level_func(e)                 
            if currlevel<desired_level:
                e._elemental_error_max_override=must_refine
            elif currlevel>=desired_level and self.prevent_unrefinement:
                e._elemental_error_max_override = max(e._elemental_error_max_override,may_not_unrefine)




class RemeshingOptions:
    """
    A class containing the remeshing sensitivity options to be used with the :py:class:`~pyoomph.equations.generic.RemeshWhen` class.
    
    Args:
        max_expansion: Maximum expansion factor of an element before remeshing is invoked.
        min_expansion: Minimum expansion factor of an element before remeshing is invoked.
        min_solves_before_remesh: Minimum number of sucessful solves before remeshing is invoked.
        reinit_initial_size_after_one_step: Flag indicating whether to reinitialize the initial size after one step.
        active: Flag indicating whether the remeshing is active.
        min_quality_decrease: Minimum quality decrease of an element before remeshing is invoked.
        on_invalid_triangulation: Flag indicating whether to remesh if the triangulation is invalid.
    """    
    def __init__(self,max_expansion:float=1.75,min_expansion:float=0.6,min_solves_before_remesh:int=0,reinit_initial_size_after_one_step:bool=False,active:bool=True,min_quality_decrease:float=0.2,on_invalid_triangulation:bool=False):
        self.max_expansion=max_expansion
        self.min_expansion=min_expansion
        self.min_solves_before_remesh=min_solves_before_remesh
        self.reinit_initial_size_after_one_step=reinit_initial_size_after_one_step
        self.min_quality_decrease=min_quality_decrease
        self.on_invalid_triangulation=on_invalid_triangulation
        self.active=active

    def keys(self) -> List[str]:
        return ['max_expansion', 'min_expansion','min_solves_before_remesh','reinit_initial_size_after_one_step','active','min_quality_decrease']

    def __getitem__(self, key:str)->Any:
        return vars(self)[key] #type:ignore


class RemeshWhen(Equations):
    """
    Checks whether the mesh has been deformed to much based on either the passed :py:class:`~pyoomph.equations.generic.RemeshingOptions` object or the passed parameters. If the mesh has been deformed too much, it will be marked for remeshing. The remeshing will be done after the current Newton solve, followed by a subsequent interpolation from the previous mesh.
    
    Args:
        remeshing_opts: An object containing the remeshing sensitivity.
        max_expansion: Maximum expansion factor of an element before remeshing is invoked.
        min_expansion: Minimum expansion factor of an element before remeshing is invoked.
        min_solves_before_remesh: Minimum number of sucessful solves before remeshing is invoked.
        reinit_initial_size_after_one_step: Flag indicating whether to reinitialize the initial size after one step.
        active: Flag indicating whether the remeshing is active.
        min_quality_decrease: Minimum quality decrease of an element before remeshing is invoked.
        on_invalid_triangulation: Flag indicating whether to remesh if the triangulation is invalid.
    """
    def __init__(self,remeshing_opts:Optional[RemeshingOptions]=None,*,max_expansion:Optional[float]=None,min_expansion:Optional[float]=None,min_solves_before_remesh:Optional[int]=0,reinit_initial_size_after_one_step:Optional[bool]=False,active:bool=True,min_quality_decrease:Optional[float]=None,on_invalid_triangulation:bool=False):

        super(RemeshWhen, self).__init__()
        if isinstance(remeshing_opts,RemeshingOptions):
            self.max_expansion=remeshing_opts.max_expansion
            self.min_expansion=remeshing_opts.min_expansion
            self.min_solves_before_remesh=remeshing_opts.min_solves_before_remesh
            self.reinit_initial_size_after_one_step=remeshing_opts.reinit_initial_size_after_one_step
            self.min_quality_decrease=remeshing_opts.min_quality_decrease
            self.on_invalid_triangulation=remeshing_opts.on_invalid_triangulation
            self.active=remeshing_opts.active
        else:
            self.max_expansion = max_expansion
            self.min_expansion = min_expansion
            self.min_solves_before_remesh = min_solves_before_remesh
            self.reinit_initial_size_after_one_step = reinit_initial_size_after_one_step
            self.on_invalid_triangulation=on_invalid_triangulation
            self.active = active
            self.min_quality_decrease = min_quality_decrease

        if self.max_expansion and self.max_expansion<=1:
            raise ValueError("max_expansion must be >1")

        if self.min_expansion and self.min_expansion>=1:
            raise ValueError("min_expansion must be <1")


    def after_newton_solve(self):
        need_remesh=False
        mesh=self.get_my_domain()._mesh 
        assert mesh is not None
        if not self.active:
            return

        if isinstance(mesh,MeshFromTemplateBase):
            since_remesh=mesh._solves_since_remesh 
            if self.min_solves_before_remesh is not None:
                if self.min_solves_before_remesh>=since_remesh:
                    if since_remesh==1:
                        if self.reinit_initial_size_after_one_step:
                            for e in mesh.elements():
                                e.set_initial_cartesian_nondim_size(e.get_current_cartesian_nondim_size())
                                e.set_initial_quality_factor(e.get_quality_factor())

                    return


        meshname:str=mesh.get_name()

        if self.max_expansion or self.min_expansion or self.min_quality_decrease:
            for e in mesh.elements():
                if self.max_expansion or self.min_expansion:
                    isize=e.get_initial_cartesian_nondim_size()
                    csize=e.get_current_cartesian_nondim_size()
                    ratio=csize/isize
                    if self.max_expansion and  ratio>self.max_expansion:
                        print("Remeshing invoked from "+meshname+" by an element expanded by a factor of "+str(ratio))
                        need_remesh=True
                        break
                    elif self.min_expansion and  ratio<self.min_expansion:
                        print("Remeshing invoked from " + meshname + " by an element shrunken by a factor of " + str(
                            ratio))
                        need_remesh=True
                        break
                if self.min_quality_decrease:
                    iquality=e.get_initial_quality_factor()
                    if iquality>0:
                        cquality=e.get_quality_factor()
                        ratio=cquality/iquality
                        #print(ratio)
    #                    exit()
                        if ratio<self.min_quality_decrease:
                            print("Remeshing invoked from " + meshname + " by an element lost quality by a factor of " + str(ratio))
                            need_remesh = True
                            break

        if self.on_invalid_triangulation and not need_remesh:
            from matplotlib import tri
            mshcache=mesh.get_problem().get_cached_mesh_data(mesh,nondimensional=False,tesselate_tri=True)
            coordinates=mshcache.get_coordinates()
            try:
                triang = tri.Triangulation(coordinates[0], coordinates[1], mshcache.elem_indices)            
                tf=triang.get_trifinder()
            except:
                need_remesh=True


        if not isinstance(mesh,MeshFromTemplateBase) or  mesh._templatemesh.remesher is None: 
            raise RuntimeError("You added a RemeshWhen object to the equations of '"+meshname+"'. However, the corresponding MeshTemplate does not have the property 'remesher' set.")

        if need_remesh:            
            self.get_current_code_generator().get_problem()._domains_to_remesh.add(mesh._templatemesh) 


class RemeshMeshSize(BaseEquations):
    """
    Can be added to boundaries or corners to set the local mesh size. The size can be a constant or a function of the point. If the size is a function, it must be a function of the point and return a float.

    Args:
        size: The local size, i.e. the typical nondimensional length of an element here. Can be a constant or a function of the point.
    """
    def __init__(self,size:Optional[Union[float,Callable[["RemesherPointEntry"],float]]]=None):
        super(RemeshMeshSize, self).__init__()
        self.size=size

    def setup_remeshing_size(self,remesher:"RemesherBase",preorder:bool):
        if self.size and preorder:
            my_name=self.get_current_code_generator().get_full_name()
            splt=my_name.split("/")
            if len(splt)==2 or len(splt)==3:
                pts=remesher._get_points_by_phys_name(my_name) 
                for l in pts:
                    for p in l:
                        if callable(self.size):
                            p.set_sizes.append(self.size(p))
                        else:
                            p.set_sizes.append(self.size)
            else:
                raise RuntimeError("Cannot use RemeshMeshSize on a domain, only at interfaces or corners")
            #print(self.get_current_code_generator().get_full_name())
            #exit()


class ProjectExpression(Equations):
    def __init__(self,scale:Union[ExpressionOrNum,str]=1,space:FiniteElementSpaceEnum="C2",field_type:Literal["scalar","vector"]="scalar",**projs:ExpressionOrNum):
        super(ProjectExpression, self).__init__()
        self.space:FiniteElementSpaceEnum=space
        self.scale=scale
        if isinstance(scale,str):
            self.scale=scale_factor(scale)
        self.field_type=field_type
        self.projs=projs.copy()

    def define_fields(self):
        for n,_ in self.projs.items():
            if self.field_type=="scalar":
                self.define_scalar_field(n,self.space,scale=self.scale,testscale=1/self.scale)
            elif self.field_type=="vector":
                self.define_vector_field(n,self.space,scale=self.scale,testscale=1/self.scale)
            else:
                raise ValueError("Unsupported field type "+self.field_type)

    def define_residuals(self):
        import pyoomph.expressions.generic
        for n,e in self.projs.items():
            f,ftest=var_and_test(n)
            self.add_residual(pyoomph.expressions.generic.weak(f,testfunction(n,dimensional=False)/scale_factor(n)))
            self.add_residual(pyoomph.expressions.generic.weak(-e,testfunction(n,dimensional=False)/scale_factor(n)))

class InitialCondition(BaseEquations):
    """
    Class representing initial conditions for a set of equations. If the initial conditions dpend on time, i.e. on ``var("time)``, it will be used to initialize the history steps before the first step. Otherwise, by default, the first time step will be calculated by a first order step.

    Args:
        degraded_start: Flag indicating whether to use degraded start (i.e. first order time stepping in the first step) or not. Defaults to "auto", meaning we degrade if the initial condition does not depend on time.
        IC_name: Name of the initial condition. Defaults to an empty string, which are the default initial conditions.
        **kwargs: Keyword arguments representing the initial conditions for each variable.
    """

    def __init__(self, *, degraded_start: Union[bool, Literal["auto"]] = "auto", IC_name: str = "", **kwargs: ExpressionOrNum):
        super(InitialCondition, self).__init__()
        self._ics = {n: 0 + v for n, v in kwargs.items()}
        self._ic_name = IC_name
        self._degraded_start = degraded_start

    def get_information_string(self):    		
        return ",".join([str(k) + "=" + str(v) for k, v in self._ics.items()])

    def define_residuals(self):
        for n, val in self._ics.items():
            assert isinstance(self._degraded_start, bool) or self._degraded_start == "auto"
            self.set_initial_condition(n, val, degraded_start=self._degraded_start, IC_name=self._ic_name)


class TemporalErrorEstimator(BaseEquations):
    """
    Adding temporal error estimators to the equations. Each field can have a different factor. If you have e.g. field "u" and "v", add a 
    
        ``TemporalErrorEstimator(u=1,v=10)``
        
    to the equations to weight the error estimator for "u" with 1 and "v" with 10. Errors in "v" will be more weighted then.
    
    Args:
        fieldfactors: A dict of field names and their corresponding temporal error weighting factors for temporal error estimation.    
    """

    def __init__(self, **fieldfactors: float):
        super(TemporalErrorEstimator, self).__init__()
        self.fieldfactors = fieldfactors.copy()

    def define_error_estimators(self):       
        for f, v in self.fieldfactors.items():                        
            self.set_temporal_error_factor(f, v)
            
            

class SubstituteVarByExpression(BaseEquations):
    """
    If not defined in this domain, all statements of ``var(name)`` in the weak formulations in this domain will be replaced by the given expressions.
    If you e.g. use a :py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionEquations`, you can combine it with ``SubstituteVarByExpression(velocity=vector(ux,uy))`` to introduce a prescribed velocity.
    
    You can do the same on the problem level, i.e. as fallback for all equations, by the :py:class:`~pyoomph.generic.problem.Problem`-method :py:meth:`~pyoomph.generic.problem.Problem.define_named_var`. With this class, you can do it on particular domains only.
    
    Attributes:
        also_on_interfaces: A flag indicating whether the substitution should also be applied on interfaces.
        def_vars: A dictionary containing the variables and their corresponding expressions to be subsitutes.
    """
    def __init__(self, also_on_interfaces: bool = True, **def_vars):
        super().__init__()
        self.def_vars = def_vars.copy()
        self.also_on_interfaces = also_on_interfaces
        
    def define_fields(self):    
        for name, val in self.def_vars.items():
            self.define_field_by_substitution(name, val, also_on_interface=self.also_on_interfaces)

class EquationCompilationFlags(BaseEquations):
    """
    Allows to control some flags for code generation when added to other equations.

    Args:
        analytical_position_jacobian (Optional[bool]): Flag indicating whether to use analytical position Jacobian (default is True).
        analytical_jacobian (Optional[bool]): Flag indicating whether to use analytical Jacobian (default is True).
        warn_on_large_numerical_factor (Optional[bool]): Flag indicating whether to warn on large numerical factor (default is False).
        debug_jacobian_epsilon (Optional[float]): Epsilon value for debugging Jacobian. Will calculate both FD and analytical Jacobian and compares them. If the difference is larger than this epsilon, it will print a warning. Only use for debugging.
        ccode_expression_mode (Optional[str]): Mode for C-code expression, e.g. "expand", "normal", "collect_common_factors", "factor".
        with_adaptivity (Optional[bool]): Flag indicating whether you allow for spatial adaptivity.
    """    
    def __init__(self,analytical_position_jacobian:Optional[bool]=None,analytical_jacobian:Optional[bool]=None,warn_on_large_numerical_factor:Optional[bool]=None,debug_jacobian_epsilon:Optional[float]=None,ccode_expression_mode:Optional[str]=None,with_adaptivity:Optional[bool]=None):
        super(EquationCompilationFlags, self).__init__()
        self.analytical_position_jacobian=analytical_position_jacobian
        self.analytical_jacobian=analytical_jacobian
        self.warn_on_large_numerical_factor=warn_on_large_numerical_factor
        self.debug_jacobian_epsilon=debug_jacobian_epsilon
        self.ccode_expression_mode=ccode_expression_mode
        self.with_adaptivity=with_adaptivity

    def before_compilation(self,codegen:"FiniteElementCodeGenerator"):
        if self.analytical_position_jacobian is not None:
            codegen.analytical_position_jacobian=self.analytical_position_jacobian
        if self.analytical_jacobian is not None:
            codegen.analytical_jacobian=self.analytical_jacobian
        if self.debug_jacobian_epsilon is not None:
            codegen.debug_jacobian_epsilon=self.debug_jacobian_epsilon
        if self.ccode_expression_mode is not None:
            codegen.ccode_expression_mode=self.ccode_expression_mode        
        if self.with_adaptivity is not None:
            codegen.with_adaptivity=self.with_adaptivity





    def define_fields(self):
        if self.warn_on_large_numerical_factor is not None:
            self.get_current_code_generator().warn_on_large_numerical_factor=self.warn_on_large_numerical_factor


class SetCoordinateSystem(Equations):
    """
    Set the default coordinate system for the current equations. It will override the coordinate system set on the problem level.

    Args:
        coord_sys (BaseCoordinateSystem): Pass an coordinate system instance from the `pyoomph.expressions.coordsys` module.
    """
    def __init__(self,coord_sys:"BaseCoordinateSystem"):
        super(SetCoordinateSystem, self).__init__()
        self.coord_sys=coord_sys

    def define_fields(self):
        master = self._get_combined_element()
        master._coordinate_system=self.coord_sys




class ApplyMappingOnAddedResidual(BaseEquations):    #
    def __init__(self,mapping:Callable[[str,"Expression"],Union["Expression",Dict[str,"Expression"]]]=lambda destination,expr:{destination:expr}):
        super(ApplyMappingOnAddedResidual, self).__init__()
        self.mapping=mapping

    def define_fields(self):
        master=self._get_combined_element()
        master._residual_mapping_functions.append(self.mapping)


class LocalExpressions(Equations):
    """
    Local expressions are additional expressions for output, evaluated on the nodes of the mesh. 
    They are not solved, but only calculated for output.
    Since it works node-wise, it might give problems, e.g. for 1/r terms at the axis of symmetry.
    An alternative is to use the `ProjectExpressions` class, which calculates such expressions by projection. However, these are degrees of freedom, i.e. it will be slower.

    Args:
        **local_expressions (ExpressionOrNum): A dict of expressions to be evaluated on the nodes of the mesh for output only.
    """
    def __init__(self, **local_expressions:ExpressionOrNum):
        super(LocalExpressions, self).__init__()
        self.local_expressions = {k:v for k,v in local_expressions.items()}

    def define_additional_functions(self):
        for k,v in self.local_expressions.items():
            self.add_local_function(k, v )
            


class BackupHistoryExpressions(Equations):
    def __init__(self,*,space:FiniteElementSpaceEnum="C2",local_expression_format="_history_expr_{name}",**exprs:Expression):
        super().__init__()
        self.history_fields=exprs
        self.space=space
        self.update=True
        self.local_expression_format=local_expression_format
        
    def define_fields(self):
        for name,expr in self.history_fields.items():
            self.define_scalar_field(name,space=self.space)
            
    def define_residuals(self):
        for name,expr in self.history_fields.items():
            self.set_Dirichlet_condition(name,True)
            self.add_local_function(self.local_expression_format.format(name=name),expr)
            
    def update_history(self,mesh:"AnySpatialMesh"):
        exprs=mesh.list_local_expressions()
        nodalfields=mesh.get_nodal_field_indices()
        for name,expr in self.history_fields.items():            
            idx=exprs.index(self.local_expression_format.format(name=name))
            vals=mesh.evaluate_local_expression_at_nodes(idx,True,False )            
            if name not in nodalfields:
                raise Exception(f"Field {name} not found in nodal fields. This must be improved for e.g. interface fields or discontinuous fields (DL/D0).")
            nindx=nodalfields[name]            
            for i,n in enumerate(mesh.nodes()):
                n.set_value(nindx,vals[i])
            
            
    def after_newton_solve(self):
        if not self.update:
            return
        mesh=self.get_mesh()
        self.update_history(mesh)
        
        


class DependentIntegralObservable:
    def __init__(self,func:Callable[...,ExpressionOrNum],*argnames:str):
        super(DependentIntegralObservable, self).__init__()
        self.func=func
        self.argnames=[*argnames]

    def __call__(self, *args:ExpressionOrNum) -> ExpressionOrNum:
        return self.func(*args)



class IntegralObservables(Equations):
    """
    Integral expressions will be evaluated by spatial integration over the mesh domain.
    E.g. an 
    
        IntegralObservables(volume=1)
        
    will calculate the volume by integration over the mesh domain. In e.g. axisymmetry, the factor 2*pi*r will be included.
    Also, the output is dimensional, i.e. if you have set the scaling to a metric quantity, you will get a result in cubic meters here.
    When combined with an IntegralObservablesOutput object, they will be written to an output file.
    
    You can also introduce dependent IntegralObservables. If you have a field "u", you can calculated the average of "u" on the domain by
    
        IntegralObservables(_denom=1,_u_integral=var("u"),u_avg=lambda _u_integral,_denom:_u_integral/_denom)
        
    Here, _denom will be the integral over 1 and _u_integral will be the integral over "u". The function u_avg will be evaluated as average.
    The parameter names in the lambda function must match the names of the integral observables. The underscore prevents writing the helper observables to output.

    Parameters:
        _coordinate_system (Optional[BaseCoordinateSystem]): The coordinate system to use. Defaults to None, i.e. the one of the equations or the problem.
        **integral_observables (Union[ExpressionOrNum, Callable[..., ExpressionOrNum]]): Integral observables to be added.
    """
    def __init__(self,_coordinate_system:Optional["BaseCoordinateSystem"]=None,_lagrangian:bool=False, **integral_observables:Union[ExpressionOrNum,Callable[...,ExpressionOrNum]]):
        super(IntegralObservables, self).__init__()
        self.integral_observables = {k:v for k,v in integral_observables.items() if not callable(v)}
        self.dependent_funcs={k:v for k,v in integral_observables.items() if callable(v)}
        self._coordinate_system=_coordinate_system
        self._lagrangian=_lagrangian

    def define_additional_functions(self):
        if self._coordinate_system is None:
            dx = self.get_dx(lagrangian=self._lagrangian)
        else:
            dx=self.get_dx(coordsys=self._coordinate_system,lagrangian=self._lagrangian)
        for k,v in self.integral_observables.items():
            #import _pyoomph
            #_pyoomph.set_verbosity_flag(1)
            self.add_integral_function(k, v * dx)
            #_pyoomph.set_verbosity_flag(0)
        for k,v in self.dependent_funcs.items():
            self.add_dependent_integral_function(k,v)



class ExtremumObservables(Equations):
    """You can add these to continuum Equations to find minima and maxima of given expressions. 
    If you want to find the minimum/maximum of a scalar quantity "u", you can add an ``ExtremumObservables("u")`` or, alternatively, give it a name like ``ExtremumObservables(my_name_for_u=var("u"))``.
    More than one argument can be passed to register multiple extremum observables. For e.g. the maximum of a norm of a vectorial variable v, you can add ``ExtremumObservables(v_norm=square_root(dot(var("v"),var("v"))))``.  
    
    Once registered, you can evaluate the extremum values by calling the ``evaluate_maximum`` or ``evaluate_minimum`` method of the corresponding mesh (available via ``problem.get_mesh(...)``)

    Args:
        Either strings as positional arguments or expressions as keyword arguments.
    """
    def __init__(self,*direct_vars:str,**named_extrema):
        super().__init__()
        self.named_extrema=named_extrema.copy()
        for varname in direct_vars:
            if not isinstance(varname,str):
                raise ValueError("ExtremumObservables must be either constructed with strings as positional args or expressions as keyword args, i.e. e.g. ExtremumObservables('u') to monitor the extrema of var('u') or ExtremumObservables(u_sqr=var('u')**2) to monitor the extrema of u**2")
            self.named_extrema[varname]=var(varname)
        

    def add_extremum_function(self,name,expr):
        import _pyoomph
        master = self._get_combined_element()
        cg = master._assert_codegen()
        if not (isinstance(expr,int) or isinstance(expr,float) or isinstance(expr,_pyoomph.Expression)) and  callable(expr):
            expr=expr()
        if isinstance(expr,(int,float)): # Does not really make sense here
            expr=_pyoomph.Expression(expr)
        cg._register_extremum_function(name, expr)        
        
    def define_residuals(self):
        for name,expr in self.named_extrema.items():
            self.add_extremum_function(name,expr)

class ODEObservables(ODEEquations):
    """
    Adds observables to ODEs. Observables are just expressions which will be also written to the output file when combined with an :py:class:`~pyoomph.output.generic.ODEFileOutput` object.
    If you have e.g. a harmonic oscillator (with variable y) and want to observe the total energy, you can add the total energy as an observable:
    
        ``HarmonicOscillator(...)+ODEObservables(Etot=1/2*partial_t(y)**2+1/2*omega**2*y**2)``
        
    Args:
        **ode_observables: Observables to be added, identified by the name.
    """

    def __init__(self, **ode_observables:ExpressionOrNum):
        super(ODEObservables, self).__init__()
        self.ode_observables = {k:v for k,v in ode_observables.items() if not callable(v)}
        self.dependent_funcs={k:v for k,v in ode_observables.items() if callable(v)}

    def define_additional_functions(self):
        dx = nondim("dx")
        for k,v in self.ode_observables.items():
            self.add_integral_function(k, v * dx)
        for k,v in self.dependent_funcs.items():
            self.add_dependent_integral_function(k,v)



class Scaling(BaseEquations):
    """
    Set the scales used for nondimensionalization on the equation level. It will override the scales set on the problem level by Problem.set_scaling(...=...).

    Args:
        **scales (ExpressionOrNum): Used scales for nondimensionalization.
    """
    def __init__(self,**kwargs:Union[ExpressionOrNum,str]):
        super(Scaling, self).__init__()
        self.scales=kwargs.copy()
    def define_scaling(self):
        super(Scaling, self).define_scaling()
        self.set_scaling(**self.scales)

class TestScaling(BaseEquations):    
    """
    Set the scales of the test functions used for nondimensionalization on the equation level.

    Args:
        **testscales (ExpressionOrNum): Used scales for nondimensionalization of the test functions.
    """    
    __test__ = False
    def __init__(self,**kwargs:Union[ExpressionOrNum,str]):
        super(TestScaling, self).__init__()
        self.scales=kwargs.copy()

    def define_scaling(self):
        super(TestScaling, self).define_scaling()
        self.set_test_scaling(**self.scales)



class ElementSpace(Equations):
    """Sets the element space of the current equations. By default, pyoomph will take the highest order element space of all fields defined on the domain.
    With this class, you can e.g. set the element space to second order ("C2"), although you only have first-order fields ("C1") defined.

    Args:
        space (FiniteElementSpaceEnum): Set the desired element space for the equations of the domain.
    """
    def __init__(self,space:FiniteElementSpaceEnum):
        super(ElementSpace, self).__init__()
        self.space=space

    def define_fields(self):        
        cg=self.get_current_code_generator()
        if self.space not in {"C2TB","C2","C1TB","C1"}:
            raise ValueError("Can only set the coordinate space to either C2TB, C2, C1TB or C1")
        cg._coordinate_space = find_dominant_element_space(cg._coordinate_space,self.space)




# Constaints of the form: integral(u+A)*dx=B
# Where A is given by get_integral_contribution
# and B is given by get_global_residual_contribution
# Used for Average and Integral constraints
# If physical dimensions are set, it only works if the these are set on problem level by problem.set_scaling(...=...), not if set in the equations
class _AverageOrIntegralConstraintBase(Equations):
    def __init__(self,*,ode_storage_domain:Optional[str]=None,only_for_stationary_solve:bool=False,set_zero_on_normal_mode_eigensolve:bool=True,scaling_factor:Union[str,ExpressionNumOrNone]=None, **kwargs:"ExpressionOrNum"):
        super().__init__()
        self.ode_storage_domain=ode_storage_domain        
        self.constraints=kwargs.copy()
        self.dimensional_dx=False
        self.only_for_stationary_solve=only_for_stationary_solve
        self.set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve
        self.scaling_factor=scaling_factor
        if isinstance(self.scaling_factor,str):
            self.scaling_factor=scale_factor(self.scaling_factor)

    def get_global_dof_storage_name(self, pathname: Optional[str] = None):
        if self.ode_storage_domain is None:
            return super().get_global_dof_storage_name(pathname)
        else:
            return self.ode_storage_domain
        
    def after_fill_dummy_equations(self, problem: "Problem", eqtree: "EquationTree",pathname:str,elem_dim:Optional[int]=None):
        from ..generic.codegen import GlobalLagrangeMultiplier
        if len(self.constraints)==0:
            return super().after_fill_dummy_equations(problem,eqtree,pathname,elem_dim)        
        odestorage=self.get_global_dof_storage_name(pathname=pathname)  
        add_eqs=None      
        for field,integral_value in self.constraints.items():
            scale_correction=problem.get_scaling(field) if self.scaling_factor is None else self.scaling_factor
            testscale=1
            if self.dimensional_dx:
                if elem_dim is None:
                    elem_dim=self.get_element_dimension()
                
                coordsys=eqtree._codegen.get_coordinate_system()
                #coordsys=self.get_combined_equations().get_coordinate_system()                
                testscale/=(0+coordsys.volumetric_scaling(problem.get_scaling("spatial"),elem_dim))
                        
            
            new_eq=GlobalLagrangeMultiplier(only_for_stationary_solve=self.only_for_stationary_solve,set_zero_on_normal_mode_eigensolve=self.set_zero_on_normal_mode_eigensolve, **{field:self.get_global_residual_contribution(field)/scale_correction})+Scaling(**{field:1})+TestScaling(**{field:testscale})
            add_eqs=new_eq if add_eqs is None else add_eqs+new_eq
            
        problem._equation_system+=add_eqs@odestorage
        return super().after_fill_dummy_equations(problem,eqtree,pathname,elem_dim)        

    def define_residuals(self):
        odestorage=self.get_global_dof_storage_name()        
        for field,integral_value in self.constraints.items():
            u,utest=var_and_test(field)
            l,ltest=var_and_test(field,domain=odestorage)
            #self.add_weak(u/scale_factor(field),ltest,dimensional_dx=self.dimensional_dx)
            #self.add_weak(self.get_integral_contribution(field)/scale_factor(field),ltest,dimensional_dx=self.dimensional_dx)
            self.add_weak(self.get_constraint(field,u),ltest,dimensional_dx=self.dimensional_dx)
            self.add_weak(l,utest/test_scale_factor(field),dimensional_dx=False)

    def get_constraint(self,field:str,u:Expression):
        return (u-self.get_integral_contribution(field))/scale_factor(field)

    def get_global_residual_contribution(self,field:str)-> ExpressionOrNum:
        raise RuntimeError("Must be implemented")
    
    def get_integral_contribution(self,field:str)-> ExpressionOrNum:
        raise RuntimeError("Must be implemented")    
    


class IntegralConstraint(_AverageOrIntegralConstraintBase):
    """
    Enforces the value of a field to have a fixed integral value by a global Lagrange multiplier.
    If you have e.g. a field "u", you can enforce the integral of "u" to be 1 by adding
    
        IntegralConstraint(u=1)
                
    Args:
        dimensional_dx (bool): Flag indicating whether the constraint is defined in dimensional or non-dimensional form.
        ode_storage_domain (Optional[str]): The storage domain for the ODEs, will default to some generated name.
        only_for_stationary_solve (bool): Flag indicating whether the constraint is only applied during stationary solves.
        set_zero_on_normal_mode_eigensolve (bool): Flag indicating whether the constraint is set to zero during angular eigensolves.
        scaling_factor (Union[str, ExpressionNumOrNone]): The scaling factor for the constraint.
        **kwargs (ExpressionOrNum): Constraints as name=value pairs.
    """       
    def __init__(self, *, dimensional_dx:bool=True,ode_storage_domain: Optional[str] = None, only_for_stationary_solve: bool = False, set_zero_on_normal_mode_eigensolve: bool = True, scaling_factor:Union[str,ExpressionNumOrNone]=None, **kwargs: ExpressionOrNum):
        super().__init__(ode_storage_domain=ode_storage_domain, only_for_stationary_solve=only_for_stationary_solve, set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve, scaling_factor=scaling_factor, **kwargs)
        self.dimensional_dx=dimensional_dx
    
    def get_global_residual_contribution(self,field:str) -> ExpressionOrNum:
        return -self.constraints[field] # Globally subtract the integral value
    
    def get_integral_contribution(self,field:str)-> ExpressionOrNum:
        return 0 # No contribution during the spatial integral

class AverageConstraint(_AverageOrIntegralConstraintBase):
    """
    Enforces the value of a field to have a fixed averaged value by a global Lagrange multiplier.
    If you have e.g. a field "u", you can enforce the average of "u" to be 1 by adding
    
        AverageConstraint(u=1)
                
    Args:
        dimensional_dx (bool): Flag indicating whether the constraint is defined in dimensional or non-dimensional form.
        ode_storage_domain (Optional[str]): The storage domain for the ODEs, will default to some generated name.
        only_for_stationary_solve (bool): Flag indicating whether the constraint is only applied during stationary solves.
        set_zero_on_normal_mode_eigensolve (bool): Flag indicating whether the constraint is set to zero during angular eigensolves.
        scaling_factor (Union[str, ExpressionNumOrNone]): The scaling factor for the constraint.
        **kwargs (ExpressionOrNum): Constraints as name=value pairs.
    """           
    def get_global_residual_contribution(self,field:str)-> ExpressionOrNum:
        return 0 # No global contribution
    
    def get_integral_contribution(self,field:str)-> ExpressionOrNum:
        return self.constraints[field] # Consider the offset for the average
    
