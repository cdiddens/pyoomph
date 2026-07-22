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
 
import weakref
from .. import _pyoomph_core as _pyoomph

from ..meshes.mesh import assert_spatial_mesh,InterfaceMesh,ODEStorageMesh
from ..expressions import AxisymmetryBreakingCoordinateSystem,AxisymmetricCoordinateSystem, find_dominant_element_space, scale_factor, vector,matrix,evaluate_in_domain,testfunction,weak,var,nondim,Expression,rational_num,minimize_functional_derivative,time_derivative_of_integral

# from ..expressions import var, get_global_symbol, nondim, vector, testfunction, scale_factor, cartesian, partial_t
from ..expressions.coordsys import ODECoordinateSystem, BaseCoordinateSystem
from ..expressions.units import assert_dimensional_value
import numpy

from ..typings import *
if TYPE_CHECKING:
    from .problem import Problem,_DofSelector 
    from ..expressions import ExpressionOrNum,ExpressionNumOrNone,FiniteElementSpaceEnum,OptionalCoordinateSystem,TimeSteppingScheme
    from ..meshes.mesh import AnySpatialMesh,AnyMesh,ODEStorageMesh
    from ..solvers.generic import GenericEigenSolver
    from ..meshes.remesher import RemesherBase
    from ..meshes.interpolator import BaseMeshToMeshInterpolator


def _check_for_valid_var_name(name:str,for_domain:bool):
    typ="domain" if for_domain else "variable" 
    if name=="":
        raise ValueError("Empty "+typ+" name")    
    elif not name.isidentifier():
        raise ValueError(typ+" names may not contain anything else than [A-Z], [a-z], _ and [0-9] (not beginning with a number). Happened at the name: '"+str(name)+"'")
    elif for_domain and name.find("__")>0:
        if not name.startswith("_meshwide_"):
            raise ValueError("Domain names may not have double underscores __, except at the beginning. Happened at the name: '"+str(name)+"'")


class FiniteElementCodeGenerator(_pyoomph.FiniteElementCode):
    def __init__(self):
        super(FiniteElementCodeGenerator, self).__init__()
        self._code:_pyoomph.DynamicBulkElementInstance | None=None
        self._name:str | None=None
        self._mesh:"AnyMesh | None"=None
        self._dependent_integral_funcs:dict[str,Callable[...,ExpressionOrNum]]={}
        self._dependent_integral_funcs_is_vector_helper:dict[str,bool] = {}
        self._external_ode_fields:dict[str,tuple["FiniteElementCodeGenerator",str]]={}
        self._named_numerical_factors:dict[str,"ExpressionOrNum"]={} #To monitor some factors and see whether the scaling is good or not
        self._dummy_codegen_for_internal_facets:FiniteElementCodeGenerator | None=None
        self._dummy_codegen_for_internal_facets_bulk:FiniteElementCodeGenerator | None=None
        self._dummy_codegen_for_internal_facets_bulk_bulk:FiniteElementCodeGenerator | None=None
        self._dummy_codegen_for_internal_facets_bulk_opp:FiniteElementCodeGenerator | None=None

        self._fields_defined_on_my_domain:dict[str,"FiniteElementSpaceEnum"]={}

        self._custom_domain_name:str | None=None

    def get_default_timestepping_scheme(self,dt_order:int):
        return self.get_equations().get_default_timestepping_scheme(dt_order,cg=self)

    def get_code(self)->_pyoomph.DynamicBulkElementInstance:
        assert self._code is not None
        return self._code

    def get_problem(self)->"Problem":
        return self._get_problem() #type:ignore

    def get_equations(self)->"BaseEquations":
        res=super().get_equations()
        assert isinstance(res,BaseEquations)
        return res

    def get_domain_name(self)->str:
        from ..meshes.mesh import InterfaceMesh        
        if self._custom_domain_name is not None:
            return self._custom_domain_name
        elif self._name is not None:
            return self._name
        if self._mesh is None:            
            return super(FiniteElementCodeGenerator, self).get_domain_name()
        elif isinstance(self._mesh,InterfaceMesh):
            return self._mesh.get_name()
        else:
            return self._mesh._name 

    def get_full_name(self)->str:
        res:str
        if self._mesh is None:
            res=super(FiniteElementCodeGenerator, self).get_domain_name()
        elif isinstance(self._mesh,InterfaceMesh):
            res=self._mesh.get_name()
        else:
            res=self._mesh._name  
        pdom=self.get_parent_domain()
        if pdom is not None:
            res=pdom.get_full_name()+"/"+res
        return res

    def get_integral_dx(self,use_scaling:bool,lagrangian:bool,coordsys:_pyoomph.CustomCoordinateSystem | None) -> Expression:
        return self.get_equations().get_dx(use_scaling=use_scaling,lagrangian=lagrangian,coordsys=coordsys)

    def _is_ode_element(self):
        eqs = self.get_equations()
        if eqs._is_ode()==True:
            return True
        else:
            return False

    def _register_external_ode_linkage(self,myfieldname:str,odecodegen:_pyoomph.FiniteElementCode,odefieldname:str):
        assert isinstance(odecodegen,FiniteElementCodeGenerator)
        #print("LINKAGE",myfieldname,odecodegen,odefieldname)
        self._external_ode_fields[myfieldname]=(odecodegen,odefieldname)

    def _perform_external_ode_linkage(self):
        #print("Performing external ODE linkage")
        for myfield,linkinfo in self._external_ode_fields.items():
            #print("info",myfield,linkinfo[0],linkinfo[1])
            source_name=linkinfo[0].get_full_name()+"/"+linkinfo[1]            
            #print("source name",source_name)
            #print("code",linkinfo[0].get_code())
            di = linkinfo[0].get_code().get_discontinuous_field_index(linkinfo[1])
            #print("di",di)            
            assert linkinfo[0]._mesh is not None
            #print("mesh",linkinfo[0]._mesh)            
            #print("nelement",linkinfo[0]._mesh.nelement())            
            #print("elempt0",linkinfo[0]._mesh.element_pt(0))            
            data = linkinfo[0]._mesh.element_pt(0).internal_data_pt(di)
            #print("data",data)            
            index=0
            #print("linking")            
            self.get_code().link_external_data(myfield, data, index,source_name)
            #print("done")            


    def _register_dependent_integral_function(self,name:str,func:Callable[...,"ExpressionOrNum"],vector_helper:bool=False):
        self._dependent_integral_funcs[name]=func
        if vector_helper:
            self._dependent_integral_funcs_is_vector_helper[name]=True

    def _resolve_based_on_domain_name(self,domainname:str)->_pyoomph.FiniteElementCode | None:
        res=self.get_problem()._equation_system.get_by_path(domainname)
        if not res:
            return None
        return res._codegen


    def _set_problem(self,problem:"Problem | None"): #type:ignore
        super()._set_problem(problem) #type:ignore

    def get_element_dimension(self) -> int:
        return self.dimension

    def calculate_error_overrides(self):
        eqs = self.get_equations()
        oldcg = eqs._get_current_codegen()
        eqs._set_current_codegen(self)
        #        print("EQS",eqs)
        eqs.calculate_error_overrides()
        eqs._set_current_codegen(oldcg)

    @overload
    def get_scaling(self, n:str,testscale:Literal[False]=...)->"Expression": ...

    @overload
    def get_scaling(self, n:str,testscale:Literal[True])->"Expression": ...

    @overload
    def get_scaling(self, n:str,testscale:Literal["from_parent"])->"Expression | None": ...

    def get_scaling(self, n:str,testscale:bool | Literal["from_parent"]=False)->"Expression | None": #type:ignore

        #print("OVERRIDE GET SCALING")
        eqs=self.get_equations()
        oldcg=eqs._get_current_codegen()
        eqs._set_current_codegen(self)
#        print("EQS",eqs)
        #print("SCAL", n, eqs, self,testscale)
        if testscale=="from_parent":
            res=eqs.get_scaling(n,testscale="from_parent")
        elif testscale==True:
            res=eqs.get_scaling(n,testscale=True)
        elif testscale==False:
            res=eqs.get_scaling(n,testscale=False)
        else:
            raise RuntimeError("Should not end here")
        #print("RET",res,testscale)
        eqs._set_current_codegen(oldcg)

        #print("EXPANDING FOR",res)
        #resn=self.expand_placeholders(res,False)
        #print("EXP ", n, res,"->",resn)

        if res is None:
            assert testscale=="from_parent"
            return res
        if not isinstance(res,_pyoomph.Expression):
            res=_pyoomph.Expression(res)
        return res

    def on_apply_boundary_conditions(self,mesh:"AnyMesh"):
        eqs=self.get_equations()
        oldcg = eqs._get_current_codegen()
        eqs._set_current_codegen(self)
        eqs.on_apply_boundary_conditions(mesh)
        eqs._set_current_codegen(oldcg)

    def get_coordinate_system(self)->"BaseCoordinateSystem":
        eqs = self.get_equations()
        if eqs._is_ode()==True: 
            return _ode_coordinate_system
        if eqs._coordinate_system is not None:  
            return eqs._coordinate_system  
        else:
            return self.get_problem().get_coordinate_system()

    def expand_additional_field(self, name:str, dimensional:bool, expression:_pyoomph.Expression,in_domain:_pyoomph.FiniteElementCode,no_jacobian:bool,no_hessian:bool,where:str)->"Expression":
        
        #print("CODEGEN: Expand additional field", name, dimensional, expression, in_domain, no_jacobian, no_hessian, where)
        eqs=self.get_equations()
        oldcg = eqs._get_current_codegen()
        eqs._set_current_codegen(self)
        #print("----------------EXPAND ",name,dimensional,in_domain,self)
        res=eqs.expand_additional_field( name, dimensional, expression,in_domain,no_jacobian,no_hessian,where)
        eqs._set_current_codegen(oldcg)
        return res

    def add_named_numerical_factor(self,**kwargs:"ExpressionOrNum"):
        for k,v in kwargs.items():
            if isinstance(v,_pyoomph.Expression):
                v=self.expand_placeholders(v,True)            
            self._named_numerical_factors[k]=v

    def expand_additional_testfunction(self, name:str, expression:"Expression",in_domain:_pyoomph.FiniteElementCode)->"Expression":
        eqs=self.get_equations()
        oldcg = eqs._get_current_codegen()
        eqs._set_current_codegen(self)                        
        res= eqs.expand_additional_testfunction(name,expression,in_domain)
        eqs._set_current_codegen(oldcg)
        return res

    def get_parent_domain(self)->"FiniteElementCodeGenerator | None":
        pd=self._get_parent_domain()
        if pd is None:
            return None
        else:
            assert isinstance(pd,FiniteElementCodeGenerator)
            return pd


    def get_default_spatial_integration_order(self)->int:
        eqs=self.get_equations()
        if isinstance(eqs, ODEEquations):
            return 0
        pdom=self.get_parent_domain()
        if pdom is not None:
            return pdom.get_default_spatial_integration_order()
        else:
            return self.get_problem().get_default_spatial_integration_order()


    def _transfer_my_fields_to_dummy_codegen(self,dummy:"FiniteElementCodeGenerator"):  
        raise NotImplementedError("This function is not implemented yet. It should transfer all fields defined on this codegen to the dummy codegen, so that the dummy codegen can be used for internal facets and still have access to all fields defined on the parent domain")  
        print("Transfer called", self.get_parent_domain(), dummy)
        if self.get_parent_domain() is not None:
           pself=self.get_parent_domain()
           if pself.get_parent_domain() is not None:
               ppself=pself.get_parent_domain()
               for fieldname,space in ppself._fields_defined_on_my_domain.items():
                if fieldname not in dummy._fields_defined_on_my_domain.keys():
                    dummy._fields_defined_on_my_domain[fieldname]=space
                    print("Transferring parent parent field",fieldname,space,"to",dummy)


           for fieldname,space in pself._fields_defined_on_my_domain.items():
            if fieldname not in dummy._fields_defined_on_my_domain.keys():
                dummy._fields_defined_on_my_domain[fieldname]=space
                print("Transferring parent field",fieldname,space,"to",dummy)

        for fieldname,space in self._fields_defined_on_my_domain.items():
            if fieldname not in dummy._fields_defined_on_my_domain.keys():
                dummy._fields_defined_on_my_domain[fieldname]=space
                print("Transferring field",fieldname,space,"to",dummy)
        exit()

class ScalingException(Exception):
    def __init__(self, msg:str, obj:"BaseEquations | None"=None):
        fullmsg = msg
        if obj is not None:
            fullmsg = fullmsg + "\nDefined Scales (on object " + str(obj) + "):\n"
            for k, v in obj._scaling.items():
                if isinstance(v, str):
                    fullmsg = fullmsg + "\t" + k + " -> " + v + " = " + str(obj.get_scaling(k)) + "\n"
                else:
                    fullmsg = fullmsg + "\t" + k + " = " + str(obj.get_scaling(k)) + "\n"
        super().__init__(fullmsg)

import inspect


class BaseEquations(_pyoomph.Equations):
    """
    These are the parent class for both :py:class:`~pyoomph.generic.codegen.ODEEquations` and :py:class:`~pyoomph.generic.codegen.Equations`. You will rarely have to use this base class directly.
    """
    with_exception_info:bool=True


    def __iter__(self)->Iterator["BaseEquations"]:
        return self._iter_helper(set())

    def _iter_helper(self,visited:set[int])->Iterator["BaseEquations"]:
        # Leaf equations: a bare (non-combined) equation just yields itself once.
        # CombinedEquations overrides __iter__ directly to iterate over its _subelements instead.
        if id(self) not in visited:
            visited.add(id(self))
            yield self

    def __new__(cls, *args:Any, **kwargs:Any):
        new_instance = super(BaseEquations, cls).__new__(cls, *args, **kwargs)
        #print("WITH EX INFO",cls.with_exception_info)
        if cls.with_exception_info:            
            stack_trace = inspect.stack()
            created_at = '%s:%d' % (stack_trace[1][1], stack_trace[1][2])
            new_instance._created_at = created_at 
        else:
            new_instance._created_at=None
        return new_instance

    def change_output_directory(self,newdir:str,eqtree:"EquationTree"):
        pass
    
    def add_weak(self,a:"ExpressionOrNum",b:"str | ExpressionOrNum",*,dimensional_dx:bool=False,lagrangian:bool=False,coordinate_system:"OptionalCoordinateSystem"=None,destination:str | None=None):
        """
        Adds the weak contribution ``(a, b)`` (i.e. the integral of ``a`` times the test function ``b``) to the residuals.

        Args:
            a: Expression to be tested.
            b: Test function, either passed directly or as the name of the field to test.
            dimensional_dx: Whether to use the dimensional (as opposed to the nondimensional) integration measure.
            lagrangian: Whether to integrate over the Lagrangian (undeformed) instead of the Eulerian domain.
            coordinate_system: Optional coordinate system override for the integration.
            destination: Optional residual destination for multiple residuals. Defaults to ``None``.

        Returns:
            self, to allow chaining further ``add_weak``/``add_residual`` calls.
        """
        if isinstance(b,str):
            b=testfunction(b)
        self.add_residual(weak(a,b,dimensional_dx=dimensional_dx,coordinate_system=coordinate_system,lagrangian=lagrangian),destination=destination)
        return self
    
    def add_dweak_dt(self,a:"ExpressionOrNum",b:"str | ExpressionOrNum",*,dimensional_dx:bool=False,lagrangian:bool=False,coordinate_system:"OptionalCoordinateSystem"=None,destination:str | None=None,scheme:"TimeSteppingScheme"="BDF1"):
        if isinstance(b,str):
            b=testfunction(b)
        self.add_residual(time_derivative_of_integral(weak(a,b,dimensional_dx=dimensional_dx,coordinate_system=coordinate_system,lagrangian=lagrangian),scheme=scheme),destination=destination)
        return self
    
    def add_functional_minimization(self,F:"ExpressionOrNum",with_respect_to:Expression | list[Expression] | None=None,*,dimensional_dx:bool=False,dimensional_testfunctions:bool=True,lagrangian:bool=False,coordinate_system:"OptionalCoordinateSystem"=None,destination:str | None=None):
        """Adds the weak form of the functional minimization of W=integral(F dOmega) to the equations.

        Args:
            F (ExpressionOrNum): Integrand of the functional.
            with_respect_to (Optional[Union[Expression,List[Expression]]], optional): Optionally only derive with respect to all shape functions appearing in the listed expressions. Defaults to None, meaning all shape functions in F.
            dimensional_dx (bool, optional): Consider spatial scaling in the weak form integral. Defaults to False.
            dimensional_testfunctions (bool, optional): Expand by dimensional testfunctions. Defaults to True.
            lagrangian (bool, optional): Weak formulation is integrated over the Lagrangian domain. Defaults to False.
            coordinate_system (OptionalCoordinateSystem, optional): Optional coordinate system. Defaults to the equations' coordinate system, then parent equations and eventually the problem coordinate system. Defaults to None.
            destination (Optional[str], optional): Residual destination identifier. Defaults to None.

        Returns:
            BaseEquations: Returns self for chaining.
        """
        dF=minimize_functional_derivative(F, only_with_respect_to=with_respect_to, dimensional_testfunctions=dimensional_testfunctions,coordinate_system=coordinate_system,lagrangian=lagrangian,dimensional_dx=dimensional_dx)
        self.add_residual(dF,destination=destination)
        return self

    def get_dx(self, use_scaling:bool=True, lagrangian:bool=False,coordsys:_pyoomph.CustomCoordinateSystem | None=None)->"Expression":
        master = self._get_combined_element()  # TODO This does not allow for dx on individual coordinate systems
        if coordsys is None:
            coordsys = master.get_coordinate_system()
        assert isinstance(coordsys,BaseCoordinateSystem)
        return coordsys.integral_dx(self.get_nodal_dimension(), self.get_element_dimension(), use_scaling,master.get_scaling("spatial"), lagrangian)

    def after_fill_dummy_equations(self,problem:"Problem",eqtree:"EquationTree",pathname:str,elem_dim:int | None=None):
        pass        

    def get_parent_domain(self)->FiniteElementCodeGenerator | None:
        """
        If this domain is a subdomain of another domain, i.e. a boundary, this function returns the parent domain. Otherwise, it returns None.
        """
        master = self._get_combined_element()
        cg=master._assert_codegen()
        res=cg._get_parent_domain()
        if res is None:
            return res
        assert isinstance(res,FiniteElementCodeGenerator)
        return res

    def get_list_of_vector_fields(self,codegen:"FiniteElementCodeGenerator")->list[dict[str,list[str]]]:
        return []
    
    def get_problem(self)->"Problem":
        mst=self.get_combined_equations()
        if mst._problem is not None:
            return mst._problem
        else:
            cg=mst._assert_codegen()
            return cg.get_problem()

    @property
    def _problem(self)->"Problem | None":
        # Stored as a weakref, not a strong reference: a pure-Python Equations object has
        # no C++-side get_problem() to fall back on (unlike meshes/codegens/MeshTemplate),
        # and is never explicitly cleared during Problem.release() - a strong reference
        # here would keep the Problem alive forever via e.g. EquationTree._equations._problem.
        return self._problem_wr() if self._problem_wr is not None else None

    @_problem.setter
    def _problem(self,p:"Problem | None"):
        self._problem_wr=weakref.ref(p) if p is not None else None

    def get_creation_info(self)->str | None:
        return self._created_at #type:ignore

    def add_exception_info(self,exception:Exception)->Exception:
        if self.with_exception_info:            
            import sys
            errmsg = '\nRaised from ' + str(self.__class__.__name__) + ' object instantiated at: "' + str(self.get_creation_info()) + '"'
            raise type(exception)(str(exception) + ' %s' % errmsg).with_traceback(sys.exc_info()[2])
        else:
            raise exception


    def get_azimuthal_r0_info(self):
        """Returns a dict [0,1,2]-> Set[str] with the names of the fields that are pinned at r=0 for azimuthal symmetry.
        Entry 0 contains the names of the fields that are pinned to zero at r=0 for normal (axisymmetric solves). This pinning is strongly enforced.
        Entry 1 contains the names of the fields that are pinned at r=0 for azimuthal eigensolves with m=1. This pinning is implemented by modifying the eigenproblem matrices.
        Entry 2 contains the names of the fields that are pinned at r=0 for azimuthal eigensolves with m>=2. This pinning is implemented by modifying the eigenproblem matrices.        
        """
        master=self._get_combined_element()
        return master._azimuthal_r0_info

    def before_precice_initialise(self,eqtree:"EquationTree"):
        pass

    def before_precice_solve(self,eqtree:"EquationTree",precice_dt:float):
        pass

    def after_precice_solve(self,eqtree:"EquationTree",precice_dt:float):
        pass

    def __init__(self):
        super().__init__()
        self._created_at:str | None
        self._final_element:BaseEquations | None = None
        self._coordinate_system:"BaseCoordinateSystem | None" = None
        self._additional_fields:dict[str,ExpressionOrNum] = {}
        self._additional_fields_also_on_interface:dict[str,ExpressionOrNum] = {}
        self._additional_testfuncs:dict[str,ExpressionOrNum] = {}
        self._additional_testfuncs_also_on_interface:dict[str,ExpressionOrNum] = {}
        self._initial_conditions:dict[str,dict[str,tuple[ExpressionOrNum,str,"BaseEquations"]]] = {}
        self._Dirichlet_conditions:dict[str,tuple[ExpressionOrNum,"BaseEquations"]] = {}
        self._code:_pyoomph.DynamicBulkElementInstance | None = None
        self._scaling:dict[str,"ExpressionOrNum | str"] = {}
        self._test_scaling:dict[str,"ExpressionOrNum | str"]={}
        self._scales_to_check_for_fields:set[str] = set()
        self._test_scales_to_check_for_fields:set[str] = set()
        #self._external_data_links:Dict[str,Tuple["ODEEquations",str]] = {}
        self._dimension = None
        self.default_timestepping_scheme:Literal["BDF2", "BDF1", "Newmark2"] | None = None
        self._problem=None
        self._residuals_for_tex:dict[str,list["Expression"]]={}
        # A list of mapping functions (lambda destination,residual_expression -> dict({destination:new_residual_expression}))
        self._residual_mapping_functions:list[Callable[[str,Expression],Expression | dict[str, Expression]]]=[]
        self._interior_facet_residuals:dict[str,Expression]={}
        self._additional_residuals:dict[str,Expression]={}
        self._fields_defined_on_my_domain:dict[str,FiniteElementSpaceEnum]={}
        #: Set this to true if you require internal facet contributions for DG methods, at best in the constructor
        self.requires_interior_facet_terms:bool=False   
        
        # Stores the data to pin for azimuthal stuff
        self._azimuthal_r0_info:dict[int,set[str]]={} # Which fields will be pinned at the azimuthal symmetry axis for a given azimuthal mode
        self._azimuthal_r0_info[0]=set()
        self._azimuthal_r0_info[1]=set()
        self._azimuthal_r0_info[2]=set()              
        
    

    def interior_facet_terms_required(self):
        return self.requires_interior_facet_terms

    def get_latex_info(self) -> dict[str, list["Expression"]]:
        return self._residuals_for_tex

    def get_combined_equations(self) -> "BaseEquations":
        return self._get_combined_element()

    def calculate_error_overrides(self):
        pass

    def _before_stationary_or_transient_solve(self, eqtree:"EquationTree", stationary:bool)->bool:
        return False # Return whether the equations have to be renumbered

    def _before_eigen_solve(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver",angular_m:int | None=None,normal_k:float | None=None)->bool:
        return False # Return whether the equations have to be renumbered

    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree:"EquationTree",eigensolver:"GenericEigenSolver", angular_mode:int | None,normal_k:float | None)->set[str | int]:
        return set()

    def _init_output(self,eqtree:"EquationTree",continue_info:dict[str, Any] | None,rank:int):
        pass

    def _do_output(self, eqtree:"EquationTree", step:int,stage:str):
        pass

    def _is_ode(self)->bool | None:
        return None

    def before_assigning_equations_preorder(self, mesh:"AnyMesh"):
        """
        This function is called whenever the equations are numbered. The equation tree is traversed and this function is called *before* applying it on all of the children of this domain.
        
        Override this method to e.g. pin redundant (overconstraining) Lagrange multipliers with the :py:meth:`InterfaceEquations.pin_redundant_lagrange_multipliers` method.

        Args:
            mesh: The mesh corresponding to the this domain.
        """        
        pass

    def before_assigning_equations_postorder(self, mesh:"AnyMesh"):
        """
        This function is called whenever the equations are numbered. The equation tree is traversed and this function is called *after* applying it on all of the children of this domain.
        
        Override this method to e.g. pin redundant (overconstraining) Lagrange multipliers with the :py:meth:`InterfaceEquations.pin_redundant_lagrange_multipliers` method.

        Args:
            mesh: The mesh corresponding to the this domain.
        """
        pass

    def after_newton_solve(self):
        pass

    # Returns true if the Newton step is okay. If we cannot take the Newton step for whatever reason, we can return False to reject the step
    def before_newton_convergence_check(self,eqtree:"EquationTree")->bool:
        return True

    def before_newton_solve(self):
        pass

    def after_remeshing(self,eqtree:"EquationTree"):
        pass

    def _release_output_files(self)->None:
        # Overridden by GenericOutput (see output/generic.py) to close any open output
        # file handles it holds, so Problem.release() can close them proactively instead
        # of leaving them for eventual garbage collection -- on Windows, a still-open
        # output file prevents deleting its containing directory (WinError 32), the same
        # class of bug fixed for the problem log file and compiled DLLs.
        pass

    def before_mesh_to_mesh_interpolation(self,eqtree:"EquationTree",interpolator:"BaseMeshToMeshInterpolator"):
        pass

    def after_mapping_on_macro_elements(self):
        pass


    def before_finalization(self,codegen:"FiniteElementCodeGenerator"):
        pass

    def before_compilation(self,codegen:"FiniteElementCodeGenerator"):
        pass

    def after_compilation(self,codegen:"FiniteElementCodeGenerator"):
        pass


    def setup_remeshing_size(self,remesher:"RemesherBase",preorder:bool):
        pass

    def get_space_of_field(self,name:str) -> str:
        cg=self._assert_codegen()
        return cg.get_space_of_field(name)

    def add_named_numerical_factor(self,**kwargs:"ExpressionOrNum"):
        mst=self._get_combined_element()
        cg=mst._assert_codegen()
        cg.add_named_numerical_factor(**kwargs)

    def sanity_check(self):
        pass

    def define_scaling(self):
        pass

    def on_apply_boundary_conditions(self,mesh:"AnyMesh"):
        pass
    
    
    def _fill_interinter_connections(self,eqtree:"EquationTree",interinter:set[str]):
        pass
    
    def before_fill_dummy_equations(self,problem:"Problem",eqtree:"EquationTree",pathname:str):
        pass

    def _assert_codegen(self)->FiniteElementCodeGenerator:
        cg=self._get_current_codegen()
        if cg is None:
            raise RuntimeError("Cannot do this operation outside the scope of a code generator. Occurend in Equations: "+str(self)+" : "+ self.get_information_string())
        assert isinstance(cg,FiniteElementCodeGenerator)
        return cg

    def _set_final_element(self, final:"BaseEquations | None"):
        self._final_element = final

    def define_field_by_substitution(self, fieldname:str, expr:"ExpressionOrNum", also_on_interface:bool=False):
        master = self._get_combined_element()
        master._additional_fields[fieldname] = expr
        if also_on_interface:
            master._additional_fields_also_on_interface[fieldname] = expr

    def define_testfunction_by_substitution(self, fieldname:str, expr:"ExpressionOrNum", also_on_interface:bool=False):
        master = self._get_combined_element()
        master._additional_testfuncs[fieldname] = expr
        if also_on_interface:
            master._additional_testfuncs_also_on_interface[fieldname] = expr

    def set_scaling(self,_field_scalings:"dict[str,ExpressionOrNum | str] | None"=None,*,allow_scales_with_fields:bool=False, **args:"ExpressionOrNum | str"):
        mst = self._get_combined_element()
        all_args:"dict[str,ExpressionOrNum | str]"=dict(args)
        if _field_scalings is not None:
            all_args.update(_field_scalings)
        for n, v in all_args.items():
            if not isinstance(v,str):
                if not isinstance(v,_pyoomph.Expression):
                    v=_pyoomph.Expression(v)
            self._scaling[n] = v
            mst._scaling[n] = v
            if not allow_scales_with_fields:
                mst._scales_to_check_for_fields.add(n)
            else:
                if n in mst._scales_to_check_for_fields:
                    mst._scales_to_check_for_fields.remove(n)

    def set_test_scaling(self,_field_scalings:"dict[str,ExpressionOrNum | str] | None"=None, *, allow_scales_with_fields:bool=False, **args:"ExpressionOrNum | str"):
        mst = self._get_combined_element()
        all_args:"dict[str,ExpressionOrNum | str]"=dict(args)
        if _field_scalings is not None:
            all_args.update(_field_scalings)
        for n, v in all_args.items():
            if not isinstance(v, (_pyoomph.Expression,str)):
                v=_pyoomph.Expression(v)
            self._test_scaling[n] = v
            mst._test_scaling[n] = v
            if not allow_scales_with_fields:
                mst._test_scales_to_check_for_fields.add(n)
                self._scales_to_check_for_fields.add(n)
            else:
                if n in mst._test_scales_to_check_for_fields:
                    mst._test_scales_to_check_for_fields.remove(n)

    def get_element_dimension(self):
        """
        Returns the element dimension of the domain where the equations are defined.
        """
        master=self._get_combined_element()
        cg=master._assert_codegen()
        return cg.dimension

    def get_nodal_dimension(self):
        """
        Returns the nodal (Eulerian) dimension of the domain where the equations are defined.
        """
        master = self._get_combined_element()
        cg = master._assert_codegen()
        return cg.get_nodal_dimension()

    def get_normal(self):
        """
        Returns the normal of this domain. This is only possible if the domain is either a boundary or a bulk domain with co-dimension 1.
        
        Note that ``var("normal")`` is essentially the same.
        """
        master = self._get_combined_element()
        cg=master._assert_codegen()
        ndim = cg.get_nodal_dimension()
        if ndim == 0:
            raise RuntimeError(
                "Normal cannot be used here... Element has no nodal dimension or is not initialised yet for normals")
        #return vector([cg._get_normal_component(i) for i in range(ndim)])
        return var("normal",domain=cg)

    def _get_combined_element(self)->"BaseEquations":
        if self._final_element is None:
            return self
        else:
            return self._final_element._get_combined_element()

    def get_current_code_generator(self) -> FiniteElementCodeGenerator:
        mst=self._get_combined_element()
        assert mst is not None
        return mst._assert_codegen()

    def get_mesh(self)->"AnyMesh":
        mesh=self.get_current_code_generator()._mesh
        assert mesh is not None
        return mesh

    def _perform_define_fields(self):
        master = self._get_combined_element()
        parent_domain=self.get_parent_domain()
        if parent_domain is not None:
            p=parent_domain.get_equations().get_azimuthal_r0_info()
            for i in range(3):
                master._azimuthal_r0_info[i]=p[i].copy()
        master.define_fields()
        master.sanity_check()

    def define_fields(self):
        """
        Inherit and specify to define fields (dependent variables), either by using :py:meth:`ODEEquations.define_ode_variable` (ODEs inherited from :py:class:`ODEEquations`) or :py:meth:`Equations.define_scalar_field`/:py:meth:`Equations.define_vector_field` (PDEs inherited from :py:class:`Equations`)
        """
        pass

    def define_residuals(self)->Expression | None:
        """
        Inherit and specify to define residuals for the equations, using :py:meth:`add_residual` or :py:meth:`add_weak`
        Any returned expression will be also added to the residuals
        """
        pass

    @overload
    def get_scaling(self, n:str,testscale:Literal[True]=...)->"ExpressionOrNum": ...

    @overload
    def get_scaling(self, n:str,testscale:Literal[False]=...)->"ExpressionOrNum": ...

    @overload
    def get_scaling(self, n:str,testscale:Literal["from_parent"])->"ExpressionOrNum | None": ...

    def get_scaling(self, n:str,testscale:bool | Literal["from_parent"]=False)->"ExpressionOrNum | None":
        master = self._get_combined_element()
        cg=master._assert_codegen()
        #print("GETTING SCALE", n, self, master, self._scaling.get(n, None), self._scaling, self._is_ode(),hasattr(self, "get_parent_domain"), cg.get_parent_domain())
        arr=self._test_scaling if testscale else self._scaling
        if arr.get(n, None) is not None:
            if isinstance(arr[n], str):
                return self.get_scaling(arr[n],testscale=testscale) #type:ignore
            else:
                return arr[n] #type:ignore
        if master != self:
            return master.get_scaling(n,testscale=testscale) #type:ignore
        elif cg.get_parent_domain() is not None:
            if testscale:
                #print("IN HERE",n,self.get_scaling("spatial"),self.get_parent_domain().get_scaling(n, testscale=True))
                ts=cg.get_parent_domain().get_scaling(n, testscale="from_parent")  #type:ignore
                if ts is None:
                    return _pyoomph.Expression(1)
                else:
                    return ts/self.get_scaling("spatial") 
            else:
                return cg.get_parent_domain().get_scaling(n, testscale=False) #type:ignore

        else:
           # print("PROBLEM SCALE ",n)
           if not testscale:
               return cg.get_problem().get_scaling(n)
           elif testscale=="from_parent":
               return None
           else:
               return _pyoomph.Expression(1)

    def set_initial_condition(self, field:str, expr:"ExpressionOrNum", degraded_start:Literal["auto"] | bool="auto",IC_name:str=""):
        #self._perform_define_fields()
        master = self._get_combined_element()
        if expr is None:
            raise RuntimeError("Cannot set initial condition to None")
        if type(expr) == float or type(expr) == int:
            expr = _pyoomph.Expression(expr)
        if degraded_start == "auto":
            degraded_startI="auto"
        elif not isinstance(degraded_start, bool): #type:ignore
            raise RuntimeError(
                "degraded_start must be a bool or 'auto' (which means that every IC without a time-dependency will be degraded")
        elif degraded_start == True:
            degraded_startI = "yes"
        else:
            degraded_startI = "no"
        if not field in master._initial_conditions.keys():
            master._initial_conditions[field]={}
        master._initial_conditions[field][IC_name] = (expr, degraded_startI,self)

    def set_Dirichlet_condition(self, field:str, expr:"ExpressionOrNum"):
        master = self._get_combined_element()
        if type(expr) == float or type(expr) == int:
            expr = _pyoomph.Expression(expr)
        master._Dirichlet_conditions[field] = (expr,self)

    def _perform_initial_and_Dirichlet_conditions(self):  # Only called at master level
        cg=self._assert_codegen()
        if _pyoomph.get_verbosity_flag() != 0:
            print("SETTING IC", repr(self))
        for n, field_ics in self._initial_conditions.items():
            for ic_name, expr in field_ics.items():
                if _pyoomph.get_verbosity_flag() != 0:
                    print("SETTING IC", ic_name, n, self.get_scaling(n), expr)
                try:
                    nondim_icexpr=expr[0] / self.get_scaling(n)
                    if not isinstance(nondim_icexpr,_pyoomph.Expression):
                        nondim_icexpr=_pyoomph.Expression(nondim_icexpr)
                    cg._set_initial_condition(n, nondim_icexpr, expr[1],ic_name)
                except Exception as e:
                    expr[2].add_exception_info(e)

        if _pyoomph.get_verbosity_flag() != 0:

            print("SETTING DIRICHLET", repr(self))
        for n, expr_comb in self._Dirichlet_conditions.items():
            expr=expr_comb[0]
            if _pyoomph.get_verbosity_flag() != 0:
                print("SETTING DIRICHLET OF ", n, self.get_scaling(n), expr)
            if expr is True:
                cg._set_Dirichlet_bc(n, _pyoomph.Expression(0),True)
            else:
                try:
                    nondim_bc=expr / self.get_scaling(n)
                    if not isinstance(nondim_bc,_pyoomph.Expression):
                        nondim_bc=_pyoomph.Expression(nondim_bc)
                    cg._set_Dirichlet_bc(n, nondim_bc,False)
                except Exception as e:
                    expr_comb[1].add_exception_info(e)


    def define_error_estimators(self):
        pass

    def define_additional_functions(self):
        pass


    def add_interior_facet_residual(self,expr:"ExpressionOrNum",*,destination:str | None=None):
        """
        Same as :py:meth:`add_residual`, but the added residuals are evaluated and considered at the interior facet domain. This is only used for DG methods and requires the property :py:attr:`requires_interior_facet_terms` to be set in the constructor of the equations.

        Args:
            expr: Expression to add to the residuals
            destination: Optional residual destination for multiple residuals. Defaults to ``None``.
        """
        
        master = self._get_combined_element()
        if not self.requires_interior_facet_terms or not master.interior_facet_terms_required():
            raise RuntimeError("Please set the property requires_interior_facet_terms=True in the __init__ of the Equations class before calling add_interior_facet_residual")
        if not isinstance(expr, _pyoomph.Expression):
            expr = _pyoomph.Expression(expr)
        dn=destination if destination is not None else ""
        if dn not in master._interior_facet_residuals.keys():
            master._interior_facet_residuals[dn]=Expression(0)
        master._interior_facet_residuals[dn]+=expr

    def add_residual(self, expr: "ExpressionOrNum | str", *, destination: str | None = None):
        """
        Adds a residual contribution to this equations.

        Args:
            expr: The expression or number to be added as a residual.
            destination: The destination of the residual. Defaults to ``None``, can be used to specify different residuals.
        """
        master = self._get_combined_element()
        if not isinstance(expr, _pyoomph.Expression):
            expr = _pyoomph.Expression(expr)
        dn = destination if destination is not None else ""
        if dn not in master._residuals_for_tex.keys():
            master._residuals_for_tex[dn] = []
        master._residuals_for_tex[dn].append(expr)
        cg = master._assert_codegen()
        contributions = {dn: expr}
        all_mappings: list[Callable[[str, Expression], Expression | dict[str, Expression]]] = (
            cg.get_problem()._residual_mapping_functions + master._residual_mapping_functions  # type:ignore
        )
        for mapping in all_mappings:
            newcontribs: dict[str, _pyoomph.Expression] = {}
            for ds, es in contributions.items():
                newmap = mapping(ds, es)
                if not isinstance(newmap, dict):
                    newmap = {ds: newmap}
                for dn, en in newmap.items():
                    if dn not in newcontribs.keys():
                        newcontribs[dn] = en
                    else:
                        newcontribs[dn] += en
            contributions = newcontribs

        for dest, expression in contributions.items():
            if dest is not None:
                cg._activate_residual(dest)
            try:
                #print("adding residual", expression, dest)
                cg._add_residual(expression, False)
            except Exception as e:
                self.add_exception_info(e)
            cg._activate_residual("")
        cg._activate_residual("")
        return self

    def _setup_combined_element(self):
        self._set_final_element(None)

    def _define_fields(self):
        self._setup_combined_element()
        master = self._get_combined_element()
#        print("IN DEFINE FIELDS. Master is ",master,self._assert_codegen())
        master._perform_define_fields()


    def _check_scalings(self):
        # Check all scalings and test scalings
        for n in self._scales_to_check_for_fields:
            scal=self.get_scaling(n)
            if scal is not None:
                scal_expa=self.expand_expression_for_debugging(scal)
                try:
                    assert_dimensional_value(scal_expa)
                except Exception as e:
                    cg=self._assert_codegen()
                    raise RuntimeError("The scale for '"+str(n)+"' on domain '"+self._assert_codegen().get_full_name()+"' is not a simple dimensional number, but:\n    "+str(scal)+"\n   expands to: "+str(scal_expa)+"\n.")
        for n in self._test_scales_to_check_for_fields:
            scal=self.get_scaling(n,testscale=True)
            if scal is not None:
                scal_expa=self.expand_expression_for_debugging(scal)
                try:
                    assert_dimensional_value(scal_expa)
                except Exception as e:
                    cg=self._assert_codegen()
                    raise RuntimeError("The test function scale for '"+str(n)+"' on domain '"+cg.get_full_name()+"' is not a simple dimensional number, but:\n    "+str(scal)+"\n   it expands to: "+str(scal_expa)+"\n.") 

    def _define_element(self):
        

        self._setup_combined_element()
        master = self._get_combined_element()
        #master._perform_define_fields()
        master.define_scaling()
        cg=self._assert_codegen()


#            raise RuntimeError("Transfer fields here")
        res=master.define_residuals()
        if res is not None:
            master.add_residual(res)
        for d,add_res in master._additional_residuals.items():
            master.add_residual(add_res,destination=d if d!="" else None)
        master.define_error_estimators()
        master._perform_initial_and_Dirichlet_conditions()
        master.define_additional_functions()
        assert master._problem is not None
        master._problem.before_compile_equations(master)
        master._check_scalings()
        

    def get_coordinate_system(self)->BaseCoordinateSystem:
        master = self._get_combined_element()
        if master._coordinate_system is not None:
            return master._coordinate_system
        elif (pdom:=master.get_current_code_generator().get_parent_domain()):
            return pdom.get_coordinate_system()
        else:
            assert master._problem is not None
            return master._problem.get_coordinate_system()

    def expand_additional_field(self, name:str, dimensional:bool, expression:_pyoomph.Expression,in_domain:_pyoomph.FiniteElementCode,no_jacobian:bool,no_hessian:bool,where:str)->"Expression":        
        #msh=self.get_mesh()
        #if msh is not None:
        #    msh=msh._name
        master = self._get_combined_element()
        try:
            cg:"FiniteElementCodeGenerator" = master._assert_codegen()

        except:
            if master._is_ode(): # ODEs might still be accessible
                tags=expression.op(1)
                print(dir(master))
                print("CODE",master._code)
                print("PROBLEM", master._problem)
                print("TAGS", tags)
                raise RuntimeError("TODO: Expand tags, see what ODE is meant by domain tag and resolve the additional test function. You could also have a typo in the name of "+str(name)+", i.e. that this field does not exist in this ODE")
            else:
                raise RuntimeError("Should not end up here")

        assert isinstance(in_domain,FiniteElementCodeGenerator)
        if _pyoomph.get_verbosity_flag() != 0:
                print("Expanding additional field ", name, dimensional,"self/in_domain",cg.get_full_name(),in_domain.get_full_name())
        
        if dimensional:
            scale = self.get_scaling(name)
        else:
            scale = 1

        only_base_mode=False
        only_perturbation_mode=False
        axibreakcsys:BaseCoordinateSystem | None=None
        typeinfo=str(expression.op(1))
        if typeinfo.find("tags="):
            tags=typeinfo[typeinfo.find("tags=")+5:-1].split(", ")
            axibreakcsys=self.get_current_code_generator().get_coordinate_system()
            if isinstance(axibreakcsys,AxisymmetryBreakingCoordinateSystem):                        
                if 'flag:only_base_mode' in tags:                        
                    only_base_mode=True                        
                elif 'flag:only_perturbation_mode' in tags:
                    only_perturbation_mode=True            

        assert cg.get_problem() is not None

        def vr(name:str,domain:"FiniteElementCodeGenerator | None"=None)->"Expression":
            if dimensional:
                return var(name,domain=domain,no_jacobian=no_jacobian,no_hessian=no_hessian)
            else:
                return nondim(name,domain=domain,no_jacobian=no_jacobian,no_hessian=no_hessian)

        if name == "coordinate":
            dim = cg.get_nodal_dimension()
            if dim == 1:
                return vector([vr("coordinate_x")])
            elif dim == 2:
                return vector([vr("coordinate_x"), vr("coordinate_y")])
            elif dim == 3:
                return vector([vr("coordinate_x"), vr("coordinate_y"), vr("coordinate_z")])
        elif name == "mesh":
            res=cg.get_coordinate_system().expand_coordinate_or_mesh_vector(cg,"mesh",dimensional=dimensional,no_jacobian=no_jacobian,no_hessian=no_hessian)
            assert res is not None
            return res
            
        elif name == "lagrangian":
            dim = cg.get_lagrangian_dimension()
            if dim == 1:
                return vector([vr("lagrangian_x")])
            elif dim == 2:
                return vector([vr("lagrangian_x"), vr("lagrangian_y")])
            elif dim == 3:
                return vector([vr("lagrangian_x"), vr("lagrangian_y"), vr("lagrangian_z")])
        elif name == "local_coordinate":
            dim = cg.get_element_dimension()
            if dim == 1:
                return vector([vr("local_coordinate_1")])
            elif dim == 2:
                return vector([vr("local_coordinate_1"), vr("local_coordinate_2")])
            elif dim == 3:
                return vector([vr("local_coordinate_1"), vr("local_coordinate_2"), vr("local_coordinate_3")])            
        elif name == "normal":
            return cg.get_coordinate_system().get_normal_vector_or_component(cg,component=None,only_base_mode=only_base_mode,only_perturbation_mode=only_perturbation_mode,where=where)
            dim = cg.get_nodal_dimension()
            if dim == 1:
                return vector([vr("normal_x",domain=cg)])
            elif dim == 2:
                return vector([vr("normal_x",domain=cg), vr("normal_y",domain=cg)])
            elif dim == 3:
                return vector([vr("normal_x",domain=cg), vr("normal_y",domain=cg), vr("normal_z",domain=cg)])
        elif name == "normal_x":
#            if cg.get_nodal_dimension() != cg.get_element_dimension() + 1:
#                raise RuntimeError("Problem to get a normal for this element at this nodal dimension: "+str(cg.get_nodal_dimension())+" and "+str(cg.get_element_dimension())+". Domain is: "+str(in_domain))
            #return cg._get_normal_component(0)
            return cg.get_coordinate_system().get_normal_vector_or_component(cg,component=0,only_base_mode=only_base_mode,only_perturbation_mode=only_perturbation_mode,where=where)
        elif name == "normal_y":
#            if cg.get_nodal_dimension() != cg.get_element_dimension() + 1:
#                raise RuntimeError("Problem to get a normal for this element at this nodal dimension")
            #return cg._get_normal_component(1)
            return cg.get_coordinate_system().get_normal_vector_or_component(cg,component=1,only_base_mode=only_base_mode,only_perturbation_mode=only_perturbation_mode,where=where)
        elif name == "normal_z":
#            if cg.get_nodal_dimension() != cg.get_element_dimension() + 1:
#                raise RuntimeError("Problem to get a normal for this element at this nodal dimension")
            #return cg._get_normal_component(2)
            return cg.get_coordinate_system().get_normal_vector_or_component(cg,component=2,only_base_mode=only_base_mode,only_perturbation_mode=only_perturbation_mode,where=where)
        elif name == "dx":
            return _pyoomph.FiniteElementCode._get_dx(cg, False,False)
        elif name == "dx_unity":
            return _pyoomph.FiniteElementCode._get_dx(cg, False,True)
        elif name == "_nodal_delta":
            return _pyoomph.FiniteElementCode._get_nodal_delta(cg)
        elif name == "dX":
            return _pyoomph.FiniteElementCode._get_dx(cg, True,False)
        elif name == "element_size_Eulerian":
            return _pyoomph.FiniteElementCode._get_element_size_symbol(cg,False,True)*(cg.get_coordinate_system().volumetric_scaling(scale_factor("spatial"),self.get_element_dimension()) if dimensional else 1)
        elif name == "cartesian_element_size_Eulerian":
            return _pyoomph.FiniteElementCode._get_element_size_symbol(cg,False,False)*((scale_factor("spatial")**self.get_element_dimension()) if dimensional else 1)        
        # Length factor of an element. Note that is is (elem_vol)**(1/actual_dim) where actual_dim is e.g. 3 in axisymm coordsys
        elif name == "element_length_h": 
            real_edim=cg.get_coordinate_system().get_actual_dimension(self.get_element_dimension())
            return vr("element_size_Eulerian",domain=cg)**rational_num(1,real_edim)
        elif name == "cartesian_element_length_h": 
            real_edim=self.get_element_dimension()
            return vr("cartesian_element_size_Eulerian",domain=cg)**rational_num(1,real_edim)        
        elif name == "element_size_Lagrangian":
            return _pyoomph.FiniteElementCode._get_element_size_symbol(cg,True,True)*(cg.get_coordinate_system().volumetric_scaling(scale_factor("lagrangian"),self.get_element_dimension()) if dimensional else 1)                
        elif name == "cartesian_element_size_Lagrangian":
            return _pyoomph.FiniteElementCode._get_element_size_symbol(cg,True,False)*((scale_factor("lagrangian")**self.get_element_dimension()) if dimensional else 1)        
        elif name == "time":
            return scale * _pyoomph.GiNaC_TimeSymbol()  # get_global_symbol("t")
        elif name in self._additional_fields.keys():
            if only_base_mode:
                assert isinstance(axibreakcsys,AxisymmetryBreakingCoordinateSystem)
                return scale * axibreakcsys.map_to_zero_epsilon(evaluate_in_domain(self._additional_fields[name],self.get_current_code_generator()))
            elif only_perturbation_mode:
                assert isinstance(axibreakcsys,AxisymmetryBreakingCoordinateSystem)
                return scale * axibreakcsys.map_to_first_order_epsilon(evaluate_in_domain(self._additional_fields[name],self.get_current_code_generator()),with_epsilon=True)
            else:
                return scale * evaluate_in_domain(self._additional_fields[name],self.get_current_code_generator())
        elif hasattr(self,"get_parent_domain") and self.get_parent_domain() is not None:
        #    print("PARENT "+str(self))
            bulk = self.get_parent_domain()
            while bulk is not None:
                bulkeq=bulk.get_equations()
                if name in bulkeq._additional_fields_also_on_interface.keys():
                    expr = bulkeq._additional_fields_also_on_interface[name]
                    expr=evaluate_in_domain(expr,self.get_current_code_generator())
                    if only_base_mode:
                        assert isinstance(axibreakcsys,AxisymmetryBreakingCoordinateSystem)
                        expr= axibreakcsys.map_to_zero_epsilon(expr)
                    elif only_perturbation_mode:
                        assert isinstance(axibreakcsys,AxisymmetryBreakingCoordinateSystem)
                        expr= axibreakcsys.map_to_first_order_epsilon(expr,with_epsilon=True)
                    if dimensional:
                        scale = bulk.get_scaling(name)
                    else:
                        scale = 1
                    return scale * expr
                bulk=bulk.get_parent_domain()
        if name == "mesh_y" and self.get_nodal_dimension() < 2:
            return _pyoomph.Expression(0.0)
        elif name == "mesh_z" and self.get_nodal_dimension() < 3:
            return _pyoomph.Expression(0.0)
        elif cg.get_problem().has_named_var(name):
            named_res=cg.get_problem().get_named_var(name)
            assert named_res is not None
            if not isinstance(named_res,_pyoomph.Expression):
                named_res=_pyoomph.Expression(named_res)
            return named_res
        cg_dom_name=cg.get_full_name()
        raise RuntimeError(
            "Cannot expand the field '" + name + "' since it is not defined in the equation or any parents.\nCurrent code generator is:"+str(cg)+" : " +cg_dom_name+"\nIn: "+str(self)+"\nAdditional fields are: "+", ".join(self._additional_fields.keys()))

    def expand_additional_testfunction(self, name:str, expression:"Expression",in_domain:_pyoomph.FiniteElementCode)->"Expression":
        master = self._get_combined_element()
        try:
            cg = master._assert_codegen()
        except:
            if master._is_ode(): # ODEs might still be accessible
                tags=expression.op(1)
                print(dir(master))
                print("CODE",master._code)
                print("PROBLEM", master._problem)
                print("TAGS", tags)
                raise RuntimeError("TODO: Expand tags, see what ODE is meant by domain tag and resolve the additional test function. You could also have a typo in the name of "+str(name)+", i.e. that this field does not exist in this ODE")
            raise RuntimeError("Cannot expand (additional) test function '"+str(name)+"' to expand "+str(expression)+".\n Probably, you want to access a spatial domain, which is not accessible (i.e. neither parent(/parent) domain, nor opposite side of interface or parent of that")
        if name == "mesh":
            dim = cg.get_nodal_dimension()
            if dim == 1:
                return vector([testfunction("mesh_x")])
            elif dim == 2:
                return vector(
                    [testfunction("mesh_x"), testfunction("mesh_y")])
            elif dim == 3:
                return vector(
                    [testfunction("mesh_x"), testfunction("mesh_y"),
                     testfunction("mesh_z")])
            else:
                raise RuntimeError("Cannot expand the testfunction " + str(name) + " with dimension " + str(dim))
        elif name in self._additional_testfuncs.keys():
            res=self._additional_testfuncs[name]
            if not isinstance(res,Expression):
                res=Expression(res)
            return evaluate_in_domain(res,cg)
        elif (not isinstance(self, ODEEquations)) and self.get_parent_domain() is not None:
            bulk = self.get_parent_domain()
            while bulk is not None:
               # print("INPUT",expression)
                bulkeq=bulk.get_equations()
                if name in bulkeq._additional_testfuncs_also_on_interface.keys():
                    tf=bulkeq._additional_testfuncs_also_on_interface[name]
                    tf=evaluate_in_domain(tf,self.get_current_code_generator())
                    #print(tf)
                    return tf
                bulk=bulk.get_parent_domain()
            raise RuntimeError("Cannot expand the testfunction "+ str(name))
        else:
            raise RuntimeError("Cannot expand the testfunction " + str(name))

    def define_external_field(self, name:str, space:"FiniteElementSpaceEnum"="D0"):
        if space == "D0":
            spaceI = "ED0"
        else:
            raise RuntimeError("External fields may only be on space D0 at the moment")
        master = self._get_combined_element()
        master._register_field(name, spaceI)

    def set_temporal_error_factor(self, name:str, factor:float):
        master = self._get_combined_element()
        cg=master._assert_codegen()
        if isinstance(master,Equations):
            if name in master._vectorfields.keys(): 
                v=master._vectorfields[name] 
                for f in v:
                    cg._set_temporal_error(f, factor)
            else:
                cg._set_temporal_error(name, factor)
        else:
            cg._set_temporal_error(name, factor)

    def get_default_timestepping_scheme(self, order:int,cg:FiniteElementCodeGenerator | None=None)->Literal["BDF2","BDF1","Newmark2"]:
        if order == 2:
            return "Newmark2"        

        if self.default_timestepping_scheme is not None:
            return self.default_timestepping_scheme
        master = self._get_combined_element()
        if master.default_timestepping_scheme is not None:
            return master.default_timestepping_scheme

        if isinstance(self, ODEEquations):
            assert self._problem is not None
            return cast(Literal["BDF2","BDF1","Newmark2"],self._problem.get_default_timestepping_scheme(order))
        elif cg is not None:
            pdom=cg.get_parent_domain()
            if pdom is not None:
                return cast(Literal["BDF2","BDF1","Newmark2"],pdom.get_default_timestepping_scheme(order)) 
            else:
                assert self._problem is not None
                return cast(Literal["BDF2","BDF1","Newmark2"],self._problem.get_default_timestepping_scheme(order))
        elif (pdom:=self.get_parent_domain()) is not None:
            return cast(Literal["BDF2","BDF1","Newmark2"],pdom.get_default_timestepping_scheme(order)) 
        else:
            assert self._problem is not None
            return cast(Literal["BDF2","BDF1","Newmark2"],self._problem.get_default_timestepping_scheme(order))





    def get_information_string(self)->str:
        return ""

    def _tree_string(self, indent:str) -> str:
        addinfo = self.get_information_string()
        return self.__class__.__name__ + (": " + addinfo if addinfo != "" else "")

    def __matmul__(self, other:str | list[str] | tuple[str, ...] | set[str])->"EquationTree":
        if isinstance(other, (list,tuple,set,)):
            res=EquationTree(None, None)
            cmb=self if isinstance(self,CombinedEquations) else CombinedEquations(self)
            for d in other:
                if d is None or d==".":
                    res+=cmb
                else:
                    res+=cmb@d
            return res
            #return sum([self @ d for d in other], EquationTree(None, None))
        if isinstance(other, str): #type:ignore
            splt = other.split("/")
            splt = [x for x in splt if x]  # Remove empties
            if len(splt) == 0:
                raise ValueError("Please restrict equations with a non-empty domain name")
            root = EquationTree(None, parent=None)
            mynode = EquationTree(self, parent=root)
            _check_for_valid_var_name(splt[-1],True)
            root._children[splt[-1]] = mynode
            res = root
            for k in reversed(splt[:-1]):
                res = res @ k
            return res
        else:
            raise ValueError(
                "Please combine equation with a string (name of the domain) to restrict the equations to a domain")

    def get_my_domain(self)->FiniteElementCodeGenerator:
        master = self._get_combined_element()
        cg = master._assert_codegen()
        return cg

    @overload
    def get_equation_of_type(self, typ:type["BaseEquations"], *, exact_type:bool=False,always_as_list:Literal[True])->list["BaseEquations"]: ...

    @overload
    def get_equation_of_type(self, typ:type["BaseEquations"], *, exact_type:bool=False,always_as_list:Literal[False]=False)->"BaseEquations | None": ...

    def get_equation_of_type(self, typ:type["BaseEquations"], *, exact_type:bool=False,always_as_list:bool=False)->'list["BaseEquations"] | BaseEquations | None':
        if exact_type:
            if type(self) is typ:
                if always_as_list:
                    return [self]
                else:
                    return self
        else:
            if isinstance(self, typ):
                if always_as_list:
                    return [self]
                else:
                    return self
        if always_as_list:
            return cast(list["BaseEquations"],[])
        else:
            return None

    def _register_field(self,name:str,space:str):
        _check_for_valid_var_name(name,False)
        cg=self._assert_codegen()
        cg._register_field(name,space)


    @overload
    def __add__(self, other:"BaseEquations")->"CombinedEquations": ...

    @overload
    def __add__(self, other:"EquationTree")->"EquationTree": ...

    def __add__(self, other:"Literal[0] | BaseEquations | EquationTree")->"CombinedEquations | EquationTree | BaseEquations":
        if other==0:
            return self
        if isinstance(other, BaseEquations):
            if isinstance(self,CombinedEquations):
                if isinstance(other,CombinedEquations):
                    return CombinedEquations(*(self._subelements+other._subelements)) 
                else:
                    return CombinedEquations(*(self._subelements+[other])) 
            elif isinstance(other,CombinedEquations):
                return CombinedEquations(*([self]+other._subelements)) 
            else:
                return CombinedEquations(self, other)
        elif isinstance(other, EquationTree): #type:ignore
            mytree = EquationTree(self, None)
            return mytree + other
        else:
            raise RuntimeError("Cannot add (+) Equation and " + other.__class__.__name__)


    def add_local_function(self,name:str,expr:'ExpressionOrNum | Callable[[], "ExpressionOrNum"]')->tuple[list[str],int]:
        """
        Adds a local function for the output. This are not degrees of freedom but only calculated node-wise on output.
        The same can be achieved by using LocalExpressions(...) instead.

        Args:
            name (str): name of the local expressions
            expr (Union[ExpressionOrNum,Callable[[],ExpressionOrNum]]): Expression to be evaluated on the nodes on output.

        Returns:
            Tuple[List[str],int]: If the expression is a vector, it just returns the vector component names and 0. For a tensor, it returns the tensor components and the dimension of the tensor.
        """
        
        
        master = self._get_combined_element()
        cg = master._assert_codegen()
        if not (isinstance(expr,int) or isinstance(expr,float) or isinstance(expr,_pyoomph.Expression)) and  callable(expr):
            expr=expr()
        if isinstance(expr,(int,float)):
            expr=_pyoomph.Expression(expr)
        entries,diminfo=cg._register_local_function(name, expr)
        if diminfo==0: # vector
            assert isinstance(master,Equations)
            master._vectorfields[name]=[]
            for jc in range(len(entries)):
                    master._vectorfields[name].append(entries[jc])
        elif diminfo>0: # tensor
            assert isinstance(master,Equations)
            master._tensorfields[name]=[]
            cnt=0
            for ic in range(diminfo):
                row=[]
                for jc in range(len(entries)//diminfo):
                    row.append(entries[cnt])
                    cnt+=1
                master._tensorfields[name].append(row)
        return entries,diminfo

    def add_integral_function(self, name:str, expr:"ExpressionOrNum"):
        master = self._get_combined_element()
        cg=master._assert_codegen()
        if not isinstance(expr,_pyoomph.Expression):
            expr=_pyoomph.Expression(expr)
        res=cg._register_integral_function(name, expr)
        if len(res)>0:
            # assemble the vector expression
            argnames=[x for x in res if x!=""]
            codestr="numpy.array(["+ ",".join([x if x!="" else "0" for x in res]) +"])"
            lambda_code="lambda "+",".join(argnames)+" : "+codestr
            lambda_func=eval(lambda_code,{"numpy":numpy})
            cg._register_dependent_integral_function(name, lambda_func,vector_helper=True)


    def add_dependent_integral_function(self,name:str,func:Callable[...,"ExpressionOrNum"]):
        master = self._get_combined_element()
        cg = master._assert_codegen()
        cg._register_dependent_integral_function(name, func)  


    def expand_expression_for_debugging(self,expr:"ExpressionOrNum",raise_error:bool=True,collect_units:bool=True,unit_error:bool=True,with_mode_expansion:bool=True) -> Expression:
        master = self._get_combined_element()
        cg = master._assert_codegen()
        if not isinstance(expr,_pyoomph.Expression):
            expr=_pyoomph.Expression(expr)
        csys=self.get_coordinate_system()

        old_setting:bool | None=None
        if isinstance(csys,AxisymmetryBreakingCoordinateSystem) and with_mode_expansion:
            old_setting=csys.expand_with_modes_for_python_debugging
            csys.expand_with_modes_for_python_debugging=True
        expanded=cg.expand_placeholders(expr,raise_error)
        if isinstance(csys,AxisymmetryBreakingCoordinateSystem) and with_mode_expansion:
            assert old_setting is not None
            csys.expand_with_modes_for_python_debugging=old_setting

        if collect_units:
            factor, unit, rest, success = _pyoomph.GiNaC_collect_units(expanded)
            if unit_error and not success:
                raise RuntimeError("Cannot collect the units of "+str(expanded)+". FACTOR, UNIT, REST are\n"+str(factor)+"\n"+str(unit)+"\n"+str(rest))
            expanded=factor*unit*rest

        return expanded

#########


class EquationTree:
    def __init__(self, eqs:BaseEquations | None=None, parent:"EquationTree | None"=None):
        super(EquationTree, self).__init__()
        self._name:str
        self._equations = eqs
        self._parent = parent
        self._codegen:"FiniteElementCodeGenerator | None"=None
        self._mesh:"AnyMesh | None"=None
        self._children:dict[str,"EquationTree"] = {}

    def get_mesh(self)->"AnyMesh":
        assert self._mesh is not None
        return self._mesh

    def get_equations(self)->BaseEquations:
        assert self._equations is not None
        return self._equations
    
    def get_children(self) -> dict[str, "EquationTree"]:
        return self._children

    def get_code_gen(self) -> FiniteElementCodeGenerator:
        assert self._codegen is not None
        return self._codegen


    # Set my equations (and potentially also bulk,etc. for the codegenerator of this domain)
    def setup_codegen_to_equations(self,with_bulk_and_opp=True,reset_info:dict[str, FiniteElementCodeGenerator | None] | None=None)->dict[str,FiniteElementCodeGenerator | None]:
        # _get_current_codegen() is declared (in the C++ stub) to return the base FiniteElementCode,
        # but in practice it always holds a FiniteElementCodeGenerator (the only concrete Python subclass in use).
        if reset_info is None:
            if with_bulk_and_opp:
                res:dict[str,FiniteElementCodeGenerator | None]={}
                res["."]=cast(FiniteElementCodeGenerator,self.get_code_gen().get_equations()._get_current_codegen())
                self.get_equations()._set_current_codegen(self._codegen)
                #print(self._codegen,self.get_equations()._get_current_codegen())
                oppi = self.get_code_gen()._get_opposite_interface()
                if oppi is not None:
                    assert isinstance(oppi, FiniteElementCodeGenerator)
                    res["|."]=cast(FiniteElementCodeGenerator,oppi.get_equations()._get_current_codegen())
                    oppi.get_equations()._set_current_codegen(oppi)
                    oppblk = oppi.get_parent_domain()
                    if oppblk is not None:
                        res["|.."]=cast(FiniteElementCodeGenerator,oppblk.get_equations()._get_current_codegen())
                        oppblk.get_equations()._set_current_codegen(oppblk)
                blk = self.get_code_gen().get_parent_domain()
                if blk is not None:
                    res[".."] = cast(FiniteElementCodeGenerator,blk.get_equations()._get_current_codegen())
                    blk.get_equations()._set_current_codegen(blk)
                    blkblk=blk.get_parent_domain()
                    if blkblk is not None:
                        res["../.."] = cast(FiniteElementCodeGenerator,blkblk.get_equations()._get_current_codegen())
                        blkblk.get_equations()._set_current_codegen(blkblk)
                return res
            else:
                res2=cast(FiniteElementCodeGenerator,self.get_code_gen().get_equations()._get_current_codegen())
                self.get_equations()._set_current_codegen(self._codegen)
                return {".":res2}
        else:
            if with_bulk_and_opp:
                res={}
                oppi = self.get_code_gen()._get_opposite_interface()
                if oppi is not None:
                    assert isinstance(oppi, FiniteElementCodeGenerator)
                    res["|."]=cast(FiniteElementCodeGenerator,oppi.get_equations()._get_current_codegen())
                    oppi.get_equations()._set_current_codegen(reset_info.get("|.",None))
                    oppblk = oppi.get_parent_domain()
                    if oppblk is not None:
                        res["|.."]=cast(FiniteElementCodeGenerator,oppblk.get_equations()._get_current_codegen())
                        oppblk.get_equations()._set_current_codegen(reset_info.get("|..",None))
                blk = self.get_code_gen().get_parent_domain()
                if blk is not None:
                    res[".."] = cast(FiniteElementCodeGenerator,blk.get_equations()._get_current_codegen())
                    blk.get_equations()._set_current_codegen(reset_info.get("..",None))
                    blkblk=blk.get_parent_domain()
                    if blkblk is not None:
                        res["../.."] = cast(FiniteElementCodeGenerator,blkblk.get_equations()._get_current_codegen())
                        blkblk.get_equations()._set_current_codegen(reset_info.get("../..",None))
                res["."]=cast(FiniteElementCodeGenerator,self.get_code_gen().get_equations()._get_current_codegen())
                self.get_equations()._set_current_codegen(reset_info.get(".",None))
                return res
            else:
                res2=cast(FiniteElementCodeGenerator,self.get_code_gen().get_equations()._get_current_codegen())
                self.get_equations()._set_current_codegen(reset_info.get(".",None))
                return {".":res2}


    def _change_output_directory(self,newdir:str):
        if (self._mesh is not None) and (self._equations is not None):            
            assert self._codegen is not None
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations.change_output_directory(newdir,self)
            self._equations._set_current_codegen(oldcg)
            
        for _,c in self._children.items():
            c._change_output_directory(newdir)

    def _before_assigning_equations(self,dof_selector:"_DofSelector | None"):                
        if (self._mesh is not None) and (self._equations is not None):            
            assert self._codegen is not None
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations.before_assigning_equations_preorder(self._mesh)
            self._equations._set_current_codegen(oldcg)

        if dof_selector is not None:
            dof_selector._apply_on_domain(self._mesh)  

        for _,c in self._children.items():
            c._before_assigning_equations(dof_selector)
        if (self._mesh is not None) and (self._equations is not None):
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations.before_assigning_equations_postorder(self._mesh)
            self._equations._set_current_codegen(oldcg)

    def _after_remeshing(self):
        if (self._mesh is not None) and (self._equations is not None):
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen) 
            self._equations.after_remeshing(self)
            self._equations._set_current_codegen(oldcg)
        for _,c in self._children.items():
            c._after_remeshing()

    def _before_mesh_to_mesh_interpolation(self,interpolator:"BaseMeshToMeshInterpolator"):
        if (self._mesh is not None) and (self._equations is not None):
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen) 
            self._equations.before_mesh_to_mesh_interpolation(self,interpolator)
            self._equations._set_current_codegen(oldcg)
        for _,c in self._children.items():
            c._before_mesh_to_mesh_interpolation(interpolator)


    def _setup_remeshing_size(self,remesher:"RemesherBase",preorder:bool):
        if preorder:
            for _, c in self._children.items():
                c._setup_remeshing_size(remesher, preorder)
        if (self._equations is not None):
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations.setup_remeshing_size(remesher,preorder)
            self._equations._set_current_codegen(oldcg)
        if not preorder:
            for _,c in self._children.items():
                c._setup_remeshing_size(remesher,preorder)


    def _after_mapping_on_macro_elements(self):
        if (self._mesh is not None) and (self._equations is not None):
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen) 
            self._equations.after_mapping_on_macro_elements()
            self._equations._set_current_codegen(oldcg)
        for _,c in self._children.items():
            c._after_mapping_on_macro_elements()


    def _before_newton_solve(self):
        if self._equations is not None:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen) 
            self._equations.before_newton_solve()
            self._equations._set_current_codegen(oldcg)
        for _,c in self._children.items():
            c._before_newton_solve()

    def _after_newton_solve(self):
        if self._equations is not None:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations.after_newton_solve()
            self._equations._set_current_codegen(oldcg)
        for _,c in self._children.items():
            c._after_newton_solve()

    def _before_newton_convergence_check(self)->bool:
        res=True
        if self._equations is not None:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            res=self._equations.before_newton_convergence_check(self)
            self._equations._set_current_codegen(oldcg)
        for _,c in self._children.items():
            res=c._before_newton_convergence_check() and res
        return res


    def _before_precice_initialise(self):
        if self._equations is not None:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations.before_precice_initialise(self)
            self._equations._set_current_codegen(oldcg)
        for _,c in self._children.items():
            c._before_precice_initialise()

    def _before_precice_solve(self,precice_dt:float):
        if self._equations is not None:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations.before_precice_solve(self,precice_dt)
            self._equations._set_current_codegen(oldcg)
        for _,c in self._children.items():
            c._before_precice_solve(precice_dt)

    def _after_precice_solve(self,precice_dt:float):
        if self._equations is not None:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations.after_precice_solve(self,precice_dt)
            self._equations._set_current_codegen(oldcg)
        for _,c in self._children.items():
            c._after_precice_solve(precice_dt)


    def _init_output(self,continue_info:dict[str, Any] | None=None,rank:int=0):
        if self._equations:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations._init_output(self,continue_info=continue_info,rank=rank) 
            self._equations._set_current_codegen(oldcg)
        for _,child in self._children.items():
            child._init_output(continue_info=continue_info,rank=rank)

    def _before_stationary_or_transient_solve(self, stationary:bool)->bool:
        must_reapply:bool=False
        if self._equations:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            res=self._equations._before_stationary_or_transient_solve(self, stationary)
            if res is True:
                must_reapply=True
            self._equations._set_current_codegen(oldcg)
        for _, child in self._children.items():
            must_reapply=child._before_stationary_or_transient_solve(stationary=stationary) or must_reapply
        return must_reapply

    def _before_eigen_solve(self, eigensolver:"GenericEigenSolver",angular_m:int | None=None,normal_k:float | None=None)->bool:
        must_reapply = False
        if self._equations:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            res = self._equations._before_eigen_solve(self, eigensolver,angular_m,normal_k) 
            if res is True:
                must_reapply = True
            self._equations._set_current_codegen(oldcg)
        for _, child in self._children.items():
            must_reapply = child._before_eigen_solve(eigensolver,angular_m,normal_k) or must_reapply
        return must_reapply

    def _get_forced_zero_dofs_for_eigenproblem(self,eigensolver:"GenericEigenSolver",angular_mode:int | None,normal_k:float | None)->set[str | int]:
        res:set[str | int]=set()
        if self._equations:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            res.update(self._equations._get_forced_zero_dofs_for_eigenproblem(self, eigensolver,angular_mode,normal_k)) 
            self._equations._set_current_codegen(oldcg)
        for _, child in self._children.items():
            res.update(child._get_forced_zero_dofs_for_eigenproblem(eigensolver,angular_mode,normal_k))
        return res

    def _do_output(self,step:int,stage:str):
        if self._equations:
            oldcg = self._equations._get_current_codegen()
            self._equations._set_current_codegen(self._codegen)
            self._equations._do_output(self,step,stage) 
            self._equations._set_current_codegen(oldcg)
        for _,child in self._children.items():
            child._do_output(step,stage)

    def _has_sub_equations_defined(self):
        if self._equations is not None:
            return True
        else:
            for _,v in self._children.items():
                if v._has_sub_equations_defined():
                    return True
        return False

    def _fill_dummy_equations(self,problem:"Problem",is_bulk_root:bool=True,pathname:str=""):
        if len(self._children)>0 and self._equations is None:
            if self._has_sub_equations_defined() and not is_bulk_root:
                self._equations=DummyEquations()
                self._equations._problem=problem
        if self._equations:
            self._equations._problem = problem             
            self._equations.before_fill_dummy_equations(problem,self,pathname)
            if self._equations.interior_facet_terms_required():
                if "_internal_facets_" not in self._children.keys():
                    self._children["_internal_facets_"]=EquationTree(DummyEquations(),self)
                    self._children["_internal_facets_"].get_equations()._problem=problem
        #for dn in list(self._children.keys()):
        for dn,v in self._children.items():
            #v=self._children[dn] # Cannot use .items() here
            #print(dn,v)
            v._fill_dummy_equations(problem,False,pathname=(dn if is_bulk_root else pathname+"/"+dn))
        

    def _fill_interinter_connections(self,iconns:set[str]):
        if self._equations:
            myiconns=set([x for x in iconns if x.startswith(self.get_full_path().lstrip("/"))])
            self._equations._fill_interinter_connections(self,myiconns)
        for _,v in self._children.items():
            v._fill_interinter_connections(iconns)

    def _set_parent_to_equations(self,problem:"Problem"):
        if self._codegen is not None:            
            self._codegen._set_problem(problem)
            for _,v in self._children.items():
                if v._codegen is not None:
                    v._codegen._set_bulk_element(self._codegen)
        for _, v in self._children.items():
            v._set_parent_to_equations(problem)

    def _create_dummy_domains_for_DG(self,problem:"Problem",elemdim=None):

        if elemdim is None and self._codegen is not None:
            elemdim=self._codegen.get_element_dimension()
            parent=self.get_parent()
            if parent is not None and parent._codegen is not None:
                elemdim=parent._codegen.get_element_dimension()-1
        #print("############")
        #print("ELEM DIM",elemdim)
        #print(self)
        #print("############")
        if self._equations is not None:
            assert self._codegen is not None
            cg_self=self._codegen
            if cg_self._name=="_internal_facets_":
                #print("Creating dummy domains for DG, current path:",self.get_full_path(),", elemdim:",elemdim)
                def generate_dummy_domain(source:EquationTree):
                    dummy=FiniteElementCodeGenerator()
                    dummy._set_equations(source.get_equations())
                    dummy._set_problem(problem)
                    dummy._name=source.get_my_path_name()
                    dummy._custom_domain_name=dummy._name
                    cg=source.get_code_gen()
                    nodal_dim=cg.get_nodal_dimension()
                    parent_domain=cg.get_parent_domain()
                    while nodal_dim==0 and parent_domain is not None:
                        cg=parent_domain
                        nodal_dim=cg.get_nodal_dimension()
                        parent_domain=cg.get_parent_domain()
                    dummy._set_nodal_dimension(nodal_dim)
                    return dummy

                assert self._parent is not None
                dummy=generate_dummy_domain(self) # Opposite DG facet
                dummy_p=generate_dummy_domain(self._parent) # Opposite bulk facet

                cg_self._set_opposite_interface(dummy)
                dummy._set_bulk_element(dummy_p)

                cg_self._dummy_codegen_for_internal_facets=dummy
                cg_self._dummy_codegen_for_internal_facets_bulk=dummy_p
                # TODO: This is a bit problematic
                parent=self.get_parent()
                if parent is not None:
                    grandparent=parent.get_parent()
                    if grandparent is not None and grandparent._equations is not None:
                        #print(grandparent,grandparent.get_equations())
                        dummy_pp=generate_dummy_domain(grandparent)
                        dummy_p._set_bulk_element(dummy_pp)
                        cg_self._dummy_codegen_for_internal_facets_bulk_bulk=dummy_pp
                        #dummy_po=generate_dummy_domain(grandparent)
                        #dummy_po._set_bulk_element(dummy_pp)
                        #cg_self._dummy_codegen_for_internal_facets_bulk_opp=dummy_po



                if elemdim is None or elemdim<0:
                    raise RuntimeError("Element dimension was not set correctly here...")
                bulk_bulk=cg_self._dummy_codegen_for_internal_facets_bulk_bulk
                if bulk_bulk is not None:
                    bulk_bulk._find_all_accessible_spaces()
                    bulk_bulk._do_define_fields(elemdim+2)

                bulk=cg_self._dummy_codegen_for_internal_facets_bulk
                assert bulk is not None
                bulk._find_all_accessible_spaces()
                #print("Calling do define fields on ",bulk.get_full_name(),bulk.get_domain_name(),"with",elemdim+1)
                #cg_self._transfer_my_fields_to_dummy_codegen(bulk)
                opp=cg_self._get_opposite_interface()
                assert opp is not None
                bulk._set_opposite_interface(opp)
                bulk._do_define_fields(elemdim+1)

                facets=cg_self._dummy_codegen_for_internal_facets
                assert facets is not None
                facets._coordinates_as_dofs=bulk._coordinates_as_dofs
                facets._coordinate_space=bulk._coordinate_space
#                facets._define_fields()
                facets._find_all_accessible_spaces()
                #facets.define_scaling()
                facets._do_define_fields(elemdim)

            backup=self.setup_codegen_to_equations()
            self._equations.after_fill_dummy_equations(problem,self,self.get_full_path(),elem_dim=elemdim)
            self.setup_codegen_to_equations(reset_info=backup)

        for _, v in self._children.items():
            v._create_dummy_domains_for_DG(problem,elemdim=None if elemdim is None else elemdim-1)


    #This will create new equations multiple occuring equations (Important, since the same equation might occur on different nodal dims, etc)
    def _finalize_equations(self,problem:"Problem",second_loop:bool=False):
        if self._equations is not None:
            if self._codegen is None:
                self._codegen=FiniteElementCodeGenerator()                
                self._codegen.ccode_expression_mode=problem.default_ccode_expression_mode
                if problem.debug_jacobian_by_fd_epsilon is not None and problem.debug_jacobian_by_fd_epsilon>0.0:
                    self._codegen.debug_jacobian_epsilon=problem.debug_jacobian_by_fd_epsilon
                self._codegen._name=self.get_my_path_name()
                self._codegen.set_latex_printer(problem.latex_printer)
                self._codegen._set_problem(problem) 
                if second_loop and self._equations._is_ode():
                    self._codegen._set_equations(self._equations)
                    backup=self.setup_codegen_to_equations(with_bulk_and_opp=False)                    
                    self.setup_codegen_to_equations(reset_info=backup)
                    meshname=self.get_my_path_name()
                    #print(meshname)
                    #print("Creating ODE storage mesh for ",meshname)
                    mesh=ODEStorageMesh(problem,self,meshname)
                    self.get_code_gen()._mesh=mesh 
                    problem._meshdict[meshname]=mesh

        if self._codegen:
            self._codegen._set_problem(problem)
            # _codegen is only ever created above, inside the "if self._equations is not None:" branch,
            # so its presence implies self._equations is also set.
            assert self._equations is not None
            self._codegen._set_equations(self._equations)
        for _,v in self._children.items():
            v._finalize_equations(problem,second_loop=second_loop)


    def get_parent(self) -> "EquationTree | None":
        return self._parent

    def get_full_path(self,for_child:"EquationTree | None"=None,sep:str="/")->str:
        if self._parent is not None:
            trunk=self._parent.get_full_path(self,sep=sep)
        else:
            trunk=""
            if for_child is None:
                return sep
        if for_child is not None:
            for k,v in self._children.items():
                if v is for_child:
                    return trunk+sep+k
        while sep!="/" and trunk.startswith(sep):
            trunk=trunk[len(sep):]
        return trunk

    def get_my_path_name(self) -> str:
        if self._parent is None:
            return "/"
        else:
            for k,v in self._parent._children.items():
                if v==self:
                    return k
        raise RuntimeError("Error in equation tree")

    def get_by_path(self,path:str)->"EquationTree | None":
        if path=="":
            return self
        pth=path.split("/")
        chld=self._children.get(pth[0])
        if chld is None:
            return None
        else:
            return chld.get_by_path("/".join(pth[1:]))

    def _create_dummy_equations_at_path(self,path:str,root:"EquationTree",problem:"Problem"):
        if (self._equations is None) and (self!=root):
            self._equations=DummyEquations()            
            self._equations._problem=problem
        if path=="":
            return
        pth=path.split("/")
        chld=self._children.get(pth[0])
        if chld is None:
            self._children[pth[0]]=EquationTree(DummyEquations(),parent=self)
            self._children[pth[0]].get_equations()._problem=problem
        self._children.get(pth[0])._create_dummy_equations_at_path("/".join(pth[1:]),root,problem) #type:ignore

    def get_child(self, name:str) -> "EquationTree":
        res = self._children.get(name)
        if res is None:
            raise ValueError("No sub-equation path '" + name + "' found at '" + self.get_full_path() + "'")
        return res

    def __matmul__(self, other:str)->"EquationTree":
        if isinstance(other, str): #type:ignore
            splt = other.split("/")
            splt = [x for x in splt if x]  # Remove empties
            if len(splt) == 0:
                raise ValueError("Please restrict equations with a non-empty domain name")
            res = EquationTree(None, parent=None)
            res._children[splt[-1]] = self
            if not (self._parent is None):
                raise RuntimeError("Part of an equation tree is multiple times present in the entire tree")            
            res._name = splt[-1]
            _check_for_valid_var_name(res._name,True)
            self._parent = res
            for k in reversed(splt[:-1]):
                res = res @ k
            return res
        else:
            raise ValueError(
                "Please combine equation with a string (name of the domain) to restrict the equations to a domain")

    def __radd__(self, other:Literal[0])->"EquationTree":
        if other==0:
            return self
        else:
            raise RuntimeError("Cannot add "+str(other)+" and "+str(self))

    def __add__(self, other:"EquationTree | BaseEquations | Literal[0]")->"EquationTree":
        if other==0:
            return self
        if isinstance(other,EquationTree):
            res=EquationTree(self._equations,parent=None)
            if other._equations:
                res._equations=(res._equations+other._equations if res._equations is not None else other._equations)
            for k,v in self._children.items():
                if k in other._children.keys():
                    res._children[k]=v+other._children[k]
                else:
                    res._children[k]=v
                res._children[k]._parent=res
            for k,v in other._children.items():
                if not k in self._children.keys():
                    res._children[k]=v
                    res._children[k]._parent = res
            return res
        elif isinstance(other,BaseEquations): #type:ignore
            return self+EquationTree(other,parent=None)
        else:
            raise RuntimeError("Cannot combine a EquationTree by adding "+repr(self)+" and "+repr(other))

    def numerical_factors_to_string(self,indent:str="")->str:
        pth = self.get_my_path_name()
        res = indent
        if self._equations is not None:
            res += "--" + pth + " : " #+ self._equations._tree_string(indent + " " * (len(pth) + 6))
            assert self._codegen is not None
            for k,v in self._codegen._named_numerical_factors.items():
                res = res + "\n" + indent + " " * (len(pth) + 6) + str(k)+" = "+str(v)
        elif self._parent is not None:
            res += "--" + pth
        else:
            res += pth
        for k, v in self._children.items():
            res = res + "\n" + v.numerical_factors_to_string(indent + (" " * 2 if pth != "/" else "") + "|")
        return res

    def _tree_string(self,indent:str="") -> str:
        pth=self.get_my_path_name()
        res=indent
        if self._equations is not None:
            res+="--"+pth+" : "+self._equations._tree_string(indent+" "*(len(pth)+6)) 
        elif self._parent is not None:
            res+="--"+pth
        else:
            res+=pth
        for _,v in self._children.items():
            res=res+"\n"+v._tree_string(indent+(" "*2 if pth!="/" else "")+"|")
        return res

    def __str__(self) -> str:
        return self._tree_string()




class Equations(BaseEquations):
    """
    Equations to be solved on a domain, i.e. including spatial coordinates.
    Add unknown fields by overriding the :py:meth:`~BaseEquations.define_fields` method and residuals by overriding the :py:meth:`~BaseEquations.define_residuals` method.
    
    See :py:class:`~BaseEquations` for further methods.
    """
    def __init__(self):
        super().__init__()
        self._coordinates_as_dofs = False
        self._vectorfields:dict[str,list[str]]={}
        self._tensorfields:dict[str,list[list[str]]]={}

    def get_global_dof_storage_name(self,pathname:str | None=None):
        if pathname is None:
            pathname=self.get_current_code_generator().get_full_name()
        dofstorage="_meshwide__"+pathname.lstrip("/").replace("/","__")
        return dofstorage

    def get_weak_dirichlet_terms_for_DG(self,fieldname:str,value:"ExpressionOrNum")->"ExpressionNumOrNone":
        """
        Returns the weak Dirichlet terms for a discontinuous Galerkin (DG) formulation. When using a :py:class:`~pyoomph.meshes.bcs.DirichletBC` with ``prefer_weak_for_DG``, this method is called. If it returns not ``None``, the :py:class:`~pyoomph.meshes.bcs.DirichletBC` is not enforced strongly, but on the basis of the given interface residuals.

        Args:
            fieldname: Name of the field for which the weak Dirichlet terms are to be returned.
            value: The desired Dirichlet condition.
        """
        return None

    def get_mesh(self)->"AnySpatialMesh":
        from ..meshes.mesh import MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh
        mesh=super().get_mesh()
        assert isinstance(mesh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh))
        return mesh

    def get_list_of_vector_fields(self,codegen:"FiniteElementCodeGenerator")->list[dict[str,list[str]]]:
        vector_fields:list[dict[str,list[str]]]=[]
        current=self
        if hasattr(current, "_vectorfields"):
            vector_fields.append(current._vectorfields)
        #check_spaces = {"C2TB", "C2", "C1"}
        #allfields = codegen.get_all_fieldnames(check_spaces)
        parent = codegen._get_parent_domain()
        while parent is not None:
            assert isinstance(parent,FiniteElementCodeGenerator)
            peqs=parent.get_equations()
            
            if isinstance(peqs,Equations):
                vector_fields.append(peqs._vectorfields) 
            parent = parent._get_parent_domain()
        return vector_fields


    def _is_ode(self)->bool | None:
        return False

    @overload
    def get_opposite_side_of_interface(self,raise_error_if_none:Literal[True]=...)->FiniteElementCodeGenerator: ...

    @overload
    def get_opposite_side_of_interface(self,raise_error_if_none:Literal[False])->FiniteElementCodeGenerator | None: ...

    def get_opposite_side_of_interface(self,raise_error_if_none:bool=True)->FiniteElementCodeGenerator | None:
        """
        Returns the interface domain at the opposite side of this interface.

        Args:
            raise_error_if_none: If there is no opposite side set, raise an error. Otherwise, just ``None`` is returned.

        Returns:
            The interface domain at the opposite side.
        """
        master = self._get_combined_element()
        cg=master._assert_codegen()
        if cg._get_opposite_interface() is None:
            if raise_error_if_none:
                raise RuntimeError("The interface has no opposite side set")
            return None
        if master.get_parent_domain() is None:
            raise RuntimeError("Can only have opposite interface sides on interfaces, not on bulk equations")
        res=cg._get_opposite_interface()
        assert isinstance(res,FiniteElementCodeGenerator)
        return res

    @overload
    def get_opposite_parent_domain(self,raise_error_if_none:Literal[True]=...)->FiniteElementCodeGenerator: ...

    @overload
    def get_opposite_parent_domain(self,raise_error_if_none:Literal[False])->FiniteElementCodeGenerator | None: ...

    def get_opposite_parent_domain(self,raise_error_if_none:bool=True)->FiniteElementCodeGenerator | None:
        """
        Returns the bulk domain at the opposite side of this interface.

        Args:
            raise_error_if_none: If there is no opposite side set, raise an error. Otherwise, just ``None`` is returned.
        
        Returns:
            The bulk domain at the opposite side.
        """
        if raise_error_if_none:
            opp_inter=self.get_opposite_side_of_interface(raise_error_if_none=True)
        else:
            opp_inter=self.get_opposite_side_of_interface(raise_error_if_none=False)
        if opp_inter is None:
            if raise_error_if_none:
                raise RuntimeError("The interface has no opposite side set")
            return None
        res=opp_inter.get_parent_domain()
        if raise_error_if_none and (res is None):
            raise RuntimeError("The interface has no opposite bulk")
        return res


    def activate_coordinates_as_dofs(self, coordinate_space: str | None = None) -> None:
        """
        Activates the coordinates as degrees of freedom (dofs) for a moving mesh. You must then add residuals in the define_residuals method for the field "mesh",
        e.g. 
        
            def define_residuals(self):
             x,xtest=var_and_test("mesh")
             X=var("lagrangian")
             self.add_weak(grad(x-X,lagrangian=True),grad(xtest,lagrangian=True),lagrangian=True)
             
        for a Laplace-smoothed mesh

        Args:
            coordinate_space (Optional[str]): The coordinate space to be set as the coordinate space for the element. Valid options are "C2TB", "C2", "C1TB", or "C1". If not provided, the coordinate space will not be set.

        Raises:
            ValueError: If the provided coordinate space is not one of the valid options.

        Returns:
            None
        """
        master = self._get_combined_element()  # TODO This does not allow for dx on individual coordinate systems
        cg = master._assert_codegen()
        cg._coordinates_as_dofs = True
        if coordinate_space is not None:
            cg._coordinate_space = coordinate_space
            if coordinate_space not in ["C2TB", "C2", "C1TB", "C1"]:
                raise ValueError("Can only set the coordinate space to either C2TB, C2, C1TB or C1")
        rcomponent="_x"
        zcomponent="_y"
        csys=self.get_coordinate_system()
        if isinstance(csys,AxisymmetricCoordinateSystem):
            if csys.use_x_as_symmetry_axis:
                rcomponent="_y"
                zcomponent="_x"
        if self.get_nodal_dimension()<2:
            zcomponent=None
        # x-axis is always pinned
        master._azimuthal_r0_info[0].add("mesh"+rcomponent)
        master._azimuthal_r0_info[1].add("mesh"+rcomponent)
        master._azimuthal_r0_info[2].add("mesh"+rcomponent)
        if zcomponent is not None:
            master._azimuthal_r0_info[1].add("mesh"+zcomponent)
            master._azimuthal_r0_info[2].add("mesh"+zcomponent)
    
    def _internal_define_scalar_field(self,name:str, space:"FiniteElementSpaceEnum", scale:"ExpressionOrNum | str | None"=None, testscale:"ExpressionOrNum | str | None"=None, discontinuous_refinement_exponent:float | None=None,allow_scales_with_fields:bool=False):
        master = self._get_combined_element()
        pdom=master.get_parent_domain()
        if pdom is not None:
            # Check a bit what is possible
            if space=="C2TB" or space=="D2TB":
                if pdom._coordinate_space!="C2TB":
                    raise self.add_exception_info(RuntimeError("You tried to define a "+str(space)+" field '"+str(name)+"' at an interface attached to a bulk domain with element space "+str(pdom._coordinate_space)+". This does not work"))
            elif space=="C2" or space=="D2":
                if pdom._coordinate_space not in {"C2TB","C2"}:
                    raise self.add_exception_info(RuntimeError("You tried to define a "+str(space)+" field '"+str(name)+"' at an interface attached to a bulk domain with element space "+str(pdom._coordinate_space)+". This does not work"))
            elif space=="C1TB" or space=="D1TB":    
                if pdom._coordinate_space=="C1":
                    raise self.add_exception_info(RuntimeError("You tried to define a "+str(space)+" field '"+str(name)+"' at an interface attached to a bulk domain with element space "+str(pdom._coordinate_space)+". This does not work"))
                elif pdom._coordinate_space=="C2" or pdom._coordinate_space=="C1TB":
                    if pdom.dimension==3:
                        raise self.add_exception_info(RuntimeError("You tried to define a "+str(space)+" field '"+str(name)+"' at an interface attached to 3d bulk domain with element space "+str(pdom._coordinate_space)+". This does not work, since 3d tetrahedral elements of "+str(pdom._coordinate_space)+" do not provide the face bubble node for "+str(space)+" on 2d facets. Consider upgrading the 3d space to C2TB using an ElementSpace('C2TB') for the 3d domain or adjust the facet space to "+("C1" if space=="C1TB" else "D1")+"."))
            
            
        if _pyoomph.get_verbosity_flag() != 0:
            print("REGISTER", name, self, master, self == master, space)
        master._register_field(name, space)
        self._fields_defined_on_my_domain[name]=space
        master._fields_defined_on_my_domain[name]=space
        cg = master._assert_codegen()
        if discontinuous_refinement_exponent is not None:
            if discontinuous_refinement_exponent!=0:
                if space!="D0":
                    raise RuntimeError("Discontinuous refinement exponents only work for D0 at the moment")
                cg._set_discontinuous_refinement_exponent(name,discontinuous_refinement_exponent)
        # cg._coordinate_space (a plain str C++ property) can legitimately be "" (not yet set) here;
        # find_dominant_element_space() explicitly handles that empty-string sentinel, even though it
        # is not part of the FiniteElementSpaceEnum literal, so a validating cast would wrongly reject it.
        cg._coordinate_space=find_dominant_element_space(cast("FiniteElementSpaceEnum",cg._coordinate_space),space)
        cg._fields_defined_on_my_domain[name]=space
                
        if scale is not None:
            self.set_scaling({name: scale},allow_scales_with_fields=allow_scales_with_fields)
        if testscale is not None:
            self.set_test_scaling({name:testscale},allow_scales_with_fields=allow_scales_with_fields)
            
        # Scalar fields are pinned by default for |m|=1 and |m|>=2
        if space!="D0" and space!="DL":
            master._azimuthal_r0_info[1].add(name)
            master._azimuthal_r0_info[2].add(name)


    def define_scalar_field(self, name:str | list[str], space:"FiniteElementSpaceEnum",scale:"ExpressionOrNum | str | None"=None,testscale:"ExpressionOrNum | str | None"=None,discontinuous_refinement_exponent:float | None=None,allow_scales_with_fields:bool=False):
        """
        Define a scalar field on this domain. Must be called within the specified implementation of the method :py:meth:`~BaseEquations.define_fields`.

        Args:
            name (str): The name of the vector field.
            space (FiniteElementSpaceEnum): The finite element space on which the vector field is defined.
            scale (ExpressionNumOrNone): The scale for the vector field for nondimensionalization. Defaults to None.
            testscale (ExpressionNumOrNone): The scale for the test function of the vector field for nondimensionalization. Defaults to None.
            discontinuous_refinement_exponent (Optional[float]): The exponent for the discontinuous refinement. Defaults to None.
            allow_scales_with_fields (bool): Whether to allow scales/testscales with fields included. Defaults to False.
        """
        if not isinstance(name, str):
            for n in name:
                self.define_scalar_field(n, space, scale=scale, testscale=testscale,discontinuous_refinement_exponent=discontinuous_refinement_exponent,allow_scales_with_fields=allow_scales_with_fields)
            return        
        # BaseCoordinateSystem.define_scalar_field is annotated with scale/testscale:ExpressionOrNum|None,
        # but it just forwards them unchanged to Equations._internal_define_scalar_field, which does accept
        # a str (a named scale reference) as well - so passing a str through here is safe at runtime.
        self.get_coordinate_system().define_scalar_field(name, space, self,scale,testscale,discontinuous_refinement_exponent,allow_scales_with_fields=allow_scales_with_fields) #type:ignore

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", dim:int | None=None,scale:"ExpressionNumOrNone"=None,testscale:"ExpressionNumOrNone"=None,allow_scales_with_fields:bool=False):
        """
        Define a vector field on this domain. Must be called within the specified implementation of the method :py:meth:`~BaseEquations.define_fields`.

        Args:
            name (str): The name of the vector field.
            space (FiniteElementSpaceEnum): The finite element space on which the vector field is defined.
            dim (Optional[int]): The dimension of the vector field. If not provided, it defaults to the nodal dimension.
            scale (ExpressionNumOrNone): The scale for the vector field for nondimensionalization. Defaults to None.
            testscale (ExpressionNumOrNone): The scale for the test function of the vector field for nondimensionalization. Defaults to None.
        """
                   
        dim = dim if dim is not None else self.get_nodal_dimension()  # TODO: Here, it should be nodal_dimension!
        v, vtest,comps = self.get_coordinate_system().define_vector_field(name, space, dim, self)
        also_on_interface = space in {"C1","D1","C2","D2","C1TB","D1TB","C2TB","D2TB"}
        mst=self._get_combined_element()
        assert isinstance(mst,Equations)
        mst._vectorfields[name]=comps
        self.define_field_by_substitution(name, vector(*v), also_on_interface=also_on_interface)
        self.define_testfunction_by_substitution(name, vector(*vtest),
                                                 also_on_interface=also_on_interface)
        if scale is not None:
            self.set_scaling({name:scale}, allow_scales_with_fields=allow_scales_with_fields)
        if testscale is not None:
            self.set_test_scaling({name:testscale}, allow_scales_with_fields=allow_scales_with_fields)

        # Vector fields are pinned by default for |m|=1 and |m|>=2
        if space!="D0":
            rcomponent="_x"
            zcomponent="_y"
            csys=self.get_coordinate_system()
            if isinstance(csys,AxisymmetricCoordinateSystem):
                if csys.use_x_as_symmetry_axis:
                    rcomponent="_y"
                    zcomponent="_x"
            if dim<2:
                zcomponent=None
            mst._azimuthal_r0_info[0].add(name+rcomponent)
            if name+rcomponent in mst._azimuthal_r0_info[1]:
                mst._azimuthal_r0_info[1].remove(name+rcomponent)
            mst._azimuthal_r0_info[2].add(name+rcomponent)
            if zcomponent is not None:
                mst._azimuthal_r0_info[1].add(name+zcomponent)
                mst._azimuthal_r0_info[2].add(name+zcomponent)
            if isinstance(csys,AxisymmetryBreakingCoordinateSystem):
                mst._azimuthal_r0_info[0].add(name+"_phi")          
                if name+"_phi" in mst._azimuthal_r0_info[1]:
                    mst._azimuthal_r0_info[1].remove(name+"_phi")      
                mst._azimuthal_r0_info[2].add(name+"_phi")
                
    

    def define_tensor_field(self, name:str, space:"FiniteElementSpaceEnum", dim:int | None=None,scale:"ExpressionNumOrNone"=None,testscale:"ExpressionNumOrNone"=None, symmetric:bool=False,allow_scales_with_fields:bool=False):
        dim = dim if dim is not None else self.get_nodal_dimension()  # TODO: Here, it should be nodal_dimension!
        t, ttest,comps = self.get_coordinate_system().define_tensor_field(name, space, dim, self, symmetric)
        also_on_interface:bool = space in { "C1","C2","C1TB","C2TB","D2TB","D2","D1","D1TB"}
        mst=self._get_combined_element()
        assert isinstance(mst,Equations)
        mst._tensorfields[name]=comps
        # list is invariant, but matrix() only reads its argument, so a list[list[Expression]] is safe here
        # even though it is not (structurally) assignable to list[list[ExpressionOrNum]].
        self.define_field_by_substitution(name, matrix(cast("list[list[ExpressionOrNum]]",t)), also_on_interface=also_on_interface)
        self.define_testfunction_by_substitution(name, matrix(cast("list[list[ExpressionOrNum]]",ttest)),also_on_interface=also_on_interface)
        if scale is not None:
            self.set_scaling({name:scale}, allow_scales_with_fields=allow_scales_with_fields)
        if testscale is not None:
            self.set_test_scaling({name:testscale}, allow_scales_with_fields=allow_scales_with_fields)
        # TODO: set the azimuthal r=0 info for tensor fields



    def get_nodal_delta(self) -> Expression:
        return nondim("_nodal_delta")

    def add_spatial_error_estimator(self, expr:"Expression",for_base:bool=True,for_eigen:bool=True):
        master = self._get_combined_element()
        cg=master._assert_codegen()
        if for_base:
            cg._add_Z2_flux(expr,False)
        if for_eigen:
            cg._add_Z2_flux(expr,True)




class DummyEquations(Equations):
    def define_fields(self):
        pass
    def define_residuals(self):
        pass


_ode_coordinate_system = ODECoordinateSystem()


class ODEEquations(BaseEquations):
    """
    Class representing a set of ordinary differential equations (ODEs).
    Add unknowns by overriding the :py:meth:`~BaseEquations.define_fields` method and residuals by overriding the :py:meth:`~BaseEquations.define_residuals` method.
    
    See :py:class:`~BaseEquations` for further methods.
    """

    def __init__(self):
        """
        Call this method to initialize the ODE equations, e.g. usually you pass parameters here which can be then used in the equations within the define_residuals method.
        Also, you must call super().__init__() in the derived class before anything else.
        """
        super(ODEEquations, self).__init__()
        self._pinned_dofs: dict[str, bool | float] = {}

    def pin(self, **kwargs: bool | float):
        """
        Pins the specified degrees of freedom (DOFs).

        Args:
            **kwargs (Union[bool, float]): Keyword arguments representing the DOFs to be pinned and their values. If assigning True, we just fix the current value, otherwise, the provided value is set.
        """
        self._pinned_dofs.update(kwargs)

    def unpin(self, *args: str):
        """
        Unpins the specified degrees of freedom (DOFs).

        Args:
            *args(str): Variable length argument representing the DOFs to be unpinned.
        """
        for k in args:
            if k in self._pinned_dofs.keys():
                self._pinned_dofs[k] = False

    def _is_ode(self) -> bool | None:
        """
        Checks if the equations are ordinary differential equations (ODEs).

        Returns:
            bool: True if the equations are ODEs, False otherwise.
        """
        return True

    def get_mesh(self) -> "ODEStorageMesh":
        """
        Returns the ODE storage mesh.

        Returns:
            ODEStorageMesh: The ODE storage mesh.
        """
        from ..meshes.mesh import ODEStorageMesh
        mesh = super().get_mesh()
        assert isinstance(mesh, ODEStorageMesh)
        return mesh

    def get_coordinate_system(self) -> BaseCoordinateSystem:
        """
        Returns the base coordinate system.

        Returns:
            BaseCoordinateSystem: The base coordinate system.
        """
        return _ode_coordinate_system

    def define_ode_variable(self, *names: str, scale: "ExpressionOrNum | None" = None,
                            testscale: "ExpressionOrNum | None" = None) -> None:
        """
        Defines the ODE variables.

        Args:
            *names: Variable length argument representing the names of the ODE variable(s).
            scale: Optional scaling factor for the ODE variable(s).
            testscale: Optional scaling factor for the test functions associated with the ODE variable(s).
        """
        for name in names:
            master = self._get_combined_element()
            if _pyoomph.get_verbosity_flag() != 0:
                print("REGISTER", name, self, master, self == master)
            master._register_field(name, "D0")
            master._fields_defined_on_my_domain[name] = "D0"
            self._fields_defined_on_my_domain[name] = "D0"
        if scale is not None:
            self.set_scaling({n: scale for n in names})
        if testscale is not None:
            self.set_test_scaling({n: testscale for n in names})
        # cg = master._assert_codegen()

    def expand_additional_field(self, name: str, dimensional: bool, expression: _pyoomph.Expression,
                                in_domain: _pyoomph.FiniteElementCode, no_jacobian: bool, no_hessian: bool,
                                where: str) -> _pyoomph.Expression:
        """
        Expands additional fields for the ODEs.

        Args:
            name: The name of the additional field.
            dimensional: A boolean indicating if the field is dimensional.
            expression: The expression defining the additional field.
            in_domain: The finite element code representing the domain.
            no_jacobian: A boolean indicating if the Jacobian should not be computed.
            no_hessian: A boolean indicating if the Hessian should not be computed.
            where: The location where the additional field is expanded.

        Returns:
            Expression: The expanded additional field.
        """
        if _pyoomph.get_verbosity_flag() != 0:
            print("ADD field", name)
        if name == "mesh" or name == "lagrangian":
            return vector(0)
        elif name == "mesh_x" or name == "mesh_y" or name == "mesh_z" or name == "coordinate_x" or \
                name == "coordinate_y" or name == "coordinate_z":
            return _pyoomph.Expression(0)
        elif name == "lagrangian_x" or name == "lagrangian_y" or name == "langrangian_z":
            return _pyoomph.Expression(0)
        elif name == "local_coordinate_1" or name == "local_coordinate_2" or name == "local_coordinate_3":
            return _pyoomph.Expression(0)
        else:
            return super(ODEEquations, self).expand_additional_field(name, dimensional, expression,
                                                                     in_domain, no_jacobian, no_hessian, where)

    def get_parent_domain(self):
        """
        Returns the parent domain.

        Returns:
            None: The parent domain.
        """
        return None

    def on_apply_boundary_conditions(self, mesh: "AnyMesh"):
        """
        Applies boundary conditions to the ODEs.

        Args:
            mesh: The mesh to which the boundary conditions are applied.
        """
        from ..meshes.mesh import ODEStorageMesh
        assert isinstance(mesh, ODEStorageMesh)
        e = mesh.get_element()
        _, inds = e._ode_elem_to_numpy()
        for k, v in self._pinned_dofs.items():
            if not (k in inds.keys()):
                raise RuntimeError("Cannot pin the degree " + str(k) + " since it is not defined on this ODE: "
                                   "Possible degrees are: " + ",".join(inds.keys()))
            index = inds[k]
            if not v is False:
                e.internal_data_pt(index).pin(0)
                if v is not True:
                    fval = v  # TODO NONDIM!
                    e.internal_data_pt(index).set_value(0, fval)
            else:
                e.internal_data_pt(index).unpin(0)


class CombinedEquations(Equations):

    def get_weak_dirichlet_terms_for_DG(self,fieldname:str,value:"ExpressionOrNum")->"ExpressionNumOrNone":
        res=None
        for e in self._subelements:
            if isinstance(e,Equations):
                contrib=e.get_weak_dirichlet_terms_for_DG(fieldname,value)
                if contrib is not None:
                    if res is None:
                        res=contrib
                    else:
                        res+=contrib
        return res

        
    
    def setup_remeshing_size(self,remesher:"RemesherBase",preorder:bool):
        for e in self._subelements:
            e.setup_remeshing_size(remesher,preorder)
            
    def change_output_directory(self,newdir:str,eqtree:"EquationTree"):
        for e in self._subelements:
            e.change_output_directory(newdir,eqtree)

    def after_fill_dummy_equations(self,problem:"Problem",eqtree:"EquationTree",pathname:str,elem_dim:int | None=None):
        for e in self._subelements:
            e.after_fill_dummy_equations(problem,eqtree,pathname,elem_dim)

    def before_precice_initialise(self,eqtree:"EquationTree"):
        for e in self._subelements:
            e.before_precice_initialise(eqtree)


    def before_precice_solve(self,eqtree:"EquationTree",precice_dt:float):
        for e in self._subelements:
            e.before_precice_solve(eqtree,precice_dt)

    def after_precice_solve(self,eqtree:"EquationTree",precice_dt:float):
        for e in self._subelements:
            e.after_precice_solve(eqtree,precice_dt)


    def calculate_error_overrides(self):
        for e in self._subelements:
            e.calculate_error_overrides()

    def before_mesh_to_mesh_interpolation(self,eqtree:"EquationTree",interpolator:"BaseMeshToMeshInterpolator"):
        for e in self._subelements:
            e.before_mesh_to_mesh_interpolation(eqtree,interpolator)

    def interior_facet_terms_required(self):        
        for e in self._subelements:
            if e.requires_interior_facet_terms:
                return True
        return False

    def _is_ode(self)->bool | None:
        res=None
        for e in self._subelements:
            isode=e._is_ode()
            if isode==True:
                if res is None:
                    res=True
                elif res==False:
                    info=[repr(ei)+" is ODE: "+str(ei._is_ode()) for ei in self._subelements]
                    raise RuntimeError("Combined Equations and ODEEquations does not work yet:\n"+"\n".join(info))
            elif isode==False:
                if res is None:
                    res=False
                elif res==True:
                    info = [repr(ei) + " is ODE: " + str(ei._is_ode()) for ei in self._subelements]
                    raise RuntimeError("Combined Equations and ODEEquations does not work yet:\n" + "\n".join(info))
        return res
        

    def _tree_string(self,indent:str) -> str:
        res="Combined Equations:"
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                res=res+"\n"+indent+e._tree_string(indent)
        return res
    
    def _fill_interinter_connections(self,eqtree:"EquationTree",interinter:set[str]):
        for e in self._subelements:
            if isinstance(e, BaseEquations):
                e._fill_interinter_connections(eqtree,interinter)

    def _init_output(self,eqtree:"EquationTree",continue_info:dict[str, Any] | None,rank:int):
        for e in self._subelements:
            if isinstance(e,BaseEquations): #type:ignore
                e._init_output(eqtree,continue_info,rank)

    def _do_output(self,eqtree:"EquationTree",step:int,stage:str):
        for e in self._subelements:
            if isinstance(e,BaseEquations): #type:ignore
                e._do_output(eqtree,step,stage)

    def _before_stationary_or_transient_solve(self, eqtree:"EquationTree", stationary:bool)->bool:
        must_reapply=False
        for e in self._subelements:
            if isinstance(e,BaseEquations): #type:ignore
                must_reapply=must_reapply or e._before_stationary_or_transient_solve(eqtree, stationary)
        return must_reapply

    def _before_eigen_solve(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver",angular_m:int | None=None,normal_k:float | None=None) -> bool:
        must_reapply=False
        for e in self._subelements:
            if isinstance(e,BaseEquations):  #type:ignore
                must_reapply=must_reapply or e._before_eigen_solve(eqtree, eigensolver,angular_m,normal_k)
        return must_reapply

    def _get_forced_zero_dofs_for_eigenproblem(self,eqtree:"EquationTree", eigensolver:"GenericEigenSolver",angular_mode:int | None,normal_k:float | None)->set[str | int]:
        res:set[str | int] = set()
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                upd=e._get_forced_zero_dofs_for_eigenproblem(eqtree, eigensolver,angular_mode,normal_k)
                #print("UPDATE WITH",upd)
                res.update(upd)
                #print("UPDATE IS",res)
        return res


    def on_apply_boundary_conditions(self,mesh:"AnyMesh"):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e,BaseEquations): #type:ignore
                e.on_apply_boundary_conditions(mesh)
        self._rstr_final_elem(bck)

    def before_fill_dummy_equations(self,problem:"Problem",eqtree:"EquationTree",pathname:str):
        #bck = self._bckup_final_elem()
        #self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e,BaseEquations): #type:ignore
                e.before_fill_dummy_equations(problem,eqtree,pathname)
        #self._rstr_final_elem(bck)
        

    def before_finalization(self,codegen:FiniteElementCodeGenerator):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                e.before_finalization(codegen)
        self._rstr_final_elem(bck)

    def before_compilation(self,codegen:FiniteElementCodeGenerator):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e,BaseEquations): #type:ignore
                e.before_compilation(codegen)
        self._rstr_final_elem(bck)

    def after_compilation(self,codegen:FiniteElementCodeGenerator):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e,BaseEquations): #type:ignore
                e.after_compilation(codegen)
        self._rstr_final_elem(bck)

    def after_newton_solve(self):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                e.after_newton_solve()
        self._rstr_final_elem(bck)

    def before_newton_convergence_check(self,eqtree:"EquationTree")->bool:
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        res=True
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                res=e.before_newton_convergence_check(eqtree) and res
        self._rstr_final_elem(bck)   
        return res     

    def before_newton_solve(self):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                e.before_newton_solve()
        self._rstr_final_elem(bck)

    def before_assigning_equations_preorder(self, mesh:"AnyMesh"):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                e.before_assigning_equations_preorder(mesh)
        self._rstr_final_elem(bck)

    def before_assigning_equations_postorder(self, mesh:"AnyMesh"):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations):  #type:ignore
                e.before_assigning_equations_postorder(mesh)
        self._rstr_final_elem(bck)

    def after_remeshing(self,eqtree:"EquationTree"):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations):  #type:ignore
                e.after_remeshing(eqtree)
        self._rstr_final_elem(bck)

    def _release_output_files(self)->None:
        for e in self._subelements:
            if isinstance(e, BaseEquations):  #type:ignore
                e._release_output_files()

    def after_mapping_on_macro_elements(self):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations):  #type:ignore
                e.after_mapping_on_macro_elements()
        self._rstr_final_elem(bck)

    def sanity_check(self):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations):  #type:ignore
                e.sanity_check()
        self._rstr_final_elem(bck)

    def __init__(self, *args:BaseEquations):
        super(CombinedEquations, self).__init__()
        self._subelements:list[BaseEquations] = [*args]

    def _bckup_final_elem(self):
        return [e._get_combined_element() for e in self._subelements]

    def _rstr_final_elem(self,bck:list[BaseEquations]):
        for e,fe in zip(self._subelements,bck):
            e._set_final_element(fe)

    def _setup_combined_element(self):
        for e in self._subelements:
            e._set_final_element(self)


    @overload
    def get_equation_of_type(self, typ:type[BaseEquations], *, exact_type:bool=False,always_as_list:Literal[True])->list["BaseEquations"]: ...

    @overload
    def get_equation_of_type(self, typ:type[BaseEquations], *, exact_type:bool=False,always_as_list:Literal[False]=False)->"BaseEquations | None": ...

    def get_equation_of_type(self, typ:type[BaseEquations], *, exact_type:bool=False,always_as_list:bool=False)->'list["BaseEquations"] | BaseEquations | None':
        res:list["BaseEquations"] | None = None
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                localL = e.get_equation_of_type(typ, exact_type=exact_type,always_as_list=always_as_list) #type:ignore
                if localL is not None:
                    if not isinstance(localL, list):
                        local = [localL]
                    else:
                        local= localL
                    if res is None:
                        res = local
                    elif isinstance(res, list): #type:ignore
                        res = res + local
                    else:
                        res = [res] + local
        if isinstance(res, list) and len(res) == 1 and not always_as_list:
            return res[0]
        if res is None:
            return [] if always_as_list else None
        if always_as_list and not (isinstance(res,list)):  #type:ignore
            res=[res]
        return res

    def _set_final_element(self, final:BaseEquations | None):
        for e in self._subelements:
            e._set_final_element(final)
        self._final_element = final

    def define_scaling(self):
        bck=self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                e.define_scaling()
        self._rstr_final_elem(bck)

    def define_fields(self):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if _pyoomph.get_verbosity_flag() != 0:
                print("DEF SUB", e)
            if isinstance(e, BaseEquations): #type:ignore
                e.define_fields()
        self._rstr_final_elem(bck)

    def define_residuals(self):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        retres=None
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                ret=e.define_residuals()
                if ret is not None:
                    if retres is None:
                        retres=ret
                    else:
                        retres+=ret
        self._rstr_final_elem(bck)
        return retres

    def define_error_estimators(self):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                e.define_error_estimators()
        self._rstr_final_elem(bck)

    def define_additional_functions(self):
        bck = self._bckup_final_elem()
        self._setup_combined_element()
        for e in self._subelements:
            if isinstance(e, BaseEquations): #type:ignore
                e.define_additional_functions()
        self._rstr_final_elem(bck)
        
    def get_coordinate_system(self) -> BaseCoordinateSystem:
        if self._is_ode() is True:
            return _ode_coordinate_system
        else:
            return super(CombinedEquations, self).get_coordinate_system()

    def __iter__(self):
        return iter(self._subelements)


class InterfaceEquations(Equations):
    """
    Same as normal :py:class:`~pyoomph.generic.codegen.Equations` but with some extra functions for equations defined on interfaces.    
    """
    
    #: If set to a particular :py:class:`~pyoomph.generic.codegen.Equations` class, pyoomph will check whether we have indeed these equations in the bulk
    required_parent_type:type[Equations] | None = None 
    #: If set to a particular :py:class:`~pyoomph.generic.codegen.Equations` class, pyoomph will check whether we have indeed these equations at the opposite bulk side of this interface
    required_opposite_parent_type:type[Equations] | None=None

    def get_mesh(self)->"InterfaceMesh":
        from ..meshes.mesh import InterfaceMesh
        mesh=super().get_mesh()
        assert isinstance(mesh,InterfaceMesh)
        return mesh

    def get_parent_domain(self)->FiniteElementCodeGenerator:
        res=super().get_parent_domain()
        if res is None:
            raise self.add_exception_info(RuntimeError("You apparently used InterfaceEquations in the bulk"))
        assert res is not None
        return res

    def sanity_check(self):
        super(InterfaceEquations, self).sanity_check()
        if self.get_parent_domain() is None:
            raise RuntimeError("Cannot use InterfaceEquations in the bulk")
        if self.required_parent_type is not None:
            pt=self.get_parent_domain().get_equations().get_equation_of_type(self.required_parent_type)
            if pt is None or (isinstance(pt,list) and len(pt)==0):
                raise RuntimeError(
                    "Interface equation " + self.__class__.__name__ + " need to be attached on a domain having the bulk equations " + self.required_parent_type.__name__)
        if self.required_opposite_parent_type is not None:
            pt=self.get_opposite_parent_domain(raise_error_if_none=False)
            if pt is not None:
                pt=pt.get_equations().get_equation_of_type(self.required_opposite_parent_type)
            if pt is None or (isinstance(pt,list) and len(pt)==0):
                raise RuntimeError(
                    "Interface equation " + self.__class__.__name__ + " need to be attached on a domain with an opposite side having the bulk equations " + self.required_opposite_parent_type.__name__)

    def get_parent_equations(self, of_type:type[Equations] | None=None):
        """
        Returns the :py:class:`Equations` in the parent bulk domain of a given type. 
        When setting the attribute :py:attr:`~pyoomph.generic.codegen.InterfaceEquations.required_parent_type`, ``of_type`` can be omitted to get the expected parent equations.
        
        This method is useful to e.g. get the mass density from a Navier-Stokes equation in the bulk domain, e.g. for mass transfer processes at the interface.

        Args:
            of_type: The type of the equations to be returned. If not provided, the :py:attr:`~pyoomph.generic.codegen.InterfaceEquations.required_parent_type` has to be set.
        """
        if of_type is None:
            if self.required_parent_type is None:
                raise RuntimeError("Need to set required_parent_type to used get_parent_equations without argument")
            of_type = self.required_parent_type
        return self.get_parent_domain().get_equations().get_equation_of_type(of_type)

    def get_opposite_parent_equations(self, of_type:type["Equations"] | None=None):
        """
        Returns the :py:class:`Equations` in the parent bulk domain at the opposite side of this interface. 
        When setting the attribute :py:attr:`~pyoomph.generic.codegen.InterfaceEquations.required_opposite_parent_type`, ``of_type`` can be omitted to get the expected parent equations.
        
        This method is useful to e.g. get the mass density from a Navier-Stokes equation in the opposite bulk domain, e.g. for mass transfer processes at the interface.

        Args:
            of_type: The type of the equations to be returned. If not provided, the :py:attr:`~pyoomph.generic.codegen.InterfaceEquations.required_opposite_parent_type` has to be set.
        """
        if of_type is None:
            if self.required_opposite_parent_type is None:
                raise RuntimeError("Need to set required_opposite_parent_type to used get_parent_equations without argument")
            of_type = self.required_opposite_parent_type
        return self.get_opposite_parent_domain().get_equations().get_equation_of_type(of_type)

    def pin_redundant_lagrange_multipliers(self,mesh:"InterfaceMesh",lagr:str,depvars:str | list[str] | tuple[str, ...],opposite_interface:str | list[str] | tuple[str, ...]=[]):
        """
        Allows to pin redundant (overconstraining) Lagrange multipliers. A field of Lagrange multipliers usually enforces some constraint depending on ``depvars`` (and poentially degrees at the ``opposite_interface``).
        If all these degrres are pinned, the Lagrange multiplier ``lagr`` is pinned and set to zero as well. 

        Args:
            mesh: The current mesh must be passed
            lagr: Name of the Lagrange multiplier field to be automatically pinned if necessary
            depvars: Single or multiple variables that occur in the constraint.
            opposite_interface: Optional dependencies on the opposite side of the interface.
        """
        if not isinstance(depvars, (list, tuple)):
            depvars=[depvars]
        if opposite_interface is None:
            opposite_interface=[]
        if not isinstance(opposite_interface, (list, tuple)):
            opposite_interface=[opposite_interface]

        if isinstance(lagr, (list, tuple)):
            for l in lagr:
                self.pin_redundant_lagrange_multipliers(mesh,l,depvars,opposite_interface=opposite_interface)
            return
        else:
            pmesh=mesh._eqtree._parent
            assert pmesh is not None
            bulkmesh:"AnySpatialMesh" = assert_spatial_mesh(pmesh._mesh)
            #print(mesh,bulkmesh)
            interfid = bulkmesh.has_interface_dof_id(lagr)
            dg_space:str | None=None
            if interfid < 0:
                assert mesh._eqtree._codegen is not None
                dg_space=mesh._eqtree._codegen.get_space_of_field(lagr)
                if dg_space=="":
                    raise RuntimeError(f"Something strange here. We have the bulk mesh '{bulkmesh.get_name()}' and it does not have the interface id '{lagr}'") 
                elif dg_space not in {"D2TB","D2","D1"}:
                    raise RuntimeError(f"Something strange here. We have the bulk mesh '{bulkmesh.get_name()}' the Lagrange multiplier field '{lagr}' is defined on unsupported space {dg_space}") 

## NEW PART with DG fields
        if True:
            def expand_depvars(depvars:str | list[str] | tuple[str, ...],msh:"AnySpatialMesh | None"):
                depvars=[depvars] if isinstance(depvars,str) else depvars
                depvars_expanded=[]
                if len(depvars)==0:
                    return depvars_expanded
                assert msh is not None
                cgen=msh._codegen
                assert cgen is not None 
                ccode=cgen.get_code()
                ndim = cgen.get_nodal_dimension()
                for dv in depvars:
                    if dv in ccode.get_nodal_field_indices().keys():
                        depvars_expanded.append(dv)
                    elif dv == "mesh":                    
                        for direct in range(ndim):
                            depvars_expanded.append("mesh_"+(["x","y","z"][direct]))
                    elif dv == "mesh_x" or dv=="mesh_y" or dv=="mesh_z":
                        depvars_expanded.append(dv)
                    else:
                        current:"AnySpatialMesh | None" = msh
                        while current is not None:
                            assert current._codegen is not None
                            ceqs=cast(Equations,current._codegen.get_equations()) 
                            if not isinstance(ceqs,Equations):
                                #print(f"Something strange here. We have the mesh  and it does not have the expected equations."+str(ceqs)+" Looking for "+str(dv))
                                if isinstance(current,InterfaceMesh):                                    
                                    current = current._parent 
                                else:
                                    current = None
                                continue
                            assert isinstance(ceqs,Equations)
                            if dv in ceqs._vectorfields.keys():
                                vcomps = ceqs._vectorfields[dv]
                                for vc in vcomps:
                                    depvars_expanded.append(vc)
                                break
                            else:
                                if isinstance(current,InterfaceMesh):                                    
                                    current = current._parent 
                                else:
                                    current = None
                return depvars_expanded
            
            depvars_expanded=expand_depvars(depvars,mesh)
            depvars_opp=expand_depvars(opposite_interface,mesh._opposite_interface_mesh)
            for e in mesh.elements():
                lagr_data=e.get_field_data_list(lagr,True)


                #print(lagr,lagr_data)
                checkdata=[e.get_field_data_list(cd,True) for cd in depvars_expanded ]
#                print("CHECKING LAGR",lagr,interfid,e,lagr_data);
#               print("WTITH",depvars_expanded)
#                print("WHICH ARE",checkdata)

                opp_e=e.get_opposite_interface_element()
                if opp_e:
                    checkopp=[opp_e.get_field_data_list(cd,True) for cd in depvars_opp ]
                    opp_node_index_map={opp_e.node_pt(ni):ni for ni in range(opp_e.nnode())}
                else:
                    checkopp=[]
                    opp_node_index_map={}
                for nodeind,(l_pt,ni) in enumerate(lagr_data):
#                    print("LOOP",nodeind,l_pt,ni)
                    if ni>=0:
#                        print("AT",e.node_pt(nodeind).x(0),e.node_pt(nodeind).x(1))

                        all_pinned=True
                        for cd in checkdata:
                            if cd[nodeind][1]>=0:
                                if not cd[nodeind][0].is_pinned(cd[nodeind][1]):
                                    all_pinned=False
                                    break
                                #else:
                                #    print("PINNED",cd[nodeind][0],"AT INDEX",cd[nodeind][1],e.node_pt(nodeind).is_pinned(cd[nodeind][1]))
                        #if all_pinned:
                        #    print("ALL PINNED",e.get_Eulerian_midpoint())
                        #    if e.get_Eulerian_midpoint()[0]<0.264:
                        #        exit()
                        if all_pinned and len(checkopp)>0:
                            oppnode=e.opposite_node_pt(nodeind)
                            if oppnode is not None:
                                oppi=opp_node_index_map.get(oppnode,-1)
                                if oppi>=0:
                                    for cd in checkopp:
                                        if cd[oppi][1]>=0:
                                            if not cd[oppi][0].is_pinned(cd[oppi][1]):
                                                all_pinned=False
                                                break

                        if all_pinned:
                            #print("PINNING LAGR ",lagr,l_pt," at index ",ni,"since all of ",depvars_opp,"are pinned",checkopp,opp_e,opposite_interface )
                            l_pt.pin(ni)
                            l_pt.set_value(ni,0)

            return
## END OF NEW PART                
        else:
### OLD PART,  #########################
            def get_inds_and_pos(msh:"AnySpatialMesh",dvs:Sequence[str]):
                cgen=msh._codegen 
                assert cgen is not None 
                ccode=cgen.get_code()
                nodalfields = ccode.get_nodal_field_indices()
                inds_to_check:list[int] = []
                pos_to_check:list[int] = []
                for dv in dvs:
                    if dv in nodalfields.keys():
                        inds_to_check.append(nodalfields[dv])
                    else:  # Check vector
                        if dv == "mesh":
                            ndim = cgen.get_nodal_dimension()
                            for direct in range(ndim):
                                pos_to_check.append(direct)
                        elif dv =="mesh_x":
                            pos_to_check.append(0)
                        elif dv =="mesh_y":
                            pos_to_check.append(1)
                        elif dv =="mesh_z":
                            pos_to_check.append(2)
                        else:
                            current:AnySpatialMesh | None = msh
                            while True:
                                assert current._codegen is not None
                                ceqs=cast(Equations,current._codegen.get_equations()) 
                                assert isinstance(ceqs,Equations)
                                if dv in ceqs._vectorfields.keys():
                                    vcomps = ceqs._vectorfields[dv]
                                    for vc in vcomps:
                                        inds_to_check.append(nodalfields[vc])
                                    break
                                else:
                                    if isinstance(current,InterfaceMesh):                                    
                                        current = current._parent 
                                    else:
                                        current = None
                                    if current is None:
                                        raise RuntimeError("Cannot find the variable " + str(
                                            dv) + " to pin redundant Lagrange multipliers for " + lagr)
                return  inds_to_check,pos_to_check


            inds_to_check,pos_to_check=get_inds_and_pos(mesh,depvars)

            if len(opposite_interface)==0:
                if interfid>=0:
                    for n in mesh.nodes():
                        if all(n.is_pinned(i) for i in inds_to_check) and all(n.position_is_pinned(i) for i in pos_to_check):
                            lind = n.additional_value_index(interfid)
                            n.pin(lind)
                            n.set_value(lind, 0)
                else:
                    for e in mesh.elements():
                        dg_data=e.get_field_data_list(k,False)
                        l_data=e.get_field_data_list(lagr,False)
                        for dg,l in zip(dg_data,l_data):
                            if dg[0].is_pinned(dg[1]):
                                l[0].pin(l[1])
                                l[0].set_value(l[1],0)

            else:
                assert mesh._opposite_interface_mesh is not None
                opp_inds_to_check, opp_pos_to_check = get_inds_and_pos(mesh._opposite_interface_mesh, opposite_interface) 
                for ni,no in mesh.nodes_on_both_sides():
                    if no:
                        opp_pinned=all(no.is_pinned(i) for i in opp_inds_to_check) and all(no.position_is_pinned(i) for i in opp_pos_to_check)
                    else:
                        opp_pinned=True
                    if opp_pinned and all(ni.is_pinned(i) for i in inds_to_check) and all(ni.position_is_pinned(i) for i in pos_to_check):
                        lind = ni.additional_value_index(interfid)
                        ni.pin(lind)
                        ni.set_value(lind, 0)
### END OF OLD PART,  #########################


class SpatialErrorEstimator(Equations):
    def __init__(self, estimator_expression:'ExpressionOrNum | list["ExpressionOrNum"]'):
        super(SpatialErrorEstimator, self).__init__()
        self.estimator_expression = estimator_expression

    def define_error_estimators(self):
        if isinstance(self.estimator_expression, list):
            for e in self.estimator_expression:
                if not isinstance(e,_pyoomph.Expression):
                    e=_pyoomph.Expression(e)
                self.add_spatial_error_estimator(e)
        else:
            e=self.estimator_expression
            if not isinstance(e,_pyoomph.Expression):
                    e=_pyoomph.Expression(e)
            self.add_spatial_error_estimator(e)


class GlobalLagrangeMultiplier(ODEEquations):
    """
    This class allows to define a global Lagrange multiplier that are used to enforce global constraints. It is just a normal :py:class:`~pyoomph.generic.codegen.ODEEquations` but with some additional features, i.e. it can be e.g. deactivated on transient solves.
    """
    def __init__(self,*args:str,only_for_stationary_solve:bool=False,set_zero_on_normal_mode_eigensolve:bool=True,**kwargs:"ExpressionOrNum"):
        super(GlobalLagrangeMultiplier, self).__init__()
        self._entries:dict[str,ExpressionOrNum]=OrderedDict({})
        if "set_zero_on_angular_eigensolve" in kwargs.keys():
            raise RuntimeError("set_zero_on_angular_eigensolve is not supported anymore. Please use set_zero_on_normal_mode_eigensolve instead")
        self.only_for_stationary_solve=only_for_stationary_solve
        self.set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve
        for a in args:
            self._entries[a]=0
        for a,v in kwargs.copy().items():
            self._entries[a]=v

    def define_fields(self):
        super().define_fields()
        for k in self._entries.keys():            
            self.define_ode_variable(k)

    def define_residuals(self):   
        super().define_residuals()     
        for k,v in self._entries.items():
            #print(v,k)
            self.add_weak(v,testfunction(k))
            if self.only_for_stationary_solve:
                self.set_Dirichlet_condition(k,0)
        #exit()

    def after_compilation(self,codegen:"FiniteElementCodeGenerator"):
        super(GlobalLagrangeMultiplier, self).after_compilation(codegen)
        assert codegen._mesh is not None 
        if self.only_for_stationary_solve:
            for k, _ in self._entries.items():
                # Do not activate by default to allow for initial conditions                
                codegen._mesh._set_dirichlet_active(k,False)  

    def _before_stationary_or_transient_solve(self, eqtree:"EquationTree", stationary:bool)->bool:
        must_reapply=False
        if self.set_zero_on_normal_mode_eigensolve:
            pr=self.get_mesh().get_problem()
            from ..generic.bifurcation_tools import _NormalModeBifurcationTrackerBase
            if pr.get_bifurcation_tracking_mode() == "azimuthal" or (pr.get_custom_assembler() is not None and isinstance(pr.get_custom_assembler(),_NormalModeBifurcationTrackerBase)):             
                #if self.get_mesh().get_problem()._azimuthal_mode_param_m.value!=0:
                return False  # Don't do anything in this case. It would mess up everything!
        mesh=eqtree._mesh
        assert mesh is not None
        for k in self._entries.keys():
            if self.only_for_stationary_solve:
                if mesh._get_dirichlet_active(k) == stationary: 
                    mesh._set_dirichlet_active(k, not stationary)
                    must_reapply = True
            else:
                if mesh._get_dirichlet_active(k)==True: 
                    mesh._set_dirichlet_active(k,False) 
                    must_reapply=True
        return must_reapply

    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver", angular_mode:int | None,normal_k:float | None)->set[str | int]:
        if (not self.set_zero_on_normal_mode_eigensolve) or (angular_mode is None and normal_k is None):
            return set()
        elif angular_mode is not None:
            angular_mode=int(angular_mode)
            fullpath = eqtree.get_full_path().lstrip("/")
            if angular_mode == 0:
                return set()
            elif angular_mode == 1 or angular_mode == -1:
                for_my_m = self._entries.keys()
            else:
                for_my_m = self._entries.keys()
            lst=[fullpath + "/" + k for k in for_my_m]
            res:set[str | int] = set(lst)
            return res
        elif normal_k is not None:
            if normal_k == 0:
                return set()
            else:
                fullpath = eqtree.get_full_path().lstrip("/")
                lst=[fullpath + "/" + k for k in self._entries.keys()]
                res:set[str | int] = set(lst)
                return res
        # angular_mode is None and normal_k is None is already handled above; this is unreachable
        # but kept so the function has an explicit return on every static code path.
        return set()

    def get_information_string(self) -> str:
        return ", ".join([str(n) + " with contrib. " + str(v) for n, v in self._entries.items()])



class ScalarField(Equations):
    """
    Introduces a scalar field with the given name and the given space. Residuals can be either added in the constructor or by combining with :py:class:`~pyoomph.generic.codegen.WeakContribution`.
    
    Args:
        name: Name of the scalar field
        space: Space of the scalar field
        scale: Optional scaling of the field (default is 1)
        testscale: Optional scaling of the test function (default is 1)
        residual: Optional residual to be added. Formulate it in terms of the scalar field and the test function.
    """
    def __init__(self,name:str,space:"FiniteElementSpaceEnum",scale:"ExpressionOrNum | None"=None,testscale:"ExpressionOrNum | None"=None,residual:"ExpressionOrNum | None"=None):
        super(ScalarField, self).__init__()
        self.name=name
        self.space:"FiniteElementSpaceEnum"=space
        self.scale=scale
        self.testscale=testscale
        self.residual=residual

    def define_fields(self):
        self.define_scalar_field(self.name,self.space,scale=self.scale,testscale=self.testscale)

    def define_residuals(self):
        if self.residual is not None:
            self.add_residual(self.residual)

class VectorField(Equations):
    """
    Introduces a vector field with the given name and the given space. Residuals can be either added in the constructor or by combining with :py:class:`~pyoomph.generic.codegen.WeakContribution`.
    
    Args:
        name: Name of the scalar field
        space: Space of the scalar field
        scale: Optional scaling of the field (default is 1)
        testscale: Optional scaling of the test function (default is 1)
        residual: Optional residual to be added. Formulate it in terms of the scalar field and the test function.
        dim: Vector dimension. If not set, it will be taken by the dimension of the mesh coordinates, i.e. the nodal dimension
    """
    def __init__(self,name:str,space:"FiniteElementSpaceEnum",scale:"ExpressionOrNum | None"=None,testscale:"ExpressionOrNum | None"=None,residual:"ExpressionOrNum | None"=None,dim:int | None=None):
        super(VectorField, self).__init__()
        self.name=name
        self.space:"FiniteElementSpaceEnum"=space
        self.scale=scale
        self.testscale=testscale
        self.residual=residual
        self.dim=dim

    def define_fields(self):
        self.define_vector_field(self.name,self.space,scale=self.scale,testscale=self.testscale,dim=self.dim)

    def define_residuals(self):
        if self.residual is not None:
            self.add_residual(self.residual)


class WeakContribution(BaseEquations):
    """
    A class to add an arbitrary weak contribution to the equations. This is useful to add additional terms to the equations that are not covered by the standard weak formulation. Essentially, it just adds ``weak(a,b)`` to the residuals.

    Args:
        a: The lhs of the :py:func:`~pyoomph.expressions.generic.weak` contribution.
        b: The rhs (usually a :py:func:`~pyoomph.expressions.generic.testfunction`) of the :py:func:`~pyoomph.expressions.generic.weak` contribution.
        dimensional_dx: If set to ``True``, the weak contribution is treated as a dimensional contribution, i.e. spatial integration dx will carry dimension.
        lagrangian: If set to ``True``, the weak contribution is integrated in the Lagrangian frame of reference.
        coordinate_system: The coordinate system in which the weak contribution is defined. If not set, the coordinate system of the equations or the problem is used.
        destination: The residual destination of the weak contribution. Can be used to define multiple residuals.
    """
    def __init__(self,a:"ExpressionOrNum | str",b:"Expression | str",dimensional_dx:bool=False,lagrangian:bool=False,coordinate_system:BaseCoordinateSystem | None=None,destination:str | None=None):
        super(WeakContribution, self).__init__()
        self.dimensional_dx=dimensional_dx
        self.coordinate_system=coordinate_system
        self.lagrangian=lagrangian
        self.destination=destination
        if isinstance(b,str):
            self.b=testfunction(b)
        else:
            self.b=b
        if isinstance(a,str):
            self.a=var(a)
        else:
            self.a=a

    def define_residuals(self):
        self.add_residual(weak(self.a,self.b,dimensional_dx=self.dimensional_dx,lagrangian=self.lagrangian,coordinate_system=self.coordinate_system),destination=self.destination)


class ResidualContribution(BaseEquations):
    """
    A class to add an arbitrary residual contribution to the equations. This is useful to add additional terms to the equations that are not covered by the standard weak formulation. Essentially, it just adds ``r`` to the residuals.

    Args:
        r: The residual to add (can be e.g. a :py:func:`~pyoomph.expressions.generic.weak` contribution).        
        destination: The residual destination of the weak contribution. Can be used to define multiple residuals.
    """
    def __init__(self,r:"ExpressionOrNum | str",destination:str | None=None):
        super(ResidualContribution, self).__init__()        
        self.destination=destination
        self.r=r

    def define_residuals(self):
        self.add_residual(self.r,destination=self.destination)

class ForceZeroOnEigenSolve(BaseEquations):
    def __init__(self,default:Iterable[str],*,for_nonzero_angular:Iterable[str] | None=None):
        super(ForceZeroOnEigenSolve, self).__init__()
        self.default=default
        self.for_nonzero_angular=for_nonzero_angular

    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree:EquationTree,eigensolver:"GenericEigenSolver", angular_mode:int | None,normal_k:float | None)->set[str | int]:
        if angular_mode is not None:
            if normal_k is not None:
                raise RuntimeError("Cannot set both angular_mode and normal_k")
            angular_mode=int(angular_mode)
            if angular_mode==0:
                topin:set[str]=set(self.default)
            else:
                assert self.for_nonzero_angular is not None
                topin=set(self.for_nonzero_angular)
        elif normal_k is not None:
            if normal_k==0.0:
                topin=set(self.default)
            else:
                assert self.for_nonzero_angular is not None
                topin=set(self.for_nonzero_angular)
        else:
            topin=set(self.default)


        fullpath=eqtree.get_full_path().lstrip("/")
        res:set[str | int]=set([ fullpath+"/"+k for k in topin])
        return res



class ConstrainFieldsToC1Space(Equations):
    """Constrains a higher order field to the first order C1 space. Useful in combination with either the where parameter or (un)constrain a field e.g. a boundary

    Args:
        *args: The names of the fields to constrain to C1 space.
        unconstrain_instead: If set to True, the specified fields will be unconstrained from C1 space instead of being constrained. Default is False.
        where: An optional function to specify where the constraints should be applied. Nondimensional nodal positions are passed to this function, and it should return True for nodes where the constraint should be applied. If None, the constraint is applied to all nodes.
    """
    def __init__(self, *args:str,unconstrain_instead:bool=False,where:Callable[[list[float]], bool] | None=None):
        super().__init__()
        self._constrained_fields = []
        self._unconstrain_instead = unconstrain_instead
        self._where = where
        for field in args:
            self._constrained_fields.append(field)
            
    def before_assigning_equations_preorder(self, mesh):
        BULKFIELD_CONSTRAIN_TO_C1 = 0
        INTERFACE_CONSTRAIN_TO_C1 = 1
        POSITION_CONSTRAIN_TO_C1 = 2
        coordmap={"mesh_x":0,"coordinate:x":0,"mesh_y":1,"coordinate_y":1,"mesh_z":2,"coordinate_z":2}
        modes:dict[str, tuple[int, int]] = {}
        for field in self._constrained_fields:
            if field in coordmap.keys():
                modes[field] = (POSITION_CONSTRAIN_TO_C1, coordmap[field])
                continue
            # Continuous bulk field
            contifi=mesh.get_code_gen().get_code().get_nodal_field_index(field)
            if contifi>=0:
                modes[field] = (BULKFIELD_CONSTRAIN_TO_C1, contifi)
            else:
                # additional interface field?
                contifi=mesh.has_interface_dof_id(field)
                if contifi>=0:
                    modes[field] = (INTERFACE_CONSTRAIN_TO_C1, contifi)
                else:
                    raise RuntimeError(f"Field {field} is not a bulk or interface field, cannot constrain to C1 space")
                
        for e in mesh.elements():
            for nindex in e.non_vertex_node_indices():
                n = e.node_pt(nindex)
                if self._where is not None:
                    x=[n.x(i) for i in range(n.ndim())]
                    if not self._where(x):
                        continue
                for field,(mode,index) in modes.items():  
                    if self._unconstrain_instead:
                        n.remove_additional_dof_constraint(index, mode)
                    else:                  
                        n.set_additional_dof_constraint(index, mode)
                    
        return super().before_assigning_equations_preorder(mesh)


class UnconstrainFieldsFromC1Space(ConstrainFieldsToC1Space):
    """Unconstrains a higher order field from the first order C1 space. Useful in combination with either the where parameter or (un)constrain a field e.g. a boundary

    Args:
        *args: The names of the fields to constrain to C1 space.        
        where: An optional function to specify where the constraints should be applied. Nondimensional nodal positions are passed to this function, and it should return True for nodes where the constraint should be applied. If None, the constraint is applied to all nodes.
    """
    def __init__(self, *args:str,where:Callable[[list[float]], bool] | None=None):
        super().__init__(*args,unconstrain_instead=True,where=where)