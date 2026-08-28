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
 
import contextlib
import fnmatch
import weakref
from .._deprecation import deprecated_kwargs as _deprecated_kwargs
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
    from ..equations.additional import EquationCompilationFlags


def _check_for_valid_var_name(name:str,for_domain:bool):
    typ="domain" if for_domain else "variable" 
    if name=="":
        raise ValueError("Empty "+typ+" name")    
    elif not name.isidentifier():
        raise ValueError(typ+" names may not contain anything else than [A-Z], [a-z], _ and [0-9] (not beginning with a number). Happened at the name: '"+str(name)+"'")
    elif for_domain and name.find("__")>0:
        if not name.startswith("_meshwide_"):
            raise ValueError("Domain names may not have double underscores __, except at the beginning. Happened at the name: '"+str(name)+"'")


# Wildcards in an @-restriction, e.g. DirichletBC(u=0)@"*" or @"wall*". A component containing any of
# these is a glob, everything else stays a literal name checked by _check_for_valid_var_name. Kept
# separate from that function on purpose: it also validates *variable* names, where a glob must never
# become legal.
_DOMAIN_PATTERN_CHARS = "*?["


def _is_domain_name_pattern(name:str)->bool:
    """Whether this component of an @-restriction path is a glob to be expanded against the geometry
    rather than a literal domain/boundary name. Globs never cross a '/': the restriction string is
    split on '/' first, so a pattern always matches exactly one path component."""
    return any(c in name for c in _DOMAIN_PATTERN_CHARS)


def _check_domain_name_pattern(name:str):
    # A pattern is expanded much later (at Problem.initialise), so a typo that can never match would
    # otherwise only surface there. Reject the cases that are provably dead right where they are written.
    if name.startswith("!"):
        raise ValueError("Exclusion patterns such as '"+str(name)+"' are not supported. Use a positive pattern, or list the domain names explicitly.")
    allowed=set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_*?[]!-")
    bad=sorted(set(name)-allowed)
    if bad:
        raise ValueError("Domain name patterns may only contain [A-Z], [a-z], _, [0-9] and the wildcard characters * ? [ ] ! -. Happened at the pattern: '"+str(name)+"' (offending characters: "+"".join(bad)+")")
    # fnmatch treats an unbalanced '[' as a literal, which can then never match an identifier - i.e. a
    # silently dead pattern. Nested '[' is equally meaningless.
    in_bracket=False
    for c in name:
        if c=="[":
            if in_bracket:
                raise ValueError("Nested '[' in the domain name pattern: '"+str(name)+"'")
            in_bracket=True
        elif c=="]":
            in_bracket=False
    if in_bracket:
        raise ValueError("Unbalanced '[' in the domain name pattern: '"+str(name)+"'")
    if name.find("__")>0 and not name.startswith("_meshwide_"):
        raise ValueError("Domain names may not have double underscores __, except at the beginning, so the pattern '"+str(name)+"' can never match anything.")


def _check_domain_name_or_pattern(name:str)->bool:
    """Validates one component of an @-restriction path. Returns whether it is a glob pattern."""
    if name=="":
        raise ValueError("Empty domain name")
    if _is_domain_name_pattern(name):
        _check_domain_name_pattern(name)
        return True
    if name.startswith("!"):
        raise ValueError("Exclusion patterns such as '"+str(name)+"' are not supported. Use a positive pattern, or list the domain names explicitly.")
    _check_for_valid_var_name(name,True)
    return False


_KwargValue=TypeVar("_KwargValue")

def sorted_field_kwargs(kwargs:dict[str,_KwargValue])->dict[str,_KwargValue]:
    """Return keyword arguments naming fields in a fixed (alphabetical) order.

    Equations that take their fields as ``**kwargs`` get whatever order the caller's dict happens to
    have. That is the literal source order when the call is written out by hand, but a *random* one,
    differing from process to process, as soon as the caller builds the arguments from a set::

        DirichletBC(**{"massfrac_"+c:True for c in mixture.required_adv_diff_fields})

    Python randomizes the iteration order of a set of strings per process (PYTHONHASHSEED), so
    without this the conditions would be stated - and the generated code written - in a different
    order on every run. Normalizing here fixes that for every caller at once.

    **Only for equations whose kwargs order is not otherwise observable.** Classes that DEFINE
    fields from their kwargs (:py:class:`~pyoomph.equations.ode.ODEEquations`,
    :py:class:`~pyoomph.equations.generic.ProjectExpression`, the Lagrange multiplier constraints)
    must NOT use this: there the order sets the dof numbering, so sorting would renumber the dofs of
    every existing script and invalidate the state files those scripts have already written. The
    same holds for the observable classes, whose kwargs order is the column order of their output
    files. Callers of those must sort their own arguments if they build them from a set.
    """
    return {k:kwargs[k] for k in sorted(kwargs.keys())}


class FiniteElementCodeGenerator(_pyoomph.FiniteElementCode):
    def __init__(self):
        super(FiniteElementCodeGenerator, self).__init__()
        self._code:_pyoomph.DynamicJITCode | None=None
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
        return self.get_equations()._get_default_timestepping_scheme(dt_order,cg=self)

    def get_code(self)->_pyoomph.DynamicJITCode:
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


    def get_element_dimension(self) -> int:
        return self.dimension

    def calculate_error_overrides(self):
        eqs = self.get_equations()
        oldcg = eqs._get_current_codegen()
        eqs._set_current_codegen(self)
        #        print("EQS",eqs)
        eqs.calculate_error_overrides()
        eqs._set_current_codegen(oldcg)

    # The C++ base has get_scaling(name, testscale:bool); "from_parent" (and the None it may then
    # return) is an addition of this class, which no override can be compatible with.
    @overload # type: ignore[override]
    def get_scaling(self, n:str,testscale:Literal[False]=...)->"Expression": ...

    @overload
    def get_scaling(self, n:str,testscale:Literal[True])->"Expression": ...

    @overload
    def get_scaling(self, n:str,testscale:Literal["from_parent"])->"Expression | None": ...

    def get_scaling(self, n:str,testscale:bool | Literal["from_parent"]=False)->"Expression | None": # type: ignore[override]

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
        res=eqs._expand_additional_field( name, dimensional, expression,in_domain,no_jacobian,no_hessian,where)
        eqs._set_current_codegen(oldcg)
        return res

    def _add_named_numerical_factor(self,**kwargs:"ExpressionOrNum"):
        for k,v in kwargs.items():
            if isinstance(v,_pyoomph.Expression):
                v=self.expand_placeholders(v,True)            
            self._named_numerical_factors[k]=v

    def expand_additional_testfunction(self, name:str, expression:"Expression",in_domain:_pyoomph.FiniteElementCode)->"Expression":
        eqs=self.get_equations()
        oldcg = eqs._get_current_codegen()
        eqs._set_current_codegen(self)                        
        res= eqs._expand_additional_testfunction(name,expression,in_domain)
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
        # An EquationTree node overrides __iter__ to iterate over its own equations instead.
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

    def _change_output_directory(self,newdir:str,eqtree:"EquationTree"):
        pass
    
    @_deprecated_kwargs(coordinate_system="coordsys")
    def add_weak(self,a:"ExpressionOrNum",b:"str | ExpressionOrNum",*,dimensional_dx:bool=False,lagrangian:bool=False,coordsys:"OptionalCoordinateSystem"=None,destination:str | None=None):
        """
        Adds the weak contribution ``(a, b)`` (i.e. the integral of ``a`` times the test function ``b``) to the residuals.

        Args:
            a: Expression to be tested.
            b: Test function, either passed directly or as the name of the field to test.
            dimensional_dx: Whether to use the dimensional (as opposed to the nondimensional) integration measure.
            lagrangian: Whether to integrate over the Lagrangian (undeformed) instead of the Eulerian domain.
            coordsys: Optional coordinate system override for the integration. The former name ``coordinate_system`` is deprecated, but still accepted.
            destination: Optional residual destination for multiple residuals. Defaults to ``None``.

        Returns:
            self, to allow chaining further ``add_weak``/``add_residual`` calls.
        """
        if isinstance(b,str):
            b=testfunction(b)
        self.add_residual(weak(a,b,dimensional_dx=dimensional_dx,coordsys=coordsys,lagrangian=lagrangian),destination=destination)
        return self
    
    @_deprecated_kwargs(coordinate_system="coordsys")
    def add_dweak_dt(self,a:"ExpressionOrNum",b:"str | ExpressionOrNum",*,dimensional_dx:bool=False,lagrangian:bool=False,coordsys:"OptionalCoordinateSystem"=None,destination:str | None=None,scheme:"TimeSteppingScheme"="BDF1",apply_on_others:bool=True):
        """
        Adds d/dt of the weak contribution ``(a, b)``, i.e. the time derivative of the whole integral, so
        that the change of the integration domain of a moving mesh is taken into account as well.

        Args:
            apply_on_others: Whether the history terms also take the normal, the Eulerian element sizes and the Eulerian spatial derivatives in grad/div from the mesh of the corresponding history step. Defaults to True, since each history term belongs to the configuration the element had then. Has no effect unless the mesh moves.
        """
        if isinstance(b,str):
            b=testfunction(b)
        self.add_residual(time_derivative_of_integral(weak(a,b,dimensional_dx=dimensional_dx,coordsys=coordsys,lagrangian=lagrangian),scheme=scheme,apply_on_others=apply_on_others),destination=destination)
        return self
    
    @_deprecated_kwargs(coordinate_system="coordsys")
    def add_functional_minimization(self,F:"ExpressionOrNum",with_respect_to:Expression | list[Expression] | None=None,*,dimensional_dx:bool=False,dimensional_testfunctions:bool=True,lagrangian:bool=False,coordsys:"OptionalCoordinateSystem"=None,destination:str | None=None):
        """Adds the weak form of the functional minimization of W=integral(F dOmega) to the equations.

        Args:
            F (ExpressionOrNum): Integrand of the functional.
            with_respect_to (Optional[Union[Expression,List[Expression]]], optional): Optionally only derive with respect to all shape functions appearing in the listed expressions. Defaults to None, meaning all shape functions in F.
            dimensional_dx (bool, optional): Consider spatial scaling in the weak form integral. Defaults to False.
            dimensional_testfunctions (bool, optional): Expand by dimensional testfunctions. Defaults to True.
            lagrangian (bool, optional): Weak formulation is integrated over the Lagrangian domain. Defaults to False.
            coordsys (OptionalCoordinateSystem, optional): Optional coordinate system. Defaults to the equations' coordinate system, then parent equations and eventually the problem coordinate system. Defaults to None. The former name ``coordinate_system`` is deprecated, but still accepted.
            destination (Optional[str], optional): Residual destination identifier. Defaults to None.

        Returns:
            BaseEquations: Returns self for chaining.
        """
        dF=minimize_functional_derivative(F, only_with_respect_to=with_respect_to, dimensional_testfunctions=dimensional_testfunctions,coordsys=coordsys,lagrangian=lagrangian,dimensional_dx=dimensional_dx)
        self.add_residual(dF,destination=destination)
        return self

    def get_dx(self, use_scaling:bool=True, lagrangian:bool=False,coordsys:_pyoomph.CustomCoordinateSystem | None=None)->"Expression":
        master = self._master()  # TODO This does not allow for dx on individual coordinate systems
        if coordsys is None:
            coordsys = master.get_coordinate_system()
        assert isinstance(coordsys,BaseCoordinateSystem)
        return coordsys.integral_dx(self.get_nodal_dimension(), self.get_element_dimension(), use_scaling,master.get_scaling("spatial"), lagrangian)

    def _after_fill_dummy_equations(self,problem:"Problem",eqtree:"EquationTree",pathname:str,elem_dim:int | None=None):
        pass        

    def get_parent_domain(self)->FiniteElementCodeGenerator | None:
        """
        If this domain is a subdomain of another domain, i.e. a boundary, this function returns the parent domain. Otherwise, it returns None.
        """
        master = self._master()
        cg=master._assert_codegen()
        res=cg._get_parent_domain()
        if res is None:
            return res
        assert isinstance(res,FiniteElementCodeGenerator)
        return res

    def _get_list_of_vector_fields(self,codegen:"FiniteElementCodeGenerator")->list[dict[str,list[str]]]:
        return []

    def _get_list_of_tensor_fields(self,codegen:"FiniteElementCodeGenerator")->list[dict[str,list[list[str]]]]:
        return []

    def _expand_vectorial_entries(self,entries:dict[str,"ExpressionOrNum"],what:str)->dict[str,"ExpressionOrNum"]:
        """Split vector-valued entries into one entry per component of the corresponding vector field.

        ``DirichletBC(velocity=vector(0,1))`` and ``InitialCondition(velocity=vector(...))`` both name
        a FIELD in their keyword arguments, but both act on the scalar fields the code generator knows
        about -- ``velocity_x``, ``velocity_y`` -- rather than on the vector ``velocity`` that
        :py:meth:`define_vector_field` composed out of them. This maps the one onto the other.

        The mapping is POSITIONAL, and it has to be: component *i* of the value goes to component *i*
        of the field, which is exactly the correspondence ``var("velocity")`` is itself built with
        (``define_vector_field`` substitutes it by ``vector(*components)`` in that same order). So it
        holds in every coordinate system -- an axisymmetric ``velocity`` takes its radial and axial
        components in the order the coordinate system declared them, not in x/y order -- without this
        code having to know anything about coordinate systems.

        ``vector()`` always pads to :py:func:`GiNaC_vector_dim` components, so a 2d field is routinely
        handed a third one. Padding is accepted, a non-zero component the field has no slot for is not:
        that is a real mistake (a 3-component condition on a 2-component field) and it would otherwise
        be dropped in silence.

        The value does not have to be an explicit ``vector(...)``. ``var("lagrangian")`` and friends are
        deferred field symbols that only become a matrix once the code generator resolves them, so an
        entry naming a vector field whose value is not a matrix yet is resolved here before it is split
        -- which is what makes ``DirichletBC(mesh=var("lagrangian"))`` work.

        Call it once the equations are attached to a domain (from ``define_residuals`` onwards): a
        condition stated on an interface finds its vector fields on the PARENT domain, and nothing can
        be resolved, before that.

        Args:
            entries: field name -> value, as the equation received them.
            what: what the entries are, for the error messages ("Dirichlet condition", ...).

        Returns:
            The same mapping with every vectorial entry replaced by its per-component entries.
        """
        def _is_vectorial(v:"ExpressionOrNum")->bool:
            return isinstance(v,_pyoomph.Expression) and _pyoomph.GiNaC_is_a_matrix(v)

        cg0=self.get_current_code_generator()

        # Walk the code generators rather than asking the combined element for its list: when this
        # condition is the ONLY equation on its boundary there is nothing to combine it with, so the
        # "combined element" is this very object -- a BaseEquations, which has no _vectorfields and
        # would report an empty list. The domains themselves always know.
        known:dict[str,list[str]]={}
        # Typed as the C++ base, which is what _get_parent_domain hands back; only get_equations and
        # the parent link are used here, and both live there.
        cg:"_pyoomph.FiniteElementCode | None"=cg0
        while cg is not None:
            vfs=getattr(cg.get_equations(),"_vectorfields",None)
            if vfs:
                for name,comps in vfs.items():
                    known.setdefault(name,comps) # the most local definition wins over the parent domains'
            cg=cg._get_parent_domain()
        # The position fields are vectors in exactly the same sense -- var("mesh") resolves to a vector
        # just as var("velocity") does -- but the element gets them from the C++ side instead of from
        # define_vector_field, so they appear in no _vectorfields dict and would be rejected below.
        # Added last, so a user-defined vector field of the same name keeps precedence, and only where
        # there are nodes at all (an ODE domain has no position fields).
        nodal_dim=cg0.get_nodal_dimension()
        if nodal_dim>0:
            for posname in ("mesh","coordinate","lagrangian"):
                known.setdefault(posname,[posname+"_"+c for c in ("x","y","z")[:nodal_dim]])

        if not any(_is_vectorial(v) or (n in known) for n,v in entries.items()):
            return entries

        res:dict[str,"ExpressionOrNum"]={}
        for name,val in entries.items():
            comps=known.get(name)
            if not _is_vectorial(val):
                if comps is None or not isinstance(val,_pyoomph.Expression):
                    # A number is never a vector, and a name we know nothing about is left to the code
                    # generator, which reports it along with the fields it does have.
                    res[name]=val
                    continue
                # Resolve with the very call the code generator itself makes on this expression a few
                # steps later (expand_initial_or_Dirichlet), so the two cannot disagree about what
                # var("lagrangian") means here. raise_error=False: an expression that stays unresolved
                # is passed on untouched and fails where it would have failed before this existed.
                try:
                    resolved=cg0.expand_placeholders(val,False)
                except Exception:
                    res[name]=val
                    continue
                if not _is_vectorial(resolved):
                    res[name]=val
                    continue
                val=resolved
            assert isinstance(val,_pyoomph.Expression)
            if comps is None:
                raise RuntimeError("Got a vectorial "+what+" for '"+name+"', but '"+name+
                                   "' is not a vector field on domain "+str(cg0.get_full_name())+
                                   ". Vector fields here: "+(", ".join(sorted(known.keys())) if known else "(none)")+
                                   ". For a scalar field, pass a scalar; for a single component, name it explicitly (e.g. '"+name+"_x').")
            ncomp=val.nops()
            if ncomp<len(comps):
                raise RuntimeError("The vectorial "+what+" for '"+name+"' has "+str(ncomp)+
                                   " component(s), but the field has "+str(len(comps))+" ("+", ".join(comps)+").")
            for i,c in enumerate(comps):
                res[c]=val[i]
            for i in range(len(comps),ncomp):
                if not val[i].is_zero():
                    raise RuntimeError("The vectorial "+what+" for '"+name+"' has a non-zero component "+
                                       str(i)+" ("+str(val[i])+"), but the field only has "+str(len(comps))+
                                       " component(s) here ("+", ".join(comps)+").")
        return res

    def get_problem(self)->"Problem":
        mst=self._master()
        # The code generator's pointer is C++-side and is re-stamped whenever the problem is
        # rebuilt, so it wins over the stored one. The stored one is only there for the window
        # before any code generator exists (the dummy-equation and interface-connection passes).
        cg=mst._get_current_codegen()
        if isinstance(cg,FiniteElementCodeGenerator):
            p=cg.get_problem()
            if p is not None:
                return p
        if mst._problem is not None:
            return mst._problem
        return mst._assert_codegen().get_problem()

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

    def _get_creation_info(self)->str | None:
        return self._created_at #type:ignore

    def _add_exception_info(self,exception:Exception)->Exception:
        if self.with_exception_info:            
            import sys
            errmsg = '\nRaised from ' + str(self.__class__.__name__) + ' object instantiated at: "' + str(self._get_creation_info()) + '"'
            raise type(exception)(str(exception) + ' %s' % errmsg).with_traceback(sys.exc_info()[2])
        else:
            raise exception


    def get_azimuthal_r0_info(self):
        """Returns a dict [0,1,2]-> Set[str] with the names of the fields that are pinned at r=0 for azimuthal symmetry.
        Entry 0 contains the names of the fields that are pinned to zero at r=0 for normal (axisymmetric solves). This pinning is strongly enforced.
        Entry 1 contains the names of the fields that are pinned at r=0 for azimuthal eigensolves with m=1. This pinning is implemented by modifying the eigenproblem matrices.
        Entry 2 contains the names of the fields that are pinned at r=0 for azimuthal eigensolves with m>=2. This pinning is implemented by modifying the eigenproblem matrices.        
        """
        master=self._master()
        return master._azimuthal_r0_info

    def _before_precice_initialise(self,eqtree:"EquationTree"):
        pass

    def _before_precice_solve(self,eqtree:"EquationTree",precice_dt:float):
        pass

    def _after_precice_solve(self,eqtree:"EquationTree",precice_dt:float):
        pass

    def __init__(self):
        super().__init__()
        self._created_at:str | None
        #: Every domain this instance was added to, weakly. One instance may sit in several,
        #: which is why it records all of them and resolves the active one per call rather than
        #: storing "the" owner. Weak, so that dropping a domain does not keep it alive.
        self._owner_trees:"list[weakref.ref[EquationTree]]" = []
        self._coordinate_system:"BaseCoordinateSystem | None" = None
        self._additional_fields:dict[str,ExpressionOrNum] = {}
        self._additional_fields_also_on_interface:dict[str,ExpressionOrNum] = {}
        self._additional_testfuncs:dict[str,ExpressionOrNum] = {}
        self._additional_testfuncs_also_on_interface:dict[str,ExpressionOrNum] = {}
        self._initial_conditions:dict[str,dict[str,tuple[ExpressionOrNum,str,"BaseEquations"]]] = {}
        self._Dirichlet_conditions:dict[str,tuple[ExpressionOrNum,"BaseEquations"]] = {}
        self._code:_pyoomph.DynamicJITCode | None = None
        self._scaling:dict[str,"ExpressionOrNum | str"] = {}
        self._test_scaling:dict[str,"ExpressionOrNum | str"]={}
        self._scales_to_check_for_fields:set[str] = set()
        self._test_scales_to_check_for_fields:set[str] = set()
        #self._external_data_links:Dict[str,Tuple["ODEEquations",str]] = {}
        self._dimension = None
        self.default_timestepping_scheme:Literal["BDF2", "BDF1", "Newmark2"] | None = None
        self._problem=None
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
        
    

    def _interior_facet_terms_required(self):
        return self.requires_interior_facet_terms

    def get_combined_equations(self) -> "BaseEquations":
        return self._master()

    def calculate_error_overrides(self):
        pass

    def _before_stationary_or_transient_solve(self, eqtree:"EquationTree", stationary:bool)->bool:
        return False # Return whether the equations have to be renumbered

    def _before_eigen_solve(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver",angular_m:int | None=None,normal_k:float | None=None)->bool:
        return False # Return whether the equations have to be renumbered

    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree:"EquationTree",eigensolver:"GenericEigenSolver", angular_mode:int | float | None,normal_k:float | None)->set[str | int]:
        return set()

    def _init_output(self,eqtree:"EquationTree",continue_info:dict[str, Any] | None,rank:int):
        pass

    def _do_output(self, eqtree:"EquationTree", step:int,stage:str,only_every_step:bool=False):
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

    def after_transient_solve(self):
        """Called once per **accepted** timestep, at the end of :py:meth:`~pyoomph.generic.problem.Problem.solve`
        with a time step.

        Unlike :py:meth:`after_newton_solve`, this does not fire for stationary solves, for
        arclength continuation steps, for the Newton solves a temporal-adaptivity retry discards,
        or for the intermediate solves of spatial adaptation. Anything that must advance exactly
        once per step in time - tracer particle advection, say - belongs here.
        """
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

    def _before_mesh_to_mesh_interpolation(self,eqtree:"EquationTree",interpolator:"BaseMeshToMeshInterpolator"):
        pass

    def after_mapping_on_macro_elements(self):
        pass


    def before_finalization(self,codegen:"FiniteElementCodeGenerator"):
        pass

    def _before_compilation(self,codegen:"FiniteElementCodeGenerator"):
        pass

    def after_compilation(self,codegen:"FiniteElementCodeGenerator"):
        pass

    def _register_refinement_directives(self,codegen:"FiniteElementCodeGenerator"):
        """Declare persistent refinement criteria (see :py:meth:`pyoomph.meshes.mesh.BaseMesh._add_refinement_directive_to_level`)
        on ``codegen._mesh``.

        Separate from :py:meth:`after_compilation` because a directive lives on the mesh *object*,
        not in the compiled code: remeshing (and loading a state file written with a different mesh
        template) replaces every mesh but reuses the code, so ``after_compilation`` is not called
        again and the directives of the replaced mesh would simply be gone. Everything stating a
        directive therefore does so here and lets ``after_compilation`` call it, so the mesh
        replacement can call it once more on the new mesh.

        Must not cache the mesh on ``self``: one equation instance can be attached to several
        domains, and is then called once per mesh with a different codegen each time."""
        pass


    def setup_remeshing_size(self,remesher:"RemesherBase",preorder:bool):
        pass

    def get_space_of_field(self,name:str) -> str:
        cg=self._assert_codegen()
        return cg.get_space_of_field(name)

    def _add_named_numerical_factor(self,**kwargs:"ExpressionOrNum"):
        mst=self._master()
        cg=mst._assert_codegen()
        cg._add_named_numerical_factor(**kwargs)

    def sanity_check(self):
        pass

    def define_scaling(self):
        pass

    def on_apply_boundary_conditions(self,mesh:"AnyMesh"):
        pass
    
    
    def _fill_interinter_connections(self,eqtree:"EquationTree",interinter:set[str]):
        pass
    
    def _before_fill_dummy_equations(self,problem:"Problem",eqtree:"EquationTree",pathname:str):
        pass

    def _assert_codegen(self)->FiniteElementCodeGenerator:
        cg=self._get_current_codegen()
        if cg is None:
            raise RuntimeError("Cannot do this operation outside the scope of a code generator. Occurend in Equations: "+str(self)+" : "+ self.get_information_string())
        assert isinstance(cg,FiniteElementCodeGenerator)
        return cg

    def define_field_by_substitution(self, fieldname:str, expr:"ExpressionOrNum", also_on_interface:bool=False):
        master = self._master()
        master._additional_fields[fieldname] = expr
        if also_on_interface:
            master._additional_fields_also_on_interface[fieldname] = expr

    def define_testfunction_by_substitution(self, fieldname:str, expr:"ExpressionOrNum", also_on_interface:bool=False):
        master = self._master()
        master._additional_testfuncs[fieldname] = expr
        if also_on_interface:
            master._additional_testfuncs_also_on_interface[fieldname] = expr

    def set_scaling(self,_field_scalings:"dict[str,ExpressionOrNum | str] | None"=None,*,allow_scales_with_fields:bool=False, **args:"ExpressionOrNum | str"):
        mst = self._master()
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
        mst = self._master()
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
        master=self._master()
        cg=master._assert_codegen()
        return cg.dimension

    def get_nodal_dimension(self):
        """
        Returns the nodal (Eulerian) dimension of the domain where the equations are defined.
        """
        master = self._master()
        cg = master._assert_codegen()
        return cg.get_nodal_dimension()

    def get_normal(self):
        """
        Returns the normal of this domain. This is only possible if the domain is either a boundary or a bulk domain with co-dimension 1.
        
        Note that ``var("normal")`` is essentially the same.
        """
        master = self._master()
        cg=master._assert_codegen()
        ndim = cg.get_nodal_dimension()
        if ndim == 0:
            raise RuntimeError(
                "Normal cannot be used here... Element has no nodal dimension or is not initialised yet for normals")
        #return vector([cg._get_normal_component(i) for i in range(ndim)])
        return var("normal",domain=cg)

    def _master(self)->"BaseEquations":
        """The domain this instance is currently acting on, which owns the state its equations
        share: scalings, fields defined by substitution, initial and Dirichlet conditions.

        Resolved per call, never stored. One Equations instance may be added to several domains -
        eqs @ ["left","right"], a glob expansion, a reused sub-equation - and each of them must
        see its own state, so the answer is "whichever domain is being compiled or traversed
        right now", not "the domain this was last put into"."""
        cg=self._get_current_codegen()
        if cg is not None:
            # Typed C++-side as _pyoomph.Equations; every domain actually carries an EquationTree.
            eqs=cast("BaseEquations | None",cg.get_equations())
            if eqs is not None:
                return eqs
        # Not bound: something in another domain reached in, e.g. through
        # InterfaceEquations.get_parent_equations(). Its domain is bound even though this
        # instance is not, so look for it among the domains holding this instance.
        active=[]
        for ref in self._owner_trees:
            node=ref()
            if node is not None and node._get_current_codegen() is not None:
                active.append(node)
        if len(active)==1:
            return active[0]
        elif len(active)>1:
            # Several at once happens while an interface is compiled, which binds its bulk, the
            # opposite interface and the opposite bulk too. Innermost wins.
            for node in reversed(_domains_being_traversed):
                if any(node is a for a in active):
                    return node
            return active[-1]
        return self

    def _get_combined_element(self)->"BaseEquations":
        # Former name of _master(), kept because it leaked into user subclasses.
        return self._master()

    def get_current_code_generator(self) -> FiniteElementCodeGenerator:
        mst=self._master()
        assert mst is not None
        return mst._assert_codegen()

    def get_mesh(self)->"AnyMesh":
        mesh=self.get_current_code_generator()._mesh
        assert mesh is not None
        return mesh

    def _perform_define_fields(self):
        master = self._master()
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
        master = self._master()
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
        master = self._master()
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
        master = self._master()
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
                    expr[2]._add_exception_info(e)

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
                    expr_comb[1]._add_exception_info(e)


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
        
        master = self._master()
        if not self.requires_interior_facet_terms or not master._interior_facet_terms_required():
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
        master = self._master()
        if not isinstance(expr, _pyoomph.Expression):
            expr = _pyoomph.Expression(expr)
        dn = destination if destination is not None else ""
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
                self._add_exception_info(e)
            cg._activate_residual("")
        cg._activate_residual("")
        return self

    def _define_fields(self):
        self._master()._perform_define_fields()


    def _check_scalings(self):
        # Check all scalings and test scalings. sorted, so that a problem with more than one bad
        # scaling reports the same one on every MPI rank rather than whichever the hash seed put
        # first - the ranks then abort with different messages about the same problem.
        for n in sorted(self._scales_to_check_for_fields):
            scal=self.get_scaling(n)
            if scal is not None:
                scal_expa=self.expand_expression_for_debugging(scal)
                try:
                    assert_dimensional_value(scal_expa)
                except Exception as e:
                    cg=self._assert_codegen()
                    raise RuntimeError("The scale for '"+str(n)+"' on domain '"+self._assert_codegen().get_full_name()+"' is not a simple dimensional number, but:\n    "+str(scal)+"\n   expands to: "+str(scal_expa)+"\n.")
        for n in sorted(self._test_scales_to_check_for_fields):
            scal=self.get_scaling(n,testscale=True)
            if scal is not None:
                scal_expa=self.expand_expression_for_debugging(scal)
                try:
                    assert_dimensional_value(scal_expa)
                except Exception as e:
                    cg=self._assert_codegen()
                    raise RuntimeError("The test function scale for '"+str(n)+"' on domain '"+cg.get_full_name()+"' is not a simple dimensional number, but:\n    "+str(scal)+"\n   it expands to: "+str(scal_expa)+"\n.") 

    def _define_element(self):
        

        master = self._master()
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
        master = self._master()
        if master._coordinate_system is not None:
            return master._coordinate_system
        elif (pdom:=master.get_current_code_generator().get_parent_domain()):
            return pdom.get_coordinate_system()
        else:
            assert master._problem is not None
            return master._problem.get_coordinate_system()

    def _expand_additional_field(self, name:str, dimensional:bool, expression:_pyoomph.Expression,in_domain:_pyoomph.FiniteElementCode,no_jacobian:bool,no_hessian:bool,where:str)->"Expression":        
        #msh=self.get_mesh()
        #if msh is not None:
        #    msh=msh._name
        master = self._master()
        try:
            cg:"FiniteElementCodeGenerator" = master._assert_codegen()

        except:
            if master._is_ode(): # ODEs might still be accessible
                tagexpr=expression.op(1)
                print(dir(master))
                print("CODE",master._code)
                print("PROBLEM", master._problem)
                print("TAGS", tagexpr)
                raise RuntimeError("TODO: Expand tags, see what ODE is meant by domain tag and resolve the additional test function. You could also have a typo in the name of "+str(name)+", i.e. that this field does not exist in this ODE")
            else:
                raise RuntimeError("Should not end up here")

        assert isinstance(in_domain,FiniteElementCodeGenerator)
        if _pyoomph.get_verbosity_flag() != 0:
                print("Expanding additional field ", name, dimensional,"self/in_domain",cg.get_full_name(),in_domain.get_full_name())
        
        scale:ExpressionOrNum = self.get_scaling(name) if dimensional else 1

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
                    raw = bulkeq._additional_fields_also_on_interface[name]
                    expr:Expression=evaluate_in_domain(raw if isinstance(raw,Expression) else Expression(raw),self.get_current_code_generator())
                    if only_base_mode:
                        assert isinstance(axibreakcsys,AxisymmetryBreakingCoordinateSystem)
                        expr= axibreakcsys.map_to_zero_epsilon(expr)
                    elif only_perturbation_mode:
                        assert isinstance(axibreakcsys,AxisymmetryBreakingCoordinateSystem)
                        expr= axibreakcsys.map_to_first_order_epsilon(expr,with_epsilon=True)
                    scale = bulk.get_scaling(name) if dimensional else 1
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

    def _expand_additional_testfunction(self, name:str, expression:"Expression",in_domain:_pyoomph.FiniteElementCode)->"Expression":
        master = self._master()
        try:
            cg = master._assert_codegen()
        except:
            if master._is_ode(): # ODEs might still be accessible
                tagexpr=expression.op(1)
                print(dir(master))
                print("CODE",master._code)
                print("PROBLEM", master._problem)
                print("TAGS", tagexpr)
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
            tfres=self._additional_testfuncs[name]
            return evaluate_in_domain(tfres if isinstance(tfres,Expression) else Expression(tfres),cg)
        elif (not isinstance(self, ODEEquations)) and self.get_parent_domain() is not None:
            bulk = self.get_parent_domain()
            while bulk is not None:
               # print("INPUT",expression)
                bulkeq=bulk.get_equations()
                if name in bulkeq._additional_testfuncs_also_on_interface.keys():
                    rawtf=bulkeq._additional_testfuncs_also_on_interface[name]
                    return evaluate_in_domain(rawtf if isinstance(rawtf,Expression) else Expression(rawtf),self.get_current_code_generator())
                bulk=bulk.get_parent_domain()
            raise RuntimeError("Cannot expand the testfunction "+ str(name))
        else:
            raise RuntimeError("Cannot expand the testfunction " + str(name))

    def set_temporal_error_factor(self, name:str, factor:float):
        master = self._master()
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

    def _get_default_timestepping_scheme(self, order:int,cg:FiniteElementCodeGenerator | None=None)->Literal["BDF2","BDF1","Newmark2"]:
        if order == 2:
            return "Newmark2"        

        if self.default_timestepping_scheme is not None:
            return self.default_timestepping_scheme
        master = self._master()
        if master.default_timestepping_scheme is not None:
            return master.default_timestepping_scheme

        # get_problem() rather than self._problem: only the domain is stamped with the problem,
        # so an equation asking on its own behalf used to assert here instead of resolving.
        if isinstance(self, ODEEquations):
            return cast(Literal["BDF2","BDF1","Newmark2"],self.get_problem().get_default_timestepping_scheme(order))
        elif cg is not None:
            pdom=cg.get_parent_domain()
            if pdom is not None:
                return cast(Literal["BDF2","BDF1","Newmark2"],pdom.get_default_timestepping_scheme(order))
            else:
                return cast(Literal["BDF2","BDF1","Newmark2"],self.get_problem().get_default_timestepping_scheme(order))
        elif (pdom:=self.get_parent_domain()) is not None:
            return cast(Literal["BDF2","BDF1","Newmark2"],pdom.get_default_timestepping_scheme(order))
        else:
            return cast(Literal["BDF2","BDF1","Newmark2"],self.get_problem().get_default_timestepping_scheme(order))





    def get_information_string(self)->str:
        return ""

    def _tree_string(self, indent:str) -> str:
        addinfo = self.get_information_string()
        return self.__class__.__name__ + (": " + addinfo if addinfo != "" else "")

    def __matmul__(self, other:str | list[str] | tuple[str, ...] | set[str])->"EquationTree":
        """Restricts these equations to a domain, boundary or interface, e.g. ``PoissonEquation()@"domain"``
        or ``DirichletBC(u=0)@"left"``.

        The name may be a path, ``@"domain/left"`` being the same as ``(eqs@"left")@"domain"``, or a
        list/tuple/set of names, ``@["left","right"]``, which attaches the very same Equations object
        to each of them.

        A path component may also be an fnmatch-style glob - ``@"*"``, ``@"wall*"``, ``@"[lr]*"``,
        ``@"domain/*"`` - which is expanded once during :py:meth:`~pyoomph.generic.problem.Problem.initialise`
        against the names the mesh templates actually provide: bulk domains at the top level, the
        boundaries of a bulk domain below that, and the intersections of those boundaries (contact
        lines) one level deeper. A glob never crosses a ``/`` and a glob matching nothing is an error.
        A leading ``!`` is reserved and rejected. Note that on a multi-domain mesh the interface
        *between* two bulk domains is a genuine boundary of both, so ``@"*"`` includes it.

        Globs are only interpreted here; path lookups such as
        :py:meth:`~pyoomph.generic.problem.Problem.get_mesh` stay literal.
        """
        if isinstance(other, (list,tuple,set,)):
            res=EquationTree(None, None)
            for d in other:
                # The same instance goes to every domain named here; each of them resolves it
                # independently, so no wrapping is needed to keep them apart.
                if d is None or d==".":
                    res+=self
                else:
                    res+=self@d
            return res
            #return sum([self @ d for d in other], EquationTree(None, None))
        if isinstance(other, str): #type:ignore
            splt = other.split("/")
            splt = [x for x in splt if x]  # Remove empties
            if len(splt) == 0:
                raise ValueError("Please restrict equations with a non-empty domain name")
            root = EquationTree(None, parent=None)
            mynode = EquationTree(self, parent=root)
            _check_domain_name_or_pattern(splt[-1])
            root._children[splt[-1]] = mynode
            res = root
            for k in reversed(splt[:-1]):
                res = res @ k
            return res
        else:
            raise ValueError(
                "Please combine equation with a string (name of the domain) to restrict the equations to a domain")

    def get_my_domain(self)->FiniteElementCodeGenerator:
        master = self._master()
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


    def __radd__(self, other:"Literal[0]")->"EquationTree | BaseEquations":
        # So that sum() over equations works; only EquationTree had this before.
        if other==0:
            return self
        raise RuntimeError("Cannot add "+str(other)+" and "+str(self))

    def __add__(self, other:"Literal[0] | BaseEquations | EquationTree")->"EquationTree | BaseEquations":
        """Adding equations yields an unrestricted domain holding both.

        There is no separate class for a combination: a domain is a list of equations, and one
        that has not been restricted to a name yet is simply a node without a parent, which
        @ "name" then places exactly as it places a single equation."""
        if other==0:
            return self
        # EquationTree is a BaseEquations too, so it has to be tested first - otherwise
        # "eqs + tree" would put a whole tree into a domain's equation list.
        if isinstance(other, EquationTree):
            return EquationTree(self, None) + other
        if isinstance(other, BaseEquations):
            return EquationTree([self, other], None)
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
        
        
        master = self._master()
        cg = master._assert_codegen()
        if not (isinstance(expr,int) or isinstance(expr,float) or isinstance(expr,_pyoomph.Expression)) and  callable(expr):
            expr=expr()
        if isinstance(expr,(int,float)):
            expr=_pyoomph.Expression(expr)
        elif isinstance(expr,_pyoomph.GiNaC_GlobalParam):
            expr=0+expr # a parameter is not an Expression yet; wrapped it stays live, i.e. the output follows it
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

    #: Spaces a field on the interior-facet skeleton may live on (cf. the check in
    #: _internal_define_scalar_field, which is the one that fires when the declaration is replayed).
    _internal_facet_field_spaces={"D0","DL","D1","D1TB","D2","D2TB"}

    def _defer_to_internal_facets(self,kind:str,space:"FiniteElementSpaceEnum | None",args:tuple[Any,...],kwargs:dict[str,Any]):
        """Record a declaration made with at_internal_facets=True on the skeleton node.

        define_fields() runs far too late to create the '_internal_facets_' domain - the node has to
        exist by _fill_dummy_equations, i.e. before any code generator or InterfaceMesh is built - so
        this only ever fills a skeleton that is already there. requires_interior_facet_terms in
        __init__ is what puts it there, and a class formulating a facet weak form needs that anyway.
        """
        master=self._master()
        cg=master._assert_codegen()
        me=self.__class__.__name__
        if isinstance(master,EquationTree) and master._codegen is not None and master._codegen is not cg:
            # A dummy pass: _create_dummy_domains_for_DG builds throwaway generators for the two sides
            # of a facet and runs this domain's define_fields() on them a second time, so that the
            # skeleton's residuals can resolve the opposite-side fields. The real pass has already
            # recorded everything, and recording again would double every entry.
            return
        if cg.get_domain_name()=="_internal_facets_":
            raise self._add_exception_info(RuntimeError("at_internal_facets=True was used in "+me+", which is already attached to the interior-facet skeleton '_internal_facets_'. Just drop the argument."))
        if space is not None and space not in self._internal_facet_field_spaces:
            raise self._add_exception_info(NotImplementedError("Continuous fields on the interior-facet skeleton '_internal_facets_' are not supported (tried to define '"+str(args[0])+"' on space "+str(space)+" with at_internal_facets=True in "+me+"). Use a discontinuous facet space instead, i.e. "+", ".join(sorted(self._internal_facet_field_spaces))+"."))
        facets=master._children.get("_internal_facets_") if isinstance(master,EquationTree) else None
        if facets is None:
            raise self._add_exception_info(RuntimeError("at_internal_facets=True was used in "+me+" on domain '"+cg.get_full_name()+"', but this domain has no interior-facet skeleton to put the field on. Set self.requires_interior_facet_terms=True in the __init__ of "+me+" (which add_interior_facet_residual requires anyway), or add the subdomain by hand with eqs+=Equations()@'_internal_facets_'."))
        facets._pending_internal_facet_defs.append((kind,args,kwargs))

    def set_facet_recovery(self,fieldname:str,expr:"ExpressionOrNum",at_internal_facets:bool=False):
        """
        Defines how a discontinuous (``DL`` or ``D0``) field of this domain is filled on elements that
        are created by a spatial adaptation or a remeshing and therefore receive no values from the old
        mesh.

        This is meant for fields on the interior-facet skeleton (``"_internal_facets_"``): when the bulk
        is refined, facets appear *inside* the former elements, and no amount of interpolation from the
        old skeleton can produce a value there - the old skeleton has nothing at that position. For an
        HDG-style trace the right answer comes from the bulk instead, e.g.::

            class MyFacetEqs(Equations):
                def define_fields(self):
                    self.define_scalar_field("lam","DL")
                def define_residuals(self):
                    lam,lamtest=var_and_test("lam")
                    self.add_residual(weak(lam-avg(var("u")),lamtest))
                    self.set_facet_recovery("lam",avg(var("u")))

        Without it, such facets keep the zero they were allocated with (and a one-time warning is
        printed); they are corrected by the next solve, which is why zero remains the default.

        The expression is evaluated on the new element, i.e. on the CURRENT state only, and least-squares
        fitted onto the field's own space (for ``D0``: averaged). The result is written to every stored
        time level, so that a facet which just came into existence does not produce a spurious time
        derivative. Elements that DO get values from the old mesh are untouched by it, and so are fields
        without a recovery expression - if any field of the domain lacks one, the element still counts as
        unrestored and shows up in ``mesh.get_discontinuous_unrestored_elements()``.

        Args:
            fieldname: name of the ``DL``/``D0`` field to recover.
            expr: expression evaluated on the new element, e.g. ``avg(var("u"))``.
            at_internal_facets: register this on the ``"_internal_facets_"`` child of this domain
                instead of on this domain itself, so that a bulk equation class can declare the
                recovery of a skeleton field it declared with ``at_internal_facets=True``. Returns
                ``None`` in that case, since the local function is only created on replay.
        """
        if at_internal_facets:
            self._defer_to_internal_facets("recovery",None,(fieldname,expr),{})
            return None
        return self.add_local_function("__facet_recovery_"+fieldname,expr)

    def add_integral_function(self, name:str, expr:"ExpressionOrNum"):
        master = self._master()
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
        master = self._master()
        cg = master._assert_codegen()
        cg._register_dependent_integral_function(name, func)  


    def expand_expression_for_debugging(self,expr:"ExpressionOrNum",raise_error:bool=True,collect_units:bool=True,unit_error:bool=True,with_mode_expansion:bool=True) -> Expression:
        master = self._master()
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

    def _get_global_dof_storage_name(self,pathname:str | None=None):
        if pathname is None:
            pathname=self.get_current_code_generator().get_full_name()
        dofstorage="_meshwide__"+pathname.lstrip("/").replace("/","__")
        return dofstorage

    def get_weak_dirichlet_terms_for_DG(self,fieldname:str,value:"ExpressionOrNum")->"ExpressionNumOrNone":
        """
        Returns the weak Dirichlet terms for a discontinuous Galerkin (DG) formulation. When using a :py:class:`~pyoomph.equations.generic.DirichletBC` with ``prefer_weak_for_DG``, this method is called. If it returns not ``None``, the :py:class:`~pyoomph.equations.generic.DirichletBC` is not enforced strongly, but on the basis of the given interface residuals.

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

    def _get_list_of_vector_fields(self,codegen:"FiniteElementCodeGenerator")->list[dict[str,list[str]]]:
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

    def _get_list_of_tensor_fields(self,codegen:"FiniteElementCodeGenerator")->list[dict[str,list[list[str]]]]:
        tensor_fields:list[dict[str,list[list[str]]]]=[]
        current=self
        if hasattr(current, "_tensorfields"):
            tensor_fields.append(current._tensorfields)
        parent = codegen._get_parent_domain()
        while parent is not None:
            assert isinstance(parent,FiniteElementCodeGenerator)
            peqs=parent.get_equations()
            if isinstance(peqs,Equations):
                tensor_fields.append(peqs._tensorfields)
            parent = parent._get_parent_domain()
        return tensor_fields


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
        master = self._master()
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
        opp_inter:"FiniteElementCodeGenerator | None"
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
        master = self._master()  # TODO This does not allow for dx on individual coordinate systems
        cg = master._assert_codegen()
        cg._coordinates_as_dofs = True
        if coordinate_space is not None:
            cg._coordinate_space = coordinate_space
            if coordinate_space not in ["C2TB", "C2", "C1TB", "C1"]:
                raise ValueError("Can only set the coordinate space to either C2TB, C2, C1TB or C1")
        rcomponent="_x"
        zcomponent:str | None="_y"
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
        master = self._master()
        pdom=master.get_parent_domain()
        if pdom is not None:
            # Check a bit what is possible
            if space=="C2TB" or space=="D2TB":
                if pdom._coordinate_space!="C2TB":
                    raise self._add_exception_info(RuntimeError("You tried to define a "+str(space)+" field '"+str(name)+"' at an interface attached to a bulk domain with element space "+str(pdom._coordinate_space)+". This does not work"))
            elif space=="C2" or space=="D2":
                if pdom._coordinate_space not in {"C2TB","C2"}:
                    raise self._add_exception_info(RuntimeError("You tried to define a "+str(space)+" field '"+str(name)+"' at an interface attached to a bulk domain with element space "+str(pdom._coordinate_space)+". This does not work"))
            elif space=="C1TB" or space=="D1TB":
                # A bubble space needs a bubble NODE, and a C2 element does not have one: on the simplex
                # families BulkElementTri2dC2 / BulkElementTetra3dC2 the C1TB row of
                # Nodal_Space_Index_To_Element_Index_Map is literally empty (src/elements_2d.cpp). This
                # check used to let a C2 parent through in 2d, and indexing that empty row SEGFAULTED in
                # interpolate_newly_constructed_additional_dof rather than reporting anything -- see
                # dev_docs/interface_refinement_coupling.md section 14.5. Kept as a backstop: the
                # negotiation in get_interface_field_connection_space now caps the space with
                # largest_facet_space() and should never reach here.
                if pdom._coordinate_space not in {"C1TB","C2TB"}:
                    raise self._add_exception_info(RuntimeError("You tried to define a "+str(space)+" field '"+str(name)+"' at an interface attached to a bulk domain with element space "+str(pdom._coordinate_space)+". This does not work, since elements of "+str(pdom._coordinate_space)+" have no bubble node for "+str(space)+". Use "+("C1" if space=="C1TB" else "D1")+" for the facet field, or raise the bulk domain to C1TB/C2TB with an ElementSpace()."))
                elif pdom._coordinate_space=="C1TB" and pdom.dimension==3:
                    raise self._add_exception_info(RuntimeError("You tried to define a "+str(space)+" field '"+str(name)+"' at an interface attached to 3d bulk domain with element space "+str(pdom._coordinate_space)+". This does not work, since 3d tetrahedral elements of "+str(pdom._coordinate_space)+" do not provide the face bubble node for "+str(space)+" on 2d facets. Consider upgrading the 3d space to C2TB using an ElementSpace('C2TB') for the 3d domain or adjust the facet space to "+("C1" if space=="C1TB" else "D1")+"."))
            
            
        cg = master._assert_codegen()
        if cg.get_domain_name()=="_internal_facets_" and space not in {"D0","DL","D1","D1TB","D2","D2TB"}:
            # A continuous field on the interior-facet skeleton would need one shared dof per facet
            # node, i.e. oomph-lib "additional values" on BoundaryNodes. Interior nodes are plain
            # pyoomph::Node, so InterfaceElementBase::add_interface_dofs null-derefs on them (it used
            # to segfault before this check existed; the C++ backstop there now throws as well).
            raise self._add_exception_info(NotImplementedError("Continuous fields on the interior-facet skeleton '_internal_facets_' are not supported (tried to define '"+str(name)+"' on space "+str(space)+"). Use a discontinuous facet space instead, i.e. D0, DL, D1, D1TB, D2 or D2TB."))

        if _pyoomph.get_verbosity_flag() != 0:
            print("REGISTER", name, self, master, self == master, space)
        master._register_field(name, space)
        self._fields_defined_on_my_domain[name]=space
        master._fields_defined_on_my_domain[name]=space
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


    def define_scalar_field(self, name:str | list[str], space:"FiniteElementSpaceEnum",scale:"ExpressionOrNum | str | None"=None,testscale:"ExpressionOrNum | str | None"=None,discontinuous_refinement_exponent:float | None=None,allow_scales_with_fields:bool=False,at_internal_facets:bool=False):
        """
        Define a scalar field on this domain. Must be called within the specified implementation of the method :py:meth:`~BaseEquations.define_fields`.

        Args:
            name (str): The name of the vector field.
            space (FiniteElementSpaceEnum): The finite element space on which the vector field is defined.
            scale (ExpressionNumOrNone): The scale for the vector field for nondimensionalization. Defaults to None.
            testscale (ExpressionNumOrNone): The scale for the test function of the vector field for nondimensionalization. Defaults to None.
            discontinuous_refinement_exponent (Optional[float]): The exponent for the discontinuous refinement. Defaults to None.
            allow_scales_with_fields (bool): Whether to allow scales/testscales with fields included. Defaults to False.
            at_internal_facets (bool): Define the field on the ``"_internal_facets_"`` skeleton of this domain instead of on this domain itself. The space must then be discontinuous (``D0``, ``DL``, ``D1``, ``D1TB``, ``D2``, ``D2TB``) and the domain must have a skeleton, i.e. ``self.requires_interior_facet_terms`` must be set in ``__init__``. Defaults to False.
        """
        if not isinstance(name, str):
            for n in name:
                self.define_scalar_field(n, space, scale=scale, testscale=testscale,discontinuous_refinement_exponent=discontinuous_refinement_exponent,allow_scales_with_fields=allow_scales_with_fields,at_internal_facets=at_internal_facets)
            return        
        if at_internal_facets:
            self._defer_to_internal_facets("scalar",space,(name,space),dict(scale=scale,testscale=testscale,discontinuous_refinement_exponent=discontinuous_refinement_exponent,allow_scales_with_fields=allow_scales_with_fields))
            return
        
        # BaseCoordinateSystem.define_scalar_field is annotated with scale/testscale:ExpressionOrNum|None,
        # but it just forwards them unchanged to Equations._internal_define_scalar_field, which does accept
        # a str (a named scale reference) as well - so passing a str through here is safe at runtime.
        self.get_coordinate_system().define_scalar_field(name, space, self,scale,testscale,discontinuous_refinement_exponent,allow_scales_with_fields=allow_scales_with_fields) #type:ignore

    def define_vector_field(self, name:str, space:"FiniteElementSpaceEnum", dim:int | None=None,scale:"ExpressionNumOrNone"=None,testscale:"ExpressionNumOrNone"=None,allow_scales_with_fields:bool=False,at_internal_facets:bool=False):
        """
        Define a vector field on this domain. Must be called within the specified implementation of the method :py:meth:`~BaseEquations.define_fields`.

        Args:
            name (str): The name of the vector field.
            space (FiniteElementSpaceEnum): The finite element space on which the vector field is defined.
            dim (Optional[int]): The dimension of the vector field. If not provided, it defaults to the nodal dimension.
            scale (ExpressionNumOrNone): The scale for the vector field for nondimensionalization. Defaults to None.
            testscale (ExpressionNumOrNone): The scale for the test function of the vector field for nondimensionalization. Defaults to None.
            at_internal_facets (bool): Define the field on the ``"_internal_facets_"`` skeleton of this domain instead of on this domain itself. See :py:meth:`define_scalar_field`. Defaults to False.
        """
        if at_internal_facets:
            # Deferred before dim is resolved, so that the default comes from the skeleton (it is the
            # same nodal dimension, but the skeleton is the domain the field belongs to).
            self._defer_to_internal_facets("vector",space,(name,space),dict(dim=dim,scale=scale,testscale=testscale,allow_scales_with_fields=allow_scales_with_fields))
            return
                   
        dim = dim if dim is not None else self.get_nodal_dimension()  # TODO: Here, it should be nodal_dimension!
        v, vtest,comps = self.get_coordinate_system().define_vector_field(name, space, dim, self)
        also_on_interface = space in {"C1","D1","C2","D2","C1TB","D1TB","C2TB","D2TB"}
        mst=self._master()
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
            zcomponent:str | None="_y"
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
                
    

    def define_tensor_field(self, name:str, space:"FiniteElementSpaceEnum", dim:int | None=None,scale:"ExpressionNumOrNone"=None,testscale:"ExpressionNumOrNone"=None, symmetric:bool=False,allow_scales_with_fields:bool=False,at_internal_facets:bool=False):
        if at_internal_facets:
            self._defer_to_internal_facets("tensor",space,(name,space),dict(dim=dim,scale=scale,testscale=testscale,symmetric=symmetric,allow_scales_with_fields=allow_scales_with_fields))
            return
        dim = dim if dim is not None else self.get_nodal_dimension()  # TODO: Here, it should be nodal_dimension!
        t, ttest,comps = self.get_coordinate_system().define_tensor_field(name, space, dim, self, symmetric)
        also_on_interface:bool = space in { "C1","C2","C1TB","C2TB","D2TB","D2","D1","D1TB"}
        mst=self._master()
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

    def add_spatial_error_estimator(self, expr:"Expression",for_base:bool=True,for_eigen:bool=True,group:str="",normalize_relative:float=1.0,weight:float=1.0):
        master = self._master()
        cg=master._assert_codegen()
        if for_base:
            cg._add_Z2_flux(expr,False,group,normalize_relative,weight)
        if for_eigen:
            cg._add_Z2_flux(expr,True,group,normalize_relative,weight)




class DummyEquations(Equations):
    def define_fields(self):
        pass
    def define_residuals(self):
        pass


class _InternalFacetFieldDeclarations(Equations):
    """Replays the at_internal_facets=True declarations recorded on this skeleton node.

    Attached to every '_internal_facets_' node by _fill_dummy_equations and inert when nothing was
    declared. The recorded list is complete by the time this runs, because the declaring domain
    always defines its fields first: a bulk domain in MeshFromTemplateBase.__init__, before the
    interface meshes of its children exist at all, and an interface in _finalise_creation, which
    walks the codim-1 domains before the codim-2 ones (see pyoomph/meshes/mesh.py)."""

    def _recorded(self)->"list[tuple[str,tuple[Any,...],dict[str,Any]]]":
        master=self._master()
        assert isinstance(master,EquationTree), "the replay equation only ever sits on a skeleton domain"
        return master._pending_internal_facet_defs

    def define_fields(self):
        for kind,args,kwargs in self._recorded():
            if kind=="scalar":
                self.define_scalar_field(*args,**kwargs)
            elif kind=="vector":
                self.define_vector_field(*args,**kwargs)
            elif kind=="tensor":
                self.define_tensor_field(*args,**kwargs)

    def define_residuals(self):
        # set_facet_recovery is add_local_function underneath, which needs the fields in place -
        # same reason its docstring example puts it in define_residuals.
        for kind,args,kwargs in self._recorded():
            if kind=="recovery":
                self.set_facet_recovery(*args,**kwargs)


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
            master = self._master()
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

    def _expand_additional_field(self, name: str, dimensional: bool, expression: _pyoomph.Expression,
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
            return super(ODEEquations, self)._expand_additional_field(name, dimensional, expression,
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
            raise self._add_exception_info(RuntimeError("You apparently used InterfaceEquations in the bulk"))
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


        def expand_depvars(depvars:str | list[str] | tuple[str, ...],msh:"AnySpatialMesh | None"):
            depvars=[depvars] if isinstance(depvars,str) else depvars
            depvars_expanded:list[str]=[]
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
                        ceqs=current._codegen.get_equations()
                        # Only a spatial domain registers vector fields. The test used to be
                        # isinstance(ceqs,Equations), which separated spatial from ODE domains only
                        # because ODEEquations derives from BaseEquations and not from Equations -
                        # an EquationTree node is an Equations whatever it holds, so ask directly.
                        if ceqs._is_ode() is True or not hasattr(ceqs,"_vectorfields"):
                            if isinstance(current,InterfaceMesh):
                                current = current._parent
                            else:
                                current = None
                            continue
                        ceqs=cast(Equations,ceqs)
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
            checkdata=[e.get_field_data_list(cd,True) for cd in depvars_expanded ]
            opp_e=e.get_opposite_interface_element()
            if opp_e:
                checkopp=[opp_e.get_field_data_list(cd,True) for cd in depvars_opp ]
                opp_node_index_map={opp_e.node_pt(ni):ni for ni in range(opp_e.nnode())}
            else:
                checkopp=[]
                opp_node_index_map={}
            for nodeind,(l_pt,ni) in enumerate(lagr_data):
                if ni>=0:
                    all_pinned=True
                    for cd in checkdata:
                        if cd[nodeind][1]>=0:
                            if not cd[nodeind][0].is_pinned(cd[nodeind][1]):
                                all_pinned=False
                                break                         
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
                        l_pt.pin(ni)
                        l_pt.set_value(ni,0)



#: Domains currently being compiled or traversed, innermost last. Only consulted to break a tie
#: when one Equations instance sits in several domains that are bound at the same time.
_domains_being_traversed:"list[EquationTree]" = []


def _as_equation_list(eqs:"BaseEquations | list[BaseEquations] | None")->"list[BaseEquations]":
    """Normalise whatever was handed to an EquationTree node into its own list.

    The copy matters: __add__ and _clone_structure pass an existing node's list straight into a
    new node, and an alias would let a later addition (see AxisymmetryBC) mutate the source tree."""
    if eqs is None:
        return []
    elif isinstance(eqs,EquationTree):
        # A node is a BaseEquations itself, so this has to come first - a tree handed in here
        # would otherwise be stored as if it were a single equation.
        raise RuntimeError("Cannot put an EquationTree into the equations of a domain")
    elif isinstance(eqs,BaseEquations):
        eqs=[eqs]
    return list(eqs)


class EquationTree(Equations):
    """One domain of the equation tree, and at the same time the equation object of that domain.

    Being an Equations is what lets a node carry the state its equations share - scalings, fields
    defined by substitution, initial and Dirichlet conditions - and be the single object the code
    generator is given, without a separate combining equation class in between."""

    #: Nodes are created in bulk (one per path component, per clone, per glob match) and
    #: BaseEquations.__new__ walks the interpreter stack for each one, which reads source files.
    with_exception_info:bool=False

    def __init__(self, eqs:"BaseEquations | list[BaseEquations] | None"=None, parent:"EquationTree | None"=None):
        super(EquationTree, self).__init__()
        #: All equations added to this domain. Merging two domains concatenates the lists, which
        #: is what makes a separate combining equation class unnecessary.
        self._equations = eqs
        self._parent = parent
        self._codegen:"FiniteElementCodeGenerator | None"=None
        self._children:dict[str,"EquationTree"] = {}
        self._compilation_flags:"EquationCompilationFlags | None"=None
        #: Declarations made from the parent domain with at_internal_facets=True, replayed by
        #: _InternalFacetFieldDeclarations when this (skeleton) node defines its own fields. Kept
        #: on the node rather than on the declaring equation because one Equations instance may sit
        #: on several domains (eqs @ ["left","right"]) and each skeleton must get exactly one copy.
        self._pending_internal_facet_defs:list[tuple[str,tuple[Any,...],dict[str,Any]]]=[]

    @property
    def _mesh(self)->"AnyMesh | None":
        """The mesh built for this domain, stored once, on the code generator.

        A node used to keep a second copy. The two were written from different places - an
        InterfaceMesh set the node's but not the generator's, and only
        rebuild_global_mesh_from_list() re-stamped the latter - so between a remesh and that
        call the generator pointed at a destroyed mesh. Both warnings about that ordering
        (Problem._reregister_refinement_directives, _register_refinement_directives) refer to it.
        """
        return self._codegen._mesh if self._codegen is not None else None

    @_mesh.setter
    def _mesh(self,mesh:"AnyMesh | None"):
        assert self._codegen is not None, "A domain gets its code generator before its mesh"
        self._codegen._mesh=mesh

    @property
    def _equations(self)->"list[BaseEquations]":
        return self._equation_list

    @_equations.setter
    def _equations(self,eqs:"BaseEquations | list[BaseEquations] | None"):
        # Normalised and copied here rather than at the call sites: __add__ and _clone_structure
        # hand an existing node's list straight over, and an alias would let a later addition
        # mutate the source tree.
        self._equation_list=_as_equation_list(eqs)
        for e in self._equation_list:
            # Building a tree makes plenty of short-lived nodes (every + creates one), so the
            # dead entries they leave behind are dropped here rather than accumulating.
            e._owner_trees=[r for r in e._owner_trees if r() is not None]
            if not any(r() is self for r in e._owner_trees):
                e._owner_trees.append(weakref.ref(self))

    def _absorb_equation_state(self):
        """Take over what the equations set on themselves before any domain existed.

        Scalings set in a constructor land on the equation, but everything reads them off the
        domain. That used to be the same object whenever a domain held exactly one equation."""
        for eq in self._equations:
            self._scaling.update(eq._scaling)
            self._test_scaling.update(eq._test_scaling)
            self._scales_to_check_for_fields|=eq._scales_to_check_for_fields
            self._test_scales_to_check_for_fields|=eq._test_scales_to_check_for_fields
            if self.default_timestepping_scheme is None:
                self.default_timestepping_scheme=eq.default_timestepping_scheme

    @contextlib.contextmanager
    def _on_this_domain(self):
        """Bind this domain's code generator while its equations run a hook.

        Everything an equation resolves through _master() - scalings, additional fields, the code
        generator itself - is relative to the domain it is running on, so the binding has to be
        restored afterwards: an interface hook routinely runs while its bulk is bound as well.
        """
        eqs=self._equations
        # The combining wrapper of a merged domain has to be bound as well, not just the
        # equations themselves: it is what they resolve their master to, and reading anything
        # off it (a scaling, the mesh) goes through its code generator.
        targets=list(eqs)+[self]
        old=[t._get_current_codegen() for t in targets]
        for t in targets:
            t._set_current_codegen(self._codegen)
        _domains_being_traversed.append(self)
        try:
            yield eqs
        finally:
            _domains_being_traversed.pop()
            for t,o in zip(targets,old):
                t._set_current_codegen(o)

    def _dispatch(self,hook:str,*args:Any,needs_mesh:bool=False)->list[Any]:
        """Run `hook` on every equation of this domain and collect the results.

        The list is built eagerly on purpose: several hooks combine the results with `and`/`or`,
        and every equation has to run regardless of what an earlier one returned."""
        if not self._equations or (needs_mesh and self._mesh is None):
            return []
        with self._on_this_domain() as eqs:
            return [getattr(e,hook)(*args) for e in eqs]

    def get_compilation_flags(self,problem:"Problem")->"EquationCompilationFlags":
        """The code generation flags in effect on this domain.

        Any :py:class:`~pyoomph.equations.additional.EquationCompilationFlags` added here wins, all
        settings it leaves at ``None`` are inherited from the parent domain and eventually from
        :py:attr:`~pyoomph.generic.problem.Problem.equation_compilation_flags`.
        """
        if self._compilation_flags is None:
            from ..equations.additional import EquationCompilationFlags
            if self._parent is not None:
                flags=self._parent.get_compilation_flags(problem)
            else:
                flags=problem.equation_compilation_flags
            for eq in self._equations:
                for own in eq.get_equation_of_type(EquationCompilationFlags,always_as_list=True):
                    flags=cast(EquationCompilationFlags,own).with_defaults_from(flags)
            self._compilation_flags=flags
        return self._compilation_flags

    # Wider than Equations.get_mesh, which promises a spatial mesh: a node may be an ODE
    # domain, and then its mesh is an ODEStorageMesh.
    def get_mesh(self)->"AnyMesh": # type: ignore[override]
        assert self._mesh is not None
        return self._mesh

    def __iter__(self)->Iterator["BaseEquations"]:
        return iter(self._equations)

    # ---- what the code generator and the meshes now call straight on the domain --------------
    # Each opens _on_my_current_codegen(), so an equation running one of these resolves its
    # master back to this domain even though it may also belong to others.
    def define_fields(self):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                if _pyoomph.get_verbosity_flag() != 0:
                    print("DEF SUB", eq)
                eq.define_fields()

    def define_scaling(self):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq.define_scaling()

    def define_residuals(self):
        res=None
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                contrib=eq.define_residuals()
                if contrib is not None:
                    res=contrib if res is None else res+contrib
        return res

    def define_error_estimators(self):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq.define_error_estimators()

    def define_additional_functions(self):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq.define_additional_functions()

    def sanity_check(self):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq.sanity_check()

    def calculate_error_overrides(self):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq.calculate_error_overrides()

    def on_apply_boundary_conditions(self,mesh:"AnyMesh"):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq.on_apply_boundary_conditions(mesh)

    def before_finalization(self,codegen:"FiniteElementCodeGenerator"):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq.before_finalization(codegen)

    def _before_compilation(self,codegen:"FiniteElementCodeGenerator"):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq._before_compilation(codegen)

    def after_compilation(self,codegen:"FiniteElementCodeGenerator"):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq.after_compilation(codegen)

    def _register_refinement_directives(self,codegen:"FiniteElementCodeGenerator"):
        with self._on_my_current_codegen() as eqs:
            for eq in eqs:
                eq._register_refinement_directives(codegen)

    def _release_output_files(self)->None:
        for eq in self._equations:
            eq._release_output_files()

    @contextlib.contextmanager
    def _on_my_current_codegen(self):
        """Bind this domain's equations to whichever code generator the node is bound to.

        Not self._codegen: while the dummy domains for interior facets are defined, the node is
        deliberately bound to a dummy generator instead of its own. Restored afterwards, so that
        an instance shared with another domain is not left pointing at this one.
        """
        cg=self._get_current_codegen()
        old=[e._get_current_codegen() for e in self._equations]
        for e in self._equations:
            e._set_current_codegen(cg)
        _domains_being_traversed.append(self)
        try:
            yield self._equations
        finally:
            _domains_being_traversed.pop()
            for e,o in zip(self._equations,old):
                e._set_current_codegen(o)

    @overload
    def get_equation_of_type(self, typ:type["BaseEquations"], *, exact_type:bool=False,always_as_list:Literal[True])->list["BaseEquations"]: ...

    @overload
    def get_equation_of_type(self, typ:type["BaseEquations"], *, exact_type:bool=False,always_as_list:Literal[False]=False)->"BaseEquations | None": ...

    def get_equation_of_type(self, typ:type["BaseEquations"], *, exact_type:bool=False,always_as_list:bool=False)->'list["BaseEquations"] | BaseEquations | None':
        """Search this domain's equations. The node itself is never a candidate."""
        res:list["BaseEquations"]=[]
        for eq in self._equations:
            found=eq.get_equation_of_type(typ,exact_type=exact_type,always_as_list=True)
            res+=found
        if always_as_list:
            return res
        if len(res)==1:
            return res[0]
        return res if res else None

    def _interior_facet_terms_required(self)->bool:
        return any(eq._interior_facet_terms_required() for eq in self._equations)

    def get_weak_dirichlet_terms_for_DG(self,fieldname:str,value:"ExpressionOrNum")->"ExpressionNumOrNone":
        res=None
        for eq in self._equations:
            # Only spatial equations have this hook; ODEEquations and the plain BaseEquations
            # helpers (Scaling, InitialCondition, ...) do not.
            if not isinstance(eq,Equations):
                continue
            contrib=eq.get_weak_dirichlet_terms_for_DG(fieldname,value)
            if contrib is not None:
                res=contrib if res is None else res+contrib
        return res

    def get_coordinate_system(self)->BaseCoordinateSystem:
        if self._is_ode() is True:
            return _ode_coordinate_system
        return super(EquationTree, self).get_coordinate_system()

    def _is_ode(self)->bool | None:
        """Whether this domain is an ODE domain. None while nothing has an opinion."""
        res=None
        for eq in self._equations:
            isode=eq._is_ode()
            if isode is None:
                continue
            if res is None:
                res=isode
            elif res!=isode:
                info=[repr(e)+" is ODE: "+str(e._is_ode()) for e in self._equations]
                raise RuntimeError("Combined Equations and ODEEquations does not work yet:\n"+"\n".join(info))
        return res

    def get_equations(self)->"EquationTree":
        """The equation object of this domain, which is the node itself."""
        assert self._equations
        return self
    
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


    # The tree walkers below (and the ones after them) share their name with the BaseEquations
    # hook they dispatch to, but take one argument less: a node *is* the eqtree its equations are
    # handed. Nothing ever calls them through that base signature, because a node lives in another
    # node's _children, never in its _equations, so _dispatch never reaches one.
    def _change_output_directory(self,newdir:str): # type: ignore[override]
        self._dispatch("_change_output_directory",newdir,self,needs_mesh=True)
        for _,c in self._children.items():
            c._change_output_directory(newdir)

    def _before_assigning_equations(self,dof_selector:"_DofSelector | None"):
        self._dispatch("before_assigning_equations_preorder",self._mesh,needs_mesh=True)

        if dof_selector is not None:
            dof_selector._apply_on_domain(self._mesh)

        for _,c in self._children.items():
            c._before_assigning_equations(dof_selector)
        self._dispatch("before_assigning_equations_postorder",self._mesh,needs_mesh=True)

    def _after_remeshing(self):
        self._dispatch("after_remeshing",self,needs_mesh=True)
        for _,c in self._children.items():
            c._after_remeshing()

    def _register_all_refinement_directives(self):
        """Re-state the refinement criteria of this subtree on the meshes it currently points at.

        Called after a mesh replacement (remeshing, or a state file with a different template) for
        the subtree of each replaced domain only - the new meshes carry no directives yet, whereas
        a domain that was not replaced still has its own and would collect a duplicate per remesh."""
        self._dispatch("_register_refinement_directives",self._codegen,needs_mesh=True)
        for _,c in self._children.items():
            c._register_all_refinement_directives()

    def _before_mesh_to_mesh_interpolation(self,interpolator:"BaseMeshToMeshInterpolator"): # type: ignore[override]
        self._dispatch("_before_mesh_to_mesh_interpolation",self,interpolator,needs_mesh=True)
        for _,c in self._children.items():
            c._before_mesh_to_mesh_interpolation(interpolator)

    def _setup_remeshing_size(self,remesher:"RemesherBase",preorder:bool):
        if preorder:
            for _, c in self._children.items():
                c._setup_remeshing_size(remesher, preorder)
        self._dispatch("setup_remeshing_size",remesher,preorder)
        if not preorder:
            for _,c in self._children.items():
                c._setup_remeshing_size(remesher,preorder)

    def _after_mapping_on_macro_elements(self):
        self._dispatch("after_mapping_on_macro_elements",needs_mesh=True)
        for _,c in self._children.items():
            c._after_mapping_on_macro_elements()

    def _before_newton_solve(self):
        self._dispatch("before_newton_solve")
        for _,c in self._children.items():
            c._before_newton_solve()

    def _after_newton_solve(self):
        self._dispatch("after_newton_solve")
        for _,c in self._children.items():
            c._after_newton_solve()

    def _after_transient_solve(self):
        self._dispatch("after_transient_solve")
        for _,c in self._children.items():
            c._after_transient_solve()

    def _before_newton_convergence_check(self)->bool:
        # all() over an already built list: a rejecting equation must not stop the others running
        res=all(self._dispatch("before_newton_convergence_check",self))
        for _,c in self._children.items():
            res=c._before_newton_convergence_check() and res
        return res

    def _before_precice_initialise(self): # type: ignore[override]
        self._dispatch("_before_precice_initialise",self)
        for _,c in self._children.items():
            c._before_precice_initialise()

    def _before_precice_solve(self,precice_dt:float): # type: ignore[override]
        self._dispatch("_before_precice_solve",self,precice_dt)
        for _,c in self._children.items():
            c._before_precice_solve(precice_dt)

    def _after_precice_solve(self,precice_dt:float): # type: ignore[override]
        self._dispatch("_after_precice_solve",self,precice_dt)
        for _,c in self._children.items():
            c._after_precice_solve(precice_dt)

    # Same one-argument-less situation as above, for the output and eigen hooks.
    def _init_output(self,continue_info:dict[str, Any] | None=None,rank:int=0): # type: ignore[override]
        self._dispatch("_init_output",self,continue_info,rank)
        for _,child in self._children.items():
            child._init_output(continue_info=continue_info,rank=rank)

    def _before_stationary_or_transient_solve(self, stationary:bool)->bool: # type: ignore[override]
        must_reapply=any(r is True for r in self._dispatch("_before_stationary_or_transient_solve",self,stationary))
        for _, child in self._children.items():
            must_reapply=child._before_stationary_or_transient_solve(stationary=stationary) or must_reapply
        return must_reapply

    def _before_eigen_solve(self, eigensolver:"GenericEigenSolver",angular_m:int | None=None,normal_k:float | None=None)->bool: # type: ignore[override]
        must_reapply=any(r is True for r in self._dispatch("_before_eigen_solve",self,eigensolver,angular_m,normal_k))
        for _, child in self._children.items():
            must_reapply = child._before_eigen_solve(eigensolver,angular_m,normal_k) or must_reapply
        return must_reapply

    def _get_forced_zero_dofs_for_eigenproblem(self,eigensolver:"GenericEigenSolver",angular_mode:int | float | None,normal_k:float | None)->set[str | int]: # type: ignore[override]
        res:set[str | int]=set()
        for upd in self._dispatch("_get_forced_zero_dofs_for_eigenproblem",self,eigensolver,angular_mode,normal_k):
            res.update(upd)
        for _, child in self._children.items():
            res.update(child._get_forced_zero_dofs_for_eigenproblem(eigensolver,angular_mode,normal_k))
        return res

    def _do_output(self,step:int,stage:str,only_every_step:bool=False): # type: ignore[override]
        self._dispatch("_do_output",self,step,stage,only_every_step)
        for _,child in self._children.items():
            child._do_output(step,stage,only_every_step)

    def _has_sub_equations_defined(self):
        if self._equations:
            return True
        else:
            for _,v in self._children.items():
                if v._has_sub_equations_defined():
                    return True
        return False

    def _expand_domain_name_patterns(self,candidates_getter:Callable[["EquationTree",int],tuple[set[str],str]],depth:int=0):
        """Replaces every glob child (e.g. "*", "wall*", "[lr]*") by one clone of its subtree per matching
        name, merging into an explicit sibling of the same name if there already is one. Called once from
        Problem._link_geometry_and_equations, after the mesh templates have defined their geometry - which
        is the earliest moment any real domain or boundary name exists - and before anything else looks at
        the tree, so everything downstream only ever sees literal names."""
        # Insertion order of _children mirrors the order the user wrote the restrictions in, and that
        # order decides which of two conditions on the same boundary is applied last and hence wins.
        # It therefore has to be carried through the expansion, both for the patterns themselves and
        # for the merge below.
        order=[k for k in self._children.keys()]
        patterns=[k for k in order if _is_domain_name_pattern(k)]
        if patterns:
            # The candidates come from the mesh templates only, never from self._children. That is what
            # makes a pattern unable to match a name another pattern just produced, so expansion at one
            # node is order-independent and cannot loop.
            candidates,descr=candidates_getter(self,depth)
            for pat in patterns:
                node=self._children.pop(pat)
                matches=sorted(n for n in candidates if fnmatch.fnmatchcase(n,pat))
                if not matches:
                    raise RuntimeError("The domain name pattern '"+pat+"' at '"+self.get_full_path()+"' does not match anything, i.e. '"+self.get_full_path().rstrip("/")+"/"+pat+"' does not exist. "+descr+" are "+str(sorted(candidates)))
                patpos=order.index(pat)
                fresh:list[str]=[]
                for name in matches:
                    # Each match gets its own tree nodes but shares the Equations objects - exactly the
                    # semantics eqs@["a","b"] already has.
                    clone=node._clone_structure()
                    existing=self._children.get(name)
                    if existing is None:
                        merged=clone
                        fresh.append(name)
                    elif patpos<order.index(name):
                        merged=clone._merge_with(existing)
                    else:
                        merged=existing._merge_with(clone)
                    merged._parent=self
                    self._children[name]=merged
                # The pattern gives up its slot to the names it produced; names that were already there
                # keep the slot they had.
                order[patpos:patpos+1]=fresh
            self._children={k:self._children[k] for k in order}
        # After the expansion, so that a clone which itself contains a pattern (@"*/*") is descended into
        for child in list(self._children.values()):
            child._expand_domain_name_patterns(candidates_getter,depth+1)

    def _fill_dummy_equations(self,problem:"Problem",is_bulk_root:bool=True,pathname:str=""):
        # The node is the master of its domain, so it needs the Problem as much as the equations
        # do - get_coordinate_system() and the timestepping defaults read it off the master.
        self._problem=problem
        if len(self._children)>0 and not self._equations:
            if self._has_sub_equations_defined() and not is_bulk_root:
                self._equations=[DummyEquations()]
        if self._equations:
            for eq in self._equations:
                eq._before_fill_dummy_equations(problem,self,pathname)
            if any(eq._interior_facet_terms_required() for eq in self._equations):
                if "_internal_facets_" not in self._children.keys():
                    facets=EquationTree(DummyEquations(),self)
                    facets._problem=problem
                    self._children["_internal_facets_"]=facets
            # Whichever skeleton is there - auto-created above or written out by the user - gets the
            # replay equation, so that define_*_field(...,at_internal_facets=True) in the equations
            # of this domain has somewhere to land. It defines nothing when nothing was declared.
            skeleton=self._children.get("_internal_facets_")
            if skeleton is not None:
                skeleton._equations=skeleton._equations+[_InternalFacetFieldDeclarations()]
        #for dn in list(self._children.keys()):
        for dn,v in self._children.items():
            #v=self._children[dn] # Cannot use .items() here
            #print(dn,v)
            v._fill_dummy_equations(problem,False,pathname=(dn if is_bulk_root else pathname+"/"+dn))
        

    def _fill_interinter_connections(self,iconns:set[str]): # type: ignore[override]
        if self._equations:
            myiconns=set([x for x in iconns if x.startswith(self.get_full_path().lstrip("/"))])
            for eq in self._equations:
                eq._fill_interinter_connections(self,myiconns)
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
        if self._equations:
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
                    source.get_compilation_flags(problem).apply_to_codegen(dummy)
                    return dummy

                assert self._parent is not None
                dummy=generate_dummy_domain(self) # Opposite DG facet
                # The facet dummy mirrors the skeleton's own fields, but its elements are never added
                # to any mesh, so those fields never get equation numbers. Flag it so that reading them
                # through '|-' is rejected at code generation time instead of silently yielding zero.
                dummy._is_internal_facet_opposite_dummy=True
                dummy_p=generate_dummy_domain(self._parent) # Opposite bulk facet

                cg_self._set_opposite_interface(dummy)
                dummy._set_bulk_element(dummy_p)

                cg_self._dummy_codegen_for_internal_facets=dummy
                cg_self._dummy_codegen_for_internal_facets_bulk=dummy_p
                # TODO: This is a bit problematic
                parent=self.get_parent()
                if parent is not None:
                    grandparent=parent.get_parent()
                    if grandparent is not None and grandparent._equations:
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
            for eq in self._equations:
                eq._after_fill_dummy_equations(problem,self,self.get_full_path(),elem_dim=elemdim)
            self.setup_codegen_to_equations(reset_info=backup)

        for _, v in self._children.items():
            v._create_dummy_domains_for_DG(problem,elemdim=None if elemdim is None else elemdim-1)


    #This will create new equations multiple occuring equations (Important, since the same equation might occur on different nodal dims, etc)
    def _finalize_equations(self,problem:"Problem",second_loop:bool=False):
        if self._equations:
            if self._codegen is None:
                self._codegen=FiniteElementCodeGenerator()                
                # Inherited from the parent domains and the problem. The flags of this domain itself
                # are applied again in _before_compilation, but the warning threshold among them has
                # to be in place already while the residuals are assembled.
                self.get_compilation_flags(problem).apply_to_codegen(self._codegen)
                self._codegen._name=self.get_my_path_name()
                self._codegen.set_latex_printer(problem.latex_printer)
                self._codegen._set_problem(problem) 
                if second_loop and self._is_ode():
                    self._absorb_equation_state()
                    self._codegen._set_equations(self)
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
            # _codegen is only ever created above, inside the "if self._equations:" branch,
            # so its presence implies self._equations is also set.
            assert self._equations
            self._absorb_equation_state()
            self._codegen._set_equations(self)
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

    def _adopt_nodes_added_after_fill(self,problem:"Problem"):
        """Give ``_problem`` to every node that still lacks one.

        _fill_dummy_equations() hands the Problem to each node it walks, but some equations GRAFT a
        whole new domain on afterwards: _AverageOrIntegralConstraintBase._after_fill_dummy_equations
        does ``problem._equation_system += add_eqs @ odestorage`` to park its Lagrange multiplier in an
        ODE storage domain, and by then the walk is long past. The grafted node then reached
        _define_element() with ``_problem`` still None and tripped its assertion - AverageConstraint on
        a moving mesh, i.e. the free-stream GCL case of tests/test_tensor_index_conventions.py.

        Done as a sweep rather than in the constraint itself so that any other equation grafting a
        domain from _after_fill_dummy_equations() is covered too; the node is the master of its domain
        and get_coordinate_system()/the timestepping defaults all read the Problem off it.
        """
        if self._problem is None:
            self._problem=problem
        for child in self._children.values():
            child._adopt_nodes_added_after_fill(problem)

    def _create_dummy_equations_at_path(self,path:str,root:"EquationTree",problem:"Problem"):
        if (not self._equations) and (self!=root):
            self._equations=[DummyEquations()]
            self._problem=problem
        if path=="":
            return
        pth=path.split("/")
        chld=self._children.get(pth[0])
        if chld is None:
            node=EquationTree(DummyEquations(),parent=self)
            node._problem=problem
            self._children[pth[0]]=node
        self._children.get(pth[0])._create_dummy_equations_at_path("/".join(pth[1:]),root,problem) #type:ignore

    def get_child(self, name:str) -> "EquationTree":
        res = self._children.get(name)
        if res is None:
            raise ValueError("No sub-equation path '" + name + "' found at '" + self.get_full_path() + "'")
        return res

    def _clone_structure(self)->"EquationTree":
        """Creates a copy of this (sub)tree. Only the EquationTree nodes are new objects, the Equations
        stored within are shared with the original. This is exactly the same semantics as restricting a
        single Equations object to several domains at once, i.e. eqs@["domA","domB"], where the very same
        Equations object ends up in both domains."""
        if self._codegen is not None:
            raise RuntimeError("Cannot reuse an equation tree that is already part of an initialized problem")
        res = EquationTree(self._equations, parent=None)
        for k, v in self._children.items():
            child = v._clone_structure()
            child._parent = res
            res._children[k] = child
        return res

    def __matmul__(self, other:str | list[str] | tuple[str, ...] | set[str])->"EquationTree":
        """Restricts an already assembled equation tree to a further domain, e.g.
        ``(eqs + DirichletBC(u=0)@"left") @ "domain"``. Accepts the same names, paths, lists and glob
        patterns as :py:meth:`BaseEquations.__matmul__`."""
        if isinstance(other, (list,tuple,set,)):
            # Restricting to several domains at once: each domain gets its own copy of the tree structure,
            # but shares the Equations objects (as it is also the case for BaseEquations@[...])
            res = EquationTree(None, None)
            for d in other:
                if d is None or d==".": #type:ignore
                    res += self._clone_structure()
                else:
                    res += self._clone_structure() @ d
            return res
        if isinstance(other, str): #type:ignore
            splt = other.split("/")
            splt = [x for x in splt if x]  # Remove empties
            if len(splt) == 0:
                raise ValueError("Please restrict equations with a non-empty domain name")
            if not (self._parent is None):
                # Already part of another tree, i.e. the same equations are meant to be used at several
                # places. Work on a copy then, so that each place has its own tree nodes.
                return self._clone_structure() @ other
            res = EquationTree(None, parent=None)
            res._children[splt[-1]] = self
            _check_domain_name_or_pattern(splt[-1])
            self._parent = res
            for k in reversed(splt[:-1]):
                res = res @ k
            return res
        else:
            raise ValueError(
                "Please combine equation with a string (name of the domain) or a list/tuple/set of strings (names of several domains) to restrict the equations to a domain")

    def __radd__(self, other:"Literal[0] | BaseEquations")->"EquationTree":
        if other==0:
            return self
        # Once EquationTree derives from BaseEquations, CPython hands "eqs + tree" to the right
        # operand first (subclass overriding the nb_add slot wins), so this - not
        # BaseEquations.__add__ - is what runs for the commonest idiom in the whole library,
        # eqs + SomeBC() @ "boundary". Left operand stays leftmost so that child insertion
        # order, which decides which of two conditions on a boundary is applied last, is kept.
        elif isinstance(other,BaseEquations):
            return EquationTree(other,parent=None)+self
        else:
            raise RuntimeError("Cannot add "+str(other)+" and "+str(self))

    def _merge_with(self,other:"EquationTree")->"EquationTree":
        """The actual merge behind ``__add__``, without the placement check.

        It hands the children of *both* operands to the new node, i.e. it rewrites their ``_parent``.
        That is only sound while nobody else still holds the operands, which is true for the recursion
        here and for the pattern expansion, but not for a node that has already been placed - hence the
        check sits in ``__add__`` rather than here."""
        res=EquationTree(self._equations,parent=None)
        res._equations=res._equations+other._equations
        for k,v in self._children.items():
            if k in other._children.keys():
                res._children[k]=v._merge_with(other._children[k])
            else:
                res._children[k]=v
            res._children[k]._parent=res
        for k,v in other._children.items():
            if not k in self._children.keys():
                res._children[k]=v
                res._children[k]._parent = res
        return res

    def _raise_if_already_placed(self,lead:str):
        if self._parent is None:
            return
        raise RuntimeError(
            lead+" an equation tree that has already been placed at '"+self.get_full_path()+
            "', e.g. by an earlier @ 'domain' or by adding it to the Problem.\n"            
            "Assemble all equations of a domain first, and place the result afterwards.")

    def __add__(self, other:"EquationTree | BaseEquations | Literal[0]")->"EquationTree":
        if other==0:
            return self
        if isinstance(other,EquationTree):
            # Being placed is exactly what makes the merge below unsafe, so neither operand may be.
            self._raise_if_already_placed("Cannot add to")
            other._raise_if_already_placed("Cannot add")
            return self._merge_with(other)
        elif isinstance(other,BaseEquations): #type:ignore
            return self+EquationTree(other,parent=None)
        else:
            raise RuntimeError("Cannot combine a EquationTree by adding "+repr(self)+" and "+repr(other))

    def numerical_factors_to_string(self,indent:str="")->str:
        pth = self.get_my_path_name()
        res = indent
        if self._equations:
            res += "--" + pth + " : "
            assert self._codegen is not None
            for k,v in self._codegen._named_numerical_factors.items():
                res = res + "\n" + indent + " " * (len(pth) + 6) + str(k)+" = "+str(v)
        elif self._parent is not None:
            res += "--" + pth
        else:
            res += pth
        for k, child in self._children.items():
            res = res + "\n" + child.numerical_factors_to_string(indent + (" " * 2 if pth != "/" else "") + "|")
        return res

    def _tree_string(self,indent:str="") -> str:
        pth=self.get_my_path_name()
        res=indent
        if self._equations:
            sub=indent+" "*(len(pth)+6)
            # One line per equation of the domain. A domain holding several used to print through
            # a combining equation, which contributed a "Combined Equations:" header line of its
            # own; the equations are listed directly now.
            if len(self._equations)==1:
                res+="--"+pth+" : "+self._equations[0]._tree_string(sub)
            else:
                res+="--"+pth+" : "+("\n"+sub).join(e._tree_string(sub) for e in self._equations)
        elif self._parent is not None:
            res+="--"+pth
        else:
            res+=pth
        for _,v in self._children.items():
            res=res+"\n"+v._tree_string(indent+(" "*2 if pth!="/" else "")+"|")
        return res

    def __str__(self) -> str:
        return self._tree_string()




from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
