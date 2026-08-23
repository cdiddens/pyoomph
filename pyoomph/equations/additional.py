"""
The rarer generic equation classes, deliberately kept out of the top-level namespace. Unlike
:py:mod:`pyoomph.equations.generic`, this module is *not* pulled in by ``from pyoomph import *``
and must be imported explicitly::

    from pyoomph.equations.additional import *

* How a domain is assembled: :py:class:`SpatialIntegrationOrder`,
  :py:class:`EquationCompilationFlags`, :py:class:`SetCoordinateSystem`,
  :py:class:`ApplyMappingOnAddedResidual`, :py:class:`SubstituteVarByExpression`
* Declarative refinement criteria: :py:class:`RefineMaxElementSize`,
  :py:class:`RefineAccordingToElement`
* Residual terms and history: :py:class:`ResidualContribution`,
  :py:class:`BackupHistoryExpressions`
* Special-purpose boundary conditions: :py:class:`InactiveDirichletBC`,
  :py:class:`AxisymmetryBCForScalarD0Field`, :py:class:`PinMeshAtDistanceToInterface`,
  :py:class:`InteriorBoundaryOrientation`
"""
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

from ..generic.codegen import Equations, BaseEquations, InterfaceEquations, FiniteElementCodeGenerator
from .generic import DirichletBC, PinWhere
from ..utils.smallest_circle import make_circle
from ..expressions.generic import Expression, ExpressionOrNum, FiniteElementSpaceEnum, weak, var

from ..typings import *
if TYPE_CHECKING:
    from ..expressions.coordsys import BaseCoordinateSystem
    from ..generic.problem import Problem
    from ..meshes import AnySpatialMesh, AnyMesh
    from ..generic.problem import EquationTree
    from ..solvers.generic import GenericEigenSolver
    from ..meshes.mesh import Element


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


class RefineMaxElementSize(Equations):
    """
    Refines until no element exceeds the given nondimensional Cartesian size (area in 2d, volume in
    3d), irrespective of any error estimator.

    Args:
        max_nondim_cartesian_size: The size no element may exceed.
    """
    def __init__(self,max_nondim_cartesian_size:float):
        super(RefineMaxElementSize, self).__init__()
        self.max_nondim_size=max_nondim_cartesian_size

    def _register_refinement_directives(self,codegen):
        mesh=codegen._mesh
        assert mesh is not None
        # A C++ refinement directive, for the same reason as RefineToLevel: it is a pure function of the
        # element's own size, so evaluating it in C++ covers halo copies too and needs no synchronisation.
        mesh._add_refinement_directive_max_element_size(float(self.max_nondim_size))

    def after_compilation(self,codegen):
        self._register_refinement_directives(codegen)



# Refine an element to a level specified by a callback with the element


class RefineAccordingToElement(Equations):
    """
    Refines each element to the level a Python callback returns for it, e.g.
    ``RefineAccordingToElement(lambda e: 3 if e.get_Eulerian_midpoint()[0]<0.5 else 1)``. Unlike
    :py:class:`RefineMaxElementSize`, the criterion is evaluated in Python and hence can be anything.

    Args:
        level_func: Called with each element, returns the refinement level it should have.
        prevent_unrefinement: Also keep elements from being unrefined below that level.
    """
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


class SubstituteVarByExpression(BaseEquations):
    """
    If not defined in this domain, all statements of ``var(name)`` in the weak formulations in this domain will be replaced by the given expressions.
    If you e.g. use a :py:class:`~pyoomph.equations.advection_diffusion.AdvectionDiffusionEquations`, you can combine it with ``SubstituteVarByExpression(velocity=vector(ux,uy))`` to introduce a prescribed velocity.
    
    You can do the same on the problem level, i.e. as fallback for all equations, by the :py:class:`~pyoomph.generic.problem.Problem`-method :py:meth:`~pyoomph.generic.problem.Problem.define_named_var`. With this class, you can do it on particular domains only.
    
    Args:
        also_on_interfaces: Whether the substitution is also applied on the interfaces of this domain.
        **def_vars: The substitutions as ``name=expression`` pairs.
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
        analytical_position_jacobian: Whether to derive the moving-mesh (nodal position) Jacobian symbolically.
        analytical_jacobian: Whether to derive the Jacobian symbolically instead of by finite differences.
        warn_on_large_numerical_factor: Report a numerical coefficient in the expanded residual exceeding this magnitude, which usually means a scale is off. A positive value warns, a negative one raises, 0 switches the check off.
        debug_jacobian_epsilon: Assemble both the analytical and the finite-difference Jacobian and report entries differing by more than this. For debugging only - it is expensive.
        ccode_expression_mode: How the generated C expressions are rearranged: ``"expand"``, ``"normal"``, ``"collect_common_factors"`` or ``"factor"``.
        with_adaptivity: Whether to generate the code required for spatial adaptivity.
        jacobian_hoist_min_cost: How large a Jacobian/Hessian coefficient must be before it is named above the trial loop. ``-1`` follows the global setting.
        split_rjm_by_flag: Whether the residual/Jacobian/mass function is emitted once per assembly mode instead of one body branching at runtime.
        split_rjm_by_hang: Whether the residual/Jacobian/mass function additionally gets a twin without the hanging-node machinery, used for elements in which nothing hangs. Requires ``split_rjm_by_flag`` and ``with_adaptivity``.

    Every argument defaults to ``None``, which means "inherit": the setting is then taken from the
    domain this one is nested in, and eventually from :py:attr:`~pyoomph.generic.problem.Problem.equation_compilation_flags`.
    Adding e.g. ``EquationCompilationFlags(analytical_jacobian=False)`` to a bulk domain therefore
    also disables the analytical Jacobian on all its interfaces, while everything else stays as set
    on the problem.
    """

    #: The settings controlled here. Each one is a read/write property of the code generator of the
    #: same name, so they can be transferred generically.
    _flag_names = ("analytical_position_jacobian", "analytical_jacobian", "warn_on_large_numerical_factor", "debug_jacobian_epsilon", "ccode_expression_mode", "with_adaptivity", "jacobian_hoist_min_cost", "split_rjm_by_flag", "split_rjm_by_hang")

    def __init__(self,analytical_position_jacobian:bool | None=None,analytical_jacobian:bool | None=None,warn_on_large_numerical_factor:float | None=None,debug_jacobian_epsilon:float | None=None,ccode_expression_mode:str | None=None,with_adaptivity:bool | None=None,jacobian_hoist_min_cost:int | None=None,split_rjm_by_flag:bool | None=None,split_rjm_by_hang:bool | None=None):
        super(EquationCompilationFlags, self).__init__()
        self.analytical_position_jacobian=analytical_position_jacobian
        self.analytical_jacobian=analytical_jacobian
        self.warn_on_large_numerical_factor=warn_on_large_numerical_factor
        self.debug_jacobian_epsilon=debug_jacobian_epsilon
        self.ccode_expression_mode=ccode_expression_mode
        self.with_adaptivity=with_adaptivity
        self.jacobian_hoist_min_cost=jacobian_hoist_min_cost
        self.split_rjm_by_flag=split_rjm_by_flag
        self.split_rjm_by_hang=split_rjm_by_hang

    def with_defaults_from(self,fallback:"EquationCompilationFlags | None")->"EquationCompilationFlags":
        """Returns a copy of these flags where every setting left at ``None`` is taken from ``fallback``."""
        if fallback is None:
            return self
        res=EquationCompilationFlags()
        for n in self._flag_names:
            own=getattr(self,n)
            setattr(res,n,getattr(fallback,n) if own is None else own)
        return res

    def apply_to_codegen(self,codegen:"FiniteElementCodeGenerator"):
        """Writes every setting that is not ``None`` to the code generator, leaving the others alone."""
        for n in self._flag_names:
            v=getattr(self,n)
            if v is not None:
                setattr(codegen,n,v)

    def _before_compilation(self,codegen:"FiniteElementCodeGenerator"):
        self.apply_to_codegen(codegen)

    def define_fields(self):
        # Also set in _before_compilation, but the warning fires while the residuals are assembled,
        # i.e. long before that.
        if self.warn_on_large_numerical_factor is not None:
            self.get_current_code_generator().warn_on_large_numerical_factor=self.warn_on_large_numerical_factor


class SetCoordinateSystem(Equations):
    """
    Set the default coordinate system for the current equations. It will override the coordinate system set on the problem level.

    Args:
        coord_sys: A coordinate system instance from :py:mod:`pyoomph.expressions.coordsys`, e.g. ``axisymmetric``.
    """
    def __init__(self,coord_sys:"BaseCoordinateSystem"):
        super(SetCoordinateSystem, self).__init__()
        self.coord_sys=coord_sys

    def define_fields(self):
        master = self._master()
        master._coordinate_system=self.coord_sys


class ApplyMappingOnAddedResidual(BaseEquations):    #
    """
    Installs a mapping that is applied to every residual added afterwards, e.g. to send a contribution
    to another residual than the one it was written for.

    Args:
        mapping: Called with the destination name and the expression, returns either the mapped
            expression or a dict of destination name to expression.
    """
    def __init__(self,mapping:Callable[[str,"Expression"],'Expression | dict[str, "Expression"]']=lambda destination,expr:{destination:expr}):
        super(ApplyMappingOnAddedResidual, self).__init__()
        self.mapping=mapping

    def define_fields(self):
        master=self._master()
        master._residual_mapping_functions.append(self.mapping)


class BackupHistoryExpressions(Equations):
    """
    Stores the value of arbitrary expressions in fields of their own, so that their time history is
    available later on - the history of an expression is otherwise not accessible, only that of the
    fields it is built from.

    Args:
        space: Space of the storage fields. Defaults to ``"C2"``.
        local_expression_format: Naming pattern of the local expression bound to each stored field.
        **exprs: Name to expression to keep the history of.
    """
    def __init__(self,*,space:FiniteElementSpaceEnum="C2",local_expression_format="_history_expr_{name}",**exprs:Expression):
        super().__init__()
        self.history_fields=exprs
        self.space:FiniteElementSpaceEnum=space
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


class ResidualContribution(BaseEquations):
    """
    A class to add an arbitrary residual contribution to the equations. This is useful to add additional terms to the equations that are not covered by the standard weak formulation. Essentially, it just adds ``r`` to the residuals.

    Args:
        r: The residual to add, e.g. a :py:func:`~pyoomph.expressions.generic.weak` contribution.
        destination: The residual destination. Can be used to fill more than one residual.
    """
    def __init__(self,r:"ExpressionOrNum | str",destination:str | None=None):
        super(ResidualContribution, self).__init__()        
        self.destination=destination
        self.r=r

    def define_residuals(self):
        self.add_residual(self.r,destination=self.destination)


class InactiveDirichletBC(DirichletBC):
    """
    Same as 'DirichletBC', but it starts deactivated, i.e. the Neumann term will be active by default.

    To activate the Dirichlet condition, call ``set_dirichlet_active(...)`` on the mesh obtained from
    :py:meth:`~pyoomph.generic.problem.Problem.get_mesh`, followed by
    :py:meth:`~pyoomph.generic.problem.Problem.reapply_boundary_conditions`::

        problem.get_mesh("domain/interface").set_dirichlet_active(u=True)  # activate the BC
        problem.reapply_boundary_conditions()  # renumber the equations and apply the BCs
        problem.solve()  # solve with the active DirichletBC

    """
    def __init__(self, *, prefer_weak_for_DG: bool = True, **kwargs: ExpressionOrNum):
        super().__init__(prefer_weak_for_DG=prefer_weak_for_DG, **kwargs)
        self._init_setup_for_mesh:set[AnyMesh]=set()
    
    def before_assigning_equations_preorder(self, mesh: "AnyMesh"):
        if mesh in self._init_setup_for_mesh: # Only init it once during problem init. Someone might have switched it later on
            return super().before_assigning_equations_preorder(mesh)        
        # Expanded, like define_residuals: set_dirichlet_active names the SCALAR fields, so a
        # vectorial entry has to be split here too or its components would stay active.
        mesh.set_dirichlet_active(**{k:False for k in self._expanded_dcs().keys()})
        self._init_setup_for_mesh.add(mesh) # Don't ever set it again for this mesh
        # TODO: Check redefine_problem and/or remeshing
        return super().before_assigning_equations_preorder(mesh)

    def after_remeshing(self, eqtree: "EquationTree"):
        raise RuntimeError("Check remeshing settings here...")
        return super().after_remeshing(eqtree)


class AxisymmetryBCForScalarD0Field(InterfaceEquations):
    """
    The counterpart of :py:class:`~pyoomph.equations.generic.AxisymmetryBC` for elementally constant
    (``"D0"``) scalar fields: it zeroes them in the eigenproblem of any nonzero azimuthal mode, which
    the nodal treatment of the ordinary axisymmetry condition cannot reach.

    Args:
        *fields: Names of the discontinuous fields to constrain.
    """
    def __init__(self,*fields:str):
        super().__init__()
        self.fields=[f for f in fields]

    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree: "EquationTree", eigensolver: "GenericEigenSolver", angular_mode: int | float | None,normal_k:float | None) -> set[str | int]:
        eqs:set[str | int]=set()
        if angular_mode!=0:
            assert eqtree._mesh is not None
            for ie in eqtree._mesh.elements():
                be=ie.get_bulk_element()
                for f in self.fields:
                    fi=be.get_jit_code().get_discontinuous_field_index(f)
                    if fi<0:
                        raise RuntimeError("Discontinuous parent field '"+str(f)+"' not known here")                
                    eqs.add(be.internal_data_pt(fi).eqn_number(0))
        return eqs


class PinMeshAtDistanceToInterface(PinWhere):
    """
    Pins the mesh coordinates of all nodes further than ``distance`` away from the given interfaces,
    i.e. lets the mesh move only in a band around them. The distance is measured against the smallest
    circle enclosing the interfaces, re-fitted whenever the conditions are applied.

    Args:
        interface_names: Name(s) of the interfaces to measure the distance from.
        distance: Beyond this distance the mesh is frozen.
        mode: How the interface is approximated. Only ``"smallest_circle"`` is implemented.
    """
    def __init__(self, interface_names:str | set[str], distance:ExpressionOrNum, mode:str="smallest_circle"):
        super(PinMeshAtDistanceToInterface, self).__init__(where=lambda : False, mesh_x=True, mesh_y=True)
        if isinstance(interface_names, str):
            self.interface_names = {interface_names}
        else:
            self.interface_names = set(self.interface_names)
        self.distance = distance
        self._circle_x = 0
        self._circle_y = 0
        self._circle_radius = 0

    def _build_where_func(self):
        assert self.mesh is not None
        pts:list[tuple[float,float]] = []
        for inter in self.interface_names:
            for n in self.mesh.boundary_nodes(inter):
                pts.append((n.x(0), n.x(1),))
        self._circle_x, self._circle_y, self._circle_radius = make_circle(pts)
        self._circle_radius += float(self.distance / self.mesh.get_problem().get_scaling("spatial"))
        self.where:Callable[[float,float],bool] = lambda x, y: (x - self._circle_x) ** 2 + (y - self._circle_y) ** 2 > self._circle_radius ** 2

    def apply(self):
        if not self.active:
            return
        self._build_where_func()
        super(PinMeshAtDistanceToInterface, self).apply()


class InteriorBoundaryOrientation(InterfaceEquations):
    """
    Named interior boundaries within a domain are by default double-layered, i.e. interface elements are added from both sides.
    This can usually cause problems. In order to avoid this, we have to specify the orientation of the boundary, i.e. only interface elements are added from one side, namely where the indicator function is positive.
    For a unit circle ``"circle"`` embedded in a domain, you could e.g. add
    ``InteriorBoundaryOrientation(dot(var("coordinate"),var("normal")))@"circle"`` to only add interface
    elements with an outward pointing normal.

    Args:
        indicator: Interface elements are kept on the side where this expression is positive.
    """    
    def __init__(self,indicator:ExpressionOrNum):
        super().__init__()
        self.indicator=indicator
        
    def define_residuals(self):
        self.add_local_function("__interface_constraint",self.indicator)


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
