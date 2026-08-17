"""
The physics-independent equation classes of pyoomph: the pieces that are combined with actual
physics (:py:mod:`pyoomph.equations.navier_stokes`, :py:mod:`pyoomph.equations.poisson`, ...) to
state a problem.

* Fields and residual terms: :py:class:`ScalarField`, :py:class:`VectorField`,
  :py:class:`WeakContribution`, :py:class:`ElementSpace`
* Boundary and nodal conditions: :py:class:`DirichletBC`, :py:class:`NeumannBC`,
  :py:class:`EnforcedBC`, :py:class:`EnforcedDirichlet`, :py:class:`AxisymmetryBC`,
  :py:class:`PeriodicBC`, :py:class:`PythonDirichletBC`, :py:class:`PinWhere`,
  :py:class:`UnpinDofs`
* Initial conditions and nondimensionalisation: :py:class:`InitialCondition`,
  :py:class:`Scaling`, :py:class:`TestScaling`
* Constraints and couplings: :py:class:`IntegralConstraint`, :py:class:`AverageConstraint`,
  :py:class:`GlobalLagrangeMultiplier`, :py:class:`ConnectFieldsAtInterface`,
  :py:class:`ConstrainFieldsToC1Space`, :py:class:`ForceZeroOnEigenSolve`
* Diagnostics and derived quantities: :py:class:`IntegralObservables`,
  :py:class:`ExtremumObservables`, :py:class:`ODEObservables`, :py:class:`LocalExpressions`,
  :py:class:`ProjectExpression`
* Adaptivity and remeshing: :py:class:`SpatialErrorEstimator`, :py:class:`TemporalErrorEstimator`,
  :py:class:`RefineToLevel`, :py:class:`RemeshWhen`, :py:class:`RemeshMeshSize`

Everything here is reachable from ``from pyoomph import *``. The rarer relatives live in
:py:mod:`pyoomph.equations.additional`, which has to be imported explicitly.
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
 
 

# Not "from .. import var_and_test,var": this module is imported while the pyoomph package itself is
# still being set up, so it must not depend on what the top-level __init__ has bound so far.
from .. import _pyoomph_core as _pyoomph
from ..expressions import var_and_test,var
from ..generic.codegen import  InterfaceEquations,Equations,BaseEquations,ODEEquations,FiniteElementCodeGenerator,sorted_field_kwargs
from ..expressions.generic import ExpressionOrNum,ExpressionNumOrNone,FiniteElementSpaceEnum, grad,nondim, scale_factor,test_scale_factor,Expression,assert_valid_finite_element_space, testfunction,find_dominant_element_space,weak

#Connects one or multiple fields at both sides of the interfaces via Lagrange multipliers
#i.e. it ensures the same Neumann flux on both sides, whereas the magnitude of this flux is given by the Lagrange multiplier
#which is automatically chosen that way that the condition <inner>=<outer> is satisfied.
from ..meshes.mesh import MeshFromTemplateBase,Element,InterfaceMesh,MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,assert_spatial_mesh

from ..typings import *
import inspect
import numpy
import scipy.spatial #type:ignore

if TYPE_CHECKING:
    from ..meshes.remesher import RemesherBase,RemesherPointEntry
    from ..expressions.coordsys import BaseCoordinateSystem
    from ..meshes import AnyMesh,AnySpatialMesh
    from ..generic.problem import Problem,EquationTree
    from ..solvers.generic import GenericEigenSolver
    from ..meshes.mesh import Node
    


# TODO Check this
def get_interface_field_connection_space(inside_space:FiniteElementSpaceEnum | Literal[""],outside_space:FiniteElementSpaceEnum | Literal[""],use_highest_space:bool=False)->FiniteElementSpaceEnum | Literal[""]:
    if outside_space == "":
        return inside_space
    elif inside_space == "":
        return outside_space    
    if outside_space[0]!=inside_space[0]:
        raise RuntimeError("TODO: Think about what space is lower/higher ") #TODO: Is e.g. D2 lower or higher than C2TB? hard to tell
    space_order:list[FiniteElementSpaceEnum]=["D2TB","C2TB","D2","C2","D1TB","C1TB","D1","C1"]
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
        check_consistent_scaling: Flag indicating whether to check for consistent scaling of the fields on both sides. Defaults to True.
    """
    def __init__(self,fields:str | dict[str, str] | list[str],*,lagr_mult_prefix:str="_lagr_conn_",use_highest_space:bool=False,check_consistent_scaling:bool=True):
        super(ConnectFieldsAtInterface, self).__init__()
        self.lagr_mult_prefix=lagr_mult_prefix
        self.use_highest_space=use_highest_space
        self.check_consistent_scaling=check_consistent_scaling
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
            if self.check_consistent_scaling:
                testdiff=test_scale_factor(finner)-test_scale_factor(fouter,domain=self.get_opposite_side_of_interface())
                testdiff=self.expand_expression_for_debugging(testdiff,raise_error=False,unit_error=False)
                if not testdiff.is_zero():
                    testscale_inside=self.expand_expression_for_debugging(test_scale_factor(finner))
                    testscale_outside=self.expand_expression_for_debugging(test_scale_factor(fouter,domain=self.get_opposite_side_of_interface()))
                    
                    raise self.add_exception_info(RuntimeError("When connecting fields "+str(finner)+" and "+str(fouter)+" at the interface, the test function scaling is inconsistent.\nPlease either set check_consistent_scaling=False or ensure that the test function scaling is consistent.\n   test_scale("+str(finner)+")_inside = "+str(testscale_inside)+"\n   test_scale("+str(fouter)+")_outside = "+str(testscale_outside)))
                testdiff=scale_factor(finner)-scale_factor(fouter,domain=self.get_opposite_side_of_interface())
                testdiff=self.expand_expression_for_debugging(testdiff,raise_error=False,unit_error=False)
                if not testdiff.is_zero():
                    scale_inside=self.expand_expression_for_debugging(scale_factor(finner))
                    scale_outside=self.expand_expression_for_debugging(scale_factor(fouter,domain=self.get_opposite_side_of_interface()))
                    raise self.add_exception_info(RuntimeError("When connecting fields "+str(finner)+" and "+str(fouter)+" at the interface, the scaling is inconsistent.\nPlease either set check_consistent_scaling=False or ensure that the scaling is consistent.\n   scale("+str(finner)+")_inside = "+str(scale_inside)+"\n   scale("+str(fouter)+")_outside = "+str(scale_outside)))


            l, l_test=var_and_test(self.lagr_mult_prefix+finner+"_"+fouter)
            inside, inside_test=var_and_test(finner)
            outside, outside_test=var_and_test(fouter,domain=self.get_opposite_side_of_interface())
            scal=self.get_scaling(finner)
            self.add_residual((inside-outside)/scal*l_test*dx) #TODO: Possibly nodal connection?
            self.add_residual(l*inside_test*dx)
            self.add_residual(-l*outside_test*dx)
           
    def before_assigning_equations_postorder(self, mesh: "AnyMesh") -> None:
        # ConnectFieldsAtInterface is an InterfaceEquations, so whenever this is invoked
        # (i.e. the equations were actually assigned to some mesh), that mesh is the
        # corresponding InterfaceMesh. Statically, before_assigning_equations_postorder is
        # declared generically on AnyMesh since it is shared by domain and interface equations.
        assert isinstance(mesh,InterfaceMesh)
        for finner,fouter in self.fields.items():
            lname=self.lagr_mult_prefix+finner+"_"+fouter
            self.pin_redundant_lagrange_multipliers(mesh,lname,finner,fouter)

        super().before_assigning_equations_postorder(mesh)

    def with_removed_overconstraining(self,*corners:str):
        return self+sum([_ConnectFieldsAtInterfaceRemoveOverconstraining(self.fields)@corner for corner in corners]) #type:ignore


class _ConnectFieldsAtInterfaceRemoveOverconstraining(InterfaceEquations):
    required_parent_type = ConnectFieldsAtInterface
    def __init__(self,fields:str | dict[str, str] | list[str]):
        super(_ConnectFieldsAtInterfaceRemoveOverconstraining, self).__init__()
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
    Spatial error estimators are used to estimate where a mesh should be refined. You can either pass
    variable name(s) and numerical factor(s)::

        SpatialErrorEstimator(u=1, v=10)

    In that case, the jumps of the gradients ``grad(u)`` and ``10*grad(v)`` will be used as error
    estimators. Alternatively, you can also provide custom expressions as estimators, e.g. for
    discontinuous fields, it might be better to just add::

        SpatialErrorEstimator(5*var("u"))

    so that the jump in ``"u"`` is used, after weighting by the factor 5, as error estimator.
    Error estimator expressions must be nondimensional.

    ``for_which`` controls whether these error estimators are used for the base solution, potential
    eigenfunctions or both.

    ``normalize_relative`` selects what the resulting numbers mean, i.e. what
    :py:attr:`~pyoomph.generic.problem.Problem.max_permitted_error` is compared against:

    * ``1`` (default) gives the **relative** error: each element's error is divided by the recovered
      flux norm of the whole mesh, so it is that element's share of this mesh's total. Dimensionless,
      but blind to how well resolved the mesh is overall -- a well resolved and a badly resolved
      domain report errors of similar magnitude, and refining the mesh does not by itself make the
      numbers smaller.
    * ``0`` gives the **absolute** error: the raw integrated flux jump, which really does shrink as
      the mesh is refined and is comparable between meshes and between adaptation steps. In exchange,
      its magnitude depends on the element size and on the field's own scale, so
      ``max_permitted_error``/``min_permitted_error`` have to be chosen for the problem at hand
      rather than left at the defaults.
    * values in between divide by ``norm**normalize_relative``, which is the geometric blend of the
      two (``err/norm**p`` is exactly ``(err/norm)**p * err**(1-p)``).

    Note that ``normalize_relative`` is the only knob that can change the *scale* of the errors of a
    group: a common factor on the estimator expressions themselves cancels out of the relative
    normalisation exactly (``SpatialErrorEstimator(u=2)`` is identical to
    ``SpatialErrorEstimator(u=1)``); only the ratios between the fields *within one group* matter
    there. To scale a group against the others, use ``weight``, which is applied after the
    normalisation and therefore does not cancel.

    The norm being divided out is a whole-group quantity, so ``normalize_relative`` and ``weight``
    belong to the group rather than to the individual expressions. That is what ``group`` is for::

        eqs += SpatialErrorEstimator(u=1)                                        # group ""
        eqs += SpatialErrorEstimator(Gamma=1, group="surf", normalize_relative=0, weight=3)

    Each group gets its **own** recovered-flux norm, and the elemental error is the **maximum** over
    the groups. Two criteria in different groups can therefore neither dilute nor mask each other,
    and adding a group can only ever cause more refinement and less unrefinement, never the reverse
    -- which is what makes it safe for independent pieces of a model to contribute error criteria
    without knowing about each other. Everything lands in the single unnamed group ``""`` unless a
    group is named, which is the historical behaviour of one joint norm over all fields.

    Vetoes ("refine this regardless", "do not unrefine this") deliberately do *not* live here: they
    would have to lower an element's error and so break the monotonicity that makes the maximum safe.
    Use :py:class:`RefineToLevel`, :py:class:`~pyoomph.equations.additional.RefineMaxElementSize` or
    :py:class:`~pyoomph.equations.additional.RefineAccordingToElement` for those.

    Args:
        *fluxes: Estimators with factor 1, either an expression used as it is, e.g. ``5*var("u")``,
            or a field name, which then uses the jump of its gradient.
        for_which: Whether the estimators act on the base solution, on eigenfunctions or on both.
        group: Name of the group this criterion belongs to. Each group gets its own recovered-flux
            norm, and the elemental error is the maximum over the groups.
        normalize_relative: Exponent of the norm divided out, see above. ``1`` gives the relative,
            ``0`` the absolute error.
        weight: Applied after the normalisation, i.e. it does not cancel and can scale this group
            against the others.
        **kwargs: Field names with their numerical factors, e.g. ``u=1, v=10``, which use the jump
            of the weighted gradient of that field.
    """

    def __init__(self,*fluxes:str | Expression,for_which:Literal["both","base","eigen"]="both",group:str="",normalize_relative:float=1.0,weight:float=1.0,**kwargs:ExpressionOrNum):
        super(SpatialErrorEstimator, self).__init__()
        self.fluxes:dict[str | Expression,ExpressionOrNum]={x:1.0 for x in fluxes}
        for lhs,rhs in sorted_field_kwargs(kwargs).items():
            self.fluxes[lhs]=rhs
        normalize_relative=float(normalize_relative)
        if not (0.0<=normalize_relative<=1.0):
            raise ValueError("normalize_relative must be between 0 (absolute error) and 1 (relative error), got "+str(normalize_relative))
        if float(weight)<=0.0:
            raise ValueError("weight must be positive, got "+str(weight))
        self.group=group
        self.normalize_relative=normalize_relative
        self.weight=float(weight)
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
        # Tensor fields have to be taken apart: grad() of one is not a vector gradient, so naming it
        # here used to raise "matrix::operator(): index out of range" from vector_gradient. Each
        # component contributes its own criterion to the same group, which is what naming the
        # components by hand would have done anyway.
        combined=self._get_combined_element()
        tensorfields=getattr(combined,"_tensorfields",{}) if isinstance(combined,Equations) else {}
        for flux,factor in self.fluxes.items():
            if isinstance(flux,str):
                if flux=="normal":
                    jflux=nondim("normal") #Normal is not derived
                elif flux=="mesh":
                    jflux=grad(nondim("mesh"),nondim=True,lagrangian=True)
                elif flux in tensorfields:
                    for component in sorted({c for row in tensorfields[flux] for c in row if c}):
                        self.add_spatial_error_estimator(factor*grad(nondim(component),nondim=True),
                                                         for_base=self.for_base_solution,for_eigen=self.for_eigenfunction,
                                                         group=self.group,normalize_relative=self.normalize_relative,weight=self.weight)
                    continue
                else:
                    jflux=grad(nondim(flux),nondim=True)
            else:
                jflux=flux
            # No uniqueness guard here any more: the group carries the settings, so several
            # SpatialErrorEstimators on one domain just contribute several criteria, combined by the
            # maximum. Two objects that name the SAME group but disagree about how it is normalised
            # are still a mistake, and FiniteElementCode::add_Z2_flux rejects that.
            self.add_spatial_error_estimator(factor*jflux,for_base=self.for_base_solution,for_eigen=self.for_eigenfunction,
                                             group=self.group,normalize_relative=self.normalize_relative,weight=self.weight)

    def get_information_string(self) -> str:
        res=", ".join(map(str,self.fluxes))
        extra=[]
        if self.group!="":
            extra.append("group='"+self.group+"'")
        if self.normalize_relative!=1.0:
            extra.append("normalize_relative="+str(self.normalize_relative))
        if self.weight!=1.0:
            extra.append("weight="+str(self.weight))
        if extra:
            res+=" ("+", ".join(extra)+")"
        return res


class RefineToLevel(Equations):
    """
    Refine elements to a certain level. If the level is set to "max", the elements will be refined to the maximum level set by e.g. :py:attr:`~pyoomph.generic.problem.Problem.max_refinement_level`.
    """
    def __init__(self, level:Literal["max"] | int="max"):
        super(RefineToLevel, self).__init__()
        self.level:Literal["max"] | int = level

    def register_refinement_directives(self,codegen):
        mesh=codegen._mesh
        assert mesh is not None
        # Registered as a C++ refinement directive rather than evaluated by a Python loop over the
        # elements on every adapt. Same values, but evaluated for every element this process holds --
        # halo copies included -- so a halo copy reaches the same verdict as the element it copies,
        # instead of being one more rank-local override for the halo exchange to repair afterwards.
        # See pyoomph::Mesh::apply_refinement_directives.
        mesh._add_refinement_directive_to_level(-1 if self.level=="max" else int(self.level))

    def after_compilation(self,codegen):
        self.register_refinement_directives(codegen)
        mesh=codegen._mesh
        assert mesh is not None
        # Only MeshFromTemplate1d/2d/3d actually carry _initial_uniform_refinement_level.
        # The previous "not isinstance(mesh,InterfaceMesh)" check also let ODEStorageMesh
        # through, which does not have this attribute at all and would raise an
        # AttributeError at runtime if RefineToLevel were ever attached to an ODE domain.
        if isinstance(mesh,MeshFromTemplateBase):
            problem=mesh.get_problem()
            mesh._initial_uniform_refinement_level=max(mesh._initial_uniform_refinement_level,self.level if self.level!="max" else (problem.initial_adaption_steps if problem.initial_adaption_steps is not None else problem.max_refinement_level) )
        

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

    def keys(self) -> list[str]:
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
    def __init__(self,remeshing_opts:RemeshingOptions | None=None,*,max_expansion:float | None=None,min_expansion:float | None=None,min_solves_before_remesh:int | None=0,reinit_initial_size_after_one_step:bool | None=False,active:bool=True,min_quality_decrease:float | None=None,on_invalid_triangulation:bool=False):

        super(RemeshWhen, self).__init__()
        if isinstance(remeshing_opts,RemeshingOptions):
            self.max_expansion:float | None=remeshing_opts.max_expansion
            self.min_expansion:float | None=remeshing_opts.min_expansion
            self.min_solves_before_remesh:int | None=remeshing_opts.min_solves_before_remesh
            self.reinit_initial_size_after_one_step:bool | None=remeshing_opts.reinit_initial_size_after_one_step
            self.min_quality_decrease:float | None=remeshing_opts.min_quality_decrease
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

        # get_cached_mesh_data only accepts a "real" spatial mesh (not an ODEStorageMesh).
        # If mesh isn't a MeshFromTemplateBase (i.e. RemeshWhen ended up on a domain that
        # cannot be remeshed at all), skip straight to the check below, which raises a
        # clear RuntimeError instead of failing deep inside get_cached_mesh_data.
        if self.on_invalid_triangulation and not need_remesh and isinstance(mesh,MeshFromTemplateBase):
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
    def __init__(self,size:float | Callable[["RemesherPointEntry"], float] | None=None):
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
    """
    Projects an expression onto a finite element space. 
    The projected field can be used in the equations as a variable.
    
    Args:
        scale: Scale of the projected field. A string is resolved as ``scale_factor(scale)``, i.e. it
            refers to a registered scale by name. Defaults to 1.
        space: Finite element space to project onto. Defaults to "C2".
        destination: Residual destination for the projection. Defaults to None, i.e. the default residual.
        field_type: Type of the projected field. Can be "scalar" or "vector". Defaults to "scalar".
        coordinate_system: Coordinate system for the projection. If None, the coordinate system of the domain/problem will be used. Defaults to None.
        **projs: Keyword arguments representing the expressions to project. The keys are the names of the projected fields, and the values are the expressions to project.
        
    """
    def __init__(self,scale:ExpressionOrNum | str=1,space:FiniteElementSpaceEnum="C2",destination:str | None=None,field_type:Literal["scalar","vector"]="scalar",coordinate_system:"BaseCoordinateSystem | None"=None, **projs:ExpressionOrNum):
        super(ProjectExpression, self).__init__()
        self.space:FiniteElementSpaceEnum=space
        self.scale:ExpressionOrNum=scale_factor(scale) if isinstance(scale,str) else scale
        self.field_type=field_type
        self.projs=projs.copy()
        self.coordinate_system=coordinate_system
        self.destination=destination
        
    def define_fields(self):
        for n,_ in self.projs.items():
            if self.field_type=="scalar":
                self.define_scalar_field(n,self.space,scale=self.scale,testscale=1/self.scale)
            elif self.field_type=="vector":
                self.define_vector_field(n,self.space,scale=self.scale,testscale=1/self.scale)
            else:
                raise ValueError("Unsupported field type "+self.field_type)

    def define_residuals(self):
        from ..expressions.generic import weak
        for n,e in self.projs.items():
            f,ftest=var_and_test(n)
            self.add_residual(weak(f,testfunction(n,dimensional=False)/scale_factor(n),coordinate_system=self.coordinate_system),destination=self.destination)
            self.add_residual(weak(-e,testfunction(n,dimensional=False)/scale_factor(n),coordinate_system=self.coordinate_system),destination=self.destination)

class InitialCondition(BaseEquations):
    """
    Class representing initial conditions for a set of equations. If the initial conditions depend on time, i.e. on ``var("time")``, it will be used to initialize the history steps before the first step. Otherwise, by default, the first time step will be calculated by a first order step.

    Args:
        degraded_start: Flag indicating whether to use degraded start (i.e. first order time stepping in the first step) or not. Defaults to "auto", meaning we degrade if the initial condition does not depend on time.
        IC_name: Name of the initial condition. Defaults to an empty string, which are the default initial conditions.
        **kwargs: Keyword arguments representing the initial conditions for each variable.

    A vector field may be given as a whole, i.e. ``InitialCondition(velocity=vector(0,1))`` instead of
    ``InitialCondition(velocity_x=0, velocity_y=1)``. The value is split component by component onto
    the field's components in their own order, so this is also correct in e.g. an axisymmetric
    coordinate system. This includes the position fields and values that are not written as an explicit
    ``vector(...)``, i.e. ``InitialCondition(mesh=var("lagrangian"))``. See
    :py:meth:`~pyoomph.generic.codegen.BaseEquations.expand_vectorial_entries`.
    """

    def __init__(self, *, degraded_start: bool | Literal["auto"] = "auto", IC_name: str = "", **kwargs: ExpressionOrNum):
        super(InitialCondition, self).__init__()
        self._ics: dict[str, ExpressionOrNum] = {n: Expression(0 + v) for n, v in sorted_field_kwargs(kwargs).items()}
        self._ic_name = IC_name
        self._degraded_start = degraded_start

    def get_information_string(self):    		
        return ",".join([str(k) + "=" + str(v) for k, v in self._ics.items()])

    def define_residuals(self):
        # Vector-valued entries split into their components. Not done in __init__: which components
        # "velocity" has is a property of the domain this condition ends up on, which is not known
        # until the equations are attached.
        for n, val in self.expand_vectorial_entries(self._ics, "initial condition").items():
            assert isinstance(self._degraded_start, bool) or self._degraded_start == "auto"
            self.set_initial_condition(n, val, degraded_start=self._degraded_start, IC_name=self._ic_name)
            if self.get_problem().project_initial_conditions:
                self.add_weak(var(n)-val,testfunction(n,dimensional=False)/scale_factor(n),destination="_IC_"+self._ic_name) 


class TemporalErrorEstimator(BaseEquations):
    """
    Adds temporal error estimators, which drive the adaptive time stepping of
    :py:meth:`~pyoomph.generic.problem.Problem.run` with ``temporal_error=...``. Each field can have a
    different factor. If you have e.g. fields ``"u"`` and ``"v"``, add::

        TemporalErrorEstimator(u=1, v=10)

    to weight the error estimator of ``"u"`` with 1 and of ``"v"`` with 10, i.e. errors in ``"v"``
    count ten times as much.

    Args:
        **fieldfactors: Field names with their temporal error weighting factors.
    """

    def __init__(self, **fieldfactors: float):
        super(TemporalErrorEstimator, self).__init__()
        self.fieldfactors = sorted_field_kwargs(fieldfactors)

    def define_error_estimators(self):       
        for f, v in self.fieldfactors.items():                        
            self.set_temporal_error_factor(f, v)
            
            

class LocalExpressions(Equations):
    """
    Local expressions are additional expressions for output, evaluated on the nodes of the mesh. 
    They are not solved, but only calculated for output.
    Since it works node-wise, it might give problems, e.g. for 1/r terms at the axis of symmetry.
    An alternative is :py:class:`ProjectExpression`, which calculates such expressions by projection.
    However, these are degrees of freedom, i.e. it will be slower.

    Args:
        **local_expressions (ExpressionOrNum): A dict of expressions to be evaluated on the nodes of the mesh for output only.
    """
    def __init__(self, **local_expressions:ExpressionOrNum):
        super(LocalExpressions, self).__init__()
        self.local_expressions = {k:v for k,v in local_expressions.items()}

    def define_additional_functions(self):
        if self._get_combined_element()._is_ode():
            raise self.add_exception_info( RuntimeError("LocalExpressions cannot be used with ODE equations. Use IntegralObservables instead."))
        for k,v in self.local_expressions.items():
            self.add_local_function(k, v )
            

    def _is_ode(self):
        return None

class DependentIntegralObservable:
    """
    An :py:class:`IntegralObservables` entry that is computed from other observables instead of being
    integrated itself, e.g. ``U=DependentIntegralObservable(lambda U_sqr,Area: square_root(U_sqr/Area),"U_sqr","Area")``.
    Passing a plain ``lambda`` with named arguments to :py:class:`IntegralObservables` does the same.

    Args:
        func: Called with the values of the observables named in ``argnames``.
        *argnames: Names of the observables to feed into ``func``.
    """
    def __init__(self,func:Callable[...,ExpressionOrNum],*argnames:str):
        super(DependentIntegralObservable, self).__init__()
        self.func=func
        self.argnames=[*argnames]

    def __call__(self, *args:ExpressionOrNum) -> ExpressionOrNum:
        return self.func(*args)



class IntegralObservables(Equations):
    """
    Integral expressions will be evaluated by spatial integration over the mesh domain, e.g.::

        IntegralObservables(volume=1)

    will calculate the volume by integration over the mesh domain. In e.g. axisymmetry, the factor 2*pi*r will be included.
    Also, the output is dimensional, i.e. if you have set the scaling to a metric quantity, you will get a result in cubic meters here.
    When combined with an :py:class:`~pyoomph.output.generic.IntegralObservableOutput` object, they will be written to an output file.
    
    You can also introduce dependent integral observables. If you have a field ``"u"``, you can
    calculate its average over the domain by::

        IntegralObservables(_denom=1, _u_integral=var("u"), u_avg=lambda _u_integral, _denom: _u_integral/_denom)

    Here, ``_denom`` is the integral over 1 and ``_u_integral`` the integral over ``"u"``, so ``u_avg``
    evaluates to the average. The parameter names of the lambda must match the names of the other
    integral observables. The leading underscore keeps the helper observables out of the output.

    Note that the integration measure of an integral observable is always the current one: the
    ``apply_on_integral_dx`` of :py:func:`~pyoomph.expressions.generic.evaluate_in_past` has no effect
    here, so an integrand evaluated in the past is still integrated over the present mesh. History
    evaluation of the integrand itself, including ``apply_on_others`` for the geometry entering it,
    does work.

    Args:
        _coordinate_system: The coordinate system to use. Defaults to None, i.e. the one of the
            equations or the problem.
        _lagrangian: Integrate over the Lagrangian instead of the Eulerian domain. Defaults to False.
        **integral_observables: The observables, either expressions to integrate or callables of other
            observables (see above).
    """
    def __init__(self,_coordinate_system:"BaseCoordinateSystem | None"=None,_lagrangian:bool=False, **integral_observables:ExpressionOrNum | Callable[..., ExpressionOrNum]):
        super(IntegralObservables, self).__init__()
        is_dependent_func=lambda v: callable(v) and not isinstance(v,Expression)
        self.integral_observables = {k:v for k,v in integral_observables.items() if not is_dependent_func(v)}
        self.dependent_funcs={k:v for k,v in integral_observables.items() if is_dependent_func(v)}
        self._coordinate_system=_coordinate_system
        self._lagrangian=_lagrangian

    def define_additional_functions(self):
        if self._coordinate_system is None:
            dx = self.get_dx(lagrangian=self._lagrangian)
        else:
            dx=self.get_dx(coordsys=self._coordinate_system,lagrangian=self._lagrangian)
        for k,v in self.integral_observables.items():
            #import pyoomph._pyoomph_core as _pyoomph
            #_pyoomph.set_verbosity_flag(1)
            # self.integral_observables was filtered (in __init__) to only contain the
            # non-callable entries of integral_observables, but pyright cannot track that
            # invariant through the dict comprehension, so it still sees the full
            # ExpressionOrNum | Callable[...,ExpressionOrNum] union here.
            assert isinstance(v,(int,float,Expression))
            self.add_integral_function(k, v * dx)
            #_pyoomph.set_verbosity_flag(0)
        for k,v in self.dependent_funcs.items():
            # Symmetric to the above: self.dependent_funcs was filtered to only the callable
            # entries.
            assert callable(v)
            self.add_dependent_integral_function(k,v)

    def _is_ode(self):
        return None

class ExtremumObservables(Equations):
    """
    Add these to continuum equations to find minima and maxima of given expressions.
    If you want to find the minimum/maximum of a scalar quantity "u", you can add an ``ExtremumObservables("u")`` or, alternatively, give it a name like ``ExtremumObservables(my_name_for_u=var("u"))``.
    More than one argument can be passed to register multiple extremum observables. For e.g. the maximum of a norm of a vectorial variable v, you can add ``ExtremumObservables(v_norm=square_root(dot(var("v"),var("v"))))``.  
    
    Once registered, you can evaluate the extremum values by calling the ``evaluate_maximum`` or ``evaluate_minimum`` method of the corresponding mesh (available via ``problem.get_mesh(...)``)

    Args:
        *direct_vars: Names of variables to monitor, e.g. ``"u"``.
        **named_extrema: Expressions to monitor, keyed by the name under which the extremum is reported, e.g. ``u_sqr=var("u")**2``.
    """
    def __init__(self,*direct_vars:str,**named_extrema:ExpressionOrNum):
        super().__init__()
        self.named_extrema=named_extrema.copy()
        for varname in direct_vars:
            if not isinstance(varname,str):
                raise ValueError("ExtremumObservables must be either constructed with strings as positional args or expressions as keyword args, i.e. e.g. ExtremumObservables('u') to monitor the extrema of var('u') or ExtremumObservables(u_sqr=var('u')**2) to monitor the extrema of u**2")
            self.named_extrema[varname]=var(varname)


    def add_extremum_function(self,name:str,expr:"ExpressionOrNum | Callable[[], ExpressionOrNum]"):
        from .. import _pyoomph_core as _pyoomph
        master = self._get_combined_element()
        cg = master._assert_codegen()
        # Equivalent to the original "not(isinstance(expr,int) or isinstance(expr,float) or
        # isinstance(expr,Expression)) and callable(expr)": int/float instances are never
        # callable, so the int/float isinstance checks were redundant once callable(expr) is
        # required. Written this way (callable-check first), pyright can narrow expr's type.
        if callable(expr) and not isinstance(expr,_pyoomph.Expression):
            expr=expr()
        if isinstance(expr,(int,float)): # Does not really make sense here
            expr=_pyoomph.Expression(expr)
        cg._register_extremum_function(name, Expression(expr))
        
    def define_residuals(self):
        for name,expr in self.named_extrema.items():
            self.add_extremum_function(name,expr)

class ODEObservables(ODEEquations):
    """
    Adds observables to ODEs. Observables are just expressions which will be also written to the output file when combined with an :py:class:`~pyoomph.output.generic.ODEFileOutput` object.
    If you have e.g. a harmonic oscillator (with variable ``y``) and want to monitor the total energy,
    add it as an observable::

        HarmonicOscillator(...) + ODEObservables(Etot=1/2*partial_t(y)**2 + 1/2*omega**2*y**2)

    Args:
        **ode_observables: Observables to be added, identified by the name. Can also be a callable
            taking no arguments and returning an ExpressionOrNum, evaluated lazily as a dependent observable.
    """

    def __init__(self, **ode_observables:ExpressionOrNum | Callable[..., ExpressionOrNum]):
        super(ODEObservables, self).__init__()
        is_callable=lambda v: callable(v) and not isinstance(v,Expression)
        self.ode_observables = {k:v for k,v in ode_observables.items() if not is_callable(v)}
        self.dependent_funcs={k:v for k,v in ode_observables.items() if is_callable(v)}

    def define_additional_functions(self):
        dx = nondim("dx")
        for k,v in self.ode_observables.items():
            # self.ode_observables was filtered (above) to only the non-callable entries,
            # but pyright cannot track that invariant through the dict comprehension.
            assert isinstance(v,(int,float,Expression))
            self.add_integral_function(k, v * dx)
        for k,v in self.dependent_funcs.items():
            assert callable(v)
            self.add_dependent_integral_function(k,v)



class Scaling(BaseEquations):
    """
    Set the scales used for nondimensionalization on the equation level. It will override the scales set on the problem level by Problem.set_scaling(...=...).

    Args:
        **kwargs: Scales to use for nondimensionalization, as ``field_name=scale`` pairs. A string value refers to another scale by name.
    """
    def __init__(self,**kwargs:ExpressionOrNum | str):
        super(Scaling, self).__init__()
        self.scales=sorted_field_kwargs(kwargs)
    def define_scaling(self):
        super(Scaling, self).define_scaling()
        self.set_scaling(self.scales)

class TestScaling(BaseEquations):    
    """
    Set the scales of the test functions used for nondimensionalization on the equation level.

    Args:
        **kwargs: Test function scales, as ``field_name=scale`` pairs. A string value refers to another scale by name.
    """    
    __test__ = False
    def __init__(self,**kwargs:ExpressionOrNum | str):
        super(TestScaling, self).__init__()
        self.scales=sorted_field_kwargs(kwargs)

    def define_scaling(self):
        super(TestScaling, self).define_scaling()
        self.set_test_scaling(self.scales)



class ElementSpace(Equations):
    """
    Sets the element space of the current equations. By default, pyoomph will take the highest order element space of all fields defined on the domain.
    With this class, you can e.g. set the element space to second order ("C2"), although you only have first-order fields ("C1") defined.

    Args:
        space (FiniteElementSpaceEnum): Set the desired element space for the equations of the domain.
    """
    def __init__(self,space:FiniteElementSpaceEnum):
        super(ElementSpace, self).__init__()
        self.space:FiniteElementSpaceEnum=space

    def define_fields(self):
        cg=self.get_current_code_generator()
        if self.space not in {"C2TB","C2","C1TB","C1"}:
            raise ValueError("Can only set the coordinate space to either C2TB, C2, C1TB or C1")
        # cg._coordinate_space (a plain str C++ property) can legitimately be "" (not yet
        # set) here; find_dominant_element_space() explicitly handles that empty-string
        # sentinel. See the identical cast in generic/codegen.py's _internal_define_scalar_field.
        cg._coordinate_space = find_dominant_element_space(cast("FiniteElementSpaceEnum",cg._coordinate_space),self.space)




# Constaints of the form: integral(u+A)*dx=B
# Where A is given by get_integral_contribution
# and B is given by get_global_residual_contribution
# Used for Average and Integral constraints
# If physical dimensions are set, it only works if the these are set on problem level by problem.set_scaling(...=...), not if set in the equations
class _AverageOrIntegralConstraintBase(Equations):
    def __init__(self,*,ode_storage_domain:str | None=None,only_for_stationary_solve:bool=False,set_zero_on_normal_mode_eigensolve:bool=True,scaling_factor:str | ExpressionNumOrNone=None, **kwargs:"ExpressionOrNum"):
        super().__init__()
        self.ode_storage_domain=ode_storage_domain        
        self.constraints=kwargs.copy()
        self.dimensional_dx=False
        self.only_for_stationary_solve=only_for_stationary_solve
        self.set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve
        self.scaling_factor:ExpressionNumOrNone=scale_factor(scaling_factor) if isinstance(scaling_factor,str) else scaling_factor

    def get_global_dof_storage_name(self, pathname: str | None = None):
        if self.ode_storage_domain is None:
            return super().get_global_dof_storage_name(pathname)
        else:
            return self.ode_storage_domain
        
    def after_fill_dummy_equations(self, problem: "Problem", eqtree: "EquationTree",pathname:str,elem_dim:int | None=None):
        if len(self.constraints)==0:
            return super().after_fill_dummy_equations(problem,eqtree,pathname,elem_dim)        
        odestorage=self.get_global_dof_storage_name(pathname=pathname)  
        add_eqs=None      
        for field,integral_value in self.constraints.items():
            scale_correction=problem.get_scaling(field) if self.scaling_factor is None else self.scaling_factor
            testscale:ExpressionOrNum=1
            if self.dimensional_dx:
                if elem_dim is None:
                    elem_dim=self.get_element_dimension()
                
                codegen=eqtree._codegen
                assert codegen is not None
                coordsys=codegen.get_coordinate_system()
                #coordsys=self.get_combined_equations().get_coordinate_system()
                testscale/=(0+coordsys.volumetric_scaling(problem.get_scaling("spatial"),elem_dim))


            new_eq=GlobalLagrangeMultiplier(only_for_stationary_solve=self.only_for_stationary_solve,set_zero_on_normal_mode_eigensolve=self.set_zero_on_normal_mode_eigensolve, **{field:self.get_global_residual_contribution(field)/scale_correction})+Scaling(**{field:1})+TestScaling(**{field:testscale})
            add_eqs=new_eq if add_eqs is None else add_eqs+new_eq

        # self.constraints is non-empty here (checked above), so the loop ran at least
        # once and add_eqs was assigned.
        assert add_eqs is not None
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
    If you have e.g. a field ``"u"``, you can enforce the integral of ``"u"`` to be 1 by adding::

        IntegralConstraint(u=1)

    Args:
        dimensional_dx: Whether the integration measure carries its physical dimension. Defaults to True.
        ode_storage_domain: The storage domain for the Lagrange multipliers. Defaults to a generated name.
        only_for_stationary_solve: Apply the constraint during stationary solves only.
        set_zero_on_normal_mode_eigensolve: Force the multiplier to zero in a normal mode eigensolve.
        scaling_factor: Scale of the Lagrange multiplier, either an expression or the name of a scale.
        **kwargs: The constraints as ``field_name=value`` pairs.
    """       
    def __init__(self, *, dimensional_dx:bool=True,ode_storage_domain: str | None = None, only_for_stationary_solve: bool = False, set_zero_on_normal_mode_eigensolve: bool = True, scaling_factor:str | ExpressionNumOrNone=None, **kwargs: ExpressionOrNum):
        super().__init__(ode_storage_domain=ode_storage_domain, only_for_stationary_solve=only_for_stationary_solve, set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve, scaling_factor=scaling_factor, **kwargs)
        self.dimensional_dx=dimensional_dx
    
    def get_global_residual_contribution(self,field:str) -> ExpressionOrNum:
        return -self.constraints[field] # Globally subtract the integral value
    
    def get_integral_contribution(self,field:str)-> ExpressionOrNum:
        return 0 # No contribution during the spatial integral

class AverageConstraint(_AverageOrIntegralConstraintBase):
    r"""
    Enforces the value of a field to have a fixed averaged value by a global Lagrange multiplier.
    If you have e.g. a field ``"u"``, you can enforce the average of ``"u"`` to be 1 by adding::

        AverageConstraint(u=1)

    Unlike :py:class:`IntegralConstraint`, there is no ``dimensional_dx``: the constraint is
    :math:`\int (u-\text{target})\,\mathrm{d}x=0`, whose root a constant factor on the measure does
    not move.

    Args:
        ode_storage_domain: The storage domain for the Lagrange multipliers. Defaults to a generated name.
        only_for_stationary_solve: Apply the constraint during stationary solves only.
        set_zero_on_normal_mode_eigensolve: Force the multiplier to zero in a normal mode eigensolve.
        scaling_factor: Scale of the Lagrange multiplier, either an expression or the name of a scale.
        **kwargs: The constraints as ``field_name=value`` pairs.
    """           
    def get_global_residual_contribution(self,field:str)-> ExpressionOrNum:
        return 0 # No global contribution
    
    def get_integral_contribution(self,field:str)-> ExpressionOrNum:
        return self.constraints[field] # Consider the offset for the average



##########################################################################################
# Generic equation classes. These are not tied to any particular physics, but they are
# equations the user instantiates - unlike the base classes (Equations, InterfaceEquations,
# ODEEquations, ...) and the code generation machinery, which stay in pyoomph.generic.codegen.
##########################################################################################

class WeakContribution(BaseEquations):
    """
    A class to add an arbitrary weak contribution to the equations. This is useful to add additional terms to the equations that are not covered by the standard weak formulation. Essentially, it just adds ``weak(a,b)`` to the residuals.

    Args:
        a: The first argument of the :py:func:`~pyoomph.expressions.generic.weak` contribution. A
            string is taken as a field name, i.e. ``"u"`` means ``var("u")``.
        b: The second argument, usually a :py:func:`~pyoomph.expressions.generic.testfunction`. A
            string is taken as a field name, i.e. ``"u"`` means ``testfunction("u")``.
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
        self.b:Expression=testfunction(b) if isinstance(b,str) else b
        self.a:ExpressionOrNum=var(a) if isinstance(a,str) else a

    def define_residuals(self):
        self.add_residual(weak(self.a,self.b,dimensional_dx=self.dimensional_dx,lagrangian=self.lagrangian,coordinate_system=self.coordinate_system),destination=self.destination)


class ScalarField(Equations):
    """
    Introduces a scalar field with the given name and the given space. Residuals can be either added in the constructor or by combining with :py:class:`~pyoomph.equations.generic.WeakContribution`.
    
    Args:
        name: Name of the scalar field
        space: Space of the scalar field
        scale: Optional scaling of the field. Defaults to None, i.e. the scale registered for ``name``.
        testscale: Optional scaling of the test function. Defaults to None, i.e. the test scale registered for ``name``.
        residual: Optional residual to be added. Formulate it in terms of the scalar field and its test function.
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
    Introduces a vector field with the given name and the given space. Residuals can be either added in the constructor or by combining with :py:class:`~pyoomph.equations.generic.WeakContribution`.
    
    Args:
        name: Name of the vector field
        space: Space of the vector field
        scale: Optional scaling of the field. Defaults to None, i.e. the scale registered for ``name``.
        testscale: Optional scaling of the test function. Defaults to None, i.e. the test scale registered for ``name``.
        residual: Optional residual to be added. Formulate it in terms of the vector field and its test function.
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


class GlobalLagrangeMultiplier(ODEEquations):
    """
    Defines global degrees of freedom, typically Lagrange multipliers enforcing a global constraint.
    It is just an :py:class:`~pyoomph.generic.codegen.ODEEquations` with a few extras, e.g. it can be
    deactivated on transient solves. Each entry adds one global unknown together with the residual
    contribution given as its value::

        problem += (GlobalLagrangeMultiplier(volume_lagr=-V0) + Scaling(volume_lagr=1*pascal)) @ "globals"

    The remaining part of the constraint - here the integral over the domain - is usually added with a
    :py:class:`WeakContribution` projected onto ``testfunction("volume_lagr", domain="globals")``.

    Args:
        *args: Names of multipliers whose residual contribution is added elsewhere.
        only_for_stationary_solve: Pin the multipliers to zero during transient solves, so the
            constraint acts on stationary solves only.
        set_zero_on_normal_mode_eigensolve: Force them to zero in a normal mode eigensolve, where a
            global multiplier is usually meaningless. Defaults to True.
        **kwargs: Multiplier name with the residual contribution added to its test function.
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

    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver", angular_mode:int | float | None,normal_k:float | None)->set[str | int]:
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
                return set(lst)
        # angular_mode is None and normal_k is None is already handled above; this is unreachable
        # but kept so the function has an explicit return on every static code path.
        return set()

    def get_information_string(self) -> str:
        return ", ".join([str(n) + " with contrib. " + str(v) for n, v in self._entries.items()])


class EnforcedBC(InterfaceEquations):
    """
    Enforce rather arbitrary boundary conditions by a field of Lagrange multipliers, e.g.::

        EnforcedBC(u=var("u")-var("v")) @ "boundary"

    will set ``u=v`` on the boundary by adjusting ``u``. The value must be the constraint in residual
    form, i.e. the expression to drive to zero - here ``u-v``.

    Args:
        only_for_stationary_solve: Apply the conditions during stationary solves only. Defaults to False.
        set_zero_on_normal_mode_eigensolve: Force the multipliers to zero in an azimuthal eigensolve.
            Defaults to False.
        domain: Domain of the adjusted field, if it is not the one this condition is added to.
        space: Space of the Lagrange multiplier field. Defaults to the space of the adjusted field.
        coordinate_system: Coordinate system of the constraint integral. Defaults to the one of the
            equations or the problem.
        **constraints: Name of the field to adjust, with the constraint expression in residual form.
    """
 
    def __init__(self,*, only_for_stationary_solve:bool=False, set_zero_on_normal_mode_eigensolve=False,domain:str | None=None,space:FiniteElementSpaceEnum | None=None,coordinate_system:BaseCoordinateSystem | None=None,**constraints:Expression):
        super(EnforcedBC, self).__init__()
        self.constraints = constraints.copy()
        self.lagrangian:bool = False
        self.only_for_stationary_solve=only_for_stationary_solve
        self.set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve
        self.domain=domain
        self.space=space
        self.coordsys=coordinate_system

    def get_lagrange_multiplier_name(self, varname:str)->str:
        return "_lagr_enf_bc_" + varname

    def define_fields(self):
        allowed_spaces= {"C1","C1TB","C2","C2TB","D1","D1TB","D2","D2TB"}
        for k, _ in self.constraints.items():
            if self.space is not None:
                    sp = self.space                            
            else:
                if self.domain is not None:
                    raise RuntimeError("Please specify the FEM space of the Lagrange multipliers when using domain")
            
                sp = self.get_parent_domain().get_space_of_field(k)
                if sp == "":
                    ppdom=self.get_parent_domain().get_parent_domain()
                    if ppdom is not None:
                        sp = ppdom.get_space_of_field(k)
                        if sp == "":
                            # Test if it is a vector field
                            # print(dir(self.get_parent_domain().get_equations()))
                            # expanded = self.expand_additional_field(k, True, 0, self.get_current_code_generator(),False,False)
                            # peqs = self.get_parent_domain().get_equations()
                            raise RuntimeError("Cannot use EnforcedBC on an unknown field " + k)
                if sp not in allowed_spaces:
                    if sp == "Pos":
                        sp = self.get_current_code_generator()._coordinate_space
                    else:
                        raise RuntimeError("EnforcedBC only works the following bulk spaces:"+", ".join(allowed_spaces)+". problem for field " + k + " on space " + sp)
            self.define_scalar_field(self.get_lagrange_multiplier_name(k), cast(FiniteElementSpaceEnum, sp), scale=1 / test_scale_factor(k,domain=self.domain),testscale=1 / scale_factor(k,domain=self.domain))
        
        aziinfo=self.get_azimuthal_r0_info()
        for k in self.constraints.keys():
            ln=self.get_lagrange_multiplier_name(k)
            for i in [0,1,2]:
                if k in aziinfo[i]:
                    aziinfo[i].add(ln)
                else:
                    if ln in aziinfo[i]:
                        aziinfo[i].remove(ln)


    def define_residuals(self):
        for k, v in self.constraints.items():
            lagr_name=self.get_lagrange_multiplier_name(k)
            l, ltest = var_and_test(lagr_name)  # get the Lagrange multiplier
            utest = testfunction(k,domain=self.domain)
            self.add_residual(weak(v, ltest, lagrangian=self.lagrangian,coordinate_system=self.coordsys))  # Enforce the constraint
            self.add_residual(weak(l, utest,lagrangian=self.lagrangian,coordinate_system=self.coordsys))  # Lagrange multiplier pair to enforce it
            if self.only_for_stationary_solve:
                self.set_Dirichlet_condition(lagr_name,0)

    def before_assigning_equations_postorder(self, mesh:"AnyMesh"):
        # Pin redundant Lagrange multipliers
        assert isinstance(mesh,InterfaceMesh)
        assert mesh._eqtree._parent is not None #type: ignore
        bulkmesh = mesh._eqtree._parent._mesh #type: ignore
        assert bulkmesh is not None
        codeinst_inside=mesh.get_code_gen().get_code()
        #codeinst_inside = mesh.element_pt(0).get_code_instance()
        for k, _ in self.constraints.items():            
            index = [codeinst_inside.get_nodal_field_index(k)]  # TODO: Vectors
            #print("Index is ",index," for field ",k)
            psindex = None
            nfi=None
            spaceDG=None
            if any(i < 0 for i in index):
                if k == "mesh_x":
                    psindex = 0
                elif k == "mesh_y":
                    psindex = 1
                elif k == "mesh_z":
                    psindex = 2
                elif mesh.has_interface_dof_id(k)>=0:
                    nfi=mesh.has_interface_dof_id(k)
                else:
                    spc=mesh._eqtree.get_code_gen().get_space_of_field(k)
                    if spc in {"D2TB","D2","D1"}:
                        spaceDG=spc
                    else:
                        raise RuntimeError("Cannot find a nodal index for field " + k+". Defined on space: "+spc)
            lname = self.get_lagrange_multiplier_name(k)
            if spaceDG is None:
                interfid = bulkmesh.has_interface_dof_id(lname)
                if interfid < 0:
                    raise RuntimeError(
                        f"Something strange here. We have the bulk mesh '{bulkmesh.get_name()}' and it does not have the interface id '{lname}'")  #
                for n in mesh.nodes():
                    if psindex is not None:
                        if n.variable_position_pt().is_pinned(psindex):
                            lind = n.additional_value_index(interfid)
                            n.pin(lind)
                            n.set_value(lind, 0)
                    elif nfi is not None:
                        nfind = n.additional_value_index(nfi)
                        if n.is_pinned(nfind):
                            lind = n.additional_value_index(interfid)
                            n.pin(lind)
                            n.set_value(lind, 0)
                    elif all(n.is_pinned(i) for i in index):
                        lind = n.additional_value_index(interfid)
                        n.pin(lind)
                        n.set_value(lind, 0)
            else:
                for e in mesh.elements():
                    dg_data=e.get_field_data_list(k,False)
                    l_data=e.get_field_data_list(lname,False)
                    for dg,l in zip(dg_data,l_data):
                        if dg[0].is_pinned(dg[1]):
                            l[0].pin(l[1])
                            l[0].set_value(l[1],0)

    def after_compilation(self,codegen:"FiniteElementCodeGenerator"):
        super().after_compilation(codegen)
        assert codegen._mesh is not None 
        if self.only_for_stationary_solve:
            for k, _ in self.constraints.items():
                # Do not activate by default to allow for initial conditions                
                lagr_name=self.get_lagrange_multiplier_name(k)
                codegen._mesh._set_dirichlet_active(lagr_name,False)                      

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
        for k in self.constraints.keys():
            lagr_name=self.get_lagrange_multiplier_name(k)
            if self.only_for_stationary_solve:
                if mesh._get_dirichlet_active(lagr_name) == stationary: 
                    mesh._set_dirichlet_active(lagr_name, not stationary)
                    must_reapply = True
            else:
                if mesh._get_dirichlet_active(lagr_name)==True: 
                    mesh._set_dirichlet_active(lagr_name,False) 
                    must_reapply=True
        return must_reapply
    
    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver", angular_mode:int | float | None,normal_k:float | None)->set[str | int]:
        if (not self.set_zero_on_normal_mode_eigensolve) or (angular_mode is None and normal_k is None):
            return cast("set[str | int]",set())
        else:
            if angular_mode is not None and normal_k is not None:
                raise RuntimeError("Cannot have both angular and normal mode set")
            if angular_mode is not None:
                mode:int | float=int(angular_mode)
            elif normal_k is not None:
                mode=normal_k
            else:
                raise RuntimeError("Neither angular_mode nor normal_k is set")
            fullpath = eqtree.get_full_path().lstrip("/")
            if mode == 0:
                return cast("set[str | int]",set())
            else:
                for_my_m = [self.get_lagrange_multiplier_name(k) for k in self.constraints.keys()]
            lst=[fullpath + "/" + k for k in for_my_m]
            res:set[str | int] = set(lst)
            return res


class EnforcedDirichlet(EnforcedBC):
    """
    Enforces a Dirichlet condition by Lagrange multipliers instead of by pinning, i.e. the value stays
    a degree of freedom. Unlike :py:class:`DirichletBC`, the prescribed value may therefore depend on
    unknowns. The two lines::

        EnforcedDirichlet(u=var("v")) @ "boundary"
        EnforcedBC(u=var("u")-var("v")) @ "boundary"

    are equivalent - this class only subtracts the current value for you.

    Args:
        only_for_stationary_solve: Apply the conditions during stationary solves only. Defaults to False.
        set_zero_on_normal_mode_eigensolve: Force the multipliers to zero in an azimuthal eigensolve.
            Defaults to False.
        domain: Domain of the adjusted field, if it is not the one this condition is added to.
        space: Space of the Lagrange multiplier field. Defaults to the space of the adjusted field.
        coordinate_system: Coordinate system of the constraint integral. Defaults to the one of the
            equations or the problem.
        **constraints: Name of the field to adjust, with the value it is enforced to.
    """
    
    def __init__(self,*, only_for_stationary_solve:bool=False, set_zero_on_normal_mode_eigensolve=False,domain:str | None=None,space:FiniteElementSpaceEnum | None=None,coordinate_system:BaseCoordinateSystem | None=None, **constraints:Expression):
        from ..expressions import var        
        new_kwargs={k:var(k,domain=domain)-v for k,v in constraints.items()}
        super(EnforcedDirichlet, self).__init__(only_for_stationary_solve=only_for_stationary_solve, set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve,coordinate_system=coordinate_system ,**new_kwargs.copy(),domain=domain,space=space)


class ForceZeroOnEigenSolve(BaseEquations):
    """
    Forces the given degrees of freedom to zero in the eigenproblem only, i.e. the base solution keeps
    them. Required whenever a dof must not take part in a perturbation, e.g. a Lagrange multiplier that
    is meaningless for a nonzero azimuthal mode.

    Args:
        default: Names of the fields to zero out.
        for_nonzero_angular: Names to use instead when the azimuthal mode is not zero. Defaults to
            ``default``.
    """
    def __init__(self,default:Iterable[str],*,for_nonzero_angular:Iterable[str] | None=None):
        super(ForceZeroOnEigenSolve, self).__init__()
        self.default=default
        self.for_nonzero_angular=for_nonzero_angular

    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree:EquationTree,eigensolver:"GenericEigenSolver", angular_mode:int | float | None,normal_k:float | None)->set[str | int]:
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
    """
    Constrains a higher order field to the first order C1 space. Useful in combination with either the where parameter or (un)constrain a field e.g. a boundary

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
                    # NB: the binding signature is (mode, index) - see src/nanobind/mesh.cpp and the
                    # matching call in ConstrainPositionsToC1Space. Passing them the other way round
                    # silently mislabels the constraint (it only happened to work for a bulk field at
                    # value index 0, where mode==index==0, and did nothing for interface fields).
                    if self._unconstrain_instead:
                        n.remove_additional_dof_constraint(mode, index)
                    else:
                        n.set_additional_dof_constraint(mode, index)
                    
        return super().before_assigning_equations_preorder(mesh)


class UnconstrainFieldsFromC1Space(ConstrainFieldsToC1Space):
    """
    Unconstrains a higher order field from the first order C1 space. Useful in combination with either the where parameter or (un)constrain a field e.g. a boundary

    Args:
        *args: The names of the fields to unconstrain from the C1 space.
        where: An optional function to specify where the unconstraining should be applied. Nondimensional nodal positions are passed to this function, and it should return True for nodes where the constraint should be applied. If None, the constraint is applied to all nodes.
    """
    def __init__(self, *args:str,where:Callable[[list[float]], bool] | None=None):
        super().__init__(*args,unconstrain_instead=True,where=where)


##########################################################################################
# Boundary and nodal conditions (formerly pyoomph.meshes.bcs - they are equations, not
# mesh infrastructure).
##########################################################################################

class NeumannBC(InterfaceEquations):
    """
    Imposes a Neumann boundary condition. What the flux means depends on the bulk equations, i.e. on how
    the integration by parts was performed for their weak form. For a Poisson equation implemented by
    the residual ``weak(grad(u),grad(utest))``, the condition::

        NeumannBC(u=1) @ "boundary"

    does not mean setting ``u=1``, but rather ``dot(grad(u),var("normal"))=-1``, with the normal vector
    pointing out of the domain at the boundary.

    Args:
        **fluxes: The imposed fluxes as ``field_name=flux`` pairs.
    """

    def __init__(self, **fluxes:ExpressionOrNum):
        super(NeumannBC, self).__init__()
        self.fluxes = sorted_field_kwargs(fluxes)

    def define_residuals(self):
        for name, flux in self.fluxes.items():
            test = testfunction(name)
            self.add_residual(weak(flux, test))


class DirichletBC(BaseEquations):
    """
    Class to impose one or more Dirichlet boundary condition.

    Args:
        prefer_weak_for_DG (bool, optional): Flag indicating whether to prefer weak contributions for Discontinuous Galerkin (DG) methods. If set and the bulk equations provide a specific implementation of :py:meth:`~pyoomph.generic.codegen.Equations.get_weak_dirichlet_terms_for_DG`, these terms are used to enforce the condition in a weak sense. Otherwise strongly. Defaults to True.
        **kwargs (ExpressionOrNum): Keyword arguments representing the Dirichlet conditions, where the keys are the variable names and the values are the corresponding expressions or numbers. Expressions for strong DirichletBCs may not depend on unknowns.

    A vector field may be given as a whole, i.e. ``DirichletBC(velocity=vector(0,1))`` instead of
    ``DirichletBC(velocity_x=0, velocity_y=1)``. The value is split component by component onto the
    field's components in their own order, so this is also correct in e.g. an axisymmetric coordinate
    system. This includes the position fields and values that are not written as an explicit
    ``vector(...)``, i.e. ``DirichletBC(mesh=var("lagrangian"))`` to hold the mesh at its undeformed
    position. See :py:meth:`~pyoomph.generic.codegen.BaseEquations.expand_vectorial_entries`.
    """

    def __init__(self, *, prefer_weak_for_DG: bool = True, **kwargs: ExpressionOrNum):
        super(DirichletBC, self).__init__()
        self._dcs: dict[str, ExpressionOrNum] = sorted_field_kwargs(kwargs)
        self.prefer_weak_for_DG = prefer_weak_for_DG

    def _expanded_dcs(self) -> dict[str, ExpressionOrNum]:
        # Vector-valued entries split into their components. Not done in __init__: which components
        # "velocity" has is a property of the domain this condition ends up on (and, for an interface
        # condition, of its parent), which is not known until the equations are attached.
        return self.expand_vectorial_entries(self._dcs, "Dirichlet condition")

    def define_residuals(self):
        pdom = self.get_parent_domain()
        peqs = pdom.get_equations() if pdom is not None else None
        if not isinstance(peqs, Equations):
            peqs = None
        for n, val in self._expanded_dcs().items():
            if self.prefer_weak_for_DG and (peqs is not None) and (val is not True):
                # Check if some equation is defining weak contributions instead
                weak_DBC = peqs.get_weak_dirichlet_terms_for_DG(n, val)
                if weak_DBC is not None:
                    self.add_residual(weak_DBC)  # Add the weak Dirichlet
                    continue
            # Otherwise, strong Dirichlet
            self.set_Dirichlet_condition(n, val)

    def get_information_string(self) -> str:
        return ", ".join([str(n) + "=" + str(v) for n, v in self._dcs.items()])


class AxisymmetryBC(InterfaceEquations):
    r"""
    Add this to the axis of symmetry to automatically enforce the boundary condition required by symmetry.
    Also automatically sets the correct boundary conditions for azimuthal eigenvalue problems.

    For normal solving, it sets radial (and azimuthal) components of vector fields (also ``mesh_x``) to
    zero. For azimuthal eigenvalue problems, it depends on the azimuthal mode :math:`m`:

    * :math:`m=0`: as for normal solving.
    * :math:`|m|=1`: scalar fields and axial vector components are set to zero, radial and azimuthal
      components are not.
    * :math:`|m|\geq 2`: scalar fields and all vector components are set to zero.

    If you write an equation, where you want to change this behavior, you can manually change the conditions by obtaining the (writeable) field information via :py:meth:`~pyoomph.generic.codegen.Equations.get_azimuthal_r0_info` after the definition via :py:meth:`~pyoomph.generic.codegen.Equations.define_scalar_field` or :py:meth:`~pyoomph.generic.codegen.Equations.define_vector_field`.

    Must also be added to intersections of other boundaries with the axis of symmetry when those
    boundaries define additional fields::

        bulk = NavierStokesEquations(...)
        bulk += AxisymmetryBC() @ "axis"
        bulk += NavierStokesFreeSurface() @ "interface"
        bulk += AxisymmetryBC() @ "interface/axis"  # the free surface introduces new fields here

    This is, however, done automatically as long as ``recurse`` is set.

    Args:
        verbose: Report which fields are pinned on the axis.
        recurse: Also apply the condition on the intersections of this axis with other interfaces.
    """
    def __init__(self,verbose:bool=True,recurse:bool=True):
        super().__init__()
        self.verbose=verbose
        self.recurse=recurse
        
    def _fill_interinter_connections(self, eqtree:"EquationTree", interinter):
        if self.recurse:
            from ..generic.codegen import EquationTree
            # Now find the reversed connections. We get e.g. domain/axis/interface, but we must add it to domain/interface/axis
            revconns=list()
            eqtree_parent=eqtree.get_parent()
            assert eqtree_parent is not None
            trunk=eqtree_parent.get_full_path().lstrip("/")
            myname=eqtree.get_my_path_name()
            for conn in interinter:
                rest=conn[len(eqtree.get_full_path().lstrip("/")):].lstrip("/")
                path=trunk+"/"+rest+"/"+myname
                revconns.append(path)
            revconns.sort(key=lambda x: x.count("/")) # Sort by number of slashes to get it in good order
            root=eqtree
            while True:
                root_parent=root.get_parent()
                if root_parent is None:
                    break
                root=root_parent
            for rc in revconns:
                splt=rc.split("/")
                dom=root
                is_present=True
                for s in splt[:-1]:
                    if s in dom._children:
                        dom=dom.get_child(s)
                    else:
                        is_present=False
                        break
                if not is_present:
                    continue # Nothing to be done. There is no interface added
                if splt[-1] in dom._children:
                    iface=dom.get_child(splt[-1])
                    if iface.get_equations() is not None:
                        axieq_list=iface.get_equations().get_equation_of_type(AxisymmetryBC,always_as_list=True)
                        if len(axieq_list)>0:
                            continue # Already added
                        else:
                            assert iface._equations is not None
                            oldeqs=iface._equations
                            iface._equations=iface._equations+AxisymmetryBC(verbose=self.verbose,recurse=self.recurse)
                            iface._equations._problem=oldeqs._problem
                else:
                    new_child=EquationTree(AxisymmetryBC(verbose=self.verbose,recurse=self.recurse),dom)
                    dom._children[splt[-1]]=new_child
                    assert new_child._equations is not None
                    assert dom._equations is not None
                    new_child._equations._problem=dom._equations._problem
                    
            
        return super()._fill_interinter_connections(eqtree, interinter)
                                
    
    def define_residuals(self):
        if self.verbose:
            print("AxisymmetryBC: Setting zero DirichletBCs at",self.get_current_code_generator().get_full_name(),"for",self.get_azimuthal_r0_info()[0])
        for k in self.get_azimuthal_r0_info()[0]:            
            self.set_Dirichlet_condition(k,0)
                
                
    def _before_stationary_or_transient_solve(self, eqtree:"EquationTree", stationary:bool)->bool:
        must_reapply=False
        #if self.get_mesh().get_problem().get_bifurcation_tracking_mode() == "azimuthal": 
        from ..generic.bifurcation_tools import _NormalModeBifurcationTrackerBase
        pr=self.get_mesh().get_problem()
        if pr.get_bifurcation_tracking_mode() == "azimuthal" or (pr.get_custom_assembler() is not None and isinstance(pr.get_custom_assembler(),_NormalModeBifurcationTrackerBase)): 
            return False  # Don't do anything in this case. It would mess up everything!
        
        mesh=eqtree._mesh
        assert mesh is not None        
        activated_bcs=set()
        for k in self.get_azimuthal_r0_info()[0]:
            if mesh._get_dirichlet_active(k) == False: 
                activated_bcs.add(k)                
                mesh._set_dirichlet_active(k, True)
                must_reapply = True 
        if len(activated_bcs)>0 and self.verbose:
            print("AxisymmetryBC: Activating zero DirichletBCs at",self.get_current_code_generator().get_full_name(),"for",activated_bcs)
        return must_reapply
    
           
    def _before_eigen_solve(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver",angular_m:float | None=None,normal_k:float | None=None) -> bool:
        if angular_m is None or angular_m==0:
            return False
        must_reapply = False        
        assert eqtree._mesh is not None 
        deactivated_bcs=set()
        for k in self.get_azimuthal_r0_info()[0]:            
            if eqtree._mesh._get_dirichlet_active(k):
                deactivated_bcs.add(k)
                eqtree._mesh._set_dirichlet_active(k, False) 
                must_reapply = True 
        if len(deactivated_bcs)>0 and self.verbose:            
            print("AxisymmetryBC: Deactivating strong zero DirichletBCs at",self.get_current_code_generator().get_full_name(),"for",deactivated_bcs)
        return must_reapply    
                        
    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver", angular_mode:int | float | None,normal_k:float | None)->set[str | int]:
        if angular_mode is None:
            return set()

        angular_mode=int(angular_mode)

        # Note: angular_mode==0, abs(angular_mode)==1 and abs(angular_mode)>1 are exhaustive for any int,
        # so info is always assigned below (written as if/elif/else so that this is also clear to the type checker).
        if angular_mode==0:
            info=self.get_azimuthal_r0_info()[0]
        elif abs(angular_mode)==1:
            info=self.get_azimuthal_r0_info()[1]
        else:
            info=self.get_azimuthal_r0_info()[2]
        res:set[str | int]=set([eqtree.get_full_path().lstrip("/")+"/"+m for m in info])
        if len(info)>0 and self.verbose:
            print("AxisymmetryBC (mode m="+str(angular_mode)+"): Imposed zero by matrix manipulation at",self.get_current_code_generator().get_full_name(),"for",info)
        return res
            



# Scalar fields on DG space => set the corresponding eigenfunctions


class PeriodicBC(InterfaceEquations):
    """
    Introduces a periodic boundary condition between two interfaces. It will hold for all continuous fields!
    The mesh must be generated that way that for each node on this interface, there is a corresponding node on the other interface when adding offset to the position.

    Args:
        other_interface: The name of the other interface to which this boundary is periodic.
        offset: Added to the position of a node here to find its counterpart on ``other_interface``.

    """

    def __init__(self, other_interface: str, offset: list[ExpressionOrNum] | None = None):
        super(PeriodicBC, self).__init__()
        self.other_interface = other_interface        
        self.offset:list[ExpressionOrNum]
        if offset is None:
            raise RuntimeError("Please supply an offset")
        elif not isinstance(offset,(list,tuple)):
            self.offset=[offset]
        else:
            self.offset = list(offset)

    def before_finalization(self, codegen: "FiniteElementCodeGenerator"):
       
        bulkdom = self.get_parent_domain()
        if bulkdom.get_nodal_dimension()!=len(self.offset):
            raise RuntimeError("The offset of the PeriodicBC must have the same dimension as the nodal dimension of the mesh")
        while bulkdom.get_parent_domain() is not None:
            raise RuntimeError("Cannot yet apply periodic boundaries on interfaces on interfaces")
        pmesh = bulkdom.get_equations().get_mesh()
        assert isinstance(pmesh, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        bnames = pmesh.get_boundary_names()
        my_name = self.get_mesh().get_name()
        ss = self.get_scaling("spatial")
        offs = [float(o / ss) for o in self.offset]
        if my_name not in bnames:
            raise RuntimeError("Cannot find boundary '" + my_name + "' in bulk mesh")
        if self.other_interface not in bnames:
            raise RuntimeError("Cannot find boundary '" + self.other_interface + "' in bulk mesh")
        my_nodes_by_pos: dict[tuple[float, ...], _pyoomph.Node] = {}
        for n in pmesh.boundary_nodes(my_name):
            ps = [n.x(i) + offs[i] for i in range(n.ndim())]
            my_nodes_by_pos[tuple(ps)] = n

        dataG: list[list[float]] = []
        master_nodes: list[_pyoomph.Node] = []
        for n in pmesh.boundary_nodes(self.other_interface):
            ps = [n.x(i) for i in range(n.ndim())]
            dataG.append(ps)
            master_nodes.append(n)
        data = numpy.array(dataG)  # type:ignore
        if len(data) != len(my_nodes_by_pos):  # type:ignore
            raise RuntimeError("Mismatch in number of nodes for a periodic boundary")
        kdtree = scipy.spatial.KDTree(data)  # type:ignore

        slave_to_master:dict[_pyoomph.Node,_pyoomph.Node]=dict()
        master_to_slave:dict[_pyoomph.Node,_pyoomph.Node]=dict()
        for pos, nslave in my_nodes_by_pos.items():
            qres = kdtree.query(pos)  # type:ignore
            if qres[0] > 1e-6:  # type:ignore
                raise RuntimeError("Cannot find a periodic node at the position " + str(pos))
            nmaster: Node = master_nodes[qres[1]]  # type:ignore
            if len(nmaster.get_boundary_indices()) >= 2:
                if not len(nslave.get_boundary_indices()) >= 2:
                    raise RuntimeError(
                        "A periodic node on a single boundary shall be copied from a master that lies on more than one boundary")
                pmesh._periodic_corner_node_info[nslave] = nmaster  # type:ignore
                slave_to_master[nslave]=nmaster
                master_to_slave[nmaster]=nslave
                continue
            elif len(nslave.get_boundary_indices()) >= 2:
                raise RuntimeError(
                    "A periodic corner node located on multiple boundaries shall be attached to a periodic master on a single boundary")
            slave_to_master[nslave]=nmaster
            master_to_slave[nmaster]=nslave
            nslave._make_periodic(nmaster, pmesh)
            
        # If we have a quad tree, we must also connect the quad tree here
        if pmesh.refinement_possible():
            myind=bnames.index(my_name)
            oppind=bnames.index(self.other_interface)
            oppnodes_to_oppelem:dict[tuple[_pyoomph.Node,...],tuple[_pyoomph.OomphGeneralisedElement,int]]=dict()
            for oppelem,direct in pmesh.boundary_elements(self.other_interface,with_directions=True):
                oppnodes_to_oppelem[tuple(oppelem.boundary_nodes(oppind))]=(oppelem,direct)
            
            for myelem,direct in pmesh.boundary_elements(my_name,with_directions=True):                
                my_nodes_on_bind=myelem.boundary_nodes(myind)                            
                search_for=tuple(slave_to_master[n] for n in my_nodes_on_bind)
                opp=oppnodes_to_oppelem.get(search_for,None)
                if opp is None:            
                    raise RuntimeError("Cannot identify the corresponding periodic boundary element on the other interface.")
                myelem._connect_periodic_tree(opp[0],direct,opp[1])


class PythonDirichletBC(Equations):
    """
    Pins (or unpins) degrees of freedom directly from Python whenever the boundary conditions are
    applied, i.e. for conditions that cannot be stated as an expression because they depend on the
    nodes themselves. Also the base class of :py:class:`PinWhere` and :py:class:`UnpinDofs`.

    Args:
        **kwargs: Fields to pin. The value is either the value to pin to, or ``True`` to pin each
            node at whatever value it currently has.
    """
    def __init__(self, **kwargs:ExpressionOrNum | Literal[True] | tuple[Callable[..., ExpressionOrNum], list[int], Expression]):
        super(PythonDirichletBC, self).__init__()
        # Merged in from the former BoundaryCondition base class, which had no other purpose left
        self.mesh:"AnySpatialMesh | None" = None
        self.active = True
        self.vals = sorted_field_kwargs(kwargs)
        self.unpin_instead:bool = False

    def _is_ode(self):
        return None

    def get_information_string(self) -> str:
        return ", ".join([str(k) + "=" + str(v) for k, v in self.vals.items()])

    def setup(self):
        assert self.mesh is not None
        self.indexvals:dict[int,float | Literal[True] | tuple[Callable[..., ExpressionOrNum], list[int], Expression]] = {}
        self.indexval_arginds = {}
        self.additional_vals:dict[int,float | Literal[True] | tuple[Callable[..., ExpressionOrNum], list[int], Expression]] = {}
        self.pinnedpositions:dict[int,float | Literal[True] | tuple[Callable[..., ExpressionOrNum], list[int], Expression]] = {}
        self.internal_vals:dict[int,float | Literal[True] | tuple[Callable[..., ExpressionOrNum], list[int], Expression]] = {}
        codeinst = self.mesh.element_pt(0).get_code_instance()

        currcodegen = self.get_current_code_generator()

        vals:dict[str,ExpressionOrNum | Literal[True] | tuple[Callable[..., ExpressionOrNum], list[int], Expression]] = {}

        for k, val in self.vals.items():
            if k == "mesh_x" or k == "mesh_y" or k == "mesh_z":
                vals[k] = val
                continue
            nodalfield = codeinst.get_nodal_field_index(k)
            if nodalfield < 0:
                internalfield = codeinst.get_discontinuous_field_index(k)
                if internalfield < 0:
                    interfid = self.mesh.has_interface_dof_id(k)
                    if interfid == -1:
                        # Last chance: is is an additional field, e.g a vector:
                        if k == "mesh":
                            dim = currcodegen.get_nodal_dimension()
                            if dim == 1:
                                vals[k + "_x"] = val
                                continue
                            elif dim == 2:
                                vals[k + "_x"] = val
                                vals[k + "_y"] = val
                                continue
                            elif dim == 3:
                                vals[k + "_x"] = val
                                vals[k + "_y"] = val
                                vals[k + "_z"] = val
                                continue
                        replaced = None
                        if k in currcodegen.get_equations()._additional_fields.keys():
                            replaced = self._additional_fields[k]
                        elif self.get_parent_domain() is not None:
                            bulk = self.get_parent_domain()
                            while bulk is not None:
                                bulkeq = bulk.get_equations()
                                if k in bulkeq._additional_fields_also_on_interface.keys():
                                    replaced = bulkeq._additional_fields_also_on_interface[k]
                                    break
                                bulk = bulk.get_parent_domain()
                            if replaced is not None:
                                if not isinstance(replaced,Expression):
                                    replaced=Expression(replaced)
                                if _pyoomph.GiNaC_is_a_matrix(replaced):
                                    for index in range(replaced.nops()):
                                        if not replaced[index].is_zero():
                                            print(replaced[index])
                                    raise RuntimeError("Cannot set a boundary condition by expanding yet")
                                else:
                                    raise RuntimeError("Cannot set a boundary condition for an unknown field " + str(k))
                            else:
                                raise RuntimeError("Cannot set a boundary condition for an unknown field " + str(k))
                    else:
                        vals[k] = val
                else:
                    vals[k] = val
            else:
                vals[k] = val

        assert self.mesh._codegen is not None 
        for k, val in vals.items():
            if k == "mesh_x" or k == "mesh_y" or k == "mesh_z":
                
                scal = self.mesh._codegen.get_scaling("spatial") 
                if val is True:
                    fval = True
                else:
                    assert not isinstance(val,tuple)
                    fval = float(val / scal)
                if k == "mesh_x":
                    self.pinnedpositions[0] = fval
                    continue
                elif k == "mesh_y":
                    self.pinnedpositions[1] = fval
                    continue
                elif k == "mesh_z":
                    self.pinnedpositions[2] = fval
                    continue
            nodalfield = codeinst.get_nodal_field_index(k)
            scal = self.mesh._codegen.get_scaling(k)  
            fval:float | bool | tuple[Callable[..., ExpressionOrNum], list[int], Expression]
            if not isinstance(val, bool) or val != True:
                try:
                    fval = float(val / scal) #type:ignore ##TODO Functions here
                except:
                    if callable(val):
                        arg_inds:list[int] = []
                        for a in inspect.signature(val).parameters:
                            if a == "mesh_x" or a == "coordinate_x":
                                arg_inds.append(-1)
                            elif a == "mesh_y" or a == "coordinate_y":
                                arg_inds.append(-2)
                            else:
                                raise RuntimeError("Lambda argument " + a + " not yet resolved")
                        fval = (val, arg_inds, scal)
                    else:
                        raise
            else:
                fval=True
            
            if nodalfield < 0:
                internalfield = codeinst.get_discontinuous_field_index(k)
                if internalfield < 0:
                    # Last chance: Get the index from an additional interface field
                    interfid = self.mesh.has_interface_dof_id(k)
                    if interfid == -1:
                        raise RuntimeError(
                            "Cannot find a nodal field, and elemental field or and additional interface field with name '" + k + "' to set a DirichletBC")
                    else:
                        self.additional_vals[interfid] = True if (isinstance(val, bool) and val == True) else fval
                else:
                    self.internal_vals[internalfield] = True if (isinstance(val, bool) and val == True) else fval

            else:
                self.indexvals[nodalfield] = True if (isinstance(val, bool) and val == True) else fval

    def apply(self):
        if not self.active:
            return
        assert self.mesh is not None
        for n in self.mesh.nodes():
            for i, val in self.indexvals.items():
                if self.unpin_instead:
                    n.unpin(i)
                else:
                    n.pin(i)
                    if not (isinstance(val, bool) and val == True):
                        if isinstance(val, tuple) and callable(val[0]):
                            arglst = [0.0] * len(val[1])
                            for j, ind in enumerate(val[1]):
                                if ind == -1:
                                    arglst[j] = n.x(0)
                                elif ind == -2:
                                    arglst[j] = n.x(1)
                                elif ind == -3:
                                    arglst[j] = n.x(2)
                            v = float(val[0](*arglst) / val[2])
                            n.set_value(i, v)
                        else:
                            assert isinstance(val,float)
                            n.set_value(i, val)
            for i, val in self.pinnedpositions.items():
                assert not isinstance(val,tuple)
                if self.unpin_instead:
                    n.unpin_position(i)
                else:
                    n.pin_position(i)
                    if not (isinstance(val, bool) and val == True):
                        n.set_x(i, val)
            for id, val in self.additional_vals.items():
                i = n.additional_value_index(id)
                assert not isinstance(val,tuple)
                if i >= 0:
                    if self.unpin_instead:
                        n.unpin(i)
                    else:
                        n.pin(i)
                        if not (isinstance(val, bool) and val == True):
                            n.set_value(i, val)

        if len(self.internal_vals) > 0:
            for ei in range(self.mesh.nelement()):
                e = self.mesh.element_pt(ei)
                # TODO: Where
                for idi, val in self.internal_vals.items():
                    d = e.internal_data_pt(idi)
                    for vi in range(d.nvalue()):
                        if self.unpin_instead:
                            d.unpin(vi)
                        else:
                            d.pin(vi)
                            if not (isinstance(val, bool) and val == True):
                                raise RuntimeError("TODO: Setting DL values")

    def on_apply_boundary_conditions(self, mesh:"AnyMesh"):
        mesh=assert_spatial_mesh(mesh)
        if (self.mesh is None) or (self.mesh != mesh):
            self.mesh = mesh
            self.setup()  # the dof indices are resolved once per mesh, the values applied every time
        self.apply()


class PinWhere(PythonDirichletBC):
    """
    Pins the given fields, but only at the nodes where ``where`` returns ``True``, e.g.
    ``PinWhere(lambda x,y: x**2+y**2<0.25, u=0)``. The callback gets the nondimensional nodal
    coordinates. Use it for conditions that cannot be stated on a named boundary.

    Args:
        where: Predicate on the nodal coordinates deciding where to pin.
        **kwargs: Fields to pin, either to a value or to ``True`` to keep the current one.
    """
    def __init__(self, where:Callable[...,bool], **kwargs:ExpressionOrNum | Literal[True]):
        super(PinWhere, self).__init__(**kwargs)
        self.where = where

    def apply(self):
        if not self.active:
            return
        assert self.mesh is not None
        if len(self.additional_vals) > 0:
            raise RuntimeError("Cannot use PinWhere yet on interface fields")
        
        for n in self.mesh.nodes():
            # Check the where condition
            xv:list[float] = []
            for xi in range(n.ndim()):
                xv.append(n.x(xi))
            if self.where(*xv) == False:
                continue
            
            for i, val in self.indexvals.items():            
                if self.unpin_instead:
                    n.unpin(i)
                else:
                    n.pin(i)
                    if not (isinstance(val, bool) and val == True):                
                        assert isinstance(val,float)
                        n.set_value(i, val)
            for i, val in self.pinnedpositions.items():                                                
                if self.unpin_instead:
                    n.unpin_position(i)
                else:
                    n.pin_position(i)
                    if not (isinstance(val, bool) and val == True) and isinstance(val,float):
                        n.set_x(i, val)


# Pin the mesh (2d only) for points that are further away than distance from the interface
# At the moment, we only allow to make the smallest circle around all interface points, add the distance to the radius
# and take this as reference. However, it will not co-move (we always have to do a setup_pinning to refresh)
# Further modes (e.g. "pointwise" may follow)


class UnpinDofs(PythonDirichletBC):
    """
    The inverse of :py:class:`DirichletBC`: frees the given degrees of freedom again, e.g. to release a
    boundary that a previously added condition has pinned.

    Args:
        **kwargs: Fields to unpin, e.g. ``UnpinDofs(u=True)``. The values are ignored.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unpin_instead=True


class StaticCondensation(Equations):
    """
    Declares which degrees of freedom of this domain static condensation may eliminate from the linear
    system. Contributes no residuals: it only states a selection, on the domain it is added to.

    **Experimental.** Serial and distributed (``--distribute``) runs are both supported. Element-local
    unknowns - a Crouzeix-Raviart bubble velocity, the
    gradient modes of a discontinuous (DL) pressure, a field projected onto a D0/DG space - couple to
    nothing outside their own element and can be eliminated by a small dense Schur complement before the
    Jacobian reaches the solver, and reconstructed from the retained increment afterwards. The
    elimination is exact: the same solution and the same Newton iteration count as without it. On a
    Crouzeix-Raviart Navier-Stokes system it saves roughly half of the factorisation time.

    The classical Crouzeix-Raviart elimination needs the bubble velocities and the pressure gradient
    modes **together** - neither half is invertible on its own - while the constant pressure mode has to
    stay a global unknown::

        eqs = NavierStokesEquations(mode="CR", dynamic_viscosity=1, mass_density=100)
        eqs += StaticCondensation(velocity="bubble", pressure="DL_gradients")
        self.add_equations(eqs @ "domain")

    ``"DL_gradients"`` is just a name for "every value of that ``"DL"`` field except the constant", so
    it expands to ``[1,2]`` in 2d and ``[1,2,3]`` in 3d without the script having to know which.

    Taking the constant mode along (``pressure=[0,1,2]``) is refused with an explanation, and so is
    condensing the pressure on its own - the continuity equation contains no pressure at all, so that
    block is structurally singular.

    Without any argument, every element-private degree of freedom of this domain is selected, i.e. every
    element-internal Data that no other element reads. That is the convenient form for auxiliary fields
    projected onto a discontinuous space::

        eqs += StaticCondensation()

    Internal Data that some other element does adopt as external data - an interface element on a free
    surface, an interior-facet DG coupling - is excluded from that automatic selection, since it is not
    element-local at all; a field named explicitly is not filtered that way, because the elimination
    handles those correctly too and the user asked for them.

    **Switching it on and off.** Adding this class anywhere in the equation tree sets
    ``problem.use_static_condensation`` for you. Assigning that switch explicitly always wins, in either
    direction, so ``self.use_static_condensation = False`` in ``define_problem()`` is a kill switch that
    disables the feature wholesale without touching the equations. Several instances on different
    domains compose - the selections are unioned - and so do rules declared at problem level with
    :py:meth:`~pyoomph.generic.problem.Problem.condense_dofs`.

    **Limitations.** Only Newton solves benefit: residual evaluations, eigenvalue and Hessian
    assemblies, and arclength continuation always see the full system. An MPI run, Jacobian reuse and
    the globally convergent (line search) Newton method are refused with an error rather than silently
    ignored. Adding it to an interface or ODE domain is an error. See
    ``dev_docs/static_condensation.md`` for the design and the measurements.

    Args:
        *fields: Fields to condense entirely, e.g. ``StaticCondensation("my_projection_field")``.
        **field_specs: Fields to condense in part. The value is ``"all"`` or ``True`` for the whole
            field, ``"internal"`` for the element-internal Data of an elemental (DL/D0/DG) field,
            ``"bubble"`` for the cell-interior bubble nodes of a nodal C1TB/C2TB field,
            ``"DL_gradients"`` for the gradient modes of a ``"DL"`` field (i.e. all of its values
            except the constant, resolved to ``[1,2]`` or ``[1,2,3]`` from the dimension of the
            domain it is added to), or an explicit list of value indices, e.g. ``pressure=[1,2]``.
    """

    def __init__(self,*fields:str,**field_specs:"str | bool | Sequence[int]"):
        super().__init__()
        self._rules:list[tuple[str,tuple[int,...],str]]=[]
        for f in fields:
            if not isinstance(f,str): #type:ignore
                raise ValueError("StaticCondensation takes field names as positional arguments, but got "+repr(f))
            self._rules.append((self._check_field_name(f),(),"all"))
        for f,spec in field_specs.items():
            values,part=self._parse_spec(f,spec)
            self._rules.append((self._check_field_name(f),values,part))
        # No argument at all means "every element-private dof of this domain", which is a rule of its
        # own rather than a list of fields - it names no field, and the C++ side auto-detects.
        self._element_private=(len(self._rules)==0)

    @staticmethod
    def _check_field_name(field:str)->str:
        if "/" in field:
            raise ValueError("StaticCondensation names fields of the domain it is added to, so '"+field+"' cannot contain a '/'. Add the class to that domain instead, or use Problem.condense_dofs() for a rule stated at problem level.")
        return field

    @staticmethod
    def _parse_spec(field:str,spec:"str | bool | Sequence[int]")->tuple[tuple[int,...],str]:
        if isinstance(spec,bool):   # before the Sequence/int cases: True is an int as far as Python is concerned
            if spec:
                return (),"all"
            raise ValueError("StaticCondensation("+field+"=False) is not a way to exclude a field - just leave it out.")
        if isinstance(spec,str):
            if spec in ("all","internal","bubble","DL_gradients"):
                # "DL_gradients" cannot be turned into value indices here: which ones they are depends
                # on the dimension of the elements, which is not known until the equation is attached
                # to a domain. It is resolved in on_apply_boundary_conditions() below.
                return (),spec
            raise ValueError("Unknown static condensation spec '"+spec+"' for field '"+field+"'. Use 'all', 'internal', 'bubble', 'DL_gradients', True, or a list of value indices.")
        if isinstance(spec,(list,tuple,set)):
            values:list[int]=[]
            for v in spec: #type:ignore
                if isinstance(v,bool) or not isinstance(v,int): #type:ignore
                    raise ValueError("StaticCondensation("+field+"=...) expects a list of integer value indices, but got "+repr(v))
                values.append(int(v))
            if len(values)==0:
                raise ValueError("StaticCondensation("+field+"=[]) selects nothing. Pass the value indices to condense, or True for the whole field.")
            return tuple(sorted(set(values))),"all"
        raise ValueError("Cannot interpret the static condensation spec "+repr(spec)+" for field '"+field+"'. Use 'all', 'internal', 'bubble', True, or a list of value indices.")

    def _resolve_rules(self,mesh:"AnySpatialMesh")->list[tuple[str,tuple[int,...],str]]:
        """Turn the dimension-dependent specs into concrete value indices, now that the mesh is known."""
        dim=mesh.get_dimension()
        resolved:list[tuple[str,tuple[int,...],str]]=[]
        for field,values,part in self._rules:
            if part=="DL_gradients":
                # A DL field carries one constant plus one gradient mode per spatial direction, so the
                # gradients are values 1..dim. Value 0, the constant, is deliberately left out: for the
                # Crouzeix-Raviart elimination it has to remain a global unknown.
                if dim<1:
                    raise RuntimeError("StaticCondensation("+field+"='DL_gradients') needs a spatial domain, but '"+mesh.get_full_name()+"' has element dimension "+str(dim)+".")
                resolved.append((field,tuple(range(1,dim+1)),"all"))
            else:
                resolved.append((field,values,part))
        return resolved

    def _is_ode(self):
        # Like PythonDirichletBC: this contributes no equations, so it must not decide whether the
        # domain is an ODE domain. It cannot be USED on one, which the checks below say explicitly
        # rather than leaving it to a confusing "cannot mix ODEs and PDEs".
        return None

    def get_information_string(self)->str:
        if self._element_private:
            return "element-private dofs"
        return ", ".join([f+"="+(str(list(v)) if v else p) for f,v,p in self._rules])

    def before_finalization(self,codegen:"FiniteElementCodeGenerator"):
        # Fires once per domain the equation is attached to, at code generation time, i.e. before the
        # mesh exists and independently of whether that mesh ends up holding any element - so a
        # misplaced instance is reported even where the registration hook below would never run.
        if codegen.get_parent_domain() is not None:
            raise RuntimeError("StaticCondensation cannot be added to an interface domain ('"+codegen.get_full_name()+"'): condensation eliminates element-local dofs of a bulk domain, and an interface element's own internal data belongs to a facet rather than to a cell. Add it to the bulk domain instead.")

    def on_apply_boundary_conditions(self,mesh:"AnyMesh"):
        # The registration hook, chosen because it (a) fires once the mesh exists, from
        # Problem.setup_pinning() during initialisation, and (b) fires again from
        # reapply_boundary_conditions() after adaptation, remeshing and reading a state file, which is
        # what refreshes the mesh a rule names once remeshing has replaced it. It is the same hook the
        # rest of the "act on the mesh, contribute no weak form" family (PythonDirichletBC, PinWhere,
        # UnpinDofs) uses, and it hands the mesh over directly.
        #
        # It also fires very often, so the registration below has to be free when nothing changed:
        # Problem._declare_static_condensation_rules() compares against what has been pushed to the C++
        # rule list and pushes nothing when the rules and the resolved mesh are the same. That matters -
        # every edit of that list bumps the rules revision, which is part of the Jacobian structure id,
        # so a naive re-registration would rebuild the condensation plan and force a fresh symbolic
        # factorisation on every solve. When the mesh HAS been replaced (remeshing destroys the old one,
        # and a C++ rule holds the mesh it names), the comparison fails and the rules are restated - so
        # this call is also what repairs them.
        from ..meshes.mesh import ODEStorageMesh
        if isinstance(mesh,InterfaceMesh):
            raise RuntimeError("StaticCondensation cannot be added to an interface domain ('"+mesh.get_full_name()+"'). Add it to the bulk domain instead.")
        if isinstance(mesh,ODEStorageMesh):
            raise RuntimeError("StaticCondensation cannot be added to an ODE domain ('"+mesh.get_full_name()+"'): there are no element-local degrees of freedom to eliminate there.")
        mesh=assert_spatial_mesh(mesh)
        problem=mesh.get_problem()
        domain=mesh.get_full_name()
        rules=[("",(),"element_private")] if self._element_private else self._resolve_rules(mesh)
        problem._declare_static_condensation_rules((StaticCondensation,id(self),domain),domain,rules) #type:ignore
        # Adding the class to the tree is the request; the problem-level switch stays available as a
        # kill switch, and an explicit assignment to it - True or False - always wins over this.
        problem._auto_enable_static_condensation() #type:ignore


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
