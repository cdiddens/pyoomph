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
 
 
from ..typings import NPFloatArray
from ..equations.generic import SpatialErrorEstimator, TemporalErrorEstimator
from ..expressions import CustomMultiReturnExpression, square_root, symbolic_diff, var_and_test,var,grad,dot,maximum,minimum,subexpression
from ..expressions.generic import ExpressionNumOrNone, ExpressionOrNum, FiniteElementSpaceEnum, dyadic, identity_matrix, partial_t, scale_factor, weak
from ..generic import *
from .navier_stokes import NavierStokesEquations #type:ignore
from .cahn_hilliard import smooth_clamp  # the C1 clamp SimpleNSCH already uses for rho and mu
from ..materials.generic import AnyFluidProperties, AnyFluidFluidInterface, LiquidGasInterfaceProperties, LiquidLiquidInterfaceProperties
from ..meshes.mesh import AnySpatialMesh,AnyMesh,MeshFromTemplate2d,Element,Node
from ..typings import *

if TYPE_CHECKING:
    # AnyFluidProperties is a string-valued TypeAlias (see materials/generic.py) whose members
    # (PureLiquidProperties, PureGasProperties, ...) are only resolvable via a wildcard import -
    # needed so tools that resolve forward references in type annotations (e.g. sphinx_autodoc_typehints)
    # can look them up in this module's namespace too, same as materials/generic.py itself does.
    # cahn_hilliard.py carries the same import for the same reason. It became necessary here when
    # CompositionNSCHEquations was given a docstring: an undocumented function is not autodoc'd, so
    # nothing tried to resolve its annotations before.
    from ..materials.generic import *


# Piecewise potential to prevent overshooting
# Returns f and f'
class PiecewiseNSCHPotential(CustomMultiReturnExpression):
    def __init__(self) -> None:
        super().__init__()
    
    def get_num_returned_scalars(self, nargs: int) -> int:
        assert nargs==1
        return 2
    
    def eval(self, flag: int, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray) -> None:
        phi=arg_list[0]
        if phi<-1:
            result_list[0]=(phi+1)**2
            result_list[1]=2*(phi+1)
        elif phi>1:
            result_list[0]=(phi-1)**2
            result_list[1]=2*(phi - 1)
        else:
            result_list[0]=(phi**2-1)**2/4    
            result_list[1]=phi*(phi**2 - 1)
        if flag:
            derivative_matrix[0]=result_list[1]
            if abs(phi)>1:            
                derivative_matrix[1]=2
            else:            
                derivative_matrix[1]=3*phi**2 - 1
            #self.debug_python_derivatives_with_FD(arg_list,result_list,derivative_matrix)

    # The two results are the first and second derivative of the double-well potential, so their
    # own derivatives are its third and fourth: a parabola and a line.
    def eval_second_derivatives(self, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray, second_derivative_tensor: NPFloatArray) -> None:
        self.eval(1, arg_list, result_list, derivative_matrix)
        phi = arg_list[0]
        if abs(phi) > 1:
            second_derivative_tensor[0, 0, 0] = 2.0
            second_derivative_tensor[1, 0, 0] = 0.0
        else:
            second_derivative_tensor[0, 0, 0] = 3 * phi ** 2 - 1
            second_derivative_tensor[1, 0, 0] = 6 * phi

    def generate_c_code_second_derivatives(self) -> str:
        return """
        CURRENT_MULTIRET_FUNCTION(PYOOMPH_MULTIRET_FLAG_DERIVATIVES, arg_list, result_list, derivative_matrix, nargs, nret);
        {
          const double phi=arg_list[0];
          if (phi<-1 || phi>1)
          {
            second_derivative_tensor[0]=2.0;
            second_derivative_tensor[1]=0.0;
          }
          else
          {
            second_derivative_tensor[0]=3.0*phi*phi - 1.0;
            second_derivative_tensor[1]=6.0*phi;
          }
        }
        """

    def generate_c_code(self) -> str:
        return """
        const double phi=arg_list[0];
        if (phi<-1)
        {
            result_list[0]=pow(phi+1,2);
            result_list[1]=2*(phi+1);
        }
        else if (phi>1)
        {
            result_list[0]=pow(phi-1,2);
            result_list[1]=2*(phi - 1);
        }
        else
        {                
            result_list[0]=pow(phi*phi-1,2)/4.0;
            result_list[1]=phi*(phi*phi - 1.0);
        }
        if (flag)
        {
            derivative_matrix[0]=result_list[1];
            if (phi<-1)
            {
                derivative_matrix[1]=2.0;
            }
            else if (phi>1)
            {
                derivative_matrix[1]=2.0;
            }
            else
            {
                derivative_matrix[1]=3.0*phi*phi - 1.0;
            }
        }
         
        """
            
    

    
    

class CompositionNSCHPhaseField(Equations):
    def __init__(self,epsilon:ExpressionOrNum,mobility:ExpressionOrNum,sigma_nsch:ExpressionOrNum,*,space:FiniteElementSpaceEnum="C2",phase_name:str="phi",potential_name:str="mu",partial_integrate_advection:bool=False,swap_test_functions:bool=True,mobility_for_scale:ExpressionNumOrNone=None,skew_symmetric_advection:bool=False,piecewise_potential:bool=False,potential_func:ExpressionNumOrNone=None):
        super().__init__()
        self.space:FiniteElementSpaceEnum=space
        self.phase_name=phase_name
        self.potential_name=potential_name
        self.mobility=mobility
        self.sigma_nsch=sigma_nsch
        self.epsilon=epsilon
        self.partial_integrate_advection=partial_integrate_advection
        self.swap_test_functions=swap_test_functions
        self.mobility_for_scale=mobility_for_scale
        self.skew_symmetric_advection=skew_symmetric_advection
        self.piecewise_potential=piecewise_potential
        self.potential_func=potential_func

    def define_fields(self):
        if self.swap_test_functions:
            testscale_phi=1/scale_factor(self.potential_name)
            testscale_mu=scale_factor("temporal")
        else:
            testscale_phi=scale_factor("temporal")
            testscale_mu=1/scale_factor(self.potential_name)
        self.define_scalar_field(self.phase_name,self.space,testscale=testscale_phi)        
        if self.mobility_for_scale is not None:
            scale_mobility=self.mobility_for_scale
        else:
            scale_mobility=self.mobility
        self.define_scalar_field(self.potential_name,self.space,scale=scale_factor("spatial")/scale_mobility*scale_factor("velocity"), testscale=testscale_mu)

    def define_residuals(self):
        phi,phi_test=var_and_test(self.phase_name)
        mu,mu_test=var_and_test(self.potential_name)   
        u,u_test=var_and_test("velocity")
        if self.swap_test_functions:
            phi_eq_test=mu_test     
            mu_eq_test=phi_test
        else:
            phi_eq_test=phi_test     
            mu_eq_test=mu_test
        
        if self.skew_symmetric_advection:
            self.add_residual(weak(partial_t(phi),phi_eq_test))
            self.add_residual(weak(dot(u,grad(phi)),phi_eq_test)/2-weak(u*phi,grad(phi_eq_test))/2) # Note that this induces additional boundary fluxes
            self.add_residual(weak(self.mobility*grad(mu),grad(phi_eq_test)))
        else:
            if self.partial_integrate_advection:
                self.add_residual(weak(partial_t(phi),phi_eq_test)) # Note that this induces additional boundary fluxes
                self.add_residual(weak(-u*phi+self.mobility*grad(mu),grad(phi_eq_test)))
            else:
                self.add_residual(weak(partial_t(phi)+dot(u,grad(phi)),phi_eq_test))
                self.add_residual(weak(self.mobility*grad(mu),grad(phi_eq_test)))


        if self.potential_func is None:
            potential=(phi**2-1)**2/4
            potential_prime=phi*(phi**2 - 1)
        else:
            potential=0+self.potential_func
            potential_prime=symbolic_diff(potential,var("phi"))
        if self.piecewise_potential:
            potential,potential_prime=PiecewiseNSCHPotential()(var("phi"))

        #self.add_residual(weak(mu-self.sigma_nsch/self.epsilon*potential_prime,mu_eq_test))
        self.add_residual(weak(mu,mu_eq_test))
        self.add_residual(weak(-self.sigma_nsch/self.epsilon*potential_prime,mu_eq_test))
        self.add_residual(-weak(self.sigma_nsch*self.epsilon*grad(phi),grad(mu_eq_test)))

        xi=-self.sigma_nsch*self.epsilon*dyadic(grad(phi),grad(phi))+identity_matrix()*(self.sigma_nsch*self.epsilon/2*dot(grad(phi),grad(phi))+self.sigma_nsch/self.epsilon*potential)
        self.add_residual(weak(xi,grad(u_test)))


def CompositionNSCHEquations(positive_props:AnyFluidProperties,negative_props:AnyFluidProperties,epsilon:ExpressionOrNum,mobility:ExpressionOrNum,interface_props:AnyFluidFluidInterface | None=None,phase_field_name:str="phi",partial_integrate_advection:bool=False,swap_test_functions:bool=True,velocity_error_factor:float=100,skew_symmetric_advection:bool=False,piecewise_potential:bool=False,mobility_for_scale:ExpressionNumOrNone=None,potential_func:ExpressionNumOrNone=None,temporal_error_factors:Dict[str,float] | None=None):
    """Navier-Stokes coupled to a Cahn-Hilliard phase field, as one set of equations.

    ``temporal_error_factors`` names the fields whose temporal error drives
    :py:meth:`~pyoomph.generic.problem.Problem.run` with ``temporal_error=...``, e.g.
    ``{"velocity_x":1, "velocity_y":1, "phi":1}``. ``None`` (the default) adds no estimator, which
    is what this factory has always done - but be aware of what that means: with no estimator
    anywhere the global norm is clamped to 1e-12 by oomph-lib, so the step is never rejected and dt
    is multiplied by ``DTSF_max_increase`` every step until something else stops it. Only the output
    step and ``maxstep`` then bound it, and a run that looks adaptive is not.
    """
    # SMOOTH, not minimum/maximum. Those differentiate branch-wise, as
    # dx*heaviside(x-y)+dy*heaviside(y-x) (see pyoomph.expressions.maximum), so a hard clamp makes
    # the viscosity and the density C0-but-not-C1 in phi - and they enter the momentum equations
    # only. A Cahn-Hilliard phase field routinely overshoots |phi|>1 next to a sharp interface;
    # once it does, the Newton direction is computed on one branch of the kink while the residual
    # at the new iterate falls on the other, so the iteration two-cycles at a residual that does
    # not shrink with dt. Measured on the printhead with use_NSCH=True: the residual alternated
    # between 1.78e-3 and 2.13e-3 for all ten iterations, entirely on velocity_x/velocity_y with
    # phi and mu three to four orders smaller, while dt was cut by six orders with no effect, until
    # the run died on oomph-lib's minimum dt.
    # smooth_clamp is the quintic smoothstep SimpleNSCH already uses for exactly these two
    # coefficients (cahn_hilliard.get_density_for_ns / get_viscosity_for_ns). Its derivative
    # vanishes at both ends, so the composite is C1 across the kink, and positivity - which is what
    # the clamp is for - is preserved.
    phi_clamp=subexpression(smooth_clamp(var(phase_field_name),-1,1))
    mu=subexpression(positive_props.dynamic_viscosity*(1+phi_clamp)/2+negative_props.dynamic_viscosity*(1-phi_clamp)/2)
    #rho=subexpression(positive_props.mass_density*(1+var(phase_field_name))/2+negative_props.mass_density*(1-var(phase_field_name)))
    rho=subexpression(positive_props.mass_density*(1+phi_clamp)/2+negative_props.mass_density*(1-phi_clamp)/2)

    if interface_props is None:
        # positive_props and negative_props are both AnyFluidProperties (liquid or gas, never solid), so
        # get_interface_properties (invoked via __or__) can only take the "liquid_gas" or "liquid_liquid"
        # branch and thus can only return a LiquidGasInterfaceProperties or LiquidLiquidInterfaceProperties
        # instance (i.e. AnyFluidFluidInterface). The declared return type of __or__ is a broader union
        # (it also covers BaseInterfaceProperties/LiquidSolidInterfaceProperties for the solid-involving
        # cases), which pyright cannot narrow away here, hence the assert to make the invariant explicit.
        computed_interface_props=positive_props | negative_props
        assert isinstance(computed_interface_props,(LiquidGasInterfaceProperties,LiquidLiquidInterfaceProperties))
        interface_props=computed_interface_props
    sigma_nsch=3/(2*square_root(2))*interface_props.surface_tension
    res:Equations=CompositionNSCHPhaseField(epsilon=epsilon,mobility=mobility, sigma_nsch=sigma_nsch,phase_name=phase_field_name,partial_integrate_advection=partial_integrate_advection,swap_test_functions=swap_test_functions,skew_symmetric_advection=skew_symmetric_advection,piecewise_potential=piecewise_potential,mobility_for_scale=mobility_for_scale,potential_func=potential_func)
    res+=NavierStokesEquations(dynamic_viscosity=mu,mass_density=rho,boussinesq=True)
    #res+=SpatialErrorEstimator(evaluate_in_past(var(phase_field_name)),**{phase_field_name:1.0,"velocity":100})
    res+=SpatialErrorEstimator(**{phase_field_name:1.0,"velocity":velocity_error_factor}) #type:ignore
    if temporal_error_factors:
        res+=TemporalErrorEstimator(**temporal_error_factors)
    #res+=SpatialErrorEstimator(**{phase_field_name:1.0})

    #u,u_test=var_and_test("velocity")
    #J=(negative_props.mass_density-positive_props.mass_density)*mobility*grad(var("mu"))
    #res+=WeakContribution(dyadic(J,u),grad(u_test))
    
    return res




def _phase_field_equation_types():
    """The equation classes that own a phase field named by ``phase_name`` (default ``phi``).

    Late import: ``low_order_NSCH`` pulls in the package root, so importing it at module level here
    would close an import cycle. Both helpers below look the phase field up BY TYPE, so without
    this the simple first-order NSCH would raise "Could not find a single phase field".
    """
    from .low_order_NSCH import LowOrderNSCH
    return (CompositionNSCHPhaseField, LowOrderNSCH)


class RefinePhaseFieldGradients(Equations):
    def __init__(self, level:Literal["max"] | int="max",bound=0.8):
        super(RefinePhaseFieldGradients, self).__init__()
        self.level = level
        self.bound=bound

    def calculate_error_overrides(self):
        mesh=self.get_current_code_generator()._mesh 
        assert mesh is not None
        must_refine = 100 * mesh.max_permitted_error
        may_not_unrefine = 0.5 * (mesh.max_permitted_error+mesh.min_permitted_error)
        
        eq=self.get_combined_equations().get_equation_of_type(_phase_field_equation_types(),always_as_list=True)
        currmesh=mesh
        while len(eq)==0 and currmesh is not None:
            currmesh=currmesh.get_bulk_mesh()
            if currmesh is None:
                raise RuntimeError("Cannot find a phase-field equation (CompositionNSCHPhaseField or LowOrderNSCH) in this domain")
            eq=currmesh._eqtree.get_equations().get_equation_of_type(_phase_field_equation_types(),always_as_list=True)
        
        if len(eq)!=1:
            raise RuntimeError("Cannot find a unique phase-field equation in this domain")
        
        eq=eq[0]

        phi_index=mesh.get_code_gen().get_code().get_nodal_field_index(getattr(eq,"phase_name","phi"))
        for e in mesh.elements():
            refine_this_elem=False
            for ni in range(e.nnode()):
                 ph=e.node_pt(ni).value(phi_index)
                 if abs(ph)<self.bound:
                      refine_this_elem=True
                      break
            if not refine_this_elem:
                 continue
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
            



# Marking disjunct domains by an integer D0 field
# Can be used e.g. in integral expressions or similar
class DisjunctDomainMarkerNSCH(Equations):
    def __init__(self,name:str,direction:Literal["up","down"]="up",mark_threshold:float=-0.1) -> None:
        super().__init__()
        self.name=name
        self.direction:Literal["up","down"]=direction # Direction of increasing marker
        self.mark_threshold=mark_threshold

    def define_fields(self) -> None:
        self.define_scalar_field(self.name,"D0")

    def define_residuals(self) -> None:
        self.set_Dirichlet_condition(self.name,True) # Do not solve for it. Will be set by hand

    def _update_marker(self,mesh:AnySpatialMesh):
        phase_field_eqs=self.get_combined_equations().get_equation_of_type(_phase_field_equation_types(),always_as_list=True)
        if len(phase_field_eqs)!=1:
            raise RuntimeError("Could not find a single phase field")
        phase_field_eq=phase_field_eqs[0]
        # LowOrderNSCH hardcodes the name instead of exposing it.
        phase_ind=mesh.get_code_gen().get_code().get_nodal_field_index(getattr(phase_field_eq,"phase_name","phi"))

        if mesh.nelement()==0:
            return
        marker_index=mesh.element_pt(0).get_jit_code().get_discontinuous_field_index(self.name)
        # unhandled_nodes is an insertion-ordered dict rather than a set, and checknodes a stack
        # rather than a set: both hold Node objects, and a set of those iterates by id(), i.e. by
        # allocation address. Which connected component became droplet 0 - and which node seeded it
        # when several share the extremal y - was therefore not reproducible between two runs, let
        # alone between two MPI ranks marking the same geometry. The dict is only used as a set.
        # Reset all markers
        unhandled_nodes:dict[Node,None]={}
        unhandled_elems:set[Element]=set()
        nodes2elem:dict[Node,list[Element]]={}
        # Create the look-up tables for unhandles nodes and node->elements map
        for e in mesh.elements():
            e.internal_data_pt(marker_index).set_value(0,-1)
            unhandled_elems.add(e)
            for ni in range(e.nnode()):
                n=e.node_pt(ni)
                if n.value(phase_ind)>self.mark_threshold:
                    unhandled_nodes[n]=None
                    if n not in nodes2elem.keys():
                        nodes2elem[n]=[]
                    nodes2elem[n].append(e)
        
        # Start over numbering the droplets
        domain_index=0
        self._max_droplet_index=0
        while len(unhandled_nodes)>0: # We still have nodes which do not belong to any domain
            # Find the node with maximum or minimum y
            ym=1e20*(-1 if self.direction=="down" else 1)
            startnode=None
            for n in unhandled_nodes:
                if (n.x(1)<ym if self.direction=="up" else n.x(1)>ym):
                    startnode=n
                    ym=n.x(1)
            if startnode is None:
                break

            # Flood-fill like algorithm
            checknodes:list[Node]=[startnode] # seed the start node
            while len(checknodes)>0:
                nn=checknodes.pop() # get one node out of the bucket
                if nn in unhandled_nodes: # only check further if the node was not handled before
                    unhandled_nodes.pop(nn)
                    for e in nodes2elem[nn]: # go over all elements the node is part of
                        if e in unhandled_elems:
                            e.internal_data_pt(marker_index).set_value(0,domain_index) # mark the element
                            self._max_droplet_index=domain_index
                            unhandled_elems.remove(e)
                            for ni in range(e.nnode()):
                                n=e.node_pt(ni)
                                if n in unhandled_nodes:
                                    checknodes.append(n)
            domain_index+=1

    def after_newton_solve(self):
        mesh=self.get_mesh()
        assert isinstance(mesh,MeshFromTemplate2d)
        self._update_marker(mesh)
        super().after_newton_solve()
    
    def on_apply_boundary_conditions(self, mesh: "AnyMesh"):
        assert isinstance(mesh,MeshFromTemplate2d)
        self._update_marker(mesh)
        return super().on_apply_boundary_conditions(mesh)
        


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
