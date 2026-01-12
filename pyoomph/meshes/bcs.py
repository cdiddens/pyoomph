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
 

import _pyoomph
import inspect

from ..expressions.generic import Expression, ExpressionOrNum, FiniteElementSpaceEnum
from ..meshes.mesh import InterfaceMesh, MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d, assert_spatial_mesh
from ..generic.codegen import Equations,  InterfaceEquations, BaseEquations
from ..expressions import testfunction, weak, var_and_test, test_scale_factor, scale_factor
from ..utils.smallest_circle import make_circle
import scipy.spatial #type:ignore
import numpy
from ..expressions.coordsys import AxisymmetryBreakingCoordinateSystem,AxisymmetricCoordinateSystem,BaseCoordinateSystem


from ..typings import *
if TYPE_CHECKING:
    from ..meshes.mesh import AnySpatialMesh,Node,AnyMesh
    from ..solvers.generic import GenericEigenSolver
    from ..generic.codegen import EquationTree,FiniteElementCodeGenerator

class BoundaryCondition(Equations):
    def __init__(self):
        super(BoundaryCondition, self).__init__()
        self.mesh = None
        self.active = True
        self.mesh:Optional["AnySpatialMesh"]=None

    def setup(self):
        pass

    def set_mesh(self, mesh:"AnySpatialMesh"):
        self.mesh = mesh
        self.setup()

    def apply(self):
        pass

    def on_apply_boundary_conditions(self, mesh:"AnyMesh"):
        mesh=assert_spatial_mesh(mesh)
        if (self.mesh is None) or (self.mesh != mesh):
            self.set_mesh(mesh)
        self.apply()


class NeumannBC(InterfaceEquations):
    """
    Class to impose Neumann boundary condition. The particular meaning of the Neumann flux depends on the bulk equations, i.e. how the integration by parts was performed for the weak formulation of the bulk equations.
    For a Poisson equation implemented by the residual weak(grad(u),grad(utest)), the Neumann condition 

        ``NeumannBC(u=1) @ "boundary"``

    does not neither mean setting ``u=1``, but rather dot(grad(u),var("normal"))=-1, where normal vector is pointing outward the domain at the boundary.

    Parameters:
    	**fluxes: Dictionary of fluxes, where the keys are the names of the fluxes and the values are expressions or numbers.
    """

    def __init__(self, **fluxes:ExpressionOrNum):
        super(NeumannBC, self).__init__()
        self.fluxes = fluxes.copy()

    def define_residuals(self):
        for name, flux in self.fluxes.items():
            test = testfunction(name)
            self.add_residual(weak(flux, test))


# Automatic Neumann condition either takes a single flux as pos. arg (when there is only one equation of the required_parent_type)
# or by named fluxes
class AutomaticNeumannCondition(InterfaceEquations):
    neumann_sign = 1

    def __init__(self, flux:Optional[ExpressionOrNum]=None, **named_fluxes:ExpressionOrNum):
        super(AutomaticNeumannCondition, self).__init__()
        self.flux = flux
        self.named_fluxes = named_fluxes.copy()
        if (self.flux is None) and (len(self.named_fluxes) != 0):
            raise RuntimeError("AutomaticNeumannCondition must be either constructed with a single pos. arg or with **kwargs")

    def get_parent_var_name(self, parent_eq:BaseEquations)->str:
        assert hasattr(parent_eq,"name")
        assert isinstance(parent_eq.name,str) #type:ignore
        return parent_eq.name #type:ignore

    def define_residuals(self):
        bulkeqs = self.get_parent_equations()
        if isinstance(bulkeqs, (list, tuple)):
            if self.flux is not None:
                raise RuntimeError(
                    "Cannot use " + self.__class__.__name__ + " with a positional argument only if there are multiple bulk equations of type " + str(
                        self.required_parent_type))
            else:
                for k, v in self.named_fluxes.items():
                    _, u_test = var_and_test(k)
                    self.add_residual(self.neumann_sign * weak(v, u_test))
        else:
            assert self.flux is not None
            assert bulkeqs is not None
            _, u_test = var_and_test(self.get_parent_var_name(bulkeqs))
            self.add_residual(self.neumann_sign * weak(self.flux, u_test))


class EnforcedBC(InterfaceEquations):
    """
    Enforce rather arbitrary boundary conditions by a field of Lagrange multipliers.
    As an example,
    
        EnforcedBC(u=var("u")-var("v")) @ "boundary"
        
    will set u=v on the boundary by adjusting u. The rhs must be a constraint in residual formulation, here u-v=0.

    Args:
        only_for_stationary_solve (bool, optional): Flag indicating if the enforced boundary conditions should only be applied during stationary solves. Defaults to False.
        set_zero_on_normal_mode_eigensolve (bool, optional): Flag indicating if the enforced boundary conditions should be set to zero during azimuthal eigensolves. Defaults to False.
        **constraints (Expression): Keyword arguments representing the enforced boundary conditions as pair of variable name to adjust and constraint expression to fulfill in residual form.
    """
 
    def __init__(self,*, only_for_stationary_solve:bool=False, set_zero_on_normal_mode_eigensolve=False,domain:Optional[str]=None,space:Optional[FiniteElementSpaceEnum]=None,coordinate_system:Optional[BaseCoordinateSystem]=None,**constraints:Expression):
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
        codeinst_inside = mesh.element_pt(0).get_code_instance()
        for k, _ in self.constraints.items():            
            index = [codeinst_inside.get_nodal_field_index(k)]  # TODO: Vectors
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
            pr=self.get_mesh()._problem
            from ..generic.bifurcation_tools import _NormalModeBifurcationTrackerBase
            if pr.get_bifurcation_tracking_mode() == "azimuthal" or (pr.get_custom_assembler() is not None and isinstance(pr.get_custom_assembler(),_NormalModeBifurcationTrackerBase)): 
                #if self.get_mesh()._problem._azimuthal_mode_param_m.value!=0:
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
    
    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver", angular_mode:Optional[int],normal_k:Optional[float])->Set[Union[str,int]]:
        if (not self.set_zero_on_normal_mode_eigensolve) or (angular_mode is None and normal_k is None):
            return cast(Set[str],set())
        else:
            if angular_mode is not None and normal_k is not None:
                raise RuntimeError("Cannot have both angular and normal mode set")
            if angular_mode is not None:
                mode=int(angular_mode)
            elif normal_k is not None:
                mode=normal_k
            fullpath = eqtree.get_full_path().lstrip("/")
            if mode == 0:
                return cast(Set[str],set())
            else:
                for_my_m = [self.get_lagrange_multiplier_name(k) for k in self.constraints.keys()]
            lst=[fullpath + "/" + k for k in for_my_m]
            res:Set[str] = set(lst) 
            return res    

class DirichletBC(BaseEquations):
    """
    Class to impose one or more Dirichlet boundary condition.

    Args:
        prefer_weak_for_DG (bool, optional): Flag indicating whether to prefer weak contributions for Discontinuous Galerkin (DG) methods. If set and the bulk equations provide a specific implementation of get_weak_dirichlet_terms_for_DG, these terms are used to enforce the condition in a weak sense. Otherwise, just stronly. Defaults to True.
        **kwargs (ExpressionOrNum): Keyword arguments representing the Dirichlet conditions, where the keys are the variable names and the values are the corresponding expressions or numbers. Expressions for strong DirichletBCs may not depend on unknowns.

    """

    def __init__(self, *, prefer_weak_for_DG: bool = True, **kwargs: ExpressionOrNum):
        super(DirichletBC, self).__init__()
        self._dcs: Dict[str, ExpressionOrNum] = {}
        self._dcs.update(kwargs)
        self.prefer_weak_for_DG = prefer_weak_for_DG

    def define_residuals(self):
        pdom = self.get_parent_domain()
        peqs = pdom.get_equations() if pdom is not None else None
        if not isinstance(peqs, Equations):
            peqs = None
        for n, val in self._dcs.items():
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



class EnforcedDirichlet(EnforcedBC):
    """
    Enforces a DirichletBC by Lagrange multipliers.
    As an example,
    
        ``EnforcedDirichlet(u=var("v")) @ "boundary"``
        
    will just be the same as :py:class:`~pyoomph.meshes.bcs.EnforcedBC` ``(u=var("u")-var("v")) @ "boundary"``
        
    
    Args:
        only_for_stationary_solve (bool, optional): Flag indicating if the enforced boundary conditions should only be applied during stationary solves. Defaults to False.
        set_zero_on_normal_mode_eigensolve (bool, optional): Flag indicating if the enforced boundary conditions should be set to zero during azimuthal eigensolves. Defaults to False.
        **constraints (Expression): Keyword arguments representing the enforced boundary conditions as pair of variable name to adjust and constraint expression to fulfill in residual form.
    """
    
    def __init__(self,*, only_for_stationary_solve:bool=False, set_zero_on_normal_mode_eigensolve=False,domain:Optional[str]=None,space:Optional[FiniteElementSpaceEnum]=None,coordinate_system:Optional[BaseCoordinateSystem]=None, **constraints:Expression):
        from ..expressions import var        
        new_kwargs={k:var(k,domain=domain)-v for k,v in constraints.items()}
        super(EnforcedDirichlet, self).__init__(only_for_stationary_solve=only_for_stationary_solve, set_zero_on_normal_mode_eigensolve=set_zero_on_normal_mode_eigensolve,coordinate_system=coordinate_system ,**new_kwargs.copy(),domain=domain,space=space)


class InactiveDirichletBC(DirichletBC):
    """
    Same as 'DirichletBC', but it starts deactivated, i.e. the Neumann term will be active by default.

    To activate the Dirichlet condition, you must call the set_dirichlet_active(...) method of the Mesh class, which you can obtain by problem.get_mesh(...).
    Afterwards, it is important to call Problem.reapply_boundary_conditions(), e.g.
    
        problem.get_mesh("domain/interface").set_dirichlet_active(u=True) # activate the BC
        problem.reapply_boundary_conditions() # Renumber the equations and apply BCs
        problem.solve() # solve with active DirichletBC
    
    """
    def __init__(self, **kwargs: ExpressionOrNum):
        super().__init__(**kwargs)
        self._init_setup_for_mesh:Set[AnyMesh]=set()
    
    def before_assigning_equations_preorder(self, mesh: "AnyMesh"):
        if mesh in self._init_setup_for_mesh: # Only init it once during problem init. Someone might have switched it later on
            return super().before_assigning_equations_preorder(mesh)        
        mesh.set_dirichlet_active(**{k:False for k in self._dcs.keys()})        
        self._init_setup_for_mesh.add(mesh) # Don't ever set it again for this mesh
        # TODO: Check redefine_problem and/or remeshing
        return super().before_assigning_equations_preorder(mesh)

    def after_remeshing(self, eqtree: "EquationTree"):
        raise RuntimeError("Check remeshing settings here...")
        return super().after_remeshing(eqtree)

    


class AxisymmetryBC(InterfaceEquations):
    r"""
    Add this to the axis of symmetry to automatically enforce the boundary condition required by symmetry.
    Also automatically sets the correct boundary conditions for azimuthal eigenvalue problems.

    For normal solving, it sets radial (and azimuthal) components of vector fields (also mesh_x) to zero.
    For azimuthal eigenvalue problems, it depends on the azimuthal mode m:
    
        :math:`m=0`: As for normal solving.
        :math:`|m|=1`:  scalar fields and axial vector components are set to zero, radial and azimuthal components are not.
        :math:`|m|\geq 2`: scalar fields and all vector components are set to zero
    
    If you write an equation, where you want to change this behavior, you can manually change the conditions by obtaining the (writeable) field information via :py:meth:`~pyoomph.generic.codegen.Equations.get_azimuthal_r0_info` after the definition via :py:meth:`~pyoomph.generic.codegen.Equations.define_scalar_field` or :py:meth:`~pyoomph.generic.codegen.Equations.define_vector_field`.

    Notes:
        Must be also added to intersections of other boundaries with the axis of symmetry, when the other boundaries define additional fields, e.g.
        
            bulk=NavierStokesEquations(...)
            bulk+=AxisymmetryBC()@"axis"
            bulk+=NavierStokesFreeSurface()@"interface"
            bulk+=AxisymmetryBC()@"interface/axis" # This is important, since the free surface introduces new fields at the interface
            
        This is, however, done automatically if the recurse flag is set to True.
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
            trunk=eqtree.get_parent().get_full_path().lstrip("/")
            myname=eqtree.get_my_path_name()
            for conn in interinter:
                rest=conn[len(eqtree.get_full_path().lstrip("/")):].lstrip("/")
                path=trunk+"/"+rest+"/"+myname
                revconns.append(path)
            revconns.sort(key=lambda x: x.count("/")) # Sort by number of slashes to get it in good order
            root=eqtree
            while root.get_parent() is not None:
                root=root.get_parent()
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
                            oldeqs=dom._children[splt[-1]]._equations
                            dom._children[splt[-1]]._equations+=AxisymmetryBC(verbose=self.verbose,recurse=self.recurse)
                            dom._children[splt[-1]]._equations._problem=oldeqs._problem
                else:
                    dom._children[splt[-1]]=EquationTree(AxisymmetryBC(verbose=self.verbose,recurse=self.recurse),dom)
                    dom._children[splt[-1]]._equations._problem=dom._equations._problem
                    
            
        return super()._fill_interinter_connections(eqtree, interinter)
                                
    
    def define_residuals(self):
        if self.verbose:
            print("AxisymmetryBC: Setting zero DirichletBCs at",self.get_current_code_generator().get_full_name(),"for",self.get_azimuthal_r0_info()[0])
        for k in self.get_azimuthal_r0_info()[0]:            
            self.set_Dirichlet_condition(k,0)
                
                
    def _before_stationary_or_transient_solve(self, eqtree:"EquationTree", stationary:bool)->bool:
        must_reapply=False
        #if self.get_mesh()._problem.get_bifurcation_tracking_mode() == "azimuthal": 
        from ..generic.bifurcation_tools import _NormalModeBifurcationTrackerBase
        pr=self.get_mesh()._problem
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
    
           
    def _before_eigen_solve(self, eqtree:"EquationTree", eigensolver:"GenericEigenSolver",angular_m:Optional[float]=None,normal_k:Optional[float]=None) -> bool:
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
                        
    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree, eigensolver, angular_mode, normal_k):
        if angular_mode is None:
            return set()
        
        angular_mode=int(angular_mode)
            
        info=None
        if angular_mode==0:
            info=self.get_azimuthal_r0_info()[0]            
        elif abs(angular_mode)==1:
            info=self.get_azimuthal_r0_info()[1]
        elif abs(angular_mode)>1:
            info=self.get_azimuthal_r0_info()[2]
        if info is None:
            res=set() 
        else:
            res=set([eqtree.get_full_path().lstrip("/")+"/"+m for m in info])
        if len(info)>0 and self.verbose:            
            print("AxisymmetryBC (mode m="+str(angular_mode)+"): Imposed zero by matrix manipulation at",self.get_current_code_generator().get_full_name(),"for",info)
        return res
            



# Scalar fields on DG space => set the corresponding eigenfunctions
class AxisymmetryBCForScalarD0Field(InterfaceEquations):
    def __init__(self,*fields:str):
        super().__init__()
        self.fields=[f for f in fields]

    def _get_forced_zero_dofs_for_eigenproblem(self, eqtree: "EquationTree", eigensolver: "GenericEigenSolver", angular_mode: Union[int,None],normal_k:Optional[float]) -> Set[Union[str,int]]:
        eqs=set()
        if angular_mode!=0:
            for ie in eqtree._mesh.elements():
                be=ie.get_bulk_element()
                for f in self.fields:
                    fi=be.get_code_instance().get_discontinuous_field_index(f)
                    if fi<0:
                        raise RuntimeError("Discontinuous parent field '"+str(f)+"' not known here")                
                    eqs.add(be.internal_data_pt(fi).eqn_number(0))
        return eqs


class PeriodicBC(InterfaceEquations):
    """
    Introduces a periodic boundary condition between two interfaces. It will hold for all continuous fields!
    The mesh must be generated that way that for each node on this interface, there is a corresponding node on the other interface when adding offset to the position.

    Attributes:
        other_interface (str): The name of the other interface to which this boundary is periodic.
        offset (Optional[List[ExpressionOrNum]]): The offset to find the corresponding nodes on the other interface.

    """

    def __init__(self, other_interface: str, offset: Optional[List[ExpressionOrNum]] = None):
        super(PeriodicBC, self).__init__()
        self.other_interface = other_interface        
        if offset is None:
            raise RuntimeError("Please supply an offset")
        elif not isinstance(offset,(list,tuple)):
            self.offset=[offset]
        else:
            self.offset = offset

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
        my_nodes_by_pos: Dict[Tuple[float, ...], _pyoomph.Node] = {}
        for n in pmesh.boundary_nodes(my_name):
            ps = [n.x(i) + offs[i] for i in range(n.ndim())]
            my_nodes_by_pos[tuple(ps)] = n

        dataG: List[List[float]] = []
        master_nodes: List[_pyoomph.Node] = []
        for n in pmesh.boundary_nodes(self.other_interface):
            ps = [n.x(i) for i in range(n.ndim())]
            dataG.append(ps)
            master_nodes.append(n)
        data = numpy.array(dataG)  # type:ignore
        if len(data) != len(my_nodes_by_pos):  # type:ignore
            raise RuntimeError("Mismatch in number of nodes for a periodic boundary")
        kdtree = scipy.spatial.KDTree(data)  # type:ignore

        slave_to_master:Dict[_pyoomph.Node,_pyoomph.Node]=dict()
        master_to_slave:Dict[_pyoomph.Node,_pyoomph.Node]=dict()
        for ps, nslave in my_nodes_by_pos.items():
            qres = kdtree.query(ps)  # type:ignore
            if qres[0] > 1e-6:  # type:ignore
                raise RuntimeError("Cannot find a periodic node at the position " + str(ps))
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
            oppnodes_to_oppelem:Dict[_pyoomph.Node,_pyoomph.Element]=dict()
            for oppelem,direct in pmesh.boundary_elements(self.other_interface,with_directions=True):
                oppnodes_to_oppelem[tuple(oppelem.boundary_nodes(oppind))]=(oppelem,direct)
            
            for myelem,direct in pmesh.boundary_elements(my_name,with_directions=True):                
                my_nodes_on_bind=myelem.boundary_nodes(myind)                            
                search_for=tuple(slave_to_master[n] for n in my_nodes_on_bind)
                oppelem=oppnodes_to_oppelem.get(search_for,None)
                if oppelem is None:            
                    raise RuntimeError("Cannot identify the corresponding periodic boundary element on the other interface.")
                myelem._connect_periodic_tree(oppelem[0],direct,oppelem[1])
        


class PythonDirichletBC(BoundaryCondition):
    def __init__(self, **kwargs:Union[ExpressionOrNum,Literal[True],Tuple[Callable[...,ExpressionOrNum],List[int],Expression]]):
        super(PythonDirichletBC, self).__init__()
        self.vals = kwargs.copy()
        self.unpin_instead:bool = False

    def _is_ode(self):
        return None

    def get_information_string(self) -> str:
        return ", ".join([str(k) + "=" + str(v) for k, v in self.vals.items()])

    def setup(self):
        assert self.mesh is not None
        self.indexvals:Dict[int,Union[float,Literal[True],Tuple[Callable[...,ExpressionOrNum],List[int],Expression]]] = {}
        self.indexval_arginds = {}
        self.additional_vals:Dict[int,Union[float,Literal[True],Tuple[Callable[...,ExpressionOrNum],List[int],Expression]]] = {}
        self.pinnedpositions:Dict[int,Union[float,Literal[True],Tuple[Callable[...,ExpressionOrNum],List[int],Expression]]] = {}
        self.internal_vals:Dict[int,Union[float,Literal[True],Tuple[Callable[...,ExpressionOrNum],List[int],Expression]]] = {}
        codeinst = self.mesh.element_pt(0).get_code_instance()

        currcodegen = self.get_current_code_generator()

        vals:Dict[str,Union[ExpressionOrNum,Literal[True],Tuple[Callable[...,ExpressionOrNum],List[int],Expression]]] = {}

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
            fval:Union[float,bool,Tuple[Callable[...,ExpressionOrNum],List[int],Expression]]
            if not isinstance(val, bool) or val != True:
                try:
                    fval = float(val / scal) #type:ignore ##TODO Functions here
                except:
                    if callable(val):
                        arg_inds:List[int] = []
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


class PinWhere(PythonDirichletBC):
    def __init__(self, where:Callable[...,bool], **kwargs:Union[ExpressionOrNum,Literal[True]]):
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
            xv:List[float] = []
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
class PinMeshAtDistanceToInterface(PinWhere):
    def __init__(self, interface_names:Union[str,Set[str]], distance:ExpressionOrNum, mode:str="smallest_circle"):
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
        pts:List[Tuple[float,float]] = []
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


class UnpinDofs(PythonDirichletBC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unpin_instead=True

class InteriorBoundaryOrientation(InterfaceEquations):
    """
    Named interior boundaries within a domain are by default double-layered, i.e. interface elements are added from both sides.
    This can usually cause problems. In order to avoid this, we have to specify the orientation of the boundary, i.e. only interface elements are added from one side, namely where the indicator function is positive.
    For a unit circle ``"circle"`` embedded in a domain, you could e.g. add ``InteriorBoundaryOrientation(dot(var("coordinate"),var("normal)))@"circle"`` to only add interface elements with an outward pointing normal.

    Args:
        InterfaceEquations (_type_): _description_
    """    
    def __init__(self,indicator:ExpressionOrNum):
        super().__init__()
        self.indicator=indicator
        
    def define_residuals(self):
        self.add_local_function("__interface_constraint",self.indicator)