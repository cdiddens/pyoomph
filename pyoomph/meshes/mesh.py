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
 
import inspect
import os.path

from ..generic.mpi import mpi_barrier

from ..typings import *


import numpy

import _pyoomph

from ..expressions.generic import Expression, ExpressionOrNum, is_zero, NameStrSequence




import itertools


if TYPE_CHECKING:
    from ..generic.problem import Problem, Z2ErrorEstimator
    from ..output.states import DumpFile
    from .remesher import RemesherBase
    from ..generic.codegen import EquationTree, FiniteElementCodeGenerator


Node = _pyoomph.Node
Element=_pyoomph.OomphGeneralisedElement
AnySpatialMesh = Union["InterfaceMesh", "MeshFromTemplate1d",
                       "MeshFromTemplate2d", "MeshFromTemplate3d"]
AnyMesh = Union[AnySpatialMesh, "ODEStorageMesh"]


def assert_spatial_mesh(mesh: Optional[Union[AnyMesh, "MeshFromTemplateBase"]]) -> AnySpatialMesh:
    if mesh is None:
        raise RuntimeError("Mesh is None")
    elif isinstance(mesh, ODEStorageMesh):
        raise RuntimeError("Expected spatial mesh, but got ODEStorageMesh")
    elif isinstance(mesh, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d, InterfaceMesh)):
        return cast(AnySpatialMesh,mesh)
    else:
        raise RuntimeError("Should not end up here")


class BaseMesh:
    def __init__(self):
        super(BaseMesh, self).__init__()
        # self._interfacial_elements=dict()
        self._interfacemeshes: Dict[str, "InterfaceMesh"] = dict()
        self._outputscales = {}
        self.initial_uniform_refinements = 0
        self._initial_interface_refinement = {}
        # Tracer particles -> name to tracer instance
        self._tracers: Dict[str, _pyoomph.TracerCollection] = {}
        self._codegen: Optional["FiniteElementCodeGenerator"]
        self._problem: "Problem"
        self._eqtree: "EquationTree"

    def get_code_gen(self) -> "FiniteElementCodeGenerator":
        assert self._codegen is not None
        return self._codegen

    def get_eqtree(self) -> "EquationTree":
        return self._eqtree

    def get_tracers(self, name: str = "tracers", error_on_missing: bool = True) -> Optional[_pyoomph.TracerCollection]:
        if name not in self._tracers.keys():
            if error_on_missing:
                raise RuntimeError("Cannot find tracers " +
                                   str(name)+" on this mesh")
            return None
        else:
            return self._tracers[name]

    def set_dirichlet_active(self, **kwargs: bool):
        for k, v in kwargs.items():
            if (v is True) or (v is False):
                assert isinstance(self, _pyoomph.Mesh)
                self._set_dirichlet_active(k, v)
            else:
                raise ValueError(
                    "Please set Dirichlet active either to True or False")

    def boundary_intersection_nodes(self, bname1: str, bname2: str) -> List[Node]:
        assert isinstance(self, _pyoomph.Mesh)
        imesh = self.get_mesh(bname1)
        assert imesh is not None
        res: Set[Node] = set()
        i2 = self.get_boundary_index(bname2)
        for e in imesh.boundary_elements(bname2):
            nn = e.nnode()
            for i in range(nn):
                n = e.node_pt(i)
                if n.is_on_boundary(i2):
                    res.add(n)
        return list(res)

    def nodes(self) -> Iterator[_pyoomph.Node]:
        assert isinstance(self, _pyoomph.Mesh)
        numnodes = self.nnode()
        for i in range(numnodes):
            yield self.node_pt(i)

    def elements(self) -> Iterator[_pyoomph.OomphGeneralisedElement]:
        assert isinstance(self, _pyoomph.Mesh)
        numelems = self.nelement()
        for i in range(numelems):
            yield self.element_pt(i)

    @overload
    def boundary_elements(
        self, b: str, with_directions: Literal[False] = ...) -> Iterator[_pyoomph.OomphGeneralisedElement]: ...

    @overload
    def boundary_elements(
        self, b: str, with_directions: Literal[True]) -> Iterator[Tuple[_pyoomph.OomphGeneralisedElement, int]]: ...

    def boundary_elements(self, b: str, with_directions: bool = False) -> Union[Iterator[_pyoomph.OomphGeneralisedElement], Iterator[Tuple[_pyoomph.OomphGeneralisedElement, int]]]:
        assert isinstance(self, _pyoomph.Mesh)
        bn = self.get_boundary_names()
        bn = bn.index(b)
        numelems = self.nboundary_element(bn)
        if with_directions:
            for i in range(numelems):
                yield self.boundary_element_pt(bn, i), self.face_index_at_boundary(bn, i)
        else:
            for i in range(numelems):
                yield self.boundary_element_pt(bn, i)

    def boundary_nodes(self, b: str) -> Iterable[_pyoomph.Node]:
        assert isinstance(self, _pyoomph.Mesh)
        bn = self.get_boundary_names()
        bn = bn.index(b)
        numelems = self.nboundary_node(bn)
        for i in range(numelems):
            yield self.boundary_node_pt(bn, i)

    @overload
    def get_mesh(self, name: str, return_None_if_not_found: Literal[False] = ...) -> Union["MeshFromTemplate1d",
                                                                                           "MeshFromTemplate2d", "MeshFromTemplate3d", "InterfaceMesh"]: ...

    @overload
    def get_mesh(self, name: str, return_None_if_not_found: Literal[True]) -> Union["MeshFromTemplate1d",
                                                                                    "MeshFromTemplate2d", "MeshFromTemplate3d", "InterfaceMesh", None]: ...

    def get_mesh(self, name: str, return_None_if_not_found: bool = False) -> Union["MeshFromTemplate1d", "MeshFromTemplate2d", "MeshFromTemplate3d", "InterfaceMesh", None]:
        splt = name.split("/")
        if len(splt) == 1:
            if not (name in self._interfacemeshes.keys()):
                if return_None_if_not_found:
                    return None
                else:
                    raise RuntimeError(
                        "No interface mesh constructed on interface " + name)
            return self._interfacemeshes[name]
        else:
            if not (splt[0] in self._interfacemeshes.keys()):
                if return_None_if_not_found:
                    return None
                else:
                    raise RuntimeError(
                        "Cannot get mesh " + name + " since parent mesh " + splt[0] + " is constructed on the interface")
            if return_None_if_not_found:
                return self._interfacemeshes[splt[0]].get_mesh("/".join(splt[1:]), return_None_if_not_found=True)
            else:
                return self._interfacemeshes[splt[0]].get_mesh("/".join(splt[1:]), return_None_if_not_found=False)

    def _pre_compile_interface_equations(self, tree_depth: int):
        if tree_depth == 0:
            for _, imsh in self._interfacemeshes.items():
                imsh._pre_compile()
                mpi_barrier()
        else:
            for _, imsh in self._interfacemeshes.items():
                imsh._pre_compile_interface_equations(tree_depth-1)
                mpi_barrier()

    def _compile_interface_equations(self, tree_depth: int):
        if tree_depth == 0:
            for n in sorted(self._interfacemeshes.keys()):
                imsh=self._interfacemeshes[n]
                imsh._compile()
                mpi_barrier()
        else:
            for n in sorted(self._interfacemeshes.keys()):
                imsh=self._interfacemeshes[n]
                imsh._compile_interface_equations(tree_depth-1)
                mpi_barrier()

    def _generate_interface_elements(self, tree_depth: int):
        if tree_depth == 0:
            for n in sorted(self._interfacemeshes.keys()):
                imsh=self._interfacemeshes[n]
                assert imsh._codegen is not None
                imsh._codegen._perform_external_ode_linkage()
                imsh.ensure_external_data()
                assert imsh._codegen._code is not None
                imsh._codegen._code._exchange_mesh(imsh)
                imsh._setup_output_scales()
                assert isinstance(self, _pyoomph.Mesh)
                self.generate_interface_elements(n, imsh, imsh._codegen._code)
                # imsh.nullify_selected_bulk_dofs()  # TODO
        else:
            for n in sorted(self._interfacemeshes.keys()):
                imsh=self._interfacemeshes[n]
                imsh._generate_interface_elements(tree_depth-1)

    def evaluate_observable(self, name: str) -> ExpressionOrNum:
        assert isinstance(self, _pyoomph.Mesh)
        lst = self.list_integral_functions()
        assert self._codegen is not None
        deps = self._codegen._dependent_integral_funcs
        cmb: Set[str] = set()
        cmb.update(lst)
        cmb.update(deps.keys())

        if name in lst:
            res = self._evaluate_integral_function(name)
        elif name in self._codegen._dependent_integral_funcs.keys():
            l = deps[name]
            args: List[ExpressionOrNum] = []
            for a in inspect.signature(l).parameters:
                if not (a in cmb):
                    raise RuntimeError("During evaluation of integral observable "+name +
                                       ": Cannot evaluate the observable "+name+". Possible are "+", ".join(cmb))
                args.append(self.evaluate_observable(a))
            res = l(*args)

        else:

            raise ValueError("Integral observable "+name +
                             " not defined on this mesh. Possible integral observables on this mesh are: "+", ".join(cmb))
        return res

    def evaluate_all_observables(self) -> Dict[str, ExpressionOrNum]:
        assert isinstance(self, _pyoomph.Mesh)
        lst = self.list_integral_functions()
        assert self._codegen is not None
        deps = self._codegen._dependent_integral_funcs
        res: Dict[str, ExpressionOrNum] = {}
        for l in lst:
            res[l] = self._evaluate_integral_function(l)
        args: Dict[str, ExpressionOrNum] = {k: v for k, v in res.items()}
        args["time"] = self._problem.get_current_time()
        remaining: Set[str] = set(deps.keys())
        while len(remaining) > 0:
            torem: Set[str] = set()
            for r in remaining:
                # Check if we can evaluate
                l = deps[r]
                all_present = True
                arglist: List[ExpressionOrNum] = []
                for a in inspect.signature(l).parameters:
                    if not a in args.keys():
                        all_present = False
                    else:
                        arglist.append(args[a])
                if all_present:
                    torem.add(r)
                    depres = l(*arglist)
                    args[r] = depres
                    res[r] = depres
            if len(torem) == 0:
                raise RuntimeError(
                    "Cannot evaluate the dependent integral functions, probably due to unknown or circular arguments : "+str(remaining))
            remaining = remaining-torem
        # Now remove the vector helpers
        for k in self._codegen._dependent_integral_funcs_is_vector_helper.keys():
            del res[k]
        # And expand all numpy arrays
        newres: Dict[str, ExpressionOrNum] = {}
        for k, v in res.items():
            if isinstance(v, numpy.ndarray):
                for i, direct, compo in zip([0, 1, 2], ["x", "y", "z"], v):
                    if not (is_zero(compo) and i >= self._codegen.get_nodal_dimension()):
                        newres[k+"_"+direct] = compo

            else:
                newres[k] = v
        return newres


    @overload
    def get_maximum_value_of_field(self,fieldname:str,minimum_instead:bool=False,dimensional:Literal[True]=...)->ExpressionOrNum: ...

    @overload
    def get_maximum_value_of_field(self,fieldname:str,minimum_instead:bool=False,dimensional:Literal[False]=...)->float: ...

    def get_maximum_value_of_field(self,fieldname:str,minimum_instead:bool=False,dimensional:bool=True) ->ExpressionOrNum:
        assert self._codegen is not None
        func=min if minimum_instead else max 
        contind=self._codegen.get_code().get_nodal_field_index(fieldname)
        if contind>=0:
            maxim=None
            for n in self.nodes():
                maxim=n.value(contind) if maxim is None else func(maxim,n.value(contind))
            if maxim is None:
                raise RuntimeError("Empty mesh")
            else:
                return maxim*(self._codegen.get_scaling(fieldname) if dimensional else 1)
        else:
            discind=self._codegen.get_code().get_discontinuous_field_index(fieldname)
            if discind<0:
                raise RuntimeError("Cannot find the field '"+str(fieldname)+"' in the mesh")
            maxim=None
            
            for e in self.elements():                
                # On DL, this only respects the center value
                maxim=e.internal_data_pt(discind).value(0) if maxim is None else func(maxim,e.internal_data_pt(discind).value(0))
            if maxim is None:
                raise RuntimeError("Empty mesh")
            else:
                return maxim*(self._codegen.get_scaling(fieldname) if dimensional else 1)

######################################################

class MeshTemplateOppositeInterfaceConnection:
    def __init__(self, sideA: str, sideB: str, problem:"Problem", matchfunc: Optional[Callable[[Sequence[float], Sequence[float]], float]] = None):
        self._sideA = sideA
        self._sideB = sideB
        self._problem=problem
        if matchfunc:
            self._match_pos_func = matchfunc
            self._use_kdtree = False
        else:
            self._match_pos_func: Callable[[Sequence[float], Sequence[float]], float] = lambda a, b: sum([pow(a[j] - b[j], 2) for j in range(len(b))])
            self._use_kdtree = True

    def __str__(self) -> str:
        return "MeshTemplateOppositeInterfaceConnection("+str(self._sideA)+","+str(self._sideB)+")"

    def _connect_opposite_interfaces(self, eqtree_root: "EquationTree"):
        sideA = eqtree_root.get_by_path(self._sideA)
        sideB = eqtree_root.get_by_path(self._sideB)
        if sideA is None or sideB is None:  # TODO: Ensure dummy equations!
            return
        assert sideA._codegen is not None
        assert sideB._codegen is not None
        sideA._codegen._set_opposite_interface(sideB._codegen)
        sideB._codegen._set_opposite_interface(sideA._codegen)

    def _ensure_opposite_tree_node(self, eqtree_root: "EquationTree"):
        sideA = eqtree_root.get_by_path(self._sideA)
        sideB = eqtree_root.get_by_path(self._sideB)
        if sideA is None and sideB is None:  # Nothing to be done
            return
        elif sideA is not None and sideB is not None:
            return
        elif sideA is None:
            # Only create if parent is there
            ppth = self._sideA.split("/")[0:-1]
            if eqtree_root.get_by_path("/".join(ppth)):
                eqtree_root._create_dummy_equations_at_path(
                    self._sideA, eqtree_root,self._problem)
        else:
            ppth = self._sideB.split("/")[0:-1]
            if eqtree_root.get_by_path("/".join(ppth)):
                eqtree_root._create_dummy_equations_at_path(
                    self._sideB, eqtree_root,self._problem)

    def _connect_elements(self, eqtree_root: "EquationTree"):
        sideA = eqtree_root.get_by_path(self._sideA)
        sideB = eqtree_root.get_by_path(self._sideB)
        if not (sideA and sideB):
            return
        if not (sideA._mesh and sideB._mesh):
            return
        meshA = sideA._mesh
        meshB = sideB._mesh
        assert isinstance(meshA, InterfaceMesh)
        assert isinstance(meshB, InterfaceMesh)
        meshA._opposite_interface_mesh = meshB
        meshB._opposite_interface_mesh = meshA

        if self._use_kdtree:
            assert isinstance(meshA, InterfaceMesh)
            assert isinstance(meshB, InterfaceMesh)
            meshA._connect_interface_elements_by_kdtree(meshB)
            return
        assert not isinstance(meshA, _pyoomph.ODEStorageMesh)
        assert not isinstance(meshB, _pyoomph.ODEStorageMesh)

        posBmap: Dict[Tuple[Tuple[float, ...], ...],
                      _pyoomph.OomphGeneralisedElement] = {}
        for eB in meshB.elements():
            pos: List[Tuple[float, ...]] = []
            for nvj in range(eB.nvertex_node()):
                v = eB.vertex_node_pt(nvj)
                pos.append(tuple([v.x(xi) for xi in range(v.ndim())]))
            pos = sorted(pos)
            posBmap[tuple(pos)] = eB

        offset_vector=meshA.get_opposite_interface_offset_vector()
        rev_offset=[-o for o in offset_vector]
        print("OFFSET VECTOR",offset_vector)
        for eA in meshA.elements():
            pos2find: List[List[float]] = []
            for nvi in range(eA.nvertex_node()):
                v = eA.vertex_node_pt(nvi)
                pos2find.append([v.x(xi) for xi in range(v.ndim())])
            pos2find = sorted(pos2find)
            found = False
            for pB, eB in posBmap.items():
                if len(pB) == len(pos2find):
                    dist = 0
                    for i in range(len(pos2find)):
                        dist += self._match_pos_func(pos2find[i], pB[i])
                    # print(pB,len(pB),dist)
                    if dist < 1e-8:
                        eA.set_opposite_interface_element(eB,offset_vector)
                        eB.set_opposite_interface_element(eA,rev_offset)
                        found = True
                        break
            if not found:
                debug_entries: List[Tuple[float,Tuple[Tuple[float, ...], ...]]] = []
                for pB, eB in posBmap.items():
                    if len(pB) == len(pos2find):
                        dist = 0.0
                        for i in range(len(pos2find)):
                            dist += self._match_pos_func(pos2find[i], pB[i])
                        debug_entries.append((dist, pB))
                for e in sorted(debug_entries, key=lambda a: a[0]):
                    print(e[1], "dist=", e[0])
                # from ..output.meshio import _MeshFileOutput
                # debugoutA=_MeshFileOutput(problem=meshA._problem, mesh=meshA,ftrunk="DEBUG_MeshA",write_pvd=False)
                # debugoutA.init()
                # debugoutB = _MeshFileOutput(problem=meshB._problem,mesh=meshB, ftrunk="DEBUG_MeshB",write_pvd=False)
                # debugoutB.init()
                # debugoutA.output(0)
                # debugoutB.output(0)
                raise RuntimeError("Cannot connect the interface element at " +
                                   str(pos2find)+" to the required opposite side")


class MeshTemplate(_pyoomph.MeshTemplate):
    """
    A class to construct meshes by defining nodes with the :py:meth:`add_node` or :py:meth:`add_node_unique` method. 
    Elements must be specified by first creating one or multiple domains with the :py:meth:`new_domain` method and adding elements on each domain.
    Nodes can be also marked to be on particular boundaries with the :py:meth:`add_node_on_boundary` method.
    """
    def __init__(self):
        super(MeshTemplate, self).__init__()
        self._domains: Dict[str, _pyoomph.MeshTemplateElementCollection] = {}
        self._problem = None
        self._geometry_defined = False
        #: The minimum permitted error for the spatial error estimator. If ``None``, we use the value from the :py:class:`~pyoomph.generic.problem.Problem` object.
        self.min_permitted_error = None
        #: The maximum permitted error for the spatial error estimator. If ``None``, we use the value from the :py:class:`~pyoomph.generic.problem.Problem` object.
        self.max_permitted_error = None
        #: The maximum refinement level for spatial adaptivity. If ``None``, we use the value from the :py:class:`~pyoomph.generic.problem.Problem` object.
        self.max_refinement_level = None
        #: The minimum refinement level for spatial adaptivity. If ``None``, we use the value from the :py:class:`~pyoomph.generic.problem.Problem` object.
        self.min_refinement_level = None
        self._opposite_interface_connections: List[MeshTemplateOppositeInterfaceConnection] = [
        ]
        self._meshfile = None
        #: Must be set to allow for remeshing.
        self.remesher: Optional["RemesherBase"] = None
        self.auto_find_opposite_interface_connections = True
        self._template_override = None
        self._interior_boundaries: Set[str] = set()
        self._macrobounds: List[_pyoomph.MeshTemplateCurvedEntityBase] = []
        self._fntrunk:Optional[str]=None # To be set for remeshing
        self.all_nodes_as_boundary_nodes:bool=False

    def get_problem(self) -> "Problem":
        return self._problem
    
    def set_boundary_as_interior(self, *args: str):
        for name in args:
            self._interior_boundaries.add(name)

    def define_state_file(self, state: "DumpFile",additional_info={}) -> "MeshTemplate":
        mshfile = self.get_template()._meshfile
        if mshfile is None:
            mshfile = ""
        else:
            mshfile = os.path.relpath(mshfile, os.path.dirname(state.fname))
        found_mshfile = state.string_data(lambda: mshfile, lambda s: s)
        if not state.save and found_mshfile != "":
            print("Template is using msh file "+found_mshfile+". Consider to change it with the additional_info dict by setting additional_info['exchange_msh_file']['"+found_mshfile+"']=...")
        if "exchange_msh_file" in additional_info:
            if found_mshfile in additional_info["exchange_msh_file"]:
                found_mshfile=additional_info["exchange_msh_file"][found_mshfile]
#        else:
#            print("Writing meshfile "+found_mshfile)

        has_remesher = 1 if self.remesher is not None else 0
        state.int_data(lambda: has_remesher,
                       lambda r: state.assert_equal(has_remesher, r))
        if has_remesher:
            assert self.remesher is not None
            self.remesher._cnt = state.int_data(
                lambda: self.remesher._cnt, lambda s: s)  # type:ignore
        if found_mshfile != mshfile:
            # We need to load the remeshed version here
            import pyoomph.meshes.gmsh
            statedir = os.path.dirname(state.fname)
            fffound_mshfile = os.path.join(statedir, found_mshfile)
            newtempl = pyoomph.meshes.gmsh.GmshTemplate(fffound_mshfile)
            newtempl.remesher = self.remesher
            assert self._problem is not None
            newtempl._do_define_geometry(self._problem)
            self._template_override = newtempl
            newtempl.get_template()._meshfile = fffound_mshfile
            # self._meshfile=found_mshfile
        return self.get_template()
        # print("IN STATE FILE "+mshfile,state.fname,)
        # exit()

    def get_opposite_interface(self, side: str) -> Optional[str]:
        for ic in self._opposite_interface_connections:
            if ic._sideA == side:
                return ic._sideB
            elif ic._sideB == side:
                return ic._sideA
        return None

    # Called from C on automatic finding
    def _add_opposite_interface_connection(self, sideA: str, sideB: str):
        self.add_opposite_interface_connection(sideA, sideB)

    def add_opposite_interface_connection(self, sideA: str, sideB: str, matchfunc: Optional[Callable[[Sequence[float], Sequence[float]], float]] = None):
        self._opposite_interface_connections.append(
            MeshTemplateOppositeInterfaceConnection(sideA, sideB, self._problem, matchfunc))

    def _connect_opposite_interfaces(self, eqtree_root: "EquationTree"):
        for conn in self._opposite_interface_connections:
            conn._connect_opposite_interfaces(eqtree_root)

    def _ensure_opposite_eq_tree_nodes(self, eqtree_root: "EquationTree"):
        for conn in self._opposite_interface_connections:
            conn._ensure_opposite_tree_node(eqtree_root)

    def _connect_opposite_elements(self, eqtree_root: "EquationTree"):
        for conn in self._opposite_interface_connections:
            conn._connect_elements(eqtree_root)

    def get_template(self) -> "MeshTemplate":
        if self._template_override is None:
            return self
        else:
            return self._template_override

    def define_geometry(self) -> None:
        """
        This method must be specialized in a derived class to define the geometry, i.e. the nodes, domains and elements of the mesh.
        """
        raise RuntimeError("Please implement the function define_geometry")

    def available_domains(self) -> Set[str]:
        """
        Returns a list of all available domains constructed with :py:meth:`new_domain`.
        """
        if not self._geometry_defined:
            raise RuntimeError(
                "Can only check the available domains after _do_define_geometry")
        return set(self._domains.keys())

    def has_domain(self, name: str) -> bool:
        """
        Test if a domain with the given name is available, i.e. constructed with :py:meth:`new_domain` before.
        """
        if not self._geometry_defined:
            raise RuntimeError(
                "Can only check the available domains after _do_define_geometry")
        return name in self._domains.keys()

    def get_domain(self, name: str) -> _pyoomph.MeshTemplateElementCollection:
        """
        Get a domain by name constructed with the method :py:meth:`new_domain` before.
        """
        if not self._geometry_defined:
            raise RuntimeError(
                "Can only get a domain after _do_define_geometry")
        return self._domains[name]

    def _do_define_geometry(self, problem: "Problem"):
        if not self._geometry_defined:
            self._geometry_defined = True
            self._problem = problem
            self._set_problem(problem)
            self.define_geometry()
            if self.auto_find_opposite_interface_connections:
                self._find_opposite_interface_connections()

    def new_domain(self, name: str, nodal_dimension: Optional[int] = None) -> _pyoomph.MeshTemplateElementCollection:
        """
        Create a new domain with the given name. With the help of this domain, elements can be added to the mesh.
        """
        if not self.has_domain(name):
            self._domains[name] = self.new_bulk_element_collection(name)
            if nodal_dimension is not None:
                self._domains[name].set_nodal_dimension(nodal_dimension)
        else:
            raise RuntimeError("Domain with name '" + name +
                               "' already in the mesh template")
        if self.all_nodes_as_boundary_nodes:
            self._domains[name].set_all_nodes_as_boundary_nodes()
        return self._domains[name]

    @overload
    def nondim_size(self, a: ExpressionOrNum) -> float: ...

    @overload
    def nondim_size(self, a: List[ExpressionOrNum]) -> List[float]: ...

    def nondim_size(self, a: Union[ExpressionOrNum, List[ExpressionOrNum]]) -> Union[float, List[float]]:
        """
        Nondimensionalize a coordinate or a length scale by dividing by the spatial scale of the problem.
        
        Args:
            a: The coordinate or length scale to nondimensionalize.
            
        Returns:
            The arguments divided by the spatial scale of the problem.
        """
        if isinstance(a, list):
            resL: List[float] = []
            for b in a:
                resL.append(self.nondim_size(b))
            return resL
        res: float
        if isinstance(a, float) or isinstance(a, int) or isinstance(a,_pyoomph.GiNaC_GlobalParam) or isinstance(a,numpy.floating) or isinstance(a,numpy.integer):
            assert self._problem is not None
            res = (float(a / self._problem.get_scaling("spatial")))
        elif isinstance(a, _pyoomph.Expression):  # type:ignore
            assert self._problem is not None
            res = ((a / self._problem.get_scaling("spatial")).float_value())        
        else:
            raise ValueError("Strange spatial argument for a mesh:"+str(a)+" of type "+str(type(a)))
        return res

    def add_nodes(self, *args: Sequence[float]) -> Optional[Union[int , Tuple[int, ...]]]:
        res: List[int] = []
        for a in args:
            res.append(self.add_node(*a))
        if len(res) == 0:
            return
        elif len(res) == 1:
            return res[0]
        else:
            return tuple(res)

    def create_curved_entity(self, typ: str, *args: Any, **kwargs: Any)-> _pyoomph.MeshTemplateCurvedEntityBase:
        """
        Creates a curved entity on the mesh. Currently only the type ``"circle_arc"`` is supported, which requires the start and end point as positional arguments and either the ``center`` or ``through_point`` as keyword argument.

        Args:
            typ: Type of the curved entity. Currently only ``"circle_arc"`` is supported.
            args: Positional arguments for the curved entity.
            kwargs: Keyword arguments for the curved entity.

        Returns:
            The created curved entity to be used in :py:meth:`add_facet_to_curve_entity`.
        """
        store_entity: bool = kwargs.get("store_entity", True)
        res = None
        if typ == "circle_arc":
            if len(args) != 2 or (kwargs.get("center") is None and kwargs.get("through_point") is None):
                raise RuntimeError(
                    "circle_arg must have two positional args {start,end} and either through_point or center as kwarg")
            if kwargs.get("center") is not None:
                if kwargs.get("through_point") is not None:
                    raise RuntimeError(
                        "Either pass center or through_point as kwarg")
                center = kwargs.get("center")
            else:
                raise RuntimeError("TODO: do the through_point")
            start, end = args[0], args[1]
            if isinstance(center, int):
                center = self.get_node_position(center)
            if isinstance(start, int):
                start = self.get_node_position(start)
            if isinstance(end, int):
                end = self.get_node_position(end)
            res = _pyoomph.CurvedEntityCircleArc(
                center, start, end)  # type:ignore
        else:
            raise RuntimeError("Unknown type "+str(typ))
        if store_entity:
            self._macrobounds.append(res)
        return res


class MeshFromTemplateBase(BaseMesh):
    def __init__(self, problem: "Problem", templatemesh: MeshTemplate, domainname: str, eqtree: "EquationTree", previous_mesh: Optional["MeshFromTemplateBase"] = None):
        super(MeshFromTemplateBase, self).__init__()

        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))

        self._problem = problem
        self._templatemesh: MeshTemplate = templatemesh
        self._name = domainname
        self._eqtree: "EquationTree" = eqtree
        self._eqtree._mesh = self
        # self._equations = eqtree._equations
        self._codegen = eqtree.get_code_gen()
        self._codegen._mesh = self
        self.ignore_initial_condition = False
        self._set_problem(problem, self._codegen._code)
        self._error_estimator: Z2ErrorEstimator  # =None
        self._solves_since_remesh = 0  # Counting the number of solves since last remesh
        self._periodic_corner_node_info: Dict[Node, Node] = {}

        T = TypeVar("T")

        def a_or_b(a: Optional[T], b: Optional[T]) -> T:
            res = b if a is None else a
            assert res is not None
            return res

        if previous_mesh is None:
            self.min_permitted_error = a_or_b(
                templatemesh.min_permitted_error, self._problem.min_permitted_error)
            self.max_permitted_error = a_or_b(
                templatemesh.max_permitted_error, self._problem.max_permitted_error)
            self.max_refinement_level = a_or_b(
                templatemesh.max_refinement_level, self._problem.max_refinement_level)
            self.min_refinement_level = a_or_b(
                templatemesh.min_refinement_level, self._problem.min_refinement_level)
        else:
            self.min_permitted_error = previous_mesh.min_permitted_error
            self.max_permitted_error = previous_mesh.max_permitted_error
            self.max_refinement_level = previous_mesh.max_refinement_level
            self.min_refinement_level = previous_mesh.min_refinement_level
            assert isinstance(self, _pyoomph.Mesh)
            assert isinstance(previous_mesh, _pyoomph.Mesh)
            self._setup_information_from_old_mesh(previous_mesh)

        if previous_mesh is None:
            coll = self._templatemesh.get_domain(self._name)
            edim = coll.get_element_dimension()
            assert self._codegen is not None
            self._codegen._set_nodal_dimension(coll.nodal_dimension())
            self._codegen._set_lagrangian_dimension(
                coll.lagrangian_dimension())
            ocg = self._codegen.get_equations()._get_current_codegen()
            self._codegen.get_equations()._set_current_codegen(self._codegen)
            self._codegen._do_define_fields(edim)
            self._codegen._index_fields()
            self._codegen.get_equations()._set_current_codegen(ocg)
            self._was_remeshed = False
        else:
            self._was_remeshed = True
        self._interfacemeshes = {}
        for n, eqtree in self._eqtree.get_children().items():
            pinter = None
            if previous_mesh is not None:
                pinter = previous_mesh.get_mesh(n)
                assert isinstance(pinter, InterfaceMesh)
            self._interfacemeshes[n] = InterfaceMesh(
                self._problem, self, n, eqtree, previous_mesh=pinter)

    def get_problem(self) -> "Problem":
        assert self._problem is not None
        return self._problem

    def get_bulk_mesh(self):
        return None



    def _link_periodic_corner_nodes(self):
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        if len(self._periodic_corner_node_info) == 0:
            return
        newmap: Dict[Node, Node] = {}
        for islv, imst in self._periodic_corner_node_info.items():
            rmst = imst
            visited = set([imst])
            while self._periodic_corner_node_info.get(rmst) is not None:
                rmst = self._periodic_corner_node_info.get(rmst)
                assert rmst is not None
                if rmst in visited:
                    raise RuntimeError("Looped periodic corner nodes map")
                visited.add(rmst)
            newmap[islv] = rmst
        for slv, mst in newmap.items():
            slv._make_periodic(mst, self)

    def setup_initial_conditions_with_interfaces(self, resetting_first_step: bool, ic_name: str):
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        if self.ignore_initial_condition:
            return
        assert_spatial_mesh(self).setup_initial_conditions(
            resetting_first_step, ic_name)
        for _, im in self._interfacemeshes.items():
            im.setup_initial_conditions_with_interfaces(
                resetting_first_step, ic_name)

    def get_name(self) -> str:
        return self._name

    def get_full_name(self) -> str:
        return self.get_name()

    def _reset_elemental_error_max_override(self):
        for e in self.elements():
            e._elemental_error_max_override = 0.0

    def _merge_my_error_with_elemental_max_override(self) -> List[float]:
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        res = self.get_elemental_errors()
        # print("IN MERGE",self.get_name(),res)
        for i, e in enumerate(self.elements()):
            res[i] = max(e._elemental_error_max_override, res[i])
            # e._elemental_error_max_override=res[i]
        # TODO: Elements that are only at one interface
        return res

    def get_nodal_field_indices(self) -> Dict[str, int]:
        return self.get_code_gen().get_code().get_nodal_field_indices()

    def recreate_boundary_information(self):
        import pyoomph

        if self.refinement_possible() or (pyoomph.get_dev_option("allow_tri_refine") and self.get_dimension() == 2):
            self.setup_tree_forest()

        self.setup_boundary_element_info()

        for interior_bound in self._templatemesh.get_template()._interior_boundaries:
            try:
                bindex = self.get_boundary_index(interior_bound)
                self.setup_interior_boundary_elements(bindex)
            except:
                pass

    def _setup_output_scales(self):
        codegen = self.get_code_gen()
        code = codegen.get_code()
        _, unit, _, _ = _pyoomph.GiNaC_collect_units(
            codegen.expand_placeholders(codegen.get_scaling("spatial"), False))
        self.set_output_scale("spatial", unit, code)  # TODO
        for k in itertools.chain(code.get_nodal_field_indices().keys(), code.get_elemental_field_indices().keys()):
            s = codegen.get_scaling(k)
            s = codegen.expand_placeholders(s, False)
            _, unit, _, _ = _pyoomph.GiNaC_collect_units(s)
            self.set_output_scale(k, unit, code)
        
    def _finalise_creation(self):
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        self.generate_from_template(
            self._templatemesh.get_template().get_domain(self._name))
        # if self.refinement_possible():
        import pyoomph

        if self.refinement_possible() or (pyoomph.get_dev_option("allow_tri_refine") and self.get_dimension() == 2):
            self.setup_tree_forest()

        self.setup_boundary_element_info()

        for interior_bound in self._templatemesh.get_template()._interior_boundaries:
            try:
                bindex = self.get_boundary_index(interior_bound)
                self.setup_interior_boundary_elements(bindex)
            except:
                pass

        self._error_estimator = _pyoomph.Z2ErrorEstimator()
        self._error_estimator.use_Lagrangian = False
        self.set_spatial_error_estimator_pt(self._error_estimator)
        codegen = self.get_code_gen()
        code = codegen.get_code()
        # This will allocate the Dirichlet BC active buffer
        self._set_problem(self.get_problem(), code)
        # default to SI units in output #TODO
        self._setup_output_scales()
#        self.perform_set_output_scales(self._equations._code)  #TODO

        bn = self.get_boundary_names()
        for b, imsh in self._interfacemeshes.items():
            if not (b in bn) and b!="_internal_facets_":
                raise RuntimeError("Boundary " + b +
                                   " not in mesh '"+str(self.get_full_name())+"', i.e. '"+str(self.get_full_name())+"/"+b+"' does not exist. Boundaries of '"+str(self.get_full_name())+"' are "+str(bn))
            ieqs = imsh.get_eqtree().get_equations()
            icg = imsh.get_eqtree().get_code_gen()
            if icg._code is not None:

                if (not self._was_remeshed) and (ieqs._problem != self.get_problem() or icg.get_parent_domain() != self._codegen or icg.get_nodal_dimension() != self._eqtree.get_code_gen().get_nodal_dimension()):
                    raise RuntimeError(
                        "Cannot add one interface element instance to different bulk equations. Create a new interface element instance instead")
            else:
                icg._set_problem(self._problem)
                assert self._codegen
                icg._set_nodal_dimension(self._codegen.get_nodal_dimension())
                icg._set_lagrangian_dimension(
                    self._codegen.get_lagrangian_dimension())
                icg._coordinate_space = self._codegen._coordinate_space
                icg._do_define_fields(self._codegen.dimension - 1)

        # Second loop for interface meshes on integrace meshes
        for _, imsh1 in self._interfacemeshes.items():
            for b, imsh in imsh1._interfacemeshes.items():
                if not (b in bn) and b!="_internal_facets_":
                    raise RuntimeError("Boundary " + b + " not in mesh")
                ieqs = imsh._eqtree._equations
                icg = imsh._eqtree._codegen
                assert icg is not None
                assert ieqs is not None
                if icg._code is not None:
                    if (not self._was_remeshed) and (ieqs._problem != self.get_problem() or icg.get_parent_domain() != self._codegen or icg.get_nodal_dimension() != self._eqtree.get_code_gen().get_nodal_dimension()):
                        print("was remeshed", self._was_remeshed)
                        print(ieqs._problem,self.get_problem())
                        if ieqs is not None and self._eqtree._codegen is not None:
                            print(ieqs._problem, self.get_problem())
                            print(icg.get_parent_domain(), self._codegen)
                            print(icg.get_nodal_dimension(),
                                self._eqtree._codegen.get_nodal_dimension())
                        raise RuntimeError(
                            "Cannot add one interface element instance to different bulk equations. Create a new interface element instance instead. Boundary "+b)
                else:
                    assert self._codegen is not None
                    icg._set_problem(self._problem)
                    icg._set_nodal_dimension(
                        self._codegen.get_nodal_dimension())
                    icg._set_lagrangian_dimension(
                        self._codegen.get_lagrangian_dimension())
                    icg._coordinate_space = self._codegen._coordinate_space
                    icg._do_define_fields(self._codegen.dimension - 2)

    def _compile_bulk_equations(self) -> _pyoomph.DynamicBulkElementInstance:
        assert self._problem is not None
        assert self._codegen is not None
        eqs = self._eqtree.get_equations()
        self._codegen._set_problem(self._problem)
        mesh = self._eqtree._mesh
        if mesh is not None:
            assert isinstance(
                mesh, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
            templ = mesh._templatemesh
            # Get point to evaluate the IC and DBC to check whether it is a numeric value (Can prevent problems if somethink like 1/x is used)
            if templ is not None:
                templ = templ.get_template()
                dom = templ.get_domain(self._name)
                refpos = dom._get_reference_position_for_IC_and_DBC(set())
                refnorm=[0.1,0.1,0.1] # TODO: Get a right reference normal
                t = self._problem.time_pt().time()
                self._codegen._set_reference_point_for_IC_and_DBC(
                    refpos[0], refpos[1], refpos[2], t,refnorm[0],refnorm[1],refnorm[2])
        self._eqtree._equations._set_current_codegen(self._eqtree._codegen)
        #self._problem.before_compile_equations(self._eqtree._equations)
        eqs.before_finalization(self._codegen)
        self._codegen._finalise()
        eqs.before_compilation(self._codegen)        
        self._codegen._code = self._problem.compile_bulk_element_code(
            self._codegen, assert_spatial_mesh(self), self._name)
        self._templatemesh.get_domain(
            self._name).set_element_code(self._codegen.get_code())
        self._finalise_creation()
#        self._transfer_mesh_functions()
        eqs.after_compilation(self._codegen)
        mpi_barrier()
        return self._codegen.get_code()

    def _construct_after_remesh(self):
        assert self._codegen is not None
        self._templatemesh.get_domain(
            self._name).set_element_code(self._codegen.get_code())
        self._finalise_creation()
        # self._transfer_mesh_functions()

    def get_dimension(self) -> int:
        raise NotImplementedError("Please specify")

    def define_state_file(self, state: "DumpFile",additional_info={}):
        # Write/load the template information
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))

        old_ordering = True
        # Refinement pattern
        if state.save:
            refinementS: List[NPInt32Array] = self.get_refinement_pattern()
            nref = len(refinementS)
            state.int_data(lambda: nref, lambda v: v)
            for n in range(nref):
                state.numpy_data(lambda: refinementS[n], lambda v: v)
        else:
            nref = state.int_data(lambda: 0, lambda v: v)
            refinementL: List[List[int]] = []
            for n in range(nref):
                refinementL.append(list(state.numpy_data(
                    lambda: refinementL[n], lambda v: v)))  # type:ignore
            # print("REFINEMEHT",refinement,self.nelement())
            self.refine_base_mesh(refinementL)
            self.reorder_nodes(old_ordering)

        # Check the element num and the node num
        nelem = self.nelement()
        nnode = self.nnode()
        nodaldim = self.get_code_gen().get_nodal_dimension()
        lagrdim = self.get_code_gen().get_lagrangian_dimension()
        nelem = state.int_data(
            lambda: nelem, lambda n: state.assert_equal(n, nelem))  # type:ignore
        nnode = state.int_data(
            lambda: nnode, lambda n: state.assert_equal(n, nnode))  # type:ignore
        nodaldim = state.int_data(
            lambda: nodaldim, lambda n: state.assert_equal(n, nodaldim))  # type:ignore
        lagrdim = state.int_data(
            lambda: lagrdim, lambda n: state.assert_equal(n, lagrdim))  # type:ignore

        # Now store the nodal data

        # Create the interfaces to make sure that the additional dofs gets assigned
        if not state.save:
            for _, im in self._interfacemeshes.items():
                im.rebuild_after_adapt()

        if state.save:
            mdata = self._save_state()
            state.numpy_data(lambda: mdata, lambda v: v)  # type:ignore
        else:
            mdata = state.numpy_data(lambda: 0, lambda v: v)  # type:ignore
            # print("LOAD DATA",mdata)
            self._load_state(mdata)  # type:ignore

        numtracercols = len(self._tracers)
        state.int_data(lambda: numtracercols,
                       lambda n: state.assert_equal(n, numtracercols))
        for tname in sorted(self._tracers.keys()):
            tcol = self.get_tracers(tname)
            assert tcol is not None
            state.string_data(
                lambda: tname, lambda tn: state.assert_equal(tn, tname))
            if state.save:
                pdata, tdata = tcol._save_state()
                state.numpy_data(lambda: pdata, lambda v: v)  # type:ignore
                state.numpy_data(lambda: tdata, lambda v: v)  # type:ignore
            else:
                pdata = state.numpy_data(lambda: 0, lambda v: v)  # type:ignore
                tdata = state.numpy_data(lambda: 0, lambda v: v)  # type:ignore
                tcol._load_state(pdata, tdata)  # type:ignore
    
    
    def _evaluate_extremum_wrapper(self,name:Union[str,List[str]],sign:int,dimensional:bool=True,as_float:bool=False,return_x:bool=True):
        if not isinstance(name,str):
            if return_x:
                raise RuntimeError("Please set return_x=False for multiple extremum evaluations (or call them one by one)")
            return [self._evaluate_extremum_wrapper(n,sign,dimensional=dimensional,as_float=as_float,return_x=False) for n in name]
        flags=0
        if dimensional:
            flags|=1
        val,s,elem=self._evaluate_extremum(name,sign,flags)        
        #print(val,s,elem)
        if return_x:
            x=elem.get_interpolated_position_at_s(0,s,False)
            #print("X",x)
            if dimensional:
                SS=self.get_problem().get_scaling("spatial")
                x=[xc*SS for xc in x]
        if not dimensional:
            val=float(val)
        else:
            if as_float:
                factor, _, _, _ = _pyoomph.GiNaC_collect_units(val)
                val = float(factor)
                if return_x:
                    xn=[]
                    for xc in x:
                        factor, _, _, _ = _pyoomph.GiNaC_collect_units(xc)
                        xn.append(float(factor))
                    x=xn                
        if return_x:
            return val,x
        else:
            return val
                    
    def evaluate_maximum(self,name:Union[str,List[str]],dimensional:bool=True,as_float:bool=False,return_x:bool=False)->Union[ExpressionOrNum,List[ExpressionOrNum],Tuple[ExpressionOrNum,List[ExpressionOrNum]]]:
        return self._evaluate_extremum_wrapper(name,1,dimensional=dimensional,as_float=as_float,return_x=return_x)
           
    def evaluate_minimum(self,name:Union[str,List[str]],dimensional:bool=True,as_float:bool=False,return_x:bool=False)->Union[ExpressionOrNum,List[ExpressionOrNum],Tuple[ExpressionOrNum,List[ExpressionOrNum]]]:
        return self._evaluate_extremum_wrapper(name,-1,dimensional=dimensional,as_float=as_float,return_x=return_x)


class MeshFromTemplate1d(_pyoomph.TemplatedMeshBase1d, MeshFromTemplateBase):
    def __init__(self, problem: "Problem", templatemesh: MeshTemplate, domainname: str, elementtype: "EquationTree", previous_mesh: Optional[MeshFromTemplateBase] = None):
        super(MeshFromTemplate1d, self).__init__()
        MeshFromTemplateBase.__init__(
            self, problem, templatemesh, domainname, elementtype, previous_mesh=previous_mesh)

    def get_dimension(self) -> int:
        return 1

    def get_problem(self) -> "Problem":
        from ..generic.problem import Problem
        pr = self._get_problem()
        assert isinstance(pr, Problem)
        return pr


class MeshFromTemplate2d(_pyoomph.TemplatedMeshBase2d, MeshFromTemplateBase):
    def __init__(self, problem: "Problem", templatemesh: MeshTemplate, domainname: str, elementtype: "EquationTree", previous_mesh: Optional[MeshFromTemplateBase] = None):
        super(MeshFromTemplate2d, self).__init__()
        MeshFromTemplateBase.__init__(
            self, problem, templatemesh, domainname, elementtype, previous_mesh=previous_mesh)

    def get_dimension(self)->int:
        return 2

    def get_problem(self) -> "Problem":
        from ..generic.problem import Problem
        pr = self._get_problem()
        assert isinstance(pr, Problem)
        return pr


class MeshFromTemplate3d(_pyoomph.TemplatedMeshBase3d, MeshFromTemplateBase):
    def __init__(self, problem: "Problem", templatemesh: MeshTemplate, domainname: str, elementtype: "EquationTree", previous_mesh: Optional[MeshFromTemplateBase] = None):
        super(MeshFromTemplate3d, self).__init__()
        MeshFromTemplateBase.__init__(
            self, problem, templatemesh, domainname, elementtype, previous_mesh=previous_mesh)

    def get_dimension(self)->int:
        return 3

    def get_problem(self) -> "Problem":
        from ..generic.problem import Problem
        pr = self._get_problem()
        assert isinstance(pr, Problem)
        return pr


def MeshFromTemplate(problem: "Problem", templatemesh: MeshTemplate, domainname: str, eqtree: "EquationTree", previous_mesh: Optional[MeshFromTemplateBase] = None) -> Union[MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d]:
    if not templatemesh.has_domain(domainname):
        raise RuntimeError("There is no domain '" +
                           domainname + "' defined in this mesh")
    coll = templatemesh.get_domain(domainname)

    edim = coll.get_element_dimension()

    # print("COLL ", domainname, coll, edim)

    if edim == -1:
        raise RuntimeError("The domain '" + domainname + "' has no elements")
    elif edim == 1:
        return MeshFromTemplate1d(problem, templatemesh, domainname, eqtree, previous_mesh=previous_mesh)
    elif edim == 2:
        return MeshFromTemplate2d(problem, templatemesh, domainname, eqtree, previous_mesh=previous_mesh)
    else:
        return MeshFromTemplate3d(problem, templatemesh, domainname, eqtree, previous_mesh=previous_mesh)


######################################################

class InterfaceMesh(_pyoomph.InterfaceMesh, BaseMesh):
    def __init__(self, problem: "Problem", parent: "AnySpatialMesh", intername: str, eqtree: "EquationTree", previous_mesh: Optional["InterfaceMesh"] = None):
        super(InterfaceMesh, self).__init__()
        BaseMesh.__init__(self)
        # _pyoomph.InterfaceMesh.__init__(self,problem)
        # super().__init__(problem)
        self._problem = problem
        self._set_problem(problem, eqtree.get_code_gen()._code)
        self._parent: "AnySpatialMesh" = parent
        self._opposite_interface_mesh: Optional["InterfaceMesh"] = None
        self._interface_name: str = intername
        self._codegen = eqtree.get_code_gen()
        self._eqtree: "EquationTree" = eqtree
        self._eqtree._mesh = self
        self._error_estimator = _pyoomph.Z2ErrorEstimator()
        self._error_estimator.use_Lagrangian = False
        self.ignore_initial_condition = False
        self.set_spatial_error_estimator_pt(self._error_estimator)
        if previous_mesh is not None:
            self._setup_information_from_old_mesh(previous_mesh)

        for n, eqtree in self._eqtree.get_children().items():
            pinter = None
            if previous_mesh is not None:
                pinter = previous_mesh.get_mesh(n)
                assert isinstance(pinter, InterfaceMesh)
            self._interfacemeshes[n] = InterfaceMesh(
                self._problem, self, n, eqtree, previous_mesh=pinter)

    def get_problem(self) -> "Problem":
        from ..generic.problem import Problem
        pr = self._get_problem()
        assert isinstance(pr, Problem)
        return pr

    def refinement_possible(self) -> bool:
        p = self
        while isinstance(p, InterfaceMesh):
            p = p._parent
        return p.refinement_possible()

    def get_dimension(self) -> int:
        return self._parent.get_dimension()-1

    def setup_initial_conditions_with_interfaces(self, resetting_first_step: bool, ic_name: str):
        if self.ignore_initial_condition:
            return
        self.setup_initial_conditions(resetting_first_step, ic_name)
        for _, im in self._interfacemeshes.items():
            im.setup_initial_conditions_with_interfaces(
                resetting_first_step, ic_name)
        # for n,im in self._interfacemeshes.items():
         #   im.setup_initial_conditions_with_interfaces(self)

    def get_name(self) -> str:
        return self._interface_name

    def get_full_name(self) -> str:
        myname = self.get_name()
        pname = self._parent.get_full_name()
        return pname+"/"+myname

    def _override_bulk_errors_where_necessary(self):
        nelem = self.nelement()
        if nelem == 0:
            return
        el0 = self.element_pt(0)
        opposite_required = (self.element_pt(0).get_opposite_bulk_element(
        ) is not None) and (self._opposite_interface_mesh is not None)
        # print("OPP REQ",opposite_required)
        own_error_estim = True
        if el0.num_Z2_flux_terms() == 0:
            own_error_estim = False
        # print(dir(el0))
        if el0.nnode() <= 1:  # TODO: That's not the best here... Probably find some other way to refine. Z2 required dim>0
            own_error_estim = False
        if own_error_estim:
            self._enable_adaptation()
            ierrs = self.get_elemental_errors()
            self._disable_adaptation()
        else:
            # ierrs:NPIntArray=numpy.zeros((nelem)) #type:ignore
            ierrs = [0.0]*nelem
        # Merge with elemental override
        for i, e in enumerate(self.elements()):
            ierrs[i] = max(e._elemental_error_max_override, ierrs[i])
        if opposite_required:
            for i, e in enumerate(self.elements()):
                ierrs[i] = max(e.get_opposite_interface_element(
                )._elemental_error_max_override, ierrs[i])

        imax_err = self.max_permitted_error
        imin_err = self.min_permitted_error
        # print(ierrs)

        def do_on_bulk_mesh(bmesh: AnySpatialMesh, iname: str, opposite: bool):
            must_brefine = 100 * bmesh.max_permitted_error
            may_not_unrefine = 0.5 * \
                (bmesh.max_permitted_error+bmesh.min_permitted_error)
            for i, ie in enumerate(self.elements()):
                if ierrs[i] > imax_err:
                    if opposite:
                        ie.get_opposite_bulk_element()._elemental_error_max_override = must_brefine
                    else:
                        ie.get_bulk_element()._elemental_error_max_override = must_brefine
                elif ierrs[i] > imin_err:
                    if opposite:
                        ov = ie.get_opposite_bulk_element()
                        ov._elemental_error_max_override = max(
                            ov._elemental_error_max_override, may_not_unrefine)
                    else:
                        ov = ie.get_bulk_element()
                        ov._elemental_error_max_override = max(
                            ov._elemental_error_max_override, may_not_unrefine)
            bnames = bmesh.get_boundary_names()
            if iname!="_internal_facets_":
                bind = bnames.index(iname)
                bmesh._enlarge_elemental_error_max_override_to_only_nodal_connected_elems(bind)

        do_on_bulk_mesh(self._parent, self._interface_name, False)

        if opposite_required:
            assert self._opposite_interface_mesh is not None
            do_on_bulk_mesh(self._opposite_interface_mesh._parent,
                            self._opposite_interface_mesh._interface_name, True)

    def get_nodal_field_indices(self) -> Dict[str, int]:
        return self.get_code_gen().get_code().get_nodal_field_indices()

    def _setup_output_scales(self):
        codegen = self.get_code_gen()
        code = codegen.get_code()
        _, unit, _, _ = _pyoomph.GiNaC_collect_units(
            codegen.expand_placeholders(codegen.get_scaling("spatial"), False))
        self.set_output_scale("spatial", unit, code)  # TODO
        for k in itertools.chain(code.get_nodal_field_indices().keys(), code.get_elemental_field_indices().keys()):
            s = codegen.get_scaling(k)
            s = codegen.expand_placeholders(s, False)
            _, unit, _, _ = _pyoomph.GiNaC_collect_units(s)
            self.set_output_scale(k, unit, code)

    def _pre_compile(self):
        self.get_code_gen()._index_fields()


    def _compile(self):
        from ..generic.codegen import FiniteElementCodeGenerator
        name: str = self._interface_name
        curri: AnySpatialMesh = self._parent
        boundname_set: Set[str] = {name}
        while isinstance(curri, InterfaceMesh):
            assert curri._interface_name is not None
            name = curri._interface_name + "__" + name
            boundname_set.add(curri._interface_name)
            curri = cast(AnySpatialMesh, curri._parent)
        # assert isinstance(curri,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d))
        assert isinstance(curri, MeshFromTemplateBase)
        if not self.get_problem().is_quiet():
            print("Generating interface code "+curri._name+" "+name)
        templ: Optional[MeshTemplate] = curri._templatemesh
        # Get point to evaluate the IC and DBC to check whether it is a numeric value (Can prevent problems if somethink like 1/x is used)
        if templ is not None:
            templ = templ.get_template()
            dom = templ.get_domain(curri._name)
            bnames = curri.get_boundary_names()
            if boundname_set=={"_internal_facets_"}:
                # Just select anything
                #print(boundname_set,bnames)
                bind_set = {0} #TODO: improve this potentially for codim>1 interface mesh
            elif "_internal_facets_" in boundname_set:
                bind_set = {bnames.index(n) for n in boundname_set if n!="_internal_facets_"}
            else:
                bind_set = {bnames.index(n) for n in boundname_set}
            refpos = dom._get_reference_position_for_IC_and_DBC(bind_set)
            refnorm=[0.1,0.1,0.1] # TODO: Get a right reference norm
            t = self._problem.time_pt().time()
            self.get_code_gen()._set_reference_point_for_IC_and_DBC(
                refpos[0], refpos[1], refpos[2], t,refnorm[0],refnorm[1],refnorm[2])

        oppi = self.get_code_gen()._get_opposite_interface()

        if oppi is not None:
            assert isinstance(oppi, FiniteElementCodeGenerator)
            old_oppcg = oppi.get_equations()._get_current_codegen()
            oppi.get_equations()._set_current_codegen(oppi)
            oppblk = oppi.get_parent_domain()
#            assert isinstance(oppblk, FiniteElementCodeGenerator)
            if oppblk is not None:
                old_oppblkcg = oppblk.get_equations()._get_current_codegen()                
                oppblk.get_equations()._set_current_codegen(oppblk)

        blk = self.get_code_gen().get_parent_domain()
        assert blk is not None
        oldblk = blk.get_equations()._get_current_codegen()
        blk.get_equations()._set_current_codegen(blk)

        oldmy = self._eqtree.get_equations()._get_current_codegen()
        self._eqtree.get_equations()._set_current_codegen(self._codegen)

        assert self._codegen is not None
        self._codegen._mesh = self
        eqs = self._eqtree.get_equations()


        #self._problem.before_compile_equations(self._eqtree)
        eqs.before_finalization(self._codegen)
        self._codegen._finalise()


        # Transfer the facet contributions
        if "_internal_facets_" in self._eqtree._children.keys():
            internal_eqs=self._eqtree.get_child("_internal_facets_").get_equations()            
            for destination,int_contrib in eqs._interior_facet_residuals.items():
                if destination in internal_eqs._additional_residuals.keys():
                        internal_eqs._additional_residuals[destination]+=int_contrib
                else:
                        internal_eqs._additional_residuals[destination]=int_contrib                        

        eqs._set_current_codegen(self._codegen)
        eqs.before_compilation(self._codegen)

        self._codegen._code = self._codegen.get_problem().compile_bulk_element_code(self._codegen, self, curri._name + "__" + name)  

        self._set_problem(self.get_problem(),
                          self._codegen._code)  # type:ignore
        if oppi is not None:
            oppi.get_equations()._set_current_codegen(old_oppcg)  # type:ignore
            oppblk.get_equations()._set_current_codegen(old_oppblkcg)  # type:ignore

        blk.get_equations()._set_current_codegen(oldblk)
        self._eqtree._equations._set_current_codegen(oldmy)  # type:ignore

        self._eqtree._equations.after_compilation(self._codegen)  # type:ignore
        mpi_barrier()

    def nodes(self) -> Iterator[_pyoomph.Node]:
        uniqnods: Set[Node] = set()
        for e in self.elements():
            nn = e.nnode()
            for ni in range(nn):
                n = e.node_pt(ni)
                uniqnods.add(n)
        for n in uniqnods:
            yield n

    def boundary_nodes(self, b: str) -> Iterable[_pyoomph.Node]:
        uniqnods: Set[Node] = set()
        bind = self.get_boundary_index(b)
        for e in self.boundary_elements(b):
            nn = e.nnode()
            for ni in range(nn):
                n = e.node_pt(ni)
                if n.is_on_boundary(bind):
                    uniqnods.add(n)
        return uniqnods

    def nodes_on_both_sides(self) -> Generator[Tuple[_pyoomph.Node, _pyoomph.Node], None, None]:
        nodemap: Dict[Node, Node] = {}
        for e in self.elements():
            nn = e.nnode()
            for ni in range(nn):
                n = e.node_pt(ni)
                # print(e.get_opposite_bulk_element())
                no = e.opposite_node_pt(ni)
                nodemap[n] = no
        for ni, no in nodemap.items():
            yield ni, no

    def _reset_elemental_error_max_override(self):
        for e in self.elements():
            e._elemental_error_max_override = 0.0


    def _evaluate_extremum_wrapper(self,name:Union[str,List[str]],sign:int,dimensional:bool=True,as_float:bool=False,return_x:bool=True):
        if not isinstance(name,str):
            if return_x:
                raise RuntimeError("Please set return_x=False for multiple extremum evaluations (or call them one by one)")
            return [self._evaluate_extremum_wrapper(n,sign,dimensional=dimensional,as_float=as_float,return_x=False) for n in name]
        flags=0
        if dimensional:
            flags|=1
        val,s,elem=self._evaluate_extremum(name,sign,flags)        
        #print(val,s,elem)
        if return_x:
            x=elem.get_interpolated_position_at_s(0,s,False)
            #print("X",x)
            if dimensional:
                SS=self.get_problem().get_scaling("spatial")
                x=[xc*SS for xc in x]
        if not dimensional:
            val=float(val)
        else:
            if as_float:
                factor, _, _, _ = _pyoomph.GiNaC_collect_units(val)
                val = float(factor)
                if return_x:
                    xn=[]
                    for xc in x:
                        factor, _, _, _ = _pyoomph.GiNaC_collect_units(xc)
                        xn.append(float(factor))
                    x=xn                
        if return_x:
            return val,x
        else:
            return val
                    
    def evaluate_maximum(self,name:Union[str,List[str]],dimensional:bool=True,as_float:bool=False,return_x:bool=False)->Union[ExpressionOrNum,List[ExpressionOrNum],Tuple[ExpressionOrNum,List[ExpressionOrNum]]]:
        return self._evaluate_extremum_wrapper(name,1,dimensional=dimensional,as_float=as_float,return_x=return_x)
           
    def evaluate_minimum(self,name:Union[str,List[str]],dimensional:bool=True,as_float:bool=False,return_x:bool=False)->Union[ExpressionOrNum,List[ExpressionOrNum],Tuple[ExpressionOrNum,List[ExpressionOrNum]]]:
        return self._evaluate_extremum_wrapper(name,-1,dimensional=dimensional,as_float=as_float,return_x=return_x)


class ODEStorageMesh(_pyoomph.ODEStorageMesh):
    """
    A sort of a mesh storing ODE values. This is not a real mesh, but a container for ODE values.
    """
    def __init__(self, problem: "Problem", eqtree: "EquationTree", domainname: str):
        super().__init__()
        self._problem: "Problem" = problem
        self._eqtree: Optional["EquationTree"] = eqtree
        self._eqtree._mesh = self  # type:ignore
        self._codegen = eqtree._codegen  # type:ignore
        self._name = domainname
        ocg = self._codegen.get_equations()._get_current_codegen()  # type:ignore
        self._codegen.get_equations()._set_current_codegen(self._codegen)  # type:ignore
        self._codegen._do_define_fields(0)  # type:ignore
        self._codegen._index_fields()  # type:ignore
        self._codegen.get_equations()._set_current_codegen(ocg)  # type:ignore
        self._element = None
        self.ignore_initial_condition=False
        for _, eqtree in self._eqtree.get_children().items():
            raise RuntimeError("ODE domains may not have children yet")

    def get_code_gen(self) -> "FiniteElementCodeGenerator":
        assert self._codegen is not None
        return self._codegen

    def get_problem(self) -> "Problem":
        return self._problem

    def get_bulk_mesh(self):
        return None
    
    def _setup_output_scales(self):
        ocg = self._codegen.get_equations()._get_current_codegen()  
        self._codegen.get_equations()._set_current_codegen(self._codegen)  
        _, indices = self._element.to_numpy()
        scales: List[ExpressionOrNum] = [1.0] * len(indices)
        for k, i in indices.items():
            s = self._eqtree.get_equations().get_scaling(k)
            if isinstance(s, Expression):
                factor, _, _, _ = _pyoomph.GiNaC_collect_units(s)  
                s = float(factor)
            scales[i]=s
        self._codegen.get_equations()._set_current_codegen(ocg) 

    def _compile_bulk_equations(self) -> _pyoomph.DynamicBulkElementInstance:
        assert self._eqtree is not None
        assert self._codegen is not None
        self._codegen._set_problem(self._problem)  

        ocg = self._codegen.get_equations()._get_current_codegen()  
        self._codegen.get_equations()._set_current_codegen(self._codegen)  

        eqs=self._eqtree.get_equations()
        #self._problem.before_compile_equations(self._eqtree)
        eqs.before_finalization(self._codegen)  
        eqs._problem=self._problem
        self._codegen._finalise()  
        self._codegen.get_equations()._set_current_codegen(self._codegen)  

        eqs.before_compilation(self._codegen)  
        self._codegen._code = self._problem.compile_bulk_element_code(self._codegen, self, self._name)  
        self._element = _pyoomph.BulkElementODE0d.construct_new(self._codegen.get_code(), self._problem.timestepper)  
        self._set_problem(self._problem, self._codegen._code)  
        self._setup_output_scales()
        self._add_ODE("ODE", self._element)
        # self._transfer_mesh_functions()
        self._codegen.get_equations()._set_current_codegen(ocg) 
        self._eqtree.get_equations().after_compilation(self._codegen)
        mpi_barrier()
        return self.get_code_gen().get_code()

    def setup_initial_conditions_with_interfaces(self, resetting_first_step: bool, ic_name: str):
        if self.ignore_initial_condition:
            return
        self.setup_initial_conditions(resetting_first_step, ic_name)

    def elements(self) -> Iterator[_pyoomph.OomphGeneralisedElement]:
        numelems = self.nelement()
        for i in range(numelems):
            yield self.element_pt(i)

    def evaluate_all_observables(self) -> Dict[str, ExpressionOrNum]:
        return BaseMesh.evaluate_all_observables(self)  # type:ignore

    def get_element(self) -> _pyoomph.BulkElementODE0d:
        return self._get_ODE("ODE")

    @overload
    def get_value(self, name: str, *,dimensional: bool=..., as_float: Literal[False]=...) -> _pyoomph.Expression: ...

    @overload
    def get_value(self, name: str, *, dimensional: bool=..., as_float: Literal[True]) -> float: ...

    @overload
    def get_value(self, name: NameStrSequence, *,dimensional: bool=..., as_float: Literal[False]=...) -> Tuple[_pyoomph.Expression, ...]: ...
    
    @overload
    def get_value(self, name: NameStrSequence, *,dimensional: bool=..., as_float: Literal[True]) -> Tuple[float, ...]: ...

    def get_value(self, name: Union[str, NameStrSequence], *, dimensional: bool = True, as_float: bool = False) -> Union[_pyoomph.Expression, float, Tuple[float, ...], Tuple[_pyoomph.Expression, ...]]:
        """
        Get the value(s) associated with the given name(s) from the ODE.

        Args:
            name (Union[str, pyoomph.expressions.NameStrSequence]): The name(s) of the value(s) to retrieve.
            dimensional (bool, optional): Whether to return the value(s) in dimensional form. Defaults to True.
            as_float (bool, optional): Whether to return the value(s) as float(s). Defaults to False.

        Returns:
            Union[ExpressionOrNum, Tuple[ExpressionOrNum, ...]]: The value(s) associated with the given name(s).

        Raises:
            RuntimeError: If the ODE has no value with the given name(s).
        """
        assert self._eqtree is not None
        ode = self._get_ODE("ODE")
        vals, inds = ode.to_numpy()
        if isinstance(name, str):
            names = [name]
        else:
            names = name
        res = []
        for n in names:
            if n not in inds.keys():
                raise RuntimeError("The ODE has no value " + str(n))
            entry = vals[inds[n]]
            # Force tiny onces to zero
            if abs(entry)<1e-200:
                entry=0.0
            # Scaling
            if dimensional:
                S = self._eqtree.get_code_gen().get_scaling(n)
                entry *= S
                entry = self.get_code_gen().expand_placeholders(entry, False)
                if as_float:
                    factor, _, _, _ = _pyoomph.GiNaC_collect_units(entry)
                    entry = float(factor)
            res.append(entry)  # type:ignore
        if len(res) == 1:
            return res[0]  # type:ignore
        else:
            return res  # type:ignore

    def set_value(self, dimensional: bool = True, **namvals: ExpressionOrNum) -> None:
        """
        Set the current values of ODE variables.

        Args:
            dimensional (bool, optional): Flag indicating whether the values should be set in dimensional form. 
                                          Defaults to True.
            **namvals: Keyword arguments representing the names and values of the ODE variables to be set.

        Raises:
            RuntimeError: If the ODE variable does not exist.
            RuntimeError: If the value cannot be converted to the required unit.

        Returns:
            None
        """
        assert self._eqtree is not None
        ode = self._get_ODE("ODE")
        _, inds = ode.to_numpy()
        for n, v in namvals.items():
            if not n in inds.keys():
                raise RuntimeError("The ODE has no value "+str(n))
            val = v
            if dimensional:
                S = self._eqtree.get_code_gen().get_scaling(n)
                val /= S
                try:
                    val = float(val)
                except:
                    _, unit, _, _ = _pyoomph.GiNaC_collect_units(S)
                    raise RuntimeError("Cannot convert the value "+str(v) +
                                       " to the required unit of "+str(unit)+" to set "+str(n))
            assert isinstance(val, (float, int))
            ode.internal_data_pt(inds[n]).set_value(0, val)

    def define_state_file(self, state: "DumpFile",additional_info={}):
        ode = self._get_ODE("ODE")
        _, inds = ode.to_numpy()
        inds_sorted = list(sorted(list(inds)))
        numinds = len(inds_sorted)
        numinds = state.int_data(
            lambda: numinds, lambda l: state.assert_equal(l, numinds))  # type:ignore
        for nind in range(numinds):
            fname = inds_sorted[nind]
            fname = state.string_data(lambda: fname, lambda s: s)
            assert fname in inds.keys()
            ind = inds[fname]
            data = ode.internal_data_pt(ind)
            assert data.nvalue() == 1
            ntstorage = data.ntstorage()
            ntstorage = state.int_data(
                lambda: ntstorage, lambda t: state.assert_equal(t, ntstorage))  # type:ignore
            for nt in range(ntstorage):
                state.float_data(lambda: data.value_at_t(
                    nt, 0), lambda v: data.set_value_at_t(nt, 0, v))  # type:ignore

    def get_name(self) -> str:
        return self._name

    def get_full_name(self) -> str:
        return self._name

    def set_dirichlet_active(self, **kwargs: bool):
        for k, v in kwargs.items():
            if (v is True) or (v is False):
                self._set_dirichlet_active(k, v)
            else:
                raise ValueError(
                    "Please set Dirichlet active either to True or False")
