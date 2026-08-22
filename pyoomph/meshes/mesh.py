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
 
import abc
import inspect
import math
import os.path
import weakref

from ..generic.mpi import mpi_barrier, get_mpi_nproc, get_mpi_any, get_mpi_rank, get_mpi_world_comm, mpi_share_root_failure

from ..typings import *


import numpy

from .. import _pyoomph_core as _pyoomph

from ..expressions.generic import Expression, ExpressionOrNum, is_zero, NameStrSequence

from .ordering import SortAlongAxis, sort_line_segments

import itertools


if TYPE_CHECKING:
    from ..generic.problem import Problem, Z2ErrorEstimator
    from ..output.states import DumpFile
    from .remesher import RemesherBase
    from ..generic.codegen import EquationTree, FiniteElementCodeGenerator


Node = _pyoomph.Node
Element=_pyoomph.OomphGeneralisedElement
AnySpatialMesh:TypeAlias = "InterfaceMesh | MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d"
AnyMesh:TypeAlias = "AnySpatialMesh | ODEStorageMesh"
# Union of the concrete "mesh built from a MeshTemplate" classes. Used in type annotations
# in place of MeshFromTemplateBase: nanobind does not support combining a bound C++ base
# (e.g. _pyoomph.TemplatedMeshBase1d) with an additional Python base in the same class, so
# MeshFromTemplateBase is no longer a real (nominal) base of MeshFromTemplate1d/2d/3d -- see
# the _install_mixin() docstring below for how shared behavior and isinstance() still work.
BulkTemplateMesh:TypeAlias = "MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d"


def assert_spatial_mesh(mesh: "AnyMesh | MeshFromTemplateBase | None") -> AnySpatialMesh:
    if mesh is None:
        raise RuntimeError("Mesh is None")
    elif isinstance(mesh, ODEStorageMesh):
        raise RuntimeError("Expected spatial mesh, but got ODEStorageMesh")
    elif isinstance(mesh, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d, InterfaceMesh)):
        return cast(AnySpatialMesh,mesh)
    else:
        raise RuntimeError("Should not end up here")


def _install_mixin(target_cls: type, mixin_cls: type) -> type:
    """
    Copies every method/attribute defined anywhere in mixin_cls's MRO onto target_cls
    (without overriding anything target_cls already defines itself), and registers
    target_cls as a virtual subclass of every ABC found in mixin_cls's MRO.

    This replaces plain Python multiple inheritance (e.g. ``class MeshFromTemplate1d(
    _pyoomph.TemplatedMeshBase1d, MeshFromTemplateBase)``) for classes that also derive
    from a nanobind-bound C++ type: nanobind only supports a single linear chain of C++
    bases and does not support combining a bound base with an additional Python base in
    the same class. Copying the mixin's methods directly into target_cls's __dict__
    gives every instance the same attribute lookup result as real inheritance would,
    without adding mixin_cls to target_cls.__bases__. The ABC registration keeps
    isinstance(x, mixin_cls) (used throughout the codebase, e.g. to recognize any
    MeshFromTemplate1d/2d/3d instance as a MeshFromTemplateBase) working, even though
    mixin_cls is no longer a real ancestor. Static type checkers do not see this virtual
    relationship, so use a Union type alias (e.g. BulkTemplateMesh) instead of mixin_cls
    itself in type annotations for values that are actually instances of target_cls.
    """
    for base in reversed(mixin_cls.__mro__):
        if base in (object, abc.ABC):
            continue
        for name, value in vars(base).items():
            if name in ("__dict__", "__weakref__", "__module__", "__doc__"):
                continue
            if name not in target_cls.__dict__:
                setattr(target_cls, name, value)
        if isinstance(base, abc.ABCMeta):
            base.register(target_cls)
    return target_cls


class BaseMesh(abc.ABC):
    def __init__(self):
        # self._interfacial_elements=dict()
        self._interfacemeshes: dict[str, "InterfaceMesh"] = dict()
        self._outputscales = {}
        self.initial_uniform_refinements = 0
        self._initial_interface_refinement = {}
        # Tracer particles -> name to tracer instance
        self._tracers: dict[str, _pyoomph.TracerCollection] = {}
        self._codegen: "FiniteElementCodeGenerator | None"
        self._eqtree: "EquationTree"

    def get_code_gen(self) -> "FiniteElementCodeGenerator":
        assert self._codegen is not None
        return self._codegen

    def get_eqtree(self) -> "EquationTree":
        return self._eqtree

    def get_problem(self) -> "Problem":
        # Overridden by every concrete mesh class (MeshFromTemplateBase, InterfaceMesh,
        # ODEStorageMesh); declared here only so that BaseMesh methods (like
        # evaluate_all_observables below) can call self.get_problem() without a static
        # type checker complaining that BaseMesh itself has no such attribute.
        raise NotImplementedError("Please specify")

    def get_tracers(self, name: str = "tracers", error_on_missing: bool = True) -> _pyoomph.TracerCollection | None:
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

    def boundary_intersection_nodes(self, bname1: str, bname2: str) -> list[Node]:
        assert isinstance(self, _pyoomph.Mesh)
        imesh = self.get_mesh(bname1)
        assert imesh is not None
        res: set[Node] = set()
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
        self, b: str, with_directions: Literal[True]) -> Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]: ...

    def boundary_elements(self, b: str, with_directions: bool = False) -> Iterator[_pyoomph.OomphGeneralisedElement] | Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]:
        assert isinstance(self, _pyoomph.Mesh)
        bind = self.get_boundary_names().index(b)
        numelems = self.nboundary_element(bind)
        if with_directions:
            for i in range(numelems):
                yield self.boundary_element_pt(bind, i), self.face_index_at_boundary(bind, i)
        else:
            for i in range(numelems):
                yield self.boundary_element_pt(bind, i)

    def boundary_nodes(self, b: str) -> Iterable[_pyoomph.Node]:
        assert isinstance(self, _pyoomph.Mesh)
        bind = self.get_boundary_names().index(b)
        numelems = self.nboundary_node(bind)
        for i in range(numelems):
            yield self.boundary_node_pt(bind, i)

    @overload
    def get_mesh(self, name: str, return_None_if_not_found: Literal[False] = ...) -> "MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d | InterfaceMesh": ...

    @overload
    def get_mesh(self, name: str, return_None_if_not_found: Literal[True]) -> "MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d | InterfaceMesh | None": ...

    def get_mesh(self, name: str, return_None_if_not_found: bool = False) -> "MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d | InterfaceMesh | None":
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
        cmb: set[str] = set()
        cmb.update(lst)
        cmb.update(deps.keys())

        res:ExpressionOrNum
        if name in lst:
            res = self._evaluate_integral_function(name)
        elif name in self._codegen._dependent_integral_funcs.keys():
            l = deps[name]
            args: list[ExpressionOrNum] = []
            for a in inspect.signature(l).parameters:
                if not (a in cmb):
                    raise RuntimeError("During evaluation of integral observable "+name +
                                       ": Cannot evaluate the observable "+name+". Possible are "+", ".join(sorted(cmb)))
                args.append(self.evaluate_observable(a))
            res = l(*args)

        else:

            raise ValueError("Integral observable "+name +
                             " not defined on this mesh. Possible integral observables on this mesh are: "+", ".join(sorted(cmb)))
        return res

    def evaluate_all_observables(self) -> dict[str, ExpressionOrNum]:
        assert isinstance(self, _pyoomph.Mesh)
        lst = self.list_integral_functions()
        assert self._codegen is not None
        deps = self._codegen._dependent_integral_funcs
        res: dict[str, ExpressionOrNum] = {}
        for name in lst:
            res[name] = self._evaluate_integral_function(name)
        args: dict[str, ExpressionOrNum] = {k: v for k, v in res.items()}
        args["time"] = self.get_problem().get_current_time()
        # A list in declaration order, not a set: the order in which the dependent observables are
        # resolved here is the insertion order of res, and hence the column order of the observable
        # output files. Iterating a set of strings made those columns move from run to run, since
        # Python randomizes that order per process.
        remaining: list[str] = list(deps.keys())
        while len(remaining) > 0:
            torem: set[str] = set()
            for r in remaining:
                # Check if we can evaluate
                l = deps[r]
                all_present = True
                arglist: list[ExpressionOrNum] = []
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
            remaining = [r for r in remaining if r not in torem]
        # Now remove the vector helpers
        for k in self._codegen._dependent_integral_funcs_is_vector_helper.keys():
            del res[k]
        # And expand all numpy arrays
        newres: dict[str, ExpressionOrNum] = {}
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
    def get_maximum_value_of_field(self,fieldname:str,minimum_instead:bool=False,*,dimensional:Literal[False])->float: ...

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
    def __init__(self, sideA: str, sideB: str, problem:"Problem", matchfunc: Callable[[Sequence[float], Sequence[float]], float] | None = None):
        self._sideA = sideA
        self._sideB = sideB
        # Stored as a weakref, not a strong reference: this connection is owned (via
        # MeshTemplate._opposite_interface_connections) by a MeshTemplate that is itself
        # kept alive by the Problem's nb::keep_alive of its meshes - a strong back-reference
        # here would form the same kind of uncollectible cycle fixed for meshes/codegens.
        self._problem_wr=weakref.ref(problem)
        self._match_pos_func: Callable[[Sequence[float], Sequence[float]], float]
        if matchfunc:
            self._match_pos_func = matchfunc
            self._use_kdtree = False
        else:
            self._match_pos_func = lambda a, b: sum([pow(a[j] - b[j], 2) for j in range(len(b))])
            self._use_kdtree = True

    def __str__(self) -> str:
        return "MeshTemplateOppositeInterfaceConnection("+str(self._sideA)+","+str(self._sideB)+")"

    @property
    def _problem(self)->"Problem":
        p=self._problem_wr()
        assert p is not None, "The Problem this connection belonged to has already been destroyed"
        return p

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

        posBmap: dict[tuple[tuple[float, ...], ...],
                      _pyoomph.OomphGeneralisedElement] = {}
        for eB in meshB.elements():
            pos: list[tuple[float, ...]] = []
            for nvj in range(eB.nvertex_node()):
                v = eB.vertex_node_pt(nvj)
                pos.append(tuple([v.x(xi) for xi in range(v.ndim())]))
            pos = sorted(pos)
            posBmap[tuple(pos)] = eB

        offset_vector=meshA.get_opposite_interface_offset_vector()
        rev_offset=[-o for o in offset_vector]
        print("OFFSET VECTOR",offset_vector)
        for eA in meshA.elements():
            pos2find: list[list[float]] = []
            for nvi in range(eA.nvertex_node()):
                v = eA.vertex_node_pt(nvi)
                pos2find.append([v.x(xi) for xi in range(v.ndim())])
            pos2find = sorted(pos2find)
            found = False
            for pB, eB in posBmap.items():
                if len(pB) == len(pos2find):
                    dist = 0.0
                    for i in range(len(pos2find)):
                        dist += self._match_pos_func(pos2find[i], pB[i])
                    # print(pB,len(pB),dist)
                    if dist < 1e-8:
                        eA.set_opposite_interface_element(eB,offset_vector)
                        eB.set_opposite_interface_element(eA,rev_offset)
                        found = True
                        break
            if not found:
                debug_entries: list[tuple[float,tuple[tuple[float, ...], ...]]] = []
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
    Nodes can be also marked to be on particular boundaries with the :py:meth:`add_facet_to_boundary` method.
    """
    def __init__(self):
        super(MeshTemplate, self).__init__()
        self._domains: dict[str, _pyoomph.MeshTemplateElementCollection] = {}
        self._geometry_defined = False
        #: The minimum permitted error for the spatial error estimator. If ``None``, we use the value from the :py:class:`~pyoomph.generic.problem.Problem` object.
        self.min_permitted_error = None
        #: The maximum permitted error for the spatial error estimator. If ``None``, we use the value from the :py:class:`~pyoomph.generic.problem.Problem` object.
        self.max_permitted_error = None
        #: The maximum refinement level for spatial adaptivity. If ``None``, we use the value from the :py:class:`~pyoomph.generic.problem.Problem` object.
        self.max_refinement_level = None
        #: The minimum refinement level for spatial adaptivity. If ``None``, we use the value from the :py:class:`~pyoomph.generic.problem.Problem` object.
        self.min_refinement_level = None
        self._opposite_interface_connections: list[MeshTemplateOppositeInterfaceConnection] = [
        ]
        self._meshfile = None
        #: Must be set to allow for remeshing.
        self.remesher: "RemesherBase | None" = None
        self.auto_find_opposite_interface_connections = True
        self._template_override = None
        self._interior_boundaries: set[str] = set()
        self._macrobounds: list[_pyoomph.MeshTemplateCurvedEntityBase] = []
        self._fntrunk:str | None=None # To be set for remeshing
        self.all_nodes_as_boundary_nodes:bool=False
        
    def _reset(self):
        super(MeshTemplate, self)._reset()
        self._domains = {}
        self._geometry_defined = False
        self._template_override = None
        self._interior_boundaries = set()
        self._macrobounds = []

    def get_problem(self) -> "Problem":
        return self._get_problem() #type:ignore

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
        # We must follow the flag stored in the file here, not our own: since MeshedMeshTemplate attaches a
        # RemesherViaRecreation by default, a template can now have a remesher while an older state file - written
        # when the same script had none - does not contain the remesher counter. Asserting equality would make all
        # those files unloadable, and reading the counter unconditionally would desynchronize the entire file.
        has_remesher = state.int_data(lambda: has_remesher, lambda r: r)
        if has_remesher:
            if self.remesher is None:
                raise RuntimeError("The state file was written with a remesher attached to the mesh template, but the template does not have one anymore")
            self.remesher._cnt = state.int_data(
                lambda: self.remesher._cnt, lambda s: s)  # type:ignore
        if found_mshfile != mshfile:
            # We need to load the remeshed version here
            from . import gmsh
            statedir = os.path.dirname(state.fname)
            fffound_mshfile = os.path.join(statedir, found_mshfile)
            newtempl = gmsh.GmshTemplate(fffound_mshfile)
            newtempl.remesher = self.remesher
            newtempl._do_define_geometry(self.get_problem())
            self._template_override = newtempl
            newtempl.get_template()._meshfile = fffound_mshfile
            # self._meshfile=found_mshfile
        return self.get_template()
        # print("IN STATE FILE "+mshfile,state.fname,)
        # exit()

    def get_opposite_interface(self, side: str) -> str | None:
        for ic in self._opposite_interface_connections:
            if ic._sideA == side:
                return ic._sideB
            elif ic._sideB == side:
                return ic._sideA
        return None

    # Called from C on automatic finding
    def _add_opposite_interface_connection(self, sideA: str, sideB: str):
        self.add_opposite_interface_connection(sideA, sideB)

    def add_opposite_interface_connection(self, sideA: str, sideB: str, matchfunc: Callable[[Sequence[float], Sequence[float]], float] | None = None):
        self._opposite_interface_connections.append(
            MeshTemplateOppositeInterfaceConnection(sideA, sideB, self.get_problem(), matchfunc))

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

    def _remeshing_can_change_the_mesh(self) -> bool:
        # Whether remeshing this template can give a mesh different from the current one. Only a
        # MeshedMeshTemplate can tell (see there), any other remesher is always taken at face value.
        return True

    def available_domains(self) -> set[str]:
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
            self._set_problem(problem)
            self.define_geometry()
            if self.auto_find_opposite_interface_connections:
                self._find_opposite_interface_connections()

    def new_domain(self, name: str, nodal_dimension: int | None = None) -> _pyoomph.MeshTemplateElementCollection:
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
    def nondim_size(self, a: list[ExpressionOrNum]) -> list[float]: ...

    def nondim_size(self, a: ExpressionOrNum | list[ExpressionOrNum]) -> float | list[float]:
        """
        Nondimensionalize a coordinate or a length scale by dividing by the spatial scale of the problem.
        
        Args:
            a: The coordinate or length scale to nondimensionalize.
            
        Returns:
            The arguments divided by the spatial scale of the problem.
        """
        if isinstance(a, list):
            resL: list[float] = []
            for b in a:
                resL.append(self.nondim_size(b))
            return resL
        res: float
        spatial = self.get_problem().get_scaling("spatial")
        try:
            if isinstance(a, float) or isinstance(a, int) or isinstance(a,_pyoomph.GiNaC_GlobalParam) or isinstance(a,numpy.floating) or isinstance(a,numpy.integer):
                res = (float(a / spatial))
            elif isinstance(a, _pyoomph.Expression):  # type:ignore
                res = ((a / spatial).float_value())
            else:
                raise ValueError("Strange spatial argument for a mesh:"+str(a)+" of type "+str(type(a)))
        except RuntimeError as e:
            # Mixing a dimensionless mesh coordinate with a dimensional spatial scale only surfaced as
            # "Cannot convert meter^(-1) to double" from the float conversion, which does not say which
            # of the two sides is the problem.
            raise RuntimeError("Cannot nondimensionalise the mesh coordinate/size "+str(a)+" with the spatial scale "+str(spatial)+" of the problem.\n"
                               "If the problem uses dimensional scales (set_scaling(spatial=...)), all mesh coordinates and sizes must be given dimensionally as well, e.g. size=1*meter.\n"
                               "Original error: "+str(e)) from e
        return res

    def add_nodes(self, *args: Sequence[float]) -> int | tuple[int, ...] | None:
        res: list[int] = []
        for a in args:
            res.append(self.add_node(*a))
        if len(res) == 0:
            return None
        elif len(res) == 1:
            return res[0]
        else:
            return tuple(res)

    def create_curved_entity(self, typ: str, *args: Any, **kwargs: Any)-> _pyoomph.MeshTemplateCurvedEntityBase:
        """
        Creates a curved entity, i.e. the exact geometry that facets of this mesh lie on. Nodes created
        by spatial refinement are then placed on that geometry instead of on the straight-sided
        interpolation between the coarse mesh's nodes.

        Supported types, with the arguments each expects (points may be given either as node indices or
        as coordinate lists):

        * ``"circle_arc"``: the arc's ``start`` and ``end`` as positional arguments, plus ``center``.
        * ``"sphere_part"``: one ``point_on_sphere`` as positional argument, plus ``center``. The
          parametrisation is the outward unit normal, so a patch may be of any size short of half the
          sphere and may contain the poles.
        * ``"cylinder_arc"``: the arc's ``start`` and ``end`` as positional arguments, plus ``center``.
          The cylinder's axis follows from those three points.

        Args:
            typ: One of ``"circle_arc"``, ``"sphere_part"``, ``"cylinder_arc"``.
            args: Positional arguments for the curved entity, as listed above.
            kwargs: Keyword arguments for the curved entity, in particular ``center``.

        Returns:
            The created curved entity to be used in :py:meth:`add_facet_to_boundary`.
        """
        store_entity: bool = kwargs.get("store_entity", True)
        res: "_pyoomph.MeshTemplateCurvedEntityBase"
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
        elif typ == "sphere_part":
            if len(args) != 1 or kwargs.get("center") is None:
                raise RuntimeError(
                    "sphere_part must have one positional arg {point_on_sphere} and center as kwarg")
            center, onsphere = kwargs.get("center"), args[0]
            if isinstance(center, int):
                center = self.get_node_position(center)
            if isinstance(onsphere, int):
                onsphere = self.get_node_position(onsphere)
            res = _pyoomph.CurvedEntitySpherePart(center, onsphere)  # type:ignore
        elif typ == "cylinder_arc":
            if len(args) != 2 or kwargs.get("center") is None:
                raise RuntimeError(
                    "cylinder_arc must have two positional args {start,end} and center as kwarg")
            center, start, end = kwargs.get("center"), args[0], args[1]
            if isinstance(center, int):
                center = self.get_node_position(center)
            if isinstance(start, int):
                start = self.get_node_position(start)
            if isinstance(end, int):
                end = self.get_node_position(end)
            res = _pyoomph.CurvedEntityCylinderArc(center, start, end)  # type:ignore
        else:
            raise RuntimeError("Unknown type "+str(typ))
        if store_entity:
            self._macrobounds.append(res)
        return res


class MeshedMeshTemplate(MeshTemplate):
    """
    Base class of all :py:class:`MeshTemplate` classes that let an external mesh generator (e.g. Gmsh) create the mesh
    from a geometry description given in :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry`.

    Since the geometry is described by a method that can be called again at any time, remeshing is done by recreation:
    a :py:class:`~pyoomph.meshes.remesher.RemesherViaRecreation` is attached by default, which calls
    :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry` again whenever remeshing is required. It is therefore
    up to you to describe both the initial and the remeshed geometry in that single method: use :py:meth:`is_remeshing`
    (or its complement :py:meth:`is_first_time`) to distinguish the two cases and
    :py:meth:`get_boundary_coordinates` to reconstruct the boundaries that have moved meanwhile.

    A :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry` that never asks either of the two would describe the
    very same geometry again, i.e. recreating it cannot give a different mesh. Such a template is therefore skipped
    when :py:meth:`~pyoomph.generic.problem.Problem.force_remesh` gathers the meshes to remesh by itself, but not when
    it is asked for this particular mesh, e.g. by a :py:class:`~pyoomph.equations.generic.RemeshWhen`.

    If you prefer the mesh generator to reconstruct the geometry automatically from the deformed mesh, overwrite the
    :py:attr:`~pyoomph.meshes.mesh.MeshTemplate.remesher` attribute by e.g. a
    :py:class:`~pyoomph.meshes.remesher.Remesher2d` instead.

    .. note::
        On a mesh distributed with ``--distribute``, :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry` is a
        **collective** region: :py:meth:`get_boundary_coordinates` gathers the boundary from all ranks, since no rank
        holds more than its own part of it. Every rank must therefore reach the same calls, with the same arguments and
        in the same order - do not branch on the MPI rank inside ``define_geometry``, and do not ask for a boundary on
        some ranks only. A ``define_geometry`` that raises on one rank is caught and turned into an error on all of
        them; one whose ranks disagree on which collectives to enter can only hang. See dev_docs/distributed_remeshing.md.
    """

    def __init__(self):
        super().__init__()
        # Deferred import: remesher.py imports gmsh.py, which imports this module, so importing it at module level
        # would be circular.
        from .remesher import RemesherViaRecreation
        self.remesher = RemesherViaRecreation(self)
        # Remember the remesher we attached ourselves, so that we can tell it from one the user has chosen deliberately
        self._auto_remesher = self.remesher
        self._within_define_geometry = False
        # Set by is_first_time(), i.e. whenever define_geometry asks whether it is remeshing. A define_geometry that
        # never asks will describe the very same geometry again, so recreating it cannot give a different mesh
        self._has_remeshing_path = False

    def is_first_time(self) -> bool:
        """Will return ``True``, if the mesh is being generated for the first time. Otherwise, it will return ``False``, which means that the mesh is being remeshed. You can use this to define different geometries for the initial mesh and the remeshed mesh.

        May only be called from within :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry`, since only there the answer is meaningful.

        Returns:
            Whether it is the first time the mesh is generated or not. ``True`` means first time, ``False`` means remeshing.
        """
        self._assert_within_define_geometry("is_first_time() and is_remeshing()")
        # Asking the question is what tells us that this template describes a different geometry when remeshed
        self._has_remeshing_path = True
        return not self.get_problem().is_initialised()

    def _assert_within_define_geometry(self, what: str):
        if not self._within_define_geometry:
            raise RuntimeError(what+" may only be called from within the define_geometry method of the mesh template. Elsewhere, no mesh is being created, so there is nothing to tell apart. If you require the information later on, store it in an attribute during define_geometry.")

    def _remeshing_can_change_the_mesh(self) -> bool:
        # Used by Problem.force_remesh() when it collects the meshes to remesh on its own: since we attach a
        # RemesherViaRecreation to every meshed template, remeshing all of them would also rebuild those templates
        # whose define_geometry does not react on remeshing at all - which just burns time to arrive at the same mesh.
        # A remesher the user has set deliberately is of course never skipped.
        if self.remesher is not self._auto_remesher:
            return True
        return self._has_remeshing_path

    def is_remeshing(self) -> bool:
        """Complement of :py:meth:`is_first_time`, i.e. ``True`` whenever :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.define_geometry` is called to remesh an already existing mesh.

        Returns:
            Whether the mesh is currently being recreated for remeshing.
        """
        return not self.is_first_time()

    def get_boundary_coordinates(self, name: str, sort_along_axis: "SortAlongAxis | None" = None, start_near_point: tuple["ExpressionOrNum", "ExpressionOrNum"] | None = None, nondimensional: bool = False) -> list[list[tuple["ExpressionOrNum", "ExpressionOrNum"]]]:
        """Returns a list of boundary segments, which are lists of (x,y) coordinates (dimensional or not can be controlled by the nondimensional argument). The segments are sorted and reversed based on the sort_along_axis or start_near_point arguments. If both are None, the order is arbitrary.

        On a mesh distributed with ``--distribute`` this is a **collective** call that returns the
        whole boundary on every rank, not just this rank's part of it. Every rank must therefore
        reach it, with the same arguments and in the same order - see :py:class:`MeshedMeshTemplate`.

        Args:
            name: Name of the boundary, e.g. "domain1/boundary1"
            sort_along_axis: Sort the segments along a given axis, e.g. "x+" means sort along x in increasing order, "y-" means sort along y in decreasing order. Defaults to None.
            start_near_point: Sort the segments by their distance to this point, closest first, and start each segment at its end closer to the point. May carry units. Defaults to None.
            nondimensional: Whether to return nondimensional coordinates. Defaults to False.

        Returns:
            A list of boundary segments, which are each a list of (x,y) coordinates. These are plain
            floats only for ``nondimensional``; otherwise they carry the spatial scaling and are
            therefore dimensional expressions.
        """
        self._assert_within_define_geometry("get_boundary_coordinates()")
        self._has_remeshing_path = True # The geometry obviously depends on the mesh we are replacing
        problem = self.get_problem()
        if not problem.is_initialised():
            raise RuntimeError("Cannot get boundary coordinates before the first mesh is generated")

        mesh, merge = self._resolve_mesh_for_boundary_coordinates(name)
        payload:list[list[tuple[float,float]]] | None
        if not merge:
            # Serial, or mpirun without --distribute: this rank holds the whole boundary already.
            segs, pts = self._sorted_boundary_segments(mesh, sort_along_axis, start_near_point)
            assert segs is not None and pts is not None  # only a global request returns None
            payload = [[(float(pts[0, i]), float(pts[1, i])) for i in seg] for seg in segs]
        else:
            # Each rank sees only its own partition of the boundary, which is not a piece of the
            # answer - it is a different, truncated geometry, and the segments are cut wherever the
            # partition happens to run. So merge the mesh data (collective, result on rank 0), sort
            # there, and hand the polylines to everybody.
            #
            # The broadcast has to be here rather than inside the merge: a repeated request is a
            # cache hit on rank 0, which broadcasts nothing (see meshdatamerge.py §3.4b), and the
            # other ranks would then wait for a request that never comes.
            comm = get_mpi_world_comm()
            assert comm is not None  # needs_merging implies more than one process
            payload = None
            error: BaseException | None = None
            if get_mpi_rank() == 0:
                # Everything from here to the broadcast happens on rank 0 alone, so it has to end for
                # all ranks or for none - the sorting rejects a bad sort_along_axis, and the merge
                # itself checks that the nodes it identified really do coincide.
                try:
                    segs, pts = self._sorted_boundary_segments(mesh, sort_along_axis, start_near_point,
                                                               global_mesh=True)
                    assert segs is not None and pts is not None  # rank 0 is the one that gets the merged data
                    payload = [[(float(pts[0, i]), float(pts[1, i])) for i in seg] for seg in segs]
                except BaseException as e:
                    error = e
            else:
                self._sorted_boundary_segments(mesh, sort_along_axis, start_near_point, global_mesh=True)
            # Only the nondimensional numbers travel, so a dimensional scaling (a GiNaC expression)
            # never has to survive being pickled - it is applied below, on every rank.
            payload = comm.bcast(payload, root=0)
            mpi_share_root_failure(error, context="building the boundary coordinates of '"+name+"' from the merged mesh data")
            assert payload is not None

        SS = 1 if nondimensional else problem.get_scaling("spatial")
        return [[(x*SS, y*SS) for x, y in seg] for seg in payload]

    def _resolve_mesh_for_boundary_coordinates(self, name: str):
        """The mesh named by get_boundary_coordinates, and whether its data has to be merged.

        Both answers are agreed on across the ranks, because both decide whether this rank enters the
        merge collective that follows, and a rank that decides differently from the others hangs them
        rather than failing:

        * a name this rank cannot resolve becomes an error on all of them, instead of unwinding one
          rank alone - the same asymmetry MeshedMeshTemplate._do_define_geometry guards against one
          level further out;
        * ``needs_merging`` is asked of every rank rather than trusted locally. It reads
          ``is_mesh_distributed()`` off *this* mesh, and an interface mesh whose partition happens to
          hold no element of it is exactly the kind of place where that could come out differently.
        """
        problem = self.get_problem()
        collective = bool(problem.is_distributed()) and get_mpi_nproc() > 1
        mesh = None
        error: BaseException | None = None
        try:
            mesh = problem.get_mesh(name)
        except BaseException as e:
            error = e
        if not collective:
            if error is not None:
                raise error
            return mesh, False
        if get_mpi_any(error is not None):
            if error is not None:
                raise error
            raise RuntimeError("get_boundary_coordinates('"+name+"'): another MPI rank could not resolve "
                               "that mesh. This rank raises here rather than entering the collective merge "
                               "alone; the real error is reported by the rank that saw it.")
        # Deferred, like everywhere else that touches the merge: a serial run must not pull in mpi4py
        # through this path.
        from .meshdatamerge import needs_merging
        assert mesh is not None # get_mesh() returns a mesh or raises, and a raise has been dealt with above
        return mesh, get_mpi_any(needs_merging(mesh))

    def _sorted_boundary_segments(self, mesh, sort_along_axis: "SortAlongAxis | None", start_near_point: tuple["ExpressionOrNum", "ExpressionOrNum"] | None, global_mesh: bool = False):
        """The boundary's line segments, oriented and ordered, together with the coordinate array.

        With ``global_mesh`` the extraction is collective and only rank 0 gets data back; the other
        ranks still have to call it, which is why it returns ``(None, None)`` there instead of
        refusing."""
        data = self.get_problem().get_cached_mesh_data(mesh, nondimensional=True, global_mesh=global_mesh)
        if data is None:
            return None, None
        pts = data.get_coordinates()
        segs, _ = data.get_interface_line_segments()
        segs = sort_line_segments(pts, segs, sort_along_axis=sort_along_axis, start_near_point=start_near_point, spatial_unit=self.get_problem().get_scaling("spatial"), whom="get_boundary_coordinates()")
        return segs, pts

    def _do_define_geometry(self, problem: "Problem", filename_trunk: str | None = None):
        # RemesherViaRecreation passes a fresh file name trunk for each remeshing round, so that the meshes written by
        # the backend do not overwrite each other. Backends that do not write any file can simply ignore it.
        if filename_trunk is not None:
            self._fntrunk = filename_trunk
        self._within_define_geometry = True
        error: BaseException | None = None
        try:
            super()._do_define_geometry(problem)
        except BaseException as e:
            error = e
        finally:
            self._within_define_geometry = False
        # Whatever the backend does with the geometry afterwards is collective (the barriers and the
        # run_on_rank_zero write in generate_mesh_to_file), so a rank that unwinds from here alone
        # leaves the others waiting for it forever. Agree on the outcome instead, symmetrically
        # rather than rooted at rank 0, precisely because the rank that fails need not be rank 0.
        #
        # This catches a define_geometry that raises once all ranks are through its own collectives -
        # a geometry only one rank finds invalid, say. It cannot catch a raise BEFORE one of them:
        # get_boundary_coordinates() is collective on a distributed mesh, so a rank that never
        # reaches it hangs the others inside its merge and this point is never reached. That is what
        # the "do not branch on rank" contract on MeshedMeshTemplate is for; see
        # dev_docs/distributed_remeshing.md, stage 1.
        if get_mpi_nproc() > 1:
            if get_mpi_any(error is not None) and error is None:
                raise RuntimeError("Another MPI rank failed inside the define_geometry() of "+type(self).__name__ +
                                   ". This rank succeeded and raises here so that the job ends rather than waiting "
                                   "for a rank that is gone; the real error is reported by the rank that saw it.")
        if error is not None:
            raise error


def _evaluate_extremum_impl(mesh:"AnySpatialMesh",name:str | list[str],sign:int,dimensional:bool,as_float:bool,return_x:bool):
    """Shared body of ``evaluate_maximum``/``evaluate_minimum`` for bulk and interface meshes.

    ``sign`` is +1 for a maximum and -1 for a minimum, matching ``Mesh::evaluate_extremum``.

    On a distributed mesh this is **collective**: an extremum is a property of the whole mesh, while
    ``Mesh::evaluate_extremum`` samples only the elements this rank holds (its halo copies merely
    repeat a neighbour's answer). Every rank therefore has to reach it, with the same arguments -
    ``rayleigh_plateau.py`` derives its *time step* from one of these, so ranks that answered
    differently did not fail, they quietly stopped being one simulation.
    """
    if not isinstance(name,str):
        if return_x:
            raise RuntimeError("Please set return_x=False for multiple extremum evaluations (or call them one by one)")
        return [cast(ExpressionOrNum, _evaluate_extremum_impl(mesh,n,sign,dimensional,as_float,False)) for n in name]
    flags=0
    if dimensional:
        flags|=1

    x:list[float] | None=None
    if mesh.nelement():
        val,s,elem=mesh._evaluate_extremum(name,sign,flags)
        if return_x:
            x=[float(xc) for xc in elem.get_interpolated_position_at_s(0,s,False)]
        fval=float(_pyoomph.GiNaC_collect_units(val)[0]) if dimensional else float(val)
    else:
        # A rank holding no element of this mesh has nothing to offer - which happens for an
        # interface mesh whose boundary lies entirely in another partition. -inf loses a maximum
        # search and +inf loses a minimum one, so it needs no special case in the comparison.
        fval=-math.inf if sign>0 else math.inf

    if mesh.is_mesh_distributed() and get_mpi_nproc()>1:
        comm=get_mpi_world_comm()
        assert comm is not None
        # One allgather of a float and a handful of coordinates, rather than a reduction followed by
        # a broadcast: min()/max() over the gathered list settles a tie by the lowest rank, and does
        # so identically on every rank.
        gathered=comm.allgather((fval,x))
        pick=max if sign>0 else min
        best=pick(range(len(gathered)),key=lambda r:gathered[r][0])
        fval,x=gathered[best]
        if fval in (math.inf,-math.inf):
            raise RuntimeError("evaluate_maximum/evaluate_minimum('"+name+"'): no rank holds any element of '"+
                               mesh.get_full_name()+"', so there is no extremum to report.")

    # The unit comes from the registered expression rather than from the value just evaluated, so
    # that a rank holding no element of this mesh - which has no value to read it off - can still
    # build the dimensional result. Only the number travels between the ranks; a GiNaC expression
    # could not, and would not have to.
    unit:ExpressionOrNum=1
    if dimensional and not as_float:
        unit=mesh.get_code_gen()._get_extremum_expression_unit_factor(name)

    if return_x and x is not None and dimensional:
        SS=mesh.get_problem().get_scaling("spatial")
        assert isinstance(SS, Expression)
        xdim:list[ExpressionOrNum]=[xc*SS for xc in x]
    else:
        xdim=list(x) if x is not None else None  # type:ignore

    if not dimensional:
        outval:ExpressionOrNum=fval
    elif as_float:
        outval=fval
        if return_x:
            assert xdim is not None
            xn:list[ExpressionOrNum]=[]
            for xc in xdim:
                assert isinstance(xc, Expression)
                factor, _, _, _ = _pyoomph.GiNaC_collect_units(xc)
                xn.append(float(factor))
            xdim=xn
    else:
        outval=fval*unit
    if return_x:
        assert xdim is not None
        return outval,xdim
    else:
        return outval


class MeshFromTemplateBase(BaseMesh):
    # Restated with the types of the C++ Mesh properties they really are: assigning them in __init__
    # would otherwise define plain attributes here, which the concrete meshes then inherit twice with
    # two different types (once from this mixin, once from the C++ base).
    min_permitted_error:float
    max_permitted_error:float
    max_refinement_level:int
    min_refinement_level:int

    def __init__(self, problem: "Problem", templatemesh: MeshTemplate, domainname: str, eqtree: "EquationTree", previous_mesh: "BulkTemplateMesh | None" = None):
        # Not super().__init__(): self is a MeshFromTemplate1d/2d/3d instance, which (due to the
        # nanobind single-inheritance restriction, see _install_mixin) is not a real subclass of
        # MeshFromTemplateBase/BaseMesh, so super(MeshFromTemplateBase, self) would fail its MRO check.
        BaseMesh.__init__(self)

        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))

        self._templatemesh: MeshTemplate = templatemesh
        self._name = domainname
        self._eqtree: "EquationTree" = eqtree
        self._codegen = eqtree.get_code_gen()
        self._eqtree._mesh = self
        self.ignore_initial_condition = False
        self._set_problem(problem, self._codegen._code)
        self._error_estimator: Z2ErrorEstimator  # =None
        self._solves_since_remesh = 0  # Counting the number of solves since last remesh
        self._periodic_corner_node_info: dict[Node, Node] = {}
        self._initial_uniform_refinement_level=0

        T = TypeVar("T")

        def a_or_b(a: T | None, b: T | None) -> T:
            res = b if a is None else a
            assert res is not None
            return res

        if previous_mesh is None:
            self.min_permitted_error = a_or_b(
                templatemesh.min_permitted_error, problem.min_permitted_error)
            self.max_permitted_error = a_or_b(
                templatemesh.max_permitted_error, problem.max_permitted_error)
            self.max_refinement_level = a_or_b(
                templatemesh.max_refinement_level, problem.max_refinement_level)
            self.min_refinement_level = a_or_b(
                templatemesh.min_refinement_level, problem.min_refinement_level)
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
            pinter:"InterfaceMesh | None" = None
            if previous_mesh is not None:
                prev = previous_mesh.get_mesh(n)
                assert isinstance(prev, InterfaceMesh)
                pinter = prev
            self._interfacemeshes[n] = InterfaceMesh(
                problem, self, n, eqtree, previous_mesh=pinter)

    def get_problem(self) -> "Problem":
        return self._get_problem() #type:ignore

    def get_bulk_mesh(self):
        return None



    def _link_periodic_corner_nodes(self):
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        if len(self._periodic_corner_node_info) == 0:
            return
        newmap: dict[Node, Node] = {}
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

    def _merge_my_error_with_elemental_max_override(self) -> NPFloatArray:
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        res = self.get_elemental_errors()
        # print("IN MERGE",self.get_name(),res)
        for i, e in enumerate(self.elements()):
            res[i] = max(e._elemental_error_max_override, res[i])
            # e._elemental_error_max_override=res[i]
        # TODO: Elements that are only at one interface
        return res

    def get_nodal_field_indices(self) -> dict[str, int]:
        return self.get_code_gen().get_code().get_nodal_field_indices()

    def recreate_boundary_information(self):
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        from .. import get_dev_option

        if self.refinement_possible() or (get_dev_option("allow_tri_refine") and self.get_dimension() == 2):
            self.setup_tree_forest()

        self.setup_boundary_element_info()

        for interior_bound in self._templatemesh.get_template()._interior_boundaries:
            try:
                bindex = self.get_boundary_index(interior_bound)
                self.setup_interior_boundary_elements(bindex)
            except:
                pass

    def _setup_output_scales(self):
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        codegen = self.get_code_gen()
        code = codegen.get_code()
        _, unit, _, _ = _pyoomph.GiNaC_collect_units(
            codegen.expand_placeholders(codegen.get_scaling("spatial"), False))
        self.set_output_scale("spatial", unit, code)  # TODO
        for k in itertools.chain(code.get_nodal_field_indices().keys(), code.get_elemental_field_indices().keys()):
            s = codegen.get_scaling(k)
            s = codegen.expand_placeholders(s, False)
            _factor, unit, _rest, _success = _pyoomph.GiNaC_collect_units(s)
            if not (_rest-1).is_zero():
                raise RuntimeError(
                    "Cannot set output scale for field "+k+" to "+str(s)+" because it has a non-unit factor "+str(_factor)+" and a non-unit rest "+str(_rest))
            self.set_output_scale(k, unit, code)
        
    def _finalise_creation(self):
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        self.generate_from_template(
            self._templatemesh.get_template().get_domain(self._name))
        # if self.refinement_possible():
        from .. import get_dev_option

        if self.refinement_possible() or (get_dev_option("allow_tri_refine") and self.get_dimension() == 2):
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
                icg._set_problem(self.get_problem())
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
                ieqs = imsh._eqtree.get_equations()
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
                    icg._set_problem(self.get_problem())
                    icg._set_nodal_dimension(
                        self._codegen.get_nodal_dimension())
                    icg._set_lagrangian_dimension(
                        self._codegen.get_lagrangian_dimension())
                    icg._coordinate_space = self._codegen._coordinate_space
                    icg._do_define_fields(self._codegen.dimension - 2)

    def _compile_bulk_equations(self) -> _pyoomph.DynamicJITCode:
        problem = self.get_problem()
        assert problem is not None
        assert self._codegen is not None
        eqs = self._eqtree.get_equations()
        self._codegen._set_problem(problem)
        mesh = self._eqtree._mesh
        if mesh is not None:
            # isinstance against the ABC (rather than the 3 concrete classes) so mypy still
            # knows about MeshFromTemplateBase's members (see _install_mixin); equivalent at
            # runtime since only MeshFromTemplate1d/2d/3d are registered against it.
            assert isinstance(mesh, MeshFromTemplateBase)
            templ = mesh._templatemesh
            # Get point to evaluate the IC and DBC to check whether it is a numeric value (Can prevent problems if somethink like 1/x is used)
            if templ is not None:
                templ = templ.get_template()
                dom = templ.get_domain(self._name)
                refpos = dom._get_reference_position_for_IC_and_DBC(set())
                refnorm=[0.1,0.1,0.1] # TODO: Get a right reference normal
                t = problem.time_pt().time()
                self._codegen._set_reference_point_for_IC_and_DBC(
                    refpos[0], refpos[1], refpos[2], t,refnorm[0],refnorm[1],refnorm[2])
        eqs._set_current_codegen(self._eqtree._codegen)
        #problem.before_compile_equations(self._eqtree._equations)
        eqs.before_finalization(self._codegen)
        self._codegen._finalise()
        eqs.before_compilation(self._codegen)
        self._codegen._code = problem._compile_bulk_element_code(
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

        if state.save or state.version_at_least(0,1,0):
            self._define_state_file_structural(state)
        else:
            self._define_state_file_legacy(state)

    def _define_state_file_structural(self, state: "DumpFile"):
        # The refinement is stored as the shape of each root's tree and every node/element by a key
        # that does not mention the partition, so the file is the same whether it was written serially
        # or on any number of processes, and either can read the other.
        # See pyoomph/meshes/meshstate.py and dev_docs/distributed_state_files.md.
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        from .meshstate import save_mesh_state, load_mesh_state
        # The base element numbers have to exist before anything can be addressed by them. On a
        # distributed mesh they were assigned while it was still whole - before the initial
        # distribution, or in Problem._redistribute_after_remeshing() for a mesh a remesh replaced -
        # and must not be touched here, where only the local share is visible. Otherwise assign them
        # now: besides being idempotent for a mesh that has them, this covers the meshes that did not
        # exist when the problem was initialised - the ones a remesh built, and the ones the loader
        # itself builds when the state file carries a different mesh template. Both are built from a
        # template, in its element order, so writer and reader agree on the numbers.
        if not self.is_mesh_distributed():
            self.assign_global_base_element_indices()
        if state.save:
            save_mesh_state(self,state)
        else:
            load_mesh_state(self,state)
        self._define_tracer_state_file(state)

    def _define_state_file_legacy(self, state: "DumpFile"):
        # Reader for state files written before version 0.1.0: the refinement as oomph-lib's level-wise
        # element numbers and one flat nodal blob in the mesh's own traversal order, both of which only
        # mean anything to a process holding exactly this mesh. Still writes as well (nothing calls it
        # to write any more, but that is what makes the two formats comparable in a benchmark).
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        old_ordering = True
        # Refinement pattern
        if state.save:
            refinementS: list[NPInt32Array] = self.get_refinement_pattern()
            nref = len(refinementS)
            state.int_data(lambda: nref, lambda v: v)
            for n in range(nref):
                state.numpy_data(lambda: refinementS[n], lambda v: v)
        else:
            nref = state.int_data(lambda: 0, lambda v: v)
            refinementL: list[list[int]] = []
            for n in range(nref):
                refinementL.append(list(state.numpy_data(
                    lambda: refinementL[n], lambda v: v)))  # type:ignore
            # print("REFINEMEHT",refinement,self.nelement())
            while not self.unrefine_uniformly(): # Unrefine until we hit the rock bottom
                pass
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

        self._define_tracer_state_file(state)

    def _define_tracer_state_file(self, state: "DumpFile"):
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        # No MPI refusal here any more: TracerCollection._save_state gathers every process's
        # particles and sorts them by their persistent identity, and _load_state hands the whole set
        # to every process, each of which keeps the ones it owns. So the file says nothing about the
        # partitioning and can be written at one process count and read at another.
        numtracercols = len(self._tracers)
        state.int_data(lambda: numtracercols,
                       lambda n: state.assert_equal(n, numtracercols))
        # The rolling position history came in with 0.1.3. Older files simply do not have it, and
        # the two arrays are laid out differently with and without it, so the reader and the writer
        # have to agree on this flag - which is what the version condition gives.
        with_history = state.version_at_least(0, 1, 3)
        for tname in sorted(self._tracers.keys()):
            tcol = self.get_tracers(tname)
            assert tcol is not None
            state.string_data(
                lambda: tname, lambda tn: state.assert_equal(tn, tname))
            if state.save:
                pdata, tdata = tcol._save_state(with_history)
                state.numpy_data(lambda: pdata, lambda v: v)  # type:ignore
                state.numpy_data(lambda: tdata, lambda v: v)  # type:ignore
            else:
                pdata = state.numpy_data(lambda: 0, lambda v: v)  # type:ignore
                tdata = state.numpy_data(lambda: 0, lambda v: v)  # type:ignore
                tcol._load_state(pdata, tdata, with_history)  # type:ignore



    def _evaluate_extremum_wrapper(self,name:str | list[str],sign:int,dimensional:bool=True,as_float:bool=False,return_x:bool=True):
        assert isinstance(
            self, (MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d))
        return _evaluate_extremum_impl(self,name,sign,dimensional,as_float,return_x)


    def evaluate_maximum(self,name:str | list[str],dimensional:bool=True,as_float:bool=False,return_x:bool=False)->ExpressionOrNum | Sequence[ExpressionOrNum] | tuple[ExpressionOrNum, Sequence[ExpressionOrNum]]:
        """Evaluate the maximum of a quantity defined by ExtremumObservables on the mesh.
        
        Args:
            name: The name of the observable or a list of names
            dimensional: If True, return the value(s) with units, otherwise return float(s)
            as_float: If True, return the value(s) as float(s) (without units)
            return_x: If True, also return the position(s) of the maximum(s)
        
        Returns:
            If return_x is False, returns the maximum value(s) as ExpressionOrNum or list of ExpressionOrNum.
            If return_x is True, returns a tuple of (maximum value(s), position(s)) where position(s) is a list of coordinates corresponding to the maximum value(s).
        """
        return self._evaluate_extremum_wrapper(name,1,dimensional=dimensional,as_float=as_float,return_x=return_x)
           
    def evaluate_minimum(self,name:str | list[str],dimensional:bool=True,as_float:bool=False,return_x:bool=False)->ExpressionOrNum | Sequence[ExpressionOrNum] | tuple[ExpressionOrNum, Sequence[ExpressionOrNum]]:
        """Evaluate the minimum of a quantity defined by ExtremumObservables on the mesh.
        
        Args:
            name: The name of the observable or a list of names
            dimensional: If True, return the value(s) with units, otherwise return float(s)
            as_float: If True, return the value(s) as float(s) (without units)
            return_x: If True, also return the position(s) of the minimum(s)
        
        Returns:
            If return_x is False, returns the minimum value(s) as ExpressionOrNum or list of ExpressionOrNum.
            If return_x is True, returns a tuple of (minimum value(s), position(s)) where position(s) is a list of coordinates corresponding to the minimum value(s).
        """
        return self._evaluate_extremum_wrapper(name,-1,dimensional=dimensional,as_float=as_float,return_x=return_x)


if TYPE_CHECKING:
    # _install_mixin() (see its docstring above) copies MeshFromTemplateBase's members onto
    # MeshFromTemplate1d/2d/3d at runtime without adding it to __bases__, since nanobind does
    # not support combining a bound C++ base with an additional Python base in the same class.
    # A static type checker has no such restriction - here (TYPE_CHECKING only, never executed)
    # each concrete class instead really inherits from both, so every MeshFromTemplateBase/
    # BaseMesh member (get_code_gen, _interfacemeshes, _eqtree, get_full_name, ...) is visible
    # automatically, without having to hand-declare each one as they get used (as happened
    # before for get_code_gen). Only members the *C++* base also defines under the same name
    # (e.g. boundary_elements, whose auto-generated binding is less precise than the overloads
    # below) still need restating directly on the concrete class: Python's MRO would otherwise
    # let the C++ base's version - first in this bases tuple - win over MeshFromTemplateBase's.
    # The ignores are for the refinement settings: the C++ base has them as properties and the mixin
    # restates them as the attributes they are assigned as, which mypy will not accept as the same
    # member. At runtime there is only one of each - the C++ property - since the mixin is not a base
    # of these classes at all (see _install_mixin).
    class _MeshFromTemplate1dTypingBase(_pyoomph.TemplatedMeshBase1d, MeshFromTemplateBase): pass # type: ignore[misc]
    class _MeshFromTemplate2dTypingBase(_pyoomph.TemplatedMeshBase2d, MeshFromTemplateBase): pass # type: ignore[misc]
    class _MeshFromTemplate3dTypingBase(_pyoomph.TemplatedMeshBase3d, MeshFromTemplateBase): pass # type: ignore[misc]
else:
    _MeshFromTemplate1dTypingBase = _pyoomph.TemplatedMeshBase1d
    _MeshFromTemplate2dTypingBase = _pyoomph.TemplatedMeshBase2d
    _MeshFromTemplate3dTypingBase = _pyoomph.TemplatedMeshBase3d


class MeshFromTemplate1d(_MeshFromTemplate1dTypingBase):
    if TYPE_CHECKING:
        @overload
        def boundary_elements(self, b: str, with_directions: Literal[False] = ...) -> Iterator[_pyoomph.OomphGeneralisedElement]: ...
        @overload
        def boundary_elements(self, b: str, with_directions: Literal[True]) -> Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]: ...
        def boundary_elements(self, b: str, with_directions: bool = ...) -> Iterator[_pyoomph.OomphGeneralisedElement] | Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]: ...

    def __init__(self, problem: "Problem", templatemesh: MeshTemplate, domainname: str, elementtype: "EquationTree", previous_mesh: BulkTemplateMesh | None = None):
        super(MeshFromTemplate1d, self).__init__()
        # self is not nominally a MeshFromTemplateBase (see _install_mixin); the explicit
        # unbound call still works fine at runtime since MeshFromTemplateBase.__init__ only
        # duck-types on self's attributes.
        MeshFromTemplateBase.__init__(
            self, problem, templatemesh, domainname, elementtype, previous_mesh=previous_mesh)  # type: ignore[arg-type]

    def get_dimension(self) -> int:
        return 1

    def get_problem(self) -> "Problem":
        from ..generic.problem import Problem
        pr = self._get_problem()
        assert isinstance(pr, Problem)
        return pr


_install_mixin(MeshFromTemplate1d, MeshFromTemplateBase)


class MeshFromTemplate2d(_MeshFromTemplate2dTypingBase):
    if TYPE_CHECKING:
        @overload
        def boundary_elements(self, b: str, with_directions: Literal[False] = ...) -> Iterator[_pyoomph.OomphGeneralisedElement]: ...
        @overload
        def boundary_elements(self, b: str, with_directions: Literal[True]) -> Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]: ...
        def boundary_elements(self, b: str, with_directions: bool = ...) -> Iterator[_pyoomph.OomphGeneralisedElement] | Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]: ...

    def __init__(self, problem: "Problem", templatemesh: MeshTemplate, domainname: str, elementtype: "EquationTree", previous_mesh: BulkTemplateMesh | None = None):
        super(MeshFromTemplate2d, self).__init__()
        MeshFromTemplateBase.__init__(
            self, problem, templatemesh, domainname, elementtype, previous_mesh=previous_mesh)  # type: ignore[arg-type]

    def get_dimension(self)->int:
        return 2

    def get_problem(self) -> "Problem":
        from ..generic.problem import Problem
        pr = self._get_problem()
        assert isinstance(pr, Problem)
        return pr


_install_mixin(MeshFromTemplate2d, MeshFromTemplateBase)


class MeshFromTemplate3d(_MeshFromTemplate3dTypingBase):
    if TYPE_CHECKING:
        @overload
        def boundary_elements(self, b: str, with_directions: Literal[False] = ...) -> Iterator[_pyoomph.OomphGeneralisedElement]: ...
        @overload
        def boundary_elements(self, b: str, with_directions: Literal[True]) -> Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]: ...
        def boundary_elements(self, b: str, with_directions: bool = ...) -> Iterator[_pyoomph.OomphGeneralisedElement] | Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]: ...

    def __init__(self, problem: "Problem", templatemesh: MeshTemplate, domainname: str, elementtype: "EquationTree", previous_mesh: BulkTemplateMesh | None = None):
        super(MeshFromTemplate3d, self).__init__()
        MeshFromTemplateBase.__init__(
            self, problem, templatemesh, domainname, elementtype, previous_mesh=previous_mesh)  # type: ignore[arg-type]

    def get_dimension(self)->int:
        return 3

    def get_problem(self) -> "Problem":
        from ..generic.problem import Problem
        pr = self._get_problem()
        assert isinstance(pr, Problem)
        return pr


_install_mixin(MeshFromTemplate3d, MeshFromTemplateBase)


def MeshFromTemplate(problem: "Problem", templatemesh: MeshTemplate, domainname: str, eqtree: "EquationTree", previous_mesh: BulkTemplateMesh | None = None) -> MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d:
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

if TYPE_CHECKING:
    # See the identical block above MeshFromTemplate1d for the general rationale. BaseMesh's
    # members (_interfacemeshes, get_code_gen, elements, ...) are now visible automatically
    # via this typing-only base; only boundary_elements/get_mesh (also defined by the *C++*
    # base, less precisely) still need restating directly on InterfaceMesh below.
    class _InterfaceMeshTypingBase(_pyoomph.InterfaceMesh, BaseMesh): pass
else:
    _InterfaceMeshTypingBase = _pyoomph.InterfaceMesh


class InterfaceMesh(_InterfaceMeshTypingBase):
    """
    A mesh that is added to the boundary to add Neumann terms or setting Dirichlet conditions or add new fields directly on the interface, like e.g. surfactants.
    """
    if TYPE_CHECKING:
        @overload
        def boundary_elements(self, b: str, with_directions: Literal[False] = ...) -> Iterator[_pyoomph.OomphGeneralisedElement]: ...
        @overload
        def boundary_elements(self, b: str, with_directions: Literal[True]) -> Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]: ...
        def boundary_elements(self, b: str, with_directions: bool = ...) -> Iterator[_pyoomph.OomphGeneralisedElement] | Iterator[tuple[_pyoomph.OomphGeneralisedElement, int]]: ...
        @overload
        def get_mesh(self, name: str, return_None_if_not_found: Literal[False] = ...) -> "MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d | InterfaceMesh": ...
        @overload
        def get_mesh(self, name: str, return_None_if_not_found: Literal[True]) -> "MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d | InterfaceMesh | None": ...
        def get_mesh(self, name: str, return_None_if_not_found: bool = ...) -> "MeshFromTemplate1d | MeshFromTemplate2d | MeshFromTemplate3d | InterfaceMesh | None": ...
        def _pre_compile_interface_equations(self, tree_depth: int) -> None: ...
        def _compile_interface_equations(self, tree_depth: int) -> None: ...
        def _generate_interface_elements(self, tree_depth: int) -> None: ...

    def __init__(self, problem: "Problem", parent: "AnySpatialMesh", intername: str, eqtree: "EquationTree", previous_mesh: "InterfaceMesh | None" = None):
        super(InterfaceMesh, self).__init__()
        # self is not nominally a BaseMesh (see _install_mixin); the explicit unbound call
        # still works fine at runtime since BaseMesh.__init__ only duck-types on self's attributes.
        BaseMesh.__init__(self)  # type: ignore[arg-type]
        # _pyoomph.InterfaceMesh.__init__(self,problem)
        # super().__init__(problem)
        self._set_problem(problem, eqtree.get_code_gen()._code)
        self._parent: "AnySpatialMesh" = parent
        self._opposite_interface_mesh: "InterfaceMesh | None" = None
        self._interface_name: str = intername
        self._codegen = eqtree.get_code_gen()
        self._eqtree: "EquationTree" = eqtree
        self._eqtree._mesh = self
        self._error_estimator = _pyoomph.Z2ErrorEstimator()
        self._error_estimator.use_Lagrangian = False
        self.ignore_initial_condition = False
        self.set_spatial_error_estimator_pt(self._error_estimator)
        # Inherit the refinement thresholds from the bulk mesh this interface hangs off (which in
        # turn got them from the template or the Problem). Only MeshFromTemplate* used to set them,
        # so an interface silently ran on oomph-lib's own defaults, 1e-3/1e-5 - the max coinciding
        # with pyoomph's default and the min not (1e-4). Nobody chose that, and it only started to
        # matter once an interface could estimate its own error. Settable afterwards per interface.
        self.min_permitted_error = parent.min_permitted_error
        self.max_permitted_error = parent.max_permitted_error
        if previous_mesh is not None:
            self._setup_information_from_old_mesh(previous_mesh)
            # Carry the tracer collections over to the replacement mesh. Remeshing does this for the
            # bulk meshes explicitly (Problem.remesh_handler_during_solve), but an interface mesh is
            # rebuilt as a new object rather than replaced in _meshdict, so without this the
            # collections - and every particle in them - simply disappeared. The dict is shared
            # rather than copied, for the same reason the bulk path shares it: the TracerParticles
            # equation may still be holding the old mesh, and both then see the same collections.
            # Re-pointing each collection at this mesh has to wait until it has elements, which is
            # what TracerParticles.after_remeshing does.
            self._tracers = previous_mesh._tracers

        for n, eqtree in self._eqtree.get_children().items():
            pinter:"InterfaceMesh | None" = None
            if previous_mesh is not None:
                prev = previous_mesh.get_mesh(n)
                assert isinstance(prev, InterfaceMesh)
                pinter = prev
            self._interfacemeshes[n] = InterfaceMesh(
                problem, self, n, eqtree, previous_mesh=pinter)

    def get_problem(self) -> "Problem":
        from ..generic.problem import Problem
        pr = self._get_problem()
        assert isinstance(pr, Problem)
        return pr

    def refinement_possible(self) -> bool:
        p:"AnySpatialMesh" = self
        while isinstance(p, InterfaceMesh):
            p = p._parent
        return p.refinement_possible()

    def get_dimension(self) -> int:
        return self._parent.get_dimension()-1

    def get_bulk_mesh(self) -> "AnySpatialMesh":
        # The nanobind C++ binding no longer exposes get_bulk_mesh() (see mesh.cpp): with the
        # MeshHandle machinery there is no bare mesh pointer nanobind could wrap for a return
        # value, and it would also not be the same Python object as self._parent anyway. This is
        # already stored directly, so just return it.
        return self._parent

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

    def get_nodal_field_indices(self) -> dict[str, int]:
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
        boundname_set: set[str] = {name}
        while isinstance(curri, InterfaceMesh):
            assert curri._interface_name is not None
            name = curri._interface_name + "__" + name
            boundname_set.add(curri._interface_name)
            curri = cast(AnySpatialMesh, curri._parent)
        # assert isinstance(curri,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d))
        assert isinstance(curri, MeshFromTemplateBase)
        templ: MeshTemplate | None = curri._templatemesh
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
            t = self.get_problem().time_pt().time()
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

        self._codegen._code = self._codegen.get_problem()._compile_bulk_element_code(self._codegen, self, curri._name + "__" + name)  

        self._set_problem(self.get_problem(),
                          self._codegen._code)  # type:ignore
        if oppi is not None:
            oppi.get_equations()._set_current_codegen(old_oppcg)  # type:ignore
            oppblk.get_equations()._set_current_codegen(old_oppblkcg)  # type:ignore

        blk.get_equations()._set_current_codegen(oldblk)
        self._eqtree.get_equations()._set_current_codegen(oldmy)

        self._eqtree.get_equations().after_compilation(self._codegen)
                
        mpi_barrier()

    def nodes(self) -> Iterator[_pyoomph.Node]:
        uniqnods: set[Node] = set()
        for e in self.elements():
            nn = e.nnode()
            for ni in range(nn):
                n = e.node_pt(ni)
                uniqnods.add(n)
        for n in uniqnods:
            yield n

    def boundary_nodes(self, b: str) -> Iterable[_pyoomph.Node]:
        uniqnods: set[Node] = set()
        bind = self.get_boundary_index(b)
        for e in self.boundary_elements(b):
            nn = e.nnode()
            for ni in range(nn):
                n = e.node_pt(ni)
                if n.is_on_boundary(bind):
                    uniqnods.add(n)
        return uniqnods

    def nodes_on_both_sides(self) -> Generator[tuple[_pyoomph.Node, _pyoomph.Node], None, None]:
        nodemap: dict[Node, Node] = {}
        for e in self.elements():
            nn = e.nnode()
            for ni in range(nn):
                n = e.node_pt(ni)
                # print(e.get_opposite_bulk_element())
                no = e.opposite_node_pt(ni)
                nodemap[n] = no
        for nfrom, nto in nodemap.items():
            yield nfrom, nto

    def _reset_elemental_error_max_override(self):
        for e in self.elements():
            e._elemental_error_max_override = 0.0


    def _evaluate_extremum_wrapper(self,name:str | list[str],sign:int,dimensional:bool=True,as_float:bool=False,return_x:bool=True):
        return _evaluate_extremum_impl(self,name,sign,dimensional,as_float,return_x)


    def evaluate_maximum(self,name:str | list[str],dimensional:bool=True,as_float:bool=False,return_x:bool=False)->ExpressionOrNum | Sequence[ExpressionOrNum] | tuple[ExpressionOrNum, Sequence[ExpressionOrNum]]:
        """Evaluate the maximum of a quantity defined by ExtremumObservables on the mesh.
        
        Args:
            name: The name of the observable or a list of names
            dimensional: If True, return the value(s) with units, otherwise return float(s)
            as_float: If True, return the value(s) as float(s) (without units)
            return_x: If True, also return the position(s) of the maximum(s)
        
        Returns:
            If return_x is False, returns the maximum value(s) as ExpressionOrNum or list of ExpressionOrNum.
            If return_x is True, returns a tuple of (maximum value(s), position(s)) where position(s) is a list of coordinates corresponding to the maximum value(s).
        """
        return self._evaluate_extremum_wrapper(name,1,dimensional=dimensional,as_float=as_float,return_x=return_x)
           
    def evaluate_minimum(self,name:str | list[str],dimensional:bool=True,as_float:bool=False,return_x:bool=False)->ExpressionOrNum | Sequence[ExpressionOrNum] | tuple[ExpressionOrNum, Sequence[ExpressionOrNum]]:
        """Evaluate the minimum of a quantity defined by ExtremumObservables on the mesh.
        
        Args:
            name: The name of the observable or a list of names
            dimensional: If True, return the value(s) with units, otherwise return float(s)
            as_float: If True, return the value(s) as float(s) (without units)
            return_x: If True, also return the position(s) of the minimum(s)
        
        Returns:
            If return_x is False, returns the minimum value(s) as ExpressionOrNum or list of ExpressionOrNum.
            If return_x is True, returns a tuple of (minimum value(s), position(s)) where position(s) is a list of coordinates corresponding to the minimum value(s).
        """
        return self._evaluate_extremum_wrapper(name,-1,dimensional=dimensional,as_float=as_float,return_x=return_x)


_install_mixin(InterfaceMesh, BaseMesh)


class ODEStorageMesh(_pyoomph.ODEStorageMesh):
    """
    A sort of a mesh storing ODE values. This is not a real mesh, but a container for ODE values.
    """
    def __init__(self, problem: "Problem", eqtree: "EquationTree", domainname: str):
        super().__init__()
        #print("ODEStorageMesh: Creating ODE storage mesh for domain", domainname,get_mpi_rank())
        self._set_problem(problem, None)
        self._eqtree: "EquationTree | None" = eqtree
        self._eqtree._mesh = self  # type:ignore
        self._codegen = eqtree._codegen  # type:ignore
        self._name = domainname
        ocg = self._codegen.get_equations()._get_current_codegen()  # type:ignore
        self._codegen.get_equations()._set_current_codegen(self._codegen)  # type:ignore
        self._codegen._do_define_fields(0)  # type:ignore
        self._codegen._index_fields()  # type:ignore
        self._codegen.get_equations()._set_current_codegen(ocg)  # type:ignore
        self._element:"_pyoomph.OomphGeneralisedElement | None" = None
        self.ignore_initial_condition=False
        for _, eqtree in self._eqtree.get_children().items():
            raise RuntimeError("ODE domains may not have children yet")

    def get_code_gen(self) -> "FiniteElementCodeGenerator":
        assert self._codegen is not None
        return self._codegen

    def get_eqtree(self) -> "EquationTree":
        assert self._eqtree is not None
        return self._eqtree

    def get_problem(self) -> "Problem":
        return self._get_problem() #type:ignore

    def get_bulk_mesh(self):
        return None
    
    def _setup_output_scales(self):
        codegen = self.get_code_gen()
        eqtree = self.get_eqtree()
        ocg = codegen.get_equations()._get_current_codegen()
        codegen.get_equations()._set_current_codegen(codegen)
        _, indices = self.get_element()._ode_elem_to_numpy()
        scales: list[ExpressionOrNum] = [1.0] * len(indices)
        for k, i in indices.items():
            s = eqtree.get_equations().get_scaling(k)
            if isinstance(s, Expression):
                factor, _, _, _ = _pyoomph.GiNaC_collect_units(s)
                s = float(factor)
            scales[i]=s
        codegen.get_equations()._set_current_codegen(ocg)

    def _compile_bulk_equations(self) -> _pyoomph.DynamicJITCode:
        assert self._eqtree is not None
        assert self._codegen is not None
        problem=self.get_problem()
        self._codegen._set_problem(problem)

        ocg = self._codegen.get_equations()._get_current_codegen()
        self._codegen.get_equations()._set_current_codegen(self._codegen)

        eqs=self._eqtree.get_equations()
        #problem.before_compile_equations(self._eqtree)
        eqs.before_finalization(self._codegen)
        self._codegen._finalise()
        self._codegen.get_equations()._set_current_codegen(self._codegen)

        eqs.before_compilation(self._codegen)
        self._codegen._code = problem._compile_bulk_element_code(self._codegen, self, self._name)
        #self._element = _pyoomph.BulkElementODE0d.construct_new(self._codegen.get_code(), problem.timestepper)
        #self._element.set_must_be_kept_as_halo(True) # ODE Dofs are always halo dofs, so they can be accessed from everywhere
        self._set_problem(problem, self._codegen._code)
        self._element=self._create_ode_element(problem.timestepper)
        self._setup_output_scales()
#        self._add_ODE("ODE", self._element)
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

    def evaluate_all_observables(self) -> dict[str, ExpressionOrNum]:
        return BaseMesh.evaluate_all_observables(self)  # type:ignore

    #def get_element(self) -> _pyoomph.BulkElementODE0d:
    def get_element(self) -> Element:
        #print("GET ELEMENT",self._element)
        assert self._element is not None
        return self._element
        #return self._get_ODE("ODE")

    @overload
    def get_value(self, name: str, *,dimensional: bool=..., as_float: Literal[False]=...) -> _pyoomph.Expression: ...

    @overload
    def get_value(self, name: str, *, dimensional: bool=..., as_float: Literal[True]) -> float: ...

    @overload
    def get_value(self, name: NameStrSequence, *,dimensional: bool=..., as_float: Literal[False]=...) -> tuple[_pyoomph.Expression, ...]: ...
    
    @overload
    def get_value(self, name: NameStrSequence, *,dimensional: bool=..., as_float: Literal[True]) -> tuple[float, ...]: ...

    def get_value(self, name: str | NameStrSequence, *, dimensional: bool = True, as_float: bool = False) -> _pyoomph.Expression | float | tuple[float, ...] | tuple[_pyoomph.Expression, ...]:
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
        ode = self.get_element()
        vals, inds = ode._ode_elem_to_numpy()
        names:Sequence[str] = [name] if isinstance(name, str) else name
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
        ode = self.get_element()
        _, inds = ode._ode_elem_to_numpy()
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
        ode = self.get_element()
        _, inds = ode._ode_elem_to_numpy()
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


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
