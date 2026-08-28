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
 
from dataclasses import dataclass, fields as dataclass_fields, replace as dataclass_replace

from .. import _pyoomph_core as _pyoomph
from ..typings import *
import numpy


from .mesh import AnySpatialMesh, InterfaceMesh, MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d, MeshFromTemplateBase
from ..expressions import ExpressionOrNum,Expression
from ..expressions.units import unit_to_string

MeshDataEigenModes:TypeAlias=Literal["abs","real","imag","merge","angle"]


@dataclass(frozen=True)
class MeshDataCacheKey:
    """Identifies one option set of extracted mesh data, i.e. one slot in the :py:class:`MeshDataCacheStorage`.

    Everything that changes the *content* of a :py:class:`MeshDataCacheEntry` must be a field here -
    otherwise two different requests would share a cache slot and the second one would silently get
    the first one's data. Do not access the fields positionally; that is what this class replaced.

    ``operator`` is an arbitrary object and is hashed by identity, so two structurally identical
    operator instances still get separate cache slots.
    """
    nondimensional:bool=False
    tesselate_tri:bool=True
    eigenvector:int | tuple[int,...] | None=None
    eigenmode:MeshDataEigenModes="abs"
    history_index:int=0
    with_halos:bool=False
    operator:"MeshDataCacheOperatorBase | None"=None
    discontinuous:bool=False
    add_eigen_to_mesh_positions:bool=True
    global_mesh:bool=False

    @classmethod
    def create(cls,eigenvector:int | Sequence[int] | None=None,**kwargs:Any) -> "MeshDataCacheKey":
        """Builds a key, normalizing the eigenvector selection. Sequences become a sorted tuple: they
        must be hashable to serve as a key, and [1,0] must land in the same slot as [0,1]."""
        ev:int | tuple[int,...] | None
        if eigenvector is None or isinstance(eigenvector,int):
            ev=eigenvector
        else:
            ev=tuple(sorted(set(eigenvector)))
        return cls(eigenvector=ev,**kwargs)

    def as_kwargs(self) -> dict[str,Any]:
        # Not dataclasses.asdict: that deep-copies everything that is not itself a dataclass, i.e. it
        # would clone the operator and thereby break its identity (and any state it carries).
        return {f.name:getattr(self,f.name) for f in dataclass_fields(self)}

    @property
    def depends_on_eigen(self) -> bool:
        """Whether the data behind this key changes when the eigenvectors change, i.e. whether
        ``invalidate_cached_mesh_data(only_eigens=True)`` must flush it."""
        return self.eigenvector is not None or (self.operator is not None and self.operator.depends_on_eigen())


class MeshDataCacheEntry:
    # Not pinned to int32 like the mesh's own to_numpy output: the globally merged data assembled in
    # meshdatamerge is indexed across all processes and is built as int64.
    elem_indices:NPAnyIntArray
    elem_types:NPAnyIntArray

    def __init__(self,msh:AnySpatialMesh,key:MeshDataCacheKey):
        assert isinstance(msh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh))

        self.mesh=msh
        self.key=key
        # Mirrored as attributes since operators and output classes read them off the entry
        self.nondimensional=key.nondimensional
        self.tesselate_tri=key.tesselate_tri
        self.eigenvector:int | tuple[int,...] | None = key.eigenvector
        self.eigenmode = key.eigenmode
        self.history_index=key.history_index
        self.with_halos=key.with_halos
        self.operator=key.operator
        self.discontinuous=key.discontinuous
        self.add_eigen_to_mesh_positions=key.add_eigen_to_mesh_positions
        self.merged_eigendata:dict[int,dict[str,Any]]={}
        if self.eigenmode not in {"abs","real","imag","merge","angle"}:
            raise RuntimeError("Unknown eigenmode "+str(self.eigenmode))
        backup_dofs:NPFloatArray | None=None
        backup_pinned:NPFloatArray | None=None
        if isinstance(self.eigenvector,int):
            if self.eigenmode=="merge":
                self.eigenvector=(self.eigenvector,)
            else:
                backup_dofs, backup_pinned = self.mesh.get_problem().set_eigenfunction_as_dofs(self.eigenvector, mode=self.eigenmode,additive_mesh_positions=self.add_eigen_to_mesh_positions)

        if isinstance(self.eigenvector,tuple):
            # Tuples arrive here from MeshDataCacheKey.create, which normalizes any sequence of
            # eigenvector indices. Before the key was normalized this check only looked at list/set
            # and was therefore dead: a list of eigenvectors with a non-merge eigenmode used to
            # produce plain (non-eigen) data instead of raising.
            if self.eigenmode!="merge":
                raise RuntimeError("Multiple eigenvectors in MeshDataCache only works if eigenmode is set to 'merge'")

        if self.eigenmode=="merge" and self.eigenvector is not None:
            for ev in cast(Sequence[int],self.eigenvector):
                backup = self.mesh.get_problem().set_eigenfunction_as_dofs(ev,mode="real",additive_mesh_positions=self.add_eigen_to_mesh_positions)
                if backup_dofs is None:
                    backup_dofs, backup_pinned=backup
                real_nodal_values, elem_indices, elem_types, nodal_field_inds, real_D0_data, real_DL_data, elemental_field_inds = msh.to_numpy(key.tesselate_tri, key.nondimensional, key.history_index,key.discontinuous) #type:ignore
                self.mesh.get_problem().set_eigenfunction_as_dofs(ev, mode="imag",additive_mesh_positions=self.add_eigen_to_mesh_positions)
                imag_nodal_values, elem_indices, elem_types, nodal_field_inds, imag_D0_data, imag_DL_data, elemental_field_inds = msh.to_numpy(key.tesselate_tri, key.nondimensional, key.history_index,key.discontinuous) #type:ignore
                self.merged_eigendata[ev]={"nodal_values":(real_nodal_values,imag_nodal_values),"DL_data":(real_DL_data,imag_DL_data),"D0_data":(real_D0_data,imag_D0_data)}
            if backup_dofs is not None:
                assert backup_pinned is not None
                self.mesh.get_problem().set_all_values_at_current_time(backup_dofs, backup_pinned, not self.add_eigen_to_mesh_positions)
        assert isinstance(msh,(MeshFromTemplateBase,InterfaceMesh))

        self.nodal_values, self.elem_indices, self.elem_types, self.nodal_field_inds, self.D0_data, self.DL_data, self.elemental_field_inds = msh.to_numpy(key.tesselate_tri,key.nondimensional,key.history_index,key.discontinuous)

        self._dropped_node_rows:NPBoolArray | None=None # set below if halo node rows were removed
        if (not self.with_halos) and msh.is_mesh_distributed():
            # A row of elem_indices is a SUB-element, not a mesh element: with tesselate_tri a Quad9
            # becomes 8 triangles. So the owning element of a row must be looked up (it used to be
            # assumed to be element_pt(row), which indexed past nelement() as soon as anything was
            # split). D0/DL are indexed by the same rows and are dropped along with them - otherwise
            # the elemental fields end up belonging to different elements than the connectivity.
            src=msh.get_numpy_element_source_indices(key.tesselate_tri,key.discontinuous)
            owned=numpy.array([msh.element_pt(i).non_halo_proc_ID()<0 for i in range(msh.nelement())],dtype=bool)
            keep=owned[src] if len(src) else numpy.zeros((0,),dtype=bool)
            self.elem_indices=self.elem_indices[keep]
            self.elem_types=self.elem_types[keep]
            if not key.discontinuous:
                self.D0_data=self.D0_data[keep]
                self.DL_data=self.DL_data[keep]
                for eigendata in self.merged_eigendata.values():
                    eigendata["D0_data"]=tuple(d[keep] for d in eigendata["D0_data"])
                    eigendata["DL_data"]=tuple(d[keep] for d in eigendata["DL_data"])
            else:
                # Here every element has its own private copy of its nodes, so the node rows (and D0/DL,
                # which are indexed by them) belong to one element each and the halo elements' blocks
                # have to go as well - they are referenced by nothing once their elements are dropped.
                node_keep=numpy.repeat(owned,[msh.element_pt(i).nnode() for i in range(msh.nelement())]) if msh.nelement() else numpy.zeros((0,),dtype=bool)
                self._dropped_node_rows=node_keep # local expressions are evaluated later and must follow suit
                renumber=numpy.cumsum(node_keep)-1
                valid=(self.elem_indices>=0)&(self.elem_indices<len(node_keep))
                self.elem_indices=numpy.where(valid,renumber[numpy.clip(self.elem_indices,0,max(len(node_keep)-1,0))],0)
                self.nodal_values=self.nodal_values[node_keep]
                self.D0_data=self.D0_data[node_keep]
                self.DL_data=self.DL_data[node_keep]
                for eigendata in self.merged_eigendata.values():
                    for what in ("nodal_values","D0_data","DL_data"):
                        eigendata[what]=tuple(d[node_keep] for d in eigendata[what])


        if isinstance(self.eigenvector, int) :
            assert backup_dofs is not None and backup_pinned is not None # taken together with the eigenfunction above
            self.mesh.get_problem().set_all_values_at_current_time(backup_dofs, backup_pinned, not self.add_eigen_to_mesh_positions)

        self.interface_lines_segs:list[list[int]] | None=None
        self.interface_lines_segs_ninter:int | None=None

        self.nodal_local_exprs:dict[str,NPFloatArray]={}
        self.local_expr_indices:dict[str,int]={n:i for i,n in enumerate(self.mesh.list_local_expressions())}

        vector_fields = msh.get_eqtree().get_equations()._get_list_of_vector_fields(self.mesh.get_eqtree().get_code_gen())
        self.vector_fields:dict[str,list[str]] = {k: v for a in vector_fields for k, v in a.items()}
        tensor_fields = msh.get_eqtree().get_equations()._get_list_of_tensor_fields(self.mesh.get_eqtree().get_code_gen())
        self.tensor_fields:dict[str,list[list[str]]] = {k: v for a in tensor_fields for k, v in a.items()}
        self._add_implicit_vector_fields()

        self._additional_eigendata:dict[int,tuple[str,str,str]]={} # Index to pair of Re,Im
        self.is_global=False # This entry holds this rank's part of a distributed mesh, see from_arrays
        if self.operator is not None:
            self.operator.apply(self)

    #: Vectors the mesh writes itself instead of an equation declaring them, so they appear in no
    #: _vectorfields and used to be written as loose scalars - "normal_x", "normal_y" rather than a
    #: single "normal", and likewise the eigenperturbation of the position. Their components carry
    #: the same x/y/z suffixes as any other vector field, so registering them here is enough for
    #: everything downstream to treat them alike: the writer groups them, MeshDataCombineWith-
    #: Eigenfunction prefixes them, and the extrusions rotate them with the frame.
    IMPLICIT_VECTOR_FIELDS = ("coordinate", "lagrangian", "normal")

    def _add_implicit_vector_fields(self)->None:
        for stem in self.IMPLICIT_VECTOR_FIELDS:
            if stem in self.vector_fields:
                continue
            components=[stem+"_"+c for c in ("x","y","z") if stem+"_"+c in self.nodal_field_inds]
            if len(components)>1:
                self.vector_fields[stem]=components

    @classmethod
    def from_arrays(cls,msh:AnySpatialMesh,key:MeshDataCacheKey,nodal_values:NPFloatArray,elem_indices:NPAnyIntArray,elem_types:NPAnyIntArray,nodal_field_inds:dict[str,int],D0_data:NPFloatArray,DL_data:NPFloatArray,elemental_field_inds:dict[str,int],merged_eigendata:dict[int,dict[str,Any]],nodal_local_exprs:dict[str,NPFloatArray],local_expr_indices:dict[str,int],vector_fields:dict[str,list[str]],tensor_fields:dict[str,list[list[str]]] | None=None) -> "MeshDataCacheEntry":
        """Builds an entry from ready-made arrays instead of extracting them from the mesh.

        Used for the globally merged data of a distributed mesh (see
        :py:mod:`pyoomph.meshes.meshdatamerge`), whose arrays span all processes and therefore cannot
        come out of any single rank's mesh. ``msh`` is still kept, for the equation tree, the code
        generator and the units - all of which are the same on every rank.

        Local expressions must be passed in already evaluated: evaluating them lazily would ask the
        local mesh, i.e. return this rank's partition, silently misaligned with the merged nodes.
        """
        self=cls.__new__(cls)
        self.mesh=msh
        self.key=key
        self.nondimensional=key.nondimensional
        self.tesselate_tri=key.tesselate_tri
        self.eigenvector=key.eigenvector
        self.eigenmode=key.eigenmode
        self.history_index=key.history_index
        self.with_halos=key.with_halos
        self.operator=key.operator
        self.discontinuous=key.discontinuous
        self.add_eigen_to_mesh_positions=key.add_eigen_to_mesh_positions
        self.nodal_values=nodal_values
        self.elem_indices=elem_indices
        self.elem_types=elem_types
        self.nodal_field_inds=nodal_field_inds
        self.D0_data=D0_data
        self.DL_data=DL_data
        self.elemental_field_inds=elemental_field_inds
        self.merged_eigendata=merged_eigendata
        self.nodal_local_exprs=nodal_local_exprs
        self.local_expr_indices=local_expr_indices
        self.vector_fields=vector_fields
        self.tensor_fields=tensor_fields if tensor_fields is not None else {}
        self.interface_lines_segs=None
        self.interface_lines_segs_ninter=None
        self._additional_eigendata={}
        self.is_global=True
        return self

    def _evaluate_local_expression(self,name:str)->NPFloatArray:
        """Evaluates a local expression at the nodes, in the same row layout as ``nodal_values``.

        The mesh always evaluates over all of its node rows, so on a distributed mesh with
        discontinuous data the halo elements' node blocks - which were dropped from nodal_values as
        their elements are not ours - have to be dropped here too, or the values line up with the
        wrong nodes."""
        values=numpy.array(self.mesh.evaluate_local_expression_at_nodes(self.local_expr_indices[name],self.nondimensional,self.discontinuous)) #type:ignore
        if self._dropped_node_rows is not None and len(values)==len(self._dropped_node_rows):
            values=values[self._dropped_node_rows]
        return values #type:ignore

    def get_coordinates(self,lagrangian:bool=False)->NPFloatArray:
        if lagrangian:
            coordinates = [self.nodal_values[:, self.nodal_field_inds["lagrangian_x"]]]
            if "lagrangian_y" in self.nodal_field_inds.keys():
                coordinates.append(self.nodal_values[:, self.nodal_field_inds["lagrangian_y"]])
            if "lagrangian_z" in self.nodal_field_inds.keys():
                coordinates.append(self.nodal_values[:, self.nodal_field_inds["lagrangian_z"]])
            return numpy.array(coordinates,dtype=numpy.float64) #type:ignore
        else:
            coordinates = [self.nodal_values[:, self.nodal_field_inds["coordinate_x"]]]
            if "coordinate_y" in self.nodal_field_inds.keys():
                coordinates.append(self.nodal_values[:, self.nodal_field_inds["coordinate_y"]])
            if "coordinate_z" in self.nodal_field_inds.keys():
                coordinates.append(self.nodal_values[:, self.nodal_field_inds["coordinate_z"]])
            return numpy.array(coordinates,dtype=numpy.float64) #type:ignore

    def get_default_output_fields(self,rem_underscore:bool=True,rem_lagrangian:bool=True) -> list[str]:
        maxind=max(self.nodal_field_inds.values())
        maxindconti=maxind+1
        if len(self.elemental_field_inds)>0:
            maxind+=max(self.elemental_field_inds.values())+1
        srt=[""]*(maxind+1)
        for k,v in self.nodal_field_inds.items():
            srt[v]=k
        for k,v in self.elemental_field_inds.items():
            srt[v+maxindconti]=k


        if len(self.local_expr_indices.values())>0:
            maxind=max(self.local_expr_indices.values())
            le = [""] * (maxind + 1)
            for k,v in self.local_expr_indices.items():
                le[v]=k
            srt=srt+le
        srt = [s for s in srt if s != ""]


        if rem_lagrangian: # Kill the lagrangians
            srt=[s for s in srt if s not in {"lagrangian_x","lagrangian_y","lagrangian_z"}]
        if rem_underscore: # and the underscore
            srt = [s for s in srt if not s.startswith("_")]

        return srt

    @overload
    def get_unit(self,field:str,as_string:Literal[False]=...,with_brackets:bool=...)->ExpressionOrNum: ...

    @overload
    def get_unit(self,field:list[str],as_string:Literal[False]=...,with_brackets:bool=...)->list[ExpressionOrNum]: ...

    @overload
    def get_unit(self,field:str,as_string:Literal[True],with_brackets:bool=...)->str: ...

    @overload
    def get_unit(self,field:list[str],as_string:Literal[True],with_brackets:bool=...)->list[str]: ...

    def get_unit(self,field:str | list[str],as_string:bool=False,with_brackets:bool=True)->ExpressionOrNum | list[ExpressionOrNum] | str | list[str]:
        if isinstance(field,list):
            if as_string:
                return [self.get_unit(f,as_string=True,with_brackets=with_brackets) for f in field]
            else:
                return [self.get_unit(f,as_string=False,with_brackets=with_brackets) for f in field]
        if self.nondimensional or (field=="normal_x" or field=="normal_y" or field=="normal_z"):
            return "" if as_string else 1
        s:ExpressionOrNum
        if (field in self.local_expr_indices.keys()) and not (field is self.nodal_field_inds.keys()):
            s=self.mesh.get_code_gen()._get_local_expression_unit_factor(field)
        else:
            # Asked of the code generator, not of the equations directly: Equations.get_scaling needs
            # a code generator to be the current one and raises otherwise, which made get_unit usable
            # only from inside a scope that happened to have set one - text output has one, a plotter
            # does not. The code generator's own get_scaling sets and restores that scope around the
            # very same call, so the answer is unchanged wherever the old form worked at all.
            s = self.mesh.get_code_gen().get_scaling(field)
        if not isinstance(s,Expression): #type:ignore
            s=Expression(s)
        s = self.mesh.get_code_gen().expand_placeholders(s, False)
        _, unit, _, _ = _pyoomph.GiNaC_collect_units(s)
        res:"str | ExpressionOrNum"=1
        try:
            float(unit)
        except:
            if as_string:
                res = str(unit_to_string(unit, estimate_prefix=False))
            else:
                res=unit
        if res==1 and as_string:
            return ""
        else:
            if as_string:
                if with_brackets:
                    return "["+str(res)+"]"
                else:
                    return res
            else:
                return res




    def get_data(self,name:str | list[str] | list[list[str]],additional_eigenvector:int | None=None,eigen_real_imag:int | None=None)->NPFloatArray | None:
        assert isinstance(self.mesh,(InterfaceMesh,MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d))
        if isinstance(name, list):
            if isinstance(name[0], list): #tensor data
                tensor_mdata:list[list[NPFloatArray | None]]=[]
                nonzero_length=-1
                for row in name:
                    rowdata:list[NPFloatArray | None]=[]
                    for entry in row:
                        d=self.get_data(entry,additional_eigenvector,eigen_real_imag)
                        if d is None:
                            rowdata.append(None)
                        else:
                            rowdata.append(d)
                            if nonzero_length==-1:
                                nonzero_length=len(d)
                            elif nonzero_length!=len(d):
                                raise RuntimeError("Inconsistent data!")
                    tensor_mdata.append(rowdata)
                if nonzero_length==-1:
                    raise RuntimeError("Tensor data "+str(name)+" does not contain anything")
                zer=numpy.zeros((nonzero_length,))
                for i,datarow in enumerate(tensor_mdata):
                    for j,dataentry in enumerate(datarow):
                        if dataentry is None:
                            tensor_mdata[i][j]=zer
                return numpy.array(tensor_mdata) #type:ignore
            else:
                mdata:list[NPFloatArray | None]=[]
                nonzero_length=-1
                for n in name:
                    d=self.get_data(n,additional_eigenvector,eigen_real_imag)
                    if d is None:
                        mdata.append(None)
                    else:
                        mdata.append(d)
                        if nonzero_length==-1:
                            nonzero_length=len(d)
                        elif nonzero_length!=len(d):
                            raise RuntimeError("Inconsistent data!")
                if nonzero_length==-1:
                    raise RuntimeError("Vector data "+str(name)+" does not contain anything")
                zer=numpy.zeros((nonzero_length,))
                for i,dataentry in enumerate(mdata):
                    if dataentry is None:
                        mdata[i]=zer
                return numpy.array(mdata) #type:ignore

        if additional_eigenvector is not None:
            if additional_eigenvector not in self.merged_eigendata.keys():
                raise RuntimeError("Eigenvector " + str(additional_eigenvector) + " not allocated")
            eigendata = self.merged_eigendata[additional_eigenvector]
            if eigen_real_imag not in {0,1}:
                raise RuntimeError("eigen_real_imag must be either 0 for real or 1 for imag")
            if name in self.nodal_field_inds.keys():
                return eigendata["nodal_values"][eigen_real_imag][:, self.nodal_field_inds[name]]
            else:
                raise RuntimeError("Cannot get additional eigenvector data on non-nodal fields yet")

        if name is None:
            return None
        elif name in self.nodal_field_inds.keys():
            data:NPFloatArray = self.nodal_values[:, self.nodal_field_inds[name]]
        elif name in self.local_expr_indices.keys():
            if name in self.nodal_local_exprs.keys():
                data=self.nodal_local_exprs[name]
            elif self.is_global:
                # Evaluating it here would ask the local mesh, i.e. deliver this rank's partition for
                # a node array that spans all of them. Merged entries get all local expressions filled
                # in eagerly, so a missing one means it was not evaluated at merge time.
                raise RuntimeError("The local expression '"+name+"' was not gathered into this globally merged mesh data and cannot be evaluated on it afterwards")
            else:
                if isinstance(self.eigenvector, int):
                    base = self._evaluate_local_expression(name)
                    eps=1e-8
                    if self.eigenmode=="real" or self.eigenmode=="imag":
                        backup_dofs, backup_pinned,aampl = self.mesh.get_problem().set_eigenfunction_as_dofs(self.eigenvector,mode=self.eigenmode,perturb_amplitude=eps)
                        perturbed = self._evaluate_local_expression(name)
                        self.mesh.get_problem().set_all_values_at_current_time(backup_dofs, backup_pinned, not self.add_eigen_to_mesh_positions)
                        self.nodal_local_exprs[name] = (numpy.array(perturbed) - numpy.array(base)) * aampl / eps #type:ignore
                    else:
                        backup_dofs, backup_pinned, aampl_real = self.mesh.get_problem().set_eigenfunction_as_dofs(self.eigenvector, mode="real", perturb_amplitude=eps)
                        real_perturbed = self._evaluate_local_expression(name)
                        _, _, aampl_imag = self.mesh.get_problem().set_eigenfunction_as_dofs(self.eigenvector, mode="imag", perturb_amplitude=eps)
                        imag_perturbed = self._evaluate_local_expression(name)
                        self.mesh.get_problem().set_all_values_at_current_time(backup_dofs, backup_pinned , not self.add_eigen_to_mesh_positions)
                        le_real=(numpy.array(real_perturbed) - numpy.array(base)) * aampl_real / eps #type:ignore
                        le_imag = (numpy.array(imag_perturbed) - numpy.array(base)) * aampl_imag / eps #type:ignore
                        le_complex=le_real+(0+1j)*le_imag #type:ignore
                        if self.eigenmode=="abs":
                            le_result=numpy.absolute(le_complex) #type:ignore
                        elif self.eigenmode=="angle":
                            le_result = numpy.angle(le_complex) #type:ignore
                        else:
                            raise RuntimeError("Unknown eigenmode "+str(self.eigenmode))
                        self.nodal_local_exprs[name] = le_result #type:ignore


                else:
                    self.nodal_local_exprs[name]=numpy.array(self._evaluate_local_expression(name)) #type:ignore
                data=self.nodal_local_exprs[name]
        #elif name in self.elemental_field_inds.keys():
        #    if self.discontinuous:
        #        raise RuntimeError("DG finding here")
        #    else:
        #        return None
        else:
            return None
        return numpy.array(data) #type:ignore


    def get_interface_line_segments(self) -> tuple[list[list[int]], int]:
        if self.discontinuous:
            raise RuntimeError("get_interface_line_segments does not work for discontinuous caches")
        if self.interface_lines_segs is not None:
            assert self.interface_lines_segs_ninter is not None
            return self.interface_lines_segs,self.interface_lines_segs_ninter
        lines:list[list[int]] = []

        # Merge connected lines
        elms = [tuple([i for i in e]) for e in self.elem_indices]
        if not elms:
            # No elements of this interface on this process, which under --distribute is the normal
            # state of every rank the interface does not reach. There are no segments and no
            # intermediate node count to derive, and the walk below cannot start: ninter stayed None
            # and the assertion after the loop killed the whole job (droplet_spread_* under
            # --mpirun 2 --distribute). Callers unpack (segments, ninter) and iterate the segments,
            # so an empty list is the answer they already handle.
            self.interface_lines_segs=lines
            self.interface_lines_segs_ninter=0
            return lines,0
        elms_at_points:dict[int,list[int]] = {}
        inbetween_pts:dict[tuple[int,int],list[int]] = {}
        ninter=None
        for e in elms:
            elms_at_points.setdefault(e[0], []).append(e[-1])  #type:ignore
            elms_at_points.setdefault(e[-1], []).append(e[0]) #type:ignore
            inbetween_pts[(e[0], e[-1])] = list(e[1:-1]) #type:ignore
            # The reverse direction, for when the walk traverses this element from its far end. Two
            # things were wrong here: the key was (e[-1], e[1]) rather than (e[-1], e[0]), so the
            # entry was filed under a pair that is not this element's endpoints at all, and
            # reversed() is a one-shot iterator that yields nothing the second time it is read. The
            # effect was that a backwards traversal fell back to the FORWARD list and inserted the
            # intermediate nodes in the wrong order - invisible for one intermediate node per element
            # (C2), wrong for any higher-order space.
            inbetween_pts[(e[-1], e[0])] = list(reversed(e[1:-1])) #type:ignore
            if ninter is None:
                ninter=len(e[1:-1])
            else:
                if ninter!=len(e[1:-1]):
                    raise RuntimeError("Strange intermediate points...")
        assert ninter is not None
        starnode_history:list[int]=[]
        while len(elms_at_points) > 0:
            for n, neighs in elms_at_points.items():
                if len(neighs) == 1:
                    startnode = n
                    starnode_history.append(startnode)
                    break
            else:
                #print("SEEMS TO BE LOOPED! Startnode history "+str(starnode_history) )
                #print(elms_at_points)
                startnode = list(elms_at_points.keys())[0]  # Just any node. Seems to be looped

            currentcurve:list[int] = []
            currentnode = startnode

            while len(elms_at_points) > 0:
                #print(elms_at_points)
                while True:
                    currentcurve.append(currentnode)
                    if len(elms_at_points.get(currentnode, [])) == 0:
                        #print("No elem found",currentcurve)
                        for n, neighs in elms_at_points.items():
                            if len(neighs) == 1:
                                startnode = n
                                starnode_history.append(startnode)
                                break
                        else:
                            #print("SEEMS TO BE LOOPED! Startnode history " + str(starnode_history))
                            if len(elms_at_points)==0:
                                break
                            print(elms_at_points)
                            startnode = list(elms_at_points.keys())[0]  # Just any node. Seems to be looped
                        lines.append(currentcurve)
                        currentcurve=[]
                        currentnode = startnode
                        break
                    nextnode = elms_at_points[currentnode][0]
                    elms_at_points[currentnode].remove(nextnode)
                    if len(elms_at_points[currentnode]) == 0:
                        elms_at_points.pop(currentnode)
                    elms_at_points[nextnode].remove(currentnode)
                    if len(elms_at_points[nextnode]) == 0:
                        elms_at_points.pop(nextnode)
                    inbetween = inbetween_pts.get((currentnode, nextnode,),
                                                  inbetween_pts.get((nextnode, currentnode,), None))
                    if inbetween is not None:
                        for i in inbetween:
                            currentcurve.append(i)
                    currentnode = nextnode
                    if currentnode == startnode:
                        #print("LOOP")
                        currentcurve.append(startnode)  # Indicate a loop
                        break
                if len(currentcurve) > 0:
                    lines.append(currentcurve)
                    # Reset. `lines` holds a REFERENCE, so continuing to append here kept growing a
                    # curve that had already been emitted: after a closed loop was completed, the
                    # next fragment's nodes were tacked onto it. That is how a remeshed circular
                    # boundary came back as a loop whose last entries jumped half way across it,
                    # which in turn inflated any arclength computed from it by ~1.6x.
                    currentcurve = []
                if len(elms_at_points) > 0:
                    for n, neighs in elms_at_points.items():
                        if len(neighs) == 1:
                            startnode = n
                            break
                    else:
                        startnode = next(iter(elms_at_points.keys()))
                    currentnode = startnode
        self.interface_lines_segs=lines
        self.interface_lines_segs_ninter=ninter
        return lines,ninter
        

class MeshDataCache:
    """Holds the extracted data of one option set (:py:class:`MeshDataCacheKey`), one entry per mesh."""
    def __init__(self,tesselate_tri:bool=True,nondimensional:bool=False,eigenvector:int | Sequence[int] | None=None,eigenmode:MeshDataEigenModes="abs",history_index:int=0,with_halos:bool=False,operator:"MeshDataCacheOperatorBase | None"=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True,global_mesh:bool=False):
        # None as a value marks "merged elsewhere": with global_mesh the merged data only exists on
        # rank 0, but every rank stores something so that the hit/miss decision stays the same on all
        # of them - a rank that silently skipped the merge would leave the others in a dead gather.
        self._cache:dict[AnySpatialMesh,MeshDataCacheEntry | None]=dict()
        self.key=MeshDataCacheKey.create(tesselate_tri=tesselate_tri,nondimensional=nondimensional,eigenvector=eigenvector,eigenmode=eigenmode,history_index=history_index,with_halos=with_halos,operator=operator,discontinuous=discontinuous,add_eigen_to_mesh_positions=add_eigen_to_mesh_positions,global_mesh=global_mesh)

    # The options live in self.key; these keep the previous attribute access working
    @property
    def tesselate_tri(self)->bool: return self.key.tesselate_tri
    @property
    def nondimensional(self)->bool: return self.key.nondimensional
    @property
    def eigenvector(self)->int | tuple[int,...] | None: return self.key.eigenvector
    @property
    def eigenmode(self)->MeshDataEigenModes: return self.key.eigenmode
    @property
    def history_index(self)->int: return self.key.history_index
    @property
    def with_halos(self)->bool: return self.key.with_halos
    @property
    def operator(self)->"MeshDataCacheOperatorBase | None": return self.key.operator
    @property
    def discontinuous(self)->bool: return self.key.discontinuous
    @property
    def add_eigen_to_mesh_positions(self)->bool: return self.key.add_eigen_to_mesh_positions
    @property
    def global_mesh(self)->bool: return self.key.global_mesh

    def clear(self):
        self._cache=dict()

    def get_data(self,msh:AnySpatialMesh) -> MeshDataCacheEntry | None:

        if not (msh in self._cache.keys()):
            #print("CREATING MESH DATA",msh.get_full_name(),self.key)
            msh._setup_output_scales()

            if self.key.global_mesh:
                from .meshdatamerge import merge_global_mesh_data # imported here so a serial run never pulls in mpi4py through this path
                self._cache[msh] = merge_global_mesh_data(msh,self.key)
            else:
                self._cache[msh] = MeshDataCacheEntry(msh,self.key)
        else:
            pass
            #print("REUSING MESH DATA",msh.get_full_name(),self.key)
        #print(self._cache[msh].get_data("theta"))
        return self._cache[msh]


class MeshDataCacheStorage:
    def __init__(self):
        self._storage:dict[MeshDataCacheKey,MeshDataCache]={}


    def clear(self,only_eigens:bool=False):
        remkeys:list[MeshDataCacheKey]=[]
        for k,v in self._storage.items():
            if only_eigens:
                if k.depends_on_eigen:
                    v.clear()
                    remkeys.append(k)
            else:
                v.clear()
        if only_eigens:
            for k in remkeys:
                self._storage.pop(k, None)
        else:
            self._storage={}
        #print("STORAGE AFTER CLEAR",self._storage)

    def get_data(self,msh:AnySpatialMesh,nondimensional:bool,tesselate_tri:bool,eigenvector:int | Sequence[int] | None=None,eigenmode:MeshDataEigenModes="abs",history_index:int=0,with_halos:bool=False,operator:"MeshDataCacheOperatorBase | None"=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True,global_mesh:bool=False) -> MeshDataCacheEntry | None:
        key=MeshDataCacheKey.create(nondimensional=nondimensional,tesselate_tri=tesselate_tri,eigenvector=eigenvector,eigenmode=eigenmode,history_index=history_index,with_halos=with_halos,operator=operator,discontinuous=discontinuous,add_eigen_to_mesh_positions=add_eigen_to_mesh_positions,global_mesh=global_mesh)
        if key.global_mesh:
            from .meshdatamerge import needs_merging
            if not needs_merging(msh):
                # Serial, or mpirun without --distribute: every rank holds the entire mesh already, so
                # the global data IS the local data. Redirected to the same cache slot rather than
                # duplicating it, and no communication happens at all in this case.
                key=dataclass_replace(key,global_mesh=False)
            elif key.with_halos:
                raise RuntimeError("global_mesh=True cannot be combined with with_halos=True: halos only exist to stand in for the parts of the mesh a rank does not own, and the merged mesh is not missing any")
            elif key.operator is not None:
                raise NotImplementedError("Mesh data operators are not yet supported together with global_mesh=True. They must be applied to the merged data (an extrusion of one partition is not a part of the extrusion of the whole mesh), and the eigenfunction operator issues nested cache requests that would have to be resolved collectively first")
        if not key in self._storage.keys():
            #print("CREATING",key)
            msh._setup_output_scales()

            self._storage[key]=MeshDataCache(**key.as_kwargs())

        else:
            pass
            #print("REUSING",key)

        return self._storage[key].get_data(msh)




# --------------------------------------------------------------------------------------------
# Shared machinery of the two extrusion operators (Cartesian and rotational).
#
# Both used to walk every element in Python and, for each one, loop over every segment appending
# lists - and then grow the D0/DL/angle accumulators with numpy.concatenate ONCE PER ELEMENT, which
# is quadratic. On a 4000-element quad9 mesh with 64 segments that accumulation alone dominated
# everything else by an order of magnitude. The connectivity each branch produced is a fixed
# pattern, so it is expressed as a template table here and evaluated for all elements of a kind at
# once.
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _ExtrusionSubElement:
    """One output element produced from one source element at one segment offset.

    Each column is ``(local_node, offset_factor, offset_delta)`` and resolves to

        elem_indices[src, local_node] + (offset_factor*offs + offset_delta) * stride

    (taken modulo the total node count when the extrusion wraps). ``offset_factor==0`` pins the
    column to the base layer; that is how the rotational extrusion collapses the nodes sitting on
    the symmetry axis, which are shared by every segment.
    """
    elem_type:int
    nodes:tuple[tuple[int,int,int],...]


@dataclass(frozen=True)
class _ExtrusionTemplate:
    """How one kind of source element is extruded. ``step`` is the stride through the segment
    offsets: 0 means "produce one element for the whole extrusion" (used for elements degenerating
    onto the rotation axis), and ``None`` means "derive it from the template", see :py:meth:`stride`.
    """
    step:int | None
    subs:tuple[_ExtrusionSubElement,...]

    @property
    def nnode(self)->int:
        return len(self.subs[0].nodes)

    @property
    def stride(self)->int:
        """How many segment offsets one element consumes.

        Derived from the template rather than taken from the operator's phi_increm, because the two
        disagree exactly when the cells are not the ones the coordinate space implies: tesselate_tri
        splits a C2 mesh into LINEAR triangles, which span one layer, while phi_increm is still 2
        because the space is C2. Striding 2 over a template that spans 1 left every second slab
        unfilled - a solid with gaps through it.
        """
        if self.step is not None:
            return self.step
        return max(delta for sub in self.subs for _node,_factor,delta in sub.nodes)


def _sub(elem_type:int,*nodes:tuple[int,int,int])->_ExtrusionSubElement:
    return _ExtrusionSubElement(elem_type,tuple(nodes))


def _compact_field_indices(field_inds:dict[str,int])->None:
    """Renumbers the indices to 0..n-1, keeping the relative order.

    Not cosmetic: the extruded nodal_values array is built by walking the names in index order, so a
    gap left by a removed field would shift every later name off its own column. The extrusions used
    to call this after every single removal, which made it quadratic in the number of fields for no
    gain - the relabeling is order-preserving, so one call before the array is built is equivalent.
    """
    for newindex,(name,_) in enumerate(sorted(field_inds.items(),key=lambda item:item[1])):
        field_inds[name]=newindex


def _vector_components_present(vfield:str,field_inds:dict[str,int])->list[str]:
    """The x/y/z components of a vector field that survive an extrusion.

    Both extrusions fold the in-plane component ("_phi" azimuthally, "_normal" for the Cartesian
    mode) into the Cartesian ones and may add a "_z". The list stored in vector_fields has to be
    rebuilt from what is left afterwards: it used to keep naming the folded-away component and to
    miss the added one, so entry.vector_fields disagreed with entry.nodal_field_inds.
    """
    return [vfield+c for c in ("_x","_y","_z") if vfield+c in field_inds]


_TENSOR_DIRS=("x","y","z")


def _tensor_component_names(tname:str)->list[list[str]]:
    """The nine Cartesian component names of an extruded tensor, row-major."""
    return [[tname+"_"+_TENSOR_DIRS[i]+_TENSOR_DIRS[j] for j in range(3)] for i in range(3)]


def _padded_tensor_grid(grid:list[list[str]])->list[list[str]]:
    """A component-name grid padded to 3x3, absent entries as "".

    Grids arrive in three shapes: ndim x ndim from define_tensor_field in Cartesian coordinates,
    a full 3x3 with the azimuthal entry on the diagonal in axisymmetry, and a full 3x3 with holes
    from add_local_function, which leaves a symbolically zero component unregistered.
    """
    out=[["","",""],["","",""],["","",""]]
    for i,row in enumerate(grid[:3]):
        for j,name in enumerate(row[:3]):
            out[i][j]=name if name else ""
    return out


def _grid_is_symmetric(grid:list[list[str]])->bool:
    """Whether a padded name grid names the same field at (i,j) and (j,i).

    That is how both producers record symmetry: define_tensor_field repeats the upper-triangle name
    in the lower triangle, and register_local_expression does the same rather than registering a
    duplicate expression.
    """
    return all(grid[i][j]==grid[j][i] for i in range(3) for j in range(3))


def _outer_sum_operator(coefs:tuple[NPFloatArray,...]):
    """sum_k outer(coefs[k], arg_k).flatten(), the shape every extruded component takes.

    coefs is bound as a default: these closures are called long after the loop that built them has
    finished, the same trap the wavenumbers in the eigen operators fell into.
    """
    def op(*args:NPFloatArray,coefs:tuple[NPFloatArray,...]=coefs)->NPFloatArray:
        total=numpy.outer(coefs[0],args[0]).flatten()
        for c,a in zip(coefs[1:],args[1:]):
            total+=numpy.outer(c,a).flatten()
        return total #type:ignore
    return op


def _emit_tensor_frame_operators(tname:str,sources:list[list[list[tuple[str,NPFloatArray]]]],
                                 Q:NPFloatArray,nrows:int,nnodes:int,
                                 new_field_inds:dict[str,int],field_operators:dict[str,Any],
                                 symmetric:bool=False)->list[list[str]]:
    """Writes the nine Cartesian components of one tensor into ``field_operators``.

    ``sources[i][j]`` reconstructs the (i,j) component of the tensor **in its own frame** as a list
    of (base field name, weight over the extrusion rows): one unit-weighted entry for a base-state
    field, a cos/sin pair for an eigenmode. An empty list means the component is identically zero,
    which is how a symbolically zero one arrives - it is never registered as a field at all, so it
    cannot be named as an operator argument.

    ``Q[r]`` is the basis matrix at extrusion row r, so what gets written is ``Q T Q^T``. Q is the
    very matrix the vector operators of the same extrusion already apply, only once per index
    instead of once - deriving it from them is what keeps vectors and tensors consistent.
    """
    outnames=_tensor_component_names(tname)
    for a in range(3):
        for b in range(3):
            if symmetric and b<a:
                # Q T Q^T preserves symmetry exactly, so the lower triangle is the upper one under
                # a different name. Aliasing rather than recomputing keeps an extruded symmetric
                # tensor at six components, the way it was written before the extrusion.
                outnames[a][b]=outnames[b][a]
                continue
            contrib:dict[str,NPFloatArray]={}
            for i in range(3):
                for j in range(3):
                    if not sources[i][j]:
                        continue
                    qq=Q[:,a,i]*Q[:,b,j]
                    if not numpy.any(qq): # a structural zero of Q, exact rather than rounded
                        continue
                    for srcname,weight in sources[i][j]:
                        coef=qq*weight
                        if not numpy.any(coef):
                            continue
                        contrib[srcname]=contrib[srcname]+coef if srcname in contrib else coef
            name=outnames[a][b]
            if contrib:
                args=list(contrib.keys())
                field_operators[name]=[_outer_sum_operator(tuple(contrib[k] for k in args))]+args
            else:
                field_operators[name]=[lambda n=nrows*nnodes: numpy.zeros(n)] #type:ignore
            if name not in new_field_inds:
                new_field_inds[name]=max(new_field_inds.values())+1
    return outnames


def _extrude_tensor_fields(tensor_fields:dict[str,list[list[str]]],Q:NPFloatArray,nrows:int,nnodes:int,
                           available:set[str],
                           new_field_inds:dict[str,int],field_operators:dict[str,Any],
                           eigen:"Sequence[tuple[str,str,str,NPFloatArray,NPFloatArray]]"=())->dict[str,list[list[str]]]:
    """Rotates every tensor field of an extrusion into the Cartesian frame.

    Without this a tensor's components fall through to the plain tile of the dispatcher below, i.e.
    T_xx at phi=90 degrees still holds T_rr and the extruded tensor is wrong everywhere off the
    starting angle - the same bug vectors had before their rotation was written.

    ``eigen`` names the (real prefix, imaginary prefix, result prefix, real weights, imaginary
    weights) of each eigenmode. A tensor under one of those prefixes is reconstructed from its real
    and imaginary halves and rotated in one step, since both are weights on the same outer product.
    The imaginary weight carries the minus of Re[u*exp(I*m*phi)] = cos*Re - sin*Im; the caller passes
    it in already signed.

    ``available`` is the set of field names that actually carry data. A grid can name components
    that do not: MeshDataCombineWithEigenfunction prefixes a whole grid whether or not the
    eigenfunction has every component, which for a local expression it never does. Such a name is
    treated as absent, and a tensor left with nothing is dropped rather than written as zeros. The
    vector path relies on the same filtering, only spelled as a guard at each use.
    """
    def clean(grid:list[list[str]])->list[list[str]]:
        padded=_padded_tensor_grid(grid)
        return [[(name if name in available else "") for name in row] for row in padded]

    result:dict[str,list[list[str]]]={}
    consumed:set[str]=set()
    handled:set[str]=set()

    for prefixRe,prefixIm,prefixRes,cosw,sinw in eigen:
        for tname in list(tensor_fields.keys()):
            if not tname.startswith(prefixRe):
                continue
            stem=tname[len(prefixRe):]
            if prefixIm+stem not in tensor_fields:
                continue
            grid_re=clean(tensor_fields[tname])
            grid_im=clean(tensor_fields[prefixIm+stem])
            if not any(name for row in grid_re for name in row) and not any(name for row in grid_im for name in row):
                handled.add(tname)
                handled.add(prefixIm+stem)
                continue
            sources:list[list[list[Any]]]=[[[] for _j in range(3)] for _i in range(3)]
            for i in range(3):
                for j in range(3):
                    entry=[]
                    if grid_re[i][j]:
                        entry.append((grid_re[i][j],cosw))
                        consumed.add(grid_re[i][j])
                        # The generic scalar loop of the caller reconstructs EVERY field under the
                        # real prefix into one under the result prefix, component by component, so
                        # the ones whose names this rotation does not reuse - the azimuthal "_xa",
                        # "_ya", "_aa" of a tensor - are left over as stray scalars unless they are
                        # dropped here too.
                        consumed.add(prefixRes+grid_re[i][j][len(prefixRe):])
                    if grid_im[i][j]:
                        entry.append((grid_im[i][j],sinw))
                        consumed.add(grid_im[i][j])
                    sources[i][j]=entry
            result[prefixRes+stem]=_emit_tensor_frame_operators(prefixRes+stem,sources,Q,nrows,nnodes,
                                                                new_field_inds,field_operators,
                                                                _grid_is_symmetric(grid_re) and _grid_is_symmetric(grid_im))
            handled.add(tname)
            handled.add(prefixIm+stem)

    ones=numpy.ones(nrows)
    for tname,grid in tensor_fields.items():
        if tname in handled:
            continue
        padded=clean(grid)
        if not any(name for row in padded for name in row):
            continue
        sources=[[([(padded[i][j],ones)] if padded[i][j] else []) for j in range(3)] for i in range(3)]
        for row in padded:
            for name in row:
                if name:
                    consumed.add(name)
        result[tname]=_emit_tensor_frame_operators(tname,sources,Q,nrows,nnodes,
                                                   new_field_inds,field_operators,
                                                   _grid_is_symmetric(padded))

    # A source component that is not also one of the written ones - the azimuthal "_aa" of an
    # axisymmetric tensor, or either half of an eigenmode - would otherwise be tiled through as a
    # stray scalar that no longer means anything.
    written={name for grid in result.values() for row in grid for name in row}
    for name in consumed-written:
        new_field_inds.pop(name,None)
        field_operators.pop(name,None)
    return result


# Keyed by a template name; the mapping from element type (plus, on the rotation axis, which of its
# nodes lie on the axis) to a name is done in _extrusion_template_groups.
_EXTRUSION_TEMPLATES:dict[str,_ExtrusionTemplate]={
    # Point -> line along the extrusion direction
    "point1":_ExtrusionTemplate(None,(_sub(1,(0,1,0),(0,1,1)),)),
    "point2":_ExtrusionTemplate(None,(_sub(2,(0,1,0),(0,1,1),(0,1,2)),)),
    # A point sitting on the rotation axis traces out nothing at all, so it stays a single vertex
    "point_axis":_ExtrusionTemplate(0,(_sub(0,(0,0,0)),)),

    # LineC1 -> two triangles, or one if an end sits on the rotation axis
    "line2":_ExtrusionTemplate(1,(_sub(3,(0,1,0),(1,1,0),(1,1,1)),
                                  _sub(3,(0,1,0),(1,1,1),(0,1,1)))),
    "line2_ax0":_ExtrusionTemplate(1,(_sub(3,(0,0,0),(1,1,0),(1,1,1)),)),
    "line2_ax1":_ExtrusionTemplate(1,(_sub(3,(1,0,0),(0,1,0),(0,1,1)),)),

    # LineC2 -> eight triangles (the quadratic edge is tesselated), or four at the axis
    "line3":_ExtrusionTemplate(2,(_sub(3,(0,1,0),(1,1,0),(1,1,1)),
                                  _sub(3,(0,1,1),(0,1,0),(1,1,1)),
                                  _sub(3,(0,1,2),(0,1,1),(1,1,1)),
                                  _sub(3,(0,1,2),(1,1,1),(1,1,2)),
                                  _sub(3,(1,1,1),(1,1,0),(2,1,0)),
                                  _sub(3,(1,1,1),(2,1,0),(2,1,1)),
                                  _sub(3,(1,1,1),(2,1,1),(2,1,2)),
                                  _sub(3,(1,1,1),(2,1,2),(1,1,2)))),
    "line3_ax0":_ExtrusionTemplate(2,(_sub(3,(0,0,0),(1,1,0),(1,1,2)),
                                      _sub(3,(2,1,1),(1,1,0),(2,1,0)),
                                      _sub(3,(2,1,2),(1,1,2),(2,1,1)),
                                      _sub(3,(2,1,1),(1,1,2),(1,1,0)))),
    "line3_ax2":_ExtrusionTemplate(2,(_sub(3,(2,0,0),(1,1,2),(1,1,0)),
                                      _sub(3,(0,1,1),(1,1,0),(0,1,0)),
                                      _sub(3,(0,1,2),(1,1,2),(0,1,1)),
                                      _sub(3,(0,1,1),(1,1,2),(1,1,0)))),

    # Quad4 -> Hex8, Quad9 -> Hex27
    "quad4":_ExtrusionTemplate(None,(_sub(11,(2,1,0),(3,1,0),(0,1,0),(1,1,0),
                                             (2,1,1),(3,1,1),(0,1,1),(1,1,1)),)),
    "quad9":_ExtrusionTemplate(None,(_sub(14,*[(n,1,i) for i in range(3)
                                                       for n in (6,7,8,3,4,5,0,1,2)]),)),

    # Tri3 -> Wedge6, Tri6/Tri7 -> Wedge15 (the seventh, central node of a Tri7 is dropped)
    "tri3":_ExtrusionTemplate(None,(_sub(7,(0,1,1),(1,1,1),(2,1,1),(0,1,0),(1,1,0),(2,1,0)),)),
    "tri6":_ExtrusionTemplate(None,(_sub(77,(0,1,0),(1,1,0),(2,1,0),(0,1,2),(1,1,2),(2,1,2),
                                            (3,1,0),(4,1,0),(5,1,0),(3,1,2),(4,1,2),(5,1,2),
                                            (0,1,1),(1,1,1),(2,1,1)),)),
}

# Nodes actually read per source element type. elem_indices is zero-padded to the widest element in
# the mesh, so the axis test below must not look at the padding - column 0 of the padding would
# alias node 0, which is on the axis surprisingly often.
_EXTRUSION_NNODE:dict[int,int]={0:1, 1:2, 2:3, 3:3, 66:3, 6:4, 8:9, 9:6, 99:7}


def _extrusion_template_groups(elem_types:NPAnyIntArray,elem_indices:NPAnyIntArray,*,phi_increm:int,axis_nodes:"NPBoolArray | None",allow_line_c1:bool,collapse_axis_points:bool)->list[tuple[str,NPIntArray]]:
    """Splits the elements into groups that share one template, as (template name, row indices)."""
    groups:list[tuple[str,NPIntArray]]=[]
    for et in numpy.unique(elem_types):
        et=int(et)
        rows=numpy.flatnonzero(elem_types==et)
        if et==1 and not allow_line_c1:
            raise RuntimeError("Cartesian extrusion does not work with LineC1 elements")
        if et in {0,1,2} and axis_nodes is not None:
            nnode=_EXTRUSION_NNODE[et]
            on_axis=axis_nodes[elem_indices[rows][:,:nnode]]
            if et==0:
                if collapse_axis_points:
                    parts=[("point_axis",on_axis[:,0]),
                           ("point2" if phi_increm==2 else "point1",~on_axis[:,0])]
                else:
                    parts=[("point2" if phi_increm==2 else "point1",numpy.ones(len(rows),dtype=bool))]
            else:
                # The order matters: an element with BOTH ends on the axis takes the ax0 template,
                # matching the if/elif cascade this replaced. So does an element with only its mid
                # node on the axis, which falls into the "other end" template.
                name="line2" if et==1 else "line3"
                far=1 if et==1 else 2
                any_axis=on_axis.any(axis=1)
                parts=[(name+"_ax0",on_axis[:,0]),
                       (name+"_ax"+str(far),any_axis & ~on_axis[:,0]),
                       (name,~any_axis)]
            for name,mask in parts:
                if mask.any():
                    groups.append((name,rows[mask]))
            continue
        if et==0:
            groups.append(("point2" if phi_increm==2 else "point1",rows))
        elif et==1:
            groups.append(("line2",rows))
        elif et==2:
            groups.append(("line3",rows))
        elif et==8:
            groups.append(("quad9",rows))
        elif et==6:
            groups.append(("quad4",rows))
        elif et in {3,66}:
            groups.append(("tri3",rows))
        elif et in {9,99}:
            groups.append(("tri6",rows))
        else:
            raise RuntimeError("Implement element type "+str(et))
    return groups


def _extrude_element_connectivity(elem_types:NPAnyIntArray,elem_indices:NPAnyIntArray,*,stride:int,upper_limit:int,phi_increm:int,phi_row_for_step:Callable[[int],NPFloatArray],modulus:int | None,axis_nodes:"NPBoolArray | None"=None,allow_line_c1:bool=True,collapse_axis_points:bool=False)->tuple[NPIntArray,NPIntArray,NPFloatArray,NPIntArray]:
    """Extrudes a 0d/1d/2d connectivity into one dimension higher.

    Returns ``(new_elem_types, new_elem_indices, elemental_angles, counts)``, where ``counts[i]`` is
    how many output elements source element ``i`` produced. The rows are laid out exactly as the
    element-by-element loop this replaced produced them - source-major, then segment offset, then
    sub-element - because the output writers pair these rows with D0/DL by position
    (see :py:func:`pyoomph.output.meshio._convert_mesh_to_meshio`).

    ``phi_row_for_step`` is the only thing the two extrusion operators differ in: it maps a segment
    step to the angle (or axial position) assigned to each ring of created elements.
    """
    nelem=len(elem_types)
    if nelem==0:
        # numpy.array([]) would be float64 here, and the callers assign this straight onto the cache
        # entry where an integer connectivity is expected.
        return numpy.zeros((0,),dtype=int),numpy.zeros((0,0),dtype=int),numpy.zeros((0,)),numpy.zeros((0,),dtype=int)

    groups=_extrusion_template_groups(elem_types,elem_indices,phi_increm=phi_increm,axis_nodes=axis_nodes,allow_line_c1=allow_line_c1,collapse_axis_points=collapse_axis_points)

    phi_row_cache:dict[int,NPFloatArray]={}
    def phi_row(step:int)->NPFloatArray:
        if step not in phi_row_cache:
            phi_row_cache[step]=phi_row_for_step(step)
        return phi_row_cache[step]

    # First pass: sizes only, so the output arrays can be allocated once and every group can be
    # written straight into its final rows.
    counts=numpy.zeros((nelem,),dtype=int)
    resolved:list[tuple[_ExtrusionTemplate,NPIntArray,NPIntArray,NPFloatArray]]=[]
    maxl=0
    for name,rows in groups:
        tmpl=_EXTRUSION_TEMPLATES[name]
        step=tmpl.stride
        if step==0:
            offs=numpy.zeros((1,),dtype=int)
            row=numpy.array([phi_row(phi_increm)[0]])
        else:
            offs=numpy.arange(0,upper_limit,step)
            row=phi_row(step)
            if len(row)!=len(offs):
                raise RuntimeError("Inconsistent extrusion of '"+name+"' elements: "+str(len(offs))+" segment offsets but "+str(len(row))+" angles. upper_limit="+str(upper_limit)+" is not a multiple of the step "+str(step)+".")
        k=len(tmpl.subs)
        counts[rows]=len(offs)*k
        maxl=max(maxl,tmpl.nnode)
        resolved.append((tmpl,rows,offs,row))

    starts=numpy.concatenate(([0],numpy.cumsum(counts)))[:-1]
    total=int(counts.sum())
    new_elem_types=numpy.zeros((total,),dtype=int)
    new_elem_indices=numpy.zeros((total,maxl),dtype=int)
    elemental_phis=numpy.zeros((total,))

    for tmpl,rows,offs,row in resolved:
        k=len(tmpl.subs)
        nn=tmpl.nnode
        M=len(offs)
        loc=numpy.array([[c[0] for c in s.nodes] for s in tmpl.subs],dtype=int)       # (k,nn)
        fac=numpy.array([[c[1] for c in s.nodes] for s in tmpl.subs],dtype=int)
        dlt=numpy.array([[c[2] for c in s.nodes] for s in tmpl.subs],dtype=int)
        shift=(offs[:,None,None]*fac[None]+dlt[None])*stride                          # (M,k,nn)
        dest=(starts[rows][:,None]+numpy.arange(M*k)[None,:]).ravel()
        sub_types=numpy.array([s.elem_type for s in tmpl.subs],dtype=int)
        new_elem_types[dest]=numpy.tile(sub_types,len(rows)*M)
        elemental_phis[dest]=numpy.tile(numpy.repeat(row,k),len(rows))
        # Chunked so the (Nt,M,k,nn) temporary stays bounded on large meshes
        chunk=max(1,(1<<23)//max(M*k*nn,1))
        for lo in range(0,len(rows),chunk):
            rw=rows[lo:lo+chunk]
            idx=elem_indices[rw][:,loc][:,None]+shift[None]                            # (Nt,M,k,nn)
            if modulus is not None:
                idx=idx%modulus
            new_elem_indices[dest[lo*M*k:(lo+len(rw))*M*k],:nn]=idx.reshape(-1,nn)

    return new_elem_types,new_elem_indices,elemental_phis,counts


class MeshDataCacheOperatorBase:
    """A generic operator that can be applied to a :py:class:`~pyoomph.output.meshio.MeshDataCacheEntry` before it is e.g. written to a file.
    Specialized classes can e.g. merge an eigenfunction to the data or extrude the data azimuthally or along a Cartesian direction.
    """
    def __init__(self):
        super(MeshDataCacheOperatorBase, self).__init__()

    def apply(self,base:MeshDataCacheEntry)->None:
        raise RuntimeError("Specify")

    def _materialise_local_expressions(self,base:MeshDataCacheEntry)->None:
        """Folds every local expression into the nodal field block.

        An extrusion rewrites nodal_values and nodal_field_inds, but a local expression lives in
        nodal_local_exprs and is evaluated against the ORIGINAL mesh, so it kept the original node
        count and the writer died with "len(points) = 784, but len(point_data[...]) = 49". That hit
        scalars, vectors and tensors alike, i.e. every extrusion of any output carrying a
        LocalExpressions. Materialising them first turns them into ordinary columns that the
        extrusion below transforms like any other field.
        """
        if not base.local_expr_indices:
            return
        columns:list[NPFloatArray]=[]
        nextind=max(base.nodal_field_inds.values())+1 if base.nodal_field_inds else 0
        for name in sorted(base.local_expr_indices.keys(),key=lambda n: base.local_expr_indices[n]):
            data=base.get_data(name)
            if data is None:
                continue
            base.nodal_field_inds[name]=nextind
            nextind+=1
            columns.append(numpy.asarray(data))
        if columns:
            base.nodal_values=numpy.column_stack([base.nodal_values]+columns) #type:ignore
        base.local_expr_indices={}
        base.nodal_local_exprs={}

    def depends_on_eigen(self)->bool:
        return False

    def __add__(self, other:"MeshDataCacheOperatorBase")->"MeshDataCacheCombinedOperator":
        return MeshDataCacheCombinedOperator(self,other)

    def _get_elem_dim(self,base:MeshDataCacheEntry) -> Literal[0, 1, 2, 3]:
        result:Literal[0, 1, 2, 3] | None=None
        et=set(base.elem_types)
        et3d={14,11,10,100,4}
        et2d={6,8,9,99,3}
        et1d = {1,2}
        et0d = {0}
        if len(et.intersection(et3d))>0:
            result=3
        if len(et.intersection(et2d))>0:
            if result is not None:
                raise RuntimeError("Got element types with different dimensions: "+str(et))
            result=2
        if len(et.intersection(et1d))>0:
            if result is not None:
                raise RuntimeError("Got element types with different dimensions: "+str(et))
            result=1
        if len(et.intersection(et0d))>0:
            if result is not None:
                raise RuntimeError("Got element types with different dimensions: " + str(et))
            result = 0
        if result is None:
            raise RuntimeError("Cannot determine element dimension "+str(et))
        return result


class MeshDataCacheCombinedOperator(MeshDataCacheOperatorBase):
    def __init__(self,*lst:MeshDataCacheOperatorBase):
        super(MeshDataCacheCombinedOperator, self).__init__()
        self._lst=list(lst)

    def apply(self,base:MeshDataCacheEntry):
        for op in self._lst:
            op.apply(base)

    def depends_on_eigen(self)->bool:
        for op in self._lst:
            if op.depends_on_eigen():
                return True
        return False

class MeshDataCombineWithEigenfunction(MeshDataCacheOperatorBase):
    """
    Can be added as ``operator`` to :py:class:`~pyoomph.output.meshio.MeshFileOutput` to combine the solution with the eigenfunction data. Both will be written to the same file and can be postprocessed in e.g. Paraview.

    Args:
        eigenindex: Index of the eigenfunction to combine with the solution. Can be a single index or a list of indices.
        eigen_prefix_real: Prefix for the real part of the eigenfunction data.
        eigen_prefix_imag: Prefix for the imaginary part of the eigenfunction data.
        eigen_prefix_merged: Prefix for the merged eigenfunction data.
        add_eigen_to_mesh_positions: If True, the eigenfunction data will be added to the mesh positions. 
    """
    def __init__(self,eigenindex:int | Sequence[int],eigen_prefix_real:str="EigenRe_",eigen_prefix_imag:str="EigenIm_",eigen_prefix_merged:str="Eigen_",add_eigen_to_mesh_positions=False):
        super(MeshDataCombineWithEigenfunction, self).__init__()
        
        if isinstance(eigenindex,int):
            self.eigenindex=[eigenindex]
        else:
            self.eigenindex=list(eigenindex)
        self.eigen_prefix_real=eigen_prefix_real
        self.eigen_prefix_imag=eigen_prefix_imag
        self.eigen_prefix_merged=eigen_prefix_merged
        self.add_eigen_to_mesh_positions=add_eigen_to_mesh_positions
        
    def depends_on_eigen(self) -> bool:
        return True

    def apply(self,base:MeshDataCacheEntry):
        hidden_fields={"lagrangian_x","lagrangian_y","lagrangian_z"}
        if not base.mesh.get_eqtree().get_code_gen()._coordinates_as_dofs:
            hidden_fields=hidden_fields.union({"coordinate_x","coordinate_y","coordinate_z"})
        evs=base.mesh.get_problem().get_last_eigenvalues()
        if len(base._additional_eigendata)>0: #type:ignore
            raise RuntimeError("Already added other eigenfunctions to the mesh data operator. Please combine them in one MeshDataCombineWithEigenfunction([index1,index2,...])")
        if evs is None:
            return
        
        for eigenindex in self.eigenindex:
            if eigenindex>=len(evs):
                continue
            eigenreal=base.mesh.get_problem().get_cached_mesh_data(base.mesh,eigenmode="real",eigenvector=eigenindex,tesselate_tri=base.tesselate_tri,history_index=base.history_index,with_halos=base.with_halos,discontinuous=base.discontinuous,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions)
            eigenimag=base.mesh.get_problem().get_cached_mesh_data(base.mesh,eigenmode="imag",eigenvector=eigenindex,tesselate_tri=base.tesselate_tri,history_index=base.history_index,with_halos=base.with_halos,discontinuous=base.discontinuous,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions)
            if len(base.elem_types)!=len(eigenreal.elem_types):
                raise RuntimeError("Mismatching element count")


            def process(eigendata:MeshDataCacheEntry,prefix:str) -> str:
                # Everything here appends the eigenfunction's fields behind the base fields. It used
                # to do so one column at a time with numpy.c_ / numpy.concatenate, i.e. reallocating
                # and copying the whole block once per field - O(nfields^2 * nnodes). Selecting all
                # the columns first and stacking once is the same result for one allocation.
                new_nodal_field_inds=base.nodal_field_inds.copy()
                if len(self.eigenindex)>1:
                    prefix+=str(eigenindex)+"_"
                sel=[index for fn,index in eigendata.nodal_field_inds.items() if fn not in hidden_fields]
                nxt=max(new_nodal_field_inds.values())+1
                for j,fn in enumerate(fn for fn in eigendata.nodal_field_inds if fn not in hidden_fields):
                    # A prefixed name colliding with an existing one overwrites its index but still
                    # adds a column, which is why the counter runs over all selected fields
                    new_nodal_field_inds[prefix+fn]=nxt+j
                new_nodal_values=numpy.hstack((base.nodal_values,eigendata.nodal_values[:,sel])) if sel else base.nodal_values.copy()

                new_elem_field_inds={}
                rev_field_inds={i:n for n,i in base.elemental_field_inds.items()}
                rev_field_inds_eig={i:n for n,i in eigendata.elemental_field_inds.items()}
                cnt=0

                num_DL=base.DL_data.shape[1]
                for iDL in range(num_DL):
                    new_elem_field_inds[rev_field_inds[iDL]]=cnt
                    cnt+=1
                sel_DL=[]
                for iDL in range(eigendata.DL_data.shape[1]):
                    if rev_field_inds_eig[iDL] in hidden_fields:
                        continue
                    # rev_field_inds_eig, not rev_field_inds: the name has to come from the
                    # eigendata whose column is being appended. It used to read the base's map,
                    # which happens to agree only because both entries come from the same mesh and
                    # therefore share their DL ordering.
                    new_elem_field_inds[prefix+rev_field_inds_eig[iDL]]=cnt
                    sel_DL.append(iDL)
                    cnt+=1
                # concatenate on axis 1 covers both layouts: DL_data is (nelem,nfields,nvalues) for
                # continuous data and (nelem,nfields) for discontinuous
                new_DL_data=numpy.concatenate((base.DL_data,eigendata.DL_data[:,sel_DL]),axis=1) if sel_DL else base.DL_data.copy()

                for iD0 in range(base.D0_data.shape[1]):
                    new_elem_field_inds[rev_field_inds[iD0+num_DL]]=cnt
                    cnt+=1
                sel_D0=[]
                for iD0 in range(eigendata.D0_data.shape[1]):
                    if rev_field_inds_eig[iD0+eigendata.DL_data.shape[1]] in hidden_fields:
                        continue
                    new_elem_field_inds[prefix+rev_field_inds_eig[iD0+eigendata.DL_data.shape[1]]]=cnt
                    sel_D0.append(iD0)
                    cnt+=1
                new_D0_data=numpy.concatenate((base.D0_data,eigendata.D0_data[:,sel_D0]),axis=1) if sel_D0 else base.D0_data.copy()

                for vector_name,compo_names in eigendata.vector_fields.items():
                    base.vector_fields[prefix+vector_name]=[prefix+compo_name for compo_name in compo_names]
                for tensor_name,compo_rows in eigendata.tensor_fields.items():
                    # An empty name marks a component that is identically zero and was never
                    # registered, so it must stay empty rather than become a prefixed non-name.
                    base.tensor_fields[prefix+tensor_name]=[[(prefix+c if c else "") for c in row] for row in compo_rows]
                base.nodal_values = new_nodal_values
                base.nodal_field_inds = new_nodal_field_inds
                base.elemental_field_inds=new_elem_field_inds

                base.D0_data=new_D0_data
                base.DL_data=new_DL_data
                return prefix

            preRe=process(eigenreal,self.eigen_prefix_real)
            preIm=process(eigenimag, self.eigen_prefix_imag)
            preMerge=self.eigen_prefix_merged
            if len(self.eigenindex) > 1:
                preMerge += str(eigenindex) + "_"
            base._additional_eigendata[eigenindex]=(preRe,preIm,preMerge) #type:ignore




class MeshDataCartesianExtrusion(MeshDataCacheOperatorBase):
    """
    Can be added as ``operator`` to :py:class:`~pyoomph.output.meshio.MeshFileOutput` to extrude the mesh in the z-direction. Most useful combined with :py:class:`MeshDataCombineWithEigenfunction` and Cartesian normal mode stability analysis.
    
    Args:
        n_segments: Number of segments in the z-direction.
        default_length: Default length of the extrusion (when no wave number is available).
        phase: Axial offset the eigenmode expansion starts at. Until this was fixed it shifted only
            the elemental (D0/DL) fields, not the nodal ones.
        apply_k_mode_expansion: If True, the extrusion will consider the exp(i*k*z) factor of the eigenfunction.
        use_k_for_length: If True, the length of the extrusion will be determined by the wave number (if available, otherwise default_length).
        numperiods: Number of periods to extrude (in terms of either default_length or 2*pi/k).
    """
    def __init__(self,n_segments:int=32,default_length=1,phase:float=0.0,apply_k_mode_expansion:bool=True,use_k_for_length:bool=True,numperiods:float=1):
        super(MeshDataCartesianExtrusion, self).__init__()
        self.n_segments=n_segments
        self.default_length=default_length
        self.phase=phase
        self.apply_k_mode_expansion=apply_k_mode_expansion
        self.use_k_for_length=use_k_for_length
        self.numperiods=numperiods
        
    def apply(self,base:MeshDataCacheEntry):
        self._materialise_local_expressions(base)
        n_segments=self.n_segments
        phi_increm=1
        if base.mesh._eqtree.get_code_gen()._coordinate_space not in {"C1","C1TB"}: 
            n_segments*=2        
            phi_increm=2
        
            

        # Getting the length
        L=self.default_length
        last_eigenmodes_k=base.mesh.get_problem().get_last_eigenmodes_k()
        if last_eigenmodes_k is not None:
            if self.use_k_for_length:
                kcommon=None
                for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                    if eigenindex<len(last_eigenmodes_k):
                        k=last_eigenmodes_k[eigenindex] #type:ignore
                        if kcommon is None:
                            kcommon=k
                        elif kcommon!=k:
                            kcommon=None
                            break
                if kcommon is not None:
                    L=2*numpy.pi/kcommon*self.numperiods
        #print("GOT L",L,"k",2*numpy.pi/L,kcommon)
        zs=numpy.linspace(0,L,n_segments+1,endpoint=True)
        
        
        stride = base.nodal_values.shape[0]

        new_nodal_values:list[NPFloatArray]=[]
        new_nodal_field_inds=base.nodal_field_inds.copy()
        # name -> either a plain source field name or [operator, *its argument field names]
        field_operators:dict[str,Any]={}

        vector_fields=base.vector_fields.copy()
#        vector_fields["coordinate"]=["coordinate_x","coordinate_y"]
        rev_vector_fields={}
        for a, b in vector_fields.items():
            for c in b:
                rev_vector_fields[c] = a


        if "coordinate_y" in base.nodal_field_inds:
            new_nodal_field_inds["coordinate_z"]=max(new_nodal_field_inds.values())+1
            if "lagrangian_z" in base.nodal_field_inds:
                new_nodal_field_inds["lagrangian_z"] = max(new_nodal_field_inds.values()) + 1
            if "normal_x" in base.nodal_field_inds:
                new_nodal_field_inds["normal_z"] = max(new_nodal_field_inds.values()) + 1
            

            #field_operators["coordinate_z"] = [lambda cy: numpy.tile(cy, n_segments+1), "coordinate_y"] #type:ignore
            field_operators["coordinate_z"] = [lambda : numpy.repeat(zs,len(base.nodal_values[:,0])).flatten()] #type:ignore
            if "lagrangian_z" in base.nodal_field_inds:
                field_operators["lagrangian_z"] = [lambda cy: numpy.tile(cy, n_segments+1), "lagrangian_y"] #type:ignore
            field_operators["normal_z"] = [lambda ny: numpy.tile(ny,n_segments+1), "normal_y"] #type:ignore
        elif "coordinate_x" not in base.nodal_field_inds:
            # 0d to 1d
            new_nodal_field_inds["coordinate_x"] = max(new_nodal_field_inds.values()) + 1
            new_nodal_field_inds["lagrangian_x"] = max(new_nodal_field_inds.values()) + 1
            #if "normal_x" in base.nodal_field_inds:
            #    new_nodal_field_inds["normal_x"] = max(new_nodal_field_inds.values()) + 1
            field_operators["coordinate_x"] = [lambda : numpy.repeat(zs,len(base.nodal_values[:,0])).flatten()] #type:ignore
            field_operators["lagrangian_x"] = field_operators["coordinate_x"]
        else:
            new_nodal_field_inds["coordinate_y"] = max(new_nodal_field_inds.values()) + 1
            new_nodal_field_inds["lagrangian_y"] = max(new_nodal_field_inds.values()) + 1
            if "normal_x" in base.nodal_field_inds:
                new_nodal_field_inds["normal_y"] = max(new_nodal_field_inds.values()) + 1
            field_operators["coordinate_y"] = [lambda : numpy.repeat(zs,len(base.nodal_values[:,0])).flatten()] #type:ignore
            field_operators["lagrangian_y"] = field_operators["coordinate_y"]


        completed_eigen_vector_fields=set() #type:ignore
        if self.apply_k_mode_expansion and base.mesh.get_problem().get_last_eigenmodes_k() is not None: #type:ignore
            for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                prefixRe=prefixPair[0]
                prefixIm = prefixPair[1]
                prefixRes = prefixPair[2]
                for fn,findex in base.nodal_field_inds.items(): #type:ignore
                    if fn.startswith(prefixRe):
                        fnRe=fn
                        fnIm=prefixIm+fn[len(prefixRe):]
                        fnRes=prefixRes+fn[len(prefixRe):]
                        del new_nodal_field_inds[fnRe]
                        del new_nodal_field_inds[fnIm]
                        new_nodal_field_inds[fnRes]=max(new_nodal_field_inds.values()) + 1
                        k=base.mesh.get_problem().get_last_eigenmodes_k()[eigenindex] #type:ignore
                        phis=numpy.linspace(0,2*numpy.pi*self.numperiods/k,n_segments+1,endpoint=True)+self.phase
                        
                        
                        # k/phis (m/phis) are bound as defaults: these closures are only called much later, after
                        # every loop has finished, so with more than one entry in _additional_eigendata
                        # they all used to be evaluated with the LAST eigenindex's wavenumber.
                        # Same convention as the rotational one, and as the vector operators below,
                        # which already had the minus. Verified with d/dx_extra = I*k.
                        field_operators[fnRes] = [lambda RealPart,ImagPart,k=k,phis=phis : numpy.outer(numpy.cos(k*phis), RealPart).flatten() - numpy.outer(numpy.sin(k*phis), ImagPart).flatten(), fnRe,fnIm] #type:ignore

                        if fnRe in rev_vector_fields:
                            ReVector=rev_vector_fields[fnRe] #type:ignore
                            ImVector=rev_vector_fields[fnIm] #type:ignore
                            ResVector=prefixRes+rev_vector_fields[fnRe][len(prefixRe):] #type:ignore
                            vector_fields[ResVector]=[prefixRes+compofn[len(prefixRe):] for compofn in vector_fields[ReVector]] #type:ignore
                            del vector_fields[ReVector] #type:ignore
                            del vector_fields[ImVector] #type:ignore
                            rev_vector_fields = {}
                            for a, b in vector_fields.items():
                                for c in b:
                                    rev_vector_fields[c] = a
                            #print(vector_fields[ResVector])
                            #raise RuntimeError("HEREH")
                            #field_operators[fnRes] = [lambda RealPart, ImagPart: numpy.outer(numpy.cos(m * phis),RealPart).flatten() + numpy.outer(numpy.sin(m * phis), ImagPart).flatten(), fnRe, fnIm]
                # Second iteration to patch the vectors
                for vecname,veccompos in vector_fields.items():
                    if vecname.startswith(prefixRes):
                        composRes = [fn for fn in veccompos]
                        composIm = [prefixIm + fn[len(prefixRes):] for fn in veccompos]
                        composRe = [prefixRe + fn[len(prefixRes):] for fn in veccompos]
                        r_index=None
                        phi_index=None
                        #print("PATCHING VECTOR",vecname)
                        for cindex,componame in enumerate(composRes):
                            if componame.endswith("_x"):
                                r_index=cindex
                            elif componame.endswith("_normal"):
                                phi_index=cindex

                        k=base.mesh.get_problem().get_last_eigenmodes_k()[eigenindex] #type:ignore
                        if r_index is not None and phi_index is not None:
                            #print("RINDEX",r_index,phi_index)
                            #print("K",eigenindex,k)
                            phis=numpy.linspace(0,2*numpy.pi*self.numperiods/k,n_segments+1,endpoint=True)+self.phase
                            def get_x_component(ReR,ImR,ReP,ImP,k=k,phis=phis): #type:ignore
                                #print("XCOMPONENT",vecname, len(ReR),len(ImR),len(ReP),len(ImP))
                                
                                return numpy.outer(numpy.cos(k * phis),ReR).flatten()-numpy.outer(numpy.sin(k * phis),ImR).flatten()
                                #Vr_cos_phi=numpy.outer(numpy.cos(k * phis)*numpy.cos(phis),ReR).flatten()+numpy.outer(numpy.sin(k * phis)*numpy.cos(phis),ImR).flatten() #type:ignore
                                #Vphi_sin_phi=numpy.outer(numpy.cos(k * phis)*numpy.sin(phis),ReP).flatten()+numpy.outer(numpy.sin(k * phis)*numpy.sin(phis),ImP).flatten() #type:ignore
                                #return Vr_cos_phi+0*Vphi_sin_phi #type:ignore
                            def get_y_component(ReR,ImR,ReP,ImP,k=k,phis=phis): #type:ignore
                                #print("YCOMPONENT",vecname,len(ReR),len(ImR),len(ReP),len(ImP))             
                                #print("MAGS YCOMPONENT",vecname,numpy.amax(numpy.absolute(ReR)),numpy.amax(numpy.absolute(ImR)),numpy.amax(numpy.absolute(ReP)),numpy.amax(numpy.absolute(ImP)))
                                return numpy.outer(numpy.cos(k * phis),ReP).flatten()-numpy.outer(numpy.sin(k * phis),ImP).flatten()
                                #Vr_sin_phi=numpy.outer(numpy.cos(k * phis)*numpy.sin(phis),ReR).flatten()+numpy.outer(numpy.sin(k * phis)*numpy.sin(phis),ImR).flatten() #type:ignore
                                #Vphi_cos_phi=numpy.outer(numpy.cos(k * phis)*numpy.cos(phis),ReP).flatten()+numpy.outer(numpy.sin(k * phis)*numpy.cos(phis),ImP).flatten() #type:ignore
                                #return Vr_sin_phi-Vphi_cos_phi #type:ignore
                            field_operators[composRes[r_index]]=[get_x_component,composRe[r_index],composIm[r_index],composRe[phi_index],composIm[phi_index]] 
                            field_operators[composRes[phi_index]] = [get_y_component, composRe[r_index],composIm[r_index], composRe[phi_index],composIm[phi_index]]
                            missing_dir=["x","y","z"][len(composRes)-1]
                            yname=vecname+"_"+missing_dir
                            #print("YNAME",yname)
                            field_operators[yname]=field_operators.pop(composRes[phi_index]) #type:ignore
                            new_nodal_field_inds[yname]=new_nodal_field_inds.pop(composRes[phi_index])
                            completed_eigen_vector_fields.add(vecname) #type:ignore
                            #if len(composRes)>2:
                            #    new_nodal_field_inds[vecname + "_z"] = max(new_nodal_field_inds.values()) + 1
                            #    field_operators[vecname+"_z"]= [lambda ReVy,ImVy: numpy.outer(numpy.cos(m * phis), ReVy).flatten()+numpy.outer(numpy.sin(m * phis), ImVy).flatten(),prefixRe + veccompos[0][len(prefixRes):-len("_x")] + "_y",prefixIm + veccompos[0][len(prefixRes):-len("_x")] + "_y"] #type:ignore
                            vector_fields[vecname]=[vecname+component for component in ["_x","_y","_z"][0:len(composRes)]]
                            #print(new_nodal_field_inds,vector_fields)
                        else:
                            # No "_normal" component to fold in. This extrusion translates rather
                            # than turning, so the scalar loop above already produced every component
                            # of this vector correctly and the plain vector loop below must be kept
                            # off it: that one composes its operator from a SOURCE field of the same
                            # name, which for an eigen result does not exist. "normal" and
                            # "coordinate" started arriving here once the mesh-written vectors were
                            # registered as vector fields at all.
                            completed_eigen_vector_fields.add(vecname) #type:ignore
                            vector_fields[vecname]=_vector_components_present(vecname,new_nodal_field_inds)
                    # There used to be an else branch here setting field_operators[vecname+"_y"]
                    # from vecname+"_normal". It was dead: it fires only for the non-eigen vector
                    # fields, which the loop below overwrites in every case where the entry would
                    # ever be read.

        for vfield,components in vector_fields.items(): #type:ignore
            if vfield in completed_eigen_vector_fields:
                continue
            if vfield+"_x" in new_nodal_field_inds:
                if vfield+"_y" in new_nodal_field_inds:
                    field_operators[vfield+"_y"]= [lambda vy: numpy.tile(vy,n_segments+1), vfield+"_y"] #type:ignore
                    if vfield+"_normal" in new_nodal_field_inds:
                        new_nodal_field_inds[vfield+"_z"] = max(new_nodal_field_inds.values()) + 1
                        field_operators[vfield+"_z"]= [lambda vy: numpy.tile(vy,n_segments+1), vfield+"_normal"] #type:ignore
                else:
                    field_operators[vfield+"_x"]= [lambda vy: numpy.tile(vy,n_segments+1), vfield+"_x"] #type:ignore
                    if vfield+"_normal" in new_nodal_field_inds:
                        new_nodal_field_inds[vfield + "_y"] = max(new_nodal_field_inds.values()) + 1
                        field_operators[vfield+"_y"]= [lambda vy: numpy.tile(vy,n_segments+1), vfield+"_normal"] #type:ignore
                if vfield+"_normal" in new_nodal_field_inds:
                    del new_nodal_field_inds[vfield+"_normal"]
                vector_fields[vfield]=_vector_components_present(vfield,new_nodal_field_inds)

        # Tensors. This extrusion translates along the new axis rather than turning about one, so
        # the basis does not rotate and Q is the identity: a tensor's slot 2 already IS the extra
        # direction (matrix() pads to 3x3 and no coordinate system names a tensor component
        # "_normal", unlike a vector). The base state therefore needs nothing the plain tile of the
        # dispatcher below would not do - what does need doing is the eigenmode reconstruction,
        # which otherwise leaves the real and imaginary halves lying around unrecombined.
        Q=numpy.broadcast_to(numpy.eye(3),(n_segments+1,3,3))
        eigen_tensor_modes=[]
        if self.apply_k_mode_expansion and base.mesh.get_problem().get_last_eigenmodes_k() is not None: #type:ignore
            for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                k=base.mesh.get_problem().get_last_eigenmodes_k()[eigenindex] #type:ignore
                kphis=numpy.linspace(0,2*numpy.pi*self.numperiods/k,n_segments+1,endpoint=True)+self.phase
                eigen_tensor_modes.append((prefixPair[0],prefixPair[1],prefixPair[2],
                                           numpy.cos(k*kphis),-numpy.sin(k*kphis)))
        tensor_fields=_extrude_tensor_fields(getattr(base,"tensor_fields",{}),Q,n_segments+1,stride,
                                             set(base.nodal_field_inds.keys()),
                                             new_nodal_field_inds,field_operators,eigen_tensor_modes)

        _compact_field_indices(new_nodal_field_inds)
        for name,index in sorted(new_nodal_field_inds.items(),key=lambda item: item[1]): #type:ignore
            if name in field_operators.keys():
                op=field_operators[name] #type:ignore
                #print("Applying operator for "+name,op)
                if op is not None:
                    for arg in op[1:]: #type:ignore
                        if arg not in base.nodal_field_inds:
                            raise RuntimeError("Cannot resolve argument "+arg+" for tranformation of "+name+"\n"+str(op)+"\nAvailable: "+str(base.nodal_field_inds)) #type:ignore
                    args=[base.nodal_values[:,base.nodal_field_inds[n]] for n in op[1:]] #type:ignore
                    newdata=op[0](*args) #type:ignore
                else:
                    newdata=None
            else:
                newdata=numpy.tile(base.nodal_values[:,base.nodal_field_inds[name]], n_segments+1) #type:ignore
            if new_nodal_values is not None:
                new_nodal_values.append(newdata) #type:ignore

        base.nodal_field_inds=new_nodal_field_inds
        # column_stack, not transpose(array(...)): the latter leaves a Fortran-ordered view, so
        # every later nodal_values[:,i] became a strided gather
        base.nodal_values=numpy.column_stack(new_nodal_values) #type:ignore
        base.vector_fields=vector_fields
        base.tensor_fields=tensor_fields


        #if base.tesselate_tri:
        #    raise RuntimeError("Cartesian extrusion cannot be combined with tesselate_tri=True yet")
        if base.discontinuous and (base.D0_data.shape[1]>0 or base.DL_data.shape[1]>0):
            raise RuntimeError("Cartesian extrusion does not work with discontinuous=True, at least if D0 or DL fields are defined")
        upper_limit=n_segments

        def phi_row_for_step(step:int)->NPFloatArray:
            # The axial position of the centre of each ring of created elements. This used to run
            # over [0,2*pi] while the nodal fields are expanded over [0,L] with L=2*pi*numperiods/k,
            # so cos(k*...) gave the D0/DL eigen fields k periods where the nodal ones had
            # numperiods - they only agreed when k happened to equal numperiods.
            M=upper_limit//step
            return numpy.linspace(0,L,M,endpoint=False)+L/(2*M)+self.phase

        base.elem_types,base.elem_indices,elemental_phis,counts=_extrude_element_connectivity(
            base.elem_types,base.elem_indices,stride=stride,upper_limit=upper_limit,
            phi_increm=phi_increm,phi_row_for_step=phi_row_for_step,
            # Unlike the rotational extrusion this one does not close onto itself, so no wrapping
            modulus=None,allow_line_c1=False)

        # Each output element inherits the elemental data of the element it was extruded from
        if base.DL_data.shape[1]>0:
            base.DL_data=numpy.repeat(base.DL_data,counts,axis=0)
        if base.D0_data.shape[1]>0:
            base.D0_data=numpy.repeat(base.D0_data,counts,axis=0)

        # Rotate DL and D0 with m if necessary:        
        if self.apply_k_mode_expansion and base.mesh.get_problem().get_last_eigenmodes_k() is not None: #type:ignore
            remove_indices=[]
            remove_indices_DL=[]
            remove_indices_D0=[]
            rename_indices={}
            for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                prefixRe=prefixPair[0]
                prefixIm = prefixPair[1]
                prefixRes = prefixPair[2]
                k=base.mesh.get_problem().get_last_eigenmodes_k()[eigenindex] #type:ignore
                cs=numpy.cos(k*elemental_phis)
                sn=numpy.sin(k*elemental_phis)
                for dgfieldname,dgfieldind in base.elemental_field_inds.items():
                    if dgfieldname.startswith(prefixRe):
                        imfieldindex=base.elemental_field_inds[prefixIm+dgfieldname[len(prefixRe):]]
                        
                        #raise RuntimeError("Strange, but for some reason the D0/DL without discontinuous=True is broken")
                        if dgfieldind<base.DL_data.shape[1]:
                            # DL field                                                                       
                            base.DL_data[:,dgfieldind,0]=base.DL_data[:,dgfieldind,0]*cs+base.DL_data[:,imfieldindex,0]*sn                            
                            remove_indices_DL.append(imfieldindex)
                            #base.DL_data[:,dgfieldind,0]=cs
                            #base.DL_data[:,dgfieldind,0]=elemental_phis
                        else:
                            base.D0_data[:,dgfieldind-base.DL_data.shape[1]]=base.D0_data[:,dgfieldind-base.DL_data.shape[1]]*cs+base.D0_data[:,imfieldindex-base.DL_data.shape[1]]*sn
                            remove_indices_D0.append(imfieldindex-base.DL_data.shape[1])
                            #base.D0_data[:,dgfieldind-base.DL_data.shape[1]]=cs
                            #base.D0_data[:,dgfieldind-base.DL_data.shape[1]]=elemental_phis
                        # Rename to the result
                        rename_indices[prefixRes+dgfieldname[len(prefixRe):]]=dgfieldname
                        remove_indices.append(imfieldindex)
            for new_name,old_name in rename_indices.items():
                base.elemental_field_inds[new_name]=base.elemental_field_inds.pop(old_name)
            if len(remove_indices_D0)>0:
                base.D0_data=numpy.delete(base.D0_data,numpy.array(remove_indices_D0),axis=1)
            if len(remove_indices_DL)>0:
                base.DL_data=numpy.delete(base.DL_data,numpy.array(remove_indices_DL),axis=1)
            if len(remove_indices)>0:
                new_inds={}
                rev_inds={i:n for n,i in base.elemental_field_inds.items()}
                cnt=0
                # max()+1: without it the highest-numbered elemental field was silently dropped
                for i in range(max(rev_inds.keys())+1):
                    if i in remove_indices:
                        continue
                    new_inds[rev_inds[i]]=cnt
                    cnt+=1                
                base.elemental_field_inds=new_inds
                
        
        

        #print(field_operators)
        #exit()
        
        
        

class MeshDataRotationalExtrusion(MeshDataCacheOperatorBase):
    """
    Can be added as ``operator`` to :py:class:`~pyoomph.output.meshio.MeshFileOutput` to extrude the mesh along the azimuthal phi-coordinate. Most useful combined with :py:class:`MeshDataCombineWithEigenfunction` and azimuthal normal mode stability analysis.

    Args:
        n_segments: Number of segments to extrude the mesh to.
        angle: Angle to extrude the mesh to. If larger than 2*pi, it will be cut off at 2*pi.
        start_angle: Angle to start the extrusion at.
        rotate_eigendata_with_mode_m: If True, the eigendata will be rotated with the azimuthal mode number m. This is useful for azimuthal normal mode stability analysis.
        collapse_axis_points: If True, a point element sitting on the symmetry axis stays a single
            vertex instead of being swept into a ring of zero-length line elements. Set it to False
            to reproduce the output of versions before this was fixed.
    """
    def __init__(self,n_segments:int=32,angle:float=2*numpy.pi,start_angle:float=0.0,rotate_eigendata_with_mode_m:bool=True,collapse_axis_points:bool=True):
        super(MeshDataRotationalExtrusion, self).__init__()
        self.collapse_axis_points=collapse_axis_points
        self.n_segments=n_segments
        self.angle=float(angle)
        if self.angle>2*numpy.pi:
            self.angle=2*numpy.pi
        self.start_angle=float(start_angle)
        self.rotate_eigendata_with_mode_m=rotate_eigendata_with_mode_m

    def apply(self,base:MeshDataCacheEntry):
        self._materialise_local_expressions(base)
        n_segments=self.n_segments
        phi_increm=1
        if base.mesh._eqtree.get_code_gen()._coordinate_space not in {"C1","C1TB"}: 
            n_segments*=2
            phi_increm=2

        closed=(self.angle>=2*numpy.pi-1e-10)
        phis=numpy.linspace(0,self.angle,n_segments,endpoint=not closed)+self.start_angle #type:ignore

        r_pos=base.nodal_values[:, base.nodal_field_inds["coordinate_x"]]
        min_radial_index=int(numpy.argmin(r_pos))
        if r_pos[min_radial_index]<-1.0e-9:
            raise RuntimeError("Cannot rotationally extrude meshes with negative x-coordinates (radius)")
        # Nodes on the symmetry axis are shared by all segments instead of being copied per segment
        axis_nodes=r_pos<=1.0e-9


        stride = base.nodal_values.shape[0]

        new_nodal_values:list[NPFloatArray]=[]
        new_nodal_field_inds=base.nodal_field_inds.copy()
        # name -> either a plain source field name or [operator, *its argument field names]
        field_operators:dict[str,Any]={}

        field_operators["coordinate_x"] = [lambda cx: numpy.outer(numpy.cos(phis), cx).flatten(), "coordinate_x"] #type:ignore
        field_operators["coordinate_y"] = [lambda cx: numpy.outer(numpy.sin(phis), cx).flatten(), "coordinate_x"] #type:ignore
        field_operators["lagrangian_x"] = [lambda cx: numpy.outer(numpy.cos(phis), cx).flatten(), "lagrangian_x"] #type:ignore
        field_operators["lagrangian_y"] = [lambda cx: numpy.outer(numpy.sin(phis), cx).flatten(), "lagrangian_x"] #type:ignore
        field_operators["normal_x"] = [lambda nx: numpy.outer(numpy.cos(phis), nx).flatten(), "normal_x"] #type:ignore
        field_operators["normal_y"] = [lambda nx: numpy.outer(numpy.sin(phis), nx).flatten(), "normal_x"] #type:ignore

        vector_fields=base.vector_fields.copy()
#        vector_fields["coordinate"]=["coordinate_x","coordinate_y"]
        rev_vector_fields={}
        for a, b in vector_fields.items():
            for c in b:
                rev_vector_fields[c] = a

        completed_eigen_vector_fields=set() #type:ignore
        ms_by_prefix:dict[str,float]={} # so the mesh-position block below does not need a leaked loop variable
        if self.rotate_eigendata_with_mode_m and base.mesh.get_problem().get_last_eigenmodes_m() is not None: #type:ignore
            for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                ms_by_prefix[prefixPair[0]]=base.mesh.get_problem().get_last_eigenmodes_m()[eigenindex] #type:ignore
                prefixRe=prefixPair[0]
                prefixIm = prefixPair[1]
                prefixRes = prefixPair[2]
                for fn,findex in base.nodal_field_inds.items(): #type:ignore
                    if fn.startswith(prefixRe):
                        fnRe=fn
                        fnIm=prefixIm+fn[len(prefixRe):]
                        fnRes=prefixRes+fn[len(prefixRe):]
                        del new_nodal_field_inds[fnRe]
                        del new_nodal_field_inds[fnIm]
                        new_nodal_field_inds[fnRes]=max(new_nodal_field_inds.values()) + 1
                        m=base.mesh.get_problem().get_last_eigenmodes_m()[eigenindex] #type:ignore
                        # k/phis (m/phis) are bound as defaults: these closures are only called much later, after
                        # every loop has finished, so with more than one entry in _additional_eigendata
                        # they all used to be evaluated with the LAST eigenindex's wavenumber.
                        # The stored pair is (Re,Im) of the complex amplitude of exp(I*m*phi), so the physical field
                        # is Re[u*exp(I*m*phi)] = cos*Re - SIN*Im. Verified against the coordinate system itself rather
                        # than against another reconstruction: d/dphi of a perturbation is a factor I*m, so projecting
                        # w = r*grad(u)_phi must give Re_w = -m*Im_u and Im_w = +m*Re_u, which it does to 3e-9. This used
                        # to read +sin, i.e. it rendered the conjugate mode - a mirror image, which still looks like a
                        # plausible mode and is why it went unnoticed.
                        field_operators[fnRes] = [lambda RealPart,ImagPart,m=m,phis=phis : numpy.outer(numpy.cos(m*phis), RealPart).flatten() - numpy.outer(numpy.sin(m*phis), ImagPart).flatten(), fnRe,fnIm] #type:ignore

                        if fnRe in rev_vector_fields:
                            ReVector=rev_vector_fields[fnRe] #type:ignore
                            ImVector=rev_vector_fields[fnIm] #type:ignore
                            ResVector=prefixRes+rev_vector_fields[fnRe][len(prefixRe):] #type:ignore
                            vector_fields[ResVector]=[prefixRes+compofn[len(prefixRe):] for compofn in vector_fields[ReVector]] #type:ignore
                            del vector_fields[ReVector] #type:ignore
                            del vector_fields[ImVector] #type:ignore
                            rev_vector_fields = {}
                            for a, b in vector_fields.items():
                                for c in b:
                                    rev_vector_fields[c] = a
                            #print(vector_fields[ResVector])
                            #raise RuntimeError("HEREH")
                            #field_operators[fnRes] = [lambda RealPart, ImagPart: numpy.outer(numpy.cos(m * phis),RealPart).flatten() + numpy.outer(numpy.sin(m * phis), ImagPart).flatten(), fnRe, fnIm]
                # The wavenumber of THIS prefix, taken from the table rather than from the scalar
                # loop above: m was only assigned there if some field matched prefixRe, so reading
                # the leaked loop variable here would be undefined for a prefix with vectors only.
                m=ms_by_prefix[prefixRe]
                # Second iteration to patch the vectors.
                #
                # These are FUSED operators, reading the real and imaginary halves and doing both
                # the mode reconstruction and the frame rotation at once: the dispatcher can only
                # read base fields, so an operator cannot chain onto the reconstruction the scalar
                # loop above produced. A swirl-free vector needs this as much as a swirling one -
                # its components are still radial and axial rather than Cartesian. "normal" and
                # "coordinate", which the mesh writes itself rather than an equation declaring them,
                # are exactly that, and letting them fall through to the plain vector loop below
                # asked it for a source field named after the RESULT.
                for vecname,veccompos in vector_fields.items():
                    if vecname.startswith(prefixRes):
                        composRes = [fn for fn in veccompos]
                        composIm = [prefixIm + fn[len(prefixRes):] for fn in veccompos]
                        composRe = [prefixRe + fn[len(prefixRes):] for fn in veccompos]
                        r_index=phi_index=axial_index=None
                        for cindex,componame in enumerate(composRes):
                            if componame.endswith("_x"):
                                r_index=cindex
                            elif componame.endswith("_phi"):
                                phi_index=cindex
                            elif componame.endswith("_y"):
                                axial_index=cindex

                        if r_index is not None:
                            # V = Re[(Vr*r_hat + Vphi*phi_hat)*exp(I*m*phi)] with the RIGHT-handed
                            # phi_hat = (-sin phi, cos phi, 0) -- the same one the base vector loop
                            # further down uses. Both signs were wrong here and they composed into a
                            # clean mirror image of the true mode, so the base swirl and the
                            # perturbation swirl were drawn with opposite handedness in one picture.
                            def _mode(Re,Im,m=m,phis=phis): #type:ignore
                                return numpy.cos(m*phis)[:,None]*Re[None,:]-numpy.sin(m*phis)[:,None]*Im[None,:] #type:ignore
                            if phi_index is not None:
                                def get_x_component(ReR,ImR,ReP,ImP,phis=phis): #type:ignore
                                    return (numpy.cos(phis)[:,None]*_mode(ReR,ImR)-numpy.sin(phis)[:,None]*_mode(ReP,ImP)).flatten() #type:ignore
                                def get_y_component(ReR,ImR,ReP,ImP,phis=phis): #type:ignore
                                    return (numpy.sin(phis)[:,None]*_mode(ReR,ImR)+numpy.cos(phis)[:,None]*_mode(ReP,ImP)).flatten() #type:ignore
                                mode_args=[composRe[r_index],composIm[r_index],composRe[phi_index],composIm[phi_index]]
                            else:
                                def get_x_component(ReR,ImR,phis=phis): #type:ignore
                                    return (numpy.cos(phis)[:,None]*_mode(ReR,ImR)).flatten() #type:ignore
                                def get_y_component(ReR,ImR,phis=phis): #type:ignore
                                    return (numpy.sin(phis)[:,None]*_mode(ReR,ImR)).flatten() #type:ignore
                                mode_args=[composRe[r_index],composIm[r_index]]
                            # the source-named components go away; the Cartesian ones take over
                            for stale in composRes:
                                field_operators.pop(stale,None)
                                new_nodal_field_inds.pop(stale,None)
                            for suffix,op in (("_x",get_x_component),("_y",get_y_component)):
                                field_operators[vecname+suffix]=[op]+mode_args
                                if vecname+suffix not in new_nodal_field_inds:
                                    new_nodal_field_inds[vecname+suffix]=max(new_nodal_field_inds.values())+1
                            if axial_index is not None:
                                # the axial component only picks up the mode factor, not the rotation
                                field_operators[vecname+"_z"]=[lambda ReVy,ImVy,m=m,phis=phis: numpy.outer(numpy.cos(m*phis),ReVy).flatten()-numpy.outer(numpy.sin(m*phis),ImVy).flatten(),
                                                               composRe[axial_index],composIm[axial_index]] #type:ignore
                                if vecname+"_z" not in new_nodal_field_inds:
                                    new_nodal_field_inds[vecname+"_z"]=max(new_nodal_field_inds.values())+1
                            completed_eigen_vector_fields.add(vecname) #type:ignore
                            vector_fields[vecname]=_vector_components_present(vecname,new_nodal_field_inds)

                
            # Also assemble the eigenperturbation of the position. The field names are hardcoded, so
            # this only ever fires for the default eigen_prefix_real with a single eigenindex; m is
            # looked up rather than read off the loop above, which had left it dangling.
            if "EigenRe_coordinate_x" in base.nodal_field_inds and "EigenRe_" in ms_by_prefix:
                m=ms_by_prefix["EigenRe_"]

                # The mesh perturbation has no azimuthal component of its own (the commented-out
                # Vphi terms), so only the mode factor applies -- with the same minus as everywhere
                # else, since these are the (Re,Im) of an exp(I*m*phi) amplitude too.
                def _mode_c(Re,Im,m=m,phis=phis): #type:ignore
                    return numpy.cos(m*phis)[:,None]*Re[None,:]-numpy.sin(m*phis)[:,None]*Im[None,:] #type:ignore
                def get_x_component(ReR,ImR,phis=phis): #type:ignore
                    return (numpy.cos(phis)[:,None]*_mode_c(ReR,ImR)).flatten() #type:ignore
                def get_y_component(ReR,ImR,phis=phis): #type:ignore
                    return (numpy.sin(phis)[:,None]*_mode_c(ReR,ImR)).flatten() #type:ignore
                field_operators["Eigen_coordinate_x"]= [get_x_component,"EigenRe_coordinate_x","EigenIm_coordinate_x"] #type:ignore
                field_operators["Eigen_coordinate_y"]= [get_y_component,"EigenRe_coordinate_x","EigenIm_coordinate_x"] #type:ignore
                if "EigenRe_coordinate_y" in base.nodal_field_inds:
                    field_operators["Eigen_coordinate_z"]= [lambda ReVy,ImVy,m=m,phis=phis: numpy.outer(numpy.cos(m * phis), ReVy).flatten()-numpy.outer(numpy.sin(m * phis), ImVy).flatten(),"EigenRe_coordinate_y","EigenIm_coordinate_y"] #type:ignore
                    new_nodal_field_inds["Eigen_coordinate_z"] = max(new_nodal_field_inds.values()) + 1
                    vector_fields["Eigen_coordinate"]=["Eigen_coordinate"+component for component in ["_x","_y","_z"]]
                else:                
                    new_nodal_field_inds["Eigen_coordinate_y"] = max(new_nodal_field_inds.values()) + 1
                    vector_fields["Eigen_coordinate"]=["Eigen_coordinate"+component for component in ["_x","_y"]]
                completed_eigen_vector_fields.add("Eigen_coordinate")

        for vfield,components in vector_fields.items(): #type:ignore
            if vfield in completed_eigen_vector_fields:
                continue
            if vfield+"_x" in new_nodal_field_inds:
                if vfield+"_y" in new_nodal_field_inds:
                    new_nodal_field_inds[vfield+"_z"] = max(new_nodal_field_inds.values()) + 1
                    field_operators[vfield+"_z"]= [lambda vy: numpy.tile(vy,n_segments), vfield+"_y"] #type:ignore
                else:
                    new_nodal_field_inds[vfield + "_y"] = max(new_nodal_field_inds.values()) + 1
                if vfield+"_phi" in new_nodal_field_inds:
                    field_operators[vfield + "_x"] = [lambda vx,vphi: numpy.outer(numpy.cos(phis), vx).flatten()-numpy.outer(numpy.sin(phis), vphi).flatten(),vfield + "_x",vfield + "_phi"] #type:ignore
                    field_operators[vfield + "_y"] = [lambda vx,vphi: numpy.outer(numpy.sin(phis), vx).flatten()+numpy.outer(numpy.cos(phis), vphi).flatten(),vfield + "_x",vfield+"_phi"] #type:ignore

                    if vfield+"_phi" in new_nodal_field_inds:
                        del new_nodal_field_inds[vfield+"_phi"]

                else:
                    field_operators[vfield + "_x"] = [lambda vx: numpy.outer(numpy.cos(phis), vx).flatten(),vfield + "_x"] #type:ignore
                    field_operators[vfield + "_y"] = [lambda vx: numpy.outer(numpy.sin(phis), vx).flatten(),vfield + "_x"] #type:ignore
                vector_fields[vfield]=_vector_components_present(vfield,new_nodal_field_inds)

        # Tensors, one index at a time with the same rotation. A tensor's slots are positional,
        # unlike a vector, whose azimuthal component is the separately named "_phi" - so Q's columns
        # are the images of the slots in their own order, and that order depends on the mesh. On a
        # bulk mesh the slots are (r, z, phi); on a RADIAL one there is no axial direction at all and
        # they are (r, phi), which is the layout define_tensor_field hands out ("_aa" at [1][1], not
        # [2][2]) and the one directional_tensor_derivative assumes with azi = 2 if ndim == 2 else 1.
        # Using the bulk Q on a radial mesh sent the azimuthal slot to z, i.e. e_phi (x) e_phi came
        # out as e_z (x) e_z, constant instead of turning with phi.
        # Like the vector path this assumes the mesh x is the radius: neither consults
        # use_x_as_symmetry_axis, which would swap the first two slots of the bulk layout. That flag
        # is now refused outright by the azimuthal normal mode system, so only a plain axisymmetric
        # output can still carry it - and such a mesh is already extruded about the wrong axis here,
        # vectors included, since r_pos above reads coordinate_x.
        cs,sn=numpy.cos(phis),numpy.sin(phis)
        zeros_row=numpy.zeros_like(cs)
        ones_row=numpy.ones_like(cs)
        if "coordinate_y" in base.nodal_field_inds:
            Q=numpy.stack([numpy.stack([ cs,zeros_row,-sn],axis=1),
                           numpy.stack([ sn,zeros_row, cs],axis=1),
                           numpy.stack([zeros_row,ones_row,zeros_row],axis=1)],axis=1)
        else:
            # (r, phi) and an unused third slot: a plain rotation about the axis
            Q=numpy.stack([numpy.stack([ cs,-sn,zeros_row],axis=1),
                           numpy.stack([ sn, cs,zeros_row],axis=1),
                           numpy.stack([zeros_row,zeros_row,ones_row],axis=1)],axis=1)
        eigen_tensor_modes=[]
        if self.rotate_eigendata_with_mode_m and base.mesh.get_problem().get_last_eigenmodes_m() is not None: #type:ignore
            for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                m=base.mesh.get_problem().get_last_eigenmodes_m()[eigenindex] #type:ignore
                eigen_tensor_modes.append((prefixPair[0],prefixPair[1],prefixPair[2],
                                           numpy.cos(m*phis),-numpy.sin(m*phis)))
        tensor_fields=_extrude_tensor_fields(getattr(base,"tensor_fields",{}),Q,len(phis),stride,
                                             set(base.nodal_field_inds.keys()),
                                             new_nodal_field_inds,field_operators,eigen_tensor_modes)

        if "coordinate_y" in base.nodal_field_inds:
            new_nodal_field_inds["coordinate_z"]=max(new_nodal_field_inds.values())+1
            if "lagrangian_z" in base.nodal_field_inds:
                new_nodal_field_inds["lagrangian_z"] = max(new_nodal_field_inds.values()) + 1
            if "normal_x" in base.nodal_field_inds:
                new_nodal_field_inds["normal_z"] = max(new_nodal_field_inds.values()) + 1
            

            field_operators["coordinate_z"] = [lambda cy: numpy.tile(cy, n_segments), "coordinate_y"] #type:ignore
            if "lagrangian_z" in base.nodal_field_inds:
                field_operators["lagrangian_z"] = [lambda cy: numpy.tile(cy, n_segments), "lagrangian_y"] #type:ignore
            field_operators["normal_z"] = [lambda ny: numpy.tile(ny,n_segments), "normal_y"] #type:ignore
        else:
            new_nodal_field_inds["coordinate_y"] = max(new_nodal_field_inds.values()) + 1
            new_nodal_field_inds["lagrangian_y"] = max(new_nodal_field_inds.values()) + 1
            if "normal_x" in base.nodal_field_inds:
                new_nodal_field_inds["normal_y"] = max(new_nodal_field_inds.values()) + 1


        _compact_field_indices(new_nodal_field_inds)
        for name,index in sorted(new_nodal_field_inds.items(),key=lambda item: item[1]): #type:ignore
            if name in field_operators.keys():
                op=field_operators[name] #type:ignore
                if op is not None:
                    for arg in op[1:]: #type:ignore
                        if arg not in base.nodal_field_inds:
                            raise RuntimeError("Cannot resolve argument "+arg+" for tranformation of "+name+"\n"+str(op)+"\nAvailable: "+str(base.nodal_field_inds)) #type:ignore
                    args=[base.nodal_values[:,base.nodal_field_inds[n]] for n in op[1:]] #type:ignore
                    newdata=op[0](*args) #type:ignore
                else:
                    newdata=None
            else:
                newdata=numpy.tile(base.nodal_values[:,base.nodal_field_inds[name]], n_segments) #type:ignore
            if new_nodal_values is not None:
                new_nodal_values.append(newdata) #type:ignore

        base.nodal_field_inds=new_nodal_field_inds
        # column_stack, not transpose(array(...)): the latter leaves a Fortran-ordered view, so
        # every later nodal_values[:,i] became a strided gather
        base.nodal_values=numpy.column_stack(new_nodal_values) #type:ignore
        base.vector_fields=vector_fields
        base.tensor_fields=tensor_fields


        if base.tesselate_tri:
            raise RuntimeError("rotational extrusion cannot be combined with tesselate_tri=True yet")
        if base.discontinuous and (base.D0_data.shape[1]>0 or base.DL_data.shape[1]>0):
            raise RuntimeError("rotational extrusion does not work with discontinuous=True, at least if D0 or DL fields are defined")

        upper_limit=n_segments-(0 if closed else phi_increm)

        def phi_row_for_step(step:int)->NPFloatArray:
            row=numpy.linspace(0,self.angle,upper_limit//step,endpoint=not closed)+self.start_angle
            return row+row[-1]/(2*len(row))

        base.elem_types,base.elem_indices,elemental_phis,counts=_extrude_element_connectivity(
            base.elem_types,base.elem_indices,stride=stride,upper_limit=upper_limit,
            phi_increm=phi_increm,phi_row_for_step=phi_row_for_step,
            # The extrusion wraps around onto its own first layer when it closes the full circle
            modulus=base.nodal_values.shape[0],axis_nodes=axis_nodes,
            collapse_axis_points=self.collapse_axis_points)

        # Each output element inherits the elemental data of the element it was extruded from.
        # Guarded on the field count because with no fields at all the arrays keep the old row
        # count, and some callers still compare that against the pre-extrusion element count.
        if base.DL_data.shape[1]>0:
            base.DL_data=numpy.repeat(base.DL_data,counts,axis=0)
        if base.D0_data.shape[1]>0:
            base.D0_data=numpy.repeat(base.D0_data,counts,axis=0)

        # Rotate DL and D0 with m if necessary:        
        if self.rotate_eigendata_with_mode_m and base.mesh.get_problem().get_last_eigenmodes_m() is not None: #type:ignore
            remove_indices=[]
            remove_indices_DL=[]
            remove_indices_D0=[]
            rename_indices={}
            for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                prefixRe=prefixPair[0]
                prefixIm = prefixPair[1]
                prefixRes = prefixPair[2]
                m=base.mesh.get_problem().get_last_eigenmodes_m()[eigenindex] #type:ignore
                cs=numpy.cos(m*elemental_phis)
                sn=numpy.sin(m*elemental_phis)
                for dgfieldname,dgfieldind in base.elemental_field_inds.items():
                    if dgfieldname.startswith(prefixRe):
                        imfieldindex=base.elemental_field_inds[prefixIm+dgfieldname[len(prefixRe):]]
                        
                        #raise RuntimeError("Strange, but for some reason the D0/DL without discontinuous=True is broken")
                        if dgfieldind<base.DL_data.shape[1]:
                            # DL field                                                                       
                            base.DL_data[:,dgfieldind,0]=base.DL_data[:,dgfieldind,0]*cs+base.DL_data[:,imfieldindex,0]*sn                            
                            remove_indices_DL.append(imfieldindex)
                            #base.DL_data[:,dgfieldind,0]=cs
                            #base.DL_data[:,dgfieldind,0]=elemental_phis
                        else:
                            base.D0_data[:,dgfieldind-base.DL_data.shape[1]]=base.D0_data[:,dgfieldind-base.DL_data.shape[1]]*cs+base.D0_data[:,imfieldindex-base.DL_data.shape[1]]*sn
                            remove_indices_D0.append(imfieldindex-base.DL_data.shape[1])
                            #base.D0_data[:,dgfieldind-base.DL_data.shape[1]]=cs
                            #base.D0_data[:,dgfieldind-base.DL_data.shape[1]]=elemental_phis
                        # Rename to the result
                        rename_indices[prefixRes+dgfieldname[len(prefixRe):]]=dgfieldname
                        remove_indices.append(imfieldindex)
            for new_name,old_name in rename_indices.items():
                base.elemental_field_inds[new_name]=base.elemental_field_inds.pop(old_name)
            if len(remove_indices_D0)>0:
                base.D0_data=numpy.delete(base.D0_data,numpy.array(remove_indices_D0),axis=1)
            if len(remove_indices_DL)>0:
                base.DL_data=numpy.delete(base.DL_data,numpy.array(remove_indices_DL),axis=1)
            if len(remove_indices)>0:
                new_inds={}
                rev_inds={i:n for n,i in base.elemental_field_inds.items()}
                cnt=0
                # max()+1: without it the highest-numbered elemental field was silently dropped
                for i in range(max(rev_inds.keys())+1):
                    if i in remove_indices:
                        continue
                    new_inds[rev_inds[i]]=cnt
                    cnt+=1                
                base.elemental_field_inds=new_inds


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
