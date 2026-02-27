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
from ..typings import *
import numpy


from .mesh import AnySpatialMesh, InterfaceMesh, MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d, MeshFromTemplateBase 
from ..expressions import ExpressionOrNum,Expression, num
from ..expressions.units import unit_to_string

MeshDataEigenModes=Literal["abs","real","imag","merge","angle"]

class MeshDataCacheEntry:
    def __init__(self,msh:AnySpatialMesh,tesselate_tri:bool=True,nondimensional:bool=False,eigenvector:Optional[Union[int,Sequence[int]]]=None,eigenmode:MeshDataEigenModes="abs",history_index:int=0,with_halos:bool=False,operator:Optional["MeshDataCacheOperatorBase"]=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True):
        assert isinstance(msh,(MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d,InterfaceMesh))
        
        self.mesh=msh
        self.eigenvector = eigenvector
        self.eigenmode = eigenmode
        self.history_index=history_index
        self.with_halos=with_halos
        self.merged_eigendata:Dict[int,Dict[str,Any]]={}
        self.discontinuous=discontinuous
        self.add_eigen_to_mesh_positions=add_eigen_to_mesh_positions
        if self.eigenmode not in {"abs","real","imag","merge","angle"}:
            raise RuntimeError("Unknown eigenmode "+str(self.eigenmode))
        if isinstance(self.eigenvector,int):
            if eigenmode=="merge":
                self.eigenvector=[self.eigenvector]
            else:
                backup_dofs, backup_pinned = self.mesh.get_problem().set_eigenfunction_as_dofs(self.eigenvector, mode=self.eigenmode,additive_mesh_positions=self.add_eigen_to_mesh_positions)

        if isinstance(self.eigenvector,(list,set)):
            if eigenmode!="merge":
                raise RuntimeError("Multiple eigenvectors in MeshDataCache only works if eigenmode is set to 'merge'")

        if eigenmode=="merge" and eigenvector!=None:
            backup_dofs:Optional[NPFloatArray]=None
            backup_pinned:Optional[NPFloatArray]=None
            for ev in cast(List[int],self.eigenvector):
                backup = self.mesh.get_problem().set_eigenfunction_as_dofs(ev,mode="real",additive_mesh_positions=self.add_eigen_to_mesh_positions)
                if backup_dofs is None:
                    backup_dofs, backup_pinned=backup
                real_nodal_values, elem_indices, elem_types, nodal_field_inds, real_D0_data, real_DL_data, elemental_field_inds = msh.to_numpy(tesselate_tri, nondimensional, history_index,discontinuous) #type:ignore
                self.mesh.get_problem().set_eigenfunction_as_dofs(ev, mode="imag",additive_mesh_positions=self.add_eigen_to_mesh_positions)
                imag_nodal_values, elem_indices, elem_types, nodal_field_inds, imag_D0_data, imag_DL_data, elemental_field_inds = msh.to_numpy(tesselate_tri, nondimensional, history_index,discontinuous) #type:ignore
                self.merged_eigendata[ev]={"nodal_values":(real_nodal_values,imag_nodal_values),"DL_data":(real_DL_data,imag_DL_data),"D0_data":(real_D0_data,real_D0_data)}
            if backup_dofs is not None:
                assert backup_pinned is not None
                self.mesh.get_problem().set_all_values_at_current_time(backup_dofs, backup_pinned, not self.add_eigen_to_mesh_positions)
        assert isinstance(msh,(MeshFromTemplateBase,InterfaceMesh))
        
        self.nodal_values, self.elem_indices, self.elem_types, self.nodal_field_inds, self.D0_data, self.DL_data, self.elemental_field_inds = msh.to_numpy(tesselate_tri,nondimensional,history_index,discontinuous)
        
        if (not self.with_halos) and msh.is_mesh_distributed():
            newei:List[List[int]]=[]
            newet:List[int]=[]
            for i,(ei,et) in enumerate(zip(self.elem_indices,self.elem_types)):
                #print("Element",i,"has proc ID",i,ei,et)
                if msh.element_pt(i).non_halo_proc_ID()<0:
                    newei.append(ei)
                    newet.append(et)
            self.elem_indices:NPIntArray =numpy.array(newei,dtype=numpy.int32)  #type:ignore
            self.elem_types:NPIntArray=numpy.array(newet,dtype=numpy.int32) #type:ignore


        if isinstance(self.eigenvector, int) :
            self.mesh.get_problem().set_all_values_at_current_time(backup_dofs, backup_pinned, not self.add_eigen_to_mesh_positions) #type:ignore

        self.nondimensional=nondimensional
        self.interface_lines_segs:Optional[List[List[int]]]=None
        self.interface_lines_segs_ninter:Optional[int]=None

        self.nodal_local_exprs:Dict[str,NPFloatArray]={}
        self.local_expr_indices:Dict[str,int]={n:i for i,n in enumerate(self.mesh.list_local_expressions())}

        self.operator=operator
        self.tesselate_tri=tesselate_tri

        vector_fields = msh.get_eqtree().get_equations().get_list_of_vector_fields(self.mesh.get_eqtree().get_code_gen())
        self.vector_fields:Dict[str,List[str]] = {k: v for a in vector_fields for k, v in a.items()}

        self._additional_eigendata:Dict[int,Tuple[str,str,str]]={} # Index to pair of Re,Im
        if self.operator is not None:
            self.operator.apply(self)

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

    def get_default_output_fields(self,rem_underscore:bool=True,rem_lagrangian:bool=True) -> List[str]:
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
    def get_unit(self,field:List[str],as_string:Literal[False]=...,with_brackets:bool=...)->List[ExpressionOrNum]: ...

    @overload
    def get_unit(self,field:str,as_string:Literal[True]=...,with_brackets:bool=...)->str: ...

    @overload
    def get_unit(self,field:List[str],as_string:Literal[True]=...,with_brackets:bool=...)->List[str]: ...

    def get_unit(self,field:Union[str,List[str]],as_string:bool=False,with_brackets:bool=True)->Union[ExpressionOrNum,List[ExpressionOrNum],str,List[str]]:
        if isinstance(field,list):
            if as_string:
                return [self.get_unit(f,as_string=True,with_brackets=with_brackets) for f in field]
            else:
                return [self.get_unit(f,as_string=False,with_brackets=with_brackets) for f in field]
        if self.nondimensional or (field=="normal_x" or field=="normal_y" or field=="normal_z"):
            return "" if as_string else 1
        if (field in self.local_expr_indices.keys()) and not (field is self.nodal_field_inds.keys()):
            s=self.mesh.get_code_gen()._get_local_expression_unit_factor(field)
        else:
            s = self.mesh.get_code_gen().get_equations().get_scaling(field)
        if not isinstance(s,Expression): #type:ignore
            s=Expression(s)
        s = self.mesh.get_code_gen().expand_placeholders(s, False)
        _, unit, _, _ = _pyoomph.GiNaC_collect_units(s)
        res=1
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




    def get_data(self,name:Union[str,List[str],List[List[str]]],additional_eigenvector:Optional[int]=None,eigen_real_imag:Optional[int]=None)->Optional[NPFloatArray]:
        assert isinstance(self.mesh,(InterfaceMesh,MeshFromTemplate1d,MeshFromTemplate2d,MeshFromTemplate3d))
        if isinstance(name, list):
            if isinstance(name[0], list): #tensor data
                mdata:List[List[NPFloatArray]]=[]
                nonzero_length=-1
                for row in name:
                    rowdata:List[NPFloatArray]=[]
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
                    mdata.append(rowdata)
                if nonzero_length==-1:
                    raise RuntimeError("Tensor data "+str(name)+" does not contain anything")
                zer=numpy.zeros((nonzero_length,))
                for i,row in enumerate(mdata):
                    for j,entry in enumerate(row):
                        if entry is None:
                            mdata[i][j]=zer
                return numpy.array(mdata) #type:ignore
            else:
                mdata:List[NPFloatArray]=[]
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
                for i,entry in enumerate(mdata):
                    if entry is None:
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
            else:
                if isinstance(self.eigenvector, int):
                    base = self.mesh.evaluate_local_expression_at_nodes(self.local_expr_indices[name], self.nondimensional,self.discontinuous)
                    eps=1e-8
                    if self.eigenmode=="real" or self.eigenmode=="imag":
                        backup_dofs, backup_pinned,aampl = self.mesh.get_problem().set_eigenfunction_as_dofs(self.eigenvector,mode=self.eigenmode,perturb_amplitude=eps)
                        perturbed = self.mesh.evaluate_local_expression_at_nodes(self.local_expr_indices[name],self.nondimensional,self.discontinuous)
                        self.mesh.get_problem().set_all_values_at_current_time(backup_dofs, backup_pinned, not self.add_eigen_to_mesh_positions)
                        self.nodal_local_exprs[name] = (numpy.array(perturbed) - numpy.array(base)) * aampl / eps #type:ignore
                    else:
                        backup_dofs, backup_pinned, aampl_real = self.mesh.get_problem().set_eigenfunction_as_dofs(self.eigenvector, mode="real", perturb_amplitude=eps)
                        real_perturbed = self.mesh.evaluate_local_expression_at_nodes(self.local_expr_indices[name],self.nondimensional,self.discontinuous)
                        _, _, aampl_imag = self.mesh.get_problem().set_eigenfunction_as_dofs(self.eigenvector, mode="imag", perturb_amplitude=eps)
                        imag_perturbed = self.mesh.evaluate_local_expression_at_nodes(self.local_expr_indices[name],self.nondimensional,self.discontinuous)
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
                    self.nodal_local_exprs[name]=numpy.array(self.mesh.evaluate_local_expression_at_nodes(self.local_expr_indices[name],self.nondimensional,self.discontinuous)) #type:ignore
                data=self.nodal_local_exprs[name]
        #elif name in self.elemental_field_inds.keys():
        #    if self.discontinuous:
        #        raise RuntimeError("DG finding here")
        #    else:
        #        return None
        else:
            return None
        return numpy.array(data) #type:ignore


    def get_interface_line_segments(self) -> Tuple[List[List[int]], int]:
        if self.discontinuous:
            raise RuntimeError("get_interface_line_segments does not work for discontinuous caches")
        if self.interface_lines_segs is not None:
            assert self.interface_lines_segs_ninter is not None
            return self.interface_lines_segs,self.interface_lines_segs_ninter
        lines:List[List[int]] = []

        # Merge connected lines
        elms = [tuple([i for i in e]) for e in self.elem_indices]
        elms_at_points:Dict[int,List[int]] = {}
        inbetween_pts:Dict[Tuple[int,int],List[int]] = {}
        ninter=None
        for e in elms:
            elms_at_points.setdefault(e[0], []).append(e[-1])  #type:ignore
            elms_at_points.setdefault(e[-1], []).append(e[0]) #type:ignore
            inbetween_pts[(e[0], e[-1])] = e[1:-1] #type:ignore
            inbetween_pts[(e[-1], e[1])] = reversed(e[1:-1]) #type:ignore
            if ninter is None:
                ninter=len(e[1:-1])
            else:
                if ninter!=len(e[1:-1]):
                    raise RuntimeError("Strange intermediate points...")
        assert ninter is not None
        starnode_history:List[int]=[]
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

            currentcurve:List[int] = []
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
        self.interface_lines_segs=lines
        self.interface_lines_segs_ninter=ninter
        return lines,ninter
        

class MeshDataCache:
    def __init__(self,tesselate_tri:bool=True,nondimensional:bool=False,eigenvector:Optional[Union[int,Sequence[int]]]=None,eigenmode:MeshDataEigenModes="abs",history_index:int=0,with_halos:bool=False,operator:Optional["MeshDataCacheOperatorBase"]=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True):
        self._cache=dict()
        self.tesselate_tri=tesselate_tri
        self.nondimensional=nondimensional
        self.eigenvector=eigenvector
        self.eigenmode:MeshDataEigenModes=eigenmode
        self.history_index=history_index
        self.with_halos=with_halos
        self.operator=operator
        self.discontinuous=discontinuous
        self.add_eigen_to_mesh_positions=add_eigen_to_mesh_positions

    def clear(self):
        self._cache:Dict[AnySpatialMesh,MeshDataCacheEntry]=dict()

    def get_data(self,msh:AnySpatialMesh) -> MeshDataCacheEntry:
        
        if not (msh in self._cache.keys()):
            #print("CREATING MESH DATA",msh.get_full_name(),self.tesselate_tri,self.nondimensional)
            msh._setup_output_scales()
            
            self._cache[msh] = MeshDataCacheEntry(msh,tesselate_tri=self.tesselate_tri,nondimensional=self.nondimensional,eigenvector=self.eigenvector,eigenmode=self.eigenmode,history_index=self.history_index,with_halos=self.with_halos,operator=self.operator,discontinuous=self.discontinuous,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions)
        else:
            pass
            #print("REUSING MESH DATA",msh.get_full_name(),self.tesselate_tri,self.nondimensional)
        #print(self._cache[msh].get_data("theta"))
        return self._cache[msh]


class MeshDataCacheStorage:
    def __init__(self):
        self._storage:Dict[Tuple[Any,...],MeshDataCache]={}


    def clear(self,only_eigens:bool=False):
        remkeys:List[str]=[]
        for k,v in self._storage.items():
            if only_eigens:
                if k[2] is not None or (k[6] is not None and k[6].depends_on_eigen()):
                    v.clear()
                    remkeys.append(k) #type:ignore
            else:
                v.clear()
        if only_eigens:
            for k in remkeys:
                self._storage.pop(k, None) #type:ignore
        else:
            self._storage={}
        #print("STORAGE AFTER CLEAR",self._storage)

    def get_data(self,msh:AnySpatialMesh,nondimensional:bool,tesselate_tri:bool,eigenvector:Optional[Union[int,Sequence[int]]]=None,eigenmode:MeshDataEigenModes="abs",history_index:int=0,with_halos:bool=False,operator:Optional["MeshDataCacheOperatorBase"]=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True) -> MeshDataCacheEntry:
        if isinstance(eigenvector,list):
            eigenvector=tuple(set(eigenvector))
        key:Tuple[Any, ...] = (nondimensional, tesselate_tri,eigenvector,eigenmode,history_index,with_halos,operator,discontinuous,add_eigen_to_mesh_positions)
        if not key in self._storage.keys():                        
            #print("CREATING",key)
            msh._setup_output_scales()
            
            self._storage[key]=MeshDataCache(tesselate_tri=tesselate_tri, nondimensional=nondimensional,eigenvector=eigenvector,eigenmode=eigenmode,history_index=history_index,with_halos=with_halos,operator=operator,discontinuous=discontinuous,add_eigen_to_mesh_positions=add_eigen_to_mesh_positions)
            
        else:
            pass
            #print("REUSING",key)
        
        return self._storage[key].get_data(msh)




class MeshDataCacheOperatorBase:
    def __init__(self):
        super(MeshDataCacheOperatorBase, self).__init__()

    def apply(self,base:MeshDataCacheEntry)->None:
        raise RuntimeError("Specify")
    
    def depends_on_eigen(self)->bool:
        return False

    def __add__(self, other:"MeshDataCacheOperatorBase")->"MeshDataCacheCombinedOperator":
        return MeshDataCacheCombinedOperator(self,other)

    def _get_elem_dim(self,base:MeshDataCacheEntry) -> Literal[0, 1, 2, 3]:
        result=None
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
    def __init__(self,eigenindex:Union[int,Sequence[int]],eigen_prefix_real:str="EigenRe_",eigen_prefix_imag:str="EigenIm_",eigen_prefix_merged:str="Eigen_",add_eigen_to_mesh_positions=False):
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
                new_nodal_values=base.nodal_values.copy()
                new_nodal_field_inds=base.nodal_field_inds.copy()
                if len(self.eigenindex)>1:
                    prefix+=str(eigenindex)+"_"
                for fn,index in eigendata.nodal_field_inds.items():
                    if fn in hidden_fields:
                        #print("SKIPPING ",fn,hidden_fields)
                        continue
                    new_nodal_field_inds[prefix+fn] = max(new_nodal_field_inds.values()) + 1
                    new_nodal_values=numpy.c_[new_nodal_values,eigendata.nodal_values[:,index]]

              
                new_elem_field_inds={}
                rev_field_inds={i:n for n,i in base.elemental_field_inds.items()}
                rev_field_inds_eig={i:n for n,i in eigendata.elemental_field_inds.items()}
                cnt=0

                num_DL=base.DL_data.shape[1]        
                if not base.discontinuous:
                    new_DL_data=numpy.zeros((base.DL_data.shape[0],0,base.DL_data.shape[2]))                
                    for iDL in range(num_DL):
                        new_elem_field_inds[rev_field_inds[iDL]]=cnt
                        new_DL_data=numpy.concatenate((new_DL_data[:,:,:],base.DL_data[:,iDL:iDL+1,:]),axis=1)
                        cnt+=1
                    for iDL in range(eigendata.DL_data.shape[1]):
                        if rev_field_inds_eig[iDL] in hidden_fields:
                            continue
                        new_elem_field_inds[prefix+rev_field_inds[iDL]]=cnt
                        new_DL_data=numpy.concatenate((new_DL_data[:,:,:],eigendata.DL_data[:,iDL:iDL+1,:]),axis=1)
                        cnt+=1
                else:
                    new_DL_data=numpy.zeros((base.DL_data.shape[0],0))  
                    for iDL in range(num_DL):
                        new_elem_field_inds[rev_field_inds[iDL]]=cnt
                        new_DL_data=numpy.concatenate((new_DL_data[:,:],base.DL_data[:,iDL:iDL+1]),axis=1)
                        cnt+=1
                    for iDL in range(eigendata.DL_data.shape[1]):
                        if rev_field_inds_eig[iDL] in hidden_fields:
                            continue
                        new_elem_field_inds[prefix+rev_field_inds[iDL]]=cnt
                        new_DL_data=numpy.concatenate((new_DL_data[:,:],eigendata.DL_data[:,iDL:iDL+1]),axis=1)
                        cnt+=1              

                new_D0_data=numpy.zeros((base.D0_data.shape[0],0))
                for iD0 in range(base.D0_data.shape[1]):
                    new_elem_field_inds[rev_field_inds[iD0+base.DL_data.shape[1]]]=cnt
                    new_D0_data=numpy.concatenate((new_D0_data[:,:],base.D0_data[:,iD0:iD0+1]),axis=1)
                    cnt+=1                    
                for iD0 in range(eigendata.D0_data.shape[1]):
                    if rev_field_inds_eig[iD0+eigendata.DL_data.shape[1]] in hidden_fields:
                        continue
                    new_elem_field_inds[prefix+rev_field_inds_eig[iD0+eigendata.DL_data.shape[1]]]=cnt
                    new_D0_data=numpy.concatenate((new_D0_data[:,:],eigendata.D0_data[:,iD0:iD0+1]),axis=1)
                    cnt+=1
                
                for vector_name,compo_names in eigendata.vector_fields.items():
                    if len(self.eigenindex) == 1:
                        base.vector_fields[prefix+vector_name]=[prefix+compo_name for compo_name in compo_names]
                    else:
                        base.vector_fields[prefix+vector_name]=[prefix+compo_name for compo_name in compo_names]
                        #base.vector_fields[prefix +str(eigenindex)+"_"+ vector_name] = [prefix +str(eigenindex)+"_"+ compo_name for compo_name in compo_names]
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
        phase: Phase of the extrusion to start with.
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
        n_segments=self.n_segments
        phi_increm=1
        if base.mesh._eqtree.get_code_gen()._coordinate_space not in {"C1","C1TB"}: 
            n_segments*=2        
            phi_increm=2
        
            

        # Getting the length
        L=self.default_length
        if base.mesh._problem.get_last_eigenmodes_k() is not None:
            if self.use_k_for_length:                
                kcommon=None
                for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                    if eigenindex<len(base.mesh._problem.get_last_eigenmodes_k()):
                        k=base.mesh._problem.get_last_eigenmodes_k()[eigenindex] #type:ignore
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

        new_nodal_values=[]
        new_nodal_field_inds=base.nodal_field_inds.copy()
        field_operators={}

        new_D0_data=[]
        new_DL_data=[]
        
        
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
        if self.apply_k_mode_expansion and base.mesh._problem.get_last_eigenmodes_k() is not None: #type:ignore
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
                        newindex = 0
                        for name, index in sorted(new_nodal_field_inds.items(), key=lambda item: item[1]): #type:ignore
                            new_nodal_field_inds[name] = newindex
                            newindex += 1
                        new_nodal_field_inds[fnRes]=max(new_nodal_field_inds.values()) + 1
                        k=base.mesh._problem.get_last_eigenmodes_k()[eigenindex] #type:ignore
                        phis=numpy.linspace(0,2*numpy.pi*self.numperiods/k,n_segments+1,endpoint=True)
                        
                        
                        field_operators[fnRes] = [lambda RealPart,ImagPart : numpy.outer(numpy.cos(k*phis), RealPart).flatten() + numpy.outer(numpy.sin(k*phis), ImagPart).flatten(), fnRe,fnIm] #type:ignore

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

                        k=base.mesh._problem.get_last_eigenmodes_k()[eigenindex] #type:ignore
                        if r_index is not None and phi_index is not None:
                            #print("RINDEX",r_index,phi_index)
                            #print("K",eigenindex,k)
                            phis=numpy.linspace(0,2*numpy.pi*self.numperiods/k,n_segments+1,endpoint=True)
                            def get_x_component(ReR,ImR,ReP,ImP): #type:ignore
                                #print("XCOMPONENT",vecname, len(ReR),len(ImR),len(ReP),len(ImP))
                                
                                return numpy.outer(numpy.cos(k * phis),ReR).flatten()-numpy.outer(numpy.sin(k * phis),ImR).flatten()
                                #Vr_cos_phi=numpy.outer(numpy.cos(k * phis)*numpy.cos(phis),ReR).flatten()+numpy.outer(numpy.sin(k * phis)*numpy.cos(phis),ImR).flatten() #type:ignore
                                #Vphi_sin_phi=numpy.outer(numpy.cos(k * phis)*numpy.sin(phis),ReP).flatten()+numpy.outer(numpy.sin(k * phis)*numpy.sin(phis),ImP).flatten() #type:ignore
                                #return Vr_cos_phi+0*Vphi_sin_phi #type:ignore
                            def get_y_component(ReR,ImR,ReP,ImP): #type:ignore
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
                        field_operators[vecname+"_y"]=[lambda a:numpy.tile(a,n_segments+1),vecname+"_normal"] 
                        #print("SKIPPING VECTOR",vecname)
                        pass


        if True:
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
                            newindex=0
                            for name, index in sorted(new_nodal_field_inds.items(), key=lambda item: item[1]): #type:ignore
                                new_nodal_field_inds[name]=newindex
                                newindex+=1


        
                

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
        base.nodal_values=numpy.transpose(numpy.array(new_nodal_values)) #type:ignore        
        base.vector_fields=vector_fields
        


        #if base.tesselate_tri:
        #    raise RuntimeError("Cartesian extrusion cannot be combined with tesselate_tri=True yet")
        if base.discontinuous and (base.D0_data.shape[1]>0 or base.DL_data.shape[1]>0):
            raise RuntimeError("Cartesian extrusion does not work with discontinuous=True, at least if D0 or DL fields are defined")
        elem_types=base.elem_types


        upper_limit=n_segments#-phi_increm

        mod_length=base.nodal_values.shape[0]
        new_elem_types=[]
        new_elem_indices=[]
        elemental_phis=numpy.zeros((0,))
        
        mp = lambda i, o: (i + o * stride) #% mod_length #type:ignore
        for d0dl_index,(elemtype,eis) in enumerate(zip(elem_types,base.elem_indices)):            
            old_num_elems=len(new_elem_types)       
            if elemtype==1: # LineC1                         
                raise RuntimeError("Cartesian extrusion does not work with LineC1 elements")
            elif elemtype == 2:  # LineC2                
                    for offs in range(0,upper_limit,2):
                        new_elem_indices.append([mp(eis[0], offs), mp(eis[1], offs), mp(eis[1], offs + 1)]) #type:ignore
                        new_elem_indices.append([mp(eis[0], offs+1), mp(eis[0], offs), mp(eis[1], offs + 1)]) #type:ignore
                        new_elem_indices.append([mp(eis[0], offs + 2), mp(eis[0], offs+1), mp(eis[1], offs + 1)]) #type:ignore
                        new_elem_indices.append([mp(eis[0], offs + 2), mp(eis[1], offs + 1),mp(eis[1], offs + 2)]) #type:ignore
                        #new_elem_types+=[3,3,3,3] #type:ignore
                        new_elem_indices.append([mp(eis[1], offs + 1),mp(eis[1], offs), mp(eis[2], offs) ]) #type:ignore
                        new_elem_indices.append([mp(eis[1], offs + 1), mp(eis[2], offs), mp(eis[2], offs+1)]) #type:ignore
                        new_elem_indices.append([mp(eis[1], offs + 1), mp(eis[2], offs+1), mp(eis[2], offs + 2)]) #type:ignore
                        new_elem_indices.append([mp(eis[1], offs + 1), mp(eis[2], offs + 2), mp(eis[1], offs + 2)]) #type:ignore
                        #print(new_elem_indices[-1],mod_length)
                    #raise RuntimeError("This causes troubles ")
                        new_elem_types += [3,3,3,3,3,3,3,3] #type:ignore
                        elemental_phi_row=numpy.linspace(0,2*numpy.pi,upper_limit//2,endpoint=True)+self.phase  
                        elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==0: # Point -> Line
                for offs in range(0, upper_limit, phi_increm):
                    if phi_increm==2: # second order
                        new_elem_indices.append([mp(eis[0], offs), mp(eis[0], offs+1), mp(eis[0], offs + 2)]) #type:ignore
                        new_elem_types.append(2) #type:ignore
                    else:
                        new_elem_indices.append([mp(eis[0], offs), mp(eis[0], offs + 1)]) #type:ignore
                        new_elem_types.append(1) #type:ignore
                    elemental_phi_row=numpy.linspace(0,2*numpy.pi,upper_limit//phi_increm,endpoint=True)+self.phase  
                    elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==8: # Quad9 -> Tris at the center and hex27 in bulk
                for offs in range(0, upper_limit, phi_increm):
                    #if zero_radial_index in eis:
                    #    pass
                    #else:
                    hex27inds=[]
                    for i in range(3):
                        hex27inds += [mp(eis[6], offs + i), mp(eis[7], offs + i), mp(eis[8], offs + i)] #type:ignore
                        hex27inds += [mp(eis[3], offs + i), mp(eis[4], offs + i), mp(eis[5], offs + i)] #type:ignore
                        hex27inds+=[mp(eis[0], offs+i), mp(eis[1], offs +i), mp(eis[2], offs+i)] #type:ignore
                    new_elem_indices.append(hex27inds) #type:ignore
                    new_elem_types.append(14) #type:ignore
                elemental_phi_row=numpy.linspace(0,2*numpy.pi,upper_limit//phi_increm,endpoint=True)+self.phase  
                elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==6: # Quad4 -> Tris at the center and hex in bulk
                for offs in range(0, upper_limit, phi_increm):
                        # TODO: Tri at center
                        hexinds=[]
                        for i in range(2):
                            hexinds += [mp(eis[2], offs + i), mp(eis[3], offs + i)] #type:ignore
                            hexinds+=[mp(eis[0], offs+i), mp(eis[1], offs +i)] #type:ignore
                        new_elem_indices.append(hexinds) #type:ignore
                        new_elem_types.append(11) #type:ignore
                elemental_phi_row=numpy.linspace(0,self.angle,upper_limit//phi_increm,endpoint=not closed)+self.start_angle  
                elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==3 or elemtype==66: # Tri3 -> Tetras
                for offs in range(0, upper_limit, phi_increm):
                        # TODO: Special tetra at center
                        new_elem_indices.append([mp(eis[0], offs+1),mp(eis[1], offs+1),mp(eis[2], offs+1),mp(eis[0],offs),mp(eis[1],offs),mp(eis[2], offs)]) #type:ignore
                        new_elem_types+=[7] #type:ignore
                elemental_phi_row=numpy.linspace(0,self.angle,upper_limit//phi_increm,endpoint=not closed)+self.start_angle  
                elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==9 or elemtype==99:
                for offs in range(0, upper_limit, phi_increm):
                        new_elem_indices.append([mp(eis[0],offs),mp(eis[1],offs),mp(eis[2], offs), #type:ignore
                                                 mp(eis[0], offs + 2), mp(eis[1], offs + 2), mp(eis[2], offs + 2),
                                                 mp(eis[3],offs),mp(eis[4],offs),mp(eis[5], offs),
                                                 mp(eis[3], offs + 2), mp(eis[4], offs + 2), mp(eis[5], offs + 2),
                                                 mp(eis[0], offs + 1), mp(eis[1], offs + 1), mp(eis[2], offs + 1)
                                                 ]) #type:ignore
                        new_elem_types+=[77] #type:ignore
                elemental_phi_row=numpy.linspace(0,2*numpy.pi,upper_limit//phi_increm,endpoint=True)+self.phase  
                elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            else:
                raise RuntimeError("Implement element type "+str(elemtype))
            
            # DL/D0 Data
            num_created_elems=len(new_elem_types)-old_num_elems            
            #print(num_created_elems//upper_limit,eis,elemtype)
            if num_created_elems % len(elemental_phi_row)!=0:
                print("ERROR NUM CREATED ELEMENTS:",num_created_elems,"LEN OF THE ELEMENTAL ANGLE ROW",len(elemental_phi_row),"UPPER LIMIT",upper_limit)
                #raise RuntimeError("See above")
            
            new_elemental_phis=numpy.repeat(elemental_phi_row,num_created_elems//len(elemental_phi_row))
            #print("LENS",len(new_elemental_phis),num_created_elems,num_created_elems//len(elemental_phi_row),num_created_elems)
            #print(num_created_elems,len(new_elemental_phis),len(elemental_phi_row),elemtype)
            elemental_phis=numpy.r_[elemental_phis,new_elemental_phis] #type:ignore
            #start=0 # elemental_phis[-1] if len(elemental_phis)>0 else 0
            #end=start+1
            #elemental_phis=numpy.r_[elemental_phis,numpy.linspace(start,end,num_created_elems)]

            #print(len(numpy.repeat(elemental_phi_row,num_created_elems//upper_limit)),num_created_elems)

            if base.DL_data.shape[1]>0:
                dlrow=[]
                for dlfield in range(base.DL_data.shape[1]):  
                    dlrow+=[[base.DL_data[d0dl_index][dlfield]]*len(new_elemental_phis)]
                dlrow=numpy.transpose(numpy.array(dlrow))
                if len(new_DL_data)==0:
                    new_DL_data=dlrow
                else:
                    new_DL_data=numpy.concatenate((new_DL_data,dlrow),axis=1)

            if base.D0_data.shape[1]>0:
                d0row=[]
                for d0field in range(base.D0_data.shape[1]):            
                    d0row+=[[base.D0_data[d0dl_index][d0field]]*len(new_elemental_phis)]
                d0row=numpy.transpose(numpy.array(d0row))
                if len(new_D0_data)==0:
                    new_D0_data=d0row
                else:
                    new_D0_data=numpy.r_[new_D0_data,d0row]



        base.elem_types=numpy.array(new_elem_types) #type:ignore
        maxl=0
        for l in new_elem_indices: #type:ignore
            maxl=max(maxl,len(l)) #type:ignore
        base.elem_indices=numpy.zeros((len(new_elem_indices),maxl),dtype=int) #type:ignore
        for i,ne in enumerate(new_elem_indices): #type:ignore
            for j,e in enumerate(ne): #type:ignore
                base.elem_indices[i,j]=e #type:ignore

        if len(new_DL_data)>0:
            base.DL_data=numpy.transpose(new_DL_data,axes=(1,2,0))
            
        if len(new_D0_data)>0:            
            base.D0_data=new_D0_data
            assert base.D0_data.shape[0]==len(elemental_phis)

        # Rotate DL and D0 with m if necessary:        
        if self.apply_k_mode_expansion and base.mesh._problem.get_last_eigenmodes_k() is not None: #type:ignore
            remove_indices=[]
            remove_indices_DL=[]
            remove_indices_D0=[]
            rename_indices={}
            for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                prefixRe=prefixPair[0]
                prefixIm = prefixPair[1]
                prefixRes = prefixPair[2]
                k=base.mesh._problem.get_last_eigenmodes_k()[eigenindex] #type:ignore
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
                for i in range(max(rev_inds.keys())):
                    if i in remove_indices:
                        continue
                    new_inds[rev_inds[i]]=cnt
                    cnt+=1                
                base.elemental_field_inds=new_inds
                
        
        

        #print(field_operators)
        print("DONE HRE",base.nodal_field_inds)
        #exit()
        
        
        

class MeshDataRotationalExtrusion(MeshDataCacheOperatorBase):
    """
    Can be added as ``operator`` to :py:class:`~pyoomph.output.meshio.MeshFileOutput` to extrude the mesh along the azimuthal phi-coordinate. Most useful combined with :py:class:`MeshDataCombineWithEigenfunction` and azimuthal normal mode stability analysis.

    Args:
        n_segments: Number of segments to extrude the mesh to.
        angle: Angle to extrude the mesh to. If larger than 2*pi, it will be cut off at 2*pi.
        start_angle: Angle to start the extrusion at.
        rotate_eigendata_with_mode_m: If True, the eigendata will be rotated with the azimuthal mode number m. This is useful for azimuthal normal mode stability analysis.
    """
    def __init__(self,n_segments:int=32,angle:float=2*numpy.pi,start_angle:float=0.0,rotate_eigendata_with_mode_m:bool=True):
        super(MeshDataRotationalExtrusion, self).__init__()
        self.n_segments=n_segments
        self.angle=float(angle)
        if self.angle>2*numpy.pi:
            self.angle=2*numpy.pi
        self.start_angle=float(start_angle)
        self.rotate_eigendata_with_mode_m=rotate_eigendata_with_mode_m

    def apply(self,base:MeshDataCacheEntry):
        n_segments=self.n_segments
        phi_increm=1
        if base.mesh._eqtree.get_code_gen()._coordinate_space not in {"C1","C1TB"}: 
            n_segments*=2
            phi_increm=2

        closed=(self.angle>=2*numpy.pi-1e-10)
        phis=numpy.linspace(0,self.angle,n_segments,endpoint=not closed)+self.start_angle #type:ignore

        r_pos=base.nodal_values[:, base.nodal_field_inds["coordinate_x"]]
        zero_radial_index=numpy.argsort(r_pos)[0] #type:ignore
        if r_pos[zero_radial_index]<-1.0e-9:
            raise RuntimeError("Cannot rotationally extrude meshes with negative x-coordinates (radius)")
        elif r_pos[zero_radial_index]>1.0e-9:
            zero_radial_index = set({})
        else:
            zero_radial_index=set(numpy.argwhere(r_pos<=1.0e-9)[:,0]) #type:ignore


        stride = base.nodal_values.shape[0]

        new_nodal_values=[]
        new_nodal_field_inds=base.nodal_field_inds.copy()
        field_operators={}

        new_D0_data=[]
        new_DL_data=[]

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
        if self.rotate_eigendata_with_mode_m and base.mesh._problem.get_last_eigenmodes_m() is not None: #type:ignore
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
                        newindex = 0
                        for name, index in sorted(new_nodal_field_inds.items(), key=lambda item: item[1]): #type:ignore
                            new_nodal_field_inds[name] = newindex
                            newindex += 1
                        new_nodal_field_inds[fnRes]=max(new_nodal_field_inds.values()) + 1
                        m=base.mesh._problem.get_last_eigenmodes_m()[eigenindex] #type:ignore
                        field_operators[fnRes] = [lambda RealPart,ImagPart : numpy.outer(numpy.cos(m*phis), RealPart).flatten() + numpy.outer(numpy.sin(m*phis), ImagPart).flatten(), fnRe,fnIm] #type:ignore

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
                        for cindex,componame in enumerate(composRes):
                            if componame.endswith("_x"):
                                r_index=cindex
                            elif componame.endswith("_phi"):
                                phi_index=cindex                                

                        if r_index is not None and phi_index is not None:
                            def get_x_component(ReR,ImR,ReP,ImP): #type:ignore
                                Vr_cos_phi=numpy.outer(numpy.cos(m * phis)*numpy.cos(phis),ReR).flatten()+numpy.outer(numpy.sin(m * phis)*numpy.cos(phis),ImR).flatten() #type:ignore
                                Vphi_sin_phi=numpy.outer(numpy.cos(m * phis)*numpy.sin(phis),ReP).flatten()+numpy.outer(numpy.sin(m * phis)*numpy.sin(phis),ImP).flatten() #type:ignore
                                return Vr_cos_phi+Vphi_sin_phi #type:ignore
                            def get_y_component(ReR,ImR,ReP,ImP): #type:ignore
                                Vr_sin_phi=numpy.outer(numpy.cos(m * phis)*numpy.sin(phis),ReR).flatten()+numpy.outer(numpy.sin(m * phis)*numpy.sin(phis),ImR).flatten() #type:ignore
                                Vphi_cos_phi=numpy.outer(numpy.cos(m * phis)*numpy.cos(phis),ReP).flatten()+numpy.outer(numpy.sin(m * phis)*numpy.cos(phis),ImP).flatten() #type:ignore
                                return Vr_sin_phi-Vphi_cos_phi #type:ignore
                            field_operators[composRes[r_index]]=[get_x_component,composRe[r_index],composIm[r_index],composRe[phi_index],composIm[phi_index]] 
                            field_operators[composRes[phi_index]] = [get_y_component, composRe[r_index],composIm[r_index], composRe[phi_index],composIm[phi_index]]
                            yname=vecname+"_y"
                            field_operators[yname]=field_operators.pop(composRes[phi_index]) #type:ignore
                            new_nodal_field_inds[yname]=new_nodal_field_inds.pop(composRes[phi_index])
                            completed_eigen_vector_fields.add(vecname) #type:ignore
                            if len(composRes)>2:
                                new_nodal_field_inds[vecname + "_z"] = max(new_nodal_field_inds.values()) + 1
                                field_operators[vecname+"_z"]= [lambda ReVy,ImVy: numpy.outer(numpy.cos(m * phis), ReVy).flatten()+numpy.outer(numpy.sin(m * phis), ImVy).flatten(),prefixRe + veccompos[0][len(prefixRes):-len("_x")] + "_y",prefixIm + veccompos[0][len(prefixRes):-len("_x")] + "_y"] #type:ignore
                            vector_fields[vecname]=[vecname+component for component in ["_x","_y","_z"][0:len(composRes)]]
                            #print(new_nodal_field_inds,vector_fields)

                
            # Also assemble the eigenperturbation of the position
            if "EigenRe_coordinate_x" in base.nodal_field_inds:
                
                def get_x_component(ReR,ImR): #type:ignore
                    Vr_cos_phi=numpy.outer(numpy.cos(m * phis)*numpy.cos(phis),ReR).flatten()+numpy.outer(numpy.sin(m * phis)*numpy.cos(phis),ImR).flatten() #type:ignore
                    #Vphi_sin_phi=numpy.outer(numpy.cos(m * phis)*numpy.sin(phis),ReP).flatten()+numpy.outer(numpy.sin(m * phis)*numpy.sin(phis),ImP).flatten() #type:ignore
                    return Vr_cos_phi#+Vphi_sin_phi #type:ignore
                def get_y_component(ReR,ImR): #type:ignore
                    Vr_sin_phi=numpy.outer(numpy.cos(m * phis)*numpy.sin(phis),ReR).flatten()+numpy.outer(numpy.sin(m * phis)*numpy.sin(phis),ImR).flatten() #type:ignore
                    #Vphi_cos_phi=numpy.outer(numpy.cos(m * phis)*numpy.cos(phis),ReP).flatten()+numpy.outer(numpy.sin(m * phis)*numpy.cos(phis),ImP).flatten() #type:ignore
                    return Vr_sin_phi#-Vphi_cos_phi #type:ignore
                field_operators["Eigen_coordinate_x"]= [get_x_component,"EigenRe_coordinate_x","EigenIm_coordinate_x"] #type:ignore
                field_operators["Eigen_coordinate_y"]= [get_y_component,"EigenRe_coordinate_x","EigenIm_coordinate_x"] #type:ignore
                if "EigenRe_coordinate_y" in base.nodal_field_inds:
                    field_operators["Eigen_coordinate_z"]= [lambda ReVy,ImVy: numpy.outer(numpy.cos(m * phis), ReVy).flatten()+numpy.outer(numpy.sin(m * phis), ImVy).flatten(),"EigenRe_coordinate_y","EigenIm_coordinate_y"] #type:ignore
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
                        newindex=0
                        for name, index in sorted(new_nodal_field_inds.items(), key=lambda item: item[1]): #type:ignore
                            new_nodal_field_inds[name]=newindex
                            newindex+=1

                else:
                    field_operators[vfield + "_x"] = [lambda vx: numpy.outer(numpy.cos(phis), vx).flatten(),vfield + "_x"] #type:ignore
                    field_operators[vfield + "_y"] = [lambda vx: numpy.outer(numpy.sin(phis), vx).flatten(),vfield + "_x"] #type:ignore

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
        base.nodal_values=numpy.transpose(numpy.array(new_nodal_values)) #type:ignore        
        base.vector_fields=vector_fields


        if base.tesselate_tri:
            raise RuntimeError("rotational extrusion cannot be combined with tesselate_tri=True yet")
        if base.discontinuous and (base.D0_data.shape[1]>0 or base.DL_data.shape[1]>0):
            raise RuntimeError("rotational extrusion does not work with discontinuous=True, at least if D0 or DL fields are defined")
        elem_types=base.elem_types


        upper_limit=n_segments-(0 if closed else phi_increm)

        mod_length=base.nodal_values.shape[0]
        new_elem_types=[]
        new_elem_indices=[]
        elemental_phis=numpy.zeros((0,))
        
        mp = lambda i, o: (i + o * stride) % mod_length #type:ignore
        for d0dl_index,(elemtype,eis) in enumerate(zip(elem_types,base.elem_indices)):            
            old_num_elems=len(new_elem_types)       
            if elemtype==1: # LineC1
                
                    if len(zero_radial_index.intersection(eis))>0:
                        for offs in range(upper_limit):
                            if eis[0] in zero_radial_index:
                                new_elem_indices.append([eis[0],mp(eis[1],offs),mp(eis[1],offs+1)]) #type:ignore
                                new_elem_types.append(3) #type:ignore
                            elif eis[1] in zero_radial_index:
                                new_elem_indices.append([eis[1], mp(eis[0],offs), mp(eis[0],offs+1)]) #type:ignore
                                new_elem_types.append(3) #type:ignore
                        elemental_phi_row=numpy.linspace(0,self.angle,upper_limit,endpoint=not closed)+self.start_angle  
                        elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
                    else: # Two triangles
                        for offs in range(upper_limit):
                            new_elem_indices.append([mp(eis[0],offs), mp(eis[1],offs), mp(eis[1],offs+1)]) #type:ignore
                            new_elem_types.append(3) #type:ignore
                            new_elem_indices.append([mp(eis[0],offs),  mp(eis[1],offs+1),mp(eis[0],offs+1)]) #type:ignore
                            new_elem_types.append(3) #type:ignore
                        elemental_phi_row=numpy.linspace(0,self.angle,upper_limit,endpoint=not closed)+self.start_angle  
                        elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))

            elif elemtype == 2:  # LineC2                
                    if len(zero_radial_index.intersection(eis))>0:
                        # One tesselated triangle only
                        for offs in range(0,upper_limit,2):
                            if eis[0] in zero_radial_index:
                                new_elem_indices.append([eis[0], mp(eis[1], offs), mp(eis[1], offs + 2)]) #type:ignore
                                new_elem_indices.append([mp(eis[2], offs + 1), mp(eis[1], offs), mp(eis[2], offs)]) #type:ignore
                                new_elem_indices.append([mp(eis[2], offs + 2), mp(eis[1], offs+2), mp(eis[2], offs+1)]) #type:ignore
                                new_elem_indices.append([mp(eis[2], offs + 1), mp(eis[1], offs+2), mp(eis[1], offs)]) #type:ignore
                                new_elem_types+=[3,3,3,3] #type:ignore
                            else:          
                                new_elem_indices.append([eis[2], mp(eis[1], offs+2), mp(eis[1], offs)]) #type:ignore
                                new_elem_indices.append([mp(eis[0], offs + 1), mp(eis[1], offs), mp(eis[0], offs)]) #type:ignore
                                new_elem_indices.append([mp(eis[0], offs + 2), mp(eis[1], offs+2), mp(eis[0], offs+1)]) #type:ignore
                                new_elem_indices.append([mp(eis[0], offs + 1), mp(eis[1], offs+2), mp(eis[1], offs)]) #type:ignore
                                new_elem_types+=[3,3,3,3] #type:ignore
                        elemental_phi_row=numpy.linspace(0,self.angle,upper_limit//2,endpoint=not closed)+self.start_angle  
                        elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
                    else:
                        for offs in range(0,upper_limit,2):
                            new_elem_indices.append([mp(eis[0], offs), mp(eis[1], offs), mp(eis[1], offs + 1)]) #type:ignore
                            new_elem_indices.append([mp(eis[0], offs+1), mp(eis[0], offs), mp(eis[1], offs + 1)]) #type:ignore
                            new_elem_indices.append([mp(eis[0], offs + 2), mp(eis[0], offs+1), mp(eis[1], offs + 1)]) #type:ignore
                            new_elem_indices.append([mp(eis[0], offs + 2), mp(eis[1], offs + 1),mp(eis[1], offs + 2)]) #type:ignore
                            #new_elem_types+=[3,3,3,3] #type:ignore
                            new_elem_indices.append([mp(eis[1], offs + 1),mp(eis[1], offs), mp(eis[2], offs) ]) #type:ignore
                            new_elem_indices.append([mp(eis[1], offs + 1), mp(eis[2], offs), mp(eis[2], offs+1)]) #type:ignore
                            new_elem_indices.append([mp(eis[1], offs + 1), mp(eis[2], offs+1), mp(eis[2], offs + 2)]) #type:ignore
                            new_elem_indices.append([mp(eis[1], offs + 1), mp(eis[2], offs + 2), mp(eis[1], offs + 2)]) #type:ignore
                        #raise RuntimeError("This causes troubles ")
                            new_elem_types += [3,3,3,3,3,3,3,3] #type:ignore
                        elemental_phi_row=numpy.linspace(0,self.angle,upper_limit//2,endpoint=not closed)+self.start_angle  
                        elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==0: # Point -> Line
                for offs in range(0, upper_limit, phi_increm):
                    if eis[0] == zero_radial_index:
                        new_elem_indices.append([eis[0]]) #type:ignore
                        new_elem_types.append(0) #type:ignore
                    else:
                        if phi_increm==2: # second order
                            new_elem_indices.append([mp(eis[0], offs), mp(eis[0], offs+1), mp(eis[0], offs + 2)]) #type:ignore
                            new_elem_types.append(2) #type:ignore
                        else:
                            new_elem_indices.append([mp(eis[0], offs), mp(eis[0], offs + 1)]) #type:ignore
                            new_elem_types.append(1) #type:ignore
                    elemental_phi_row=numpy.linspace(0,self.angle,upper_limit//phi_increm,endpoint=not closed)+self.start_angle  
                    elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==8: # Quad9 -> Tris at the center and hex27 in bulk
                for offs in range(0, upper_limit, phi_increm):
                    #if zero_radial_index in eis:
                    #    pass
                    #else:
                    hex27inds=[]
                    for i in range(3):
                        hex27inds += [mp(eis[6], offs + i), mp(eis[7], offs + i), mp(eis[8], offs + i)] #type:ignore
                        hex27inds += [mp(eis[3], offs + i), mp(eis[4], offs + i), mp(eis[5], offs + i)] #type:ignore
                        hex27inds+=[mp(eis[0], offs+i), mp(eis[1], offs +i), mp(eis[2], offs+i)] #type:ignore
                    new_elem_indices.append(hex27inds) #type:ignore
                    new_elem_types.append(14) #type:ignore
                elemental_phi_row=numpy.linspace(0,self.angle,upper_limit//phi_increm,endpoint=not closed)+self.start_angle  
                elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==6: # Quad4 -> Tris at the center and hex in bulk
                for offs in range(0, upper_limit, phi_increm):
                        # TODO: Tri at center
                        hexinds=[]
                        for i in range(2):
                            hexinds += [mp(eis[2], offs + i), mp(eis[3], offs + i)] #type:ignore
                            hexinds+=[mp(eis[0], offs+i), mp(eis[1], offs +i)] #type:ignore
                        new_elem_indices.append(hexinds) #type:ignore
                        new_elem_types.append(11) #type:ignore
                elemental_phi_row=numpy.linspace(0,self.angle,upper_limit//phi_increm,endpoint=not closed)+self.start_angle  
                elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==3 or elemtype==66: # Tri3 -> Tetras
                for offs in range(0, upper_limit, phi_increm):
                        # TODO: Special tetra at center
                        new_elem_indices.append([mp(eis[0], offs+1),mp(eis[1], offs+1),mp(eis[2], offs+1),mp(eis[0],offs),mp(eis[1],offs),mp(eis[2], offs)]) #type:ignore
                        new_elem_types+=[7] #type:ignore
                elemental_phi_row=numpy.linspace(0,self.angle,upper_limit//phi_increm,endpoint=not closed)+self.start_angle  
                elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            elif elemtype==9 or elemtype==99:
                for offs in range(0, upper_limit, phi_increm):
                        new_elem_indices.append([mp(eis[0],offs),mp(eis[1],offs),mp(eis[2], offs), #type:ignore
                                                 mp(eis[0], offs + 2), mp(eis[1], offs + 2), mp(eis[2], offs + 2),
                                                 mp(eis[3],offs),mp(eis[4],offs),mp(eis[5], offs),
                                                 mp(eis[3], offs + 2), mp(eis[4], offs + 2), mp(eis[5], offs + 2),
                                                 mp(eis[0], offs + 1), mp(eis[1], offs + 1), mp(eis[2], offs + 1)
                                                 ]) #type:ignore
                        new_elem_types+=[77] #type:ignore
                elemental_phi_row=numpy.linspace(0,self.angle,upper_limit//phi_increm,endpoint=not closed)+self.start_angle  
                elemental_phi_row+=elemental_phi_row[-1]/(2*len(elemental_phi_row))
            else:
                raise RuntimeError("Implement element type "+str(elemtype))
            
            # DL/D0 Data
            num_created_elems=len(new_elem_types)-old_num_elems            
            #print(num_created_elems//upper_limit,eis,elemtype)
            if num_created_elems % len(elemental_phi_row)!=0:
                print("ERROR NUM CREATED ELEMENTS:",num_created_elems,"LEN OF THE ELEMENTAL ANGLE ROW",len(elemental_phi_row),"UPPER LIMIT",upper_limit)
                raise RuntimeError("See above")
            
            new_elemental_phis=numpy.repeat(elemental_phi_row,num_created_elems//len(elemental_phi_row))
            #print("LENS",len(new_elemental_phis),num_created_elems,num_created_elems//len(elemental_phi_row),num_created_elems)
            #print(num_created_elems,len(new_elemental_phis),len(elemental_phi_row),elemtype)
            elemental_phis=numpy.r_[elemental_phis,new_elemental_phis] #type:ignore
            #start=0 # elemental_phis[-1] if len(elemental_phis)>0 else 0
            #end=start+1
            #elemental_phis=numpy.r_[elemental_phis,numpy.linspace(start,end,num_created_elems)]

            #print(len(numpy.repeat(elemental_phi_row,num_created_elems//upper_limit)),num_created_elems)

            if base.DL_data.shape[1]>0:
                dlrow=[]
                for dlfield in range(base.DL_data.shape[1]):  
                    dlrow+=[[base.DL_data[d0dl_index][dlfield]]*len(new_elemental_phis)]
                dlrow=numpy.transpose(numpy.array(dlrow))
                if len(new_DL_data)==0:
                    new_DL_data=dlrow
                else:
                    new_DL_data=numpy.concatenate((new_DL_data,dlrow),axis=1)

            if base.D0_data.shape[1]>0:
                d0row=[]
                for d0field in range(base.D0_data.shape[1]):            
                    d0row+=[[base.D0_data[d0dl_index][d0field]]*len(new_elemental_phis)]
                d0row=numpy.transpose(numpy.array(d0row))
                if len(new_D0_data)==0:
                    new_D0_data=d0row
                else:
                    new_D0_data=numpy.r_[new_D0_data,d0row]



        base.elem_types=numpy.array(new_elem_types) #type:ignore
        maxl=0
        for l in new_elem_indices: #type:ignore
            maxl=max(maxl,len(l)) #type:ignore
        base.elem_indices=numpy.zeros((len(new_elem_indices),maxl),dtype=int) #type:ignore
        for i,ne in enumerate(new_elem_indices): #type:ignore
            for j,e in enumerate(ne): #type:ignore
                base.elem_indices[i,j]=e #type:ignore

        if len(new_DL_data)>0:
            base.DL_data=numpy.transpose(new_DL_data,axes=(1,2,0))
            
        if len(new_D0_data)>0:            
            base.D0_data=new_D0_data
            assert base.D0_data.shape[0]==len(elemental_phis)

        # Rotate DL and D0 with m if necessary:        
        if self.rotate_eigendata_with_mode_m and base.mesh._problem.get_last_eigenmodes_m() is not None: #type:ignore
            remove_indices=[]
            remove_indices_DL=[]
            remove_indices_D0=[]
            rename_indices={}
            for eigenindex,prefixPair in base._additional_eigendata.items(): #type:ignore
                prefixRe=prefixPair[0]
                prefixIm = prefixPair[1]
                prefixRes = prefixPair[2]
                m=base.mesh._problem.get_last_eigenmodes_m()[eigenindex] #type:ignore
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
                for i in range(max(rev_inds.keys())):
                    if i in remove_indices:
                        continue
                    new_inds[rev_inds[i]]=cnt
                    cnt+=1                
                base.elemental_field_inds=new_inds
