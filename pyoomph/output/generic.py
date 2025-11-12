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
 
import os
from pathlib import Path,PurePath
from ..expressions.generic import Expression,  ExpressionOrNum, GlobalParameter
from ..expressions.units import unit_to_string

from ..meshes.mesh import ODEStorageMesh


from ..generic.codegen import BaseEquations
import inspect
import _pyoomph

from scipy.io import savemat,loadmat #type:ignore

from ..typings import *
import numpy

if TYPE_CHECKING:
    from ..generic.codegen import EquationTree
    from ..generic.problem import Problem
    from ..meshes.mesh import AnySpatialMesh
    from ..meshes.meshdatacache import MeshDataCacheEntry, MeshDataCacheOperatorBase, MeshDataEigenModes


class _BaseOutputter:
    def __init__(self):
        self._stages:Optional[Set[str]]=None
        self._eqtree:"EquationTree"
        self._mpi_rank:int
        self.problem:"Problem"
        pass

    def after_remeshing(self,eqtree:"EquationTree"):
        pass

    def init(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]]=None,rank:int=0):
        self._eqtree=eqtree
        self._mpi_rank=rank
        pass

    def output(self,step:int)->None:
        raise NotImplementedError("Not implemented")

    def delete_files_from_previous_simulation(self)->None:
        pass

    def get_time(self,nondimensional:bool=False)->float:
        return self.problem.get_current_time(dimensional=not nondimensional,as_float=True)

    def clean_up(self)->None:
        pass

    def set_active_on_stages(self,stages:Optional[Union[str,Set[str]]]):
        if stages is not None:
            if isinstance(stages,str):
                stages={stages}
            elif isinstance(stages,set): #type:ignore
                try:
                    stages=set(stages)
                except:
                    stages=set({stages}) #type:ignore
        self._stages=stages #type:ignore

    def get_active_on_stages(self) -> Optional[Set[str]]:
        return self._stages


class _BaseNumpyOutput(_BaseOutputter):
    def __init__(self,mesh:"AnySpatialMesh"):
        super().__init__()
        self.mesh=mesh

    def clean_up(self):
        pass

    def after_remeshing(self,eqtree:"EquationTree"):
        #Refresh the mesh!
        m=eqtree.get_mesh()
        assert not isinstance(m,ODEStorageMesh)
        self.mesh=m

    def get_cached_mesh_data(self,mesh:"AnySpatialMesh",nondimensional:bool=False,tesselate_tri:bool=False,eigenvector:Optional[Union[int,Sequence[int]]]=None,eigenmode:"MeshDataEigenModes"="abs",history_index:int=0,with_halos:bool=False,operator:Optional["MeshDataCacheOperatorBase"]=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True)->"MeshDataCacheEntry":
        pr = self.mesh.get_problem()
        cache = pr.get_cached_mesh_data(mesh, tesselate_tri=tesselate_tri, nondimensional=nondimensional,eigenvector=eigenvector,eigenmode=eigenmode,history_index=history_index,with_halos=with_halos,operator=operator,discontinuous=discontinuous,add_eigen_to_mesh_positions=add_eigen_to_mesh_positions)
        return cache




####################


class _TextOutput(_BaseNumpyOutput):
    def __init__(self,mesh:"AnySpatialMesh",*fields:str,ftrunk:str="txtout",in_subdir:bool=True,file_ext:Optional[Union[str,List[str]]]=None,eigenvector:Optional[int]=None,eigenmode:"MeshDataEigenModes"="abs",nondimensional:bool=False,hide_lagrangian:bool=True,hide_underscore:bool=True,reverse_segment_if:Optional[Callable[[List[int],NPFloatArray],bool]]=None,sort_segments_by:Optional[Callable[[List[int],NPFloatArray],float]]=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True,operator:Optional["MeshDataCacheOperatorBase"]=None,tesselate_tri:bool=True):
        super().__init__(mesh)
        self.fname_trunk=ftrunk
        self._orbit_subdir=None
        self.in_subdir=in_subdir
        self.file_ext=file_ext
        self.fields:List[str]=[*fields]
        #self._additional_outs=[]
        self.eigenvector=eigenvector
        self.eigenvector_mode:"MeshDataEigenModes"=eigenmode
        self.nondimensional=nondimensional
        self.hide_lagrangian=hide_lagrangian
        self.hide_underscore = hide_underscore
        self.reverse_segment_if=reverse_segment_if
        self.sort_segments_by=sort_segments_by
        self.discontinuous=discontinuous
        self.add_eigen_to_mesh_positions=add_eigen_to_mesh_positions
        self.operator=operator
        self.tesselate_tri=tesselate_tri



    def init(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]]=None,rank:int=0):
        super().init(eqtree,continue_info,rank)
        if isinstance(self.mesh,str):
            self.mesh=self.problem.get_mesh(self.mesh)
        if self.in_subdir and rank==0:
                Path(os.path.join(self.problem.get_output_directory()),self.fname_trunk).mkdir(parents=True, exist_ok=True) 
        if self.file_ext is None:
            self.file_ext=self.problem.default_1d_file_extension

    def get_filename(self, step:int):
        assert self.file_ext is not None
        if isinstance(self.file_ext, (list,set)):
            res:List[str] = []
            for e in self.file_ext:
                fname = self.fname_trunk + "_{:06d}".format(step) + "." + e
                if self.in_subdir:
                    fname = os.path.join(self.problem.get_output_directory(self._orbit_subdir), self.fname_trunk, fname)
                else:
                    fname = os.path.join(self.problem.get_output_directory(self._orbit_subdir), fname)
                res.append(fname)
            return res
        else:
            fname = self.fname_trunk + "_{:06d}".format(step) + "." + self.file_ext
            if self.in_subdir:
                fname = os.path.join(self.problem.get_output_directory(self._orbit_subdir), self.fname_trunk, fname)
            else:
                fname = os.path.join(self.problem.get_output_directory(self._orbit_subdir), fname)
            return fname

    def change_output_directory(self,newdir:str,eqtree:"EquationTree"):
        basedir=self.problem.get_output_directory()
        if Path(basedir).samefile(newdir):
            self._orbit_subdir=None
        else:
            self._orbit_subdir=Path.relative_to(Path(newdir),Path(basedir))
            Path(os.path.join(self.problem.get_output_directory(self._orbit_subdir)),self.fname_trunk).mkdir(parents=True, exist_ok=True) 
        

    def output(self,step:int):
        mesh = self.mesh
        if (not mesh.is_mesh_distributed()) and self._mpi_rank > 0:
            return
        if self.eigenvector is not None:
            if self.eigenvector >= len(self.mesh.get_problem()._last_eigenvectors): #type:ignore
                return  # No output hrere
        cache=self.get_cached_mesh_data(self.mesh,nondimensional=self.nondimensional,tesselate_tri=self.tesselate_tri,eigenvector=self.eigenvector,eigenmode=self.eigenvector_mode,discontinuous=self.discontinuous,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions,operator=self.operator)
        if len(self.fields)==0:
            self.fields=cache.get_default_output_fields(rem_lagrangian=self.hide_lagrangian,rem_underscore=self.hide_underscore)
        header:List[str] = []
        timeinfo = self.get_time(nondimensional=self.nondimensional)
        fname = self.get_filename(step)
        datag:List[NPFloatArray]=[]
        for i,f in enumerate(self.fields):
            d=cache.get_data(f)
            if d is not None:
                datag.append(d)
                header.append(f + cache.get_unit(f,as_string=True))
        if self.discontinuous:
            numDL=cache.DL_data.shape[1]
            for k,v in cache.elemental_field_inds.items():
                if k in self.fields:
                    header.append(k + cache.get_unit(k,as_string=True))
                    if v>=numDL:
                        datag.append(cache.D0_data[:,v-numDL])
                    else:
                        datag.append(cache.DL_data[:,v])

        data:NPFloatArray=numpy.array(datag).transpose()  #type:ignore
        if mesh.get_element_dimension()==1 and not self.discontinuous:
            lsegs_in,_=cache.get_interface_line_segments() 
            lsegs=lsegs_in.copy()
            coords:List[NPFloatArray]=[]
            for c in ["x","y","z"]:
                if "coordinate_"+c in self.fields:
                    coords.append(data[:,self.fields.index("coordinate_"+c)])                        
            coordsA:NPFloatArray=numpy.array(coords)
            if self.reverse_segment_if is not None:
                for i,l in enumerate(lsegs):
                    if self.reverse_segment_if(l,coordsA):
                        lsegs[i]=list(reversed(l))
            if self.sort_segments_by is not None:
                lsegs=list(sorted(lsegs,key=lambda k : self.sort_segments_by(k,coordsA)))



            sortdata:List[NPFloatArray]=[] 
            for i,ls in enumerate(lsegs): 
                sortdata.append(data[ls]) 
                if i+1<len(lsegs): 
                    sortdata.append([[numpy.nan]*len(data[0,:])])  #type:ignore
            data:NPFloatArray=numpy.vstack(sortdata) #type:ignore


        params:Dict[str,str] = {}
        for n in self.mesh.get_problem().get_global_parameter_names():
            params[n] =str(self.mesh.get_problem().get_global_parameter(n).value)
        if self.eigenvector is not None:
            eigeninfostr="OUTPUT_IS_"+str(self.eigenvector_mode)+"_OF_EIGENVALUE_"+str(self.eigenvector)
            if self.mesh.get_problem()._last_eigenvalues_m is not None and self.eigenvector<len(self.mesh.get_problem()._last_eigenvalues_m): #type:ignore
                eigeninfostr+="_AND_ANGULAR_MODE_"+str(self.mesh.get_problem()._last_eigenvalues_m[self.eigenvector]) #type:ignore
            params[eigeninfostr]=str(self.mesh.get_problem()._last_eigenvalues[self.eigenvector]) #type:ignore
        if isinstance(fname, list):
            for f in fname:
                save_by_extension(f, data, header, timeinfo,params,cache.elem_indices if cache.discontinuous else None)
        else:
            save_by_extension(fname, data, header, timeinfo,params,cache.elem_indices if cache.discontinuous else None)
        self.clean_up()



    def delete_files_from_previous_simulation(self):
        try:
            step=0
            while True:
                fn=self.get_filename(step)
                if isinstance(fn,list):
                    for f in fn:
                        os.remove(f)
                else:
                    os.remove(fn)
                step+=1
        except:
            pass



####################

def save_by_extension(fname:str,data:NPFloatArray,header:List[str],timeinfo:float,params:Dict[str,str],discontinuous_elem_indices:Optional[NPInt32Array]=None):
    _,ext=os.path.splitext(fname)
    if ext in [".mat",".MAT"]:
        mdict={}
        for i,fn in enumerate(header):
            if "[" in fn:
                fn=fn[0:fn.find("[")]
            mdict[fn]=data[:,i] #type:ignore
        if timeinfo is not None:
            mdict["current_time"]=timeinfo
        for pn,v in params.items():
            mdict[pn]=v
        if discontinuous_elem_indices:
            raise RuntimeError("Cannot output MATLAB in discontinuous mode yet")
        savemat(fname,mdict,appendmat=False)
    else:
        headerr="\t".join(header)
        if timeinfo is not None:
            headerr+="\t@time="+str(timeinfo)
        elif len(params)>0:
            headerr += "\t@"
        for pn,v in params.items():
            headerr+="\t"+pn+"="+str(v)
        if discontinuous_elem_indices is None:
            numpy.savetxt(fname,data,header=headerr.lstrip(),delimiter="\t") #type:ignore
        else:
            with open(fname,"w") as f:
                f.write("#"+headerr.lstrip()+"\n")
                for e in discontinuous_elem_indices:
                    for j in e:
                        if j==-1:
                            break
                        f.write("\t".join(map(str,data[j,:]))+"\n")
                    f.write("\n")


class _OutputTxtAlongLine(_BaseOutputter):
    def __init__(self,*fields:str,coords:Optional[Union[NPFloatArray,List[Sequence[ExpressionOrNum]]]]=None,start:Optional[List[ExpressionOrNum]]=None,end:Optional[List[ExpressionOrNum]]=None,N:Optional[int]=None,isovalue:Optional[Tuple[str,ExpressionOrNum]]=None,mesh:Optional["AnySpatialMesh"]=None,ftrunk:str="along_line",in_subdir:bool=True,file_ext:Optional[Union[str,List[str]]]=None,hide_lagrangian:bool=True,hide_underscore:bool=True,eigenvector:Optional[int]=None,eigenmode:"MeshDataEigenModes"="abs",NaN_outside:bool=False):
        super().__init__()
        if mesh is None:
            raise ValueError("Need to supply at least a mesh")
        self.fname_trunk=ftrunk
        self.in_subdir=in_subdir
        assert mesh is not None
        self.mesh=mesh
        self.problem=mesh.get_problem()
        self.file_ext=file_ext
        self.fields:List[str]=[*fields]
        self.use_tri_interpolator=True
        self.hide_lagrangian=hide_lagrangian
        self.hide_underscore = hide_underscore
        self.eigenvector=eigenvector
        self.eigenmode:"MeshDataEigenModes"=eigenmode
        self.NaN_outside=NaN_outside
        
        if isovalue is not None:
            if coords is not None or start is not None or end is not None:
                raise RuntimeError("Cannot specify coords, start or end if isovalue is set")
            self.isovalue=isovalue
        elif coords is None:            
            if (start is None) or (end is  None) or (N is None):
                raise RuntimeError("Require to specify start, end and N if no coords are given")
                        
            #ss=p.get_scaling("spatial")
            meshdata = self.mesh.get_problem().get_cached_mesh_data(self.mesh, tesselate_tri=True, nondimensional=False)
            ss=meshdata.get_unit("spatial")

            s:NPFloatArray=numpy.array([float(x/ss) for x in start]) #type:ignore
            e:NPFloatArray = numpy.array([float(x/ss) for x in end]) #type:ignore
            l:NPFloatArray=numpy.linspace(0,1,N,endpoint=True) #type:ignore
            self.coords:NPFloatArray=numpy.tensordot(1-l,s,axes=0)+numpy.tensordot(l,e,axes=0)
            self.isovalue=None
        else:
            if (start is not None) or (end is not None) or (N is not None):
                raise RuntimeError("Cannot specify start, end or N if coords are given")
            meshdata = self.mesh.get_problem().get_cached_mesh_data(self.mesh, tesselate_tri=True, nondimensional=False)
            ss=meshdata.get_unit("spatial")
            coords=numpy.array(coords/ss,dtype=numpy.float64)
            self.coords:NPFloatArray=numpy.array(coords) #type:ignore
            self.isovalue=None

    def init(self, eqtree:"EquationTree", continue_info:Optional[Dict[str,Any]]=None, rank:int=0):
        super().init(eqtree, continue_info, rank)
        if isinstance(self.mesh, str):
            self.mesh = self.problem.get_mesh(self.mesh)
        if self.in_subdir and rank == 0:
            Path(os.path.join(self.problem.get_output_directory()), self.fname_trunk).mkdir(parents=True, exist_ok=True)
        if self.file_ext is None:
            self.file_ext=self.problem.default_1d_file_extension




    def get_filename(self,step:int) -> Union[List[str] ,str]:
        assert self.file_ext is not None
        if isinstance(self.file_ext,list):
            res:List[str]=[]
            for e in self.file_ext:
                fname = self.fname_trunk + "_{:06d}".format(step) + "." + e
                if self.in_subdir:
                    fname = os.path.join(self.problem.get_output_directory(), self.fname_trunk, fname)
                else:
                    fname = os.path.join(self.problem.get_output_directory(), fname)
                res.append(fname)
            return res
        else:
            fname = self.fname_trunk + "_{:06d}".format(step) + "." + self.file_ext
            if self.in_subdir:
                fname = os.path.join(self.problem.get_output_directory(), self.fname_trunk, fname)
            else:
                fname = os.path.join(self.problem.get_output_directory(), fname)
            return fname

    def after_remeshing(self,eqtree:"EquationTree"):
        m=eqtree.get_mesh()
        assert not isinstance(m,ODEStorageMesh)
        self.mesh=m


    def get_data_and_descs(self)->Tuple[NPFloatArray,List[str]]:

        if self.use_tri_interpolator:
            meshdata = self.mesh.get_problem().get_cached_mesh_data(self.mesh, tesselate_tri=True, nondimensional=False,eigenmode=self.eigenmode,eigenvector=self.eigenvector)
            
            coordinates:NPFloatArray=meshdata.get_coordinates()
            import matplotlib.tri as tri

            triang = tri.Triangulation(coordinates[0,:], coordinates[1,:], meshdata.elem_indices)
            if self.isovalue is not None:
                import matplotlib.pyplot as plt
                isodata=meshdata.get_data(self.isovalue[0])-float(self.isovalue[1]) # TODO: Nondimensionalize cast to float
                isol=plt.tricontour(triang,isodata,[0.0])
                self.coords=[]
                for path in isol.allsegs[0]:
                    for p in path:
                        self.coords.append(p)
                self.coords=numpy.array(self.coords)
                plt.close()
                del isol

            fields=meshdata.get_default_output_fields(rem_lagrangian=self.hide_lagrangian,rem_underscore=self.hide_underscore)
            fields=[f for f in fields if f not in meshdata.elemental_field_inds.keys()] # Does not work for elemental fields
            dataL:List[NPFloatArray]=[]
            for f in fields:
                inter=tri.LinearTriInterpolator(triang, meshdata.get_data(f))
                inter=inter(self.coords[:,0],self.coords[:,1]) #type:ignore
                if self.NaN_outside:                    
                    if f not in {"coordinate_x","coordinate_y","coordinate_z"}:                        
                        inter[inter.mask] = numpy.nan
                    elif f=="coordinate_x":
                        inter=self.coords[:,0]
                    elif f=="coordinate_y":
                        inter=self.coords[:,1]
                    elif f=="coordinate_z":
                        raise RuntimeError("Z coord")
                else:
                    inter = inter[~inter.mask] #type:ignore
                dataL.append(numpy.array(inter,dtype=numpy.float64)) #type:ignore
            data:NPFloatArray=numpy.array(dataL).transpose() #type:ignore
            units=meshdata.get_unit(fields,with_brackets=True,as_string=True)
            descs=[fields[i]+units[i] for i in range(len(fields))]

            return cast(NPFloatArray,data),descs #type:ignore
        else:
            raise RuntimeError("Not implemented")
            data, mask, descs = self.mesh.get_values_at_zetas(self.coords, True) #type:ignore
            fullmask = numpy.transpose([mask] * len(data[0])) #type:ignore
            data:NPFloatArray = numpy.ma.masked_array(data, fullmask).transpose() #type:ignore
            return data,descs

    def output(self,step:int):
        mesh=self.mesh
        if (not mesh.is_mesh_distributed()) and self._mpi_rank>0:
            return
        if self.eigenvector is not None:
            if self.eigenvector >= len(self.mesh.get_problem()._last_eigenvectors): #type:ignore
                return  # No output hrere

        data,header=self.get_data_and_descs()
        fname=self.get_filename(step)
        params = {}
        for n in self.mesh.get_problem().get_global_parameter_names():
            params[n] = self.mesh.get_problem().get_global_parameter(n).value
        if isinstance(fname,list):
            for fn in fname:
                save_by_extension(fn, data, header=header,timeinfo=mesh.get_problem().get_current_time(as_float=True),params=params)
        else:            
            save_by_extension(fname,data,header=header,timeinfo=mesh.get_problem().get_current_time(as_float=True),params=params)
        self.clean_up()

    def delete_files_from_previous_simulation(self):
        try:
            step = 0
            while True:
                fn = self.get_filename(step)
                if isinstance(fn, list):
                    for f in fn:
                        os.remove(f)
                else:
                    os.remove(fn)
                step += 1
        except:
            pass





class _GridFileOutput(_BaseOutputter):
    def __init__(self,*fields:str,lower:List[Sequence[ExpressionOrNum]],upper:List[ExpressionOrNum],N:Optional[List[int]]=None,dx:Optional[List[ExpressionOrNum]],mesh:Optional["AnySpatialMesh"]=None,ftrunk:str="grid_out",in_subdir:bool=True,file_ext:Optional[Union[str,List[str]]]=None,hide_lagrangian:bool=True,hide_underscore:bool=True,eigenvector:Optional[int]=None,eigenmode:"MeshDataEigenModes"="abs"):
        super().__init__()
        if mesh is None:
            raise ValueError("Need to supply at least a mesh")
        self.fname_trunk=ftrunk
        self.in_subdir=in_subdir
        assert mesh is not None
        self.mesh=mesh
        self.problem=mesh.get_problem()
        self.file_ext=file_ext
        self.fields:List[str]=[*fields]
        self.use_tri_interpolator=True
        self.hide_lagrangian=hide_lagrangian
        self.hide_underscore = hide_underscore
        self.eigenvector=eigenvector
        self.eigenmode:"MeshDataEigenModes"=eigenmode
        self.lower=lower
        self.upper=upper
        self.dx=dx
        self.N=N
        pr=self.mesh.get_problem()
        meshdata = pr.get_cached_mesh_data(self.mesh, tesselate_tri=True, nondimensional=False)
        ss=meshdata.get_unit("spatial")
        self.coords_per_dir=[]
        if self.dx is not None:
            for ll,uu,step in zip(self.lower,self.upper,self.dx):
                ll_nd=float(ll/ss)
                uu_nd=float(uu/ss)
                step_nd=float(dx/ss)
                self.coords_per_dir.append(numpy.arange(ll_nd,uu_nd,step_nd))
        else:
            for ll,uu,N in zip(self.lower,self.upper,self.N):
                ll_nd=float(ll/ss)
                uu_nd=float(uu/ss)
                self.coords_per_dir.append(numpy.linspace(ll_nd,uu_nd,num=N,endpoint=True))
        if len(self.coords_per_dir)!=2:
            raise RuntimeError("Only works for 2d")
        self.coords_x,self.coords_y=numpy.meshgrid(numpy.array(self.coords_per_dir[0]),numpy.array(self.coords_per_dir[1]))
        self.coords_x,self.coords_y=self.coords_x.flatten(),self.coords_y.flatten()
        
        

        

    def init(self, eqtree:"EquationTree", continue_info:Optional[Dict[str,Any]]=None, rank:int=0):
        super().init(eqtree, continue_info, rank)
        if isinstance(self.mesh, str):
            self.mesh = self.problem.get_mesh(self.mesh)
        if self.in_subdir and rank == 0:
            Path(os.path.join(self.problem.get_output_directory()), self.fname_trunk).mkdir(parents=True, exist_ok=True)
        if self.file_ext is None:
            self.file_ext=self.problem.default_1d_file_extension


    def get_filename(self,step:int) -> Union[List[str] ,str]:
        assert self.file_ext is not None
        if isinstance(self.file_ext,list):
            res:List[str]=[]
            for e in self.file_ext:
                fname = self.fname_trunk + "_{:06d}".format(step) + "." + e
                if self.in_subdir:
                    fname = os.path.join(self.problem.get_output_directory(), self.fname_trunk, fname)
                else:
                    fname = os.path.join(self.problem.get_output_directory(), fname)
                res.append(fname)
            return res
        else:
            fname = self.fname_trunk + "_{:06d}".format(step) + "." + self.file_ext
            if self.in_subdir:
                fname = os.path.join(self.problem.get_output_directory(), self.fname_trunk, fname)
            else:
                fname = os.path.join(self.problem.get_output_directory(), fname)
            return fname

    def after_remeshing(self,eqtree:"EquationTree"):
        m=eqtree.get_mesh()
        assert not isinstance(m,ODEStorageMesh)
        self.mesh=m


    def get_data_and_descs(self)->Tuple[NPFloatArray,List[str]]:

        if self.use_tri_interpolator:
            meshdata = self.mesh.get_problem().get_cached_mesh_data(self.mesh, tesselate_tri=True, nondimensional=False,eigenmode=self.eigenmode,eigenvector=self.eigenvector)
            
            coordinates:NPFloatArray=meshdata.get_coordinates()
            import matplotlib.tri as tri

            triang = tri.Triangulation(coordinates[0,:], coordinates[1,:], meshdata.elem_indices)
            
            fields=meshdata.get_default_output_fields(rem_lagrangian=self.hide_lagrangian,rem_underscore=self.hide_underscore)
            dataL:List[NPFloatArray]=[]
            for f in fields:
                inter=tri.LinearTriInterpolator(triang, meshdata.get_data(f))                
                inter=inter(self.coords_x,self.coords_y) #type:ignore
                if True:                    
                    if f not in {"coordinate_x","coordinate_y","coordinate_z"}:                        
                        inter[inter.mask] = numpy.nan
                    elif f=="coordinate_x":
                        inter=self.coords_x
                    elif f=="coordinate_y":
                        inter=self.coords_y
                    elif f=="coordinate_z":
                        raise RuntimeError("Z coord")
                else:
                    inter = inter[~inter.mask] #type:ignore
                dataL.append(numpy.array(inter,dtype=numpy.float64)) #type:ignore
            data:NPFloatArray=numpy.array(dataL).transpose() #type:ignore
            units=meshdata.get_unit(fields,with_brackets=True,as_string=True)
            descs=[fields[i]+units[i] for i in range(len(fields))]

            return cast(NPFloatArray,data),descs #type:ignore
        else:
            raise RuntimeError("Not implemented")
            data, mask, descs = self.mesh.get_values_at_zetas(self.coords, True) #type:ignore
            fullmask = numpy.transpose([mask] * len(data[0])) #type:ignore
            data:NPFloatArray = numpy.ma.masked_array(data, fullmask).transpose() #type:ignore
            return data,descs

    def output(self,step:int):
        mesh=self.mesh
        if (not mesh.is_mesh_distributed()) and self._mpi_rank>0:
            return
        if self.eigenvector is not None:
            if self.eigenvector >= len(self.mesh.get_problem()._last_eigenvectors): #type:ignore
                return  # No output hrere

        data,header=self.get_data_and_descs()
        fname=self.get_filename(step)
        params = {}
        for n in self.mesh.get_problem().get_global_parameter_names():
            params[n] = self.mesh.get_problem().get_global_parameter(n).value
        if isinstance(fname,list):
            for fn in fname:
                save_by_extension(fn, data, header=header,timeinfo=mesh.get_problem().get_current_time(as_float=True),params=params)
        else:            
            save_by_extension(fname,data,header=header,timeinfo=mesh.get_problem().get_current_time(as_float=True),params=params)
        self.clean_up()

    def delete_files_from_previous_simulation(self):
        try:
            step = 0
            while True:
                fn = self.get_filename(step)
                if isinstance(fn, list):
                    for f in fn:
                        os.remove(f)
                else:
                    os.remove(fn)
                step += 1
        except:
            pass




#############################

class _BaseODEOutput(_BaseOutputter):
    def __init__(self):
        super().__init__()
        self._element:_pyoomph.BulkElementODE0d
        self._odemesh:ODEStorageMesh

    def init(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]]=None,rank:int=0):
        self._eqtree=eqtree
        self._mpi_rank=rank
        self._element=self._odemesh._get_ODE("ODE")
        pass

    def get_ODE_values(self)->Tuple[NPFloatArray,Dict[str,int]]:
        values,fieldinds=self._element.to_numpy()
        return values,fieldinds

    def output(self,step:int)->None:
        values,_=self.get_ODE_values()
        print("ODE:",values)

    def get_additional_values(self):
        return None,None

#######################

class _ODEFileOutput(_BaseODEOutput):
    def __init__(self,odemesh:ODEStorageMesh,eqtree:"EquationTree",fname:Optional[str]=None,first_column:List[Union[str,GlobalParameter]]=["time"],continue_info:Optional[Dict[str,Any]]=None,in_units:Dict[str,ExpressionOrNum]={},hide_underscore:bool=False):
        super().__init__()
        self.fname=fname
        self._odemesh=odemesh
        self.in_units=in_units
        self.hide_underscore=hide_underscore
        if self.hide_underscore:
            raise RecursionError("TODO: Hiding underscore")
        self.file:Any=None
        self._element=self._odemesh._get_ODE("ODE")
        self._eqtree=eqtree
        self.first_column=first_column
        self.continue_info=continue_info
        
    def change_output_directory(self,newdir:str,eqtree:"EquationTree"):
        oldname=self.fname
        self.fname = os.path.join(newdir, os.path.basename(self.fname))        
        if self.fname!=oldname:
            self.file.close()
            if os.path.exists(self.fname):  
                self.init(eqtree,{"TODO":"Fill further information here"},self._mpi_rank)
            else:
                self.init(eqtree,None,self._mpi_rank)

    def init(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]]=None,rank:int=0):
        super().init(eqtree,continue_info,rank)
        assert self.fname is not None
        if continue_info is None:
            self.file = open(self.fname, "w")
        else:            
            self.file=open(self.fname,"a")

        values, fieldinds = self.get_ODE_values()
        obs=self._eqtree.get_mesh().evaluate_all_observables()
        compiled_ifuncs = self._eqtree.get_mesh().list_integral_functions()
        descs=[""]*(len(values)+len(obs))
        for d,ind in fieldinds.items():
            descs[ind]=d
        for i,n in enumerate(obs.keys(),start=len(values)):
            descs[i]=n
        #TODO Scales
        _, indices = self._element.to_numpy()
        scales:List[ExpressionOrNum] = [1.0] * (len(indices)+len(obs))
        for k, i in indices.items():
            s = self._eqtree.get_equations().get_scaling(k)
            if not isinstance(s,Expression):
                s=Expression(s)
            s = self._eqtree.get_equations().get_current_code_generator().expand_placeholders(s,True)
            factor, unit, rest, success = _pyoomph.GiNaC_collect_units(s)
            scales[i] = float(factor)
            if k in self.in_units.keys():
                scales[i]*=float(unit/self.in_units[k])
                descs[i] = descs[i] + "[" + str(self.in_units[k]) + "]"
            else:
                try:
                    float(unit)
                except:
                    descs[i]=descs[i]+"["+unit_to_string(unit,estimate_prefix=False)+"]"
        for (i, n),v in zip(enumerate(obs.keys(), start=len(values)),obs.values()):
            if n in compiled_ifuncs:
                ieunit = self._eqtree.get_mesh().get_code_gen()._get_integral_function_unit_factor(n)
                _, unit, rest, _ = _pyoomph.GiNaC_collect_units(ieunit)
                try:
                    float(unit)
                except:
                    descs[i] = descs[i] + "[" + unit_to_string(unit,estimate_prefix=False) + "]"
                    scales[i]=1/unit
            else:
                if isinstance(v,_pyoomph.Expression):
                    factor, unit, rest, success = _pyoomph.GiNaC_collect_units(v)
                    scales[i] = unit
                    if factor.is_zero():
                        raise RuntimeError("TODO: Find a good way to detemine a unit here...")
                    try:
                        float(unit)
                    except:
                        descs[i] = descs[i] + "[" + unit_to_string(unit,estimate_prefix=False) + "]"
                else:
                    scales[i]=1.0

        self._scales = scales

        firstcols:List[str]=[]
        for fc in self.first_column:
            if fc=="time":
                tscale=self._eqtree.get_equations().get_scaling("temporal")
                if not isinstance(tscale,Expression):
                    tscale=Expression(tscale)
                factor, unit, rest, success = _pyoomph.GiNaC_collect_units(tscale)
                try:
                    float(unit)
                    tscale=""
                except:
                    tscale= "[" + unit_to_string(unit,estimate_prefix=False) + "]"
                firstcols.append("time"+tscale)
            elif isinstance(fc,GlobalParameter):
                firstcols.append(fc.get_name())
            else:
                raise RuntimeError(repr(fc))

        if continue_info is None:
            self.file.write("#"+"\t".join(firstcols+descs)+"\n")
        self.firsttime=True

    def output(self,step:int):
        values,_=self.get_ODE_values()
        obs=self._eqtree.get_mesh().evaluate_all_observables()
        
        _, indices = self._element.to_numpy()
        self._scales:List[ExpressionOrNum] = [1.0] * (len(indices)+len(obs))
        for k, i in indices.items():
            s = self._eqtree.get_equations().get_scaling(k)
            if not isinstance(s,Expression):
                s=Expression(s)
            s = self._eqtree.get_equations().get_current_code_generator().expand_placeholders(s,True)
            factor, unit, rest, success = _pyoomph.GiNaC_collect_units(s)
            self._scales[i] = float(factor)
            
   
        values[:]=values[:]*self._scales[:len(values)]  #type:ignore         
        obsv=numpy.array([v for v in obs.values()]) #type:ignore         
        

        obsv[:]=obsv[:]*self._scales[len(values):len(values)+len(obsv)]
        obsv=numpy.array(list(map(float,obsv))) #type:ignore 
        #try:
        #    obsv=obsv.astype(numpy.float)
        #except:
        #    pass
        values=numpy.concatenate([values,obsv]) #type:ignore 
        addv,_=self.get_additional_values() #type:ignore 
        if (addv is not None) and len(addv)>0:
            addstr="\t"+"\t".join(map(str,addv))
        else:
            addstr=""

        firstcols:List[str]=[]
        for fc in self.first_column:
            if fc=="time":
                firstcols.append(str(self.get_time()))
            elif isinstance(fc,GlobalParameter):
                firstcols.append(str(fc.value))
            else:
                raise RuntimeError(repr(fc))

        self.file.write("\t".join(firstcols+list(map(str,values)))+addstr+"\n")
        self.file.flush()
######################

class GenericOutput(BaseEquations):
    def __init__(self):
        super(GenericOutput, self).__init__()
        self._outputter:Dict["EquationTree",_BaseOutputter]={} #Map from eqtree node to an outputter object

    def after_remeshing(self,eqtree:"EquationTree"):
        for _,out in self._outputter.items():
            out.after_remeshing(eqtree)

    def _construct_outputter_for_eq_tree(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]],mpirank:int)->_BaseOutputter:
        raise NotImplementedError("Implement this")

    def _expand_filename(self,eqtree:"EquationTree",filename:Optional[str]=None,extension:str="",add_problem_outdir:bool=True):
        outdir = eqtree.get_mesh().get_problem().get_output_directory()
        if filename is None:
            if add_problem_outdir:
                fname = os.path.join(outdir, eqtree.get_full_path(eqtree, sep="__") + extension)
            else:
                fname=eqtree.get_full_path(eqtree, sep="__") + extension
            return fname
        else:
            if len(self._outputter) > 1:# or (not eqtree in self._outputter.keys()):

                raise RuntimeError("There are multiple outputs written to the same file "+filename)
            if add_problem_outdir:
                return os.path.join(outdir, filename)
            else:
                return filename

    def _init_output(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]]=None,rank:int=0):
        super()._init_output(eqtree,continue_info,rank)
        self._outputter[eqtree]=self._construct_outputter_for_eq_tree(eqtree,continue_info,rank)
        self._outputter[eqtree].problem = eqtree.get_mesh().get_problem()
        self._outputter[eqtree].init(eqtree,continue_info,rank)


    def _do_output(self, eqtree:"EquationTree", step:int,stage:str):
        self._outputter[eqtree].output(step)
        
    def change_output_directory(self, dir:str,eqtree:"EquationTree"):
        self._outputter[eqtree].change_output_directory(dir,eqtree)


class ODEFileOutput(GenericOutput):
    """
    ODEFileOutput writes the variables of all ODE unknowns at the current time to a text file.

    Args:
        filename: The name of the output file. Default is None, meaning that the output file will be named after the equation tree node.
        first_column: The value(s) to be written in the first column of the output file. Default is "time".
        in_units: A dictionary specifying the units of the variables to be written in the output file. Default is an empty dictionary, i.e. base SI units.
        hide_underscore: A flag indicating whether to hide variable names starting with an underscore in the output file. Default is False.
    """
    
    def __init__(self,filename:Optional[str]=None,first_column:Optional[Union[str,GlobalParameter,List[Union[str,GlobalParameter]]]]="time",in_units:Dict[str,ExpressionOrNum]={},hide_underscore:bool=False):
        super(ODEFileOutput, self).__init__()
        self.filename=filename        
        self.in_units=in_units
        self.hide_underscore=hide_underscore
        if not isinstance(first_column,list):
            if first_column is None:
                self.first_column=[]
            else:
                self.first_column=[first_column]
        else:
            self.first_column=first_column

    def _construct_outputter_for_eq_tree(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]],mpirank:int) -> _ODEFileOutput:
        fn=self._expand_filename(eqtree,self.filename,".txt")
        mesh=eqtree.get_mesh()
        assert isinstance(mesh,ODEStorageMesh)
        return _ODEFileOutput(mesh,eqtree,fname=fn,first_column=self.first_column,continue_info=continue_info,in_units=self.in_units,hide_underscore=self.hide_underscore)

    def _is_ode(self):
        return True


class TextFileOutput(GenericOutput):
    """
    A class for writing the degrees of freedom at the current time step to a text file. Will be invoked whenever Problem.output is called.

    Args:
        filetrunk (Optional[str]): The file trunk name. If not set, it will take the filename from the domain we added this equation to.
        filename (Optional[str]): Same as filetrunk, but for backwards compatibility.
        nondimensional (bool): Flag indicating whether the output should be nondimensional. Default is False.
        hide_underscore (bool): Flag indicating whether to hide variables starting with an underscore. Default is True.
        hide_lagrangian (bool): Flag indicating whether to hide Lagrangian coordinates. Default is True.
        eigenvector (Optional[int]): The eigenvector index. If set, we write the eigenvector at this index instead of the solution. Only writing output when the eigenvector at this index is calculated. Default is None.
        eigenmode (MeshDataEigenModes): The eigenmode type ("abs","real","imag"). Default is "abs".
        reverse_segment_if (Optional[Callable[[List[int], NPFloatArray], bool]]): A function to reverse individual segments of a segregated 1d line embedded in higher spaces. Default is None.
        sort_segments_by (Optional[Callable[[List[int], NPFloatArray], float]]): A function to sort such segments based on a condition. Otherwise, the ordering is more or less random. Default is None.
        discontinuous (bool): Flag indicating whether discontinuous output should be written. In that case, each node can be written multiple times, potential with different values. Default is False.
        add_eigen_to_mesh_positions (bool): When outputting an eigenvector on a moving mesh, do we want to add the original mesh coordinates to the eigensolution or not. Default is True.
    """



    def __init__(self,filetrunk:Optional[str]=None,filename:Optional[str]=None, nondimensional:bool=False,hide_underscore:bool=True,hide_lagrangian:bool=True,eigenvector:Optional[int]=None,eigenmode:"MeshDataEigenModes"="abs",reverse_segment_if:Optional[Callable[[List[int],NPFloatArray],bool]]=None,sort_segments_by:Optional[Callable[[List[int],NPFloatArray],float]]=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True,operator:Optional["MeshDataCacheOperatorBase"]=None,tesselate_tri:bool=True):
        super(TextFileOutput, self).__init__()
        if filetrunk is not None and filename is not None:
            raise RuntimeError("Please set either filename or filetrunk - both are the same, just for backwards compatibility")
        elif filetrunk is not None:
            self.filename=filetrunk
        else:
            self.filename=filename
        self.nondimensional=nondimensional
        self.hide_underscore=hide_underscore
        self.hide_lagrangian = hide_lagrangian
        self.eigenvector=eigenvector
        self.eigenmode:"MeshDataEigenModes"=eigenmode
        self.sort_segments_by=sort_segments_by
        self.reverse_segment_if=reverse_segment_if
        self.discontinuous=discontinuous
        self.add_eigen_to_mesh_positions=add_eigen_to_mesh_positions
        self.operator=operator
        self.tesselate_tri=tesselate_tri

    def _construct_outputter_for_eq_tree(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]],mpirank:int) -> _TextOutput:
        fn=self._expand_filename(eqtree,self.filename,"",add_problem_outdir=False)
        mesh=eqtree.get_mesh()
        assert not isinstance(mesh,ODEStorageMesh)
        return _TextOutput(mesh,ftrunk=fn,nondimensional=self.nondimensional,hide_underscore=self.hide_underscore,hide_lagrangian=self.hide_lagrangian,eigenvector=self.eigenvector,eigenmode=self.eigenmode,sort_segments_by=self.sort_segments_by,reverse_segment_if=self.reverse_segment_if,discontinuous=self.discontinuous,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions,operator=self.operator,tesselate_tri=self.tesselate_tri)

    def _is_ode(self):
        return False



class TextFileOutputAlongLine(GenericOutput):
    def __init__(self,filename:Optional[str]=None,coords:Optional[Union[NPFloatArray,List[Sequence[ExpressionOrNum]]]]=None,start:Optional[List[ExpressionOrNum]]=None,end:Optional[List[ExpressionOrNum]]=None,N:Optional[int]=None,isovalue:Optional[Tuple[str,ExpressionOrNum]]=None,file_ext:Optional[Union[str,List[str]]]=None,eigenvector:Optional[int]=None,eigenmode:"MeshDataEigenModes"="abs",NaN_outside:bool=False):
        super(TextFileOutputAlongLine, self).__init__()
        self.filename=filename
        self.file_ext=file_ext
        self.start=start
        self.end=end
        self.N=N
        self.coords=coords
        self.eigenvector=eigenvector
        self.eigenmode:"MeshDataEigenModes"=eigenmode
        self.isovalue=isovalue
        self.NaN_outside=NaN_outside

    def _construct_outputter_for_eq_tree(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]],mpirank:int) -> _OutputTxtAlongLine:
        fn=self._expand_filename(eqtree,self.filename,"",add_problem_outdir=False)
        mesh=eqtree.get_mesh()
        assert not isinstance(mesh,ODEStorageMesh)        
        return _OutputTxtAlongLine(mesh=mesh,ftrunk=fn,start=self.start,end=self.end,isovalue=self.isovalue,coords=self.coords,N=self.N,file_ext=self.file_ext,eigenvector=self.eigenvector,eigenmode=self.eigenmode,NaN_outside=self.NaN_outside)

    def _is_ode(self):
        return False



class GridFileOutput(GenericOutput):
    def __init__(self,lower:Union[NPFloatArray,List[Sequence[ExpressionOrNum]]],upper:Optional[List[ExpressionOrNum]],N:Union[int,List[int]]=None,dx:Union[ExpressionOrNum,List[ExpressionOrNum]]=None,filename:Optional[str]=None,file_ext:Optional[Union[str,List[str]]]=None,eigenvector:Optional[int]=None,eigenmode:"MeshDataEigenModes"="abs"):
        super(GridFileOutput, self).__init__()
        self.lower=lower
        self.upper=upper
        if len(self.lower)!=len(self.upper):
            raise RuntimeError("Start coordinate vector 'lower' must have the same length as the end coordinate vector 'upper'")        
        if N is None and dx is None:
            raise RuntimeError("Must either set dx or N")
        elif N is not None and dx is not None:
            raise RuntimeError("Cannot set N and dx simultaneously, just set one")
        elif N is not None:
            if not isinstance(N,(list,tuple)):
                N=[N]*len(self.lower)
            self.N=N
            self.dx=None
        else:
            if not isinstance(dx,(list,tuple)):
                dx=[dx]*len(self.lower)
            self.dx=dx
            self.N=None
        self.filename=filename
        self.file_ext=file_ext            
        self.N=N
        self.eigenvector=eigenvector
        self.eigenmode:"MeshDataEigenModes"=eigenmode


    def _construct_outputter_for_eq_tree(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]],mpirank:int) -> _OutputTxtAlongLine:
        fn=self._expand_filename(eqtree,self.filename,"",add_problem_outdir=False)
        mesh=eqtree.get_mesh()
        assert not isinstance(mesh,ODEStorageMesh)        
        return _GridFileOutput(mesh=mesh,ftrunk=fn,lower=self.lower,upper=self.upper,N=self.N,dx=self.dx,file_ext=self.file_ext,eigenvector=self.eigenvector,eigenmode=self.eigenmode)

    def _is_ode(self):
        return False




class _IntegralObservableOutput(_BaseOutputter):
    def __init__(self, mesh:"AnySpatialMesh", ftrunk:str,continue_info:Optional[Dict[str,Any]],file_ext:Optional[Union[str,List[str]]]=None,first_column:List[str]=["time"]):
        super(_IntegralObservableOutput, self).__init__()
        self._mesh=mesh
        self._filetrunk=ftrunk
        self._file_ext=file_ext
        self._units:Dict[str,Expression]={}
        self._iexprs:Optional[List[str]]=None
        self._files:Dict[str,Any]={}
        self._continue_info=continue_info
        self.first_column=first_column

    def after_remeshing(self,eqtree:"EquationTree"):
        self._mesh=eqtree.get_mesh()

    def _eval_all_integral_funcs(self)->Dict[str,Expression]:
        res:Dict[str,Expression]={}
        if self._iexprs is None:
            return res
        for n in self._iexprs:
            try:
                rs=self._mesh._evaluate_integral_function(n)
            except:
                print("IN INTEGRAL FUNCTION "+n)
                raise

            res[n]=rs
        return res


    def change_output_directory(self,newdir:str,eqtree:"EquationTree"):
        print("TODO: Change output path in IntegralObservables")
        

    def _eval_dependent_funcs(self,intres:Dict[str,Expression]) -> Dict[str, float]:
        import pyoomph.equations.generic
        deps=self._mesh.get_code_gen()._dependent_integral_funcs #type:ignore
        args:Dict[str,Expression]={k:v for k,v in intres.items()}
        res:Dict[str,Expression] = {k: v for k, v in intres.items()}
        args["time"]=self._mesh.get_problem().get_current_time(dimensional=True,as_float=False)
        remaining=set(deps.keys())
        while len(remaining)>0:
            torem:Set[str]=set()
            for r in remaining:
                #Check if we can evaluate
                l=deps[r]
                all_present=True

                if isinstance(l,pyoomph.equations.generic.DependentIntegralObservable):
                    reqargs=l.argnames
                    func_to_call=l.func
                else:
                    reqargs=inspect.signature(l).parameters
                    func_to_call=l
                arglist:List[ExpressionOrNum]=[]

                for a in reqargs:
                    if not a in args.keys():
                        all_present=False
                    else:
                        arglist.append(args[a])
                if all_present:
                    torem.add(r)
                    depres=func_to_call(*arglist)
                    if not isinstance(depres,Expression):
                        depres=Expression(depres)
                    args[r]=depres
                    res[r]=depres
            if len(torem)==0:
                raise RuntimeError("Cannot evaluate the dependent integral functions, probably due to unknown or circular arguments : "+str(remaining))
            remaining = remaining-torem


        for n,v in res.items():
            if not n in self._units.keys():
                if not isinstance(v,numpy.ndarray):
                    if v == 0 or (isinstance(v, _pyoomph.Expression) and v.is_zero()): #type:ignore  # TODO: Unit cannot be detemined that way!
                        if n in intres.keys():
                            ieunit=self._mesh.get_code_gen()._get_integral_function_unit_factor(n)
                            fact, unit, rest, reslt = _pyoomph.GiNaC_collect_units(ieunit)
                            self._units[n] = unit
                        else:
                            pass #Can't do anything right now
                    else:
                        if not isinstance(v,_pyoomph.Expression): #type:ignore
                            v=_pyoomph.Expression(v)
                        fact,unit,rest,reslt=_pyoomph.GiNaC_collect_units(v)
                        self._units[n]=unit
                else:
                    for i,direct,cmp in zip([0,1,2],["x","y","z"],v):
                        if cmp == 0 or (isinstance(cmp,_pyoomph.Expression) and cmp.is_zero()):  # TODO: Unit cannot be detemined that way!
                            if n in intres.keys():
                                ieunit = self._mesh.get_code_gen()._get_integral_function_unit_factor(n)
                                fact, unit, rest, reslt = _pyoomph.GiNaC_collect_units(ieunit)
                                self._units[n]=unit
                                self._units[n+"_"+direct] = unit
                            else:
                                pass  # Can't do anything right now
                        else:
                            fact, unit, rest, reslt = _pyoomph.GiNaC_collect_units(cmp)
                            self._units[n] = unit
                            self._units[n+"_"+direct] = unit


        #Second loop, try fo set the remainin units by calling things with the units

        for n,v in res.items():
            if not n in self._units.keys():
                arglist=[]
                if v == 0 or (isinstance(v, _pyoomph.Expression) and v.is_zero()): #type:ignore
                    l = deps[n]
                    problem=False
                    for a in inspect.signature(l).parameters:
                        if not a in self._units.keys():
                            #Cannot do anything here
                            problem=True
                            break
                        else:
                            arglist.append(self._units[a])
                    if not problem:
                        depres = l(*arglist)
                        if depres == 0 or (isinstance(depres, _pyoomph.Expression) and depres.is_zero()):
                            pass
                        else:
                            if not isinstance(depres,Expression):
                                depres=Expression(depres)
                            fact, unit, rest, reslt = _pyoomph.GiNaC_collect_units(depres)
                            self._units[n]=unit
                    else:
                        pass #TODO: Further ways to get the unit??


        nondim_res:Dict[str,float]={}
        for n,v in res.items():
            if n in self._mesh.get_code_gen()._dependent_integral_funcs_is_vector_helper.keys(): 
                continue
            if isinstance(v,numpy.ndarray):
                for i,direct,cmp in zip([0,1,2],["x","y","z"],v):
                    if i>=self._mesh.get_code_gen().get_nodal_dimension():
                        break
                    if not n in self._units.keys():
                        nondim_res[n+"_"+direct] = cmp
                    else:
                        vef = (cmp / self._units.get(n, 1)).evalf()
                        try:
                            nondim_res[n+"_"+direct] = float(vef)
                        except:
                            fact, unit, rest, reslt = _pyoomph.GiNaC_collect_units(vef)
                            vef = fact * unit * rest
                            nondim_res[n+"_"+direct] = float(vef)

            else:
                if not n in self._units.keys():
                    nondim_res[n] = float(v)
                else:
                    vef=(v / self._units.get(n, 1)).evalf()
                    try:
                        nondim_res[n] = float(vef)
                    except:
                        fact, unit, rest, reslt = _pyoomph.GiNaC_collect_units(vef)
                        vef=fact*unit*rest
                        nondim_res[n] = float(vef)


        return nondim_res

    def _eval_all(self) -> Dict[str, float]:
        intres=self._eval_all_integral_funcs()
        all=self._eval_dependent_funcs(intres)
        #self._eqtree.get_mesh().evaluate_all_observables()
        return all

    def output(self,step:int):
        fexts=self._file_ext
        assert fexts is not None
        if not isinstance(fexts,list):
            fexts=[fexts]
        for ext in fexts:
            if ext in ["mat","MAT"]:
                continue
            if ext not in self._files.keys():
                _filename=self._filetrunk+"."+ext
                if self._continue_info is None:
                    self._files[ext]=open(_filename,"wt")
                else:
                    self._files[ext] = open(_filename, "at")
                    #TODO: Trim here the output
                    #print(self._continue_info)
                    #raise RuntimeError("TODO")
        firsttime=False
        if self._iexprs is None:
            firsttime=True
            self._iexprs=self._mesh.list_integral_functions()
            all=self._eval_all()

            firstcols:List[str]=[]
            for fc in self.first_column:
                if fc == "time":
                    tscale = self._eqtree.get_equations().get_scaling("temporal")
                    if not isinstance(tscale,Expression):
                        tscale=Expression(tscale)
                    _, unit, _, _ = _pyoomph.GiNaC_collect_units(tscale)
                    try:
                        float(unit)
                        tscale = ""
                    except:
                        tscale = "[" + unit_to_string(unit, estimate_prefix=False) + "]"
                    firstcols.append("time" + tscale)
                elif isinstance(fc, _pyoomph.GiNaC_GlobalParam):
                    firstcols.append(fc.get_name())
                elif isinstance(fc,str) and (fc in self._mesh.get_problem().get_global_parameter_names()): #type:ignore
                    firstcols.append(fc)
                else:
                    raise RuntimeError(repr(fc))

            desc=firstcols
            for n,v in sorted(all.items()):
                if n[0]!="_":
                    entry=n
                    if self._units.get(n,1)!=1:
                        entry+="["+unit_to_string(self._units.get(n,1),estimate_prefix=False)+"]"
                    desc.append(entry)
            if self._continue_info is None:
                for f in self._files.values():
                    f.write("#"+"\t".join(desc)+"\n")
            self._descs=desc
        else:
            all = self._eval_all()

        line:List[str]=[]
        for fc in self.first_column:
            if fc=="time":
                line.append(str(self.get_time()))
            elif isinstance(fc,_pyoomph.GiNaC_GlobalParam):
                line.append(str(fc.value))
            elif isinstance(fc, str) and (fc in self._mesh.get_problem().get_global_parameter_names()): #type:ignore
                line.append(str(self._mesh.get_problem().get_global_parameter(fc).value))
            else:
                raise RuntimeError(repr(fc))

        #line=[self.get_time()]
        for n, v in sorted(all.items()):
            if n[0] != "_":
                line.append(str(v))
        for f in self._files.values():
            f.write("\t".join(map(str,line))+"\n")
            f.flush()
        for ext in fexts:
            if ext in ["mat","MAT"]:
                _filename=self._filetrunk+"."+ext
                mdict={}
                if os.path.exists(_filename) and not firsttime:
                    mdict=loadmat(_filename) #type:ignore
                for i,d in enumerate(self._descs):
                    if "[" in d:
                        d = d[0:d.find("[")]
                    if d in mdict:
                        #print(d)
                        olddata=mdict[d] #type:ignore
                        if len(olddata.shape)>1: #type:ignore
                            olddata=olddata[0,:] #type:ignore
                        else:
                            olddata=numpy.array(olddata[:]) #type:ignore
                        newdata=numpy.array(line[i]) #type:ignore
                        if len(newdata.shape)<len(olddata.shape):
                            newdata=numpy.array([newdata],dtype="float64") #type:ignore
                        #print(olddata,newdata)
                        mdict[d]=numpy.concatenate([olddata,newdata]) #type:ignore
                    else:
                        mdict[d]=numpy.array([line[i]],dtype="float64") #type:ignore
                savemat(_filename,mdict,appendmat=True)

    def init(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]]=None,rank:int=0):
        super().init(eqtree,continue_info,rank)
        if self._file_ext is None:
            self._file_ext=self.problem.default_1d_file_extension


class IntegralObservableOutput(GenericOutput):
    """
    Outputs all integral observables on this domain to a text file.

    Args:
        filename: The name of the output file (without extension). Default is None, meaning that the output file will be named after the domain.
        file_ext: The file extension. Default is None, meaning that the default file extension from the problem will be used.
        first_column: The value(s) to be written in the first column of the output file. Default is ``"time"``.        
    """
    def __init__(self, filename:Optional[str]=None, file_ext:Optional[Union[str,List[str]]]=None,first_column:List[str]=["time"]):
        super(IntegralObservableOutput, self).__init__()
        self.filename = filename
        self.file_ext = file_ext
        self.first_column=first_column

    def _construct_outputter_for_eq_tree(self, eqtree:"EquationTree", continue_info:Optional[Dict[str,Any]], mpirank:int) -> _IntegralObservableOutput:
        fn = self._expand_filename(eqtree, self.filename,  "_IntObsv")
        mesh=eqtree.get_mesh()
        assert not isinstance(mesh,ODEStorageMesh)
        return _IntegralObservableOutput(mesh=mesh, ftrunk=fn , file_ext=self.file_ext,continue_info=continue_info,first_column=self.first_column)

    def _is_ode(self):
        return None

#ODEObservableOutput=IntegralObservableOutput
