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
 
from ..typings import *
import numpy


from .generic import *
from .generic import _BaseNumpyOutput #type:ignore
import os
from pathlib import Path
import meshio #type:ignore

from ..meshes.meshdatacache import MeshDataEigenModes,MeshDataCacheOperatorBase,MeshDataCombineWithEigenfunction,MeshDataCartesianExtrusion,MeshDataRotationalExtrusion

# Hack, because the meshio version does not have a meshio._mesh.topological_dimension["wedge15"] set!
class Wedge15Cellblock(meshio.CellBlock):
	def __init__(self,cell_type,data,tags=None): #type:ignore
		super().__init__("wedge18",data,tags=tags) #type:ignore
		self.type = cell_type #type:ignore


import xml.etree.ElementTree as ET



def _convert_mesh_to_meshio(problem:"Problem",cache,eigenvector:Optional[Union[int,List[int]]]=None,eigenmode:MeshDataEigenModes="abs",tesselate_tri:bool=False,hide_lagrangian:bool=True,hide_underscore:bool=True):
	from .. import get_mpi_nproc #type:ignore




	
	#print("GEETING CACHE",self.operator,cache,cache.operator)
	#print("GOT CACHE",cache,cache.nodal_values)
	nodal_data=cache.nodal_values
	field_names=cache.nodal_field_inds
	elemtypes=cache.elem_types
	elemental_fields=cache.elemental_field_inds
	eleminds=cache.elem_indices
	numDL=cache.DL_data.shape[1]
	numD0 = cache.D0_data.shape[1] #type:ignore

	if "coordinate_x" in field_names.keys():
		x:NPFloatArray=nodal_data[:,field_names["coordinate_x"]] 
	else:
		x:NPFloatArray = numpy.zeros(len(nodal_data)) #type:ignore	
	if "coordinate_y" in field_names.keys():
		y:NPFloatArray = nodal_data[:, field_names["coordinate_y"]] #type:ignore
	else:
		y:NPFloatArray = 0*x
	if "coordinate_z" in field_names.keys():
		z:NPFloatArray = nodal_data[:, field_names["coordinate_z"]] #type:ignore
	else:
		z:NPFloatArray = 0*x
	field_data={}
	outfields=cache.get_default_output_fields(rem_lagrangian=hide_lagrangian,rem_underscore=hide_underscore)

	group_vector_fields=True
	rev_vector_fields:Dict[str,str]={}
	if group_vector_fields:
		vector_fields = cache.vector_fields			
		for a,_ in vector_fields.items():
			for c in ["_x","_y","_z"]:
				rev_vector_fields[a+c]=a
	vector_fields_written:Set[str]=set()

	for n in outfields:
		if n!="coordinate_x" and n!="coordinate_y" and n!="coordinate_z":
			if group_vector_fields and n in rev_vector_fields:
				vector_name=rev_vector_fields[n]
				if vector_name in vector_fields_written:
					continue
				data:List[NPFloatArray]=[]
				for k in "_x","_y","_z":
					if vector_name+k in outfields:
						#print("GETTING DATA",vector_name,k,cache.get_data(vector_name+k))
						data.append(cache.get_data(vector_name+k)) #type:ignore
					else:
						if len(data)>0:
							data.append(numpy.zeros(len(data[0]))) #type:ignore
				if len(data)>0:
					field_data[vector_name]=numpy.transpose(numpy.array(data)) #type:ignore
					vector_fields_written.add(vector_name)
				continue

			field_data[n]=cache.get_data(n)
			#print("MESHIO FIELDADTA",n,field_data[n])
			if field_data[n] is None:
				del field_data[n]

	
	if cache.discontinuous:
		for k,v in elemental_fields.items():
			if k in outfields:
				if v>=numDL:
					field_data[k]=cache.D0_data[:,v-numDL]
				else:
					field_data[k]=cache.DL_data[:,v] 

	points=numpy.transpose(numpy.array([x,y,z])) #type:ignore
	cells = []
	cell_data:Dict[str,List[NPFloatArray]] = {}


	present_elem_types,inds = numpy.unique(elemtypes,return_inverse=True) #type:ignore
	#print("ELEMTYPES",elemtypes)
	pointperm = numpy.array([0], dtype=int) #type:ignore
	lperm = numpy.array([0, 1], dtype=int) #type:ignore
	lperm3 = numpy.array([0, 2, 1], dtype=int) #type:ignore
	triperm = numpy.array([0, 1, 2], dtype=int) #type:ignore
	triC1TBperm = numpy.array([0, 1, 2], dtype=int) #type:ignore			
	tetraperm = numpy.array([0, 1, 2,3], dtype=int) #type:ignore
	wedgeperm=numpy.array([0,1,2,3,4,5],dtype=int) #type:ignore
	wedge18perm=numpy.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],dtype=int) #type:ignore
	wedge12perm=numpy.array([0,1,2,3,4,5,6,7,8,9,10,11],dtype=int) #type:ignore
	#tetra10perm = numpy.array([0, 1, 2, 3,4,5,6,7,8,9], dtype=int)
	tetra10perm = numpy.array([0, 1, 2, 3, 4, 7, 5, 6,  9,8], dtype=int) #type:ignore
	triperm6 = numpy.array([0, 1, 2, 3, 4, 5], dtype=int) #type:ignore
	triperm7 = numpy.array([0, 1, 2, 3, 4, 5,6], dtype=int) #type:ignore
	quadperm = numpy.array([0, 1, 3, 2], dtype=int) #type:ignore
	quad9perm = numpy.array([0, 2, 8, 6, 1, 5, 7, 3, 4], dtype=int) #type:ignore
	hexahedronperm = numpy.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=int) #type:ignore
	#hexahedronperm27 = numpy.array([0, 8, 1, 11, 24, 9, 3, 10,2,16,22,17,20,26,21,19,23,18,4,12,5,15,25,13,7,14,6], dtype=int)
	hexahedronperm27 = numpy.array([0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15, 12, 14, 10, 16, 4, 22, 13]) #type:ignore
	for et in present_elem_types:
		elinds=numpy.argwhere(elemtypes==et) #type:ignore
		#print(et,elinds)
		#rint((et,elemental_fields))
		if not cache.discontinuous:
			for k,v in elemental_fields.items():
				if k in outfields:
					if cell_data.get(k) is None:
						cell_data[k]=[]
					if v>=numDL:
						cell_data[k].append(cache.D0_data[elinds,v-numDL])
					else:
						cell_data[k].append(cache.DL_data[elinds,v,0]) #TODO: Slopes
		if et==0:
			cells.append(("vertex",eleminds[elinds, pointperm])) #type:ignore
		elif et==1: #line
			cells.append(("line", eleminds[elinds, lperm])) #type:ignore
		elif et==2: #line3
			cells.append(("line3", eleminds[elinds, lperm3])) #type:ignore
		elif et==3: #tri
			cells.append(("triangle", eleminds[elinds, triperm])) #type:ignore
		elif et==66: #tri C1TB (ignore the center node)
			cells.append(("triangle", eleminds[elinds, triC1TBperm])) #type:ignore					
		elif et==4: #tetra
			cells.append(("tetra", eleminds[elinds, tetraperm])) #type:ignore
		elif et == 9:  # tri6
			cells.append(("triangle6", eleminds[elinds, triperm6])) #type:ignore
		elif et == 99:  # tri6
			if "topological_dimension" not in dir(meshio._mesh) or "triangle7" in meshio._mesh.topological_dimension.keys(): #type:ignore
				cells.append(("triangle7", eleminds[elinds, triperm7])) #type:ignore
			else:
				cells.append(("triangle6", eleminds[elinds, triperm6])) #type:ignore
		elif et==10 or et==100: #tetra10 (or with bubbles, which are not possible)
			cells.append(("tetra10", eleminds[elinds, tetra10perm])) #type:ignore
		elif et==6: #quad
			cells.append(("quad", eleminds[elinds, quadperm])) #type:ignore
		elif et==8: #quad9
			cells.append(("quad9", eleminds[elinds,quad9perm])) #type:ignore
		elif et==11: #hexahedron
			cells.append(("hexahedron", eleminds[elinds,hexahedronperm])) #type:ignore
		elif et==14: #hexahedron27
			cells.append(("hexahedron27", eleminds[elinds,hexahedronperm27])) #type:ignore
		elif et==7: # wegde (only from rotational extrusion at the moment)
			cells.append(("wedge",eleminds[elinds,wedgeperm])) #type:ignore
		elif et==77: # wedge12 (only from rotational extrusion at the moment)
			if "topological_dimension" not in dir(meshio._mesh) or "wedge12" in meshio._mesh.topological_dimension.keys(): #type:ignore
				cells.append(("wedge12", eleminds[elinds, wedge12perm])) #type:ignore
			else:
				cells.append(("wedge", eleminds[elinds, wedgeperm])) #type:ignore

			#cells.append(Wedge15Cellblock("wedge15",numpy.asarray(eleminds[elinds[:,0],:])))
		else:
			raise RuntimeError("Unknown element type "+str(et))

	return meshio.Mesh(points, cells, point_data=field_data,cell_data=cell_data) #type:ignore


def pretty_xml(element:ET.Element, indent:str, newline:str, level:int=0):  
    if element is not None:  
        if (element.text is None) or element.text.isspace():  
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    temp = list(element)  
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  
            subelement.tail = newline + indent * (level + 1)
        else: 
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1) 


class _MeshFileOutput(_BaseNumpyOutput):
	def __init__(self,mesh:"AnySpatialMesh",ftrunk:str="output",in_subdir:bool=True,file_ext:str="vtu",tesselate_tri:bool=False,write_pvd:Optional[bool]=None,eigenvector:Optional[Union[int,List[int]]]=None,eigenmode:"MeshDataEigenModes"="abs",nondimensional:bool=False,hide_lagrangian:bool=True,hide_underscore:bool=True,history_index:int=0,operator:Optional["MeshDataCacheOperatorBase"]=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True):
		super().__init__(mesh)
		self.fname_trunk=ftrunk
		self.file_ext=file_ext
		self.in_subdir=in_subdir
		self.nondimensional=nondimensional
		self.tesselate_tri=tesselate_tri
		self.hide_lagrangian=hide_lagrangian
		self.hide_underscore = hide_underscore
		if write_pvd is None:
			write_pvd= file_ext=="vtu"
		if write_pvd==True:
			self.write_pvd_file:Optional[str]=os.path.join(self.mesh.get_problem().get_output_directory(),self.fname_trunk+".pvd")
		else:
			self.write_pvd_file=None
		self.eigenvector =eigenvector
		self.eigenmode:"MeshDataEigenModes" = eigenmode
		self.history_index=history_index
		self.active=True		
		self.operator=operator
		self.pvdcollection:ET.Element
		self.discontinuous=discontinuous
		self.add_eigen_to_mesh_positions=add_eigen_to_mesh_positions


	def init(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]]=None,rank:int=0):
		super().init(eqtree,continue_info,rank)
		self.mpi_rank=rank
		if isinstance(self.mesh,str):
			self.mesh=self.problem.get_mesh(self.mesh)
		if self.in_subdir:
				Path(os.path.join(self.problem.get_output_directory()),self.fname_trunk).mkdir(parents=True, exist_ok=True)
		if self.write_pvd_file:
			if (continue_info is None) or (not os.path.isfile(self.write_pvd_file)):
				self.pvddata = ET.Element("VTKFile")
				self.pvddata.set("type", "Collection")
				self.pvddata.set("version", "0.1")
				self.pvdtree = ET.ElementTree(self.pvddata)
				self.pvdcollection = ET.SubElement(self.pvddata, "Collection")
			else:
				self.pvdtree=ET.parse(self.write_pvd_file)
				self.pvddata=self.pvdtree.getroot()
				cll=self.pvddata.find("Collection")
				assert isinstance(cll,ET.Element)
				self.pvdcollection=cll


	def write_pvd(self,new_filename:str,all_files:Optional[List[str]]=None):
		assert self.write_pvd_file is not None
		if all_files is None:
			all_files=[new_filename]
		for i,f in enumerate(all_files):
			pvd_entry=ET.SubElement(self.pvdcollection,"DataSet")
			pvd_entry.set("timestep",str(self.mesh.get_problem().get_current_time(dimensional=not self.nondimensional, as_float=True)))
			pvd_entry.set("part",str(i))
			pvd_entry.set("file",f)
		pretty_xml(self.pvdtree.getroot(),"\t","\n")
		self.pvdtree.write(self.write_pvd_file)

	def clean_up(self):
		super().clean_up()


	def output(self,step:int):
		from .. import get_mpi_nproc #type:ignore

		mesh=self.mesh

		if self.active is False:
			return


		additional_eigenvectors:List[int]=[]
		evarg_for_cache=self.eigenvector
		if isinstance(self.eigenvector,(list,set,tuple)):
			for e in self.eigenvector:
				if e < len(self.problem._last_eigenvectors): #type:ignore
					additional_eigenvectors.append(e)
				if len(additional_eigenvectors)==0:
					return
			evarg_for_cache=additional_eigenvectors
		elif self.eigenvector is not None:
			if self.eigenvector>=len(self.problem._last_eigenvectors): #type:ignore
				return # No output hrere
			if self.eigenmode == "merge":
				evarg_for_cache = [self.eigenvector]
				additional_eigenvectors=[self.eigenvector]



		if (not mesh.is_mesh_distributed()) and self.mpi_rank>0:
			return

		#eleminds,elemtypes,D0_data,DL_data,elemental_fields,nodal_data,field_names=self.get_nodal_values(self.mesh,with_elem_indices=True,with_discontinuous=True,tesselate_tri=self.tesselate_tri,hide_fields=self.hide_fields,eigenvector=self.eigenvector,eigenvector_mode=self.eigenvector_mode,nondimensional=self.nondimensional)
		cache=self.get_cached_mesh_data(self.mesh,nondimensional=self.nondimensional,tesselate_tri=self.tesselate_tri,eigenvector=evarg_for_cache,eigenmode=self.eigenmode,history_index=self.history_index,operator=self.operator,discontinuous=self.discontinuous,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions)
		#print("GEETING CACHE",self.operator,cache,cache.operator)
		#print("GOT CACHE",cache,cache.nodal_values)
		nodal_data=cache.nodal_values
		field_names=cache.nodal_field_inds
		elemtypes=cache.elem_types
		elemental_fields=cache.elemental_field_inds
		eleminds=cache.elem_indices
		numDL=cache.DL_data.shape[1]
		numD0 = cache.D0_data.shape[1] #type:ignore

		if "coordinate_x" in field_names.keys():
			x:NPFloatArray=nodal_data[:,field_names["coordinate_x"]] 
		else:
			x:NPFloatArray = numpy.zeros(len(nodal_data)) #type:ignore	
		if "coordinate_y" in field_names.keys():
			y:NPFloatArray = nodal_data[:, field_names["coordinate_y"]] #type:ignore
		else:
			y:NPFloatArray = 0*x
		if "coordinate_z" in field_names.keys():
			z:NPFloatArray = nodal_data[:, field_names["coordinate_z"]] #type:ignore
		else:
			z:NPFloatArray = 0*x
		field_data={}
		outfields=cache.get_default_output_fields(rem_lagrangian=self.hide_lagrangian,rem_underscore=self.hide_underscore)

		group_vector_fields=True
		rev_vector_fields:Dict[str,str]={}
		if group_vector_fields:
			vector_fields = cache.vector_fields			
			for a,_ in vector_fields.items():
				for c in ["_x","_y","_z"]:
					rev_vector_fields[a+c]=a
		vector_fields_written:Set[str]=set()

		for n in outfields:
			if n!="coordinate_x" and n!="coordinate_y" and n!="coordinate_z":
				if group_vector_fields and n in rev_vector_fields:
					vector_name=rev_vector_fields[n]
					if vector_name in vector_fields_written:
						continue
					data:List[NPFloatArray]=[]
					for k in "_x","_y","_z":
						if vector_name+k in outfields:
							#print("GETTING DATA",vector_name,k,cache.get_data(vector_name+k))
							data.append(cache.get_data(vector_name+k)) #type:ignore
						else:
							if len(data)>0:
								data.append(numpy.zeros(len(data[0]))) #type:ignore
					if len(data)>0:
						field_data[vector_name]=numpy.transpose(numpy.array(data)) #type:ignore
						vector_fields_written.add(vector_name)
					continue

				field_data[n]=cache.get_data(n)
				#print("MESHIO FIELDADTA",n,field_data[n])
				if field_data[n] is None:
					del field_data[n]
				if additional_eigenvectors is not None:
					for eigenv in additional_eigenvectors:
						re_name="EIGEN_"+str(eigenv)+"_REAL_"+n
						im_name = "EIGEN_" + str(eigenv) + "_IMAG_"+n
						field_data[re_name] = cache.get_data(n,additional_eigenvector=eigenv,eigen_real_imag=0)
						field_data[im_name] = cache.get_data(n, additional_eigenvector=eigenv, eigen_real_imag=1)
		
		if cache.discontinuous:
			for k,v in elemental_fields.items():
				if k in outfields:
					if v>=numDL:
						field_data[k]=cache.D0_data[:,v-numDL]
					else:
						field_data[k]=cache.DL_data[:,v] 

		points=numpy.transpose(numpy.array([x,y,z])) #type:ignore
		cells = []
		cell_data:Dict[str,List[NPFloatArray]] = {}

		if self.tesselate_tri and self.mesh.get_dimension()>1:
			if self.mesh.get_dimension()==3:
				raise RuntimeError("TODO")
			cells = [("triangle", eleminds)]
			#print(eleminds)
			#print(len(cache.D0_data),len(eleminds))
			if not cache.discontinuous:
				for k, v in elemental_fields.items():
					if k in outfields:
						if cell_data.get(k) is None:
							cell_data[k] = []
						if v >= numDL:
							cell_data[k].append(cache.D0_data[:, v - numDL]) #type:ignore
						else:
							cell_data[k].append(cache.DL_data[:, v, 0])	#type:ignore #TODO: Slopes

		else:
			present_elem_types,inds = numpy.unique(elemtypes,return_inverse=True) #type:ignore
			#print("ELEMTYPES",elemtypes)
			pointperm = numpy.array([0], dtype=int) #type:ignore
			lperm = numpy.array([0, 1], dtype=int) #type:ignore
			lperm3 = numpy.array([0, 2, 1], dtype=int) #type:ignore
			triperm = numpy.array([0, 1, 2], dtype=int) #type:ignore
			triC1TBperm = numpy.array([0, 1, 2], dtype=int) #type:ignore			
			tetraperm = numpy.array([0, 1, 2,3], dtype=int) #type:ignore
			wedgeperm=numpy.array([0,1,2,3,4,5],dtype=int) #type:ignore
			wedge18perm=numpy.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],dtype=int) #type:ignore
			wedge12perm=numpy.array([0,1,2,3,4,5,6,7,8,9,10,11],dtype=int) #type:ignore
			#tetra10perm = numpy.array([0, 1, 2, 3,4,5,6,7,8,9], dtype=int)
			tetra10perm = numpy.array([0, 1, 2, 3, 4, 7, 5, 6,  9,8], dtype=int) #type:ignore
			triperm6 = numpy.array([0, 1, 2, 3, 4, 5], dtype=int) #type:ignore
			triperm7 = numpy.array([0, 1, 2, 3, 4, 5,6], dtype=int) #type:ignore
			quadperm = numpy.array([0, 1, 3, 2], dtype=int) #type:ignore
			quad9perm = numpy.array([0, 2, 8, 6, 1, 5, 7, 3, 4], dtype=int) #type:ignore
			hexahedronperm = numpy.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=int) #type:ignore
			#hexahedronperm27 = numpy.array([0, 8, 1, 11, 24, 9, 3, 10,2,16,22,17,20,26,21,19,23,18,4,12,5,15,25,13,7,14,6], dtype=int)
			hexahedronperm27 = numpy.array([0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15, 12, 14, 10, 16, 4, 22, 13]) #type:ignore
			for et in present_elem_types:
				elinds=numpy.argwhere(elemtypes==et) #type:ignore
				#print(et,elinds)
				#rint((et,elemental_fields))
				if not cache.discontinuous:
					for k,v in elemental_fields.items():
						if k in outfields:
							if cell_data.get(k) is None:
								cell_data[k]=[]
							if v>=numDL:
								cell_data[k].append(cache.D0_data[elinds,v-numDL])
							else:
								cell_data[k].append(cache.DL_data[elinds,v,0]) #TODO: Slopes
				if et==0:
					cells.append(("vertex",eleminds[elinds, pointperm])) #type:ignore
				elif et==1: #line
					cells.append(("line", eleminds[elinds, lperm])) #type:ignore
				elif et==2: #line3
					cells.append(("line3", eleminds[elinds, lperm3])) #type:ignore
				elif et==3: #tri
					cells.append(("triangle", eleminds[elinds, triperm])) #type:ignore
				elif et==66: #tri C1TB (ignore the center node)
					cells.append(("triangle", eleminds[elinds, triC1TBperm])) #type:ignore					
				elif et==4: #tetra
					cells.append(("tetra", eleminds[elinds, tetraperm])) #type:ignore
				elif et == 9:  # tri6
					cells.append(("triangle6", eleminds[elinds, triperm6])) #type:ignore
				elif et == 99:  # tri6
					if "topological_dimension" not in dir(meshio._mesh) or "triangle7" in meshio._mesh.topological_dimension.keys(): #type:ignore
						cells.append(("triangle7", eleminds[elinds, triperm7])) #type:ignore
					else:
						cells.append(("triangle6", eleminds[elinds, triperm6])) #type:ignore
				elif et==10 or et==100: #tetra10 (or with bubbles, which are not possible)
					cells.append(("tetra10", eleminds[elinds, tetra10perm])) #type:ignore
				elif et==6: #quad
					cells.append(("quad", eleminds[elinds, quadperm])) #type:ignore
				elif et==8: #quad9
					cells.append(("quad9", eleminds[elinds,quad9perm])) #type:ignore
				elif et==11: #hexahedron
					cells.append(("hexahedron", eleminds[elinds,hexahedronperm])) #type:ignore
				elif et==14: #hexahedron27
					cells.append(("hexahedron27", eleminds[elinds,hexahedronperm27])) #type:ignore
				elif et==7: # wegde (only from rotational extrusion at the moment)
					cells.append(("wedge",eleminds[elinds,wedgeperm])) #type:ignore
				elif et==77: # wedge12 (only from rotational extrusion at the moment)
					if "topological_dimension" not in dir(meshio._mesh) or "wedge12" in meshio._mesh.topological_dimension.keys(): #type:ignore
						cells.append(("wedge12", eleminds[elinds, wedge12perm])) #type:ignore
					else:
						cells.append(("wedge", eleminds[elinds, wedgeperm])) #type:ignore

					#cells.append(Wedge15Cellblock("wedge15",numpy.asarray(eleminds[elinds[:,0],:])))
				else:
					raise RuntimeError("Unknown element type "+str(et))

		meshout = meshio.Mesh(points, cells, point_data=field_data,cell_data=cell_data) #type:ignore
		allfiles =None
		if self.mesh.is_mesh_distributed():
			fname = self.fname_trunk + "_{:06d}_{:d}".format(step,self.mpi_rank) + "." + self.file_ext
			allfiles=[self.fname_trunk + "_{:06d}_{:d}".format(step,i) + "." + self.file_ext for i in range(get_mpi_nproc())]
		else:
			fname = self.fname_trunk + "_{:06d}".format(step) + "." + self.file_ext
		if self.in_subdir:
			rel_filename = os.path.join(self.fname_trunk, fname)
			fname = os.path.join(self.problem.get_output_directory(), self.fname_trunk, fname)
			if allfiles is not None:
				for i,f in enumerate(allfiles):
					allfiles[i] = os.path.join(self.fname_trunk, f)
		else:
			rel_filename = fname
			fname = os.path.join(self.problem.get_output_directory(), fname)

		#print(mesh)
		meshio.write(fname, meshout) #type:ignore

		if self.write_pvd_file and self.mpi_rank==0:
			self.write_pvd(rel_filename,allfiles)

		self.clean_up()





class MeshFileOutput(GenericOutput):    

	"""
    A class for writing the solution at the current time step to a mesh file. Will be invoked whenever Problem.output is called.

    Args:
        filetrunk (Optional[str]): The file trunk name. If not set, it will take the filename from the domain we added this equation to.
        tesselate_tri (bool): Flag indicating whether the output should be split into first order triangles/tetrahedrons. Default is False.
		file_ext (str): The file extension. Default is "vtu".
        eigenvector (Optional[int]): The eigenvector index. If set, we write the eigenvector at this index instead of the solution. Only writing output when the eigenvector at this index is calculated. Default is None.
        eigenmode (MeshDataEigenModes): The eigenmode type ("abs","real","imag"). Default is "abs".
        nondimensional (bool): Flag indicating whether the output should be nondimensional. Default is False.
        hide_underscore (bool): Flag indicating whether to hide variables starting with an underscore. Default is True.
        hide_lagrangian (bool): Flag indicating whether to hide Lagrangian coordinates. Default is True.
		history_index (int): The history index to output. Default is 0.
		operator (Optional[MeshDataCacheOperatorBase]): An operator to apply to the mesh data before outputting, see e.g. MeshDataCombineWithEigenfunction or MeshDataRotationalExtrusion from pyoomph.meshes.meshdatacache. Default is None.
        discontinuous (bool): Flag indicating whether discontinuous output should be written. In that case, each node can be written multiple times, potential with different values. Default is False.
        add_eigen_to_mesh_positions (bool): When outputting an eigenvector on a moving mesh, do we want to add the original mesh coordinates to the eigensolution or not. Default is True.
    """
    
	OpCombineWithEigenfunction=MeshDataCombineWithEigenfunction
	OpCartesianExtrusion=MeshDataCartesianExtrusion
	OpRotationalExtrusion=MeshDataRotationalExtrusion

	def __init__(self,filetrunk:Optional[str]=None,tesselate_tri:bool=False,file_ext:str="vtu",eigenvector:Optional[Union[int,List[int]]]=None,eigenmode:"MeshDataEigenModes"="abs",nondimensional:bool=False,hide_lagrangian:bool=True,hide_underscore:bool=True,history_index:int=0,operator:Optional["MeshDataCacheOperatorBase"]=None,discontinuous:bool=False,add_eigen_to_mesh_positions:bool=True):
		super(MeshFileOutput, self).__init__()
		self.filetrunk=filetrunk
		self.tesselate_tri=tesselate_tri
		self.fileext=file_ext
		self.eigenvector=eigenvector
		self.eigenmode:"MeshDataEigenModes"=eigenmode
		self.nondimensional=nondimensional
		self.hide_lagrangian=hide_lagrangian
		self.hide_underscore=hide_underscore
		self.history_index=history_index
		self.active=True
		self._my_outputter:List[_MeshFileOutput]=[]		
		self.operator=operator
		self.discontinuous=discontinuous
		self.add_eigen_to_mesh_positions=add_eigen_to_mesh_positions


	def _construct_outputter_for_eq_tree(self,eqtree:"EquationTree",continue_info:Optional[Dict[str,Any]],mpirank:int) -> _MeshFileOutput:
		fn=self._expand_filename(eqtree,self.filetrunk,"",add_problem_outdir=False)
		mesh=eqtree.get_mesh()
		assert not isinstance(mesh,ODEStorageMesh)
		res=_MeshFileOutput(mesh=mesh,ftrunk=fn,file_ext=self.fileext,in_subdir=True,tesselate_tri=self.tesselate_tri,eigenvector=self.eigenvector,eigenmode=self.eigenmode,nondimensional=self.nondimensional,hide_lagrangian=self.hide_lagrangian,hide_underscore=self.hide_underscore,history_index=self.history_index,operator=self.operator,discontinuous=self.discontinuous,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions)
		self._my_outputter.append(res)
		res.active=self.active
		return res

	def deactivate(self):
		for o in self._my_outputter:
			o.active=False
		self.active=False



	def _is_ode(self) -> Optional[bool]:
		return False

	