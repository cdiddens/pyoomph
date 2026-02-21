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
 

import precice

from ..expressions import *
from ..generic.codegen import EquationTree, ODEStorageMesh, BaseEquations, Equations, ODEEquations
from ..generic.problem import Problem

from scipy import spatial
import gc

# This is just a helper class which collects a few methods
class _PyoomphPreciceAdapater:
    def __init__(self) -> None:
        self._initialized:bool=False
        
    def initialize_problem(self,problem:Problem):        
        problem._precice_interface=precice.Participant(problem.precice_participant,problem.precice_config_file,0,1)
        problem._equation_system._before_precice_initialise()
        problem._precice_interface.initialize()
        self._initialized=True

    def coupled_run(self,problem:Problem,maxstep:Optional[float]=None,temporal_error:Optional[float]=None,output_initially:bool=True,fast_dof_backup:bool=False):
        problem._activate_solver_callback()
        
        if output_initially:
            problem.output()
        while problem._precice_interface.is_coupling_ongoing():
            if problem._precice_interface.requires_writing_checkpoint():
                if fast_dof_backup:
                    raise RuntimeError("Fast dof backup not implemented yet")
                else:
                    problem.save_state(problem.get_output_directory("precice_checkpoint.dump"))
                
            precice_dt = problem._precice_interface.get_max_time_step_size()
            dt=precice_dt
            if maxstep:
                dt = min(precice_dt, maxstep)

            problem._equation_system._before_precice_solve(dt)
         
            if temporal_error is None:
                problem.unsteady_newton_solve(dt,True)
            else:
                current_time=problem.get_current_time(as_float=True,dimensional=False)
                problem.adaptive_unsteady_newton_solve(dt,temporal_error,True)
                dt=problem.get_current_time(as_float=True,dimensional=False)-current_time
            
            problem._equation_system._after_precice_solve(dt)
                                    
            problem._precice_interface.advance(dt)
                                    
            if problem._precice_interface.requires_reading_checkpoint():
                problem.load_state(problem.get_output_directory("precice_checkpoint.dump"))
                gc.collect()
                gc.collect()
                gc.collect() # Just to be sure
            else:                    
                if problem._precice_interface.is_time_window_complete():
                    problem.actions_after_newton_solve()
                    problem.output()
            

    def generate_precice_config_file(self,problem:Problem):
        import xml.etree.ElementTree as ET
        root=ET.Element("precice-configuration")        
        
        data_kwargs_default={"waveform-degree":"1"}       
        mesh_kwargs_default={}
        mapping_kwargs_default={"mode":"nearest-neighbor","constraint":"consistent"}
        
        mesh_read_data={}
        mesh_write_data={}
        precice_read_data={}
        precice_write_data={}
        provided_meshes=set()
        received_meshes=set()
        pyoomph_mesh_name_to_provide_name={}
        def recursive_scan_data(eqtree:EquationTree):
            if eqtree._equations is not None:
                read_datas=eqtree._equations.get_equation_of_type(PreciceReadData,always_as_list=True)
                for read_data in read_datas:                                 
                    read_data=cast(PreciceReadData,read_data)
                    meshname=eqtree.get_full_path().lstrip("/")
                    for n,e in read_data.entries.items():                        
                        data_kwargs=data_kwargs_default.copy()
                        dataentry=ET.SubElement(root,"data"+":"+("scalar" if read_data.vector_dim is None else "vector"),name=e,**data_kwargs)
                        if meshname not in mesh_read_data:
                            mesh_read_data[meshname]=[]                        
                        mesh_read_data[meshname].append(e)                        
                        
                
                write_datas=eqtree._equations.get_equation_of_type(PreciceWriteData,always_as_list=True)
                for write_data in write_datas:                                 
                    write_data=cast(PreciceWriteData,write_data)
                    meshname=eqtree.get_full_path().lstrip("/")
                    for n,e in write_data.entries.items():    
                        data_kwargs=data_kwargs_default.copy()                    
                        dataentry=ET.SubElement(root,"data"+":"+("scalar" if write_data.vector_dim is None else "vector"),name=n,**data_kwargs)
                        if meshname not in mesh_write_data:
                            mesh_write_data[meshname]=[]
                        mesh_write_data[meshname].append(n)
                        
            for path,subeqs in eqtree._children.items():
                recursive_scan_data(subeqs)    
                        
        def recursive_scan_meshes(eqtree:EquationTree):
            if eqtree._equations is not None:
                mesh_provides=eqtree._equations.get_equation_of_type(PreciceProvideMesh,always_as_list=True)
                for mesh_provide in mesh_provides:                    
                    mesh_provide=cast(PreciceProvideMesh,mesh_provide)
                    meshname=eqtree.get_full_path().lstrip("/")
                    if eqtree._equations.get_combined_equations()._is_ode() is True:
                        meshdim=2 # Must be 2 for ODEs
                    else:
                        mesh=problem.get_mesh(meshname)
                        n=next(mesh.nodes())
                        meshdim=n.ndim()
                        if meshdim<2:
                            meshdim=2
                    mesh_kwargs=mesh_kwargs_default.copy()
                    meshentry=ET.SubElement(root,"mesh",name=mesh_provide.name,dimensions=str(meshdim), **mesh_kwargs)
                    use_data=[]
                    if meshname in mesh_read_data.keys():                        
                        use_data+=mesh_read_data[meshname]
                        precice_read_data[mesh_provide.name]=mesh_read_data[meshname].copy()
                    if meshname in mesh_write_data.keys():
                        use_data+=mesh_write_data[meshname]                                 
                        precice_write_data[mesh_provide.name]=mesh_write_data[meshname].copy()           
                    if len(use_data)==0:
                        raise RuntimeError("Provided Mesh "+meshname+" (preCICE name '"+str(mesh_provide.name)+"') has no data to read or write")
                    else:
                        use_data=set(use_data)
                        for ud in use_data:
                            ET.SubElement(meshentry,"use-data",name=ud)
                    provided_meshes.add(mesh_provide.name)
                    if meshname in pyoomph_mesh_name_to_provide_name.keys():
                        raise RuntimeError("Mesh "+meshname+" has multiple PreciceProvideMesh entries")
                    pyoomph_mesh_name_to_provide_name[meshname]=mesh_provide.name
                    
                mesh_receives=eqtree._equations.get_equation_of_type(PreciceReceiveMesh,always_as_list=True)
                for mesh_receive in mesh_receives:                    
                    mesh_receive=cast(PreciceProvideMesh,mesh_receive)
                    meshname=eqtree.get_full_path().lstrip("/")
                    if eqtree._equations.get_combined_equations()._is_ode() is True:
                        meshdim=2 # Must be 2 for ODEs
                    else:
                        mesh=problem.get_mesh(meshname)
                        n=next(mesh.nodes())
                        meshdim=n.ndim()
                        if meshdim<2:
                            meshdim=2
                    mesh_kwargs=mesh_kwargs_default.copy()
                    meshentry=ET.SubElement(root,"mesh",name=mesh_receive.name,dimensions=str(meshdim), **mesh_kwargs)
                    use_data=[]
                    if meshname in mesh_read_data.keys():                        
                        use_data+=mesh_read_data[meshname]
                    if meshname in mesh_write_data.keys():
                        use_data+=mesh_write_data[meshname]                                            
                    if len(use_data)==0:
                        raise RuntimeError("Received Mesh "+meshname+" (preCICE name '"+str(mesh_receive.name)+"') has no data to read or write")
                    else:
                        use_data=set(use_data)
                        for ud in use_data:
                            ET.SubElement(meshentry,"use-data",name=ud)
                    received_meshes.add((mesh_receive.name,mesh_receive.from_participant,meshname))
                    
            for path,subeqs in eqtree._children.items():
                recursive_scan_meshes(subeqs)
        recursive_scan_data(problem._equation_system)
        recursive_scan_meshes(problem._equation_system)
        
        if problem.precice_participant is None or problem.precice_participant=="":
            raise RuntimeError("Please set a preCICE participant name by the Problem property precice_participant")
        
        participant=ET.SubElement(root,"participant",name=problem.precice_participant)
        for meshname in provided_meshes:
            ET.SubElement(participant,"provide-mesh",name=meshname)
        for entry in received_meshes:
            ET.SubElement(participant,"receive-mesh",name=entry[0],**{"from":entry[1]})
        for meshname in provided_meshes:
            for data in precice_write_data.get(meshname,[]):
                ET.SubElement(participant,"write-data",name=data,mesh=meshname)
            for data in precice_read_data.get(meshname,[]):
                ET.SubElement(participant,"read-data",name=data,mesh=meshname)
        for entry in received_meshes:
            if entry[2] not in pyoomph_mesh_name_to_provide_name.keys():
                print("WARNING: Received mesh "+str(entry[0])+" is not provided by this participant")
                #raise RuntimeError("Received mesh "+str(entry[0])+" is not provided by this participant")
            else:
                # Mapping
                mapping_kwargs=mapping_kwargs_default.copy()
                mapping_kwargs["from"]=entry[0]
                mapping_kwargs["to"]=pyoomph_mesh_name_to_provide_name[entry[2]]
                mode=mapping_kwargs.pop("mode","nearest-neighbor")
                ET.SubElement(participant,"mapping:"+mode,direction="read",**mapping_kwargs)
                
            
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(problem.get_output_directory("generated_precice_config.xml"))


class PreciceProvideMesh(BaseEquations):
    """This class exports the domain or interface where it is added to to preCICE. 
    Afterwards, it can be used in combination with :py:class:`~pyoomph.solvers.precice_adapter.PreciceWriteData` and :py:class:`~pyoomph.solvers.precice_adapter.PreciceReadData`.
    
    Args:
        precice_mesh_name: The name of the mesh in the preCICE config file
        use_lagrangian_coords: If True, the Lagrangian coordinates are used to map the nodes to preCICE, otherwise the Eulerian coordinates
    """
    def __init__(self,precice_mesh_name:str,use_lagrangian_coords=True):
        super().__init__()
        self.name=precice_mesh_name
        self.use_lagrangian_coords=use_lagrangian_coords

    def before_precice_initialise(self,eqtree:"EquationTree"):
        mesh=eqtree.get_mesh()
        pr=mesh.get_problem()
        interface=pr._precice_interface
                                        
        xml_dimension = interface.get_mesh_dimensions(self.name)
        if isinstance(mesh,ODEStorageMesh):
            mesh._precice_node_to_vertex_id=None # No mapping needed for ODEs
            grid = numpy.zeros([1, xml_dimension])          
        else:
            my_nodes=[n for n in mesh.nodes()]
            mesh._precice_node_to_vertex_id={n:i for i,n in enumerate(my_nodes)}    
            
            grid = numpy.zeros([len(my_nodes), xml_dimension])
            if len(my_nodes)==0:
                raise ValueError("No nodes in mesh")
            mesh_dim=my_nodes[0].ndim()
            if xml_dimension<mesh_dim:
                raise RuntimeError("Mesh dimension mismatch: "+str(xml_dimension)+" in preCICE config vs. "+str(mesh_dim))
            iterate_dim=min(xml_dimension,my_nodes[0].ndim())
            if self.use_lagrangian_coords:
                for i,n in enumerate(my_nodes):
                    for j in range(iterate_dim):
                        grid[i,j] = n.x_lagr(j)
            else:
                for i,n in enumerate(my_nodes):
                    for j in range(iterate_dim):
                        grid[i,j] = n.x(j)
            
        mesh._precice_vertex_ids=interface.set_mesh_vertices(self.name, grid)
        if not isinstance(mesh._precice_vertex_ids,numpy.ndarray):
            mesh._precice_vertex_ids=numpy.array(mesh._precice_vertex_ids)
            
        if interface.requires_mesh_connectivity_for(self.name):
            for e in mesh.elements():
                all_nodes_have_vertex_id=True
                for ni in range(e.nnode()):
                    n=e.node_pt(ni)
                    if n not in mesh._precice_node_to_vertex_id:
                        all_nodes_have_vertex_id=False
                        break
                if not all_nodes_have_vertex_id:
                    raise RuntimeError("Not all nodes in element have vertex id...")
                typus=e.get_meshio_type_index()
                
                if typus==2: # LineC2
                    interface.set_mesh_edge(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)])
                    interface.set_mesh_edge(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)])                    
                elif typus==1: # LineC1
                    interface.set_mesh_edge(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)])
                elif typus==0: # Points have no edges
                    pass 
                elif typus==3: # TriC1
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)])
                elif typus==9: # TriC2
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(5)])                    
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                    
                    # Strange order, no matter how I permute the nodes, it is always backface in preCICE vtu out
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(2)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(1)])                                        
                elif typus==99: # TriC2TB
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                                            
                    # Strange order, no matter how I permute the nodes, it is always backface in preCICE vtu out
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(6)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                    
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                                            
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(6)])                                                                                
                elif typus==66: # TriC1TB
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                                        
                    interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                                                            
                    
                    
                elif typus==8: # QuadC2
                    if xml_dimension==3:
                        def _add_quad(i1,i2,i3,i4):
                            interface.set_mesh_quad(self.name,mesh._precice_node_to_vertex_id[e.node_pt(i1)],mesh._precice_node_to_vertex_id[e.node_pt(i2)],mesh._precice_node_to_vertex_id[e.node_pt(i3)],mesh._precice_node_to_vertex_id[e.node_pt(i4)])
                        _add_quad(0,1,3,4)
                        _add_quad(1,2,4,5)
                        _add_quad(3,4,6,7)
                        _add_quad(4,5,7,8)
                        
                    else:
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(3)])
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(4)])                        
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(5)])                                                
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(5)])                                                                        
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(6)])
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(3)])
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(8)])
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(5)],mesh._precice_node_to_vertex_id[e.node_pt(4)])
                elif typus==6: # QuadC1
                    if xml_dimension==3:
                        interface.set_mesh_quad(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                        
                    else:
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)])                                        
                        interface.set_mesh_triangle(self.name,mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(3)])                                                            
                #elif typus==11: # BrickC1
                    #interface.set_mesh_tetrahedron(self.name,mesh._precice_node_to_vertex_id[e.node_pt(0)],mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(4)])
                    #interface.set_mesh_tetrahedron(self.name,mesh._precice_node_to_vertex_id[e.node_pt(4)],mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(2)])
                    #interface.set_mesh_tetrahedron(self.name,mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(7)],mesh._precice_node_to_vertex_id[e.node_pt(6)],mesh._precice_node_to_vertex_id[e.node_pt(4)])
                    #interface.set_mesh_tetrahedron(self.name,mesh._precice_node_to_vertex_id[e.node_pt(1)],mesh._precice_node_to_vertex_id[e.node_pt(2)],mesh._precice_node_to_vertex_id[e.node_pt(3)],mesh._precice_node_to_vertex_id[e.node_pt(4)])
                else:
                    raise RuntimeError("Mesh elements are only supported for points, lines, triangle and quads, not yet for e.g. 3d elements. Got type index "+str(typus))
                
                                            

    def after_remeshing(self, eqtree: EquationTree):
        
        if not get_pyoomph_precice_adapter()._initialized:
            return # Just skip it here for the first time. In can happen e.g. in --runmode c
        if self.get_combined_equations()._is_ode():
            return # All fine for ODEs
        mesh=eqtree.get_mesh()
        pr=mesh.get_problem()
        interface=pr._precice_interface
        if not hasattr(mesh,"_precice_vertex_ids") or not hasattr(mesh,"_precice_node_to_vertex_id"):
            raise ValueError("Something strange. Did you remesh? This does not work with preCICE")
        my_nodes=[n for n in mesh.nodes()]
        if len(my_nodes)!=len(mesh._precice_node_to_vertex_id):
            raise ValueError("Number of nodes mismatch: "+str(len(my_nodes))+" vs. "+str(len(mesh._precice_node_to_vertex_id))+". Did you remesh? This does not work with preCICE")
        for n in my_nodes:
            if n not in mesh._precice_node_to_vertex_id:
                raise ValueError("Node not in preCICE mesh. Did you remesh? This does not work with preCICE")
        


class PreciceReceiveMesh(BaseEquations):
    """
    This class is not necessary for running, but helps to automatically generate a preCICE config file.
    Add it at an interface or domain so that the automatic generation of the preCICE config file includes the mesh as <receive-mesh> tag.
    """
    def __init__(self,name:str,from_participant:str):
        super().__init__()
        self.name=name
        self.from_participant=from_participant
            
            

class PreciceWriteData(BaseEquations):
    """
    Must be combined with :py:class:`~pyoomph.solvers.precice_adapter.PreciceProvideMesh` on the same domain or interface.
    
    Writes data to preCICE. Data can be an arbitrary scalar or vectorial expression. In the latter case, the vector_dim parameter must be set.

    Args:
        scaling: Optional scaling factor (e.g. units) for the data
        by_projection: If True, the data is first projected to nodal data and then written to preCICE. This is useful if the data is not available at the nodes.
        projection_space: If by_projection is True, the space to which the data is projected. If None, the coordinate space of the mesh is used.
        vector_dim: The dimension of the vector field. If None, a scalar field is assumed.
        kwargs: The data to be written. The keys are the names of the data in preCICE, the values are the expressions.
    """
    def __init__(self,scaling:ExpressionNumOrNone=None,by_projection:bool=False,projection_space:Optional[FiniteElementSpaceEnum]=None,vector_dim:Optional[int]=None,**kwargs:ExpressionOrNum):
        super().__init__()
        self.entries=kwargs.copy()
        self.by_projection=by_projection
        self.scaling=scaling
        self.projection_space=projection_space
        self.vector_dim=vector_dim
        
    def _sanitize_name(self,name:str):
        return "_precice_write_proj_"+name.replace(" ","_").replace("-","_").replace("/","_")

    def define_scalar_field(self,name:str,space:FiniteElementSpaceEnum,scale:Optional[ExpressionNumOrNone]=None,testscale:Optional[ExpressionNumOrNone]=None):
        # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
        return Equations.define_scalar_field(self,name,space,scale=scale,testscale=testscale)

    def define_fields(self):
        if self.get_combined_equations()._is_ode():
            self.by_projection=True
        if self.by_projection:
            if self.get_combined_equations()._is_ode():
                space="D0"            
            elif not self.projection_space:
                space=self.get_current_code_generator()._coordinate_space
            else:
                space=self.projection_space
            for name in self.entries.keys():
                # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
                testscale=None
                if self.scaling is not None:
                    testscale=1/self.scaling
                if self.vector_dim is None:
                    Equations.define_scalar_field(self,self._sanitize_name(name),space,scale=self.scaling,testscale=testscale)
                else:
                    Equations.define_vector_field(self,self._sanitize_name(name),space,scale=self.scaling,dim=self.vector_dim,testscale=testscale)                    

    def define_residuals(self):
        if self.get_combined_equations()._is_ode():
            self.by_projection=True
        if self.by_projection:
            for name,expr in self.entries.items():
                vname=self._sanitize_name(name)
                v,vtest=var_and_test(vname)
                self.add_weak(v-expr,vtest)
        else:
            for n,v in self.entries.items():
                self.add_local_function("_precice_write_"+n,v)
    

    def _do_write_data(self,precice_dt:float):
        mesh=self.get_mesh()
        if not hasattr(mesh,"_precice_vertex_ids"):
            raise ValueError("PreciceProvideMesh not set, please add it before PreciceWriteData")
        interface=mesh.get_problem()._precice_interface
        provider=self.get_combined_equations().get_equation_of_type(PreciceProvideMesh,always_as_list=True)
        if len(provider)!=1:
            raise ValueError("PreciceProvideMesh not set or set multiple times on this domain, please add a single one")
        provider=provider[0]
        assert isinstance(provider,PreciceProvideMesh)
        vertex_ids=mesh._precice_vertex_ids
                
        if self.by_projection:            
            for write_name in self.entries.keys():
                pyoomph_name=self._sanitize_name(write_name)
                
                if self.get_combined_equations()._is_ode():
                    mynodes=[None]
                else:
                    mynodes=[n for n in mesh.nodes()]
                
                if self.vector_dim is None:
                
                    buffer=numpy.zeros([len(mynodes)])
                    if self.get_combined_equations()._is_ode():
                        index=mesh.get_code_gen().get_code().get_elemental_field_indices()[pyoomph_name]
                        buffer[0]=mesh.element_pt(0).internal_data_pt(index).value(0)
                    else:
                        if mesh.has_interface_dof_id(pyoomph_name)>=0:
                            index=mesh.has_interface_dof_id(pyoomph_name)
                            for i,n in enumerate(mynodes):
                                buffer[mesh._precice_node_to_vertex_id[n]]=n.value(n.additional_value_index(index))
                        else:
                            index=mesh.get_nodal_field_indices()[pyoomph_name]
                            if index<0:
                                raise ValueError("Field "+pyoomph_name+" not found in mesh. TODO: Elemental fields")
                            for i,n in enumerate(mynodes):
                                buffer[mesh._precice_node_to_vertex_id[n]]=n.value(index,buffer)
                else:
                    buffer=[]
                    for vindex in range(self.vector_dim):
                        buffer_row=numpy.zeros([len(mynodes)])
                        component_name=pyoomph_name+"_"+["x","y","z"][vindex]
                        if mesh.has_interface_dof_id(component_name)>=0:
                            index=mesh.has_interface_dof_id(component_name)
                            for i,n in enumerate(mynodes):
                                buffer_row[mesh._precice_node_to_vertex_id[n]]=n.value(n.additional_value_index(index))
                        else:
                            index=mesh.get_nodal_field_indices()[component_name]
                            for i,n in enumerate(mynodes):
                                buffer_row[mesh._precice_node_to_vertex_id[n]]=n.value(index,buffer)
                        buffer.append(buffer_row)
                    buffer=numpy.array(buffer).T
                                
                interface.write_data(provider.name, write_name, vertex_ids, buffer)                                    
        else:
            nodemap=mesh.fill_node_index_to_node_map()
            inds=numpy.array([mesh._precice_node_to_vertex_id[n] for n in nodemap])            
            for write_name in self.entries.keys():
                if self.vector_dim is None:
                    expr_index=mesh.list_local_expressions().index("_precice_write_"+write_name)
                    buffer=mesh.evaluate_local_expression_at_nodes(expr_index,True,False)                
                    buffer=numpy.array(buffer)                                                            
                    if self.get_combined_equations()._is_ode():
                        expr=mesh.element_pt(0).evalulate_local_expression_at_midpoint(expr_index)
                        buffer=numpy.array([expr])
                        #print("WRITE DATA",write_name,mesh._precice_vertex_ids,buffer)
                        interface.write_data(provider.name, write_name, mesh._precice_vertex_ids, buffer)
                    else:                    
                        interface.write_data(provider.name, write_name, vertex_ids[inds], buffer)
                else:                    
                    directs=["x","y","z"]
                    buffer=[]
                    for vindex in range(self.vector_dim):                    
                        expr_index=mesh.list_local_expressions().index("_precice_write_"+write_name+"_"+directs[vindex])
                        buffer_row=mesh.evaluate_local_expression_at_nodes(expr_index,True,False)                
                        buffer_row=numpy.array(buffer_row)                                        
                        buffer.append(buffer_row)
                    buffer=numpy.array(buffer).T                
                    interface.write_data(provider.name, write_name, vertex_ids[inds], buffer)

   
    def before_precice_initialise(self, eqtree: "EquationTree"):
        if self.get_problem()._precice_interface.requires_initial_data():
            self._do_write_data(0)
        

    def after_precice_solve(self, eqtree: "EquationTree",precice_dt:float):
        self._do_write_data(precice_dt)

            

class PreciceReadData(BaseEquations):
    """Must be combined with :py:class:`~pyoomph.solvers.precice_adapter.PreciceProvideMesh` on the same domain or interface.
    
    Reads data from preCICE. Data can be an arbitrary scalar or vectorial expression. In the latter case, the vector_dim parameter must be set.
    
    For each data to be read, a corresponding field is constructed in pyoomph on this domain. This field is pinned, i.e. not a degree of freedom, and its values are updated via preCICE.

    Args:
        scaling: Optional scaling factor (e.g. units) for the data
        vector_dim: The dimension of the vector field. If None, a scalar field is assumed.
        kwargs: The data to be read. The keys are the names of the data in pyoomph, the values are the names of the fields in preCICE.
    """
    def __init__(self,*,scaling:ExpressionNumOrNone=None,vector_dim:Optional[int]=None,**kwargs:str):
        super().__init__()
        self.entries=kwargs.copy()
        for n,e in self.entries.items():
            if not isinstance(e,str):
                raise ValueError("PreciceReadData requires mapping from string (pyoomph var name) to string (precice config name), but got "+str(type(e))+" in "+str(n)+" = "+str(e))
        self.scaling=scaling
        self.vector_dim=vector_dim

    def define_scalar_field(self,name:str,space:FiniteElementSpaceEnum,scale:Optional[ExpressionNumOrNone]=None):
        # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
        return Equations.define_scalar_field(self,name,space,scale=scale)
    
    def _internal_define_scalar_field(self,name:str,space:FiniteElementSpaceEnum,scale:Optional[ExpressionNumOrNone]=None,testscale:Optional[ExpressionNumOrNone]=None,discontinuous_refinement_exponent:Optional[int]=None):
        return Equations._internal_define_scalar_field(self,name,space,scale=scale,testscale=testscale,discontinuous_refinement_exponent=discontinuous_refinement_exponent)

    def define_fields(self):
        if self.get_combined_equations()._is_ode():
            if self.vector_dim is not None:
                raise RuntimeError("Vector fields not supported for ODEs")
            else:
                for name in self.entries.keys():
                    # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
                    ODEEquations.define_ode_variable(self,name,scale=self.scaling)
        else:
            for name in self.entries.keys():
                # This is a bit dirty, but I cannot see how it can be done differently, except for providing different classes for ODEEquations and Equations
                if self.vector_dim is None:
                    Equations.define_scalar_field(self,name,self.get_current_code_generator()._coordinate_space,scale=self.scaling)
                else:
                    Equations.define_vector_field(self,name,self.get_current_code_generator()._coordinate_space,scale=self.scaling,dim=self.vector_dim)

    def define_residuals(self):
        for name in self.entries.keys():
            if self.vector_dim is None:
                self.set_Dirichlet_condition(name,True) # Fix it
            else:
                directs=["x","y","z"]
                for i in range(self.vector_dim):
                    self.set_Dirichlet_condition(name+"_"+directs[i],True)

    def _do_read_data(self,precice_dt:float):
        mesh=self.get_mesh()
        if not hasattr(mesh,"_precice_vertex_ids"):
            raise ValueError("PreciceProvideMesh not set, please add it before PreciceReadData")
        interface=mesh.get_problem()._precice_interface
        provider=self.get_combined_equations().get_equation_of_type(PreciceProvideMesh,always_as_list=True)
        if len(provider)!=1:
            raise ValueError("PreciceProvideMesh not set or set multiple times on this domain, please add a single one")
        provider=provider[0]
        assert isinstance(provider,PreciceProvideMesh)
        vertex_ids=mesh._precice_vertex_ids        
                
        for pyoomph_name,precice_name in self.entries.items():
            buffer=interface.read_data(provider.name, precice_name, vertex_ids, precice_dt)
            if self.vector_dim is None:
                if len(buffer.shape)>1:
                    raise RuntimeError("Expected scalar, got vector")
                if self.get_combined_equations()._is_ode():
                    index=mesh.get_code_gen().get_code().get_elemental_field_indices()[pyoomph_name]
                    #print("READ DATA",pyoomph_name,index,buffer[0])
                    mesh.element_pt(0).internal_data_pt(index).set_value(0,buffer[0])                    
                else:
                    if mesh.has_interface_dof_id(pyoomph_name)>=0:
                        index=mesh.has_interface_dof_id(pyoomph_name)
                        for i,n in enumerate(mesh.nodes()):                    
                            n.set_value(n.additional_value_index(index) ,buffer[mesh._precice_node_to_vertex_id[n]])
                    else:
                        index=mesh.get_nodal_field_indices()[pyoomph_name]
                        if index<0:
                            raise RuntimeError("TODO: Elemental fields?")
                        for i,n in enumerate(mesh.nodes()):
                            n.set_value(index,buffer[mesh._precice_node_to_vertex_id[n]])
            else:
                if len(buffer.shape)==1:
                    raise RuntimeError("Expected vector, got scalar")
                if buffer.shape[1]!=self.vector_dim:
                    raise RuntimeError("Expected vector of dimension "+str(self.vector_dim)+", got "+str(buffer.shape[1]))
                directs=["x","y","z"]
                for vindex in range(self.vector_dim):
                    if mesh.has_interface_dof_id(pyoomph_name+"_"+directs[vindex])>=0:
                        index=mesh.has_interface_dof_id(pyoomph_name+"_"+directs[vindex])
                        for i,n in enumerate(mesh.nodes()):                    
                            n.set_value(n.additional_value_index(index) ,buffer[mesh._precice_node_to_vertex_id[n],vindex])
                    else:
                        index=mesh.get_nodal_field_indices()[pyoomph_name+"_"+directs[vindex]]
                        for i,n in enumerate(mesh.nodes()):
                            n.set_value(index,buffer[mesh._precice_node_to_vertex_id[n],vindex])

    def before_precice_solve(self, eqtree: EquationTree, precice_dt: float):
        self._do_read_data(precice_dt)
             




# Instance of the helper class
_pyoomph_precice_adapter=_PyoomphPreciceAdapater()
# Access function to the helper class instance. Called by the Problem class
def get_pyoomph_precice_adapter():
    return _pyoomph_precice_adapter
    

