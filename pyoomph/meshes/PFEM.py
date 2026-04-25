#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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
 
from ..generic.problem import Problem
from .mesh import MeshFromTemplate2d
import numpy
from ..typings import *
from scipy.spatial import Delaunay

class PFEMMeshUpdater:
    def __init__(self,problem:Problem,meshname:str) -> None:
        super().__init__()
        self.problem=problem
        self.meshname=meshname
        self.default_alpha=0.1

    def get_mesh(self)->MeshFromTemplate2d:
        mesh=self.problem.get_mesh(self.meshname)
        assert isinstance(mesh,MeshFromTemplate2d)
        return mesh

    def alpha_shape(self,tri, alpha:Optional[float]=None, only_outer=True, do_not_prune_at_x0_eps:Optional[float]=None):
        """
        Compute the alpha shape (concave hull) of a set of points.
        :param points: np.array of shape (n,2) points.
        :param alpha: alpha value.
        :param only_outer: boolean value to specify if we keep only the outer border
        or also inner edges.
        :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
        """

        def add_edge(edges, i, j):
            """
            Add an edge between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                assert (j, i) in edges, "Can't go twice over same directed edge right?"
                if only_outer:
                    # if both neighboring triangles are in shape, it's not a boundary edge
                    edges.remove((j, i))
                return
            edges.add((i, j))

        innersimplices:List[List[int]]=[]
        edges = set()
        # Loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.simplices:
            pa = tri.points[ia]
            pb = tri.points[ib]
            pc = tri.points[ic]
            # Computing radius of triangle circumcircle
            # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
            a = numpy.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = numpy.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = numpy.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            s = (a + b + c) / 2.0
            area = numpy.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            if do_not_prune_at_x0_eps is not None:
                eps=do_not_prune_at_x0_eps
                if (pa[0]<eps and (pb[0]<eps or pc[0]<eps)) or (pb[0]<eps and pc[0]<eps):
                    if not (pa[0]<eps and pb[0]<eps and pc[0]<eps):
                        circum_r=0
            if alpha is None:
                alpha=self.get_local_alpha(pa[0],pa[1])
                alpha=max(alpha,self.get_local_alpha(pb[0],pb[1]))
                alpha=max(alpha,self.get_local_alpha(pc[0],pc[1]))
            if circum_r < alpha:
                add_edge(edges, ia, ib)
                add_edge(edges, ib, ic)
                add_edge(edges, ic, ia)
                innersimplices.append([ia,ib,ic])
        return edges,innersimplices
    
    def delaunay_with_alpha_shape(self,coords:List[Tuple[float,float]],alpha:Optional[float]=None, do_not_prune_at_x0_eps:Optional[float]=None):
        tri=Delaunay(coords)
        edges,innersimplices=self.alpha_shape(tri,alpha,do_not_prune_at_x0_eps=do_not_prune_at_x0_eps)        
        dom=self.get_mesh()
        # Create a new mesh
        for s in innersimplices:
            n1=dom.node_pt(s[0])
            n2=dom.node_pt(s[1])
            n3=dom.node_pt(s[2])
            dom.add_tri_C1(n1,n2,n3)        
        return edges

    def define_new_mesh(self,coords:List[Tuple[float,float]]):
        raise RuntimeError("Implement this here by overriding this method")
    
    def get_local_alpha(self,x,y):
        return self.default_alpha
    
    def insert_or_remove_points_where_needed(self):
        nodal_remap=[]
        mesh=self.get_mesh() 
        
        min_elem_size=0.0001
        
        for e in mesh.elements():   
            assert e.nnode()==3
            # Test if the element is too small    
            s=e.get_current_cartesian_nondim_size()
            if s<min_elem_size:
                #num_on_boundary=0
                #for i in range(e.nnode()):
                #    if e.node_pt(i).is_on_boundary():
                #        num_on_boundary+=1                                
                for i in range(e.nnode()):
                    if not e.node_pt(i).is_on_boundary():
                        e.node_pt(i).set_obsolete()
                        
            for edge in range(3):
                n1=e.node_pt(edge)
                n2=e.node_pt((edge+1)%3)
                if n1.is_obsolete() or n2.is_obsolete():
                    continue
                s1=e.local_coordinate_of_node(edge)
                s2=e.local_coordinate_of_node((edge+1)%3)
                x1,y1=n1.x(0),n1.x(1)
                x2,y2=n2.x(0),n2.x(1)
                dist=numpy.sqrt((x1-x2)**2+(y1-y2)**2)
                if dist<0.05:                
                                        
                    if n1.is_on_boundary() and n2.is_on_boundary():
                        if len(n1.get_boundary_indices().union(n2.get_boundary_indices())):
                            mesh.add_node_to_mesh(e.create_interpolated_node(0.5*(numpy.array(s1)+numpy.array(s2)),True))
                            n1.set_obsolete()                    
                            n2.set_obsolete()
                    elif not n1.is_on_boundary():
                        n1.set_obsolete()                    
                    elif  n2.is_on_boundary():
                        n2.set_obsolete()
                    #if n1.is_obsolete() and n2.is_obsolete():
                    #    mesh.add_node_to_mesh(e.create_interpolated_node(0.5*(numpy.array(s1)+numpy.array(s2)),True))
        
        for e in mesh.elements():
            assert e.nnode()==3
            for edge in range(3):
                n1=e.node_pt(edge)
                n2=e.node_pt((edge+1)%3)
                s1=e.local_coordinate_of_node(edge)
                s2=e.local_coordinate_of_node((edge+1)%3)
                x1,y1=n1.x(0),n1.x(1)
                x2,y2=n2.x(0),n2.x(1)
                dist=numpy.sqrt((x1-x2)**2+(y1-y2)**2)
                if dist>0.2:                
                    mesh.add_node_to_mesh(e.create_interpolated_node(0.5*(numpy.array(s1)+numpy.array(s2)),True))
                
                

    def update_mesh(self):
        dom=self.get_mesh()
        
        
        self.insert_or_remove_points_where_needed()
        nodal_remap=[]
        for i in range(dom.nnode()):
            if not dom.node_pt(i).is_obsolete():
                nodal_remap.append(i)        
        
    
        coords=[(dom.node_pt(j).x(0),dom.node_pt(j).x(1),) for j in nodal_remap]
        # Remove interfaces and remove the nodes from the boundary       
        for n in dom.nodes():
            binds=n.get_boundary_indices()
            for b in binds:
                n.remove_from_boundary(b)
        for inam,imesh in dom._interfacemeshes.items():
            imesh.clear_before_adapt()
        for inam,imesh in dom._interfacemeshes.items():
            for inam2,imesh2 in imesh._interfacemeshes.items():
                imesh2.clear_before_adapt()
        for inam,imesh in dom._interfacemeshes.items():
            for inam2,imesh2 in imesh._interfacemeshes.items():
                for inam3,imesh3 in imesh2._interfacemeshes.items():
                    imesh3.clear_before_adapt()
        

        
        dom.flush_element_storage()        
        dom.prune_dead_nodes(False)        
        self.define_new_mesh(coords)
        
        
        
        dom.recreate_boundary_information()        
        isolated_nodes=set([dom.node_pt(ni) for ni in range(dom.nnode())])
        for e in dom.elements():
            for ni in range(e.nnode()):
                if e.node_pt(ni) in isolated_nodes:
                    isolated_nodes.remove(e.node_pt(ni))
        for n in isolated_nodes:
            n.set_obsolete()
        #print("ISO",isolated_nodes,dom.prune_dead_nodes())
        dom.prune_dead_nodes(False)
        dom.recreate_boundary_information()        


        for inam,imesh in dom._interfacemeshes.items():
            imesh.rebuild_after_adapt()
        for inam,imesh in dom._interfacemeshes.items():
            for inam2,imesh2 in imesh._interfacemeshes.items():
                imesh2.rebuild_after_adapt()
        for inam,imesh in dom._interfacemeshes.items():
            for inam2,imesh2 in imesh._interfacemeshes.items():
                for inam3,imesh3 in imesh2._interfacemeshes.items():
                    imesh3.rebuild_after_adapt()                
                    
        dom.recreate_boundary_information()        

        
        dom.get_problem().rebuild_global_mesh()
        dom.get_problem().reapply_boundary_conditions()
        dom.get_problem().invalidate_cached_mesh_data()
        dom.get_problem().invalidate_eigendata()
        
        
