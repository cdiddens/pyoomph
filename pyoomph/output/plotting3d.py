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
 
import os
from . plotting import BasePlotter
from ..typings import *
import pyvista

if TYPE_CHECKING:
    from ..generic.problem import Problem
    from ..meshes.meshdatacache import MeshDataEigenModes
    from ..output.meshio import AnySpatialMesh


class _PyVistaPlotPartBase:        
    def _add_to_plotter(self,plotter:"PyVistaPlotter"):
        pass

class _PyVistaPlotPartMesh(_PyVistaPlotPartBase):    
    def __init__(self,what:str):
        super().__init__()
        self.what=what
        
    def _add_to_plotter(self,plotter:"PyVistaPlotter",pl:pyvista.Plotter):
        mshio=plotter._get_meshio_data(self.what)
        pvmesh=pyvista.from_meshio(mshio)  # This will convert the meshio data to a PyVista mesh
        act=pl.add_mesh(pvmesh,show_edges=True,show_scalar_bar=True,render_points_as_spheres=True,point_size=5,opacity=0.5)
        act.prop.culling="back"
        
        
        
        

class PyVistaPlotter(BasePlotter):
    def __init__(self, problem:Optional["Problem"]=None,filetrunk:str="plot_{:05d}",fileext:Union[str,List[str]]="svg",eigenvector:Optional[int]=None,eigenmode:"MeshDataEigenModes"="abs",):
        super().__init__(problem=problem,eigenvector=eigenvector,eigenmode=eigenmode)
        self.filetrunk=filetrunk
        self.fileext=fileext
        self._parts:List[_PyVistaPlotPartBase]=[]
        self.add_eigen_to_mesh_positions=False
        self._output_dir="_plots"

    def _get_mesh_data(self,msh:Union[str,"AnySpatialMesh"],problem_name:str=""):        
        return self.get_problem(problem_name=problem_name).get_cached_mesh_data(msh,nondimensional=False,tesselate_tri=False,eigenvector=self.eigenvector,eigenmode=self.eigenmode,add_eigen_to_mesh_positions=self.add_eigen_to_mesh_positions)
        
                
    def add_plot(self,what:str):
        self._parts.append(_PyVistaPlotPartMesh(what))
        
    def _after_plot(self):
        pl = pyvista.Plotter(off_screen=True)
        for p in self._parts:
            p._add_to_plotter(self,pl)
        #pl.show()
        pdir=os.path.join(self._problem.get_output_directory(),self._output_dir)
        os.makedirs(pdir,exist_ok=True)
        pl.save_graphic(os.path.join(pdir,self.filetrunk.format(self.get_problem()._output_step)+"."+self.fileext),raster=True,painter=False)
        
        
        
    def _get_meshio_data(self,msh:Union[str,"AnySpatialMesh"],problem_name:str=""):
        from pyoomph.output.meshio import _convert_mesh_to_meshio
        return _convert_mesh_to_meshio(self.get_problem(), self._get_mesh_data(msh,problem_name=problem_name))


