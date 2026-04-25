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
 
from ..typings import *
import numpy

from .mesh import *

from ..expressions import ExpressionOrNum


class DropletMesh3d(MeshTemplate):
    def __init__(self,h_center:ExpressionOrNum=0.25,r:ExpressionOrNum=1,Nr:int=5,Nphi:int=5,spherical_cap:bool=False):
        super(DropletMesh3d, self).__init__()
        self.r=r
        self.rn_straight=0.4
        self.corner_inward_slant=0.2
        self.h0=h_center
        self._dom=None
        self._x000:NPFloatArray
        self._x100:NPFloatArray
        self._x010:NPFloatArray
        self._x110:NPFloatArray
        self._lsouth:Callable[[List[float]],NPFloatArray]=lambda s : (1-s[0])*self._x000+s[0]*self._x100
        self._lnorth:Callable[[List[float]],NPFloatArray] = lambda s: (1 - s[0]) * self._x010 + s[0] * self._x110
        self._lwest:Callable[[List[float]],NPFloatArray] = lambda s: (1 - s[1]) * self._x000 + s[1] * self._x010
        self._least:Callable[[List[float]],NPFloatArray]= lambda s: (1 - s[1]) * self._x100 + s[1] * self._x110
        self._hf:Callable[[float],float]=lambda x: x # Will be changed later
        self.spherical_cap=spherical_cap
        self._rot_x:Callable[[Sequence[float]],Sequence[float]]=lambda x:x
        self._inward_slant=0.2
        self.Nr=Nr
        self.Np=Nphi
        self.tetra=False
        self._nodes=[]
        #self._node_calls=0
        #self._node_max = 0

    def _uniq_node(self,x:Sequence[float]) -> int:
        res=self.add_node_unique( x[0], x[1], x[2])
        #self._node_calls+=1
        #self._node_max=max(self._node_max,res)
        #print(self._node_calls,self._node_max)
        return res


    def add_brick(self,pf:Sequence[int]):
        domain=self._dom
        assert domain is not None
        if self.tetra:
            domain.add_tetra_3d_C1(pf[0], pf[2], pf[1], pf[4])
            domain.add_tetra_3d_C1(pf[1], pf[7], pf[2], pf[3])
            domain.add_tetra_3d_C1(pf[4], pf[1], pf[5], pf[7])
            domain.add_tetra_3d_C1(pf[1], pf[4], pf[7], pf[2])
            domain.add_tetra_3d_C1(pf[2], pf[4], pf[6], pf[7])
        else:
            domain.add_brick_3d_C1(*pf)

    def add_brick_curved(self,sl:List[float],sh:List[float],end:bool=False):
        def n(s:List[float]) -> int:
            x=0.5*(self._lsouth(s)*(1-s[1])+self._lnorth(s)*s[1]+self._lwest(s)*(1-s[0])+self._least(s)*s[0])
            x*=1-(s[2]*(1-s[2])*self._inward_slant)*numpy.sqrt(s[0]**2)
            x[2]=self._hf(x)*s[2]
            x=self._rot_x(x)
            res=self._uniq_node(x)
            if s[2]*s[2]<1e-20:
                self.add_nodes_to_boundary("droplet_substrate",[res])
            elif (s[2]-1.0)**2<1e-20:
                self.add_nodes_to_boundary("droplet_gas",[res])
            return res

        if end==False:
            pts = [n([sl[0], sl[1], sh[2]]), n([sl[0], sh[1], sh[2]]), n([sh[0], sl[1], sh[2]]),
                    n([sh[0], sh[1], sh[2]])]
            pts+=[n([sl[0], sl[1],sl[2]]), n([sl[0], sh[1],sl[2]]), n([sh[0], sl[1],sl[2]]), n([sh[0], sh[1],sl[2]])]

            self.add_brick(pts)
        else:
            nout1=n([sh[0], sl[1], sl[2]])
            nout2=n([sh[0], sh[1], sl[2]])

            pts = [n([sl[0], sl[1], sh[2]]), n([sl[0], sh[1], sh[2]]), n([sl[0], sl[1], 1]),
                    n([sl[0], sh[1], 1])]
            pts += [n([sl[0], sl[1], sl[2]]), n([sl[0], sh[1], sl[2]]), nout1, nout2]
            self.add_nodes_to_boundary("droplet_gas", [nout1,nout2])
            self.add_brick(pts)
            pass

    def add_curved_part(self) -> float:
        last_sx, last_sy=0.0,0.0
        for j,sx in enumerate(numpy.linspace(0,1,self.Nr,endpoint=False)): #type:ignore
            if j>=1:
                for i,sy in enumerate(numpy.linspace(0,1,self.Np)): #type:ignore
                    if i>=1:
                        self.add_brick_curved([last_sx,last_sy,0],[sx,sy,0.5])
                        self.add_brick_curved([last_sx, last_sy, 0.5], [sx, sy, 1])
                    last_sy = sy
            last_sx=sx
        for i, sy in enumerate(numpy.linspace(0, 1, self.Np)): #type:ignore
            if i >= 1:
                self.add_brick_curved([last_sx, last_sy, 0], [1, sy, 0.5],end=True)
            last_sy = sy
        return last_sx

    def add_quarter_section(self):
        cis=1.0-self.corner_inward_slant
        self._x000 = numpy.array([self.rn_straight * self.r, 0, 0]) #type:ignore
        self._x100 = numpy.array([self.r, 0, 0]) #type:ignore
        self._x010 = numpy.array([self.rn_straight * self.r*cis, self.rn_straight * self.r*cis, 0]) #type:ignore
        self._x110 = numpy.array([self.r * numpy.sqrt(0.5), self.r * numpy.sqrt(0.5), 0]) #type:ignore
        self._least = lambda s: numpy.sin((1 - s[1]) * numpy.pi / 3) / numpy.sin(numpy.pi / 3) * self._x100 + numpy.sin(
            s[1] * numpy.pi / 3) / numpy.sin(numpy.pi / 3) * self._x110

        self._inward_slant = 0.2

        self.add_curved_part()

        self._x000 = numpy.array([self.rn_straight * self.r * cis, self.rn_straight * self.r * cis, 0]) #type:ignore
        self._x100 = numpy.array([self.r * numpy.sqrt(0.5), self.r * numpy.sqrt(0.5), 0]) #type:ignore
        self._x010 = numpy.array([ 0,self.rn_straight * self.r, 0]) #type:ignore
        self._x110 = numpy.array([0, self.r, 0]) #type:ignore

        self._least = lambda s: numpy.sin((1 - s[1]) * numpy.pi / 3) / numpy.sin(numpy.pi / 3) * self._x100 + numpy.sin(
            s[1] * numpy.pi / 3) / numpy.sin(numpy.pi / 3) * self._x110

        last_sx = self.add_curved_part()

        # Central part
        self._least = lambda s: (1 - s[1]) * self._x100 + s[1] * self._x110
        self._inward_slant = 0.0
        self._x000 = numpy.array([0, 0, 0]) #type:ignore
        self._x100 = numpy.array([self.rn_straight * self.r, 0, 0]) #type:ignore
        self._x010 = numpy.array([0, self.rn_straight * self.r, 0]) #type:ignore
        self._x110 = numpy.array([self.rn_straight * self.r*cis, self.rn_straight * self.r*cis, 0]) #type:ignore
        last_sy=-1        
        for j, sx in enumerate(numpy.linspace(0, 1, self.Np)): #type:ignore
            if j >= 1:
                for i, sy in enumerate(numpy.linspace(0, 1, self.Np)): #type:ignore
                    if i >= 1:
                        self.add_brick_curved([last_sx, last_sy, 0], [sx, sy, 0.5])
                        self.add_brick_curved([last_sx, last_sy, 0.5], [sx, sy, 1])
                    last_sy = sy
            last_sx = sx

    def define_geometry(self):
        self.r=self.nondim_size(self.r)
        self.h0=self.nondim_size(self.h0)

        if self.spherical_cap:
            rc=(self.r*self.r+self.h0*self.h0)/(2*self.h0)
            zoffs=-rc+self.h0
            self._hf=lambda x : numpy.sqrt(rc**2-(x[0]**2+x[1]**2))+zoffs #type:ignore
        else:
            self._hf = lambda x: self.h0 * (1 - (x[0] ** 2 + x[1] ** 2) / self.r ** 2) #type:ignore

        droplet=self.new_domain("droplet")
        self._dom=droplet
        #North east
        self._rot_x=lambda  x : x
        self.add_quarter_section()

        #North west
        self._rot_x = lambda x: numpy.array([-x[1],x[0],x[2]]) #type:ignore
        self.add_quarter_section()

        # south west
        self._rot_x = lambda x: numpy.array([-x[0], -x[1], x[2]]) #type:ignore
        self.add_quarter_section()
        # south east
        self._rot_x = lambda x: numpy.array([x[1], -x[0], x[2]]) #type:ignore
        self.add_quarter_section()

