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
 
import _pyoomph

from ..typings import *

import numpy


class CurvedEntityCircle(_pyoomph.MeshTemplateCurvedEntity):
   def __init__(self, center:Sequence[float], radius:float):
      super().__init__(1)
      self.center:NPFloatArray = numpy.array([center[i] if i<len(center) else 0.0 for i in range(3)],dtype=numpy.float64) #type:ignore
      self.radius = radius

   def parametric_to_pos(self, t:int, param:NPFloatArray, pos:NPFloatArray):
      pos[:] = self.center
      pos[0] += self.radius * numpy.cos(param[0])
      pos[1] += self.radius * numpy.sin(param[0])

   def pos_to_parametric(self, t:int, pos:NPFloatArray, param:NPFloatArray):
      diff_x = pos[0] - self.center[0]
      diff_y = pos[1] - self.center[1]
      param[0] = numpy.arctan2(diff_y, diff_x)

   def ensure_periodicity(self, param:NPFloatArray):
      # print("DIFF",abs(parametrics[0]-parametrics[1]),numpy.pi)
      if abs(param[0] - param[1]) >= numpy.pi:
         param[0] = numpy.mod(param[0], 2 * numpy.pi)
         param[1] = numpy.mod(param[1], 2 * numpy.pi)
         # raise NotImplementedError("Handle the periodic case here")
         # pass
      # exit()

