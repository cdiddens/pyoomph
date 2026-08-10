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
 
from .. import _pyoomph_core as _pyoomph

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
      # The polar angle is only defined modulo 2*pi, so a facet crossing arctan2's branch cut on the
      # negative x axis arrives with endpoints near +pi and -pi. Shift every node onto the branch
      # nearest the first one's, so the facet blends the short way round. Taking a mod of each value
      # separately would only move the cut somewhere else instead of removing it.
      param[1:] -= 2 * numpy.pi * numpy.round((param[1:] - param[0]) / (2 * numpy.pi))


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
