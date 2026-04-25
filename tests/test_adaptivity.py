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

# Import pyoomph
from pyoomph import *
from pyoomph.expressions import *
# Also import the predefined harmonic oscillator equation
from pyoomph.meshes.simplemeshes import CircularMesh
from pyoomph.equations.poisson import *


class PoissonProblem(Problem):	
	def define_problem(self):
		self+=CircularMesh(radius=1,segments=["NE"])
		eqs=PoissonEquation(source=1)+DirichletBC(u=0)@"circumference"
		anasol=0.25*(1-dot(var("coordinate"),var("coordinate")))
		eqs+=IntegralObservables(error=(var("u")-anasol)**2)
		self+=eqs@"domain"
		

def test_without_adapt():
	with PoissonProblem() as problem:
		
		problem.solve()
		err=float(problem.get_mesh("domain").evaluate_observable("error"))
		assert err<1e-7
  
def test_with_adapt():
	with PoissonProblem() as problem:
		problem+=RefineToLevel(2)@"domain"
		problem+=RefineToLevel(4)@"domain/circumference"
		problem+=MeshFileOutput()@"domain"
		problem.solve()
		problem.output()
		err=float(problem.get_mesh("domain").evaluate_observable("error"))
		assert err<1e-10
  
