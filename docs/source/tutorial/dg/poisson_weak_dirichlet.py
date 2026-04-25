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


from pyoomph import *
from pyoomph.expressions import *

class PoissonEquations(Equations):
   def __init__(self,f,space,alpha_DG=4):
       super().__init__()
       self.f=f
       self.space=space
       self.requires_interior_facet_terms=is_DG_space(self.space,allow_DL_and_D0=True)
       self.alpha_DG=alpha_DG

   def define_fields(self):
        self.define_scalar_field("u",self.space)

   def define_residuals(self):
       u,v=var_and_test("u")
       # Both continuous and discontinuous spaces
       self.add_residual(weak(grad(u),grad(v))-weak(self.f,v))
       if is_DG_space(self.space,allow_DL_and_D0=True):
         # Discontinuous penalization         
         h_avg=avg(var("cartesian_element_length_h"))
         n=var("normal") # will default to n^+ if used without any restriction in facets

         facet_terms= weak(self.alpha_DG/h_avg*jump(u),jump(v)) 
         facet_terms-=weak(jump(u)*n,avg(grad(v)))
         facet_terms-=weak(avg(grad(u)),jump(v)*n)          
         self.add_interior_facet_residual(facet_terms)
         
   def get_weak_dirichlet_terms_for_DG(self, fieldname, value):
      if fieldname!="u" or not is_DG_space(self.space,allow_DL_and_D0=True):
         return None
      else:
         u,v=var_and_test("u",domain="..") # bind the bulk field to get bulk gradients
         n=var("normal") # exterior normal
         h=var("cartesian_element_length_h",domain="..") # element size of the bulk element
         facet_terms=weak(self.alpha_DG/h*(u-value),v)
         facet_terms-=weak((u-value)*n,grad(v))
         facet_terms-=weak(grad(u),v*n)
         return facet_terms


class PoissonProblem(Problem):
    def __init__(self):
      super().__init__()
      x=var("coordinate")        
      self.f=500.0*exp(-((x[0] - 0.5)** 2 + (x[1] )**2)/ 0.02) 
      self.space="D1"
      self.prefer_weak_dirichlet=True
      self.alpha_DG=4
      self.N=8

    def define_problem(self):
      self+=RectangularQuadMesh(N=self.N) 
      eqs=MeshFileOutput(discontinuous=True)
      eqs+=PoissonEquations(self.f,self.space,self.alpha_DG)
      eqs+=DirichletBC(u=0,prefer_weak_for_DG=self.prefer_weak_dirichlet)@["left","right","top","bottom"]
      self+=eqs@"domain"

with PoissonProblem() as problem:
    problem.solve()
    problem.output()
