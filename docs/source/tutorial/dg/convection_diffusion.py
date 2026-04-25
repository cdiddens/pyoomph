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
from pyoomph.equations.SUPG import * # To calculate the element size


class ConvectionDiffusionEquation(Equations):
    def __init__(self, u, D ,space="C2",alpha_DG=2):
        super(ConvectionDiffusionEquation, self).__init__()
        self.u = u  # advection velocity
        self.D = D  # diffusivity
        self.space=space
        # Activate interior facet terms if the space is discontinuous
        self.requires_interior_facet_terms=is_DG_space(self.space)
        self.alpha_DG=alpha_DG # penalty parameter for DG

    def define_fields(self):
        self.define_scalar_field("c", self.space) 

    def define_residuals(self):
        c, ctest = var_and_test("c")        
        # Conventional form, used for CG spaces
        self.add_weak(partial_t(c), ctest)  
        self.add_weak(self.D * grad(c) -self.u*c, grad(ctest)) 
        
        if is_DG_space(self.space):
            # Additional facet terms for DG spaces            
            h_avg=avg(var("cartesian_element_length_h")) # length of an element:
            n=var("normal") # in facet terms, the normal vector is the facet normal. For the element normal, var("normal",domain="..") can be used.
            # if used without any restriction, i.e. outside from jump or average, it will default to n^+
            
            # Upwind scheme. See whether the velocity is in the direction of the normal vector, otherwise, it will be zero
            un_upwind=(dot(self.u, n) + absolute(dot(self.u, n)))/2
            
            # Assemble the facet terms:
            facet_terms=weak(self.D*(self.alpha_DG/h_avg)*jump(c),jump(ctest))
            facet_terms=-weak(self.D*jump(c)*n,avg(grad(ctest)))
            facet_terms=-weak(self.D*avg(grad(c)),jump(ctest)*n)
            facet_terms+=weak( jump(un_upwind*c,at_facet=True) ,jump(ctest))

            # And add them to the skeleton mesh of the facets
            self.add_interior_facet_residual(facet_terms)
                

class OneDimAdvectionDiffusionProblem(Problem):
    def __init__(self):
        super(OneDimAdvectionDiffusionProblem, self).__init__()
        self.u=vector(1,0)
        self.D=0.0001
        self.space="D1"

    def define_problem(self):
        self.add_mesh(LineMesh(N=100,size=100,minimum=-20)) # coarse mesh from [-20:80]

        eqs=TextFileOutput(discontinuous=True)
        eqs+=ConvectionDiffusionEquation(u=self.u,D=self.D,space=self.space)

        x=var("coordinate_x")
        cinit=exp(-x**2*0.25)
        eqs+=InitialCondition(c=cinit)

        self.add_equations(eqs@"domain")


if __name__=="__main__":
    with OneDimAdvectionDiffusionProblem() as problem:
        problem.run(50,outstep=1,maxstep=0.1)