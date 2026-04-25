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

class TuringEquations(Equations):
    def __init__(self,d,f,g):
        super().__init__()
        self.d=d # Diffusivity ratio
        self.f,self.g=f,g # Reaction terms
        
    def define_fields(self):
        self.define_scalar_field("u","C2") # activator
        self.define_scalar_field("v","C2") # inhibitor
        
    def define_residuals(self):
        u,utest=var_and_test("u")
        v,vtest=var_and_test("v")
        self.add_weak(partial_t(u)-self.f,utest).add_weak(grad(u),grad(utest)) 
        self.add_weak(partial_t(v)-self.g,vtest).add_weak(self.d*grad(v),grad(vtest))
        
        
class TuringProblem(Problem):
    def __init__(self):
        super().__init__()
        # Parameters, see https://dx.doi.org/10.1007/s10994-023-06334-9
        self.d,self.a,self.b,self.c=self.define_global_parameter(d=40,a=0.01,b=1.2,c = 0.7)        
        # Reaction terms
        self.f=self.a-self.b*var("u")+var("u")**2/(var("v")*(1+self.c*var("u")**2))
        self.g=var("u")**2-var("v")
        
    def define_problem(self):        
        # We add a 0d mesh here. So we do not have any spatial information
        from pyoomph.meshes.simplemeshes import PointMesh
        self+=PointMesh()
        eqs=TuringEquations(self.d,self.f,self.g)
        # Some reasonable guess here (not the exact solution)
        eqs+=InitialCondition(u=(self.a + 1)/self.b, v=(self.a + 1)**2/self.b**2)
        self+=eqs@"domain"

if __name__=="__main__":
    with TuringProblem() as problem:
        # Take a quick C compiler, to speed up the code generation
        problem.set_c_compiler("tcc")
        # Important part: This will add the N+1 dimension allowing for perturbations like exp(i*k*x+lambda*t)
        problem.setup_for_stability_analysis(additional_cartesian_mode=True)
        # Solve for the flat stationary solution
        problem.solve()
        # And scan over the k values, solve the normal mode eigenvalue problem and write the results to a file
        output=problem.create_text_file_output("dispersion.txt",header=["k","ReLamba1","ReLambda2","ImLambda1","ImLambda2"])    
        for k in numpy.linspace(0,1,400):
            problem.solve_eigenproblem(2,normal_mode_k=k)
            evs=problem.get_last_eigenvalues()
            output.add_row(k,numpy.real(evs[0]),numpy.real(evs[1]),numpy.imag(evs[0]),numpy.imag(evs[1]))
            
            
            