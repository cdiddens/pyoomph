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


class ConvectionDiffusionEquationWithSUPG(Equations):
    def __init__(self, u, D,with_SUPG=True):
        super(ConvectionDiffusionEquationWithSUPG, self).__init__()
        self.u = u  # advection velocity
        self.D = D  # diffusivity
        self.scheme="TPZ" # Time scheme, trapezoidal rule
        self.with_SUPG=with_SUPG # do we activate SUPG?

    def define_fields(self):
        self.define_scalar_field("c", "C1") # Take the coarse space C1

    def get_supg_tau(self):
        # We must find an equation of the type ElementSizeForSUPG, which calculates the element size
        elsize_eqs = self.get_combined_equations().get_equation_of_type(ElementSizeForSUPG, always_as_list=True)
        if len(elsize_eqs)!=1: # User must combine it with a single ElementSizeForSUPG instance
            raise RuntimeError("SUPG only works if combined with a single ElementSizeForSUPG equation")
        elsize_eqs=elsize_eqs[0] # get the ElementSizeForSUPG object, which is combined with this equation
        h = elsize_eqs.get_element_h() + 1e-15 # element size, add a tiny offset to prevent errors
        u_mag=square_root(dot(self.u,self.u))+1e-15 # velocity magnitude , add a tiny offset to prevent errors
        Pe_h=u_mag*h/(2*self.D) # Mesh Peclet number
        beta=1/tanh(Pe_h)-1/Pe_h # coefficient activating SUPG if Pe becomes large
        tau = subexpression(beta*h/(2*u_mag)) # returning the tau coefficient
        return tau

    def define_residuals(self):
        c, ctest = var_and_test("c")
        # This term occurs multiple times, so wrap it into a subexpression for performance gain
        radv = subexpression(time_scheme(self.scheme,partial_t(c) + dot(self.u, grad(c))))
        self.add_residual(weak(radv, ctest))  # time derivative and advection
        self.add_residual(time_scheme(self.scheme,weak(self.D * grad(c), grad(ctest))))  # diffusion
        if self.with_SUPG: # SUPG stabilization
            self.add_residual(time_scheme(self.scheme,weak(radv,self.get_supg_tau() * dot(self.u, grad(ctest)))))


class OneDimAdvectionDiffusionProblem(Problem):
    def __init__(self):
        super(OneDimAdvectionDiffusionProblem, self).__init__()
        self.u=vector(1,0)
        self.D=0.0001
        self.with_SUPG=True

    def define_problem(self):
        self.add_mesh(LineMesh(N=100,size=100,minimum=-20)) # coarse mesh from [-20:80]

        eqs=TextFileOutput()
        eqs+=ConvectionDiffusionEquationWithSUPG(u=self.u,D=self.D,with_SUPG=self.with_SUPG)
        if self.with_SUPG:
            eqs+=ElementSizeForSUPG() # We must add the element size

        x=var("coordinate_x")
        cinit=exp(-x**2*0.25)
        eqs+=InitialCondition(c=cinit)

        eqs+=DirichletBC(c=0)@"left"
        eqs += DirichletBC(c=0) @ "right"

        self.add_equations(eqs@"domain")


if __name__=="__main__":
    with OneDimAdvectionDiffusionProblem() as problem:
        problem.with_SUPG=True
        problem.run(50,outstep=1,maxstep=0.1)