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


from pyoomph import *
from pyoomph.expressions import *

class PitchForkNormalForm(ODEEquations):
    def __init__(self,r,sign=-1):
        super(PitchForkNormalForm, self).__init__()
        self.r=r
        self.sign=sign

    def define_fields(self):
        self.define_ode_variable("x")

    def define_residuals(self):
        x,x_test=var_and_test("x") #Shortcut to get var("x") and testfunction("x")
        self.add_residual((partial_t(x)-(self.r*x+self.sign*x**3))*x_test)


class PitchForkProblem(Problem):
    def __init__(self):
        super(PitchForkProblem, self).__init__()
        #Bind r to a global parameter with initial value 1
        self.r=self.define_global_parameter(r=1) 
		  # also introduce a sign parameter to switch super and subcritial
        self.sign=self.define_global_parameter(sign=-1)
        self.x0=1

    def define_problem(self):
        eq=PitchForkNormalForm(r=self.r,sign=self.sign)
        eq+=InitialCondition(x=self.x0)
        eq+=ODEFileOutput()
        self.add_equations(eq@"pitchfork")


if __name__=="__main__":
    with PitchForkProblem() as problem:

        # Find start solution
        problem.r.value=1
        problem.get_ode("pitchfork").set_value(x=1)
        problem.solve()
        problem.output()

        # Initialize ds (the first step is in direction of the parameter r, i.e. we decrease r first)
        ds=-0.02

        # File to write the parameter r, the value of x and the eigenvalue
        fold_with_eigen_file=open(os.path.join(problem.get_output_directory(),"super_with_eigen_1.txt"),"w")
        # Function to write the current state into the file
        def write_to_eigen_file():
            eigenvals,eigenvects=problem.solve_eigenproblem(1)
            line=[problem.r.value,problem.get_ode("pitchfork").get_value("x",as_float=True),eigenvals[0].real,eigenvals[0].imag]
            fold_with_eigen_file.write("\t".join(map(str,line))+"\n")
            fold_with_eigen_file.flush()

        write_to_eigen_file() # write the first state



        # Important. Otherwise it will be to small at the birfurcation
        problem.set_arc_length_parameter(scale_arc_length=False)

        while problem.get_ode("pitchfork").get_value("x",as_float=True)>-1 and problem.r.value>-1:
            ds=problem.arclength_continuation(problem.r,ds,max_ds=0.01)
            problem.output()
            write_to_eigen_file() # write the updated state


        # Start over at x=0, r=1
        problem.reset_arc_length_parameters()
        problem.get_ode("pitchfork").set_value(x=0)
        problem.r.value=1
        problem.solve()

        # New output file for the second branch
        fold_with_eigen_file = open(os.path.join(problem.get_output_directory(), "super_with_eigen_2.txt"), "w")
        write_to_eigen_file()  # write the first state
        ds = -0.02
        while problem.r.value > -1:
            ds = problem.arclength_continuation(problem.r, ds, max_ds=0.005)
            problem.output()
            write_to_eigen_file()  # write the updated state


        # Start over at x=0, r=1, but flip sign
        problem.reset_arc_length_parameters()
        problem.get_ode("pitchfork").set_value(x=0)
        problem.r.value=1
        problem.sign.value=1
        problem.solve()

        # New output file for the second branch
        fold_with_eigen_file = open(os.path.join(problem.get_output_directory(), "sub_with_eigen_1.txt"), "w")
        write_to_eigen_file()  # write the first state
        ds = -0.02
        while problem.r.value > -1:
            ds = problem.arclength_continuation(problem.r, ds, max_ds=0.005)
            problem.output()
            write_to_eigen_file()  # write the updated state


        problem.reset_arc_length_parameters()
        problem.get_ode("pitchfork").set_value(x=1)
        problem.r.value=-1
        problem.solve()

        # New output file for the second branch
        fold_with_eigen_file = open(os.path.join(problem.get_output_directory(), "sub_with_eigen_2.txt"), "w")
        write_to_eigen_file()  # write the first state
        ds = 0.02
        while problem.r.value >= -1:
            ds = problem.arclength_continuation(problem.r, ds, max_ds=0.005)
            problem.output()
            write_to_eigen_file()  # write the updated state





