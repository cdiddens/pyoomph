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


from bifurcation_fold_param_change import *

if __name__=="__main__":
    with FoldProblem() as problem:

        # Find start solution
        problem.r.value=1
        problem.get_ode("fold").set_value(x=1)
        problem.solve()
        problem.output()

        # Initialize ds (the first step is in direction of the parameter r, i.e. we decrease r first)
        ds=-0.02

        # File to write the parameter r, the value of x and the eigenvalue
        fold_with_eigen_file=open(os.path.join(problem.get_output_directory(),"fold_with_eigen.txt"),"w")
        # Function to write the current state into the file
        def write_to_eigen_file():
            eigenvals,eigenvects=problem.solve_eigenproblem(1)
            line=[problem.r.value,problem.get_ode("fold").get_value("x",as_float=True),eigenvals[0].real,eigenvals[0].imag]
            fold_with_eigen_file.write("\t".join(map(str,line))+"\n")
            fold_with_eigen_file.flush()

        write_to_eigen_file() # write the first state

        while problem.get_ode("fold").get_value("x",as_float=True)>-1:
            ds=problem.arclength_continuation(problem.r,ds,max_ds=0.005)
            problem.output()
            write_to_eigen_file() # write the updated state


