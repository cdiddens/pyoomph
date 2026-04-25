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


from rising_bubble import *
from pyoomph.output.plotting import MatplotlibPlotter

# Define a plotter as usual. We don't have to care about the eigendynamics here 
class RisingBubblePlotter(MatplotlibPlotter):
    def define_plot(self):
        pr=cast("RisingBubbleProblem",self.get_problem())
        self.set_view(-3*pr.R,-5*pr.R,3*pr.R,2*pr.R)
        cb_v=self.add_colorbar("velocity",cmap="viridis",position="top right")
        self.add_plot("domain/velocity",colorbar=cb_v,transform=[None,"mirror_x"])
        self.add_plot("domain/velocity",mode="streamlines",transform=[None,"mirror_x"])
        self.add_plot("domain/interface",transform=[None,"mirror_x"])
        
with RisingBubbleProblem() as problem:        
    # Same as before in the rising bubble problem, except that we just solve at Bo=4
    problem.set_c_compiler("system").optimize_for_max_speed()
    problem.set_eigensolver("slepc").use_mumps()        
    problem.setup_for_stability_analysis(azimuthal_stability=True,analytic_hessian=False)    
    problem.Mo=6.2e-7 
    problem.Bo.value=4 
    m=1 
    lambd=0.1+0.67j 
                        
    problem.run(10,startstep=0.1,outstep=False,temporal_error=1)
    problem.solve(max_newton_iterations=20,spatial_adapt=4)    
    problem.solve_eigenproblem(1,azimuthal_m=m,shift=lambd,target=lambd)
    problem.refine_eigenfunction(use_startvector=True)
    
    # Create an animation of the eigenfunction using the RisingBubblePlotter class
    problem.create_eigendynamics_animation("eigenanim",RisingBubblePlotter(),max_amplitude=1,numperiods=4,numouts=4*25)    
