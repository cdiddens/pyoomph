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


from droplet_spread_free_slip import * # Load the problem without slip

class SlipLength(InterfaceEquations):
 	# must be attached to a domain with NavierStokesEquations
	required_parent_type = NavierStokesEquations

	def __init__(self, slip_length):
		super(SlipLength, self).__init__()
		self.slip_length = slip_length # store the slip length

	def define_residuals(self):
		n = var("normal")
		u, utest = var_and_test("velocity")
		utang = u - dot(u, n) * n # tangential velocity
		utest_tang = utest - dot(utest, n) * n # tangential test function
		mu=self.get_parent_equations().dynamic_viscosity # get mu from the parent equations
		factor = mu / (self.slip_length) # add the weak contribution
		self.add_residual(weak(factor * utang, utest_tang))
	
	
# Inherit from the problem without slip
class DropletSpreadingWithSliplength(DropletSpreadingProblem):
	def __init__(self):
		super(DropletSpreadingWithSliplength,self).__init__()
		self.slip_length=0.001 # tiny slip length
		
		
	def define_problem(self):
		super(DropletSpreadingWithSliplength,self).define_problem() # define the old problem
		self.add_equations(SlipLength(self.slip_length)@"domain/substrate") # add a slip length to the substrate
		
		# Refinement
		self.max_refinement_level=6 # level 4 is already the base refinement, allow additional refinment
		self.add_equations(SpatialErrorEstimator(velocity=1)@"domain") # allow for refinement to resolve the strong stresses near the contact line

		
if __name__=="__main__":
	with DropletSpreadingWithSliplength() as problem:
		problem.run(50,outstep=True,startstep=0.25,spatial_adapt=1)	
