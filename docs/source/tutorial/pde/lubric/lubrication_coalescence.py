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


from lubrication_spreading import * # Import the previous example problem
		
class DropletCoalescence(DropletSpreading):	
	def __init__(self):
		super(DropletCoalescence,self).__init__()
		self.distance=2.5 # droplet distance
		self.Lx=7.5
		self.max_refinement_level=6
			
					
	def define_problem(self):
		self.add_mesh(RectangularQuadMesh(N=[10,5],size=[self.Lx,self.Lx/2],lower_left=[-self.Lx*0.5,0])) 
		
		h=var("h") # Building disjoining pressure
		disjoining_pressure=5*self.sigma*self.hp**2*self.theta_eq**2*(h**3 - self.hp**3)/(3*h**6)
		
		eqs=LubricationEquations(sigma=self.sigma,disjoining_pressure=disjoining_pressure) # equations
		eqs+=MeshFileOutput() # output	
		x=var("coordinate")
		dist1=x-vector(-self.distance/2,0) # distance to the centers of the droplets
		dist2=x-vector(self.distance/2,0)
		h1=self.h_center*(1-dot(dist1,dist1)/self.R**2) # height functions of the droplets
		h2=self.h_center*(1-dot(dist2,dist2)/self.R**2)		
		h_init=maximum(maximum(h1,h2),self.hp) # Initial height: maximum of h1, h2 and precursor
		eqs+=InitialCondition(h=h_init) 
		
		eqs+=SpatialErrorEstimator(h=1) # refine based on the height field
		
		self.add_equations(eqs@"domain") # adding the equation

		
if __name__=="__main__":
	with DropletCoalescence() as problem:
		problem.run(1000,outstep=True,startstep=0.01,maxstep=10,temporal_error=1,spatial_adapt=1)
