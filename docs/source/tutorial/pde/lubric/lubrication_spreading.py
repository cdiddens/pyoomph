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


from lubrication import *
		
class DropletSpreading(Problem):	
	def __init__(self):
		super(DropletSpreading,self).__init__()
		self.hp=0.0075 # precursor height
		self.sigma=1 # surface tension
		self.R,self.h_center=1,0.5 # initial radius and height of the droplet
		self.theta_eq=pi/8  # equilibrium contact angle
			
					
	def define_problem(self):
		self.set_coordinate_system("axisymmetric")	
		self.add_mesh(LineMesh(N=500,size=5)) # simple line mesh		
		
		h=var("h") # Building disjoining pressure
		disjoining_pressure=5*self.sigma*self.hp**2*self.theta_eq**2*(h**3 - self.hp**3)/(3*h**6)
		
		eqs=LubricationEquations(sigma=self.sigma,disjoining_pressure=disjoining_pressure) # equations
		eqs+=TextFileOutput() # output	
		h_init=maximum(self.h_center*(1-(var("coordinate_x")/self.R)**2),self.hp) # Initial height
		eqs+=InitialCondition(h=h_init) 
		
		self.add_equations(eqs@"domain") # adding the equation

		
if __name__=="__main__":
	with DropletSpreading() as problem:
		problem.run(1000,outstep=True,startstep=0.01,maxstep=10,temporal_error=1)
