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


from wave_eq import * # Import the wave equation from the previous example
from pyoomph.meshes.simplemeshes import CircularMesh # Import the circle mesh

# Required for Bessel functions
import scipy.special


# Expose the Bessel function from scipy to pyoomph
class BesselJ(CustomMathExpression):
	def __init__(self,m):
		super(BesselJ,self).__init__()
		self.m=m # index of the Bessel function
		
	def eval(self,arg_arry):
		return scipy.special.jv(self.m,arg_arry[0]) # Return the scipy result
		
			
class WaveProblemCircularMesh(Problem):
	def __init__(self):
		super(WaveProblemCircularMesh, self).__init__()
		self.c=1 # speed
		self.R=10 # domain length
		self.m=3 # angular mode				
		self.alpha=1 # coefficient of cos
		self.beta=0 # coefficient of sin
		self.radial_amplitudes=[1,-0.4,0.8] # radial amplitudes of R_mn
		
	def define_problem(self):
		self.add_mesh(CircularMesh(radius=self.R)) # Circular mesh
		
		eqs=WaveEquation(self.c) # equation
		eqs+=MeshFileOutput() # output
		eqs+=DirichletBC(u=0)@"circumference" # fixed knots at the rim
		eqs+=RefineToLevel(4) # the CircularMesh is by default coarse, refine it 4 times
			
		# Initial condition
		x, y = var(["coordinate_x","coordinate_y"]) # Cartesian coordinates
		r, theta = square_root(x**2+y**2), atan2(y,x)  # polar coordinates
		J_m=BesselJ(self.m) # bind the Bessel function with integer index m
		bessel_roots=scipy.special.jn_zeros(self.m, len(self.radial_amplitudes)) # get the Bessel roots lambda_mn
				
		Theta=self.alpha*cos(self.m*theta)+self.beta*sin(self.m*theta) # angular solution	
		R=sum([A*J_m(r*lambd/self.R) for A,lambd in zip(self.radial_amplitudes,bessel_roots)]) # radial solution
		eqs+=InitialCondition(u=Theta*R) # setting the initial condition
		
		self.add_equations(eqs@"domain") # adding the equation

		
if __name__=="__main__":
	with WaveProblemCircularMesh() as problem:
		problem.run(20,outstep=True,startstep=0.1)
