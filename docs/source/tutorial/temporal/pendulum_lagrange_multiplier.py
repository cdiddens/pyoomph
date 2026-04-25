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



from pyoomph import * # Import pyoomph 
from pyoomph.expressions import * # Import some additional things to express e.g. partial_t

#Newtons law of motion in 2d, i.e.
# m*partial_t^2 x = force_x
# m*partial_t^2 y = force_y
class NewtonsLaw2d(ODEEquations):
	def __init__(self,*,mass=1,force_vector=vector([0,-1])):
		super(NewtonsLaw2d,self).__init__()
		self.force_vector=force_vector
		self.mass=mass

	# Here, we use BDF2 time stepping, i.e. we split the system into a 4d system of first order ODEs
	def define_fields(self):
		self.define_ode_variable("x") 
		self.define_ode_variable("y") 		
		self.define_ode_variable("xdot") #partial_t x
		self.define_ode_variable("ydot") #partial_t y
		
	def define_residuals(self):
		x,y=var(["x","y"])
		xdot,ydot=var(["xdot","ydot"])
		# Motion equations
		self.add_residual( (self.mass*partial_t(xdot)-self.force_vector[0])*testfunction(x))
		self.add_residual( (self.mass*partial_t(ydot)-self.force_vector[1])*testfunction(y))
		# Definition of xdot and ydot
		self.add_residual( (partial_t(x)-xdot)*testfunction(xdot))
		self.add_residual( (partial_t(y)-ydot)*testfunction(ydot))

#Pendulum constraint: Enforcing sqrt(x**2+y**2)=L via a Lagrange multiplier
class PendulumConstraint(ODEEquations):
	def __init__(self,*,L=1):
		super(PendulumConstraint,self).__init__()
		self.L=L
		
	def define_fields(self):
		self.define_ode_variable("lambda_pendulum") #Lagrange multiplier
		
	def define_residuals(self):
		x,y,lambda_pendulum=var(["x","y","lambda_pendulum"])
		currentL=square_root(x**2+y**2) #Current length
		currentL=subexpression(currentL) #Wrap it into a subexpression, since it occurs multiple times in the equations
		self.add_residual(lambda_pendulum*x/currentL*testfunction(x)) #additional forces
		self.add_residual(lambda_pendulum*y/currentL*testfunction(y))	
		self.add_residual((currentL-self.L)*testfunction(lambda_pendulum)) #constraint equation to solve for the Lagrange multiplier



class PendulumProblem(Problem):

	def __init__(self):
		super(PendulumProblem,self).__init__() 
		self.gvector=vector([0,-1]) #Default gravity direction, g is assumed to be 1
		self.L=1 #pendulum length
		self.mass=1
	
	def define_problem(self):
		eqs=NewtonsLaw2d(force_vector=self.mass*self.gvector,mass=self.mass)
		eqs+=PendulumConstraint(L=self.L)
		phi0=0.9*pi #Initial phi
		x0=self.L*sin(phi0) #Initial position
		y0=-self.L*cos(phi0)		
		eqs+=InitialCondition(x=x0,y=y0)  #Set the initial position
		eqs+=ODEFileOutput()  #Output
		eqs+=ODEObservables(phi=atan2(var("x"),-var("y"))) #Calculate phi from x and y
		self.add_equations(eqs@"pendulum") 		

if __name__=="__main__":
	with PendulumProblem() as problem:
		# We need many outputs, i.e. a small dt for the time stepping scheme to be nearly energy-conserving
		problem.run(endtime=100,numouts=10000)
