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
 
 
from ..generic import ODEEquations
from ..expressions import var,testfunction,partial_t,ExpressionOrNum,scale_factor
from ..typings import *

class HarmonicOscillator(ODEEquations):
	"""
	Represents a harmonic oscillator defined by the second-order ordinary differential equation (ODE):
 
	.. math::
	
		\\partial_t^2 y + 2\\delta \\partial_t y + \\omega^2 y = f(t)
	
	where :math:`y` is the dependent variable, :math:`t` is the independent variable (time), :math:`\\delta` is the damping coefficient,
	and :math:`\\omega` is the angular frequency. f(t) is an optional forcing term.
 
 	Args:
		omega: The angular frequency of the harmonic oscillator. Default is 1.
		damping: The damping coefficient of the harmonic oscillator. Default is 0.
		driving: Driving term f
		name: The name of the dependent variable. Default is "y".
		first_derivative_name: The name of the first derivative of the dependent variable. Default is None, meaning that the equation is a second-order ODE.
	"""
	def __init__(self,*,omega:ExpressionOrNum=1,damping:ExpressionOrNum=0,driving:ExpressionOrNum=0,name:str="y",first_derivative_name:Optional[str]=None):
		super(HarmonicOscillator,self).__init__()
		self.omega=omega
		self.damping=damping
		self.driving=driving
		self.name=name
		self.first_derivative_name=first_derivative_name
		
	def define_fields(self):
		self.define_ode_variable(self.name,testscale=scale_factor("temporal")**2/scale_factor(self.name))
		if self.first_derivative_name is not None:
			self.define_ode_variable(self.first_derivative_name,testscale=scale_factor("temporal")/scale_factor(self.name))
		
		
	def define_residuals(self):
		y=var(self.name)
		y_test=testfunction(self.name)
		if self.first_derivative_name is None:
			EQ_y=partial_t(y,2)+2*self.damping*partial_t(y)+self.omega**2 *y-self.driving
			self.add_residual(EQ_y*y_test)
		else:
			yp=var(self.first_derivative_name)
			yptest = testfunction(self.first_derivative_name)
			EQ_y = partial_t(yp) + 2 * self.damping * yp + self.omega ** 2 * y
			EQ_yp = partial_t(y) - yp
			self.add_residual(EQ_y * y_test+EQ_yp*yptest)

	
