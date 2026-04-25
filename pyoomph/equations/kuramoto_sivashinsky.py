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
 
 
from ..generic import Equations
from ..expressions import *  # Import grad et al


class KuramotoSivashinskyEquations(Equations):
	"""
		Represents the Kuramoto-Sivashinsky equation, which is a fourth order partial differential equation given by:

			dh/dt = b*h + c*h^2 + a1*curv - a2*laplace(curv) + a3*|grad(h)|^2
			curv = laplacian(h)
	
		where h is the height field, curv is the curvature, a1, a2, a3, b and c are constants, laplace is the Laplace operator and grad is the gradient of h.

		Args:
			a1 (ExpressionOrNum): Coefficient of the (anti)-diffusive term.
			a2 (ExpressionOrNum): Coefficient of the fourth order term.
			a3 (ExpressionOrNum): Coefficient of the KS-nonlinearity.
			b (ExpressionOrNum): Coefficient of h.
			c (ExpressionOrNum): Coefficient of h^2.
			space (FiniteElementSpaceEnum): Finite element space.
			curvspace (FiniteElementSpaceEnum): Finite element space for the curvature.
			swap_test_functions: Swap the test functions of height and curvature.
	"""
		
	def __init__(self,*,a1:ExpressionOrNum=-1,a2:ExpressionOrNum=-1,a3:ExpressionOrNum=1,b:ExpressionOrNum=0,c:ExpressionOrNum=0,space:FiniteElementSpaceEnum="C2",curvspace:Optional[FiniteElementSpaceEnum]=None,swap_test_functions:bool=False,fieldname:str="height",curvfieldname:str="curvature"):
		super().__init__() #Really important, otherwise it will crash
		self.a1=a1
		self.a2=a2
		self.a3=a3
		self.b=b
		self.c=c
		self.space:FiniteElementSpaceEnum=space
		self.curvspace:FiniteElementSpaceEnum=curvspace if curvspace is not None else self.space
		self.fieldname=fieldname
		self.curvfieldname=curvfieldname
		self.swap_test_functions=swap_test_functions  

	def define_fields(self):
		self.define_scalar_field(self.fieldname,space=self.space)
		self.define_scalar_field(self.curvfieldname,space=self.curvspace)

	def define_residuals(self):
		h,h_test=var_and_test(self.fieldname)
		curv,curv_test=var_and_test(self.curvfieldname)
		if self.swap_test_functions:
			h_test,curv_test=curv_test,h_test
		self.add_residual( weak(partial_t(h) - self.b*h - self.c*h**2 - self.a1*curv - self.a3*dot(grad(h),grad(h)), h_test) + self.a2*weak(grad(curv),grad(h_test)) )
		self.add_residual( weak(curv,curv_test) + weak(grad(h),grad(curv_test)) )


class KuramotoSivashinskyBoundary(Equations):
	"""
		Represents the Neumann boundary conditions for the Kuramoto-Sivashinsky equation, given by:

			dot(grad(h),n) = 0
	"""
	def define_residuals(self):
		peqs=self.get_parent_domain().get_equations().get_equation_of_type(KuramotoSivashinskyEquations,always_as_list=True)
		if len(peqs)!=1:
			raise ValueError("KuramotoSivashinskyBoundary requires exactly one KuramotoSivashinskyEquations in the parent domain")

		hbulk, _ = var_and_test(peqs[0].fieldname, domain=self.get_parent_domain())
		_, curv_test = var_and_test(peqs[0].curvfieldname)
		n = self.get_normal()
		self.add_residual(-weak(dot(n,grad(hbulk)),curv_test))
