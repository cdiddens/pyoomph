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
 
 
from .. import *
from ..expressions import *


class HelmholtzEquation(Equations):
    """
        Represents the Helmholtz equation, which is a second order elliptic partial differential equation given by:

            laplace(u) + k^2 u = 0
        
        where u is the unknown field, k is a constant and laplace is the Laplace operator.

        Args:
            name (str): Name of the unknown field.
            k (ExpressionOrNum): Wavenumber.
            complex (bool): If True, the equation is complex.
            space (FiniteElementSpaceEnum): Finite element space.
            coeff (ExpressionOrNum): Coefficient of the unknown field.
            test_coeff (ExpressionOrNum): Coefficient of the test field.
    """
    def __init__(self,name:str="u",k:ExpressionOrNum=1,complex:bool=False,space:FiniteElementSpaceEnum="C2",coeff:ExpressionOrNum=1,test_coeff:ExpressionOrNum=1):
        super(HelmholtzEquation, self).__init__()
        self.name=name
        self.complex=complex
        self.space:FiniteElementSpaceEnum=space
        self.k=k
        self.coeff=coeff
        self.test_coeff=test_coeff

    def define_fields(self):
        def def_field(n:str):
            self.define_scalar_field(n, self.space, testscale=scale_factor("spatial") ** 2 / scale_factor(self.name))
        if self.complex:
            def_field(self.name + "_Re")
            def_field(self.name + "_Im")
        else:
            def_field(self.name)

    def define_residuals(self):
        if self.complex:
            I=imaginary_i
            uR, uRtest = var_and_test(self.name+"_Re")
            uI, uItest = var_and_test(self.name + "_Im")
            u=uR+I*uI
            utest = uRtest + I * uItest
        else:
            u,utest=var_and_test(self.name)
        eq=weak(contract(self.coeff,grad(u)),contract(self.test_coeff,grad(utest)))-self.k**2*weak(u,utest)

        if self.complex:
            self.add_residual(real_part(eq)+imag_part(eq))
        else:
            self.add_residual(eq)