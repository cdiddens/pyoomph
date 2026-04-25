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


from pyoomph import *
from pyoomph.equations.poisson import *

class TwoDomainMesh1d(MeshTemplate):
	def __init__(self,Ntot=100,xI=1,L=2,left_domain_name="domainA",right_domain_name="domainB"):
		super(TwoDomainMesh1d,self).__init__()
		self.Ntot, self.xI, self.L = Ntot, xI, L
		self.left_domain_name,self.right_domain_name=left_domain_name,right_domain_name

	def define_geometry(self):
		xI=self.nondim_size(self.xI)
		L=self.nondim_size(self.L)
		NA=round(self.Ntot*xI/L) # number of elements on domainA calcuated from total number 
		
		domainA=self.new_domain(self.left_domain_name)
		domainB=self.new_domain(self.right_domain_name)
		
		# Generate nodes 
		nodesA=[self.add_node_unique(x) for x in numpy.linspace(0,xI,NA)]
		for x0,x1 in zip(nodesA, nodesA[1:]):
			domainA.add_line_1d_C1(x0,x1)	# and elements by pairs of nodes
			
		# same for domainB
		nodesB=[self.add_node_unique(x) for x in numpy.linspace(xI,L,self.Ntot-NA)]
		for x0,x1 in zip(nodesB, nodesB[1:]):
			domainB.add_line_1d_C1(x0,x1)
			
		# marking boundaries
		self.add_nodes_to_boundary("left",[nodesA[0]])
		self.add_nodes_to_boundary("interface",[nodesB[0]]) # coordsB[0] is acutally = coordsA[-1]
		self.add_nodes_to_boundary("right",[nodesB[-1]])
		

class ConnectTAtInterface(InterfaceEquations):		
	def define_fields(self):
		self.define_scalar_field("lambda","C2") # Lagrange multiplier
		
	def define_residuals(self):
		my_field,my_test=var_and_test("T") # T on the domain where this InterfaceEquations object is attached to
		opp_field,opp_test=var_and_test("T",domain=self.get_opposite_side_of_interface()) # T on the interface, but evaluated in the opposite domain
		lagr,lagr_test=var_and_test("lambda") # Lagrange multiplier
		self.add_residual(weak(my_field-opp_field,lagr_test)) # constraint T_my-T_opp=0
		self.add_residual(weak(lagr,my_test)) # Lagrange Neumann contribution to the inside domain
		self.add_residual(weak(-lagr,opp_test)) # Lagrange Neumann contribution to the outside domain
		
		
class TwoDomainTemperatureConduction(Problem):
	def __init__(self):
		super(TwoDomainTemperatureConduction,self).__init__()
		self.conductivityA=0.5	# thermal conductivity of domain A
		self.conductivityB=2	# thermal conductivity of domain B
		
	def define_problem(self):
		self.add_mesh(TwoDomainMesh1d())
		
		# Assemble equations domainA
		eqsA=TextFileOutput()
		eqsA+=PoissonEquation(name="T",space="C2",coefficient=self.conductivityA,source=0)
		eqsA+=DirichletBC(T=0)@"left"

		# and equations of domainB
		eqsB=TextFileOutput()
		eqsB+=PoissonEquation(name="T",space="C2",coefficient=self.conductivityB,source=0)		
		eqsB+=DirichletBC(T=1)@"right"	
		
		# Interface connection. Must be added to one side of the interface, i.e. alternatively to eqsB
		eqsA+=ConnectTAtInterface()@"interface"
		
		self.add_equations(eqsA@"domainA")
		self.add_equations(eqsB@"domainB")		

		
if __name__=="__main__":
	with TwoDomainTemperatureConduction() as problem:
		problem.solve()
		problem.output()
		

