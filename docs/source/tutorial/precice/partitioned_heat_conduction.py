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


from pyoomph import *
from pyoomph.expressions import *

# Import the preCICE adapter of pyoomph
from pyoomph.solvers.precice_adapter import *

# Heat conduction equation with a source term
class HeatEquation(Equations):
    def __init__(self,f):
        super().__init__()
        self.f=f
        
    def define_fields(self):
        self.define_scalar_field("u","C2")
        
    def define_residuals(self):
        u,v=var_and_test("u")
        self.add_weak(partial_t(u),v).add_weak(grad(u),grad(v)).add_weak(-self.f,v)


# Generic heat conduction problem. Can be run without preCICE on the full domain or as Dirichlet or Neumann participant
class HeatConductionProblem(Problem):
    def __init__(self):
        super().__init__()
        self.alpha=3 # Parameters
        self.beta=1.2    
        # Config file
        self.precice_config_file="precice-config.xml"   
        
    def get_f(self):
        # Source term
        return self.beta-2-2*self.alpha
    
    def get_u_analyical(self):
        # Analytical solution
        return 1+var("coordinate_x")**2+self.alpha*var("coordinate_y")**2+self.beta*var("time")
        
    def define_problem(self):
        # Depending on the participant, set up the coupling equations
        
        # First of all, we must provide the mesh at the coupling interface to preCICE
        # If we run without preCICE, this equation part is not used, so it will be just discarded
        coupling_eqs=PreciceProvideMesh(self.precice_participant+"-Mesh")
        if self.precice_participant=="Dirichlet":
            # Dirichlet participant: We take the left part and use the right boundary as coupling boundary
            x_offset,box_length=0,1
            coupling_boundary="right"
            # Here we write the heat flux and read the temperature
            # Note that we cannot use PreciceWriteData(Heat-Flux=partial_x(var("u",domain=".."))) 
            # because "Heat-Flux" is not a valid keyword argument. Therefore, we use the **{...} syntax
            # It will calculate the gradient of u in x-direction and write it to the preCICE data "Heat-Flux"
            coupling_eqs+=PreciceWriteData(**{"Heat-Flux":partial_x(var("u",domain=".."))})
            # It reads the preCICE field "Temperature" and stores it in the field "uD"
            coupling_eqs+=PreciceReadData(uD="Temperature")            
            # We cannot set a Dirichlet boundary condition depending on a variable, so we use an enforced boundary condition
            # We adjust u so that u-uD=0 at the coupling boundary. This is equivalent to setting u=uD
            coupling_eqs+=EnforcedBC(u=var("u")-var("uD"))
        elif self.precice_participant=="Neumann":
            # The Neuamnn participant is on the right side and uses the left boundary as coupling boundary
            x_offset,box_length=1,1
            coupling_boundary="left"
            # It writes the preCICE field "Temperature" by evaluating the variable u
            coupling_eqs+=PreciceWriteData(Temperature=var("u"))
            # It reads the preCICE field "Heat-Flux" and stores it in the field "flux"
            coupling_eqs+=PreciceReadData(flux="Heat-Flux")            
            # This flux is used as Neumann boundary condition. 
            coupling_eqs+=NeumannBC(u=var("flux"))
        elif self.precice_participant=="":
            # If we run without preCICE, we use the full domain and have no coupling boundary
            x_offset=0
            box_length=2
            coupling_boundary=None
        else:
            raise Exception("Unknown participant. Choose 'Dirichlet', 'Neumann' or an empty string")
        
        # Create the corresponding mesh
        N0=11
        self+=RectangularQuadMesh(size=[box_length,1],N=[box_length*N0,N0],lower_left=[x_offset,0])
        
        # Assemble the base equations
        eqs=MeshFileOutput()
        eqs+=HeatEquation(self.get_f())
        eqs+=InitialCondition(u=self.get_u_analyical())
        
        # Add the coupling equations and the Dirichlet boundary conditions
        # All potential Dirichlet boundaries        
        dirichlet_bounds=set(["bottom","top","left","right"])
        if coupling_boundary:
            # Of course, the coupling boundary must not be set as Dirichlet boundary
            dirichlet_bounds.remove(coupling_boundary)        
            eqs+=coupling_eqs@coupling_boundary
        eqs+=DirichletBC(u=self.get_u_analyical())@dirichlet_bounds            
        
        # Calculate the error
        eqs+=IntegralObservables(error=(var("u")-self.get_u_analyical())**2)
        eqs+=IntegralObservableOutput()
                            
        self+=eqs@"domain"
        
        
if __name__=="__main__":
    problem=HeatConductionProblem()
    problem.initialise() # After this, precice_participant could have been set via command line, e.g. by  -P precice_participant=Neumann
    if problem.precice_participant=="":
        # Just run it manually without preCICE
        problem.run(1,outstep=0.1)
    else:
        # Run it with preCICE. Time stepping is taken from the config file
        problem.precice_run()