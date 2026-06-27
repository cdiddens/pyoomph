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
# Using the predefined Navier-Stokes and ALE equations from pyoomph
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *
# We require the remeshing utities from pyoomph
from pyoomph.meshes.remesher import *
# And the interface zeta coordinate utilities to get smooth interpolation of the interface upon remeshing
from pyoomph.meshes.zeta import *

        
class DomainMeshAxi(RemeshableGmshTemplate2d):                  
    def define_geometry(self):
        self.mesh_mode="tris"
        
        # Setting some GMSH global options 
        pr=cast(RayleighPlateauProblem, self.get_problem())        
        # Never make a finer mesh than the minimum radius threshold would allow
        self.gmsh_options["Mesh.MeshSizeMin"]=pr.epsilon/pr.min_elements_per_radius 
        # Make sure that we cannot go coarser than the maximum elements per radius
        self.gmsh_options["Mesh.MeshSizeMax"]=1.0/pr.max_elements_per_radius 
        # Make sure that we have enough elements to resolve the curvature of the interface
        self.gmsh_options["Mesh.MeshSizeFromCurvature"]=8*pr.min_elements_per_radius
                
        # The end points on the axis are always the same
        p00,p0H=self.point(0,0),self.point(0,pr.L)
        
        if self.is_first_time():            
            # We just create a rectangle as initial mesh and perturb it afterwards
            pR0,pRH=self.point(1,0),self.point(1,pr.L)                        
            interface=self.line(pR0,pRH,name="interface")
        else:     
            # Remeshing: We need the current interface shape
            interface_segments=self.get_boundary_coordinates("liquid/interface",sort_along_axis="y+")                   
            # There is only one segment in the interface, but we still need to create a list of Gmsh points 
            pts=[self.point(x,y) for x,y in interface_segments[0]]
            # We create a spline through the points to define the interface
            interface=self.spline(pts,name="interface")
            # the end points of the interface are just used for the top and bottom corners
            pR0,pRH=pts[0],pts[-1]                        
        
        # Common part: Axis of symmetry and the top and bottom lines and the surface
        axis=self.create_lines(pR0,"bottom",p00,"axisymm",p0H,"top",pRH)[1]
        self.plane_surface("bottom","axisymm", "top","interface",name="liquid")
        
        # Now define mesh size fields of gmsh (see https://gmsh.info/doc/texinfo/#Gmsh-mesh-size-fields)
        
        # First, calculate the distance from the interface to the axis of symmetry.
        # This is just the radius, so we just use a "MathEval" with the x-coordinate. 
        # We divide by the minimum elements per radius to get the desired mesh size.
        dist_interf_to_axis=self.add_mesh_size_field("MathEval",F="x/"+str(pr.min_elements_per_radius))
        # However, this resolution is only meant for the interface, so we restrict it with a "Restrict" field to the interface curve
        restr_interf=self.add_mesh_size_field("Restrict",InField=dist_interf_to_axis,CurvesList=[interface])
        # For the resolution on the axis, we need to calculate the distance from the axis to the interface. 
        # This is done with a "Distance" field, but it is important to increase the "Sampling" parameter here
        axis_to_inter=self.add_mesh_size_field("Distance",CurvesList=[interface],Sampling=1000)
        # Convert the distance to a mesh size using "MathEval". 
        # We can access the value of the distance field with "F<field_id>", where <field_id> is the id of the distance field.
        axis_to_inter_size=self.add_mesh_size_field("MathEval",F="F"+str(axis_to_inter)+"/"+str(pr.min_elements_per_radius*0.75))
        # Again, we restrict this mesh size to the axis of symmetry with a "Restrict" field
        restr_axis=self.add_mesh_size_field("Restrict",InField=axis_to_inter_size,CurvesList=[axis])
        # Now we just take the minimum of the two restricted mesh size fields to get the final mesh size field
        combined=self.add_mesh_size_field("Min",FieldsList=[restr_interf,restr_axis])
        # And tell gmsh to use this one
        self.set_mesh_size_background_field(combined)
                        
            
class RayleighPlateauProblem(Problem):
    def __init__(self):
        super().__init__()
        self.L=4.0 # Length of the domain
        self.Oh=0.07 # Ohnesorge number
        self.k=pi/4 # Wavenumber of the perturbation        
        self.a=0.1 # Amplitude of the perturbation        
        self.min_elements_per_radius=2 # Resolution of the mesh, typical elements per radius (higher means finer mesh)
        self.max_elements_per_radius=4 # Maximum elements per radius (higher means finer mesh)        
        self.epsilon=0.0004 # Minimum radius threshold for stopping the simulation
        
    def define_problem(self):
        self.add_mesh(DomainMeshAxi())
        self.set_coordinate_system("axisymmetric") 
        
        # Equations
        eqs=NavierStokesEquations(mass_density=1,dynamic_viscosity=self.Oh)
        eqs+=HyperelasticSmoothedMesh()
        eqs+=MeshFileOutput()
        # We still must tell pyoomph that it should remesh when necessary
        eqs+=RemeshWhen(RemeshingOptions())
        
        # Initial perturbation
        X,Y=var(["lagrangian_x","lagrangian_y"])
        eqs+=InitialCondition(mesh_x = X*(1+self.a*cos(self.k*Y)))
                   
        # Boundary conditions     
        eqs+=AxisymmetryBC()@"axisymm"
        eqs+=DirichletBC(mesh_y=True)@["top","bottom"]
        eqs+=DirichletBC(velocity_y=0)@["bottom","top"]
        eqs+=NavierStokesFreeSurface(surface_tension=1)@"interface"                        
        
        # Monitor the minimum radius and its position along the interface
        eqs+=ExtremumObservables(min_r=var("mesh_x"))@"interface"
                
        # This improves the interpolation at interfaces upon remeshing
        eqs+=AssignZetaCoordinatesByEulerianCoordinate("x")@"bottom"
        eqs+=AssignZetaCoordinatesByEulerianCoordinate("x")@"top"
        eqs+=AssignZetaCoordinatesByEulerianCoordinate("y")@"axisymm"
        eqs+=AssignZetaCoordinatesByArclength(sort_along_axis="y+")@"interface"                        
                                   
        self.add_equations(eqs@"liquid")

    def get_minimum_radius_and_position(self):
        # Obtain the minimum radius and its position along the interface
        # evaluate_minimum with return_x=True returns a tuple of (min_value, (x,y)), so we unpack it accordingly
        min_r,min_z=self.get_mesh("liquid/interface").evaluate_minimum("min_r",return_x=True)[1]
        return min_r,min_z
    
    def minimum_radius_dependent_run(self,dt_factor=0.1,maxstep=0.1):
        # This is a generator function that yields the current time and adjusts the time step based on the minimum radius
        while True:
            r_min,_=self.get_minimum_radius_and_position()                                    
            if r_min<self.epsilon:                
                return # Stop the run if the minimum radius is below the threshold
            
            dt=dt_factor*r_min # Time step is proportional to the minimum radius

            if dt>maxstep:
                dt=maxstep # Limit the maximum time step to avoid too large steps

            # Run a simulation over the time interval [t, t+dt] and yield the current time after each step
            # We might take multiple steps within this interval if necessary, but we only yield after the full interval
            self.run(self.get_current_time()+dt,outstep=False,maxstep=dt,temporal_error=1)
            yield problem.get_current_time() 
            

if __name__=="__main__":
    with RayleighPlateauProblem() as problem:
        # Control the time step adaptation factors 
        problem.DTSF_max_increase_factor=1.25 # increase dt by 25% for successful steps
        problem.DTSF_min_decrease_factor=0.75 # decrease dt by 25% for failed steps

        problem.initialise()        
        # Monitor the minimum radius and its position along the interface and write to a text file
        minimum_out=problem.create_text_file_output("minimum.txt",header=["t","r_min","z_min"])
        
        for t in problem.minimum_radius_dependent_run():
            # Loop over the generator function that performs runs with minimum radius-dependent time steps
            problem.output()
            minimum_out.add_row(t,*problem.get_minimum_radius_and_position())
