#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
#
#  @section LICENSE
#
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt
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
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================

from pyoomph import *
from pyoomph.expressions import *

from pyoomph.output.plotting import MatplotlibPlotter

class PMLPlotter(MatplotlibPlotter):
    def define_plot(self):
        pr=cast(HelmholtzProblem, self.get_problem())
        S=pr.L+pr.d_PML
        self.set_view(-S,-S,S,S)  # Set the view limits
        cb_u=self.add_colorbar("u", position="top center")
        cb_u.invisible=True
        self.add_plot("domain/u",colorbar=cb_u)
        self.add_plot("domain",mode="outlines")

# Mesh with a circular hole and optionally PML boundary regions
class RectangularMeshWithHoleAndPMLBoundary(GmshTemplate):
    def define_geometry(self):
        pr=cast(HelmholtzProblem,self.get_problem())
        self.mesh_mode="tris"
        self.default_resolution=0.2
        self.create_circle_lines(self.point(0,0),pr.a,mesh_size=0.1,line_name="circle")
        bnds=self.create_lines(self.point(-pr.L,-pr.L),"bottom",self.point(pr.L,-pr.L),"right",self.point(pr.L,pr.L),"top",self.point(-pr.L,pr.L),"left")
        self.plane_surface(*bnds,holes=[["circle"]], name="domain")
        # Create the PML boundary regions
        if pr.use_PML:
            corner_lines=[]
            # Iterate over the boundary lines and create PML regions by sort of extrusion
            for b in bnds:
                p1=b.points[0] # the first point of the boundary line
                p2=b.points[1] # the second point of the boundary line
                t=(numpy.array(p2.x)-numpy.array(p1.x))/(2*pr.L) # tangent vector
                p3=self.point(p1.x[0]+t[1]*pr.d_PML, p1.x[1]-t[0]*pr.d_PML) # extruded points
                p4=self.point(p2.x[0]+t[1]*pr.d_PML, p2.x[1]-t[0]*pr.d_PML)
                corner_lines.append([self.line(p1,p3),self.line(p2,p4)]) # store the normal lines for the corners later
                # Transfinite mesh lines for the PML regions
                self.make_lines_transfinite(*(corner_lines[-1]),numnodes=pr.N_PML,coeff=pr.mesh_coeff_PML)
                surf=self.plane_surface(b,corner_lines[-1][0],self.line(p3,p4,name="PML_outer"),corner_lines[-1][1], name="domain")
                # and transfinite surfaces with quads
                self.make_surface_transfinite(surf,corners=[p1,p2,p3,p4])
                self.set_recombined_surfaces(surf)
                
            # add the corner regions
            for i,b in enumerate(bnds):
                pcx=numpy.array(b.points[0].x) # inner corner point
                pcx+=numpy.sign(pcx)*pr.d_PML 
                pc=self.point(pcx[0], pcx[1]) # add an outer corner node
                l3=self.line(corner_lines[i-1][1].points[1],pc,name="PML_outer") # connect the outer corner
                l4=self.line(corner_lines[i][0].points[1],pc,name="PML_outer")
                # and again transfinite regions here
                self.make_lines_transfinite(l3,l4,numnodes=pr.N_PML,coeff=pr.mesh_coeff_PML)
                surf=self.plane_surface(corner_lines[i-1][1],corner_lines[i][0],l3,l4, name="domain")
                self.make_surface_transfinite(surf,corners=[corner_lines[i-1][1].points[1],pc,corner_lines[i][0].points[1],corner_lines[i][0].points[0]])
                self.set_recombined_surfaces(surf)
                

        

class HelmholtzEquation(Equations):
    def __init__(self, k,gamma_x=1, gamma_y=1):
        super().__init__()
        self.k = k # wavenumber
        self.gamma_x = gamma_x # PML complex coordinate transformation coefficient in x-direction
        self.gamma_y = gamma_y # PML complex coordinate transformation coefficient in y-direction
        
    def has_PML(self):
        # Check if PML is used, i.e., if the gamma_x or gamma_y are not equal to 1
        return self.gamma_x != 1 or self.gamma_y != 1
        
    def define_fields(self):
        # Define the scalar fields for the Helmholtz equation
        self.define_scalar_field('u', 'C2')
        if self.has_PML():
            # If PML is used, define an additional scalar field for the imaginary part
            self.define_scalar_field('u_Im', 'C2')
        
    def define_residuals(self):        
        u,v=var_and_test("u")        
        if not self.has_PML():
            # Standard Helmholtz equation without PML
            self.add_weak(grad(u),grad(v)).add_weak(-self.k**2 * u, v)
        else:
            # Helmholtz equation with PML, only works for 2D Cartesian coordinates like here
            if self.get_nodal_dimension()!=2 or self.get_coordinate_system().get_id_name() != "Cartesian":
                raise ValueError("PML Helmholtz equations only implemented for Cartesian 2D problems")
            
            uIm,vIm=var_and_test("u_Im")
            # complex field and test function
            U,V=u+imaginary_i()*uIm,v+imaginary_i()*vIm
            # scaled gradient 
            mygrad=lambda f: vector(self.gamma_y/self.gamma_x*grad(f)[0], self.gamma_x/self.gamma_y*grad(f)[1])
            # complex weak form
            R=weak(mygrad(U),grad(V))-weak(self.k**2 * U, V*self.gamma_x*self.gamma_y) 
            # add real and imaginary parts separately
            self.add_residual(real_part(R)+imag_part(R))
        
        
class HelmholtzProblem(Problem):
    def __init__(self):
        super().__init__()
        self.k = square_root(50) # wavenumber
        self.a=0.2 # radius of the circle
        self.L=2 # half the side length of the square domain
        self.use_PML=True # use PML or not
        self.N_PML=5 # number of nodes in the PML region
        self.d_PML=0.2 # thickness of the PML region
        self.mesh_coeff_PML=1 # node placement coefficient for the PML region
        
    def define_problem(self):
        self+=RectangularMeshWithHoleAndPMLBoundary()
        
        if self.use_PML:
            x,y=var(["coordinate_x","coordinate_y"])            
            # Inverse PML distance coefficients. Diverge at the far boundary
            sigma_x=subexpression(maximum(1/((self.L+self.d_PML)-x),1/((self.L+self.d_PML)+x)))
            sigma_y=subexpression(maximum(1/((self.L+self.d_PML)-y),1/((self.L+self.d_PML)+y)))
            # PML complex coordinate transformations, only active in the PML region by the indicator functions
            gamma_x=1+var("PML_indicator_x")*imaginary_i()/self.k*sigma_x
            gamma_y=1+var("PML_indicator_y")*imaginary_i()/self.k*sigma_y            
            # Indicator functions for PML, elementally constant, set to 1 in PML region, 0 in physical domain
            pml_eqs=ScalarField("PML_indicator_x", "D0")+DirichletBC(PML_indicator_x=(heaviside(absolute(x)-self.L)))
            pml_eqs+=ScalarField("PML_indicator_y", "D0")+DirichletBC(PML_indicator_y=(heaviside(absolute(y)-self.L)))                                    
            pml_eqs+=DirichletBC(u_Im=0)@"circle"
            pml_eqs+=DirichletBC(u=0,u_Im=0)@"PML_outer"
        else:
            pml_eqs=0 # No PML equations if not used
            gamma_x, gamma_y=1, 1 # No PML scaling factors if not used
            
        eqs=HelmholtzEquation(self.k,gamma_x,gamma_y)       
        eqs+=MeshFileOutput()
        eqs+=DirichletBC(u=0.1)@"circle"
        eqs+=pml_eqs # Add PML equations if used
                
        self+=eqs@"domain"
        
        
with HelmholtzProblem() as problem:
    problem+=PMLPlotter(fileext="pdf")  # Add the PML plotter
    problem.solve()    
    problem.output()