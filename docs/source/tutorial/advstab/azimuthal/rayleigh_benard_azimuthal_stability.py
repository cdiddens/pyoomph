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
from pyoomph.equations.navier_stokes import * # Navier-Stokes for the flow
from pyoomph.equations.advection_diffusion import * # Advection-diffusion for the temperature
from pyoomph.utils.num_text_out import * # Output for the critical Rayleigh as function of the aspect ratio


class RBConvectionProblem(Problem):
    def __init__(self):
        super().__init__()
        # Aspect ratio, Rayleigh and Prandtl number with defaults
        self.Gamma = self.define_global_parameter(Gamma=1)  
        self.Ra = self.define_global_parameter(Ra=1)  
        self.Pr =self.define_global_parameter(Pr=1)   
                
    def define_problem(self):        
        # Axisymmetric coordinate system
        self.set_coordinate_system(axisymmetric)
        # Scale radial coordinate with aspect ratio parameter
        self.set_scaling(coordinate_x=self.Gamma)
        # Axisymmetric cross-section as mesh. 
        # We use R=1 and H=1, but due to the radial scaling, we can modify the effective radius
        self+=RectangularQuadMesh(size=[1, 1], N=20)

        RaPr=self.Ra*self.Pr # Shortcut for Ra*Pr
        # Equations: Navier-Stokes. We scale the pressure also with RaPr, 
        # so that the hydrostatic pressure due to the bulk-force is independent on the value of Ra*Pr
        NS=NavierStokesEquations(mass_density=1, dynamic_viscosity=self.Pr,bulkforce=RaPr*var("T")*vector(0, 1), pressure_factor=RaPr)
        # Since u*n is set at all walls, we have a nullspace in the pressure
        # This offset is fixed by an integral constraint <p>=0
        # One could also set it via a DirichletBC(pressure=0) at e.g. a single corner, 
        # but this yields problems in the azimuthal stability analysis then 
        # The pressure integral constraint is automatically deactivated when m!=0, since <p>=0 
        # holds automatically when p = p^(m)*exp(I*m*phi) for m!=0
        eqs = NS.with_pressure_integral_constraint(self,integral_value=0,set_zero_on_normal_mode_eigensolve=True)
        
        # And advection-diffusion for temperature
        eqs += AdvectionDiffusionEquations(fieldnames="T",diffusivity=1, space="C1")

        # Boundary conditions
        eqs += DirichletBC(T=0)@"bottom"
        eqs += DirichletBC(T=-1)@"top"
        # The NoSlipBC will actually also set velocity_phi=0 automatically
        eqs += NoSlipBC()@["top", "right", "bottom"]
        # Here, the magic happens regarding the m-dependent boundary conditions
        eqs += AxisymmetryBC()@"left"

        # Output
        eqs+=MeshFileOutput()

        # Add the system to the problem
        self+=eqs@"domain"


if __name__=="__main__":
    with RBConvectionProblem() as problem:
        # Magic function: It will perform all necessary adjustments, i.e.
        #   -expand fields and test functions with exp(i*m*phi)
        #   -consider phi-components in vector fields, i.e. here velocity
        #   -incorporate phi-derivatives in grad and div
        #   -generate the base residual, Jacobian, mass matrix and Hessian, but also
        #    the corresponding versions for the azimuthal mode m!=0
        problem.setup_for_stability_analysis(azimuthal_stability=True,analytic_hessian=True)

        # Solve once to get the right pressure and temperature field. Velocity can stay zero here
        with problem.select_dofs() as dofs:
            dofs.unselect("domain/velocity_x","domain/velocity_y")
            problem.solve()
        
        # Iterate over all desired modes m
        for m in [0,1,2,3]:
            problem.Gamma.value=0.5 # Start at some aspect ratio
            # Find a good guess for the critical Ra by eigenvalue bisection
            problem.Ra.value=10
            # We increase by steps of 200, but we don't have to solve the system, since the stationary solution is independent of Ra
            # we also pass the mode we want to solve for
            for currentRa,currentEigen in problem.find_bifurcation_via_eigenvalues("Ra",initstep=200,do_solve=False,neigen=4,azimuthal_m=m,epsilon=1e-2):
                print("Currently at Ra=",currentRa,"with eigenvalue",currentEigen)

            # Activate the bifurcation tracking. For mode m=0, we can have fold, pitchfork, etc.
            # For azimuthal modes m!=0, this distinguishment is not easily possible, since everything is complex anymways
            problem.activate_bifurcation_tracking("Ra",bifurcation_type="pitchfork" if m==0 else "azimuthal")

            # Find a solution at the cricical Ra and write output
            problem.solve()
            problem.output_at_increased_time()
            txtout = NumericalTextOutputFile(problem.get_output_directory(f"curve_m_{m}.txt"))
            txtout.header("Gamma", "Ra")
            txtout.add_row(problem.Gamma.value,problem.Ra.value)

            # Arclength continuation in the aspect ratio
            ds_max = 0.05
            ds=ds_max

            while problem.get_global_parameter("Gamma").value < 3.0:
                ds = problem.arclength_continuation("Gamma", ds,max_ds=ds_max)
                # Resetting the arclength parameters. Since the stationary solution does not change at all, this is beneficial.
                # Arclength continuation otherwise monitors how the solution changes, but it does not at all... 
                problem.reset_arc_length_parameters()   
                txtout.add_row(problem.Gamma.value,problem.Ra.value)

            # Deactivate for the next loop
            problem.deactivate_bifurcation_tracking()
