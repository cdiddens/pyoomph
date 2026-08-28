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


# Import pyoomph and operators like grad
from pyoomph import *
from pyoomph.expressions import *
# Import the bifurcation GUI
from pyoomph.utils.bifurcation_gui import BifurcationGUI
# Plot the fields themselves beside the bifurcation diagram
from pyoomph.output.plotting1d import MatplotlibPlotter1D




class LubricationEquations(Equations):
    def __init__(self,W,D1,D2,r,beta1,beta2,beta_mu,delta,space="C1"):
        super().__init__()
        self.space=space         
        self.W,self.D1,self.D2,self.r,self.beta1,self.beta2,self.beta_mu,self.delta=W,D1,D2,r,beta1,beta2,beta_mu,delta
        
    def define_fields(self):
        # Define the required fields
        for field in ["h","laplh","Gamma1","Gamma2"]:
            self.define_scalar_field(field,self.space)        
        
    def define_residuals(self):
        # Define the symbolical residuals
        h,htest=var_and_test("h")
        laplh,laplhtest=var_and_test("laplh")
        Gamma1,Gamma1test=var_and_test("Gamma1")
        Gamma2,Gamma2test=var_and_test("Gamma2")
        
        # Define some fluxes. Since they appear multiple times, we wrap them in subexpressions.
        # Thereby, the expressions and deriatives thereof are only calculated once and reused, which speeds it up.
        Jp=subexpression(grad(self.W*(1/h**3-1/h**6)-laplh))
        jr=subexpression(self.r*(self.delta*Gamma1**2*Gamma2-self.delta**3*Gamma1**3))
        
        # calculate laplh=laplacian(h)
        self.add_weak(laplh,laplhtest).add_weak(grad(h),grad(laplhtest))
        
        # Height evolution equation
        self.add_weak(partial_t(h),htest).add_weak(h**3/3*Jp+h**2/2*grad(Gamma1+Gamma2),grad(htest))
                        
        # Surface equations
        self.add_weak(partial_t(Gamma1),Gamma1test)\
            .add_weak(h**2/2*Gamma1*Jp + (h*Gamma1+self.D1)*grad(Gamma1)+h*Gamma1*grad(Gamma2),grad(Gamma1test))\
            .add_weak(-jr,Gamma1test).add_weak(self.beta1*log(Gamma1*self.delta)+self.beta_mu,Gamma1test)
            
        self.add_weak(partial_t(Gamma2),Gamma2test)\
            .add_weak(h**2/2*Gamma2*Jp + h*Gamma2*grad(Gamma1) + (h*Gamma2+self.D2)*grad(Gamma2),grad(Gamma2test))\
            .add_weak(jr,Gamma2test).add_weak(self.beta2*log(Gamma2/self.delta)-self.beta_mu,Gamma2test)
                    
    
    
class LubricationPlotter(MatplotlibPlotter1D):
    """h on the left axis, the two surfactant coverages on the right one.

    They differ by more than a decade, so one scale would flatten the coverages onto the axis.
    """
    def define_plot(self):        
        # Fixed ranges for the normal plot, auto ranges for the eigenvector plot
        is_eigen=self.eigenvector is not None
        ranges={} if is_eigen else dict(ymin=0,ymax=25,y2min=0,y2max=1.5) 
        self.set_axes(xlabel="$x$",ylabel="$h$",y2label="$\\Gamma_1,\\ \\Gamma_2$",
                      grid=True,legend=True,legend_position="upper right",
                      # Room on the right for the second axis and its label.
                      margins=(0.10,0.13,0.90,0.95),
                      # Colour each label like its curves; with two scales on one graph there is
                      # otherwise nothing saying which axis belongs to which line.
                      ylabel_color="navy",y2label_color="darkred",
                      **ranges)
        self.add_plot("domain/h",color="navy",linewidth=2,label="$h$")
        self.add_plot("domain/Gamma1",color="darkred",linewidth=2,linestyle="dashed",
                      use_y2=True,label="$\\Gamma_1$")
        self.add_plot("domain/Gamma2",color="darkorange",linewidth=2,linestyle="dotted",
                      use_y2=True,label="$\\Gamma_2$")


class LubricationProblem(Problem):
    def __init__(self):
        super().__init__()        
        self.W, self.delta, self.r = 3, 1, 0.8
        self.D1,self.D2 = 1.4, 0.1
        self.beta1, self.beta2 = 1, 0.5        
        self.beta_mu=self.define_global_parameter(beta_mu=1)         
        # Fine diagram settings
        #self.space="C2"
        #self.N=500
        # Periodic orbit settings
        self.space="C1"
        self.N=85
        self.L=100
        self.V=816
        self.fix_in_center=True # Fix <x*h>=0 by a translation velocity
        
                        
    def define_problem(self):        
        self+=LineMesh(size=self.L,minimum=-self.L/2,N=self.N)
        
        eqs=TextFileOutput()        
        eqs+=LubricationEquations(self.W,self.D1,self.D2,self.r,self.beta1,self.beta2,self.beta_mu,self.delta,self.space)
        eqs+=PeriodicBC("right",offset=[self.L])@"left"        
        eqs+=IntegralConstraint(h=self.V,only_for_stationary_solve=True) # Fix the volume for stationary solutions
        
        # Some reasonable initial conditions
        eqs+=InitialCondition(Gamma1=0.6,Gamma2=2) # Some guesses        
        xsqr=dot(var("coordinate"),var("coordinate"))
        eqs+=InitialCondition(h=maximum(20*(1-xsqr/(30**2)),1))
                
        # Calculate the integral of h**2 and then divide by L and take the sqrt
        eqs+=IntegralObservables(_h_sqr_int=var("h")**2,hsqrnorm=lambda _h_sqr_int: square_root(_h_sqr_int/self.L))
        # Also write it to a file
        eqs+=IntegralObservableOutput()
        
        if self.fix_in_center:
            # Now we add a constraint for h'(0)=0 and we move in the co-moving frame
            # Lagrange multiplier is the velocity U
            U,Utest=self.add_global_dof("U")
            # U is determined by <x*h>=0
            eqs+=WeakContribution(var("coordinate_x")*var("h"),Utest)
            # And we must correct for the shift in the time derivatives
            eqs+=WeakContribution(U*partial_x(var("h")),testfunction("h"))
            eqs+=WeakContribution(U*partial_x(var("Gamma1")),testfunction("Gamma1"))
            eqs+=WeakContribution(U*partial_x(var("Gamma2")),testfunction("Gamma2"))
        
        self+=eqs@"domain"

        self.plotter=LubricationPlotter(self)



if __name__=="__main__":
    # Create the problem
    problem=LubricationProblem()
    
    # Branch switchting etc requires the analytic Hessian, so we must set it up for stability analysis
    problem.setup_for_stability_analysis(analytic_hessian=True)            
    # Create the GUI for the bifurcation analysis
    # Select the parameter to vary: here beta_mu
    gui=BifurcationGUI(problem,"beta_mu")
    # Select the default observable to plot: here the integral of h**2 added by the IntegralObservables
    gui.set_initial_observable("domain/hsqrnorm")
    gui.set_initial_view(1.2525,1.31,11.05,11.29)
    # Eigen-settings: number of eigenvalues to compute and shift for the shift-invert method
    gui.neigen=50
    gui.shift=1
    
    # The first time, we must construct an initial starting point
    # We do so by first solving and then continuing in beta_mu
    if gui.must_init():        
        problem.solve()
        problem.go_to_param(beta_mu=1.255)
    # If you have restarted the GUI, the block above is skipped.
    # Finally, start the GUI with an initial parameter step (beta_mu is increased by 0.005 in the first step)
    gui.start(0.002)
    
    