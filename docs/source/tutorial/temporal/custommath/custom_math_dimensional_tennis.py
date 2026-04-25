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
from pyoomph.expressions.units import *

# Force of a tennis racket as function of the ball position and the racket position
# This will create a function Force(ball_position,racket_position)
# Both positions must have the unit of meter
# And the result is a force measured in Newton
class TennisRacket(CustomMathExpression):
    def __init__(self,*,direction=1,spring_constant=1000*newton/meter):
        super(TennisRacket, self).__init__()
        self.direction=direction # sign of the force
        # Store a non-dimensional value of the spring constant
        self.k_in_N_per_m=float(spring_constant/(newton/meter))

    # Input arguments are converted to numerical values by treating the input as meter (for both arguments, ball and racket position)
    def get_argument_unit(self,index:int):
        return meter # return meter, no matter whether index==0 or index==1

    # The result is obtained by multiplying the result of 'eval' by newton
    def get_result_unit(self):
        return newton

    # This routine is now entirely nondimensional
    def eval(self,arg_array):
        # get the input values (numerical float values!)
        ball_pos_in_m=arg_array[0] # measured int meter
        racket_pos_in_m=arg_array[1] # measured int meter
        # calculate the distance (also in meter)
        distance_in_m=self.direction*(ball_pos_in_m-racket_pos_in_m)
        if distance_in_m>=0: # in front of the racket
            return 0.0 # numerical float value of the force [in newton]
        else:
            # Force of the racket on the ball
            force_in_newton=-self.direction*self.k_in_N_per_m*distance_in_m
            return force_in_newton # result is treated in newton


class NewtonsLaw1d(ODEEquations):
    def __init__(self,mass,force):
        super(NewtonsLaw1d, self).__init__()
        self.mass=mass
        self.force=force

    def define_fields(self):
        # bind the scale factors (defined on problem level)
        T=scale_factor("temporal")
        X=scale_factor("spatial")        
        # we set the scales as well as the test function scales here locally in the equation class
        self.define_ode_variable("x",scale=X,testscale=T**2/X) # same test scale as in the dimensional harmonic oscillator before
        self.define_ode_variable("xdot",scale=X/T,testscale=T/X) # velocity scales as X/T, test scale T/X will cancel this out

    def define_residuals(self):
        x,xdot=var(["x","xdot"])
        residual=(partial_t(xdot)-self.force/self.mass)*testfunction(x)
        residual+=(partial_t(x)-xdot)*testfunction(xdot)
        self.add_residual(residual)



class TennisProblem(Problem):
    def __init__(self):
        super(TennisProblem, self).__init__()
        self.top_racket_force=TennisRacket(direction=-1,spring_constant=5*newton/meter)
        self.bottom_racket_force=TennisRacket(direction=1,spring_constant=20*newton/meter)
        self.top_position=10*meter
        self.bottom_position=-10*meter
        self.ball_mass=60*gram
        self.ball_pos0=0*meter
        self.ball_velo0=10*meter/second

    def define_problem(self):
        self.set_scaling(spatial=1*meter,temporal=1*second)
        ball_pos=var("x")
        racket_force=self.top_racket_force(ball_pos,self.top_position)
        racket_force+=self.bottom_racket_force(ball_pos,self.bottom_position)
        racket_force=subexpression(racket_force)

        ball_eq=NewtonsLaw1d(mass=self.ball_mass,force=racket_force)
        ball_eq+=InitialCondition(x=self.ball_pos0,xdot=self.ball_velo0)
        ball_eq += ODEObservables(top_position_in_m=self.top_position/meter,bottom_position_in_m=self.bottom_position/meter)
        ball_eq+=ODEFileOutput()
        ball_eq+=TemporalErrorEstimator(x=1,xdot=1)

        self.add_equations(ball_eq@"ball")

if __name__=="__main__":
    with TennisProblem() as problem:
        # UNCOMMENT: Let the players move up and down
        # t=var("time")
        # problem.bottom_position=-10*meter+4*meter*sin(2*pi * 0.25*hertz*t)
        # problem.top_position = 10 * meter + 6 * meter * cos(2*pi * 0.1*hertz*t)
        problem.run(endtime=20*second,outstep=True,temporal_error=0.0025,startstep=0.01*second)

