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


# Load the previous code 
from hopf_switch import *

if __name__=="__main__":
    with LorenzProblem() as problem:        
        problem.setup_for_stability_analysis(analytic_hessian=True)        
        
        # Go above the Hopf bifurcation into the chaos
        problem.rho.value=28
        problem+=InitialCondition(x=1,y=1,z=28)@"lorenz"
        # Solve for the unstable pitchfork branch        
        problem.solve()
        # Perturb it by the eigenfunction to leave the pitchfork by time integration
        problem.solve_eigenproblem(0)        
        problem.perturb_dofs(100*problem.get_last_eigenvectors()[0])
        # First run without any outputs
        problem.run(endtime=10,maxstep=0.0125,outstep=False,startstep=0.0001,temporal_error=0.005,do_not_set_IC=True)
        # Then write the dynamics from t=10 to t=60 to the output. A long stretch is required, since we
        # need an almost closed part of the trajectory in it (see below)
        problem.run(endtime=60,startstep=0.0125,outstep=True,do_not_set_IC=True)

        # Build the guess from the written output
        def build_orbit_guess():
            # Load the output
            data=numpy.loadtxt(problem.get_output_directory("lorenz.txt"))
            dt=data[1,0]-data[0,0]
            # It matters a lot which stretch of the chaos we take as guess. A fixed window, e.g. just
            # t=10 to t=13, lands wherever the chaos happens to be and is then typically far from closing
            # on itself. So we search for the best almost closed stretch of the trajectory.
            # The period range restricts the search to the orbit family we are after here (T close to 3).
            X=data[:,1:]
            best=None
            for k in range(int(round(2.8/dt)),int(round(3.4/dt))+1):
                dist=numpy.linalg.norm(X[k:]-X[:-k],axis=1)
                i=int(numpy.argmin(dist))
                if best is None or dist[i]<best[0]:
                    best=(dist[i],i,k)
            _,istart,nT=best
            segment=data[istart:istart+nT] # The almost closed stretch of the trajectory
            Tguess=nT*dt # Get a guess for the period (will be T~3)
            # Apply a low pass filter to make the guess periodic
            fft=numpy.fft.rfft(segment[:,1:],axis=0)
            fft[fft.shape[0]//16:,:]=0.0   # Just let only 1/16 of the frequencies pass
            # n must be passed, otherwise irfft silently drops the last sample for an odd segment length
            smoothed=numpy.fft.irfft(fft,n=nT,axis=0)
            numpy.savetxt(problem.get_output_directory("lorenz_smoothed.txt"),numpy.column_stack([segment[:,0],smoothed]),header="t x y z")
            return Tguess,smoothed

        # Under mpirun, only rank 0 writes lorenz.txt, but every rank would read it back - and
        # nothing synchronises the two. A rank arriving while rank 0 is still writing sees a shorter
        # trajectory and picks a different stretch, hence a different number of history dofs, so the
        # ranks would build orbit systems of different sizes and hang in the first collective of the
        # solve below. Rank 0 alone builds the guess (and writes the smoothed file) and broadcasts it.
        # Without MPI this is just the function call itself.
        Tguess,smoothed=get_mpi_bcast(build_orbit_guess() if get_mpi_rank()==0 else None)
        # Start with the smoothed data
        problem.set_current_dofs(smoothed[0])        
        # And give the smoothed recorded time history as orbit guess
        orbit=problem.activate_periodic_orbit_handler(Tguess,history_dofs=smoothed[1:],mode="bspline",order=3,GL_order=3)
        # Solve for the real orbit
        problem.solve()
            
        # Continue the orbit in rho
        def output_orbit_rho():
            orbit.output_orbit("orbit_at_rho_{:.3f}.txt".format(problem.rho.value))
                    
        output_orbit_rho()        
        ds=-0.01
        while problem.rho.value>20:
            ds=problem.arclength_continuation("rho",ds,max_ds=0.1)            
            output_orbit_rho()
            