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



#Import the parallel parameter scanner
from pyoomph.utils.paramscan import *
from pyoomph.expressions.units import * #Import the units (meter etc)

if __name__=="__main__":

	#Create a parameter scanner, give the script to run and the max number of processes to run simultaneously
	scanner=ParallelParameterScan("dimensional_oscillator_with_units.py",max_procs=4) 
	
	for k_in_N_per_m in [0.1,0.2,0.5,1,2,5]: #Scan the spring constant
		sim=scanner.new_sim("dim_osci_seq_run_k_"+str(k_in_N_per_m))
				
		#Modify the parameters
		sim.initial_displacement=-10*centi*meter
		sim.mass=1*kilogram
		sim.spring_constant=k_in_N_per_m*newton/meter
			
			
	#Run all (and rerun also already finished sims)
	scanner.run_all(skip_done=False) 
