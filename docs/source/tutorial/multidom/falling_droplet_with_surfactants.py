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


from falling_droplet import *
from pyoomph.expressions.phys_consts import *


class SurfactantTransportEquation(InterfaceEquations):
    def __init__(self):
        super(SurfactantTransportEquation, self).__init__()
        self.D=1e-9*meter**2/second # diffusivity

    def define_fields(self):
        self.define_scalar_field("Gamma","C2",testscale=scale_factor("temporal")/scale_factor("Gamma"))

    def define_residuals(self):
        u=var("velocity") # velocity at the interface
        G,Gtest=var_and_test("Gamma")
        self.add_residual(weak(partial_t(G)+div(u*G),Gtest))
        self.add_residual(weak(self.D*grad(G), grad(Gtest)))


if __name__ == "__main__":
    with FallingDropletProblem() as problem:
        Gamma0=1*micro*mol/meter**2
        problem.set_scaling(Gamma=Gamma0)

        add_ieqs=SurfactantTransportEquation()+InitialCondition(Gamma=Gamma0)+MeshFileOutput()
        problem.additional_equations+=add_ieqs@"droplet/droplet_outside"

        T=20*celsius
        problem.surface_tension=50*milli*newton/meter-gas_constant*T*var("Gamma")

        problem.run(0.5*second,startstep=0.05*second,outstep=True)  # solve and output


