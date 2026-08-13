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


class HybridizedPoissonEquation(Equations):
    # -grad^2(u)=f, but on a fully discontinuous space. Without a coupling across the facets, this is
    # just a collection of independent element-local problems.
    def __init__(self,source,space="D1"):
        super().__init__()
        self.source=source
        self.space=space

    def define_fields(self):
        self.define_scalar_field("u",self.space)

    def define_residuals(self):
        u,v=var_and_test("u")
        self.add_residual(weak(grad(u),grad(v))-weak(self.source,v))


class HDGCoupling(Equations):
    # The unknown lam lives on the interior facets themselves. It is the Lagrange multiplier enforcing
    # jump(u)=0 and, at the same time, the flux transmitted from one element to the other.
    def __init__(self,space="D0"):
        super().__init__()
        self.space=space # only discontinuous spaces, i.e. "D0", "DL", "D1", "D1TB", "D2", "D2TB"

    def define_fields(self):
        self.define_scalar_field("lam",self.space)

    def define_residuals(self):
        lam,mu=var_and_test("lam")
        u,v=var("u"),testfunction("u") # the bulk field, as seen from the facet
        self.add_residual(weak(lam,jump(v))+weak(jump(u),mu))
        # Only relevant with spatial adaptivity or remeshing: facets appearing inside a refined element
        # get nothing from the old skeleton. Instead of leaving them at zero, build lam from the bulk.
        self.set_facet_recovery("lam",-dot(var("normal"),avg(grad(u))))


class HybridizedPoissonProblem(Problem):
    def __init__(self):
        super().__init__()
        x=var("coordinate_x")
        self.exact=sin(pi*x)          # manufactured solution ...
        self.source=pi**2*sin(pi*x)   # ... and the corresponding source term
        self.N=10
        self.space="D1"

    def define_problem(self):
        self+=LineMesh(N=self.N)
        eqs=HybridizedPoissonEquation(self.source,self.space)
        eqs+=TextFileOutput(discontinuous=True)
        eqs+=IntegralObservables(uerr2=(var("u")-self.exact)**2)
        eqs+=DirichletBC(u=self.exact)@["left","right"]

        feqs=HDGCoupling()
        # jump(u) must vanish and lam must approach the exact flux -n*grad(u_exact).
        # A facet of a 1d mesh is a point, so these "integrals" are plain sums over all facets.
        feqs+=IntegralObservables(jump2=jump(var("u"))**2,
                                  fluxerr2=(var("lam")+var("normal")[0]*pi*cos(pi*var("coordinate_x")))**2)
        eqs+=feqs@"_internal_facets_" # the reserved name of the interior facet skeleton

        self+=eqs@"domain"
        self.max_refinement_level=1

    def report(self,when):
        bulk=self.get_mesh("domain").evaluate_all_observables()
        facet=self.get_mesh("domain/_internal_facets_").evaluate_all_observables()
        unfilled=self.get_mesh("domain/_internal_facets_").get_discontinuous_unrestored_elements()
        print(when+": L2 error of u = %.3e, |jump(u)| = %.3e, error of lam = %.3e, unfilled facets = %d"
            %(abs(float(bulk["uerr2"]))**0.5,abs(float(facet["jump2"]))**0.5,
                abs(float(facet["fluxerr2"]))**0.5,len(unfilled)))


if __name__=="__main__":
    with HybridizedPoissonProblem() as problem:
        problem.solve()
        problem.report("solution")
        # Any adaptation rebuilds the skeleton from scratch: surviving facets keep their values, the
        # ones created inside the refined elements are filled by the recovery expression above.
        problem.refine_uniformly()
        problem.report("after refinement, before solving")
        problem.solve()
        problem.report("after refinement, solved")
        problem.output()
