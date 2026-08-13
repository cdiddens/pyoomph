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
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


class HDGPoissonEquations(Equations):
    # -grad^2(u)=f on a fully discontinuous space. No facet terms at all: everything that connects the
    # elements lives on the skeleton, which is what makes the element blocks condensable.
    def __init__(self, source, space="D2"):
        super().__init__()
        self.source, self.space = source, space

    def define_fields(self):
        self.define_scalar_field("u", self.space)

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) - weak(self.source, v))


class HDGCoupling(Equations):
    # The unknown uhat on each interior facet is the single-valued trace of u there. Each element sees
    # only its own u and the uhat of its own facets, never the neighbour's u - that is what
    # "hybridizable" means, and it is why the bulk unknowns can be eliminated element by element.
    #
    # Per element K, the symmetric hybridized interior-penalty form adds
    #     - int_dK  dn(u) (v - vhat)  - int_dK (u - uhat) dn(v)  + int_dK tau (u - uhat)(v - vhat)
    # to the bulk term int_K grad(u).grad(v). The second term restores symmetry, the third stabilizes.
    # Both are consistent, since u = uhat at the exact solution.
    def __init__(self, tau, space="D2"):
        super().__init__()
        self.tau, self.space = tau, space

    def define_fields(self):
        self.define_scalar_field("uhat", self.space)

    def define_residuals(self):
        uhat, vhat = var_and_test("uhat")
        u, v = var("u"), testfunction("u")
        n = var("normal")   # outward normal of the element this facet is attached to

        # The two one-sided values. avg(a)+jump(a)/2 is the near side, avg(a)-jump(a)/2 the far one,
        # since jump(a)=a_near-a_far and avg(a)=(a_near+a_far)/2.
        u_n, u_f = avg(u) + jump(u) / 2, avg(u) - jump(u) / 2
        v_n, v_f = avg(v) + jump(v) / 2, avg(v) - jump(v) / 2
        gu_n, gu_f = avg(grad(u)) + jump(grad(u)) / 2, avg(grad(u)) - jump(grad(u)) / 2
        gv_n, gv_f = avg(grad(v)) + jump(grad(v)) / 2, avg(grad(v)) - jump(grad(v)) / 2

        # The sum over the two elements sharing this facet. The far element's outward normal is -n.
        r = -weak(dot(n, gu_n), v_n - vhat) - weak(-dot(n, gu_f), v_f - vhat)
        r += -weak(u_n - uhat, dot(n, gv_n)) - weak(u_f - uhat, -dot(n, gv_f))
        r += weak(self.tau * (u_n - uhat), v_n - vhat) + weak(self.tau * (u_f - uhat), v_f - vhat)
        self.add_residual(r)

        # Only relevant under adaptivity/remeshing: a facet created inside a refined element has no
        # predecessor to take a value from, so reconstruct the trace from the bulk solution instead.
        self.set_facet_recovery("uhat", avg(u))


class HDGDirichletBC(InterfaceEquations):
    # The skeleton holds the INTERIOR facets only, so the exterior boundary gets its own terms: the
    # same expression with uhat replaced by the prescribed value, i.e. Nitsche's method.
    def __init__(self, value, tau):
        super().__init__()
        self.value, self.tau = value, tau

    def define_residuals(self):
        n = var("normal")
        # The gradient must be bound through the parent domain. A "D1"/"D2" field can be read directly
        # on an interface, but grad() of it is then the SURFACE gradient - dot(n,grad(u)) would quietly
        # vanish, and the boundary condition would contribute nothing.
        u, v = var_and_test("u", domain=self.get_parent_domain())
        self.add_residual(-weak(dot(n, grad(u)), v) - weak(u - self.value, dot(n, grad(v)))
                          + weak(self.tau * (u - self.value), v))


class HDGPoissonProblem(Problem):
    def __init__(self):
        super().__init__()
        self.N = 8                 # elements per direction
        self.space = "D2"          # bulk space
        self.facet_space = "D2"    # trace space on the skeleton, usually of the same order
        self.tau_factor = 10       # stabilization, tau = tau_factor/h
        self.condense = True       # eliminate the bulk unknowns, leaving the trace system

        x, y = var("coordinate_x"), var("coordinate_y")
        self.exact = sin(pi * x) * sin(pi * y)      # manufactured solution ...
        self.source = 2 * pi**2 * self.exact        # ... and the matching source term

    def define_problem(self):
        # Adaptivity off on purpose: the convergence table below compares a fixed sequence of meshes.
        # Every discontinuous facet space survives an adaptation, so this is not a restriction.
        self.max_refinement_level = 0
        self.initial_adaption_steps = 0

        self += RectangularQuadMesh(N=self.N)
        tau = self.tau_factor * self.N              # ~ tau_factor/h on the unit square

        eqs = HDGPoissonEquations(self.source, self.space)
        eqs += IntegralObservables(err2=(var("u") - self.exact)**2)
        eqs += HDGCoupling(tau, self.facet_space) @ "_internal_facets_"
        eqs += HDGDirichletBC(self.exact, tau) @ ["left", "right", "top", "bottom"]

        if self.condense:
            # Note that condense_element_private_dofs() would NOT select u: the facet elements read it
            # as external data, so it has to be named explicitly.
            #
            # This works under mpirun in both modes. Replicated (no --distribute), the elimination is
            # served because u is element-internal: oomph-lib numbers each element's own values
            # consecutively, so the row split can be cut between the blocks instead of through them.
            # With --distribute the trace is a shared facet unknown owned by one process, which needs
            # facet_space="DL" (a nodal one cannot be carried through the mesh rebuild that
            # distributing performs, and distribute() refuses it with that message).
            eqs += StaticCondensation("u")

        self += eqs @ "domain"

    def report(self):
        mesh = self.get_mesh("domain")
        nfacet = self.get_mesh("domain/_internal_facets_").nelement()
        stats = self._get_static_condensation_stats()
        condensed = stats.get("n_selected", 0)
        err = abs(float(mesh.evaluate_all_observables()["err2"]))**0.5
        print(f"  elements                : {mesh.nelement()}")
        print(f"  interior facets         : {nfacet}")
        print(f"  degrees of freedom      : {self.ndof()}")
        if condensed:
            # Under MPI the block count is this process's share of them (and includes the ones it only
            # holds a coupled row of), while the dof count is the whole problem's - so say which.
            from pyoomph.generic.mpi import get_mpi_nproc
            where = " on this process" if get_mpi_nproc() > 1 else ""
            print(f"  condensed away          : {condensed} in {stats['n_components']} blocks "
                  f"of {stats['component_size_max']}{where}")
            print(f"  seen by the solver      : {self.ndof() - condensed}")
        print(f"  L2 error of u           : {err:.4e}")


if __name__ == "__main__":
    with HDGPoissonProblem() as problem:
        problem.solve()
        problem.report()
        problem.output()
