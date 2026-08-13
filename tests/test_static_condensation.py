#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
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

# Correctness gates for static condensation, i.e. for problem.condense_dofs() /
# problem.use_static_condensation. See dev_docs/static_condensation.md for the design.
#
# What is being protected here:
#
#   Condensation eliminates the user-selected element-local dofs (Crouzeix-Raviart bubble velocities,
#   the gradient modes of a DL pressure, a projected D0/DG field) from the assembled Jacobian by a small
#   dense Schur complement, hands the solver a smaller system, and reconstructs the eliminated dofs from
#   the retained increment after the Newton update. It is a pure algebraic identity, so the ONLY
#   acceptable outcome is that the switch changes nothing about the answer -- same iteration count, same
#   dofs, eliminated ones included. Three failure modes matter and none of them announces itself:
#
#     * a WRONG elimination -- a slot list pointing at the wrong entry, a missed E-set member -- gives a
#       consistent-looking but different linear system, i.e. a Newton method that converges somewhere
#       else (or, on a linear problem, converges in one step to the wrong answer);
#     * a MISSING reconstruction leaves the eliminated dofs at their previous values, which on a linear
#       problem is invisible in the residual history of the retained ones;
#     * a SILENT DECLINE (the plan not applying, the gate not engaging) reports success while measuring
#       nothing -- so every equivalence test here also asserts _last_jacobian_was_condensed().
#
#   The tests are therefore organised as: what gets selected, what the plan makes of it, what is refused,
#   the algebra against a scipy referee at a fixed state, full Newton equivalence, and non-interference
#   with everything that must keep seeing the full system.
#
# Note on the linear solver: the equivalence tests set scipy's SuperLU explicitly. A condensed matrix is
# not the same matrix, so a solver whose own error is well above the Newton tolerance -- pardiso leaves
# ~1e-6 on these systems -- reproduces the converged state only to ~1e-8 and does not reproduce a
# residual history at all. That is the solver's accuracy, not the elimination's, but it makes for a flaky
# test. (Forcing a solver is fine in a test; it is not fine in a tutorial, where it changes what is being
# computed.)

import contextlib

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.navier_stokes import NavierStokesEquations, StokesEquations
from pyoomph.meshes.simplemeshes import RectangularQuadMesh


# ---------------------------------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------------------------------

class _InterfacePressureCoupling(Equations):
    """Requirement 2 live: a field on the top boundary that reads the bulk DL pressure, and feeds back
    into the bulk continuity equation.

    The interface element therefore adopts the adjacent bulk elements' pressure Data as external data
    and writes both the (lam, p) block -- a retained row referencing a condensed column -- and the
    (p, lam) block, i.e. the condensed dofs' OWN rows, from an element that does not own them. A
    per-element pre-scatter Schur complement (NGSolve's approach) is wrong exactly there, which is why
    pyoomph condenses the assembled matrix instead."""

    def define_fields(self):
        self.define_scalar_field("lam", "C2")

    def define_residuals(self):
        lam, lamtest = var_and_test("lam")
        p, ptest = var_and_test("pressure", domain="..")
        self.add_residual(weak(lam - 0.1 * p, lamtest) + weak(0.05 * lam, ptest))


class _CRCavity(Problem):
    """Lid-driven cavity discretised with Crouzeix-Raviart elements on triangles: C2TB velocities, whose
    cell-interior bubble node belongs to exactly one element, and a DL pressure, i.e. one constant plus
    the gradient modes per element. That is the discretisation static condensation was designed around --
    and the one where the naive "condense the pressure" is structurally singular."""

    def __init__(self, N=3, Re=None, interface=False, param=False, eqtree_condensation=False):
        super().__init__()
        self.N, self.Re, self.interface, self.param = N, Re, interface, param
        # The same selection stated in the equation tree instead of at problem level (section 7).
        self.eqtree_condensation = eqtree_condensation

    def define_problem(self):
        self.add_mesh(RectangularQuadMesh(N=self.N, split_in_tris="left"))
        if self.param:
            # The same problem with the Reynolds number as a continuable global parameter.
            self.get_global_parameter("Re").value = self.Re
            self.Re = self.get_global_parameter("Re")
        if self.Re is None:
            # mass_density=0 is not expressible (the continuity equation divides by rho), so the linear
            # case is Stokes rather than Navier-Stokes at Re=0.
            ns = StokesEquations(dynamic_viscosity=1, mode="CR")
        else:
            ns = NavierStokesEquations(dynamic_viscosity=1, mass_density=self.Re, mode="CR")
        eqs = ns + ns.create_pressure_fixation(value=0)
        if self.eqtree_condensation:
            eqs += StaticCondensation(velocity="bubble", pressure=[1, 2])
        eqs += InitialCondition(velocity_x=0, velocity_y=0)
        for b in ["left", "right", "bottom"]:
            eqs += DirichletBC(velocity_x=0, velocity_y=0) @ b
        eqs += DirichletBC(velocity_x=1, velocity_y=0) @ "top"
        if self.interface:
            eqs += _InterfacePressureCoupling() @ "top"
        self.add_equations(eqs @ "domain")

    def declare_condensation(self):
        """The classical CR elimination: the bubble velocities together with the pressure gradient
        modes. Neither half is invertible on its own, and value 0 -- the constant pressure mode -- has
        to stay a global unknown."""
        self.condense_dofs("domain/velocity", part="bubble")
        self.condense_dofs("domain/pressure", values=[1, 2])


class _HDGPoisson(Problem):
    """Hybridized DG Poisson: the two sides of a facet interact ONLY through the facet unknown, so the
    bulk field is element-local after all and must condense into one block per element.

    This is the case the whole side-aware contribution class exists for. It is the same equations as
    _DGPoisson above except that the coupling goes through `uhat` rather than through jump(u), plus the
    stabilisation term without which each element-local block would be a pure Neumann problem and hence
    singular. `with_jump=True` puts an interior-penalty term back and must return the problem to one
    percolating component -- the check that the coupling table follows the equations rather than having
    been hardcoded sparse."""

    def __init__(self, N=4, with_jump=False):
        super().__init__()
        self.N, self.with_jump = N, with_jump

    def define_problem(self):
        from pyoomph.meshes.simplemeshes import LineMesh
        self.add_mesh(LineMesh(N=self.N))
        x = var("coordinate_x")
        tau = 10.0 * self.N
        exact, source = sin(pi * x), pi**2 * sin(pi * x)
        with_jump = self.with_jump

        class Bulk(Equations):
            def define_fields(self):
                self.define_scalar_field("u", "D1")

            def define_residuals(self):
                u, v = var_and_test("u")
                self.add_residual(weak(grad(u), grad(v)) - weak(source, v))

        class Facet(Equations):
            def define_fields(self):
                self.define_scalar_field("uhat", "D0")

            def define_residuals(self):
                uh, mu = var_and_test("uhat")
                u, v = var("u"), testfunction("u")
                n = var("normal")
                # One-sided values, built from avg/jump of the bulk quantities.
                up, um = avg(u) + jump(u) / 2, avg(u) - jump(u) / 2
                vp, vm = avg(v) + jump(v) / 2, avg(v) - jump(v) / 2
                gup, gum = avg(grad(u)) + jump(grad(u)) / 2, avg(grad(u)) - jump(grad(u)) / 2
                gvp, gvm = avg(grad(v)) + jump(grad(v)) / 2, avg(grad(v)) - jump(grad(v)) / 2
                r = -weak(dot(n, gup), vp - mu) - weak(-dot(n, gum), vm - mu)
                r += -weak(up - uh, dot(n, gvp)) - weak(um - uh, -dot(n, gvm))
                r += weak(tau * (up - uh), vp - mu) + weak(tau * (um - uh), vm - mu)
                if with_jump:
                    r += weak(jump(u), jump(v))   # couples the two sides DIRECTLY: no longer hybridized
                self.add_residual(r)

        class Boundary(InterfaceEquations):
            def define_residuals(self):
                u, v = var_and_test("u")
                n = var("normal")
                self.add_residual(-weak(dot(n, grad(u)), v) - weak(u - exact, dot(n, grad(v)))
                                  + weak(tau * (u - exact), v))

        eqs = Bulk() + Facet() @ "_internal_facets_" + Boundary() @ ["left", "right"]
        self.add_equations(eqs @ "domain")


class _DGPoisson(Problem):
    """Interior-penalty DG Poisson: a D1 field whose facet terms couple every element to its neighbours,
    so selecting the whole field gives ONE connected component spanning the mesh. The size guard exists
    for this, and this is what it is meant to catch."""

    def define_problem(self):
        # 32 elements x 3 values = 96 dofs in the single component, i.e. above the default limit of 64.
        # (At N=3 the same selection is one component of 54 and is accepted, which is the guard being a
        # threshold rather than a diagnosis: it bounds the cost of the dense inversions, nothing more.)
        self.add_mesh(RectangularQuadMesh(N=4, split_in_tris="left"))
        from pyoomph.equations.poisson import PoissonEquation
        self.add_equations(PoissonEquation(space="D1", source=1) @ "domain")


# ---------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------

def _deterministic_solver(p):
    """A direct solver that solves to round-off, so that a residual history is a statement about the
    elimination rather than about the factorisation.

    scipy's SuperLU rather than "umfpack": scipy is a hard dependency, so this is available everywhere,
    and pyoomph's "umfpack" backend factorises with splu (SuperLU) in any case -- it only flips scipy's
    global use_umfpack flag, which then affects spsolve() elsewhere in the process."""
    p.set_linear_solver("superlu")


@contextlib.contextmanager
def _prepared(condense, **kwargs):
    """A quiet, initialised _CRCavity with the CR selection declared and condensation on or off. The
    rules are declared in BOTH arms so that the only difference between them is the master switch.

    A context manager rather than a factory, so that the problem is released even if the setup itself
    raises: several tests build several problems in one process, which only works because each one has
    been torn down (`Problem.__exit__` -> release()) before the next is created."""
    p = _CRCavity(**kwargs)
    with p:
        p.quiet()
        p.initialise()
        _deterministic_solver(p)
        p.declare_condensation()
        p.use_static_condensation = bool(condense)
        yield p


def _selected(p):
    return numpy.asarray(sorted(p._get_static_condensation_dofs()), dtype=int)


def _dofs(p):
    return numpy.asarray(p.get_current_dofs()[0], dtype=float).copy()


def _internal_values(p, meshname="domain"):
    """Every element-internal value: the condensed DL gradient modes AND the constant mode that stays
    global. Redundant with the dof vector for the unknowns, but it also covers anything that stopped
    being a dof, which is where a reconstruction bug would hide."""
    mesh = p.get_mesh(meshname)
    out = []
    for e in range(mesh.nelement()):
        el = mesh.element_pt(e)
        for j in range(el.ninternal_data()):
            d = el.internal_data_pt(j)
            out.append([d.value(k) for k in range(d.nvalue())])
    return numpy.asarray(out, dtype=float)


def _assembled(p):
    """(residual, J) of one assembly, as dense-free scipy CSR, copied: the arrays handed back are views
    onto oomph's buffers and the next assembly reallocates them."""
    r, J = p.assemble_jacobian(with_residual=True)
    return numpy.asarray(r, dtype=float).copy(), J.tocsr().copy()


def _schur_referee(r, J, L):
    """Recompute, in scipy, everything the kernel is supposed to have done: the Schur complement of the
    retained block, the condensed residual, and the two per-component operators the Newton step
    reconstructs from. Small meshes, so the L block goes dense -- the point is independence from the
    implementation, not speed."""
    n = J.shape[0]
    mask = numpy.zeros(n, dtype=bool)
    mask[L] = True
    E = numpy.flatnonzero(~mask)
    A_LL = J[L][:, L].toarray()
    A_LE = J[L][:, E].toarray()
    A_EL = J[E][:, L].toarray()
    A_EE = J[E][:, E].toarray()
    X = numpy.linalg.solve(A_LL, A_LE)      # A_LL^-1 A_LE
    y = numpy.linalg.solve(A_LL, r[L])      # A_LL^-1 r_L
    return E, A_EE - A_EL @ X, r[E] - A_EL @ y, X, y


# ---------------------------------------------------------------------------------------------------
# 1 -- what gets selected
# ---------------------------------------------------------------------------------------------------

def test_selection_counts_on_crouzeix_raviart():
    """The counts are exact and per element, which is the whole claim of the selection stage: the bubble
    node of a C2TB triangle belongs to one element (2 velocity values), and a 2D DL pressure has one
    constant plus two gradient modes."""
    with _CRCavity(N=3) as p:
        p.quiet()
        p.initialise()
        nel = p.get_mesh("domain").nelement()
        assert nel == 18

        p.condense_dofs("domain/velocity", part="bubble")
        assert p._get_static_condensation_stats()["n_selected"] == 2 * nel

        p._clear_static_condensation_rules()
        p.condense_dofs("domain/pressure", values=[1, 2])
        assert p._get_static_condensation_stats()["n_selected"] == 2 * nel

        p._clear_static_condensation_rules()
        p.condense_dofs("domain/pressure")
        # One pressure value is fixed by create_pressure_fixation, and a pinned value is never selected.
        assert p._get_static_condensation_stats()["n_selected"] == 3 * nel - 1

        # Auto-detection must find exactly the same thing: the DL pressure is the only internal Data.
        p._clear_static_condensation_rules()
        p.condense_element_private_dofs()
        assert p._get_static_condensation_stats()["n_selected"] == 3 * nel - 1


def test_DL_gradients_is_the_dimension_of_the_domain():
    """"DL_gradients" is the same selection as spelling the value indices out, without the script
    having to know the dimension: 1..dim, i.e. every value of the DL field except the constant. Value 0
    has to stay global (test_numeric_pivot_guard_catches_the_constant_pressure_mode is why), so this is
    the form that cannot be got wrong by porting a 2D script to 3D."""
    with _CRCavity(N=3) as p:
        p.quiet()
        p.initialise()
        p.condense_dofs("domain/pressure", values=[1, 2])
        explicit = _selected(p)

        p._clear_static_condensation_rules()
        p.condense_dofs("domain/pressure", part="DL_gradients")
        assert numpy.array_equal(_selected(p), explicit)


def test_data_adopted_by_an_interface_is_not_element_private():
    """The case that rules out a per-element Schur complement, and the one thing condense_element_private
    _dofs() has to get right: internal Data another element reads is not element-local. The explicitly
    named rule is deliberately NOT filtered -- the user asked for those dofs, and the post-assembly
    elimination handles them correctly (see the algebra test below)."""
    with _CRCavity(N=3, interface=True) as p:
        p.quiet()
        p.initialise()
        nel = p.get_mesh("domain").nelement()
        p.condense_element_private_dofs()
        # The three elements along the top adopt their pressure Data out to the interface: 3 x 3 values.
        assert p._get_static_condensation_stats()["n_selected"] == 3 * nel - 1 - 9

        p._clear_static_condensation_rules()
        p.condense_dofs("domain/pressure")
        assert p._get_static_condensation_stats()["n_selected"] == 3 * nel - 1


def test_selection_is_reresolved_after_refinement():
    """Rules are the durable objects; the (Data*, value) pairs are derived and thrown away on every
    renumbering. Refinement is the cheapest way to check that, and a stale selection would either keep
    the old count or dereference freed Data."""
    with _CRCavity(N=3) as p:
        p.quiet()
        p.initialise()
        p.declare_condensation()
        before = p._get_static_condensation_stats()["n_selected"]
        nel = p.get_mesh("domain").nelement()
        p.refine_uniformly()
        assert p.get_mesh("domain").nelement() == 4 * nel
        assert p._get_static_condensation_stats()["n_selected"] == 4 * before


# ---------------------------------------------------------------------------------------------------
# 2 -- the elimination plan
# ---------------------------------------------------------------------------------------------------

def test_plan_is_one_four_by_four_block_per_element():
    """Size 4 is exactly the classical CR block: two bubble velocity values plus the two DL gradient
    modes. If the components came out bigger, the selection would be coupling across elements and the
    dense inversions would not be cheap; if smaller, the block would be singular."""
    with _prepared(True, N=3) as p:
        assert p._build_condensation_plan()
        stats = p._get_static_condensation_stats()
        nel = p.get_mesh("domain").nelement()
        assert stats["n_components"] == nel
        assert stats["component_size_min"] == stats["component_size_max"] == 4
        sizes = p._get_condensation_component_sizes()
        assert len(sizes) == nel
        assert all(nL == 4 for nL, _nE in sizes)
        # ...and the retained sets are genuinely small too, i.e. the fill-in is local.
        assert max(nE for _nL, nE in sizes) < 20


def test_condensed_matrix_is_smaller_than_the_full_one():
    with _prepared(True, N=4) as p:
        assert p._build_condensation_plan()
        stats = p._get_static_condensation_stats()
        assert stats["condensed_nnz"] < stats["full_nnz"] / 1.5
        # The Schur complements do produce entries the full pattern never had -- that is the price --
        # but on CR it is a small one: the pressure-pressure couplings the constant modes acquire.
        assert 0 < stats["n_fill_in_entries"] < 0.05 * stats["full_nnz"]
        # Every condensed entry is either copied over, part of a Schur block, or an identity diagonal.
        assert stats["n_passthrough"] + stats["n_fill_in_entries"] + len(_selected(p)) == stats["condensed_nnz"]


def test_structure_id_changes_on_toggle_and_on_a_rule_change():
    """The full pattern does not depend on either, but what a solver is HANDED under that id does: the
    condensed matrix. Reusing a symbolic factorisation across a toggle would factorise one pattern and
    back-substitute on another."""
    with _prepared(False, N=3) as p:
        p.solve()
        sid = p.jacobian_structure_id
        assert sid != 0
        p.use_static_condensation = True
        sid_on = p.jacobian_structure_id
        assert sid_on != sid
        p.condense_dofs("domain/pressure", values=[1])   # a redundant rule, but still a rule change
        assert p.jacobian_structure_id != sid_on


# ---------------------------------------------------------------------------------------------------
# 3 -- refusals. Every one of these THROWS: silently not eliminating what the user asked for would be a
# performance mystery rather than an error.
# ---------------------------------------------------------------------------------------------------

def test_pressure_only_selection_is_refused_as_structurally_singular():
    """The classic mistake, and the discovery that shaped the whole selection API: the continuity
    equation weak(div(u), q) contains no pressure at all, so the rows of a pressure-only block are
    identically zero. The message has to name the field, or the user cannot act on it."""
    with _CRCavity(N=3) as p:
        p.quiet()
        p.initialise()
        p.condense_dofs("domain/pressure", values=[1, 2])
        p.use_static_condensation = True
        with pytest.raises(Exception) as excinfo:
            p._build_condensation_plan()
        msg = str(excinfo.value)
        assert "structurally singular" in msg
        assert "pressure" in msg, "the refusal must name the offending field"
        assert "bubble" in msg, "...and say what to do instead"


def test_component_size_guard_refuses_a_selection_that_percolates():
    """A component is inverted densely, so a selection whose coupling graph spans the mesh is not a
    condensation but a second, worse, direct solve."""
    with _prepared(True, N=3) as p:
        p.static_condensation_max_component_size = 3   # the CR block is 4
        with pytest.raises(Exception) as excinfo:
            p._build_condensation_plan()
        msg = str(excinfo.value)
        assert "4 mutually coupled dofs" in msg
        assert "static_condensation_max_component_size = 3" in msg


@pytest.mark.slow
def test_size_guard_fires_on_a_genuinely_percolating_selection():
    """The same guard on the case it was written for rather than on a lowered threshold: an
    interior-penalty DG field is coupled across every facet, so the whole mesh is one component. Marked
    slow only because it compiles a second, otherwise unused, set of elements."""
    with _DGPoisson() as p:
        p.quiet()
        p.initialise()
        p.condense_dofs("domain/u")
        p.use_static_condensation = True
        with pytest.raises(Exception) as excinfo:
            p._build_condensation_plan()
        msg = str(excinfo.value)
        assert "do not decompose into small element-local blocks" in msg
        assert "96 mutually coupled dofs" in msg


def _facet_tables(p):
    """(names, jacobian table) of the interior-facet element code."""
    el = p.get_mesh("domain/_internal_facets_").element_pt(0)
    names, jac, _mass = el._get_contribution_tables()
    return list(names), jac


def test_hdg_bulk_field_condenses_into_one_block_per_element():
    """The point of the side-aware contribution class. The two sides of a facet talk only through the
    facet unknown, so the bulk field really is element-local and the solver must be left with the trace
    system alone -- which is the definition of an HDG method, not an optimisation of it.

    Before the split, both sides shared the class "domain/u", contributes_to_jacobian[u][u] conflated
    u+ vs u+ with u+ vs u-, the pruned pattern kept an edge across every facet, and this came out as ONE
    component spanning the whole mesh."""
    with _HDGPoisson(N=6) as p:
        p.quiet()
        p.initialise()
        p.condense_dofs("domain/u")
        p.use_static_condensation = True
        p._build_condensation_plan()
        st = p._get_static_condensation_stats()
        nel = p.get_mesh("domain").nelement()
        nfacet = p.get_mesh("domain/_internal_facets_").nelement()
        assert st["n_components"] == nel
        # D1 in 1d: two values per element, and nothing from a neighbour.
        assert st["component_size_min"] == st["component_size_max"] == 2
        assert st["n_selected"] == 2 * nel
        # What the solver is handed is exactly the skeleton.
        assert p.ndof() - st["n_selected"] == nfacet


def test_hdg_facet_table_states_the_two_sides_do_not_couple():
    """The table underneath the previous test, asserted directly: the near and far copies of the bulk
    field are separate classes, they do not couple, and the facet unknown still couples to both."""
    with _HDGPoisson(N=4) as p:
        p.quiet()
        p.initialise()
        names, jac = _facet_tables(p)
        assert "domain/u" in names and "domain/u@opposite" in names
        near, far = names.index("domain/u"), names.index("domain/u@opposite")
        uhat = names.index("domain/_internal_facets_/uhat")
        assert jac[near][near] and jac[far][far]
        assert not jac[near][far] and not jac[far][near]
        # The trace equation is what carries the coupling, in both directions and to both sides.
        assert jac[uhat][near] and jac[uhat][far]
        assert jac[near][uhat] and jac[far][uhat]


def test_hdg_dofs_are_attributed_to_the_side_they_belong_to():
    """The other half of the mechanism: the table would be useless if the element could not say which
    of its local dofs belongs to which side. The facet element adopts the far side's attribution and
    translates it into its own @opposite class."""
    with _HDGPoisson(N=4) as p:
        p.quiet()
        p.initialise()
        names, _ = _facet_tables(p)
        near, far = names.index("domain/u"), names.index("domain/u@opposite")
        el = p.get_mesh("domain/_internal_facets_").element_pt(0)
        cidx = list(el._get_dof_contribution_indices())
        assert -1 not in cidx, "every dof of a facet element must be attributable"
        # Two D1 values on each side, plus the single D0 facet value.
        assert cidx.count(near) == 2 and cidx.count(far) == 2
        assert cidx.count(names.index("domain/_internal_facets_/uhat")) == 1


def test_a_jump_term_makes_the_hdg_problem_percolate_again():
    """The control that proves the table follows the equations. Adding an interior-penalty jump term
    couples the two sides directly, so the very same problem must go back to one component -- if this
    ever passes, the sparsity has been hardcoded rather than derived."""
    with _HDGPoisson(N=4, with_jump=True) as p:
        p.quiet()
        p.initialise()
        names, jac = _facet_tables(p)
        near, far = names.index("domain/u"), names.index("domain/u@opposite")
        assert jac[near][far] and jac[far][near]
        p.condense_dofs("domain/u")
        p.use_static_condensation = True
        p.static_condensation_max_component_size = 100000
        p._build_condensation_plan()
        st = p._get_static_condensation_stats()
        assert st["n_components"] == 1
        assert st["component_size_max"] == 2 * p.get_mesh("domain").nelement()


def test_hdg_condensation_does_not_change_the_solution():
    """Exactness, on the case whose whole point is the elimination."""
    ref = None
    for condense in (False, True):
        with _HDGPoisson(N=8) as p:
            p.quiet()
            p.initialise()
            _deterministic_solver(p)
            if condense:
                p.condense_dofs("domain/u")
            p.use_static_condensation = condense
            p.solve()
            if condense:
                assert p._last_jacobian_was_condensed()
            dofs = numpy.array(p.get_current_dofs()[0], dtype=float)
            if ref is None:
                ref = dofs
            else:
                assert numpy.allclose(dofs, ref, rtol=0, atol=1e-11)


def test_the_side_split_is_invisible_at_problem_level():
    """The per-side classes are an element-local statement. Globally there is one field `u`, and the
    empty-row analysis must not see a phantom one -- a far-side class typically has no residual tested
    against it at all."""
    with _HDGPoisson(N=4) as p:
        p.quiet()
        p.initialise()
        info, _good = p._get_jacobian_information_string()
        info = str(info)
        fields = info.split("Jacobian Structure")[0]
        assert "@opposite" not in fields
        # ... but the split is reported, so it is discoverable from the file.
        assert "Split by facet side" in info and "domain/u" in info.split("Split by facet side")[1]


def test_numeric_pivot_guard_catches_the_constant_pressure_mode():
    """Taking DL value 0 along with the bubbles passes the STRUCTURAL check -- the pattern does contain a
    pressure-to-bubble coupling -- and breaks down numerically, because the integral of div(u_bubble)
    against a constant test function vanishes. Hence a relative pivot test inside the factorisation, and
    hence not oomph's DenseLU, which substitutes 1e-20 for a vanishing pivot and returns an increment of
    order 1e20 instead of an error."""
    with _CRCavity(N=3) as p:
        p.quiet()
        p.initialise()
        p.condense_dofs("domain/velocity", part="bubble")
        p.condense_dofs("domain/pressure", values=[0, 1, 2])
        p.use_static_condensation = True
        p._debug_force_condensed_assembly = True
        with pytest.raises(Exception) as excinfo:
            p.assemble_jacobian(with_residual=True)
        msg = str(excinfo.value)
        assert "numerically singular" in msg
        assert "CONSTANT mode" in msg, "the message must point at the actual mistake"


def test_line_search_is_refused():
    """The globally convergent Newton method rescales the retained increment AFTER the solve, while the
    eliminated dofs would be reconstructed from the unscaled one. Refused rather than silently wrong --
    and the refusal is not one-way: switching the line search off again condenses as before.

    (The other two refusals of the same family are not testable from here: an MPI run needs mpirun --
    the pattern in tests/mpi_structural_worker.py -- and Jacobian reuse has no Python binding at all, so
    that one is compile-checked only.)"""
    with _prepared(True, N=3) as p:
        with pytest.raises(Exception) as excinfo:
            p.solve(globally_convergent_newton=True)
        assert "globally convergent" in str(excinfo.value)
        p.solve()
        assert p._last_jacobian_was_condensed()


# ---------------------------------------------------------------------------------------------------
# 4 -- the algebra, refereed by scipy at a fixed state
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("interface", [False, True], ids=["plain", "interface_adopts_pressure"])
def test_condensed_system_is_the_schur_complement(interface):
    """The core identity, checked entry by entry against an independent implementation:

        J~[E,E] = A_EE - A_EL A_LL^-1 A_LE,   r~[E] = r_E - A_EL A_LL^-1 r_L,
        J~[L,:] = identity,  J~[:,L] = 0,     r~[L] = 0

    and, on top of that, that solving the condensed system and reconstructing gives the full increment.
    The interface variant is requirement 2 live: a non-owning element writes both the row and the column
    of a condensed dof, which is exactly what a per-element pre-scatter Schur complement gets wrong.

    Both arms assemble at the same (initial) state, so the two Jacobians describe the same linearisation."""
    with _CRCavity(N=3, interface=interface) as p_full:
        p_full.quiet()
        p_full.initialise()
        p_full.declare_condensation()
        L = _selected(p_full)
        r, J = _assembled(p_full)
        assert p_full._get_frozen_sparsity_nnz() == J.nnz, \
            "the frozen path must have engaged -- the plan is expressed as positions in its value array"

    with _prepared(True, N=3, interface=interface) as p_cond:
        p_cond._debug_force_condensed_assembly = True
        assert numpy.array_equal(_selected(p_cond), L)
        rc, Jc = _assembled(p_cond)
        assert p_cond._last_jacobian_was_condensed()

    E, S, rE, X, y = _schur_referee(r, J, L)
    scale = max(abs(J).max(), 1.0)

    assert abs(Jc[E][:, E].toarray() - S).max() <= 1e-11 * scale
    assert abs(rc[E] - rE).max() <= 1e-11 * max(abs(r).max(), 1.0)

    # The eliminated rows are identity rows with a zero right-hand side, so the solver returns dx_L = 0.
    assert abs(rc[L]).max() == 0.0
    Lrows = Jc[L]
    assert Lrows.nnz == len(L)
    assert numpy.array_equal(Lrows.indices, L)
    assert numpy.array_equal(Lrows.data, numpy.ones(len(L)))
    assert Jc[:, L][E].nnz == 0, "no retained row may still reference an eliminated dof"

    # dx-equivalence: what the two systems hand back to the Newton step must agree, on the retained dofs
    # directly and on the eliminated ones after the reconstruction dx_L = y - X dx_E.
    # use_umfpack=False explicitly: scipy's global flag is process-wide and pyoomph's "umfpack" backend
    # flips it on, so a test run that touched that backend would otherwise send this through a package
    # that may not be installed at all.
    from scipy.sparse.linalg import spsolve
    dx = spsolve(J.tocsc(), r, use_umfpack=False)
    dxc = spsolve(Jc.tocsc(), rc, use_umfpack=False)
    assert abs(dxc[L]).max() == 0.0
    tol = 1e-9 * max(abs(dx).max(), 1.0)
    assert abs(dxc[E] - dx[E]).max() <= tol
    assert abs((y - X @ dxc[E]) - dx[L]).max() <= tol


# ---------------------------------------------------------------------------------------------------
# 5 -- full Newton equivalence. The switch may change nothing at all.
# ---------------------------------------------------------------------------------------------------

def _newton_arms(run, **kwargs):
    """Run `run(p)` twice, once with condensation off and once on, and return the two final states."""
    out = []
    for condense in (False, True):
        with _prepared(condense, **kwargs) as p:
            run(p)
            assert p._last_jacobian_was_condensed() == condense, \
                "condensation %s engage" % ("did not" if condense else "must not")
            out.append((_dofs(p), _internal_values(p), list(p.get_last_residual_convergence()),
                        _selected(p)))
    return out


@pytest.mark.parametrize("Re", [None, 100.0], ids=["stokes", "navier_stokes_re100"])
def test_steady_newton_solve_is_unchanged(Re):
    (d0, i0, h0, L0), (d1, i1, h1, L1) = _newton_arms(lambda p: p.solve(), N=4, Re=Re)
    assert numpy.array_equal(L0, L1)
    assert len(h0) == len(h1), "the iteration count changed: %s vs %s" % (h0, h1)
    assert numpy.allclose(d0, d1, rtol=0, atol=1e-9)
    assert numpy.allclose(i0, i1, rtol=0, atol=1e-9)
    # ...and the eliminated dofs really were reconstructed rather than left where they started.
    assert abs(d1[L1]).max() > 1e-3
    assert numpy.allclose(h0, h1, rtol=1e-6, atol=1e-12)


def test_transient_solve_is_unchanged():
    """Three BDF2 steps: the reconstruction writes through value_pt(), so this is what checks that the
    eliminated dofs' time history is shifted along with everybody else's."""

    def run(p):
        for _ in range(3):
            p.solve(timestep=0.05)

    (d0, i0, h0, L0), (d1, i1, h1, L1) = _newton_arms(run, N=4, Re=100.0)
    assert len(h0) == len(h1)
    assert numpy.allclose(d0, d1, rtol=0, atol=1e-9)
    assert numpy.allclose(i0, i1, rtol=0, atol=1e-9)


def test_interface_adoption_solve_is_unchanged():
    """The same equivalence with a non-owning element writing the condensed dofs' rows and columns."""
    (d0, i0, h0, _L0), (d1, i1, h1, _L1) = _newton_arms(lambda p: p.solve(), N=4, interface=True)
    assert len(h0) == len(h1)
    assert numpy.allclose(d0, d1, rtol=0, atol=1e-9)
    assert numpy.allclose(i0, i1, rtol=0, atol=1e-9)


def test_solve_after_refinement_is_unchanged():
    """Adaptation renumbers everything, so the rules have to be re-resolved and the plan rebuilt between
    the two solves. A stale plan would be caught by the pattern check in the kernel, a stale SELECTION
    would not be."""

    def run(p):
        p.solve()
        p.refine_uniformly()
        p.solve()

    (d0, i0, h0, L0), (d1, i1, h1, L1) = _newton_arms(run, N=3)
    assert numpy.array_equal(L0, L1)
    assert len(h0) == len(h1)
    assert numpy.allclose(d0, d1, rtol=0, atol=1e-9)
    assert numpy.allclose(i0, i1, rtol=0, atol=1e-9)


def test_relaxation_factor_is_applied_to_the_reconstruction():
    """A damped Newton step is what actually tests that the reconstruction uses the SAME relaxation
    factor: at 0.7 the residual history is a geometric sequence, and eliminated dofs moving by a
    different fraction of the increment show up immediately in the second entry."""

    def run(p):
        p.newton_relaxation_factor = 0.7
        p.max_newton_iterations = 60
        p.solve()

    (d0, i0, h0, _L0), (d1, i1, h1, _L1) = _newton_arms(run, N=3)
    assert len(h0) == len(h1) > 5, "expected a damped, multi-step convergence"
    assert numpy.allclose(d0, d1, rtol=0, atol=1e-9)
    assert numpy.allclose(i0, i1, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------------------------------
# 6 -- non-interference: everything that is not a flagged Newton solve keeps the full system
# ---------------------------------------------------------------------------------------------------

def test_bare_assembly_and_residuals_stay_full():
    """The primary gate. A bare assemble_jacobian() has no reconstruction step after it, so condensing
    it would silently drop the eliminated dofs from whatever the caller does next; and get_residuals()
    feeds the Newton convergence check, which is what makes "identical iteration counts" meaningful."""
    with _prepared(True, N=3) as p:
        p.solve()
        assert p._last_jacobian_was_condensed()
        full_nnz = p._get_static_condensation_stats()["full_nnz"]
        J = p.assemble_jacobian(with_residual=False)
        assert not p._last_jacobian_was_condensed()
        assert J.nnz == full_nnz
        assert len(numpy.asarray(p.get_residuals())) == p.ndof()


def test_eigenvalues_are_unaffected_by_a_condensed_solve():
    """Eigenproblems assemble through a different assembly handler and are declined, so the eigenvalues
    after a condensed solve must be the ones after an uncondensed solve -- both that the state is the
    same and that the eigen assembly itself was not condensed (M_LL is not zero here, so a condensed
    eigenproblem would be a different spectrum, which is why v1 does not attempt it).

    The eigen assembly is checked by the SIZE of the matrix it produced, not by
    _last_jacobian_was_condensed(): that flag is written by get_jacobian() alone, which the eigenproblem
    assembly does not go through, so it would still be reporting on the preceding Newton step."""
    spectra = []
    for condense in (False, True):
        with _prepared(condense, N=3, Re=1.0) as p:
            p.set_eigensolver("scipy")
            p.solve()
            assert p._last_jacobian_was_condensed() == condense
            full_nnz = p._get_static_condensation_stats()["full_nnz"] if condense else None
            n, _, _, _Mv, _Mc, _Mr, _, _, Jv, _Jc, _Jr = p.assemble_eigenproblem_matrices(0.0)
            assert n == p.ndof()
            if condense:
                assert len(Jv) == full_nnz, "the eigenproblem Jacobian must be the full one"
            p.solve_eigenproblem(4)
            spectra.append(numpy.sort_complex(numpy.asarray(p.get_last_eigenvalues())))
    assert numpy.allclose(spectra[0], spectra[1], rtol=1e-8, atol=1e-8)


def test_arclength_continuation_is_not_condensed_but_still_correct():
    """Continuation runs an augmented system with its own dof-update loop, which the reconstruction hook
    does not see, so it is deliberately left uncondensed -- and must therefore still track the same
    solution branch as a problem without any of this."""
    trajectories = []
    for condense in (False, True):
        with _prepared(condense, N=3, Re=50.0, param=True) as p:
            p.solve()
            assert p._last_jacobian_was_condensed() == condense
            traj = []
            for _ in range(3):
                p.arclength_continuation("Re", 15.0)
                assert not p._last_jacobian_was_condensed(), "continuation must not condense"
                traj.append((float(p.get_global_parameter("Re").value),
                             float(numpy.max(_dofs(p)))))
            trajectories.append(traj)
    for a, b in zip(*trajectories):
        assert a == pytest.approx(b, rel=0, abs=1e-8)


def test_switch_without_rules_is_inert():
    with _CRCavity(N=3) as p:
        p.quiet()
        p.initialise()
        _deterministic_solver(p)
        p.use_static_condensation = True    # ...but nothing selected
        p.solve()
        assert not p._last_jacobian_was_condensed()
        assert p._get_static_condensation_stats()["n_selected"] == 0


# ---------------------------------------------------------------------------------------------------
# 7 -- the equation-tree interface. StaticCondensation states the selection on the domain it belongs
# to, which is how everything else in pyoomph is said, and is the interface users are meant to use;
# condense_dofs() is the plumbing underneath it. What has to be true of it:
#
#   * it selects exactly what the problem-level calls select, and solves to the same answer;
#   * it switches the feature on by itself, while an explicit use_static_condensation stays the last
#     word -- in particular False, which is the documented kill switch;
#   * re-registering, which the hook does on every reapply_boundary_conditions(), must be FREE. Every
#     edit of the C++ rule list bumps the rules revision, which is part of the Jacobian structure id,
#     so a naive re-registration would silently rebuild the plan and force a fresh symbolic
#     factorisation on every solve -- a pure performance bug, invisible in any answer;
#   * after remeshing/adaptation the rules must name the CURRENT meshes.
# ---------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _prepared_eqtree(**kwargs):
    """The cavity with the selection declared in the equation tree and nothing else: no
    condense_dofs(), and no assignment to use_static_condensation anywhere."""
    p = _CRCavity(eqtree_condensation=True, **kwargs)
    with p:
        p.quiet()
        p.initialise()
        _deterministic_solver(p)
        yield p


def _uncondensed_vs_eqtree(run, **kwargs):
    """Run `run(p)` on the plain (switch off) problem and on the equation-tree one, and return both
    final states. The first arm declares the same rules at problem level so that the two selections
    are comparable; only the switch differs."""
    out = []
    for prepare in (lambda: _prepared(False, **kwargs), lambda: _prepared_eqtree(**kwargs)):
        with prepare() as p:
            run(p)
            out.append((_dofs(p), _internal_values(p), list(p.get_last_residual_convergence()),
                        _selected(p), p._last_jacobian_was_condensed()))
    assert out[0][4] is False and out[1][4] is True, "the equation-tree arm did not condense"
    return out[0], out[1]


def test_equation_tree_selects_what_the_problem_level_api_selects():
    """Same two rules, same dofs, same plan -- stated on the domain rather than at problem level. And
    the switch comes on by itself: nothing in this problem assigns use_static_condensation."""
    with _prepared_eqtree(N=3) as p:
        assert p.use_static_condensation
        assert not p._use_static_condensation_is_explicit
        nel = p.get_mesh("domain").nelement()
        stats = p._get_static_condensation_stats()
        assert stats["n_rules"] == 2
        assert stats["n_selected"] == 4 * nel      # 2 bubble velocity values + 2 DL gradient modes
        assert p._build_condensation_plan()
        stats = p._get_static_condensation_stats()
        assert stats["n_components"] == nel
        assert stats["component_size_min"] == stats["component_size_max"] == 4
        from_tree = _selected(p)

    with _prepared(True, N=3) as p:
        assert numpy.array_equal(_selected(p), from_tree)


def test_equation_tree_steady_solve_is_unchanged():
    (d0, i0, h0, L0, _), (d1, i1, h1, L1, _) = _uncondensed_vs_eqtree(lambda p: p.solve(), N=4, Re=100.0)
    assert numpy.array_equal(L0, L1)
    assert len(h0) == len(h1), "the iteration count changed: %s vs %s" % (h0, h1)
    assert numpy.allclose(d0, d1, rtol=0, atol=1e-9)
    assert numpy.allclose(i0, i1, rtol=0, atol=1e-9)
    assert abs(d1[L1]).max() > 1e-3, "the eliminated dofs were not reconstructed"


def test_equation_tree_solve_after_refinement_is_unchanged():
    """Refinement renumbers everything and the registration hook fires again on the way through
    reapply_boundary_conditions(). A rule pointing at a superseded mesh, or a selection left stale,
    would show up here rather than in the counts."""

    def run(p):
        p.solve()
        p.refine_uniformly()
        p.solve()

    (d0, i0, h0, L0, _), (d1, i1, h1, L1, _) = _uncondensed_vs_eqtree(run, N=3)
    assert numpy.array_equal(L0, L1)
    assert len(L1) == 4 * 4 * 18, "the rules were not re-resolved on the refined mesh"
    assert len(h0) == len(h1)
    assert numpy.allclose(d0, d1, rtol=0, atol=1e-9)
    assert numpy.allclose(i0, i1, rtol=0, atol=1e-9)


def test_reregistration_leaves_the_jacobian_structure_alone():
    """The idempotency requirement, and the only test that can catch it: the answers are unaffected
    either way, and a rules revision that ticks on every solve merely throws away the plan and the
    solver's symbolic factorisation each time."""
    with _prepared_eqtree(N=3) as p:
        # All four solves transient, so that nothing but the re-registration can move the id: the
        # first stationary-to-transient transition renumbers on its own account.
        p.solve(timestep=0.05)
        sid = p.jacobian_structure_id
        assert sid != 0
        rebuilds = p._get_static_condensation_stats()["plan_rebuilds"]
        assert rebuilds == 1
        for _ in range(3):
            p.solve(timestep=0.05)
        assert p._last_jacobian_was_condensed()
        assert p.jacobian_structure_id == sid, "the rules were restated even though nothing changed"
        assert p._get_static_condensation_stats()["plan_rebuilds"] == rebuilds


def test_the_problem_switch_overrules_the_equation_tree_in_both_directions():
    """Auto-enabling must not take the switch away from the user: use_static_condensation = False is
    the kill switch that disables the feature without touching the equations. The rules stay
    declared, so it really is the switch that is being tested and not an empty selection."""
    with _CRCavity(N=3, eqtree_condensation=True) as p:
        p.quiet()
        p.use_static_condensation = False     # before initialise, i.e. before anything registers
        p.initialise()
        _deterministic_solver(p)
        assert p._use_static_condensation_is_explicit
        assert not p.use_static_condensation
        assert p._get_static_condensation_stats()["n_selected"] == 4 * p.get_mesh("domain").nelement()
        p.solve()
        assert not p._last_jacobian_was_condensed()
        # ...and it is not a one-way street: switching it on again picks the same rules up.
        p.use_static_condensation = True
        p.solve()
        assert p._last_jacobian_was_condensed()


def test_static_condensation_on_an_interface_is_refused():
    """Condensation eliminates element-local dofs of a bulk domain; an interface element's own
    internal data belongs to a facet. Refused with a message rather than quietly selecting nothing."""
    class _OnAnInterface(_CRCavity):
        def define_problem(self):
            super().define_problem()
            self.add_equations(StaticCondensation("pressure") @ "domain/top")

    with pytest.raises(Exception) as excinfo:
        with _OnAnInterface(N=3) as p:
            p.quiet()
            p.initialise()
    msg = str(excinfo.value)
    assert "interface domain" in msg
    assert "domain/top" in msg


@pytest.mark.parametrize("args,kwargs", [
    (("domain/pressure",), {}),          # a path, not a field of this domain
    ((), {"u": "nonsense"}),
    ((), {"u": []}),
    ((), {"u": False}),
    ((), {"u": 1.5}),
    ((), {"u": [0, 1.5]}),
], ids=["path", "unknown_part", "empty_list", "false", "float", "float_in_list"])
def test_malformed_specs_are_refused_at_construction(args, kwargs):
    """Cheap, and worth having: a mistyped spec that only surfaced when the mesh is resolved would be
    reported far from where it was written."""
    with pytest.raises(ValueError):
        StaticCondensation(*args, **kwargs)


# --- the no-argument form: element-private dofs, restricted to its own domain -----------------------

class _TwoProjectionDomains(Problem):
    """Two independent bulk domains, each carrying one element-private (D0) unknown per element.

    Nothing couples them, so the only thing this measures is WHICH domain a rule applies to: the
    problem-wide element-private rule takes both, StaticCondensation() added to one domain takes that
    one. Deliberately tiny -- it is a scoping test, not a numerics test."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode    # "eqtree", "problem_wide" or "problem_restricted"

    def define_problem(self):
        from pyoomph.equations.generic import ProjectExpression
        for name in ("domain", "other"):
            self.add_mesh(RectangularQuadMesh(N=2, name=name, lower_left=[0, 0] if name == "domain" else [2, 0]))
            eqs = ProjectExpression(q=1, space="D0")
            if self.mode == "eqtree" and name == "domain":
                eqs += StaticCondensation()
            self.add_equations(eqs @ name)


def test_no_argument_form_takes_the_element_private_dofs_of_its_own_domain():
    with _TwoProjectionDomains("eqtree") as p:
        p.quiet()
        p.initialise()
        _deterministic_solver(p)
        nel = p.get_mesh("domain").nelement()
        assert nel == 4 and p.get_mesh("other").nelement() == 4
        assert p.use_static_condensation, "the no-argument form must switch condensation on too"
        assert p._get_static_condensation_stats()["n_selected"] == nel
        p.solve()
        assert p._last_jacobian_was_condensed()
        assert numpy.allclose(_dofs(p), 1.0), "q = 1 elementwise, eliminated dofs included"

    # The same rule at problem level is problem-wide unless a domain is named.
    with _TwoProjectionDomains("problem_wide") as p:
        p.quiet()
        p.initialise()
        p.condense_element_private_dofs()
        assert p._get_static_condensation_stats()["n_selected"] == 8
        p._clear_static_condensation_rules()
        p.condense_element_private_dofs(domain="domain")
        assert p._get_static_condensation_stats()["n_selected"] == 4


def test_rules_are_restated_against_the_meshes_remeshing_replaces(tmp_path):
    """Remeshing does not adapt the mesh, it builds a new one and destroys the old, so a rule naming a
    domain has to be restated afterwards -- a C++ rule holds the mesh it was given, and that mesh is
    gone. Both routes are covered: the equation's registration hook fires again through
    reapply_boundary_conditions(), and Problem.actions_after_remeshing() re-synchronises whatever was
    declared at problem level, which nothing else would restate."""

    class _Segment(GmshTemplate):
        def define_geometry(self):
            self.default_resolution = 0.3
            p00, p10, p01 = self.point(0, 0), self.point(1, 0), self.point(0, 1)
            self.create_lines(p00, "bottom", p10, "diag", p01, "left", p00)
            self.plane_surface("bottom", "diag", "left", name="domain")

    from pyoomph.equations.generic import ProjectExpression

    for eqtree in (True, False):
        p = Problem()
        with p:
            p.set_output_directory(str(tmp_path / ("eqtree" if eqtree else "problem_level")))
            p.quiet()
            p += _Segment()
            eqs = ProjectExpression(q=1, space="D0")
            if eqtree:
                eqs += StaticCondensation()
            p.additional_equations += eqs @ "domain"
            p.initialise()
            _deterministic_solver(p)
            if not eqtree:
                p.condense_dofs("domain/q")
                p.use_static_condensation = True
            nel = p.get_mesh("domain").nelement()
            assert nel > 4
            assert p._get_static_condensation_stats()["n_selected"] == nel
            p.solve()
            assert p._last_jacobian_was_condensed()

            p.force_remesh()
            nel = p.get_mesh("domain").nelement()
            assert p._get_static_condensation_stats()["n_selected"] == nel, \
                "the rule still names the mesh that remeshing destroyed"
            p.solve()
            assert p._last_jacobian_was_condensed()
            assert numpy.allclose(_dofs(p), 1.0)
