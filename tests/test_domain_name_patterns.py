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

# Glob domain names in the @-restriction:
#
#     DirichletBC(u=0) @ "*"        instead of  DirichletBC(u=0) @ ["left","right","top","bottom"]
#     DirichletBC(u=0) @ "wall*"    /  "[lr]*"  /  "domain/*"  /  "left/*"
#
# A pattern cannot be resolved where it is written -- inside define_problem no mesh template has built
# its geometry yet, so no domain or boundary name exists. It therefore survives in the equation tree as
# an ordinary child key and is expanded in Problem._link_geometry_and_equations, right after the
# geometry loop. The oracle for the plain case is that the glob and the enumerated list are
# bit-identical, which is stronger than "it runs" and is what catches a boundary being missed.
#
# Two behaviours here are worth pinning rather than assuming, because both were found the hard way:
# the expanded clones share their Equations objects - which is safe because nothing per-domain is
# stored on an equation, see tests/test_shared_equations.py - and a pattern hands its slot in the
# child order to the names it produced, so an explicitly named sibling stays where it was written
# and therefore still wins over the wildcard.

import numpy
import pytest

from pyoomph import Problem, DirichletBC, ODEEquations
from pyoomph.equations.poisson import PoissonEquation
from pyoomph.generic.codegen import _check_domain_name_or_pattern, _check_for_valid_var_name, _is_domain_name_pattern
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh


class _Poisson(Problem):
    """A unit square with the Dirichlet condition restricted by whatever `restrict` says."""

    def __init__(self, restrict, extra=None, N=4):
        super().__init__()
        self.restrict, self.extra, self.N = restrict, extra, N

    def define_problem(self):
        self += RectangularQuadMesh(size=[1, 1], N=[self.N, self.N])
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ self.restrict
        if self.extra is not None:
            eqs = eqs + self.extra
        self += eqs @ "domain"


def _boundaries(tmp_path, restrict, **kw):
    """The child names the pattern expanded to, after the tree has been built."""
    with _Poisson(restrict, **kw) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        return sorted(p._equation_system.get_children()["domain"].get_children())


def _solve(tmp_path, restrict, **kw):
    with _Poisson(restrict, **kw) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.solve()
        return p.ndof(), numpy.array(p.get_current_dofs()[0])


# ---------------------------------------------------------------- construction-time name checking


@pytest.mark.parametrize("name,is_pattern", [
    ("left", False), ("_meshwide_x", False),
    ("*", True), ("wall*", True), ("[lr]*", True), ("b?ttom", True), ("[!lr]*", True),
])
def test_pattern_detection(name, is_pattern):
    assert _is_domain_name_pattern(name) is is_pattern
    assert _check_domain_name_or_pattern(name) is is_pattern


@pytest.mark.parametrize("name,msg", [
    ("", "Empty domain name"),
    ("do main", "may only contain|may not contain"),
    ("a__b*", "double underscores"),
    ("[left", r"Unbalanced"),
    ("[[a]]*", r"Nested"),
    ("!left", "Exclusion patterns"),
    ("wall+*", "may only contain"),
])
def test_broken_patterns_are_rejected_where_they_are_written(name, msg):
    # A pattern is only expanded at initialise(), so one that can never match has to be caught here --
    # otherwise the user learns about the typo much later, or not at all.
    with pytest.raises(ValueError, match=msg):
        _check_domain_name_or_pattern(name)


def test_variable_names_are_unaffected():
    # _check_for_valid_var_name also guards *variable* names, where a glob must never become legal.
    # That is the whole reason the pattern check is a separate function rather than a flag on it.
    with pytest.raises(ValueError):
        _check_for_valid_var_name("u*", False)


# ---------------------------------------------------------------------------------- what "*" matches


def test_star_matches_every_boundary(tmp_path):
    assert _boundaries(tmp_path, "*") == ["bottom", "left", "right", "top"]


@pytest.mark.parametrize("pattern,expected", [
    ("l*", ["left"]),
    ("[lr]*", ["left", "right"]),
    ("?ight", ["right"]),
    ("[!lr]*", ["bottom", "top"]),
])
def test_more_specific_globs(tmp_path, pattern, expected):
    assert _boundaries(tmp_path, pattern) == expected


def test_glob_in_a_path(tmp_path):
    # "domain/*" is the same restriction written as one path rather than two @s.
    class P(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(size=[1, 1], N=[4, 4])
            self += PoissonEquation(source=1) @ "domain" + DirichletBC(u=0) @ "domain/*"

    with P() as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        assert sorted(p._equation_system.get_children()["domain"].get_children()) == ["bottom", "left", "right", "top"]


def test_star_at_the_root_matches_every_bulk_domain(tmp_path):
    class P(Problem):
        def define_problem(self):
            self += LineMesh(N=10, size=1, name=lambda x: "inner" if x < 0.5 else "outer")
            self += PoissonEquation(source=1) @ "*"
            self += DirichletBC(u=0) @ "inner/left"
            self += DirichletBC(u=0) @ "outer/right"

    with P() as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.solve()
        assert sorted(p._equation_system.get_children()) == ["inner", "outer"]


def test_glob_on_an_interface_matches_its_intersections(tmp_path):
    # One level deeper the candidates are the contact lines, taken from _find_interface_intersections().
    class P(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(size=[1, 1], N=[4, 4])
            self += (PoissonEquation(source=1) + DirichletBC(u=0) @ "left" + DirichletBC(u=0) @ "left/*") @ "domain"

    with P() as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        left = p._equation_system.get_children()["domain"].get_children()["left"]
        assert sorted(left.get_children()) == ["bottom", "top"]


def test_star_includes_the_interface_between_two_bulk_domains(tmp_path):
    # Documented surprise rather than a defect: on a multi-domain mesh the interface between the two
    # domains is a genuine registered boundary (LineMesh names it "<dom1>_<dom2>"), adjacent to both,
    # so "all boundaries of this domain" legitimately contains it.
    class P(Problem):
        def define_problem(self):
            self += LineMesh(N=10, size=1, name=lambda x: "inner" if x < 0.5 else "outer")
            self += (PoissonEquation(source=1) + DirichletBC(u=0) @ "*") @ "inner"
            self += PoissonEquation(source=1) @ "outer"
            self += DirichletBC(u=0) @ "outer/right"

    with P() as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        assert sorted(p._equation_system.get_children()["inner"].get_children()) == ["inner_outer", "left"]


# ------------------------------------------------------------------- equivalence with the list form


def test_star_is_bit_identical_to_the_enumerated_list(tmp_path):
    ndof_g, sol_g = _solve(tmp_path / "glob", "*", N=6)
    ndof_l, sol_l = _solve(tmp_path / "list", ["left", "right", "top", "bottom"], N=6)
    assert ndof_g == ndof_l
    # Guards against a vacuous comparison: an all-zero solution would match whatever was pinned.
    assert numpy.max(numpy.abs(sol_l)) > 1e-3
    assert numpy.max(numpy.abs(sol_g - sol_l)) == 0.0


def test_expanded_equations_match_the_list_form(tmp_path):
    # The clones share their Equations objects, exactly as eqs @ ["a","b"] does. Each domain
    # resolves such an instance on its own, so the two spellings produce the same tree.
    with _Poisson("*") as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        glob = str(p._equation_system)
    with _Poisson(["left", "right", "top", "bottom"]) as p:
        p.set_output_directory(str(tmp_path / "list"))
        p.quiet()
        p.initialise()
        lst = str(p._equation_system)
    # The two differ only in the order the children were inserted in.
    assert sorted(glob.splitlines()) == sorted(lst.splitlines())


def test_an_explicit_boundary_still_overrides_the_wildcard(tmp_path):
    # The pattern hands its slot in the child order to the names it produced, so a boundary the user
    # named explicitly keeps the position it was written at and is therefore applied last.
    with _Poisson("*", extra=DirichletBC(u=1) @ "left", N=6) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.solve()
        on_left = {round(n.value(0), 10) for n in p.get_mesh("domain/left").nodes()}
        on_right = {round(n.value(0), 10) for n in p.get_mesh("domain/right").nodes()}
    assert on_left == {1.0}
    assert on_right == {0.0}


# ------------------------------------------------------------------------------------------- errors


def test_a_pattern_matching_nothing_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="does not match anything") as exc:
        _boundaries(tmp_path, "nomatch*")
    # The message has to name the alternatives, like the existing unknown-boundary error does.
    assert "left" in str(exc.value) and "bottom" in str(exc.value)


def test_a_pattern_without_a_mesh_template_is_an_error(tmp_path):
    class P(Problem):
        def define_problem(self):
            self += RectangularQuadMesh(size=[1, 1], N=[4, 4])
            self += PoissonEquation(source=1) @ "domain"
            self += _Oscillator() @ "osc"
            self += DirichletBC(y=0) @ "osc/*"

    with pytest.raises(RuntimeError, match="no mesh template with a domain named"):
        with P() as p:
            p.set_output_directory(str(tmp_path))
            p.quiet()
            p.initialise()


def test_a_pattern_below_the_supported_depth_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="only supported for bulk domains"):
        _boundaries(tmp_path, "left/bottom/*")


class _Oscillator(ODEEquations):
    def define_fields(self):
        self.define_ode_variable("y")

    def define_residuals(self):
        from pyoomph.expressions import var, partial_t, testfunction
        y = var("y")
        self.add_residual((partial_t(y) + y) * testfunction(y))


# ------------------------------------------------------------------------ stability of the expansion


def test_the_expansion_is_deterministic(tmp_path):
    def tree(where):
        with _Poisson("*") as p:
            p.set_output_directory(str(where))
            p.quiet()
            p.initialise()
            return open(str(where / "_ccode" / "_equation_tree.txt")).read()

    assert tree(tmp_path / "a") == tree(tmp_path / "b")


def test_expanding_twice_changes_nothing(tmp_path):
    # _link_geometry_and_equations runs a second time when a problem is redefined, on a tree that may
    # already have been expanded. The pass has to be idempotent for that, which it is because a pattern
    # is removed as it expands.
    with _Poisson("*") as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.initialise()
        before = str(p._equation_system)
        p._equation_system._expand_domain_name_patterns(p._domain_name_pattern_candidates)
        assert str(p._equation_system) == before
