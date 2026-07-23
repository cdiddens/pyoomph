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

import re
import warnings
from .. import _pyoomph_core as _pyoomph

# Characters that are special to LaTeX and must be escaped whenever we fall back to
# auto-generating a name (i.e. it was not given an explicit entry in LaTeXPrinter.replace).
_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "^": r"\^{}",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "~": r"\~{}",
}

# A plain run of letters/digits is already valid (italicized) LaTeX math, so it is
# passed through as-is instead of being wrapped in \mathrm{...}.
_SIMPLE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


def _escape_latex(s):
    return "".join(_LATEX_SPECIAL_CHARS.get(ch, ch) for ch in s)


class LaTeXEquationTreeNode:
    """One node (domain or interface) of the equation tree, mirroring the pyoomph EquationTree
    structure just for the purpose of grouping rendered LaTeX by domain/interface."""

    def __init__(self, parent, name):
        self._name = name
        self._parent = parent
        self._children = {}
        self._residuals = {}  # test_name -> rendered residual LaTeX (the "as compiled" form)
        self._subexpr_definitions = {}  # cvar -> rendered definition LaTeX, in first-seen order
        self._as_entered = []  # rendered "as entered" (still-held grad/div/dot/...) residual snapshots

    def get_full_name(self):
        if self._parent:
            return self._parent.get_full_name() + "/" + self._name
        else:
            return self._name

    def write_contribution(self, texwriter, f, level):
        texwriter.write_equation_tree_contribution(self, f, level)
        for _, c in self._children.items():
            c.write_contribution(texwriter, f, level + 1)


class LaTeXPrinter(_pyoomph.LaTeXPrinter):
    def __init__(self):
        super(LaTeXPrinter, self).__init__()
        self._eqtree = {}
        self.replace = {}
        self.replace["pressure"] = "p"
        self.replace["velocity_x"] = "u_x"
        self.replace["velocity_y"] = "u_y"
        self.replace["velocity_z"] = "u_z"
        self.replace["coordinate_x"] = "x"
        self.replace["coordinate_y"] = "y"
        self.replace["coordinate_z"] = "z"

    # ------------------------------------------------------------------
    # Name handling: self.replace lets users override how a raw field/domain
    # name is rendered; anything not in there is auto-escaped into \mathrm{...}
    # so the output always compiles, even if it is not particularly pretty.
    # ------------------------------------------------------------------
    def expand_tex_name(self, varname):
        if varname in self.replace:
            return self.replace[varname]
        if _SIMPLE_NAME_RE.match(varname):
            return varname
        return r"\mathrm{" + _escape_latex(varname) + "}"

    def fieldname_to_testfunction(self, fieldname, info):
        fn = self.expand_tex_name(fieldname)
        return r"\psi_{" + fn + "}"

    def domain_annotation(self, info, code):
        # If a field/testfunction is evaluated on a domain different from the one
        # whose residual we are currently rendering (e.g. a bulk field accessed from
        # an interface, or vice versa), mark it so the reader knows it is not local.
        dom = info.get("domain", "")
        if code is None or not dom:
            return ""
        try:
            local_dom = code.get_domain_name()
        except AttributeError:
            return ""
        if dom != local_dom:
            return r"\big|_{" + self.expand_tex_name(dom) + "}"
        return ""

    def augment_partial_t(self, varstr, order):
        if order == 2:
            return r"\partial_t^2{" + varstr + "}"
        else:
            return r"\partial_t{" + varstr + "}"

    def augment_partial_x(self, varstr, dir):
        return r"\partial_{" + dir + r"}{" + varstr + "}"

    def augment_basis_derivatives(self, res, info):
        basis = info.get("basis", "")
        if basis.startswith("d/dx "):
            res = self.augment_partial_x(res, "x")
        if basis.startswith("d/dy "):
            res = self.augment_partial_x(res, "y")
        if basis.startswith("d/dz "):
            res = self.augment_partial_x(res, "z")
        return res

    def _augment_derived(self, base, info):
        # Shared helper for symbols that can represent a first/second derivative
        # w.r.t. a nodal coordinate direction (element size, normal, dx/dX, ...).
        dirs = ["x", "y", "z"]
        d1 = info.get("derived_in_direction", "none")
        d2 = info.get("derived_in_direction2", "none")
        if d1 == "none":
            return base
        d1i = int(d1)
        dname1 = dirs[d1i] if d1i < len(dirs) else d1
        if d2 != "none":
            d2i = int(d2)
            dname2 = dirs[d2i] if d2i < len(dirs) else d2
            return r"\partial_{" + dname1 + r"}\partial_{" + dname2 + r"}\left(" + base + r"\right)"
        return r"\partial_{" + dname1 + r"}\left(" + base + r"\right)"

    def field(self, info, code=None):
        res = self.expand_tex_name(info.get("name", ""))
        res = self.augment_basis_derivatives(res, info)
        timediff = info.get("timediff") or ""
        if timediff == "d/dt ":
            res = self.augment_partial_t(res, 1)
        elif timediff.startswith("d^2/dt^2"):
            res = self.augment_partial_t(res, 2)
        res += self.domain_annotation(info, code)
        return res

    def testfunction(self, info, code=None):
        res = self.fieldname_to_testfunction(info.get("name", ""), info)
        res = self.augment_basis_derivatives(res, info)
        res += self.domain_annotation(info, code)
        return res

    def residual_symbol(self, testname):
        return r"\mathcal{R}_{" + testname + "}"

    # ------------------------------------------------------------------
    # Handlers for the remaining leaf symbol types, dispatched to by
    # _get_LaTeX_expression() below based on info["typ"].
    # ------------------------------------------------------------------
    def spatial_integral_symbol(self, info, code=None):
        base = r"{\rm d}X" if info.get("lagrangian", "false") == "true" else r"{\rm d}x"
        return self._augment_derived(base, info)

    def element_size_symbol(self, info, code=None):
        base = r"h_X" if info.get("lagrangian", "false") == "true" else r"h_x"
        if info.get("with_coordsys", "true") == "false":
            base = base + r"^{\rm cart}"
        return self._augment_derived(base, info)

    def normal_symbol(self, info, code=None):
        dirs = ["x", "y", "z"]
        d = int(info.get("direction", "0"))
        dname = dirs[d] if d < len(dirs) else str(d)
        base = r"n_{" + dname + "}"
        return self._augment_derived(base, info)

    def nodal_delta_symbol(self, info, code=None):
        return r"\delta_{\rm nodal}"

    def subexpression(self, info, code=None):
        cvar = info.get("cvar", "")
        idx = cvar.rsplit("_", 1)[-1]
        return "S_{" + _escape_latex(idx) + "}"

    # ------------------------------------------------------------------
    # Handlers for the "as entered" (still-held, unexpanded) vector-calculus operators - see
    # expand_held_calc_ops() on the C++ side for why these exist as distinct leaf types at all.
    # ------------------------------------------------------------------
    def held_grad(self, info, code=None):
        return r"\nabla\left(" + info.get("arg", "") + r"\right)"

    def held_div(self, info, code=None):
        return r"\nabla\cdot\left(" + info.get("arg", "") + r"\right)"

    def held_directional_derivative(self, info, code=None):
        return r"\left(" + info.get("direction", "") + r"\cdot\nabla\right)\left(" + info.get("arg", "") + r"\right)"

    def held_dot(self, info, code=None):
        return r"\left(" + info.get("a", "") + r"\right)\cdot\left(" + info.get("b", "") + r"\right)"

    def held_double_dot(self, info, code=None):
        return r"\left(" + info.get("a", "") + r"\right):\left(" + info.get("b", "") + r"\right)"

    def held_contract(self, info, code=None):
        return r"\left(" + info.get("a", "") + r"\right)\odot\left(" + info.get("b", "") + r"\right)"

    def held_weak(self, info, code=None):
        dx = r"{\rm d}X" if info.get("lagrangian", "false") == "true" else r"{\rm d}x"
        return r"\left(" + info.get("a", "") + r"\right)\left(" + info.get("b", "") + r"\right)" + dx

    def multiret_callback(self, info, code=None):
        id_name = info.get("id_name", "")
        index = info.get("index", "?")
        retindex = info.get("retindex", "0")
        args = info.get("args", "")
        if id_name and id_name != "unknown multi-ret cb":
            name = self.expand_tex_name(id_name)
        else:
            name = r"\mathrm{cb}_{" + _escape_latex(str(index)) + "}"
        res = name + r"_{" + _escape_latex(str(retindex)) + r"}\left(" + args + r"\right)"
        if info.get("derived_by_arg", "none") != "none":
            res = r"\partial_{" + _escape_latex(info["derived_by_arg"]) + r"}" + res
        return res

    # ------------------------------------------------------------------
    # Sinks called from C++: _add_LaTeX_expression records a fully rendered
    # (side-effecting) expression, _get_LaTeX_expression renders a single leaf.
    # ------------------------------------------------------------------
    def _get_tree_node(self, code):
        domain_name = code.get_full_name()
        parent_list = domain_name.split("/")
        childlist = self._eqtree
        entry = None
        parent_tree = None
        for p in parent_list:
            if p not in childlist:
                entry = LaTeXEquationTreeNode(parent_tree, p)
                childlist[p] = entry
            else:
                entry = childlist[p]
            childlist = entry._children
            parent_tree = entry
        assert entry is not None
        return entry

    def _add_LaTeX_expression(self, info, tex, code):
        typ = info.get("typ")
        if typ == "final_residual":
            entry = self._get_tree_node(code)
            entry._residuals[info.get("test_name", "")] = tex
        elif typ == "subexpression_definition":
            entry = self._get_tree_node(code)
            entry._subexpr_definitions[info.get("cvar", "")] = tex
        elif typ == "as_entered_residual":
            entry = self._get_tree_node(code)
            entry._as_entered.append(tex)
        else:
            warnings.warn("LaTeXPrinter._add_LaTeX_expression: unhandled typ " + str(typ))

    def _get_LaTeX_expression(self, info, code):
        typ = info.get("typ")
        if typ is not None and hasattr(self, typ):
            return getattr(self, typ)(info, code)
        warnings.warn("LaTeXPrinter._get_LaTeX_expression: unhandled typ " + str(typ))
        return r"\mathrm{unknown}"

    # ------------------------------------------------------------------
    # File writing
    # ------------------------------------------------------------------
    def write_use_package(self, f):
        f.write(r"\usepackage{breqn}" + "\n")
        f.write(r"\breqnsetup{breakdepth={5}}" + "\n")

    def write_header(self, f):
        f.write(r"\documentclass{article}" + "\n")
        self.write_use_package(f)
        f.write(r"\begin{document}" + "\n")

    def write_residual_contributions(self, f):
        for _, root in self._eqtree.items():
            root.write_contribution(self, f, 0)

    def write_equation_tree_contribution(self, eqtree, f, level):
        secname = "sub" * level
        typ = "Domain" if level == 0 else "Boundary "
        f.write("\\" + secname + "section{" + typ + " \\textsl{" + eqtree.get_full_name() + "}}\n")
        if eqtree._as_entered:
            f.write("\\" + secname + "subsection{As entered}\n")
            for tex in eqtree._as_entered:
                f.write(r"\begin{dmath*}" + "\n")
                f.write(tex)
                f.write(r"\end{dmath*}" + "\n")
        if eqtree._subexpr_definitions:
            f.write("\\" + secname + "subsection{Auxiliary expressions}\n")
            for cvar, tex in eqtree._subexpr_definitions.items():
                idx = cvar.rsplit("_", 1)[-1]
                f.write(r"\begin{dmath*}" + "\n")
                f.write("S_{" + _escape_latex(idx) + "}=" + tex)
                f.write(r"\end{dmath*}" + "\n")
        if eqtree._residuals:
            f.write("\\" + secname + "subsection{Residual contributions}\n")
            for testname, residual in eqtree._residuals.items():
                f.write(r"\begin{dmath*}" + "\n")
                tex_testname = self.expand_tex_name(testname)
                f.write(self.residual_symbol(tex_testname) + "=" + residual)
                f.write(r"\end{dmath*}" + "\n")

    def write_footer(self, f):
        f.write(r"\end{document}")

    def write_to_file(self, fname):
        with open(fname, "w") as f:
            self.write_header(f)
            self.write_residual_contributions(f)
            self.write_footer(f)
