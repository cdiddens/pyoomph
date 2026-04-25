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
 
import _pyoomph

class LaTeXEquationTreeNode:
    def __init__(self,parent,name):
        super(LaTeXEquationTreeNode, self).__init__()
        self._name=name
        self._parent=parent
        self._children={}
        self._residual_contributions={}

    def write_contribution(self,texwriter,f,level):
        texwriter.write_equation_tree_contribution(self,f,level)
        for n,c in self._children.items():
            c.write_contribution(texwriter,f,level+1)

    def get_full_name(self):
        if self._parent:
            return self._parent.get_full_name()+"/"+self._name
        else:
            return self._name


class LaTeXPrinter(_pyoomph.LaTeXPrinter):
    def __init__(self):
        super(LaTeXPrinter, self).__init__()
        self._eqtree={}
        self.replace={}
        self.replace["pressure"] = "p"
        self.replace["velocity_x"] = "u_x"
        self.replace["velocity_y"] = "u_y"

    def fieldname_to_testfunction(self,fieldname,info):
        fn = self.expand_tex_name(fieldname)
        return r"\psi_{"+fn+"}"


    # Expansion of the spatial integral symbol
    def spatial_integral_symbol(self,info):
        if info.get("lagrangian","false")=="true":
            return r"{\rm d}X"
        else:
            return r"{\rm d}x"

    def augment_partial_t(self,varstr,order):
        if order==2:
            return r"\partial_t^2{" + varstr+"}"
        else:
            return r"\partial_t{"+varstr+"}"

    def augment_partial_x(self,varstr,dir):
        return r"\partial_{"+dir+r"}{"+varstr+"}"

    def expand_tex_name(self,varname):
        if varname in self.replace.keys():
            return self.replace[varname]
        else:
            return varname

    def augment_basis_derivatives(self,res,info):
        if info.get("basis").startswith("d/dx "):
            res=self.augment_partial_x(res,"x")
        if info.get("basis").startswith("d/dy "):
            res=self.augment_partial_x(res,"y")
        if info.get("basis").startswith("d/dz "):
            res = self.augment_partial_x(res, "z")
        return res

    def field(self,info):
        res=info.get("name")
        res=self.expand_tex_name(res)
        res=self.augment_basis_derivatives(res,info)
        if info.get("timediff")=="d/dt ":
            res=self.augment_partial_t(res,1)
        return res

    def testfunction(self,info):
        res = self.fieldname_to_testfunction(info.get("name"),info)
        res = self.augment_basis_derivatives(res,info)
        return res

    def residual_symbol(self,testname):
        return "\\mathcal{R}_{"+testname+"}"

    def _add_LaTeX_expression(self,info,tex,code):
        print("TEX",info,tex,code)
        if info.get("typ")=="final_residual":
            domain_name=code.get_full_name()
            parent_list=domain_name.split("/")
            childlist=self._eqtree
            entry=None
            parent_tree=None
            for p in parent_list:
                if p not in childlist.keys():
                    entry=LaTeXEquationTreeNode(parent_tree,p)
                    childlist[p]=entry
                    childlist=entry._children
                else:
                    entry=childlist[p]
                    childlist = childlist[p]._children
                parent_tree=entry

            contribution=""
            if not contribution in entry._residual_contributions.keys():
                entry._residual_contributions[contribution]={}
            entry._residual_contributions[contribution][info.get("test_name")]=tex

        else:
            raise RuntimeError(str(["ADDING LATEX",info,tex]))

    def _get_LaTeX_expression(self,info,code):
        #print("GETTING LATEX",info)
        if "typ" in info.keys():
            if hasattr(self,info["typ"]):
                return getattr(self,info["typ"])(info)
        return "I MUST RETURN HERE"

    def write_use_package(self,f):
        f.write(r"\usepackage{breqn}"+"\n")
        f.write(r"\breqnsetup{breakdepth={5}}"+"\n")

    def write_header(self,f):
        f.write(r"\documentclass{article}"+"\n")
        self.write_use_package(f)
        f.write(r"\begin{document}"+"\n")


    def write_residual_contributions(self,f):
        for name,root in self._eqtree.items():
            print("NAME",name)
            root.write_contribution(self,f,0)


    def write_equation_tree_contribution(self,eqtree,f,level):
        secname="sub"*(level)
        if level==0:
            typ="Domain"
        else:
            typ="Boundary "
        f.write("\\"+secname+"section{"+typ+" \\textsl{"+eqtree.get_full_name()+"}}\n")
        for contrib,contribution in eqtree._residual_contributions.items():
            for testname,residual in contribution.items():
                f.write(r"\begin{dmath*}"+"\n")
                tex_testname=self.expand_tex_name(testname)
                f.write(self.residual_symbol(tex_testname)+"="+residual)
                f.write(r"\end{dmath*}"+"\n")


    def write_footer(self,f):
        f.write(r"\end{document}")

    def write_to_file(self,fname):
        with open(fname,"w") as f:
            self.write_header(f)
            self.write_residual_contributions(f)
            self.write_footer(f)
