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
 
from .problem import Problem,GenericProblemHooks
# Layouts for the global dof numbering (problem.dof_ordering); see pyoomph/generic/dof_ordering.py
from .dof_ordering import NodalBlockOrdering,ElementBlockOrdering
# Only the base classes live in codegen now - the equation classes a user actually instantiates
# (WeakContribution, ScalarField, GlobalLagrangeMultiplier, ...) moved to pyoomph.equations.generic,
# from where the top-level "from pyoomph import *" picks them up. They are deliberately not
# re-imported here: pyoomph.generic is imported first while the package is still initialising, and
# pulling pyoomph.equations in at that point would make the import order circular.
from .codegen import Equations,ODEEquations,ScalingException,InterfaceEquations

__all__ = ["Problem", "GenericProblemHooks","Equations",
           "ScalingException",
           "ODEEquations","InterfaceEquations",
           "NodalBlockOrdering","ElementBlockOrdering"]
