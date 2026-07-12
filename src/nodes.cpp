/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. 

The main author may be contacted at c.diddens@utwente.nl

================================================================================*/


#include "nodes.hpp"
// #include "meshtemplate.hpp"
#include "problem.hpp"
#include "exception.hpp"

namespace pyoomph
{

	// Explicit template instantiation of pyoomph's Node type (see the typedef in
	// nodes.hpp), so that its member functions are compiled once here rather than in every
	// translation unit that uses pyoomph::Node.
	template class NodeWithFieldIndices<oomph::SolidNode>;

}
