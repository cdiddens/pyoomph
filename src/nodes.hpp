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

The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl

================================================================================*/

#pragma once

#include "oomph_lib.hpp"

namespace pyoomph
{

	class Problem;
	class NodeAccess;
	class FieldDescriptor;
	class Mesh;
	class BulkElementBase;
	class NodeWithFieldIndicesBase
	{
	protected:
		friend class MeshTemplate;
		friend class NodeAccess;
		friend class BulkElementBase;

	public:
		virtual ~NodeWithFieldIndicesBase() = default;
	};

	template <class NODE_TYPE>
	class NodeWithFieldIndices : public NODE_TYPE, public NodeWithFieldIndicesBase
	{
	public:
		NodeWithFieldIndices();

		NodeWithFieldIndices(oomph::TimeStepper *const &time_stepper_pt, const unsigned &n_lagrangian, const unsigned &n_lagrangian_type, const unsigned &n_dim, const unsigned &Nposition_type, const unsigned &initial_n_value) : NODE_TYPE(time_stepper_pt, n_lagrangian, n_lagrangian_type, n_dim, Nposition_type, initial_n_value), NodeWithFieldIndicesBase() {}

		NodeWithFieldIndices(const unsigned &n_lagrangian, const unsigned &n_lagrangian_type, const unsigned &n_dim, const unsigned &Nposition_type, const unsigned &initial_n_value) : NODE_TYPE(n_lagrangian, n_lagrangian_type, n_dim, Nposition_type, initial_n_value), NodeWithFieldIndicesBase() {}

		virtual void resize(const unsigned &n_value)
		{
			NODE_TYPE::resize(n_value);										
		}

		virtual int additional_value_index(unsigned interf_id)
		{
			oomph::BoundaryNodeBase *bn = dynamic_cast<oomph::BoundaryNodeBase *>(this);
			if (!bn)
				return -1;
			std::map<unsigned, unsigned> *&mp = bn->index_of_first_value_assigned_by_face_element_pt();
			if (!mp)
				return -1;
			if (!(*mp).count(interf_id))
				return -1;
			return (*mp)[interf_id];
		}
	};


	typedef NodeWithFieldIndices<oomph::SolidNode> Node;
	class BoundaryNode : public oomph::BoundaryNode<pyoomph::Node>
	{
	public:
		// std::map<void*,std::set<int>> nullified_dofs; //Nullify the dofs on element/element class indiced by the pointer, negative dofs are for positions
		std::map<unsigned, unsigned> *get_additional_dof_map() { return Index_of_first_value_assigned_by_face_element_pt; }
		bool has_additional_dof(const unsigned index)
		{
			//std::cout << "has_additional_dof  " << Index_of_first_value_assigned_by_face_element_pt << std::endl << std::flush;
			if (!Index_of_first_value_assigned_by_face_element_pt)
				return false;
			return Index_of_first_value_assigned_by_face_element_pt->count(index);
		}

		BoundaryNode(const unsigned &n_lagrangian, const unsigned &n_lagrangian_type, const unsigned &n_dim, const unsigned &Nposition_type, const unsigned &initial_n_value) : oomph::BoundaryNode<pyoomph::Node>(n_lagrangian, n_lagrangian_type, n_dim, Nposition_type, initial_n_value) {}

		BoundaryNode(oomph::TimeStepper *const &time_stepper_pt, const unsigned &n_lagrangian, const unsigned &n_lagrangian_type, const unsigned &n_dim, const unsigned &Nposition_type, const unsigned &initial_n_value) : oomph::BoundaryNode<pyoomph::Node>(time_stepper_pt, n_lagrangian, n_lagrangian_type, n_dim, Nposition_type, initial_n_value) {}
	};

};
