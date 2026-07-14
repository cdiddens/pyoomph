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

#pragma once

#include "oomph_lib.hpp"

namespace pyoomph
{

  enum AdditionalDofConstraintMode : unsigned
  {
    CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1,
    INTERFACE_DOF_CONSTRAIN_TO_C1,
    POSITION_CONSTRAIN_TO_C1
  };


  struct AdditionalDofConstrainingInfo
  {
    AdditionalDofConstraintMode mode; // Mode, e.g. what to do with this
    unsigned index; // Index, if mode==CONTINUOUS_BASE_DOF_CONSTRAIN_TO_C1, it indicates a value index, if mode==INTERFACE_DOF_CONSTRAIN_TO_C1, it indicates an interface id, if mode==POSITION_CONSTRAIN_TO_C1, it indicates a coordinate index
    AdditionalDofConstrainingInfo *next; // Next in linked list
    AdditionalDofConstrainingInfo(unsigned index, AdditionalDofConstraintMode mode) : index(index), mode(mode), next(NULL) {}
  };

  class Problem;
  class NodeAccess;
  class FieldDescriptor;
  class Mesh;
  class BulkElementBase;
  // Empty tag/marker base class that lets pyoomph's node types (see NodeWithFieldIndices
  // below) be identified/friended independently of the oomph-lib node template they wrap.
  class NodeWithFieldIndicesBase
  {
  protected:
    friend class MeshTemplate;
    friend class NodeAccess;
    friend class BulkElementBase;
    AdditionalDofConstrainingInfo *additional_dof_constraints = NULL; // Linked list of additional dofs that constrain this node's, to e.g. reduce from C2 to C1 locally
  public:
    virtual void add_additional_dof_constraint(unsigned index, AdditionalDofConstraintMode mode);
    virtual void remove_additional_dof_constraint(unsigned index, AdditionalDofConstraintMode mode);
    virtual const AdditionalDofConstrainingInfo *get_additional_dof_constraints() const { return additional_dof_constraints; }
    virtual void flush_additional_dof_constraints();
    virtual ~NodeWithFieldIndicesBase();
  };
  
  // Mixin that adds pyoomph-specific bookkeeping on top of an oomph-lib node type
  // (NODE_TYPE, e.g. oomph::SolidNode - see the Node typedef below). Currently this only
  // adds additional_value_index(), which looks up the extra per-interface value index a
  // FaceElement may have assigned to this node (via a BoundaryNodeBase), returning -1 if
  // the node is not a boundary node or has no such assignment.
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

    // Look up the index (within this node's value storage) of the first value that a
    // FaceElement with interface id `interf_id` was assigned on this (boundary) node.
    // Returns -1 if this node is not a boundary node, or has no such assignment.
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


  // pyoomph's standard node type: an oomph::SolidNode (i.e. a node that carries both
  // Eulerian and Lagrangian position, for use with moving/deforming meshes) extended with
  // the field-index bookkeeping of NodeWithFieldIndices.
  typedef NodeWithFieldIndices<oomph::SolidNode> Node;
  // pyoomph's boundary node type, adding storage/lookup for extra ("additional") dof
  // indices that FaceElements attached to this boundary node assign beyond the bulk node's
  // own values (e.g. Lagrange multipliers or surface-only fields living on the boundary).
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
