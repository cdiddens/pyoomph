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

#include <array>
#include <vector>
#include <utility>
#include "oomph_lib.hpp"

namespace pyoomph
{

  // PYOOMPH_DISABLE_TOPOLOGICAL_INTERFACE_KEYS=1 forces the coupled-interface machinery back onto
  // position matching. Only for negative tests: the mixed-order cases must FAIL with it set, and a test
  // that passes both ways is measuring nothing. Defined in mesh.cpp.
  bool topological_interface_keys_disabled();
  // Re-read the environment variable behind it; called once per mesh per adaptation.
  void refresh_topological_interface_key_setting();

  // --- The digest behind pyoomph::Node::interface_topological_id (defined in mesh.cpp) ---
  // A node identity is 128 bits mixed over a canonical description of where the node comes from, so it
  // stays a fixed size at any refinement depth. Both the MeshTemplate (level-0 nodes) and the refinement
  // sweep build ids with these, and they MUST agree: a C2 domain's midside node is a template node while
  // the C1 domain's node at the same place is created by a refinement, and those two have to digest to
  // the same value or the two sides stop recognising each other.
  std::array<unsigned long long, 2> topo_digest_of_template_index(std::size_t template_index);
  // The identity of a point given as an exact C1 combination of TEMPLATE nodes; see
  // pyoomph::Node::interface_topological_expansion. Canonicalises (sorts, merges, drops zeros) in place.
  std::array<unsigned long long, 2> topo_digest_of_expansion(std::vector<std::pair<std::size_t, double>> &expansion);
  // ... and of an entity that has no such dyadic description (a triangle's centroid bubble, weight 1/3):
  // deterministic and distinct, but deliberately NOT comparable with a refinement-created node, since no
  // refinement ever creates a node there.
  std::array<unsigned long long, 2> topo_digest_of_corner_set(const std::vector<std::size_t> &sorted_corners);
  // Identity for a point whose C1 description is not dyadic (a centroid bubble); see mesh.cpp.
  std::array<unsigned long long, 2> topo_digest_of_opaque_expansion(std::vector<std::pair<std::size_t, double>> expansion);
  // A C1 shape weight as an exact dyadic integer; throws if it is not one.
  long long topo_weight_exact(double w);
  // ... and the non-throwing question, for deciding whether a node can be described this way at all.
  bool topo_weight_is_dyadic(double w);

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
    AdditionalDofConstrainingInfo(unsigned index, AdditionalDofConstraintMode mode) : mode(mode), index(index), next(NULL) {}
  };

  class Problem;
  class NodeAccess;
  class FieldDescriptor;
  class Mesh;
  class BulkElementBase;
  class BoundaryNode;
  // Empty tag/marker base class that lets pyoomph's node types (see NodeWithFieldIndices
  // below) be identified/friended independently of the oomph-lib node template they wrap.
  class NodeWithFieldIndicesBase
  {
  protected:
    friend class MeshTemplate;
    friend class NodeAccess;
    friend class BulkElementBase;
    AdditionalDofConstrainingInfo *additional_dof_constraints = NULL; // Linked list of additional dofs that constrain this node's, to e.g. reduce from C2 to C1 locally
    // Flattened one-level C1-corner expansion for a node that is constrained to C1 (i.e. a non-vertex
    // node in its home element): the element's C1 vertex nodes whose average defines this node's
    // constrained value, each with weight 1/ncorners. Empty for unconstrained nodes. Purely
    // geometric, so it is shared by all constrained value indices and by position constraints. It is
    // (re)computed by BulkElementBase::setup_additional_dof_constraints after constraints are applied
    // (and hence after every mesh adaptation) and lets a constrained node's dof be recursively
    // flattened into real free dofs even when the node is reached as a master from a *neighbouring*
    // element (where it may be a C1 vertex and its own corners are not locally known). This is what
    // makes ConstrainFieldsToC1Space / ConstrainPositionsToC1Space compose with genuine adaptive
    // hanging nodes. See dev_docs/adaptive_refinement.md section 3.
    std::vector<std::pair<oomph::Node*, double>> c1_constraint_corners;
    // Where this node came from, as TOPOLOGY: the (father node, rounded father-shape weight) pairs of the
    // point at which a refinement created it -- exactly the key the per-round shared-node registries use
    // (RefineablePyramidElement::Shared_node_registry and friends). Empty for a node that was not born of
    // a refinement, and for the element families whose sharing is resolved by a tree walk instead
    // (2d triangles, tets outside a mixed forest).
    //
    // The point of storing it is the CROSS-ROUND case. Two elements meeting at a facet compute the same
    // key for a node on it, because both compute it from their own element at the level where the node is
    // born and those two elements share the facet's nodes -- but only if they are split in the SAME round
    // does the per-round registry see both. When one side was refined rounds ago, the key it used is gone.
    // Keeping it ON the node lets the snapshot be rebuilt from the live mesh at the start of every round
    // (Mesh::split_elements_if_required), which is safe by construction: a node that is still in the mesh
    // cannot have generating nodes that are not, since unrefinement removes the finer nodes first.
    // This replaces matching by POSITION, which is only as good as the positions are -- and a hanging
    // node's position is a cache of its masters that goes stale whenever something writes the dof vector
    // from outside the Newton solver.
    std::vector<std::pair<oomph::Node*, long long>> refinement_generating_key;
    // Cross-domain topological identity of this node, as a 128-bit digest. {0,0} means "not assigned".
    //
    // refinement_generating_key above is the same idea WITHIN one mesh, and says why: matching by
    // POSITION is only as good as the positions are. Across two coupled domains it cannot be used --
    // the two sides' nodes are distinct objects, so father-node POINTERS mean nothing to each other --
    // and yet that is exactly where position matching hurts most: a C2 domain's interface is a
    // quadratic curve through three nodes while a C1 domain's is the chord between two of them, so a
    // refinement promotes an off-chord midside node to a vertex on one side and creates a chord
    // midpoint on the other (dev_docs/interface_refinement_coupling.md section 14.3).
    //
    // The identity is therefore built from what the two domains DO share, which is the MeshTemplate:
    //   * a node created by the generator is stamped with its MeshTemplateNode index. Both domains read
    //     the same template, so the same interface node gets the same number in both;
    //   * a node created by a refinement is the C1 (vertex-linear) interpolation of its father's vertex
    //     nodes at a dyadic local coordinate, so its identity is the canonical set of
    //     (father vertex id, exact dyadic weight) pairs. Only the corners of the shared facet have a
    //     non-zero weight there, and those agree by the line above -- whatever the element family, the
    //     element space, or where the node physically ended up.
    // Digested to 128 bits so the id is a fixed size at any refinement depth (a collision over the ~1e5
    // keys in play is ~1e-29). Assigned by Mesh::assign_interface_topological_ids().
    std::array<unsigned long long, 2> interface_topological_id = {0ULL, 0ULL};
    // The identity in its unreduced form: this node's position as an exact C1 combination of TEMPLATE
    // nodes, (template node index, weight), sorted by index with the weights summed.
    //
    // It has to be flattened all the way to the template, not expressed over the node's immediate
    // parents, or it is not canonical. A node a quarter of the way along a level-0 edge is reached by a
    // C2 domain as "the midside node of a level-1 element" (weights 3/4, 1/4 over the two level-0
    // corners) and by a C1 domain as "the midpoint of a level-1 edge" (weights 1/2, 1/2 over a level-0
    // corner and the level-0 midpoint). Digesting the immediate parents gives those two different
    // values; flattening gives both 3/4, 1/4. The weights are dyadic, hence exact in double.
    //
    // Empty for a node whose position has no such description -- a triangle's centroid bubble is 1/3 of
    // each corner, which no refinement ever produces, so it gets an opaque id instead and never appears
    // in anyone else's expansion (only C1 CORNERS do).
    std::vector<std::pair<std::size_t, double>> interface_topological_expansion;
  public:
    // Is this node a pyoomph::BoundaryNode? Overridden there to return `this`; the base returns NULL.
    // Every node in a mesh is a pyoomph::Node (see the Node typedef below), so a caller holding an
    // oomph::Node* answers "is it a boundary node, and if so which one" with a static_cast to
    // pyoomph::Node plus this virtual call, instead of a dynamic_cast down a diamond that costs a
    // few hundred cycles per query in the dof-numbering and interface-setup loops.
    virtual pyoomph::BoundaryNode *as_boundary_node() { return NULL; }
    const pyoomph::BoundaryNode *as_boundary_node() const { return const_cast<NodeWithFieldIndicesBase *>(this)->as_boundary_node(); }

    // See refinement_generating_key. Set once, when a refinement creates the node.
    virtual void set_refinement_generating_key(const std::vector<std::pair<oomph::Node*, long long>> &k) { refinement_generating_key = k; }
    virtual const std::vector<std::pair<oomph::Node*, long long>> &get_refinement_generating_key() const { return refinement_generating_key; }
    // See interface_topological_id / interface_topological_expansion.
    void set_interface_topological_id(const std::array<unsigned long long, 2> &id) { interface_topological_id = id; }
    void set_interface_topological_expansion(const std::vector<std::pair<std::size_t, double>> &e) { interface_topological_expansion = e; }
    const std::vector<std::pair<std::size_t, double>> &get_interface_topological_expansion() const { return interface_topological_expansion; }
    const std::array<unsigned long long, 2> &get_interface_topological_id() const { return interface_topological_id; }
    bool has_interface_topological_id() const { return interface_topological_id[0] || interface_topological_id[1]; }
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

    void resize(const unsigned &n_value) override
    {
      NODE_TYPE::resize(n_value);
    }

    // Look up the index (within this node's value storage) of the first value that a
    // FaceElement with interface id `interf_id` was assigned on this (boundary) node.
    // Returns -1 if this node is not a boundary node, or has no such assignment.
    virtual int additional_value_index(unsigned interf_id)
    {
      oomph::BoundaryNodeBase *bn = this->as_boundary_node();
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
  //
  // INVARIANT: every node of every pyoomph mesh is a pyoomph::Node (or the pyoomph::BoundaryNode
  // below, which derives from it). Nodes are only ever made by BulkElementBase::construct_node /
  // construct_boundary_node, by the mesh templates, and - under MPI - by the external-halo master
  // reconstruction in missing_masters.hpp; all of them build this type. Because oomph::Node ->
  // oomph::SolidNode -> pyoomph::Node is a plain non-virtual chain, the downcast from an
  // oomph::Node* is therefore a static_cast, and that is how it is written throughout: a
  // dynamic_cast here is not a safety net, it is just a slower way to compute the same pointer,
  // and it is on the per-node path of assembly, dof numbering and hanging-node interpolation.
  // (The one oomph::Node in the codebase that is NOT one of these is bifurcation.cpp's TimeNode,
  // which lives in its own standalone collocation mesh that no mesh/element code ever sees.)
  typedef NodeWithFieldIndices<oomph::SolidNode> Node;
  // pyoomph's boundary node type, adding storage/lookup for extra ("additional") dof
  // indices that FaceElements attached to this boundary node assign beyond the bulk node's
  // own values (e.g. Lagrange multipliers or surface-only fields living on the boundary).
  class BoundaryNode : public oomph::BoundaryNode<pyoomph::Node>
  {
  public:
    using NodeWithFieldIndicesBase::as_boundary_node; // keep the const overload visible past the override
    pyoomph::BoundaryNode *as_boundary_node() override { return this; }

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
