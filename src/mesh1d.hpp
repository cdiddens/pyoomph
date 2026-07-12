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

#include "mesh.hpp"

namespace pyoomph
{

  // DynamicTree specialization for 1d (binary-tree-refined) elements, combining oomph-lib's
  // BinaryTree geometry with pyoomph's dynamic (JIT-compiled) element support.
  class DynamicBinaryTree : public virtual oomph::BinaryTree, public virtual DynamicTree
  {
  protected:
    DynamicBinaryTree(oomph::RefineableElement *const &object_pt) : oomph::Tree(object_pt), oomph::BinaryTree(object_pt), DynamicTree(object_pt) {}

    DynamicBinaryTree(oomph::RefineableElement *const &object_pt, oomph::Tree *const &father_pt, const int &son_type) : oomph::Tree(object_pt), oomph::BinaryTree(object_pt, father_pt, son_type), DynamicTree(object_pt)
    {
      this->Father_pt = father_pt;
      this->Son_type = son_type;
      Level = father_pt->level() + 1;
    }

    // Factory used by oomph-lib's tree-refinement code to create a son tree of the correct dynamic type.
    oomph::Tree *construct_son(oomph::RefineableElement *const &object_pt,
                               Tree *const &father_pt, const int &son_type)
    {
      DynamicBinaryTree *temp_binary_pt = new DynamicBinaryTree(object_pt, father_pt, son_type);
      return temp_binary_pt;
    }
  };

  // Root of a DynamicBinaryTree, i.e. the tree node associated with a top-level (unrefined) 1d element.
  class DynamicBinaryTreeRoot : virtual public DynamicBinaryTree, public virtual DynamicTreeRoot
  {

  public:
    DynamicBinaryTreeRoot(oomph::RefineableElement *const &object_pt) : oomph::Tree(object_pt), oomph::BinaryTree(object_pt), DynamicTree(object_pt), DynamicBinaryTree(object_pt), oomph::TreeRoot(object_pt), DynamicTreeRoot(object_pt)
    {
    }
  };

  // 1d specialization of TemplatedMeshBase: builds/refines meshes of 1d (line) elements using a
  // binary tree forest for h-refinement.
  class TemplatedMeshBase1d : public virtual TemplatedMeshBase
  {

  public:
    TemplatedMeshBase1d() : pyoomph::Mesh(), TemplatedMeshBase()
    {
      oomph::BinaryTree::setup_static_data();
    }
    /// Broken copy constructor
    TemplatedMeshBase1d(const TemplatedMeshBase1d &dummy) : pyoomph::Mesh(), TemplatedMeshBase()
    {
      oomph::BrokenCopy::broken_copy("TemplatedMeshBase1d");
    }

    /// Broken assignment operator
    void operator=(const TemplatedMeshBase1d &)
    {
      oomph::BrokenCopy::broken_assign("TemplatedMeshBase1d");
    }

    /// Destructor:
    virtual ~TemplatedMeshBase1d() {}

    virtual void setup_tree_forest()
    {
      setup_binary_tree_forest();
    }

    // Build a fresh BinaryTreeForest from scratch, turning each current (leaf) element into the root
    // of its own binary tree. Discards any pre-existing forest/refinement-history information.
    void setup_binary_tree_forest()
    {
      if (this->Forest_pt != 0)
        delete this->Forest_pt;
      oomph::Vector<oomph::TreeRoot *> trees_pt;

      const unsigned n_element = this->nelement();

      for (unsigned e = 0; e < n_element; e++)
      {
        BulkElementBase *el_pt = dynamic_cast<BulkElementBase *>(this->element_pt(e));
        trees_pt.push_back(new DynamicBinaryTreeRoot(el_pt));
      }

      this->Forest_pt = new oomph::BinaryTreeForest(trees_pt);
      //     for (auto & t : trees_pt) dynamic_cast<oomph::BinaryTree*>(t)->self_test();
    }

    // Populate this mesh's elements, nodes and boundaries from a MeshTemplateElementCollection: for each
    // template element, instantiate the corresponding JIT-compiled element (factory_element); for each
    // template node with an already-created oomph node, add it to Node_pt and register it on whichever
    // boundaries it belongs to (remapping template boundary indices to compact per-mesh indices via
    // bound_map, since only boundaries actually touched by this collection are kept).
    void generate_from_template(MeshTemplateElementCollection *coll)
    {

      MeshTemplate *templ = coll->get_template();
      templ->flush_oomph_nodes();

      int nb = 0;
      set_nboundary(nb);
      std::vector<int> bound_map(templ->get_boundary_names().size(), -1);

      for (unsigned int e = 0; e < coll->get_elements().size(); e++)
      {
        auto &tel = coll->get_elements()[e];
        this->Element_pt.push_back(templ->factory_element(tel, coll));
      }

      for (unsigned int n = 0; n < templ->get_nodes().size(); n++)
      {
        if (templ->get_nodes()[n]->oomph_node)
          this->Node_pt.push_back(templ->get_nodes()[n]->oomph_node);
        else
          continue;
        oomph::BoundaryNodeBase *bn = dynamic_cast<oomph::BoundaryNodeBase *>(Node_pt.back());
        if (bn) // Add the node to the boundary
        {
          for (unsigned int b : templ->get_nodes()[n]->on_boundaries)
          {
            if (bound_map[b] == -1)
            {
              bound_map[b] = nb;
              nb++;
              this->set_nboundary(nb);
            }
            add_boundary_node(bound_map[b], Node_pt.back());
          }
        }
      }
      this->boundary_names.resize(nb);
      for (unsigned int i = 0; i < templ->get_boundary_names().size(); i++)
      {
        if (bound_map[i] > -1)
        {
          this->boundary_names[bound_map[i]] = templ->get_boundary_names()[i];
        }
      }
      templ->link_periodic_nodes();
      setup_facets_from_template(templ,bound_map);
    }

    void setup_boundary_element_info(std::ostream &outfile);
    void setup_boundary_element_info() override;
	 void fill_internal_facet_buffers(std::vector<BulkElementBase*> & internal_elements, std::vector<int> & internal_face_dir,std::vector<BulkElementBase*> & opposite_elements,std::vector<int> & opposite_face_dir,std::vector<int> & opposite_already_at_index) override;
  };

}
