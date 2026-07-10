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

  class DynamicOcTree : public virtual oomph::OcTree, public virtual DynamicTree
  {
  protected:
    DynamicOcTree(oomph::RefineableElement *const &object_pt) : oomph::Tree(object_pt), oomph::OcTree(object_pt), DynamicTree(object_pt) {}

    DynamicOcTree(oomph::RefineableElement *const &object_pt, oomph::Tree *const &father_pt, const int &son_type) : oomph::Tree(object_pt), oomph::OcTree(object_pt, father_pt, son_type), DynamicTree(object_pt)
    {
      this->Father_pt = father_pt;
      this->Son_type = son_type;
      Level = father_pt->level() + 1;
      this->Root_pt = father_pt->root_pt();
    }

    Tree *construct_son(oomph::RefineableElement *const &object_pt,
                        Tree *const &father_pt, const int &son_type)
    {
      DynamicOcTree *temp_Oc_pt = new DynamicOcTree(object_pt, father_pt, son_type);
      return temp_Oc_pt;
    }
  };

  class DynamicOcTreeRoot : virtual public DynamicOcTree, public virtual DynamicTreeRoot, public virtual oomph::OcTreeRoot
  {

  public:
    DynamicOcTreeRoot(oomph::RefineableElement *const &object_pt) : oomph::Tree(object_pt), oomph::OcTree(object_pt), DynamicTree(object_pt), DynamicOcTree(object_pt), oomph::TreeRoot(object_pt), DynamicTreeRoot(object_pt), oomph::OcTreeRoot(object_pt)
    {
    }
  };

  class TemplatedMeshBase3d : public virtual TemplatedMeshBase
  {
  private:
    bool issued_tri_refinement_warning = false;
  public:
    bool refinement_possible();
    /*
    TemplatedMeshBase3d(MeshTemplate * templ) : pyoomph::Mesh(),TemplatedMeshBase()
    {
      oomph::OcTree::setup_static_data();
      generate_from_template(templ);
      if (refinement_possible())
      {
        setup_tree_forest();
      }
      else
      {
        this->disable_adaptation();
      }
       std::ofstream outfile;
      setup_boundary_element_info(outfile);
    }
    */

    TemplatedMeshBase3d() : pyoomph::Mesh(), TemplatedMeshBase()
    {
      oomph::OcTree::setup_static_data();
    }

    /// Broken copy constructor
    TemplatedMeshBase3d(const TemplatedMeshBase3d &dummy) : pyoomph::Mesh(), TemplatedMeshBase()
    {
      oomph::BrokenCopy::broken_copy("TemplatedMeshBase3d");
    }

    /// Broken assignment operator
    void operator=(const TemplatedMeshBase3d &)
    {
      oomph::BrokenCopy::broken_assign("TemplatedMeshBase3d");
    }

    /// Destructor:
    virtual ~TemplatedMeshBase3d() {}

    virtual void setup_tree_forest()
    {
      setup_octree_forest();
    }

    void setup_octree_forest()
    {
      if (this->Forest_pt != 0)
      {
        // Get all the tree nodes
        oomph::Vector<oomph::Tree *> all_tree_nodes_pt;
        this->Forest_pt->stick_all_tree_nodes_into_vector(all_tree_nodes_pt);
        unsigned local_min_ref = 0;
        unsigned local_max_ref = 0;
        this->get_refinement_levels(local_min_ref, local_max_ref);
        unsigned min_ref = local_min_ref;
#ifdef OOMPH_HAS_MPI
        if (Comm_pt != 0)
        {
          int int_local_min_ref = local_min_ref;
          if (this->nelement() == 0)
          {
            int_local_min_ref = INT_MAX;
          }
          int int_min_ref = 0;
          MPI_Allreduce(&int_local_min_ref, &int_min_ref, 1,
                        MPI_INT, MPI_MIN,
                        Comm_pt->mpi_comm());
          min_ref = int_min_ref;
        }
#endif

        if (this->nelement() == 0)
        {
          // Flush the Forest's current trees
          this->Forest_pt->flush_trees();

          delete this->Forest_pt;

          // Empty dummy vector to build empty forest
          oomph::Vector<oomph::TreeRoot *> trees_pt;

          // Make a new (empty) Forest
          this->Forest_pt = new oomph::OcTreeForest(trees_pt);

          return;
        }

        // Vector to store trees for new Forest
        oomph::Vector<oomph::TreeRoot *> trees_pt;

        // Loop over tree nodes (e.g. elements)
        unsigned n_tree_nodes = all_tree_nodes_pt.size();
        for (unsigned e = 0; e < n_tree_nodes; e++)
        {
          oomph::Tree *tree_pt = all_tree_nodes_pt[e];

          // If the object_pt has been flushed then we don't want to keep
          // this tree
          if (tree_pt->object_pt() != 0)
          {
            // Get the refinement level of the current tree node
            oomph::RefineableElement *el_pt = dynamic_cast<oomph::RefineableElement *>(tree_pt->object_pt());
            unsigned level = el_pt->refinement_level();

            // If we are below the minimum refinement level, remove tree
            if (level < min_ref)
            {
              // Flush sons for this tree
              tree_pt->flush_sons();

              // Delete the tree (no recursion)
              delete tree_pt;

              // Delete the element
              delete el_pt;
            }
            else if (level == min_ref)
            {
              // Get the sons (if there are any) and store them
              unsigned n_sons = tree_pt->nsons();
              oomph::Vector<oomph::Tree *> backed_up_sons(n_sons);
              for (unsigned i_son = 0; i_son < n_sons; i_son++)
              {
                backed_up_sons[i_son] = tree_pt->son_pt(i_son);
              }

              // Make the element into a new treeroot
              DynamicOcTreeRoot *tree_root_pt = new DynamicOcTreeRoot(el_pt);

              // Pass sons
              tree_root_pt->set_son_pt(backed_up_sons);

              // Loop over sons and make the new treeroot their father
              for (unsigned i_son = 0; i_son < n_sons; i_son++)
              {
                oomph::Tree *son_pt = backed_up_sons[i_son];

                // Tell the son about its new father (which is also the root)
                son_pt->set_father_pt(tree_root_pt);
                son_pt->root_pt() = tree_root_pt;

                // ...and then tell all the descendants too
                oomph::Vector<oomph::Tree *> all_sons_pt;
                son_pt->stick_all_tree_nodes_into_vector(all_sons_pt);
                unsigned n = all_sons_pt.size();
                for (unsigned i = 0; i < n; i++)
                {
                  all_sons_pt[i]->root_pt() = tree_root_pt;
                }
              }

              // Add tree root to the trees_pt vector
              trees_pt.push_back(tree_root_pt);

              // Now kill the original (non-root) tree: First
              // flush sons for this tree
              tree_pt->flush_sons();

              // ...then delete the tree (no recursion)
              delete tree_pt;
            }
          }
          else // tree_pt->object_pt() is null, so delete tree
          {
            // Flush sons for this tree
            tree_pt->flush_sons();

            // Delete the tree (no recursion)
            delete tree_pt;
          }
        }

        // Flush the Forest's current trees
        this->Forest_pt->flush_trees();

        // Delete the old Forest
        delete this->Forest_pt;

        // Make a new Forest with the trees_pt roots created earlier
        this->Forest_pt = new oomph::OcTreeForest(trees_pt);
      }
      else // Create a new Forest from scratch in the "usual" uniform way
      {
        // Turn elements into individual octrees and plant in forest
        oomph::Vector<oomph::TreeRoot *> trees_pt;
        unsigned nel = nelement();
        for (unsigned iel = 0; iel < nel; iel++)
        {
          // Get pointer to full element type
          BulkElementBase *el_pt = dynamic_cast<BulkElementBase *>(element_pt(iel));

          // Build associated octree(root) -- pass pointer to corresponding
          // finite element and add the pointer to vector of octree (roots):
          DynamicOcTreeRoot *octree_root_pt = new DynamicOcTreeRoot(el_pt);
          trees_pt.push_back(octree_root_pt);
        }
        // Plant OcTreeRoots in OcTreeForest
        this->Forest_pt = new oomph::OcTreeForest(trees_pt);
      }
    }

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

    void setup_boundary_element_info_bricks(std::ostream &outfile);
    void setup_boundary_element_info_tris(std::ostream &outfile);
    void setup_boundary_element_info(std::ostream &outfile);
    void setup_boundary_element_info() override;
  };

}
