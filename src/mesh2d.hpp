/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha

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

#include "mesh.hpp"

namespace pyoomph
{

  class DynamicQuadTreeForest : public oomph::QuadTreeForest
  {
  public:
    DynamicQuadTreeForest()
    {
      throw oomph::OomphLibError("Don't call an empty constructor for a QuadTreeForest object", OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
    }

    DynamicQuadTreeForest(oomph::Vector<oomph::TreeRoot *> &trees_pt);

    DynamicQuadTreeForest(const DynamicQuadTreeForest &dummy)
    {
      oomph::BrokenCopy::broken_copy("DynamicQuadTreeForest");
    }

    void operator=(const DynamicQuadTreeForest &)
    {
      oomph::BrokenCopy::broken_assign("DynamicQuadTreeForest");
    }

  protected:
    void find_neighbours() override;
  };

  class DynamicQuadTree : public virtual oomph::QuadTree, public virtual DynamicTree
  {
  protected:
    DynamicQuadTree(oomph::RefineableElement *const &object_pt) : oomph::Tree(object_pt), oomph::QuadTree(object_pt), DynamicTree(object_pt) {}

    DynamicQuadTree(oomph::RefineableElement *const &object_pt, oomph::Tree *const &father_pt, const int &son_type) : oomph::Tree(object_pt), oomph::QuadTree(object_pt, father_pt, son_type), DynamicTree(object_pt)
    {
      this->Father_pt = father_pt;
      this->Son_type = son_type;
      Level = father_pt->level() + 1;
      this->Root_pt = father_pt->root_pt();
    }

    Tree *construct_son(oomph::RefineableElement *const &object_pt,
                        Tree *const &father_pt, const int &son_type)
    {
      DynamicQuadTree *temp_Quad_pt = new DynamicQuadTree(object_pt, father_pt, son_type);
      return temp_Quad_pt;
    }
  };

  class DynamicQuadTreeRoot : virtual public DynamicQuadTree, public virtual DynamicTreeRoot, public virtual oomph::QuadTreeRoot
  {

  public:
    DynamicQuadTreeRoot(oomph::RefineableElement *const &object_pt) : oomph::Tree(object_pt), oomph::QuadTree(object_pt), DynamicTree(object_pt), DynamicQuadTree(object_pt), oomph::TreeRoot(object_pt), DynamicTreeRoot(object_pt), oomph::QuadTreeRoot(object_pt)
    {
    }
  };

  class TemplatedMeshBase2d : public virtual TemplatedMeshBase
  {
  public:
    virtual unsigned add_tri_C1( Node* & n1, Node* & n2, Node* & n3);
    virtual unsigned add_tri_C1TB( Node* & n1, Node* & n2, Node* & n3, Node* & n4);
    
    virtual void setup_interior_boundary_elements(unsigned bindex);
    bool refinement_possible()
    {
      bool allquads = true;
      for (unsigned int i = 0; i < this->nelement(); i++)
      {
        allquads = allquads && (dynamic_cast<oomph::QuadElementBase *>(this->element_pt(i)) != NULL);
      }
      if (allquads)
      {
        return true;
      }
      else
      {
        if (this->max_refinement_level())
        {
          std::cerr << "WARNING: Found a tri or something in the mesh "<< this->domainname << " -> cannot be adaptive right now. Requires to implement a good tree for mixed meshes" << std::endl;
        }
        return false;
      }
    }

    /*
    TemplatedMeshBase2d(MeshTemplate * templ) : pyoomph::Mesh(),TemplatedMeshBase()
    {
      std::cout << "SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS" << std::endl;
      oomph::QuadTree::setup_static_data();
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

    TemplatedMeshBase2d() : pyoomph::Mesh(), TemplatedMeshBase()
    {
      oomph::QuadTree::setup_static_data();
    }

    /// Broken copy constructor
    TemplatedMeshBase2d(const TemplatedMeshBase2d &dummy) : pyoomph::Mesh(), TemplatedMeshBase()
    {
      oomph::BrokenCopy::broken_copy("TemplatedMeshBase2d");
    }

    /// Broken assignment operator
    void operator=(const TemplatedMeshBase2d &)
    {
      oomph::BrokenCopy::broken_assign("TemplatedMeshBase2d");
    }

    /// Destructor:
    virtual ~TemplatedMeshBase2d() {}

    virtual void setup_tree_forest()
    {
      if (refinement_possible())  setup_quadtree_forest();      
      else {
        std::cout << "WARNING: Found a tri or something in the mesh "<< this->domainname << " -> cannot be adaptive right now. Requires to implement a good tree for mixed meshes" << std::endl;
        this->disable_adaptation();
      }
//      std::cout << "TREE FORESET SET UP " << this->Forest_pt << "  " << dynamic_cast<DynamicQuadTreeForest *>(this->Forest_pt) << std::endl;
    }

    void setup_quadtree_forest()
    {
      if (this->Forest_pt != 0)
      {
        oomph::Vector<oomph::Tree *> all_tree_nodes_pt;
        this->Forest_pt->stick_all_tree_nodes_into_vector(all_tree_nodes_pt);
        unsigned local_min_ref = 0;
        unsigned local_max_ref = 0;
        this->get_refinement_levels(local_min_ref, local_max_ref);

#ifdef OOMPH_HAS_MPI
        int int_local_min_ref = local_min_ref;
        if (this->nelement() == 0)
        {
          int_local_min_ref = INT_MAX;
        }
        int int_min_ref = 0;
        MPI_Allreduce(&int_local_min_ref, &int_min_ref, 1,
                      MPI_INT, MPI_MIN,
                      Comm_pt->mpi_comm());

        unsigned min_ref = unsigned(int_min_ref);
#else
        unsigned min_ref = local_min_ref;
#endif
        if (this->nelement() == 0)
        {
          this->Forest_pt->flush_trees();
          delete this->Forest_pt;
          oomph::Vector<oomph::TreeRoot *> trees_pt;
          this->Forest_pt = new pyoomph::DynamicQuadTreeForest(trees_pt);
          return;
        }

        oomph::Vector<oomph::TreeRoot *> trees_pt;
        unsigned n_tree_nodes = all_tree_nodes_pt.size();
        for (unsigned e = 0; e < n_tree_nodes; e++)
        {
          oomph::Tree *tree_pt = all_tree_nodes_pt[e];
          if (tree_pt->object_pt() != 0)
          {
            oomph::RefineableElement *el_pt = dynamic_cast<oomph::RefineableElement *>(tree_pt->object_pt());
            unsigned level = el_pt->refinement_level();
            if (level < min_ref)
            {
              tree_pt->flush_sons();
              delete tree_pt;
              delete el_pt;
            }
            else if (level == min_ref)
            {
              unsigned n_sons = tree_pt->nsons();
              oomph::Vector<oomph::Tree *> backed_up_sons(n_sons);
              for (unsigned i_son = 0; i_son < n_sons; i_son++)
              {
                backed_up_sons[i_son] = tree_pt->son_pt(i_son);
              }
              DynamicQuadTreeRoot *tree_root_pt = new DynamicQuadTreeRoot(el_pt);

              // Pass sons
              tree_root_pt->set_son_pt(backed_up_sons);

              // Loop over sons and make the new treeroot their father
              for (unsigned i_son = 0; i_son < n_sons; i_son++)
              {
                oomph::Tree *son_pt = backed_up_sons[i_son];
                son_pt->set_father_pt(tree_root_pt);
                son_pt->root_pt() = tree_root_pt;
                oomph::Vector<oomph::Tree *> all_sons_pt;
                son_pt->stick_all_tree_nodes_into_vector(all_sons_pt);
                unsigned n = all_sons_pt.size();
                for (unsigned i = 0; i < n; i++)
                {
                  all_sons_pt[i]->root_pt() = tree_root_pt;
                }
              }
              trees_pt.push_back(tree_root_pt);
              tree_pt->flush_sons();
              delete tree_pt;
            }
          }
          else
          {
            tree_pt->flush_sons();
            delete tree_pt;
          }
        }
        this->Forest_pt->flush_trees();
        delete this->Forest_pt;
        this->Forest_pt = new pyoomph::DynamicQuadTreeForest(trees_pt);
      }

      else // Create a new Forest from scratch in the "usual" uniform way
      {
        oomph::Vector<oomph::TreeRoot *> trees_pt;
        unsigned n_element = this->nelement();
        for (unsigned e = 0; e < n_element; e++)
        {
          // Get pointer to full element type
          BulkElementBase *el_pt = dynamic_cast<BulkElementBase *>(this->element_pt(e));
          trees_pt.push_back(new DynamicQuadTreeRoot(el_pt));
          //				std::cout << "TREE CREATED " << trees_pt.back() << "  " << trees_pt.back()->root_pt() << std::endl;
        }
        this->Forest_pt = new pyoomph::DynamicQuadTreeForest(trees_pt);
      }
    }

    void generate_from_template(MeshTemplateElementCollection *coll)
    {
      //      std::cout << "GEN FROM TEMPLATE " << std::endl;
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
    }

    void setup_boundary_element_info_quads(std::ostream &outfile);
    void setup_boundary_element_info_tris(std::ostream &outfile);
    void setup_boundary_element_info(std::ostream &outfile);
    void setup_boundary_element_info() override;
	 void fill_internal_facet_buffers(std::vector<BulkElementBase*> & internal_elements, std::vector<int> & internal_face_dir,std::vector<BulkElementBase*> & opposite_elements,std::vector<int> & opposite_face_dir,std::vector<int> & opposite_already_at_index) override;        
  };

}
