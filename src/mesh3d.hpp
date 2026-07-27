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

  // DynamicTree specialization for 3d brick-refined (OcTree) elements.
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

    // Factory used by oomph-lib's tree-refinement code to create a son tree of the correct dynamic type.
    Tree *construct_son(oomph::RefineableElement *const &object_pt,
                        Tree *const &father_pt, const int &son_type) override
    {
      DynamicOcTree *temp_Oc_pt = new DynamicOcTree(object_pt, father_pt, son_type);
      return temp_Oc_pt;
    }
  };

  // Root of a DynamicOcTree, i.e. the tree node associated with a top-level (unrefined) brick element.
  class DynamicOcTreeRoot : virtual public DynamicOcTree, public virtual DynamicTreeRoot, public virtual oomph::OcTreeRoot
  {

  public:
    DynamicOcTreeRoot(oomph::RefineableElement *const &object_pt) : oomph::Tree(object_pt), oomph::OcTree(object_pt), DynamicTree(object_pt), DynamicOcTree(object_pt), oomph::TreeRoot(object_pt), DynamicTreeRoot(object_pt), oomph::OcTreeRoot(object_pt)
    {
    }
  };

  // OcTreeForest specialization that (a) skips oomph's brick compass neighbour finding for
  // tetrahedral forests -- tets use geometric node-sharing/hanging instead -- while delegating to
  // it for brick forests, and (b) skips the quad-coordinate neighbour self-test for tets.
  class DynamicOcTreeForest : public oomph::OcTreeForest
  {
  public:
    DynamicOcTreeForest(oomph::Vector<oomph::TreeRoot *> &trees_pt) : oomph::OcTreeForest(trees_pt, true)
    {
      if (trees_pt.size() == 0) return;
      find_neighbours();
      // Brick forests need oomph's rotation/orientation equivalents for the compass coordinate descent;
      // tet forests bypass that descent entirely (shared-node face correspondence + the affine map), so
      // skip it -- and it assumes brick geometry.
      if (!(ntree() > 0 && dynamic_cast<oomph::TElementBase *>(Trees_pt[0]->object_pt())))
        construct_up_right_equivalents();
    }

    DynamicOcTreeForest(const DynamicOcTreeForest &) : oomph::OcTreeForest()
    {
      oomph::BrokenCopy::broken_copy("DynamicOcTreeForest");
    }

    void operator=(const DynamicOcTreeForest &)
    {
      oomph::BrokenCopy::broken_assign("DynamicOcTreeForest");
    }

    void check_all_neighbours(oomph::DocInfo &doc_info) override
    {
      // Tets and wedges bypass oomph's brick (hex) compass neighbour self-test: they do not use the OcTree's
      // compass neighbour structure (tets have their own topological finder; wedges skip it for now), and the
      // brick test would misread their 4/6-node elements as an 8-node hex.
      if (ntree() > 0 && dynamic_cast<oomph::TElementBase *>(Trees_pt[0]->object_pt())) return;
      if (ntree() > 0 && dynamic_cast<oomph::RefineableWedgeElement *>(Trees_pt[0]->object_pt())) return;
      if (ntree() > 0 && dynamic_cast<oomph::RefineablePyramidElement *>(Trees_pt[0]->object_pt())) return;
      oomph::OcTreeForest::check_all_neighbours(doc_info);
    }

  protected:
    void find_neighbours() override
    {
      // A MIXED forest (its roots are not all the same element family) shares interface nodes via the
      // weight-augmented registry and installs 2:1 hanging via the mesh-level post_adapt pass, so NO octree
      // neighbour pointers are needed -- and oomph's brick compass finder must not run (it would misread a
      // 5-node pyramid / 6-node wedge as an 8-node hex) nor the tet finder (it throws on a non-tet neighbour).
      // Skipping here also makes the pure registry mixes (tet+wedge etc.) robust to root ordering.
      if (ntree() > 1)
      {
        oomph::FiniteElement *o0 = Trees_pt[0]->object_pt();
        const bool b0 = dynamic_cast<oomph::BrickElementBase *>(o0) != nullptr;
        const bool t0 = dynamic_cast<oomph::TElementBase *>(o0) != nullptr;
        const bool w0 = dynamic_cast<oomph::RefineableWedgeElement *>(o0) != nullptr;
        const bool p0 = dynamic_cast<oomph::RefineablePyramidElement *>(o0) != nullptr;
        for (unsigned i = 1; i < ntree(); i++)
        {
          oomph::FiniteElement *oi = Trees_pt[i]->object_pt();
          if ((dynamic_cast<oomph::BrickElementBase *>(oi) != nullptr) != b0 ||
              (dynamic_cast<oomph::TElementBase *>(oi) != nullptr) != t0 ||
              (dynamic_cast<oomph::RefineableWedgeElement *>(oi) != nullptr) != w0 ||
              (dynamic_cast<oomph::RefineablePyramidElement *>(oi) != nullptr) != p0)
            return; // mixed forest -> registry node-sharing + mesh-level hanging
        }
      }
      // Tetrahedral forests: topological FACE neighbour finding, the 3d analogue of
      // DynamicQuadTreeForest::find_neighbours (mesh2d.cpp). Two tet roots are neighbours across a face
      // iff they share that face's three corner (vertex) nodes; the tet's four faces reuse four of the
      // six OcTree face slots (L,R,D,U; face f is opposite vertex f). The tet neighbour/hang path bypasses
      // oomph's compass coordinate descent (shared-node correspondence + the exact affine map), so only
      // this topological adjacency is needed -- exactly as for triangles.
      if (ntree() > 0 && dynamic_cast<oomph::TElementBase *>(Trees_pt[0]->object_pt()))
      {
        using namespace oomph::OcTreeNames;
        static const int FACE_SLOT[4] = {L, R, D, U};          // tet face f -> OcTree face slot
        static const int FACE_CORNER[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
        const unsigned numtrees = ntree();
        // Trees sharing a common corner node are potential face neighbours.
        std::map<oomph::Node *, std::set<unsigned>> tree_at_vertex;
        for (unsigned i = 0; i < numtrees; i++)
        {
          oomph::TElementBase *ti = dynamic_cast<oomph::TElementBase *>(Trees_pt[i]->object_pt());
          if (!ti) throw_runtime_error("Strange element in tet tree forest");
          for (unsigned v = 0; v < 4; v++) tree_at_vertex[ti->vertex_node_pt(v)].insert(i);
        }
        oomph::Vector<std::set<unsigned>> potential(numtrees);
        for (auto &kv : tree_at_vertex)
          for (unsigned a : kv.second)
            for (unsigned b : kv.second)
              if (a != b) potential[a].insert(b);
        for (unsigned i = 0; i < numtrees; i++)
        {
          oomph::TElementBase *ti = dynamic_cast<oomph::TElementBase *>(Trees_pt[i]->object_pt());
          for (unsigned j : potential[i])
          {
            oomph::FiniteElement *obj_j = Trees_pt[j]->object_pt();
            if (!dynamic_cast<oomph::TElementBase *>(obj_j)) throw_runtime_error("Mixed tet/brick neighbours not yet supported");
            for (int f = 0; f < 4; f++)
            {
              bool all = true;
              for (int c = 0; c < 3 && all; c++)
                if (obj_j->get_node_number(ti->vertex_node_pt(FACE_CORNER[f][c])) == -1) all = false;
              if (all) Trees_pt[i]->neighbour_pt(FACE_SLOT[f]) = Trees_pt[j];
            }
          }
        }
        return;
      }
      // Wedge forests: skip neighbour finding for now. Uniform (1->8) wedge refinement stays conforming and
      // shares new father-boundary nodes via the father-node-keyed registry (RefineableWedgeElement::build),
      // so no inter-tree neighbour pointers are needed. (Non-uniform 2:1 wedge hanging -- a later milestone --
      // will need a topological wedge face/edge neighbour finder, as tets have above. oomph's brick compass
      // find_neighbours below must NOT run on wedges: it would misread the 6-node wedge as an 8-node hex.)
      if (ntree() > 0 && dynamic_cast<oomph::RefineableWedgeElement *>(Trees_pt[0]->object_pt()))
      {
        return;
      }
      // Pyramid forests: skip neighbour finding for now (like wedges). Uniform pyramid refinement stays
      // conforming (the mixed 6-pyr+4-tet offspring share their father-boundary nodes via the cross-shape
      // registry in RefineablePyramidElement::build_son_from_pyramid_father), so no inter-tree neighbour
      // pointers are needed. oomph's brick compass find_neighbours must NOT run: it would misread the 5-node
      // pyramid as an 8-node hex. (Non-uniform pyramid hanging -- a later milestone -- needs a topological
      // cross-shape face/edge neighbour finder.)
      if (ntree() > 0 && dynamic_cast<oomph::RefineablePyramidElement *>(Trees_pt[0]->object_pt()))
      {
        return;
      }
      oomph::OcTreeForest::find_neighbours(); // brick forests keep oomph's compass neighbouring
    }
  };

  // 3d specialization of TemplatedMeshBase: builds/refines meshes of brick (hex) or tetrahedral
  // elements. Bricks use an octree forest for h-refinement; tets are not tree-refineable (see
  // refinement_possible()).
  class TemplatedMeshBase3d : public virtual TemplatedMeshBase
  {
  private:
    bool issued_tri_refinement_warning = false; // Guards against spamming the "cannot refine tets" warning repeatedly
  public:
    // Whether this mesh's element type actually supports tree-based h-refinement (true for bricks,
    // false for plain tetrahedra).
    bool refinement_possible() override; 
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
    TemplatedMeshBase3d(const TemplatedMeshBase3d &) : oomph::Mesh(), pyoomph::Mesh(), TemplatedMeshBase()
    {
      oomph::BrokenCopy::broken_copy("TemplatedMeshBase3d");
    }

    /// Broken assignment operator
    void operator=(const TemplatedMeshBase3d &)
    {
      oomph::BrokenCopy::broken_assign("TemplatedMeshBase3d");
    }

    /// Destructor:
    ~TemplatedMeshBase3d() override {}

    void setup_tree_forest() override
    {
      setup_octree_forest();
    }

    // After (non-uniform) refinement, install hanging nodes for tetrahedral meshes. A hanging node
    // lies in the interior of a coarser neighbour's edge; it is constrained by that edge's
    // interpolation (linear for C1). No-op for pure-brick meshes (oomph-lib handles those). See .cpp.
    void post_adapt_setup_hanging_nodes() override;

    // Enforce 2:1 refinement balancing for tetrahedral meshes: iteratively refine any leaf tet that
    // is >1 refinement level coarser than a face/edge neighbour (detected by a node existing at the
    // quarter point of one of its edges), until the mesh is balanced. This guarantees all hanging is
    // single-level (edge/face on real coarse corners, no hanging chains), making arbitrary refinement
    // patterns -- and C2 tet face-interior hanging -- machine-zero. No-op for non-tet meshes. See .cpp.
    void enforce_refinement_balance() override;

    // (Re)build the OcTreeForest. If a forest already exists, this "flattens" it down to the coarsest
    // common refinement level present (min_ref, reduced across MPI ranks if applicable) by promoting
    // each tree node at that level to a new tree root and discarding levels below it; if no forest
    // exists yet, one is created from scratch with one tree root per current element.
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
          this->Forest_pt = new pyoomph::DynamicOcTreeForest(trees_pt);

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
              oomph::Vector<oomph::Tree *> backed_up_sons;
              backed_up_sons.reserve(n_sons);
              for (unsigned i_son = 0; i_son < n_sons; i_son++)
              {
                backed_up_sons.push_back(tree_pt->son_pt(i_son));
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
        this->Forest_pt = new pyoomph::DynamicOcTreeForest(trees_pt);
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
        this->Forest_pt = new pyoomph::DynamicOcTreeForest(trees_pt);
      }
    }

    // Populate this mesh's elements, nodes and boundaries from a MeshTemplateElementCollection; see
    // TemplatedMeshBase1d::generate_from_template for the (identical) algorithm description.
    void generate_from_template(MeshTemplateElementCollection *coll) override
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
      // Turn the template's facet records into per-element face boundary tags right away; from here
      // on the tags are carried (and inherited on refinement) by the elements themselves.
      seed_face_boundaries_from_facets();
    }

    void setup_boundary_element_info_bricks(std::ostream &outfile);
    void setup_boundary_element_info_tris(std::ostream &outfile);
    void setup_boundary_element_info(std::ostream &outfile) override;
    void setup_boundary_element_info() override;
  };

}
