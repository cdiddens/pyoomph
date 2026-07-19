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


// Lagrangian tracer particles that are advected through a (possibly moving/deforming)
// mesh by evaluating an element-provided advection velocity field in local ("s") coordinates.
// A TracerParticle lives inside exactly one TracerCollection at a time and tracks its
// element, local coordinate and global position; TracerCollection owns the particles,
// (re-)locates them in the mesh via a k-d tree, and drives the per-timestep advection.

#pragma once
#include <vector>
#include <stack>
#include "mesh.hpp"
namespace pyoomph
{

  class TracerCollection;



  // A single Lagrangian tracer particle. Position is tracked both as global coordinates
  // (pos) and as local/element coordinates (s) within the current element (elem); the two
  // are kept in sync by update_position_from_s(). Particles are owned/indexed by exactly
  // one TracerCollection (in_collection, collection_index) at a time.
  class TracerParticle
  {
  protected:
    friend class TracerCollection;
    oomph::Vector<double> pos, s;
    BulkElementBase *elem;
    bool active;
    TracerCollection *in_collection;
    unsigned collection_index;
    int tag;
    double timefrac; // Fraction in [0,1] of the current timestep already advected (0=start of step, 1=done)
    virtual void set_coordinate_dimension(unsigned d);
    virtual void advect(double dt); // Advect this particle through the full timestep dt, starting from timefrac
    virtual void update_position_from_s();

  public:
    TracerParticle() : pos(), s(), elem(NULL), active(true), in_collection(NULL), collection_index(0), tag(0), timefrac(0) {}
    virtual ~TracerParticle() {}
  };

  class TracerCollection;

  // Bookkeeping for a tracer transfer interface: when a particle leaves through a mesh
  // boundary that is registered as a transfer interface, it is handed over to the
  // TracerCollection stored here (see TracerCollection::set_transfer_interface).
  class TracerTransferInterfaceInfo
  {
  public:
    TracerCollection *other_collection;
  };

  // Owns a set of TracerParticle objects that live on a single Mesh, and drives their
  // advection. Uses a k-d tree (last_lagrangian_kdtree) to (re-)locate particles' elements
  // whenever the mesh's Lagrangian k-d tree has changed (e.g. after mesh adaptation).
  // Free slots left behind by removed tracers are recycled via free_indices so that
  // tracer indices stay stable for as long as possible (important for save/load state).
  class TracerCollection
  {
  protected:
    friend class TracerParticle;
    pyoomph::Mesh *mesh;
    std::string tracer_name; // Name of the tracer advection field to look up in the generated code's function table
    pyoomph::MeshKDTree *last_lagrangian_kdtree; // k-d tree used the last time elements were located; compared against mesh's current one to detect staleness
    unsigned nodal_dim, elem_dim;
    unsigned tracer_code_index; // Index of this tracer's advection field within the element code's function table
    std::vector<TracerParticle *> tracers; // Indexed storage; NULL entries are free slots (see free_indices)
    std::map<unsigned, TracerTransferInterfaceInfo> transfer_interfaces; // Maps mesh boundary index -> collection to transfer particles into when they cross it
    std::stack<unsigned> free_indices; // Recycled indices into `tracers` left by removed particles
    virtual unsigned get_free_index();
    virtual std::vector<unsigned> get_allocated_indices();
    virtual double get_time(unsigned index = 0);

  public:
    TracerCollection(std::string name) : mesh(NULL), tracer_name(name), last_lagrangian_kdtree(NULL), nodal_dim(0), elem_dim(0), tracer_code_index(0) {}
    virtual unsigned get_tracer_code_index() { return tracer_code_index; }
    virtual void set_mesh(pyoomph::Mesh *m);
    virtual void clear(bool kill_contents);
    unsigned add_tracer(TracerParticle *p);
    TracerParticle *add_tracer(std::vector<double> pos, int tag);
    virtual void remove_tracer(TracerParticle *p);
    virtual TracerParticle *remove_tracer(unsigned index);
    virtual std::vector<double> get_positions();
    virtual std::vector<int> get_tags();
    unsigned get_coordinate_dimension() { return nodal_dim; }
    virtual void advect_all(); // Advect all tracers by the current timestep, re-locating elements first if the mesh has changed
    virtual void prepare_advection(); // Reset all tracers' timefrac to 0 before a new timestep starts
    virtual void locate_elements(); // Rebuild a fresh k-d tree and find the containing element/local coordinate for every tracer
    virtual void get_new_element(TracerParticle *p); // Find the element containing p after it left its previous element, using Lagrangian (undeformed) coordinates
    virtual void _save_state(std::vector<double> &posarr, std::vector<int> &tagarr); // Serialize positions/tags of all currently allocated tracers (e.g. for checkpointing)
    virtual void _load_state(std::vector<double> &posarr, std::vector<int> &tagarr); // Restore tracers from a previously saved (posarr, tagarr) pair, replacing current contents
    virtual void set_transfer_interface(unsigned boundary_index, TracerCollection *opp); // Register that particles crossing mesh boundary `boundary_index` should be handed over to collection `opp`
  };

}
