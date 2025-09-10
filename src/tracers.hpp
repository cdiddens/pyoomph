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
#include <vector>
#include <stack>
#include "mesh.hpp"
namespace pyoomph
{

  class TracerCollection;

  

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
    double timefrac;
    virtual void set_coordinate_dimension(unsigned d);
    virtual void advect(double dt);
    virtual void update_position_from_s();

  public:
    TracerParticle() : pos(), s(), elem(NULL), active(true), in_collection(NULL), collection_index(0), tag(0), timefrac(0) {}
    virtual ~TracerParticle() {}
  };

  class TracerCollection;

  class TracerTransferInterfaceInfo
  {
  public:
    TracerCollection *other_collection;
  };

  class TracerCollection
  {
  protected:
    friend class TracerParticle;
    pyoomph::Mesh *mesh;
    std::string tracer_name;
    pyoomph::MeshKDTree *last_lagrangian_kdtree;
    unsigned nodal_dim, elem_dim;
    unsigned tracer_code_index;
    std::vector<TracerParticle *> tracers;
    std::map<unsigned, TracerTransferInterfaceInfo> transfer_interfaces;
    std::stack<unsigned> free_indices;
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
    virtual void advect_all();
    virtual void prepare_advection();
    virtual void locate_elements();
    virtual void get_new_element(TracerParticle *p);
    virtual void _save_state(std::vector<double> &posarr, std::vector<int> &tagarr);
    virtual void _load_state(std::vector<double> &posarr, std::vector<int> &tagarr);
    virtual void set_transfer_interface(unsigned boundary_index, TracerCollection *opp);
  };

}
