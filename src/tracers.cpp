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


#include "tracers.hpp"
#include "exception.hpp"
#include "elements.hpp"
#include <iostream>

namespace pyoomph
{

  void TracerParticle::set_coordinate_dimension(unsigned d)
  {
    pos.resize(d, 0.0);
  }

  void TracerParticle::advect(double dt)
  {
    if (this->timefrac > 1 - 1e-15)
      return;
    if (!elem)
    {
      // throw_runtime_error("Not without any element");
      return;
    }

    // double dt_full=dt*(1-this->timefrac);
    // double dt_step=dt_full;

    double desired_smag_scale = 0.1; //~10 steps per element

    oomph::Vector<double> svelo(s.size());
    unsigned tracer_index = in_collection->get_tracer_code_index();
    double over_leave_factor = 0.001;

    // On the range [0:1] (where 1 means the desired dt)
    double min_dt = 1e-5;
    double max_dt = 5e-2;


    double s_plane_distance;
    oomph::Vector<double> s_normal(s.size());

    while (this->timefrac < 1 - 1e-16)
    {

      double this_dt = 1 - this->timefrac;
      // if (collection_index==0) std::cout << "ADV " << this->timefrac << "  " <<this_dt << std::endl;
      if (this_dt > max_dt)
        this_dt = max_dt;
      elem->eval_tracer_advection_in_s_space(tracer_index, this->timefrac, s, svelo);
      double svmag = 0.0;
      for (unsigned int i = 0; i < svelo.size(); i++)
        svmag += svelo[i] * svelo[i];
      svmag = sqrt(svmag) * this_dt * dt;
      // if (collection_index==0) std::cout << "   SVMAG " << svmag << "  " << desired_smag_scale   << std::endl;
      if (svmag > desired_smag_scale)
      {
        this_dt *= desired_smag_scale / (svmag * dt);
      }
      double factor_to_leave = elem->factor_when_local_coordinate_becomes_invalid(s, svelo, s_normal, s_plane_distance);
      // Check if we would leave
      // if (collection_index==0) std::cout << "   FACT T LEAVE  " << factor_to_leave<< "  " <<  this_dt << "   " << dt   << std::endl;
      // if (collection_index==0) std::cout << "   SVELO " << svelo[0]<< "  " <<  svelo[1]   << std::endl;
      if (factor_to_leave < this_dt)
      {
        if (factor_to_leave < over_leave_factor)
          factor_to_leave = over_leave_factor;
        // Perform only until then
        this_dt = factor_to_leave * (1 + over_leave_factor);
      }
      if (this_dt < min_dt)
      {
        this_dt = min_dt;
      }

      if (this->timefrac + this_dt > 1)
        this_dt = 1 - this->timefrac;

      oomph::Vector<double> sold = s;
      if (false)
      {
        for (unsigned int i = 0; i < s.size(); i++)
        {
          s[i] += svelo[i] * this_dt * dt;
        }
      }
      else
      {
        // Make an RK2 step
        oomph::Vector<double> s1(s.size());
        oomph::Vector<double> svelo1(s.size());
        oomph::Vector<double> s2(s.size());
        oomph::Vector<double> svelo2(s.size());
        for (unsigned int i = 0; i < s.size(); i++)
        {
          s1[i] = s[i] + 0.5 * this_dt * dt * svelo[i];
        }
        elem->eval_tracer_advection_in_s_space(tracer_index, this->timefrac + this_dt * 0.5, s1, svelo1);
        for (unsigned int i = 0; i < s.size(); i++)
        {
          s2[i] = s[i] + this_dt * dt * (2 * svelo1[i] - svelo[i]);
        }
        elem->eval_tracer_advection_in_s_space(tracer_index, this->timefrac + this_dt, s2, svelo2);
        for (unsigned int i = 0; i < s.size(); i++)
        {
          s[i] += this_dt * dt * (svelo[i] / 6 + svelo1[i] * 4 / 6 + svelo2[i] / 6);
        }
      }

      if (!elem->local_coord_is_valid(s))
      {
        // Check whether we left a mesh boundary here
        std::set<unsigned> allbounds;
        for (unsigned int in = 0; in < this->elem->nnode(); in++)
        {
          std::set<unsigned> *nodebounds;
          elem->node_pt(in)->get_boundaries_pt(nodebounds);
          if (nodebounds)
          {
            for (auto b : (*nodebounds))
              allbounds.insert(b);
          }
        }
        // Now go over all nodes and find the boundary where all are located on if they lie on the passing s plane
        for (unsigned int in = 0; in < this->elem->nnode(); in++)
        {
          oomph::Vector<double> snode(s.size());
          elem->local_coordinate_of_node(in, snode);
          double dist = -s_plane_distance;
          for (unsigned int is = 0; is < s.size(); is++)
            dist += s_normal[is] * snode[is];
          if (abs(dist) < 1e-7)
          {
            std::set<unsigned> *nodebounds;
            elem->node_pt(in)->get_boundaries_pt(nodebounds);
            if (!nodebounds)
            {
              allbounds.clear();
              break;
            }
            else
            {
              std::set<unsigned> nbounds, intersect;
              nbounds = *nodebounds;
              std::set_intersection(nbounds.begin(), nbounds.end(), allbounds.begin(), allbounds.end(), std::inserter(intersect, intersect.begin()));
              allbounds.swap(intersect);
              if (allbounds.empty())
                break;
            }
          }
        }

        if (!allbounds.empty())
        {
          for (auto bind : allbounds)
          {
            if (this->in_collection->transfer_interfaces.count(bind))
            {
              this->timefrac += this_dt;
              auto *newcoll = this->in_collection->transfer_interfaces[bind].other_collection;

              if (newcoll->mesh->get_lagrangian_kdtree() != newcoll->last_lagrangian_kdtree)
              {
                newcoll->locate_elements();
              }

              this->in_collection->remove_tracer(this);
              newcoll->add_tracer(this);
              newcoll->get_new_element(this);

              this->advect(dt);
              if (this->elem)
                this->update_position_from_s();

              return;
            }
          }

          // Mode: Stay inside, tangential motion at interface:
          //   find projected s
          oomph::Vector<double> sstep(s.size());
          // double sdot = 0.0;
          oomph::Vector<double> sproj(s.size());
          for (unsigned int is = 0; is < s.size(); is++)
            sstep[is] = s[is] - sold[is];

          /*  //std::cout << "PARTICLE " << this<< " IS LEAVING THROUGH BOUNDARY " ; for (auto b : allbounds) std::cout << b << "  " ; std::cout << " WITH SNORMAL "<< s_normal[0] << "  " << s_normal[1] << "  AND  SSTEP " << sstep[0] << "  " << sstep[1];  std::cout << std::endl;

            for (unsigned int is=0;is<s.size();is++) sdot+=sstep[is]*s_normal[is];

            for (unsigned int is=0;is<s.size();is++)
            {
             s[is]=sold[is]+(sstep[is]-s_normal[is]*sdot);
            }
             */

          /*
          double dn=-s_plane_distance;
          for (unsigned int is=0;is<s.size();is++)
          {
           dn+=s_normal[is]*sold[is];
          }
          for (unsigned int is=0;is<s.size();is++)
          {
           s[is]=sold[is]-s_normal[is]*dn;
          }
          if (elem->local_coord_is_valid(s))
          {
           this->timefrac+=this_dt;
           continue;
          }
          else
          {
           in_collection->get_new_element(this);
           if (!this->elem) {
            this->timefrac+=this_dt;
            std::cout << "   BUT LEFT " << std::endl;
            return;
           }
          }*/
        }

        in_collection->get_new_element(this);
        if (!this->elem)
        {
          this->timefrac += this_dt;
          // TODO: See if it can be transferred or projected at a boundary
          return;
        }
      }

      this->timefrac += this_dt;
    }
  }

  void TracerParticle::update_position_from_s()
  {
    if (elem)
    {
      elem->interpolated_x(s, pos);
    }
  }

  void TracerCollection::get_new_element(TracerParticle *p)
  {
    oomph::Vector<double> xi;
    xi.resize(p->s.size());
    BulkElementBase *oldelem = p->elem;
    for (unsigned int i = 0; i < p->s.size(); i++)
    {
      xi[i] = oldelem->interpolated_xi(p->s, i);
    }
    // oldelem->interpolated_xi(p->s,xi); // WHY EVER...

    /*    p->update_position_from_s();
        std::cout << "TRACER PARTICLE AT " << p->pos[0] << "  " << p->pos[1] << std::endl;
        std::cout << "  XI " << xi[0] << "  " << xi[1] << std::endl;
        std::cout << "  S " << p->s[0] << "  " << p->s[1] << std::endl;
        std::cout << "  NODE CORNERS" << std::endl;
        for (unsigned ni=0;ni<oldelem->nnode();ni++)
        {
            std::cout << "  " <<ni << "  " << dynamic_cast<Node*>(oldelem->node_pt(ni))->xi(0) << "  " << dynamic_cast<Node*>(oldelem->node_pt(ni))->xi(1)  << std::endl;
        }
      */

    p->elem = last_lagrangian_kdtree->find_element(xi, p->s);
  }

  unsigned TracerCollection::get_free_index()
  {
    if (free_indices.empty())
    {
      unsigned ret = tracers.size();
      tracers.push_back(NULL);
      return ret;
    }
    else
    {
      unsigned ret = free_indices.top();
      free_indices.pop();
      return ret;
    }
  }

  void TracerCollection::clear(bool kill_contents)
  {
    if (kill_contents)
    {
      for (unsigned int i = 0; i < tracers.size(); i++)
      {
        if (tracers[i])
        {
          // TODO: Throw if in other collection
          if (tracers[i]->in_collection == this)
            delete tracers[i];
          else
          {
            throw_runtime_error("Tracer in wrong collection...");
          }
        }
      }
    }
    tracers.clear();
    free_indices = {};
  }

  unsigned TracerCollection::add_tracer(TracerParticle *p)
  {
    unsigned ret = this->get_free_index();
    tracers[ret] = p;
    if (p->in_collection)
    {
      if (p->in_collection == this)
      {
        throw_runtime_error("Tracer already added to this collection");
      }
      else
      {
        throw_runtime_error("Tracer already part of another collection");
      }
    }
    p->in_collection = this;
    p->collection_index = ret;
    p->set_coordinate_dimension(this->get_coordinate_dimension());
    return ret;
  }

  void TracerCollection::remove_tracer(TracerParticle *p)
  {
    if (!p->in_collection)
    {
      throw_runtime_error("Cannot remove tracer which is in no collection");
    }
    else if (p->in_collection != this)
    {
      throw_runtime_error("Cannot remove tracer since it is in another collection");
    }
    this->remove_tracer(p->collection_index);
  }

  TracerParticle *TracerCollection::remove_tracer(unsigned index)
  {
    if (index >= tracers.size())
    {
      throw_runtime_error("Tracer index out of bounds");
    }
    if (!tracers[index])
      return NULL;
    TracerParticle *ret = tracers[index];
    free_indices.push(index);
    tracers[index] = NULL;
    if (ret->in_collection == this)
    {
      ret->in_collection = NULL;
    }
    return ret;
  }

  std::vector<unsigned> TracerCollection::get_allocated_indices()
  {
    std::vector<unsigned> res;
    res.reserve(tracers.size());
    for (unsigned int i = 0; i < tracers.size(); i++)
      if (tracers[i])
        res.push_back(i);
    return res;
  }

  std::vector<double> TracerCollection::get_positions()
  {
    unsigned cd = this->get_coordinate_dimension();
    std::vector<unsigned> inds = get_allocated_indices();
    std::vector<double> ret;
    ret.reserve(inds.size() * cd);
    for (const auto &i : inds)
    {
      if (tracers[i]->elem)
        tracers[i]->update_position_from_s();
      for (unsigned int j = 0; j < cd; j++)
        ret.push_back(tracers[i]->pos[j]);
    }
    return ret;
  }

  std::vector<int> TracerCollection::get_tags()
  {
    std::vector<unsigned> inds = get_allocated_indices();
    std::vector<int> ret;
    ret.reserve(inds.size());
    for (const auto &i : inds)
      ret.push_back(tracers[i]->tag);
    return ret;
  }

  void TracerCollection::set_mesh(pyoomph::Mesh *m)
  {
    mesh = m;
    nodal_dim = m->get_nodal_dimension();
    int ed = m->get_element_dimension();
    if (ed < 0)
      throw_runtime_error("Cannot set tracer mesh with a negative element dimension");
    elem_dim = ed;
    if (nodal_dim != elem_dim)
      throw_runtime_error("Tracers can only be added to domains with zero co-dimension");

    if (m->nelement())
    {
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(m->element_pt(0));
      DynamicBulkElementInstance *ci = be->get_code_instance();
      auto *ft = ci->get_func_table();
      for (unsigned int ind = 0; ind < ft->numtracer_advections; ind++)
      {
        if (std::string(ft->tracer_advection_names[ind]) == this->tracer_name)
        {
          tracer_code_index = ind;
          break;
        }
      }
    }

    // std::cout << " TRACER SET MESH " << nodal_dim << "  " << elem_dim << std::endl;
  }

  double TracerCollection::get_time(unsigned index)
  {
    oomph::TimeStepper *ts = NULL;
    if (mesh->nnode())
      ts = mesh->node_pt(0)->time_stepper_pt();
    else if (mesh->nelement())
    {
      auto *elpt = mesh->element_pt(0);
      auto *bebase = dynamic_cast<pyoomph::BulkElementBase *>(elpt);
      if (bebase)
      {
        if (bebase->nnode())
          ts = bebase->node_pt(0)->time_stepper_pt();
        else if (bebase->ninternal_data())
          ts = bebase->internal_data_pt(0)->time_stepper_pt();
        else if (bebase->nexternal_data())
          ts = bebase->external_data_pt(0)->time_stepper_pt();
      }
    }
    if (ts)
    {
      return ts->time_pt()->time(index);
    }
    return 0.0;
  }

  TracerParticle *TracerCollection::add_tracer(std::vector<double> pos, int tag)
  {
    TracerParticle *tp = new TracerParticle();
    tp->pos.resize(pos.size());
    for (unsigned int i = 0; i < pos.size(); i++)
      tp->pos[i] = pos[i];
    tp->tag = tag;
    tp->timefrac = 0.0;
    this->add_tracer(tp);
    return tp;
  }

  void TracerCollection::locate_elements()
  {
    if (!mesh)
      throw_runtime_error("Can only locate elements of the tracers when a mesh was set");
    pyoomph::MeshKDTree kdtree(mesh, false, 0);
    for (unsigned int i = 0; i < tracers.size(); i++)
      if (tracers[i])
      {
        tracers[i]->elem = kdtree.find_element(tracers[i]->pos, tracers[i]->s);
        if (!tracers[i]->elem)
          this->remove_tracer(i);
      }
    last_lagrangian_kdtree = mesh->get_lagrangian_kdtree();

    if (mesh->nelement())
    {
      BulkElementBase *be = dynamic_cast<BulkElementBase *>(mesh->element_pt(0));
      DynamicBulkElementInstance *ci = be->get_code_instance();
      auto *ft = ci->get_func_table();
      for (unsigned int ind = 0; ind < ft->numtracer_advections; ind++)
      {
        if (std::string(ft->tracer_advection_names[ind]) == this->tracer_name)
        {
          tracer_code_index = ind;
          break;
        }
      }
    }
  }

  void TracerCollection::prepare_advection()
  {
    if (mesh->get_lagrangian_kdtree() != last_lagrangian_kdtree)
    {
      locate_elements();
    }
    for (unsigned int i = 0; i < tracers.size(); i++)
      if (tracers[i])
      {
        tracers[i]->timefrac = 0.0; // Particles are at the old step
      }
  }

  void TracerCollection::advect_all()
  {
    if (mesh->get_lagrangian_kdtree() != last_lagrangian_kdtree)
    {
      locate_elements();
    }
    double dt = this->get_time() - this->get_time(1);
    for (unsigned int i = 0; i < tracers.size(); i++)
      if (tracers[i])
      {
        tracers[i]->advect(dt);
        if (tracers[i])
        {
          if (tracers[i]->elem)
            tracers[i]->update_position_from_s();
        }
      }
    for (unsigned int i = 0; i < tracers.size(); i++)
      if (tracers[i])
      {
        if (!tracers[i]->elem)
          this->remove_tracer(i);
      }
  }

  void TracerCollection::_save_state(std::vector<double> &posarr, std::vector<int> &tagarr)
  {
    std::vector<unsigned> inds = get_allocated_indices();
    unsigned cd = this->get_coordinate_dimension();
    posarr.resize(cd * inds.size(), 0.0);

    tagarr.resize(inds.size(), 0);
    unsigned arrind = 0;
    for (auto ind : inds)
    {
      tagarr[arrind] = tracers[ind]->tag;
      for (unsigned d = 0; d < cd; d++)
        posarr[cd * arrind + d] = tracers[ind]->pos[d];
      arrind++;
    }
  }
  void TracerCollection::_load_state(std::vector<double> &posarr, std::vector<int> &tagarr)
  {
    clear(true);
    unsigned cd = this->get_coordinate_dimension();
    std::vector<double> pos(cd, 0.0);
    for (unsigned int i = 0; i < tagarr.size(); i++)
    {
      for (unsigned d = 0; d < cd; d++)
        pos[d] = posarr[i * cd + d];
      add_tracer(pos, tagarr[i]);
    }
  }

  void TracerCollection::set_transfer_interface(unsigned boundary_index, TracerCollection *opp)
  {
    transfer_interfaces[boundary_index] = TracerTransferInterfaceInfo();
    transfer_interfaces[boundary_index].other_collection = opp;
  }

}
