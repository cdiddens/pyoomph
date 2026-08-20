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

#include "tracers.hpp"

#include "elements.hpp"
#include "exception.hpp"
#include "pointlocator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace pyoomph
{

  namespace
  {
    // Dense solve for n <= 3 by Gaussian elimination with partial pivoting. Everything here is
    // 1x1, 2x2 or 3x3, so this is both the fastest and the least error-prone option; the previous
    // code carried three hand-expanded adjugate formulas, one per dimension, and simply threw for
    // anything that was not square.
    bool solve_small(double *A, unsigned n, double *b, double *xout)
    {
      unsigned piv[3] = {0, 1, 2};
      for (unsigned c = 0; c < n; c++)
      {
        unsigned best = c;
        double bestv = std::abs(A[piv[c] * n + c]);
        for (unsigned r = c + 1; r < n; r++)
        {
          const double v = std::abs(A[piv[r] * n + c]);
          if (v > bestv)
          {
            bestv = v;
            best = r;
          }
        }
        if (bestv < 1e-300)
          return false;
        std::swap(piv[c], piv[best]);
        for (unsigned r = c + 1; r < n; r++)
        {
          const double f = A[piv[r] * n + c] / A[piv[c] * n + c];
          if (f == 0.0)
            continue;
          for (unsigned k = c; k < n; k++)
            A[piv[r] * n + k] -= f * A[piv[c] * n + k];
          b[piv[r]] -= f * b[piv[c]];
        }
      }
      for (int c = (int)n - 1; c >= 0; c--)
      {
        double sum = b[piv[c]];
        for (unsigned k = c + 1; k < n; k++)
          sum -= A[piv[c] * n + k] * xout[k];
        xout[c] = sum / A[piv[c] * n + c];
      }
      return true;
    }

    // Bogacki-Shampine 3(2), FSAL. Third order with an embedded second-order estimate, three new
    // derivative evaluations per accepted step.
    //
    // Not a 5(4) method on purpose: the advection field is only C0 across element faces, so a
    // higher-order method cannot realise its order on any sub-step that straddles one, while
    // costing six stages times up to three history levels of generated-code calls each.
    // Progress watchdog of advect_one: every PROGRESS_CHECK_INTERVAL sub-steps the particle has to
    // have covered at least PROGRESS_PER_CHECK of the timestep. The two are two orders of magnitude
    // apart from both the pathology they catch (about 2e-7 per hundred sub-steps, measured) and the
    // hardest legitimate trajectory the sub-step budget allows (1e-4 per hundred), so neither the
    // element size nor the velocity enters the choice.
    const unsigned long PROGRESS_CHECK_INTERVAL = 100;
    const double PROGRESS_PER_CHECK = 1e-6;

    const double BS_A21 = 0.5;
    const double BS_A32 = 0.75;
    const double BS_B1 = 2.0 / 9.0, BS_B2 = 1.0 / 3.0, BS_B3 = 4.0 / 9.0;
    const double BS_E1 = 7.0 / 24.0, BS_E2 = 0.25, BS_E3 = 1.0 / 3.0, BS_E4 = 0.125;
  }

  // ------------------------------------------------------------------------------------------
  // TracerTimeConfig
  // ------------------------------------------------------------------------------------------

  TracerTimeConfig TracerTimeConfig::from_mesh(Mesh *m, int requested_order, unsigned nlevel_available)
  {
    TracerTimeConfig cfg;

    const oomph::TimeStepper *ts = nullptr;
    if (m->nnode())
      ts = m->node_pt(0)->time_stepper_pt();
    else if (m->nelement())
    {
      auto *be = dynamic_cast<BulkElementBase *>(m->element_pt(0));
      if (be)
      {
        if (be->nnode())
          ts = be->node_pt(0)->time_stepper_pt();
        else if (be->ninternal_data())
          ts = be->internal_data_pt(0)->time_stepper_pt();
        else if (be->nexternal_data())
          ts = be->external_data_pt(0)->time_stepper_pt();
      }
    }
    if (!ts)
      return cfg; // nothing to advect over

    const double t0 = ts->time_pt()->time(0);
    const double t1 = ts->time_pt()->time(1);
    cfg.t_current = t0;
    cfg.dt = t0 - t1;
    if (cfg.dt <= 0.0)
      return cfg;

    unsigned want = 2;
    if (requested_order >= 0)
      want = std::min<unsigned>((unsigned)requested_order + 1, 3);
    else
      want = 3;
    want = std::min(want, nlevel_available);
    // Time::ndt() is the number of stored timestep SIZES, while time(t) walks back over them, so
    // level t is available for t <= ndt(). Capping at ndt() itself silently forced linear
    // interpolation on every problem.
    want = std::min<unsigned>(want, ts->time_pt()->ndt() + 1);

    cfg.tau_of_level[0] = 1.0;
    cfg.tau_of_level[1] = 0.0;
    if (want >= 3)
    {
      const double dtprev = t1 - ts->time_pt()->time(2);
      // An impulsive start leaves t(1) == t(2); the quadratic basis is then singular and the
      // "third level" carries no information anyway.
      if (dtprev <= 1e-12 * cfg.dt)
        want = 2;
      else
        cfg.tau_of_level[2] = -dtprev / cfg.dt;
    }
    cfg.nlevel = want;
    cfg.set_tau(0.0);
    return cfg;
  }

  void TracerTimeConfig::set_tau(double tau)
  {
    for (unsigned k = 0; k < 3; k++)
    {
      w[k] = 0.0;
      dwdtau[k] = 0.0;
    }
    if (nlevel <= 1)
    {
      w[0] = 1.0;
      return;
    }
    // Lagrange basis and its derivative, written out rather than differentiated numerically so
    // that dX/dtau is the exact derivative of the interpolant J is taken from. Those two being
    // the same interpolant is what makes a Lagrangian mesh give an identically zero derivative.
    for (unsigned k = 0; k < nlevel; k++)
    {
      double num = 1.0, den = 1.0, deriv = 0.0;
      for (unsigned m = 0; m < nlevel; m++)
      {
        if (m == k)
          continue;
        num *= (tau - tau_of_level[m]);
        den *= (tau_of_level[k] - tau_of_level[m]);
      }
      for (unsigned m = 0; m < nlevel; m++)
      {
        if (m == k)
          continue;
        double term = 1.0;
        for (unsigned j = 0; j < nlevel; j++)
        {
          if (j == k || j == m)
            continue;
          term *= (tau - tau_of_level[j]);
        }
        deriv += term;
      }
      w[k] = num / den;
      dwdtau[k] = deriv / den;
    }
  }

  // ------------------------------------------------------------------------------------------
  // TracerCollection: setup and storage
  // ------------------------------------------------------------------------------------------

  TracerCollection::~TracerCollection()
  {
    clear();
    drop_locators();
  }

  void TracerCollection::drop_locators()
  {
    for (unsigned k = 0; k < 2; k++)
    {
      delete locator[k];
      locator[k] = nullptr;
    }
    has_locator_generation = false;
  }

  bool TracerCollection::generation_changed() const
  {
    return !has_locator_generation || mesh->get_topology_generation() != locator_generation;
  }

  MeshPointLocator *TracerCollection::get_adjacency_locator()
  {
    if (!mesh)
      throw_runtime_error("Tracer collection has no mesh");
    // A new topology generation invalidates the element pointers themselves, so nothing cached is
    // usable then. Otherwise take whatever is already built, however old its geometry: the walk
    // only ever asks it which elements share a node with which, and that does not move.
    if (!generation_changed())
    {
      if (locator[0])
        return locator[0];
      if (locator[1])
        return locator[1];
    }
    return get_locator(0);
  }

  MeshPointLocator *TracerCollection::get_locator(unsigned time_level)
  {
    if (!mesh)
      throw_runtime_error("Tracer collection has no mesh");
    if (generation_changed() || geometry_stale)
    {
      drop_locators();
      locator_generation = mesh->get_topology_generation();
      has_locator_generation = true;
      geometry_stale = false;
    }
    if (time_level > 1)
      throw_runtime_error("Tracers only locate in time levels 0 and 1");
    if (!locator[time_level])
    {
      LocatorSetup ls;
      ls.space = LocatorSpace::Eulerian;
      ls.time_index = time_level;
      locator[time_level] = new MeshPointLocator(mesh, ls);
    }
    return locator[time_level];
  }

  void TracerCollection::set_mesh(Mesh *m)
  {
    mesh = m;
    drop_locators();
    mark_geometry_stale();
    nodal_dim = m->get_nodal_dimension();
    const int ed = m->get_element_dimension();
    if (ed < 0)
      throw_runtime_error("Cannot attach tracers to a mesh with a negative element dimension");
    elem_dim = (unsigned)ed;
    if (nodal_dim < elem_dim || nodal_dim - elem_dim > 1)
    {
      throw_runtime_error("Tracers can only live on a bulk domain (codimension 0) or on an interface "
                          "of it (codimension 1), but this mesh has element dimension " +
                          std::to_string(elem_dim) + " in " + std::to_string(nodal_dim) + " dimensions");
    }
    for (auto *p : tracers)
    {
      p->elem = nullptr;
      p->x.resize(nodal_dim, 0.0);
      p->s.assign(elem_dim, 0.0);
    }
    resolve_code_indices();
  }

  void TracerCollection::set_num_payloads(unsigned n)
  {
    n_payload = n;
    for (auto *p : tracers)
      p->payload.resize(n, 0.0);
    resolve_code_indices();
  }

  // Each tracer name registers one generated-code entry per nodal time-history level, under the
  // name with an "@k" suffix for k > 0, and likewise one per payload. A level that was not
  // registered caps the time-interpolation order rather than being an error: a code generated
  // before a payload was added, or with only two levels, is still usable.
  void TracerCollection::resolve_code_indices()
  {
    code_index[0] = code_index[1] = code_index[2] = -1;
    payload_code_index.assign(3 * n_payload, -1);
    if (!mesh || !mesh->nelement())
      return;
    auto *be = dynamic_cast<BulkElementBase *>(mesh->element_pt(0));
    if (!be)
      return;
    auto *ft = be->get_jit_code()->get_func_table();
    for (unsigned ind = 0; ind < ft->numtracer_advections; ind++)
    {
      const std::string nm(ft->tracer_advection_names[ind]);
      for (unsigned k = 0; k < 3; k++)
      {
        const std::string want = (k == 0 ? tracer_name : tracer_name + "@" + std::to_string(k));
        if (nm == want)
          code_index[k] = (int)ind;
        for (unsigned pi = 0; pi < n_payload; pi++)
        {
          const std::string wantp = want + "/payload" + std::to_string(pi);
          if (nm == wantp)
            payload_code_index[k * n_payload + pi] = (int)ind;
        }
      }
    }
  }

  void TracerCollection::clear()
  {
    for (auto *p : tracers)
      delete p;
    tracers.clear();
    for (auto *p : pending_reinject)
      delete p;
    pending_reinject.clear();
    for (auto *p : dead)
      delete p;
    dead.clear();
  }

  void TracerCollection::retire(TracerParticle *p)
  {
    // Without a window there is nothing to fade, so this is exactly the delete it always was.
    if (history_window <= 0.0 || !p->hist_n)
    {
      delete p;
      return;
    }
    p->elem = nullptr; // it is in no element any more, and nothing may try to use one
    dead.push_back(p);
  }

  void TracerCollection::prune_dead(double tnow)
  {
    if (dead.empty())
      return;
    const unsigned stride = 1 + nodal_dim;
    std::vector<TracerParticle *> keep;
    keep.reserve(dead.size());
    for (auto *p : dead)
    {
      // Unlike a living particle, which always keeps its newest sample because that sample IS its
      // current position, a dead one is pruned all the way down and then forgotten.
      while (p->hist_n > 0)
      {
        const unsigned cap = (unsigned)(p->hist.size() / stride);
        const unsigned oldest = (p->hist_head + cap - p->hist_n) % cap;
        if (tnow - p->hist[(size_t)oldest * stride] <= history_window)
          break;
        p->hist_n--;
      }
      if (p->hist_n)
        keep.push_back(p);
      else
        delete p;
    }
    dead.swap(keep);
  }

  std::vector<long long> TracerCollection::get_dead_ids() const
  {
    std::vector<long long> ret;
    ret.reserve(dead.size());
    for (auto *p : dead)
      ret.push_back((long long)p->id);
    return ret;
  }

  // Builds a particle at `pos` and places it. Returns null (having deleted it) if the point does not
  // lie in a non-halo element of this process's part of the mesh.
  TracerParticle *TracerCollection::make_and_place(const std::vector<double> &pos, int tag,
                                                  const std::vector<double> &payload_init)
  {
    TracerParticle *p = new TracerParticle();
    p->x.assign(nodal_dim, 0.0);
    for (unsigned i = 0; i < std::min<unsigned>(nodal_dim, (unsigned)pos.size()); i++)
      p->x[i] = pos[i];
    p->s.assign(elem_dim, 0.0);
    p->payload.assign(n_payload, 0.0);
    for (unsigned i = 0; i < std::min<unsigned>(n_payload, (unsigned)payload_init.size()); i++)
      p->payload[i] = payload_init[i];
    p->tag = tag;

    if (!place_globally(p, 0) || (p->elem && p->elem->is_halo()))
    {
      // A halo element is somebody else's; letting both keep the particle would advect it twice and
      // report it twice.
      delete p;
      return nullptr;
    }
    // The projection may have moved an interface particle onto the surface; make that the position
    // rather than keeping the requested point, so that the normal offset starts at zero.
    if (get_codimension() == 1 && p->elem)
    {
      oomph::Vector<double> s(p->s.size());
      for (unsigned a = 0; a < p->s.size(); a++)
        s[a] = p->s[a];
      oomph::Vector<double> xs;
      TracerTimeConfig frozen;
      p->elem->tracer_geometry_at_s(s, 1, frozen.w, frozen.dwdtau, &xs, nullptr, nullptr);
      for (unsigned i = 0; i < nodal_dim; i++)
        p->x[i] = xs[i];
    }
    return p;
  }

  TracerId TracerCollection::add_tracer(const std::vector<double> &pos, int tag,
                                        const std::vector<double> &payload_init)
  {
    if (!mesh)
      throw_runtime_error("Cannot add tracers before a mesh was set");
    TracerParticle *p = make_and_place(pos, tag, payload_init);
    if (!p)
      return 0;
    // Rank-tagged so that particles created independently on different processes cannot collide.
    p->id = (((TracerId)mpi_rank() + 1) << 48) | next_id;
    next_id++;
    tracers.push_back(p);
    return p->id;
  }

  unsigned TracerCollection::add_tracers_collective(const std::vector<double> &pos,
                                                    const std::vector<int> &tags,
                                                    const std::vector<double> &payload_init,
                                                    const std::vector<long long> &ids)
  {
    if (!mesh)
      throw_runtime_error("Cannot add tracers before a mesh was set");
    if (!nodal_dim)
      return 0;
    const unsigned n = (unsigned)(pos.size() / nodal_dim);

    // Every process tries every candidate; a candidate is claimed by the lowest-numbered process
    // that holds it in a non-halo element. The tie only arises for a point exactly on a shared
    // face, which the locator's inside tolerance accepts on both sides.
    const int nproc = mpi_nproc(), rank = mpi_rank();
    std::vector<int> claimant(n, nproc);
    std::vector<TracerParticle *> mine(n, nullptr);
    std::vector<double> one(nodal_dim, 0.0), pay(n_payload, 0.0);
    for (unsigned i = 0; i < n; i++)
    {
      for (unsigned d = 0; d < nodal_dim; d++)
        one[d] = pos[(size_t)i * nodal_dim + d];
      for (unsigned d = 0; d < n_payload && (size_t)i * n_payload + d < payload_init.size(); d++)
        pay[d] = payload_init[(size_t)i * n_payload + d];
      TracerParticle *p = make_and_place(one, i < tags.size() ? tags[i] : 0, pay);
      if (p)
      {
        claimant[i] = rank;
        mine[i] = p;
      }
    }

#ifdef OOMPH_HAS_MPI
    if (is_distributed() && n)
    {
      std::vector<int> reduced(n, nproc);
      MPI_Allreduce(claimant.data(), reduced.data(), (int)n, MPI_INT, MPI_MIN,
                    mesh->communicator_pt()->mpi_comm());
      claimant.swap(reduced);
    }
#endif

    unsigned nowhere = 0;
    for (unsigned i = 0; i < n; i++)
    {
      if (claimant[i] >= nproc)
      {
        nowhere++;
        delete mine[i];
        continue;
      }
      if (claimant[i] != rank)
      {
        delete mine[i]; // another process claimed it
        continue;
      }
      // The identity is the candidate's index in the list, which every process agrees on, so the
      // particle set and its ids are the same however the mesh happens to be partitioned.
      mine[i]->id = (i < ids.size()) ? (TracerId)ids[i] : (next_id + i);
      tracers.push_back(mine[i]);
      next_id = std::max(next_id, mine[i]->id + 1);
    }
    if (ids.empty())
      next_id += n;
    return nowhere;
  }

  bool TracerCollection::remove_tracer(TracerId id)
  {
    for (unsigned i = 0; i < tracers.size(); i++)
    {
      if (tracers[i]->id == id)
      {
        delete tracers[i];
        tracers.erase(tracers.begin() + i);
        return true;
      }
    }
    return false;
  }

  std::vector<double> TracerCollection::get_positions()
  {
    std::vector<double> ret;
    ret.reserve(tracers.size() * nodal_dim);
    for (auto *p : tracers)
      for (unsigned i = 0; i < nodal_dim; i++)
        ret.push_back(p->x[i]);
    return ret;
  }

  std::vector<long long> TracerCollection::get_ids() const
  {
    std::vector<long long> ret;
    ret.reserve(tracers.size());
    for (auto *p : tracers)
      ret.push_back((long long)p->id);
    return ret;
  }

  std::vector<int> TracerCollection::get_tags() const
  {
    std::vector<int> ret;
    ret.reserve(tracers.size());
    for (auto *p : tracers)
      ret.push_back(p->tag);
    return ret;
  }

  std::vector<double> TracerCollection::get_payloads() const
  {
    std::vector<double> ret;
    ret.reserve(tracers.size() * n_payload);
    for (auto *p : tracers)
      for (unsigned i = 0; i < n_payload; i++)
        ret.push_back(p->payload[i]);
    return ret;
  }

  // Unwrap one particle's ring into chronological (t, x...) samples, oldest first.
  std::vector<double> TracerCollection::history_of(const TracerParticle *p) const
  {
    const unsigned stride = 1 + nodal_dim;
    std::vector<double> ret;
    if (!p->hist_n)
      return ret;
    ret.reserve(p->hist_n * stride);
    const unsigned cap = (unsigned)(p->hist.size() / stride);
    for (unsigned k = 0; k < p->hist_n; k++)
    {
      const unsigned slot = (p->hist_head + cap - p->hist_n + k) % cap;
      for (unsigned j = 0; j < stride; j++)
        ret.push_back(p->hist[slot * stride + j]);
    }
    return ret;
  }

  // The inverse: fill p's ring from `count` chronological samples, keeping the newest ones that fit.
  void TracerCollection::set_history(TracerParticle *p, const double *samples, unsigned count)
  {
    const unsigned stride = 1 + nodal_dim;
    p->hist_n = 0;
    p->hist_head = 0;
    if (!history_capacity)
    {
      p->hist.clear();
      return;
    }
    p->hist.assign((size_t)history_capacity * stride, 0.0);
    const unsigned take = std::min(count, history_capacity);
    if (!take)
      return;
    const double *src = samples + (size_t)(count - take) * stride;
    for (unsigned k = 0; k < take; k++)
      for (unsigned j = 0; j < stride; j++)
        p->hist[(size_t)k * stride + j] = src[(size_t)k * stride + j];
    p->hist_n = take;
    p->hist_head = take % history_capacity;
  }

  std::vector<double> TracerCollection::get_history_of(TracerId id) const
  {
    for (auto *p : tracers)
      if (p->id == id)
        return history_of(p);
    // Dead too: a fading trail is asked for by exactly this call.
    for (auto *p : dead)
      if (p->id == id)
        return history_of(p);
    return std::vector<double>();
  }

  void TracerCollection::set_transfer_interface(unsigned boundary_index, TracerCollection *opp)
  {
    transfer_interfaces[boundary_index] = TracerTransferInterfaceInfo();
    transfer_interfaces[boundary_index].other_collection = opp;
  }

  void TracerCollection::add_periodic_wrap(const std::vector<double> &shift)
  {
    bool nonzero = false;
    for (double v : shift)
      if (v != 0.0)
        nonzero = true;
    if (!nonzero)
      throw_runtime_error("A periodic wrap of tracers '" + tracer_name +
                          "' must have a non-zero shift; a zero one would put the particle back "
                          "exactly where it just left the mesh");
    for (const auto &have : periodic_wraps)
    {
      bool same = have.size() == shift.size();
      for (unsigned i = 0; same && i < shift.size(); i++)
        if (have[i] != shift[i])
          same = false;
      if (same)
        return; // registering both ends of a periodic pair must not double up
    }
    periodic_wraps.push_back(shift);
  }

  void TracerCollection::clear_periodic_wraps()
  {
    periodic_wraps.clear();
  }

  // ------------------------------------------------------------------------------------------
  // MPI
  // ------------------------------------------------------------------------------------------

  int TracerCollection::mpi_nproc() const
  {
#ifdef OOMPH_HAS_MPI
    if (mesh && mesh->communicator_pt())
      return mesh->communicator_pt()->nproc();
#endif
    return 1;
  }

  int TracerCollection::mpi_rank() const
  {
#ifdef OOMPH_HAS_MPI
    if (mesh && mesh->communicator_pt())
      return mesh->communicator_pt()->my_rank();
#endif
    return 0;
  }

  bool TracerCollection::is_distributed() const
  {
#ifdef OOMPH_HAS_MPI
    return mesh && mesh->is_mesh_distributed() && mpi_nproc() > 1;
#else
    return false;
#endif
  }

  // Everything a particle needs to continue on another process, as one fixed-length record of
  // doubles: position, local coordinate, payloads, sub-step state and the whole position history.
  // Fixed length so the exchange is a single Alltoallv with a stride both sides agree on without
  // negotiating it.
  unsigned TracerCollection::record_stride() const
  {
    return nodal_dim + n_payload + 3 + 1 + history_capacity * (1 + nodal_dim);
  }

  void TracerCollection::pack(const TracerParticle *p, double *out) const
  {
    unsigned k = 0;
    for (unsigned i = 0; i < nodal_dim; i++)
      out[k++] = p->x[i];
    for (unsigned i = 0; i < n_payload; i++)
      out[k++] = p->payload[i];
    out[k++] = p->timefrac;
    out[k++] = p->next_h;
    out[k++] = p->completed_step_time;
    const unsigned cap = history_capacity;
    out[k++] = (double)p->hist_n;
    for (unsigned j = 0; j < p->hist_n; j++)
    {
      const unsigned stride = 1 + nodal_dim;
      const unsigned slot = (p->hist_head + cap - p->hist_n + j) % cap;
      for (unsigned c = 0; c < stride; c++)
        out[k++] = p->hist.empty() ? 0.0 : p->hist[slot * stride + c];
    }
    while (k < record_stride())
      out[k++] = 0.0;
  }

  TracerParticle *TracerCollection::unpack(const double *in, TracerId id, int tag)
  {
    TracerParticle *p = new TracerParticle();
    unsigned k = 0;
    p->x.assign(nodal_dim, 0.0);
    for (unsigned i = 0; i < nodal_dim; i++)
      p->x[i] = in[k++];
    p->payload.assign(n_payload, 0.0);
    for (unsigned i = 0; i < n_payload; i++)
      p->payload[i] = in[k++];
    p->timefrac = in[k++];
    p->next_h = in[k++];
    p->completed_step_time = in[k++];
    p->s.assign(elem_dim, 0.0);
    p->id = id;
    p->tag = tag;
    const unsigned nh = (unsigned)(in[k++] + 0.5);
    const unsigned stride = 1 + nodal_dim;
    if (nh)
    {
      p->hist.assign((size_t)history_capacity * stride, 0.0);
      for (unsigned j = 0; j < nh && j < history_capacity; j++)
        for (unsigned c = 0; c < stride; c++)
          p->hist[j * stride + c] = in[k + j * stride + c];
      p->hist_n = std::min(nh, history_capacity);
      p->hist_head = p->hist_n % history_capacity;
    }
    return p;
  }

  unsigned TracerCollection::exchange_migrants()
  {
#ifdef OOMPH_HAS_MPI
    if (!is_distributed())
      return 0;
    const int nproc = mpi_nproc();
    MPI_Comm comm = mesh->communicator_pt()->mpi_comm();
    const unsigned stride = record_stride();

    // Partition the local particles into keepers and migrants, by whether the element they ended in
    // is a halo of somebody else's.
    std::vector<std::vector<double>> send_d((size_t)nproc);
    std::vector<std::vector<long long>> send_i((size_t)nproc);
    std::vector<TracerParticle *> keep;
    keep.reserve(tracers.size());
    unsigned nsent = 0;
    for (auto *p : tracers)
    {
      int dest = -1;
      if (p->elem && p->elem->is_halo())
        dest = p->elem->non_halo_proc_ID();
      if (dest < 0 || dest >= nproc || dest == mpi_rank())
      {
        keep.push_back(p);
        continue;
      }
      const size_t base = send_d[dest].size();
      send_d[dest].resize(base + stride, 0.0);
      pack(p, send_d[dest].data() + base);
      send_i[dest].push_back((long long)p->id);
      send_i[dest].push_back((long long)p->tag);
      delete p;
      nsent++;
    }
    tracers.swap(keep);

    std::vector<int> scount(nproc), rcount(nproc), sdispl(nproc, 0), rdispl(nproc, 0);
    for (int r = 0; r < nproc; r++)
      scount[r] = (int)(send_d[r].size() / stride);
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, comm);

    std::vector<double> sbuf, rbuf;
    std::vector<long long> sids, rids;
    std::vector<int> sd(nproc), rd(nproc), sdi(nproc), rdi(nproc), scd(nproc), rcd(nproc), sci(nproc), rci(nproc);
    int so = 0, ro = 0, soi = 0, roi = 0;
    for (int r = 0; r < nproc; r++)
    {
      scd[r] = scount[r] * (int)stride;
      rcd[r] = rcount[r] * (int)stride;
      sci[r] = scount[r] * 2;
      rci[r] = rcount[r] * 2;
      sd[r] = so;
      rd[r] = ro;
      sdi[r] = soi;
      rdi[r] = roi;
      so += scd[r];
      ro += rcd[r];
      soi += sci[r];
      roi += rci[r];
      sbuf.insert(sbuf.end(), send_d[r].begin(), send_d[r].end());
      sids.insert(sids.end(), send_i[r].begin(), send_i[r].end());
    }
    rbuf.assign((size_t)std::max(ro, 1), 0.0);
    rids.assign((size_t)std::max(roi, 1), 0);
    MPI_Alltoallv(sbuf.empty() ? nullptr : sbuf.data(), scd.data(), sd.data(), MPI_DOUBLE,
                  rbuf.data(), rcd.data(), rd.data(), MPI_DOUBLE, comm);
    MPI_Alltoallv(sids.empty() ? nullptr : sids.data(), sci.data(), sdi.data(), MPI_LONG_LONG,
                  rids.data(), rci.data(), rdi.data(), MPI_LONG_LONG, comm);

    const unsigned nrecv = (unsigned)(ro / (int)stride);
    for (unsigned i = 0; i < nrecv; i++)
    {
      TracerParticle *p = unpack(&rbuf[(size_t)i * stride], (TracerId)rids[2 * i], (int)rids[2 * i + 1]);
      // The sender's position is at the end of its sub-stepping, so the level-0 configuration is
      // the right one to place it in.
      if (place_globally(p, 0))
        tracers.push_back(p);
      else
      {
        stat_lost++;
        retire(p);
      }
    }

    unsigned total = 0;
    unsigned mine = nsent;
    MPI_Allreduce(&mine, &total, 1, MPI_UNSIGNED, MPI_SUM, comm);
    return total;
#else
    return 0;
#endif
  }

  unsigned long TracerCollection::nglobal() const
  {
    unsigned long mine = (unsigned long)tracers.size();
#ifdef OOMPH_HAS_MPI
    if (is_distributed())
    {
      unsigned long total = 0;
      MPI_Allreduce(&mine, &total, 1, MPI_UNSIGNED_LONG, MPI_SUM, mesh->communicator_pt()->mpi_comm());
      return total;
    }
#endif
    return mine;
  }

  // Gathers are written once, over a per-particle block of doubles, and the four public views just
  // pick columns out of it. Sorting by the (never recycled, partition-independent) id is what makes
  // the result the same on every process and independent of how the mesh was split.
  namespace
  {
    struct GatheredRow
    {
      long long id;
      std::vector<double> vals;
      bool operator<(const GatheredRow &o) const { return id < o.id; }
    };
  }

  std::vector<double> TracerCollection::gather_positions() const
  {
    std::vector<double> local;
    local.reserve(tracers.size() * nodal_dim);
    for (auto *p : tracers)
      for (unsigned i = 0; i < nodal_dim; i++)
        local.push_back(p->x[i]);
    return gather_rows(local, nodal_dim);
  }

  std::vector<double> TracerCollection::gather_payloads() const
  {
    std::vector<double> local;
    local.reserve(tracers.size() * n_payload);
    for (auto *p : tracers)
      for (unsigned i = 0; i < n_payload; i++)
        local.push_back(p->payload[i]);
    return gather_rows(local, n_payload);
  }

  std::vector<long long> TracerCollection::gather_ids() const
  {
    std::vector<double> none;
    std::vector<long long> ids;
    gather_rows(none, 0, &ids);
    return ids;
  }

  std::vector<int> TracerCollection::gather_tags() const
  {
    std::vector<double> local;
    local.reserve(tracers.size());
    for (auto *p : tracers)
      local.push_back((double)p->tag);
    std::vector<double> all = gather_rows(local, 1);
    std::vector<int> ret;
    ret.reserve(all.size());
    for (double v : all)
      ret.push_back((int)(v < 0 ? v - 0.5 : v + 0.5));
    return ret;
  }

  std::vector<double> TracerCollection::gather_rows(const std::vector<double> &local, unsigned ncol,
                                                    std::vector<long long> *ids_out) const
  {
    std::vector<long long> local_ids;
    local_ids.reserve(tracers.size());
    for (auto *p : tracers)
      local_ids.push_back((long long)p->id);

    std::vector<long long> all_ids = local_ids;
    std::vector<double> all = local;

#ifdef OOMPH_HAS_MPI
    if (is_distributed())
    {
      MPI_Comm comm = mesh->communicator_pt()->mpi_comm();
      const int nproc = mpi_nproc();
      int mine = (int)local_ids.size();
      std::vector<int> counts(nproc, 0);
      MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
      std::vector<int> displ(nproc, 0), dcounts(nproc, 0), ddispl(nproc, 0);
      int tot = 0, dtot = 0;
      for (int r = 0; r < nproc; r++)
      {
        displ[r] = tot;
        tot += counts[r];
        dcounts[r] = counts[r] * (int)ncol;
        ddispl[r] = dtot;
        dtot += dcounts[r];
      }
      all_ids.assign((size_t)std::max(tot, 1), 0);
      all.assign((size_t)std::max(dtot, 1), 0.0);
      MPI_Allgatherv(local_ids.empty() ? nullptr : local_ids.data(), mine, MPI_LONG_LONG,
                     all_ids.data(), counts.data(), displ.data(), MPI_LONG_LONG, comm);
      if (ncol)
        MPI_Allgatherv(local.empty() ? nullptr : local.data(), mine * (int)ncol, MPI_DOUBLE,
                       all.data(), dcounts.data(), ddispl.data(), MPI_DOUBLE, comm);
      all_ids.resize((size_t)tot);
      all.resize((size_t)dtot);
    }
#endif

    std::vector<unsigned> order(all_ids.size());
    for (unsigned i = 0; i < order.size(); i++)
      order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](unsigned a, unsigned b) { return all_ids[a] < all_ids[b]; });

    if (ids_out)
    {
      ids_out->clear();
      ids_out->reserve(order.size());
      for (unsigned i : order)
        ids_out->push_back(all_ids[i]);
    }
    std::vector<double> ret;
    ret.reserve(order.size() * ncol);
    for (unsigned i : order)
      for (unsigned c = 0; c < ncol; c++)
        ret.push_back(all[(size_t)i * ncol + c]);
    return ret;
  }

  // ------------------------------------------------------------------------------------------
  // Locating
  // ------------------------------------------------------------------------------------------

  bool TracerCollection::place_globally(TracerParticle *p, unsigned time_level)
  {
    MeshPointLocator *loc = get_locator(time_level);
    std::vector<double> flat(p->x.begin(), p->x.begin() + nodal_dim);
    LocationSet ls = loc->locate_batch(flat, 1);
    BulkElementBase *e = nullptr;
    std::vector<double> s;
    if (!ls.resolve_local(0, e, s) || !e)
    {
      p->elem = nullptr;
      return false;
    }
    p->elem = e;
    p->s.assign(s.begin(), s.begin() + elem_dim);
    p->refdomain = reference_domain_kind(e);
    e->tracer_prepare_element();

    // On an interface, snap the stored position onto the located point. The locate was a
    // closest-point projection, so the two differ by however far the particle was off the surface -
    // which is zero during a run, but not after a remesh, where the new interface discretises the
    // same boundary slightly differently. Without this a particle would sit fractionally off the
    // surface until the next sub-step re-anchored it, and anything reading positions in between
    // would see the invariant broken.
    if (nodal_dim != elem_dim)
    {
      oomph::Vector<double> sv(p->s.size());
      for (unsigned a = 0; a < p->s.size(); a++)
        sv[a] = p->s[a];
      oomph::Vector<double> xs;
      TracerTimeConfig frozen;
      e->tracer_geometry_at_s(sv, 1, frozen.w, frozen.dwdtau, &xs, nullptr, nullptr);
      for (unsigned i = 0; i < nodal_dim; i++)
        p->x[i] = xs[i];
    }

    stat_global_locates++;
    return true;
  }

  // Newton (codimension 0) or Gauss-Newton on the normal equations (codimension 1) for
  // X(s, tau) = target, restarted in a neighbouring element whenever the iterate leaves the
  // reference domain.
  //
  // The walk rather than a search is deliberate. Within a sub-step the configuration is the
  // time-interpolated one, which no locator is built for; a particle that has moved further than
  // one element in one sub-step is telling the controller the sub-step was too long, and shrinking
  // it is both cheaper and more accurate than locating it globally in the wrong configuration.
  bool TracerCollection::place_at(TracerParticle *p, const TracerTimeConfig &cfg, const double *target,
                                  double *x_on_elem)
  {
    if (!p->elem)
      return false;

    const unsigned nd = nodal_dim, ed = elem_dim;
    const bool square = (nd == ed);

    // Magnitude of the coordinates, which sets the rounding noise of the residual below.
    double xmag = 1.0;
    for (unsigned i = 0; i < nd; i++)
      xmag = std::max(xmag, std::abs(target[i]));

    // Newton (square) / Gauss-Newton on the normal equations (codimension 1) in one element.
    // Returns true when the iteration converged AND landed inside the reference domain.
    auto try_element = [&](BulkElementBase *e, const std::vector<double> &seed,
                           std::vector<double> &sout) -> bool {
        oomph::Vector<double> sv(seed.size());
      for (unsigned a = 0; a < seed.size(); a++)
        sv[a] = seed[a];
      oomph::Vector<double> xs;
      oomph::DenseMatrix<double> J;
      bool converged = false;
      for (unsigned it = 0; it < 25; it++)
      {
        e->tracer_geometry_at_s(sv, cfg.nlevel, cfg.w, cfg.dwdtau, &xs, &J, nullptr);
        double r[3] = {0.0, 0.0, 0.0};
        for (unsigned i = 0; i < nd; i++)
          r[i] = target[i] - xs[i];

        double A[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, b[3] = {0, 0, 0}, ds[3] = {0, 0, 0};
        if (square)
        {
          // X(s + ds)_i ~ X(s)_i + sum_a J(a,i) ds_a, so the system matrix is J transposed.
          for (unsigned i = 0; i < nd; i++)
          {
            for (unsigned a = 0; a < ed; a++)
              A[i * ed + a] = J(a, i);
            b[i] = r[i];
          }
        }
        else
        {
          // Least squares: (J J^T) ds = J r. The tangential correction is solved for and the
          // normal part of r is deliberately left in the residual - that is the offset from the
          // surface, which is exactly what must not be corrected away.
          for (unsigned a = 0; a < ed; a++)
          {
            for (unsigned c = 0; c < ed; c++)
            {
              double v = 0.0;
              for (unsigned i = 0; i < nd; i++)
                v += J(a, i) * J(c, i);
              A[a * ed + c] = v;
            }
            double v = 0.0;
            for (unsigned i = 0; i < nd; i++)
              v += J(a, i) * r[i];
            b[a] = v;
          }
        }
        if (!solve_small(A, ed, b, ds))
          return false;

        double dsnorm = 0.0;
        for (unsigned a = 0; a < ed; a++)
        {
          sv[a] += ds[a];
          dsnorm += ds[a] * ds[a];
        }
        // Newton cannot push |ds| below the rounding noise of the residual, which is of order
        // eps*|x|, divided by the scale of the element, |dX/ds|. On a mesh of 0.1-sized elements at
        // x ~ 4 that floor is around 1e-14 itself, so the fixed 1e-14 that used to stand here was
        // simply unreachable: the iteration spent all 25 rounds bouncing on the floor, reported
        // failure, and a particle sitting comfortably inside its own element was dropped as lost.
        // The further from the origin, the more of them - which is exactly what was observed.
        // The smallest row norm of J is used, not the largest entry, so that a flat or stretched
        // element - where one reference direction resolves far less than the others - relaxes the
        // threshold rather than tightening it. What comes out is around 1e-12 in reference
        // coordinates for the mesh above, still orders of magnitude finer than anything the
        // sub-step controller can resolve.
        double jscale = 1e300;
        for (unsigned a = 0; a < ed; a++)
        {
          double rn = 0.0;
          for (unsigned i = 0; i < nd; i++)
            rn += J(a, i) * J(a, i);
          jscale = std::min(jscale, std::sqrt(rn));
        }
        const double stol = std::max(1e-14, 64.0 * std::numeric_limits<double>::epsilon() * xmag /
                                                std::max(jscale, 1e-300));
        if (std::sqrt(dsnorm) < stol)
        {
          converged = true;
          break;
        }
        // A wild iterate means this element is not the one; do not spend 25 iterations proving it.
        for (unsigned a = 0; a < ed; a++)
          if (std::abs(sv[a]) > 20.0)
            return false;
      }
      if (!converged)
        return false;
      sout.assign(sv.begin(), sv.begin() + ed);
      return inside_reference_domain(reference_domain_kind(e), e, ed, sout.data(), 1e-8);
    };

    auto accept = [&](BulkElementBase *e, const std::vector<double> &sfound) {
      if (e != p->elem)
      {
        p->elem = e;
        p->refdomain = reference_domain_kind(e);
        e->tracer_prepare_element();
        stat_walks++;
      }
      p->s = sfound;
      if (x_on_elem)
      {
        oomph::Vector<double> svf(sfound.size());
        for (unsigned a = 0; a < sfound.size(); a++)
          svf[a] = sfound[a];
        oomph::Vector<double> xf;
        e->tracer_geometry_at_s(svf, cfg.nlevel, cfg.w, cfg.dwdtau, &xf, nullptr, nullptr);
        for (unsigned i = 0; i < nd; i++)
          x_on_elem[i] = xf[i];
      }
    };

    std::vector<double> sfound;
    if (try_element(p->elem, p->s, sfound))
    {
      accept(p->elem, sfound);
      return true;
    }

    // Left the element: it must be one of the neighbours, otherwise the sub-step was too long.
    std::vector<BulkElementBase *> candidates;
    get_adjacency_locator()->neighbour_elements(p->elem, candidates);
    BulkElementBase *from = p->elem;
    for (auto *cand : candidates)
    {
      if (cand == from)
        continue;
      // Seed at the centre of the candidate's reference domain: two or three Newton steps from
      // there suffice for any element reachable within one sub-step.
      std::vector<double> seed(ed, 0.0);
      switch (reference_domain_kind(cand))
      {
      case RefDomain::Simplex:
        for (unsigned a = 0; a < ed; a++)
          seed[a] = 1.0 / (ed + 1.0);
        break;
      case RefDomain::Prism:
        seed[0] = seed[1] = 1.0 / 3.0;
        seed[2] = 0.5 * (cand->s_min() + cand->s_max());
        break;
      case RefDomain::Pyramid:
        seed[2] = 0.5 * (cand->s_min() + cand->s_max());
        seed[0] = seed[1] = 0.5 * (1.0 - seed[2]);
        break;
      default:
        for (unsigned a = 0; a < ed; a++)
          seed[a] = 0.5 * (cand->s_min() + cand->s_max());
        break;
      }
      if (try_element(cand, seed, sfound))
      {
        accept(cand, sfound);
        return true;
      }
    }
    return false;
  }

  // ------------------------------------------------------------------------------------------
  // Advection
  // ------------------------------------------------------------------------------------------

  bool TracerCollection::eval_derivative(TracerParticle *p, TracerTimeConfig &cfg, double tau,
                                         const double *y, double *dydtau, double *dpdtau)
  {
    cfg.set_tau(tau);
    if (!place_at(p, cfg, y, nullptr))
      return false;

    const unsigned nd = nodal_dim, ed = elem_dim;
    oomph::Vector<double> sv(p->s.size());
    for (unsigned a = 0; a < p->s.size(); a++)
      sv[a] = p->s[a];

    // The advection field, blended over the history levels with the same weights the configuration
    // uses. Blending the field and the geometry with one set of weights is what makes a mesh moving
    // exactly with the flow produce an identically zero derivative.
    double v[3] = {0.0, 0.0, 0.0};
    oomph::Vector<double> xvelo;
    for (unsigned k = 0; k < cfg.nlevel; k++)
    {
      if (code_index[k] < 0)
        throw_runtime_error("Tracer '" + tracer_name + "' has no generated code for history level " +
                            std::to_string(k));
      if (cfg.w[k] == 0.0)
        continue;
      p->elem->eval_tracer_advection_at_s((unsigned)code_index[k], sv, xvelo);
      for (unsigned i = 0; i < 3; i++)
      {
        if (i < nd)
          v[i] += cfg.w[k] * xvelo[i];
        else if (std::abs(xvelo[i]) > 1e-12)
        {
          throw_runtime_error("The advection field of tracer '" + tracer_name + "' has a non-zero "
                              "component " + std::to_string(i) + ", which is out of the plane of a " +
                              std::to_string(nd) + "-dimensional mesh. A tracer lives in the mesh and "
                              "cannot follow it.");
        }
      }
    }

    if (nd == ed)
    {
      // Bulk: J J^+ = I, the mesh velocity cancels analytically and never has to be computed.
      for (unsigned i = 0; i < nd; i++)
        dydtau[i] = cfg.dt * v[i];
    }
    else
    {
      // Interface: tangential part of v plus normal part of the interface's own motion.
      oomph::DenseMatrix<double> J;
      oomph::Vector<double> dXdtau;
      p->elem->tracer_geometry_at_s(sv, cfg.nlevel, cfg.w, cfg.dwdtau, nullptr, &J, &dXdtau);

      // P = J^T (J J^T)^-1 J is the projector onto the tangent space. Applied to the combination
      // dt*v - dXdtau it gives the tangential advection; the rest is the surface's own motion.
      double d[3];
      for (unsigned i = 0; i < nd; i++)
        d[i] = cfg.dt * v[i] - dXdtau[i];

      double A[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, b[3] = {0, 0, 0}, c[3] = {0, 0, 0};
      for (unsigned a = 0; a < ed; a++)
      {
        for (unsigned q = 0; q < ed; q++)
        {
          double vv = 0.0;
          for (unsigned i = 0; i < nd; i++)
            vv += J(a, i) * J(q, i);
          A[a * ed + q] = vv;
        }
        double vv = 0.0;
        for (unsigned i = 0; i < nd; i++)
          vv += J(a, i) * d[i];
        b[a] = vv;
      }
      if (!solve_small(A, ed, b, c))
        return false;
      for (unsigned i = 0; i < nd; i++)
      {
        double t = 0.0;
        for (unsigned a = 0; a < ed; a++)
          t += c[a] * J(a, i);
        dydtau[i] = t + dXdtau[i];
      }
    }

    for (unsigned pi = 0; pi < n_payload; pi++)
    {
      double acc = 0.0;
      for (unsigned k = 0; k < cfg.nlevel; k++)
      {
        const int ci = payload_code_index[k * n_payload + pi];
        if (ci < 0 || cfg.w[k] == 0.0)
          continue;
        p->elem->eval_tracer_advection_at_s((unsigned)ci, sv, xvelo);
        acc += cfg.w[k] * xvelo[0];
      }
      dpdtau[pi] = cfg.dt * acc;
    }
    return true;
  }

  // Offer a particle to every collection registered as a transfer target.
  //
  // The old implementation worked out which mesh boundary the particle had crossed by intersecting
  // the boundary sets of the element's nodes. That is unnecessary: a particle that has left this
  // mesh either lies in a registered neighbour or it does not, and simply asking each of them is
  // both shorter and correct for a particle that leaves through a corner where two boundaries meet.
  TracerCollection *TracerCollection::try_transfer(TracerParticle *p, unsigned depth)
  {
    for (auto &kv : transfer_interfaces)
    {
      TracerCollection *other = kv.second.other_collection;
      if (!other || other == this)
        continue;
      if (other->adopt(p, depth + 1))
        return other;
    }
    return nullptr;
  }

  bool TracerCollection::place_periodic_image(TracerParticle *p)
  {
    const std::vector<double> before = p->x;
    for (const auto &shift : periodic_wraps)
    {
      for (unsigned i = 0; i < nodal_dim; i++)
        p->x[i] = before[i] + (i < shift.size() ? shift[i] : 0.0);
      p->elem = nullptr;
      if (place_globally(p, 0) && !p->elem->is_halo())
      {
        // The trail is a path through the plotted coordinates, and a wrapped path is not continuous
        // there: keeping the samples from before the jump would draw a line straight back across
        // the whole domain on the next output.
        p->hist_n = 0;
        p->hist_head = 0;
        return true;
      }
    }
    p->x = before;
    p->elem = nullptr;
    return false;
  }

  TracerCollection::WrapResult TracerCollection::wrap_position(TracerParticle *p)
  {
    if (periodic_wraps.empty())
      return WrapResult::NotPlaced;
    if (place_periodic_image(p))
      return WrapResult::PlacedHere;
    if (!is_distributed())
      return WrapResult::NotPlaced;
    // The periodic image of a point at one end of the domain is at the other end, which under a
    // partitioning that knows nothing about the periodicity is somebody else's part of the mesh
    // entirely - not a halo of this one, so exchange_migrants() cannot reach it either. Park the
    // particle at the position it LEFT from, unshifted, and let the collective round apply the
    // shifts on whichever process turns out to hold the image.
    pending_reinject.push_back(p);
    return WrapResult::ParkedForReinjection;
  }

  unsigned TracerCollection::exchange_reinjections()
  {
#ifdef OOMPH_HAS_MPI
    if (is_distributed())
    {
      MPI_Comm comm = mesh->communicator_pt()->mpi_comm();
      const int nproc = mpi_nproc(), rank = mpi_rank();
      const unsigned stride = record_stride();

      int mine = (int)pending_reinject.size();
      std::vector<int> counts(nproc, 0);
      MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
      int total = 0;
      std::vector<int> displ(nproc, 0), dcount(nproc, 0), ddispl(nproc, 0), icount(nproc, 0), idispl(nproc, 0);
      for (int r = 0; r < nproc; r++)
      {
        displ[r] = total;
        total += counts[r];
      }
      if (!total)
        return 0;
      int dtot = 0, itot = 0;
      for (int r = 0; r < nproc; r++)
      {
        dcount[r] = counts[r] * (int)stride;
        ddispl[r] = dtot;
        dtot += dcount[r];
        icount[r] = counts[r] * 2;
        idispl[r] = itot;
        itot += icount[r];
      }

      std::vector<double> sbuf((size_t)std::max<int>(mine * (int)stride, 1), 0.0);
      std::vector<long long> sids((size_t)std::max(2 * mine, 1), 0);
      for (int i = 0; i < mine; i++)
      {
        pack(pending_reinject[i], sbuf.data() + (size_t)i * stride);
        sids[2 * i] = (long long)pending_reinject[i]->id;
        sids[2 * i + 1] = (long long)pending_reinject[i]->tag;
      }

      std::vector<double> rbuf((size_t)std::max(dtot, 1), 0.0);
      std::vector<long long> rids((size_t)std::max(itot, 1), 0);
      MPI_Allgatherv(sbuf.data(), mine * (int)stride, MPI_DOUBLE,
                     rbuf.data(), dcount.data(), ddispl.data(), MPI_DOUBLE, comm);
      MPI_Allgatherv(sids.data(), 2 * mine, MPI_LONG_LONG,
                     rids.data(), icount.data(), idispl.data(), MPI_LONG_LONG, comm);

      // The same claim protocol as seeding: every process tries every record and the lowest-numbered
      // one holding the image keeps it, so the outcome does not depend on the partitioning.
      std::vector<int> claimant((size_t)total, nproc);
      std::vector<TracerParticle *> cand((size_t)total, nullptr);
      for (int i = 0; i < total; i++)
      {
        TracerParticle *p = unpack(&rbuf[(size_t)i * stride], (TracerId)rids[2 * i], (int)rids[2 * i + 1]);
        if (place_periodic_image(p))
        {
          claimant[i] = rank;
          cand[i] = p;
        }
        else
          delete p;
      }
      std::vector<int> reduced((size_t)total, nproc);
      MPI_Allreduce(claimant.data(), reduced.data(), total, MPI_INT, MPI_MIN, comm);

      unsigned taken = 0;
      for (int i = 0; i < total; i++)
      {
        if (reduced[i] == rank && cand[i])
        {
          tracers.push_back(cand[i]);
          taken++;
          continue;
        }
        delete cand[i];
      }
      stat_reinjected += taken;

      // Only now can the originals go: whether one is gone for good, and so has a trail to leave
      // behind on the process that parked it, is not known until the claim has been reduced. Mine
      // are the block at displ[rank].
      for (int i = 0; i < mine; i++)
      {
        TracerParticle *p = pending_reinject[i];
        if (reduced[displ[rank] + i] >= nproc)
        {
          stat_lost++;
          retire(p); // no process holds its periodic image; it is out of the domain for good
        }
        else
          delete p; // rebuilt from the wire by whichever process claimed it
      }
      pending_reinject.clear();
      return (unsigned)total;
    }
#endif
    // Not distributed: wrap_position never parks anything, so this only ever drains a leftover.
    for (auto *p : pending_reinject)
    {
      stat_lost++;
      retire(p);
    }
    pending_reinject.clear();
    return 0;
  }

  // A confined particle cannot leave its interface, it can only reach the end of it - so when it
  // has run out of local coordinate there and no neighbouring domain wants it, it is pinned to that
  // end rather than dropped. The closest point of a curve to a target beyond its end IS that end,
  // so the clamp is the projection the interface formulation asks for, not a fudge. Dropping it
  // instead - which is what a bulk particle leaving its domain gets - would delete a particle that
  // never left anything, and at a symmetry axis, where what pushes it off the end is rounding
  // noise, that is simply wrong.
  //
  // The position is taken at tau = 1, not at the tau where the particle got stuck: a pinned
  // particle IS the end of the interface for the rest of the step and has to co-move with it.
  // Freezing it where it stalled would make it lag a receding interface by part of a step's motion
  // every step, and it would walk off the surface over time.
  bool TracerCollection::pin_to_interface_end(TracerParticle *p, TracerTimeConfig &cfg)
  {
    if (!p->elem || get_codimension() != 1)
      return false;
    clamp_to_reference_domain(p->refdomain, p->elem, elem_dim, p->s.data());
    cfg.set_tau(1.0);
    oomph::Vector<double> sv(p->s.size());
    for (unsigned a = 0; a < p->s.size(); a++)
      sv[a] = p->s[a];
    oomph::Vector<double> xs;
    p->elem->tracer_geometry_at_s(sv, cfg.nlevel, cfg.w, cfg.dwdtau, &xs, nullptr, nullptr);
    for (unsigned i = 0; i < nodal_dim; i++)
      p->x[i] = xs[i];
    p->timefrac = 1.0;
    p->completed_step_time = cfg.t_current;
    return true;
  }

  bool TracerCollection::adopt(TracerParticle *p, unsigned depth)
  {
    // Bounded because a particle sitting exactly on a shared interface could otherwise be passed
    // back and forth between two domains without its timefrac advancing.
    if (depth > 8 || !mesh)
      return false;
    resolve_code_indices();
    unsigned nlevel_available = 0;
    while (nlevel_available < 3 && code_index[nlevel_available] >= 0)
      nlevel_available++;
    if (!nlevel_available)
      return false;
    p->elem = nullptr;
    p->s.assign(elem_dim, 0.0);
    p->payload.resize(n_payload, 0.0);
    if (!place_globally(p, 0))
      return false;
    if (p->elem->is_halo())
      return false; // let the owning process take it, on the next migration round

    TracerTimeConfig cfg = TracerTimeConfig::from_mesh(mesh, time_interpolation_order, nlevel_available);
    if (cfg.dt <= 0.0 || advect_one(p, cfg, depth))
    {
      if (!transferred_away)
        tracers.push_back(p);
      return true;
    }
    // It left this domain too, and try_transfer inside advect_one already had its chance.
    return false;
  }

  bool TracerCollection::advect_one(TracerParticle *p, TracerTimeConfig &cfg, unsigned depth)
  {
    const unsigned nd = nodal_dim;
    const bool interface = (get_codimension() == 1);

    double y[3] = {0.0, 0.0, 0.0};
    for (unsigned i = 0; i < nd; i++)
      y[i] = p->x[i];
    std::vector<double> pay(p->payload);

    double tau = p->timefrac;
    double h = (fixed_substeps > 0) ? (1.0 / fixed_substeps) : p->next_h;
    h = std::min(h, 1.0 - tau);

    double k1[3], k2[3], k3[3], k4[3];
    std::vector<double> pk1(n_payload), pk2(n_payload), pk3(n_payload), pk4(n_payload);
    double ytmp[3];
    std::vector<double> ptmp(n_payload);

    bool have_k1 = false;
    unsigned long guard = 0;
    // Progress watchdog, see the stall test below.
    unsigned long next_progress_check = PROGRESS_CHECK_INTERVAL;
    double tau_at_last_check = tau;
    transferred_away = false;

    while (tau < 1.0 - 1e-15)
    {
      if (++guard > max_substeps)
        throw_runtime_error("Tracer '" + tracer_name + "' did not finish its timestep within " +
                            std::to_string(max_substeps) + " sub-steps");

      // A particle pressed against the end of its element with nowhere left to walk to makes the
      // controller oscillate rather than collapse: every sub-step large enough to move it is
      // rejected for want of a place to put it, the halved one is accepted, and the controller
      // grows h straight back because the error estimate is tiny. tau then creeps forward at
      // rounding scale - a million sub-steps covering 2e-3 of one timestep was measured at the
      // apex of an evaporating droplet, where the drift pushing the particle off the end of the
      // free surface is itself rounding noise - and the sub-step guard above was the only thing
      // that ended it, as a hard error after a minute of work.
      //
      // h alone cannot detect that: it never gets small, it oscillates. So measure the pathology
      // itself - ground covered per sub-step - which needs no assumption about element sizes or
      // velocities. A particle that legitimately needs the full max_substeps budget still covers
      // a hundred times more than this per check.
      bool stalled = false;
      if (guard >= next_progress_check)
      {
        stalled = (tau - tau_at_last_check < PROGRESS_PER_CHECK);
        tau_at_last_check = tau;
        next_progress_check = guard + PROGRESS_CHECK_INTERVAL;
      }

      if (h < 1e-12 || stalled)
      {
        // The sub-step collapsed, so the particle really is leaving this mesh rather than merely
        // having overshot. Store where it got to before offering it to a neighbouring domain.
        for (unsigned i = 0; i < nd; i++)
          p->x[i] = y[i];
        p->payload = pay;
        p->timefrac = tau;
        if (!transfer_interfaces.empty() && try_transfer(p, depth))
        {
          transferred_away = true;
          return true; // the receiving collection owns it now
        }
        if (interface && pin_to_interface_end(p, cfg))
        {
          stat_pinned++;
          return true;
        }
        return false;
      }
      if (tau + h > 1.0)
        h = 1.0 - tau;

      if (!have_k1)
      {
        if (!eval_derivative(p, cfg, tau, y, k1, pk1.data()))
          return false;
        have_k1 = true;
      }

      for (unsigned i = 0; i < nd; i++)
        ytmp[i] = y[i] + h * BS_A21 * k1[i];
      for (unsigned i = 0; i < n_payload; i++)
        ptmp[i] = pay[i] + h * BS_A21 * pk1[i];
      if (!eval_derivative(p, cfg, tau + BS_A21 * h, ytmp, k2, pk2.data()))
      {
        h *= 0.5;
        stat_rejected++;
        continue;
      }

      for (unsigned i = 0; i < nd; i++)
        ytmp[i] = y[i] + h * BS_A32 * k2[i];
      if (!eval_derivative(p, cfg, tau + BS_A32 * h, ytmp, k3, pk3.data()))
      {
        h *= 0.5;
        stat_rejected++;
        continue;
      }

      double ynew[3] = {0.0, 0.0, 0.0};
      for (unsigned i = 0; i < nd; i++)
        ynew[i] = y[i] + h * (BS_B1 * k1[i] + BS_B2 * k2[i] + BS_B3 * k3[i]);
      std::vector<double> pnew(n_payload);
      for (unsigned i = 0; i < n_payload; i++)
        pnew[i] = pay[i] + h * (BS_B1 * pk1[i] + BS_B2 * pk2[i] + BS_B3 * pk3[i]);

      // FSAL: the derivative at the end of this step is also the first stage of the next one.
      if (!eval_derivative(p, cfg, tau + h, ynew, k4, pk4.data()))
      {
        h *= 0.5;
        stat_rejected++;
        continue;
      }

      double err = 0.0, scale = 0.0;
      if (fixed_substeps <= 0)
      {
        for (unsigned i = 0; i < nd; i++)
        {
          const double e = h * ((BS_B1 - BS_E1) * k1[i] + (BS_B2 - BS_E2) * k2[i] +
                                (BS_B3 - BS_E3) * k3[i] - BS_E4 * k4[i]);
          const double sc = atol + rtol * std::max(std::abs(y[i]), std::abs(ynew[i]));
          err += (e / sc) * (e / sc);
          scale += 1.0;
        }
        err = std::sqrt(err / std::max(1.0, scale));
      }

      if (fixed_substeps <= 0 && err > 1.0)
      {
        h *= std::max(0.2, 0.9 * std::pow(err, -1.0 / 3.0));
        stat_rejected++;
        continue;
      }

      // Accepted.
      for (unsigned i = 0; i < nd; i++)
        y[i] = ynew[i];
      pay = pnew;
      tau += h;
      stat_substeps++;

      if (interface)
      {
        // Re-anchor onto the interface. The unprojected iterate is already O(h^4) off the surface,
        // so this costs no order, and it is what keeps the normal offset at machine zero over
        // thousands of steps instead of letting it random-walk.
        cfg.set_tau(tau);
        double yproj[3];
        if (place_at(p, cfg, y, yproj))
          for (unsigned i = 0; i < nd; i++)
            y[i] = yproj[i];
      }

      for (unsigned i = 0; i < nd; i++)
        k1[i] = k4[i];
      pk1 = pk4;
      have_k1 = !interface; // the re-anchor moved y, so the stored derivative no longer belongs to it

      if (fixed_substeps <= 0)
      {
        const double fac = (err > 0.0) ? std::min(5.0, 0.9 * std::pow(err, -1.0 / 3.0)) : 5.0;
        h *= fac;
      }
      p->timefrac = tau;
    }

    for (unsigned i = 0; i < nd; i++)
      p->x[i] = y[i];
    p->payload = pay;
    // Left at 1, not reset to 0: between MPI migration rounds this is what says the particle has
    // finished its step, and advect_all clears it once before the rounds begin.
    p->timefrac = 1.0;
    p->completed_step_time = cfg.t_current;
    p->next_h = std::min(1.0, std::max(1e-6, h));
    return true;
  }

  void TracerCollection::advect_all()
  {
    if (!mesh)
      throw_runtime_error("Cannot advect tracers before a mesh was set");
    stat_substeps = stat_rejected = stat_walks = stat_global_locates = 0;
    stat_lost = 0;
    stat_migrated = 0;
    stat_transferred = 0;
    stat_wrapped = stat_reinjected = stat_pinned = 0;
    // The solve that produced this step moved the nodes, so any locator built during the previous
    // one describes a configuration that no longer exists.
    mark_geometry_stale();

    unsigned nlevel_available = 0;
    while (nlevel_available < 3 && code_index[nlevel_available] >= 0)
      nlevel_available++;
    if (nlevel_available == 0)
    {
      resolve_code_indices();
      while (nlevel_available < 3 && code_index[nlevel_available] >= 0)
        nlevel_available++;
      if (nlevel_available == 0)
        throw_runtime_error("No generated code found for tracer '" + tracer_name + "'");
    }

    TracerTimeConfig cfg = TracerTimeConfig::from_mesh(mesh, time_interpolation_order, nlevel_available);
    if (cfg.dt <= 0.0)
      return; // stationary: nothing to advect over

    // At tau = 0 the particles sit in the configuration of history level 1 - their positions are
    // from the end of the previous step, which is what level 1 now holds.
    if (generation_changed())
      relocate_all(1);

    for (auto *p : tracers)
      if (p->completed_step_time < cfg.t_current)
        p->timefrac = 0.0;

    // Advect, then hand over anything that ended in somebody else's halo and let its new owner
    // finish it. Rounds rather than a single pass because a receiving process may itself find the
    // particle in a halo of a third one; in practice one round settles it, since the halo layer is
    // one element deep and the exchange happens at the end of the step.
    //
    // Collective from here on: every process must run the same number of rounds, including
    // processes holding no particles at all - hence the Allreduce inside exchange_migrants(),
    // which is what makes the loop condition agree everywhere.
    for (unsigned round = 0; round < max_migration_rounds; round++)
    {
      std::vector<TracerParticle *> survivors;
      survivors.reserve(tracers.size());
      for (auto *p : tracers)
      {
        bool ok = (p->elem != nullptr);
        transferred_away = false;
        if (ok && p->completed_step_time >= cfg.t_current)
        {
          // Handed to this collection by another domain part-way through this very step, and
          // already finished there. Advecting it again would double its displacement.
          survivors.push_back(p);
          continue;
        }
        if (ok && p->timefrac < 1.0 - 1e-15)
          ok = advect_one(p, cfg);
        // A particle that has run out of the mesh is offered the periodic images before it is given
        // up on - after the transfer interfaces have had their chance inside advect_one, so a
        // neighbouring domain still wins over a wrap. It then finishes the rest of its step from
        // the image, which is what keeps the wrap from costing the step's accuracy.
        bool parked = false;
        for (unsigned wraps = 0; !ok && !transferred_away && wraps < max_periodic_wraps; wraps++)
        {
          const WrapResult w = wrap_position(p);
          if (w == WrapResult::ParkedForReinjection)
          {
            parked = true;
            break;
          }
          if (w != WrapResult::PlacedHere)
            break;
          stat_wrapped++;
          ok = advect_one(p, cfg);
        }
        if (parked)
          continue; // pending_reinject owns it now
        if (!ok)
        {
          stat_lost++;
          retire(p);
          continue;
        }
        if (transferred_away)
        {
          stat_transferred++;
          continue; // another domain's collection owns it now
        }
        survivors.push_back(p);
      }
      tracers.swap(survivors);

      if (!is_distributed())
      {
        exchange_reinjections(); // a no-op serially, beyond draining a leftover
        break;
      }
      stat_migrated += exchange_migrants();
      // Collective, and before the unfinished count below, because a reinjected particle re-enters
      // with its step only part-way done and the loop has to run another round for it.
      exchange_reinjections();
      unsigned unfinished = 0;
      for (auto *p : tracers)
        if (p->timefrac < 1.0 - 1e-15)
          unfinished++;
#ifdef OOMPH_HAS_MPI
      unsigned total = 0;
      MPI_Allreduce(&unfinished, &total, 1, MPI_UNSIGNED, MPI_SUM,
                    mesh->communicator_pt()->mpi_comm());
      unfinished = total;
#endif
      if (!unfinished)
        break;
      if (round + 1 == max_migration_rounds)
      {
        // Never spin silently: an unfinished particle here means it is bouncing between processes,
        // which is a defect rather than something to wait out.
        throw_runtime_error("Tracers '" + tracer_name + "': " + std::to_string(unfinished) +
                            " particle(s) did not finish their timestep within " +
                            std::to_string(max_migration_rounds) + " migration rounds");
      }
    }

    // Positions are now those of history level 0, so a locator built for level 1 is stale even
    // though the topology has not changed.
    const double tnow = mesh->nnode() ? mesh->node_pt(0)->time_stepper_pt()->time_pt()->time(0) : 0.0;
    // Outside the history_window guard on purpose: a window that has just been switched off still
    // has to let the trails it created finish fading rather than stranding them forever.
    prune_dead(tnow);
    if (history_window > 0.0)
    {
      const unsigned stride = 1 + nodal_dim;
      for (auto *p : tracers)
      {
        if (p->hist.size() != (size_t)history_capacity * stride)
        {
          // Re-ring rather than reset: this fires when a restored state meets an equation asking
          // for a different capacity, and simply reassigning the buffer left hist_n and hist_head
          // pointing into it as if the old samples were still there.
          const std::vector<double> old = history_of(p);
          set_history(p, old.data(), (unsigned)(old.size() / stride));
          if (!history_capacity)
            continue;
        }
        p->hist[p->hist_head * stride] = tnow;
        for (unsigned i = 0; i < nodal_dim; i++)
          p->hist[p->hist_head * stride + 1 + i] = p->x[i];
        p->hist_head = (p->hist_head + 1) % history_capacity;
        if (p->hist_n < history_capacity)
          p->hist_n++;
        // Drop samples that have fallen out of [t - history_window, t].
        while (p->hist_n > 1)
        {
          const unsigned oldest = (p->hist_head + history_capacity - p->hist_n) % history_capacity;
          if (tnow - p->hist[oldest * stride] <= history_window)
            break;
          p->hist_n--;
        }
      }
    }
  }

  void TracerCollection::relocate_all(unsigned time_level)
  {
    if (!mesh)
      throw_runtime_error("Cannot locate tracers before a mesh was set");
    // Nothing asks for a relocation unless the mesh changed under the particles, and a state file
    // restores a whole new configuration without touching the topology at all.
    mark_geometry_stale();
    std::vector<TracerParticle *> survivors;
    survivors.reserve(tracers.size());
    for (auto *p : tracers)
    {
      p->elem = nullptr;
      if (place_globally(p, time_level))
        survivors.push_back(p);
      else
      {
        stat_lost++;
        retire(p);
      }
    }
    tracers.swap(survivors);
  }

  std::string TracerCollection::step_statistics() const
  {
    std::ostringstream oss;
    oss << tracers.size() << " tracers, " << stat_substeps << " sub-steps";
    if (stat_rejected)
      oss << ", " << stat_rejected << " rejected";
    if (stat_walks)
      oss << ", " << stat_walks << " element changes";
    if (stat_global_locates)
      oss << ", " << stat_global_locates << " global locates";
    if (stat_migrated)
      oss << ", " << stat_migrated << " migrated";
    if (stat_transferred)
      oss << ", " << stat_transferred << " handed to another domain";
    if (stat_wrapped)
      oss << ", " << stat_wrapped << " wrapped periodically";
    if (stat_reinjected)
      oss << ", " << stat_reinjected << " reinjected from another process";
    if (stat_pinned)
      oss << ", " << stat_pinned << " pinned at an interface end";
    if (!dead.empty())
      oss << ", " << dead.size() << " trails fading";
    if (stat_lost)
      oss << ", " << stat_lost << " LOST";
    return oss.str();
  }

  // ------------------------------------------------------------------------------------------
  // State files
  // ------------------------------------------------------------------------------------------

  // COLLECTIVE. Writes the whole particle set, id-sorted, so the file says nothing about how the
  // mesh was partitioned and can be read back at any process count.
  //
  // With `with_history` the rolling position history goes in as well. It is what the trail plots
  // are drawn from, and leaving it out was not merely a cosmetic loss: a restored state came back
  // with every particle in the right place and no trail at all, and the trails then grew back from
  // scratch rather than continuing. Since the number of samples differs per particle, the counts go
  // into `tagarr` (which becomes three entries per particle) and the samples are appended to
  // `posarr` after the fixed-size blocks, so neither array needs a worst-case stride.
  void TracerCollection::_save_state(std::vector<double> &posarr, std::vector<long long> &tagarr,
                                     bool with_history)
  {
    const std::vector<double> allpos = gather_positions();
    const std::vector<double> allpay = gather_payloads();
    const std::vector<long long> allids = gather_ids();
    const std::vector<int> alltags = gather_tags();
    const unsigned n = (unsigned)allids.size();
    const unsigned stride = nodal_dim + n_payload;
    const unsigned hstride = 1 + nodal_dim;

    // Gathering the histories needs one uniform row length, so ask everybody for the longest one.
    unsigned hmax = 0;
    if (with_history)
      for (auto *p : tracers)
        hmax = std::max(hmax, p->hist_n);
#ifdef OOMPH_HAS_MPI
    if (with_history && is_distributed())
    {
      unsigned reduced = 0;
      MPI_Allreduce(&hmax, &reduced, 1, MPI_UNSIGNED, MPI_MAX,
                    mesh->communicator_pt()->mpi_comm());
      hmax = reduced;
    }
#endif

    std::vector<double> allhist;
    const unsigned hcol = hmax ? 1 + hmax * hstride : 0;
    if (hcol)
    {
      std::vector<double> local((size_t)tracers.size() * hcol, 0.0);
      for (unsigned i = 0; i < tracers.size(); i++)
      {
        const std::vector<double> h = history_of(tracers[i]);
        local[(size_t)i * hcol] = (double)(h.size() / hstride);
        for (unsigned k = 0; k < h.size(); k++)
          local[(size_t)i * hcol + 1 + k] = h[k];
      }
      allhist = gather_rows(local, hcol);
    }

    size_t htotal = 0;
    std::vector<unsigned> hn(n, 0);
    for (unsigned i = 0; i < n && hcol; i++)
    {
      hn[i] = (unsigned)allhist[(size_t)i * hcol];
      htotal += hn[i];
    }

    posarr.assign((size_t)n * stride + htotal * hstride, 0.0);
    tagarr.assign((size_t)n * (with_history ? 3 : 2), 0);
    size_t hoff = (size_t)n * stride;
    for (unsigned i = 0; i < n; i++)
    {
      const unsigned ts = with_history ? 3 : 2;
      tagarr[(size_t)ts * i] = allids[i];
      tagarr[(size_t)ts * i + 1] = alltags[i];
      for (unsigned d = 0; d < nodal_dim; d++)
        posarr[(size_t)i * stride + d] = allpos[(size_t)i * nodal_dim + d];
      for (unsigned d = 0; d < n_payload; d++)
        posarr[(size_t)i * stride + nodal_dim + d] = allpay[(size_t)i * n_payload + d];
      if (!with_history)
        continue;
      tagarr[(size_t)ts * i + 2] = (long long)hn[i];
      for (unsigned k = 0; k < hn[i] * hstride; k++)
        posarr[hoff + k] = allhist[(size_t)i * hcol + 1 + k];
      hoff += (size_t)hn[i] * hstride;
    }
  }

  // COLLECTIVE. The file holds the whole particle set, so every process reads all of it and keeps
  // the ones it owns - which is also what makes a file written at one process count readable at
  // another.
  void TracerCollection::_load_state(const std::vector<double> &posarr,
                                     const std::vector<long long> &tagarr, bool with_history)
  {
    clear();
    next_id = 1;
    // The mesh was restored from the same file a moment ago, so it is in a configuration no
    // locator built during this session knows about.
    mark_geometry_stale();
    const unsigned stride = nodal_dim + n_payload;
    const unsigned hstride = 1 + nodal_dim;
    const unsigned ts = with_history ? 3 : 2;
    const unsigned n = (unsigned)(tagarr.size() / ts);
    std::vector<double> pos((size_t)n * nodal_dim, 0.0), pay((size_t)n * n_payload, 0.0);
    std::vector<int> tags(n, 0);
    std::vector<long long> ids(n, 0);
    std::vector<size_t> hoff(n, 0);
    std::vector<unsigned> hn(n, 0);
    size_t off = (size_t)n * stride;
    for (unsigned i = 0; i < n; i++)
    {
      for (unsigned d = 0; d < nodal_dim; d++)
        pos[(size_t)i * nodal_dim + d] = posarr[(size_t)i * stride + d];
      for (unsigned d = 0; d < n_payload; d++)
        pay[(size_t)i * n_payload + d] = posarr[(size_t)i * stride + nodal_dim + d];
      ids[i] = tagarr[(size_t)ts * i];
      tags[i] = (int)tagarr[(size_t)ts * i + 1];
      if (!with_history)
        continue;
      hn[i] = (unsigned)tagarr[(size_t)ts * i + 2];
      hoff[i] = off;
      off += (size_t)hn[i] * hstride;
    }
    add_tracers_collective(pos, tags, pay, ids);

    if (!with_history)
      return;
    // Only the particles this process ended up owning get their history back, so index them by id.
    std::map<TracerId, unsigned> byid;
    for (unsigned i = 0; i < n; i++)
      byid[(TracerId)ids[i]] = i;
    for (auto *p : tracers)
    {
      auto it = byid.find(p->id);
      if (it == byid.end() || !hn[it->second])
        continue;
      set_history(p, &posarr[hoff[it->second]], hn[it->second]);
    }
  }

}
