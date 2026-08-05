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

// Implementation of the mesh point locator - see pointlocator.hpp for the API and
// dev_docs/mesh_point_locator.md for the design.
//
// State of this file: the index construction is real; the matching and evaluation paths are still
// being brought over from Mesh::nodal_interpolate_from / MeshKDTree::find_element and throw until
// they are. Nothing calls into here yet, so the old facility is still the one in use; it is removed
// once every call site listed in the header has been migrated.

#include "pointlocator.hpp"

#include "elements.hpp"
#include "exception.hpp"
#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace pyoomph
{

  namespace
  {
    // BulkElementBase::zeta_nodal reads two static flags to decide which coordinate it reports.
    // Setting them has to be scoped, and it has to RESTORE the previous values rather than reset to
    // zero: MeshKDTree resets to zero unconditionally, which silently discards the setting that
    // Mesh::nodal_interpolate_from installed around it. That is harmless today only because the two
    // never nest - which the locator design deliberately makes them do.
    struct ZetaFlagGuard
    {
      unsigned old_time, old_type;
      ZetaFlagGuard(unsigned time_index, bool lagrangian)
          : old_time(BulkElementBase::zeta_time_history), old_type(BulkElementBase::zeta_coordinate_type)
      {
        BulkElementBase::zeta_time_history = time_index;
        BulkElementBase::zeta_coordinate_type = (lagrangian ? 0 : 1);
      }
      ~ZetaFlagGuard()
      {
        BulkElementBase::zeta_time_history = old_time;
        BulkElementBase::zeta_coordinate_type = old_type;
      }
    };
  }

  MeshPointLocator::MeshPointLocator(Mesh *source_mesh, const LocatorSetup &_setup)
      : source(source_mesh), setup(_setup)
  {
    if (!source || !source->nelement())
    {
      throw_runtime_error("Cannot build a point locator on an empty mesh");
    }

    BulkElementBase *e0 = dynamic_cast<BulkElementBase *>(source->element_pt(0));
    if (!e0)
    {
      throw_runtime_error("Cannot build a point locator on a mesh of non-pyoomph elements");
    }
    element_dim = e0->dim();

    switch (setup.space)
    {
    case LocatorSpace::Eulerian:
      space_dim = e0->nodal_dimension();
      break;
    case LocatorSpace::Lagrangian:
      space_dim = e0->nlagrangian();
      break;
    case LocatorSpace::BoundaryZeta:
      // The interface's intrinsic boundary coordinate has one component per element dimension.
      space_dim = element_dim;
      break;
    }

    // Codimension decides how a point is matched at all: an equal-dimension source can be inverted,
    // a codimension-1 source (a curve in 2d, a surface in 3d) is overdetermined and has to be
    // projected instead. This is the whole of the "no unique zeta for a surface in 3d" problem - in
    // Project mode no chart is needed.
    if (space_dim == element_dim)
    {
      mode = LocatorMode::Invert;
    }
    else if (space_dim == element_dim + 1)
    {
      mode = LocatorMode::Project;
    }
    else
    {
      throw_runtime_error("Point locator supports equal-dimension or codimension-1 source meshes only");
    }

    if (!setup.period.empty() && setup.period.size() != space_dim)
    {
      throw_runtime_error("Locator period must have one entry per coordinate component");
    }

    build_index();
    build_element_boxes();
    build_affine_inverses();
  }

  MeshPointLocator::~MeshPointLocator()
  {
    delete tree;
  }

  // Collect the distinct source nodes with their coordinate in the chosen space, and the CSR
  // node->element adjacency the candidate search walks.
  void MeshPointLocator::build_index()
  {
    ZetaFlagGuard guard(setup.time_index, setup.space == LocatorSpace::Lagrangian);

    std::vector<double> coords;
    std::vector<std::vector<BulkElementBase *>> adjacency;

    const unsigned nelem = source->nelement();
    elements_by_index.reserve(nelem);

    for (unsigned ie = 0; ie < nelem; ie++)
    {
      BulkElementBase *e = dynamic_cast<BulkElementBase *>(source->element_pt(ie));
      if (!e)
        continue;
      element_index[e] = elements_by_index.size();
      elements_by_index.push_back(e);

      for (unsigned in = 0; in < e->nnode(); in++)
      {
        pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(e->node_pt(in));
        if (!n)
          continue;
        auto found = node_lookup.find(n);
        unsigned index;
        if (found != node_lookup.end())
        {
          index = found->second;
        }
        else
        {
          index = nodes_by_index.size();
          node_lookup[n] = index;
          nodes_by_index.push_back(n);
          adjacency.resize(index + 1);
          for (unsigned d = 0; d < space_dim; d++)
          {
            coords.push_back(e->zeta_nodal(in, 0, d));
          }
        }
        adjacency[index].push_back(e);
      }
    }

    node_elem_offsets.resize(adjacency.size() + 1, 0);
    for (unsigned i = 0; i < adjacency.size(); i++)
    {
      node_elem_offsets[i + 1] = node_elem_offsets[i] + adjacency[i].size();
    }
    node_elem_entries.reserve(node_elem_offsets.back());
    for (const auto &row : adjacency)
    {
      node_elem_entries.insert(node_elem_entries.end(), row.begin(), row.end());
    }

    tree = new KDTree(coords, space_dim);
  }

  void MeshPointLocator::build_element_boxes()
  {
    ZetaFlagGuard guard(setup.time_index, setup.space == LocatorSpace::Lagrangian);

    const unsigned nelem = elements_by_index.size();
    element_bbox_min.assign(nelem * space_dim, 0.0);
    element_bbox_max.assign(nelem * space_dim, 0.0);

    for (unsigned ie = 0; ie < nelem; ie++)
    {
      BulkElementBase *e = elements_by_index[ie];
      for (unsigned d = 0; d < space_dim; d++)
      {
        double lo = e->zeta_nodal(0, 0, d), hi = lo;
        for (unsigned in = 1; in < e->nnode(); in++)
        {
          const double v = e->zeta_nodal(in, 0, d);
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
        element_bbox_min[ie * space_dim + d] = lo;
        element_bbox_max[ie * space_dim + d] = hi;
      }
    }
  }

  void MeshPointLocator::build_affine_inverses()
  {
    // TODO: straight-sided simplices invert exactly through a precomputed matrix, which removes the
    // Newton solve from the overwhelmingly common case on these meshes. Curved geometry falls back
    // to Newton seeded from the affine guess. Until this is filled in, element_is_affine stays all
    // false and try_element always takes the Newton path - correct, just slower.
    element_is_affine.assign(elements_by_index.size(), false);
  }

  bool MeshPointLocator::bbox_contains(unsigned slot, const double *x) const
  {
    double diag = 0.0;
    for (unsigned d = 0; d < space_dim; d++)
    {
      const double w = element_bbox_max[slot * space_dim + d] - element_bbox_min[slot * space_dim + d];
      diag += w * w;
    }
    const double slack = bbox_slack * sqrt(diag);
    for (unsigned d = 0; d < space_dim; d++)
    {
      if (x[d] < element_bbox_min[slot * space_dim + d] - slack)
        return false;
      if (x[d] > element_bbox_max[slot * space_dim + d] + slack)
        return false;
    }
    return true;
  }

  void MeshPointLocator::wrap_into_period(double *x) const
  {
    if (setup.period.empty())
      return;
    for (unsigned d = 0; d < space_dim; d++)
    {
      const double p = setup.period[d];
      if (p <= 0.0)
        continue;
      x[d] = x[d] - p * std::floor(x[d] / p);
    }
  }

  // Match one query point against one element. The caller has already decided this element is worth
  // trying; this does the bounding-box reject and then the actual inversion.
  bool MeshPointLocator::try_element(BulkElementBase *e, const double *x, double *s_out, double *offset_out) const
  {
    auto found = element_index.find(e);
    if (found == element_index.end())
      return false;
    if (!bbox_contains(found->second, x))
      return false;

    if (mode == LocatorMode::Project)
    {
      // Phase 2. Deliberately not silently degraded to Invert: on a codimension-1 source the Newton
      // system is not square and what comes back would be meaningless rather than merely inaccurate.
      throw_runtime_error("Closest-point projection onto a codimension-1 mesh is not implemented yet (phase 2)");
    }

    oomph::Vector<double> zeta(space_dim);
    for (unsigned d = 0; d < space_dim; d++)
      zeta[d] = x[d];

    oomph::Vector<double> s(element_dim);
    oomph::GeomObject *go = NULL;

    // Two-stage inversion. oomph's locate_zeta, when not given an initial guess, lays out a grid of
    // plot points over the element and runs a Newton solve from each (elements.cc:4794) - for a 3d
    // element that is on the order of a hundred Newton solves for one query, and it is what the old
    // path paid every time. Nearly always a single solve from the element centre converges, so try
    // that first and keep the multi-start only as the fallback, which preserves the old behaviour
    // exactly for the awkward elements that need it.
    for (unsigned d = 0; d < element_dim; d++)
      s[d] = 0.5 * (e->s_min() + e->s_max());
    e->locate_zeta(zeta, go, s, true);

    if (!go)
    {
      for (unsigned d = 0; d < element_dim; d++)
        s[d] = 0.5 * (e->s_min() + e->s_max());
      e->locate_zeta(zeta, go, s, false);
    }

    if (!go)
      return false;

    for (unsigned d = 0; d < element_dim; d++)
      s_out[d] = s[d];
    if (offset_out)
      *offset_out = 0.0; // exact by construction in Invert mode
    return true;
  }

  LocationSet MeshPointLocator::locate_batch(const std::vector<double> &coords, unsigned npoint,
                                             const std::vector<unsigned> *hint_groups) const
  {
    if (coords.size() < (size_t)npoint * space_dim)
    {
      throw_runtime_error("locate_batch was given fewer coordinates than npoint*space_dim");
    }
    if (hint_groups && hint_groups->size() < npoint)
    {
      throw_runtime_error("locate_batch was given fewer hint_groups than points");
    }

    ZetaFlagGuard guard(setup.time_index, setup.space == LocatorSpace::Lagrangian);

    LocationSet out;
    out.locator = this;
    out.npoint = npoint;
    out.handles.assign(npoint, LocationHandle());
    out.local_s_stride = element_dim;
    out.local_elements.reserve(npoint);
    out.local_s.reserve((size_t)npoint * element_dim);
    out.local_offset.reserve(npoint);

    std::vector<double> query(space_dim, 0.0);
    std::vector<double> s(element_dim, 0.0);
    double offset = 0.0;

    // Reused across points so the per-query cost has no allocation in it.
    std::vector<uint32_t> knn;
    std::vector<BulkElementBase *> tried;

    BulkElementBase *previous = NULL;
    unsigned previous_group = (unsigned)-1;

    for (unsigned i = 0; i < npoint; i++)
    {
      for (unsigned d = 0; d < space_dim; d++)
        query[d] = coords[(size_t)i * space_dim + d];
      wrap_into_period(&query[0]);

      tried.clear();
      BulkElementBase *hit = NULL;
      int how = -1;

      // (a) The previous match and everything sharing a node with it. Consecutive queries are
      // usually neighbours - all the integration points of one target element, for instance - so
      // this is the case that should dominate, and it never touches the tree.
      const bool same_group = (hint_groups == nullptr) || (previous_group == (*hint_groups)[i]);
      if (previous && same_group)
      {
        if (try_element(previous, &query[0], &s[0], &offset))
        {
          hit = previous;
          how = 0;
        }
        else
        {
          tried.push_back(previous);
          for (unsigned in = 0; in < previous->nnode() && !hit; in++)
          {
            pyoomph::Node *n = dynamic_cast<pyoomph::Node *>(previous->node_pt(in));
            if (!n)
              continue;
            auto ni = node_lookup.find(n);
            if (ni == node_lookup.end())
              continue;
            for (unsigned k = node_elem_offsets[ni->second]; k < node_elem_offsets[ni->second + 1]; k++)
            {
              BulkElementBase *cand = node_elem_entries[k];
              if (std::find(tried.begin(), tried.end(), cand) != tried.end())
                continue;
              tried.push_back(cand);
              if (try_element(cand, &query[0], &s[0], &offset))
              {
                hit = cand;
                how = 0;
                break;
              }
            }
          }
        }
      }

      // (b) The nearest source node, and the elements around it.
      if (!hit)
      {
        int ni = tree->nearest_point(query[0], space_dim > 1 ? query[1] : 0.0, space_dim > 2 ? query[2] : 0.0);
        if (ni >= 0)
        {
          for (unsigned k = node_elem_offsets[ni]; k < node_elem_offsets[ni + 1]; k++)
          {
            BulkElementBase *cand = node_elem_entries[k];
            if (std::find(tried.begin(), tried.end(), cand) != tried.end())
              continue;
            tried.push_back(cand);
            if (try_element(cand, &query[0], &s[0], &offset))
            {
              hit = cand;
              how = 1;
              break;
            }
          }
        }
      }

      // (c) Widen by k-nearest nodes, doubling k. A point on the far side of a curved boundary, or
      // just outside the mesh, is genuinely not in any element and has to exhaust the search - so
      // the widening stops at a bounded multiple rather than sweeping the whole mesh, which is what
      // MeshKDTree's mesh-wide radius did.
      if (!hit)
      {
        const unsigned max_k = std::min<unsigned>(256, (unsigned)nodes_by_index.size());
        for (unsigned k = 8; !hit && k <= max_k; k *= 2)
        {
          tree->k_nearest(k, query[0], space_dim > 1 ? query[1] : 0.0, space_dim > 2 ? query[2] : 0.0, knn);
          for (uint32_t nidx : knn)
          {
            for (unsigned kk = node_elem_offsets[nidx]; kk < node_elem_offsets[nidx + 1]; kk++)
            {
              BulkElementBase *cand = node_elem_entries[kk];
              if (std::find(tried.begin(), tried.end(), cand) != tried.end())
                continue;
              tried.push_back(cand);
              if (try_element(cand, &query[0], &s[0], &offset))
              {
                hit = cand;
                how = 2;
                break;
              }
            }
            if (hit)
              break;
          }
        }
      }

      if (hit)
      {
        out.handles[i].owner = 0; // serial: everything is local
        out.handles[i].slot = out.local_elements.size();
        out.local_elements.push_back(hit);
        for (unsigned d = 0; d < element_dim; d++)
          out.local_s.push_back(s[d]);
        out.local_offset.push_back(offset);
        if (how == 0) out.n_by_walk++;
        else if (how == 1) out.n_by_nearest_node++;
        else out.n_by_widening++;
        previous = hit;
        if (hint_groups)
          previous_group = (*hint_groups)[i];
      }
      // An unlocated point keeps owner = -1; the caller decides what that means.
    }

    return out;
  }

  std::vector<double> MeshPointLocator::locate_and_evaluate(const std::vector<double> &coords, unsigned npoint,
                                                            const EvalRequest &what) const
  {
    return this->locate_batch(coords, npoint).evaluate(what);
  }

  unsigned LocationSet::n_unlocated() const
  {
    unsigned n = 0;
    for (const auto &h : handles)
    {
      if (!h.is_located())
        n++;
    }
    return n;
  }

  double LocationSet::max_projection_offset() const
  {
    double m = 0.0;
    for (double o : local_offset)
    {
      m = std::max(m, o);
    }
    return m;
  }

  bool LocationSet::resolve_local(unsigned i, BulkElementBase *&element, std::vector<double> &s) const
  {
    if (i >= npoint)
      throw_runtime_error("resolve_local called with an out-of-range query index");
    const LocationHandle &h = handles[i];
    if (!h.is_located() || h.owner != 0)
      return false; // unlocated, or (once distributed) owned by another rank
    element = local_elements[h.slot];
    s.resize(local_s_stride);
    for (unsigned d = 0; d < local_s_stride; d++)
      s[d] = local_s[(size_t)h.slot * local_s_stride + d];
    return true;
  }

  std::string LocationSet::search_statistics() const
  {
    std::ostringstream oss;
    oss << n_by_walk << " walked, " << n_by_nearest_node << " by nearest node, "
        << n_by_widening << " widened, " << n_unlocated() << " unlocated";
    return oss.str();
  }

  unsigned LocationSet::values_per_point(const EvalRequest &) const
  {
    throw_runtime_error("LocationSet::values_per_point is not implemented yet");
  }

  std::vector<double> LocationSet::evaluate(const EvalRequest &) const
  {
    throw_runtime_error("LocationSet::evaluate is not implemented yet");
  }

}
