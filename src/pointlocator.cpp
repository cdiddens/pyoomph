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
    build_neighbours();
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

    std::map<pyoomph::Node *, unsigned> node_index;
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
        auto found = node_index.find(n);
        unsigned index;
        if (found != node_index.end())
        {
          index = found->second;
        }
        else
        {
          index = nodes_by_index.size();
          node_index[n] = index;
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

  void MeshPointLocator::build_neighbours()
  {
    // TODO: face-neighbour adjacency for the walk. Without it every query starts from the tree,
    // which is correct but costs a search per integration point instead of per element.
    elem_neighbour_offsets.assign(elements_by_index.size() + 1, 0);
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

  bool MeshPointLocator::try_element(BulkElementBase *, const double *, double *, double *) const
  {
    throw_runtime_error("MeshPointLocator::try_element is not implemented yet");
  }

  LocationSet MeshPointLocator::locate_batch(const std::vector<double> &, unsigned,
                                             const std::vector<unsigned> *) const
  {
    throw_runtime_error("MeshPointLocator::locate_batch is not implemented yet");
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

  unsigned LocationSet::values_per_point(const EvalRequest &) const
  {
    throw_runtime_error("LocationSet::values_per_point is not implemented yet");
  }

  std::vector<double> LocationSet::evaluate(const EvalRequest &) const
  {
    throw_runtime_error("LocationSet::evaluate is not implemented yet");
  }

}
