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
#include "wedges_and_pyramids.hpp"

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

  double MeshPointLocator::nodal_coordinate(BulkElementBase *e, unsigned n, unsigned d) const
  {
    switch (setup.space)
    {
    case LocatorSpace::Eulerian:
      return e->node_pt(n)->x(setup.time_index, d);
    case LocatorSpace::Lagrangian:
      return dynamic_cast<pyoomph::Node *>(e->node_pt(n))->xi(d);
    case LocatorSpace::BoundaryZeta:
    default:
      return e->zeta_nodal(n, 0, d);
    }
  }

  double MeshPointLocator::unwrap(double v, double ref, unsigned d) const
  {
    if (setup.period.empty())
      return v;
    const double P = setup.period[d];
    if (P <= 0.0)
      return v;
    return v - P * std::round((v - ref) / P);
  }

  double MeshPointLocator::element_nodal_coordinate(BulkElementBase *e, unsigned n, unsigned d) const
  {
    const double v = nodal_coordinate(e, n, d);
    if (setup.period.empty() || n == 0)
      return v;
    return unwrap(v, nodal_coordinate(e, 0, d), d);
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
      space_dim = e0->node_pt(0)->ndim();
      break;
    case LocatorSpace::Lagrangian:
      // The Lagrangian coordinates carried by the NODES, which on an interface mesh is the bulk
      // nodal dimension rather than the face element's own dimension.
      space_dim = dynamic_cast<pyoomph::Node *>(e0->node_pt(0))->nlagrangian();
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
            coords.push_back(nodal_coordinate(e, in, d));
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
        double lo = element_nodal_coordinate(e, 0, d), hi = lo;
        for (unsigned in = 1; in < e->nnode(); in++)
        {
          const double v = element_nodal_coordinate(e, in, d);
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
        element_bbox_min[ie * space_dim + d] = lo;
        element_bbox_max[ie * space_dim + d] = hi;
      }
    }
  }

  // Invert an n x n matrix in place (n <= 3). Returns false if it is singular, which for an element
  // geometry means degenerate or inverted.
  static bool small_invert(double *m, unsigned n)
  {
    if (n == 1)
    {
      if (std::abs(m[0]) < 1e-300)
        return false;
      m[0] = 1.0 / m[0];
      return true;
    }
    if (n == 2)
    {
      const double det = m[0] * m[3] - m[1] * m[2];
      if (std::abs(det) < 1e-300)
        return false;
      const double a = m[0], b = m[1], c = m[2], d = m[3];
      m[0] = d / det;
      m[1] = -b / det;
      m[2] = -c / det;
      m[3] = a / det;
      return true;
    }
    if (n == 3)
    {
      const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
      const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
      const double det = a * A + b * B + c * C;
      if (std::abs(det) < 1e-300)
        return false;
      m[0] = A / det;
      m[1] = (c * h - b * i) / det;
      m[2] = (b * f - c * e) / det;
      m[3] = B / det;
      m[4] = (a * i - c * g) / det;
      m[5] = (c * d - a * f) / det;
      m[6] = C / det;
      m[7] = (b * g - a * h) / det;
      m[8] = (a * e - b * d) / det;
      return true;
    }
    return false;
  }

  // A straight-sided simplex has an affine geometric map, so inverting it is a matrix multiply
  // giving barycentric coordinates - no iteration. That matters because oomph's locate_zeta reaches
  // its answer by Newton, and on these predominantly triangular/tetrahedral meshes nearly every
  // element qualifies.
  //
  // "Straight-sided" has to be checked, not assumed: a C2 triangle whose mid-edge nodes have been
  // moved onto a curved boundary by the macro-element mapping is a simplex with a non-affine map,
  // and would be inverted wrongly. Every node is therefore tested against the affine prediction,
  // and any element that fails keeps the Newton path.
  // Build the best affine fit x(s) ~ X0 + D u for one element, whether or not the map really is
  // affine. The origin is node 0 and the other d basis nodes are chosen greedily for the largest
  // remaining pivot, so the basis is well conditioned rather than merely non-singular. Returns
  // false only if no d+1 nodes have affinely independent local coordinates, which would mean a
  // degenerate element.
  bool MeshPointLocator::build_affine_fit_for(BulkElementBase *e, unsigned slot)
  {
    const unsigned dd = element_dim * element_dim;
    oomph::Vector<double> s(element_dim, 0.0), s0(element_dim, 0.0);
    e->local_coordinate_of_node(0, s0);

    std::vector<double> Sdiff(dd, 0.0), D((size_t)space_dim * element_dim, 0.0);
    std::vector<unsigned> chosen;

    // Greedy pivoting on the local-coordinate offsets: at each step take the node whose offset has
    // the largest component orthogonal to what is already chosen. For d <= 3 this is cheap enough
    // to do by straight Gram-Schmidt.
    std::vector<std::vector<double>> basis; // orthonormalised local offsets
    for (unsigned round = 0; round < element_dim; round++)
    {
      double best = 0.0;
      unsigned best_node = 0;
      std::vector<double> best_res;
      for (unsigned in = 1; in < e->nnode(); in++)
      {
        if (std::find(chosen.begin(), chosen.end(), in) != chosen.end())
          continue;
        e->local_coordinate_of_node(in, s);
        std::vector<double> v(element_dim);
        for (unsigned d = 0; d < element_dim; d++)
          v[d] = s[d] - s0[d];
        for (const auto &b : basis)
        {
          double dot = 0.0;
          for (unsigned d = 0; d < element_dim; d++)
            dot += v[d] * b[d];
          for (unsigned d = 0; d < element_dim; d++)
            v[d] -= dot * b[d];
        }
        double nrm = 0.0;
        for (unsigned d = 0; d < element_dim; d++)
          nrm += v[d] * v[d];
        nrm = std::sqrt(nrm);
        if (nrm > best)
        {
          best = nrm;
          best_node = in;
          best_res = v;
        }
      }
      if (best < 1e-12)
        return false; // no independent direction left
      for (unsigned d = 0; d < element_dim; d++)
        best_res[d] /= best;
      basis.push_back(best_res);
      chosen.push_back(best_node);
    }

    for (unsigned d = 0; d < space_dim; d++)
      affine_origin[(size_t)slot * space_dim + d] = element_nodal_coordinate(e, 0, d);
    for (unsigned d = 0; d < element_dim; d++)
      affine_s0[(size_t)slot * element_dim + d] = s0[d];

    // D is space_dim x element_dim - square when inverting, tall when projecting.
    D.assign((size_t)space_dim * element_dim, 0.0);
    for (unsigned k = 0; k < element_dim; k++)
    {
      e->local_coordinate_of_node(chosen[k], s);
      for (unsigned d = 0; d < element_dim; d++)
        Sdiff[d * element_dim + k] = s[d] - s0[d];
      for (unsigned d = 0; d < space_dim; d++)
        D[d * element_dim + k] = element_nodal_coordinate(e, chosen[k], d) - affine_origin[(size_t)slot * space_dim + d];
    }

    // Pseudo-inverse via the normal equations: (D^T D)^-1 D^T. For a square D this is exactly D^-1,
    // so the same code covers both modes; for a tall D it is the least-squares solve, which for an
    // affine element IS the exact closest-point projection rather than an approximation to it.
    std::vector<double> DtD((size_t)element_dim * element_dim, 0.0);
    for (unsigned a = 0; a < element_dim; a++)
      for (unsigned b = 0; b < element_dim; b++)
      {
        double v = 0.0;
        for (unsigned d = 0; d < space_dim; d++)
          v += D[d * element_dim + a] * D[d * element_dim + b];
        DtD[a * element_dim + b] = v;
      }
    if (!small_invert(&DtD[0], element_dim))
      return false; // degenerate, inverted, or collapsed element geometry

    for (unsigned a = 0; a < element_dim; a++)
      for (unsigned d = 0; d < space_dim; d++)
      {
        double v = 0.0;
        for (unsigned b = 0; b < element_dim; b++)
          v += DtD[a * element_dim + b] * D[d * element_dim + b];
        affine_inverse[(size_t)slot * element_dim * space_dim + a * space_dim + d] = v;
      }

    for (unsigned d = 0; d < (size_t)space_dim * element_dim; d++)
      affine_basis[(size_t)slot * space_dim * element_dim + d] = D[d];
    for (unsigned d = 0; d < dd; d++)
      affine_sdiff[(size_t)slot * dd + d] = Sdiff[d];
    has_affine_fit[slot] = true;
    return true;
  }

  double MeshPointLocator::element_size(unsigned slot) const
  {
    double h = 0.0;
    for (unsigned d = 0; d < space_dim; d++)
      h = std::max(h, element_bbox_max[(size_t)slot * space_dim + d] - element_bbox_min[(size_t)slot * space_dim + d]);
    return h;
  }

  // Is the fit exact? Every node - not just the ones spanning the basis - must sit where the affine
  // map puts it. This is what makes the same test work for every element family: a T6 with mid-edge
  // nodes at the midpoints passes (its quadratic terms cancel), a T6 bent onto a curved boundary
  // fails, and a quad passes exactly when it is a parallelogram, because the vertex the basis did
  // not use is itself one of the nodes being checked.
  bool MeshPointLocator::is_exactly_affine(BulkElementBase *e, unsigned slot) const
  {
    const unsigned dd = element_dim * element_dim;
    const double *D = &affine_basis[(size_t)slot * space_dim * element_dim];
    const double *X0 = &affine_origin[(size_t)slot * space_dim];
    const double *s0 = &affine_s0[(size_t)slot * element_dim];
    const double *Sdiff = &affine_sdiff[(size_t)slot * dd];

    const double scale = element_size(slot);
    if (scale <= 0.0)
      return false;

    // Sdiff maps u to s - s0; predicting a node's position from its local coordinate needs the
    // other direction.
    std::vector<double> Sinv(Sdiff, Sdiff + dd);
    if (!small_invert(&Sinv[0], element_dim))
      return false;

    oomph::Vector<double> s(element_dim, 0.0);
    for (unsigned in = 0; in < e->nnode(); in++)
    {
      e->local_coordinate_of_node(in, s);
      double u[3] = {0.0, 0.0, 0.0};
      for (unsigned k = 0; k < element_dim; k++)
        for (unsigned d = 0; d < element_dim; d++)
          u[k] += Sinv[k * element_dim + d] * (s[d] - s0[d]);
      // Compared in POSITION rather than in u, because for a codimension-1 element u carries no
      // information about the component perpendicular to the element - a curved surface would look
      // affine in u and is not.
      for (unsigned d = 0; d < space_dim; d++)
      {
        double predicted = X0[d];
        for (unsigned k = 0; k < element_dim; k++)
          predicted += D[d * element_dim + k] * u[k];
        if (std::abs(predicted - element_nodal_coordinate(e, in, d)) > 1e-10 * scale)
          return false;
      }
    }
    return true;
  }

  // A straight-edged 2d quad is exactly bilinear: x(s,t) = b0 + b1 s + b2 t + b3 s t, with
  // b3 != 0 unless it is a parallelogram. That still admits a closed-form inverse (see
  // try_element), so a general quadrilateral need not fall back to Newton either.
  bool MeshPointLocator::build_bilinear_for(BulkElementBase *e, unsigned slot)
  {
    if (element_dim != 2 || space_dim != 2)
      return false;
    if (e->nvertex_node() != 4)
      return false;

    // x(s,t) = sum_i X_i (1+s_i s)(1+t_i t)/4 requires the corners to sit at s,t = +-1, which is
    // oomph's Q convention; check rather than assume.
    double b[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    oomph::Vector<double> s(2, 0.0);
    for (unsigned k = 0; k < 4; k++)
    {
      const unsigned vk = e->get_node_number(e->vertex_node_pt(k));
      e->local_coordinate_of_node(vk, s);
      if (std::abs(std::abs(s[0]) - 1.0) > 1e-12 || std::abs(std::abs(s[1]) - 1.0) > 1e-12)
        return false;
      const double w[4] = {1.0, s[0], s[1], s[0] * s[1]};
      for (unsigned c = 0; c < 4; c++)
        for (unsigned d = 0; d < 2; d++)
          b[c][d] += 0.25 * w[c] * element_nodal_coordinate(e, vk, d);
    }

    double scale = 0.0;
    for (unsigned d = 0; d < space_dim; d++)
      scale = std::max(scale, element_bbox_max[(size_t)slot * space_dim + d] - element_bbox_min[(size_t)slot * space_dim + d]);
    if (scale <= 0.0)
      return false;

    // Exactness: every node, including the mid-edge and centre nodes of a Q9, must sit where the
    // bilinear map puts it. A Q9 bent onto a curved boundary fails here and keeps Newton.
    for (unsigned in = 0; in < e->nnode(); in++)
    {
      e->local_coordinate_of_node(in, s);
      for (unsigned d = 0; d < 2; d++)
      {
        const double predicted = b[0][d] + b[1][d] * s[0] + b[2][d] * s[1] + b[3][d] * s[0] * s[1];
        if (std::abs(predicted - element_nodal_coordinate(e, in, d)) > 1e-10 * scale)
          return false;
      }
    }

    for (unsigned c = 0; c < 4; c++)
      for (unsigned d = 0; d < 2; d++)
        bilinear_coeffs[(size_t)slot * 8 + c * 2 + d] = b[c][d];
    return true;
  }

  // Refine a local coordinate until the geometric residual is at machine precision.
  //
  // oomph's locate_zeta stops at Locate_zeta_helpers::Newton_tolerance = 1e-7 on |x(s) - zeta|
  // (elements.cc:1654), so the s it returns is only good to about that. That was invisible while
  // there was one code path, but it makes the answer depend on the starting point: seeding Newton
  // from the affine fit instead of the element centre changed interpolated values by ~1e-8 on
  // curved elements. Polishing to convergence removes both the seed dependence and the sloppiness,
  // so the result is reproducible AND more accurate than before.
  //
  // Only needed on the Newton path; the affine and bilinear inverses are exact by construction.
  double MeshPointLocator::polish_local_coordinate(BulkElementBase *e, const double *x, double *s) const
  {
    const unsigned nnode = e->nnode();
    oomph::Shape psi(nnode);
    oomph::DShape dpsids(nnode, element_dim);
    oomph::Vector<double> sv(element_dim, 0.0);

    double scale = 0.0;
    for (unsigned d = 0; d < space_dim; d++)
      scale = std::max(scale, std::abs(x[d]));
    if (scale <= 0.0)
      scale = 1.0;

    // Residual of the geometric map at a given local coordinate, and its Jacobian.
    auto residual_at = [&](const double *st, double *res, std::vector<double> *J) -> double {
      for (unsigned d = 0; d < element_dim; d++)
        sv[d] = st[d];
      if (J)
      {
        e->dshape_local(sv, psi, dpsids);
        std::fill(J->begin(), J->end(), 0.0);
      }
      else
      {
        e->shape(sv, psi);
      }
      double rnorm = 0.0;
      for (unsigned d = 0; d < element_dim; d++)
      {
        double xd = 0.0;
        for (unsigned n = 0; n < nnode; n++)
        {
          const double zn = element_nodal_coordinate(e, n, d);
          xd += psi(n) * zn;
          if (J)
            for (unsigned k = 0; k < element_dim; k++)
              (*J)[d * element_dim + k] += dpsids(n, k) * zn;
        }
        res[d] = x[d] - xd;
        rnorm = std::max(rnorm, std::abs(res[d]));
      }
      return rnorm;
    };

    // Damped Newton. An undamped step is fine on a mildly distorted element and is what runs in
    // practice, but on a strongly deformed second-order element - where the map may be far from
    // affine and the Jacobian poorly conditioned - a full step can overshoot and diverge. Halving
    // the step until the residual actually decreases costs nothing when it is not needed and keeps
    // the iteration from running away when it is. It is still not a global convergence guarantee:
    // no such guarantee exists for inverting a curved isoparametric map, which is why the callers
    // treat a large returned residual as "not this element" rather than as an answer.
    std::vector<double> J(element_dim * element_dim, 0.0);
    double res[3] = {0.0, 0.0, 0.0}, trial_res[3] = {0.0, 0.0, 0.0};
    double rnorm = residual_at(s, res, &J);

    for (unsigned iter = 0; iter < 20; iter++)
    {
      if (rnorm < 1e-15 * scale)
        break;
      std::vector<double> Jinv = J;
      if (!small_invert(&Jinv[0], element_dim))
        break; // singular Jacobian here; leave s where it is and let the caller judge the residual

      double step[3] = {0.0, 0.0, 0.0};
      for (unsigned d = 0; d < element_dim; d++)
        for (unsigned k = 0; k < element_dim; k++)
          step[d] += Jinv[d * element_dim + k] * res[k];

      bool improved = false;
      double lambda = 1.0;
      for (unsigned ls = 0; ls < 10; ls++)
      {
        double trial[3] = {0.0, 0.0, 0.0};
        for (unsigned d = 0; d < element_dim; d++)
          trial[d] = s[d] + lambda * step[d];
        const double tnorm = residual_at(trial, trial_res, nullptr);
        if (tnorm < rnorm)
        {
          for (unsigned d = 0; d < element_dim; d++)
            s[d] = trial[d];
          rnorm = residual_at(s, res, &J);
          improved = true;
          break;
        }
        lambda *= 0.5;
      }
      if (!improved)
        break; // no descent direction found: this is as close as this element gets
    }
    return rnorm / scale;
  }

  bool MeshPointLocator::inside_reference_domain(unsigned slot, BulkElementBase *e, const double *s) const
  {
    const double tol = setup.inside_tolerance;
    if (element_ref_domain[slot] == RefDomain::Simplex)
    {
      double sum = 0.0;
      for (unsigned d = 0; d < element_dim; d++)
      {
        if (s[d] < -tol)
          return false;
        sum += s[d];
      }
      return sum <= 1.0 + tol;
    }
    if (element_ref_domain[slot] == RefDomain::Box)
    {
      for (unsigned d = 0; d < element_dim; d++)
        if (s[d] < e->s_min() - tol || s[d] > e->s_max() + tol)
          return false;
      return true;
    }
    // Both of these are documented on the element classes in wedges_and_pyramids.hpp: the wedge is
    // a triangular prism, and the pyramid's cross-section shrinks with s2 ("s[0] and s[1] run from
    // 0 to 1-s[2]").
    if (element_ref_domain[slot] == RefDomain::Prism)
    {
      if (s[0] < -tol || s[1] < -tol || s[0] + s[1] > 1.0 + tol)
        return false;
      return s[2] >= e->s_min() - tol && s[2] <= e->s_max() + tol;
    }
    if (element_ref_domain[slot] == RefDomain::Pyramid)
    {
      if (s[2] < e->s_min() - tol || s[2] > e->s_max() + tol)
        return false;
      const double lim = 1.0 - s[2];
      return s[0] >= -tol && s[0] <= lim + tol && s[1] >= -tol && s[1] <= lim + tol;
    }
    return false;
  }

  void MeshPointLocator::clamp_to_reference_domain(unsigned slot, BulkElementBase *e, double *s) const
  {
    switch (element_ref_domain[slot])
    {
    case RefDomain::Simplex:
    {
      // Clamp to {s_i >= 0, sum s_i <= 1}. Not the exact Euclidean projection onto the simplex, but
      // it lands inside it, which is all the iteration needs from a step limiter.
      for (unsigned d = 0; d < element_dim; d++)
        s[d] = std::max(0.0, s[d]);
      double sum = 0.0;
      for (unsigned d = 0; d < element_dim; d++)
        sum += s[d];
      if (sum > 1.0 && sum > 0.0)
        for (unsigned d = 0; d < element_dim; d++)
          s[d] /= sum;
      break;
    }
    case RefDomain::Box:
      for (unsigned d = 0; d < element_dim; d++)
        s[d] = std::min(std::max(s[d], e->s_min()), e->s_max());
      break;
    case RefDomain::Prism:
      s[2] = std::min(std::max(s[2], e->s_min()), e->s_max());
      s[0] = std::max(0.0, s[0]);
      s[1] = std::max(0.0, s[1]);
      if (s[0] + s[1] > 1.0)
      {
        const double sum = s[0] + s[1];
        s[0] /= sum;
        s[1] /= sum;
      }
      break;
    case RefDomain::Pyramid:
    {
      s[2] = std::min(std::max(s[2], e->s_min()), e->s_max());
      const double lim = 1.0 - s[2];
      s[0] = std::min(std::max(s[0], 0.0), std::max(lim, 0.0));
      s[1] = std::min(std::max(s[1], 0.0), std::max(lim, 0.0));
      break;
    }
    default:
      break;
    }
  }

  // Closest point on a codimension-1 element: minimise |x(s) - x| over s, by damped Gauss-Newton on
  // the normal equations (J^T J) ds = J^T r. Every iterate is clamped back into the reference
  // domain, so when the unconstrained minimiser lies outside the element the iteration settles on
  // the closest point of its boundary instead of wandering off it - which is what makes the answer
  // meaningful for a query beside the interface rather than on it.
  //
  // Returns the achieved perpendicular offset in coordinate units, so the caller can both reject a
  // match that is too far away and rank competing elements by it.
  double MeshPointLocator::project_local_coordinate(BulkElementBase *e, unsigned slot, const double *x, double *s) const
  {
    const unsigned nnode = e->nnode();
    oomph::Shape psi(nnode);
    oomph::DShape dpsids(nnode, element_dim);
    oomph::Vector<double> sv(element_dim, 0.0);

    std::vector<double> J((size_t)space_dim * element_dim, 0.0);
    double r[3] = {0.0, 0.0, 0.0}, rtrial[3] = {0.0, 0.0, 0.0};

    auto residual_at = [&](const double *st, double *res, bool want_jacobian) -> double {
      for (unsigned d = 0; d < element_dim; d++)
        sv[d] = st[d];
      if (want_jacobian)
      {
        e->dshape_local(sv, psi, dpsids);
        std::fill(J.begin(), J.end(), 0.0);
      }
      else
      {
        e->shape(sv, psi);
      }
      double n2 = 0.0;
      for (unsigned d = 0; d < space_dim; d++)
      {
        double xd = 0.0;
        for (unsigned n = 0; n < nnode; n++)
        {
          const double zn = element_nodal_coordinate(e, n, d);
          xd += psi(n) * zn;
          if (want_jacobian)
            for (unsigned k = 0; k < element_dim; k++)
              J[d * element_dim + k] += dpsids(n, k) * zn;
        }
        res[d] = x[d] - xd;
        n2 += res[d] * res[d];
      }
      return std::sqrt(n2);
    };

    clamp_to_reference_domain(slot, e, s);
    double rnorm = residual_at(s, r, true);

    for (unsigned iter = 0; iter < 20; iter++)
    {
      // Normal equations. J^T J is element_dim x element_dim and is invertible whenever the element
      // is not degenerate at s.
      std::vector<double> JtJ((size_t)element_dim * element_dim, 0.0);
      double Jtr[3] = {0.0, 0.0, 0.0};
      for (unsigned a = 0; a < element_dim; a++)
      {
        for (unsigned b = 0; b < element_dim; b++)
        {
          double v = 0.0;
          for (unsigned d = 0; d < space_dim; d++)
            v += J[d * element_dim + a] * J[d * element_dim + b];
          JtJ[a * element_dim + b] = v;
        }
        double g = 0.0;
        for (unsigned d = 0; d < space_dim; d++)
          g += J[d * element_dim + a] * r[d];
        Jtr[a] = g;
      }
      if (!small_invert(&JtJ[0], element_dim))
        break;

      double step[3] = {0.0, 0.0, 0.0};
      for (unsigned a = 0; a < element_dim; a++)
        for (unsigned b = 0; b < element_dim; b++)
          step[a] += JtJ[a * element_dim + b] * Jtr[b];

      double stepnorm = 0.0;
      for (unsigned a = 0; a < element_dim; a++)
        stepnorm = std::max(stepnorm, std::abs(step[a]));
      if (stepnorm < 1e-15)
        break;

      bool improved = false;
      double lambda = 1.0;
      for (unsigned ls = 0; ls < 12; ls++)
      {
        double trial[3] = {0.0, 0.0, 0.0};
        for (unsigned a = 0; a < element_dim; a++)
          trial[a] = s[a] + lambda * step[a];
        clamp_to_reference_domain(slot, e, trial);
        const double tn = residual_at(trial, rtrial, false);
        if (tn < rnorm * (1.0 - 1e-12))
        {
          for (unsigned a = 0; a < element_dim; a++)
            s[a] = trial[a];
          rnorm = residual_at(s, r, true);
          improved = true;
          break;
        }
        lambda *= 0.5;
      }
      if (!improved)
        break; // at the constrained minimum for this element
    }
    return rnorm;
  }

  void MeshPointLocator::build_affine_inverses()
  {
    ZetaFlagGuard guard(setup.time_index, setup.space == LocatorSpace::Lagrangian);

    const unsigned nelem = elements_by_index.size();
    const unsigned dd = element_dim * element_dim;
    element_geom_kind.assign(nelem, GeomKind::General);
    element_ref_domain.assign(nelem, RefDomain::Unknown);
    has_affine_fit.assign(nelem, false);
    affine_basis.assign((size_t)nelem * space_dim * element_dim, 0.0);
    affine_inverse.assign((size_t)nelem * element_dim * space_dim, 0.0);
    affine_origin.assign((size_t)nelem * space_dim, 0.0);
    affine_s0.assign((size_t)nelem * element_dim, 0.0);
    affine_sdiff.assign((size_t)nelem * dd, 0.0);
    bilinear_coeffs.assign((size_t)nelem * 8, 0.0);
    n_affine_elements = 0;
    n_bilinear_elements = 0;


    for (unsigned ie = 0; ie < nelem; ie++)
    {
      BulkElementBase *e = elements_by_index[ie];

      if (dynamic_cast<oomph::TElementBase *>(e))
        element_ref_domain[ie] = RefDomain::Simplex;
      else if (dynamic_cast<oomph::QElementGeometricBase *>(e))
        element_ref_domain[ie] = RefDomain::Box;
      else if (dynamic_cast<oomph::WedgeElementBase *>(e))
        element_ref_domain[ie] = RefDomain::Prism;
      else if (dynamic_cast<oomph::PyramidElementBase *>(e))
        element_ref_domain[ie] = RefDomain::Pyramid;
      // else: RefDomain::Unknown - no containment test, so no exact path, but the affine fit below
      // is still built and still improves the Newton starting point.

      if (!build_affine_fit_for(e, ie))
        continue; // no usable fit at all: Newton from the element centre, as before

      if (element_ref_domain[ie] == RefDomain::Unknown)
        continue; // seeded Newton only

      if (is_exactly_affine(e, ie))
      {
        element_geom_kind[ie] = GeomKind::Affine;
        n_affine_elements++;
      }
      else if (element_ref_domain[ie] == RefDomain::Box && build_bilinear_for(e, ie))
      {
        element_geom_kind[ie] = GeomKind::Bilinear2d;
        n_bilinear_elements++;
      }
    }
  }

  bool MeshPointLocator::bbox_contains(unsigned slot, const double *x) const
  {
    double diag = 0.0, longest = 0.0;
    for (unsigned d = 0; d < space_dim; d++)
    {
      const double w = element_bbox_max[slot * space_dim + d] - element_bbox_min[slot * space_dim + d];
      diag += w * w;
      longest = std::max(longest, w);
    }
    double slack = bbox_slack * sqrt(diag);
    // A codimension-1 element's box is flat in the normal direction, so a slack proportional to the
    // box diagonal still rejects every off-surface query - which is the whole point of projecting.
    // Give it at least the distance the offset guard is willing to accept, so this cheap test never
    // pre-empts the real one.
    if (mode == LocatorMode::Project)
      slack = std::max(slack, setup.max_projection_offset_factor * longest);
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
  bool MeshPointLocator::try_element(BulkElementBase *e, const double *x_in, double *s_out, double *offset_out, bool allow_multistart) const
  {
    auto found = element_index.find(e);
    if (found == element_index.end())
      return false;
    const unsigned slot = found->second;
    const double *x = x_in;

    // On a periodic coordinate the query must be read on the same branch as the element's own
    // geometry, or a point just past the seam looks a whole period away from the element that
    // actually contains it. Everything below uses `x` through this shifted copy.
    double xw[3] = {0.0, 0.0, 0.0};
    for (unsigned d = 0; d < space_dim; d++)
      xw[d] = unwrap(x[d], nodal_coordinate(e, 0, d), d);
    x = xw;

    if (!bbox_contains(slot, x))
      return false;

    if (mode == LocatorMode::Project)
    {
      // Codimension-1 source: there is no chart to invert, so the match is the closest point on the
      // element. This is what makes 2d surfaces in 3d work at all - see dev_docs/mesh_point_locator.md.
      const unsigned dd_ = element_dim * element_dim;
      double sc[3] = {0.0, 0.0, 0.0};

      if (has_affine_fit[slot])
      {
        // For a flat element (a straight segment, a planar triangle) the least-squares solve IS the
        // exact closest point, not a starting guess, so the exactly-affine case needs no iteration.
        const double *Ainv = &affine_inverse[(size_t)slot * element_dim * space_dim];
        const double *orig = &affine_origin[(size_t)slot * space_dim];
        const double *s0 = &affine_s0[(size_t)slot * element_dim];
        const double *Sdiff = &affine_sdiff[(size_t)slot * dd_];
        double u[3] = {0.0, 0.0, 0.0};
        for (unsigned a = 0; a < element_dim; a++)
          for (unsigned d = 0; d < space_dim; d++)
            u[a] += Ainv[a * space_dim + d] * (x[d] - orig[d]);
        for (unsigned d = 0; d < element_dim; d++)
        {
          double v = s0[d];
          for (unsigned k = 0; k < element_dim; k++)
            v += Sdiff[d * element_dim + k] * u[k];
          sc[d] = v;
        }
      }
      else
      {
        oomph::Vector<double> sn(element_dim, 0.0);
        for (unsigned in = 0; in < e->nnode(); in++)
        {
          e->local_coordinate_of_node(in, sn);
          for (unsigned d = 0; d < element_dim; d++)
            sc[d] += sn[d] / e->nnode();
        }
      }

      // Clamp and iterate. For an exactly affine element this converges immediately (or does
      // nothing at all if the clamped point is already the minimum); for a curved one it is a
      // damped, domain-constrained Gauss-Newton.
      const double offset = project_local_coordinate(e, slot, x, sc);

      const double h = element_size(slot);
      if (h > 0.0 && offset > setup.max_projection_offset_factor * h)
        return false; // too far off this element to be a believable match

      for (unsigned d = 0; d < element_dim; d++)
        s_out[d] = sc[d];
      if (offset_out)
        *offset_out = offset;
      return true;
    }

    // Straight-sided simplex: the map is affine, so the barycentric coordinates come straight out
    // of a precomputed inverse and no Newton solve happens at all. Only accepted when the point
    // lands inside; outside, the fall-through below reproduces the old behaviour exactly rather
    // than relying on this being the last word.
    const GeomKind kind = element_geom_kind[slot];
    const unsigned dd = element_dim * element_dim;

    // Exactly affine: one matrix multiply, no iteration. Covers straight-sided simplices of any
    // order (a T6 with mid-edge nodes at the midpoints included) and parallelogram/parallelepiped
    // quads and hexes - which is most of a structured mesh, sheared or not.
    if (kind == GeomKind::Affine)
    {
      const double *Dinv = &affine_inverse[(size_t)slot * dd];
      const double *orig = &affine_origin[(size_t)slot * space_dim];
      const double *s0 = &affine_s0[(size_t)slot * element_dim];
      const double *Sdiff = &affine_sdiff[(size_t)slot * dd];
      double u[3] = {0.0, 0.0, 0.0};
      for (unsigned d = 0; d < element_dim; d++)
      {
        double v = 0.0;
        for (unsigned k = 0; k < element_dim; k++)
          v += Dinv[d * element_dim + k] * (x[k] - orig[k]);
        u[d] = v;
      }
      double sc[3] = {0.0, 0.0, 0.0};
      for (unsigned d = 0; d < element_dim; d++)
      {
        double v = s0[d];
        for (unsigned k = 0; k < element_dim; k++)
          v += Sdiff[d * element_dim + k] * u[k];
        sc[d] = v;
      }
      if (!inside_reference_domain(slot, e, sc))
        return false; // an exact map has no second answer
      for (unsigned d = 0; d < element_dim; d++)
        s_out[d] = sc[d];
      if (offset_out)
        *offset_out = 0.0;
      return true;
    }

    // Straight-edged but non-parallelogram 2d quad. x(s,t) = b0 + b1 s + b2 t + b3 s t is still
    // exactly invertible: eliminating s leaves a quadratic in t. b3 vanishes only for a
    // parallelogram, which the affine branch above already took, so here the quadratic is generally
    // genuine - but the linear case is kept for the nearly-degenerate one.
    if (kind == GeomKind::Bilinear2d)
    {
      const double *b = &bilinear_coeffs[(size_t)slot * 8];
      const double b1x = b[2], b1y = b[3], b2x = b[4], b2y = b[5], b3x = b[6], b3y = b[7];
      const double dx = x[0] - b[0], dy = x[1] - b[1];

      const double A = -(b2x * b3y - b2y * b3x);
      const double B = (b1x * b2y - b1y * b2x) - (b3x * dy - b3y * dx);
      const double C = -(b1x * dy - b1y * dx);

      double troots[2];
      unsigned nroot = 0;
      if (std::abs(A) < 1e-14 * (std::abs(B) + std::abs(C) + 1e-300))
      {
        if (std::abs(B) < 1e-300)
          return false;
        troots[nroot++] = -C / B;
      }
      else
      {
        const double disc = B * B - 4.0 * A * C;
        if (disc < 0.0)
          return false;
        const double sq = std::sqrt(disc);
        // The numerically stable pair: one root from the standard formula with the sign that avoids
        // cancellation, the other from the product of roots.
        const double q = -0.5 * (B + (B >= 0.0 ? sq : -sq));
        troots[nroot++] = q / A;
        if (std::abs(q) > 1e-300)
          troots[nroot++] = C / q;
      }

      // Both roots can land inside the reference square, so the containment test alone is not
      // enough to choose between them - taking the first one that fits picked the wrong branch and
      // produced either wrong values or a spurious miss. Evaluate the map at each candidate and
      // keep the one that actually reproduces the query point.
      double best_res = -1.0, best_s = 0.0, best_t = 0.0;
      for (unsigned r = 0; r < nroot; r++)
      {
        const double t = troots[r];
        // s from whichever component has the better conditioned denominator.
        const double denx = b1x + b3x * t, deny = b1y + b3y * t;
        double sv;
        if (std::abs(denx) >= std::abs(deny))
        {
          if (std::abs(denx) < 1e-300)
            continue;
          sv = (dx - b2x * t) / denx;
        }
        else
        {
          if (std::abs(deny) < 1e-300)
            continue;
          sv = (dy - b2y * t) / deny;
        }
        const double cand[2] = {sv, t};
        if (!inside_reference_domain(slot, e, cand))
          continue;
        const double rx = b[0] + b1x * sv + b2x * t + b3x * sv * t - x[0];
        const double ry = b[1] + b1y * sv + b2y * t + b3y * sv * t - x[1];
        const double res = std::max(std::abs(rx), std::abs(ry));
        if (best_res < 0.0 || res < best_res)
        {
          best_res = res;
          best_s = sv;
          best_t = t;
        }
      }
      if (best_res >= 0.0)
      {
        s_out[0] = best_s;
        s_out[1] = best_t;
        if (offset_out)
          *offset_out = 0.0;
        return true;
      }
      return false;
    }

    // Seed for whichever Newton runs below: the element's best affine fit, which is a far better
    // guess than the centre for a distorted element. Falls back to the centroid of the local node
    // coordinates, which is inside every reference domain here.
    double seed[3] = {0.0, 0.0, 0.0};
    if (has_affine_fit[slot])
    {
      const double *Dinv = &affine_inverse[(size_t)slot * dd];
      const double *orig = &affine_origin[(size_t)slot * space_dim];
      const double *s0 = &affine_s0[(size_t)slot * element_dim];
      const double *Sdiff = &affine_sdiff[(size_t)slot * dd];
      double u[3] = {0.0, 0.0, 0.0};
      for (unsigned d = 0; d < element_dim; d++)
        for (unsigned k = 0; k < element_dim; k++)
          u[d] += Dinv[d * element_dim + k] * (x[k] - orig[k]);
      for (unsigned d = 0; d < element_dim; d++)
      {
        double v = s0[d];
        for (unsigned k = 0; k < element_dim; k++)
          v += Sdiff[d * element_dim + k] * u[k];
        seed[d] = v;
      }
    }
    else
    {
      oomph::Vector<double> sn(element_dim, 0.0);
      for (unsigned in = 0; in < e->nnode(); in++)
      {
        e->local_coordinate_of_node(in, sn);
        for (unsigned d = 0; d < element_dim; d++)
          seed[d] += sn[d] / e->nnode();
      }
    }

    // Wedges and pyramids cannot go through oomph's locate_zeta at all: it needs both
    // nplot_points() and local_coord_is_valid(), and neither is implemented for those elements
    // (they throw). This is why oomph::MeshAsGeomObject could not interpolate on a wedge or pyramid
    // mesh - the old path threw rather than returning a wrong answer. Both pieces exist here
    // already, so those families get their own Newton instead: seed, polish to machine precision,
    // then test containment against the reference domain documented on the element class.
    if (element_ref_domain[slot] == RefDomain::Prism || element_ref_domain[slot] == RefDomain::Pyramid)
    {
      // First the affine seed, which is what succeeds on anything short of a badly deformed
      // element. On the second pass, fall back to a spread of starting guesses - the element's own
      // node local coordinates plus their centroid, which are guaranteed to lie in the reference
      // domain whatever its shape. This is the equivalent of oomph's plot-point multi-start, which
      // these element families cannot use (nplot_points() is not implemented for them), and it
      // exists for the same reason: a single Newton on a strongly deformed curved element is not
      // guaranteed to converge from any particular starting point.
      double sc[3] = {seed[0], seed[1], seed[2]};
      double res = polish_local_coordinate(e, x, sc);
      bool ok = (res < 1e-10 && inside_reference_domain(slot, e, sc));

      if (!ok && allow_multistart)
      {
        oomph::Vector<double> sn(element_dim, 0.0);
        double centroid[3] = {0.0, 0.0, 0.0};
        for (unsigned in = 0; in < e->nnode(); in++)
        {
          e->local_coordinate_of_node(in, sn);
          for (unsigned d = 0; d < element_dim; d++)
            centroid[d] += sn[d] / e->nnode();
        }
        for (unsigned attempt = 0; attempt <= e->nnode() && !ok; attempt++)
        {
          double start[3] = {0.0, 0.0, 0.0};
          if (attempt == 0)
          {
            for (unsigned d = 0; d < element_dim; d++)
              start[d] = centroid[d];
          }
          else
          {
            e->local_coordinate_of_node(attempt - 1, sn);
            // Pulled a little towards the centroid: starting exactly on a vertex of a curved
            // element can sit on a Jacobian degeneracy (the pyramid apex, notably).
            for (unsigned d = 0; d < element_dim; d++)
              start[d] = 0.9 * sn[d] + 0.1 * centroid[d];
          }
          for (unsigned d = 0; d < element_dim; d++)
            sc[d] = start[d];
          res = polish_local_coordinate(e, x, sc);
          ok = (res < 1e-10 && inside_reference_domain(slot, e, sc));
        }
      }

      if (!ok)
        return false; // no starting point converged inside: the point is not in this element
      for (unsigned d = 0; d < element_dim; d++)
        s_out[d] = sc[d];
      if (offset_out)
        *offset_out = 0.0;
      return true;
    }

    oomph::Vector<double> zeta(space_dim);
    for (unsigned d = 0; d < space_dim; d++)
      zeta[d] = x[d];

    oomph::Vector<double> s(element_dim);
    oomph::GeomObject *go = NULL;

    // Two-stage inversion. oomph's locate_zeta, when not given an initial guess, lays out a grid of
    // plot points over the element and runs a Newton solve from each (elements.cc:4794) - for a 3d
    // element that is on the order of a hundred Newton solves for one query, and it is what the old
    // path paid every time. Nearly always a single solve converges, so try that first and keep the
    // multi-start only as the fallback, which preserves the old behaviour exactly for the awkward
    // elements that need it.
    //
    // The starting point is the element's best affine fit rather than its centre. That matters most
    // for the elements that end up here at all - curved ones, and 3d hexes, whose trilinear inverse
    // has no closed form - because those are exactly the ones where the centre is a poor guess.
    // Clamp into the reference element: the fit is only approximate, and Newton started well
    // outside can wander into a neighbouring element's preimage.
    for (unsigned d = 0; d < element_dim; d++)
      s[d] = std::min(std::max(seed[d], e->s_min()), e->s_max());
    e->locate_zeta(zeta, go, s, true);

    // oomph's multi-start lays out a grid of plot-point initial guesses, each with its own Newton
    // solve. Two reasons it is not the default here. It is paid per REJECTED candidate, and the
    // search tries several candidates per query - on a distorted 3d hex mesh that alone was the
    // difference between 17 us and 10 us per point. And it is not available at all for wedges and
    // pyramids: it goes through nplot_points() (elements.cc:4795), which those elements do not
    // implement, so calling it throws. That is also why oomph::MeshAsGeomObject cannot locate in a
    // wedge or pyramid mesh at all - the old interpolation path simply threw on them.
    const bool multistart_supported = (element_ref_domain[slot] == RefDomain::Simplex ||
                                       element_ref_domain[slot] == RefDomain::Box);
    if (!go && allow_multistart && multistart_supported)
    {
      for (unsigned d = 0; d < element_dim; d++)
        s[d] = 0.5 * (e->s_min() + e->s_max());
      e->locate_zeta(zeta, go, s, false);
    }

    if (!go)
      return false;

    // Polish, but only keep the result if it actually improved things. locate_zeta has already
    // certified that s is inside the element to its own 1e-7 tolerance; a polish that diverges on a
    // badly deformed element must not be allowed to replace that with something worse.
    double polished[3] = {0.0, 0.0, 0.0};
    for (unsigned d = 0; d < element_dim; d++)
      polished[d] = s[d];
    const double res = polish_local_coordinate(e, x, polished);
    if (res < 1e-12 && inside_reference_domain(slot, e, polished))
    {
      for (unsigned d = 0; d < element_dim; d++)
        s_out[d] = polished[d];
    }
    else
    {
      // Keep locate_zeta's answer. It is only good to oomph's 1e-7 residual tolerance, but rejecting
      // the element outright sends the point to the nearest-node blend, which is far worse - tried,
      // and it cost the nodal interpolator two orders of magnitude on a linear field.
      for (unsigned d = 0; d < element_dim; d++)
        s_out[d] = s[d];
    }
    if (offset_out)
      *offset_out = 0.0;
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

      BulkElementBase *hit = NULL;
      int how = -1;
      unsigned found_on_pass = 0;

      // Projection differs from inversion in a way the search has to respect: a query beside a
      // surface is "in" several nearby elements to differing degrees, so the first acceptable
      // candidate is not the answer - the one with the smallest perpendicular offset is. In Invert
      // mode a hit is exact and the first one ends the search, as before.
      const bool rank_by_offset = (mode == LocatorMode::Project);
      double best_offset = 0.0;
      double best_s[3] = {0.0, 0.0, 0.0};

      // Two passes. The first only ever runs a single Newton per candidate; only if the point is
      // not found at all does the second allow oomph's multi-start, so its cost is paid once per
      // genuinely hard query rather than once per rejected candidate.
      for (unsigned pass = 0; pass < 2 && !hit; pass++)
      {
      const bool allow_multistart = (pass == 1);
      tried.clear();

      // (a) The previous match and everything sharing a node with it. Consecutive queries are
      // usually neighbours - all the integration points of one target element, for instance - so
      // this is the case that should dominate, and it never touches the tree.
      const bool same_group = (hint_groups == nullptr) || (previous_group == (*hint_groups)[i]);
      if (previous && same_group)
      {
        if (try_element(previous, &query[0], &s[0], &offset, allow_multistart))
        {
          hit = previous;
          how = 0;
          best_offset = offset;
          for (unsigned d = 0; d < element_dim; d++) best_s[d] = s[d];
        }
        if (!hit || rank_by_offset)
        {
          tried.push_back(previous);
          // In Project mode the scan must continue past the first acceptable element: the answer is
          // the NEAREST element, not any element within tolerance. Stopping at the first one made an
          // on-surface query match a neighbour whose closest point was half an element away.
          for (unsigned in = 0; in < previous->nnode() && (!hit || rank_by_offset); in++)
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
              if (try_element(cand, &query[0], &s[0], &offset, allow_multistart))
              {
                if (!hit || (rank_by_offset && offset < best_offset))
                {
                  hit = cand; how = 0; best_offset = offset;
                  for (unsigned d = 0; d < element_dim; d++) best_s[d] = s[d];
                }
                if (!rank_by_offset)
                  break;
              }
            }
          }
        }
      }

      // (b) The nearest source node, and the elements around it. Also entered in Project mode when
      // the walk already found something, since a closer element may not be in the walk's star.
      if (!hit || rank_by_offset)
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
            if (try_element(cand, &query[0], &s[0], &offset, allow_multistart))
            {
              if (!hit || (rank_by_offset && offset < best_offset))
              {
                hit = cand; how = 1; best_offset = offset;
                for (unsigned d = 0; d < element_dim; d++) best_s[d] = s[d];
              }
              if (!rank_by_offset)
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
              if (try_element(cand, &query[0], &s[0], &offset, allow_multistart))
              {
                if (!hit || (rank_by_offset && offset < best_offset))
                {
                  hit = cand; how = 2; best_offset = offset;
                  for (unsigned d = 0; d < element_dim; d++) best_s[d] = s[d];
                }
                if (!rank_by_offset)
                  break;
              }
            }
            if (hit)
              break;
          }
        }
      }

      if (hit)
        found_on_pass = pass;
      } // end of the two passes

      if (hit)
      {
        out.handles[i].owner = 0; // serial: everything is local
        out.handles[i].slot = out.local_elements.size();
        out.local_elements.push_back(hit);
        for (unsigned d = 0; d < element_dim; d++)
          out.local_s.push_back(best_s[d]);
        out.local_offset.push_back(best_offset);
        if (how == 0) out.n_by_walk++;
        else if (how == 1) out.n_by_nearest_node++;
        else out.n_by_widening++;
        if (found_on_pass == 1) out.n_needing_multistart++;
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

  double LocationSet::offset_of(unsigned i) const
  {
    if (i >= npoint || !handles[i].is_located() || handles[i].owner != 0)
      return -1.0;
    return local_offset[handles[i].slot];
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

  std::string MeshPointLocator::affine_fraction() const
  {
    std::ostringstream oss;
    oss << n_affine_elements << "/" << elements_by_index.size() << " affine";
    if (n_bilinear_elements)
      oss << ", " << n_bilinear_elements << " bilinear";
    const unsigned rest = elements_by_index.size() - n_affine_elements - n_bilinear_elements;
    if (rest)
      oss << ", " << rest << " newton";
    return oss.str();
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
    if (n_needing_multistart)
      oss << ", " << n_needing_multistart << " needed multistart";
    return oss.str();
  }

  namespace
  {
    // Sizes of each block of one time level, in the fixed order evaluate() writes them. Taken from a
    // representative source element rather than the request, because every count except field_map's
    // is a property of the source mesh's code instance.
    struct EvalLayout
    {
      unsigned ncont = 0, nDL = 0, nD0 = 0, nDG = 0, npos = 0, nlag = 0, nzeta = 0;
      unsigned per_level() const { return ncont + nDL + nD0 + nDG + npos + nlag + nzeta; }
    };

    EvalLayout eval_layout(BulkElementBase *e, const EvalRequest &what)
    {
      EvalLayout L;
      if (!e)
        return L;
      const JITFuncSpec_Table_FiniteElement_t *ft = e->get_code_instance()->get_func_table();
      if (what.continuous_fields)
      {
        // With a field_map the caller is asking for ITS OWN field layout, so the count is the map's
        // length; entries mapping to -1 are simply left at zero.
        L.ncont = (what.field_map.empty() ? e->ncont_interpolated_values()
                                          : (unsigned)what.field_map.size());
      }
      if (what.DL_fields)
        L.nDL = ft->info_DL.numfields;
      if (what.D0_fields)
        L.nD0 = ft->info_D0.numfields;
      if (what.DG_fields)
      {
        for (unsigned i = 0; i < ft->num_present_dg_spaces; i++)
          L.nDG += ft->present_dg_spaces[i]->numfields_new;
      }
      if (what.position)
        L.npos = e->nodal_dimension();
      if (what.lagrangian)
        L.nlag = e->nlagrangian();
      if (what.zeta)
        L.nzeta = e->dim();
      return L;
    }

    unsigned n_levels(const EvalRequest &what)
    {
      return what.time_levels.empty() ? 1u : (unsigned)what.time_levels.size();
    }
  }

  unsigned LocationSet::values_per_point(const EvalRequest &what) const
  {
    BulkElementBase *rep = NULL;
    for (BulkElementBase *e : local_elements)
    {
      if (e)
      {
        rep = e;
        break;
      }
    }
    if (!rep && locator && locator->get_source_mesh() && locator->get_source_mesh()->nelement())
      rep = dynamic_cast<BulkElementBase *>(locator->get_source_mesh()->element_pt(0));
    return n_levels(what) * eval_layout(rep, what).per_level();
  }

  std::vector<double> LocationSet::evaluate(const EvalRequest &what) const
  {
    const unsigned stride = this->values_per_point(what);
    std::vector<double> out((size_t)npoint * stride, 0.0);
    if (!stride || !npoint)
      return out;

    // interpolated_zeta() reports whichever coordinate the two static flags select, so a zeta
    // request has to be answered under the same setting the points were located in - otherwise the
    // caller gets Lagrangian coordinates back from a search it ran in Eulerian space.
    std::unique_ptr<ZetaFlagGuard> zguard;
    if (what.zeta && locator)
    {
      const LocatorSetup &su = locator->get_setup();
      zguard.reset(new ZetaFlagGuard(su.time_index, su.space == LocatorSpace::Lagrangian));
    }

    const std::vector<unsigned> level0(1, 0u);
    const std::vector<unsigned> &levels = (what.time_levels.empty() ? level0 : what.time_levels);

    oomph::Vector<double> vals;
    std::vector<double> dvals;
    oomph::Vector<double> s_o;
    std::vector<double> sloc;
    for (unsigned i = 0; i < npoint; i++)
    {
      BulkElementBase *e = NULL;
      if (!this->resolve_local(i, e, sloc) || !e)
        continue; // unlocated, or owned by another rank: left at zero, as documented
      const EvalLayout L = eval_layout(e, what);
      s_o.resize(sloc.size());
      for (unsigned d = 0; d < sloc.size(); d++)
        s_o[d] = sloc[d];

      double *row = &out[(size_t)i * stride];
      for (unsigned li = 0; li < levels.size(); li++)
      {
        const unsigned t = levels[li];
        double *blk = row + (size_t)li * L.per_level();
        if (L.ncont)
        {
          e->get_interpolated_values(t, s_o, vals);
          for (unsigned f = 0; f < L.ncont; f++)
          {
            const int src = (what.field_map.empty() ? (int)f : what.field_map[f]);
            if (src >= 0 && (unsigned)src < vals.size())
              blk[f] = vals[src];
          }
          blk += L.ncont;
        }
        if (L.nDL)
        {
          e->get_interpolated_fields_DL(s_o, dvals, t);
          for (unsigned f = 0; f < L.nDL && f < dvals.size(); f++)
            blk[f] = dvals[f];
          blk += L.nDL;
        }
        if (L.nD0)
        {
          e->get_interpolated_fields_D0(s_o, dvals, t);
          for (unsigned f = 0; f < L.nD0 && f < dvals.size(); f++)
            blk[f] = dvals[f];
          blk += L.nD0;
        }
        if (L.nDG)
        {
          // DG values live in the element's own internal Data, which allocate_discontinous_fields
          // lays out before the DL and D0 entries. Only the fields the element owns
          // (numfields_new); anything inherited from the bulk is external data and not the source
          // element's to report.
          const JITFuncSpec_Table_FiniteElement_t *ft = e->get_code_instance()->get_func_table();
          unsigned idata = 0, f = 0;
          for (unsigned si = 0; si < ft->num_present_dg_spaces; si++)
          {
            auto *space_info = ft->present_dg_spaces[si];
            const unsigned nn = e->get_eleminfo()->nnode_of_space[space_info->space_index];
            oomph::Shape psi(nn ? nn : 1);
            if (nn)
              e->shape_of_space(space_info->space_index, s_o, psi);
            for (unsigned fi = 0; fi < space_info->numfields_new; fi++, f++, idata++)
            {
              double acc = 0.0;
              for (unsigned l = 0; l < nn; l++)
                acc += psi[l] * e->internal_data_pt(idata)->value(t, l);
              blk[f] = acc;
            }
          }
          blk += L.nDG;
        }
        if (L.npos)
        {
          for (unsigned d = 0; d < L.npos; d++)
            blk[d] = e->interpolated_x(t, s_o, d);
          blk += L.npos;
        }
        if (L.nlag)
        {
          for (unsigned d = 0; d < L.nlag; d++)
            blk[d] = e->interpolated_xi(s_o, d);
          blk += L.nlag;
        }
        if (L.nzeta)
        {
          oomph::Vector<double> z(L.nzeta, 0.0);
          e->interpolated_zeta(s_o, z);
          for (unsigned d = 0; d < L.nzeta; d++)
            blk[d] = z[d];
        }
      }
    }
    return out;
  }

}
