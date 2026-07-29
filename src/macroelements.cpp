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

#include "macroelements.hpp"
#include "meshtemplate.hpp"

#include <algorithm>
#include <iterator>
#include <set>

namespace pyoomph
{

  unsigned macro_num_vertices(const MacroElementShape &shape)
  {
    switch (shape)
    {
    case MacroElementShape::Quad2d:
      return 4;
    case MacroElementShape::Tri2d:
      return 3;
    case MacroElementShape::Brick3d:
      return 8;
    case MacroElementShape::Tet3d:
      return 4;
    case MacroElementShape::Wedge3d:
      return 6;
    case MacroElementShape::Pyramid3d:
      return 5;
    }
    throw_runtime_error("Unknown macro element shape");
    return 0;
  }

  unsigned macro_shape_dim(const MacroElementShape &shape)
  {
    switch (shape)
    {
    case MacroElementShape::Quad2d:
    case MacroElementShape::Tri2d:
      return 2;
    case MacroElementShape::Brick3d:
    case MacroElementShape::Tet3d:
    case MacroElementShape::Wedge3d:
    case MacroElementShape::Pyramid3d:
      return 3;
    }
    throw_runtime_error("Unknown macro element shape");
    return 0;
  }

  // Must agree exactly with the corresponding oomph-lib element's own vertex shape functions and node
  // ordering, since the macro coordinate is that element's local coordinate:
  //  - QElement<2,2>: tensor-product Lagrange, node index = i0 + 2*i1, so vertices run
  //    (-1,-1), (1,-1), (-1,1), (1,1).
  //  - TElement<2,2>: TElementShape<2,2>::shape is (s0, s1, 1-s0-s1) with vertices (1,0), (0,1), (0,0),
  //    i.e. the barycentric coordinates already.
  void macro_c1_shape(const MacroElementShape &shape, const oomph::Vector<double> &s, std::vector<double> &psi)
  {
    switch (shape)
    {
    case MacroElementShape::Quad2d:
      psi.resize(4);
      psi[0] = 0.25 * (1.0 - s[0]) * (1.0 - s[1]);
      psi[1] = 0.25 * (1.0 + s[0]) * (1.0 - s[1]);
      psi[2] = 0.25 * (1.0 - s[0]) * (1.0 + s[1]);
      psi[3] = 0.25 * (1.0 + s[0]) * (1.0 + s[1]);
      return;
    case MacroElementShape::Tri2d:
      psi.resize(3);
      psi[0] = s[0];
      psi[1] = s[1];
      psi[2] = 1.0 - s[0] - s[1];
      return;
    case MacroElementShape::Brick3d:
      // QElement<3,2>: node index = i0 + 2*i1 + 4*i2.
      psi.resize(8);
      for (unsigned int v = 0; v < 8; v++)
      {
        psi[v] = 0.125 * (1.0 + ((v & 1u) ? 1.0 : -1.0) * s[0]) * (1.0 + ((v & 2u) ? 1.0 : -1.0) * s[1]) *
                 (1.0 + ((v & 4u) ? 1.0 : -1.0) * s[2]);
      }
      return;
    case MacroElementShape::Tet3d:
      // TElementShape<3,2>::shape, i.e. the barycentric coordinates, with vertices
      // (1,0,0), (0,1,0), (0,0,1), (0,0,0).
      psi.resize(4);
      psi[0] = s[0];
      psi[1] = s[1];
      psi[2] = s[2];
      psi[3] = 1.0 - s[0] - s[1] - s[2];
      return;
    case MacroElementShape::Wedge3d:
    {
      // WedgeElementShapeC1: a triangle barycentric in (s0,s1) times a linear factor in s2.
      const double l1 = 1.0 - s[0] - s[1];
      psi.resize(6);
      psi[0] = l1 * (1.0 - s[2]);
      psi[1] = s[0] * (1.0 - s[2]);
      psi[2] = s[1] * (1.0 - s[2]);
      psi[3] = l1 * s[2];
      psi[4] = s[0] * s[2];
      psi[5] = s[1] * s[2];
      return;
    }
    case MacroElementShape::Pyramid3d:
    {
      // PyramidElementShapeC1: rational, with a removable singularity at the apex s2 = 1 where the
      // whole quad base collapses to a point. The shipped shape function divides by 1-s2 unguarded
      // because its callers never evaluate exactly there; the macro map is evaluated at arbitrary
      // points of the reference domain, so take the limit explicitly.
      const double w = 1.0 - s[2];
      psi.resize(5);
      if (w < 1e-13)
      {
        psi[0] = psi[1] = psi[2] = psi[3] = 0.0;
        psi[4] = 1.0;
        return;
      }
      const double iw = 1.0 / w;
      psi[0] = (w - s[0]) * (w - s[1]) * iw;
      psi[1] = s[0] * (w - s[1]) * iw;
      psi[2] = s[0] * s[1] * iw;
      psi[3] = (w - s[0]) * s[1] * iw;
      psi[4] = s[2];
      return;
    }
    }
    throw_runtime_error("Unknown macro element shape");
  }

  void macro_reference_vertices(const MacroElementShape &shape, std::vector<std::vector<double>> &sv)
  {
    switch (shape)
    {
    case MacroElementShape::Quad2d:
      sv = {{-1.0, -1.0}, {1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}};
      return;
    case MacroElementShape::Tri2d:
      sv = {{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
      return;
    case MacroElementShape::Brick3d:
      sv.assign(8, std::vector<double>(3, 0.0));
      for (unsigned int v = 0; v < 8; v++)
      {
        sv[v][0] = (v & 1u) ? 1.0 : -1.0;
        sv[v][1] = (v & 2u) ? 1.0 : -1.0;
        sv[v][2] = (v & 4u) ? 1.0 : -1.0;
      }
      return;
    case MacroElementShape::Tet3d:
      sv = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}};
      return;
    case MacroElementShape::Wedge3d:
      sv = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}};
      return;
    case MacroElementShape::Pyramid3d:
      sv = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
      return;
    }
    throw_runtime_error("Unknown macro element shape");
  }

  const std::vector<std::pair<unsigned, unsigned>> &macro_edges(const MacroElementShape &shape)
  {
    // Vertex numbering as in macro_reference_vertices.
    static const std::vector<std::pair<unsigned, unsigned>> tri = {{0, 1}, {1, 2}, {2, 0}};
    static const std::vector<std::pair<unsigned, unsigned>> quad = {{0, 1}, {1, 3}, {3, 2}, {2, 0}};
    static const std::vector<std::pair<unsigned, unsigned>> tet = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    // Brick vertex v has coordinates given by its bits, so its edges join vertices differing in one bit.
    static const std::vector<std::pair<unsigned, unsigned>> brick = {
        {0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    static const std::vector<std::pair<unsigned, unsigned>> wedge = {
        {0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}};
    static const std::vector<std::pair<unsigned, unsigned>> pyramid = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}};
    switch (shape)
    {
    case MacroElementShape::Tri2d:
      return tri;
    case MacroElementShape::Quad2d:
      return quad;
    case MacroElementShape::Tet3d:
      return tet;
    case MacroElementShape::Brick3d:
      return brick;
    case MacroElementShape::Wedge3d:
      return wedge;
    case MacroElementShape::Pyramid3d:
      return pyramid;
    }
    throw_runtime_error("Unknown macro element shape");
    return tri;
  }

  GenericMacroElement::GenericMacroElement(oomph::Domain *domain, const unsigned &index,
                                           const MacroElementShape &shape_,
                                           const std::vector<oomph::Node *> &vertices)
      : oomph::MacroElement(domain, index), shape(shape_), vertex_nodes(vertices)
  {
    if (vertex_nodes.size() != macro_num_vertices(shape))
    {
      throw_runtime_error("GenericMacroElement got " + std::to_string(vertex_nodes.size()) +
                          " vertex nodes, but its shape has " + std::to_string(macro_num_vertices(shape)));
    }
    for (unsigned int i = 0; i < vertex_nodes.size(); i++)
    {
      if (!vertex_nodes[i])
        throw_runtime_error("GenericMacroElement got a null vertex node at index " + std::to_string(i));
    }
  }

  void GenericMacroElement::add_curved_facet(MeshTemplateFacet *facet,
                                             const std::vector<unsigned> &local_vertices,
                                             const std::vector<unsigned> &parametric_index)
  {
    if (local_vertices.size() != parametric_index.size())
      throw_runtime_error("Mismatched vertex/parametric index lists for a curved macro element facet");
    curved_facets.push_back(MacroCurvedFacet{facet, local_vertices, parametric_index});
    rebuild_edge_corrections();
  }

  void GenericMacroElement::vertex_position(const unsigned &v, const unsigned &t, const unsigned &dim,
                                            std::vector<double> &x) const
  {
    oomph::Node *n = vertex_nodes[v];
    const unsigned ndim = n->ndim();
    x.assign(dim, 0.0);
    for (unsigned int i = 0; i < dim && i < ndim; i++)
      x[i] = n->x(t, i);
  }

  // Weight and deviation of one sub-entity (a curved facet, or a shared edge being corrected for).
  bool GenericMacroElement::subentity_deviation(const MacroCurvedFacet &cf, const std::vector<double> &lambda,
                                                const unsigned &t, const unsigned &dim,
                                                double &w, std::vector<double> &deviation) const
  {
    MeshTemplateCurvedEntity *entity = cf.facet->curved_entity;
    if (!entity)
      return false;

    w = 0.0;
    for (auto &v : cf.local_vertices)
      w += lambda[v];
    // Away from this sub-entity the weight vanishes and so does its contribution. Skipping also keeps
    // the division below well posed.
    if (std::fabs(w) < 1e-14)
      return false;

    // Sub-entity-local coordinates: the vertex weights renormalised over it. When s lies on the
    // sub-entity these are its own coordinates; elsewhere they collapse onto the part of it that is
    // nearest in the barycentric sense, which is what makes the deviation die out there.
    const unsigned nfv = cf.local_vertices.size();
    std::vector<double> sigma(nfv);
    for (unsigned int k = 0; k < nfv; k++)
      sigma[k] = lambda[cf.local_vertices[k]] / w;

    // Curved image: blend the stored parametric coordinates with those weights and map.
    const unsigned npar = entity->get_parametric_dimension();
    std::vector<double> parametric(npar, 0.0);
    for (unsigned int k = 0; k < nfv; k++)
    {
      const std::vector<double> &p = cf.facet->parametrics[cf.parametric_index[k]];
      for (unsigned int i = 0; i < npar && i < p.size(); i++)
        parametric[i] += sigma[k] * p[i];
    }
    std::vector<double> curved(dim, 0.0);
    entity->parametric_to_position(t, parametric, curved);

    // ... minus the straight image of the same point.
    std::vector<double> xv;
    deviation.assign(dim, 0.0);
    for (unsigned int k = 0; k < nfv; k++)
    {
      vertex_position(cf.local_vertices[k], t, dim, xv);
      for (unsigned int i = 0; i < dim; i++)
        deviation[i] -= sigma[k] * xv[i];
    }
    for (unsigned int i = 0; i < dim && i < curved.size(); i++)
      deviation[i] += curved[i];
    return true;
  }

  // Find the sub-entities shared by two or more curved facets, which the facet sum would otherwise
  // count once per facet. In 3d two adjacent facets share an edge; in 2d they share only a vertex,
  // where every deviation already vanishes, so there is nothing to correct.
  void GenericMacroElement::rebuild_edge_corrections()
  {
    edge_corrections.clear();
    if (macro_shape_dim(shape) < 3 || curved_facets.size() < 2)
      return;

    // Candidate shared sets: the intersections of each pair of curved facets' vertex sets.
    std::set<std::vector<unsigned>> candidates;
    for (unsigned int i = 0; i < curved_facets.size(); i++)
    {
      for (unsigned int j = i + 1; j < curved_facets.size(); j++)
      {
        std::vector<unsigned> a = curved_facets[i].local_vertices, b = curved_facets[j].local_vertices;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::vector<unsigned> shared;
        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(shared));
        if (shared.size() >= 2)
          candidates.insert(shared);
      }
    }

    for (auto &shared : candidates)
    {
      // How many curved facets contain this whole set, and which was the first?
      unsigned multiplicity = 0;
      const MacroCurvedFacet *host = NULL;
      MeshTemplateCurvedEntity *first_entity = NULL;
      bool inconsistent = false;
      for (auto &cf : curved_facets)
      {
        std::vector<unsigned> sorted = cf.local_vertices;
        std::sort(sorted.begin(), sorted.end());
        if (!std::includes(sorted.begin(), sorted.end(), shared.begin(), shared.end()))
          continue;
        multiplicity++;
        if (!host)
        {
          host = &cf;
          first_entity = cf.facet->curved_entity;
        }
        else if (cf.facet->curved_entity != first_entity)
        {
          inconsistent = true;
        }
      }
      if (multiplicity < 2 || !host)
        continue;
      if (inconsistent)
      {
        // Two different curved entities claiming the same edge means the geometry itself disagrees
        // about where that edge lies; the blend cannot repair that, only report it.
        throw_runtime_error("Two different curved entities are attached to facets sharing the same "
                            "edge of one element, so they disagree about where that edge lies. Split "
                            "the element, or make the entities agree there.");
      }
      // Map the shared vertices back to the host facet's parametric slots.
      MacroCurvedFacet edge;
      edge.facet = host->facet;
      for (auto &v : shared)
      {
        for (unsigned int k = 0; k < host->local_vertices.size(); k++)
        {
          if (host->local_vertices[k] == v)
          {
            edge.local_vertices.push_back(v);
            edge.parametric_index.push_back(host->parametric_index[k]);
            break;
          }
        }
      }
      edge_corrections.push_back({edge, double(multiplicity - 1)});
    }
  }

  // x(s) = sum_v lambda_v X_v + sum_F w_F d_F - sum_E (m_E - 1) w_E d_E, the transfinite blend of
  // macroelements.hpp written in generalised barycentric coordinates.
  void GenericMacroElement::macro_map(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &r)
  {
    const unsigned dim = r.size();
    std::vector<double> lambda;
    macro_c1_shape(shape, s, lambda);

    // Straight-sided part: the element's own C1 interpolation of its vertex positions.
    std::vector<double> xv;
    for (unsigned int i = 0; i < dim; i++)
      r[i] = 0.0;
    for (unsigned int v = 0; v < lambda.size(); v++)
    {
      vertex_position(v, t, dim, xv);
      for (unsigned int i = 0; i < dim; i++)
        r[i] += lambda[v] * xv[i];
    }

    double w;
    std::vector<double> deviation;
    for (auto &cf : curved_facets)
    {
      if (!subentity_deviation(cf, lambda, t, dim, w, deviation))
        continue;
      for (unsigned int i = 0; i < dim; i++)
        r[i] += w * deviation[i];
    }
    for (auto &ec : edge_corrections)
    {
      if (!subentity_deviation(ec.first, lambda, t, dim, w, deviation))
        continue;
      for (unsigned int i = 0; i < dim; i++)
        r[i] -= ec.second * w * deviation[i];
    }
  }

  void GenericMacroElement::output(const unsigned &t, std::ostream &outfile, const unsigned &nplot)
  {
    const unsigned dim = macro_shape_dim(shape);
    oomph::Vector<double> s(dim), r(dim);
    if (shape == MacroElementShape::Tet3d || shape == MacroElementShape::Wedge3d || shape == MacroElementShape::Pyramid3d)
    {
      outfile << "ZONE" << std::endl;
      for (unsigned i = 0; i < nplot; i++)
        for (unsigned j = 0; i + j < nplot; j++)
          for (unsigned k = 0; i + j + k < nplot; k++)
          {
            s[0] = double(i) / double(nplot - 1);
            s[1] = double(j) / double(nplot - 1);
            s[2] = double(k) / double(nplot - 1);
            macro_map(t, s, r);
            outfile << r[0] << " " << r[1] << " " << r[2] << std::endl;
          }
    }
    else if (shape == MacroElementShape::Brick3d)
    {
      outfile << "ZONE I=" << nplot << ", J=" << nplot << ", K=" << nplot << std::endl;
      for (unsigned k = 0; k < nplot; k++)
      {
        s[2] = -1.0 + 2.0 * double(k) / double(nplot - 1);
        for (unsigned i = 0; i < nplot; i++)
        {
          s[1] = -1.0 + 2.0 * double(i) / double(nplot - 1);
          for (unsigned j = 0; j < nplot; j++)
          {
            s[0] = -1.0 + 2.0 * double(j) / double(nplot - 1);
            macro_map(t, s, r);
            outfile << r[0] << " " << r[1] << " " << r[2] << std::endl;
          }
        }
      }
    }
    else if (shape == MacroElementShape::Quad2d)
    {
      outfile << "ZONE I=" << nplot << ", J=" << nplot << std::endl;
      for (unsigned i = 0; i < nplot; i++)
      {
        s[1] = -1.0 + 2.0 * double(i) / double(nplot - 1);
        for (unsigned j = 0; j < nplot; j++)
        {
          s[0] = -1.0 + 2.0 * double(j) / double(nplot - 1);
          macro_map(t, s, r);
          outfile << r[0] << " " << r[1] << std::endl;
        }
      }
    }
    else
    {
      // Barycentric sampling of the reference triangle.
      outfile << "ZONE" << std::endl;
      for (unsigned i = 0; i < nplot; i++)
      {
        for (unsigned j = 0; i + j < nplot; j++)
        {
          s[0] = double(i) / double(nplot - 1);
          s[1] = double(j) / double(nplot - 1);
          macro_map(t, s, r);
          outfile << r[0] << " " << r[1] << std::endl;
        }
      }
    }
  }

  void GenericMacroElement::output_macro_element_boundaries(std::ostream &outfile, const unsigned &nplot)
  {
    const unsigned dim = macro_shape_dim(shape);
    if (dim != 2)
      throw_runtime_error("output_macro_element_boundaries is only implemented for 2d macro elements");
    std::vector<std::vector<double>> sv;
    macro_reference_vertices(shape, sv);
    const unsigned nv = sv.size();
    oomph::Vector<double> s(dim), r(dim);
    // Walk the closed polygon of reference vertices; for both 2d shapes consecutive vertices in
    // "ring" order (quad: 0,1,3,2) span exactly the facets.
    std::vector<unsigned> ring;
    if (shape == MacroElementShape::Quad2d)
      ring = {0, 1, 3, 2};
    else
      ring = {0, 1, 2};
    for (unsigned int e = 0; e < ring.size(); e++)
    {
      const std::vector<double> &a = sv[ring[e]];
      const std::vector<double> &b = sv[ring[(e + 1) % ring.size()]];
      outfile << "ZONE I=" << nplot << std::endl;
      for (unsigned int k = 0; k < nplot; k++)
      {
        const double f = double(k) / double(nplot - 1);
        for (unsigned int i = 0; i < dim; i++)
          s[i] = a[i] + f * (b[i] - a[i]);
        macro_map(0, s, r);
        outfile << r[0] << " " << r[1] << std::endl;
      }
    }
    (void)nv;
  }

  void GenericMacroElement::assemble_macro_to_eulerian_jacobian(const unsigned &t, const oomph::Vector<double> &s,
                                                                oomph::DenseMatrix<double> &jacobian)
  {
    const unsigned dim = s.size();
    const double h = 1e-6;
    oomph::Vector<double> sp(s), sm(s), rp(dim), rm(dim);
    for (unsigned int j = 0; j < dim; j++)
    {
      sp = s;
      sm = s;
      sp[j] += h;
      sm[j] -= h;
      macro_map(t, sp, rp);
      macro_map(t, sm, rm);
      for (unsigned int i = 0; i < dim; i++)
        jacobian(i, j) = (rp[i] - rm[i]) / (2.0 * h);
    }
  }

  void GenericMacroElement::assemble_macro_to_eulerian_jacobian2(const unsigned &t, const oomph::Vector<double> &s,
                                                                 oomph::DenseMatrix<double> &jacobian2)
  {
    const unsigned dim = s.size();
    const double h = 1e-4;
    oomph::Vector<double> sp(s), sm(s), r0(dim), rp(dim), rm(dim);
    macro_map(t, s, r0);
    for (unsigned int j = 0; j < dim; j++)
    {
      sp = s;
      sm = s;
      sp[j] += h;
      sm[j] -= h;
      macro_map(t, sp, rp);
      macro_map(t, sm, rm);
      for (unsigned int i = 0; i < dim; i++)
        jacobian2(i, j) = (rp[i] - 2.0 * r0[i] + rm[i]) / (h * h);
    }
  }

}
