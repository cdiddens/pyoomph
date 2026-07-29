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
    }
    throw_runtime_error("Unknown macro element shape");
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
    // In 3d, two curved facets sharing an edge each contribute that edge's deviation, so the blend
    // needs the inclusion-exclusion term of the plan's 3.2 to avoid counting it twice. That term is
    // S3 work; until then refuse the case rather than silently returning a doubled deviation. A
    // single curved facet per element -- the only 3d case reachable today -- needs no correction.
    if (macro_shape_dim(shape) == 3 && curved_facets.size() > 1)
    {
      throw_runtime_error("More than one curved facet on a 3d macro element is not supported yet "
                          "(the shared-edge correction is stage S3 of "
                          "dev_docs/macro_elements_generalisation.md)");
    }
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

  // The blend of macroelements.hpp. In 2d there is no edge-correction term: two facets of a 2d element
  // meet only at a vertex, and every facet deviation already vanishes at its own vertices.
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

    for (auto &cf : curved_facets)
    {
      MeshTemplateCurvedEntity *entity = cf.facet->curved_entity;
      if (!entity)
        continue;

      double w = 0.0;
      for (auto &v : cf.local_vertices)
        w += lambda[v];
      // Away from this facet the weight vanishes and so does its contribution. Skipping also keeps the
      // division below well posed.
      if (std::fabs(w) < 1e-14)
        continue;

      // Facet-local coordinates: the vertex weights renormalised over the facet. When s lies on the
      // facet these are its own coordinates; when it lies on a neighbouring facet they collapse onto
      // the shared sub-entity, which is exactly what makes the deviation die out there.
      const unsigned nfv = cf.local_vertices.size();
      std::vector<double> sigma(nfv);
      for (unsigned int k = 0; k < nfv; k++)
        sigma[k] = lambda[cf.local_vertices[k]] / w;

      // Curved image: blend the facet's stored parametric coordinates with those weights and map.
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

      // Straight image of the same facet point, and the deviation between the two.
      std::vector<double> straight(dim, 0.0);
      for (unsigned int k = 0; k < nfv; k++)
      {
        vertex_position(cf.local_vertices[k], t, dim, xv);
        for (unsigned int i = 0; i < dim; i++)
          straight[i] += sigma[k] * xv[i];
      }

      for (unsigned int i = 0; i < dim && i < curved.size(); i++)
        r[i] += w * (curved[i] - straight[i]);
    }
  }

  void GenericMacroElement::output(const unsigned &t, std::ostream &outfile, const unsigned &nplot)
  {
    const unsigned dim = macro_shape_dim(shape);
    oomph::Vector<double> s(dim), r(dim);
    if (shape == MacroElementShape::Brick3d)
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
