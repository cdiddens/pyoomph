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

// One MacroElement for every element shape pyoomph has.
//
// oomph-lib's MacroElement machinery describes a curved element by transfinite (Coons / Gordon-Hall)
// interpolation of its boundary parametrisations, but its implementations -- QMacroElement<2> and
// QMacroElement<3> -- are written directly in terms of a tensor-product reference domain: they name
// N/S/W/E and L/R/D/U/B/F and correct the interior with (1-eta)*diff_S + eta*diff_N + ... There is no
// useful specialisation of that to a triangle, tetrahedron, wedge or pyramid, which is why
// oomph::TMacroElement<2> only ever existed as a stub that throws.
//
// The way out is to write the same blend in coordinates that do not know the shape. Every pyoomph
// element provides its vertex (C1) shape functions, and for every shape those have the three
// properties a transfinite blend actually relies on:
//
//   * they sum to one,
//   * shape function v is 1 at vertex v and 0 at every other vertex,
//   * shape function v vanishes identically on every facet that does not contain vertex v.
//
// For a simplex they are the barycentric coordinates; for a quad/brick the bi/trilinear shapes; for a
// wedge a (triangle barycentric) x (linear) product; for a pyramid the standard rational shapes. Used
// as *generalised barycentric coordinates* they give one formula for all shapes. Writing lambda_v(s)
// for them, X_v for the vertex positions, and for a sub-entity S with vertex set V(S)
//
//   w_S(s)    = sum over v in V(S) of lambda_v(s)             (1 on S, 0 at every vertex outside it)
//   sigma_S,v = lambda_v(s) / w_S(s)                          (the facet-local coordinate, when s is on S)
//   d_S       = C_S(sigma) - L_S(sigma)                       (curved image minus straight image)
//
// the macro map is
//
//   x(s) = sum_v lambda_v X_v  +  sum over curved facets F of w_F d_F  -  sum over edges E of (m_E - 1) w_E d_E
//
// where the last sum runs over edges shared by m_E >= 2 curved facets and is empty in 2d (two edges of
// a 2d element meet only at a vertex, where every d already vanishes). Since d_F vanishes at F's own
// vertices, a curved facet's contribution automatically dies out on the facets it does not touch.
//
// This is not a weaker substitute for the Coons blend -- it *is* the Coons blend. For a quad the
// deviations telescope (the bilinear corner interpolant is linear between its own edge restrictions in
// either direction, so (1-eta)L_S + eta*L_N = f_rect = (1-xi)L_W + xi*L_E) into
//
//   x = (1-eta)C_S + eta*C_N + (1-xi)C_W + xi*C_E - f_rect
//
// which is QMacroElement<2>::macro_map line for line; in 3d the same telescoping gives Gordon-Hall.
// Verified numerically against oomph's own implementation over the whole reference domain, interior
// included -- see tests/test_curved_boundaries.py and dev_docs/macro_elements_generalisation.md 10.1.
//
// Two things fall out for free. Facet node ordering stops mattering, because each weight is carried
// together with the vertex it belongs to, so the permutation search the previous implementation needed
// (a brute-force scan over all 4! orderings for a brick face) simply has no counterpart here. And a
// facet with no curved entity contributes nothing, which makes the adjacent faces of a partially
// curved 3d element come out as ruled surfaces carrying the neighbouring curved edge -- the
// geometrically right answer, and one the old flat-face treatment got wrong.

#pragma once

#include "oomph_lib.hpp"
#include "exception.hpp"

namespace pyoomph
{
  class MeshTemplateFacet;

  // Which reference shape a macro element's coordinates live in. This is the only shape-dependent
  // ingredient of the macro map: it selects the vertex shape functions.
  enum class MacroElementShape
  {
    Quad2d,
    Tri2d,
    Brick3d,
    Tet3d,
    Wedge3d,
    Pyramid3d
  };

  // Spatial dimension of a shape's reference domain.
  unsigned macro_shape_dim(const MacroElementShape &shape);

  // Number of vertices (C1 nodes) of the reference shape.
  unsigned macro_num_vertices(const MacroElementShape &shape);

  // Vertex (C1) shape functions at local coordinate s -- the generalised barycentric coordinates
  // described above. Deliberately a free function rather than a call into a BulkElementBase: the macro
  // map is evaluated at arbitrary points of the *root* element's reference domain, and routing that
  // through an element object would tie the macro element's lifetime to that element's.
  void macro_c1_shape(const MacroElementShape &shape, const oomph::Vector<double> &s, std::vector<double> &psi);

  // Reference-domain coordinates of the vertices, i.e. the macro coordinates of an unrefined element.
  void macro_reference_vertices(const MacroElementShape &shape, std::vector<std::vector<double>> &sv);

  // The shape's edges as vertex-index pairs. Needed because an element can touch a curved surface
  // along an edge without owning a face on it -- common in an unstructured tetrahedral mesh -- and
  // must then still curve that edge, or it would place new nodes on the chord while its neighbour
  // places them on the surface. Only genuine edges: a face diagonal joins two points of the surface
  // but is not an edge of the element, and curving it would bulge the interior.
  const std::vector<std::pair<unsigned, unsigned>> &macro_edges(const MacroElementShape &shape);

  // One curved facet attached to a macro element. `facet` owns both the curved entity and the
  // parametric coordinates of its nodes; it lives in MeshTemplate::facets, which outlives the mesh.
  struct MacroCurvedFacet
  {
    MeshTemplateFacet *facet;
    std::vector<unsigned> local_vertices;   // element-local C1 vertex indices lying on this facet
    std::vector<unsigned> parametric_index; // parallel to local_vertices: index into facet->parametrics
  };

  // The shape-generic macro element. Replaces MeshTemplateQMacroElement2/TMacroElement2/QMacroElement3
  // and oomph::TMacroElement.
  class GenericMacroElement : public oomph::MacroElement
  {
  protected:
    MacroElementShape shape;
    // The element's vertex positions in C1 shape-function order, stored BY VALUE.
    //
    // Node pointers would be the obvious choice -- the map would then follow the nodes through history
    // levels and, on a moving mesh, through the solve. They are also unsafe: Mesh::distribute() deletes
    // the elements and nodes a rank does not own, and a macro element belongs to the ROOT element and is
    // shared by every son, so a son surviving on this rank can be left holding a macro element whose
    // root vertices have been freed. classify_halo_and_haloed_nodes() then calls get_x() on it and the
    // rank segfaults. (The implementation this replaced had the same hazard for the same reason; it was
    // simply never exercised, as nothing combined --distribute with a curved boundary.)
    //
    // Copying costs nothing here: the macro element is only ever consulted for a mesh whose geometry is
    // fixed. On a moving mesh the Eulerian position is a dof and the macro element deliberately does not
    // drive it (see dev_docs/macro_elements_generalisation.md 18.2), so there is no motion to follow.
    std::vector<std::vector<double>> vertex_positions;
    std::vector<MacroCurvedFacet> curved_facets;
    // The inclusion-exclusion terms of the blend, one per sub-entity (in practice an edge) shared by
    // two or more curved facets, which would otherwise contribute its deviation once per facet. Each
    // carries the shared vertices, borrowed parametrics from one of the incident facets, and the
    // multiplicity m_E - 1 to subtract. Empty in 2d, where two facets meet only at a vertex and every
    // deviation already vanishes there. Rebuilt whenever a facet is added.
    std::vector<std::pair<MacroCurvedFacet, double>> edge_corrections;
    void rebuild_edge_corrections();

    // Position of vertex v, zero-padded to the requested dimension. There is no time level: see above.
    void vertex_position(const unsigned &v, const unsigned &dim, std::vector<double> &x) const;
    // Weight w_S and deviation d_S of one sub-entity at the given generalised barycentric coordinates.
    // Returns false when the weight vanishes, i.e. the sub-entity contributes nothing here.
    bool subentity_deviation(const MacroCurvedFacet &cf, const std::vector<double> &lambda,
                             const unsigned &t, const unsigned &dim,
                             double &w, std::vector<double> &deviation) const;


  public:
    GenericMacroElement(oomph::Domain *domain, const unsigned &index, const MacroElementShape &shape_,
                        const std::vector<oomph::Node *> &vertices);
    ~GenericMacroElement() override {}

    GenericMacroElement(const GenericMacroElement &) = delete;
    void operator=(const GenericMacroElement &) = delete;

    // Attach a curved facet. `local_vertices` are this element's C1 vertex indices lying on the facet,
    // `parametric_index[i]` says which of the facet's stored parametric coordinates belongs to
    // local_vertices[i]. No ordering convention is implied or required.
    void add_curved_facet(MeshTemplateFacet *facet, const std::vector<unsigned> &local_vertices,
                          const std::vector<unsigned> &parametric_index);
    unsigned ncurved_facets() const { return curved_facets.size(); }
    const std::vector<unsigned> &curved_facet_vertices(const unsigned &i) const { return curved_facets[i].local_vertices; }
    const MacroElementShape &get_shape() const { return shape; }

    void macro_map(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &r) override;

    // Sample the mapped geometry over the reference domain (debugging/visualisation of curved meshes).
    void output(const unsigned &t, std::ostream &outfile, const unsigned &nplot) override;
    void output_macro_element_boundaries(std::ostream &outfile, const unsigned &nplot) override;

    // d r_i / d s_j and its second derivative, by central differences of macro_map. The blend is built
    // from the curved entities' parametric_to_position, which is user-supplied (and, from Python, need
    // not be differentiable in closed form), so there is nothing analytic to differentiate here.
    void assemble_macro_to_eulerian_jacobian(const unsigned &t, const oomph::Vector<double> &s,
                                             oomph::DenseMatrix<double> &jacobian) override;
    void assemble_macro_to_eulerian_jacobian2(const unsigned &t, const oomph::Vector<double> &s,
                                              oomph::DenseMatrix<double> &jacobian2) override;
  };

}
