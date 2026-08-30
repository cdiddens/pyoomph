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


// Element geometry: normals and their derivatives with respect to the nodal coordinates (the
// shape-derivative terms the analytic Jacobian needs), element diameter and quality, and the
// face/vertex node enumeration.

#include "macroelements.hpp"
#include "elements.hpp"
#include "exception.hpp"
#include "problem.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "expressions.hpp"
#include "timestepper.hpp"

namespace pyoomph
{

	// Element shape-quality measure: ratio of the smallest Jacobian determinant encountered at any
	// integration point to the element's mean Jacobian (i.e. mean element size). A value close to 1
	// indicates a well-shaped, near-uniform element; values close to 0 indicate a nearly degenerate
	// (collapsed/inverted) element.
	double BulkElementBase::get_quality_factor()
	{
		double size = 0.0;
		double weightsum = 0.0;
		double minJ = 1e40;
		for (unsigned ipt = 0; ipt < integral_pt()->nweight(); ipt++)
		{
			oomph::Vector<double> s(this->dim());
			for (unsigned int i = 0; i < this->dim(); i++)
				s[i] = integral_pt()->knot(ipt, i);
			double J = this->J_eulerian(s);
			double w = integral_pt()->weight(ipt);
			weightsum += w;
			size += J * w;
			if (J < minJ)
			{
				minJ = J;
			}
		}		
		return minJ / (size / weightsum);
	}

	// Analytically differentiates the outer unit normal vector n=dx/ds x .../|...| (line tangent
	// rotated by 90 degrees in 2d, or cross product of the two surface tangents in 3d) with respect
	// to the nodal coordinates, and optionally its second derivative. This is a direct symbolic/
	// algebraic differentiation of the normal formula (not a finite-difference approximation): each
	// case below (1d line normal in 2d, 2d surface normal in 3d) computes d(tangent)/d(node
	// coordinate) via the shape function derivatives, then applies the quotient/product rule for
	// n = t/|t| (and its second derivative) directly in index notation. Used by generated code that
	// differentiates normal-dependent boundary conditions (e.g. surface tension, moving contact
	// lines) with respect to the ALE/solid mesh position.
	void BulkElementBase::get_dnormal_dcoords_at_s(const oomph::Vector<double> &s, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2) const
	{

		unsigned nodal_dim = this->nodal_dimension();
		unsigned eldim = this->dim();

		const unsigned n_node = this->nnode();

		if (nodal_dim == 2 && eldim == 1) // Normal of a line element
		{
			oomph::Shape psi(this->nnode());
			oomph::DShape dpsi(this->nnode(), eldim);
			this->dshape_local(s, psi, dpsi);
			std::vector<double> dxds(nodal_dim, 0);
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				for (unsigned d = 0; d < nodal_dim; d++)
				{
					dxds[d] += this->nodal_position(l, d) * dpsi(l, 0);
				}
			}
			double denom = 0.0;
			for (unsigned int d = 0; d < nodal_dim; d++)
				denom += dxds[d] * dxds[d];
			if (denom < 1e-20)
				denom = 1;
			double denom_sqr=denom;
			denom = sqrt(denom);
			denom = 1 / (denom * denom * denom);
			for (unsigned int i = 0; i < nodal_dim; i++)
			{
				for (unsigned l = 0; l < n_node; l++)
				{
					for (unsigned int k = 0; k < nodal_dim; k++)
					{						
						dnormal_dcoord[i][l][k] = dpsi(l, 0) * denom * (k == 1 ? -1 : 1) * dxds[i] * dxds[1 - k];
					}
				}
			}
			if (d2normal_dcoord2)
			{			   
				for (unsigned int i = 0; i < nodal_dim; i++)
			   {
					for (unsigned l = 0; l < n_node; l++)
					{
						for (unsigned int j = 0; j < nodal_dim; j++)
						{
							for (unsigned lp = 0; lp < n_node; lp++)
							{
								for (unsigned int jp = 0; jp < nodal_dim; jp++)
								{								 
								 d2normal_dcoord2[i][l][j][lp][jp]=(i==1 ? -1 : 1)*(dpsi(l,0)*dpsi(lp,0))*denom*(( (j==jp && j!=i) ? 3*dxds[1-i] : dxds[1-(j==i ? jp : j)])-3*dxds[1-i]*dxds[j]*dxds[jp]/denom_sqr);
								}												
							}
						}
					}
				}
				
			}
		}
		else if (nodal_dim==3 && eldim==2)
		{
			const unsigned n_node = this->nnode();
			oomph::Shape psi(n_node);
			oomph::DShape dpsids(n_node, 2);
			this->dshape_local(s, psi, dpsids);
			oomph::Vector<oomph::Vector<double>> interpolated_dxds(2, oomph::Vector<double>(3, 0));
			oomph::RankFourTensor<double> dinterpolated_dxds(2, 3, n_node, 3, 0.0);

			// Tangents depend on the interface only
			for (unsigned l = 0; l < n_node; l++)
			{
				for (unsigned j = 0; j < 2; j++)
				{
					for (unsigned i = 0; i < 3; i++)
					{
						interpolated_dxds[j][i] += this->nodal_position_gen(l, 0, i) * dpsids(l, j);
					}
				}
			}

			oomph::RankThreeTensor<double> EpsilonIJK(3, 3, 3, 0.0);
			EpsilonIJK(0, 1, 2) = 1;
			EpsilonIJK(0, 2, 1) = -1;
			EpsilonIJK(1, 2, 0) = 1;
			EpsilonIJK(1, 0, 2) = -1;
			EpsilonIJK(2, 0, 1) = 1;
			EpsilonIJK(2, 1, 0) = -1;

			oomph::Vector<double> normal(3, 0.0); // Non-normalized normal
			for (unsigned int i = 0; i < 3; i++)
			{
				for (unsigned int j = 0; j < 3; j++)
				{
					for (unsigned int k = 0; k < 3; k++)
					{
						normal[i] +=  EpsilonIJK(i, j, k) * interpolated_dxds[0][j] * interpolated_dxds[1][k];
					}
				}
			}

			for (unsigned int xl = 0; xl < n_node; xl++)
			{
				for (unsigned int xi = 0; xi < 3; xi++)
				{
					for (unsigned j = 0; j < 2; j++)
					{
						for (unsigned i = 0; i < 3; i++)
						{
							dinterpolated_dxds(j, i, xl, xi) += dpsids(xl, j) * (xi == i ? 1 : 0);
						}
					}
				}
			}

			oomph::RankThreeTensor<double> dndxlm(3, n_node, 3, 0.0);
			for (unsigned int i = 0; i < 3; i++)
			{
				for (unsigned int l = 0; l < n_node; l++)
				{
					for (unsigned int m = 0; m < 3; m++)
					{
						for (unsigned int j = 0; j < 3; j++)
						{
							for (unsigned int k = 0; k < 3; k++)
							{
								dndxlm(i, l, m) +=  EpsilonIJK(i, j, k) * (dinterpolated_dxds(0, m, l, j) * interpolated_dxds[1][k] + interpolated_dxds[0][j] * dinterpolated_dxds(1, m, l, k));
							}
						}
					}
				}
			}

			double nleng = 0.0;
			for (unsigned int i = 0; i < 3; i++)
				nleng += normal[i] * normal[i];
			nleng = sqrt(nleng);
			// However, since in 2d cases, the normal might depend on the pure bulk positions, we have to calc the derivatives for the bulk nodes, although may of them are zero
			for (unsigned i = 0; i < 3; i++)
			{
				for (unsigned int l = 0; l < n_node; l++)
				{
					for (unsigned int k = 0; k < 3; k++)
					{
						double crosssum = 0.0;
						for (unsigned int j = 0; j < 3; j++)
							crosssum += normal[j] * dndxlm(j, l, k);
						dnormal_dcoord[i][l][k] = dndxlm(i, l, k) / nleng - normal[i] / (nleng * nleng * nleng) * crosssum;
					}
				}
			}

			if (d2normal_dcoord2)
			{
				throw_runtime_error("Implement second order moving mesh coordinate derivatives of the normal here");
			}

		}
		else if (eldim==0 && nodal_dim==1)
		{ 
           //Actually, this does not mean anything, but we can set the derivative to zero
		   for (unsigned int i = 0; i < nodal_dim; i++)
			{
				for (unsigned l = 0; l < n_node; l++)
				{
					for (unsigned int k = 0; k < nodal_dim; k++)
					{
						dnormal_dcoord[i][l][k] = 0.0;
					}
				}
			}
			if (d2normal_dcoord2)
			{			
				for (unsigned int i = 0; i < nodal_dim; i++)
			   {
					for (unsigned l = 0; l < n_node; l++)
					{
						for (unsigned int j = 0; j < nodal_dim; j++)
						{
							for (unsigned lp = 0; lp < n_node; lp++)
							{
								for (unsigned int jp = 0; jp < nodal_dim; jp++)
								{		
								 d2normal_dcoord2[i][l][j][lp][jp]=0.0;
								}												
							}
						}
					}
				}
			}
		}
		else
		{

			for (unsigned int i = 0; i < nodal_dim; i++)
			{
				for (unsigned l = 0; l < n_node; l++)
				{
					for (unsigned int k = 0; k < nodal_dim; k++)
					{
						dnormal_dcoord[i][l][k] = 0.0;
					}
				}
			}
			std::cerr << "Cannot calculate a dnormal_dcoords for an element of dimension " + std::to_string(eldim) + " embedded in a space of dimension " + std::to_string(nodal_dim) + " yet" << std::endl << std::flush;
			throw_runtime_error("Cannot calculate a dnormal_dcoords for an element of dimension " + std::to_string(eldim) + " embedded in a space of dimension " + std::to_string(nodal_dim) + " yet");
		}
	}

	// Computes the outer unit normal n directly from the local tangent(s) (rotated tangent in 2d, cross
	// product of the two surface tangents in 3d), for the case where this element itself is
	// dimensionally a face (e.g. called directly on an interface element, as opposed to going
	// through oomph-lib's FaceElement::outer_unit_normal). Delegates the derivative computation to
	// get_dnormal_dcoords_at_s if requested.
	void BulkElementBase::get_normal_at_s(const oomph::Vector<double> &s, oomph::Vector<double> &n, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT* PYOOMPH_RESTRICT dnormal_dcoord, double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2normal_dcoord2, unsigned history_index) const
	{
		unsigned nodal_dim = this->nodal_dimension();
		unsigned eldim = this->dim();

		n.resize(nodal_dim);
		if (nodal_dim == 2 && eldim == 1) // Normal of a line element
		{
			oomph::Shape psi(this->nnode());
			oomph::DShape dpsi(this->nnode(), eldim);
			this->dshape_local(s, psi, dpsi);
			std::vector<double> dxds(nodal_dim, 0);
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				for (unsigned d = 0; d < nodal_dim; d++)
				{
					dxds[d] += this->nodal_position(history_index, l, d) * dpsi(l, 0);
				}
			}
			double l = 0.0;
			for (unsigned int d = 0; d < nodal_dim; d++)
				l += dxds[d] * dxds[d];
			if (l < 1e-20)
				l = 1;
			l = sqrt(l);
			n[0] = -dxds[1] / l;
			n[1] = dxds[0] / l;
		}
		else if (nodal_dim==3 && eldim==2)
		{
			oomph::Shape psi(this->nnode());
			oomph::DShape dpsi(this->nnode(), eldim);
			this->dshape_local(s, psi, dpsi);
			std::vector<double> dxds1(nodal_dim, 0);
			std::vector<double> dxds2(nodal_dim, 0);
			for (unsigned int l = 0; l < this->nnode(); l++)
			{
				for (unsigned d = 0; d < nodal_dim; d++)
				{
					dxds1[d] += this->nodal_position(history_index, l, d) * dpsi(l, 0);
					dxds2[d] += this->nodal_position(history_index, l, d) * dpsi(l, 1);
				}
			}
			n[0]=dxds1[1]*dxds2[2]-dxds1[2]*dxds2[1];
			n[1]=dxds1[2]*dxds2[0]-dxds1[0]*dxds2[2];
			n[2]=dxds1[0]*dxds2[1]-dxds1[1]*dxds2[0];
			double l = 0.0;
			for (unsigned int d = 0; d < nodal_dim; d++) l += n[d] * n[d];
			if (l < 1e-20)
				l = 1;
			l = sqrt(l);
			for (unsigned int d = 0; d < nodal_dim; d++) n[d] /=l;
		}
		else if (nodal_dim==1 && eldim==0)
		{
			n[0]=1.0; // Makes only partially sense, but for PointMesh with a Cartesian normal mode expansion, it matters
		}
		else
		{
			std::cerr <<("Cannot calculate a normal for an element of dimension " + std::to_string(eldim) + " embedded in a space of dimension " + std::to_string(nodal_dim) + " yet") <<std::endl << std::flush;
			throw_runtime_error("Cannot calculate a normal for an element of dimension " + std::to_string(eldim) + " embedded in a space of dimension " + std::to_string(nodal_dim) + " yet");
		}
		if (dnormal_dcoord)
		{
			this->get_dnormal_dcoords_at_s(s, dnormal_dcoord, d2normal_dcoord2);
		}
	}

    // TODO: Use the one defined in doi:10.1016/j.cma.2006.11.013
	// Estimates the element's characteristic size (diameter) from its vertex node positions: the
	// edge length for 1d elements, or an approximation based on the diagonal(s) between opposite
	// vertices for 2d/3d elements. Used e.g. for mesh-quality diagnostics and as a length scale in
	// stabilization terms.
	double BulkElementBase::get_element_diam() const
	{

		// Element size: Choose the max. diagonal
		double h = 0;
		if (this->dim() == 1)
		{
			h = std::fabs(this->vertex_node_pt(1)->x(0) -
						  this->vertex_node_pt(0)->x(0));
		}
		else if (this->dim() == 2)
		{
			h = pow(this->vertex_node_pt(3)->x(0) -
						this->vertex_node_pt(0)->x(0),
					2) +
				pow(this->vertex_node_pt(3)->x(1) -
						this->vertex_node_pt(0)->x(1),
					2);
			double h1 = pow(this->vertex_node_pt(2)->x(0) -
								this->vertex_node_pt(1)->x(0),
							2) +
						pow(this->vertex_node_pt(2)->x(1) -
								this->vertex_node_pt(1)->x(1),
							2);
			if (h1 > h)
				h = h1;
			h = sqrt(h);
		}
		else if (this->dim() == 3)
		{
			// diagonals are from nodes 0-7, 1-6, 2-5, 3-4
			unsigned n1 = 0;
			unsigned n2 = 7;
			for (unsigned i = 0; i < 4; i++)
			{
				double h1 = pow(this->vertex_node_pt(n1)->x(0) -
									this->vertex_node_pt(n2)->x(0),
								2) +
							pow(this->vertex_node_pt(n1)->x(1) -
									this->vertex_node_pt(n2)->x(1),
								2) +
							pow(this->vertex_node_pt(n1)->x(2) -
									this->vertex_node_pt(n2)->x(2),
								2);
				if (h1 > h)
					h = h1;
				n1++;
				n2--;
			}
			h = sqrt(h);
		}
		return h;
	}

	// --- Per-face boundary tags ------------------------------------------------------------------
	// See the declarations in elements.hpp for the rationale (single source of truth for boundary
	// element identification, seeded from the mesh template's facets, propagated forward on split).

	const std::vector<unsigned> *BulkElementBase::get_face_boundaries(const int &face_index) const
	{
		auto it = face_boundaries.find(static_cast<short>(face_index));
		if (it == face_boundaries.end()) return NULL;
		return &(it->second);
	}

	void BulkElementBase::set_face_boundaries(const int &face_index, const std::vector<unsigned> &boundaries)
	{
		if (boundaries.empty()) face_boundaries.erase(static_cast<short>(face_index));
		else face_boundaries[static_cast<short>(face_index)] = boundaries;
	}

	// Every node of a local face, in the face element's own ordering. The per-family tables live in the
	// nnode_on_face_by_index()/node_index_on_face() overrides (see elements.hpp); this is just the loop.
	std::vector<oomph::Node *> BulkElementBase::get_all_nodes_of_face(const int &face_index) const
	{
		const unsigned n = this->nnode_on_face_by_index(face_index);
		std::vector<oomph::Node *> res;
		res.reserve(n);
		for (unsigned i = 0; i < n; i++) res.push_back(const_cast<BulkElementBase *>(this)->node_pt(this->node_index_on_face(face_index, i)));
		return res;
	}

}
