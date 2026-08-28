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

// Second derivatives of the shape functions with respect to the LOCAL element coordinates.
//
// Everything here fills a d2psi indexed as d2psi(l, PYOOMPH_D2_SLOT(a,b)) - the full square layout
// defined in jitbridge.h - except the d2shape_local() overrides, which implement oomph-lib's own
// virtual and therefore have to speak oomph-lib's dimension-dependent N2deriv packing
// (1D {00}; 2D {00,11,01}; 3D {00,11,22,01,02,12}). BulkElementBase::d2shape_local_pyoomph is the
// single adapter between the two conventions.
//
// A space whose basis is affine in the local coordinates (P1 on simplices, DL everywhere) has
// identically vanishing local second derivatives. That does NOT make its physical second derivative
// zero: on a curved element the whole contribution comes from the -dpsi/dx_k * K K X_{k,ab} term.

#include "elements.hpp"
#include "elements_concrete.hpp"

namespace pyoomph
{

	// Scratch matrix in oomph-lib's N2deriv packing for an element of dimension el_dim.
	static inline oomph::DShape oomph_d2_scratch(unsigned el_dim, unsigned nnode)
	{
		return oomph::DShape(nnode, (el_dim == 1 ? 1 : (el_dim == 2 ? 3 : 6)));
	}

	// ==========================================================================================
	// Sub-spaces whose basis is affine in s: only psi and dpsi have to be evaluated.
	// ==========================================================================================

	void BulkElementLine1dC2::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1(s, psi, dpsi);
		zero_d2shape(d2psi, 2);
	}

	void BulkTElementLine1dC2::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1(s, psi, dpsi);
		zero_d2shape(d2psi, 2);
	}

	void BulkElementTri2dC2::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1(s, psi, dpsi);
		zero_d2shape(d2psi, 3);
	}

	void BulkElementTri2dC1TB::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1(s, psi, dpsi);
		zero_d2shape(d2psi, 3);
	}

	void BulkElementTri2dC2TB::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1(s, psi, dpsi);
		zero_d2shape(d2psi, 3);
	}

	void BulkElementTetra3dC2::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1(s, psi, dpsi);
		zero_d2shape(d2psi, 4);
	}

	void BulkElementTetra3dC1TB::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1(s, psi, dpsi);
		zero_d2shape(d2psi, 4);
	}

	void BulkElementTetra3dC2TB::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1(s, psi, dpsi);
		zero_d2shape(d2psi, 4);
	}

	// ==========================================================================================
	// Tensor-product sub-spaces on Q elements. The one-dimensional factors are linear, so every
	// pure second derivative vanishes and only the mixed ones survive.
	// ==========================================================================================

	void BulkElementQuad2dC2::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		double psi1[2], psi2[2], dpsi1[2], dpsi2[2];
		oomph::OneDimLagrange::shape<2>(s[0], psi1);
		oomph::OneDimLagrange::shape<2>(s[1], psi2);
		oomph::OneDimLagrange::dshape<2>(s[0], dpsi1);
		oomph::OneDimLagrange::dshape<2>(s[1], dpsi2);
		zero_d2shape(d2psi, 4);
		for (unsigned i = 0; i < 2; i++)
		{
			for (unsigned j = 0; j < 2; j++)
			{
				const unsigned ind = 2 * i + j;
				psi[ind] = psi2[i] * psi1[j];
				dpsi(ind, 0) = psi2[i] * dpsi1[j];
				dpsi(ind, 1) = dpsi2[i] * psi1[j];
				const double mixed = dpsi2[i] * dpsi1[j];
				d2psi(ind, PYOOMPH_D2_SLOT(0, 1)) = mixed;
				d2psi(ind, PYOOMPH_D2_SLOT(1, 0)) = mixed;
			}
		}
	}

	void BulkElementBrick3dC2::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		double psi1[2], psi2[2], psi3[2], dpsi1[2], dpsi2[2], dpsi3[2];
		oomph::OneDimLagrange::shape<2>(s[0], psi1);
		oomph::OneDimLagrange::shape<2>(s[1], psi2);
		oomph::OneDimLagrange::shape<2>(s[2], psi3);
		oomph::OneDimLagrange::dshape<2>(s[0], dpsi1);
		oomph::OneDimLagrange::dshape<2>(s[1], dpsi2);
		oomph::OneDimLagrange::dshape<2>(s[2], dpsi3);
		zero_d2shape(d2psi, 8);
		for (unsigned i = 0; i < 2; i++)
		{
			for (unsigned j = 0; j < 2; j++)
			{
				for (unsigned k = 0; k < 2; k++)
				{
					const unsigned ind = 4 * i + 2 * j + k;
					psi[ind] = psi3[i] * psi2[j] * psi1[k];
					dpsi(ind, 0) = psi3[i] * psi2[j] * dpsi1[k];
					dpsi(ind, 1) = psi3[i] * dpsi2[j] * psi1[k];
					dpsi(ind, 2) = dpsi3[i] * psi2[j] * psi1[k];
					const double m01 = psi3[i] * dpsi2[j] * dpsi1[k];
					const double m02 = dpsi3[i] * psi2[j] * dpsi1[k];
					const double m12 = dpsi3[i] * dpsi2[j] * psi1[k];
					d2psi(ind, PYOOMPH_D2_SLOT(0, 1)) = d2psi(ind, PYOOMPH_D2_SLOT(1, 0)) = m01;
					d2psi(ind, PYOOMPH_D2_SLOT(0, 2)) = d2psi(ind, PYOOMPH_D2_SLOT(2, 0)) = m02;
					d2psi(ind, PYOOMPH_D2_SLOT(1, 2)) = d2psi(ind, PYOOMPH_D2_SLOT(2, 1)) = m12;
				}
			}
		}
	}

	// ==========================================================================================
	// Wedge and pyramid: the C1 sub-space of a C2 element comes from the linear shape class.
	// ==========================================================================================

	void BulkElementWedge3dC2::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		oomph::DShape d2o = oomph_d2_scratch(3, 6);
		oomph::WedgeElementShapeC1::d2shape_local(s, psi, dpsi, d2o);
		remap_oomph_d2shape_packing(3, 6, d2o, d2psi);
	}

	void BulkElementPyramid3dC2::d2shape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		oomph::DShape d2o = oomph_d2_scratch(3, 5);
		oomph::PyramidElementShapeC1::d2shape_local(s, psi, dpsi, d2o);
		remap_oomph_d2shape_packing(3, 5, d2o, d2psi);
	}

	// ==========================================================================================
	// Bubble-enriched sub-spaces that are NOT the element's own space, so this->d2shape_local()
	// (which is the enriched one) must not be used.
	// ==========================================================================================

	void BulkElementTri2dC2TB::d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		oomph::DShape d2o = oomph_d2_scratch(2, 6);
		oomph::TElement<2, 3>::d2shape_local(s, psi, dpsi, d2o);
		remap_oomph_d2shape_packing(2, 6, d2o, d2psi);
	}

	void BulkElementTetra3dC2TB::d2shape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		oomph::DShape d2o = oomph_d2_scratch(3, 10);
		oomph::TElement<3, 3>::d2shape_local(s, psi, dpsi, d2o);
		remap_oomph_d2shape_packing(3, 10, d2o, d2psi);
	}

	// ==========================================================================================
	// pyoomph's own MINI bubble enrichments. oomph-lib knows nothing about these, so both the
	// d2shape_local() overrides and the C1TB sub-space evaluations are written out here.
	//
	// 2d: bubble b = s0*s1*s2 with s2 = 1-s0-s1, hence
	//        b_,00 = -2*s1     b_,11 = -2*s0     b_,01 = s2 - s0 - s1
	// 3d: bubble b = 256*s0*s1*s2*s3 with s3 = 1-s0-s1-s2, hence
	//        b_,00 = -512*s1*s2                  b_,01 = 256*s2*(s3 - s0 - s1)
	//        b_,11 = -512*s0*s2                  b_,02 = 256*s1*(s3 - s0 - s2)
	//        b_,22 = -512*s0*s1                  b_,12 = 256*s0*(s3 - s1 - s2)
	// ==========================================================================================

	// Second derivatives of the 2d MINI bubble, in the (00, 11, 01) order.
	static inline void tri_mini_bubble_d2(const oomph::Vector<double> &s, double d2b[3])
	{
		const double x = s[0], y = s[1], z = 1.0 - x - y;
		d2b[0] = -2.0 * y;
		d2b[1] = -2.0 * x;
		d2b[2] = z - x - y;
	}

	// Second derivatives of the 3d MINI bubble, in the (00, 11, 22, 01, 02, 12) order.
	static inline void tet_mini_bubble_d2(const oomph::Vector<double> &s, double d2b[6])
	{
		const double s0 = s[0], s1 = s[1], s2 = s[2], s3 = 1.0 - s0 - s1 - s2;
		d2b[0] = -512.0 * s1 * s2;
		d2b[1] = -512.0 * s0 * s2;
		d2b[2] = -512.0 * s0 * s1;
		d2b[3] = 256.0 * s2 * (s3 - s0 - s1);
		d2b[4] = 256.0 * s1 * (s3 - s0 - s2);
		d2b[5] = 256.0 * s0 * (s3 - s1 - s2);
	}

	// oomph-lib virtual -> N2deriv packing {00,11,01}
	void BulkElementTri2dC1TB::d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const
	{
		this->dshape_local(s, psi, dpsids);
		double d2b[3];
		tri_mini_bubble_d2(s, d2b);
		for (unsigned k = 0; k < 3; k++)
		{
			for (unsigned l = 0; l < 3; l++) d2psids(l, k) = -9.0 * d2b[k];
			d2psids(3, k) = 27.0 * d2b[k];
		}
	}

	// pyoomph virtual -> PYOOMPH_D2_SLOT packing
	void BulkElementTri2dC2TB::d2shape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1TB(s, psi, dpsi);
		double d2b[3];
		tri_mini_bubble_d2(s, d2b);
		static const unsigned slot[3][2] = {{0, 0}, {1, 1}, {0, 1}};
		zero_d2shape(d2psi, 4);
		for (unsigned k = 0; k < 3; k++)
		{
			const unsigned a = slot[k][0], b = slot[k][1];
			for (unsigned l = 0; l < 3; l++)
			{
				d2psi(l, PYOOMPH_D2_SLOT(a, b)) = -9.0 * d2b[k];
				d2psi(l, PYOOMPH_D2_SLOT(b, a)) = -9.0 * d2b[k];
			}
			d2psi(3, PYOOMPH_D2_SLOT(a, b)) = 27.0 * d2b[k];
			d2psi(3, PYOOMPH_D2_SLOT(b, a)) = 27.0 * d2b[k];
		}
	}

	// oomph-lib virtual -> N2deriv packing {00,11,22,01,02,12}
	void BulkElementTetra3dC1TB::d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const
	{
		this->dshape_local(s, psi, dpsids);
		double d2b[6];
		tet_mini_bubble_d2(s, d2b);
		for (unsigned k = 0; k < 6; k++)
		{
			for (unsigned l = 0; l < 4; l++) d2psids(l, k) = -0.25 * d2b[k];
			d2psids(4, k) = d2b[k];
		}
	}

	// pyoomph virtual -> PYOOMPH_D2_SLOT packing
	void BulkElementTetra3dC2TB::d2shape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_C1TB(s, psi, dpsi);
		double d2b[6];
		tet_mini_bubble_d2(s, d2b);
		static const unsigned slot[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
		zero_d2shape(d2psi, 5);
		for (unsigned k = 0; k < 6; k++)
		{
			const unsigned a = slot[k][0], b = slot[k][1];
			for (unsigned l = 0; l < 4; l++)
			{
				d2psi(l, PYOOMPH_D2_SLOT(a, b)) = -0.25 * d2b[k];
				d2psi(l, PYOOMPH_D2_SLOT(b, a)) = -0.25 * d2b[k];
			}
			d2psi(4, PYOOMPH_D2_SLOT(a, b)) = d2b[k];
			d2psi(4, PYOOMPH_D2_SLOT(b, a)) = d2b[k];
		}
	}

}
