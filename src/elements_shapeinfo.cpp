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


// Filling the JIT shape buffer: fill_shape_info_at_s and its nodal-position-derivative helper, the
// Lagrangian Jacobian, element sizes, and the per-integration-point buffer preparation, plus the
// local/integral expression evaluation and tracer geometry that ride on the same buffers.

#include "macroelements.hpp"
#include "elements.hpp"
#include "exception.hpp"
#include "problem.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "expressions.hpp"
#include "timestepper.hpp"

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <iostream>
#include <map>

namespace pyoomph
{

	// PYOOMPH_PARANOID_ALE_IDENTITY: re-derive the moving-mesh shape sensitivities from closed form and
	// compare against what was actually filled. The mapping x_i = sum_q X^q_i Psi_q(s) makes them pure
	// products of quantities that are already in the buffer,
	//
	//     d( D_i psi_l ) / dX^q_j  =  - (D_j psi_l)(D_i Psi_q)  +  N_ij (D psi_l . D Psi_q)
	//     d( dx )       / dX^q_j  =  dx * (D_j Psi_q)
	//     d( n_i )      / dX^q_j  =  - n_j (D_i Psi_q)
	//
	// with D the Eulerian (on an interface: surface) gradient, Psi the GEOMETRY space's shapes, and
	// N = I - P the projector onto the normal space - identically zero on a bulk element, where the
	// tangents span everything. The whole rank-4 d_dx_shape_dcoord array is therefore redundant data,
	// which is worth exploiting but only once it has been shown to hold for every geometry pyoomph
	// builds. Off by default and strictly read-only, so a build with it compiled in behaves exactly as
	// before while the variable is unset.
	static const bool __paranoid_ale_identity = getenv("PYOOMPH_PARANOID_ALE_IDENTITY") != NULL;

	// PYOOMPH_POISON_UNREQUIRED, see the declaration in elements.hpp: fill every shape buffer family
	// that the PASSED required-shapes struct does not ask for with signalling NaN. Same precedent and
	// same discipline as PYOOMPH_PARANOID_ALE_IDENTITY above - off by default, and when on it only ever
	// writes values nobody is supposed to read.
	static const char *__poison_unrequired_mode = getenv("PYOOMPH_POISON_UNREQUIRED");
	static const bool __poison_unrequired = __poison_unrequired_mode != NULL;
	// PYOOMPH_POISON_UNREQUIRED=all poisons every family AFTER the fill, at the end of each
	// integration point. That is the tool's own positive control: a poison that silently wrote
	// nothing, or that wrote into buffers nobody reads, would report "clean" on every case just as
	// loudly as a codebase with no under-requests. Poisoning required families BEFORE the fill proves
	// nothing - the fill simply overwrites it, which is what a first attempt at this control did.
	static const bool __poison_everything = __poison_unrequired && !strcmp(__poison_unrequired_mode, "all");
	// Engagement counter, for the same reason: "0 entries poisoned" is not a clean bill of health.
	static unsigned long __poison_written = 0;
	static struct __PoisonReport
	{
		~__PoisonReport()
		{
			if (!__poison_unrequired)
				return;
			std::cout << "PYOOMPH_POISON_UNREQUIRED: " << __poison_written << " buffer entries poisoned";
			if (!__poison_written)
				std::cout << " - NOTHING was poisoned, so nothing was proven";
			std::cout << std::endl;
		}
	} __poison_report;

	// Counted and reported at exit, because a check that is never reached passes just as quietly as one
	// that holds: a run that ends with "0 comparisons" has proven nothing.
	// Keyed by identity AND by the (el_dim, nodal_dim) pair, so the report proves which geometries were
	// actually exercised - a bulk-only run says nothing about the normal-projector term.
	struct AleIdentityStat
	{
		unsigned long n = 0, bad = 0;
		double worst = 0.0;
		std::string worst_where;
	};
	static std::map<std::string, AleIdentityStat> __ale_identity_stats;
	static struct __AleIdentityReport
	{
		~__AleIdentityReport()
		{
			if (!__paranoid_ale_identity)
				return;
			if (__ale_identity_stats.empty())
			{
				std::cout << "PYOOMPH_PARANOID_ALE_IDENTITY: NO comparisons were made - nothing was proven"
						  << std::endl;
				return;
			}
			for (const auto &e : __ale_identity_stats)
			{
				std::cout << "PYOOMPH_PARANOID_ALE_IDENTITY: " << e.first << " -> " << e.second.n
						  << " comparisons, " << e.second.bad << " violations, worst relative deviation "
						  << e.second.worst << std::endl;
				if (e.second.bad)
					std::cout << "    worst at " << e.second.worst_where << std::endl;
			}
		}
	} __ale_identity_report;

	// The second nodal-coordinate derivative of a shape gradient, in closed form (see
	// dev_docs/code_generation.md 9.4.13):
	//
	//   d2(D_i psi_l)/dX^q_j dX^r_k = (D_k psi_l)(D_j Psi_r)(D_i Psi_q) + (D_j psi_l)(D_k Psi_q)(D_i Psi_r)
	//                                 - N_jk (Dpsi_l.DPsi_r)(D_i Psi_q) - N_ik (D_j psi_l)(DPsi_q.DPsi_r)
	//                                 - N_ik (D_j Psi_r)(Dpsi_l.DPsi_q) - N_jk (D_i Psi_r)(Dpsi_l.DPsi_q)
	//                                 - N_ij (D_k psi_l)(DPsi_r.DPsi_q) - N_ij (D_k Psi_q)(Dpsi_l.DPsi_r)
	//
	// with N = I - P the projector onto the normal space, which vanishes on a bulk element and leaves
	// the first line. Symmetric under (q,j)<->(r,k), as a mixed second derivative has to be: the six
	// N-terms pair up. Arguments are the FIRST derivatives, all of which the shape buffer already has -
	// which is the point, since building this the other way (the E-tensor chain plus a rank-6 scratch
	// allocated per integration point) was 24.4% of Hessian assembly.
	// `with_normal_terms` is false on a bulk element, where N vanishes and only the first line survives.
	// It is a real distinction, not tidiness: evaluating the full form everywhere costs about four times
	// the arithmetic, and measured +6.9% on a case with 144 bulk elements to 24 interface ones, because
	// the common case was paying for the rare one. The caller skips computing N and the dot products
	// entirely in that case.
	static const double __no_normal_projector[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

	static inline double d2_shape_dcoord_closed_form(unsigned i, unsigned j, unsigned j2,
													 const double *Dpsi_l, const double *Dq, const double *Dr,
													 bool with_normal_terms, const double Nproj[3][3],
													 double dot_lr, double dot_lq, double dot_qr)
	{
		double v = Dpsi_l[j2] * Dr[j] * Dq[i] + Dpsi_l[j] * Dq[j2] * Dr[i];
		if (with_normal_terms)
			v += -Nproj[j][j2] * dot_lr * Dq[i] - Nproj[i][j2] * Dpsi_l[j] * dot_qr
				 - Nproj[i][j2] * Dr[j] * dot_lq - Nproj[j][j2] * Dr[i] * dot_lq
				 - Nproj[i][j] * Dpsi_l[j2] * dot_qr - Nproj[i][j] * Dq[j2] * dot_lr;
		return v;
	}

	static void ale_identity_check(const char *what, unsigned el_dim, unsigned nodal_dim, double got, double expected, const std::string &where)
	{
		// The sensitivities scale like 1/h^2, so a purely relative tolerance is the right one; the
		// additive 1 keeps entries that ought to be zero (every bulk N-term) from tripping on noise.
		const double scale = 1.0 + std::fabs(got) + std::fabs(expected);
		const double rel = std::fabs(got - expected) / scale;
		std::ostringstream key;
		key << what << " [el_dim " << el_dim << " in " << nodal_dim << "D]";
		auto &e = __ale_identity_stats[key.str()];
		e.n++;
		if (rel > 1e-8)
			e.bad++;
		if (rel > e.worst)
		{
			e.worst = rel;
			std::ostringstream oss;
			oss << where << ": filled " << got << ", closed form gives " << expected;
			e.worst_where = oss.str();
		}
	}

	// Convenience overload that fills the element's own (default) shape_info buffer; see the
	// shape_info-taking overload below for the actual work.
	double BulkElementBase::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds,unsigned history_index) const
	{
		return fill_shape_info_at_s(s, index, required, this->shape_info, JLagr, flag, dxds,history_index);
	}

	/**
	 * When the mesh moves, we must fill in additional buffer arrays in the shape_info for the Jacobian.
	 *
	 * `interpolated_t` stores the tangent vectors, i.e.
	 *     interpolated_t(j:element_dim,i:nodal_dim)=sum_[l:numnodes] ( x^l_i * dpsi^l/ds_j )
	 *
	 * `dpsids_Element` stores the local shape derivatives (element space, i.e. max FE space in the element, C2TB/C2/C1)
	 *     dpsids_Element(l:nnode,j:element_dim )= dpsi^l/ds_j
	 *
	 * `det_Eulerian`=sqrt(det(g_{ab})) with the metric tensor g_{ab}= g(a:element_dim, b:element_dim) = sum_[i:nodal_dim] ( interpolated_t(a, i) * interpolated_t(b, i) )
	 *
	 * `aup` is the inverse of the metric tensor, i.e. g^{ab}
	 *
	 * `DXdshape_il_jb' is the resulting rank-4-tensor DXdshape(i:nodal_dim,l:numnodes,j:nodal_dim,b:element_dim).
	 *    It must return d(g^{ab}g_{a,j})/d(x_i^l) (summed over a[element_dim]) with the inverse metric tensor g^{ab} and the tangent g_{a,j}=interpolated_t(a,j)
	 *
	 * @param shape_info The destination shape information buffer
	 * @param interpolated_t local tangent vectors of the element at the integration index
	 * @param dpsids_Element stores the local shape derivatives with respect to the intrinsic coordinate s
	 * @param det_Eulerian stores the determinant of the transformation from intrinsic coordinate s to Eulerian coordinate x
	 * @param aup inverse of the metric tensor
	 * @param require_hessian indicates whether we require second order derivatives
	 * @param DXdshape_il_jb rank-4-tensor which is returned
	 */

	void BulkElementBase::fill_shape_info_at_s_dNodalPos_helper(JITShapeInfo_t *shape_info, const unsigned &index, const oomph::DenseMatrix<double> &interpolated_t, const oomph::DShape &dpsids_Element, const double det_Eulerian, const oomph::DenseMatrix<double> &aup, bool require_hessian, oomph::RankFourTensor<double> &DXdshape_il_jb,RankSixTensor * D2X2_dshape) const
	{
		unsigned el_dim = this->dim();
		unsigned n_dim = this->nodal_dimension();
		unsigned n_node = this->nnode();


		// The spatial integral contribution `dx` of the Gauss-Legendre is given by dx=det_Eulerian*integral_pt()->weight(index);
		// In particular, you get the size (length/area/volume) of the element by summing dx over all Gauss-Legendre integration points
		// If the mesh moves, dx depends on the coordinates x^l_i and we require the derivatives of dx with respect to the coordinate dofs x^l_i, i.e. i-th coordinate component of the l-th node in the element
		double dshape_dx[n_dim][n_node];
		for (unsigned l = 0; l < n_node; l++)
		{
			for (unsigned i = 0; i < n_dim; i++)
			{
				// Variable to store the information of the derivative of shape function wrt x coordinates:
				// dpsi^l/dx_i = sum_b^eldim( sum_a^eldim( g^ab * dpsi^l/ds^a * t_bi ) ). This will be used
				// for Hessian calculation.
				dshape_dx[i][l] = 0.0;

				// Store information for the derivative of dx with respect to the x coordinates:
				// d/dx^l_i(dx).
				shape_info->int_pt_weights_d_coords[i][l] = 0.0;

				for (unsigned a = 0; a < el_dim; a++)
				{
					for (unsigned b = 0; b < el_dim; b++)
					{
						dshape_dx[i][l] += aup(a, b) * dpsids_Element(l, b) * interpolated_t(a, i);
					}
				}

				// This derivative expands into:
				// sum_b^eldim( sum_a^eldim( g^ab * dpsi^l/ds^a * t_bi ) ) * sqrt(det(g^ab)) * weight(index), or simply:
				// dshape_dx[i][l] * det_Eulerian * sqrt(det(g^ab)) * weight(index).
				shape_info->int_pt_weights_d_coords[i][l] = dshape_dx[i][l] * det_Eulerian * integral_pt()->weight(index);
			}
		}
		
		// Helper tensors
		// T^{l}_{gdj}=T[l][g][d][j]
		double T[n_node][el_dim][el_dim][n_dim];
		// G^{lab}_j=G[l][a][b][j]
		double G[n_node][el_dim][el_dim][n_dim];
		
		//Fill the T tensor
		for (unsigned int l=0;l<n_node;l++)
		{
		 for (unsigned int c=0;c<el_dim;c++)
		 {
		  for (unsigned int d=0;d<el_dim;d++)
		  {
			for (unsigned int j=0;j<n_dim;j++)
			{
			  T[l][c][d][j]=dpsids_Element(l,c)*interpolated_t(d, j)+dpsids_Element(l,d)*interpolated_t(c, j);
			}
		  }
		 }
		}
		
		//Fill the G tensor
		for (unsigned int l=0;l<n_node;l++)
		{
		 for (unsigned int a=0;a<el_dim;a++)
		 {
		  for (unsigned int b=0;b<el_dim;b++)
		  {
			for (unsigned int j=0;j<n_dim;j++)
			{
			  double Gval=0.0;
			  for (unsigned int c=0;c<el_dim;c++)
			  {
			   for (unsigned int d=0;d<el_dim;d++)
			   {
			     Gval-=aup(a,c)*T[l][c][d][j]*aup(d,b);
			   }
			  }
			  G[l][a][b][j]=Gval;
			}
		  }
		 }
		}
		


		for (unsigned i = 0; i < n_dim; i++)
		{
			for (unsigned l = 0; l < n_node; l++)
			{				
				for (unsigned j = 0; j < n_dim; j++)
				{					
						for (unsigned b = 0; b < el_dim; b++)
						{
							DXdshape_il_jb(i, l, j, b) = 0.0;
							for (unsigned a = 0; a < el_dim; a++)
							{
								if (i == j)
									DXdshape_il_jb(i, l, j, b) += aup(a, b) * dpsids_Element(l, a);		
								DXdshape_il_jb(i, l, j, b) += interpolated_t(a, j) * G[l][a][b][i]; // d(g^{ab})/d(X_i^l);
							}
						}					
				}
				
			}
		}

		if (require_hessian)
		{
		 // Only the E tensor below is replaced by the closed form of 9.4.12; the rest of this block
		 // computes D_dshape_Dcoords and, from it, int_pt_weights_d2_coords - the second derivative of
		 // the integration measure, which the Hessian needs whether or not the E tensor is built.
		 // Gating the whole block on D2X2_dshape silently dropped that and made every moving-mesh
		 // Hessian wrong by a few percent; the identity check did not catch it, because it validates
		 // the E-tensor path that the flag keeps alive.
		 if (D2X2_dshape)
		 {
		   // Fill the E tensor. Note: D in the document is accessed as follows:
		   //  	$D^{lb}_{ij}=DXdshape_il_jb(j,l,i,b)
		   
		   //fill E^{ll'beta}_{ijj'}=E_hess[i][beta][l][l'][j][j']
		   for (unsigned int i=0;i<n_dim;i++) 
		   {
		     for (unsigned int b=0;b<el_dim;b++)
		     {
		       for (unsigned int l=0;l<n_node;l++)
		       {		       
		        for (unsigned int lp=0;lp<n_node;lp++)
		        {
		          for (unsigned int j=0;j<n_dim;j++)
		          {
		            for (unsigned int jp=0;jp<n_dim;jp++)
		            {
		             double Eval=0.0;
		             // First term: -D^{l'c}_{ij'}T^l_{cdj}*g^{db}
		             // and third term: -g^{ac}T^l_{cdj}*G^{l'db}_j*t_{a,i}
		             for (unsigned int c=0;c<el_dim;c++)
		             {
		              for (unsigned int d=0;d<el_dim;d++)
		              {
		               double asum=0.0;
		               for (unsigned int a=0;a<el_dim;a++)
		               {
		                asum+=aup(a,c)*G[lp][d][b][jp]*interpolated_t(a,i);
		               }
		               Eval-=T[l][c][d][j]*(DXdshape_il_jb(jp,lp,i,c)*aup(d,b) + asum);
		              }
		             }
		             // Second term, only if j=jp:
		             if (j==jp)
		             {
		               for (unsigned int c=0;c<el_dim;c++)
		               {
		                for (unsigned int d=0;d<el_dim;d++)
		                {
		                 for (unsigned int a=0;a<el_dim;a++)
		                 {
		                  Eval-=aup(a,c)*(dpsids_Element(l, c)*dpsids_Element(lp, d)+dpsids_Element(lp, c)*dpsids_Element(l, d))*aup(d,b)*interpolated_t(a,i);
		                 }
		                }
		               }
		             }
		             // Last term, only if i==j
		             if (i==j)
		             {
		              for (unsigned int a=0;a<el_dim;a++)
		              {
		                Eval+=G[lp][a][b][jp]*dpsids_Element(l,a);
		              }
		             }
		             
		             (*D2X2_dshape)(i,b,l,lp,j,jp)=Eval;

		            }
		          }
		        }
		       
		       }
		       /*
		       // Test whether it is symmetric - it should be and apparently is
		       for (unsigned int l=0;l<n_node;l++)
		       {
		        for (unsigned int lp=0;lp<n_node;lp++)
		        {		       		       		     		          
		          for (unsigned int j=0;j<n_dim;j++)
		          {
		            for (unsigned int jp=0;jp<n_dim;jp++)
		            {
		              double E1=(*D2X2_dshape)(i,b,l,lp,j,jp);
		              double E2=(*D2X2_dshape)(i,b,lp,l,jp,j);
		              double diff=E1-E2;
		              if (diff*diff>1e-6)
		              {
		                std::cout << "E["<<i<<"]["<<b<<"]  ["<<l<<"]["<<lp<<"]["<<j<<"]["<<jp<<"] = " <<E1 << " and " << E2 << "for (l,j)<->(l',j') " << std::endl;		              
		              }
		            }
		            
		          }
		        }
		       }
		       */
		     }
		   }
		 }


			// Variable to store the second derivatives of shape function wrt to coordinates, i.e.,
			// D_dshape_Dcoords[i][l][j][k] = d/dx_i^l(dpsi^k/dx_j). This can be developed into:
			// sum_b^eldim( dpsi^k/ds^b * DXdshape_il_jb(i, l, j, b) ). Used for Hessian purposes.
			double D_dshape_Dcoords[n_dim][n_node][n_dim][n_node];
			for (unsigned int i = 0; i < n_dim; i++)
			{
				for (unsigned int l = 0; l < n_node; l++)
				{
					for (unsigned int j = 0; j < n_dim; j++)
					{
						for (unsigned int k = 0; k < n_node; k++)
						{
							D_dshape_Dcoords[i][l][j][k] = 0.0;
							for (unsigned int b = 0; b < el_dim; b++)
							{
								D_dshape_Dcoords[i][l][j][k] += dpsids_Element(l, b) * DXdshape_il_jb(j, k, i, b);
								//			           std::cout << "ACCU " << i <<  " " << l << "  " << j << "  " << k << "  " << D_dshape_Dcoords[i][l][j][k] <<std::endl;
							}
						}
					}
				}
			}

			for (unsigned i = 0; i < n_dim; i++)
			{

				for (unsigned j = 0; j < n_dim; j++)
				{

					for (unsigned l = 0; l < n_node; l++)
					{

						for (unsigned k = 0; k < n_node; k++)
						{

							// The derivative of dshape_dx[i][l] * det_Eulerian * sqrt(det(g^ab))
							// wrt to the coordinates x_j^m should then be given by, applying the chain rule:
							// det_Eulerian * D_dshape_Dcoords[i][l][j][k] + (det_Eulerian * dshape_dx[j][k]) * dshape_dx[i][l],
							// where the quantities in paranthesis on the last term corresponds to the derivative of det_Eulerian
							// wrt the coordinates.
							shape_info->int_pt_weights_d2_coords[i][j][l][k] = integral_pt()->weight(index) * det_Eulerian * (dshape_dx[i][l] * dshape_dx[j][k] + D_dshape_Dcoords[i][l][j][k]);
						}
					}
				}
			}
		}
	}

	// Determinant of the Lagrangian (reference/undeformed) metric tensor at local coordinate s,
	// i.e. the analogue of the usual Eulerian Jacobian but built from the Lagrangian nodal
	// positions xi() instead of the (possibly moving) Eulerian positions. Used to integrate over
	// the reference configuration, e.g. for Lagrangian element-size or elasticity formulations.
	double BulkElementBase::J_Lagrangian(const oomph::Vector<double> &s)
	{
		unsigned el_dim = this->dim();
		unsigned n_node = this->nnode();
		unsigned n_lagr = this->nlagrangian();

		//std::cout << "NLAGR " << n_lagr << "  " << el_dim << std::endl;
		oomph::Shape psi_Element(n_node);
		oomph::DShape dpsids_Element(n_node, std::max((unsigned int)1, el_dim));
		this->dshape_local(s, psi_Element, dpsids_Element);
		oomph::DenseMatrix<double> interpolated_T(el_dim, n_lagr, 0.0);
		for (unsigned l = 0; l < n_node; l++)
		{
			for (unsigned i = 0; i < n_lagr; i++)
			{
				for (unsigned j = 0; j < el_dim; j++)
				{
					// interpolated_T(j,i) += static_cast<pyoomph::Node*>(this->node_pt(l))->xi(i)*dpsids_Element(l,j);
					interpolated_T(j, i) += this->raw_lagrangian_position_gen(l, 0, i) * dpsids_Element(l, j);
				}
			}
		}

		if (el_dim == 1)
		{
			double a11 = 0.0;
			for (unsigned int i = 0; i < n_lagr; i++)
				a11 += interpolated_T(0, i) * interpolated_T(0, i);
			return sqrt(a11);
		}
		else if (el_dim == 2)
		{
			double amet[2][2];
			for (unsigned al = 0; al < 2; al++)
			{
				for (unsigned be = 0; be < 2; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_lagr; i++)
					{
						amet[al][be] += interpolated_T(al, i) * interpolated_T(be, i);
					}
				}
			}
			double det_a = amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0];
			return sqrt(det_a);
		}
		else if (el_dim == 0)
		{
			return 1;
		}
		else if (el_dim == 3)
		{

			double amet[3][3];
			for (unsigned al = 0; al < 3; al++)
			{
				for (unsigned be = 0; be < 3; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_lagr; i++)
					{
						amet[al][be] += interpolated_T(al, i) * interpolated_T(be, i);
					}
				}
			}
			double det_a = amet[0][0] * amet[1][1] * amet[2][2] + amet[0][1] * amet[1][2] * amet[2][0] + amet[0][2] * amet[1][0] * amet[2][1] - amet[0][0] * amet[1][2] * amet[2][1] - amet[0][1] * amet[1][0] * amet[2][2] - amet[0][2] * amet[1][1] * amet[2][0];
			return sqrt(det_a);
		}
		else
		{
			throw_runtime_error("Implement for this dimension");
			return 1;
		}

		return 1;
	}
	
	
	// Computes element-size related quantities requested via `required` (Eulerian/Lagrangian
	// element size, in both the "physical" and Cartesian-only sense, i.e. without any extra
	// geometric_jacobian/JacobianForElementSize weighting) by integrating over all knots, and,
	// if the mesh moves (flag!=0), also their derivatives with respect to nodal coordinates
	// (and, if flag indicates a Hessian is required, second derivatives). These are needed
	// because element-size expressions in the generated code are not assembled point-wise like
	// normal residuals but require a pre-integrated scalar (and its coordinate sensitivities).
	void BulkElementBase::fill_shape_info_element_sizes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info,unsigned flag, unsigned history_index) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		bool require_hessian = flag > 2;
		bool require_dxdshape = (flag && functable->moving_nodes && (!functable->fd_position_jacobian)); //&& (required.dx_psi_C2 || required.dx_psi_C1 || required.dx_psi_DL)			
		bool require_dx_elemsize=require_dxdshape && (required.elemsize_Eulerian ||  required.elemsize_Eulerian_cartesian);
		// The two families are separate flags on the generated code's side (elemsize_Eulerian vs
		// elemsize_Eulerian_cartesian, chosen by the symbol's is_with_coordsys()), so a code asking only
		// for the coordinate-system-weighted size must not also pay for the Cartesian sensitivities.
		const bool need_eul_d = required.elemsize_Eulerian;
		const bool need_cart_d = required.elemsize_Eulerian_cartesian;
		if (require_dx_elemsize)
		{
		 // Fill the derivative buffer
		 for (unsigned int i=0;i<this->nodal_dimension();i++)
		 {
		  	for (unsigned int l=0;l<this->nnode();l++)
		   {
		    if (need_cart_d) shape_info->elemsize_Cart_d_coords[i][l]=0.0;
		    if (need_eul_d) shape_info->elemsize_d_coords[i][l]=0.0;
		    if (require_hessian)
		    {
				for (unsigned int j=0;j<this->nodal_dimension();j++)
			 	{
			  		for (unsigned int m=0;m<this->nnode();m++)
					{
		      		if (need_eul_d) shape_info->elemsize_d2_coords[i][j][l][m]=0.0;
  		      		if (need_cart_d) shape_info->elemsize_Cart_d2_coords[i][j][l][m]=0.0;
  		      	}
  		      }
  		    }
		   }
		 }
		 JITFuncSpec_RequiredShapes_FiniteElement_t req_dummy;
		 memset(&req_dummy,0,sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
		 req_dummy.Pos.psi=req_dummy.Pos.dx_psi=true; // Calculate these
		 double JLagr;
       for(unsigned ipt_for_esize=0;ipt_for_esize<integral_pt()->nweight();ipt_for_esize++)		 
       {  
          //double w = integral_pt()->weight(ipt_for_esize);
          oomph::Vector<double> s_for_esize(this->dim());
          for (unsigned int _i = 0; _i < this->dim(); _i++)	s_for_esize[_i] = integral_pt()->knot(ipt_for_esize, _i);
          this->fill_shape_info_at_s(s_for_esize,0,req_dummy, JLagr, flag,NULL,0); // TODO: Potentially other history indices here
          oomph::Vector<double> x_for_esize(this->nodal_dimension(),0.0);
          std::vector<double> dJdx(this->nodal_dimension(),0.0);
          std::vector<double> d2Jdx2(this->nodal_dimension()*this->nodal_dimension(),0.0);   
          double J=1.0;       
          if (required.elemsize_Eulerian)
          {          
            this->interpolated_x(s_for_esize,x_for_esize);
            J=functable->JacobianForElementSize(&eleminfo, &(x_for_esize[0]));
            if (functable->JacobianForElementSizeSpatialDerivative && flag) 
            {
              functable->JacobianForElementSizeSpatialDerivative(&eleminfo, &(x_for_esize[0]),&(dJdx[0]));
              if (functable->JacobianForElementSizeSecondSpatialDerivative && require_hessian) 
              {
                functable->JacobianForElementSizeSecondSpatialDerivative(&eleminfo, &(x_for_esize[0]),&(d2Jdx2[0]));              
              }
            }
          }                      
			 for (unsigned int i=0;i<this->nodal_dimension();i++)
			 {
			  	for (unsigned int l=0;l<this->nnode();l++)
				{
				 if (need_cart_d) shape_info->elemsize_Cart_d_coords[i][l]+=shape_info->int_pt_weights_d_coords[i][l];
				 if (required.elemsize_Eulerian)
				 {
				   shape_info->elemsize_d_coords[i][l]+=shape_info->int_pt_weights_d_coords[i][l]*J;				  
				   shape_info->elemsize_d_coords[i][l]+=shape_info->int_pt_weight[0]*dJdx[i]*shape_info->shape_Pos[l];
				 }
				 if (require_hessian)
				 {
					for (unsigned int j=0;j<this->nodal_dimension();j++)
				 	{
				  		for (unsigned int m=0;m<this->nnode();m++)
						{
	  		      		if (need_cart_d) shape_info->elemsize_Cart_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d2_coords[i][j][l][m];
	  		      		if (required.elemsize_Eulerian)
				         {
					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d2_coords[i][j][l][m]*J;
					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weight[0]*d2Jdx2[i*this->nodal_dimension()+j]*shape_info->shape_Pos[l]*shape_info->shape_Pos[m];  
					   		shape_info->elemsize_d2_coords[i][j][l][m]+=shape_info->int_pt_weights_d_coords[i][l]*dJdx[j]*shape_info->shape_Pos[m];
					   		
				         }
	  		      	}
	  		      }
	  		    }				 
				}
			 }          
       }
		}
		
		
		
		
		// For a previous configuration the determinant has to be rebuilt from the old nodal positions.
		// fill_shape_info_at_s already does exactly that and returns it; it also writes that history
		// slot of the buffer as it goes, which is harmless because the per-integration-point pass
		// rewrites the slot afterwards at the point that actually matters.
		JITFuncSpec_RequiredShapes_FiniteElement_t minimal_for_history;
		memset(&minimal_for_history, 0, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));
		auto eulerian_J_at_s = [&](const oomph::Vector<double> &s_at, unsigned ipt_at, unsigned hist) -> double
		{
			if (hist == 0) return J_eulerian_at_knot(ipt_at);
			double JLagr_dummy_hist;
			return fill_shape_info_at_s(s_at, ipt_at, minimal_for_history, shape_info, JLagr_dummy_hist, 0, NULL, hist);
		};
		if (required.elemsize_Eulerian || required.elemsize_Lagrangian)
		{
        //TODO: A bit redundant to do this for each integration point -> Move it in some other routine
		  shape_info->elemsize_Eulerian[history_index]=0.0;
		  shape_info->elemsize_Lagrangian=0.0;		  
        for(unsigned ipt_for_esize=0;ipt_for_esize<integral_pt()->nweight();ipt_for_esize++)
        {
          double w = integral_pt()->weight(ipt_for_esize);
          oomph::Vector<double> s_for_esize(this->dim());
          for (unsigned int _i = 0; _i < this->dim(); _i++)	s_for_esize[_i] = integral_pt()->knot(ipt_for_esize, _i);
          oomph::Vector<double> x_for_esize(this->nodal_dimension(),0.0);
          if (required.elemsize_Eulerian)
          {          
            this->interpolated_x(history_index,s_for_esize,x_for_esize);
            double J = eulerian_J_at_s(s_for_esize,ipt_for_esize,history_index);
            shape_info->elemsize_Eulerian[history_index] += w*J*functable->JacobianForElementSize(&eleminfo, &(x_for_esize[0]));
          }
          if (required.elemsize_Lagrangian)
          {
            this->interpolated_xi(s_for_esize,x_for_esize);
            double J = J_lagrangian_at_knot(ipt_for_esize);            
            shape_info->elemsize_Lagrangian += w*J*functable->JacobianForElementSize(&eleminfo, &(x_for_esize[0]));          
          }
        }
		}
		if (required.elemsize_Eulerian_cartesian || required.elemsize_Lagrangian_cartesian)
		{
		  shape_info->elemsize_Eulerian_cartesian[history_index]=0.0;
		  shape_info->elemsize_Lagrangian_cartesian=0.0;		  
        for(unsigned ipt_for_esize=0;ipt_for_esize<integral_pt()->nweight();ipt_for_esize++)
        {
          double w = integral_pt()->weight(ipt_for_esize);
          oomph::Vector<double> s_for_esize(this->dim());
          for (unsigned int _i = 0; _i < this->dim(); _i++)	s_for_esize[_i] = integral_pt()->knot(ipt_for_esize, _i);
          oomph::Vector<double> x_for_esize(this->nodal_dimension(),0.0);
          if (required.elemsize_Eulerian_cartesian)
          {          
            this->interpolated_x(history_index,s_for_esize,x_for_esize);
            double J = eulerian_J_at_s(s_for_esize,ipt_for_esize,history_index);
            shape_info->elemsize_Eulerian_cartesian[history_index] += w*J;
          }
          if (required.elemsize_Lagrangian_cartesian)
          {
            this->interpolated_xi(s_for_esize,x_for_esize);
            double J = J_lagrangian_at_knot(ipt_for_esize);            
            shape_info->elemsize_Lagrangian_cartesian += w*J;          
          }
        }
		}	
		
		if ( this->as_interface_element())
		{
			if (required.bulk_shapes)
			{
			 const BulkElementBase *bel = dynamic_cast<const BulkElementBase *>(this->as_interface_element()->bulk_element_pt());
			 bel->fill_shape_info_element_sizes(*(required.bulk_shapes),shape_info->bulk_shapeinfo,flag,history_index);		 
			}
			
			if (required.opposite_shapes)
			{
			 const BulkElementBase *opp = this->as_interface_element()->get_opposite_side(); // an InterfaceElementBase already IS a BulkElementBase
			 opp->fill_shape_info_element_sizes(*(required.opposite_shapes),shape_info->opposite_shapeinfo,flag,history_index);		 
			}
	   }	   
	}

	// Central shape-function evaluator: at the given local coordinate s, fills the shape_info
	// buffer with the values and physical (x/X) derivatives of every field space present in the
	// element (C2TB, C2, C1TB, C1, DL, ...), plus (if the mesh moves) the sensitivities of those
	// derivatives with respect to nodal coordinates, and (if a Hessian is requested) the second
	// such sensitivities -- everything the JIT-generated residual/Jacobian/Hessian code needs at
	// this integration/evaluation point. Returns the Eulerian Jacobian determinant det_Eulerian
	// and, via the JLagr reference parameter, the Lagrangian one.
	//
	// Strategy: first build the (inverse) metric tensor from the tangent vectors, both in the
	// Eulerian (interpolated_t) and Lagrangian (interpolated_T) configuration -- separately for
	// el_dim 0/1/2/3, since the metric determinant/inverse formulas differ by dimension -- along
	// with gab_gai[b][i] = g^{ab} g_{a,i}, the contraction used below to turn local-coordinate
	// shape derivatives dpsi/ds into physical ones dpsi/dx. If the mesh can move, also computes
	// DXdshape_il_jb = d(g^{ab} g_{a,j})/d(x_i^l) (and, for Hessians, D2X2_dshape) via
	// fill_shape_info_at_s_dNodalPos_helper(), which are then reused for every field space below.
	// Then, for each present field space, evaluates its local shape functions/derivatives
	// (dshape_local_at_s_XXX) and combines them with gab_gai (and DXdshape_il_jb) to fill the
	// corresponding shape_info->shape_XXX / dx_shape_XXX / dX_shape_XXX / dS_shape_XXX /
	// d_dx_shape_dcoord_XXX / d2_dx2_shape_dcoord_XXX arrays -- but only if that space is
	// actually required (per `required`) to avoid needless work.
	// Sensitivities of M_i^(b) = g^{ab} t_{a,i} and of Q[i][b][c] = dM_i^(b)/ds_c with respect to the
	// nodal coordinates X^m_p. Written out from
	//    dg_{ed}/dX^m_p    = Psi_{m,e} t_{d,p} + t_{e,p} Psi_{m,d}
	//    dg^{ab}/dX^m_p    = -g^{ae} (dg_{ed}/dX^m_p) g^{db}
	//    d2g_{ed}/(ds_c dX^m_p) = Psi_{m,e} X_{p,dc} + t_{e,p} Psi_{m,dc} + Psi_{m,d} X_{p,ec} + t_{d,p} Psi_{m,ec}
	// and the product rule; the s-derivative dg^{ab}/ds_c is passed in because the caller needs it too.
	void BulkElementBase::fill_d2x_dNodalPos_helper(unsigned n_node, unsigned n_dim, unsigned el_dim,
													const oomph::DenseMatrix<double> &interpolated_t,
													const oomph::DShape &dpsids_Element, const oomph::DShape &d2psids_Element,
													const oomph::DenseMatrix<double> &aup, const double (*Xkab)[MAX_N2DERIV],
													const double (*dgab_ds)[3][3],
													std::vector<double> &dM_dX, std::vector<double> &dQ_dX,
													std::vector<double> *d2M_dXdX, std::vector<double> *d2Q_dXdX) const
	{
		const bool want_hessian = (d2M_dXdX != NULL);
		// Flat index helpers. el_dim, n_dim <= 3; n_node is the element's node count.
		const unsigned ND = n_dim, ED = el_dim, NN = n_node;
		auto iM = [&](unsigned i, unsigned b, unsigned m, unsigned p) { return ((i * ED + b) * NN + m) * ND + p; };
		auto iQ = [&](unsigned i, unsigned b, unsigned c, unsigned m, unsigned p) { return (((i * ED + b) * ED + c) * NN + m) * ND + p; };
		auto iG = [&](unsigned a, unsigned b, unsigned m, unsigned p) { return ((a * ED + b) * NN + m) * ND + p; };
		auto iGs = [&](unsigned a, unsigned b, unsigned c, unsigned m, unsigned p) { return (((a * ED + b) * ED + c) * NN + m) * ND + p; };

		dM_dX.assign(ND * ED * NN * ND, 0.0);
		dQ_dX.assign(ND * ED * ED * NN * ND, 0.0);

		// dg_{ed}/dX^m_p  and  dg^{ab}/dX^m_p
		std::vector<double> dgcov_dX(ED * ED * NN * ND, 0.0), dgab_dX(ED * ED * NN * ND, 0.0);
		for (unsigned e = 0; e < ED; e++)
			for (unsigned d = 0; d < ED; d++)
				for (unsigned m = 0; m < NN; m++)
					for (unsigned p = 0; p < ND; p++)
						dgcov_dX[iG(e, d, m, p)] = dpsids_Element(m, e) * interpolated_t(d, p) + interpolated_t(e, p) * dpsids_Element(m, d);
		for (unsigned a = 0; a < ED; a++)
			for (unsigned b = 0; b < ED; b++)
				for (unsigned m = 0; m < NN; m++)
					for (unsigned p = 0; p < ND; p++)
					{
						double sum = 0.0;
						for (unsigned e = 0; e < ED; e++)
							for (unsigned d = 0; d < ED; d++)
								sum -= aup(a, e) * dgcov_dX[iG(e, d, m, p)] * aup(d, b);
						dgab_dX[iG(a, b, m, p)] = sum;
					}

		// dg_{ed}/ds_c, needed for the mixed second derivative below
		double dgcov_ds[3][3][3];
		for (unsigned e = 0; e < ED; e++)
			for (unsigned d = 0; d < ED; d++)
				for (unsigned c = 0; c < ED; c++)
				{
					double sum = 0.0;
					for (unsigned int i = 0; i < ND; i++)
						sum += interpolated_t(e, i) * Xkab[i][PYOOMPH_D2_SLOT(d, c)] + interpolated_t(d, i) * Xkab[i][PYOOMPH_D2_SLOT(e, c)];
					dgcov_ds[e][d][c] = sum;
				}

		// d2g^{ab}/(ds_c dX^m_p)
		std::vector<double> d2gab_dsdX(ED * ED * ED * NN * ND, 0.0);
		for (unsigned a = 0; a < ED; a++)
			for (unsigned b = 0; b < ED; b++)
				for (unsigned c = 0; c < ED; c++)
					for (unsigned m = 0; m < NN; m++)
						for (unsigned p = 0; p < ND; p++)
						{
							double sum = 0.0;
							for (unsigned e = 0; e < ED; e++)
								for (unsigned d = 0; d < ED; d++)
								{
									const double d2gcov = dpsids_Element(m, e) * Xkab[p][PYOOMPH_D2_SLOT(d, c)] + interpolated_t(e, p) * d2psids_Element(m, PYOOMPH_D2_SLOT(d, c)) + dpsids_Element(m, d) * Xkab[p][PYOOMPH_D2_SLOT(e, c)] + interpolated_t(d, p) * d2psids_Element(m, PYOOMPH_D2_SLOT(e, c));
									sum -= dgab_dX[iG(a, e, m, p)] * dgcov_ds[e][d][c] * aup(d, b);
									sum -= aup(a, e) * d2gcov * aup(d, b);
									sum -= aup(a, e) * dgcov_ds[e][d][c] * dgab_dX[iG(d, b, m, p)];
								}
							d2gab_dsdX[iGs(a, b, c, m, p)] = sum;
						}

		// ---- Second nodal-coordinate derivatives -------------------------------------------------
		// The same chain one level further. The covariant metric is quadratic in X, so its second
		// derivative is constant in X and the third (mixed with one s) closes the recursion:
		//    d2g_{ed}/(dX^m_p dX^m2_p2)      = delta_{p p2} (Psi_{m,e} Psi_{m2,d} + Psi_{m2,e} Psi_{m,d})
		//    d2(dg_{ed}/ds_c)/(dX^m_p dX^m2_p2)
		//        = delta_{p p2} (Psi_{m,e} Psi_{m2,dc} + Psi_{m2,e} Psi_{m,dc}
		//                      + Psi_{m,d} Psi_{m2,ec} + Psi_{m2,d} Psi_{m,ec})
		// and with A = g^{-1}, writing d for d/dX^m_p and d' for d/dX^m2_p2,
		//    d'dA     = -(d'A)(dg)A - A(d'dg)A - A(dg)(d'A)
		//    d'd dcA  = -(d'dA)(dcg)A - (dA)(d'dcg)A - (dA)(dcg)(d'A)
		//               -(d'A)(ddcg)A - A(d'd dcg)A - A(ddcg)(d'A)
		//               -(d'A)(dcg)(dA) - A(d'dcg)(dA) - A(dcg)(d'dA)
		//
		// Nodes and directions are flattened into a single coordinate index mp = m*ND + p here, since
		// everything below is a double sum over them.
		const unsigned NX = NN * ND;
		auto iGG = [&](unsigned a, unsigned b, unsigned mp, unsigned mp2) { return ((a * ED + b) * NX + mp) * NX + mp2; };
		auto iGGs = [&](unsigned a, unsigned b, unsigned c, unsigned mp, unsigned mp2) { return (((a * ED + b) * ED + c) * NX + mp) * NX + mp2; };
		// Read a [a][b][m][p]-shaped array with the flattened coordinate index
		auto flat = [&](const std::vector<double> &v, unsigned a, unsigned b, unsigned mp) { return v[((a * ED + b) * NN + mp / ND) * ND + mp % ND]; };
		auto psi_a = [&](unsigned mp, unsigned a) { return dpsids_Element(mp / ND, a); };
		auto psi_ac = [&](unsigned mp, unsigned a, unsigned c) { return d2psids_Element(mp / ND, PYOOMPH_D2_SLOT(a, c)); };
		auto dir_of = [&](unsigned mp) { return mp % ND; };
		// d(dg_{ed}/ds_c)/dX^mp, i.e. the mixed second derivative already used above
		auto d_dcgcov = [&](unsigned e, unsigned d, unsigned c, unsigned mp) {
			const unsigned p = dir_of(mp);
			return psi_a(mp, e) * Xkab[p][PYOOMPH_D2_SLOT(d, c)] + interpolated_t(e, p) * psi_ac(mp, d, c) + psi_a(mp, d) * Xkab[p][PYOOMPH_D2_SLOT(e, c)] + interpolated_t(d, p) * psi_ac(mp, e, c);
		};

		std::vector<double> d2gab_dXdX, d3gab_dsdXdX;
		if (want_hessian)
		{
			d2gab_dXdX.assign(ED * ED * NX * NX, 0.0);
			for (unsigned a = 0; a < ED; a++)
				for (unsigned b = 0; b < ED; b++)
					for (unsigned mp = 0; mp < NX; mp++)
						for (unsigned mp2 = 0; mp2 < NX; mp2++)
						{
							const bool same_dir = (dir_of(mp) == dir_of(mp2));
							double sum = 0.0;
							for (unsigned e = 0; e < ED; e++)
								for (unsigned d = 0; d < ED; d++)
								{
									const double d2gcov = same_dir ? (psi_a(mp, e) * psi_a(mp2, d) + psi_a(mp2, e) * psi_a(mp, d)) : 0.0;
									sum -= flat(dgab_dX, a, e, mp2) * flat(dgcov_dX, e, d, mp) * aup(d, b);
									sum -= aup(a, e) * d2gcov * aup(d, b);
									sum -= aup(a, e) * flat(dgcov_dX, e, d, mp) * flat(dgab_dX, d, b, mp2);
								}
							d2gab_dXdX[iGG(a, b, mp, mp2)] = sum;
						}

			d3gab_dsdXdX.assign(ED * ED * ED * NX * NX, 0.0);
			for (unsigned a = 0; a < ED; a++)
				for (unsigned b = 0; b < ED; b++)
					for (unsigned c = 0; c < ED; c++)
						for (unsigned mp = 0; mp < NX; mp++)
							for (unsigned mp2 = 0; mp2 < NX; mp2++)
							{
								const bool same_dir = (dir_of(mp) == dir_of(mp2));
								double sum = 0.0;
								for (unsigned e = 0; e < ED; e++)
									for (unsigned d = 0; d < ED; d++)
									{
										const double A_ae = aup(a, e), A_db = aup(d, b);
										const double dA_ae = flat(dgab_dX, a, e, mp), dpA_ae = flat(dgab_dX, a, e, mp2);
										const double dA_db = flat(dgab_dX, d, b, mp), dpA_db = flat(dgab_dX, d, b, mp2);
										const double dcg = dgcov_ds[e][d][c];
										const double ddcg = d_dcgcov(e, d, c, mp), dpdcg = d_dcgcov(e, d, c, mp2);
										const double d2dcg = same_dir ? (psi_a(mp, e) * psi_ac(mp2, d, c) + psi_a(mp2, e) * psi_ac(mp, d, c) + psi_a(mp, d) * psi_ac(mp2, e, c) + psi_a(mp2, d) * psi_ac(mp, e, c)) : 0.0;
										sum -= d2gab_dXdX[iGG(a, e, mp, mp2)] * dcg * A_db;
										sum -= dA_ae * dpdcg * A_db;
										sum -= dA_ae * dcg * dpA_db;
										sum -= dpA_ae * ddcg * A_db;
										sum -= A_ae * d2dcg * A_db;
										sum -= A_ae * ddcg * dpA_db;
										sum -= dpA_ae * dcg * dA_db;
										sum -= A_ae * dpdcg * dA_db;
										sum -= A_ae * dcg * d2gab_dXdX[iGG(d, b, mp, mp2)];
									}
								d3gab_dsdXdX[iGGs(a, b, c, mp, mp2)] = sum;
							}
		}

		// dM_i^(b)/dX^m_p = (dg^{ab}/dX^m_p) t_{a,i} + g^{ab} delta_{ip} Psi_{m,a}
		for (unsigned int i = 0; i < ND; i++)
			for (unsigned b = 0; b < ED; b++)
				for (unsigned m = 0; m < NN; m++)
					for (unsigned p = 0; p < ND; p++)
					{
						double sum = 0.0;
						for (unsigned a = 0; a < ED; a++)
						{
							sum += dgab_dX[iG(a, b, m, p)] * interpolated_t(a, i);
							if (i == p) sum += aup(a, b) * dpsids_Element(m, a);
						}
						dM_dX[iM(i, b, m, p)] = sum;
					}

		// dQ[i][b][c]/dX^m_p
		for (unsigned int i = 0; i < ND; i++)
			for (unsigned b = 0; b < ED; b++)
				for (unsigned c = 0; c < ED; c++)
					for (unsigned m = 0; m < NN; m++)
						for (unsigned p = 0; p < ND; p++)
						{
							double sum = 0.0;
							for (unsigned a = 0; a < ED; a++)
							{
								sum += d2gab_dsdX[iGs(a, b, c, m, p)] * interpolated_t(a, i);
								sum += dgab_dX[iG(a, b, m, p)] * Xkab[i][PYOOMPH_D2_SLOT(a, c)];
								if (i == p)
								{
									sum += dgab_ds[a][b][c] * dpsids_Element(m, a);
									sum += aup(a, b) * d2psids_Element(m, PYOOMPH_D2_SLOT(a, c));
								}
							}
							dQ_dX[iQ(i, b, c, m, p)] = sum;
						}

		if (!want_hessian)
			return;

		// d2M_i^(b) = (d'd g^{ab}) t_{a,i} + (d g^{ab}) delta_{i p2} Psi_{m2,a} + (d' g^{ab}) delta_{ip} Psi_{m,a}
		auto iM2 = [&](unsigned i, unsigned b, unsigned mp, unsigned mp2) { return ((i * ED + b) * NX + mp) * NX + mp2; };
		auto iQ2 = [&](unsigned i, unsigned b, unsigned c, unsigned mp, unsigned mp2) { return (((i * ED + b) * ED + c) * NX + mp) * NX + mp2; };
		d2M_dXdX->assign(ND * ED * NX * NX, 0.0);
		d2Q_dXdX->assign(ND * ED * ED * NX * NX, 0.0);
		for (unsigned int i = 0; i < ND; i++)
			for (unsigned b = 0; b < ED; b++)
				for (unsigned mp = 0; mp < NX; mp++)
					for (unsigned mp2 = 0; mp2 < NX; mp2++)
					{
						double sum = 0.0;
						for (unsigned a = 0; a < ED; a++)
						{
							sum += d2gab_dXdX[iGG(a, b, mp, mp2)] * interpolated_t(a, i);
							if (i == dir_of(mp2)) sum += flat(dgab_dX, a, b, mp) * psi_a(mp2, a);
							if (i == dir_of(mp)) sum += flat(dgab_dX, a, b, mp2) * psi_a(mp, a);
						}
						(*d2M_dXdX)[iM2(i, b, mp, mp2)] = sum;
					}

		// d2Q[i][b][c] = (d'd dc g^{ab}) t_{a,i} + (d dc g^{ab}) delta_{i p2} Psi_{m2,a}
		//              + (d' dc g^{ab}) delta_{ip} Psi_{m,a} + (d'd g^{ab}) X_{i,ac}
		//              + (d g^{ab}) delta_{i p2} Psi_{m2,ac} + (d' g^{ab}) delta_{ip} Psi_{m,ac}
		for (unsigned int i = 0; i < ND; i++)
			for (unsigned b = 0; b < ED; b++)
				for (unsigned c = 0; c < ED; c++)
					for (unsigned mp = 0; mp < NX; mp++)
						for (unsigned mp2 = 0; mp2 < NX; mp2++)
						{
							double sum = 0.0;
							for (unsigned a = 0; a < ED; a++)
							{
								sum += d3gab_dsdXdX[iGGs(a, b, c, mp, mp2)] * interpolated_t(a, i);
								sum += d2gab_dXdX[iGG(a, b, mp, mp2)] * Xkab[i][PYOOMPH_D2_SLOT(a, c)];
								if (i == dir_of(mp2))
								{
									sum += d2gab_dsdX[iGs(a, b, c, mp / ND, mp % ND)] * psi_a(mp2, a);
									sum += flat(dgab_dX, a, b, mp) * psi_ac(mp2, a, c);
								}
								if (i == dir_of(mp))
								{
									sum += d2gab_dsdX[iGs(a, b, c, mp2 / ND, mp2 % ND)] * psi_a(mp, a);
									sum += flat(dgab_dX, a, b, mp2) * psi_ac(mp, a, c);
								}
							}
							(*d2Q_dXdX)[iQ2(i, b, c, mp, mp2)] = sum;
						}
	}

	// PYOOMPH_DISABLE_SHAPE_FAMILY_SPLIT restores the pre-split behaviour - every family of a space
	// that is needed at all gets filled - from ONE place, so that the whole stage can be A/B'd with a
	// single binary (the PYOOMPH_DISABLE_NOHANG_DISPATCH pattern). The poison keys on the same helper,
	// so it follows the lever automatically.
	static const bool __disable_shape_family_split = getenv("PYOOMPH_DISABLE_SHAPE_FAMILY_SPLIT") != NULL;
	// Its own lever, not folded into the one above: getting this family wrong does not produce a NaN or
	// a crash, it silently drops nodal-position columns of the Jacobian, so it needs to be switched -
	// and compared against the FD Jacobian - on its own.
	static const bool __disable_dcoord_split = getenv("PYOOMPH_DISABLE_DCOORD_SPLIT") != NULL;

	static inline BulkElementBase::RequiredShapeFamilies __widen_if_lever(BulkElementBase::RequiredShapeFamilies f)
	{
		// d2x is deliberately NOT widened: before the split it was gated by the GLOBAL "does any space
		// want second derivatives" flag, and everything the second-derivative formulas need from the
		// geometry (Qibc, dM_dX, the second local derivatives of the mapping) is built under that same
		// flag. Widening it per space therefore segfaults on the first static-mesh case tried, which is
		// what the fill sites restore instead (see the space_d2x lines).
		if (__disable_shape_family_split && f.any())
			f.psi = f.dx = f.dX = true;
		if (__disable_dcoord_split && f.any())
			f.dcoord = true;
		return f;
	}

	// See the declaration. Two things are folded in here that the three copies of the predicate this
	// replaces disagreed about:
	//  * dX_psi now counts on its OWN space, not only via the dominant-space clause. A non-dominant
	//    space asked for nothing but Lagrangian (or local-coordinate) derivatives used to get NO fill
	//    at all and the generated code would have read a stale buffer. No corpus instance exists - the
	//    combination needs a Lagrangian-gradient term on a non-geometry space - but nothing prevented
	//    one, and poison mode now guards it.
	//  * the old fill-side variant carried an extra "|| (dominant_space == "" && nnode_of_space == 0)"
	//    alternative that was ANDed with nnode_of_space != 0, i.e. dead. It is simply gone.
	BulkElementBase::RequiredShapeFamilies BulkElementBase::required_shape_families(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, unsigned ispace) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		const JITFuncSpec_RequiredShapes_For_Space_t &rs = required.continuous_spaces[ispace];
		// The Pos.* flags name whichever space carries the geometry; set_remaining_shapes_appropriately
		// resolves the shape_Pos aliases onto exactly that space, so there they widen the request.
		const bool dominant = eleminfo.nnode_of_space[ispace] &&
							  !strcmp(functable->dominant_space, functable->continuous_spaces[ispace].space_name);
		RequiredShapeFamilies f;
		f.psi = rs.psi || (dominant && required.Pos.psi);
		f.dx = rs.dx_psi || (dominant && required.Pos.dx_psi);
		f.dX = rs.dX_psi || (dominant && required.Pos.dX_psi);
		f.d2x = rs.d2x_psi || (dominant && required.Pos.d2x_psi);
		f.dcoord = rs.dx_psi_dcoord || (dominant && required.Pos.dx_psi_dcoord);
		return __widen_if_lever(f);
	}

	BulkElementBase::RequiredShapeFamilies BulkElementBase::required_shape_families_DL(const JITFuncSpec_RequiredShapes_FiniteElement_t &required)
	{
		RequiredShapeFamilies f;
		f.psi = required.DL.psi;
		f.dx = required.DL.dx_psi;
		f.dX = required.DL.dX_psi;
		f.d2x = required.DL.d2x_psi;
		f.dcoord = required.DL.dx_psi_dcoord;
		return __widen_if_lever(f);
	}

	// A signalling NaN, written by explicit loops: memset cannot produce a NaN, since no repeated
	// byte value is one.
	static const double __POISON = std::numeric_limits<double>::signaling_NaN();

	static inline void __poison_scalar(double &x)
	{
		x = __POISON;
		__poison_written++;
	}
	static inline void __poison1(double *a, unsigned n)
	{
		if (!a) return;
		for (unsigned i = 0; i < n; i++) a[i] = __POISON;
		__poison_written += n;
	}
	static inline void __poison2(double **a, unsigned n, unsigned m)
	{
		if (!a) return;
		for (unsigned i = 0; i < n; i++) __poison1(a[i], m);
	}
	static inline void __poison3(double ***a, unsigned n, unsigned m, unsigned k)
	{
		if (!a) return;
		for (unsigned i = 0; i < n; i++) __poison2(a[i], m, k);
	}
	static inline void __poison4(double ****a, unsigned n, unsigned m, unsigned k, unsigned l)
	{
		if (!a) return;
		for (unsigned i = 0; i < n; i++) __poison3(a[i], m, k, l);
	}

	void BulkElementBase::poison_unrequired_shapes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *si, bool element_level) const
	{
		if (!__poison_unrequired || !si) return;
		const unsigned n_dim = this->nodal_dimension();
		const unsigned n_node = this->nnode();

		if (element_level)
		{
			// The element-level families. They are filled once per assembly, BEFORE the
			// per-integration-point fills, so they have to be poisoned from
			// prepare_shape_buffer_for_integration rather than from fill_shape_info_at_s.
			for (unsigned k = 0; k < 3; k++)
			{
				const bool want = (k == 0) || (k == 1 ? required.history_geometry1 : required.history_geometry2);
				if (!required.elemsize_Eulerian || !want) __poison_scalar(si->elemsize_Eulerian[k]);
				if (!required.elemsize_Eulerian_cartesian || !want) __poison_scalar(si->elemsize_Eulerian_cartesian[k]);
			}
			if (!required.elemsize_Lagrangian) __poison_scalar(si->elemsize_Lagrangian);
			if (!required.elemsize_Lagrangian_cartesian) __poison_scalar(si->elemsize_Lagrangian_cartesian);
			if (!required.elemsize_Eulerian)
			{
				__poison2(si->elemsize_d_coords, n_dim, n_node);
				__poison4(si->elemsize_d2_coords, n_dim, n_dim, n_node, n_node);
			}
			if (!required.elemsize_Eulerian_cartesian)
			{
				__poison2(si->elemsize_Cart_d_coords, n_dim, n_node);
				__poison4(si->elemsize_Cart_d2_coords, n_dim, n_dim, n_node, n_node);
			}
			return;
		}

		// Per-space AND per-family, keyed on the very predicate the fill uses
		// (required_shape_families), which is what makes the poison meaningful: it must poison exactly
		// what the fill declines to write, or it either misses under-requests or invents them.
		// The history slots k=1,2 are poisoned along with k=0 because the history fills key on the same
		// `required` struct, so a family nobody asked for is unfilled at every history level.
		for (unsigned int ispace = 0; ispace < NUM_CONTINUOUS_SPACES; ispace++)
		{
			const unsigned nn = eleminfo.nnode_of_space[ispace];
			const RequiredShapeFamilies fam = this->required_shape_families(required, ispace);
			if (!fam.psi)
				__poison1(si->shapes[ispace], nn);
			if (!fam.dx)
				for (unsigned k = 0; k < 3; k++) __poison2(si->dx_shapes[k][ispace], nn, n_dim);
			if (!fam.dX)
			{
				__poison2(si->dX_shapes[ispace], nn, n_dim);
				__poison2(si->dS_shapes[ispace], nn, n_dim);
			}
			if (!fam.d2x)
			{
				for (unsigned k = 0; k < 3; k++) __poison2(si->d2x_shapes[k][ispace], nn, MAX_N2DERIV);
				__poison2(si->d2S_shapes[ispace], nn, MAX_N2DERIV);
				__poison4(si->d_d2x_shape_dcoord[ispace], nn, MAX_N2DERIV, n_node, n_dim);
			}
			// Now per space: the fill still sits inside the "is this space needed at all" block, so a
			// space that is not required at all is poisoned as before, and one that is required but
			// whose gradient nothing differentiates by the nodal positions is poisoned too.
			if (!fam.any() || !fam.dcoord)
				__poison4(si->d_dx_shape_dcoord[ispace], nn, n_dim, n_node, n_dim);
		}

		{
			const RequiredShapeFamilies fam = required_shape_families_DL(required);
			const unsigned nn = eleminfo.nnode_DL;
			if (!fam.psi)
				__poison1(si->shape_DL, nn);
			if (!fam.dx)
				for (unsigned k = 0; k < 3; k++) __poison2(si->dx_shape_DL[k], nn, n_dim);
			if (!fam.dX)
			{
				__poison2(si->dX_shape_DL, nn, n_dim);
				__poison2(si->dS_shape_DL, nn, n_dim);
			}
			if (!fam.d2x)
			{
				for (unsigned k = 0; k < 3; k++) __poison2(si->d2x_shape_DL[k], nn, MAX_N2DERIV);
				__poison2(si->d2S_shape_DL, nn, MAX_N2DERIV);
				__poison4(si->d_d2x_shape_dcoord_DL, nn, MAX_N2DERIV, n_node, n_dim);
			}
			if (!fam.any())
				__poison4(si->d_dx_shape_dcoord_DL, nn, n_dim, n_node, n_dim);
		}

		if (!(required.normal || required.normal_deriv))
		{
			for (unsigned k = 0; k < 3; k++) __poison1(si->normal[k], n_dim);
			__poison3(si->d_normal_dcoord, n_dim, n_node, n_dim);
		}
		if (!required.normal_deriv)
		{
			for (unsigned k = 0; k < 3; k++) __poison2(si->dnormal_dx[k], n_dim, n_dim);
			__poison4(si->d_dnormal_dx_dcoord, n_dim, n_dim, n_node, n_dim);
		}
	}

	// The bulk/opposite sub-buffers are filled by the recursive fill_shape_info_at_s call, which
	// poisons them itself with its own sub-struct. Only the ones this pass does NOT recurse into are
	// handled here: nothing writes them at all, yet the generated code can still reach them through
	// shapeinfo->bulk_shapeinfo, which is precisely the failure mode a stale flag produces.
	void InterfaceElementBase::poison_unrequired_shapes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *si, bool element_level) const
	{
		BulkElementBase::poison_unrequired_shapes(required, si, element_level);
		if (!__poison_unrequired || !si) return;
		JITFuncSpec_RequiredShapes_FiniteElement_t nothing;
		memset(&nothing, 0, sizeof(nothing));
		if (!required.bulk_shapes && si->bulk_shapeinfo && this->bulk_element_pt())
		{
			BulkElementBase *blk = dynamic_cast<BulkElementBase *>(this->bulk_element_pt());
			if (blk) blk->poison_unrequired_shapes(nothing, si->bulk_shapeinfo, element_level);
		}
		if (!required.opposite_shapes && si->opposite_shapeinfo && this->opposite_side)
		{
			this->opposite_side->poison_unrequired_shapes(nothing, si->opposite_shapeinfo, element_level);
		}
	}

	double BulkElementBase::fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds,unsigned history_index) const
	{
		bool require_hessian = flag > 2;

		// History fills run BEFORE the current-configuration one and share the history-independent
		// buffers with it, so poisoning on their (deliberately stripped-down) required struct would
		// wipe what the real fill is about to need. Only the current configuration is poisoned.
		if (__poison_unrequired && history_index == 0)
			this->poison_unrequired_shapes(required, shape_info, false);

		unsigned el_dim = this->dim();
		unsigned n_dim = this->nodal_dimension();
		unsigned n_node = this->nnode();
		unsigned n_lagr = this->nlagrangian();

		double det_Eulerian;

		// Whether any space wants second spatial derivatives of its shape functions. Everything the
		// second-derivative blocks below need beyond the first-derivative machinery is guarded by this.
		bool require_d2x = required.Pos.d2x_psi || required.DL.d2x_psi;
		for (unsigned int isp = 0; isp < NUM_CONTINUOUS_SPACES; isp++)
			require_d2x |= required.continuous_spaces[isp].d2x_psi;
		// The spatial derivative of the normal needs the GEOMETRY's second local derivatives (X_{k,ab}
		// and everything built from them) but none of the per-FIELD second derivatives, so it widens
		// only the geometry gate below - never `space_d2x`.
		const bool require_geom_d2 = require_d2x || required.normal_deriv;

		oomph::DenseMatrix<double> interpolated_t(el_dim, n_dim, 0.0); // Tangents
		oomph::DenseMatrix<double> interpolated_T(el_dim, n_lagr, 0.0);
		oomph::Shape psi_Element(n_node);
		oomph::DShape dpsids_Element(n_node, std::max((unsigned int)1, el_dim));
		this->dshape_local(s, psi_Element, dpsids_Element);

		// Second local derivatives of the GEOMETRY shape functions and the resulting
		// X_{k,ab} = d^2 x_k / (ds_a ds_b) = sum_l X^l_k Psi_{l,ab}. Needed by every second-derivative
		// formula below, so it is built once here rather than per space.
		oomph::DShape d2psids_Element(std::max((unsigned int)1, n_node), MAX_N2DERIV);
		double Xkab[3][MAX_N2DERIV];
		if (require_geom_d2 && el_dim)
		{
			std::string why;
			if (!this->supports_second_spatial_derivatives(why))
				throw_runtime_error("Second spatial derivatives of the shape functions were requested, but " + why);
			this->d2shape_local_pyoomph(s, psi_Element, dpsids_Element, d2psids_Element);
			for (unsigned int k = 0; k < n_dim; k++)
			{
				for (unsigned a = 0; a < el_dim; a++)
				{
					for (unsigned b = 0; b < el_dim; b++)
					{
						double sum = 0.0;
						for (unsigned l = 0; l < n_node; l++)
							sum += this->nodal_position(history_index, l, k) * d2psids_Element(l, PYOOMPH_D2_SLOT(a, b));
						Xkab[k][PYOOMPH_D2_SLOT(a, b)] = sum;
					}
				}
			}
		}

		for (unsigned l = 0; l < n_node; l++)
		{
			for (unsigned i = 0; i < n_dim; i++)
			{
				for (unsigned j = 0; j < el_dim; j++)
				{
					interpolated_t(j, i) += this->nodal_position(history_index, l, i) * dpsids_Element(l, j);
				}
			}
			for (unsigned i = 0; i < n_lagr; i++)
			{
				for (unsigned j = 0; j < el_dim; j++)
				{					
					interpolated_T(j, i) += this->raw_lagrangian_position_gen(l, 0, i) * dpsids_Element(l, j);
				}
			}
		}

		if (dxds)
			*dxds = interpolated_t;

		double gab_gai[el_dim][n_dim];		// stores [g^{ab} g_a]_i . First index is b second i
		double gab_gai_Lagr[el_dim][n_dim]; // stores [g^{ab} g_a]_i . First index is b second i

		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();

		bool require_dxdshape = (flag && functable->moving_nodes && (!functable->fd_position_jacobian)); //&& (required.dx_psi_C2 || required.dx_psi_C1 || required.dx_psi_DL)
		// XXX: The last condition may not be used, since even dx depends on the coordinates
		// TODO: Add a flag, whether we have a dx contribution in the residuals. If so, we always need it for moving nodes. If not (e.g. pure Lagrangian dX), we can skip it	
 
		
		// The Eulerian inverse metric g^{ab}. The 2d/3d branches below reuse their local `aup` for the
		// Lagrangian metric afterwards, so the Eulerian one is kept separately for the second-derivative
		// blocks, which need it after the branches.
		oomph::DenseMatrix<double> aup_Euler(std::max((unsigned int)1, el_dim), std::max((unsigned int)1, el_dim), 0.0);

		oomph::RankFourTensor<double> DXdshape_il_jb; //[n_dim][n_node][n_dim][el_dim]; //this is d(g^{ab}g_{a,j})/d(x_i^l) //TODO: This could lead to stack problems due to size
      RankSixTensor * D2X2_dshape=NULL;
      // On a bulk element the second nodal-coordinate derivative of a shape gradient is a sum of two
      // triple products of FIRST derivatives (see the fill sites below), so neither this rank-6 scratch
      // nor the E-tensor chain that fills it is needed - and it was 24.4% of Hessian assembly
      // (dev_docs/code_generation.md 9.4.11). An interface picks up normal-projector terms the closed
      // form does not carry, so there the old path stays. Under PYOOMPH_PARANOID_ALE_IDENTITY both are
      // computed, so the check compares two independent derivations rather than one against itself.
      // Bulk only, and that is a measured choice rather than a gap. The E-tensor cost scales with
      // el_dim, so on an interface its loops collapse and it is cheap, while the closed form still pays
      // eight terms plus the projector and three dot products, which scale with n_dim. Switching
      // interfaces over measured +3.7% on a 2D free-surface case. The interface identity is derived and
      // validated all the same - the check below covers el_dim < n_dim - so the path that ships there is
      // now checked, which it was not before.
      const bool closed_form_d2 = (el_dim == n_dim) && !__paranoid_ale_identity;
      if (require_hessian && require_dxdshape && !closed_form_d2)
      {
        D2X2_dshape=new RankSixTensor(n_dim,el_dim,n_node,n_node,n_dim,n_dim);
      }
		if (el_dim == 1)
		{
			double a11 = 0.0;
			for (unsigned int i = 0; i < n_dim; i++)
				a11 += interpolated_t(0, i) * interpolated_t(0, i);
			for (unsigned int i = 0; i < n_dim; i++)
				gab_gai[0][i] = interpolated_t(0, i) / a11;
			det_Eulerian = sqrt(a11);
			aup_Euler(0, 0) = 1.0 / a11;

			// TODO: Only calc d(dx)/dcoords if necessary
			if (require_dxdshape)
			{
				DXdshape_il_jb.resize(n_dim, n_node, n_dim, el_dim, 0.0); // this is d(g^{ab}g_{a,j})/d(x_i^l)
				oomph::DenseMatrix<double> aup(1, 1, 1.0 / a11);
				this->fill_shape_info_at_s_dNodalPos_helper(shape_info, index, interpolated_t, dpsids_Element, det_Eulerian, aup, require_hessian, DXdshape_il_jb,D2X2_dshape);
			}

			a11 = 0.0;
			for (unsigned int i = 0; i < n_lagr; i++)
				a11 += interpolated_T(0, i) * interpolated_T(0, i);
			for (unsigned int i = 0; i < n_lagr; i++)
				gab_gai_Lagr[0][i] = interpolated_T(0, i) / a11;
			JLagr = sqrt(a11);
		}
		else if (el_dim == 2)
		{
			double amet[2][2];
			for (unsigned al = 0; al < 2; al++)
			{
				for (unsigned be = 0; be < 2; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_dim; i++)
					{
						amet[al][be] += interpolated_t(al, i) * interpolated_t(be, i);
					}
				}
			}
			double det_a = amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0];
			oomph::DenseMatrix<double> aup(2, 2);
			aup(0, 0) = amet[1][1] / det_a;
			aup(0, 1) = -amet[0][1] / det_a;
			aup(1, 0) = -amet[1][0] / det_a;
			aup(1, 1) = amet[0][0] / det_a;

			for (unsigned int b = 0; b < 2; b++)
			{
				for (unsigned int i = 0; i < n_dim; i++)
				{
					gab_gai[b][i] = aup(0, b) * interpolated_t(0, i) + aup(1, b) * interpolated_t(1, i);
				}
			}
			det_Eulerian = sqrt(det_a);
			aup_Euler = aup;

			// TODO: Only calc d(dx)/dcoords if necessary
			if (require_dxdshape)
			{
				DXdshape_il_jb.resize(n_dim, n_node, n_dim, el_dim, 0.0); // this is d(g^{ab}g_{a,j})/d(x_i^l)
				this->fill_shape_info_at_s_dNodalPos_helper(shape_info, index, interpolated_t, dpsids_Element, det_Eulerian, aup, require_hessian, DXdshape_il_jb,D2X2_dshape);
			}

			// Lagr
			for (unsigned al = 0; al < 2; al++)
			{
				for (unsigned be = 0; be < 2; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_lagr; i++)
					{
						amet[al][be] += interpolated_T(al, i) * interpolated_T(be, i);
					}
				}
			}
			det_a = amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0];
			aup(0, 0) = amet[1][1] / det_a;
			aup(0, 1) = -amet[0][1] / det_a;
			aup(1, 0) = -amet[1][0] / det_a;
			aup(1, 1) = amet[0][0] / det_a;

			for (unsigned int b = 0; b < 2; b++)
			{
				for (unsigned int i = 0; i < n_lagr; i++)
				{
					gab_gai_Lagr[b][i] = aup(0, b) * interpolated_T(0, i) + aup(1, b) * interpolated_T(1, i);
				}
			}
			JLagr = sqrt(det_a);
		}
		else if (el_dim == 0)
		{
			det_Eulerian = 1.0;
			JLagr = 1.0;			
			for (unsigned int ispace=0;ispace<NUM_CONTINUOUS_SPACES;ispace++)
			{
				for (unsigned l = 0; l < eleminfo.nnode_of_space[ispace]; l++)
				{
					shape_info->shapes[ispace][l] = 1.0;
					for (unsigned int i = 0; i < n_dim; i++)					
						shape_info->dx_shapes[history_index][ispace][l][i] = 0.0;
					for (unsigned int i = 0; i < n_lagr; i++)
						shape_info->dX_shapes[ispace][l][i] = 0.0;
					for (unsigned int i = 0; i < el_dim; i++)
						shape_info->dS_shapes[ispace][l][i] = 0.0;
					// A point element has no local coordinates, so every spatial derivative vanishes.
					if (require_d2x)
						for (unsigned int k = 0; k < MAX_N2DERIV; k++)
						{
							shape_info->d2x_shapes[history_index][ispace][l][k] = 0.0;
							shape_info->d2S_shapes[ispace][l][k] = 0.0;
						}
				}
			}			
			for (unsigned l = 0; l < eleminfo.nnode_DL; l++)
			{
				shape_info->shape_DL[l] = 1.0;
				for (unsigned int i = 0; i < n_dim; i++)
					shape_info->dx_shape_DL[history_index][l][i] = 0.0;
				for (unsigned int i = 0; i < n_lagr; i++)
					shape_info->dX_shape_DL[l][i] = 0.0;
				for (unsigned int i = 0; i < el_dim; i++)
					shape_info->dS_shape_DL[l][i] = 0.0;
				if (require_d2x)
					for (unsigned int k = 0; k < MAX_N2DERIV; k++)
					{
						shape_info->d2x_shape_DL[history_index][l][k] = 0.0;
						shape_info->d2S_shape_DL[l][k] = 0.0;
					}
			}
			for (unsigned l = 0; l < n_node; l++)
			{
				for (unsigned i = 0; i < n_dim; i++)
				{
					shape_info->int_pt_weights_d_coords[i][l] = 0.0;
				}
			}
		}
		else if (el_dim == 3)
		{

			double amet[3][3];
			for (unsigned al = 0; al < 3; al++)
			{
				for (unsigned be = 0; be < 3; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_dim; i++)
					{
						amet[al][be] += interpolated_t(al, i) * interpolated_t(be, i);
					}
				}
			}
			double det_a = amet[0][0] * amet[1][1] * amet[2][2] + amet[0][1] * amet[1][2] * amet[2][0] + amet[0][2] * amet[1][0] * amet[2][1] - amet[0][0] * amet[1][2] * amet[2][1] - amet[0][1] * amet[1][0] * amet[2][2] - amet[0][2] * amet[1][1] * amet[2][0];

			oomph::DenseMatrix<double> aup(3, 3);
			aup(0, 0) = (amet[1][1] * amet[2][2] - amet[1][2] * amet[2][1]) / det_a;
			aup(0, 1) = -(amet[0][1] * amet[2][2] - amet[0][2] * amet[2][1]) / det_a;
			aup(0, 2) = (amet[0][1] * amet[1][2] - amet[0][2] * amet[1][1]) / det_a;
			aup(1, 0) = -(amet[1][0] * amet[2][2] - amet[1][2] * amet[2][0]) / det_a;
			aup(1, 1) = (amet[0][0] * amet[2][2] - amet[0][2] * amet[2][0]) / det_a;
			aup(1, 2) = -(amet[0][0] * amet[1][2] - amet[0][2] * amet[1][0]) / det_a;
			aup(2, 0) = (amet[1][0] * amet[2][1] - amet[1][1] * amet[2][0]) / det_a;
			aup(2, 1) = -(amet[0][0] * amet[2][1] - amet[0][1] * amet[2][0]) / det_a;
			aup(2, 2) = (amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0]) / det_a;

			for (unsigned int b = 0; b < 3; b++)
			{
				for (unsigned int i = 0; i < n_dim; i++)
				{
					gab_gai[b][i] = aup(0, b) * interpolated_t(0, i) + aup(1, b) * interpolated_t(1, i) + aup(2, b) * interpolated_t(2, i);
				}
			}
			det_Eulerian = sqrt(det_a);
			aup_Euler = aup;

			// TODO: Only calc d(dx)/dcoords if necessary
			if (require_dxdshape)
			{
				DXdshape_il_jb.resize(n_dim, n_node, n_dim, el_dim, 0.0); // this is d(g^{ab}g_{a,j})/d(x_i^l)
				this->fill_shape_info_at_s_dNodalPos_helper(shape_info, index, interpolated_t, dpsids_Element, det_Eulerian, aup, require_hessian, DXdshape_il_jb,D2X2_dshape);
			}

			// Lagr
			for (unsigned al = 0; al < 3; al++)
			{
				for (unsigned be = 0; be < 3; be++)
				{
					amet[al][be] = 0.0;
					for (unsigned i = 0; i < n_lagr; i++)
					{
						amet[al][be] += interpolated_T(al, i) * interpolated_T(be, i);
					}
				}
			}
			det_a = amet[0][0] * amet[1][1] * amet[2][2] + amet[0][1] * amet[1][2] * amet[2][0] + amet[0][2] * amet[1][0] * amet[2][1] - amet[0][0] * amet[1][2] * amet[2][1] - amet[0][1] * amet[1][0] * amet[2][2] - amet[0][2] * amet[1][1] * amet[2][0];
			aup(0, 0) = (amet[1][1] * amet[2][2] - amet[1][2] * amet[2][1]) / det_a;
			aup(0, 1) = -(amet[0][1] * amet[2][2] - amet[0][2] * amet[2][1]) / det_a;
			aup(0, 2) = (amet[0][1] * amet[1][2] - amet[0][2] * amet[1][1]) / det_a;
			aup(1, 0) = -(amet[1][0] * amet[2][2] - amet[1][2] * amet[2][0]) / det_a;
			aup(1, 1) = (amet[0][0] * amet[2][2] - amet[0][2] * amet[2][0]) / det_a;
			aup(1, 2) = -(amet[0][0] * amet[1][2] - amet[0][2] * amet[1][0]) / det_a;
			aup(2, 0) = (amet[1][0] * amet[2][1] - amet[1][1] * amet[2][0]) / det_a;
			aup(2, 1) = -(amet[0][0] * amet[2][1] - amet[0][1] * amet[2][0]) / det_a;
			aup(2, 2) = (amet[0][0] * amet[1][1] - amet[0][1] * amet[1][0]) / det_a;

			for (unsigned int b = 0; b < 3; b++)
			{
				for (unsigned int i = 0; i < n_lagr; i++)
				{
					gab_gai_Lagr[b][i] = aup(0, b) * interpolated_T(0, i) + aup(1, b) * interpolated_T(1, i) + aup(2, b) * interpolated_T(2, i);
				}
			}
			JLagr = sqrt(det_a);
		}
		else
		{
			throw_runtime_error("Implement for this dimension");
		}

		// ------------------------------------------------------------------------------------------
		// Second spatial derivatives.
		//
		// With M_i^(b) := gab_gai[b][i] = g^{ab} t_{a,i} (which is the inverse Jacobian ds_b/dx_i when
		// the element has no codimension, and the pseudo-inverse giving the surface gradient
		// otherwise), the first derivative is dpsi/dx_i = M_i^(b) psi_,b and hence
		//
		//    D_ij psi := d/dx_j ( dpsi/dx_i ) = M_i^(b) M_j^(c) psi_,bc + M_j^(c) Q[i][b][c] psi_,b
		//
		// with Q[i][b][c] := dM_i^(b)/ds_c = (dg^{ab}/ds_c) t_{a,i} + g^{ab} X_{i,ac}.
		//
		// For el_dim == n_dim this collapses to the familiar
		// K_{ia}K_{jb} psi_,ab - (dpsi/dx_k) K_{ia}K_{jb} X_{k,ab} and is symmetric in i,j. For a
		// surface it does not, and it is NOT symmetric: on a unit circle it evaluates to
		// t_i t_j psi'' - n_i t_j psi'. That is the honest tangential derivative of the surface
		// gradient; its trace is exactly the Laplace-Beltrami operator (the asymmetric part is
		// trace-free), so div(grad(u)) on an interface comes out right. Only the general form is
		// implemented, so that there is a single code path to validate.
		// ------------------------------------------------------------------------------------------
		double Qibc[3][3][3];   // [i_dim][b_eldim][c_eldim]
		double dgab_ds[3][3][3]; // dg^{ab}/ds_c
		std::vector<double> dM_dX, dQ_dX, d2M_dXdX, d2Q_dXdX; // nodal-coordinate sensitivities, see fill_d2x_dNodalPos_helper
		if (require_geom_d2 && el_dim)
		{
			// dg_{ed}/ds_c = t_{e,i} X_{i,dc} + t_{d,i} X_{i,ec}, then
			// dg^{ab}/ds_c = -g^{ae} (dg_{ed}/ds_c) g^{db}.
			for (unsigned a = 0; a < el_dim; a++)
				for (unsigned b = 0; b < el_dim; b++)
					for (unsigned c = 0; c < el_dim; c++)
					{
						double sum = 0.0;
						for (unsigned e = 0; e < el_dim; e++)
							for (unsigned d = 0; d < el_dim; d++)
							{
								double dged = 0.0;
								for (unsigned int i = 0; i < n_dim; i++)
									dged += interpolated_t(e, i) * Xkab[i][PYOOMPH_D2_SLOT(d, c)] + interpolated_t(d, i) * Xkab[i][PYOOMPH_D2_SLOT(e, c)];
								sum -= aup_Euler(a, e) * dged * aup_Euler(d, b);
							}
						dgab_ds[a][b][c] = sum;
					}

			for (unsigned int i = 0; i < n_dim; i++)
				for (unsigned b = 0; b < el_dim; b++)
					for (unsigned c = 0; c < el_dim; c++)
					{
						double sum = 0.0;
						for (unsigned a = 0; a < el_dim; a++)
							sum += dgab_ds[a][b][c] * interpolated_t(a, i) + aup_Euler(a, b) * Xkab[i][PYOOMPH_D2_SLOT(a, c)];
						Qibc[i][b][c] = sum;
					}

			if (require_dxdshape)
			{
				this->fill_d2x_dNodalPos_helper(n_node, n_dim, el_dim, interpolated_t, dpsids_Element, d2psids_Element,
												aup_Euler, Xkab, dgab_ds, dM_dX, dQ_dX,
												(require_hessian ? &d2M_dXdX : NULL), (require_hessian ? &d2Q_dXdX : NULL));
			}
		}

		// Now to the parts for the spaces
		// A space's shapes are needed either because fields living in that space were explicitly
		// requested, or because it is the "dominant" (i.e. geometry-defining, Pos) space of this
		// element and Pos-shapes were requested.

		for (unsigned int ispace=0;ispace<NUM_CONTINUOUS_SPACES;ispace++)
		{
			// One shared predicate (see BulkElementBase::required_shape_families), which the poison and
			// set_remaining_shapes_appropriately key on too.
			const RequiredShapeFamilies fam = this->required_shape_families(required, ispace);

			if (fam.any())
			{
				oomph::Shape psi(eleminfo.nnode_of_space[ispace]);
				oomph::DShape dpsids(eleminfo.nnode_of_space[ispace], std::max((unsigned int)1, el_dim));
				// Per-space, not the global require_d2x: a C1 pressure next to a C2 field that wants
				// second derivatives must not pay for them. Everything the geometry side of the
				// second-derivative formulas needs is guarded by require_geom_d2, which is wider.
				const bool space_d2x = (__disable_shape_family_split ? require_d2x : fam.d2x) && el_dim && eleminfo.nnode_of_space[ispace];
				oomph::DShape d2psids(std::max((unsigned int)1, eleminfo.nnode_of_space[ispace]), MAX_N2DERIV);
				if (space_d2x)
					this->d2shape_local_of_space(ispace, s, psi, dpsids, d2psids);
				else
					this->dshape_local_of_space(ispace, s, psi, dpsids);
				for (unsigned l = 0; l < eleminfo.nnode_of_space[ispace]; l++)
				{
					// One family at a time. psi-only is by far the most common request, and it used to
					// pay for both gradient contractions; the local derivatives dpsids come out of the
					// same dshape_local_of_space call either way, so only the contractions are saved.
					if (fam.psi)
						shape_info->shapes[ispace][l] = psi[l];
					if (fam.dx)
					{
						for (unsigned int i = 0; i < n_dim; i++)
						{
							shape_info->dx_shapes[history_index][ispace][l][i] = 0.0;
							for (unsigned b = 0; b < el_dim; b++)
							{
								shape_info->dx_shapes[history_index][ispace][l][i] += gab_gai[b][i] * dpsids(l, b);
							}
						}
					}

					if (fam.dX)
					{
						// dS rides on the dX_psi flag, see jitbridge.h: the code generator classifies a
						// local-coordinate derivative as a Lagrangian one (D1XBasisFunctionLocalCoord
						// derives from D1XBasisFunctionLagr), so there is no flag of its own to key on.
						for (unsigned int i=0; i < this->dim();i++) shape_info->dS_shapes[ispace][l][i] =  dpsids(l, i);

						for (unsigned int i = 0; i < n_lagr; i++)
						{
							shape_info->dX_shapes[ispace][l][i] = 0.0;
							for (unsigned b = 0; b < el_dim; b++)
							{
								shape_info->dX_shapes[ispace][l][i] += gab_gai_Lagr[b][i] * dpsids(l, b);
							}
						}
					}

					if (space_d2x)
					{
						for (unsigned int i = 0; i < n_dim; i++)
						{
							for (unsigned int j = 0; j < n_dim; j++)
							{
								double sum = 0.0;
								for (unsigned b = 0; b < el_dim; b++)
								{
									for (unsigned c = 0; c < el_dim; c++)
									{
										sum += gab_gai[b][i] * gab_gai[c][j] * d2psids(l, PYOOMPH_D2_SLOT(b, c));
										sum += gab_gai[c][j] * Qibc[i][b][c] * dpsids(l, b);
									}
								}
								shape_info->d2x_shapes[history_index][ispace][l][PYOOMPH_D2_SLOT(i, j)] = sum;
							}
						}
						for (unsigned a = 0; a < el_dim; a++)
							for (unsigned b = 0; b < el_dim; b++)
								shape_info->d2S_shapes[ispace][l][PYOOMPH_D2_SLOT(a, b)] = d2psids(l, PYOOMPH_D2_SLOT(a, b));
					}


					// Nodal-coordinate sensitivity of the second derivative. Differentiating
					//    D_ij psi = M_i^b M_j^c psi_,bc + M_j^c Q[i][b][c] psi_,b
					// by X^m_p only touches M and Q, since psi_,b and psi_,bc are reference-element
					// quantities.
					if (space_d2x && require_dxdshape)
					{
						const unsigned ND = n_dim, ED = el_dim, NN = eleminfo.nnode;
						auto iM = [&](unsigned i2, unsigned b, unsigned m, unsigned p) { return ((i2 * ED + b) * NN + m) * ND + p; };
						auto iQ = [&](unsigned i2, unsigned b, unsigned c, unsigned m, unsigned p) { return (((i2 * ED + b) * ED + c) * NN + m) * ND + p; };
						for (unsigned int i = 0; i < n_dim; i++)
						{
							for (unsigned int j = 0; j < n_dim; j++)
							{
								const unsigned slot = PYOOMPH_D2_SLOT(i, j);
								for (unsigned m = 0; m < eleminfo.nnode; m++)
								{
									for (unsigned int p = 0; p < n_dim; p++)
									{
										double sum = 0.0;
										for (unsigned b = 0; b < el_dim; b++)
										{
											for (unsigned c = 0; c < el_dim; c++)
											{
												sum += (dM_dX[iM(i, b, m, p)] * gab_gai[c][j] + gab_gai[b][i] * dM_dX[iM(j, c, m, p)]) * d2psids(l, PYOOMPH_D2_SLOT(b, c));
												sum += (dM_dX[iM(j, c, m, p)] * Qibc[i][b][c] + gab_gai[c][j] * dQ_dX[iQ(i, b, c, m, p)]) * dpsids(l, b);
											}
										}
										shape_info->d_d2x_shape_dcoord[ispace][l][slot][m][p] = sum;
									}
								}

								// Second nodal-coordinate derivative, i.e. one more application of the
								// product rule to D_ij psi = M_i^b M_j^c psi_,bc + M_j^c Q[i][b][c] psi_,b.
								if (require_hessian)
								{
									const unsigned NX = NN * ND;
									auto iM2 = [&](unsigned i2, unsigned b, unsigned mp, unsigned mp2) { return ((i2 * ED + b) * NX + mp) * NX + mp2; };
									auto iQ2 = [&](unsigned i2, unsigned b, unsigned c, unsigned mp, unsigned mp2) { return (((i2 * ED + b) * ED + c) * NX + mp) * NX + mp2; };
									for (unsigned m = 0; m < eleminfo.nnode; m++)
										for (unsigned int p = 0; p < n_dim; p++)
											for (unsigned m2 = 0; m2 < eleminfo.nnode; m2++)
												for (unsigned int p2 = 0; p2 < n_dim; p2++)
												{
													const unsigned mp = m * ND + p, mp2 = m2 * ND + p2;
													double sum = 0.0;
													for (unsigned b = 0; b < el_dim; b++)
													{
														for (unsigned c = 0; c < el_dim; c++)
														{
															const double psi_bc = d2psids(l, PYOOMPH_D2_SLOT(b, c));
															const double psi_b = dpsids(l, b);
															sum += (d2M_dXdX[iM2(i, b, mp, mp2)] * gab_gai[c][j] + gab_gai[b][i] * d2M_dXdX[iM2(j, c, mp, mp2)]) * psi_bc;
															sum += (dM_dX[iM(i, b, m, p)] * dM_dX[iM(j, c, m2, p2)] + dM_dX[iM(i, b, m2, p2)] * dM_dX[iM(j, c, m, p)]) * psi_bc;
															sum += (d2M_dXdX[iM2(j, c, mp, mp2)] * Qibc[i][b][c] + gab_gai[c][j] * d2Q_dXdX[iQ2(i, b, c, mp, mp2)]) * psi_b;
															sum += (dM_dX[iM(j, c, m, p)] * dQ_dX[iQ(i, b, c, m2, p2)] + dM_dX[iM(j, c, m2, p2)] * dQ_dX[iQ(i, b, c, m, p)]) * psi_b;
														}
													}
													shape_info->d2_d2x2_shape_dcoord[ispace][l][slot][m][p][m2][p2] = sum;
												}
								}
							}
						}
					}
					// Only for the spaces whose Eulerian gradient an assembled entry actually
					// differentiates by the nodal positions - fam.dcoord is marked from the same set of
					// shape expansions that emits those reads. This used to run for EVERY required
					// space and is the most expensive family of a moving-mesh fill.
					if (require_dxdshape && fam.dcoord)
					{
						for (unsigned int i = 0; i < n_dim; i++)
						{
							for (unsigned l2 = 0; l2 < eleminfo.nnode; l2++)
							{
								for (unsigned int i2 = 0; i2 < n_dim; i2++)
								{
									shape_info->d_dx_shape_dcoord[ispace][l][i][l2][i2] = 0.0;
									for (unsigned int b = 0; b < el_dim; b++)
									{
										shape_info->d_dx_shape_dcoord[ispace][l][i][l2][i2] += DXdshape_il_jb(i2, l2, i, b) * dpsids(l, b); // TODO: Also for all other shapes (C1, DL)
									}
								}
							}
						}
						if (__paranoid_ale_identity)
						{
							// Both factors are rebuilt locally rather than read back from the buffer, so
							// the check cannot be satisfied by a fill order coincidence: D psi_l from this
							// space's own local derivatives, D Psi_q from dpsids_Element, which is the
							// GEOMETRY space - it is the mapping that carries the nodal-position
							// dependence, whichever space the field happens to live on.
							double Dpsi_l[3], Nproj[3][3];
							for (unsigned int i = 0; i < n_dim; i++)
							{
								Dpsi_l[i] = 0.0;
								for (unsigned b = 0; b < el_dim; b++)
									Dpsi_l[i] += gab_gai[b][i] * dpsids(l, b);
								for (unsigned int j = 0; j < n_dim; j++)
								{
									double P = 0.0; // P_ij = g^{ab} t_{a,i} t_{b,j}, the tangential projector
									for (unsigned b = 0; b < el_dim; b++)
										P += gab_gai[b][i] * interpolated_t(b, j);
									Nproj[i][j] = (i == j ? 1.0 : 0.0) - P;
								}
							}
							for (unsigned l2 = 0; l2 < eleminfo.nnode; l2++)
							{
								double Dpsi_q[3], dot = 0.0;
								for (unsigned int i = 0; i < n_dim; i++)
								{
									Dpsi_q[i] = 0.0;
									for (unsigned b = 0; b < el_dim; b++)
										Dpsi_q[i] += gab_gai[b][i] * dpsids_Element(l2, b);
								}
								for (unsigned int a = 0; a < n_dim; a++)
									dot += Dpsi_l[a] * Dpsi_q[a];
								for (unsigned int i = 0; i < n_dim; i++)
									for (unsigned int i2 = 0; i2 < n_dim; i2++)
									{
										std::ostringstream where;
										where << "(space " << ispace << ", node " << l << "/" << eleminfo.nnode_of_space[ispace]
											  << ", dir " << i << ", coord node " << l2 << ", coord dir " << i2
											  << ", el_dim " << el_dim << ", nodal_dim " << n_dim
											  << ", eleminfo.nnode " << eleminfo.nnode
											  << ", this->nnode() " << this->nnode()
											  << ", n_node " << n_node << ")";
										ale_identity_check("d_dx_shape_dcoord", el_dim, n_dim,
														   shape_info->d_dx_shape_dcoord[ispace][l][i][l2][i2],
														   -Dpsi_l[i2] * Dpsi_q[i] + Nproj[i][i2] * dot,
														   where.str());
									}
							}
						}
						if (require_hessian && closed_form_d2)
						{
							// Straight from d2_shape_dcoord_closed_form, so neither the E tensor above nor
							// its per-integration-point rank-6 allocation is needed - on interfaces too,
							// where the N-terms carry the difference.
							double Dpsi_l[3];
							for (unsigned int i = 0; i < n_dim; i++)
							{
								Dpsi_l[i] = 0.0;
								for (unsigned b = 0; b < el_dim; b++)
									Dpsi_l[i] += gab_gai[b][i] * dpsids(l, b);
							}
							for (unsigned lel = 0; lel < eleminfo.nnode; lel++)
							{
								double Dq[3];
								for (unsigned int i = 0; i < n_dim; i++)
								{
									Dq[i] = 0.0;
									for (unsigned b = 0; b < el_dim; b++)
										Dq[i] += gab_gai[b][i] * dpsids_Element(lel, b);
								}
								for (unsigned lel2 = 0; lel2 < eleminfo.nnode; lel2++)
								{
									double Dr[3];
									for (unsigned int i = 0; i < n_dim; i++)
									{
										Dr[i] = 0.0;
										for (unsigned b = 0; b < el_dim; b++)
											Dr[i] += gab_gai[b][i] * dpsids_Element(lel2, b);
									}
									for (unsigned int i = 0; i < n_dim; i++)
										for (unsigned int j = 0; j < n_dim; j++)
											for (unsigned int j2 = 0; j2 < n_dim; j2++)
												shape_info->d2_dx2_shape_dcoord[ispace][l][i][lel][j][lel2][j2] =
													d2_shape_dcoord_closed_form(i, j, j2, Dpsi_l, Dq, Dr, false, __no_normal_projector, 0.0, 0.0, 0.0);
								}
							}
						}
						else if (require_hessian)
						{
						for (unsigned int i = 0; i < n_dim; i++)
							{
								for (unsigned lel= 0; lel < eleminfo.nnode; lel++)
								{
								for (unsigned lel2= 0; lel2 < eleminfo.nnode; lel2++)
								{
									for (unsigned int j = 0; j < n_dim; j++)
									{
									for (unsigned int j2 = 0; j2 < n_dim; j2++)
									{
										shape_info->d2_dx2_shape_dcoord[ispace][l][i][lel][j][lel2][j2] = 0.0;
										for (unsigned int b = 0; b < el_dim; b++)
										{
											shape_info->d2_dx2_shape_dcoord[ispace][l][i][lel][j][lel2][j2] += (*D2X2_dshape)(i,b,lel,lel2,j,j2) * dpsids(l, b);
										}
									}
									}
								}
							}
						}
						if (__paranoid_ale_identity)
						{
							// Differentiating the first-order identity a second time gives, for a bulk
							// element (where the normal-space projector N vanishes),
							//
							//   d2(D_i psi_l)/dX^q_j dX^r_k
							//        = (D_k psi_l)(D_j Psi_r)(D_i Psi_q) + (D_j psi_l)(D_k Psi_q)(D_i Psi_r)
							//
							// - two triple products of FIRST derivatives, symmetric under (q,j)<->(r,k) as
							// it must be. If that holds, the whole rank-6 array is redundant, and it is
							// 24.4% of Hessian assembly to fill (dev_docs/code_generation.md 9.4.11).
							// On an interface the second derivative additionally carries normal-projector
							// terms. Differentiating the codim-aware first-order identity again, and using
							// that a surface gradient is tangential (so N.Dpsi = 0) together with
							// P_ij = sum_q X^q_i (D_j Psi_q), which gives
							//     dN_ij/dX^r_k = -[ N_ik (D_j Psi_r) + N_jk (D_i Psi_r) ],
							// the whole thing collapses to the bulk part plus six N-terms. It is symmetric
							// under (q,j)<->(r,k) - the six pair up - and reduces to the bulk form at N=0.
							{
								double Dpsi_l[3], Nproj[3][3];
								for (unsigned int i = 0; i < n_dim; i++)
								{
									Dpsi_l[i] = 0.0;
									for (unsigned b = 0; b < el_dim; b++)
										Dpsi_l[i] += gab_gai[b][i] * dpsids(l, b);
									for (unsigned int j = 0; j < n_dim; j++)
									{
										double P = 0.0;
										for (unsigned b = 0; b < el_dim; b++)
											P += gab_gai[b][i] * interpolated_t(b, j);
										Nproj[i][j] = (i == j ? 1.0 : 0.0) - P;
									}
								}
								for (unsigned lel = 0; lel < eleminfo.nnode; lel++)
									for (unsigned lel2 = 0; lel2 < eleminfo.nnode; lel2++)
									{
										double Dq[3], Dr[3];
										for (unsigned int i = 0; i < n_dim; i++)
										{
											Dq[i] = Dr[i] = 0.0;
											for (unsigned b = 0; b < el_dim; b++)
											{
												Dq[i] += gab_gai[b][i] * dpsids_Element(lel, b);
												Dr[i] += gab_gai[b][i] * dpsids_Element(lel2, b);
											}
										}
										double dot_lr = 0.0, dot_lq = 0.0, dot_qr = 0.0;
										for (unsigned int m = 0; m < n_dim; m++)
										{
											dot_lr += Dpsi_l[m] * Dr[m];
											dot_lq += Dpsi_l[m] * Dq[m];
											dot_qr += Dq[m] * Dr[m];
										}
										for (unsigned int i = 0; i < n_dim; i++)
											for (unsigned int j = 0; j < n_dim; j++)
												for (unsigned int j2 = 0; j2 < n_dim; j2++)
												{
													const double expected = d2_shape_dcoord_closed_form(
														i, j, j2, Dpsi_l, Dq, Dr, el_dim != n_dim, Nproj, dot_lr, dot_lq, dot_qr);
													std::ostringstream where;
													where << "(space " << ispace << ", node " << l << ", dir " << i
														  << ", coord " << lel << "/" << j << " and " << lel2 << "/" << j2
														  << ", el_dim " << el_dim << ", nodal_dim " << n_dim << ")";
													ale_identity_check("d2_dx2_shape_dcoord", el_dim, n_dim,
																	   shape_info->d2_dx2_shape_dcoord[ispace][l][i][lel][j][lel2][j2],
																	   expected, where.str());
												}
									}
							}
						}
						}
					}
				}
			}
		}
		

		// Same pattern as above, for the discontinuous-Lagrange (DL) space (no "dominant space"
		// fallback since DL fields are never used to represent the geometry).
		const RequiredShapeFamilies fam_DL = required_shape_families_DL(required);
		if (fam_DL.any())
		{
			oomph::Shape psi(eleminfo.nnode_DL);
			oomph::DShape dpsids(eleminfo.nnode_DL, std::max((unsigned int)1, el_dim));
			// The DL basis is affine in s in every dimension, so d2psids is identically zero - but the
			// physical second derivative is not, since the Q term survives on a curved element.
			const bool space_d2x_DL = (__disable_shape_family_split ? require_d2x : fam_DL.d2x) && el_dim && eleminfo.nnode_DL;
			oomph::DShape d2psids(std::max((unsigned int)1, eleminfo.nnode_DL), MAX_N2DERIV);
			if (space_d2x_DL)
				this->d2shape_local_at_s_DL(s, psi, dpsids, d2psids);
			else
				this->dshape_local_at_s_DL(s, psi, dpsids);
			for (unsigned l = 0; l < eleminfo.nnode_DL; l++)
			{
				// Split per family exactly as for the continuous spaces above.
				if (fam_DL.psi)
					shape_info->shape_DL[l] = psi[l];
				if (fam_DL.dx)
				{
					for (unsigned int i = 0; i < n_dim; i++)
					{
						shape_info->dx_shape_DL[history_index][l][i] = 0.0;
						for (unsigned b = 0; b < el_dim; b++)
						{
							shape_info->dx_shape_DL[history_index][l][i] += gab_gai[b][i] * dpsids(l, b);
						}
					}
				}

				if (fam_DL.dX)
				{
					for (unsigned int i=0; i < this->dim();i++) shape_info->dS_shape_DL[l][i] =  dpsids(l, i);

					for (unsigned int i = 0; i < n_lagr; i++)
					{
						shape_info->dX_shape_DL[l][i] = 0.0;
						for (unsigned b = 0; b < el_dim; b++)
						{
							shape_info->dX_shape_DL[l][i] += gab_gai_Lagr[b][i] * dpsids(l, b);
						}
					}
				}

				if (space_d2x_DL)
				{
					for (unsigned int i = 0; i < n_dim; i++)
					{
						for (unsigned int j = 0; j < n_dim; j++)
						{
							double sum = 0.0;
							for (unsigned b = 0; b < el_dim; b++)
							{
								for (unsigned c = 0; c < el_dim; c++)
								{
									sum += gab_gai[b][i] * gab_gai[c][j] * d2psids(l, PYOOMPH_D2_SLOT(b, c));
									sum += gab_gai[c][j] * Qibc[i][b][c] * dpsids(l, b);
								}
							}
							shape_info->d2x_shape_DL[history_index][l][PYOOMPH_D2_SLOT(i, j)] = sum;
						}
					}
					for (unsigned a = 0; a < el_dim; a++)
						for (unsigned b = 0; b < el_dim; b++)
							shape_info->d2S_shape_DL[l][PYOOMPH_D2_SLOT(a, b)] = d2psids(l, PYOOMPH_D2_SLOT(a, b));
				}


					// Nodal-coordinate sensitivity of the second derivative. Differentiating
					//    D_ij psi = M_i^b M_j^c psi_,bc + M_j^c Q[i][b][c] psi_,b
					// by X^m_p only touches M and Q, since psi_,b and psi_,bc are reference-element
					// quantities.
					if (space_d2x_DL && require_dxdshape)
					{
						const unsigned ND = n_dim, ED = el_dim, NN = eleminfo.nnode;
						auto iM = [&](unsigned i2, unsigned b, unsigned m, unsigned p) { return ((i2 * ED + b) * NN + m) * ND + p; };
						auto iQ = [&](unsigned i2, unsigned b, unsigned c, unsigned m, unsigned p) { return (((i2 * ED + b) * ED + c) * NN + m) * ND + p; };
						for (unsigned int i = 0; i < n_dim; i++)
						{
							for (unsigned int j = 0; j < n_dim; j++)
							{
								const unsigned slot = PYOOMPH_D2_SLOT(i, j);
								for (unsigned m = 0; m < eleminfo.nnode; m++)
								{
									for (unsigned int p = 0; p < n_dim; p++)
									{
										double sum = 0.0;
										for (unsigned b = 0; b < el_dim; b++)
										{
											for (unsigned c = 0; c < el_dim; c++)
											{
												sum += (dM_dX[iM(i, b, m, p)] * gab_gai[c][j] + gab_gai[b][i] * dM_dX[iM(j, c, m, p)]) * d2psids(l, PYOOMPH_D2_SLOT(b, c));
												sum += (dM_dX[iM(j, c, m, p)] * Qibc[i][b][c] + gab_gai[c][j] * dQ_dX[iQ(i, b, c, m, p)]) * dpsids(l, b);
											}
										}
										shape_info->d_d2x_shape_dcoord_DL[l][slot][m][p] = sum;
									}
								}
							}
						}
					}
				if (require_dxdshape && fam_DL.dcoord)
				{
					for (unsigned int i = 0; i < n_dim; i++)
					{
						for (unsigned l2 = 0; l2 < eleminfo.nnode; l2++)
						{
							for (unsigned int i2 = 0; i2 < n_dim; i2++)
							{
								shape_info->d_dx_shape_dcoord_DL[l][i][l2][i2] = 0.0;
								for (unsigned int b = 0; b < el_dim; b++)
								{
									shape_info->d_dx_shape_dcoord_DL[l][i][l2][i2] += DXdshape_il_jb(i2, l2, i, b) * dpsids(l, b);
								}
							}
						}
					}
					if (__paranoid_ale_identity)
					{
						// Same closed form as for the continuous spaces above - the identity is a property
						// of the mapping, not of the space the shapes belong to.
						double Dpsi_l[3], Nproj[3][3];
						for (unsigned int i = 0; i < n_dim; i++)
						{
							Dpsi_l[i] = 0.0;
							for (unsigned b = 0; b < el_dim; b++)
								Dpsi_l[i] += gab_gai[b][i] * dpsids(l, b);
							for (unsigned int j = 0; j < n_dim; j++)
							{
								double P = 0.0;
								for (unsigned b = 0; b < el_dim; b++)
									P += gab_gai[b][i] * interpolated_t(b, j);
								Nproj[i][j] = (i == j ? 1.0 : 0.0) - P;
							}
						}
						for (unsigned l2 = 0; l2 < eleminfo.nnode; l2++)
						{
							double Dpsi_q[3], dot = 0.0;
							for (unsigned int i = 0; i < n_dim; i++)
							{
								Dpsi_q[i] = 0.0;
								for (unsigned b = 0; b < el_dim; b++)
									Dpsi_q[i] += gab_gai[b][i] * dpsids_Element(l2, b);
							}
							for (unsigned int a = 0; a < n_dim; a++)
								dot += Dpsi_l[a] * Dpsi_q[a];
							for (unsigned int i = 0; i < n_dim; i++)
								for (unsigned int i2 = 0; i2 < n_dim; i2++)
								{
									std::ostringstream where;
									where << "(space DL, node " << l << ", dir " << i << ", coord node " << l2
										  << ", coord dir " << i2 << ", el_dim " << el_dim
										  << ", nodal_dim " << n_dim << ")";
									ale_identity_check("d_dx_shape_dcoord_DL", el_dim, n_dim,
													   shape_info->d_dx_shape_dcoord_DL[l][i][l2][i2],
													   -Dpsi_l[i2] * Dpsi_q[i] + Nproj[i][i2] * dot,
													   where.str());
								}
						}
					}
					if (require_hessian && closed_form_d2)
					{
						// Same closed form as the continuous spaces above - the identity is a property of the
						// mapping, not of the space the shapes belong to. Without this branch the loop below
						// would dereference a D2X2_dshape that is no longer allocated.
						double Dpsi_l[3];
						for (unsigned int i = 0; i < n_dim; i++)
						{
							Dpsi_l[i] = 0.0;
							for (unsigned b = 0; b < el_dim; b++)
								Dpsi_l[i] += gab_gai[b][i] * dpsids(l, b);
						}
						for (unsigned lel = 0; lel < eleminfo.nnode; lel++)
						{
							double Dq[3];
							for (unsigned int i = 0; i < n_dim; i++)
							{
								Dq[i] = 0.0;
								for (unsigned b = 0; b < el_dim; b++)
									Dq[i] += gab_gai[b][i] * dpsids_Element(lel, b);
							}
							for (unsigned lel2 = 0; lel2 < eleminfo.nnode; lel2++)
							{
								double Dr[3];
								for (unsigned int i = 0; i < n_dim; i++)
								{
									Dr[i] = 0.0;
									for (unsigned b = 0; b < el_dim; b++)
										Dr[i] += gab_gai[b][i] * dpsids_Element(lel2, b);
								}
								for (unsigned int i = 0; i < n_dim; i++)
									for (unsigned int j = 0; j < n_dim; j++)
										for (unsigned int j2 = 0; j2 < n_dim; j2++)
											shape_info->d2_dx2_shape_dcoord_DL[l][i][lel][j][lel2][j2] =
												d2_shape_dcoord_closed_form(i, j, j2, Dpsi_l, Dq, Dr, false, __no_normal_projector, 0.0, 0.0, 0.0);
							}
						}
					}
					else if (require_hessian)
					{
					  for (unsigned int i = 0; i < n_dim; i++)
						{
							for (unsigned lel= 0; lel < eleminfo.nnode; lel++)
							{
							 for (unsigned lel2= 0; lel2 < eleminfo.nnode; lel2++)
							 {
								for (unsigned int j = 0; j < n_dim; j++)
								{
								 for (unsigned int j2 = 0; j2 < n_dim; j2++)
								 {
									shape_info->d2_dx2_shape_dcoord_DL[l][i][lel][j][lel2][j2] = 0.0;
									for (unsigned int b = 0; b < el_dim; b++)
									{
										shape_info->d2_dx2_shape_dcoord_DL[l][i][lel][j][lel2][j2] += (*D2X2_dshape)(i,b,lel,lel2,j,j2) * dpsids(l, b);
									}
								 }
								}
							}
						  }
					  }
					}
				}
			}
		}

		if (required.normal || required.normal_deriv)
		{
			oomph::Vector<double> unit_normal(this->nodal_dimension());
			this->get_normal_at_s(s, unit_normal, (require_dxdshape ? shape_info->d_normal_dcoord : NULL), ((require_hessian && require_dxdshape) ? shape_info->d2_normal_d2coord : NULL), history_index);
			for (unsigned int i = 0; i < nodal_dimension(); i++)
				shape_info->normal[history_index][i] = unit_normal[i];

			// ------------------------------------------------------------------------------------
			// First SPATIAL derivative of the normal.
			//
			// n is characterised by n.t_a = 0 and |n| = 1. Differentiating both by s_b gives
			// (dn/ds_b).t_a = -n.X_{,ab} and n.(dn/ds_b) = 0, and for codimension 1 the tangents
			// together with n span the ambient space, so dn/ds_b is fully determined. Contracting
			// with M_j^(b) = gab_gai[b][j]:
			//
			//    B_bc      := n_k X_{k,bc}                  (second fundamental form, local indices)
			//    dn_i/dx_j  = - M_i^(c) M_j^(b) B_bc
			//
			// which is symmetric in i,j (X_{k,bc} is), i.e. minus the Weingarten map, and whose trace
			// -g^{bc} B_bc is the mean curvature. n enters linearly, so whichever orientation
			// get_normal_at_s or oomph-lib's outer_unit_normal/normal_sign() picked is inherited.
			// Being written in the metric, this is one formula for every element dimension.
			if (required.normal_deriv)
			{
				for (unsigned int i = 0; i < n_dim; i++)
					for (unsigned int j = 0; j < n_dim; j++)
						shape_info->dnormal_dx[history_index][i][j] = 0.0;

				if (el_dim) // a point normal is constant, so its spatial derivative vanishes
				{
					// Codimension 1 is what makes dn/ds_b determined: n.t_a = 0 and |n| = 1 only pin it
					// down when {t_1..t_eldim, n} spans the ambient space. For a line in 3d the normal
					// is still perfectly well defined (oomph-lib returns the co-normal), but it has an
					// out-of-surface derivative that these two conditions say nothing about, so the
					// formula below would return a plausible wrong answer rather than fail.
					if (n_dim != el_dim + 1)
						throw_runtime_error("A spatial derivative of the normal, e.g. from grad(normal) or div(normal), is only defined for a codimension of 1, but this is a " + std::to_string(el_dim) + "-dimensional element in " + std::to_string(n_dim) + " dimensions.");
					std::string why;
					if (!this->supports_second_spatial_derivatives(why))
						throw_runtime_error("A spatial derivative of the normal, e.g. from grad(normal) or div(normal), was requested, but " + why);

					double Bbc[3][3];
					for (unsigned b = 0; b < el_dim; b++)
						for (unsigned c = 0; c < el_dim; c++)
						{
							double sum = 0.0;
							for (unsigned int k = 0; k < n_dim; k++)
								sum += unit_normal[k] * Xkab[k][PYOOMPH_D2_SLOT(b, c)];
							Bbc[b][c] = sum;
						}

					for (unsigned int i = 0; i < n_dim; i++)
						for (unsigned int j = 0; j < n_dim; j++)
						{
							double sum = 0.0;
							for (unsigned b = 0; b < el_dim; b++)
								for (unsigned c = 0; c < el_dim; c++)
									sum -= gab_gai[c][i] * gab_gai[b][j] * Bbc[b][c];
							shape_info->dnormal_dx[history_index][i][j] = sum;
						}

					// Nodal-coordinate sensitivities. Note the index sets differ: dn_k/dX and the
					// output live on the BULK element's nodes, while dM_dX and Psi_{m,bc} are indexed
					// by this element's own nodes, so the latter are scattered through
					// normal_coord_node(). Bulk nodes that are not on the face keep a zero entry,
					// which is correct - the unit normal does not depend on them.
					if (require_dxdshape && history_index == 0)
					{
						const unsigned n_coord_nodes = this->n_normal_coord_nodes();
						const unsigned NN = eleminfo.nnode, ND = n_dim, ED = el_dim;
						(void)NN;
						auto iM = [&](unsigned i2, unsigned b, unsigned m, unsigned p) { return ((i2 * ED + b) * NN + m) * ND + p; };

						for (unsigned int i = 0; i < n_dim; i++)
							for (unsigned int j = 0; j < n_dim; j++)
								for (unsigned m = 0; m < n_coord_nodes; m++)
									for (unsigned int p = 0; p < n_dim; p++)
										shape_info->d_dnormal_dx_dcoord[i][j][m][p] = 0.0;

						// The part that is naturally bulk-indexed: -M M (dn_k/dX^m_p) X_{k,bc}
						for (unsigned int i = 0; i < n_dim; i++)
							for (unsigned int j = 0; j < n_dim; j++)
								for (unsigned m = 0; m < n_coord_nodes; m++)
									for (unsigned int p = 0; p < n_dim; p++)
									{
										double sum = 0.0;
										for (unsigned b = 0; b < el_dim; b++)
											for (unsigned c = 0; c < el_dim; c++)
											{
												double dB = 0.0;
												for (unsigned int k = 0; k < n_dim; k++)
													dB += shape_info->d_normal_dcoord[k][m][p] * Xkab[k][PYOOMPH_D2_SLOT(b, c)];
												sum -= gab_gai[c][i] * gab_gai[b][j] * dB;
											}
										shape_info->d_dnormal_dx_dcoord[i][j][m][p] += sum;
									}

						// The parts indexed by this element's own nodes, scattered onto bulk nodes:
						// the two dM/dX terms and the n_p Psi_{m,bc} term.
						for (unsigned mloc = 0; mloc < eleminfo.nnode; mloc++)
						{
							const unsigned m = this->normal_coord_node(mloc);
							for (unsigned int p = 0; p < n_dim; p++)
								for (unsigned int i = 0; i < n_dim; i++)
									for (unsigned int j = 0; j < n_dim; j++)
									{
										double sum = 0.0;
										for (unsigned b = 0; b < el_dim; b++)
											for (unsigned c = 0; c < el_dim; c++)
											{
												sum -= (dM_dX[iM(i, c, mloc, p)] * gab_gai[b][j] + gab_gai[c][i] * dM_dX[iM(j, b, mloc, p)]) * Bbc[b][c];
												sum -= gab_gai[c][i] * gab_gai[b][j] * unit_normal[p] * d2psids_Element(mloc, PYOOMPH_D2_SLOT(b, c));
											}
										shape_info->d_dnormal_dx_dcoord[i][j][m][p] += sum;
									}
						}

						// Second nodal-coordinate derivative. Differentiating
						//   dn_i/dx_j = -M_i^c M_j^b B_bc ,  B_bc = n_k X_{k,bc}
						// twice gives nine groups; with d2X/dXdX = 0 (X is linear in the nodal
						// positions) the only new ingredients are d2M_dXdX and d2_normal_d2coord,
						// both already available.
						//
						// The two index sets collide here on BOTH slots, so rather than classifying
						// every term as face/face, face/bulk or bulk/bulk, the face-indexed quantities
						// are first scattered into bulk-indexed temporaries. Every term below is then
						// written in one index set. Off-face bulk entries stay zero, which is correct.
						if (require_hessian && shape_info->d2_dnormal_dx_d2coord && shape_info->d2_normal_d2coord)
						{
							const unsigned NX = n_coord_nodes * ND;
							auto mp_of = [&](unsigned m, unsigned p) { return m * ND + p; };
							// dM_i^(c)/dX, scattered to bulk nodes
							std::vector<double> dMb(ND * ED * NX, 0.0);
							auto iMb = [&](unsigned i2, unsigned c, unsigned mp) { return (i2 * ED + c) * NX + mp; };
							// Psi_{m,bc}, scattered to bulk nodes
							std::vector<double> Psib(NX * MAX_N2DERIV, 0.0);
							// d2M_i^(c)/dXdX, scattered on both slots
							std::vector<double> d2Mb(ND * ED * NX * NX, 0.0);
							auto iM2b = [&](unsigned i2, unsigned c, unsigned mp, unsigned mp2) { return ((i2 * ED + c) * NX + mp) * NX + mp2; };
							const unsigned NXloc = eleminfo.nnode * ND;
							auto iM2loc = [&](unsigned i2, unsigned c, unsigned mp, unsigned mp2) { return ((i2 * ED + c) * NXloc + mp) * NXloc + mp2; };

							for (unsigned mloc = 0; mloc < eleminfo.nnode; mloc++)
							{
								const unsigned m = this->normal_coord_node(mloc);
								for (unsigned int p = 0; p < n_dim; p++)
								{
									for (unsigned int i = 0; i < n_dim; i++)
										for (unsigned c = 0; c < el_dim; c++)
											dMb[iMb(i, c, mp_of(m, p))] += dM_dX[iM(i, c, mloc, p)];
									for (unsigned b = 0; b < el_dim; b++)
										for (unsigned c = 0; c < el_dim; c++)
											Psib[mp_of(m, p) * MAX_N2DERIV + PYOOMPH_D2_SLOT(b, c)] = d2psids_Element(mloc, PYOOMPH_D2_SLOT(b, c));
								}
							}
							if (!d2M_dXdX.empty())
							{
								for (unsigned mloc = 0; mloc < eleminfo.nnode; mloc++)
									for (unsigned int p = 0; p < n_dim; p++)
										for (unsigned m2loc = 0; m2loc < eleminfo.nnode; m2loc++)
											for (unsigned int p2 = 0; p2 < n_dim; p2++)
											{
												const unsigned dst = mp_of(this->normal_coord_node(mloc), p), dst2 = mp_of(this->normal_coord_node(m2loc), p2);
												for (unsigned int i = 0; i < n_dim; i++)
													for (unsigned c = 0; c < el_dim; c++)
														d2Mb[iM2b(i, c, dst, dst2)] += d2M_dXdX[iM2loc(i, c, mp_of(mloc, p), mp_of(m2loc, p2))];
											}
							}

							for (unsigned int i = 0; i < n_dim; i++)
								for (unsigned int j = 0; j < n_dim; j++)
									for (unsigned m = 0; m < n_coord_nodes; m++)
										for (unsigned int p = 0; p < n_dim; p++)
											for (unsigned m2 = 0; m2 < n_coord_nodes; m2++)
												for (unsigned int p2 = 0; p2 < n_dim; p2++)
												{
													const unsigned mp = mp_of(m, p), mp2 = mp_of(m2, p2);
													double sum = 0.0;
													for (unsigned b = 0; b < el_dim; b++)
														for (unsigned c = 0; c < el_dim; c++)
														{
															const unsigned sl = PYOOMPH_D2_SLOT(b, c);
															// dB/dX, d'B/dX' and d'dB
															double dB = unit_normal[p] * Psib[mp * MAX_N2DERIV + sl];
															double dpB = unit_normal[p2] * Psib[mp2 * MAX_N2DERIV + sl];
															double d2B = 0.0;
															for (unsigned int k = 0; k < n_dim; k++)
															{
																dB += shape_info->d_normal_dcoord[k][m][p] * Xkab[k][sl];
																dpB += shape_info->d_normal_dcoord[k][m2][p2] * Xkab[k][sl];
																d2B += shape_info->d2_normal_d2coord[k][m][p][m2][p2] * Xkab[k][sl];
															}
															d2B += shape_info->d_normal_dcoord[p2][m][p] * Psib[mp2 * MAX_N2DERIV + sl];
															d2B += shape_info->d_normal_dcoord[p][m2][p2] * Psib[mp * MAX_N2DERIV + sl];

															sum -= d2Mb[iM2b(i, c, mp, mp2)] * gab_gai[b][j] * Bbc[b][c];
															sum -= gab_gai[c][i] * d2Mb[iM2b(j, b, mp, mp2)] * Bbc[b][c];
															sum -= (dMb[iMb(i, c, mp)] * dMb[iMb(j, b, mp2)] + dMb[iMb(i, c, mp2)] * dMb[iMb(j, b, mp)]) * Bbc[b][c];
															sum -= (dMb[iMb(i, c, mp)] * gab_gai[b][j] + gab_gai[c][i] * dMb[iMb(j, b, mp)]) * dpB;
															sum -= (dMb[iMb(i, c, mp2)] * gab_gai[b][j] + gab_gai[c][i] * dMb[iMb(j, b, mp2)]) * dB;
															sum -= gab_gai[c][i] * gab_gai[b][j] * d2B;
														}
													shape_info->d2_dnormal_dx_d2coord[i][j][m][p][m2][p2] = sum;
												}
						}
					}
				}
			}
		}
		
		if (D2X2_dshape) delete D2X2_dshape;

		return det_Eulerian;
	}


	// Points the generic "Pos" shape pointers (shape_Pos, dx_shape_Pos, ...) to whichever concrete
	// space (C2TB/C2/C1TB/C1) actually represents the element's geometry, so JIT code that reads
	// shape_info->shape_Pos etc. does not need to know which space is dominant. Falls back down
	// the priority chain C2TB -> C2 -> C1TB -> C1 depending on which nodes/spaces are present.
	void BulkElementBase::set_remaining_shapes_appropriately(JITShapeInfo_t *shape_info, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes)
	{
		// Same predicate as the fill, from the same helper: the two used to spell it out separately and
		// had to agree, or the Pos aliases would point at a space that was never filled.
		const bool required_C2TB = this->required_shape_families(required_shapes, SPACE_INDEX_C2TB).any();
		const bool required_C1TB = this->required_shape_families(required_shapes, SPACE_INDEX_C1TB).any();
		
		if (required_C2TB)
		{
			shape_info->shape_Pos = shape_info->shapes[SPACE_INDEX_C2TB];
			for (unsigned k = 0; k < 3; k++) shape_info->dx_shape_Pos[k] = shape_info->dx_shapes[k][SPACE_INDEX_C2TB];
			shape_info->dX_shape_Pos = shape_info->dX_shapes[SPACE_INDEX_C2TB];
			shape_info->dS_shape_Pos = shape_info->dS_shapes[SPACE_INDEX_C2TB];
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord[SPACE_INDEX_C2TB];
			for (unsigned k = 0; k < 3; k++) shape_info->d2x_shape_Pos[k] = shape_info->d2x_shapes[k][SPACE_INDEX_C2TB];
			shape_info->d2S_shape_Pos = shape_info->d2S_shapes[SPACE_INDEX_C2TB];
			shape_info->d_d2x_shape_dcoord_Pos = shape_info->d_d2x_shape_dcoord[SPACE_INDEX_C2TB];
			shape_info->d2_d2x2_shape_dcoord_Pos = shape_info->d2_d2x2_shape_dcoord[SPACE_INDEX_C2TB];
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord[SPACE_INDEX_C2TB];
		}
		else if (this->eleminfo.nnode_of_space[SPACE_INDEX_C2])
		{
			shape_info->shape_Pos = shape_info->shapes[SPACE_INDEX_C2];
			for (unsigned k = 0; k < 3; k++) shape_info->dx_shape_Pos[k] = shape_info->dx_shapes[k][SPACE_INDEX_C2];
			shape_info->dX_shape_Pos = shape_info->dX_shapes[SPACE_INDEX_C2];
			shape_info->dS_shape_Pos = shape_info->dS_shapes[SPACE_INDEX_C2];
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord[SPACE_INDEX_C2];
			for (unsigned k = 0; k < 3; k++) shape_info->d2x_shape_Pos[k] = shape_info->d2x_shapes[k][SPACE_INDEX_C2];
			shape_info->d2S_shape_Pos = shape_info->d2S_shapes[SPACE_INDEX_C2];
			shape_info->d_d2x_shape_dcoord_Pos = shape_info->d_d2x_shape_dcoord[SPACE_INDEX_C2];
			shape_info->d2_d2x2_shape_dcoord_Pos = shape_info->d2_d2x2_shape_dcoord[SPACE_INDEX_C2];
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord[SPACE_INDEX_C2];
		}
		else if (required_C1TB)
		{
			shape_info->shape_Pos = shape_info->shapes[SPACE_INDEX_C1TB];
			for (unsigned k = 0; k < 3; k++) shape_info->dx_shape_Pos[k] = shape_info->dx_shapes[k][SPACE_INDEX_C1TB];
			shape_info->dX_shape_Pos = shape_info->dX_shapes[SPACE_INDEX_C1TB];
			shape_info->dS_shape_Pos = shape_info->dS_shapes[SPACE_INDEX_C1TB];
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord[SPACE_INDEX_C1TB];
			for (unsigned k = 0; k < 3; k++) shape_info->d2x_shape_Pos[k] = shape_info->d2x_shapes[k][SPACE_INDEX_C1TB];
			shape_info->d2S_shape_Pos = shape_info->d2S_shapes[SPACE_INDEX_C1TB];
			shape_info->d_d2x_shape_dcoord_Pos = shape_info->d_d2x_shape_dcoord[SPACE_INDEX_C1TB];
			shape_info->d2_d2x2_shape_dcoord_Pos = shape_info->d2_d2x2_shape_dcoord[SPACE_INDEX_C1TB];		
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord[SPACE_INDEX_C1TB];
		}		
		else
		{
		  
			shape_info->shape_Pos = shape_info->shapes[SPACE_INDEX_C1];
			for (unsigned k = 0; k < 3; k++) shape_info->dx_shape_Pos[k] = shape_info->dx_shapes[k][SPACE_INDEX_C1];
			shape_info->dX_shape_Pos = shape_info->dX_shapes[SPACE_INDEX_C1];
			shape_info->dS_shape_Pos = shape_info->dS_shapes[SPACE_INDEX_C1];
			shape_info->d_dx_shape_dcoord_Pos = shape_info->d_dx_shape_dcoord[SPACE_INDEX_C1];
			for (unsigned k = 0; k < 3; k++) shape_info->d2x_shape_Pos[k] = shape_info->d2x_shapes[k][SPACE_INDEX_C1];
			shape_info->d2S_shape_Pos = shape_info->d2S_shapes[SPACE_INDEX_C1];
			shape_info->d_d2x_shape_dcoord_Pos = shape_info->d_d2x_shape_dcoord[SPACE_INDEX_C1];
			shape_info->d2_d2x2_shape_dcoord_Pos = shape_info->d2_d2x2_shape_dcoord[SPACE_INDEX_C1];
			shape_info->d2_dx2_shape_dcoord_Pos=shape_info->d2_dx2_shape_dcoord[SPACE_INDEX_C1];
		}
	}

	// Fills shape_info for a single Gauss integration point ipt: evaluates fill_shape_info_at_s()
	// at that point's local coordinate (and, if history-weighted time-integrals are required,
	// also at the previous one or two history configurations, to get their Jacobians), and stores
	// the combined integration weights (weight * Jacobian) used by JIT code to form dx/dX/etc.
	void BulkElementBase::fill_shape_buffer_for_integration_point(unsigned ipt, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag)
	{
		oomph::Vector<double> s(this->dim());
		for (unsigned int i = 0; i < this->dim(); i++)
			s[i] = integral_pt()->knot(ipt, i);
        double w = integral_pt()->weight(ipt);
		
		if (required_shapes.history_integral_dx1 || required_shapes.history_integral_dx2)
		{			
			JITFuncSpec_RequiredShapes_FiniteElement_t simplified_required_shapes;
			memset(&simplified_required_shapes, 0, sizeof(JITFuncSpec_RequiredShapes_FiniteElement_t));			
			double JLagr_dummy;			
			if (required_shapes.history_integral_dx1)
			{
			  double Jhistory = fill_shape_info_at_s(s, ipt, simplified_required_shapes,  JLagr_dummy, 0,NULL,1);
			  shape_info->int_pt_weight[1] = w * Jhistory;
			}
			if (required_shapes.history_integral_dx2)
			{
			  double Jhistory = fill_shape_info_at_s(s, ipt, simplified_required_shapes, JLagr_dummy, 0,NULL,2);
			  shape_info->int_pt_weight[2] = w * Jhistory;
			}
		}

		// Same integration point, but on the mesh as it was one or two steps ago. fill_shape_info_at_s
		// builds all the geometry from the nodal positions at the requested history level and writes the
		// configuration-dependent arrays into that level's slot, so these calls do not disturb the
		// current-time fill below - everything they share with it (the undifferentiated shapes, the
		// Lagrangian and local-coordinate derivatives) does not move with the mesh. Interfaces are
		// covered too: the nested calls carry the same history level, so each element of the chain ends
		// up with its own history slots filled.
		for (unsigned k = 1; k <= 2; k++)
		{
			bool wanted = (k == 1 ? required_shapes.history_geometry1 : required_shapes.history_geometry2);
			if (!wanted) continue;
			double JLagr_hist;
			fill_shape_info_at_s(s, ipt, required_shapes, JLagr_hist, 0, NULL, k);
		}

		double JLagr;
		// J is the factor that turns the integration weight into the physical dx below. It is
		// sqrt(det(g_ab)) built from the metric tensor, which is how pyoomph can integrate over
		// elements of lower dimension than the nodal space (interface lines in 2D, surfaces in 3D)
		// -- but it is therefore NON-NEGATIVE BY CONSTRUCTION and says nothing about orientation.
		// An element that has turned inside out has a perfectly ordinary positive J. To see the
		// inversion we need the SIGN of the mapping, which only exists when the mapping is square
		// (el_dim == nodal_dim), and which we get from the tangent matrix dx/ds that
		// fill_shape_info_at_s() already hands out through its optional out-parameter.
		//
		// This is the hottest loop in the code -- once per integration point per element per
		// assembly -- so the detection is kept strictly off to the side: when
		// detect_inverted_elements is false (the default) the branch below short-circuits on a
		// static bool before any virtual dim()/nodal_dimension() call, and the dxds scratch matrix
		// is never even constructed. Measured on a 60x60 quad mesh (14520 dofs, 240 interleaved
		// Jacobian assemblies per arm): disabled is indistinguishable from a build with this block
		// deleted entirely (within the ~2% run-to-run spread), enabled costs about +2%. That +2%
		// is almost entirely the "*dxds = interpolated_t" DenseMatrix copy-assign inside
		// fill_shape_info_at_s(), i.e. one small heap allocation per integration point -- if this
		// ever needs to be cheaper, compute the signed determinant there, where interpolated_t
		// already exists, instead of copying the matrix out.
		double J;
		if (!detect_inverted_elements)
		{
			J = fill_shape_info_at_s(s, ipt, required_shapes, JLagr, flag);
		}
		else if (this->dim() == 0 || this->dim() != this->nodal_dimension())
		{
			// No square mapping, hence no orientation to lose: interface/point elements.
			J = fill_shape_info_at_s(s, ipt, required_shapes, JLagr, flag);
		}
		else
		{
			oomph::DenseMatrix<double> dxds;
			J = fill_shape_info_at_s(s, ipt, required_shapes, JLagr, flag, &dxds);

			// Signed determinant of dx/ds. Negative means the element is inside out, zero means it
			// has collapsed; either way everything assembled from here on is meaningless. Raising
			// it now, rather than letting the garbage residual propagate, is what lets
			// adaptive_unsteady_newton_solve and the arclength loop reject the step and retry with
			// a smaller one. InvertedElementError derives from oomph::OomphLibError, so a caller
			// that knows nothing about it still sees an ordinary oomph-lib error, not a crash.
			double detJ;
			switch (this->dim())
			{
			case 1:
				detJ = dxds(0, 0);
				break;
			case 2:
				detJ = dxds(0, 0) * dxds(1, 1) - dxds(0, 1) * dxds(1, 0);
				break;
			default:
				detJ = dxds(0, 0) * (dxds(1, 1) * dxds(2, 2) - dxds(1, 2) * dxds(2, 1)) - dxds(0, 1) * (dxds(1, 0) * dxds(2, 2) - dxds(1, 2) * dxds(2, 0)) + dxds(0, 2) * (dxds(1, 0) * dxds(2, 1) - dxds(1, 1) * dxds(2, 0));
				break;
			}
			if (detJ <= 0.0)
			{
				std::ostringstream oss;
				oss << "Inverted element: the signed determinant of the Eulerian mapping dx/ds is "
					<< detJ << " (must be > 0) at integration point " << ipt << " of a "
					<< this->dim() << "-dimensional element";
				if (codeinst && codeinst->get_func_table() && codeinst->get_func_table()->domain_name)
					oss << " on domain '" << codeinst->get_func_table()->domain_name << "'";
				oss << "." << std::endl
					<< "This usually means the mesh has been distorted too far, e.g. by too large a "
					<< "time step or continuation step." << std::endl
					<< "Detection can be switched off again with set_detect_inverted_elements(False).";
				throw oomph::InvertedElementError(oss.str(), OOMPH_CURRENT_FUNCTION, OOMPH_EXCEPTION_LOCATION);
			}
		}

		if (__paranoid_ale_identity && !(J > 0.0 && std::isfinite(J)))
		{
			std::ostringstream key;
			key << "DEGENERATE MAPPING J=" << (std::isfinite(J) ? "finite" : "non-finite")
				<< " [el_dim " << this->dim() << " in " << this->nodal_dimension() << "D, nnode "
				<< this->nnode() << "]";
			auto &e = __ale_identity_stats[key.str()];
			e.n++;
			e.bad++;
			if (e.worst_where.empty())
			{
				std::ostringstream oss;
				oss << "J=" << J << " at integration point " << ipt << " of domain '"
					<< ((codeinst && codeinst->get_func_table() && codeinst->get_func_table()->domain_name)
							? codeinst->get_func_table()->domain_name : "?")
					<< "', node positions:";
				for (unsigned l = 0; l < this->nnode(); l++)
				{
					oss << " (";
					for (unsigned d = 0; d < this->nodal_dimension(); d++)
						oss << (d ? "," : "") << this->node_pt(l)->x(d);
					oss << ")";
				}
				e.worst_where = oss.str();
			}
		}
		shape_info->int_pt_weight_unity= w;
		shape_info->int_pt_weight[0] = w * J;
		shape_info->int_pt_weight_Lagrangian = w * JLagr;

		if (__poison_everything)
		{
			// Positive control, see __poison_everything: everything the generated code is about to read
			// is destroyed here, so every case MUST come out NaN. If one does not, the buffers it reads
			// are not the ones this tool poisons and its "clean" verdict is worthless.
			JITFuncSpec_RequiredShapes_FiniteElement_t nothing;
			memset(&nothing, 0, sizeof(nothing));
			this->poison_unrequired_shapes(nothing, shape_info, false);
		}

		if (__paranoid_ale_identity)
		{
			// The remaining two identities can only be checked once the integration weight is known and
			// the "Pos" aliases have been resolved (set_remaining_shapes_appropriately runs once per
			// assembly, before this). Checking them through dx_shape_Pos rather than through the space
			// they alias is deliberate: it is exactly the assumption that the position space IS the
			// geometry space that the generated code makes, and that ConstrainPositionsToC1Space could
			// in principle break.
			const JITFuncSpec_Table_FiniteElement_t *ft = codeinst->get_func_table();
			if (flag && ft->moving_nodes && !ft->fd_position_jacobian && shape_info->dx_shape_Pos[0])
			{
				const unsigned ND = this->nodal_dimension();
				const unsigned NQ = this->nnode();
				for (unsigned q = 0; q < NQ; q++)
					for (unsigned j = 0; j < ND; j++)
					{
						std::ostringstream where;
						where << "(coord node " << q << ", coord dir " << j << ", el_dim " << this->dim()
							  << ", nodal_dim " << ND << ")";
						// d(dx)/dX^q_j = dx * D_j Psi_q, i.e. d(det J)/dX = det J * dpsi_q/dx_j.
						ale_identity_check("int_pt_weights_d_coords", this->dim(), ND,
										   shape_info->int_pt_weights_d_coords[j][q],
										   shape_info->int_pt_weight[0] * shape_info->dx_shape_Pos[0][q][j],
										   where.str());
						// d(n_i)/dX^q_j = -n_j D_i Psi_q. Only meaningful where a normal exists at all.
						// Gated on normal_deriv, not on normal: d_normal_dcoord is written whenever a normal
						// is wanted at all, but the generated code only ever READS it when it needs the
						// sensitivity. Checking the write-only case reports mismatches in values nobody
						// consumes - established, not assumed: with the guard widened to `normal`, 96 of
						// 108 entries disagree on a 2D free surface while the normal_deriv cases are
						// exact. Worth a look on its own, but it is not this identity.
						if (required_shapes.normal_deriv && shape_info->d_normal_dcoord)
							for (unsigned i = 0; i < ND; i++)
							{
								std::ostringstream w2;
								w2 << "(normal comp " << i << ", coord node " << q << ", coord dir " << j
								   << ", el_dim " << this->dim() << ", nodal_dim " << ND << ")";
								ale_identity_check("d_normal_dcoord", this->dim(), ND,
												   shape_info->d_normal_dcoord[i][q][j],
												   -shape_info->normal[0][j] * shape_info->dx_shape_Pos[0][q][i],
												   w2.str());
							}
					}
			}
		}

	}

	// One-time (per residual/Jacobian assembly, not per integration point) setup of shape_info:
	// caches the number of integration points, the current time values/timesteps, and the
	// per-history-value BDF1/BDF2/Newmark2 weights used by JIT code to form time derivatives
	// (degrading to lower-order weights while too few unsteady steps have been taken yet), then
	// resolves the "Pos" shape aliases and computes element-size related quantities. Must be
	// called before fill_shape_buffer_for_integration_point() is used for the individual points.
	void BulkElementBase::prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (__poison_unrequired)
			this->poison_unrequired_shapes(required_shapes, shape_info, true);
		shape_info->n_int_pt = integral_pt()->nweight();

		const oomph::TimeStepper *tstepper = (this->nnode() ? this->node_pt(0)->time_stepper_pt() : this->internal_data_pt(0)->time_stepper_pt());
		if (tstepper->is_steady())
		{
			shape_info->timestepper_ntstorage = 0;
			for (unsigned int i = 0; i < tstepper->ntstorage(); i++)
			{
				shape_info->timestepper_weights_dt_BDF1[i] = 0;
				shape_info->timestepper_weights_dt_BDF2[i] = 0;
				shape_info->timestepper_weights_dt_Newmark2[i] = 0;
				if (functable->max_dt_order > 1)
					shape_info->timestepper_weights_d2t_Newmark2[i] = 0;
			}
			shape_info->timestepper_weights_dt_BDF2_degr = shape_info->timestepper_weights_dt_BDF2;
			shape_info->timestepper_weights_dt_Newmark2_degr = shape_info->timestepper_weights_dt_Newmark2;
		}
		else
		{
			shape_info->timestepper_ntstorage = tstepper->ntstorage();
			const MultiTimeStepper *mtstepper = dynamic_cast<const MultiTimeStepper *>(tstepper);
			if (mtstepper)
			{
				for (unsigned int i = 0; i < shape_info->timestepper_ntstorage; i++)
				{
					shape_info->timestepper_weights_dt_BDF1[i] = mtstepper->weightBDF1(1, i);
					shape_info->timestepper_weights_dt_BDF2[i] = mtstepper->weightBDF2(1, i);
					shape_info->timestepper_weights_dt_Newmark2[i] = mtstepper->weightNewmark2(1, i);
					if (functable->max_dt_order > 1)
						shape_info->timestepper_weights_d2t_Newmark2[i] = mtstepper->weightNewmark2(2, i);
				}
				unsigned unsteady_steps_done = mtstepper->get_num_unsteady_steps_done();
				if (unsteady_steps_done == 0)
				{
					shape_info->timestepper_weights_dt_BDF2_degr = shape_info->timestepper_weights_dt_BDF1;
					shape_info->timestepper_weights_dt_Newmark2_degr = shape_info->timestepper_weights_dt_BDF1;
				}				
				else
				{
					shape_info->timestepper_weights_dt_BDF2_degr = shape_info->timestepper_weights_dt_BDF2;
					shape_info->timestepper_weights_dt_Newmark2_degr = shape_info->timestepper_weights_dt_Newmark2;
				}
			}
			else
			{
				throw_runtime_error("Only the MultiTimeStepper is allowed");
			}
		}
		for (unsigned int tt = 0; tt < tstepper->time_pt()->ndt(); tt++)
		{
			shape_info->t[tt] = tstepper->time_pt()->time(tt);
			shape_info->dt[tt] = tstepper->time_pt()->dt(tt);
		}

		set_remaining_shapes_appropriately(shape_info, required_shapes);

		_currently_assembled_element = this;
		
      // Should be fine here!
      this->fill_shape_info_element_sizes(required_shapes,shape_info,flag);

		// Element sizes on the mesh as it was one or two steps ago. Unlike the shape derivatives these
		// are per element rather than per integration point, so they are done here. Each call writes only
		// its own history slot; the dx_shapes it clobbers in passing (it sweeps the integration points
		// internally) are rewritten by the per-integration-point fill that follows.
		for (unsigned k = 1; k <= 2; k++)
		{
			bool wanted = (k == 1 ? required_shapes.history_geometry1 : required_shapes.history_geometry2);
			if (!wanted) continue;
			this->fill_shape_info_element_sizes(required_shapes, shape_info, 0, k);
		}
		
	}

	// Evaluates a user-defined "integral expression" (an expression integrated over the element);
	// the actual loop over integration points happens inside the JIT-generated
	// EvalIntegralExpression, which reads the prepared shape_info buffer.
	double BulkElementBase::eval_integral_expression(unsigned index)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numintegral_expressions)
			throw_runtime_error("Cannot evaluate integral expression at too large index " + std::to_string(index));		
		this->interpolate_hang_values();
		prepare_shape_buffer_for_integration(functable->shapes_required_IntegralExprs, 0);
		return functable->EvalIntegralExpression(&eleminfo, this->shape_info, index);
	}

	// Evaluates a user-defined "local expression" (a pointwise, non-integrated expression) at a
	// given node's local coordinate.
	double BulkElementBase::eval_local_expression_at_node(unsigned index, unsigned node_index)
	{
		oomph::Vector<double> s;
		this->local_coordinate_of_node(node_index, s);
		return eval_local_expression_at_s(index, s);
	}

	// Evaluates a user-defined "local expression" at an arbitrary local coordinate s.
	double BulkElementBase::eval_local_expression_at_s(unsigned index, const oomph::Vector<double> &s)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numlocal_expressions)
			throw_runtime_error("Cannot evaluate local expression at too large index " + std::to_string(index));
		
		this->interpolate_hang_values();

		double JLagr;
		this->fill_shape_info_at_s(s, 0, codeinst->get_func_table()->shapes_required_LocalExprs, JLagr, 0);
		this->prepare_shape_buffer_for_integration(codeinst->get_func_table()->shapes_required_LocalExprs, 0);
//		set_remaining_shapes_appropriately(shape_info, codeinst->get_func_table()->shapes_required_LocalExprs);
      _currently_assembled_element = this;
	    //std::cout << "CALLING EVAL LOCAL EXPRESSION  " << this << " ELEMINFO " << &eleminfo << std::endl;
		return functable->EvalLocalExpression(&eleminfo, this->shape_info, index);
	}

	// Evaluates a user-defined "extremum expression" (used e.g. to track min/max of some field)
	// at a given node's local coordinate.
	double BulkElementBase::eval_extremum_expression_at_node(unsigned index, unsigned node_index)
	{
		oomph::Vector<double> s;
		this->local_coordinate_of_node(node_index, s);
		return eval_extremum_expression_at_s(index, s);
	}

	// Evaluates a user-defined "extremum expression" at an arbitrary local coordinate s.
	double BulkElementBase::eval_extremum_expression_at_s(unsigned index, const oomph::Vector<double> &s)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numextremum_expressions)
			throw_runtime_error("Cannot evaluate extremum expression at too large index " + std::to_string(index));
		
		this->interpolate_hang_values();

		double JLagr;
		this->fill_shape_info_at_s(s, 0, codeinst->get_func_table()->shapes_required_ExtremumExprs, JLagr, 0);
		this->prepare_shape_buffer_for_integration(codeinst->get_func_table()->shapes_required_ExtremumExprs, 0);
      _currently_assembled_element = this;	    
		return functable->EvalExtremumExpression(&eleminfo, this->shape_info, index);
	}	

	// Deliberately NOT built on fill_shape_info_at_s: this is called several times per Runge-Kutta
	// stage and needs only the local shape derivatives and the nodal position history, none of the
	// metric tensors, field interpolations and Jacobian bookkeeping that the shape buffer fill does.
	// It also has to work unchanged on an interface element, where el_dim < nodal_dim.
	void BulkElementBase::tracer_geometry_at_s(const oomph::Vector<double> &s, unsigned nlevel, const double *w, const double *dwdtau,
											   oomph::Vector<double> *x, oomph::DenseMatrix<double> *J,
											   oomph::Vector<double> *dXdtau) const
	{
		const unsigned el_dim = this->dim();
		const unsigned n_dim = this->nodal_dimension();
		const unsigned n_node = this->nnode();

		oomph::Shape psi(n_node);
		oomph::DShape dpsids(n_node, std::max((unsigned)1, el_dim));
		this->dshape_local(s, psi, dpsids);

		if (x)
			x->assign(n_dim, 0.0);
		if (J)
		{
			J->resize(el_dim, n_dim);
			J->initialise(0.0);
		}
		if (dXdtau)
			dXdtau->assign(n_dim, 0.0);

		for (unsigned l = 0; l < n_node; l++)
		{
			for (unsigned i = 0; i < n_dim; i++)
			{
				double xl = 0.0, xdotl = 0.0;
				for (unsigned k = 0; k < nlevel; k++)
				{
					const double pos = this->nodal_position(k, l, i);
					xl += w[k] * pos;
					if (dXdtau)
						xdotl += dwdtau[k] * pos;
				}
				if (x)
					(*x)[i] += psi(l) * xl;
				if (dXdtau)
					(*dXdtau)[i] += psi(l) * xdotl;
				if (J)
					for (unsigned a = 0; a < el_dim; a++)
						(*J)(a, i) += dpsids(l, a) * xl;
			}
		}
	}

	// Per-element part of a tracer evaluation: timestepper weights, time levels and element sizes,
	// none of which depend on where in the element the particle sits. Split out from
	// eval_tracer_advection_at_s because that runs several times per Runge-Kutta sub-step while a
	// particle stays in the same element, and prepare_shape_buffer_for_integration sweeps the
	// element's integration points. The old code called neither, so any partial_t() inside an
	// advection expression read whatever timestepper weights the last assembly had left behind.
	void BulkElementBase::tracer_prepare_element()
	{
		this->interpolate_hang_values();
		this->prepare_shape_buffer_for_integration(codeinst->get_func_table()->shapes_required_TracerAdvection, 0);
	}

	// Evaluates a user-defined tracer-advection velocity field, in physical/Eulerian components, at
	// an arbitrary local coordinate. Must be preceded by tracer_prepare_element() for this element.
	//
	// The history-geometry slots are filled first where the generated code asks for them, so that a
	// gradient inside a past-level advection expression is taken on the configuration that level
	// belongs to rather than on the current one.
	//
	// xvelo is sized 3: a coordinate system may produce more velocity components than the mesh has
	// dimensions (an out-of-plane swirl on an axisymmetric mesh). Those extra components cannot move
	// a particle that lives in the mesh; it is the caller's job to notice and complain rather than to
	// drop them silently.
	void BulkElementBase::eval_tracer_advection_at_s(unsigned index, const oomph::Vector<double> &s, oomph::Vector<double> &xvelo)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = codeinst->get_func_table();
		if (index >= functable->numtracer_advections)
			throw_runtime_error("Cannot evaluate tracer advection at too large index " + std::to_string(index));
		const JITFuncSpec_RequiredShapes_FiniteElement_t &required = functable->shapes_required_TracerAdvection;

		for (unsigned k = 1; k <= 2; k++)
		{
			if (k == 1 ? required.history_geometry1 : required.history_geometry2)
			{
				double JLagr_hist;
				this->fill_shape_info_at_s(s, 0, required, JLagr_hist, 0, NULL, k);
			}
		}
		double JLagr;
		this->fill_shape_info_at_s(s, 0, required, JLagr, 0);
		set_remaining_shapes_appropriately(shape_info, required);

		xvelo.assign(3, 0.0);
		_currently_assembled_element = this;
		functable->EvalTracerAdvection(&eleminfo, this->shape_info, index, &(xvelo[0]));
	}
}
