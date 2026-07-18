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


#include "periodic_bspline.hpp"
#include "exception.hpp"
#include <iostream>
namespace pyoomph
{


/*
Feel free to add more GL quadrature points and weights

import numpy
from numpy.polynomial.legendre import leggauss


xdef="std::vector<std::vector<double>> PeriodicBSplineBasis::GL_x={"
wdef="std::vector<std::vector<double>> PeriodicBSplineBasis::GL_w={"
for deg in range(1,20):
		x,w=leggauss(deg)
		xdef+="{"+",".join([f"{xi:.16f}" for xi in x])+"},"
		wdef+="{"+",".join([f"{wi:.16f}" for wi in w])+"},"

print(xdef[:-1]+"};")
print(wdef[:-1]+"};")

*/
std::vector<std::vector<double>> PeriodicBSplineBasis::GL_x={{0.0000000000000000},{-0.5773502691896257,0.5773502691896257},{-0.7745966692414834,0.0000000000000000,0.7745966692414834},{-0.8611363115940526,-0.3399810435848563,0.3399810435848563,0.8611363115940526},{-0.9061798459386640,-0.5384693101056831,0.0000000000000000,0.5384693101056831,0.9061798459386640},{-0.9324695142031521,-0.6612093864662645,-0.2386191860831969,0.2386191860831969,0.6612093864662645,0.9324695142031521},{-0.9491079123427585,-0.7415311855993945,-0.4058451513773972,0.0000000000000000,0.4058451513773972,0.7415311855993945,0.9491079123427585},{-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956498,0.5255324099163290,0.7966664774136267,0.9602898564975362},{-0.9681602395076261,-0.8360311073266358,-0.6133714327005904,-0.3242534234038089,0.0000000000000000,0.3242534234038089,0.6133714327005904,0.8360311073266358,0.9681602395076261},{-0.9739065285171717,-0.8650633666889845,-0.6794095682990244,-0.4333953941292472,-0.1488743389816312,0.1488743389816312,0.4333953941292472,0.6794095682990244,0.8650633666889845,0.9739065285171717},{-0.9782286581460570,-0.8870625997680953,-0.7301520055740494,-0.5190961292068118,-0.2695431559523450,0.0000000000000000,0.2695431559523450,0.5190961292068118,0.7301520055740494,0.8870625997680953,0.9782286581460570},{-0.9815606342467192,-0.9041172563704748,-0.7699026741943047,-0.5873179542866175,-0.3678314989981802,-0.1252334085114689,0.1252334085114689,0.3678314989981802,0.5873179542866175,0.7699026741943047,0.9041172563704748,0.9815606342467192},{-0.9841830547185881,-0.9175983992229779,-0.8015780907333099,-0.6423493394403402,-0.4484927510364468,-0.2304583159551348,0.0000000000000000,0.2304583159551348,0.4484927510364468,0.6423493394403402,0.8015780907333099,0.9175983992229779,0.9841830547185881},{-0.9862838086968123,-0.9284348836635735,-0.8272013150697650,-0.6872929048116855,-0.5152486363581541,-0.3191123689278897,-0.1080549487073437,0.1080549487073437,0.3191123689278897,0.5152486363581541,0.6872929048116855,0.8272013150697650,0.9284348836635735,0.9862838086968123},{-0.9879925180204854,-0.9372733924007060,-0.8482065834104272,-0.7244177313601701,-0.5709721726085388,-0.3941513470775634,-0.2011940939974345,0.0000000000000000,0.2011940939974345,0.3941513470775634,0.5709721726085388,0.7244177313601701,0.8482065834104272,0.9372733924007060,0.9879925180204854},{-0.9894009349916499,-0.9445750230732326,-0.8656312023878318,-0.7554044083550030,-0.6178762444026438,-0.4580167776572274,-0.2816035507792589,-0.0950125098376375,0.0950125098376375,0.2816035507792589,0.4580167776572274,0.6178762444026438,0.7554044083550030,0.8656312023878318,0.9445750230732326,0.9894009349916499},{-0.9905754753144174,-0.9506755217687678,-0.8802391537269859,-0.7815140038968014,-0.6576711592166908,-0.5126905370864769,-0.3512317634538763,-0.1784841814958479,0.0000000000000000,0.1784841814958479,0.3512317634538763,0.5126905370864769,0.6576711592166908,0.7815140038968014,0.8802391537269859,0.9506755217687678,0.9905754753144174},{-0.9915651684209309,-0.9558239495713978,-0.8926024664975557,-0.8037049589725231,-0.6916870430603532,-0.5597708310739475,-0.4117511614628426,-0.2518862256915055,-0.0847750130417353,0.0847750130417353,0.2518862256915055,0.4117511614628426,0.5597708310739475,0.6916870430603532,0.8037049589725231,0.8926024664975557,0.9558239495713978,0.9915651684209309},{-0.9924068438435844,-0.9602081521348300,-0.9031559036148179,-0.8227146565371428,-0.7209661773352294,-0.6005453046616810,-0.4645707413759609,-0.3165640999636298,-0.1603586456402254,0.0000000000000000,0.1603586456402254,0.3165640999636298,0.4645707413759609,0.6005453046616810,0.7209661773352294,0.8227146565371428,0.9031559036148179,0.9602081521348300,0.9924068438435844}};
std::vector<std::vector<double>> PeriodicBSplineBasis::GL_w={{2.0000000000000000},{1.0000000000000000,1.0000000000000000},{0.5555555555555557,0.8888888888888888,0.5555555555555557},{0.3478548451374537,0.6521451548625462,0.6521451548625462,0.3478548451374537},{0.2369268850561894,0.4786286704993662,0.5688888888888890,0.4786286704993662,0.2369268850561894},{0.1713244923791697,0.3607615730481389,0.4679139345726914,0.4679139345726914,0.3607615730481389,0.1713244923791697},{0.1294849661688706,0.2797053914892766,0.3818300505051183,0.4179591836734690,0.3818300505051183,0.2797053914892766,0.1294849661688706},{0.1012285362903767,0.2223810344533743,0.3137066458778870,0.3626837833783618,0.3626837833783618,0.3137066458778870,0.2223810344533743,0.1012285362903767},{0.0812743883615747,0.1806481606948571,0.2606106964029357,0.3123470770400028,0.3302393550012597,0.3123470770400028,0.2606106964029357,0.1806481606948571,0.0812743883615747},{0.0666713443086881,0.1494513491505804,0.2190863625159820,0.2692667193099965,0.2955242247147530,0.2955242247147530,0.2692667193099965,0.2190863625159820,0.1494513491505804,0.0666713443086881},{0.0556685671161732,0.1255803694649047,0.1862902109277344,0.2331937645919907,0.2628045445102468,0.2729250867779009,0.2628045445102468,0.2331937645919907,0.1862902109277344,0.1255803694649047,0.0556685671161732},{0.0471753363865120,0.1069393259953189,0.1600783285433461,0.2031674267230656,0.2334925365383546,0.2491470458134027,0.2491470458134027,0.2334925365383546,0.2031674267230656,0.1600783285433461,0.1069393259953189,0.0471753363865120},{0.0404840047653159,0.0921214998377286,0.1388735102197874,0.1781459807619455,0.2078160475368886,0.2262831802628971,0.2325515532308739,0.2262831802628971,0.2078160475368886,0.1781459807619455,0.1388735102197874,0.0921214998377286,0.0404840047653159},{0.0351194603317524,0.0801580871597603,0.1215185706879030,0.1572031671581934,0.1855383974779376,0.2051984637212955,0.2152638534631577,0.2152638534631577,0.2051984637212955,0.1855383974779376,0.1572031671581934,0.1215185706879030,0.0801580871597603,0.0351194603317524},{0.0307532419961186,0.0703660474881081,0.1071592204671718,0.1395706779261539,0.1662692058169938,0.1861610000155619,0.1984314853271112,0.2025782419255609,0.1984314853271112,0.1861610000155619,0.1662692058169938,0.1395706779261539,0.1071592204671718,0.0703660474881081,0.0307532419961186},{0.0271524594117540,0.0622535239386477,0.0951585116824926,0.1246289712555340,0.1495959888165768,0.1691565193950026,0.1826034150449236,0.1894506104550686,0.1894506104550686,0.1826034150449236,0.1691565193950026,0.1495959888165768,0.1246289712555340,0.0951585116824926,0.0622535239386477,0.0271524594117540},{0.0241483028685495,0.0554595293739866,0.0850361483171791,0.1118838471934036,0.1351363684685252,0.1540457610768101,0.1680041021564500,0.1765627053669925,0.1794464703562065,0.1765627053669925,0.1680041021564500,0.1540457610768101,0.1351363684685252,0.1118838471934036,0.0850361483171791,0.0554595293739866,0.0241483028685495},{0.0216160135264841,0.0497145488949692,0.0764257302548892,0.1009420441062870,0.1225552067114784,0.1406429146706506,0.1546846751262652,0.1642764837458327,0.1691423829631436,0.1691423829631436,0.1642764837458327,0.1546846751262652,0.1406429146706506,0.1225552067114784,0.1009420441062870,0.0764257302548892,0.0497145488949692,0.0216160135264841},{0.0194617882297276,0.0448142267656998,0.0690445427376411,0.0914900216224498,0.1115666455473338,0.1287539625393362,0.1426067021736064,0.1527660420658594,0.1589688433939541,0.1610544498487834,0.1589688433939541,0.1527660420658594,0.1426067021736064,0.1287539625393362,0.1115666455473338,0.0914900216224498,0.0690445427376411,0.0448142267656998,0.0194617882297276}};


// Build the periodic B-spline basis on `knots` (a strictly increasing sequence spanning
// one period, with knots.back()-knots.front() the period length L). Constructs augknots
// by mirroring/shifting k+3 knots from each end by +-L so that ordinary (non-periodic)
// B-spline evaluation on augknots automatically reproduces periodic wrap-around near the
// seam; for even order the knots are additionally shifted to their midpoints (needed so
// that the B-spline peak lies on an element rather than a knot). Finally precomputes,
// per element, the Gauss-Legendre points/weights and the corresponding shape and
// shape-derivative values so that get_integration_info can return them directly, and
// runs a sanity check on the result.
PeriodicBSplineBasis::PeriodicBSplineBasis(const std::vector<double> &knots, unsigned int order,int gl_order)
{
		if (order<1) throw_runtime_error("Order of the B-spline must be at least 1");
		if (gl_order<0) gl_order=order;
		this->gl_order=gl_order-1; // 0-based index (TODO: Do we need to add a 1 here?)
		this->knots = knots;
		this->k = order;
		double x0=knots[0];
		double L=knots[knots.size()-1]-x0;

		if (knots.size() < 4)
		{
				throw_runtime_error("Not enough knots. Need at least 4");
		}    
		if (knots.size() < 2*k)
		{
				throw_runtime_error("Not enough knots for the order of the B-spline");
		}    

		zero_offset=(k-1)/2+3;
		for (int i = 0; i < (int)k+3; i++)
		{
				this->augknots.insert(this->augknots.begin(),knots[knots.size()-i-2]-L+x0);
		}
		for (size_t i = 0; i < knots.size(); i++)
		{
				this->augknots.push_back(knots[i]);
		}
		for (int i = 0; i < (int)k+3; i++)
		{
				this->augknots.push_back(L+knots[i+1]-x0);
		}


		// Shift the knots to the central points for even order
		if (k%2==0)
		{        
			std::vector<double> newknots(augknots.size()-1);
			for (size_t i = 0; i < augknots.size()-1; i++)
			{
						newknots[i] = 0.5*(augknots[i]+augknots[i+1]);
			}
			this->augknots=newknots;
		}

		/*for (int i = 0; i < this->augknots.size(); i++)
		{
				std::cout << this->augknots[i] << std::endl;
		}*/

		// Calculate the integrals of the B-splines
	/*  integral_psi.resize(knots.size()-1);
		for (unsigned int i = 0; i < knots.size()-1; i++)
		{
				integral_psi[i]=integrate_bspline(i);
				if (i<k)
				{
						integral_psi[i] += integrate_bspline(i+knots.size()-1);
				}
		}*/
		shape_indices.resize(knots.size()-1);
		shape_values.resize(knots.size()-1);
		dshape_values.resize(knots.size()-1);
		gl_weights.resize(knots.size()-1);
		for (unsigned i=0;i<knots.size()-1;i++) // Element loop
		{
				shape_indices[i].resize(k+1);
				for (unsigned j=0;j<shape_indices[i].size();j++) // supported shape loop
				{
						//k=1: +k/2-1   -1
						//k=2: +k/2-3   -2
						//k=3: +k/2-4   -3
						//k=4
						shape_indices[i][j]=(i+j+knots.size()-k)%(knots.size()-1);
						//std::cout << "Shape indices " << i << " " << j << " : " << shape_indices[i][j] << std::endl;
				}
				int start=i+zero_offset+1;
				//int end=i+zero_offset+k+1;
				shape_values[i].resize(GL_x[gl_order].size());
				dshape_values[i].resize(GL_x[gl_order].size());
				gl_weights[i].resize(GL_x[gl_order].size());
				//std::cout << "INTEGRATING OVER " << augknots[start] << " to " << augknots[start+1] << std::endl;
				for (size_t j = 0; j < GL_x[gl_order].size(); j++)
				{
						double x=0.5*(augknots[start]+augknots[start+1])+0.5*(augknots[start+1]-augknots[start])*GL_x[gl_order][j];
						gl_weights[i][j]=GL_w[gl_order][j]*0.5*(augknots[start+1]-augknots[start]);
						shape_values[i][j].resize(k+1);
						dshape_values[i][j].resize(k+1);
						for (unsigned int l = 0; l < k+1; l++)
						{
								shape_values[i][j][l]=get_shape(shape_indices[i][l], x);
								dshape_values[i][j][l]=get_dshape(shape_indices[i][l], x);
								//if (shape_indices[i][l]==0)                std::cout << "SHAPE " << i << " " << j << " " << l << " : " << shape_values[i][j][l] << " at x= " << x <<std::endl;
						}

				}

		}

		//TODO: Can be removed
		sanity_check();
}


// Standard (Cox-de Boor) recursive B-spline evaluation w.r.t. the augmented knot vector
// augknots: order-0 is the indicator function of [augknots[i], augknots[i+1]), and higher
// orders are a convex combination of two order-(k-1) splines. Note the parameter named `k`
// here is the recursion order and shadows the member k (the basis's fixed order).
double PeriodicBSplineBasis::get_bspline(unsigned int i, unsigned int k, double x) const
{
		if (k==0)
		{
				if (augknots[i] <= x && x < augknots[i+1])
				{
						return 1.0;
				}
				else
				{
						return 0.0;
				}
		}
		else
		{
				//std::cout << "i: " << i << " k: " << k << " x: " << x << std::endl;
				double a = (x - augknots[i])/(augknots[i+k] - augknots[i]);
				double b = (augknots[i+k+1] - x)/(augknots[i+k+1] - augknots[i+1]);
				return a*get_bspline(i, k-1, x) + b*get_bspline(i+1, k-1, x);
		}
}

// Analytic derivative of get_bspline w.r.t. x, via the standard B-spline derivative identity.
double PeriodicBSplineBasis::get_dbspline(unsigned int i, unsigned int k, double x) const
{
		return k/(augknots[i+k] - augknots[i])*get_bspline(i, k-1, x) - k/(augknots[i+k+1] - augknots[i+1])*get_bspline(i+1, k-1, x);
}

/*double PeriodicBSplineBasis::integrate_bspline(int index) const
{


		int start=index+zero_offset;
		//int end=std::min((int)knots.size()-1, (int)index+(int)(k+1)/2)+zero_offset+2;
		int end=index+zero_offset+k+1;




		double gl_sum=0;
		unsigned gl_index=k-1; // TODO: Could be less or more?

		std::cout << "start: " << start << " end: " << end << std::endl;
		std::cout << "from " << this->augknots[start] << " to " << this->augknots[end]  << std::endl;
		for (int i = start; i < end; i++)
		{
				for (int j = 0; j < GL_x[gl_index].size(); j++)
				{
						double x=0.5*(augknots[i]+augknots[i+1])+0.5*(augknots[i+1]-augknots[i])*GL_x[gl_index][j];
						gl_sum += GL_w[gl_index][j]*get_shape(index, x)*0.5*(augknots[i+1]-augknots[i]);
				}
		}



		std::vector<double> xsamples;
		unsigned Nsamples=1000;
		for (int i = 0; i < Nsamples; i++)
		{
				xsamples.push_back(knots[0]+i*(knots[knots.size()-1]-knots[0])/(Nsamples-1));
		}
		double sum=0;
		for (int i = 0; i < Nsamples-1; i++)
		{
				sum += 0.5*(xsamples[i+1]-xsamples[i])*(get_shape(index, xsamples[i])+get_shape(index, xsamples[i+1]));
		}

		std::cout << "COMPARIONS " << gl_sum << " " << sum << std::endl;

		return gl_sum;

}
*/

// Numerically verify two properties that must hold for a correctly-built periodic B-spline
// partition: the shape functions integrate (summed, over all elements/GL points) to the
// total period length L (partition of unity), and their derivatives integrate to zero.
// Throws if either check is off by more than 1e-8. Called once at the end of construction.
void PeriodicBSplineBasis::sanity_check() const
{
		double integral_sum=0;
		double dintegral_sum=0;
		unsigned nelem=this->get_num_elements();

		std::vector<double> w;
		std::vector<unsigned> indices;
		std::vector<std::vector<double>> psi;
		std::vector<std::vector<double>> dpsi;
		for (unsigned int ie=0;ie<nelem;ie++)
		{
				unsigned nGL=this->get_integration_info(ie,w,indices,psi,dpsi);        
				for (unsigned int igl=0;igl<nGL;igl++)
				{
						for (unsigned int i=0;i<indices.size();i++)
						{
								integral_sum+=w[igl]*psi[igl][i];
								dintegral_sum+=w[igl]*dpsi[igl][i];
						}         
				}
		}
		double L=knots.back()-knots.front();
		if (std::abs(integral_sum-L)>1e-8) throw_runtime_error("Sanity check failed for the B-spline basis: N="+std::to_string(knots.size())+", k="+std::to_string(k)+", L="+std::to_string(L)+" != integral="+std::to_string(integral_sum));
		if (std::abs(dintegral_sum)>1e-8) throw_runtime_error("Sanity check failed for the B-spline basis: N="+std::to_string(knots.size())+", k="+std::to_string(k)+", dintegral="+std::to_string(dintegral_sum));
		
}

// Look up the precomputed Gauss-Legendre integration data for element i (0-based, in
// [0,get_num_elements())): weights, the indices of the (k+1) shape functions supported on
// this element, and their values/derivative-values at each GL point. Returns the number of
// GL points. This just returns references into arrays filled once in the constructor.
unsigned PeriodicBSplineBasis::get_integration_info(unsigned int i, std::vector<double> &w, std::vector<unsigned> &indices, std::vector<std::vector<double>> &psi, std::vector<std::vector<double>> &dpsi) const
{
				// Returns the weights, indices and shape functions for the i-th element
				// w : Gauss Legendre weights
				// indices : indices of the knots to consider
				// psi : shape functions for each Gauss-Legendre point [GL point, shape index]
				w=gl_weights[i];
				indices=shape_indices[i];
				psi=shape_values[i];
				dpsi=dshape_values[i];
				return w.size();
}

// Given an arbitrary point s (not necessarily aligned to an element boundary), first wrap
// it into the primary knot range [knots.front(), knots.back()) by adding/subtracting the
// period L, then locate the element it falls into and return the (k+1) shape function
// indices/values supported there. Used for interpolating the periodic orbit at points that
// don't coincide with the discretization's own knots.
unsigned PeriodicBSplineBasis::get_interpolation_info(double s, std::vector<unsigned> &indices, std::vector<double> &psi) const
{
				// Find all the shape functions for the interpolation points
				indices.resize(k+1);
				psi.resize(k+1);
				int start=zero_offset+1;
				double L=knots.back()-knots.front();
				// TODO: Can be improved, I guess
				while (s<augknots[start]) s+=L;
				while (s>=augknots[start+knots.size()]) s-=L;
				for (unsigned int i=0;i<knots.size();i++)
				{
						if (s>=augknots[start] && s<augknots[start+1]) break;
						start++;            
				}

				for (unsigned int j=0;j<indices.size();j++)
				{

						indices[j]=(start+j+knots.size()-k-zero_offset-1)%(knots.size()-1);
						psi[j]=this->get_shape(indices[j],s);            
				}        
				return indices.size();
}

// Value of the i-th periodic shape function at x. This is the corresponding B-spline on
// augknots (shifted by zero_offset), plus - for the few indices near either seam of the
// period (i<k or i>=knots.size()-k) - the contribution of its periodic image on the other
// side, which the non-periodic Cox-de Boor recursion alone would not otherwise include.
double PeriodicBSplineBasis::get_shape(unsigned int i,double x) const
{
		double res=this->get_bspline(i+zero_offset, k, x);

		if (i<k)
		{
				res+=this->get_bspline(knots.size()+zero_offset+i-1, k, x);
		}
		if (i>=knots.size()-k)
		{
			//   std::cout << "INFO " << i << " " << k << " " << knots.size() << std::endl;
			//  std::cout << "Getting at " << i+zero_offset+1-(knots.size()-1) << std::endl;
				res+=this->get_bspline(i+zero_offset-(knots.size()-1), k, x);
		}
		return res;
}
// Derivative w.r.t. x of get_shape; mirrors its periodic-wrap logic exactly.
double PeriodicBSplineBasis::get_dshape(unsigned int i, double x) const
{
		double res= this->get_dbspline(i+zero_offset, k, x);
		if (i<k)
		{
				res+=this->get_dbspline(knots.size()+zero_offset+i-1, k, x);
		}
		if (i>=knots.size()-k)
		{
				res+=this->get_dbspline(i+zero_offset-(knots.size()-1), k, x);
		}
		return res;
}
// Vectorized convenience overload of get_shape, evaluating at each point in x.
std::vector<double> PeriodicBSplineBasis::get_shape(unsigned int i, const std::vector<double> &x) const
{
		std::vector<double> res(x.size());
		for (size_t j = 0; j < x.size(); j++)
		{        
				res[j] = get_shape(i, x[j]);
		}
		return res;
}



// Vectorized convenience overload of get_dshape, evaluating at each point in x.
std::vector<double> PeriodicBSplineBasis::get_dshape(unsigned int i, const std::vector<double> &x) const
{
		std::vector<double> res(x.size());
		for (size_t j = 0; j < x.size(); j++)
		{
				res[j] = get_dshape(i, x[j]);
		}
		return res;
}

}