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



#pragma once

#include <vector>
namespace pyoomph
{

class PeriodicBSplineBasis
{
 protected:
   static std::vector<std::vector<double>> GL_x; // Gauss-Lengendre quadrature points
   static std::vector<std::vector<double>> GL_w; // Gauss-Lengendre quadrature weights
   std::vector<double> knots; // knots including the periodic knot at the end
   std::vector<double> augknots; // augmented knots (including the periodic knots at the beginning and the end and the shifted knots for even order)
   unsigned zero_offset;
   unsigned int k; // order of the B-spline
   int gl_order; // order of the Gauss-Legendre quadrature (-1 means from k)
   double get_bspline(unsigned int i, unsigned int k, double x) const;
   double get_dbspline(unsigned int i, unsigned int k, double x) const;   
   //std::vector<double> integral_psi; // integral of the B-splines over the periodic range [indices 0,1,...,N-1]
   std::vector<std::vector<double>> gl_weights; // Gauss-Legendre weights (same in each element, must be multiplied by the knot step)
   std::vector<std::vector<unsigned>> shape_indices; // Shape indices (for each Gauss-Legendre point)
   std::vector<std::vector<std::vector<double>>> shape_values; // Shape values (for each Gauss-Legendre point)
   std::vector<std::vector<std::vector<double>>> dshape_values; // Shape values of the first derivative (for each Gauss-Legendre point)
   void sanity_check() const;
 public:
    unsigned get_num_elements() const {return knots.size()-1;}
    unsigned get_integration_info(unsigned int i,  std::vector<double> & w,  std::vector<unsigned> & indices,  std::vector<std::vector<double>> & psi, std::vector<std::vector<double>> & dpsi) const;
    
    unsigned get_interpolation_info(double s,std::vector<unsigned> & indices,  std::vector<double> & psi) const;
   
    const std::vector<double>& get_knots() const {return knots;}
    const std::vector<double>& get_augknots() const {return augknots;}
    //const double get_integral_psi(unsigned int i) const {return integral_psi[i];}
    double integrate_bspline(int index) const; // integrate the B-spline at index i over the periodic range 
    PeriodicBSplineBasis(const std::vector<double>& knots, unsigned int order,int gl_order=-1);    
    double get_shape(unsigned int i,double x) const;
    std::vector<double> get_shape(unsigned int i,const std::vector<double> & x) const;
    double get_dshape(unsigned int i,double x) const;
    std::vector<double> get_dshape(unsigned int i,const std::vector<double> & x) const;
};

}
