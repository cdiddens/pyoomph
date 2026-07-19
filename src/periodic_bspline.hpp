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

  // A periodic (i.e. wrap-around) B-spline basis of order k on a set of knots, used e.g. by
  // PeriodicOrbitHandler (bifurcation.cpp) to discretize a periodic orbit in time.
  // Periodicity is implemented by augmenting the given knots with extra knots copied
  // (shifted by +-L, the period) from the other end of the knot range (see augknots,
  // built in the constructor); this lets the standard (non-periodic) Cox-de Boor B-spline
  // recursion (get_bspline/get_dbspline) be evaluated as usual, with the periodic wrap
  // handled afterwards by folding the few B-splines that straddle the seam back onto their
  // periodic images (see get_shape/get_dshape). The constructor also precomputes, per
  // element, Gauss-Legendre quadrature points/weights and shape (and shape-derivative)
  // values so that get_integration_info can hand out ready-to-use integration data.
  class PeriodicBSplineBasis
  {
  protected:
    static std::vector<std::vector<double>> GL_x; // Gauss-Lengendre quadrature points, indexed by [order-1][point index]
    static std::vector<std::vector<double>> GL_w; // Gauss-Lengendre quadrature weights, indexed by [order-1][point index]
    std::vector<double> knots;                     // knots including the periodic knot at the end
    std::vector<double> augknots;                  // augmented knots (including the periodic knots at the beginning and the end and the shifted knots for even order)
    unsigned zero_offset;                          // index within augknots corresponding to the start of the "real" (non-augmented) knot range
    unsigned int k;                                // order of the B-spline
    int gl_order;                                  // order of the Gauss-Legendre quadrature (-1 means from k)
    double get_bspline(unsigned int i, unsigned int k, double x) const;  // Cox-de Boor recursion: value of the i-th B-spline of order k at x, w.r.t. augknots
    double get_dbspline(unsigned int i, unsigned int k, double x) const; // Derivative of get_bspline w.r.t. x
    //std::vector<double> integral_psi; // integral of the B-splines over the periodic range [indices 0,1,...,N-1]
    std::vector<std::vector<double>> gl_weights;                  // Gauss-Legendre weights (same in each element, must be multiplied by the knot step)
    std::vector<std::vector<unsigned>> shape_indices;             // Shape indices (for each Gauss-Legendre point)
    std::vector<std::vector<std::vector<double>>> shape_values;   // Shape values (for each Gauss-Legendre point)
    std::vector<std::vector<std::vector<double>>> dshape_values;  // Shape values of the first derivative (for each Gauss-Legendre point)
    void sanity_check() const; // Verify (by direct GL integration) that the shape functions partition unity and that their derivatives integrate to zero; throws on failure

  public:
    unsigned get_num_elements() const { return knots.size() - 1; }
    unsigned get_integration_info(unsigned int i, std::vector<double> &w, std::vector<unsigned> &indices, std::vector<std::vector<double>> &psi, std::vector<std::vector<double>> &dpsi) const; // Precomputed GL weights/indices/shape+dshape values for element i

    unsigned get_interpolation_info(double s, std::vector<unsigned> &indices, std::vector<double> &psi) const; // Shape function indices/values at an arbitrary (not necessarily on-grid) point s, wrapped into the periodic range

    const std::vector<double> &get_knots() const { return knots; }
    const std::vector<double> &get_augknots() const { return augknots; }
    //const double get_integral_psi(unsigned int i) const {return integral_psi[i];}
    double integrate_bspline(int index) const; // integrate the B-spline at index i over the periodic range
    PeriodicBSplineBasis(const std::vector<double> &knots, unsigned int order, int gl_order = -1); // knots must be strictly increasing and have at least 4 entries and at least 2*order entries
    double get_shape(unsigned int i, double x) const;                                    // Value of the i-th periodic shape function at x (i in [0, get_num_elements()) )
    std::vector<double> get_shape(unsigned int i, const std::vector<double> &x) const;    // Vectorized version of get_shape
    double get_dshape(unsigned int i, double x) const;                                    // Derivative w.r.t. x of get_shape
    std::vector<double> get_dshape(unsigned int i, const std::vector<double> &x) const;   // Vectorized version of get_dshape
  };

}
