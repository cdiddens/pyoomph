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

The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl

================================================================================*/


/******************
This file strongly based on the file error_estimator.cc from the oomph-lib library, see (thirdparty/oomph-lib/include/error_estimator.cc)
*******************/


#pragma once

#include "error_estimator.h"
#include "mesh.h"
#include "quadtree.h"
#include "nodes.h"
#include "algebraic_elements.h"

namespace pyoomph
{

  //========================================================================
  /// Z2-error-estimator:  Same as in oomph-lib, but taking Lagrangian coordinates instead Eulerian
  //========================================================================
  class LagrZ2ErrorEstimator : public virtual oomph::ErrorEstimator
  {
  public:
    bool use_Lagrangian;

    /// \short Function pointer to combined error estimator function
    typedef double (*CombinedErrorEstimateFctPt)(const oomph::Vector<double> &errors);

    /// Constructor: Set order of recovery shape functions
    LagrZ2ErrorEstimator(const unsigned &recovery_order) : use_Lagrangian(true),
                                                           Recovery_order(recovery_order), Recovery_order_from_first_element(false),
                                                           Reference_flux_norm(0.0), Combined_error_fct_pt(0)
    {
    }

    /// \short Constructor: Leave order of recovery shape functions open
    /// for now -- they will be read out from first element of the mesh
    /// when the error estimator is applied
    LagrZ2ErrorEstimator() : use_Lagrangian(true), Recovery_order(0),
                             Recovery_order_from_first_element(true), Reference_flux_norm(0.0),
                             Combined_error_fct_pt(0)
    {
    }

    /// Broken copy constructor
    LagrZ2ErrorEstimator(const LagrZ2ErrorEstimator &)
    {
      oomph::BrokenCopy::broken_copy("LagrZ2ErrorEstimator");
    }

    /// Broken assignment operator
    void operator=(const LagrZ2ErrorEstimator &)
    {
      oomph::BrokenCopy::broken_assign("LagrZ2ErrorEstimator");
    }

    /// Empty virtual destructor
    virtual ~LagrZ2ErrorEstimator() {}

    /// \short Compute the elemental error measures for a given mesh
    /// and store them in a vector.
    void get_element_errors(oomph::Mesh *&mesh_pt,
                            oomph::Vector<double> &elemental_error)
    {
      // Create dummy doc info object and switch off output
      oomph::DocInfo doc_info;
      doc_info.disable_doc();
      // Forward call to version with doc.
      get_element_errors(mesh_pt, elemental_error, doc_info);
    }

    /// \short Compute the elemental error measures for a given mesh
    /// and store them in a vector.
    /// If doc_info.enable_doc(), doc FE and recovered fluxes in
    /// - flux_fe*.dat
    /// - flux_rec*.dat
    void get_element_errors(oomph::Mesh *&mesh_pt,
                            oomph::Vector<double> &elemental_error,
                            oomph::DocInfo &doc_info);

    /// Access function for order of recovery polynomials
    unsigned &recovery_order() { return Recovery_order; }

    /// Access function for order of recovery polynomials (const version)
    unsigned recovery_order() const { return Recovery_order; }

    /// Access function: Pointer to combined error estimate function
    CombinedErrorEstimateFctPt &combined_error_fct_pt()
    {
      return Combined_error_fct_pt;
    }

    ///\short  Access function: Pointer to combined error estimate function.
    /// Const version
    CombinedErrorEstimateFctPt combined_error_fct_pt() const
    {
      return Combined_error_fct_pt;
    }

    /// \short Setup patches: For each vertex node pointed to by nod_pt,
    /// adjacent_elements_pt[nod_pt] contains the pointer to the vector that
    /// contains the pointers to the elements that the node is part of.
    void setup_patches(
        oomph::Mesh *&mesh_pt,
        std::map<oomph::Node *, oomph::Vector<oomph::ElementWithZ2ErrorEstimator *> *> &
            adjacent_elements_pt,
        oomph::Vector<oomph::Node *> &vertex_node_pt);

    /// Access function for prescribed reference flux norm
    double &reference_flux_norm() { return Reference_flux_norm; }

    /// Access function for prescribed reference flux norm (const. version)
    double reference_flux_norm() const { return Reference_flux_norm; }

    /// Return a combined error estimate from all compound errors
    double get_combined_error_estimate(const oomph::Vector<double> &compound_error);

  private:
    /// \short Given the vector of elements that make up a patch,
    /// the number of recovery and flux terms, and the spatial
    /// dimension of the problem, compute
    /// the matrix of recovered flux coefficients and return
    /// a pointer to it.
    void get_recovered_flux_in_patch(
        const oomph::Vector<oomph::ElementWithZ2ErrorEstimator *> &patch_el_pt,
        const unsigned &num_recovery_terms,
        const unsigned &num_flux_terms,
        const unsigned &dim,
        oomph::DenseMatrix<double> *&recovered_flux_coefficient_pt);

    /// \short Return number of coefficients for expansion of recovered fluxes
    /// for given spatial dimension of elements.
    /// (We use complete polynomials of the specified given order.)
    unsigned nrecovery_terms(const unsigned &dim);

    /// \short Recovery shape functions as functions of the global, Eulerian
    /// coordinate x of dimension dim.
    /// The recovery shape functions are  complete polynomials of
    /// the order specified by Recovery_order.
    void shape_rec(const oomph::Vector<double> &x, const unsigned &dim,
                   oomph::Vector<double> &psi_r);

    /// \short Integation scheme associated with the recovery shape functions
    /// must be of sufficiently high order to integrate the mass matrix
    /// associated with the recovery shape functions
    oomph::Integral *integral_rec(const unsigned &dim, const bool &is_q_mesh);

    /// Order of recovery polynomials
    unsigned Recovery_order;

    /// Bool to indicate if recovery order is to be read out from
    /// first element in mesh or set globally
    bool Recovery_order_from_first_element;

    /// Doc flux and recovered flux
    void doc_flux(oomph::Mesh *mesh_pt,
                  const unsigned &num_flux_terms,
                  oomph::MapMatrixMixed<oomph::Node *, int, double> &rec_flux_map,
                  const oomph::Vector<double> &elemental_error,
                  oomph::DocInfo &doc_info);

    /// Prescribed reference flux norm
    double Reference_flux_norm;

    /// Function pointer to combined error estimator function
    CombinedErrorEstimateFctPt Combined_error_fct_pt;
  };

}
