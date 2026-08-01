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
    // If true, patches/recovery use the Lagrangian (reference) node position xi() instead
    // of the current Eulerian position x(); this makes the estimated error track the
    // (fixed) material configuration rather than a possibly moving/deforming mesh.
    bool use_Lagrangian;

    // Fit the recovery polynomial in a patch-local frame instead of in global coordinates.
    // Needed as soon as the mesh has a co-dimension (an interface: a curve in 2D, a surface in
    // 3D): the recovery basis spans the first dim GLOBAL coordinates, which parametrises such a
    // mesh only if it happens to be a graph over those axes. A boundary at x=const makes the
    // normal matrix exactly singular and the old code threw "Singular Matrix" out of DenseLU.
    // See build_recovery_frame() for what the frame is.
    bool use_local_recovery_frame_in_codim; // codim > 0; on by default, off = the old behaviour
    bool force_local_recovery_frame;        // also for codim 0, where it is only a conditioning win

    // How much of the mesh-global flux norm is divided out of the elemental errors. 1 (default) is
    // fully relative - each element's error is its share of this mesh's total, which is what the
    // estimator has always done. 0 is fully absolute: the raw integrated flux jump, comparable
    // between meshes and between adaptation steps, but on a scale that depends on the element size
    // and the field, so the permitted-error thresholds must be rechosen. In between, the errors are
    // divided by norm^normalize_relative, the geometric blend of the two.
    double normalize_relative;

    // The affine map from global coordinates into the frame the recovery polynomial is fitted in:
    // origin at the patch centroid, axes along the patch's dominant directions, scaled to O(1).
    // A complete polynomial space of order p is invariant under affine maps, so for codim 0 this
    // recovers exactly the same field as fitting in global coordinates - only better conditioned.
    struct RecoveryFrame
    {
      bool active;                                // false => to_local() just copies the first dim entries
      unsigned nodal_dim;                         // length of the global coordinate vectors it consumes
      oomph::Vector<double> origin;               // nodal_dim
      oomph::Vector<oomph::Vector<double>> axes;  // dim x nodal_dim, orthonormal
      double inv_scale;
      RecoveryFrame() : active(false), nodal_dim(0), inv_scale(1.0) {}
      void to_local(const oomph::Vector<double> &x, const unsigned &dim, oomph::Vector<double> &xlocal) const;
    };

    /// \short Function pointer to combined error estimator function
    typedef double (*CombinedErrorEstimateFctPt)(const oomph::Vector<double> &errors);

    /// Constructor: Set order of recovery shape functions
    LagrZ2ErrorEstimator(const unsigned &recovery_order) : use_Lagrangian(true),
                                                           use_local_recovery_frame_in_codim(true), force_local_recovery_frame(false),
                                                           normalize_relative(1.0),
                                                           Recovery_order(recovery_order), Recovery_order_from_first_element(false),
                                                           Reference_flux_norm(0.0), Combined_error_fct_pt(0)
    {
    }

    /// \short Constructor: Leave order of recovery shape functions open
    /// for now -- they will be read out from first element of the mesh
    /// when the error estimator is applied
    LagrZ2ErrorEstimator() : use_Lagrangian(true),
                             use_local_recovery_frame_in_codim(true), force_local_recovery_frame(false),
                             normalize_relative(1.0),
                             Recovery_order(0),
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
    ~LagrZ2ErrorEstimator() override {}

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
                            oomph::DocInfo &doc_info) override;

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

    /// \short Is the recovery for this mesh to be done in a patch-local frame? Decided from the
    /// mesh's co-dimension and the two flags above. Public because the driver has to agree with
    /// the per-patch code about the answer.
    bool recovery_frame_wanted(const unsigned &nodal_dim, const unsigned &dim) const;

    /// \short Build the local frame of a patch (see RecoveryFrame). Deterministic given the same
    /// patch in the same order, which is what lets a non-distributed parallel job rebuild a
    /// remote patch's frame rather than broadcast it.
    void build_recovery_frame(
        const oomph::Vector<oomph::ElementWithZ2ErrorEstimator *> &patch_el_pt,
        const unsigned &dim, RecoveryFrame &frame);

  private:
    /// \short Given the vector of elements that make up a patch,
    /// the number of recovery and flux terms, and the spatial
    /// dimension of the problem, compute
    /// the matrix of recovered flux coefficients and return
    /// a pointer to it. Also returns the frame the coefficients are expressed in, which the
    /// caller must keep: coefficients from different patches are only comparable within one frame.
    void get_recovered_flux_in_patch(
        const oomph::Vector<oomph::ElementWithZ2ErrorEstimator *> &patch_el_pt,
        const unsigned &num_recovery_terms,
        const unsigned &num_flux_terms,
        const unsigned &dim,
        oomph::DenseMatrix<double> *&recovered_flux_coefficient_pt,
        RecoveryFrame &frame);

    /// \short Return number of coefficients for expansion of recovered fluxes
    /// for given spatial dimension of elements.
    /// (We use complete polynomials of the specified given order.)
    unsigned nrecovery_terms(const unsigned &dim);

    /// \short Recovery shape functions as functions of the coordinate x of dimension dim - the
    /// global (Eulerian or Lagrangian) coordinate, or the patch-local one when a RecoveryFrame is
    /// in use. The recovery shape functions are complete polynomials of the order specified by
    /// Recovery_order.
    void shape_rec(const oomph::Vector<double> &x, const unsigned &dim,
                   oomph::Vector<double> &psi_r);

    /// \short Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations, returned
    /// with the eigenvalues in descending order. Used for both the patch frame (nodal_dim x
    /// nodal_dim) and the recovery system (num_recovery_terms squared); both are tiny.
    static void symmetric_eigen(const oomph::DenseMatrix<double> &A, const unsigned &n,
                                oomph::Vector<double> &evals, oomph::DenseMatrix<double> &evecs);

    /// \short Solve the (symmetric, positive semi-definite) recovery system by truncated
    /// pseudo-inverse, so that a rank-deficient patch degrades to the best lower-order fit
    /// instead of throwing "Singular Matrix" out of DenseLU.
    void solve_recovery_system(const oomph::DenseMatrix<double> &recovery_mat, const unsigned &n,
                               oomph::Vector<oomph::Vector<double>> &rhs);

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
