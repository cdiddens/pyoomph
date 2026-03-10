/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha

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

/*
##################################
This file is strongly based  on the oomph-lib library (see thirdparty/oomph-lib/include/assembly_handler.h)
##################################
*/

// OOOMPH-LIB includes
#include "bifurcation.hpp"
#include "problem.hpp"
#include "elements.h"
#include "problem.h"
#include "mesh.h"

#include "elements.hpp"

#include "periodic_bspline.hpp"

using namespace oomph;

//#define PYOOMPH_BIFURCATION_HANDLER_DEBUG

namespace oomph
{
  template<>
  oomph::GaussLegendre<1, 1>::GaussLegendre()
  {
    // Temporary storage for the integration points    
    for (unsigned i = 0; i < 1; i++)
    {
      Knot[i][0] = 0.0;
      Weight[i] = 2.0;
    }
  }

  class POCollocationFakeIntegral: public oomph::Integral
  {
    public:
      unsigned nweight() const override {return 1;}          
      double knot(const unsigned& i, const unsigned& j) const override {return -1;}
      double weight(const unsigned& i) const override { return 1;}
  };
}

namespace pyoomph
{


   void rotate_complex_eigenvector_nicely(oomph::Vector<double> &real_eigen, oomph::Vector<double> &imag_eigen)
  {
    // Get the dots of the real and imaginary parts
    double GrGr = 0.0, GiGi = 0.0, GrGi = 0.0;
    for (unsigned n = 0; n < real_eigen.size(); n++)
    {
      GrGr += real_eigen[n] * real_eigen[n];
      GiGi += imag_eigen[n] * imag_eigen[n];
      GrGi += real_eigen[n] * imag_eigen[n];
    }

    // Sample phi: We search for a phi that gives a rotated eigenvectors by multiplication with exp(-i*phi)
    // so that <Re(eigenvector),Im(eigenvector)> is zero and <Re(eigenvector),Re(eigenvector)> is maximized
    const unsigned n_phi_samples=30; // Test so many initial guesses for phi
    const unsigned n_inter=15;
    double best_phi=0.0;
    double best_GrGr=GrGr;
    for (unsigned iphi=0;iphi<n_phi_samples;iphi++)
    {
      double phi=2.0*MathematicalConstants::Pi*double(iphi)/double(n_phi_samples);      
      // If res==0, <Re(eigenvector),Im(eigenvector)> will be rotated to zero
      double res=-GiGi*sin(phi)*cos(phi) - GrGi*sin(phi)*sin(phi) + GrGi*cos(phi)*cos(phi) + GrGr*sin(phi)*cos(phi);
      unsigned iter=0;
      bool success=false;
      for  ( unsigned  iter=0; iter<n_inter;iter++)
      {
        // Newton iteration to find phi
        double J=GiGi*sin(phi)*sin(phi) - GiGi*cos(phi)*cos(phi) - 4*GrGi*sin(phi)*cos(phi) - GrGr*sin(phi)*sin(phi) + GrGr*cos(phi)*cos(phi);
        if (std::fabs(J)<1.0e-10)
        {         
          break; // Singular Jacobian
        }
        phi-=res/J;
        res=-GiGi*sin(phi)*cos(phi) - GrGi*sin(phi)*sin(phi) + GrGi*cos(phi)*cos(phi) + GrGr*sin(phi)*cos(phi);
        if (std::fabs(res)<1.0e-10)
        {
          success=true; // Found a good zero
          break;
        }
      }
      if (!success) continue;      
      // Test whether it maximizes <Re(eigenvector),Re(eigenvector)>
      double GrGr_new=GrGr*cos(phi)*cos(phi) + GiGi*sin(phi)*sin(phi) - 2*GrGi*sin(phi)*cos(phi);
      if (GrGr_new>best_GrGr)
      {
        best_GrGr=GrGr_new;
        best_phi=phi;
      }
    }

    // Rotate the eigenvector
    if (best_phi!=0.0)
    {      
      double c=cos(best_phi);
      double s=sin(best_phi);
      GrGr = 0.0, GiGi = 0.0, GrGi = 0.0;    
      for (unsigned n = 0; n < real_eigen.size(); n++)
      {
        double new_real=real_eigen[n]*c-imag_eigen[n]*s;
        double new_imag=real_eigen[n]*s+imag_eigen[n]*c;
        real_eigen[n]=new_real;
        imag_eigen[n]=new_imag;
        GrGr += real_eigen[n] * real_eigen[n];
        GiGi += imag_eigen[n] * imag_eigen[n];
        GrGi += real_eigen[n] * imag_eigen[n];
      }
      std::cout << "Rotating eigenvector by " << best_phi << " to get <Re(eigenvector),Im(eigenvector)> = " << GrGi <<" ~ 0, maximize <Re(eigenvector),Re(eigenvector)>=" << GrGr << " and <Im(eigenvector),Im(eigenvector)>= " << GiGi << std::endl;
    }
    else
    {
      std::cout << "Rotating eigenvector by " << best_phi << " to get <Re(eigenvector),Im(eigenvector)> = " << GrGi <<" ~ 0, maximize <Re(eigenvector),Re(eigenvector)>=" << GrGr << " and <Im(eigenvector),Im(eigenvector)>= " << GiGi << std::endl;
    }

    // Normalize the eigenvector to its real part
    double length_eigen_real = 0.0;
    for (unsigned n = 0; n < real_eigen.size(); n++)
    {
      length_eigen_real += real_eigen[n] * real_eigen[n];
    }
    length_eigen_real = sqrt(length_eigen_real);
    for (unsigned n = 0; n < real_eigen.size(); n++)
    {
      real_eigen[n] /= length_eigen_real;
      imag_eigen[n] /= length_eigen_real;
    }

  }


  MyHopfHandler::MyHopfHandler(Problem *const &problem_pt,
                               double *const &parameter_pt) : Solve_which_system(0), Parameter_pt(parameter_pt), Omega(0.0)
  {

    call_param_change_handler = false;
    eigenweight = 1.0;
    // Set the problem pointer
    Problem_pt = problem_pt;
    // Set the number of non-augmented degrees of freedom
    Ndof = problem_pt->ndof();

    // create the linear algebra distribution for this solver
    // currently only global (non-distributed) distributions are allowed
    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);

    // Resize the vectors of additional dofs
    Phi.resize(Ndof);
    Psi.resize(Ndof);
    C.resize(Ndof);
    Count.resize(Ndof, 0);

    // Loop over all the elements in the problem
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      // Loop over the local freedoms in an element
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        // Increase the associated global equation number counter
        ++Count[elem_pt->eqn_number(n)];
      }
    }

    // Calculate the value Phi by
    // solving the system JPhi = dF/dlambda

    // Locally cache the linear solver
    LinearSolver *const linear_solver_pt = problem_pt->linear_solver_pt();

    // Save the status before entry to this routine
    bool enable_resolve = linear_solver_pt->is_resolve_enabled();

    // We need to do a resolve
    linear_solver_pt->enable_resolve();

    // Storage for the solution
    DoubleVector x(dist_pt, 0.0);

    // Solve the standard problem, we only want to make sure that
    // we factorise the matrix, if it has not been factorised. We shall
    // ignore the return value of x.
    linear_solver_pt->solve(problem_pt, x);

    // Get the vector dresiduals/dparameter
    problem_pt->get_derivative_wrt_global_parameter(parameter_pt, x);

    // Copy rhs vector into local storage so it doesn't get overwritten
    // if the linear solver decides to initialise the solution vector, say,
    // which it's quite entitled to do!
    DoubleVector input_x(x);

    // Now resolve the system with the new RHS and overwrite the solution
    linear_solver_pt->resolve(input_x, x);

    // Restore the storage status of the linear solver
    if (enable_resolve)
    {
      linear_solver_pt->enable_resolve();
    }
    else
    {
      linear_solver_pt->disable_resolve();
    }

    // Normalise the solution x
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += x[n] * x[n];
    }
    length = sqrt(length);

    // Now add the real part of the null space components to the problem
    // unknowns and initialise it all
    // This is dumb at the moment ... fix with eigensolver?
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Phi[n]);
      C[n] = Phi[n] = -x[n] / length;
    }

    // Set the imaginary part so that the appropriate residual is
    // zero initially (eigensolvers)
    for (unsigned n = 0; n < Ndof; n += 2)
    {
      // Make sure that we are not at the end of an array of odd length
      if (n != Ndof - 1)
      {
        Psi[n] = C[n + 1];
        Psi[n + 1] = -C[n];
      }
      // If it's odd set the final entry to zero
      else
      {
        Psi[n] = 0.0;
      }
    }

    // Next add the imaginary parts of the null space components to the problem
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Psi[n]);
    }
    // Now add the parameter
    problem_pt->GetDofPtr().push_back(parameter_pt);
    // Finally add the frequency
    problem_pt->GetDofPtr().push_back(&Omega);

    // rebuild the dof dist
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                              Ndof * 3 + 2, false);
    
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);

    // delete the dist_pt
    delete dist_pt;
  }

  //====================================================================
  /// Constructor: Initialise the hopf handler,
  /// by setting initial guesses for Phi, Psi, Omega  and calculating Count.
  /// If the system changes, a new  handler must be constructed.
  //===================================================================
  MyHopfHandler::MyHopfHandler(Problem *const &problem_pt,
                               double *const &parameter_pt,
                               const double &omega,
                               const DoubleVector &phi,
                               const DoubleVector &psi) : Solve_which_system(0), Parameter_pt(parameter_pt), Omega(omega)
  {

    call_param_change_handler = false;
    eigenweight = 1.0;

    // Set the problem pointer
    Problem_pt = problem_pt;
    // Set the number of non-augmented degrees of freedom
    Ndof = problem_pt->ndof();

    // Resize the vectors of additional dofs
    Phi.resize(Ndof);
    Psi.resize(Ndof);
    C.resize(Ndof);
    Count.resize(Ndof, 0);

    // Loop over all the elements in the problem
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      // Loop over the local freedoms in an element
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        // Increase the associated global equation number counter
        ++Count[elem_pt->eqn_number(n)];
      }
    }

    // Normalise the guess for phi

    for (unsigned n = 0; n < Ndof; n++)
    {
      Phi[n]=phi[n];
      Psi[n]=psi[n];
    }
    rotate_complex_eigenvector_nicely(Phi, Psi);

    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Phi[n]);
      C[n] = Phi[n];
    }

    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Psi[n]);
    }


    // Now add the parameter
    problem_pt->GetDofPtr().push_back(parameter_pt);
    // Finally add the frequency
    problem_pt->GetDofPtr().push_back(&Omega);

    // rebuild the Dof_distribution_pt
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                              Ndof * 3 + 2, false);
    //std::cout << "Rebuild DofDistribution for " << Ndof * 3 + 2 << " dofs vs " << Problem_pt->GetDofDistributionPt()->nrow_local() << " PTR Is " << Problem_pt->GetDofDistributionPt() << std::endl;                                            
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  //=======================================================================
  /// Destructor return the problem to its original (non-augmented) state
  //=======================================================================
  MyHopfHandler::~MyHopfHandler()
  {
    // If we are using the block solver reset the problem's linear solver
    // to the original one
    BlockHopfLinearSolver *block_hopf_solver_pt =
        dynamic_cast<BlockHopfLinearSolver *>(Problem_pt->linear_solver_pt());
    if (block_hopf_solver_pt)
    {
      // Reset the problem's linear solver
      Problem_pt->linear_solver_pt() = block_hopf_solver_pt->linear_solver_pt();
      // Delete the block solver
      delete block_hopf_solver_pt;
    }
    // Now return the problem to its original size
    Problem_pt->GetDofPtr().resize(Ndof);
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                              Ndof, false);
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  void MyHopfHandler::set_eigenweight(double ew)
  {
    for (unsigned n = 0; n < Ndof; n++)
    {
      Phi[n] *= ew / eigenweight;
      Psi[n] *= ew / eigenweight;
    }
    eigenweight = ew;
  }

  //=============================================================
  /// Get the number of elemental degrees of freedom
  //=============================================================
  unsigned MyHopfHandler::ndof(GeneralisedElement *const &elem_pt)
  {
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
      // Full augmented system
    case 0:
      return (3 * raw_ndof + 2);
      break;
      // Standard non-augmented system
    case 1:
      return raw_ndof;
      break;
      // Complex system
    case 2:
      return (2 * raw_ndof);
      break;

    default:
      throw OomphLibError("Solve_which_system can only be 0,1 or 2",
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  //=============================================================
  /// Get the global equation number of the local unknown
  //============================================================
  unsigned long MyHopfHandler::eqn_number(GeneralisedElement *const &elem_pt,
                                          const unsigned &ieqn_local)
  {
    // Get the raw value
    unsigned raw_ndof = elem_pt->ndof();
    unsigned long global_eqn;
    if (ieqn_local < raw_ndof)
    {
      global_eqn = elem_pt->eqn_number(ieqn_local);
    }
    else if (ieqn_local < 2 * raw_ndof)
    {
      global_eqn = Ndof + elem_pt->eqn_number(ieqn_local - raw_ndof);
    }
    else if (ieqn_local < 3 * raw_ndof)
    {
      global_eqn = 2 * Ndof + elem_pt->eqn_number(ieqn_local - 2 * raw_ndof);
    }
    else if (ieqn_local == 3 * raw_ndof)
    {
      global_eqn = 3 * Ndof;
    }
    else
    {
      global_eqn = 3 * Ndof + 1;
    }
    return global_eqn;
  }

  //==================================================================
  /// Get the residuals
  //=================================================================
  void MyHopfHandler::get_residuals(GeneralisedElement *const &elem_pt,
                                    Vector<double> &residuals)
  {
    // Should only call get residuals for the full system
    if (Solve_which_system == 0)
    {
      // Need to get raw residuals and jacobian
      unsigned raw_ndof = elem_pt->ndof();
      if (raw_ndof == 0)
      {
        residuals.initialise(0.0);
        return;
      }

      DenseMatrix<double> jacobian(raw_ndof), M(raw_ndof);
      // Get the basic residuals, jacobian and mass matrix
      elem_pt->get_jacobian_and_mass_matrix(residuals, jacobian, M);

      bool lambda_tracking=(Parameter_pt==Problem_pt->get_lambda_tracking_real());

      // Initialise the pen-ultimate residual
      residuals[3 * raw_ndof] = -1.0 /
                                (double)(Problem_pt->mesh_pt()->nelement()) * eigenweight;
      residuals[3 * raw_ndof + 1] = 0.0;

      // Now multiply to fill in the residuals
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        residuals[raw_ndof + i] = 0.0;
        residuals[2 * raw_ndof + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          unsigned global_unknown = elem_pt->eqn_number(j);
          // Real part
          residuals[raw_ndof + i] +=
              jacobian(i, j) * Phi[global_unknown] - Omega * M(i, j) * Psi[global_unknown];
          // Imaginary part
          residuals[2 * raw_ndof + i] +=
              jacobian(i, j) * Psi[global_unknown] + Omega * M(i, j) * Phi[global_unknown];
        }
        // Get the global equation number
        unsigned global_eqn = elem_pt->eqn_number(i);

        // Real part
        residuals[3 * raw_ndof] += (Phi[global_eqn] * C[global_eqn]) /
                                   Count[global_eqn];
        // Imaginary part
        residuals[3 * raw_ndof + 1] += (Psi[global_eqn] * C[global_eqn]) /
                                       Count[global_eqn];
      }

      if (lambda_tracking)
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            unsigned global_unknown = elem_pt->eqn_number(j);
            residuals[raw_ndof + i] += (*Parameter_pt) * M(i, j) * Phi[global_unknown];          
            residuals[2 * raw_ndof + i] +=  (*Parameter_pt) * M(i, j) * Psi[global_unknown];              
          }
        }
      }
    }
    else
    {
      throw OomphLibError("Solve_which_system can only be 0",
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  void MyHopfHandler::debug_analytical_filling(oomph::GeneralisedElement *elem_pt, double eps)
  {
    if (!Problem_pt->are_hessian_products_calculated_analytically())
    {
      throw_runtime_error("Cannot do this without having analytical Hessian");
    }
    unsigned nd = this->ndof(elem_pt);
    Vector<double> fd_residuals(nd, 0.0);
    Vector<double> ana_residuals(nd, 0.0);
    DenseMatrix<double> fd_jacobian(nd, nd, 0.0);
    DenseMatrix<double> ana_jacobian(nd, nd, 0.0);
    Problem_pt->unset_analytic_hessian_products();
    this->get_jacobian(elem_pt, fd_residuals, fd_jacobian);
    Problem_pt->set_analytic_hessian_products();
    this->get_jacobian(elem_pt, ana_residuals, ana_jacobian);
    std::vector<std::string> dofnames = dynamic_cast<BulkElementBase *>(elem_pt)->get_dof_names();
    unsigned orig_ndof = dofnames.size();
    for (unsigned i = 0; i < orig_ndof; i++)
      dofnames.push_back("RE_eig[" + dofnames[i] + "]");
    for (unsigned i = 0; i < orig_ndof; i++)
      dofnames.push_back("IM_eig[" + dofnames[i] + "]");
    dofnames.push_back("PARAM");
    dofnames.push_back("OMEGA");
    std::cout << dofnames.size() << "  " << nd << std::endl;
    for (unsigned int i = 0; i < nd; i++)
    {
      double diff = fd_residuals[i] - ana_residuals[i];
      if (diff * diff > eps * eps)
      {
        std::cout << "RESIDUAL DIFF in component " << i << " of " << nd << "  :  " << diff << "  with FD/Ana " << fd_residuals[i] << " and " << ana_residuals[i] << "  ## " << dofnames[i] << std::endl;
      }
    }
    for (unsigned int i = 0; i < nd; i++)
    {
      for (unsigned int j = 0; j < nd; j++)
      {
        double diff = fd_jacobian(i, j) - ana_jacobian(i, j);
        if (diff * diff > eps * eps)
        {
          std::cout << "Jacobian DIFF at  (" << i << " , " << j << ") of " << nd << "  :  " << diff << "  with FD/Ana " << fd_jacobian(i, j) << " and " << ana_jacobian(i, j) << "  ## " << dofnames[i] << " ##wrt## " << dofnames[j] << std::endl;
        }
      }
    }
  }
  //===============================================================
  /// \short Calculate the elemental Jacobian matrix "d equation
  /// / d variable".
  //==================================================================
  void MyHopfHandler::get_jacobian(GeneralisedElement *const &elem_pt,
                                   Vector<double> &residuals,
                                   DenseMatrix<double> &jacobian)
  {

    bool lambda_tracking=(Parameter_pt==Problem_pt->get_lambda_tracking_real());
    bool ana_dparam = lambda_tracking || Problem_pt->is_dparameter_calculated_analytically(Problem_pt->GetDofPtr()[3 * Ndof]);
    bool ana_hessian = ana_dparam && Problem_pt->are_hessian_products_calculated_analytically() && dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);

    // The standard case
    if (Solve_which_system == 0)
    {
      unsigned augmented_ndof = ndof(elem_pt);
      unsigned raw_ndof = elem_pt->ndof();

      if (!ana_hessian)
      {
        if (lambda_tracking)
        {
          throw_runtime_error("Cannot track a complex eigenbranch without having analytical Hessian");
        }
        // Get the basic residuals and jacobian
        DenseMatrix<double> M(raw_ndof);
        elem_pt->get_jacobian_and_mass_matrix(residuals, jacobian, M);
        // Now fill in the actual residuals
        get_residuals(elem_pt, residuals);

        // Now the jacobian appears in other entries
        for (unsigned n = 0; n < raw_ndof; ++n)
        {
          for (unsigned m = 0; m < raw_ndof; ++m)
          {
            jacobian(raw_ndof + n, raw_ndof + m) = jacobian(n, m);
            jacobian(raw_ndof + n, 2 * raw_ndof + m) = -Omega * M(n, m);
            jacobian(2 * raw_ndof + n, 2 * raw_ndof + m) = jacobian(n, m);
            jacobian(2 * raw_ndof + n, raw_ndof + m) = Omega * M(n, m);
            unsigned global_eqn = elem_pt->eqn_number(m);
            jacobian(raw_ndof + n, 3 * raw_ndof + 1) -= M(n, m) * Psi[global_eqn];
            jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) += M(n, m) * Phi[global_eqn];
          }

          unsigned local_eqn = elem_pt->eqn_number(n);
          jacobian(3 * raw_ndof, raw_ndof + n) = C[local_eqn] / Count[local_eqn];
          jacobian(3 * raw_ndof + 1, 2 * raw_ndof + n) = C[local_eqn] / Count[local_eqn];
        }

        const double FD_step = this->FD_step;

        Vector<double> newres_p(augmented_ndof), newres_m(augmented_ndof);

        //	 DenseMatrix<double> dJduPhi(raw_ndof,raw_ndof,0.0);
        //	 DenseMatrix<double> dJduPsi(raw_ndof,raw_ndof,0.0);

        // Loop over the dofs
        for (unsigned n = 0; n < raw_ndof; n++)
        {
          // Just do the x's
          unsigned long global_eqn = eqn_number(elem_pt, n);
          double *unknown_pt = Problem_pt->GetDofPtr()[global_eqn];
          double init = *unknown_pt;
          *unknown_pt += FD_step;

          // Get the new residuals
          get_residuals(elem_pt, newres_p);

          if (!this->symmetric_FD)
          {
            for (unsigned m = 0; m < raw_ndof; m++)
            {
              jacobian(raw_ndof + m, n) =
                  (newres_p[raw_ndof + m] - residuals[raw_ndof + m]) / (FD_step); // These are in fact second order derivatives, i.e. derivatives of the jacobian

              jacobian(2 * raw_ndof + m, n) =
                  (newres_p[2 * raw_ndof + m] - residuals[2 * raw_ndof + m]) / (FD_step);
            }
          }
          else
          {
            *unknown_pt = init;
            *unknown_pt -= FD_step;
            get_residuals(elem_pt, newres_m);
            for (unsigned m = 0; m < raw_ndof; m++)
            {
              jacobian(raw_ndof + m, n) =
                  (newres_p[raw_ndof + m] - newres_m[raw_ndof + m]) / (2 * FD_step); // These are in fact second order derivatives, i.e. derivatives of the jacobian

              jacobian(2 * raw_ndof + m, n) =
                  (newres_p[2 * raw_ndof + m] - newres_m[2 * raw_ndof + m]) / (2 * FD_step);
            }
          }
          // Reset the unknown
          *unknown_pt = init;
        }

        // PARAM DERIV

        if (ana_dparam)
        {
          Vector<double> dres_dparam(augmented_ndof, 0.0);
          this->get_dresiduals_dparameter(elem_pt, Problem_pt->GetDofPtr()[3 * Ndof], dres_dparam);
          for (unsigned m = 0; m < augmented_ndof - 2; m++)
          {
            jacobian(m, 3 * raw_ndof) = dres_dparam[m];
          }
        }
        else
        {
          // Now do the global parameter
          double *unknown_pt = Problem_pt->GetDofPtr()[3 * Ndof];
          double init = *unknown_pt;
          *unknown_pt += FD_step;

          Problem_pt->actions_after_change_in_bifurcation_parameter();
          // Get the new residuals
          get_residuals(elem_pt, newres_p);

          if (!this->symmetric_FD)
          {
            for (unsigned m = 0; m < augmented_ndof - 2; m++)
            {
              jacobian(m, 3 * raw_ndof) =
                  (newres_p[m] - residuals[m]) / FD_step;
            }
          }
          else
          {
            *unknown_pt = init;
            *unknown_pt -= FD_step;
            get_residuals(elem_pt, newres_m); // XXX MOD: IS NOT USED ANYHOW
            for (unsigned m = 0; m < augmented_ndof - 2; m++)
            {
              jacobian(m, 3 * raw_ndof) =
                  (newres_p[m] - residuals[m]) / FD_step;
            }
          }
          // Reset the unknown
          *unknown_pt = init;
          Problem_pt->actions_after_change_in_bifurcation_parameter();
        }
      }
      else // ANALYTIC HESSIAN AND PARAM DERIVS
      {
        pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
        std::vector<SinglePassMultiAssembleInfo> multi_assm;
        
        residuals.initialise(0.0);
        jacobian.initialise(0.0);
        if (raw_ndof == 0)
        {
          return;
        }
          
        oomph::DenseMatrix<double> M(raw_ndof, raw_ndof, 0.0);

        oomph::DenseMatrix<double> dJdU_Eig(2 * raw_ndof, raw_ndof, 0.0), dMdU_Eig(2 * raw_ndof, raw_ndof, 0.0);
        oomph::DenseMatrix<double> dJdParam(raw_ndof, raw_ndof, 0.0), dMdParam(raw_ndof, raw_ndof, 0.0);
        oomph::Vector<double> Eig_local(2 * raw_ndof);

        oomph::Vector<double> dRdParam(raw_ndof, 0.0);
        for (unsigned int i = 0; i < raw_ndof; i++)
        {
          unsigned global_eqn = elem_pt->eqn_number(i);

          Eig_local[i] = Phi[global_eqn];
          Eig_local[raw_ndof + i] = Psi[global_eqn];
        }
        multi_assm.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian, &M));

        multi_assm.back().add_hessian(Eig_local, &dJdU_Eig, &dMdU_Eig);
        if (!lambda_tracking) multi_assm.back().add_param_deriv(Parameter_pt, &dRdParam, &dJdParam, &dMdParam);
        pyoomph_elem_pt->get_multi_assembly(multi_assm);

        // Residuals
        residuals[3 * raw_ndof] = -1.0 / (double)(Problem_pt->mesh_pt()->nelement()) * eigenweight;
        residuals[3 * raw_ndof + 1] = 0.0;
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          residuals[raw_ndof + i] = 0.0;
          residuals[2 * raw_ndof + i] = 0.0;
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            // residuals[raw_ndof + i] += jacobian(i, j) * Phi_local[j] + Omega * M(i, j) * Psi_local[j];
            // residuals[2 * raw_ndof + i] += jacobian(i, j) * Psi_local[j] - Omega * M(i, j) * Phi_local[j];
            residuals[raw_ndof + i] += jacobian(i, j) * Eig_local[j] - Omega * M(i, j) * Eig_local[raw_ndof + j];
            residuals[2 * raw_ndof + i] += jacobian(i, j) * Eig_local[raw_ndof + j] + Omega * M(i, j) * Eig_local[j];
          }
          unsigned global_eqn = elem_pt->eqn_number(i);
          residuals[3 * raw_ndof] += (Phi[global_eqn] * C[global_eqn]) / Count[global_eqn];
          residuals[3 * raw_ndof + 1] += (Psi[global_eqn] * C[global_eqn]) / Count[global_eqn];
        }
        
        // Jacobian
        for (unsigned n = 0; n < raw_ndof; ++n)
        {
          jacobian(n, 3 * raw_ndof) = dRdParam[n];
          jacobian(raw_ndof + n, 3 * raw_ndof) = 0.0;
          jacobian(raw_ndof + n, 3 * raw_ndof + 1) = 0.0;
          jacobian(2 * raw_ndof + n, 3 * raw_ndof) = 0.0;
          jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) = 0.0;
          for (unsigned m = 0; m < raw_ndof; ++m)
          {
            jacobian(raw_ndof + n, m) = dJdU_Eig(n, m) - Omega * dMdU_Eig(raw_ndof + n, m);                                           // dR[Phi]/dU
            jacobian(raw_ndof + n, raw_ndof + m) = jacobian(n, m);                                                                    // dR[Phi]/dPhi
            jacobian(raw_ndof + n, 2 * raw_ndof + m) = -Omega * M(n, m);                                                               // dR[Phi]/dPsi
            jacobian(raw_ndof + n, 3 * raw_ndof) += dJdParam(n, m) * Eig_local[m] - Omega * dMdParam(n, m) * Eig_local[raw_ndof + m]; // dR[Phi]/dParam

            jacobian(2 * raw_ndof + n, m) = dJdU_Eig(raw_ndof + n, m) + Omega * dMdU_Eig(n, m);                                           // dR[Psi]/dU
            jacobian(2 * raw_ndof + n, 2 * raw_ndof + m) = jacobian(n, m);                                                                // dR[Psi]/dPsi
            jacobian(2 * raw_ndof + n, raw_ndof + m) = Omega * M(n, m);                                                                  // dR[Psi]/dPhi
            jacobian(2 * raw_ndof + n, 3 * raw_ndof) += dJdParam(n, m) * Eig_local[raw_ndof + m] + Omega * dMdParam(n, m) * Eig_local[m]; // dR[Psi]/dParam

            jacobian(raw_ndof + n, 3 * raw_ndof + 1) += -M(n, m) * Eig_local[raw_ndof + m]; // dR[Phi]/dOmega
            jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) -= -M(n, m) * Eig_local[m];        // dR[Psi]/dOmega
          }

          unsigned local_eqn = elem_pt->eqn_number(n);
          jacobian(3 * raw_ndof, raw_ndof + n) = C[local_eqn] / Count[local_eqn];         // dR[Param]/dPhi
          jacobian(3 * raw_ndof + 1, 2 * raw_ndof + n) = C[local_eqn] / Count[local_eqn]; // dR[Omega]/dPsi
        }

        if (lambda_tracking)
        {
          for (unsigned i = 0; i < raw_ndof; i++)
          {
            for (unsigned j = 0; j < raw_ndof; j++)
            {              
              residuals[raw_ndof + i] += (*Parameter_pt)  * M(i, j) * Eig_local[ j];
              residuals[2 * raw_ndof + i] += (*Parameter_pt)  * M(i, j) * Eig_local[raw_ndof +j];

              jacobian(raw_ndof + i,j) += (*Parameter_pt)  * dMdU_Eig(i, j)* Eig_local[ j];
              jacobian(2 * raw_ndof + i,j) += (*Parameter_pt)  * dMdU_Eig(i, j)* Eig_local[raw_ndof +j];

              jacobian(raw_ndof + i,raw_ndof + j) += (*Parameter_pt)  * M(i, j);
              jacobian(2 * raw_ndof + i,2*raw_ndof+j) += (*Parameter_pt)  * M(i, j);

              jacobian(raw_ndof + i,3 * raw_ndof) +=M(i, j) * Eig_local[ j];
              jacobian(2*raw_ndof + i,3 * raw_ndof) +=M(i, j) * Eig_local[raw_ndof+j];



            }
          }
        }
      }
    } // End of standard case

    // Normal case
    else if (Solve_which_system == 1)
    {
      // Just get the normal jacobian and residuals
      elem_pt->get_jacobian(residuals, jacobian);
    }
    // Otherwise the augmented complex case
    else if (Solve_which_system == 2)
    {
      unsigned raw_ndof = elem_pt->ndof();

      // Get the basic residuals and jacobian
      DenseMatrix<double> M(raw_ndof);
      elem_pt->get_jacobian_and_mass_matrix(residuals, jacobian, M);

      // We now need to fill in the other blocks
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(n, raw_ndof + m) = Omega * M(n, m);
          jacobian(raw_ndof + n, m) = -Omega * M(n, m);
          jacobian(raw_ndof + n, raw_ndof + m) = jacobian(n, m);
        }
      }

      // Now overwrite to fill in the residuals
      // The decision take is to solve for the mass matrix multiplied
      // terms in the residuals because they require no additional
      // information to assemble.
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        residuals[n] = 0.0;
        residuals[raw_ndof + n] = 0.0;
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          unsigned global_unknown = elem_pt->eqn_number(m);
          // Real part
          residuals[n] += M(n, m) * Psi[global_unknown];
          // Imaginary part
          residuals[raw_ndof + n] -= M(n, m) * Phi[global_unknown];
        }
      }
    } // End of complex augmented case
    else
    {
      throw OomphLibError("Solve_which_system can only be 0,1 or 2",
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  //==================================================================
  /// Get the derivatives of the augmented residuals with respect to
  /// a parameter
  //=================================================================
  void MyHopfHandler::get_dresiduals_dparameter(
      GeneralisedElement *const &elem_pt,
      double *const &parameter_pt, Vector<double> &dres_dparam)
  {
    // Should only call get residuals for the full system
    if (Solve_which_system == 0)
    {
      // Need to get raw residuals and jacobian
      unsigned raw_ndof = elem_pt->ndof();

      DenseMatrix<double> djac_dparam(raw_ndof), dM_dparam(raw_ndof);
      // Get the basic residuals, jacobian and mass matrix
      elem_pt->get_djacobian_and_dmass_matrix_dparameter(
          parameter_pt, dres_dparam, djac_dparam, dM_dparam);

      // Initialise the pen-ultimate residual, which does not
      // depend on the parameter
      dres_dparam[3 * raw_ndof] = 0.0;
      dres_dparam[3 * raw_ndof + 1] = 0.0;

      // Now multiply to fill in the residuals
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        dres_dparam[raw_ndof + i] = 0.0;
        dres_dparam[2 * raw_ndof + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          unsigned global_unknown = elem_pt->eqn_number(j);
          // Real part
          dres_dparam[raw_ndof + i] +=
              djac_dparam(i, j) * Phi[global_unknown] +
              Omega * dM_dparam(i, j) * Psi[global_unknown];
          // Imaginary part
          dres_dparam[2 * raw_ndof + i] +=
              djac_dparam(i, j) * Psi[global_unknown] -
              Omega * dM_dparam(i, j) * Phi[global_unknown];
        }
      }
    }
    else
    {
      throw OomphLibError("Solve_which_system can only be 0",
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  //========================================================================
  /// Overload the derivative of the residuals and jacobian
  /// with respect to a parameter so that it breaks because it should not
  /// be required
  //========================================================================
  void MyHopfHandler::get_djacobian_dparameter(
      GeneralisedElement *const &elem_pt,
      double *const &parameter_pt,
      Vector<double> &dres_dparam,
      DenseMatrix<double> &djac_dparam)
  {
    std::ostringstream error_stream;
    error_stream << "This function has not been implemented because it is not required\n";
    error_stream << "in standard problems.\n";
    error_stream << "If you find that you need it, you will have to implement it!\n\n";

    throw OomphLibError(error_stream.str(),
                        OOMPH_CURRENT_FUNCTION,
                        OOMPH_EXCEPTION_LOCATION);
  }

  void MyHopfHandler::get_hessian_vector_products(
      GeneralisedElement *const &elem_pt,
      Vector<double> const &Y,
      DenseMatrix<double> const &C,
      DenseMatrix<double> &product)
  {
    elem_pt->get_hessian_vector_products(Y, C, product);
  }

  //==========================================================================
  /// Return the eigenfunction(s) associated with the bifurcation that
  /// has been detected in bifurcation tracking problems
  //==========================================================================
  void MyHopfHandler::get_eigenfunction(
      Vector<DoubleVector> &eigenfunction)
  {
    // There is a real and imaginary part of the null vector
    eigenfunction.resize(2);
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false);
    // Rebuild the vector
    eigenfunction[0].build(&dist, 0.0);
    eigenfunction[1].build(&dist, 0.0);
    // Set the value to be the null vector
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[0][n] = Phi[n];
      eigenfunction[1][n] = Psi[n];
    }
  }

  std::vector<std::complex<double>> MyHopfHandler::get_nicely_rotated_eigenfunction()
  {
    std::vector<std::complex<double>> eigenfunction(Ndof);
    oomph::Vector<double> PhiRot=Phi;
    oomph::Vector<double> PsiRot=Psi;
    rotate_complex_eigenvector_nicely(PhiRot,PsiRot);
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[n] = std::complex<double>(PhiRot[n], PsiRot[n]);
    }
    return eigenfunction;
  }

  //====================================================================
  /// Set to solve the standard (underlying jacobian)  system
  //===================================================================
  void MyHopfHandler::solve_standard_system()
  {
    if (Solve_which_system != 1)
    {
      Solve_which_system = 1;
      // Restrict the problem to the standard variables only
      Problem_pt->GetDofPtr().resize(Ndof);
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                                Ndof, false);
      // Remove all previous sparse storage used during Jacobian assembly
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    }
  }

  //====================================================================
  /// Set to solve the complex (jacobian and mass matrix)  system
  //===================================================================
  void MyHopfHandler::solve_complex_system()
  {
    // If we were not solving the complex system resize the unknowns
    // accordingly
    if (Solve_which_system != 2)
    {
      Solve_which_system = 2;
      // Resize to the first Ndofs (will work whichever system we were
      // solving before)
      Problem_pt->GetDofPtr().resize(Ndof);
      // Add the first (real) part of the eigenfunction back into the problem
      for (unsigned n = 0; n < Ndof; n++)
      {
        Problem_pt->GetDofPtr().push_back(&Phi[n]);
      }
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                                Ndof * 2, false);
      // Remove all previous sparse storage used during Jacobian assembly
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    }
  }

  //=================================================================
  /// Set to Solve full system system
  //=================================================================
  void MyHopfHandler::solve_full_system()
  {
    // If we are starting from another system
    if (Solve_which_system)
    {
      Solve_which_system = 0;

      // Resize to the first Ndofs (will work whichever system we were
      // solving before)
      Problem_pt->GetDofPtr().resize(Ndof);
      // Add the additional unknowns back into the problem
      for (unsigned n = 0; n < Ndof; n++)
      {
        Problem_pt->GetDofPtr().push_back(&Phi[n]);
      }
      for (unsigned n = 0; n < Ndof; n++)
      {
        Problem_pt->GetDofPtr().push_back(&Psi[n]);
      }
      // Now add the parameter
      Problem_pt->GetDofPtr().push_back(Parameter_pt);
      // Finally add the frequency
      Problem_pt->GetDofPtr().push_back(&Omega);

      //
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                                3 * Ndof + 2, false);
      // Remove all previous sparse storage used during Jacobian assembly
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    }
  }

  void MyHopfHandler::realign_C_vector()
  {

    double dot = 0.0;
    double doti = 0.0;
    double phisqr = 0.0;
    double psisqr = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      double phin = *(Problem_pt->GetDofPtr()[Ndof + n]);
      double psin = *(Problem_pt->GetDofPtr()[2 * Ndof + n]);
      dot += C[n] * phin;
      phisqr += phin * phin;
      doti += C[n] * psin;
      psisqr += psin * psin;
    }
    std::cerr << "DOT OF C and PHi is " << dot << " and PHi^2 = " << phisqr << std::endl;
    std::cerr << "DOT OF C and Psi is " << doti << " and Psi^2 = " << psisqr << std::endl;

    double lf = eigenweight / sqrt(phisqr);
    for (unsigned n = 0; n < Ndof; n++)
    {
      double *phin = (Problem_pt->GetDofPtr()[Ndof + n]);
      double *psin = (Problem_pt->GetDofPtr()[2 * Ndof + n]);
      (*phin) *= lf;
      (*psin) *= lf;
      C[n] = (*phin);
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  MyFoldHandler::MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt) : Solve_which_system(Full_augmented), Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    FD_step = 1e-8;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();

    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);

    Phi.resize(Ndof);
    Y.resize(Ndof);
    Count.resize(Ndof, 0);

    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        ++Count[elem_pt->eqn_number(n)];
      }
    }

    LinearSolver *const linear_solver_pt = problem_pt->linear_solver_pt();

    bool enable_resolve = linear_solver_pt->is_resolve_enabled();
    linear_solver_pt->enable_resolve();
    DoubleVector x(dist_pt, 0.0);
    linear_solver_pt->solve(problem_pt, x);
    problem_pt->get_derivative_wrt_global_parameter(parameter_pt, x);
    DoubleVector input_x(x);
    linear_solver_pt->resolve(input_x, x);
    if (enable_resolve)
    {
      linear_solver_pt->enable_resolve();
    }
    else
    {
      linear_solver_pt->disable_resolve();
    }
    problem_pt->GetDofPtr().push_back(parameter_pt);
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += x[n] * x[n];
    }
    length = sqrt(length);

    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Y[n]);
      Y[n] = Phi[n] = -x[n] / length;
    }
    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * 2 + 1, true);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    delete dist_pt;
  }

  MyFoldHandler::MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const DoubleVector &eigenvector) : Solve_which_system(Full_augmented), Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    FD_step = 1e-8;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);
    Phi.resize(Ndof);
    Y.resize(Ndof);
    Count.resize(Ndof, 0);
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        ++Count[elem_pt->eqn_number(n)];
      }
    }
    problem_pt->GetDofPtr().push_back(parameter_pt);
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += eigenvector[n] * eigenvector[n];
    }
    length = sqrt(length);
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Y[n]);
      Y[n] = Phi[n] = eigenvector[n] / length;
    }
    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * 2 + 1, true);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);

    delete dist_pt;
  }

  MyFoldHandler::MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const DoubleVector &eigenvector, const DoubleVector &normalisation) : Solve_which_system(Full_augmented), Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    FD_step = 1e-8;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);
    Phi.resize(Ndof);
    Y.resize(Ndof);
    Count.resize(Ndof, 0);
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        ++Count[elem_pt->eqn_number(n)];
      }
    }
    problem_pt->GetDofPtr().push_back(parameter_pt);
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += eigenvector[n] * normalisation[n];
    }
    length = sqrt(length);
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Y[n]);
      Y[n] = eigenvector[n] / length;
      Phi[n] = normalisation[n];
    }
    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * 2 + 1, true);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    delete dist_pt;
  }

  unsigned MyFoldHandler::ndof(GeneralisedElement *const &elem_pt)
  {
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
    case Full_augmented:
      return (2 * raw_ndof + 1);
      break;

    case Block_augmented_J:
      return (raw_ndof + 1);
      break;

    case Block_J:
      return raw_ndof;
      break;

    default:
      std::ostringstream error_stream;
      error_stream << "The Solve_which_system flag can only take values 0, 1, 2"
                   << " not " << Solve_which_system << "\n";
      throw OomphLibError(error_stream.str(),
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  unsigned long MyFoldHandler::eqn_number(GeneralisedElement *const &elem_pt,
                                          const unsigned &ieqn_local)
  {
    unsigned raw_ndof = elem_pt->ndof();
    unsigned long global_eqn = 0;
    if (ieqn_local < raw_ndof)
    {
      global_eqn = elem_pt->eqn_number(ieqn_local);
    }
    else if (ieqn_local == raw_ndof)
    {
      global_eqn = Ndof;
    }
    else
    {
      global_eqn = Ndof + 1 + elem_pt->eqn_number(ieqn_local - 1 - raw_ndof);
    }
    return global_eqn;
  }

  void MyFoldHandler::get_residuals(GeneralisedElement *const &elem_pt,
                                    Vector<double> &residuals)
  {
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
    case Block_J:
    {
      elem_pt->get_residuals(residuals);
    }
    break;
    case Block_augmented_J:
    {
      elem_pt->get_residuals(residuals);
      residuals[raw_ndof] = 0.0;
    }
    break;
    case Full_augmented:
    {
      DenseMatrix<double> jacobian(raw_ndof);
      DenseMatrix<double> mass_matrix(raw_ndof);
      if (Parameter_pt==Problem_pt->get_lambda_tracking_real())
      {      
        elem_pt->get_jacobian_and_mass_matrix(residuals, jacobian, mass_matrix);
      }
      else
      {
        elem_pt->get_jacobian(residuals, jacobian);
      }
      residuals[raw_ndof] = -1.0 / Problem_pt->mesh_pt()->nelement() * eigenweight;
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        residuals[raw_ndof + 1 + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y[elem_pt->eqn_number(j)];
        }
        unsigned global_eqn = elem_pt->eqn_number(i);
        residuals[raw_ndof] += (Phi[global_eqn] * Y[global_eqn]) / Count[global_eqn];
      }

      if (Parameter_pt==Problem_pt->get_lambda_tracking_real())
      {      
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            residuals[raw_ndof + 1 + i] += (*Parameter_pt)*mass_matrix(i, j) * Y[elem_pt->eqn_number(j)];
          }
        }
      }



            
    }
    break;

    default:
      std::ostringstream error_stream;
      error_stream << "The Solve_which_system flag can only take values 0, 1, 2"
                   << " not " << Solve_which_system << "\n";
      throw OomphLibError(error_stream.str(),
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  void MyFoldHandler::get_jacobian(GeneralisedElement *const &elem_pt,
                                   Vector<double> &residuals,
                                   DenseMatrix<double> &jacobian)
  {
    // If true, we do not track a fold by adjusting the parameter, but track an eigenvalue branch
    bool lambda_continuation=  (Parameter_pt==Problem_pt->get_lambda_tracking_real());      
    bool ana_dparam = lambda_continuation || Problem_pt->is_dparameter_calculated_analytically(Problem_pt->GetDofPtr()[Ndof]);    
    bool ana_hessian = ana_dparam && Problem_pt->are_hessian_products_calculated_analytically() && dynamic_cast<BulkElementBase *>(elem_pt);

    unsigned augmented_ndof = ndof(elem_pt);
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
    case Block_J:
    {
      elem_pt->get_jacobian(residuals, jacobian);
    }
    break;
    case Block_augmented_J:
    {
      get_residuals(elem_pt, residuals);
      Vector<double> newres(augmented_ndof);
      elem_pt->get_jacobian(newres, jacobian);
      const double FD_step = 1.0e-8;
      {
        double *unknown_pt = Problem_pt->GetDofPtr()[Ndof];
        double init = *unknown_pt;
        *unknown_pt += FD_step;

        Problem_pt->actions_after_change_in_bifurcation_parameter();
        get_residuals(elem_pt, newres);

        for (unsigned n = 0; n < raw_ndof; n++)
        {
          jacobian(n, augmented_ndof - 1) = (newres[n] - residuals[n]) / FD_step;
        }
        *unknown_pt = init;

        Problem_pt->actions_after_change_in_bifurcation_parameter();
      }

      for (unsigned n = 0; n < raw_ndof; n++)
      {
        unsigned local_eqn = elem_pt->eqn_number(n);
        jacobian(augmented_ndof - 1, n) = Phi[local_eqn] / Count[local_eqn];
      }
    }
    break;

    case Full_augmented:
    {


      if (ana_hessian)
      {

        

        jacobian.initialise(0.0);
        residuals.initialise(0.0);
        DenseMatrix<double> djac_dparam(raw_ndof, raw_ndof, 0.0);
        Vector<double> dres_dparam(raw_ndof, 0.0);
        DenseMatrix<double> M(raw_ndof, raw_ndof, 0.0);
        DenseMatrix<double> dJduPhiH(raw_ndof, raw_ndof, 0.0);
        DenseMatrix<double> dMduPhiH(raw_ndof, raw_ndof, 0.0);
        Vector<double> Y_local(raw_ndof);
        for (unsigned _e = 0; _e < raw_ndof; _e++)
        {
          Y_local[_e] = Y[elem_pt->eqn_number(_e)];
        }

        pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
        std::vector<SinglePassMultiAssembleInfo> assemble_info;
        
        if (!lambda_continuation)
        {
          assemble_info.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian));
          assemble_info.back().add_param_deriv(Parameter_pt, &dres_dparam, &djac_dparam);
          assemble_info.back().add_hessian(Y_local, &dJduPhiH);
        }
        else
        {
          assemble_info.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian,&M));
          assemble_info.back().add_hessian(Y_local, &dJduPhiH,&dMduPhiH);
        }
        
        pyoomph_elem_pt->get_multi_assembly(assemble_info);

        // Fill augmented residuals
        residuals[raw_ndof] = -1.0 / Problem_pt->mesh_pt()->nelement() * eigenweight;
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          residuals[raw_ndof + 1 + i] = 0.0;
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y_local[j];
          }
          unsigned global_eqn = elem_pt->eqn_number(i);
          residuals[raw_ndof] += (Phi[global_eqn] * Y[global_eqn]) / Count[global_eqn];
        }

        // And the Jacobian
        for (unsigned n = 0; n < raw_ndof; n++)
        {
          jacobian(n, raw_ndof) = dres_dparam[n];
          for (unsigned m = 0; m < raw_ndof; m++)
          {
            jacobian(raw_ndof + 1 + n, raw_ndof + 1 + m) = jacobian(n, m);
            jacobian(raw_ndof + 1 + n, raw_ndof) += djac_dparam(n, m) * Y_local[m];
            jacobian(raw_ndof + 1 + m, n) = dJduPhiH(m, n);
          }
          unsigned global_eqn = elem_pt->eqn_number(n);
          jacobian(raw_ndof, raw_ndof + 1 + n) = Phi[global_eqn] / Count[global_eqn];
        }

         if (lambda_continuation)
         {      
          for (unsigned n = 0; n < raw_ndof; n++)
          {
            for (unsigned m = 0; m < raw_ndof; m++)
            {
              residuals[raw_ndof + 1 + n] += (*Parameter_pt)*M(n, m) * Y[elem_pt->eqn_number(m)];
              jacobian(raw_ndof + 1 + n, raw_ndof + 1 + m) += (*Parameter_pt)*M(n, m);
              jacobian(raw_ndof + 1 + n, raw_ndof) += M(n, m) * Y_local[m];
              jacobian(raw_ndof + 1 + m, n) += (*Parameter_pt)*dMduPhiH(m, n);
            }
          }
         }
      }
      else
      {
        if (lambda_continuation) throw_runtime_error("Hessian must be calculated analytically for eigenbranch continuation, i.e. finite differences is not implemented yet");
        get_residuals(elem_pt, residuals);
        Vector<double> newres(raw_ndof);
        DenseMatrix<double> newjac(raw_ndof);
        elem_pt->get_jacobian(newres, jacobian);

        for (unsigned n = 0; n < raw_ndof; n++)
        {
          for (unsigned m = 0; m < raw_ndof; m++)
          {
            jacobian(raw_ndof + 1 + n, raw_ndof + 1 + m) = jacobian(n, m);
          }
        }

        if (ana_dparam)
        {
          DenseMatrix<double> djac_dparam(raw_ndof, raw_ndof, 0.0);
          Vector<double> dres_dparam(raw_ndof, 0.0);
          elem_pt->get_djacobian_dparameter(Problem_pt->GetDofPtr()[Ndof], dres_dparam, djac_dparam);
          for (unsigned n = 0; n < raw_ndof; n++)
          {
            jacobian(n, raw_ndof) = dres_dparam[n];
            for (unsigned l = 0; l < raw_ndof; l++)
            {
              jacobian(raw_ndof + 1 + n, raw_ndof) += djac_dparam(n, l) * Y[elem_pt->eqn_number(l)];
            }
          }
        }
        else
        {

          double FD_step = this->FD_step;
          {
            double *unknown_pt = Problem_pt->GetDofPtr()[Ndof];
            double init = *unknown_pt;
            *unknown_pt += FD_step;
            //            Problem_pt->actions_after_change_in_bifurcation_parameter();
            elem_pt->get_jacobian(newres, newjac);
            if (!this->symmetric_FD)
            {
              for (unsigned n = 0; n < raw_ndof; n++)
              {
                jacobian(n, raw_ndof) = (newres[n] - residuals[n]) / FD_step;
                for (unsigned l = 0; l < raw_ndof; l++)
                {
                  jacobian(raw_ndof + 1 + n, raw_ndof) += (newjac(n, l) - jacobian(n, l)) * Y[elem_pt->eqn_number(l)] /
                                                          FD_step;
                }
              }
            }
            else
            {
              *unknown_pt = init;
              *unknown_pt -= FD_step;
              Vector<double> newres_m(raw_ndof);
              DenseMatrix<double> newjac_m(raw_ndof);
              elem_pt->get_jacobian(newres_m, newjac_m);
              for (unsigned n = 0; n < raw_ndof; n++)
              {
                jacobian(n, raw_ndof) = (newres[n] - newres_m[n]) / (2 * FD_step);
                for (unsigned l = 0; l < raw_ndof; l++)
                {
                  jacobian(raw_ndof + 1 + n, raw_ndof) += (newjac(n, l) - newjac_m(n, l)) * Y[elem_pt->eqn_number(l)] /
                                                          (2 * FD_step);
                }
              }
            }
            *unknown_pt = init;
            Problem_pt->actions_after_change_in_bifurcation_parameter();
          }
        }

        for (unsigned n = 0; n < raw_ndof; n++)
        {
          unsigned long global_eqn = eqn_number(elem_pt, n);
          double *unknown_pt = Problem_pt->GetDofPtr()[global_eqn];
          double init = *unknown_pt;
          *unknown_pt += FD_step;
          //          Problem_pt->actions_before_newton_convergence_check(); /// ALICE
          elem_pt->get_jacobian(newres, newjac);
          if (!this->symmetric_FD)
          {
            // Work out the differences
            for (unsigned k = 0; k < raw_ndof; k++)
            {
              for (unsigned l = 0; l < raw_ndof; l++)
              {
                jacobian(raw_ndof + 1 + k, n) += (newjac(k, l) - jacobian(k, l)) * Y[elem_pt->eqn_number(l)] / FD_step;
              }
            }
          }
          else
          {
            *unknown_pt = init;
            *unknown_pt -= FD_step;
            Vector<double> newres_m(raw_ndof);
            DenseMatrix<double> newjac_m(raw_ndof);
            elem_pt->get_jacobian(newres_m, newjac_m);
            // Work out the differences
            for (unsigned k = 0; k < raw_ndof; k++)
            {
              for (unsigned l = 0; l < raw_ndof; l++)
              {
                jacobian(raw_ndof + 1 + k, n) += (newjac(k, l) - newjac_m(k, l)) * Y[elem_pt->eqn_number(l)] / (2 * FD_step);
              }
            }
          }
          *unknown_pt = init;
          //        Problem_pt->actions_before_newton_convergence_check(); /// ALICE
        }

        // Fill in the row corresponding to the parameter
        for (unsigned n = 0; n < raw_ndof; n++)
        {
          unsigned global_eqn = elem_pt->eqn_number(n);
          jacobian(raw_ndof, raw_ndof + 1 + n) = Phi[global_eqn] / Count[global_eqn];
        }
      }
    }
    break;

    default:
      std::ostringstream error_stream;
      error_stream << "The Solve_which_system flag can only take values 0, 1, 2"
                   << " not " << Solve_which_system << "\n";
      throw OomphLibError(error_stream.str(),
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  void MyFoldHandler::get_dresiduals_dparameter(
      GeneralisedElement *const &elem_pt,
      double *const &parameter_pt,
      Vector<double> &dres_dparam)
  {
    unsigned raw_ndof = elem_pt->ndof();
    switch (Solve_which_system)
    {
    case Block_J:
    {
      elem_pt->get_dresiduals_dparameter(parameter_pt, dres_dparam);
    }
    break;

    case Block_augmented_J:
    {
      elem_pt->get_dresiduals_dparameter(parameter_pt, dres_dparam);
      dres_dparam[raw_ndof] = 0.0;
    }
    break;
    case Full_augmented:
    {
      DenseMatrix<double> djac_dparam(raw_ndof);
      elem_pt->get_djacobian_dparameter(parameter_pt, dres_dparam, djac_dparam);
      dres_dparam[raw_ndof] = 0.0;
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        dres_dparam[raw_ndof + 1 + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          dres_dparam[raw_ndof + 1 + i] += djac_dparam(i, j) * Y[elem_pt->eqn_number(j)];
        }
      }
    }
    break;

    default:
      std::ostringstream error_stream;
      error_stream << "The Solve_which_system flag can only take values 0, 1, 2"
                   << " not " << Solve_which_system << "\n";
      throw OomphLibError(error_stream.str(),
                          OOMPH_CURRENT_FUNCTION,
                          OOMPH_EXCEPTION_LOCATION);
    }
  }

  void MyFoldHandler::get_djacobian_dparameter(GeneralisedElement *const &elem_pt, double *const &parameter_pt, Vector<double> &dres_dparam, DenseMatrix<double> &djac_dparam)
  {
    std::ostringstream error_stream;
    error_stream << "This function has not been implemented because it is not required\n";
    error_stream << "in standard problems.\n";
    error_stream << "If you find that you need it, you will have to implement it!\n\n";

    throw OomphLibError(error_stream.str(),
                        OOMPH_CURRENT_FUNCTION,
                        OOMPH_EXCEPTION_LOCATION);
  }

  void MyFoldHandler::get_hessian_vector_products(GeneralisedElement *const &elem_pt, Vector<double> const &Y, DenseMatrix<double> const &C, DenseMatrix<double> &product)
  {
    elem_pt->get_hessian_vector_products(Y, C, product);
  }

  void MyFoldHandler::get_eigenfunction(Vector<DoubleVector> &eigenfunction)
  {
    eigenfunction.resize(1);
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false);
    eigenfunction[0].build(&dist, 0.0);
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[0][n] = Y[n];
    }
  }

  MyFoldHandler::~MyFoldHandler()
  {
    AugmentedBlockFoldLinearSolver *block_fold_solver_pt = dynamic_cast<AugmentedBlockFoldLinearSolver *>(
        Problem_pt->linear_solver_pt());

    if (block_fold_solver_pt)
    {
      Problem_pt->linear_solver_pt() = block_fold_solver_pt->linear_solver_pt();
      delete block_fold_solver_pt;
    }
    Problem_pt->GetDofPtr().resize(Ndof);
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof, false);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  void MyFoldHandler::solve_augmented_block_system()
  {
    if (Solve_which_system != Block_augmented_J)
    {
      if (Solve_which_system == Block_J)
      {
        Problem_pt->GetDofPtr().push_back(Parameter_pt);
      }
      Problem_pt->GetDofPtr().resize(Ndof + 1);
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof + 1, false);
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
      Solve_which_system = Block_augmented_J;
    }
  }

  void MyFoldHandler::solve_block_system()
  {
    if (Solve_which_system != Block_J)
    {
      Problem_pt->GetDofPtr().resize(Ndof);
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof, false);
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
      Solve_which_system = Block_J;
    }
  }

  void MyFoldHandler::solve_full_system()
  {
    if (Solve_which_system != Full_augmented)
    {
      if (Solve_which_system == Block_J)
      {
        Problem_pt->GetDofPtr().push_back(Parameter_pt);
      }
      for (unsigned n = 0; n < Ndof; n++)
      {
        Problem_pt->GetDofPtr().push_back(&Y[n]);
      }
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof * 2 + 1, false);
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
      Solve_which_system = Full_augmented;
    }
  }

  void MyFoldHandler::set_eigenweight(double ew)
  {
    for (unsigned n = 0; n < Ndof; n++)
    {
      Phi[n] *= ew / eigenweight;
    }
    eigenweight = ew;
  }

  void MyFoldHandler::realign_C_vector()
  {

    double dot = 0.0;
    double phisqr = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      double phin = *(Problem_pt->GetDofPtr()[Ndof + 1 + n]);
      dot += Y[n] * phin;
      phisqr += phin * phin;
    }
    std::cerr << "DOT OF C and PHi is " << dot << " and PHi^2 = " << phisqr << std::endl;

    for (unsigned n = 0; n < Ndof; n++)
    {
      double phin = *(Problem_pt->GetDofPtr()[Ndof + 1 + n]);
      Y[n] = phin / phisqr; // Renormalize the c vector
    }
  }

  //////////////////////////////////////////////////

  MyPitchForkHandler::MyPitchForkHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &symmetry_vector) : Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    symmetryweight = 1.0;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);
    Psi.resize(Ndof);
    Y.resize(Ndof);
    C.resize(Ndof);
    Count.resize(Ndof, 0);
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    Nelement = n_element;
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        ++Count[elem_pt->eqn_number(n)];
      }
    }
    problem_pt->GetDofPtr().push_back(parameter_pt);
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += symmetry_vector[n] * symmetry_vector[n];
    }
    length = sqrt(length);
    for (unsigned n = 0; n < Ndof; n++)
    {
      problem_pt->GetDofPtr().push_back(&Y[n]);
      C[n] = Y[n] = symmetry_vector[n] / length;
    }
    for (unsigned n = 0; n < Ndof; n++)
    {
      // problem_pt->GetDofPtr().push_back(&Psi[n]);
      Psi[n] = symmetry_vector[n] / length;
    }

    if (problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      setup_U_times_Psi_residual_indices();

    if (!problem_pt->is_quiet())
    {
      double initial_orthogonality = 0.0;
      if (problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      {
        for (unsigned e = 0; e < n_element; e++)
        {
          GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
          DenseMatrix<double> psi_i_times_psi_j(elem_pt->ndof());
          initial_orthogonality += this->get_integrated_U_dot_Psi(elem_pt, psi_i_times_psi_j);
        }
      }
      else
      {
        for (unsigned n = 0; n < Ndof; n++)
        {
          initial_orthogonality += Psi[n] * (*problem_pt->GetDofPtr()[n]);
        }
      }
      std::cout << "Initial pitchfork symmetry breaking orthogonality <psi,u>=" << initial_orthogonality << std::endl;
    }
    problem_pt->GetDofPtr().push_back(&Sigma);
    Sigma = 0.0;

    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * 2 + 2, true);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    delete dist_pt;
  }

  void MyPitchForkHandler::set_eigenweight(double ew)
  {
    for (unsigned n = 0; n < Ndof; n++)
    {
      Psi[n] *= ew / eigenweight;
    }
    eigenweight = ew;
  }

  MyPitchForkHandler::~MyPitchForkHandler()
  {
    Problem_pt->GetDofPtr().resize(Ndof);
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof, false);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  double MyPitchForkHandler::get_integrated_U_dot_Psi(oomph::GeneralisedElement *const &elem_pt, DenseMatrix<double> &psi_i_times_psi_j)
  {
    unsigned raw_ndof = elem_pt->ndof();
    psi_i_times_psi_j.initialise(0.0);
    Vector<double> residuals(raw_ndof);
    this->set_assembled_residual(elem_pt, 1);
    elem_pt->get_jacobian(residuals, psi_i_times_psi_j);
    double res = 0.0;
    for (unsigned int i = 0; i < raw_ndof; i++)
    {
      unsigned eqn_i = elem_pt->eqn_number(i);
      for (unsigned int j = 0; j < raw_ndof; j++)
      {
        unsigned eqn_j = elem_pt->eqn_number(j);
        res += (*Problem_pt->global_dof_pt(eqn_i)) * Psi[eqn_j] * psi_i_times_psi_j(i, j); // Contract with mass matrix to get integral(U*Psi)
      }
    }
    this->set_assembled_residual(elem_pt, 0);
    return res;
  }

  unsigned MyPitchForkHandler::ndof(oomph::GeneralisedElement *const &elem_pt)
  {
    unsigned raw_ndof = elem_pt->ndof();
    return (2 * raw_ndof + 2);
  }

  unsigned long MyPitchForkHandler::eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local)
  {
    unsigned raw_ndof = elem_pt->ndof();
    if (ieqn_local < raw_ndof)
    {
      return elem_pt->eqn_number(ieqn_local);
    }
    // The bifurcation parameter equation
    else if (ieqn_local == raw_ndof)
    {
      return Ndof;
    }
    else if (ieqn_local < (2 * raw_ndof + 1))
    {
      return Ndof + 1 + elem_pt->eqn_number(ieqn_local - 1 - raw_ndof);
    }
    else
    {
      return 2 * Ndof + 1;
    }
  }

  void MyPitchForkHandler::get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals)
  {
    unsigned raw_ndof = elem_pt->ndof();
    DenseMatrix<double> jacobian(raw_ndof, raw_ndof, 0.0);
    Vector<double> symmetryR(raw_ndof, 0.0);
    DenseMatrix<double> symmetryA(raw_ndof, raw_ndof, 0.0);
    if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
    {
      pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
      std::vector<SinglePassMultiAssembleInfo> assemble_info;
      assemble_info.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian));
      assemble_info.push_back(SinglePassMultiAssembleInfo(resolve_assembled_residual(elem_pt, 1), &symmetryR, &symmetryA));
      pyoomph_elem_pt->get_multi_assembly(assemble_info);
    }
    else
    {
      elem_pt->get_jacobian(residuals, jacobian);
    }
    residuals[raw_ndof] = 0.0;
    residuals[2 * raw_ndof + 1] = -1.0 / Problem_pt->mesh_pt()->nelement() * eigenweight;
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      unsigned local_eqn = elem_pt->eqn_number(i);
      residuals[raw_ndof + 1 + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned local_unknown = elem_pt->eqn_number(j);
        residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y[local_unknown];
      }
      residuals[2 * raw_ndof + 1] += (Y[local_eqn] * C[local_eqn]) / Count[local_eqn];
    }
    if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
    {
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        unsigned local_eqn = elem_pt->eqn_number(i);
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          unsigned local_unknown = elem_pt->eqn_number(j);
          residuals[i] += Sigma * symmetryA(i, j) * Psi[local_unknown];
          residuals[raw_ndof] += ((*Problem_pt->global_dof_pt(local_eqn)) * symmetryA(i, j) * Psi[local_unknown]);
        }
      }
    }
    else
    {
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        unsigned local_eqn = elem_pt->eqn_number(i);
        residuals[i] += Sigma * Psi[local_eqn] / Count[local_eqn];
        residuals[raw_ndof] += ((*Problem_pt->global_dof_pt(local_eqn)) * Psi[local_eqn]) / Count[local_eqn];
      }
    }
  }

  void MyPitchForkHandler::get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
    bool ana_dparam = Problem_pt->is_dparameter_calculated_analytically(Problem_pt->GetDofPtr()[Ndof]);
    bool ana_hessian = ana_dparam && Problem_pt->are_hessian_products_calculated_analytically() && dynamic_cast<BulkElementBase *>(elem_pt);

    unsigned augmented_ndof = ndof(elem_pt);
    unsigned raw_ndof = elem_pt->ndof();
    if (ana_hessian && ana_dparam)
    {
      jacobian.initialise(0.0);
      residuals.initialise(0.0);
      DenseMatrix<double> djac_dparam(raw_ndof, raw_ndof, 0.0);
      Vector<double> dres_dparam(raw_ndof, 0.0);
      DenseMatrix<double> dJduPhiH(raw_ndof, raw_ndof, 0.0);
      Vector<double> symmetryR(raw_ndof, 0.0);
      DenseMatrix<double> symmetryA(raw_ndof, raw_ndof, 0.0);
      DenseMatrix<double> symmetryDADU_times_Psi(raw_ndof, raw_ndof, 0.0);
      Vector<double> Y_local(raw_ndof);
      for (unsigned _e = 0; _e < raw_ndof; _e++)
      {
        Y_local[_e] = Y[elem_pt->eqn_number(_e)];
      }
      Vector<double> Psi_local(raw_ndof);
      for (unsigned _e = 0; _e < raw_ndof; _e++)
      {
        Psi_local[_e] = Psi[elem_pt->eqn_number(_e)];
      }

      pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
      std::vector<SinglePassMultiAssembleInfo> assemble_info;
      assemble_info.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian));
      assemble_info.back().add_param_deriv(Parameter_pt, &dres_dparam, &djac_dparam);
      assemble_info.back().add_hessian(Y_local, &dJduPhiH);
      if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      {
        assemble_info.push_back(SinglePassMultiAssembleInfo(resolve_assembled_residual(elem_pt, 1), &symmetryR, &symmetryA));
        assemble_info.back().add_hessian(Psi_local, &symmetryDADU_times_Psi);
      }
      pyoomph_elem_pt->get_multi_assembly(assemble_info);

      residuals[2 * raw_ndof + 1] = -1.0 / Problem_pt->mesh_pt()->nelement() * eigenweight;
      // Fill augmented residuals
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        unsigned local_eqn = elem_pt->eqn_number(i);
        residuals[raw_ndof + 1 + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y_local[j];
        }

        residuals[2 * raw_ndof + 1] += (Y[local_eqn] * C[local_eqn]) / Count[local_eqn];
      }

      // And the Jacobian
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        unsigned local_eqn = elem_pt->eqn_number(n);
        jacobian(n, raw_ndof) = dres_dparam[n];
        jacobian(2 * raw_ndof + 1, raw_ndof + 1 + n) = C[local_eqn] / Count[local_eqn];
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(raw_ndof + 1 + n, raw_ndof) += djac_dparam(n, m) * Y_local[m];
          jacobian(raw_ndof + 1 + n, raw_ndof + 1 + m) = jacobian(n, m);
          jacobian(raw_ndof + 1 + n, m) = dJduPhiH(n, m);
        }
      }

      if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          unsigned eqn_i = elem_pt->eqn_number(i);
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            residuals[raw_ndof] += (*Problem_pt->global_dof_pt(eqn_i)) * symmetryA(i, j) * Psi_local[j];
            residuals[i] += Sigma * symmetryA(i, j) * Psi_local[j];
            jacobian(raw_ndof, i) += symmetryA(i, j) * Psi_local[j] + symmetryDADU_times_Psi(i, j) * (*Problem_pt->global_dof_pt(eqn_i));
            jacobian(i, j) += Sigma * symmetryDADU_times_Psi(i, j);
            jacobian(i, 2 * raw_ndof + 1) += symmetryA(i, j) * Psi_local[j];
          }
        }
      }
      else
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          unsigned local_eqn = elem_pt->eqn_number(i);
          residuals[i] += Sigma * Psi[local_eqn] / Count[local_eqn];
          jacobian(i, 2 * raw_ndof + 1) += Psi[local_eqn] / Count[local_eqn];
          residuals[raw_ndof] += ((*Problem_pt->global_dof_pt(local_eqn)) * Psi[local_eqn]) / Count[local_eqn];
          jacobian(raw_ndof, i) = Psi[local_eqn] / Count[local_eqn];
        }
      }
    }
    else
    {
      elem_pt->get_jacobian(residuals, jacobian);
      get_residuals(elem_pt,residuals); // The full residuals

      for (unsigned n = 0; n < raw_ndof; n++)
      {
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(raw_ndof + 1 + n, raw_ndof + 1 + m) = jacobian(n, m);
        }
        unsigned local_eqn = elem_pt->eqn_number(n);
        jacobian(2 * raw_ndof + 1, raw_ndof + 1 + n) = C[local_eqn] / Count[local_eqn];
      }

      if (Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes)
      {
        throw_runtime_error("Improved pitchfork tracking on nonsymmetric meshes only works with an analytically derived Hessian");
      }
      else
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          unsigned local_eqn = elem_pt->eqn_number(i);
          jacobian(i, 2 * raw_ndof + 1) = Psi[local_eqn] / Count[local_eqn];
          jacobian(raw_ndof, i) = Psi[local_eqn] / Count[local_eqn];
        }
      }
      const double FD_step = 1.0e-8;
      Vector<double> newres_p(augmented_ndof);

      for (unsigned n = 0; n < raw_ndof; ++n)
      {
        unsigned long global_eqn = Problem_pt->assembly_handler_pt()->eqn_number(elem_pt,n);
        double *unknown_pt = Problem_pt->global_dof_pt(global_eqn);
        double init = *unknown_pt;
        *unknown_pt += FD_step;
        newres_p.initialise(0.0);
        get_residuals(elem_pt, newres_p);
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(raw_ndof + 1 + m, n) = (newres_p[raw_ndof + 1 + m] - residuals[raw_ndof + 1 + m]) / (FD_step);
        }
        *unknown_pt = init;
        //		  Problem_pt->actions_before_newton_convergence_check();
      }

      {

        double *unknown_pt = Parameter_pt;
        double init = *unknown_pt;
        *unknown_pt += FD_step;
        newres_p.initialise(0.0);        
        get_residuals(elem_pt, newres_p);
        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(m, raw_ndof) = (newres_p[m] - residuals[m]) / FD_step;
        }
        for (unsigned m = raw_ndof + 1; m < augmented_ndof - 1; m++)
        {
          jacobian(m, raw_ndof) = (newres_p[m] - residuals[m]) / FD_step;
        }
        *unknown_pt = init;
        Problem_pt->actions_after_change_in_bifurcation_parameter();
      }
    }
  }

  void MyPitchForkHandler::get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam)
  {
    unsigned raw_ndof = elem_pt->ndof();
    DenseMatrix<double> djac_dparam(raw_ndof);
    elem_pt->get_djacobian_dparameter(parameter_pt, dres_dparam, djac_dparam);
    dres_dparam[raw_ndof] = 0.0;
    dres_dparam[2 * raw_ndof + 1] = 0.0;
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      dres_dparam[raw_ndof + 1 + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned local_unknown = elem_pt->eqn_number(j);
        dres_dparam[raw_ndof + 1 + i] +=
            djac_dparam(i, j) * Y[local_unknown];
      }
    }
  }
  void MyPitchForkHandler::get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam)
  {
    throw_runtime_error("implement");
  }
  void MyPitchForkHandler::get_hessian_vector_products(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product)
  {
    elem_pt->get_hessian_vector_products(Y, C, product);
  }
  void MyPitchForkHandler::get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction)
  {
    eigenfunction.resize(1);
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false);
    eigenfunction[0].build(&dist, 0.0);
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[0][n] = Y[n];
    }
  }
  void MyPitchForkHandler::solve_full_system()
  {
    // Is full system anyways.. Nothing to be done
  }

  void MyPitchForkHandler::setup_U_times_Psi_residual_indices()
  {
    pyoomph::Problem *prob = dynamic_cast<pyoomph::Problem *>(Problem_pt);
    if (!prob)
      throw_runtime_error("Not a pyoomph::Problem... Strange");
    auto codes = prob->get_bulk_element_codes();
    for (unsigned int i = 0; i < codes.size(); i++)
    {
      int orig_residual = codes[i]->get_func_table()->current_res_jac; // Store the initial residual (base state)
      int mass_matrix_residual = -1;
      if (codes[i]->_set_solved_residual("_simple_mass_matrix_of_defined_fields"))
      {
        mass_matrix_residual = codes[i]->get_func_table()->current_res_jac;
      }
      codes[i]->get_func_table()->current_res_jac = orig_residual; // Reset it
      residual_contribution_indices[codes[i]] = PitchForkResidualContributionList(codes[i], orig_residual, mass_matrix_residual);
    }
  }

  int MyPitchForkHandler::resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode)
  {
    pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el)
    {
      throw_runtime_error("Strange, not a pyoomph element");
    }
    auto *const_code = el->get_code_instance()->get_code();
    if (!residual_contribution_indices.count(const_code))
    {
      throw_runtime_error("You have not set up your residual contribution mapping in beforehand");
    }
    auto &entry = residual_contribution_indices[const_code];
    return entry.residual_indices[residual_mode];
  }

  bool MyPitchForkHandler::set_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode)
  {
    pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el)
    {
      throw_runtime_error("Strange, not a pyoomph element");
    }
    auto *const_code = el->get_code_instance()->get_code();
    if (!residual_contribution_indices.count(const_code))
    {
      throw_runtime_error("You have not set up your residual contribution mapping in beforehand");
    }
    auto &entry = residual_contribution_indices[const_code];
    // Setup the solved residual by the index (-1 means no contribution)
    entry.code->get_func_table()->current_res_jac = entry.residual_indices[residual_mode];
    return entry.residual_indices[residual_mode] >= 0;
  }


 


  // Constructors. We must pass a problem, a parameter to optimize (i.e. to change in order to get Re(eigenvalue)=0)
  // and a guess of the eigenvector
  AzimuthalSymmetryBreakingHandler::AzimuthalSymmetryBreakingHandler(Problem *const &problem_pt, double *const &parameter_pt,
                                                                     const oomph::DoubleVector &real_eigen, const oomph::DoubleVector &imag_eigen, const double &Omega_guess,bool has_imag)
      : Omega(Omega_guess), Parameter_pt(parameter_pt), has_imaginary_part(has_imag)
  {
    if (!has_imaginary_part) Omega=0.0;
    call_param_change_handler = false; // These must be false at the moment
    FD_step = 1e-8;                    // Default parameter for finite difference step
    Problem_pt = problem_pt;           // Store the problem
    // Set the number of non-augmented degrees of freedom
    Ndof = problem_pt->ndof();

    // Resize the vectors of additional dofs
    real_eigenvector.resize(Ndof,0);
    imag_eigenvector.resize(Ndof,0);
    normalization_vector.resize(Ndof,0);
    Count.resize(Ndof, 0);

    // Loop over all the elements in the problem
    unsigned n_element = problem_pt->mesh_pt()->nelement(); // Number of elements
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      // Loop over the local freedoms in an element
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        // Increase the associated global equation number counter
        ++Count[elem_pt->eqn_number(n)];
      }
    }


    for (unsigned n=0;n<Ndof;n++)
    {
      real_eigenvector[n]=real_eigen[n];
      imag_eigenvector[n]=imag_eigen[n];
    }
    rotate_complex_eigenvector_nicely(real_eigenvector,imag_eigenvector);
    //std::cout << "BIFTRACKER GOT " << (*parameter_pt) << " " << Omega << " HAS IMAG " << has_imaginary_part << std::endl;
    for (unsigned n=0;n<Ndof;n++) normalization_vector[n]=real_eigenvector[n];
    for (unsigned n=0;n<Ndof;n++)
    {
       problem_pt->GetDofPtr().push_back(&real_eigenvector[n]);
    }
    if (has_imaginary_part)
    {
      for (unsigned n=0;n<Ndof;n++)
      {
       problem_pt->GetDofPtr().push_back(&imag_eigenvector[n]);
      }
    }
    

    // Now add the parameter as degree of freedom
    problem_pt->GetDofPtr().push_back(parameter_pt);
    // Finally add the unknown imaginary part of the eigenvalue as degree of freedom
    if (has_imaginary_part) problem_pt->GetDofPtr().push_back(&Omega);

    // rebuild the Dof_distribution_pt
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), (has_imaginary_part ? Ndof *  3 + 2 : Ndof*2+1), false);
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    
  }

  // Destructor (used for cleaning up memory)
  AzimuthalSymmetryBreakingHandler::~AzimuthalSymmetryBreakingHandler()
  {
    // Now return the problem to its original size
    Problem_pt->GetDofPtr().resize(Ndof);
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                              Ndof, false);
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  // This will return the degrees of freedom of a single element of the augmented system
  // We will have to take the degrees of freedom of the original element and add a few more for the eigenvector values (Re and Im)
  unsigned AzimuthalSymmetryBreakingHandler::ndof(oomph::GeneralisedElement *const &elem_pt)
  {
    // This does not change if considering m contributions are incorporated already
    unsigned raw_ndof = elem_pt->ndof();
    {
      if (has_imaginary_part)
        return (3 * raw_ndof + 2);
      else
        return (2 * raw_ndof + 1);
    }
  }

  // This will cast the local equation number of an element to a global equation number.
  // Again, we have to consider the additional equations for the unknown eigenvector (Re and Im)
  unsigned long AzimuthalSymmetryBreakingHandler::eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local)
  {
    // Get the raw value
    unsigned raw_ndof = elem_pt->ndof();
    unsigned long global_eqn;
    if (ieqn_local < raw_ndof)
    {
      global_eqn = elem_pt->eqn_number(ieqn_local);
    }
    else if (ieqn_local < 2 * raw_ndof)
    {
      global_eqn = Ndof + elem_pt->eqn_number(ieqn_local - raw_ndof);
    }
    else if (has_imaginary_part)
    {
      if (ieqn_local < 3 * raw_ndof)
      {
        global_eqn = 2 * Ndof + elem_pt->eqn_number(ieqn_local - 2 * raw_ndof);
      }
      else if (ieqn_local == 3 * raw_ndof)
      {
        global_eqn = 3 * Ndof;
      }
      else
      {
        global_eqn = 3 * Ndof + 1;
      }
    }
    else
    {
      if (ieqn_local == 2 * raw_ndof)
      {
        global_eqn = 2 * Ndof;
      }      
    }
    return global_eqn;
  }

  // This will calculate the residual contribution of the original weak form by calling the function of the element
  void AzimuthalSymmetryBreakingHandler::get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals)
  {
    bool lambda_tracking=(Parameter_pt==Problem_pt->get_lambda_tracking_real());
    // Need to get raw residuals and jacobian
    unsigned raw_ndof = elem_pt->ndof();

    // Declare residuals, jacobian and mass matrix of real and imaginary contributions
    oomph::Vector<double> residuals_real(residuals.size(), 0);
    oomph::Vector<double> residuals_imag(residuals.size(), 0);
    DenseMatrix<double> jacobian_real(raw_ndof, raw_ndof, 0.0), M_real(raw_ndof, raw_ndof, 0.0);
    DenseMatrix<double> jacobian_imag(raw_ndof, raw_ndof, 0.0), M_imag(raw_ndof, raw_ndof, 0.0);

    // Get the base residuals, jacobian and mass matrix of real and imaginary parts
    set_assembled_residual(elem_pt, 1);
    elem_pt->get_jacobian_and_mass_matrix(residuals_real, jacobian_real, M_real);
    set_assembled_residual(elem_pt, 2);
    if (has_imaginary_part)
    {
      elem_pt->get_jacobian_and_mass_matrix(residuals_imag, jacobian_imag, M_imag);
    }
    set_assembled_residual(elem_pt, 0);
    elem_pt->get_residuals(residuals);

    // Initialise the pen-ultimate residual
    if (has_imaginary_part)
    {
      residuals[3 * raw_ndof] = -1.0 /(double)(Problem_pt->mesh_pt()->nelement())* eigenweight;
      residuals[3 * raw_ndof + 1] = 0.0;
    }
    else
    {
      residuals[2 * raw_ndof] = -1.0 /(double)(Problem_pt->mesh_pt()->nelement())* eigenweight;
    }

    // Now multiply to fill in the residuals
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      residuals[raw_ndof + i] = 0.0;
      if (has_imaginary_part) residuals[2 * raw_ndof + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned global_unknown = elem_pt->eqn_number(j);
        // First residual
        if (has_imaginary_part) 
        {
          residuals[raw_ndof + i] +=jacobian_real(i, j) * real_eigenvector[global_unknown] - jacobian_imag(i, j) * imag_eigenvector[global_unknown] - Omega * (M_real(i, j) * imag_eigenvector[global_unknown] + M_imag(i, j) * real_eigenvector[global_unknown]);        
          residuals[2 * raw_ndof + i] += jacobian_real(i, j) * imag_eigenvector[global_unknown] + jacobian_imag(i, j) * real_eigenvector[global_unknown] + Omega * (M_real(i, j) * real_eigenvector[global_unknown] + M_imag(i, j) * imag_eigenvector[global_unknown]);
        }
        else
        {
          residuals[raw_ndof + i] +=jacobian_real(i, j) * real_eigenvector[global_unknown];        
        }
      }
      // Get the global equation number
      unsigned global_eqn = elem_pt->eqn_number(i);

      if (has_imaginary_part) 
      {
        residuals[3 * raw_ndof] += (real_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
        // Imaginary eigenvector normalization
        residuals[3 * raw_ndof + 1] += (imag_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
      }
      else
      {
        residuals[2 * raw_ndof] += (real_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
      }
    }

    if (lambda_tracking)
    {
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          unsigned global_unknown = elem_pt->eqn_number(j);
          if (has_imaginary_part) 
          {
            residuals[raw_ndof + i] +=(*Parameter_pt) * (M_real(i, j) * real_eigenvector[global_unknown] - M_imag(i, j) * imag_eigenvector[global_unknown]);        
            residuals[2 * raw_ndof + i] += (*Parameter_pt) * (M_real(i, j) * imag_eigenvector[global_unknown] + M_imag(i, j) * real_eigenvector[global_unknown]);
          }
          else
          {
            residuals[raw_ndof + i] +=(*Parameter_pt) *M_real(i, j) * real_eigenvector[global_unknown];        
          }
        }
      }
    }

    //=======Correct residuals according to boundary conditions dependent on m=======//

    // Loop through the RAW dofs
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      // Get global equation number to assess whether a boundary condition applies to it
      unsigned long global_eqn = eqn_number(elem_pt, i);
      // Assess whether a boundary condition applies to dof
      if (base_dofs_forced_zero.count(global_eqn))
      {
        // Correct residual value
        residuals[i] = 0; // Base residual values **PATCH
      }
      if (eigen_dofs_forced_zero.count(global_eqn))
      {
        residuals[raw_ndof + i] = 0; // Eigenvector residual values **PATCH
        if (has_imaginary_part) residuals[2 * raw_ndof + i] = 0;
      }
    }
  }

  // Assembling the Jacobian matrix
  void AzimuthalSymmetryBreakingHandler::get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
    bool lambda_tracking=(Parameter_pt==Problem_pt->get_lambda_tracking_real());
    bool ana_dparam = lambda_tracking || Problem_pt->is_dparameter_calculated_analytically(Problem_pt->GetDofPtr()[3 * Ndof]);
    // Currently, we only calculate hessian analytically, if also ana_dparam is set (it is usually the case)
    bool ana_hessian = ana_dparam && Problem_pt->are_hessian_products_calculated_analytically() && dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);

    unsigned augmented_ndof = ndof(elem_pt);
    unsigned raw_ndof = elem_pt->ndof();

    // Declare residuals, jacobian and mass matrix of real and imaginary contributions
    oomph::Vector<double> residuals_real(residuals.size(), 0);
    oomph::Vector<double> residuals_imag(residuals.size(), 0);
    DenseMatrix<double> M(raw_ndof, raw_ndof, 0.0);
    DenseMatrix<double> jacobian_real(raw_ndof, raw_ndof, 0.0), M_real(raw_ndof, raw_ndof, 0.0);
    DenseMatrix<double> jacobian_imag(raw_ndof, raw_ndof, 0.0), M_imag(raw_ndof, raw_ndof, 0.0);

    if (!ana_hessian) // FD Hessian terms
    {
      if (lambda_tracking)
      {
   //     throw_runtime_error("Lambda tracking not implemented for finite difference Hessian");
      }
      // Get the base residuals, jacobian and mass matrix of real and imaginary parts
      set_assembled_residual(elem_pt, 1);
      elem_pt->get_jacobian_and_mass_matrix(residuals_real, jacobian_real, M_real);
      if (has_imaginary_part)
      {
        set_assembled_residual(elem_pt, 2);
        elem_pt->get_jacobian_and_mass_matrix(residuals_imag, jacobian_imag, M_imag);
      }
      set_assembled_residual(elem_pt, 0);
      elem_pt->get_jacobian_and_mass_matrix(residuals, jacobian, M);

      // Now fill in the actual residuals
      get_residuals(elem_pt, residuals);

      // Now the jacobian appears in other entries
      for (unsigned n = 0; n < raw_ndof; ++n)
      {
        for (unsigned m = 0; m < raw_ndof; ++m)
        {
          if (has_imaginary_part)
          {
            jacobian(raw_ndof + n, raw_ndof + m) = jacobian_real(n, m) - Omega * M_imag(n, m);
            jacobian(raw_ndof + n, 2 * raw_ndof + m) = -jacobian_imag(n, m) - Omega * M_real(n, m);
            jacobian(2 * raw_ndof + n, 2 * raw_ndof + m) = jacobian_real(n, m) - Omega * M_imag(n, m);
            jacobian(2 * raw_ndof + n, raw_ndof + m) = jacobian_imag(n, m) + Omega * M_real(n, m);
            unsigned global_eqn = elem_pt->eqn_number(m);
            jacobian(raw_ndof + n, 3 * raw_ndof + 1) += M_real(n, m) * imag_eigenvector[global_eqn] +
                                                        M_imag(n, m) * real_eigenvector[global_eqn];
            jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) -= M_real(n, m) * real_eigenvector[global_eqn] -
                                                            M_imag(n, m) * imag_eigenvector[global_eqn];
          }
          else
          {
            jacobian(raw_ndof + n, raw_ndof + m) = jacobian_real(n, m);
            unsigned global_eqn = elem_pt->eqn_number(m);
          }
        }

        unsigned local_eqn = elem_pt->eqn_number(n);
        if (has_imaginary_part)
        {
          jacobian(3 * raw_ndof, raw_ndof + n) = normalization_vector[local_eqn] / Count[local_eqn];
          jacobian(3 * raw_ndof + 1, 2 * raw_ndof + n) = normalization_vector[local_eqn] / Count[local_eqn];
        }
        else
        {
          jacobian(2 * raw_ndof, raw_ndof + n) = normalization_vector[local_eqn] / Count[local_eqn];
        }
      }

      const double FD_step = 1.0e-8;

      Vector<double> newres_p(augmented_ndof), newres_m(augmented_ndof);

      // Loop over the dofs
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        // Just do the x's
        unsigned long global_eqn = eqn_number(elem_pt, n);
        double *unknown_pt = Problem_pt->GetDofPtr()[global_eqn];
        double init = *unknown_pt;
        *unknown_pt += FD_step;

        // Get the new residuals
        get_residuals(elem_pt, newres_p);

        // Reset
        *unknown_pt = init;

        for (unsigned m = 0; m < raw_ndof; m++)
        {
          jacobian(raw_ndof + m, n) =
              (newres_p[raw_ndof + m] - residuals[raw_ndof + m]) / (FD_step);
          if (has_imaginary_part)
            jacobian(2 * raw_ndof + m, n) =
              (newres_p[2 * raw_ndof + m] - residuals[2 * raw_ndof + m]) /
              (FD_step);
        }
        // Reset the unknown
        *unknown_pt = init;
      }

      // Now do the global parameter
      // Either calculate the parameter derivatives analitically (ana_dparam=true) or calculate by finite difference.
      if (ana_dparam)
      {
        Vector<double> dres_dparam(augmented_ndof, 0.0);        
        this->get_dresiduals_dparameter(elem_pt, Problem_pt->GetDofPtr()[(has_imaginary_part ? 3 : 2) * Ndof], dres_dparam);
        for (unsigned m = 0; m < augmented_ndof - (has_imaginary_part ? 2 : 1); m++)
        {
          jacobian(m, (has_imaginary_part ? 3 : 2) * raw_ndof) = dres_dparam[m];
        }
      }
      else
      {
        double *unknown_pt = Problem_pt->GetDofPtr()[(has_imaginary_part ? 3 : 2) * Ndof];
        double init = *unknown_pt;
        *unknown_pt += FD_step;

        Problem_pt->actions_after_change_in_bifurcation_parameter();
        // Get the new residuals
        get_residuals(elem_pt, newres_p);

        // Reset
        *unknown_pt = init;

        // Subtract
        *unknown_pt -= FD_step;
        get_residuals(elem_pt, newres_m);

        for (unsigned m = 0; m < augmented_ndof - (has_imaginary_part ? 2 : 1); m++)
        {
          jacobian(m, (has_imaginary_part ? 3 : 2) * raw_ndof) = (newres_p[m] - residuals[m]) / FD_step;
        }
        // Reset the unknown
        *unknown_pt = init;
        Problem_pt->actions_after_change_in_bifurcation_parameter();
      }
    }
    else // Analytic Hessian version
    {

      // Cast to a pyoomph element, which has a function to calc Hessian*Vector of M and J simultaneosly. Pure oomph-lib does not have this
      pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);

      // Fill the real and imag eigenvector in local (elemental) indices
      // Vector<double> Yl(raw_ndof), Zl(raw_ndof);
      Vector<double> Eig((has_imaginary_part ? 2 : 1) * raw_ndof);
      for (unsigned _e = 0; _e < raw_ndof; _e++)
      {
        unsigned index = elem_pt->eqn_number(_e);
        Eig[_e] = real_eigenvector[index];
        if (has_imaginary_part) Eig[raw_ndof + _e] = imag_eigenvector[index];
      }

      DenseMatrix<double> JHess_real((has_imaginary_part ? 2 : 1) * raw_ndof, raw_ndof, 0.0), MHess_real((has_imaginary_part ? 2 : 1) * raw_ndof, raw_ndof, 0.0);
      DenseMatrix<double> JHess_imag((has_imaginary_part ? 2 : 1) * raw_ndof, raw_ndof, 0.0), MHess_imag((has_imaginary_part ? 2 : 1) * raw_ndof, raw_ndof, 0.0);

      DenseMatrix<double> dJ_real_dparam(raw_ndof, raw_ndof, 0.0), dJ_imag_dparam(raw_ndof, raw_ndof, 0.0);
      DenseMatrix<double> dM_real_dparam(raw_ndof, raw_ndof, 0.0), dM_imag_dparam(raw_ndof, raw_ndof, 0.0);
      Vector<double> dres_real_dparam(raw_ndof, 0.0), dres_imag_dparam(raw_ndof, 0.0);
      Vector<double> dres_dparam(raw_ndof, 0.0);

      residuals.initialise(0.0);
      jacobian.initialise(0.0);

      std::vector<SinglePassMultiAssembleInfo> assemble_info;
      int resindex;
      if ((resindex = this->resolve_assembled_residual(pyoomph_elem_pt, 0)) >= 0)
      {
        assemble_info.push_back(SinglePassMultiAssembleInfo(resindex, &residuals, &jacobian));
        if (!lambda_tracking) assemble_info.back().add_param_deriv(Parameter_pt, &dres_dparam);
      }
      if ((resindex = this->resolve_assembled_residual(pyoomph_elem_pt, 1)) >= 0)
      {
        assemble_info.push_back(SinglePassMultiAssembleInfo(resindex, &residuals_real, &jacobian_real, &M_real));
        assemble_info.back().add_hessian(Eig, &JHess_real, &MHess_real);
        if (!lambda_tracking) assemble_info.back().add_param_deriv(Parameter_pt, &dres_real_dparam, &dJ_real_dparam, &dM_real_dparam);
      }
      if (has_imaginary_part)
      {
        if ((resindex = this->resolve_assembled_residual(pyoomph_elem_pt, 2)) >= 0)
        {
          assemble_info.push_back(SinglePassMultiAssembleInfo(resindex, &residuals_imag, &jacobian_imag, &M_imag));
          assemble_info.back().add_hessian(Eig, &JHess_imag, &MHess_imag);
          if (!lambda_tracking) assemble_info.back().add_param_deriv(Parameter_pt, &dres_imag_dparam, &dJ_imag_dparam, &dM_imag_dparam);
        }
      }

      pyoomph_elem_pt->get_multi_assembly(assemble_info);

      // We will fill in the augmented residual vector once more by hand here. Otherwise, it we would call this->get_residual, we would assemble several matrices multiple times
      // Now multiply to fill in the residuals
      residuals[(has_imaginary_part ? 3 : 2) * raw_ndof] = -1.0 / (double)(Problem_pt->mesh_pt()->nelement())* eigenweight;
      if (has_imaginary_part) residuals[3 * raw_ndof + 1] = 0.0;
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        residuals[raw_ndof + i] = 0.0;
        if (has_imaginary_part) residuals[2 * raw_ndof + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          if (has_imaginary_part) 
          {
            residuals[raw_ndof + i] += jacobian_real(i, j) * Eig[j] - jacobian_imag(i, j) * Eig[raw_ndof + j] - Omega * (M_real(i, j) * Eig[raw_ndof + j] + M_imag(i, j) * Eig[j]);
            residuals[2 * raw_ndof + i] += jacobian_real(i, j) * Eig[raw_ndof + j] + jacobian_imag(i, j) * Eig[j] + Omega * (M_real(i, j) * Eig[j] + M_imag(i, j) * Eig[raw_ndof + j]);
          }
          else
          {
            residuals[raw_ndof + i] += jacobian_real(i, j) * Eig[j] ;
          }
        }
        unsigned global_eqn = elem_pt->eqn_number(i);
        residuals[(has_imaginary_part ? 3 : 2) * raw_ndof] += (real_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
        if (has_imaginary_part) residuals[3 * raw_ndof + 1] += (imag_eigenvector[global_eqn] * normalization_vector[global_eqn]) / Count[global_eqn];
      }

      for (unsigned m = 0; m < raw_ndof; m++)
      {
        // First 'row' of the Jacobian:
        // raw Jacobian is already filled by elem_pt->get_jacobian()
        // parameter derivative of the residual
        jacobian(m, (has_imaginary_part ? 3 : 2) * raw_ndof) = dres_dparam[m];

        // Second and third 'row' of the Jacobian: a lot of Hessian terms!
        jacobian(raw_ndof + m, (has_imaginary_part ? 3 : 2) * raw_ndof) = 0.0;
        if (has_imaginary_part)
        {
          jacobian(raw_ndof + m, 3 * raw_ndof + 1) = 0.0;
          jacobian(2 * raw_ndof + m, 3 * raw_ndof) = 0.0;
          jacobian(2 * raw_ndof + m, 3 * raw_ndof + 1) = 0.0;
        }
        for (unsigned int n = 0; n < raw_ndof; n++)
        {
          if (has_imaginary_part)
          {
            jacobian(raw_ndof + m, n) = JHess_real(m, n) - JHess_imag(raw_ndof + m, n) - Omega * (MHess_real(raw_ndof + m, n) + MHess_imag(m, n));
            jacobian(raw_ndof + m, raw_ndof + n) = jacobian_real(m, n) - Omega * M_imag(m, n);
            jacobian(raw_ndof + m, 2 * raw_ndof + n) = -jacobian_imag(m, n) - Omega * M_real(m, n);
            jacobian(raw_ndof + m, 3 * raw_ndof) += dJ_real_dparam(m, n) * Eig[n] - dJ_imag_dparam(m, n) * Eig[raw_ndof + n] - Omega * (dM_real_dparam(m, n) * Eig[raw_ndof + n] + dM_imag_dparam(m, n) * Eig[n]);            
            jacobian(raw_ndof + m, 3 * raw_ndof + 1) -= M_real(m, n) * Eig[raw_ndof + n] + M_imag(m, n) * Eig[n];

            jacobian(2 * raw_ndof + m, n) = JHess_real(raw_ndof + m, n) + JHess_imag(m, n) + Omega * (MHess_real(m, n) + MHess_imag(raw_ndof + m, n));
            jacobian(2 * raw_ndof + m, raw_ndof + n) = jacobian_imag(m, n) + Omega * M_real(m, n);
            jacobian(2 * raw_ndof + m, 2 * raw_ndof + n) = jacobian_real(m, n) - Omega * M_imag(m, n);
            jacobian(2 * raw_ndof + m, 3 * raw_ndof) += dJ_real_dparam(m, n) * Eig[raw_ndof + n] + dJ_imag_dparam(m, n) * Eig[n] + Omega * (dM_real_dparam(m, n) * Eig[n] + dM_imag_dparam(m, n) * Eig[raw_ndof + n]);
            jacobian(2 * raw_ndof + m, 3 * raw_ndof + 1) += M_real(m, n) * Eig[n] + M_imag(m, n) * Eig[raw_ndof + n];            
          }
          else
          {
            jacobian(raw_ndof + m, n) = JHess_real(m, n);
            jacobian(raw_ndof + m, raw_ndof + n) = jacobian_real(m, n);
            jacobian(raw_ndof + m, 2 * raw_ndof) += dJ_real_dparam(m, n) * Eig[n];
          }
        }
        unsigned local_eqn = elem_pt->eqn_number(m);
        jacobian((has_imaginary_part ? 3 : 2) * raw_ndof, raw_ndof + m) = normalization_vector[local_eqn] / Count[local_eqn];
        if (has_imaginary_part)   jacobian(3 * raw_ndof + 1, 2 * raw_ndof + m) = normalization_vector[local_eqn] / Count[local_eqn];
      }


      if (lambda_tracking)
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            unsigned global_unknown = elem_pt->eqn_number(j);
            if (has_imaginary_part) 
            {
              residuals[raw_ndof + i] +=(*Parameter_pt) * (M_real(i, j) * real_eigenvector[global_unknown] - M_imag(i, j) * imag_eigenvector[global_unknown]);        
              residuals[2 * raw_ndof + i] += (*Parameter_pt) * (M_real(i, j) * imag_eigenvector[global_unknown] + M_imag(i, j) * real_eigenvector[global_unknown]);
              //jacobian(raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i, j) * real_eigenvector[global_unknown] - MHess_imag(i, j) * imag_eigenvector[global_unknown]);        
              //jacobian(2*raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i, j) * imag_eigenvector[global_unknown] + MHess_imag(i, j) * real_eigenvector[global_unknown]);        
              jacobian(raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i, j)  - MHess_imag(i+raw_ndof, j) );        
              jacobian(2*raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i+raw_ndof, j) + MHess_imag(i, j));                    
              jacobian(raw_ndof + i,raw_ndof + j)+=(*Parameter_pt)*M_real(i, j);
              jacobian(raw_ndof + i,2*raw_ndof + j)+=-(*Parameter_pt)*M_imag(i, j);
              jacobian(2*raw_ndof + i,raw_ndof + j)+=(*Parameter_pt)*M_imag(i, j);
              jacobian(2*raw_ndof + i,2*raw_ndof + j)+=(*Parameter_pt)*M_real(i, j);
              jacobian(raw_ndof + i, 3 * raw_ndof)+=(M_real(i, j) * real_eigenvector[global_unknown] - M_imag(i, j) * imag_eigenvector[global_unknown]);
              jacobian(2*raw_ndof + i, 3 * raw_ndof)+=(M_real(i, j) * imag_eigenvector[global_unknown] + M_imag(i, j) * real_eigenvector[global_unknown]);
            }
            else
            {
              residuals[raw_ndof + i] +=(*Parameter_pt) *M_real(i, j) * real_eigenvector[global_unknown];        
              jacobian(raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i, j) );        
              jacobian(raw_ndof + i,raw_ndof + j)+=(*Parameter_pt)*M_real(i, j);
              jacobian(raw_ndof + i, 2 * raw_ndof)+=M_real(i, j) * real_eigenvector[global_unknown];        
            }
          }
        }
      }
    }

    // Get total number of dofs in the element
    unsigned ndof = elem_pt->ndof();
    // Loop through the dofs
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      // Get global equation number to assess whether a boundary condition applies to it
      unsigned long global_eqn = eqn_number(elem_pt, i);
      // Assess whether a boundary condition applies to dof
      if (base_dofs_forced_zero.count(global_eqn))
      {
        // Correct jacobian value
        for (unsigned j = 0; j < augmented_ndof; j++)
        {
          jacobian(i, j) = 0.0;
        }
        jacobian(i, i) = 1.0;
        if (ana_hessian)
        {
          residuals[i] = 0.0;
        }
      }
      if (eigen_dofs_forced_zero.count(global_eqn))
      {
        // Correct jacobian value
        for (unsigned j = 0; j < augmented_ndof; j++)
        {
          jacobian(raw_ndof + i, j) = 0.0;
          if (has_imaginary_part) jacobian(2 * raw_ndof + i, j) = 0.0;
        }
        jacobian(raw_ndof + i, raw_ndof + i) = 1.0;
        if (has_imaginary_part) jacobian(2 * raw_ndof + i, 2 * raw_ndof + i) = 1.0;
        if (ana_hessian)
        {
          residuals[raw_ndof + i] =0.0;
          if (has_imaginary_part)  residuals[2 * raw_ndof + i] = 0.0;
        }
      }
    }

    // DEBUG ANA    
    /*
    if (true)
    {
      oomph::DenseMatrix<double> J_FD(augmented_ndof,augmented_ndof,0.0);
      for (unsigned i = 0; i < augmented_ndof; i++)
      {
          
          //std::cout << " GETTTING DOF " << eqn_number(elem_pt,i)<< std::endl <<std::flush;
          unsigned global_eqn=eqn_number(elem_pt,i);        
          double *unknown_pt = Problem_pt->GetDofPtr()[global_eqn];
          double FD_step=1e-8;
          double init = *unknown_pt;
          *unknown_pt += FD_step;

          oomph::Vector<double> newres_p(augmented_ndof,0.0);
          get_residuals(elem_pt, newres_p);
          // Reset
          *unknown_pt = init;
          for (unsigned m = 0; m < augmented_ndof ; m++)
          {
            J_FD(m, i) = (newres_p[m] - residuals[m]) / FD_step;
          }              
      }

          for (unsigned i = 0; i < raw_ndof; i++)
          {
            // Get global equation number to assess whether a boundary condition applies to it
            unsigned long global_eqn = eqn_number(elem_pt, i);
            // Assess whether a boundary condition applies to dof
            if (base_dofs_forced_zero.count(global_eqn))
            {
              // Correct jacobian value
              for (unsigned j = 0; j < augmented_ndof; j++)
              {
                J_FD(i, j) = 0.0;
              }
              J_FD(i, i) = 1.0;
            }
            if (eigen_dofs_forced_zero.count(global_eqn))
            {
              // Correct jacobian value
              for (unsigned j = 0; j < augmented_ndof; j++)
              {
                J_FD(raw_ndof + i, j) = 0.0;
                if (has_imaginary_part) J_FD(2 * raw_ndof + i, j) = 0.0;
              }
              J_FD(raw_ndof + i, raw_ndof + i) = 1.0;
              if (has_imaginary_part) J_FD(2 * raw_ndof + i, 2 * raw_ndof + i) = 1.0;
            }
          }

          for (unsigned i = 0; i < augmented_ndof ; i++)
          {
            unsigned long global_eqn = eqn_number(elem_pt, i);
          for (unsigned m = 0; m < augmented_ndof ; m++)
          {
            double delta=jacobian(m, i)-J_FD(m,i);
            if (std::fabs(delta)>0.001)
            {
              std::cout << "DIFFERENCE " << (has_imaginary_part ? "WITH IMAGINARY PART" : "WITHOUT IMAGINARY PART") << " IN " << m << " " << i << ": " << delta << " ANA " << jacobian(m,i) << " FD " << J_FD(m,i) << " NDOF " << augmented_ndof << " NRAWDOF " << raw_ndof << " MPIN0 " << base_dofs_forced_zero.count(eqn_number(elem_pt,m)) << " " << eigen_dofs_forced_zero.count(eqn_number(elem_pt,m)) << " IPIN0 "  << base_dofs_forced_zero.count(global_eqn) << " " << eigen_dofs_forced_zero.count(global_eqn) << " GLOBM " << eqn_number(elem_pt,m) << " GLOBI " << global_eqn << std::endl;
            }
          }
          }              
    }
          */
    

    // DEBUG ANA
    /*
     if (ana_hessian)
     {
       oomph::Vector<double> fd_residuals(residuals.size(),0.0);
       oomph::DenseMatrix<double> fd_jacobian(residuals.size(),residuals.size(),0.0);
       Problem_pt->unset_analytic_hessian_products();
       this->get_jacobian(elem_pt,fd_residuals,fd_jacobian);
       Problem_pt->set_analytic_hessian_products();
       double delta=1e-8;
       std::string elem_info=dynamic_cast<pyoomph::BulkElementBase*>(elem_pt)->get_code_instance()->get_code()->get_file_name();
       for (unsigned int i=0;i<fd_residuals.size();i++)
       {
         std::string iwhat=(i<raw_ndof ? "raw" : (i<2*raw_ndof ? "Y" : (i<3*raw_ndof ? "Z" : (i==3*raw_ndof  ? "Param"  : "Omega" ) )));
         if (std::fabs(residuals[i]-fd_residuals[i])>delta)
         {
           std::cout << "ERROR in R : " << i << " of " << residuals.size() << " : " << residuals[i] << "  " << fd_residuals[i] << "   delta " << std::fabs(residuals[i]-fd_residuals[i]) << " in " << elem_info << std::endl;
         }
         for (unsigned int j=0;j<fd_residuals.size();j++)
         {
         std::string jwhat=(j<raw_ndof ? "raw" : (j<2*raw_ndof ? "Y" : (j<3*raw_ndof ? "Z" : (j==3*raw_ndof  ? "Param"  : "Omega" ) )));
          if (std::fabs(jacobian(i,j)-fd_jacobian(i,j))>delta)
          {
           std::cout << "ERROR in J : " << i << " , " << j << " of " << residuals.size() << " : " << jacobian(i,j) << "  " << fd_jacobian(i,j) << "   delta " << std::fabs(jacobian(i,j)-fd_jacobian(i,j))<<  " in " << elem_info  << " @ Deriv of " << iwhat << " wrto " << jwhat << std::endl;
          }
         }

       }
     }
     */
  }

  // Derivative of the augmented residuals with respect to a parameter

  void AzimuthalSymmetryBreakingHandler::get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt,
                                                                   oomph::Vector<double> &dres_dparam)
  {
    
    bool lambda_tracking=(Parameter_pt==Problem_pt->get_lambda_tracking_real());
    //if (parameter_pt==Parameter_pt) throw_runtime_error("Strange that this function is called with respect to the same parameter");
    // Need to get raw residuals and jacobian
    unsigned raw_ndof = elem_pt->ndof();
    //    if (parameter_pt!=Parameter_pt)   std::cout << "PARAM DERIV " << parameter_pt << " " <<  Parameter_pt << std::endl;

    // Declare residuals, jacobian and mass matrix of real and imaginary contributions
    oomph::Vector<double> dres_real_dparam(raw_ndof, 0);
    oomph::Vector<double> dres_imag_dparam(raw_ndof, 0);
    DenseMatrix<double> djac_dparam(raw_ndof), dM_dparam(raw_ndof);
    DenseMatrix<double> djac_real_dparam(raw_ndof), dM_real_dparam(raw_ndof);
    DenseMatrix<double> djac_imag_dparam(raw_ndof), dM_imag_dparam(raw_ndof);

    // Get the dresiduals, djacobian and dmass_matrix for base, real and imaginary jacobians
    set_assembled_residual(elem_pt, 1);
    if (lambda_tracking && parameter_pt==Parameter_pt)
    {
      elem_pt->get_jacobian_and_mass_matrix(dres_real_dparam, djac_real_dparam, dM_real_dparam);
    }
    else
    {
      elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt, dres_real_dparam, djac_real_dparam, dM_real_dparam);
    }
    if (has_imaginary_part)
    {
      set_assembled_residual(elem_pt, 2);
      if (lambda_tracking && parameter_pt==Parameter_pt)
      {
        elem_pt->get_jacobian_and_mass_matrix(dres_imag_dparam, djac_imag_dparam, dM_imag_dparam);        
      }
      else
      {
        elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt, dres_imag_dparam, djac_imag_dparam, dM_imag_dparam);
      }
    }
    set_assembled_residual(elem_pt, 0);
    if (lambda_tracking && parameter_pt==Parameter_pt)
    {
    }
    else
    {
      elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt, dres_dparam, djac_real_dparam, dM_real_dparam);
    }

    // Initialise the pen-ultimate residual, which does not
    // depend on the parameter
    dres_dparam[(has_imaginary_part ? 3 : 2) * raw_ndof] = 0.0;
    if (has_imaginary_part)  dres_dparam[3 * raw_ndof + 1] = 0.0;

    // Now multiply to fill in the residuals
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      dres_dparam[raw_ndof + i] = 0.0;
      if (has_imaginary_part) dres_dparam[2 * raw_ndof + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned global_unknown = elem_pt->eqn_number(j);
        // Real part
        if (has_imaginary_part)
        {
          if (lambda_tracking && parameter_pt==Parameter_pt)
          {
              dres_dparam[raw_ndof + i] += (dM_real_dparam(i, j) * real_eigenvector[global_unknown] - dM_imag_dparam(i, j) * imag_eigenvector[global_unknown]);        
              dres_dparam[2 * raw_ndof + i] += (dM_real_dparam(i, j) * imag_eigenvector[global_unknown] + dM_imag_dparam(i, j) * real_eigenvector[global_unknown]);    
          }
          else
          {
            dres_dparam[raw_ndof + i] +=
                djac_real_dparam(i, j) * real_eigenvector[global_unknown] - djac_imag_dparam(i, j) * imag_eigenvector[global_unknown] -
                Omega * (dM_real_dparam(i, j) * imag_eigenvector[global_unknown] + dM_imag_dparam(i, j) * real_eigenvector[global_unknown]);
            // Imaginary part
            dres_dparam[2 * raw_ndof + i] +=
                djac_real_dparam(i, j) * imag_eigenvector[global_unknown] + djac_imag_dparam(i, j) * real_eigenvector[global_unknown] +                
                Omega * (dM_real_dparam(i, j) * real_eigenvector[global_unknown] - dM_imag_dparam(i, j) * imag_eigenvector[global_unknown]);
            if (lambda_tracking)
            {
              dres_dparam[raw_ndof + i] += (*Parameter_pt) * (dM_real_dparam(i, j) * real_eigenvector[global_unknown] - dM_imag_dparam(i, j) * imag_eigenvector[global_unknown]);        
              dres_dparam[2 * raw_ndof + i] += (*Parameter_pt) * (dM_real_dparam(i, j) * imag_eigenvector[global_unknown] + dM_imag_dparam(i, j) * real_eigenvector[global_unknown]);    
            }
          }
                    
        }
        else
        {
          if (lambda_tracking && parameter_pt==Parameter_pt)
          {
            dres_dparam[raw_ndof + i] += dM_real_dparam(i, j) * real_eigenvector[global_unknown];
          }
          else
          {
            dres_dparam[raw_ndof + i] +=djac_real_dparam(i, j) * real_eigenvector[global_unknown];
            if (lambda_tracking)
            {
              dres_dparam[raw_ndof + i] += (*Parameter_pt) * dM_real_dparam(i, j) * real_eigenvector[global_unknown];
            }
          }
        }
      }
    }
  }

  // Derivative of the augmented Jacobian with respect to the parameter
  void AzimuthalSymmetryBreakingHandler::get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam)
  {
    throw_runtime_error("AzimuthalSymmetryBreakingHandler::get_djacobian_dparameter(oomph::GeneralisedElement* const &elem_pt,double* const &parameter_pt,oomph::Vector<double> &dres_dparam,oomph::DenseMatrix<double> &djac_dparam)");
    // TODO: Fill it
    // Is it required?
  }

  // Get the eigenfunction
  void AzimuthalSymmetryBreakingHandler::get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction)
  {
    // There is a real and imaginary part of the eigen vector
    eigenfunction.resize((has_imaginary_part ? 2 : 1)); // So we must return two real vectors (Re and Im)
    // build a distribution for the storage of the eigenvector parts
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false); // The eigenvectors have Ndof entries, i.e. the number of dofs of the original problem
    // Rebuild the vector
    eigenfunction[0].build(&dist, 0.0);
    if (has_imaginary_part) eigenfunction[1].build(&dist, 0.0);
    // Set the value to be the null vector
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[0][n] = real_eigenvector[n];
      if (has_imaginary_part) eigenfunction[1][n] = imag_eigenvector[n];
    }
  }

   void AzimuthalSymmetryBreakingHandler::set_eigenweight(double ew)
  {
    for (unsigned n = 0; n < Ndof; n++)
    {
      real_eigenvector[n] *= ew / eigenweight;
      imag_eigenvector[n] *= ew / eigenweight;
    }
    eigenweight = ew;
  }

  // Pyoomph has different residual contributions. The original residual along with its jacobian and the real and imag part of the azimuthal Jacobian and mass matrix. We get the indices of these contributions in beforehand.
  // We assume that all codes are initially set to the stage so that the original axisymmetric residual is solved
  void AzimuthalSymmetryBreakingHandler::setup_solved_azimuthal_contributions(std::string real_angular_J_and_M, std::string imag_angular_J_and_M)
  {
    pyoomph::Problem *prob = dynamic_cast<pyoomph::Problem *>(Problem_pt);
    if (!prob)
      throw_runtime_error("Not a pyoomph::Problem... Strange");
    // Each generated code may have different indices (i.e. not all contributions are present on each generated code)
    // Therefore, we must make a map from each generated code to the three residuals/jacobians/etc
    auto codes = prob->get_bulk_element_codes();
    for (unsigned int i = 0; i < codes.size(); i++)
    {
      int orig_residual = codes[i]->get_func_table()->current_res_jac; // Store the initial residual (base state)
      int real_azimuthal = -1;                                         // By default, no azimuthal residual present
      int imag_azimuthal = -1;
      if (codes[i]->_set_solved_residual(real_angular_J_and_M))
      {
        real_azimuthal = codes[i]->get_func_table()->current_res_jac; // Get the real residual index
      }
      if (codes[i]->_set_solved_residual(imag_angular_J_and_M))
      {
        imag_azimuthal = codes[i]->get_func_table()->current_res_jac; // Get he imaginary residual index
      }
      codes[i]->get_func_table()->current_res_jac = orig_residual; // Reset it

      // And store it in the mapping
      //std::cout << "MAPPING " << codes[i]->get_file_name() << " " << orig_residual << " " << real_azimuthal << " " << imag_azimuthal << std::endl;
      residual_contribution_indices[codes[i]] = AzimuthalSymmetryBreakingResidualContributionList(codes[i], orig_residual, real_azimuthal, imag_azimuthal);
    }
  }

  // Please reset it to the base state at the end via
  // set_assembled_residual(element,0)
  int AzimuthalSymmetryBreakingHandler::resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode)
  {
    pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el)
    {
      throw_runtime_error("Strange, not a pyoomph element");
    }
    auto *const_code = el->get_code_instance()->get_code();
    if (!residual_contribution_indices.count(const_code))
    {
      throw_runtime_error("You have not set up your residual contribution mapping in beforehand");
    }
    auto &entry = residual_contribution_indices[const_code];
    return entry.residual_indices[residual_mode];
  }

  bool AzimuthalSymmetryBreakingHandler::set_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode)
  {
    pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el)
    {
      throw_runtime_error("Strange, not a pyoomph element");
    }
    auto *const_code = el->get_code_instance()->get_code();
    if (!residual_contribution_indices.count(const_code))
    {
      throw_runtime_error("You have not set up your residual contribution mapping in beforehand");
    }
    auto &entry = residual_contribution_indices[const_code];
    // Setup the solved residual by the index (-1 means no contribution)
    entry.code->get_func_table()->current_res_jac = entry.residual_indices[residual_mode];
    return entry.residual_indices[residual_mode] >= 0;
  }



  class TimeNode : public oomph::Node
  {
    protected:
      unsigned index;
    public:
      TimeNode(double s,unsigned _index) : oomph::Node(1,1,1), index(_index) {this->x(0)=s;}
      unsigned get_index() {return index;}
  };

  

  


  //////// PERIODIC ORBIT TRACKER

  PeriodicOrbitHandler::PeriodicOrbitHandler(Problem *const &problem_pt, const double &period, const std::vector<std::vector<double>> &tadd, int bspline_order, int gl_order, std::vector<double> knots,unsigned T_constraint) : Problem_pt(problem_pt), T(period), T_constraint_mode(T_constraint), time_mesh(NULL), collocation_gl(NULL)
  {
    Ndof = problem_pt->ndof();    
    n_element = problem_pt->mesh_pt()->nelement();
    Tadd=tadd;    
    x0.resize(Ndof);
    n0.resize(Ndof);
    double nlength=0;
    if (T_constraint_mode>1) throw_runtime_error("T_constraint_mode must be 0 or 1");
    for (unsigned int i=0;i<Ndof;i++)
    {
        x0[i]=*(problem_pt->GetDofPtr()[i]); // Store the x0 for the  plane equation
        n0[i]=Tadd.back()[i]-Tadd.front()[i]; // Store the normal vector for the plane equation
        nlength+=n0[i]*n0[i];
    }
    nlength=std::sqrt(nlength);
    for (unsigned int i=0;i<Ndof;i++)    {    n0[i]/=nlength;    }
    d_plane=0;
    for (unsigned int i=0;i<Ndof;i++)    d_plane+=n0[i]*x0[i]; // Distance of the plane to the origin


    Count.resize(Ndof, 0);

    // Loop over all the elements in the problem
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = problem_pt->mesh_pt()->element_pt(e);
      // Loop over the local freedoms in an element
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        // Increase the associated global equation number counter
        ++Count[elem_pt->eqn_number(n)];
      }
    }    

    // Floquet mode: We explicitly store the 0th dofs at the last step 
    if (bspline_order==0 || bspline_order<-2 ) ////TODO: Set the floquet mode here also for time mesh mode
    {
      floquet_mode=true;
      Tadd.push_back(std::vector<double>(x0));
    }
    else floquet_mode=false;


    for (unsigned int ti=0;ti<Tadd.size();ti++)
    {      
      if (Tadd[ti].size()!=Ndof) throw_runtime_error("The size of the additional time vector must be the same as the number of dofs at index "+std::to_string(ti));
      for (unsigned int i=0;i<Ndof;i++)
      {
        problem_pt->GetDofPtr().push_back(&Tadd[ti][i]);    
      }
    }
    
    
    problem_pt->GetDofPtr().push_back(&T); 
    T_global_eqn=problem_pt->GetDofPtr().size()-1;
    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * (Tadd.size()+1)+1 , true); 
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);

    
    
    if (knots.empty())
    {
      knots.resize(Tadd.size()+(floquet_mode ? 1: 2));
      for (unsigned int i=0;i<knots.size();i++)
      {
        knots[i]=i/(knots.size()-1.0);
      }
    }
    else
    {
      if (knots.size()!=Tadd.size()+2) throw_runtime_error("The number of knots must be the same as the number of time steps");
      if (std::fabs(knots.front())>1e-10 || std::fabs(knots.back())-1>1e-10) throw_runtime_error("The first and last knot must be 0 and 1");
    }
    if (bspline_order>=1)
    {
      this->basis = new PeriodicBSplineBasis(knots, bspline_order,gl_order);
    }
    this->s_knots=knots;
    this->s_knots.front()=0.0;
    this->s_knots.back()=1.0;

    /// Setup the finite difference information here
    if (bspline_order<=0 && bspline_order>=-2)
    {
      this->FD_ds_order=1; 
      this->FD_ds_weights.resize(this->s_knots.size()-1);
      this->FD_ds_inds.resize(this->s_knots.size()-1);      
      for (unsigned int i=0;i<this->FD_ds_weights.size();i++)
      {
        if (bspline_order==-1)
        {
          // Central difference first order accurate
          this->FD_ds_weights[i].resize(2);
          this->FD_ds_inds[i].resize(2);
          double si=get_knot_value(i);
          double sip1=get_knot_value(i+1);
          double sim1=get_knot_value(i-1);
          this->FD_ds_weights[i][0]=-1.0/(sip1-sim1);
          this->FD_ds_weights[i][1]=1.0/(sip1-sim1);
          this->FD_ds_inds[i][0]=(i>0 ? i-1 : this->s_knots.size()-2);
          this->FD_ds_inds[i][1]=(i+1)%(this->s_knots.size()-1);
        }
        else if (bspline_order==-2)
        {
          // Backward difference second order
          this->FD_ds_weights[i].resize(3);
          this->FD_ds_inds[i].resize(3);
          double si=get_knot_value(i);
          double sim1=get_knot_value(i-1);
          double sim2=get_knot_value(i-2);
          double dt = si-sim1;
          double dtprev = sim1-sim2;              
          this->FD_ds_weights[i][0]=1.0 / dt + 1.0 / (dt + dtprev);
          this->FD_ds_weights[i][1]=-(dt + dtprev) / (dt * dtprev);
          this->FD_ds_weights[i][2]=dt / ((dt + dtprev) * dtprev);
          this->FD_ds_inds[i][0]=i;
   
          this->FD_ds_inds[i][1]=get_periodic_knot_index(i-1);
          this->FD_ds_inds[i][2]=get_periodic_knot_index(i-2);
        }
        else if (bspline_order==0)
        {
          this->FD_ds_weights[i].resize(1);
          //this->FD_ds_inds[i].resize(2);
          double ds=get_knot_value(i+1)-get_knot_value(i);
          this->FD_ds_weights[i][0]=1/ds;
          //this->FD_ds_weights[i][1]=-1/ds;
          //std::cout << "FILLING FOR NTSTEPS " << this->n_tsteps() << " ds " << ds <<"" << " ds0 " << get_knot_value(i) << "ds- " << get_knot_value(i-1) <<std::endl;
          
        }         
      }
    }
    else if (bspline_order<0) // Time mesh mode
    {
      unsigned order=-bspline_order-2;
      if (order==0) throw_runtime_error("Orthogonal collocation method order "+std::to_string(order)+" is not implemented");      
      if ((s_knots.size()-1)%order!=0) throw_runtime_error("The (number of knots-1) must be a multiple of the orthogonal collocation method order");
      unsigned Nelem=(s_knots.size()-1)/order;
      unsigned nnode_per_elem=order+1;
      //if (gl_order<0) gl_order=order;      
      gl_order=order;
      time_mesh=new oomph::Mesh;

      for (unsigned int i=0;i<s_knots.size();i++)
      {                           
        time_mesh->add_node_pt(new TimeNode(s_knots[i],(floquet_mode ? i : i%(s_knots.size()-1)))); // PERIODIC MODE or FLOQUET MODE
      }

      if (gl_order==0) collocation_gl=new oomph::POCollocationFakeIntegral;
      else if (gl_order==1) collocation_gl=new oomph::GaussLegendre<1,1>;
      else if (gl_order==2) collocation_gl=new oomph::GaussLegendre<1,2>;
      else if (gl_order==3) collocation_gl=new oomph::GaussLegendre<1,3>;
      else if (gl_order==4) collocation_gl=new oomph::GaussLegendre<1,4>;
      else throw_runtime_error("Orthogonal collocation method integration order is only implemented up to 4 is implemented");

      

      std::cout << "Using collocation order " << order << " and integration order " << gl_order << std::endl;
      

      for (unsigned int ie=0;ie<Nelem;ie++)
      {            
        oomph::QElementBase *el;
        if (nnode_per_elem==2) el=new oomph::QElement<1,2>;
        else if (nnode_per_elem==3) el=new oomph::QElement<1,3>;
        else if (nnode_per_elem==4) el=new oomph::QElement<1,4>;
        //else if (nnode_per_elem==5) el=new oomph::QElement<1,5>;
        else throw_runtime_error("orthogonal collocation method is only implemented up to order 3 is implemented");
                
        
        for (unsigned int in=0;in<nnode_per_elem;in++)
        {
          el->node_pt(in)=time_mesh->node_pt(ie*order+in);
        }        
        time_mesh->element_pt().push_back(el);   
      }

      if (collocation_gl->nweight()!=dynamic_cast<oomph::FiniteElement*>(time_mesh->element_pt(0))->nnode()-1) throw_runtime_error("The number of nodes per element (here "+std::to_string(dynamic_cast<oomph::FiniteElement*>(time_mesh->element_pt(0))->nnode())+") in the time mesh must be the same as the number of weights (here "+std::to_string(collocation_gl->nweight())+") plus 1 in the collocation method");
      
      /*for (unsigned int ie=0;ie<time_mesh->nelement();ie++)
      {
        oomph::QElementBase *el=dynamic_cast<oomph::QElementBase*>(time_mesh->element_pt(ie));
        if (!el) throw_runtime_error("Strange, not a QElementBase");            
        std::cout << "Element " << ie << " of " << time_mesh->nelement() <<  " = " << Nelem << " from " << el->vertex_node_pt(0)->x(0) << " to " << el->vertex_node_pt(1)->x(0) << std::endl;
        std::cout << "NWEIGHT " << el->integral_pt()->nweight() << std::endl;
        for (unsigned int igl=0;igl<el->integral_pt()->nweight();igl++)
        {
          oomph::Shape psi(el->nnode());
          oomph::DShape dpsi(el->nnode(),1);
          double w=el->integral_pt()->weight(igl);
          el->dshape_eulerian_at_knot(igl,psi,dpsi);            
          std::cout << "   GL " << igl << " with weight " << w << " has shapes " << " at " <<  std::endl;
          for (unsigned int in=0;in<el->nnode();in++)
          {
            std::cout << "     " << psi[in] << " " << dpsi(in,0) << std::endl;
          }
        }
      }
      */
      
      //throw_runtime_error("Implement Galerkin finite difference for order "+std::to_string(gl_order));
      //throw_runtime_error("Unknown finite difference mode: "+std::to_string(bspline_order));       
    }
      

    this->update_phase_constraint_information();
    std::cout << "Created PeriodicOrbitHandler with " << bspline_order  << "  BASIS " << basis << std::endl;

  }

  double PeriodicOrbitHandler::get_knot_value(int i)
  {
    double L=this->s_knots.back()-this->s_knots.front();
    double offs=0.0;
    while (i<0) { i+=this->s_knots.size()-1; offs-=L; }
    while (i>=this->s_knots.size()-1) { i-=this->s_knots.size()-1; offs+=L; }
    return this->s_knots[i]+offs;
  }

  unsigned PeriodicOrbitHandler::get_periodic_knot_index(int i)
  {    
    while (i<0) { i+=this->s_knots.size()-1; }
    while (i>=this->s_knots.size()-1) { i-=this->s_knots.size()-1;  }
    return i;
  }

  PeriodicOrbitHandler::~PeriodicOrbitHandler()
  {
    if (time_mesh) 
    {        
        delete time_mesh;
        time_mesh=NULL;
    }
    if (collocation_gl) 
    {
        delete collocation_gl;
        collocation_gl=NULL;
    }
    Problem_pt->GetDofPtr().resize(Ndof);
    Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(),
                                              Ndof, false);
    // Remove all previous sparse storage used during Jacobian assembly
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
    if (basis) delete this->basis; this->basis=NULL;
  }
  unsigned long PeriodicOrbitHandler::eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local)
  {
    unsigned raw_ndof = elem_pt->ndof();
    unsigned long global_eqn;
    unsigned nT=this->n_tsteps();  
    //std::cout << "GETTING GLOB EQ " << ieqn_local << " " << nT << " " << raw_ndof << std::endl;  
    if (ieqn_local < nT*raw_ndof)
    {
      unsigned tindex=ieqn_local/raw_ndof;
      unsigned local_eqn=ieqn_local%raw_ndof;
      global_eqn = Ndof*tindex+elem_pt->eqn_number(local_eqn);
    }
    else
    {
      //std::cout << "RETURNING " << T_global_eqn << " for " << ieqn_local << " of " << nT*raw_ndof+1 << " and " << this->ndof(elem_pt) << std::endl;
      global_eqn = T_global_eqn;
    }    
    //std::cout << " GIVES " << global_eqn << std::endl;  
    return global_eqn;
  }
  
 

  unsigned PeriodicOrbitHandler::ndof(oomph::GeneralisedElement *const &elem_pt)
  {
      unsigned nT=this->n_tsteps();    
      return elem_pt->ndof()*nT +1; 
  }

/*
  void PeriodicOrbitHandler::get_residuals_multi_shoot_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals,double *const &parameter_pt)
  {
      residuals.initialise(0.0);
      unsigned raw_ndof = elem_pt->ndof();
      DenseMatrix<double> jacobian(raw_ndof), M(raw_ndof);            
      Vector<double> current_res(raw_ndof),dof_backup(raw_ndof),dUds(raw_ndof),U(raw_ndof);             
      Vector<unsigned> glob_eqs(raw_ndof);
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();  
      unsigned ntsteps=this->n_tsteps();
      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=elem_pt->eqn_number(i);
          dof_backup[i]=*(alldofs[glob_eq]);          
          glob_eqs[i]=glob_eq;
      }
      oomph::Vector<double> dU0ds(dof_backup.size());
      
      for (unsigned int ie=0;ie<time_mesh->nelement();ie++) 
      {
        oomph::QElementBase * el=dynamic_cast<oomph::QElementBase*>(time_mesh->element_pt(ie));
        oomph::Integral * integral=el->integral_pt();
        for (unsigned int igl=0;igl<integral->nweight();igl++)
        {
          oomph::Shape psi(el->nnode());
          oomph::DShape dpsi(el->nnode(),1);
          double w =el->dshape_eulerian_at_knot(igl,psi,dpsi);          
          w*=this->n_tsteps()*200;
          for (unsigned int i=0;i<raw_ndof;i++) *(alldofs[glob_eqs[i]])=0.0;
          oomph::Vector<double> loc_coord(1);
          loc_coord[0] = el->integral_pt()->knot(igl, 0);          
          double x=el->interpolated_x(loc_coord,0);
          std::cout << " INTEGRATING " << " ie " << ie << " igl " << igl << " with weight " << w << " at local coord " << loc_coord[0] << " and s= " << x << std::endl;
          dUds.initialise(0.0);
          dU0ds.initialise(0.0);
          U.initialise(0.0);
          for (unsigned int in=0;in<el->nnode();in++)
          {
            unsigned index=dynamic_cast<TimeNode*>(el->node_pt(in))->get_index();
            //std::cout << "   NODE " << in << " with index " << index << " HAS " << psi[in] << " and " << dpsi(in,0) << std::endl;
            if (index==0) 
            { 
              for (unsigned int i=0;i<raw_ndof;i++) 
              {
                U[i]+=psi(in)*dof_backup[i];
                dUds[i]+=dpsi(in,0)*dof_backup[i];
              }
            }
            else 
            { 
              for (unsigned int i=0;i<raw_ndof;i++) 
              {
                U[i]+=psi(in)*Tadd[index-1][glob_eqs[i]];
                dUds[i]+=dpsi(in,0)*Tadd[index-1][glob_eqs[i]];
              }
            }
            if (T_constraint_mode==1)
            {
              for (unsigned int i=0;i<raw_ndof;i++) 
              {
                  dU0ds[i]+=dpsi(in,0)*du0ds[index][glob_eqs[i]];
              }
            }


          }

          std::cout << "  GIVES U = "; for (unsigned int i=0;i<raw_ndof;i++) std::cout << U[i] << "  " ;
          std::cout << " and dUds =" ; for (unsigned int i=0;i<raw_ndof;i++) std::cout << dUds[i] << "  " ; 
          std::cout << " and dU0ds =" ; for (unsigned int i=0;i<raw_ndof;i++) std::cout << dU0ds[i] << "  " ; 
          std::cout << std::endl;

          for (unsigned int i=0;i<raw_ndof;i++) *(alldofs[glob_eqs[i]])=U[i];

          current_res.initialise(0.0);
          M.initialise(0.0);
          jacobian.initialise(0.0);  
          if (!parameter_pt) elem_pt->get_jacobian_and_mass_matrix(current_res, jacobian, M);                      
          else elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt,current_res, jacobian, M);

          for (unsigned int i=0;i<raw_ndof;i++) *(alldofs[glob_eqs[i]])=dof_backup[i];

          for (unsigned in=0;in<el->nnode();in++)
          {
            unsigned index=dynamic_cast<TimeNode*>(el->node_pt(in))->get_index();
            if (floquet_mode && index==ntsteps-1) 
            {
              //index=0; 
              continue;
            }
            for (unsigned i = 0; i < raw_ndof; i++)
            {
              residuals[index*raw_ndof + i] += current_res[i]*psi[in]*w;              
              for (unsigned j=0;j<raw_ndof;j++)          
              {
                residuals[index*raw_ndof+i]+=M(i,j)/T*dUds[j]*psi[in]*w;
                //residuals[index*raw_ndof+i]+=M(i,j)/T*0.5*(dUds[j]*psi[in]- U[j]*dpsi(in,0))*w;
              }
              
            }   
            
         
          }              

          // Phase constraint
          if (!parameter_pt && T_constraint_mode==1)
          {
            for (unsigned int i=0;i<raw_ndof;i++)
            {
                residuals[raw_ndof*this->n_tsteps()]+=dU0ds[i]* U[i]/Count[glob_eqs[i]]*w;
            }     
          }

        }
      }
       

      // Fill the connection
      if (floquet_mode)
      {
        // Flush the last step
        //for (unsigned int i=0;i<raw_ndof;i++) residuals[raw_ndof*(this->n_tsteps()-1)+i]=0.0;
        if (!parameter_pt)
        {
          for (unsigned int i=0;i<raw_ndof;i++)
          {          
            residuals[(ntsteps-1)*raw_ndof+i]+=(Tadd[ntsteps-2][glob_eqs[i]]-dof_backup[i])/Count[glob_eqs[i]];
          }
        }
      }


      for (unsigned int i=0;i<raw_ndof;i++)
      {
        *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
      }

      if (!parameter_pt && T_constraint_mode==0)
      {
        double plane_eq=-d_plane;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=elem_pt->eqn_number(i);
          double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
          plane_eq+=x*n0[glob_eq]/Count[glob_eq];
        }      
        residuals[raw_ndof*this->n_tsteps()]=plane_eq;
      }          
  }
*/

void PeriodicOrbitHandler::get_residuals_collocation_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals,double *const &parameter_pt)
  {
      residuals.initialise(0.0);
      unsigned raw_ndof = elem_pt->ndof();
      DenseMatrix<double> jacobian(raw_ndof), M(raw_ndof);            
      Vector<double> current_res(raw_ndof),dof_backup(raw_ndof),dUds(raw_ndof),U(raw_ndof);             
      Vector<unsigned> glob_eqs(raw_ndof);
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();  
      unsigned ntsteps=this->n_tsteps();      
      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=elem_pt->eqn_number(i);
          dof_backup[i]=*(alldofs[glob_eq]);          
          glob_eqs[i]=glob_eq;
      }
      oomph::Vector<double> dU0ds(dof_backup.size());
      
      for (unsigned int ie=0;ie<time_mesh->nelement();ie++) 
      {
        oomph::QElementBase * el=dynamic_cast<oomph::QElementBase*>(time_mesh->element_pt(ie));
        oomph::Shape psi(el->nnode());
        oomph::DShape dpsi(el->nnode(),1);
        double deltaS=el->vertex_node_pt(1)->x(0)-el->vertex_node_pt(0)->x(0);
        for (unsigned int inode=0;inode<el->nnode()-1;inode++)
        {
                    
              double gl_s=collocation_gl->knot(inode,0);
              double w=collocation_gl->weight(inode);
              
              oomph::Vector<double> local_coord(1);              
              local_coord[0]=gl_s;
              el->dshape_eulerian(local_coord,psi,dpsi);                            
              
              
              for (unsigned int i=0;i<raw_ndof;i++) *(alldofs[glob_eqs[i]])=0.0;
              dUds.initialise(0.0);
              dU0ds.initialise(0.0);
              U.initialise(0.0);
              for (unsigned int in=0;in<el->nnode();in++)
              {
                unsigned index=dynamic_cast<TimeNode*>(el->node_pt(in))->get_index();
                //std::cout << "   NODE " << in << " with index " << index << " HAS " << psi[in] << " and " << dpsi(in,0) << std::endl;
                if (index==0) 
                { 
                  for (unsigned int i=0;i<raw_ndof;i++) 
                  {
                    U[i]+=psi(in)*dof_backup[i];
                    dUds[i]+=dpsi(in,0)*dof_backup[i];
                  }
                }
                else 
                { 
                  for (unsigned int i=0;i<raw_ndof;i++) 
                  {
                    U[i]+=psi(in)*Tadd[index-1][glob_eqs[i]];
                    dUds[i]+=dpsi(in,0)*Tadd[index-1][glob_eqs[i]];
                  }
                }
                if (T_constraint_mode==1)
                {
                  for (unsigned int i=0;i<raw_ndof;i++) 
                  {
                      dU0ds[i]+=dpsi(in,0)*du0ds[index][glob_eqs[i]];
                  }
                }


              }

              unsigned index=dynamic_cast<TimeNode*>(el->node_pt(inode))->get_index();


              for (unsigned int i=0;i<raw_ndof;i++) *(alldofs[glob_eqs[i]])=U[i];

              current_res.initialise(0.0);
              M.initialise(0.0);
              jacobian.initialise(0.0);  
              if (!parameter_pt) elem_pt->get_jacobian_and_mass_matrix(current_res, jacobian, M);                      
              else elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt,current_res, jacobian, M);

            
                
              for (unsigned i = 0; i < raw_ndof; i++)
              {
                residuals[index*raw_ndof + i] += current_res[i]*w;              
                for (unsigned j=0;j<raw_ndof;j++)          
                {
                  residuals[index*raw_ndof+i]+=M(i,j)/T*dUds[j]*w;
                }              
              }   
                        

              // Phase constraint
              if (!parameter_pt && T_constraint_mode==1)
              {
                for (unsigned int i=0;i<raw_ndof;i++)
                {
                    residuals[raw_ndof*this->n_tsteps()]+=dU0ds[i]* U[i]/Count[glob_eqs[i]]*deltaS*w;
                }     
              }

            

        }
      }
       

      // Fill the connection
      if (floquet_mode)
      {
        // Flush the last step
        //for (unsigned int i=0;i<raw_ndof;i++) residuals[raw_ndof*(this->n_tsteps()-1)+i]=0.0;
        if (!parameter_pt)
        {
          for (unsigned int i=0;i<raw_ndof;i++)
          {          
            residuals[(ntsteps-1)*raw_ndof+i]+=(Tadd[ntsteps-2][glob_eqs[i]]-dof_backup[i])/Count[glob_eqs[i]];
          }
        }
      }


      for (unsigned int i=0;i<raw_ndof;i++)
      {
        *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
      }

      if (!parameter_pt && T_constraint_mode==0)
      {
        double plane_eq=-d_plane;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=elem_pt->eqn_number(i);
          double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
          plane_eq+=x*n0[glob_eq]/Count[glob_eq];
        }      
        residuals[raw_ndof*this->n_tsteps()]+=plane_eq;
      }          
  }

  
  void PeriodicOrbitHandler::get_residuals_floquet_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals,double *const &parameter_pt)  
  {
      residuals.initialise(0.0);
      unsigned raw_ndof = elem_pt->ndof();
      DenseMatrix<double> jacobian(raw_ndof), M(raw_ndof);            
      Vector<double> current_res(raw_ndof),dof_backup(raw_ndof),U(raw_ndof),dUds(raw_ndof);             
      Vector<unsigned> glob_eqs(raw_ndof);
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();  
      unsigned ntsteps=this->n_tsteps();
      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=elem_pt->eqn_number(i);
          dof_backup[i]=*(alldofs[glob_eq]);          
          glob_eqs[i]=glob_eq;
      }
      std::vector<double> U0=dof_backup;
      std::vector<double> Uplus(U0.size());
      
      for (unsigned int ti=0;ti<ntsteps-1;ti++)  // Only loop to n-1 (the last, periodic dofs are handled via identity matrices)
      {
        double invds=this->FD_ds_weights[ti][0];
        for (unsigned int i=0;i<U0.size();i++)
        {
          Uplus[i]=Tadd[ti][glob_eqs[i]];
        }
                   
        for (unsigned int i=0;i<raw_ndof;i++)
        {
            U[i]=0.5*(U0[i]+Uplus[i]);
            dUds[i]=invds*(Uplus[i]-U0[i]);
            *(alldofs[glob_eqs[i]])=U[i];
        }
        current_res.initialise(0.0);
        M.initialise(0.0);
        jacobian.initialise(0.0);  
        if (!parameter_pt) elem_pt->get_jacobian_and_mass_matrix(current_res, jacobian, M);                      
        else elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt,current_res, jacobian, M);
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          residuals[ti*raw_ndof + i] = current_res[i];              
          for (unsigned j=0;j<raw_ndof;j++)          
          {
            residuals[ti*raw_ndof+i]+=M(i,j)*dUds[j]/T;
          }
        }   

        // Phase constraint
        if (!parameter_pt && T_constraint_mode==1)
        {
          for (unsigned int i=0;i<raw_ndof;i++)
          {
            residuals[raw_ndof*this->n_tsteps()]+=du0ds[ti][glob_eqs[i]]*U[i]/Count[glob_eqs[i]];
          }
          
        }
        U0=Uplus; // Shift the buffer
      }

      // Fill the connection
      //std::cout << "CONNECTION INDEX" << ntsteps << std::endl;
      if (!parameter_pt)
      {
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          residuals[(ntsteps-1)*raw_ndof+i]+=(Tadd[ntsteps-2][glob_eqs[i]]-dof_backup[i])/Count[glob_eqs[i]];
        }
      }


      for (unsigned int i=0;i<raw_ndof;i++)
      {
        *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
      }

      if (!parameter_pt && T_constraint_mode==0)
        {
          double plane_eq=-d_plane;
          for (unsigned int i=0;i<raw_ndof;i++)
          {
            unsigned glob_eq=elem_pt->eqn_number(i);
            double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
            plane_eq+=x*n0[glob_eq]/Count[glob_eq];
          }      
          residuals[raw_ndof*this->n_tsteps()]=plane_eq;
        }      
  }


   void PeriodicOrbitHandler::get_jacobian_floquet_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
      if (!Problem_pt->are_hessian_products_calculated_analytically())
      {
        throw_runtime_error("Cannot track periodic orbits without having analytical Hessian. Use Problem.setup_for_stability_analysis(analytic_hessian=True) before.");
      }
      residuals.initialise(0.0);
      jacobian.initialise(0.0);
      unsigned ntsteps=this->n_tsteps();
      pyoomph::BulkElementBase * pyoomph_elem_pt=dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
      auto *ft=pyoomph_elem_pt->get_code_instance()->get_func_table();
      bool has_constant_mass_matrix=false;
      if (ft->current_res_jac>=0) 
      { 
        has_constant_mass_matrix=ft->has_constant_mass_matrix_for_sure[ft->current_res_jac];   
      }
      
      unsigned raw_ndof = elem_pt->ndof();
      DenseMatrix<double> J(raw_ndof), M(raw_ndof);            
      Vector<double> current_res(raw_ndof);      
      Vector<double> dof_backup(raw_ndof);            
      Vector<unsigned> glob_eqs(raw_ndof);
      unsigned Teq=raw_ndof*this->n_tsteps();
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();
      Vector<double> U(raw_ndof,0.0),dUds(raw_ndof,0.0);          

      std::vector<SinglePassMultiAssembleInfo> multi_assm;
      multi_assm.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &current_res, &J, &M));
      oomph::DenseMatrix<double> dMdU_dUdsterm(raw_ndof,raw_ndof,0.0);
      oomph::DenseMatrix<double> dummy_dJdU_dUdsterm(raw_ndof,raw_ndof,0.0);            
      multi_assm.back().add_hessian(dUds, &dummy_dJdU_dUdsterm, &dMdU_dUdsterm);
      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=elem_pt->eqn_number(i);
          dof_backup[i]=*(alldofs[glob_eq]);          
          glob_eqs[i]=glob_eq;
      }            

      std::vector<double> U0=dof_backup;
      std::vector<double> Uplus(U0.size());
      
      for (unsigned int ti=0;ti<ntsteps-1;ti++)  // Only loop to n-1 (the last, periodic dofs are handled via identity matrices)
      {
        for (unsigned int i=0;i<U0.size();i++)
        {
          Uplus[i]=Tadd[ti][glob_eqs[i]];
        }
        double invds=this->FD_ds_weights[ti][0];
        unsigned index0,indexplus;        
        for (unsigned int i=0;i<raw_ndof;i++)
        {
            U[i]=0.5*(U0[i]+Uplus[i]);
            dUds[i]=invds*(Uplus[i]-U0[i]);
        }      
        for (unsigned int i=0;i<raw_ndof;i++)
        {
             *(alldofs[glob_eqs[i]])=U[i];
        }
        current_res.initialise(0.0);
        M.initialise(0.0);
        J.initialise(0.0);  
        if (has_constant_mass_matrix)
        {
          elem_pt->get_jacobian_and_mass_matrix(current_res, J, M);                      
        }
        else
        {            
          dMdU_dUdsterm.initialise(0.0);
          dummy_dJdU_dUdsterm.initialise(0.0);
          pyoomph_elem_pt->get_multi_assembly(multi_assm);
        }
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          residuals[ti*raw_ndof + i] = current_res[i];              
          for (unsigned j=0;j<raw_ndof;j++)          
          {
            jacobian(ti*raw_ndof+i,ti*raw_ndof+j)+=0.5*J(i,j);
            jacobian(ti*raw_ndof+i,(ti+1)*raw_ndof+j)+=0.5*J(i,j);
            
            residuals[ti*raw_ndof+i]+=M(i,j)*dUds[j]/T;
            if (!has_constant_mass_matrix)
            {
                jacobian(ti*raw_ndof+i,ti*raw_ndof+j)+=0.5*dMdU_dUdsterm(i,j)/T;
                jacobian(ti*raw_ndof+i,(ti+1)*raw_ndof+j)+=0.5*dMdU_dUdsterm(i,j)/T;
            }  
            jacobian(ti*raw_ndof+i,ti*raw_ndof+j)+=-M(i,j)*invds/T;  
            jacobian(ti*raw_ndof+i,(ti+1)*raw_ndof+j)+=M(i,j)*invds/T;  
            jacobian(ti*raw_ndof+i,Teq)+=-M(i,j)*dUds[j]/(T*T);
          }
        }   

        // Phase constraint
        if (T_constraint_mode==1)
        {
          for (unsigned int i=0;i<raw_ndof;i++)
          {
            residuals[raw_ndof*this->n_tsteps()]+=du0ds[ti][glob_eqs[i]]*U[i]/Count[glob_eqs[i]];
            jacobian(Teq,ti*raw_ndof+i)+=0.5*du0ds[ti][glob_eqs[i]]/Count[glob_eqs[i]];
            jacobian(Teq,(ti+1)*raw_ndof+i)+=0.5*du0ds[ti][glob_eqs[i]]/Count[glob_eqs[i]];
          }
          
        }
        U0=Uplus; // Shift the buffer
      }
   
      for (unsigned int i=0;i<raw_ndof;i++)
      {
        residuals[(ntsteps-1)*raw_ndof+i]+=(Tadd[ntsteps-2][glob_eqs[i]]-dof_backup[i])/Count[glob_eqs[i]];
        jacobian((ntsteps-1)*raw_ndof+i,(ntsteps-1)*raw_ndof+i)+=1.0/Count[glob_eqs[i]];
        jacobian((ntsteps-1)*raw_ndof+i,i)+=-1.0/Count[glob_eqs[i]];
      }

      for (unsigned int i=0;i<raw_ndof;i++)
      {
        *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
      }


      if (T_constraint_mode==0)
      {
        double plane_eq=-d_plane;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=glob_eqs[i];
          double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
          plane_eq+=x*n0[glob_eq]/Count[glob_eq];
        }

        // Get the plane equation
        residuals[raw_ndof*this->n_tsteps()]=plane_eq;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=glob_eqs[i];
          jacobian(Teq,i)+=n0[glob_eq]/Count[glob_eq];
        }
      }
  }

  void PeriodicOrbitHandler::get_residuals_time_nodal_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals,double *const &parameter_pt)  
  {
      residuals.initialise(0.0);
      unsigned raw_ndof = elem_pt->ndof();
      DenseMatrix<double> jacobian(raw_ndof), M(raw_ndof);            
      Vector<double> current_res(raw_ndof),dof_backup(raw_ndof);             
      Vector<unsigned> glob_eqs(raw_ndof);
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();      
      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=elem_pt->eqn_number(i);
          dof_backup[i]=*(alldofs[glob_eq]);          
          glob_eqs[i]=glob_eq;
      }     

      for (unsigned int ti=0;ti<this->n_tsteps();ti++)
      {          
        Vector<double> ddof_ds(raw_ndof,0.0);          
        for (unsigned int ii=0;ii<this->FD_ds_inds[ti].size();ii++)
        {
          unsigned index=this->FD_ds_inds[ti][ii];
          if (index>0)
          {
            index--;
            for (unsigned int i=0;i<raw_ndof;i++)
            {              
              ddof_ds[i]+=this->FD_ds_weights[ti][ii]*Tadd[index][glob_eqs[i]];
            }
          }
          else
          {
            for (unsigned int i=0;i<raw_ndof;i++)
            {              
              ddof_ds[i]+=this->FD_ds_weights[ti][ii]*dof_backup[i];
            }

          }
        }

        // Setup the dofs          
        if (ti>0)
        {
          for (unsigned int i=0;i<raw_ndof;i++)
          {
            unsigned glob_eq=glob_eqs[i];
            *(alldofs[glob_eq])=Tadd[ti-1][glob_eq];              
          }
        }            
        current_res.initialise(0.0);
        M.initialise(0.0);
        jacobian.initialise(0.0);  
        if (!parameter_pt) elem_pt->get_jacobian_and_mass_matrix(current_res, jacobian, M);                      
        else elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt,current_res, jacobian, M);
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          residuals[ti*raw_ndof + i] = current_res[i];              
          for (unsigned j=0;j<raw_ndof;j++)          
          {
            residuals[ti*raw_ndof+i]+=M(i,j)*ddof_ds[j]/T;
          }
        }  

          // Phase constraint
        if (!parameter_pt && T_constraint_mode==1)
        {
          double ds=0.5*(this->get_knot_value(ti+1)-this->get_knot_value(ti-1));
          for (unsigned int i=0;i<raw_ndof;i++)
          {
            residuals[raw_ndof*this->n_tsteps()]+=du0ds[ti][glob_eqs[i]]*(*(alldofs[glob_eqs[i]]))/Count[glob_eqs[i]]*ds;
            //jacobian(raw_ndof*this->n_tsteps(),ti*raw_ndof+i)+=du0ds[ti][glob_eqs[i]]/Count[glob_eqs[i]];            
          }
          
        }        
      }

      for (unsigned int i=0;i<raw_ndof;i++)
      {
        *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
      }

      if (!parameter_pt && T_constraint_mode==0)
      {
        double plane_eq=-d_plane;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=elem_pt->eqn_number(i);
          double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
          plane_eq+=x*n0[glob_eq]/Count[glob_eq];
        }      
        residuals[raw_ndof*this->n_tsteps()]=plane_eq;
      }

  }

  void PeriodicOrbitHandler::get_residuals_bspline_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals,double *const &parameter_pt)  
  {
      residuals.initialise(0.0);
      unsigned raw_ndof = elem_pt->ndof();
      DenseMatrix<double> jacobian(raw_ndof), M(raw_ndof);            
      Vector<double> current_res(raw_ndof),dof_backup(raw_ndof);             
      Vector<unsigned> glob_eqs(raw_ndof);
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();      
      oomph::Vector<double> dU0ds;
      if (!parameter_pt && T_constraint_mode==1) dU0ds.resize(raw_ndof,0.0);
      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=elem_pt->eqn_number(i);
          dof_backup[i]=*(alldofs[glob_eq]);          
          glob_eqs[i]=glob_eq;
      }   

      for (unsigned int ie=0;ie<this->basis->get_num_elements();ie++)
      {
        std::vector<double> w;
        std::vector<unsigned> indices;
        std::vector<std::vector<double>> psi_s;
        std::vector<std::vector<double>> dpsi_ds;
        unsigned nGL=this->basis->get_integration_info(ie,w,indices,psi_s,dpsi_ds);
        for (unsigned iGL=0;iGL<nGL;iGL++)
        {
          std::vector<double> Ulocal(raw_ndof,0.0);
          std::vector<double> dUdsLocal(raw_ndof,0.0);
          if (!parameter_pt && T_constraint_mode==1) 
          {
              dU0ds.initialise(0.0);
          }
          for (unsigned int psi_index=0;psi_index<indices.size();psi_index++)
          {
            std::vector<double> U_at_index(raw_ndof,0.0);
            if (indices[psi_index]==0) U_at_index=dof_backup;
            else 
            {
              for (unsigned int i=0;i<raw_ndof;i++)
              {
                U_at_index[i]=Tadd[indices[psi_index]-1][glob_eqs[i]];
              }
            }
            // I guess this can be optimized and filled in a rotary buffer
            for (unsigned int i=0;i<raw_ndof;i++)
            {
              Ulocal[i]+=psi_s[iGL][psi_index]*U_at_index[i];
              dUdsLocal[i]+=dpsi_ds[iGL][psi_index]*U_at_index[i];
            }

            if (!parameter_pt && T_constraint_mode==1) 
            {
              for (unsigned int i=0;i<raw_ndof;i++)
              {                
                dU0ds[i]+=dpsi_ds[iGL][psi_index]*du0ds[indices[psi_index]][glob_eqs[i]];
              }            
            }
          }

          for (unsigned int i=0;i<raw_ndof;i++)
          {
            unsigned glob_eq=elem_pt->eqn_number(i);
            *(alldofs[glob_eqs[i]])=Ulocal[i]; // Set the unknowns
          }

          current_res.initialise(0.0);
          M.initialise(0.0);
          jacobian.initialise(0.0);  
          if (!parameter_pt) elem_pt->get_jacobian_and_mass_matrix(current_res, jacobian, M);                      
          else elem_pt->get_djacobian_and_dmass_matrix_dparameter(parameter_pt,current_res, jacobian, M);

          if (!parameter_pt && T_constraint_mode==1)
          {
            for (unsigned i = 0; i < raw_ndof; i++)
            {
              double fact=dU0ds[i]/Count[glob_eqs[i]]*w[iGL];
              residuals[raw_ndof*this->n_tsteps()]+=fact*Ulocal[i];
              //for (unsigned int l2=0;l2<indices.size();l2++)
              //{
              //  unsigned ti2=indices[l2];  
                  //jacobian(raw_ndof*this->n_tsteps(),ti2*raw_ndof+i)+=fact*psi_s[iGL][l2];
              //}
            }
          }

          for (unsigned int l=0;l<indices.size();l++)
          {
            unsigned ti=indices[l];
            for (unsigned i = 0; i < raw_ndof; i++)
            {
              residuals[ti*raw_ndof + i] += current_res[i]*psi_s[iGL][l]*w[iGL];            
              for (unsigned j=0;j<raw_ndof;j++)          
              {
                residuals[ti*raw_ndof+i]+=M(i,j)*dUdsLocal[j]/T*psi_s[iGL][l]*w[iGL];                
              }
            }
          }
        }
      }

      for (unsigned int i=0;i<raw_ndof;i++)
      {
        *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
      }
      if (!parameter_pt && T_constraint_mode==0)
      {
        double plane_eq=-d_plane;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=elem_pt->eqn_number(i);
          double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
          plane_eq+=x*n0[glob_eq]/Count[glob_eq];
        }      
        residuals[raw_ndof*this->n_tsteps()]=plane_eq;    
      }
  }

  void PeriodicOrbitHandler::get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals)  
  {        
    unsigned raw_ndof=elem_pt->ndof();
    if (!raw_ndof) {residuals.initialise(0.0); return;}
      if (!this->basis)
      {                
        if (time_mesh) 
        {
          
          this->get_residuals_collocation_mode(elem_pt,residuals,NULL);          
        }
        else if (floquet_mode) this->get_residuals_floquet_mode(elem_pt,residuals,NULL);     
        else this->get_residuals_time_nodal_mode(elem_pt,residuals,NULL);                     
      }      
      else
      {       
        this->get_residuals_bspline_mode(elem_pt,residuals,NULL);
      }  
#ifdef PYOOMPH_BIFURCATION_HANDLER_DEBUG
      /// TODO: Remove
      
      
      
        oomph::DenseMatrix<double> Jdummy(residuals.size(),residuals.size(),0.0);
        oomph::Vector<double> resdummy(residuals.size(),0.0);      
        if (!basis)
        {
          if (time_mesh) 
          {
            this->get_jacobian_collocation_mode(elem_pt,resdummy,Jdummy);
          }
          else if (floquet_mode)
          {
            this->get_jacobian_floquet_mode(elem_pt,resdummy,Jdummy);
          }
          else
          {
            this->get_jacobian_time_nodal_mode(elem_pt,resdummy,Jdummy);
          }
        }
        else this->get_jacobian_bspline_mode(elem_pt,resdummy,Jdummy);
        for (unsigned int i=0;i<residuals.size();i++)
        {        
            if (std::fabs(residuals[i]-resdummy[i])>1e-10) std::cout << "RESIDUAL " << i << " " << residuals[i]-resdummy[i] << std::endl;
        }
#endif      
  }

  /*void PeriodicOrbitHandler::get_jacobian_multi_shoot_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
    if (!Problem_pt->are_hessian_products_calculated_analytically())
    {
      throw_runtime_error("Cannot track periodic orbits without having analytical Hessian. Use Problem.setup_for_stability_analysis(analytic_hessian=True) before.");
    }

    pyoomph::BulkElementBase * pyoomph_elem_pt=dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    auto *ft=pyoomph_elem_pt->get_code_instance()->get_func_table();
    bool has_constant_mass_matrix=false;
    if (ft->current_res_jac>=0) 
    { 
      has_constant_mass_matrix=ft->has_constant_mass_matrix_for_sure[ft->current_res_jac];   
    }      

    residuals.initialise(0.0);
    jacobian.initialise(0.0);
    unsigned raw_ndof = elem_pt->ndof();    
    Vector<double> current_res(raw_ndof),dof_backup(raw_ndof),dUds(raw_ndof);             
    Vector<unsigned> glob_eqs(raw_ndof);
    oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();  
    unsigned ntsteps=this->n_tsteps();

    
    std::vector<SinglePassMultiAssembleInfo> multi_assm;
    DenseMatrix<double> J(raw_ndof), M(raw_ndof);            
    multi_assm.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &current_res, &J, &M));
    oomph::DenseMatrix<double> dMdU_dUdsterm(raw_ndof,raw_ndof,0.0);
    oomph::DenseMatrix<double> dummy_dJdU_dUdsterm(raw_ndof,raw_ndof,0.0);            
    multi_assm.back().add_hessian(dUds, &dummy_dJdU_dUdsterm, &dMdU_dUdsterm);
    unsigned Teq=raw_ndof*this->n_tsteps();
    for (unsigned int i=0;i<raw_ndof;i++)
    {
        unsigned glob_eq=elem_pt->eqn_number(i);
        dof_backup[i]=*(alldofs[glob_eq]);          
        glob_eqs[i]=glob_eq;
    }
    std::vector<double> U0=dof_backup;
    oomph::Vector<double> dU0ds(U0.size());
    
    for (unsigned int ie=0;ie<time_mesh->nelement();ie++) 
    {
      oomph::QElementBase * el=dynamic_cast<oomph::QElementBase*>(time_mesh->element_pt(ie));
      oomph::Integral * integral=el->integral_pt();
      for (unsigned int igl=0;igl<integral->nweight();igl++)
      {
        oomph::Shape psi(el->nnode());
        oomph::DShape dpsi(el->nnode(),1);
        double w =el->dshape_eulerian_at_knot(igl,psi,dpsi);
        w*=this->n_tsteps();
        for (unsigned int i=0;i<raw_ndof;i++) *(alldofs[glob_eqs[i]])=0.0;
        dUds.initialise(0.0);
        dU0ds.initialise(0.0);
        for (unsigned int in=0;in<el->nnode();in++)
        {
          unsigned index=dynamic_cast<TimeNode*>(el->node_pt(in))->get_index();
          if (index==0) 
          { 
            for (unsigned int i=0;i<raw_ndof;i++) 
            {
              *(alldofs[glob_eqs[i]])+=psi(in)*dof_backup[i];
              dUds[i]+=dpsi(in,0)*dof_backup[i];
            }
          }
          else 
          { 
            for (unsigned int i=0;i<raw_ndof;i++) 
            {
              *(alldofs[glob_eqs[i]])+=psi(in)*Tadd[index-1][glob_eqs[i]];
              dUds[i]+=dpsi(in,0)*Tadd[index-1][glob_eqs[i]];
            }
          }
          if (T_constraint_mode==1) 
          {
            for (unsigned int i=0;i<raw_ndof;i++)
            {
              dU0ds[i]+=dpsi(in,0)*du0ds[index][glob_eqs[i]];
            }            
          }
        }

        current_res.initialise(0.0);
        M.initialise(0.0);
        J.initialise(0.0);  
        if (has_constant_mass_matrix)
        {
          elem_pt->get_jacobian_and_mass_matrix(current_res, J, M);
        }
        else
        {            
          dMdU_dUdsterm.initialise(0.0);
          dummy_dJdU_dUdsterm.initialise(0.0);
          pyoomph_elem_pt->get_multi_assembly(multi_assm);
        }                    

        for (unsigned in=0;in<el->nnode();in++)
        {
          unsigned index=dynamic_cast<TimeNode*>(el->node_pt(in))->get_index();
          if (floquet_mode && index==ntsteps-1) continue;
          //std::cout << " ie " << ie << " igl " << igl << " in " << in << " index " << index << " psi " << psi[in] << " dpsi " << dpsi(in,0) << " w " << w << std::endl;
          for (unsigned i = 0; i < raw_ndof; i++)
          {
            residuals[index*raw_ndof + i] += current_res[i]*psi[in]*w;                          
            for (unsigned j=0;j<raw_ndof;j++)          
            {
              residuals[index*raw_ndof+i]+=M(i,j)*dUds[j]/T*psi[in]*w;
              jacobian(index*raw_ndof+i,Teq)+=-M(i,j)*dUds[j]*psi[in]*w/(T*T); 
              
                               
            }
            for (unsigned in2=0;in2<el->nnode();in2++)
            {
              unsigned index2=dynamic_cast<TimeNode*>(el->node_pt(in2))->get_index();
              for (unsigned j=0;j<raw_ndof;j++)
              {
                jacobian(index*raw_ndof + i,index2*raw_ndof+j) += J(i,j)*psi[in2]*psi[in]*w;              
                //jacobian(index*raw_ndof+i,index2*raw_ndof+j)+=J(i,j)*psi[in]*psi[in2]*w;                      
                jacobian(index*raw_ndof+i,index2*raw_ndof+j)+=M(i,j)*(dpsi(in2,0)*psi[in])*w/T;                  
                if (!has_constant_mass_matrix)
                {
                  jacobian(index*raw_ndof+i,index2*raw_ndof+j)+=dMdU_dUdsterm(i,j)/T*psi[in]*psi[in2]*w;
                }        
              }
              
              
            }

            
          } 
          
          // Phase constraint
          //if (T_constraint_mode==1)
          //{
            //for (unsigned int i=0;i<raw_ndof;i++)
            //{
              //residuals[raw_ndof*this->n_tsteps()]+=du0ds[index][glob_eqs[i]]**(alldofs[glob_eqs[i]])/Count[glob_eqs[i]]*psi[in]*w;
            //}            
          //}

        

        
        }

        if (T_constraint_mode==1)
        {
            for (unsigned i = 0; i < raw_ndof; i++)
            {            
              residuals[raw_ndof*this->n_tsteps()]+=dU0ds[i]/Count[glob_eqs[i]]*w* *(alldofs[glob_eqs[i]]);             
            }
            for (unsigned in2=0;in2<el->nnode();in2++)
            {
              unsigned index2=dynamic_cast<TimeNode*>(el->node_pt(in2))->get_index();
              if (floquet_mode && index2==ntsteps-1) continue;
              for (unsigned i = 0; i < raw_ndof; i++)
              {
                jacobian(raw_ndof*this->n_tsteps(),index2*raw_ndof+i)+=dU0ds[i]/Count[glob_eqs[i]]*w *psi(in2);
              }
            }
        }                


      }

     
    }
      

    // Fill the connection
    if (floquet_mode)
    {
      // flush the residuals and jacobian in the last time step
      for (unsigned int i=0;i<raw_ndof;i++) 
      {
        residuals[raw_ndof*(this->n_tsteps()-1)+i]=0.0;
        for (unsigned int j=0;j<raw_ndof*this->n_tsteps();j++)
        {
          jacobian(raw_ndof*(this->n_tsteps()-1)+i,j)=0.0;
        }
        
      }
      for (unsigned int i=0;i<raw_ndof;i++)
      {          
        residuals[(ntsteps-1)*raw_ndof+i]+=(Tadd[ntsteps-2][glob_eqs[i]]-dof_backup[i])/Count[glob_eqs[i]];
        jacobian((ntsteps-1)*raw_ndof+i,(ntsteps-1)*raw_ndof+i)+=1.0/Count[glob_eqs[i]];
        jacobian((ntsteps-1)*raw_ndof+i,i)+=-1.0/Count[glob_eqs[i]];
      }
    }


    for (unsigned int i=0;i<raw_ndof;i++)
    {
      *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
    }

    if (T_constraint_mode==0)
    {
      double plane_eq=-d_plane;
      for (unsigned int i=0;i<raw_ndof;i++)
      {
        unsigned glob_eq=elem_pt->eqn_number(i);
        double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
        plane_eq+=x*n0[glob_eq]/Count[glob_eq];
      }      
      residuals[raw_ndof*this->n_tsteps()]=plane_eq;
      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=glob_eqs[i];
          jacobian(Teq,i)=n0[glob_eq]/Count[glob_eq];
      }
    }          
  }
  */

 void PeriodicOrbitHandler::get_jacobian_collocation_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals,oomph::DenseMatrix<double> & jacobian)
  {
      residuals.initialise(0.0);
      jacobian.initialise(0.0);
      unsigned raw_ndof = elem_pt->ndof();
      DenseMatrix<double> J(raw_ndof), M(raw_ndof);            
      Vector<double> current_res(raw_ndof),dof_backup(raw_ndof),dUds(raw_ndof),U(raw_ndof);             
      Vector<unsigned> glob_eqs(raw_ndof);
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();  
      unsigned ntsteps=this->n_tsteps();    
      unsigned Teq=raw_ndof*this->n_tsteps();  
      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=elem_pt->eqn_number(i);
          dof_backup[i]=*(alldofs[glob_eq]);          
          glob_eqs[i]=glob_eq;
      }
      oomph::Vector<double> dU0ds(dof_backup.size());

      pyoomph::BulkElementBase * pyoomph_elem_pt=dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
      auto *ft=pyoomph_elem_pt->get_code_instance()->get_func_table();
      bool has_constant_mass_matrix=false;
      if (ft->current_res_jac>=0) 
      { 
        has_constant_mass_matrix=ft->has_constant_mass_matrix_for_sure[ft->current_res_jac];   
      }      

      std::vector<SinglePassMultiAssembleInfo> multi_assm;
      multi_assm.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &current_res, &J, &M));
      oomph::DenseMatrix<double> dMdU_dUdsterm(raw_ndof,raw_ndof,0.0);
      oomph::DenseMatrix<double> dummy_dJdU_dUdsterm(raw_ndof,raw_ndof,0.0);            
      multi_assm.back().add_hessian(dUds, &dummy_dJdU_dUdsterm, &dMdU_dUdsterm);
      
      for (unsigned int ie=0;ie<time_mesh->nelement();ie++) 
      {
        oomph::QElementBase * el=dynamic_cast<oomph::QElementBase*>(time_mesh->element_pt(ie));
        oomph::Shape psi(el->nnode());
        oomph::DShape dpsi(el->nnode(),1);
        double deltaS=el->vertex_node_pt(1)->x(0)-el->vertex_node_pt(0)->x(0);
        for (unsigned int inode=0;inode<el->nnode()-1;inode++)
        {
                              
              double gl_s=collocation_gl->knot(inode,0);
              double w=collocation_gl->weight(inode);
              
              oomph::Vector<double> local_coord(1);              
              local_coord[0]=gl_s;
              el->dshape_eulerian(local_coord,psi,dpsi);
              
              for (unsigned int i=0;i<raw_ndof;i++) *(alldofs[glob_eqs[i]])=0.0;
              dUds.initialise(0.0);
              dU0ds.initialise(0.0);
              U.initialise(0.0);
              for (unsigned int in=0;in<el->nnode();in++)
              {
                unsigned index=dynamic_cast<TimeNode*>(el->node_pt(in))->get_index();
                //std::cout << "   NODE " << in << " with index " << index << " HAS " << psi[in] << " and " << dpsi(in,0) << std::endl;
                if (index==0) 
                { 
                  for (unsigned int i=0;i<raw_ndof;i++) 
                  {
                    U[i]+=psi(in)*dof_backup[i];
                    dUds[i]+=dpsi(in,0)*dof_backup[i];
                  }
                }
                else 
                { 
                  for (unsigned int i=0;i<raw_ndof;i++) 
                  {
                    U[i]+=psi(in)*Tadd[index-1][glob_eqs[i]];
                    dUds[i]+=dpsi(in,0)*Tadd[index-1][glob_eqs[i]];
                  }
                }
                if (T_constraint_mode==1)
                {
                  for (unsigned int i=0;i<raw_ndof;i++) 
                  {
                      dU0ds[i]+=dpsi(in,0)*du0ds[index][glob_eqs[i]];
                  }
                }


              }


              for (unsigned int i=0;i<raw_ndof;i++) *(alldofs[glob_eqs[i]])=U[i];
              unsigned index=dynamic_cast<TimeNode*>(el->node_pt(inode))->get_index();

              current_res.initialise(0.0);                                          
              M.initialise(0.0);
              J.initialise(0.0);  
              if (has_constant_mass_matrix)
              {
                elem_pt->get_jacobian_and_mass_matrix(current_res, J, M);
              }
              else
              {            
                dMdU_dUdsterm.initialise(0.0);
                dummy_dJdU_dUdsterm.initialise(0.0);
                pyoomph_elem_pt->get_multi_assembly(multi_assm);
              }                   

                          
                
              for (unsigned i = 0; i < raw_ndof; i++)
              {
                residuals[index*raw_ndof + i] += current_res[i]*w;              
                for (unsigned j=0;j<raw_ndof;j++)          
                {                 
                  residuals[index*raw_ndof+i]+=M(i,j)/T*dUds[j]*w;
                  jacobian(index*raw_ndof+i,Teq)+=-M(i,j)*dUds[j]/(T*T)*w;            
                }  
                

                for (unsigned int nn2=0;nn2<el->nnode();nn2++)
                {
                  unsigned index2=dynamic_cast<TimeNode*>(el->node_pt(nn2))->get_index();
                  for (unsigned j=0;j<raw_ndof;j++)
                  {
                    jacobian(index*raw_ndof + i,index2*raw_ndof+j) += J(i,j)*psi[nn2]*w;              
                    jacobian(index*raw_ndof+i,index2*raw_ndof+j)+=M(i,j)/T*dpsi(nn2,0)*w;                  
                    if (!has_constant_mass_matrix)
                    {
                      jacobian(index*raw_ndof+i,index2*raw_ndof+j)+=dMdU_dUdsterm(i,j)/T*dpsi(nn2,0)*w;
                    }        
                  }                  
                }
                
              }   
                        

              // Phase constraint
              if (T_constraint_mode==1)
              {
                for (unsigned int i=0;i<raw_ndof;i++)
                {
                    residuals[raw_ndof*this->n_tsteps()]+=dU0ds[i]* U[i]/Count[glob_eqs[i]]*deltaS*w;
                    for (unsigned int nn2=0;nn2<el->nnode();nn2++)
                    {
                      unsigned index2=dynamic_cast<TimeNode*>(el->node_pt(nn2))->get_index();
                      jacobian(raw_ndof*this->n_tsteps(),index2*raw_ndof+i)+=dU0ds[i]*psi[nn2]/Count[glob_eqs[i]]*deltaS*w;
                    }
                }     
              }
          

        }
      }
       

      // Fill the connection
      if (floquet_mode)
      {
        // Flush the last step
        //for (unsigned int i=0;i<raw_ndof;i++) residuals[raw_ndof*(this->n_tsteps()-1)+i]=0.0;        
          for (unsigned int i=0;i<raw_ndof;i++)
          {          
            residuals[(ntsteps-1)*raw_ndof+i]+=(Tadd[ntsteps-2][glob_eqs[i]]-dof_backup[i])/Count[glob_eqs[i]];
            jacobian((ntsteps-1)*raw_ndof+i,(ntsteps-1)*raw_ndof+i)+=1.0/Count[glob_eqs[i]];
            jacobian((ntsteps-1)*raw_ndof+i,i)+=-1.0/Count[glob_eqs[i]];
          }
      }


      for (unsigned int i=0;i<raw_ndof;i++)
      {
        *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
      }

      if (T_constraint_mode==0)
      {
        double plane_eq=-d_plane;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=elem_pt->eqn_number(i);
          double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
          plane_eq+=x*n0[glob_eq]/Count[glob_eq];
        }      
        residuals[raw_ndof*this->n_tsteps()]+=plane_eq;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
            unsigned glob_eq=glob_eqs[i];
            jacobian(raw_ndof*this->n_tsteps(),i)+=n0[glob_eq]/Count[glob_eq];
        }
      }          
  }


  void PeriodicOrbitHandler::get_jacobian_bspline_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
      if (!Problem_pt->are_hessian_products_calculated_analytically())
      {
        throw_runtime_error("Cannot track periodic orbits without having analytical Hessian. Use Problem.setup_for_stability_analysis(analytic_hessian=True) before.");
      }
      residuals.initialise(0.0);
      jacobian.initialise(0.0);
      pyoomph::BulkElementBase * pyoomph_elem_pt=dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
      auto *ft=pyoomph_elem_pt->get_code_instance()->get_func_table();
      bool has_constant_mass_matrix=false;
      if (ft->current_res_jac>=0) 
      { 
        has_constant_mass_matrix=ft->has_constant_mass_matrix_for_sure[ft->current_res_jac];   
      }      
      /*if (!has_constant_mass_matrix)
      {
        throw_runtime_error("The mass matrix must be constant for the time being for periodic orbits.");
      }*/
      unsigned raw_ndof = elem_pt->ndof();
      DenseMatrix<double> J(raw_ndof), M(raw_ndof);            
      Vector<double> current_res(raw_ndof);      
      Vector<double> dof_backup(raw_ndof);            
      Vector<unsigned> glob_eqs(raw_ndof);
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();

      oomph::Vector<double> dU0ds;
      if (T_constraint_mode==1) dU0ds.resize(raw_ndof,0.0);

      oomph::Vector<double> Ulocal(raw_ndof,0.0);
      oomph::Vector<double> dUdsLocal(raw_ndof,0.0);

      std::vector<SinglePassMultiAssembleInfo> multi_assm;
      multi_assm.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &current_res, &J, &M));
      oomph::DenseMatrix<double> dMdU_dUdsterm(raw_ndof,raw_ndof,0.0);
      oomph::DenseMatrix<double> dummy_dJdU_dUdsterm(raw_ndof,raw_ndof,0.0);            
      multi_assm.back().add_hessian(dUdsLocal, &dummy_dJdU_dUdsterm, &dMdU_dUdsterm);

      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=elem_pt->eqn_number(i);
          dof_backup[i]=*(alldofs[glob_eq]);          
          glob_eqs[i]=glob_eq;
      }            

      
      unsigned Teq=raw_ndof*this->n_tsteps();
      for (unsigned int ie=0;ie<this->basis->get_num_elements();ie++)
      {
          std::vector<double> w;
          std::vector<unsigned> indices;
          std::vector<std::vector<double>> psi_s;
          std::vector<std::vector<double>> dpsi_ds;
          unsigned nGL=this->basis->get_integration_info(ie,w,indices,psi_s,dpsi_ds);
          for (unsigned iGL=0;iGL<nGL;iGL++)
          {
            Ulocal.initialise(0.0);
            dUdsLocal.initialise(0.0);
            if (T_constraint_mode==1) 
            {
              dU0ds.initialise(0.0);
            }
            for (unsigned int psi_index=0;psi_index<indices.size();psi_index++)
            {
              std::vector<double> U_at_index(raw_ndof,0.0);
              if (indices[psi_index]==0) U_at_index=dof_backup;
              else 
              {
                for (unsigned int i=0;i<raw_ndof;i++)
                {
                  U_at_index[i]=Tadd[indices[psi_index]-1][glob_eqs[i]];
                }
              }
              // I guess this can be optimized and filled in a rotary buffer
              for (unsigned int i=0;i<raw_ndof;i++)
              {
                Ulocal[i]+=psi_s[iGL][psi_index]*U_at_index[i];
                dUdsLocal[i]+=dpsi_ds[iGL][psi_index]*U_at_index[i];
              }
              if (T_constraint_mode==1) 
              {
                for (unsigned int i=0;i<raw_ndof;i++)
                {
                  dU0ds[i]+=dpsi_ds[iGL][psi_index]*du0ds[indices[psi_index]][glob_eqs[i]];
                }            
              }
            }

            for (unsigned int i=0;i<raw_ndof;i++)
            {
              unsigned glob_eq=elem_pt->eqn_number(i);
              *(alldofs[glob_eq])=Ulocal[i]; // Set the unknowns
            }

            current_res.initialise(0.0);
            M.initialise(0.0);
            J.initialise(0.0);  
            if (has_constant_mass_matrix)
            {
              elem_pt->get_jacobian_and_mass_matrix(current_res, J, M);
            }
            else
            {            
              dMdU_dUdsterm.initialise(0.0);
              dummy_dJdU_dUdsterm.initialise(0.0);
              pyoomph_elem_pt->get_multi_assembly(multi_assm);
            }

            if (T_constraint_mode==1)
            {
              for (unsigned i = 0; i < raw_ndof; i++)
              {
                double fact=dU0ds[i]/Count[glob_eqs[i]]*w[iGL];
                residuals[raw_ndof*this->n_tsteps()]+=fact*Ulocal[i];
                for (unsigned int l2=0;l2<indices.size();l2++)
                {
                  unsigned ti2=indices[l2];  
                    jacobian(raw_ndof*this->n_tsteps(),ti2*raw_ndof+i)+=fact*psi_s[iGL][l2];
                }
              }
            }
            
            for (unsigned int l=0;l<indices.size();l++)
            {
              unsigned ti=indices[l];
              for (unsigned i = 0; i < raw_ndof; i++)
              {
                residuals[ti*raw_ndof + i] += current_res[i]*psi_s[iGL][l]*w[iGL];            
                for (unsigned j=0;j<raw_ndof;j++)          
                {                  
                  residuals[ti*raw_ndof+i]+=M(i,j)*dUdsLocal[j]/T*psi_s[iGL][l]*w[iGL];
                  //residuals[ti*raw_ndof+i]-=M(i,j)*Ulocal[j]/T*dpsi_ds[iGL][l]*w[iGL];
                  //residuals[ti*raw_ndof+i]+=M(i,j)*0.5*(dUdsLocal[j]*psi_s[iGL][l]-Ulocal[j]*dpsi_ds[iGL][l])*w[iGL]/T;                  
                  // and add the derivative with respect to T
                  //jacobian(ti*raw_ndof+i,Teq)+=-M(i,j)*0.5*(dUdsLocal[j]*psi_s[iGL][l]-Ulocal[j]*dpsi_ds[iGL][l])*w[iGL]/(T*T);                  
                  jacobian(ti*raw_ndof+i,Teq)+=-M(i,j)*dUdsLocal[j]*psi_s[iGL][l]*w[iGL]/(T*T);                  
                  
                }                
                for (unsigned int l2=0;l2<indices.size();l2++)
                {
                  unsigned ti2=indices[l2];
                  for (unsigned j=0;j<raw_ndof;j++)          
                  {                  
                    
                      jacobian(ti*raw_ndof+i,ti2*raw_ndof+j)+=J(i,j)*psi_s[iGL][l]*psi_s[iGL][l2]*w[iGL];
                      //jacobian(ti*raw_ndof+i,ti2*raw_ndof+j)+=M(i,j)*0.5*(dpsi_ds[iGL][l2]*psi_s[iGL][l]-psi_s[iGL][l2]*dpsi_ds[iGL][l])*w[iGL]/T;                  
                      jacobian(ti*raw_ndof+i,ti2*raw_ndof+j)+=M(i,j)*(dpsi_ds[iGL][l2]*psi_s[iGL][l])*w[iGL]/T;                  
                      if (!has_constant_mass_matrix)
                      {
                        jacobian(ti*raw_ndof+i,ti2*raw_ndof+j)+=dMdU_dUdsterm(i,j)/T*psi_s[iGL][l]*psi_s[iGL][l2]*w[iGL];
                      }                                                  
                  }
                }
              }
            }

          }
      }

    

      for (unsigned int i=0;i<raw_ndof;i++)
      {
        *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
      }

      if (this->T_constraint_mode==0)
      {
        double plane_eq=-d_plane;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=glob_eqs[i];
          double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
          plane_eq+=x*n0[glob_eq]/Count[glob_eq];
        }

        // Get the plane equation
        residuals[raw_ndof*this->n_tsteps()]=plane_eq;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=glob_eqs[i];
          jacobian(Teq,i)=n0[glob_eq]/Count[glob_eq];
        }
      }

  }

  void PeriodicOrbitHandler::update_phase_constraint_information()
  {
    if (T_constraint_mode==1)
    {
      unsigned ntsteps=this->n_tsteps();
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();
      if (!basis)
      {
        if (time_mesh)
        {
          du0ds.resize(ntsteps);        
        du0ds[0].resize(Ndof);
        for (unsigned int i=0;i<Ndof;i++)
        {
            du0ds[0][i]=*(alldofs[i]); // Just the U0 solution here, we do it via Gauss-Legendre in the residual/jacobian calculation
        }
        for (unsigned int ti=1;ti<ntsteps;ti++)
        {
          du0ds[ti].resize(Ndof);
          for (unsigned int i=0;i<Ndof;i++)
          {
            du0ds[ti][i]=Tadd[ti-1][i]; // Just the U0 solution here, we do it via Gauss-Legendre in the residual/jacobian calculation
          }
        }                
        }
        else if (floquet_mode)
        {

          du0ds.resize(ntsteps-1);
          du0ds[0].resize(Ndof);
          for (unsigned int i=0;i<Ndof;i++)
          {
              du0ds[0][i]=Tadd[0][i]-*(alldofs[i]); // Without 1/ds factor, since it will cancel out in the integral anyways
          }
          for (unsigned int ti=1;ti<ntsteps-1;ti++)
          {
            du0ds[ti].resize(Ndof);
            for (unsigned int i=0;i<Ndof;i++)
            {
              du0ds[ti][i]=Tadd[ti][i]-Tadd[ti-1][i]; // Without 1/ds factor, since it will cancel out in the integral anyways
            }
          }

        }
        else
        {
          du0ds.resize(ntsteps);
          for (unsigned int ti=0;ti<ntsteps;ti++)
          {          
            du0ds[ti].resize(Ndof,0.0);
            for (unsigned int ii=0;ii<this->FD_ds_inds[ti].size();ii++)
            {
              unsigned index=this->FD_ds_inds[ti][ii];
              if (index>0)
              {
                index--;
                for (unsigned int i=0;i<Ndof;i++)
                {              
                  du0ds[ti][i]+=this->FD_ds_weights[ti][ii]*Tadd[index][i];
                }
              }
              else
              {
                for (unsigned int i=0;i<Ndof;i++)
                {              
                  du0ds[ti][i]+=this->FD_ds_weights[ti][ii]* (*alldofs[i]);
                }

              }
            }
          }
        }
      }
      else
      {
        du0ds.resize(ntsteps);        
        du0ds[0].resize(Ndof);
        for (unsigned int i=0;i<Ndof;i++)
        {
            du0ds[0][i]=*(alldofs[i]); // Just the U0 solution here, we do it via Gauss-Legendre in the residual/jacobian calculation
        }
        for (unsigned int ti=1;ti<ntsteps;ti++)
        {
          du0ds[ti].resize(Ndof);
          for (unsigned int i=0;i<Ndof;i++)
          {
            du0ds[ti][i]=Tadd[ti-1][i]; // Just the U0 solution here, we do it via Gauss-Legendre in the residual/jacobian calculation
          }
        }                
      }
    }
  }

  void PeriodicOrbitHandler::get_jacobian_time_nodal_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
      //std::cout << "FD MODE STARTED " << std::endl;
      if (!Problem_pt->are_hessian_products_calculated_analytically())
      {
        throw_runtime_error("Cannot track periodic orbits without having analytical Hessian. Use Problem.setup_for_stability_analysis(analytic_hessian=True) before.");
      }
      residuals.initialise(0.0);
      jacobian.initialise(0.0);
      pyoomph::BulkElementBase * pyoomph_elem_pt=dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
      auto *ft=pyoomph_elem_pt->get_code_instance()->get_func_table();
      bool has_constant_mass_matrix=false;
      if (ft->current_res_jac>=0) 
      { 
        has_constant_mass_matrix=ft->has_constant_mass_matrix_for_sure[ft->current_res_jac];   
      }      
      /*if (!has_constant_mass_matrix)
      {
        throw_runtime_error("The mass matrix must be constant for the time being for periodic orbits.");
      }*/
      unsigned raw_ndof = elem_pt->ndof();
      DenseMatrix<double> J(raw_ndof), M(raw_ndof);            
      Vector<double> current_res(raw_ndof);      
      Vector<double> dof_backup(raw_ndof);            
      Vector<unsigned> glob_eqs(raw_ndof);
      unsigned Teq=raw_ndof*this->n_tsteps();
      oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();
      Vector<double> ddof_ds(raw_ndof,0.0);          

      std::vector<SinglePassMultiAssembleInfo> multi_assm;
      multi_assm.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &current_res, &J, &M));
      oomph::DenseMatrix<double> dMdU_dUdsterm(raw_ndof,raw_ndof,0.0);
      oomph::DenseMatrix<double> dummy_dJdU_dUdsterm(raw_ndof,raw_ndof,0.0);            
      multi_assm.back().add_hessian(ddof_ds, &dummy_dJdU_dUdsterm, &dMdU_dUdsterm);
      for (unsigned int i=0;i<raw_ndof;i++)
      {
          unsigned glob_eq=elem_pt->eqn_number(i);
          dof_backup[i]=*(alldofs[glob_eq]);          
          glob_eqs[i]=glob_eq;
      }            

      
      for (unsigned int ti=0;ti<this->n_tsteps();ti++)
      {          
          ddof_ds.initialise(0.0);
          for (unsigned int ii=0;ii<this->FD_ds_inds[ti].size();ii++)
          {
            unsigned index=this->FD_ds_inds[ti][ii];
            if (index>0)
            {
              index--;
              for (unsigned int i=0;i<raw_ndof;i++)
              {              
                ddof_ds[i]+=this->FD_ds_weights[ti][ii]*Tadd[index][glob_eqs[i]];
              }
            }
            else
            {
              for (unsigned int i=0;i<raw_ndof;i++)
              {              
                ddof_ds[i]+=this->FD_ds_weights[ti][ii]*dof_backup[i];
              }

            }
          }

          // Setup the dofs          
          if (ti>0)
          {
            for (unsigned int i=0;i<raw_ndof;i++)
            {
              unsigned glob_eq=elem_pt->eqn_number(i);
              *(alldofs[glob_eq])=Tadd[ti-1][glob_eq];              
            }
          }     
          else
          {
            for (unsigned int i=0;i<raw_ndof;i++)
            {
              unsigned glob_eq=elem_pt->eqn_number(i);
              *(alldofs[glob_eq])=dof_backup[i];
            }
          }       
          current_res.initialise(0.0);
          M.initialise(0.0);
          J.initialise(0.0);  
          if (has_constant_mass_matrix)
          {
            elem_pt->get_jacobian_and_mass_matrix(current_res, J, M);                      
          }
          else
          {            
            dMdU_dUdsterm.initialise(0.0);
            dummy_dJdU_dUdsterm.initialise(0.0);
            pyoomph_elem_pt->get_multi_assembly(multi_assm);
          }
          for (unsigned i = 0; i < raw_ndof; i++)
          {
            residuals[ti*raw_ndof + i] += current_res[i];
            
            for (unsigned j=0;j<raw_ndof;j++)          
            {
              jacobian(ti*raw_ndof+i,ti*raw_ndof+j)+=J(i,j); // Purely diagonal jacobian blocks here
              residuals[ti*raw_ndof+i]+=M(i,j)*ddof_ds[j]/T;                                     
              // and add the derivative with respect to T
              jacobian(ti*raw_ndof+i,Teq)+=-M(i,j)*ddof_ds[j]/(T*T);
              if (!has_constant_mass_matrix)
              {
                  jacobian(ti*raw_ndof+i,ti*raw_ndof+j)+=dMdU_dUdsterm(i,j)/T;               
              }     
            }

            //std::cout << "INDICES SIZE AT " << ti << "  " << this->FD_ds_inds[ti].size() << std::endl;
            for (unsigned int ii=0;ii<this->FD_ds_inds[ti].size();ii++)
            {
                unsigned index=this->FD_ds_inds[ti][ii];

                for (unsigned int j=0;j<raw_ndof;j++)
                {
                  jacobian(ti*raw_ndof+i,index*raw_ndof+j)+=M(i,j)*this->FD_ds_weights[ti][ii]/T;
                  //if (M(i,j)!=0) std::cout << " ADDING TO dR^"<<ti << "_"<<i << " / dU^"<<index << "_"<<j << " " << M(i,j)*this->FD_ds_weights[ti][ii]/T << std::endl;
                }
                
            }

          }

          if (T_constraint_mode==1)
          {
            double ds=0.5*(this->get_knot_value(ti+1)-this->get_knot_value(ti-1));
            for (unsigned int i=0;i<raw_ndof;i++)
            {
              residuals[raw_ndof*this->n_tsteps()]+=du0ds[ti][glob_eqs[i]]*(*(alldofs[glob_eqs[i]]))/Count[glob_eqs[i]]*ds;
              jacobian(raw_ndof*this->n_tsteps(),ti*raw_ndof+i)+=du0ds[ti][glob_eqs[i]]/Count[glob_eqs[i]]*ds;            
            }          
          }                  
      }

      for (unsigned int i=0;i<raw_ndof;i++)
      {
        *(this->Problem_pt->GetDofPtr()[glob_eqs[i]])=dof_backup[i];
      }

      if (T_constraint_mode==0)
      {
        double plane_eq=-d_plane;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=glob_eqs[i];
          double x=*(this->Problem_pt->GetDofPtr()[glob_eq]);
          plane_eq+=x*n0[glob_eq]/Count[glob_eq];
        }

        // Get the plane equation
        residuals[raw_ndof*this->n_tsteps()]=plane_eq;
        for (unsigned int i=0;i<raw_ndof;i++)
        {
          unsigned glob_eq=glob_eqs[i];
          jacobian(Teq,i)=n0[glob_eq]/Count[glob_eq];
        }
      }
  }

  void PeriodicOrbitHandler::get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
    unsigned raw_ndof=elem_pt->ndof();
    if (!raw_ndof) {
      residuals.initialise(0.0); 
      jacobian.initialise(0.0);
      return;
    }
    if (!basis)
    {
      if (time_mesh)
      {
        this->get_jacobian_collocation_mode(elem_pt,residuals,jacobian);
        /*
        residuals.initialise(0.0);
        this->get_residuals(elem_pt, residuals);
        unsigned raw_ndof=elem_pt->ndof();
        unsigned tot_ndof=residuals.size();
        oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();
        for (unsigned int i=0;i<tot_ndof;i++)
        {
          //unsigned glob_eq=this->eqn_number(elem_pt,i);
          unsigned glob_eq;
          unsigned tindex=i/raw_ndof;
          unsigned iindex=i%raw_ndof;
          if (i==tot_ndof-1) glob_eq=T_global_eqn;
          else glob_eq=tindex*Ndof+elem_pt->eqn_number(iindex);
          //std::cout << "GLOB EQN " << glob_eq << " of " << tot_ndof << std::endl;
          double backup=*(alldofs[glob_eq]);
          double eps=1e-8;
          *(alldofs[glob_eq])=backup+eps;
          //std::cout << "GLOB EQN " << glob_eq << " of " << tot_ndof << " BACKUP " << backup << " NEW " << *(alldofs[glob_eq]) << " PTR COMPARISON ";
          //if (i<raw_ndof) " BASE "; else if(glob_eq==T_global_eqn) std::cout << " T_PERIOD "; else std::cout << " TADD " << &(Tadd[tindex-1][elem_pt->eqn_number(iindex)]) << " VS " << alldofs[glob_eq]<< std::endl;
          oomph::Vector<double> res_p(raw_ndof*this->n_tsteps()+1,0.0);
          this->get_residuals(elem_pt, res_p);
          for (unsigned int j=0;j<tot_ndof;j++)
          {
            jacobian(j,i)=(res_p[j]-residuals[j])/eps;
          }
          *(alldofs[glob_eq])=backup;         
        }
         */
      }
      else 
      { 
        if (floquet_mode) 
        {
          this->get_jacobian_floquet_mode(elem_pt,residuals,jacobian);      
        } 
        else 
        {
          this->get_jacobian_time_nodal_mode(elem_pt,residuals,jacobian);      
        //return;
        }
      }
    }
    else
    {
      this->get_jacobian_bspline_mode(elem_pt,residuals,jacobian);
      //return;
    }
    // This is just there to debug (comment the return statements for comparison with FD), remove if everything works
#ifdef PYOOMPH_BIFURCATION_HANDLER_DEBUG
    residuals.initialise(0.0);
    this->get_residuals(elem_pt, residuals);
    unsigned tot_ndof=residuals.size();
    oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();
    for (unsigned int i=0;i<tot_ndof;i++)
    {
      //unsigned glob_eq=this->eqn_number(elem_pt,i);
      unsigned glob_eq;
      unsigned tindex=i/raw_ndof;
      unsigned iindex=i%raw_ndof;
      if (i==tot_ndof-1) glob_eq=T_global_eqn;
      else glob_eq=tindex*Ndof+elem_pt->eqn_number(iindex);
      //std::cout << "GLOB EQN " << glob_eq << " of " << tot_ndof << std::endl;
      double backup=*(alldofs[glob_eq]);
      double eps=1e-8;
      *(alldofs[glob_eq])=backup+eps;
      //std::cout << "GLOB EQN " << glob_eq << " of " << tot_ndof << " BACKUP " << backup << " NEW " << *(alldofs[glob_eq]) << " PTR COMPARISON ";
      //if (i<raw_ndof) " BASE "; else if(glob_eq==T_global_eqn) std::cout << " T_PERIOD "; else std::cout << " TADD " << &(Tadd[tindex-1][elem_pt->eqn_number(iindex)]) << " VS " << alldofs[glob_eq]<< std::endl;
      oomph::Vector<double> res_p(raw_ndof*this->n_tsteps()+1,0.0);
      this->get_residuals(elem_pt, res_p);
      for (unsigned int j=0;j<tot_ndof;j++)
      {
        jacobian(j,i)=(res_p[j]-residuals[j])/eps;
      }
      *(alldofs[glob_eq])=backup;

    }



    // Check the Jacobian
      oomph::DenseMatrix<double> ana_J(raw_ndof*this->n_tsteps()+1,raw_ndof*this->n_tsteps()+1,0.0);
      oomph::Vector<double> ana_res(raw_ndof*this->n_tsteps()+1,0.0);
      if (!basis)
      {
        if (time_mesh)
        {
          this->get_jacobian_collocation_mode(elem_pt,ana_res,ana_J);
        }
        else {
          if (floquet_mode)
          {
            this->get_jacobian_floquet_mode(elem_pt,ana_res,ana_J);
          }
          else 
          {
            this->get_jacobian_time_nodal_mode(elem_pt,ana_res,ana_J);
          }
        }
      }
      else
      {
        this->get_jacobian_bspline_mode(elem_pt,ana_res,ana_J);
      }
      for (unsigned int i=0;i<raw_ndof*this->n_tsteps()+1;i++)
      {
        if (std::fabs(ana_res[i]-residuals[i])>1e-9)
        {
          std::cout << "RESIDUAL MISMATCH " << i << " " << ana_res[i] << " " << residuals[i] << std::endl;
          std::cout << " raw_ndof " <<  raw_ndof<< " n_tsteps " << this->n_tsteps()  << " TOT DOF " << tot_ndof << " VS " << raw_ndof*this->n_tsteps()+1<< std::endl;
        }
        for (unsigned int j=0;j<raw_ndof*this->n_tsteps()+1;j++)
        {
          if (std::fabs(ana_J(i,j)-jacobian(i,j))>1e-4)
          {
            unsigned glob_eq;
            unsigned tindex=i/raw_ndof;
            unsigned iindex=i%raw_ndof;
            if (i==tot_ndof-1) glob_eq=T_global_eqn;
            else glob_eq=tindex*Ndof+elem_pt->eqn_number(iindex);

            unsigned glob_eq2;
            unsigned tindex2=j/raw_ndof;
            unsigned iindex2=j%raw_ndof;
            if (j==tot_ndof-1) glob_eq2=T_global_eqn;
            else glob_eq2=tindex2*Ndof+elem_pt->eqn_number(iindex2);

            std::cout << "JACOBIAN MISMATCH " << i << " " << j << " " << ana_J(i,j) << " " << jacobian(i,j) << " this is ";
            if (glob_eq!=T_global_eqn) std::cout << "dR^{"<<tindex<<"}_{"<<iindex<<"} / "; else std::cout << "dT_period / ";
            if (glob_eq2!=T_global_eqn) std::cout <<"dU^{"<<tindex2<<"}_{"<<iindex2<<"}" ; else std::cout << "dT_period";
            std::cout << std::endl;
            std::cout << " raw_ndof " <<  raw_ndof<< " n_tsteps " << this->n_tsteps()  << " TOT DOF " << tot_ndof << " VS " << raw_ndof*this->n_tsteps()+1<< std::endl;
          }

          jacobian(i,j)=ana_J(i,j);
        }
        
      }

#endif   
  }

  void PeriodicOrbitHandler::backup_dofs()
  {
    oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();
    if (backed_up_dofs.size()) throw_runtime_error("The dofs have already been backed up. Likely, you try have a nested loop over the periodic orbit samples, which is not supported (or you forget to call restore_dofs() after a loop)");
    backed_up_dofs.resize(Ndof);
    for (unsigned int i=0;i<Ndof;i++)
    {
      backed_up_dofs[i]=*(alldofs[i]);
    }
    /*if (this->floquet_mode)
    {
      for (unsigned int i=0;i<Ndof;i++) std::cout << " LOOP CHECK " << i << "  " << backed_up_dofs[i] << " vs " << Tadd.back()[i] << " DIFFERENCE " << (backed_up_dofs[i]-Tadd.back()[i])*10000<< "  PTSRS " << alldofs[i] << " vs " << &(Tadd.back()[i]) << std::endl;
    }*/
  }
  void PeriodicOrbitHandler::restore_dofs()
  {
    if (backed_up_dofs.size()!=Ndof) throw_runtime_error("The dofs have not been backed up");
    oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();
    for (unsigned int i=0;i<Ndof;i++)
    {
      *(alldofs[i])=backed_up_dofs[i];
    }
    backed_up_dofs.resize(0); // Clear the backup
  }
  void PeriodicOrbitHandler::set_dofs_to_interpolated_values(const double &s)
  {
    if (backed_up_dofs.size()!=Ndof) throw_runtime_error("The dofs have not been backed up");
    double clamped_s=s-floor(s);
    oomph::Vector<double *> & alldofs=this->Problem_pt->GetDofPtr();
    unsigned start=0;
    while (s_knots[start+1]<clamped_s) start++;
    if (!basis)
    {
      
      double lambda=(clamped_s-s_knots[start])/(s_knots[start+1]-s_knots[start]);
      //std::cout << "AT " << clamped_s << " START " << start << " S_KNOTS " << s_knots[start] << " " << s_knots[start+1] << " TADD SIZE " << Tadd.size() << " LAMBDA " << lambda << " T =" << clamped_s*T << std::endl;
      if (start==0)
      {
        for (unsigned int i=0;i<Ndof;i++)
        {
          *(alldofs[i])=(1-lambda)*backed_up_dofs[i]+Tadd[start][i]*lambda;
        }
      }
      else if (start>=Tadd.size())
      {
        for (unsigned int i=0;i<Ndof;i++)
        {
          *(alldofs[i])=lambda*backed_up_dofs[i]+Tadd.back()[i]*(1-lambda);
        }
      }
      else
      {
        for (unsigned int i=0;i<Ndof;i++)
        {
          *(alldofs[i])=(1-lambda)*Tadd[start-1][i]+Tadd[start][i]*lambda;
        }
      }
    }
    else
    {
      std::vector<unsigned> indices;
      std::vector<double> psi;
      unsigned numsupport=basis->get_interpolation_info(clamped_s,indices,psi);
      for (unsigned int i=0;i<Ndof;i++)
      {
          *(alldofs[i])=0.0;
      }
      for (unsigned int iindex=0;iindex<numsupport;iindex++)
      {
        if (indices[iindex]==0)
        {
          for (unsigned int i=0;i<Ndof;i++)
          {
            *(alldofs[i])+=psi[iindex]*backed_up_dofs[i];
          }
        }
        else
        {
          for (unsigned int i=0;i<Ndof;i++)
          {
            *(alldofs[i])+=psi[iindex]*Tadd[indices[iindex]-1][i];
          }
        }
      }
    }
  }

  std::vector<std::tuple<double,double>> PeriodicOrbitHandler::get_s_integration_samples()
  {
    std::vector<std::tuple<double,double>> samples;
    if (!basis)
    {
      if (time_mesh)
      {
        for (unsigned int i=0;i<s_knots.size()-1;i++)
        {
           samples.push_back(std::make_tuple((s_knots[i]+s_knots[i+1])/2,s_knots[i+1]-s_knots[i])); //TODO: Improve here
        }
      }
      else if (floquet_mode)
      {
        for (unsigned int i=0;i<s_knots.size()-1;i++)
        {
          samples.push_back(std::make_tuple((s_knots[i]+s_knots[i+1])/2,s_knots[i+1]-s_knots[i]));
        }        
      }
      else
      {        
        for (unsigned int i=0;i<s_knots.size()-1;i++)
        {
          samples.push_back(std::make_tuple(s_knots[i],0.5*(this->get_knot_value(i+1)-this->get_knot_value(i-1))));
        }  
      }
    }
    else
    {
      for (unsigned int ie=0;ie<this->basis->get_num_elements();ie++)
      {
        std::vector<double> w;
        std::vector<unsigned> indices;
        std::vector<std::vector<double>> psi_s;
        std::vector<std::vector<double>> dpsi_ds;
        unsigned nGL=this->basis->get_integration_info(ie,w,indices,psi_s,dpsi_ds);
        for (unsigned int iGL=0;iGL<nGL;iGL++)
        {
          double s=0;
          for (unsigned int is=0;is<indices.size();is++)
          {
            s+=psi_s[iGL][is]*s_knots[indices[is]];
          }
          samples.push_back(std::make_tuple(s,w[iGL]));
        }
      }    
    }
    return samples;
  }


  void PeriodicOrbitHandler::get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt,oomph::Vector<double> &dres_dparam)
  {
    unsigned raw_ndof=elem_pt->ndof();
    if (!raw_ndof) {dres_dparam.initialise(0.0); return;}
      if (!basis)
      {
        if (time_mesh) this->get_residuals_collocation_mode(elem_pt,dres_dparam,parameter_pt);     
        else if (floquet_mode) this->get_residuals_floquet_mode(elem_pt,dres_dparam,parameter_pt);     
        else this->get_residuals_time_nodal_mode(elem_pt,dres_dparam,parameter_pt);        
      } 
      else this->get_residuals_bspline_mode(elem_pt,dres_dparam,parameter_pt);    
  }


  void PeriodicOrbitHandler::get_djacobian_dparameter(GeneralisedElement *const &elem_pt,double *const &parameter_pt,Vector<double> &dres_dparam,DenseMatrix<double> &djac_dparam)
  {
    throw_runtime_error("Not implemented");

  }



//===================================================================  
  //===================================================================

  CustomMultiAssembleHandler::CustomMultiAssembleHandler(Problem *const &problem_pt,std::vector<std::string> & _what,std::vector<std::string> & _contributions,std::vector<std::string> & _params,std::vector<std::vector<double>> & _hessian_vectors,std::vector<unsigned> & _hessian_vector_indices,std::vector<int> & return_indices) : problem(problem_pt), what(_what), contributions(_contributions), params(_params), hessian_vectors(_hessian_vectors) 
  {
    if (what.size()!=contributions.size()) throw_runtime_error("The what and contributions vectors must have the same size");
    parameters.resize(what.size(),NULL);

    unsigned pindex=0;
    for (unsigned int i=0;i<what.size();i++)
    {
      if (what[i]=="dresiduals_dparameter" || what[i]=="djacobian_dparameter" || what[i]=="dmass_matrix_dparameter")
      {
        if (pindex>=params.size()) throw_runtime_error("You have not provided enough parameters for the what type '"+what[i]+"'");
        pyoomph::GlobalParameterDescriptor * parameter=problem->get_global_parameter(params[pindex]);
        parameters[i]=&parameter->value();
        pindex++;
      }      
    }
    if (pindex!=params.size()) throw_runtime_error("You have provided too many parameters");

    unsigned hvindex=0;
    hessian_vector_indices.resize(what.size(),-1);
    hessian_vector_transposed.resize(what.size(),false);
    this->transposed_hessians=false;
    bool has_nontransposed=false;
    for (unsigned int i=0;i<what.size();i++)
    {
      if (what[i]=="hessian_vector_product" || what[i]=="mass_matrix_hessian_vector_product" || what[i]=="hessian_vector_product_transposed" || what[i]=="mass_matrix_hessian_vector_product_transposed")
      {
        if (hvindex>=_hessian_vector_indices.size()) throw_runtime_error("You have not provided enough hessian vector indices for the what type '"+what[i]+"'");
        hessian_vector_indices[i]=_hessian_vector_indices[hvindex];
        hessian_vector_transposed[i]=(what[i]=="hessian_vector_product_transposed" || what[i]=="mass_matrix_hessian_vector_product_transposed");
        if (hessian_vector_transposed[i])
        {
          if (has_nontransposed) throw_runtime_error("Cannot assemble transposed and non-transposed Hessian vector products simultaneously");
          this->transposed_hessians=true;
        }
        else
        {
          if (this->transposed_hessians) throw_runtime_error("Cannot assemble transposed and non-transposed Hessian vector products simultaneously");
          has_nontransposed=true;          
        }
        if (hessian_vector_indices[i]>=hessian_vectors.size()) throw_runtime_error("Hessian vector index out of bounds");
        hvindex++;
      }      
    }
    if (hvindex!=_hessian_vector_indices.size()) throw_runtime_error("You have provided too many hessian vector indices");

    for (unsigned int i=0;i<contributions.size();i++)
    {
      bool found=false;
      for (unsigned int j=0;j<unique_contributions.size();j++)
      {
        if (unique_contributions[j]==contributions[i])
        {
          found=true;
          break;
        }
      }
      if (!found) unique_contributions.push_back(contributions[i]);
    }
    setup_residual_contribution_map();
    contribution_return_indices.resize(unique_contributions.size());
    nvector=0;
    nmatrix=0;
    for (unsigned int i=0;i<unique_contributions.size();i++)
    {
      std::string &contribution=unique_contributions[i];
      for (unsigned int j=0;j<contributions.size();j++)
      {
        if (contributions[j]==contribution)
        {
          // Now this has to be handled
          if (what[j]=="residuals")
          {
            if (contribution_return_indices[i].residual_index!=-1) throw_runtime_error("You have multiple residual requests for the same contribution '"+contribution+"'");
            contribution_return_indices[i].residual_index=nvector++;
          }
          else if (what[j]=="jacobian")
          {
            if (contribution_return_indices[i].jacobian_index!=-1) throw_runtime_error("You have multiple jacobian requests for the same contribution '"+contribution+"'");
            contribution_return_indices[i].jacobian_index=nmatrix++;
          }
          else if (what[j]=="mass_matrix")
          {
            if (contribution_return_indices[i].mass_matrix_index!=-1) throw_runtime_error("You have multiple mass matrix requests for the same contribution '"+contribution+"'");
            contribution_return_indices[i].mass_matrix_index=nmatrix++;
          }
          else if (what[j]=="dresiduals_dparameter")
          {
            if (parameters[j]==NULL) throw_runtime_error("You have not provided a parameter for the what type '"+what[j]+"'");
            if (!contribution_return_indices[i].paramderivs.count(parameters[j])) contribution_return_indices[i].paramderivs[parameters[j]]=CustomMultiAssembleReturnIndexInfo();
            if (contribution_return_indices[i].paramderivs[parameters[j]].residual_index!=-1) throw_runtime_error("You have multiple dresiduals_dparameter requests for the same parameter and contribution '"+contribution+"'");
            contribution_return_indices[i].paramderivs[parameters[j]].residual_index=nvector++;            
          }
          else if (what[j]=="djacobian_dparameter")
          {
            if (parameters[j]==NULL) throw_runtime_error("You have not provided a parameter for the what type '"+what[j]+"'");
            if (!contribution_return_indices[i].paramderivs.count(parameters[j])) contribution_return_indices[i].paramderivs[parameters[j]]=CustomMultiAssembleReturnIndexInfo();
            if (contribution_return_indices[i].paramderivs[parameters[j]].jacobian_index!=-1) throw_runtime_error("You have multiple djacobian_dparameter requests for the same parameter and contribution '"+contribution+"'");
            contribution_return_indices[i].paramderivs[parameters[j]].jacobian_index=nmatrix++;            
          }
          else if (what[j]=="dmass_matrix_dparameter")
          {
            if (parameters[j]==NULL) throw_runtime_error("You have not provided a parameter for the what type '"+what[j]+"'");
            if (!contribution_return_indices[i].paramderivs.count(parameters[j])) contribution_return_indices[i].paramderivs[parameters[j]]=CustomMultiAssembleReturnIndexInfo();
            if (contribution_return_indices[i].paramderivs[parameters[j]].mass_matrix_index!=-1) throw_runtime_error("You have multiple dmass_matrix_dparameter requests for the same parameter and contribution '"+contribution+"'");
            contribution_return_indices[i].paramderivs[parameters[j]].mass_matrix_index=nmatrix++;            
          }
          else if (what[j]=="hessian_vector_product" || what[j]=="hessian_vector_product_transposed")
          {
            if (hessian_vector_indices[j]<0) throw_runtime_error("You have not provided a hessian vector index for what type of '"+what[j]+"'");
            std::tuple<int,bool> hindex=std::make_tuple(hessian_vector_indices[j],hessian_vector_transposed[j]);
            if (!contribution_return_indices[i].hessians.count(hindex)) contribution_return_indices[i].hessians[hindex]=CustomMultiAssembleReturnIndexInfo();
            if (contribution_return_indices[i].hessians[hindex].jacobian_index!=-1) throw_runtime_error("You have multiple hessian requests for the same vector and contribution '"+contribution+"'");
            contribution_return_indices[i].hessians[hindex].jacobian_index=nmatrix++;                        
          }
          else if (what[j]=="mass_matrix_hessian_vector_product" || what[j]=="mass_matrix_hessian_vector_product_transposed")
          {
            if (hessian_vector_indices[j]<0) throw_runtime_error("You have not provided a hessian vector index for what type of '"+what[j]+"'");
            std::tuple<int,bool> hindex=std::make_tuple(hessian_vector_indices[j],hessian_vector_transposed[j]);
            if (!contribution_return_indices[i].hessians.count(hindex)) contribution_return_indices[i].hessians[hindex]=CustomMultiAssembleReturnIndexInfo();
            if (contribution_return_indices[i].hessians[hindex].mass_matrix_index!=-1) throw_runtime_error("You have multiple hessian mass matrix requests for the same vector and contribution '"+contribution+"'");
            contribution_return_indices[i].hessians[hindex].mass_matrix_index=nmatrix++;            
            contribution_return_indices[i].hessian_require_mass_matrix=true;
          }
          else
          {
            throw_runtime_error("Unknown what type '"+what[j]+"'");
          }
        }
      }
      for (auto & hc : contribution_return_indices[i].hessians) contribution_return_indices[i].hessian_vector_indices.push_back(std::get<0>(hc.first));
    }


    if (nmatrix+nvector!=what.size()) throw_runtime_error("Something went wrong here");
    return_indices.resize(what.size());
    for (unsigned int i=0;i<what.size();i++)
    {
      unsigned uci=0;
      for (unsigned int j=0;j<unique_contributions.size();j++) if (contributions[i]==unique_contributions[j]) {uci=j; break;}
      if (what[i]=="residuals") return_indices[i]=contribution_return_indices[uci].residual_index;
      else if (what[i]=="jacobian") return_indices[i]=-1-contribution_return_indices[uci].jacobian_index;
      else if (what[i]=="mass_matrix") return_indices[i]=-1-contribution_return_indices[uci].mass_matrix_index;
      else if (what[i]=="dresiduals_dparameter") return_indices[i]=contribution_return_indices[uci].paramderivs[parameters[i]].residual_index;
      else if (what[i]=="djacobian_dparameter") return_indices[i]=-1-contribution_return_indices[uci].paramderivs[parameters[i]].jacobian_index;
      else if (what[i]=="dmass_matrix_dparameter") return_indices[i]=-1-contribution_return_indices[uci].paramderivs[parameters[i]].mass_matrix_index;
      else if (what[i]=="hessian_vector_product") return_indices[i]=-1-contribution_return_indices[uci].hessians[std::make_tuple(hessian_vector_indices[i],false)].jacobian_index;
      else if (what[i]=="hessian_vector_product_transposed") return_indices[i]=-1-contribution_return_indices[uci].hessians[std::make_tuple(hessian_vector_indices[i],true)].jacobian_index;      
      else if (what[i]=="mass_matrix_hessian_vector_product") return_indices[i]=-1-contribution_return_indices[uci].hessians[std::make_tuple(hessian_vector_indices[i],false)].mass_matrix_index;
      else if (what[i]=="mass_matrix_hessian_vector_product_transposed") return_indices[i]=-1-contribution_return_indices[uci].hessians[std::make_tuple(hessian_vector_indices[i],true)].mass_matrix_index;
      else throw_runtime_error("should never arrive here")      ;      
    }
    

  }

  unsigned CustomMultiAssembleHandler::ndof(oomph::GeneralisedElement* const& elem_pt)
  {
    return elem_pt->ndof();
  }


  unsigned long CustomMultiAssembleHandler::eqn_number(oomph::GeneralisedElement* const& elem_pt, const unsigned& ieqn_local)
  {
    return elem_pt->eqn_number(ieqn_local);
  }
  
  void CustomMultiAssembleHandler::get_residuals(oomph::GeneralisedElement* const& elem_pt,Vector<double>& residuals)
  {
    throw_runtime_error("Residual called");
  }

  void CustomMultiAssembleHandler::get_jacobian(oomph::GeneralisedElement* const& elem_pt,oomph::Vector<double>& residuals,oomph::DenseMatrix<double>& jacobian)
  {
    throw_runtime_error("Jacobian called");
  }

  void CustomMultiAssembleHandler::setup_residual_contribution_map()
  {
    pyoomph::Problem *prob = dynamic_cast<pyoomph::Problem *>(problem);
    if (!prob)
      throw_runtime_error("Not a pyoomph::Problem... Strange");
    auto codes = prob->get_bulk_element_codes();
    for (unsigned int i = 0; i < codes.size(); i++)
    {
      int orig_residual = codes[i]->get_func_table()->current_res_jac; // Store the initial residual (base state)
      std::vector<int> indices(unique_contributions.size(),-1);      
      for (unsigned int ui=0;ui<unique_contributions.size();ui++)
      {
        if (codes[i]->_set_solved_residual(unique_contributions[ui]))
        {
          indices[ui] = codes[i]->get_func_table()->current_res_jac;
        }
      }      
      codes[i]->get_func_table()->current_res_jac = orig_residual; // Reset it
      residual_contribution_indices[codes[i]] = CustomMultiAssembleHandlerContributionList(codes[i], indices);
    }
    // Check whether we have an entirely empty contribution
    for (unsigned int i=0;i<unique_contributions.size();i++)
    {
      bool found=false;
      for (auto it=residual_contribution_indices.begin();it!=residual_contribution_indices.end();it++)
      {
        if (it->second.residual_indices[i]>=0)
        {
          found=true;
          break;
        }
      }
      if (!found) throw_runtime_error("You want to assemble a contribution '"+unique_contributions[i]+ "' that is not present in the problem at all");
    }
  }

  int CustomMultiAssembleHandler::resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_index)
  {
    pyoomph::BulkElementBase *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el)
    {
      throw_runtime_error("Strange, not a pyoomph element");
    }
    auto *const_code = el->get_code_instance()->get_code();
    if (!residual_contribution_indices.count(const_code))
    {
      throw_runtime_error("You have not set up your residual contribution mapping in beforehand");
    }
    auto &entry = residual_contribution_indices[const_code];
    return entry.residual_indices[residual_index];
  }

  void CustomMultiAssembleHandler::get_all_vectors_and_matrices(oomph::GeneralisedElement* const& elem_pt,oomph::Vector<oomph::Vector<double>>& vec,oomph::Vector<oomph::DenseMatrix<double>>& matrix)
  {
    unsigned n_var = elem_pt->ndof();    
    oomph::Vector<double> dummyV(n_var);    
    oomph::DenseMatrix<double> dummyM(n_var);    
    std::vector<SinglePassMultiAssembleInfo> multi_assm;
    oomph::Vector<double> hessian_vec_local(hessian_vectors.size()*n_var);
    std::vector<oomph::DenseMatrix<double>> hessian_Js(unique_contributions.size());
    std::vector<oomph::DenseMatrix<double>> hessian_Ms(unique_contributions.size());

    for (unsigned int i=0;i<vec.size();i++) vec[i].initialise(0.0);
    for (unsigned int i=0;i<matrix.size();i++) matrix[i].initialise(0.0);
    
    pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    bool has_contribs=false;

    // Fill the Hessian local vector
    for (unsigned int ih=0;ih<hessian_vectors.size();ih++)
    {
      for (unsigned int iloc=0;iloc<n_var;iloc++)
      {
        unsigned globeq=elem_pt->eqn_number(iloc);
        hessian_vec_local[ih*n_var+iloc]=hessian_vectors[ih][globeq];
      }
    }

    for (unsigned int contribution_index=0;contribution_index<unique_contributions.size();contribution_index++)
    {
      int resindex;
      if ((resindex = this->resolve_assembled_residual(pyoomph_elem_pt, contribution_index)) >= 0)
      {
        has_contribs=true;
        oomph::Vector<double> *residuals=(contribution_return_indices[contribution_index].residual_index>=0 ? &vec[contribution_return_indices[contribution_index].residual_index] : &dummyV);
        oomph::DenseMatrix<double> *jacobian=(contribution_return_indices[contribution_index].jacobian_index>=0 ? &matrix[contribution_return_indices[contribution_index].jacobian_index] : NULL);
        oomph::DenseMatrix<double> *mass_matrix=(contribution_return_indices[contribution_index].mass_matrix_index>=0 ? &matrix[contribution_return_indices[contribution_index].mass_matrix_index] : NULL);
        if (!jacobian && mass_matrix) jacobian=&dummyM;
        multi_assm.push_back(SinglePassMultiAssembleInfo(resindex, residuals, jacobian, mass_matrix));
        for (auto & paraminfo : contribution_return_indices[contribution_index].paramderivs)
        {
          residuals=(paraminfo.second.residual_index>=0 ? &vec[paraminfo.second.residual_index] : &dummyV);
          jacobian=(paraminfo.second.jacobian_index>=0 ? &matrix[paraminfo.second.jacobian_index] : NULL);
          mass_matrix=(paraminfo.second.mass_matrix_index>=0 ? &matrix[paraminfo.second.mass_matrix_index] : NULL);
          if (!jacobian && mass_matrix) jacobian=&dummyM;
          multi_assm.back().add_param_deriv(paraminfo.first, residuals,jacobian,mass_matrix);
        }
        if (!contribution_return_indices[contribution_index].hessian_vector_indices.empty())
        {
          hessian_Js[contribution_index].resize(hessian_vectors.size()*n_var,n_var,0.0);
          if (contribution_return_indices[contribution_index].hessian_require_mass_matrix)
          {
            hessian_Ms[contribution_index].resize(hessian_vectors.size()*n_var,n_var,0.0);
            multi_assm.back().add_hessian(hessian_vec_local, &hessian_Js[contribution_index], &hessian_Ms[contribution_index],this->transposed_hessians);            
          }
          else
          {
            multi_assm.back().add_hessian(hessian_vec_local, &hessian_Js[contribution_index], NULL,this->transposed_hessians);            
          }          
        }        
      }
    }
    if (!has_contribs) return;
    pyoomph_elem_pt->get_multi_assembly(multi_assm);

    if (hessian_vectors.size())
    {
      for (unsigned int contribution_index=0;contribution_index<unique_contributions.size();contribution_index++)
      {
        int resindex;
        if ((resindex = this->resolve_assembled_residual(pyoomph_elem_pt, contribution_index)) >= 0)
        {
          for (auto & hessinfo: contribution_return_indices[contribution_index].hessians)
          {
            if (hessinfo.second.jacobian_index>=0)
            {
              for (unsigned int i=0;i<n_var;i++)
              {
                for (unsigned int j=0;j<n_var;j++)
                {
                  matrix[hessinfo.second.jacobian_index](i,j)=hessian_Js[contribution_index](std::get<0>(hessinfo.first)*n_var+ i,j);
                }
              }
            }
            if (hessinfo.second.mass_matrix_index>=0)
            {
              for (unsigned int i=0;i<n_var;i++)
              {
                for (unsigned int j=0;j<n_var;j++)
                {
                  matrix[hessinfo.second.mass_matrix_index](i,j)=hessian_Ms[contribution_index](std::get<0>(hessinfo.first)*n_var+ i,j);
                }
              }
            }
          }                    
        }
      }
    }
  }

}


