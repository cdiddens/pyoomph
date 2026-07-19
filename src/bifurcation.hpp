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

/*
##################################
This file is strongly based  on the oomph-lib library (see thirdparty/oomph-lib/include/assembly_handler.h)
##################################
*/

#pragma once

#include "assembly_handler.h"
#include <set>
#include <complex>
#include "mesh.h"

namespace pyoomph
{
  class Problem;
  class DynamicBulkElementCode; // Forward decl.

  // Reimplementation of the Hopf-Handler with some changes
  // Assembly handler that augments the original residuals R(u,p) (p being the bifurcation
  // parameter) with the equations tracking a Hopf bifurcation, i.e. a pair of complex
  // conjugate eigenvalues +-i*Omega crossing the imaginary axis. The augmented unknown
  // vector consists of the original dofs u, the real and imaginary parts of the complex
  // null eigenvector of (J + i*Omega*M) (stored interleaved as Phi/Psi), and the frequency
  // Omega itself. The augmented system solved is, schematically,
  //   R(u,p)                       = 0                     (original residuals)
  //   J(u,p)*Phi - Omega*M(u,p)*Psi = 0                     (real part of eigenproblem)
  //   J(u,p)*Psi + Omega*M(u,p)*Phi = 0                     (imag part of eigenproblem)
  //   C . Phi - 1                   = 0                     (normalization, prevents Phi=Psi=0)
  //   C . Psi                       = 0
  // where C is a fixed vector picked so the normalization equations are non-degenerate.
  class MyHopfHandler : public oomph::AssemblyHandler
  {
  protected:
    unsigned Solve_which_system;   // Selects which sub-block of the augmented system is currently assembled (full, standard, or complex block; see solve_*_system())
    Problem *Problem_pt;
    double *Parameter_pt;          // The control parameter being varied to locate/track the Hopf point
    unsigned Ndof;                 // Number of dofs of the original (non-augmented) problem
    double Omega;                  // Unknown angular frequency of the oscillation at the bifurcation
    oomph::Vector<double> Phi;     // Real part of the critical (null) eigenvector
    oomph::Vector<double> Psi;     // Imaginary part of the critical (null) eigenvector
    oomph::Vector<double> C;       // Fixed normalization vector enforcing non-triviality of (Phi,Psi)
    oomph::Vector<int> Count;      // Number of elements contributing to each global equation, used to distribute/normalize shared equations
    double eigenweight;            // Scaling weight applied to the eigenvector equations relative to the base residuals

  public:
    bool call_param_change_handler;
    double FD_step = 1e-8;
    bool symmetric_FD = false;
    // Constructs the handler and initializes the eigenvector guess (Phi,Psi) and Omega from
    // an eigenvalue solve of the current Jacobian/mass matrix at the given parameter value.
    MyHopfHandler(Problem *const &problem_pt, double *const &parameter_pt);

    // Constructs the handler with an explicit initial guess for Omega and the eigenvector
    // (phi,psi), e.g. when continuing a Hopf point found previously rather than solving from scratch.
    MyHopfHandler(Problem *const &problem_pt, double *const &paramter_pt,
                  const double &omega, const oomph::DoubleVector &phi,
                  const oomph::DoubleVector &psi);

    ~MyHopfHandler() override;

    void set_eigenweight(double ew);
    unsigned get_problem_ndof() { return Ndof; }
    // Number of dofs of the augmented element: original element dofs plus its contribution
    // to Phi, Psi and (for one designated element) Omega.
    unsigned ndof(oomph::GeneralisedElement *const &elem_pt) override;

    // Maps a local dof index of the augmented element to its global equation number in the
    // augmented system, distinguishing between original-, Phi-, Psi- and Omega-dofs.
    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt,
                             const unsigned &ieqn_local) override;

    // Assembles the augmented residual vector (original residuals plus the eigenvector and
    // normalization equations described in the class comment above) for one element.
    void get_residuals(oomph::GeneralisedElement *const &elem_pt,
                       oomph::Vector<double> &residuals) override;

    // Assembles the augmented Jacobian, i.e. the derivatives of all augmented residuals
    // (base + eigen-equations + normalization) with respect to all augmented dofs
    // (u, Phi, Psi, Omega). Off-diagonal blocks require the Hessian of the base residuals
    // since J and M themselves depend on u.
    void get_jacobian(oomph::GeneralisedElement *const &elem_pt,
                      oomph::Vector<double> &residuals,
                      oomph::DenseMatrix<double> &jacobian) override;

    // Derivative of the augmented residuals with respect to the bifurcation parameter,
    // needed for arclength continuation of the Hopf point.
    void get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt,
                                   double *const &parameter_pt,
                                   oomph::Vector<double> &dres_dparam) override;

    void get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt,
                                  double *const &parameter_pt,
                                  oomph::Vector<double> &dres_dparam,
                                  oomph::DenseMatrix<double> &djac_dparam) override;

    void get_hessian_vector_products(oomph::GeneralisedElement *const &elem_pt,
                                     oomph::Vector<double> const &Y,
                                     oomph::DenseMatrix<double> const &C,
                                     oomph::DenseMatrix<double> &product) override;

    // Compares the analytically assembled Jacobian entries against a finite-difference
    // approximation (step size eps) for the given element, printing mismatches; a debugging aid.
    void debug_analytical_filling(oomph::GeneralisedElement *elem_pt, double eps);
    int bifurcation_type() const override { return 3; }

    double *bifurcation_parameter_pt() const override
    {
      return Parameter_pt;
    }

    void get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction) override;

    // Returns the complex eigenfunction Phi+i*Psi after rotating it in the complex plane
    // (multiplying by a unit-modulus phase) to a canonical, reproducible orientation, since
    // the Hopf eigenvector is only determined up to an arbitrary phase.
    std::vector<std::complex<double>> get_nicely_rotated_eigenfunction();

    const double &omega() const { return Omega; }

    // Switches Solve_which_system so that subsequent get_residuals()/get_jacobian() calls
    // only assemble the original (non-augmented) system, e.g. for a plain Newton step on u.
    void solve_standard_system();

    // Switches Solve_which_system so that only the complex eigenvector block (Phi,Psi,Omega
    // equations) is assembled, e.g. to re-solve the eigenproblem at fixed u.
    void solve_complex_system();

    // Switches Solve_which_system back to assembling the full augmented system.
    void solve_full_system();

    void realign_C_vector(); // Reset the C-vector (which enforces the non-triviality of the eigenvector)
  };

  //////////////////////////////////////////////////////////

  // List of the tree residual contribution indices of each generated C code
  // Records, for a given generated element code, which of its (possibly several) generated
  // residual-assembly variants correspond to the base-state residual and to the mass-matrix
  // residual; used by MyPitchForkHandler to switch a code-generated element between these
  // variants when assembling the different blocks of the augmented pitchfork system.
  class PitchForkResidualContributionList
  {
  public:
    DynamicBulkElementCode *code;
    std::vector<int> residual_indices; // index 0 is base state, 1 is mass matrix residual
    PitchForkResidualContributionList(DynamicBulkElementCode *_code, int _base, int _massmat) : code(_code) { residual_indices = {_base, _massmat}; }
    PitchForkResidualContributionList() {}
  };

  // Assembly handler that augments the residuals with the equations tracking a symmetry-
  // breaking pitchfork bifurcation. The augmented unknowns are the original dofs u, the
  // eigenvector Y of the null space of the Jacobian J(u,p), and the parameter p itself
  // (held fixed relative to a fold-like formulation, here a symmetry-breaking amplitude
  // Sigma is tracked instead). Schematically the augmented system is
  //   R(u,p)                = 0                (original residuals)
  //   J(u,p)*Y               = 0                (null eigenvector equation)
  //   C . Y - 1              = 0                (normalization, prevents Y=0)
  //   Psi . u - Sigma         = 0                (symmetry constraint tying u to the antisymmetric mode)
  // where Psi is a fixed symmetry (anti-symmetry) vector distinguishing the symmetric branch
  // from the bifurcating asymmetric one, and Sigma measures the amplitude of the broken-symmetry
  // component of u (Sigma=0 on the symmetric branch, at the pitchfork itself).
  class MyPitchForkHandler : public oomph::AssemblyHandler
  {
  protected:
    Problem *Problem_pt;
    unsigned Ndof;                 // Number of dofs of the original (non-augmented) problem
    double Sigma;                  // Amplitude of the symmetry-breaking component of u (unknown, tracked as part of the augmented system)
    oomph::Vector<double> Y;       // Null eigenvector of the Jacobian at the pitchfork point
    oomph::Vector<double> Psi;     // Fixed vector defining the (anti-)symmetry constraint on u
    oomph::Vector<double> C;       // Fixed normalization vector enforcing non-triviality of Y
    oomph::Vector<int> Count;      // Number of elements contributing to each global equation
    double *Parameter_pt;          // The control parameter being varied to locate/track the pitchfork
    double eigenweight, symmetryweight; // Scaling weights for the eigenvector- and symmetry-constraint equations, respectively
    unsigned Nelement;
    // Per generated-code lookup of which residual-assembly variant is the base state vs. mass matrix
    std::map<const pyoomph::DynamicBulkElementCode *, PitchForkResidualContributionList> residual_contribution_indices;
    // Builds residual_contribution_indices by inspecting all generated element codes once up front.
    void setup_U_times_Psi_residual_indices();
    // Computes the element-local contribution to the integral of U.Psi (used to fill in the
    // symmetry-constraint equation and its Jacobian) along with the psi_i*psi_j shape-function
    // products needed for its derivative.
    double get_integrated_U_dot_Psi(oomph::GeneralisedElement *const &elem_pt, oomph::DenseMatrix<double> &psi_i_times_psi_j);
    // Switches a generated-code element to assemble a particular residual variant (base state or
    // mass matrix); returns false if that element has no contribution of that kind.
    bool set_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode);
    int resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode);

  public:
    bool call_param_change_handler;
    // Constructs the handler; symmetry_vector defines the fixed Psi used in the symmetry
    // constraint. The eigenvector Y and Sigma are initialized internally.
    MyPitchForkHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &symmetry_vector);
    ~MyPitchForkHandler() override;
    void set_eigenweight(double ew);
    unsigned get_problem_ndof() { return Ndof; }
    // Number of dofs of the augmented element: original element dofs plus its contribution to Y and Sigma.
    unsigned ndof(oomph::GeneralisedElement *const &elem_pt) override;
    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local) override;
    // Assembles the augmented residual vector (original residuals plus the null-eigenvector,
    // normalization and symmetry-constraint equations) for one element.
    void get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals) override;
    // Assembles the augmented Jacobian for all augmented dofs (u, Y, Sigma).
    void get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override;
    void get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam) override;
    void get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam) override;
    void get_hessian_vector_products(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product) override;
    int bifurcation_type() const override { return 2; }
    double *bifurcation_parameter_pt() const override { return Parameter_pt; }
    void get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction) override;
    // Switches back to assembling the full augmented system (see MyHopfHandler::solve_full_system for the analogous pattern).
    void solve_full_system();
  };

  //////////////////////////////////////////////////////////

  // Assembly handler that augments the residuals with the equations tracking a fold
  // (limit point / saddle-node) bifurcation, where the Jacobian J(u,p) becomes singular
  // with a single real null eigenvector Y. The augmented unknowns are the original dofs u,
  // the null eigenvector Y, and the parameter p. Schematically:
  //   R(u,p)     = 0             (original residuals)
  //   J(u,p)*Y   = 0             (null eigenvector equation)
  //   Phi . Y - 1 = 0             (normalization, prevents Y=0)
  // where Phi is a fixed vector chosen so the normalization is non-degenerate. Unlike Hopf/
  // pitchfork tracking, no oscillation frequency or symmetry constraint is needed here since
  // a fold has a single real critical eigenvalue crossing zero.
  class MyFoldHandler : public oomph::AssemblyHandler
  {
    // Selects which part of the augmented system get_residuals()/get_jacobian() currently
    // assemble: the full augmented system, only the plain Jacobian block (for a Newton step
    // on u alone), or the eigenvector block augmented with the parameter equation.
    enum
    {
      Full_augmented,
      Block_J,
      Block_augmented_J
    };

    unsigned Solve_which_system;   // Current mode, one of the enum values above
    Problem *Problem_pt;
    unsigned Ndof;                 // Number of dofs of the original (non-augmented) problem
    oomph::Vector<double> Phi;     // Fixed normalization vector enforcing non-triviality of Y
    oomph::Vector<double> Y;       // Null eigenvector of the Jacobian at the fold point
    oomph::Vector<int> Count;      // Number of elements contributing to each global equation
    double *Parameter_pt;          // The control parameter being varied to locate/track the fold
    double eigenweight;            // Scaling weight applied to the eigenvector equations relative to the base residuals

  public:
    unsigned get_problem_ndof() { return Ndof; }
    bool call_param_change_handler;
    double FD_step = 1e-8;
    bool symmetric_FD = false;
    // Constructs the handler and initializes the eigenvector guess Y from an eigenvalue
    // solve of the current Jacobian at the given parameter value.
    MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt);
    // Constructs the handler with an explicit initial guess for the eigenvector.
    MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &eigenvector);
    // Constructs the handler with an explicit eigenvector guess and normalization vector Phi.
    MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &eigenvector, const oomph::DoubleVector &normalisation);
    ~MyFoldHandler() override;

    void set_eigenweight(double ew);

    // Number of dofs of the augmented element: original element dofs plus its contribution to Y.
    unsigned ndof(oomph::GeneralisedElement *const &elem_pt) override;

    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local) override;

    // Assembles the augmented residual vector (original residuals plus the null-eigenvector
    // and normalization equations) for one element, depending on Solve_which_system.
    void get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals) override;

    // Assembles the augmented Jacobian for the dofs selected by Solve_which_system.
    void get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override;

    void get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam) override;
    void get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam) override;

    void get_hessian_vector_products(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product) override;

    int bifurcation_type() const override { return 1; }

    double *bifurcation_parameter_pt() const override { return Parameter_pt; }

    void get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction) override;

    // Sets Solve_which_system = Block_augmented_J (eigenvector block + parameter equation only).
    void solve_augmented_block_system();
    // Sets Solve_which_system = Block_J (plain Jacobian block only, ignoring the eigen-equations).
    void solve_block_system();
    // Sets Solve_which_system = Full_augmented (assemble the complete augmented system).
    void solve_full_system();
    void realign_C_vector(); // Reset the C-vector (which enforces the non-triviality of the eigenvector)
  };

  //////////////////////////////

  // List of the tree residual contribution indices of each generated C code
  class AzimuthalSymmetryBreakingResidualContributionList
  {
  public:
    DynamicBulkElementCode *code;
    std::vector<int> residual_indices; // index 0 is base state(axisymm), 1 is real azimuthal and 2 is imag azimuthal
    AzimuthalSymmetryBreakingResidualContributionList(DynamicBulkElementCode *_code, int _base, int _real, int _imag) : code(_code) { residual_indices = {_base, _real, _imag}; }
    AzimuthalSymmetryBreakingResidualContributionList() {}
  };

  // Actual assembly handler class for axial symmetry breaking systems
  class AzimuthalSymmetryBreakingHandler : public oomph::AssemblyHandler
  {
    Problem *Problem_pt;                    // Pointer to the problem class
    unsigned Ndof;                          // Degrees of freedom of the original problem (non-augmented)
    oomph::Vector<double> real_eigenvector; // Storage for the real and imaginary eigenvector (which will be a part of the unknowns)
    oomph::Vector<double> imag_eigenvector;

    // A vector used to normalize the eigenvector (i.e. to prevent that real_eigenvector=0,
    // which trivially solves the eigenproblem with eigenvalue 0)
    oomph::Vector<double> normalization_vector;

    // Vector counting the occurrence of equations (many equations will be accessed by different elements)
    // The contributions to the normalization constraints will be hence added multiple times for these degrees
    // of freedom by different elements. Therefore, we have to normalize by the number of elements contributing
    // to each equation
    oomph::Vector<int> Count;
    double Omega;         // Imaginary part of the eigenvalue to be determined
    double *Parameter_pt; // Pointer to the critical parameter to find by this bifurcation analysis

    // Each generated C code has three residual forms: These are stored in this mapping
    std::map<const pyoomph::DynamicBulkElementCode *, AzimuthalSymmetryBreakingResidualContributionList>
        residual_contribution_indices;
    // We setup this mapping in beforehand

    // Once the mapping is set up, you can call this function to set which residual form should be assembled by the get_jacobian/get_residual/...
    // Please reset it to the base state at the end via set_assembled_residual(element,0);
    // it will return false, if there is zero residual contribution (an element might have e.g. no imaginary contribution).
    // Then you could skip this contributions and the derivatives thereof.
    bool set_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode);
    int resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode);
    // Indices (global equation numbers) of degrees of freedom which are forced to be zero due to the boundary conditions at the axis.
    // By default, all degrees at the axis are free (i.e. have an equation assigned). Depending on m, this set is filled.
    // If an equation is present in this set [ if (global_equations_forced_zero.count(global_eq)) {...} ], we do not add anything to
    // the residual (R_j=0) and we fill M_ij=0 and J_ij=0, except J_jj=1 (or any other nonzero value).
    std::set<unsigned> base_dofs_forced_zero;  // Base degrees of freedom forced to zero (e.g. velocity_x at the axis)
    std::set<unsigned> eigen_dofs_forced_zero; // Degrees of freedom of the eigenvector forced to zero (Note: the equation
                                               // indices are the base equations, not the indices of the eigendegrees!).
                                               // For e.g. m=1, equation of velocity_x at axis is part of base_dofs_forced_zero,
                                               // but not of eigen_dofs_forced_zero
    double eigenweight=1.0;                                               
  public:
    unsigned get_problem_ndof() { return Ndof; } // Returning the degrees of freedom of the original system (non-augmented)
    void set_eigenweight(double ew);

    // Setter of the forced zero degrees
    void set_global_equations_forced_zero(const std::set<unsigned> &base, const std::set<unsigned> &eigen)
    {
      base_dofs_forced_zero = base;
      eigen_dofs_forced_zero = eigen;
    }

    // These two guys will be false all the time.
    // when call_param_change_handler==true, it will call a function which is used in oomph-lib, but never in pyoomph. So we can skip it
    bool call_param_change_handler;

    double FD_step; // Finite difference step (usually small)

    bool has_imaginary_part; // If the imaginary part of the jacobian or mass matrix is present

    // Constructors. We must pass a problem, a parameter to optimize (i.e. to change in order to get Re(eigenvalue)=0) and a guess of the eigenvector and the imaginary part of the eigenvector
    AzimuthalSymmetryBreakingHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &real_eigen, const oomph::DoubleVector &imag_eigen, const double &Omega_guess,bool has_imag);
    // Destructor (used for cleaning up memory)
    ~AzimuthalSymmetryBreakingHandler() override;

    // Pyoomph has different residual contributions. The original residual along with its
    // jacobian and the real and imag part of the azimuthal Jacobian and mass matrix.
    // We get the indices of these contributions in beforehand. We assume that all codes are
    // initially set to the stage whether the original axisymmetric residual is solved
    void setup_solved_azimuthal_contributions(std::string real_angular_J_and_M, std::string imag_angular_J_and_M);

    // This will return the degrees of freedom of a single element of the augmented system
    // We will have to take the degrees of freedom of the original element and add a few more
    // for the eigenvector values (Re and Im)
    unsigned ndof(oomph::GeneralisedElement *const &elem_pt) override;

    // This will cast the local equation number of an element to a global equation number.
    // Again, we have to consider the additional equations for the unknown eigenvector (Re and Im)
    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local) override;

    // This will calculate the residual contribution of the original weak form by calling the function of the element
    // However, we will also have to add the contributions to the augmented residual form, i.e. the ones determining the eigenvector and critical parameter
    void get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals) override;

    // Same for the Jacobian: We must add the contributions for the augmented system
    void get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override;

    // Derivative of the augmented residuals with respect to the parameter
    void get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam) override;
    // Derivative of the augmented Jacobian with respect to the parameter
    void get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam) override;

    int bifurcation_type() const override { return 3; } // Internally used in oomph-lib. I assume it is best to return 3 (Hopf), since we have real and imag parts

    // Function to access the bifurcation parameter
    double *bifurcation_parameter_pt() const override { return Parameter_pt; }

    // Get the eigenfunction
    void get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction) override;
    const double &omega() const { return Omega; } // and the value of the imaginary part of the eigenvalue

    void solve_full_system();

    //  void realign_C_vector(); //Reset the C-vector (which enforces the non-triviality of the eigenvector)
  };


  // Find a periodic orbit by this
  class PeriodicBSplineBasis;

  // Assembly handler for periodic-orbit (limit-cycle) continuation. The augmented unknowns
  // are the values of the original dofs u at a discrete set of points around the orbit
  // (the "extra" copies live in Tadd, in addition to the dofs already stored in the problem's
  // current/history time levels) plus the unknown period T. Three different discretizations
  // of the periodic-in-time direction are supported, selected by 'basis'/'floquet_mode':
  //  - B-spline basis (basis!=NULL): u(s) is expanded in a periodic B-spline basis over the
  //    normalized orbit coordinate s in [0,1); see get_residuals_bspline_mode()/get_jacobian_bspline_mode().
  //  - Floquet mode (floquet_mode==true, basis==NULL): explicit dofs at s=1 (equal to s=0 by
  //    periodicity) are kept, and the discretization uses BDF2/finite differences between
  //    consecutive time nodes; see get_residuals_floquet_mode()/get_jacobian_floquet_mode().
  //  - Plain finite-difference/nodal mode (basis==NULL, floquet_mode==false): the time
  //    derivative at each nodal time point is approximated by finite differences against
  //    neighboring points (weights/indices cached in FD_ds_weights/FD_ds_inds); see
  //    get_residuals_time_nodal_mode()/get_jacobian_time_nodal_mode().
  // In all modes, in addition to the (time-)periodicity of u itself, one extra scalar
  // constraint fixes the phase/translation invariance of the orbit along time (T_constraint_mode
  // selects either a Poincare-plane constraint through (x0,n0,d_plane) or a period constraint),
  // and T itself is an unknown solved for as part of the augmented system.
  class PeriodicOrbitHandler : public oomph::AssemblyHandler
  {
  protected:
    Problem *Problem_pt;                    // Pointer to the problem class
    unsigned Ndof;                          // Degrees of freedom of the original problem (non-augmented)
    std::vector<std::vector<double>> Tadd;  // Additional time steps
    std::vector<double> x0;                 // Start point for the periodic orbit
    std::vector<double> n0;                 // Start normal for the periodic orbit
    double d_plane;                         // Plane offset for the Poincare section
    double T;                               // Period of the periodic orbit
    unsigned T_global_eqn, n_element;
    oomph::Vector<int> Count;
    PeriodicBSplineBasis *basis = NULL;     // If nonzero, we use a B-spline basis, otherwise BDF2, central FD between the nodes or
    bool floquet_mode;                      // if this is true (and basis==NULL), we use the Floquet mode, where we explictly have dofs for the periodic time point at s=1
    std::vector<double> s_knots;
    std::vector<double> backed_up_dofs;
    // When we do not have a spline basis, we do finite differences. Here, we store the coefficients and indices
    unsigned FD_ds_order;
    unsigned T_constraint_mode;             // 0: Plane constraint, 1: Period constraint
    std::vector<std::vector<double>> du0ds; // Derivatives of the start orbit for the phase constraint

    oomph::Mesh *time_mesh;
    oomph::Integral *collocation_gl;

    std::vector<std::vector<double>> FD_ds_weights;
    std::vector<std::vector<unsigned>> FD_ds_inds;
    // Per-discretization-mode residual/Jacobian assembly, dispatched to by get_residuals()/get_jacobian()
    // based on which of basis/floquet_mode/plain-FD is active (see class comment above).
    void get_jacobian_time_nodal_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    void get_residuals_time_nodal_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, double *const &parameter_pt = NULL);
    void get_jacobian_bspline_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    void get_residuals_bspline_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, double *const &parameter_pt = NULL);
    void get_jacobian_floquet_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    void get_residuals_floquet_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, double *const &parameter_pt = NULL);
    void get_jacobian_collocation_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    void get_residuals_collocation_mode(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, double *const &parameter_pt = NULL);

  public:
    // Recomputes du0ds (the derivative of the reference/starting orbit with respect to s),
    // used to build the phase constraint that removes the time-translation invariance of the orbit.
    void update_phase_constraint_information();
    unsigned get_problem_ndof() { return Ndof; } // Returning the degrees of freedom of the original system (non-augmented)
    bool is_floquet_mode() { return floquet_mode; }
    std::vector<std::tuple<double, double>> get_s_integration_samples(); // Returns tuples of (s,w), so that integral_0^1(f(U(s))*ds) ~= sum( f(U(s_i))*w_i )
    // Constructs the handler for an orbit of the given initial 'period' guess, discretized either
    // via a B-spline basis of order bspline_order (if >=0), or via nodal/Floquet finite differences
    // with Gauss-Legendre collocation order gl_order; 'tadd' holds the initial guesses for the extra
    // time-point dof copies, 'knots' the normalized s in [0,1] of the discretization points, and
    // T_constraint selects the phase-fixing constraint (0: plane, 1: period; see T_constraint_mode).
    PeriodicOrbitHandler(Problem *const &problem_pt, const double &period, const std::vector<std::vector<double>> &tadd, int bspline_order, int gl_order, std::vector<double> knots, unsigned T_constraint);
    // Destructor (used for cleaning up memory)
    ~PeriodicOrbitHandler() override;
    unsigned n_tsteps() const { return 1 + Tadd.size(); }
    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local) override;
    unsigned ndof(oomph::GeneralisedElement *const &elem_pt) override;
    // Dispatches to the residual assembly routine matching the active discretization mode.
    void get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals) override;
    // Dispatches to the Jacobian assembly routine matching the active discretization mode.
    void get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override;
    void get_dresiduals_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam) override;
    void get_djacobian_dparameter(oomph::GeneralisedElement *const &elem_pt, double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam) override;
    // Saves/restores the augmented dof vector, e.g. around trial evaluations that must not
    // permanently perturb the current orbit (used together with set_dofs_to_interpolated_values()).
    void backup_dofs();
    void restore_dofs();
    // Sets the problem's dofs to the orbit state interpolated at normalized orbit coordinate s in [0,1).
    void set_dofs_to_interpolated_values(const double &s);
    double get_knot_value(int i);
    unsigned get_periodic_knot_index(int i);
    double get_T() const { return T; }
  };

  // List of the tree residual contribution indices of each generated C code
  // Analogous to PitchForkResidualContributionList/AzimuthalSymmetryBreakingResidualContributionList:
  // records, per generated element code, which generated residual-assembly variant index
  // corresponds to which requested "contribution" name in CustomMultiAssembleHandler.
  class CustomMultiAssembleHandlerContributionList
  {
  public:
    DynamicBulkElementCode *code;
    std::vector<int> residual_indices; // index 0 is base state, 1 is mass matrix residual
    CustomMultiAssembleHandlerContributionList(DynamicBulkElementCode *_code, const std::vector<int> &resinds) : code(_code) { residual_indices = resinds; }
    CustomMultiAssembleHandlerContributionList() {}
  };

  // Bookkeeping for CustomMultiAssembleHandler: for one "contribution" name, records the index
  // (into the flat vector/matrix output arrays of get_all_vectors_and_matrices()) at which its
  // residual, jacobian and mass-matrix were placed, plus nested maps for the same information
  // per parameter (for *_dparameter requests) and per Hessian-vector-product request.
  class CustomMultiAssembleReturnIndexInfo
  {
  public:
    int residual_index = -1;
    int jacobian_index = -1;
    int mass_matrix_index = -1;
    std::map<double *, CustomMultiAssembleReturnIndexInfo> paramderivs;
    std::map<std::tuple<int, bool>, CustomMultiAssembleReturnIndexInfo> hessians;
    bool hessian_require_mass_matrix = false;
    std::vector<unsigned> hessian_vector_indices;
  };

  // Generic assembly handler used to compute an arbitrary, Python-requested combination of
  // residuals, Jacobians, mass matrices, parameter derivatives thereof, and Hessian-vector
  // products for one or more named "contributions" (residual forms) in a single element loop,
  // without needing a dedicated AssemblyHandler subclass per combination. 'what[i]' names the
  // quantity requested (e.g. "residuals", "jacobian", "mass_matrix", "dresiduals_dparameter",
  // "hessian_vector_product", ...) for 'contributions[i]', with any associated parameter name
  // taken from 'params' and any associated Hessian-contraction vector taken from hessian_vectors.
  // The constructor deduplicates the requested (what,contribution[,parameter/vector]) combinations
  // and assigns each a slot index into the flat output vector/matrix arrays filled by
  // get_all_vectors_and_matrices(); those slot indices are what setup_residual_contribution_map()
  // and the contribution_return_indices bookkeeping below resolve.
  class CustomMultiAssembleHandler : public oomph::AssemblyHandler
  {
  protected:
    Problem *problem;
    std::vector<std::string> &what;          // Requested quantity per entry, e.g. "residuals", "jacobian", "hessian_vector_product", ...
    std::vector<std::string> &contributions; // Named residual form (element residual contribution) requested per entry
    std::vector<std::string> &params;        // Parameter names for the *_dparameter entries, in order of occurrence in 'what'
    std::vector<double *> parameters;        // Resolved parameter pointers, parallel to 'what'/'contributions' (NULL where not applicable)
    std::vector<int> hessian_vector_indices;
    std::vector<std::vector<double>> &hessian_vectors; // The contraction vectors for Hessian-vector-product requests
    std::vector<bool> hessian_vector_transposed;
    bool transposed_hessians;
    std::vector<std::string> unique_contributions;                       // Deduplicated list of contribution names appearing in 'contributions'
    std::vector<CustomMultiAssembleReturnIndexInfo> contribution_return_indices; // Output-slot bookkeeping, one entry per unique_contributions
    unsigned nmatrix, nvector;                                           // Total number of matrix-valued / vector-valued outputs to allocate
    int resolve_assembled_residual(oomph::GeneralisedElement *const &elem_pt, int residual_mode);
    std::map<const pyoomph::DynamicBulkElementCode *, CustomMultiAssembleHandlerContributionList> residual_contribution_indices;
    // Builds residual_contribution_indices by inspecting all generated element codes once up front.
    void setup_residual_contribution_map();

  public:
    // Parses 'what'/'contributions'/'params'/hessian vector requests, resolves parameter/vector
    // references, deduplicates contributions and assigns output slot indices; 'return_indices' is
    // filled with, for each entry of 'what', the resolved slot index into the output arrays
    // (or a derived negative encoding for Hessian-related requests; see the .cpp for details).
    CustomMultiAssembleHandler(Problem *const &problem_pt, std::vector<std::string> &_what, std::vector<std::string> &_contributions, std::vector<std::string> &_params, std::vector<std::vector<double>> &_hessian_vectors, std::vector<unsigned> &_hessian_vector_indices, std::vector<int> &return_indices);
    ~CustomMultiAssembleHandler() override {}
    unsigned ndof(oomph::GeneralisedElement *const &elem_pt) override;
    unsigned long eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local) override;
    void get_residuals(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals) override;
    void get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian) override;
    // Assembles all requested vector- and matrix-valued quantities for one element in a single
    // pass, placing each into 'vec'/'matrix' at the slot index determined by the constructor.
    void get_all_vectors_and_matrices(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<oomph::Vector<double>> &vec, oomph::Vector<oomph::DenseMatrix<double>> &matrix) override;
    unsigned n_matrix() const { return nmatrix; }
    unsigned n_vector() const { return nvector; }
  };

}
